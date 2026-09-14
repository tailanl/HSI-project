"""World targets for official DiP's NEXT SUFFIX frame, not prefix-start.

Official t2m_prefix_collate supplies 40 suffix frames as ``motion`` and the
20 history frames separately. TrainingLoop.target_cond_modifier and the
target-location diffusion loss both call get_target_location on that suffix
alone. Its root integration therefore starts at zero XZ/yaw on the FIRST
PREDICTED frame. The known final history row's angular/linear increments
determine that next root frame; no future motion or target pose is decoded.

Only rigid coordinates change. Bodies, goals, root locks, scale and scene
geometry are never adjusted. HumanML heading follows official bug-fixed
hip+shoulder geometry; a desired world heading is atan2(world_y, world_x),
not an assumption that a legacy IK quaternion is the anatomical heading.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch


OLD_METHOD = Path(__file__).resolve().parents[2] / "closd_unihsi_sdf_stage3_20260912"


def _codec():
    source = OLD_METHOD / "code/hml_codec.py"
    previous_path, previous_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.path.insert(0, str(source.parent))
        sys.dont_write_bytecode = True
        module = importlib.import_module("hml_codec")
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
    if Path(module.__file__).resolve() != source:
        raise RuntimeError("foreign hml_codec namespace; use an isolated process")
    return module


def _history(full_history):
    if (not torch.is_tensor(full_history) or full_history.ndim != 4
            or full_history.shape[1:3] != (263, 1)
            or full_history.shape[0] < 1 or full_history.shape[-1] < 20
            or full_history.dtype != torch.float32
            or not torch.isfinite(full_history).all()):
        raise ValueError("full committed history must be finite float32 [B,263,1,T], T>=20")


def next_suffix_frame(full_history, mean, std, frame, *, vendor_root):
    """Return batched ``(world_from_local[B,3,3], translation[B,3], audit)``.

    Native row T-1 contains the displacement/angle increment into frame T.
    The appended row is a root-integration placeholder, NOT predicted joints;
    no channels of it influence next XZ/yaw. Vertical origin stays the supplied
    rigid frame's ground-origin Z, not the pelvis height. The 20-frame window
    start transform is included only to expose the distinct, wrong-for-target
    prefix-start origin in audits.
    """
    _history(full_history)
    codec = _codec()
    if not isinstance(frame, codec.RigidCanonicalFrame):
        raise ValueError("frame must be the original RigidCanonicalFrame")
    mean, std = codec._norms(mean, std, full_history)
    _, quaternion, _, motion = codec._motion_modules(str(Path(vendor_root).resolve()))
    raw = full_history[:, :, 0].permute(0, 2, 1) * std + mean
    placeholder = raw[:, -1:].clone()
    placeholder[..., :3] = 0  # never used to derive frame T; explicitly not extrapolated motion
    extended_raw = torch.cat((raw, placeholder), dim=1)
    rotation, root = motion.recover_root_rot_pos(extended_raw)
    normalized = ((extended_raw - mean) / std).permute(0, 2, 1)[:, :, None]
    headings = codec.root_heading(normalized, mean, std, frame, vendor_root=vendor_root)
    episode_rotation = torch.tensor(frame.world_from_local.copy(), dtype=raw.dtype, device=raw.device)
    episode_translation = torch.tensor(frame.translation.copy(), dtype=raw.dtype, device=raw.device)

    def at(index):
        basis = torch.eye(3, dtype=raw.dtype, device=raw.device)[None].expand(raw.shape[0], -1, -1)
        # qrot acts on row vectors; these three rotated basis rows form R.T.
        local_rotation = quaternion.qrot(quaternion.qinv(rotation[:, index])[:, None].expand(-1, 3, -1), basis).transpose(1, 2)
        world_rotation = episode_rotation[None] @ local_rotation
        origin = root[:, index].clone()
        origin[:, 1] = 0  # keep world height absolute; do NOT subtract pelvis height
        translation = origin @ episode_rotation.T + episode_translation
        return world_rotation, translation

    next_rotation, next_translation = at(raw.shape[1])
    prefix_index = raw.shape[1] - 20
    prefix_rotation, prefix_translation = at(prefix_index)
    audit = {
        "coordinate_contract": "official_get_target_location_suffix_only",
        "target_frame": "next_suffix_frame", "next_suffix_frame_index": raw.shape[1],
        "committed_history_frames": raw.shape[1], "context_len": 20,
        "prefix_window_start_index": prefix_index,
        "world_from_next_suffix": next_rotation.detach().cpu().tolist(),
        "next_suffix_translation_world": next_translation.detach().cpu().tolist(),
        "next_suffix_integrated_root_heading_world_rad": headings[:, -1].detach().cpu().tolist(),
        "world_from_prefix_start_AUDIT_ONLY": prefix_rotation.detach().cpu().tolist(),
        "prefix_start_translation_world_AUDIT_ONLY": prefix_translation.detach().cpu().tolist(),
        "prefix_start_integrated_root_heading_world_rad_AUDIT_ONLY": headings[:, prefix_index].detach().cpu().tolist(),
        "last_known_root_increment_raw": raw[:, -1, :3].detach().cpu().tolist(),
        "next_frame_source": "last committed native row angular and translation increments; no future pose",
        "future_motion_frames_read": 0, "world_body_modified": False, "scale": 1.,
        "world_up": "+Z", "target_frame_up": "+Y",
        "vertical_origin": "unchanged episode ground-origin; pelvis height is not subtracted",
        "official_source": ["data_loaders/tensors.py:t2m_prefix_collate",
                            "train/training_loop.py:target_cond_modifier",
                            "diffusion/gaussian_diffusion.py:training_losses",
                            "data_loaders/humanml/scripts/motion_process.py:get_target_location"],
    }
    return next_rotation, next_translation, audit


def build_target_conditions(full_history, mean, std, frame, world_targets, *,
                            vendor_root, heading_world=None):
    """Build official y fields plus ``audit``; do NOT pass audit into the model.

    ``world_targets`` maps only ``traj`` and/or ``pelvis`` to finite [B,3]
    world XYZ tensors. Optional ``heading_world[B]`` specifies anatomical
    forward in world XY radians. The returned three model fields are
    target_cond[B,8,3], target_joint_names (B lists), and is_heading[B] bool.
    A trajectory target has native Y=0; a pelvis target retains its actual
    native height. Unknown/disabled goal slots are exactly zero.
    """
    _history(full_history)
    if not isinstance(world_targets, dict) or not world_targets or set(world_targets) - {"traj", "pelvis"}:
        raise ValueError("world_targets must contain only traj and/or pelvis")
    batch = full_history.shape[0]
    targets = {}
    for name, value in world_targets.items():
        if not torch.is_tensor(value) or value.shape != (batch, 3) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("each world target must be a finite floating-point [B,3] tensor")
        targets[name] = value.to(device=full_history.device, dtype=full_history.dtype)
    if heading_world is not None:
        if (not torch.is_tensor(heading_world) or heading_world.shape != (batch,)
                or not heading_world.is_floating_point() or not torch.isfinite(heading_world).all()):
            raise ValueError("heading_world must be finite floating-point [B] radians")
        heading_world = heading_world.to(device=full_history.device, dtype=full_history.dtype)
    rotation, translation, audit = next_suffix_frame(full_history, mean, std, frame, vendor_root=vendor_root)
    motion = _codec()._motion_modules(str(Path(vendor_root).resolve()))[-1]
    slots = ["pelvis"] + list(motion.HML_EE_JOINT_NAMES) + ["traj", "heading"]
    if slots != ["pelvis", "left_foot", "right_foot", "left_wrist", "right_wrist", "head", "traj", "heading"]:
        raise RuntimeError("unsupported official goal-slot layout")
    condition = full_history.new_zeros((batch, len(slots), 3))
    for name, goal in targets.items():
        local = torch.einsum("bi,bij->bj", goal - translation, rotation)
        if name == "traj":
            local = local.clone()
            local[:, 1] = 0
        condition[:, slots.index(name)] = local
    is_heading = torch.full((batch,), heading_world is not None, dtype=torch.bool, device=full_history.device)
    if heading_world is not None:
        world_forward = torch.stack((heading_world.cos(), heading_world.sin(), torch.zeros_like(heading_world)), dim=-1)
        local_forward = torch.einsum("bi,bij->bj", world_forward, rotation)
        condition[:, -1, 0] = torch.atan2(local_forward[:, 0], local_forward[:, 2])
    names = [name for name in slots if name in targets]
    audit.update({"goal_slots": slots, "requested_target_names": names,
                  "heading_supplied": heading_world is not None,
                  "heading_convention": "world anatomical atan2(y,x) to official local atan2(x,z); no legacy-IK sign assumption",
                  "goal_world": {name: value.detach().cpu().tolist() for name, value in targets.items()}})
    return {"target_cond": condition, "target_joint_names": [names.copy() for _ in range(batch)],
            "is_heading": is_heading, "audit": audit}
