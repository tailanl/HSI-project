"""Optional HumanML representation consistency, not SMPL-X or physics.

The official HumanML IK/FK convention is retained, including its root-quaternion
sign and chain convention. Bone lengths are measured once from the supplied
initial body. No target, future GT, retargeting, smoothing or model is loaded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hml_codec import RigidCanonicalFrame, build_static_prefix, _motion_modules


class NativeConsistency:
    """Differentiable callback on FULL [B,263,1,T] history (T >= 20).

    ``loss`` is a convenience full-history mean. A rolling-horizon caller should
    instead select its editable frames from ``per_frame_fk_error_m2`` and
    ``per_frame_velocity_error_m2``. Velocity at t describes t -> t+1; include
    the preceding history frame when constraining a proposal's first boundary.
    The final velocity has no next frame and is explicitly masked out.
    """

    def __init__(self, initial_world_joints, mean, std, frame, vendor_root):
        if not isinstance(frame, RigidCanonicalFrame):
            raise ValueError("frame must be the codec's RigidCanonicalFrame")
        if torch.is_tensor(initial_world_joints):
            initial_world_joints = initial_world_joints.detach().cpu().numpy()
        initial = np.array(initial_world_joints, dtype=np.float64, copy=True)
        if initial.shape != (22, 3) or not np.isfinite(initial).all():
            raise ValueError("Expected one finite initial_world_joints[22,3], not future motion")
        self.mean = self._normalization(mean, "mean")
        self.std = self._normalization(std, "std")
        if not bool((self.std > 0).all()):
            raise ValueError("std must be strictly positive")
        self.frame = frame
        self.vendor_root = Path(vendor_root).resolve(strict=True)
        self._modules = _motion_modules(str(self.vendor_root))
        skeleton_module, _, params, _ = self._modules
        skeleton = skeleton_module.Skeleton(torch.from_numpy(params.t2m_raw_offsets).float(),
                                            params.t2m_kinematic_chain, "cpu")
        self.parents = tuple(skeleton.parents())
        # get_offsets_joints uses each supplied bone's length, not a mean body.
        local = torch.from_numpy(frame.to_local(initial)).float()
        self.offsets = skeleton.get_offsets_joints(local).detach().clone()
        self.bone_lengths = self.offsets[1:].norm(dim=-1)
        if not torch.isfinite(self.offsets).all() or not (self.bone_lengths > 1e-6).all():
            raise ValueError("Initial body has degenerate/nonfinite bones")
        self._skeletons = {"cpu": skeleton}
        prefix, expected_frame, codec_metadata = build_static_prefix(
            initial, self.mean, self.std, vendor_root=self.vendor_root)
        if not (np.allclose(frame.world_from_local, expected_frame.world_from_local, atol=1e-7, rtol=0)
                and np.allclose(frame.translation, expected_frame.translation, atol=1e-7, rtol=0)):
            raise ValueError("frame does not match the supplied initial body / codec convention")
        self._prefix = prefix.detach().clone()
        # Admission is an actual same-body FK/codec roundtrip, not a sign guess.
        with torch.no_grad():
            result = self(prefix)
            original = torch.from_numpy(initial).float()[None, None]
            fk_error = float((result["fk_world_joints"] - original).abs().max())
            ric_error = float((result["ric_world_joints"] - original).abs().max())
        if max(fk_error, ric_error) > 1e-5:
            raise ValueError("Official rotation FK is not admitted for this body/frame: "
                             f"FK={fk_error:.9g}m, RIC={ric_error:.9g}m")
        self.metadata = {
            "schema": "agent9.native_humanml_consistency.v1",
            "status": "initial_fk_ric_roundtrip_passed_not_motion_quality_acceptance",
            "initial_fk_max_abs_error_m": fk_error,
            "initial_ric_max_abs_error_m": ric_error,
            "initial_codec_roundtrip_error_m": codec_metadata["max_world_xyz_roundtrip_error_m"],
            "root_convention": "official recover_from_rot root quaternion unchanged, NOT inverted",
            "batch_handling": "official recover_from_rot applied per batch to preserve its T,22,6 layout",
            "bone_lengths_m": self.bone_lengths.tolist(),
            "bone_length_source": "one supplied same-shape initial J22; no uniform scaling",
            "velocity_convention": "qrot(root_quat[t], local_J[t+1]-local_J[t]); metres/frame",
            "velocity_energy": "mean of native66-vs-FK and native66-vs-RIC squared errors",
            "full_history_required": True,
            "history_prefix_verified": True,
            "contact_channels_constrained": False,
            "is_smplx_mesh": False,
            "is_physics": False,
            "future_gt_used": False,
            "vendor_source_modified": False,
        }

    @staticmethod
    def _normalization(value, name):
        if torch.is_tensor(value):
            value = value.detach().cpu()
        value = torch.as_tensor(value).clone().float()
        if value.shape != (263,) or not bool(torch.isfinite(value).all()):
            raise ValueError(name + " must be finite [263]")
        return value

    def _skeleton(self, device):
        key = str(device)
        if key not in self._skeletons:
            skeleton_module, _, params, _ = self._modules
            skeleton = skeleton_module.Skeleton(torch.from_numpy(params.t2m_raw_offsets).float(),
                                                params.t2m_kinematic_chain, device)
            skeleton.set_offset(self.offsets)
            self._skeletons[key] = skeleton
        return self._skeletons[key]

    def __call__(self, native_features):
        if not torch.is_tensor(native_features) or native_features.ndim != 4 \
                or tuple(native_features.shape[1:3]) != (263, 1) \
                or native_features.shape[0] < 1 or native_features.shape[-1] < 20:
            raise ValueError("Pass full history [B,263,1,T], B>=1, T>=20; not an isolated suffix")
        if not native_features.is_floating_point() or not bool(torch.isfinite(native_features).all()):
            raise ValueError("Native features must be finite floating point")
        prefix = self._prefix.to(device=native_features.device, dtype=native_features.dtype)
        if not torch.allclose(native_features[..., :20].detach(), prefix.expand(native_features.shape[0], -1, -1, -1),
                              atol=2e-5, rtol=1e-6):
            raise ValueError("Full history must begin with the unchanged bound 20-frame initial prefix")
        mean = self.mean.to(device=native_features.device, dtype=native_features.dtype)
        std = self.std.to(device=native_features.device, dtype=native_features.dtype)
        raw = (native_features[:, :, 0].permute(0, 2, 1) * std + mean).float()
        # Exact-zero/parallel 6D bases have no SO(3) interpretation. Reject them
        # instead of allowing upstream division by zero or collapsed skeletons.
        rotations = raw[..., 67:193].reshape(*raw.shape[:2], 21, 6)
        first, second = rotations[..., :3], rotations[..., 3:]
        first_norm, second_norm = first.norm(dim=-1), second.norm(dim=-1)
        cross_norm = torch.linalg.cross(first, second, dim=-1).norm(dim=-1)
        if not bool(((first_norm > 1e-7) & (second_norm > 1e-7)
                     & (cross_norm > 1e-7 * first_norm * second_norm)).all()):
            raise ValueError("Degenerate native rotation6D basis")
        _, quaternion, _, motion = self._modules
        skeleton = self._skeleton(native_features.device)
        # Upstream recover_from_rot flattens its rotation input, but not r_pos;
        # calling per [T,263] sequence is the official compatible batch contract.
        fk_local = torch.stack([motion.recover_from_rot(item, 22, skeleton) for item in raw])
        ric_local = motion.recover_from_ric(raw, 22)
        if not bool(torch.isfinite(fk_local).all()) or not bool(torch.isfinite(ric_local).all()):
            raise ValueError("Official HumanML recovery returned nonfinite joints")
        fk_world, ric_world = self.frame.to_world(fk_local), self.frame.to_world(ric_local)
        per_frame_fk = (fk_world - ric_world).square().mean(dim=(-1, -2))
        root_quaternion, _ = motion.recover_root_rot_pos(raw)
        velocity_rotation = root_quaternion[:, :-1, None].expand(-1, -1, 22, -1)
        fk_velocity = quaternion.qrot(velocity_rotation, fk_local[:, 1:] - fk_local[:, :-1])
        ric_velocity = quaternion.qrot(velocity_rotation, ric_local[:, 1:] - ric_local[:, :-1])
        native_velocity = raw[:, :-1, 193:259].reshape(raw.shape[0], raw.shape[1] - 1, 22, 3)
        velocity_errors = .5 * ((native_velocity - fk_velocity).square().mean(dim=(-1, -2))
                               + (native_velocity - ric_velocity).square().mean(dim=(-1, -2)))
        per_frame_velocity = torch.cat((velocity_errors, velocity_errors.new_zeros((raw.shape[0], 1))), dim=-1)
        velocity_mask = torch.ones(raw.shape[:2], dtype=torch.bool, device=raw.device)
        velocity_mask[:, -1] = False
        fk_loss, velocity_loss = per_frame_fk.mean(), velocity_errors.mean()
        loss = fk_loss + velocity_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite native consistency energy")
        return {
            "loss": loss, "fk_loss_m2": fk_loss, "velocity_loss_m2": velocity_loss,
            "per_frame_fk_error_m2": per_frame_fk,
            "per_frame_velocity_error_m2": per_frame_velocity,
            "velocity_valid_mask": velocity_mask,
            "fk_world_joints": fk_world, "ric_world_joints": ric_world,
        }
