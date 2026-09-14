"""Rigid world/HumanML3D conversion for the standalone DiP research branch.

The initial prefix uses only one supplied J22 pose, copied 21 times to obtain
20 finite-difference feature frames. No future motion, skeleton rescaling,
floor snapping, motion smoothing or SMPL-X parameter recovery is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from functools import lru_cache
import importlib
import math
from pathlib import Path
import sys
import threading

import numpy as np
import torch


_IMPORT_LOCK = threading.RLock()
_FACE_INDICES = (2, 1, 17, 16)
_WORLD_UP = np.array([0.0, 0.0, 1.0])


def _initial_array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.array(value, dtype=np.float64, copy=True)
    if value.shape != (22, 3) or not np.isfinite(value).all():
        raise ValueError("initial_world_joints must be one finite [22,3] pose, not a motion sequence")
    return value


@dataclass(frozen=True)
class RigidCanonicalFrame:
    """Column-vector convention: world = world_from_local @ local + translation.

    Rotation must map local +Y to world +Z. Row-vector tensor helpers support
    arbitrary leading dimensions and preserve Torch gradients and devices.
    """

    world_from_local: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        rotation = np.array(self.world_from_local, dtype=np.float64, copy=True)
        translation = np.array(self.translation, dtype=np.float64, copy=True)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("Rigid frame needs rotation[3,3] and translation[3]")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Rigid frame must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0):
            raise ValueError("world_from_local must be orthonormal: scaling/shearing is forbidden")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
            raise ValueError("world_from_local must be a proper rotation, not a reflection")
        if not np.allclose(rotation[:, 1], _WORLD_UP, atol=1e-7, rtol=0):
            raise ValueError("Rigid frame must map HumanML +Y to world +Z")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "world_from_local", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def from_initial_world_joints(cls, initial_world_joints):
        joints = _initial_array(initial_world_joints)
        # Match official inverse_kinematics_np(..., fix_bug=False), not the
        # differently ordered heading calculation in process_file().
        l_hip, r_hip, sdr_r, sdr_l = _FACE_INDICES
        across = (joints[r_hip] - joints[l_hip]) + (joints[sdr_r] - joints[sdr_l])
        forward = np.cross(_WORLD_UP, across)
        magnitude = np.linalg.norm(forward)
        if magnitude < 1e-8:
            raise ValueError("Cannot derive a non-degenerate legacy HumanML heading")
        forward /= magnitude
        rotation = np.column_stack((np.cross(_WORLD_UP, forward), _WORLD_UP, forward))
        # Re-express horizontal root position only. Do not infer or shift floor.
        translation = np.array([joints[0, 0], joints[0, 1], 0.0])
        return cls(rotation, translation)

    def _arguments(self, points):
        if not hasattr(points, "shape") or len(points.shape) < 1 or points.shape[-1] != 3:
            raise ValueError("points must end in three coordinates")
        if torch.is_tensor(points):
            if not points.is_floating_point():
                raise ValueError("points must be floating point")
            return (torch.tensor(self.world_from_local.copy(), dtype=points.dtype, device=points.device),
                    torch.tensor(self.translation.copy(), dtype=points.dtype, device=points.device))
        points_array = np.asarray(points)
        if points_array.dtype.kind != "f":
            raise ValueError("points must be floating point")
        return self.world_from_local.astype(points_array.dtype), self.translation.astype(points_array.dtype)

    def to_local(self, world_points):
        rotation, translation = self._arguments(world_points)
        return (world_points - translation) @ rotation

    def to_world(self, local_points):
        rotation, translation = self._arguments(local_points)
        return local_points @ rotation.T + translation

    def as_dict(self):
        forward = self.world_from_local[:, 2]
        return {
            "world_from_local": self.world_from_local.tolist(), "translation": self.translation.tolist(),
            "world_up": "+Z", "local_up": "+Y", "local_forward": "+Z", "scale": 1.0,
            "world_heading_rad": math.atan2(float(forward[1]), float(forward[0])),
            "world_heading_convention": "atan2(world_forward_y, world_forward_x)",
            "convention": "world_column = world_from_local @ local_column + translation",
        }


def _check_motion_namespace(root):
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] != "data_loaders":
            continue
        paths = list(getattr(module, "__path__", ()))
        if getattr(module, "__file__", None):
            paths.append(module.__file__)
        if not paths or any(not Path(path).resolve().is_relative_to(root) for path in paths):
            raise RuntimeError("Foreign data_loaders namespace; use an isolated codec/DiP process")


@lru_cache(maxsize=4)
def _motion_modules(vendor_root):
    root = Path(vendor_root).expanduser().resolve(strict=True)
    required = root / "data_loaders" / "humanml" / "scripts" / "motion_process.py"
    if not required.is_file():
        raise ValueError("vendor_root does not contain the official HumanML motion utilities")
    with _IMPORT_LOCK:
        _check_motion_namespace(root)
        original_path, original_bytecode = list(sys.path), sys.dont_write_bytecode
        sys.path.insert(0, str(root))
        sys.dont_write_bytecode = True
        try:
            modules = tuple(importlib.import_module(name) for name in (
                "data_loaders.humanml.common.skeleton", "data_loaders.humanml.common.quaternion",
                "data_loaders.humanml.utils.paramUtil", "data_loaders.humanml.scripts.motion_process",
            ))
            _check_motion_namespace(root)
            return modules
        finally:
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_bytecode


def _norms(mean, std, like):
    mean = torch.as_tensor(mean, dtype=like.dtype, device=like.device)
    std = torch.as_tensor(std, dtype=like.dtype, device=like.device)
    if mean.shape != (263,) or std.shape != (263,):
        raise ValueError("mean/std must each have shape [263]")
    if not torch.isfinite(mean).all().item() or not torch.isfinite(std).all().item() or not (std > 0).all().item():
        raise ValueError("mean/std must be finite, with positive std")
    return mean, std


@contextmanager
def _unbiased_static_ik(quaternion_module, skeleton_module):
    """Remove the upstream identity-quaternion roll bias only during static IK.

    qbetween_np -> qbetween resolves qnormalize in the quaternion module. Also
    restore Skeleton's directly imported name for consistency. No source edit.
    """
    def normalize(q):
        norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        if not torch.isfinite(norm).all().item() or (norm <= 1e-12).any().item():
            raise ValueError("Degenerate/antipodal quaternion in static-prefix IK")
        return q / norm

    with _IMPORT_LOCK:
        original = quaternion_module.qnormalize
        original_skeleton = skeleton_module.qnormalize
        quaternion_module.qnormalize = normalize
        skeleton_module.qnormalize = normalize
        try:
            yield
        finally:
            quaternion_module.qnormalize = original
            skeleton_module.qnormalize = original_skeleton


def decode_history(native_features, mean, std, frame, *, vendor_root):
    """Decode FULL history [B,263,1,T] to world XYZ [B,T,22,3].

    Root displacement/heading integrate from the first history frame. Never
    pass just a suffix unless its initial root transform is explicitly known.
    This is differentiable in features (including root integration); it uses
    the official RIC-position path, not rotation FK or a SMPL-X body model.
    """
    if not torch.is_tensor(native_features) or native_features.ndim != 4:
        raise ValueError("native_features must be a Torch tensor [B,263,1,T]")
    if tuple(native_features.shape[1:3]) != (263, 1) or min(native_features.shape[0], native_features.shape[-1]) < 1:
        raise ValueError("native_features must have shape [B,263,1,T], B/T positive")
    if not native_features.is_floating_point() or not isinstance(frame, RigidCanonicalFrame):
        raise ValueError("Floating-point native features and a RigidCanonicalFrame are required")
    mean, std = _norms(mean, std, native_features)
    motion = _motion_modules(str(Path(vendor_root).resolve()))[-1]
    # Official recover_root_rot_pos allocates float32 scratch tensors, matching
    # trained DiP features. The cast preserves gradients even for float64 input.
    unnormalized = (native_features[:, :, 0].permute(0, 2, 1) * std + mean).float()
    local_joints = motion.recover_from_ric(unnormalized, 22)
    return frame.to_world(local_joints)


def root_heading(native_features, mean, std, frame, *, vendor_root):
    """World XY heading in radians, from the integrated native root rotation."""
    if not torch.is_tensor(native_features) or native_features.ndim != 4 or tuple(native_features.shape[1:3]) != (263, 1):
        raise ValueError("native_features must have shape [B,263,1,T]")
    mean, std = _norms(mean, std, native_features)
    _, quaternion, _, motion = _motion_modules(str(Path(vendor_root).resolve()))
    raw = (native_features[:, :, 0].permute(0, 2, 1) * std + mean).float()
    rotation, _ = motion.recover_root_rot_pos(raw)
    forward = torch.zeros((*rotation.shape[:-1], 3), dtype=raw.dtype, device=raw.device)
    forward[..., 2] = 1
    local_forward = quaternion.qrot(quaternion.qinv(rotation), forward)
    matrix = torch.tensor(frame.world_from_local.copy(), dtype=raw.dtype, device=raw.device)
    world_forward = local_forward @ matrix.T
    return torch.atan2(world_forward[..., 1], world_forward[..., 0])


def build_static_prefix(initial_world_joints, mean, std, *, vendor_root, context_len=20):
    """Return (normalized_prefix, rigid_frame, metadata) from one supplied pose."""
    if isinstance(context_len, bool) or context_len != 20:
        raise ValueError("This DiP checkpoint requires exactly 20 prefix frames")
    joints = _initial_array(initial_world_joints)
    skeleton_module, quaternion, params, _ = _motion_modules(str(Path(vendor_root).resolve()))
    skeleton = skeleton_module.Skeleton(torch.from_numpy(params.t2m_raw_offsets).float(),
                                       params.t2m_kinematic_chain, "cpu")
    for child, parent in enumerate(skeleton.parents()):
        if child and np.linalg.norm(joints[child] - joints[parent]) < 1e-8:
            raise ValueError(f"Degenerate input bone at joint {child}")
    frame = RigidCanonicalFrame.from_initial_world_joints(joints)
    corrections = 0
    for _ in range(3):
        positions = np.repeat(frame.to_local(joints)[None], context_len + 1, axis=0)
        with _unbiased_static_ik(quaternion, skeleton_module):
            quaternions = skeleton.inverse_kinematics_np(positions, list(_FACE_INDICES),
                                                        smooth_forward=False, fix_bug=False)
        if not np.isfinite(quaternions).all():
            raise ValueError("Official IK returned non-finite rotations for the initial body")
        root_matrix = quaternion.quaternion_to_matrix_np(quaternions[0, 0])
        if np.allclose(root_matrix, np.eye(3), atol=1e-7, rtol=0):
            break
        # Correct only the reversible coordinate frame, never move world body.
        frame = RigidCanonicalFrame(frame.world_from_local @ root_matrix.T, frame.translation)
        corrections += 1
    else:
        raise RuntimeError("Could not establish a canonical identity root quaternion")
    rotations6d = quaternion.quaternion_to_cont6d_np(quaternions)
    root_rotation = quaternions[:, 0].copy()
    rotation_velocity = quaternion.qmul_np(root_rotation[1:], quaternion.qinv_np(root_rotation[:-1]))
    linear_velocity = quaternion.qrot_np(root_rotation[1:], positions[1:, 0] - positions[:-1, 0])
    local_positions = positions.copy()
    local_positions[..., 0] -= positions[:, 0:1, 0]
    local_positions[..., 2] -= positions[:, 0:1, 2]
    local_positions = quaternion.qrot_np(np.repeat(root_rotation[:, None], 22, axis=1), local_positions)
    joint_velocity = quaternion.qrot_np(np.repeat(root_rotation[:-1, None], 22, axis=1),
                                        positions[1:] - positions[:-1])
    # Matches the official velocity-only contact heuristic for identical frames.
    # These ones describe only the constructed static prefix, not future locks.
    contacts = np.ones((context_len, 4), dtype=np.float64)
    raw = np.concatenate((np.arcsin(np.clip(rotation_velocity[:, 2:3], -1, 1)),
                          linear_velocity[:, [0, 2]], positions[:-1, 0, 1:2],
                          local_positions[:-1, 1:].reshape(context_len, 63),
                          rotations6d[:-1, 1:].reshape(context_len, 126),
                          joint_velocity.reshape(context_len, 66), contacts), axis=-1)
    if raw.shape != (20, 263) or not np.isfinite(raw).all():
        raise RuntimeError("Invalid static HumanML feature construction")
    raw_tensor = torch.from_numpy(raw).to(dtype=torch.float32)
    mean_tensor, std_tensor = _norms(mean, std, raw_tensor)
    prefix = ((raw_tensor - mean_tensor) / std_tensor).T[None, :, None, :].contiguous()
    recovered = decode_history(prefix, mean_tensor, std_tensor, frame, vendor_root=vendor_root)
    error = float(np.max(np.abs(recovered.detach().numpy()[0] - joints[None])))
    if error > 1e-5:
        raise RuntimeError(f"Initial world-body roundtrip exceeded tolerance: {error:.8g} m")
    metadata = {
        "prefix_source": "one_supplied_initial_J22_pose_repeated_21_times", "future_frames_consumed": 0,
        "input_pose_provenance": "caller-provided initial state; must bind caller receipt, not assumed observed mocap",
        "input_pose_count": 1, "repeated_pose_count": 21, "feature_frames": 20, "fps": 20,
        "frame": frame.as_dict(), "canonical_root_corrections": corrections,
        "initial_root_quaternion_wxyz": root_rotation[0].tolist(),
        "max_world_xyz_roundtrip_error_m": error, "world_body_modified": False,
        "skeleton_scaled": False, "floor_shifted": False, "motion_smoothed": False,
        "legacy_IK_face_indices": list(_FACE_INDICES), "legacy_IK_fix_bug": False,
        "static_IK_qnormalize_compatibility": "temporary unbiased unit normalization replaces upstream q[-1]+=1e-4; restored after IK; no vendor edit",
        "decode_dtype": "float32, matching official recover_root_rot_pos scratch tensors",
        "contacts": "velocity-only ones for constructed static history; not measured support or future foot locks",
        "representation": "HumanML263: root4+RIC63+IKrot126+localvel66+foot4; not original SMPL-X parameters",
        "decode_path": "full-history inverse-normalization + official recover_from_ric + rigid world transform",
    }
    return prefix, frame, metadata
