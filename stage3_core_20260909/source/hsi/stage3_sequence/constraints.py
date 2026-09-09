"""Exact target-frame locks and differentiable full-pose tolerance constraints.

Motion is ``[B,T,135]``: XYZ root followed by 22 row-convention rotation-6D
groups. Target frame indices are one-based within this generated sequence;
input history is not part of the sequence. Inactive padded slots may use zero.

Projection changes only explicitly locked coordinates. Tolerance errors apply
to *all* active target coordinates, including unlocked ones. The numerical
rotation fallback makes zero/collinear inputs well-defined but does not certify
them: their validity flag remains false and they cannot satisfy the contract.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

JOINT_COUNT = 22
MOTION_DIM = 3 + JOINT_COUNT * 6


def _require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _floating(value, label: str) -> None:
    _require(isinstance(value, Tensor) and value.is_floating_point(), label + ' must be a floating tensor')


def _rot6d(value: Tensor) -> None:
    _floating(value, 'rotation6d')
    _require(value.ndim >= 1 and value.shape[-1] == 6, 'rotation6d must end in six values')


def rotation_6d_valid_mask(value: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Whether both supplied rows span a finite, nondegenerate rotation frame.

This reports recoverable rank, not that raw rows are already orthonormal. The
usual continuous 6D representation does not require unit input rows.
"""
    _rot6d(value)
    _require(math.isfinite(eps) and eps > 0, 'eps must be finite and positive')
    eps = max(eps, 4 * torch.finfo(value.dtype).eps)
    finite = torch.isfinite(value).all(dim=-1)
    safe = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    first, second = safe[..., :3], safe[..., 3:]
    norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    unit = first / norm.clamp_min(eps)
    residual = second - (unit * second).sum(dim=-1, keepdim=True) * unit
    second_threshold = eps * torch.linalg.vector_norm(second, dim=-1).clamp_min(1.)
    return finite & (norm.squeeze(-1) > eps) & (torch.linalg.vector_norm(residual, dim=-1) > second_threshold)


def rotation_6d_to_matrix(value: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Gram–Schmidt row rotation with deterministic zero/collinear fallback.

The first zero row becomes +X; a degenerate second row uses the coordinate
axis least aligned with the first. Use ``rotation_6d_valid_mask`` to distinguish
this numerical fallback from a valid supplied rotation. Nonfinite input raises.
"""
    _rot6d(value)
    _require(torch.isfinite(value).all(), 'rotation6d contains nonfinite values')
    _require(math.isfinite(eps) and eps > 0, 'eps must be finite and positive')
    eps = max(eps, 4 * torch.finfo(value.dtype).eps)
    first, second = value[..., :3], value[..., 3:]
    norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    x_axis = torch.zeros_like(first)
    x_axis[..., 0] = 1
    b1 = torch.where(norm > eps, first / norm.clamp_min(eps), x_axis)
    residual = second - (b1 * second).sum(dim=-1, keepdim=True) * b1
    norm2 = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    axis = torch.nn.functional.one_hot(b1.abs().argmin(dim=-1), num_classes=3).to(value.dtype)
    fallback = axis - (axis * b1).sum(dim=-1, keepdim=True) * b1
    fallback = fallback / torch.linalg.vector_norm(fallback, dim=-1, keepdim=True).clamp_min(eps)
    second_threshold = eps * torch.linalg.vector_norm(second, dim=-1, keepdim=True).clamp_min(1.)
    b2 = torch.where(norm2 > second_threshold, residual / norm2.clamp_min(eps), fallback)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Read the first two matrix rows, identity = [1,0,0,0,1,0]."""
    _floating(matrix, 'rotation matrix')
    _require(matrix.ndim >= 2 and matrix.shape[-2:] == (3, 3), 'rotation matrix must end in [3,3]')
    _require(torch.isfinite(matrix).all(), 'rotation matrix contains nonfinite values')
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6).clone()


def legalize_rotation_6d(value: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Return orthonormal rotation rows; this operation does not validate input."""
    return matrix_to_rotation_6d(rotation_6d_to_matrix(value, eps=eps))


def rotation_geodesic(first: Tensor, second: Tensor) -> Tensor:
    """SO(3) angle in radians, using atan2 for finite zero-angle gradients.

The norm of the relative skew vector is sin(theta). Unlike unclamped acos,
atan2 has no infinite derivative at identity. The norm's zero subgradient also
defines the exact-identity case; the cut locus at pi remains nondifferentiable.
"""
    _require(first.shape == second.shape, 'Rotation shapes differ')
    left, right = rotation_6d_to_matrix(first), rotation_6d_to_matrix(second)
    relative = left @ right.transpose(-1, -2)
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), dim=-1) * .5
    sine = torch.linalg.vector_norm(skew, dim=-1)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) * .5
    return torch.atan2(sine, cosine)


def _layout(motion: Tensor, keypose_frames: Tensor, keypose_mask: Tensor) -> tuple[int, int, int]:
    _floating(motion, 'motion')
    _require(motion.ndim == 3 and motion.shape[-1] == MOTION_DIM
             and motion.shape[0] > 0 and motion.shape[1] > 0, 'motion must be [B>0,T>0,135]')
    _require(torch.isfinite(motion).all(), 'motion contains nonfinite values')
    _require(isinstance(keypose_frames, Tensor) and keypose_frames.dtype == torch.long
             and keypose_frames.ndim == 2, 'keypose_frames must be long [B,K]')
    _require(isinstance(keypose_mask, Tensor) and keypose_mask.dtype == torch.bool
             and keypose_mask.shape == keypose_frames.shape, 'keypose_mask must be bool [B,K]')
    batch, length = motion.shape[:2]
    _require(keypose_frames.shape[0] == batch, 'Keypose and motion batch sizes differ')
    _require(keypose_frames.device == motion.device and keypose_mask.device == motion.device,
             'Frame/mask tensors must share the motion device')
    _require(not (keypose_mask[:, 1:] & ~keypose_mask[:, :-1]).any(), 'keypose_mask must be an active prefix')
    _require(((keypose_frames >= 0) & (keypose_frames <= length)).all()
             and (keypose_frames[keypose_mask] >= 1).all(), 'Active target frames must be in 1..T; only inactive padding may be zero')
    active_frames = torch.where(keypose_mask, keypose_frames, length + 1).sort(dim=1).values
    _require(not ((active_frames[:, 1:] == active_frames[:, :-1]) & (active_frames[:, 1:] <= length)).any(),
             'Duplicate active target frames are forbidden')
    return batch, length, keypose_frames.shape[1]


def _locks(motion, frames, mask, locked_root, locked_joints):
    batch, _, count = _layout(motion, frames, mask)
    for tensor, shape, label in ((locked_root, (batch, count, 3), 'locked_root'),
                                  (locked_joints, (batch, count, JOINT_COUNT), 'locked_joints')):
        _require(isinstance(tensor, Tensor) and tensor.dtype == torch.bool and tensor.shape == shape
                 and tensor.device == motion.device, label + ' must have the declared bool shape/device')
    return torch.cat((locked_root, locked_joints.unsqueeze(-1).expand(-1, -1, -1, 6).reshape(batch, count, 132)), dim=-1)


def _targets(motion, keyposes, frames, mask):
    batch, _, count = _layout(motion, frames, mask)
    _floating(keyposes, 'keyposes')
    _require(keyposes.shape == (batch, count, MOTION_DIM) and keyposes.device == motion.device
             and keyposes.dtype == motion.dtype, 'keyposes must be [B,K,135] with the motion dtype/device')
    _require(torch.isfinite(keyposes).all(), 'keyposes contain nonfinite values')


def extract_keyposes(motion: Tensor, keypose_frames: Tensor, keypose_mask: Tensor) -> Tensor:
    """Gather one-based target frames, zeroing inactive padded slots."""
    batch, _, count = _layout(motion, keypose_frames, keypose_mask)
    indices = (keypose_frames - 1).clamp_min(0).unsqueeze(-1).expand(batch, count, MOTION_DIM)
    gathered = motion.gather(1, indices)
    return torch.where(keypose_mask[..., None], gathered, torch.zeros_like(gathered))


def project_keyposes(motion: Tensor, keyposes: Tensor, keypose_frames: Tensor, keypose_mask: Tensor,
                     locked_root: Tensor, locked_joints: Tensor) -> Tensor:
    """Copy only specified root coordinates / complete six-value joint groups.

    No interpolation, tolerance clamping, pose legalizing, or in-place edit is
    performed. Gradients to overwritten motion coordinates are exactly zero.
    """
    _targets(motion, keyposes, keypose_frames, keypose_mask)
    locks = _locks(motion, keypose_frames, keypose_mask, locked_root, locked_joints)
    batch_indices = torch.arange(motion.shape[0], device=motion.device)[:, None].expand_as(keypose_frames)[keypose_mask]
    frames = keypose_frames[keypose_mask] - 1
    output = motion.clone()
    output[batch_indices, frames] = torch.where(locks[keypose_mask], keyposes[keypose_mask], motion[batch_indices, frames])
    return output


def free_dof_mask(motion: Tensor, keypose_frames: Tensor, keypose_mask: Tensor,
                  locked_root: Tensor, locked_joints: Tensor) -> Tensor:
    """True for freely optimizable motion coordinates; body shape is not a DOF."""
    locks = _locks(motion, keypose_frames, keypose_mask, locked_root, locked_joints)
    batch_indices = torch.arange(motion.shape[0], device=motion.device)[:, None].expand_as(keypose_frames)[keypose_mask]
    result = torch.ones_like(motion, dtype=torch.bool)
    result[batch_indices, keypose_frames[keypose_mask] - 1] = ~locks[keypose_mask]
    return result


def _tolerance(value, motion, shape, label):
    _floating(value, label)
    _require(value.shape == shape and value.device == motion.device,
             label + ' must have the declared shape and motion device')
    _require(torch.isfinite(value).all() and (value >= 0).all(), label + ' must be finite and nonnegative')


def _maximum(value: Tensor) -> Tensor:
    return value.max() if value.numel() else value.sum()


def constraint_metrics(motion: Tensor, keyposes: Tensor, keypose_frames: Tensor, keypose_mask: Tensor,
                       locked_root: Tensor, locked_joints: Tensor,
                       root_tolerance: Tensor, joint_tolerance: Tensor) -> dict[str, Tensor]:
    """Measure the supplied motion without projecting it or hiding deviations.

    Raw errors and tolerance hinges include every active root/joint. Separate
    locked-error arrays measure exact-lock drift, even when a nonzero tolerance
    was supplied. Padded slots contribute zero. Scalar maxima remain tensors.
    """
    _targets(motion, keyposes, keypose_frames, keypose_mask)
    _locks(motion, keypose_frames, keypose_mask, locked_root, locked_joints)
    batch, count = keypose_mask.shape
    _tolerance(root_tolerance, motion, (batch, count, 3), 'root_tolerance')
    _tolerance(joint_tolerance, motion, (batch, count, JOINT_COUNT), 'joint_tolerance')
    actual = extract_keyposes(motion, keypose_frames, keypose_mask)
    active = keypose_mask[..., None]
    rotations = actual[..., 3:].reshape(batch, count, JOINT_COUNT, 6)
    target_rotations = keyposes[..., 3:].reshape(batch, count, JOINT_COUNT, 6)
    root_error = torch.where(active, (actual[..., :3] - keyposes[..., :3]).abs(), 0.)
    joint_error = torch.where(active, rotation_geodesic(rotations, target_rotations), 0.)
    root_violation = torch.where(active, torch.relu(root_error - root_tolerance), 0.)
    joint_violation = torch.where(active, torch.relu(joint_error - joint_tolerance), 0.)
    locked_root_error = torch.where(active & locked_root, root_error, 0.)
    locked_joint_error = torch.where(active & locked_joints, joint_error, 0.)
    rotation_valid = ~active | (rotation_6d_valid_mask(rotations) & rotation_6d_valid_mask(target_rotations))
    return dict(root_error=root_error, joint_error=joint_error,
        root_violation=root_violation, joint_violation=joint_violation,
        locked_root_error=locked_root_error, locked_joint_error=locked_joint_error,
        rotation_valid=rotation_valid, rotation_invalid_count=(~rotation_valid).sum(),
        max_root_error=_maximum(root_error), max_joint_error=_maximum(joint_error),
        max_root_violation=_maximum(root_violation), max_joint_violation=_maximum(joint_violation),
        max_locked_root_error=_maximum(locked_root_error), max_locked_joint_error=_maximum(locked_joint_error),
        tolerances_satisfied=((root_violation == 0).all() & (joint_violation == 0).all() & rotation_valid.all()))


def constraint_loss(motion: Tensor, keyposes: Tensor, keypose_frames: Tensor, keypose_mask: Tensor,
                    locked_root: Tensor, locked_joints: Tensor,
                    root_tolerance: Tensor, joint_tolerance: Tensor, *,
                    root_weight: float = 1., joint_weight: float = 1., reduction: str = 'mean') -> Tensor:
    """Squared tolerance hinges over all active target coordinates.

    Each sample contributes root_weight * mean(root hinge²) plus joint_weight *
    mean(joint angle hinge²), normalized by its active slots. ``none`` returns
    [B]; ``mean`` and ``sum`` reduce those samples. Empty samples contribute zero.
    Root units are metres; joint units are radians. Loss alone is not validity.
    """
    _require(reduction in ('none', 'mean', 'sum'), 'Unknown loss reduction')
    _require(all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in (root_weight, joint_weight)),
             'Constraint weights must be finite and nonnegative')
    metrics = constraint_metrics(motion, keyposes, keypose_frames, keypose_mask,
                                 locked_root, locked_joints, root_tolerance, joint_tolerance)
    counts = keypose_mask.sum(dim=1).clamp_min(1)
    root = metrics['root_violation'].square().sum(dim=(1, 2)) / (counts * 3)
    joints = metrics['joint_violation'].square().sum(dim=(1, 2)) / (counts * JOINT_COUNT)
    values = root_weight * root + joint_weight * joints
    return values if reduction == 'none' else values.mean() if reduction == 'mean' else values.sum()
