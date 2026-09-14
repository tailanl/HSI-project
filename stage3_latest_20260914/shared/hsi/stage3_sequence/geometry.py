"""Differentiable geometry for the experimental whole-sequence Stage 3.

Coordinates are metric world XYZ (Z up); rotations use the PyTorch3D 6D
*row* convention, so identity is ``[1, 0, 0, 0, 1, 0]``.  FK region points
and bone midpoints are explicitly geometric proxies, not an SMPL-X surface
or a guarantee of full-mesh nonpenetration.  A production body adapter may
implement ``region_points`` / ``collision_points`` using actual vertices.

No target furniture is removed from the scene SDF.  Contact losses are
event-gated, whereas collision checks cover every generated frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MOTION_DIM = 135
NUM_JOINTS = 22
REGION_NAMES = (
    "pelvis", "left_thigh", "right_thigh", "left_foot", "right_foot",
    "left_hand", "right_hand", "back",
)
NUM_REGIONS = len(REGION_NAMES)
REGION_JOINTS = (0, 1, 2, 10, 11, 20, 21, 9)
FOOT_REGIONS = (3, 4)
RELATION_DIM = 14
RELATION_SLICES = {
    "scene_sdf": slice(0, 1), "target_distance": slice(1, 2),
    "target_direction": slice(2, 5), "target_normal": slice(5, 8),
    "contact_active": slice(8, 9), "outside_grid": slice(9, 10),
    "self_contact": slice(10, 11), "relative_tangential_velocity": slice(11, 14),
}


def rotation_6d_to_matrix(rotation: Tensor, eps: float = 1e-6) -> Tensor:
    """Gram--Schmidt row-basis conversion with finite degenerate fallbacks."""
    if rotation.shape[-1] != 6:
        raise ValueError("Rotation input must end in six coordinates")
    first, second = rotation[..., :3], rotation[..., 3:]
    n1 = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    fallback1 = torch.zeros_like(first)
    fallback1[..., 0] = 1
    b1 = torch.where(n1 > eps, first / n1.clamp_min(eps), fallback1)
    second = second - (b1 * second).sum(-1, keepdim=True) * b1
    n2 = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    # Pick the least-parallel coordinate axis.  This branch is only used for
    # degenerate inputs; ordinary rotations retain the standard conversion.
    axis = F.one_hot(b1.abs().argmin(-1), 3).to(rotation)
    fallback2 = F.normalize(axis - (axis * b1).sum(-1, keepdim=True) * b1, dim=-1, eps=eps)
    b2 = torch.where(n2 > eps, second / n2.clamp_min(eps), fallback2)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class BodyGeometry(Protocol):
    """A mesh-backed adapter can replace FK without changing geometry losses."""

    def region_points(self, motion: Tensor) -> Tensor:
        """Return [B,T,8,3] world points for the fixed REGION_NAMES contract."""

    def collision_points(self, motion: Tensor) -> Tensor:
        """Return [B,T,P,3] world collision points; P may differ from 8."""


class FKBody(nn.Module):
    """22-joint local-rotation FK with eight named point proxies.

    ``rest_offsets[j]`` is the rest bone from parent[j] to j, in the parent
    frame.  Parents must be topologically sorted, with joint 0 the sole root.
    User-supplied region offsets are in their attached joint's local frame.
    Defaults put thighs midway to each knee and all other regions at joints;
    notably foot points are *not* automatically anatomical sole vertices.
    """

    def __init__(self, rest_offsets: Tensor, parents: Tensor | list[int],
                 region_offsets: Tensor | None = None) -> None:
        super().__init__()
        offsets = torch.as_tensor(rest_offsets)
        if not offsets.is_floating_point():
            offsets = offsets.float()
        parent_values = torch.as_tensor(parents, dtype=torch.long).tolist()
        if offsets.shape != (NUM_JOINTS, 3) or len(parent_values) != NUM_JOINTS:
            raise ValueError("FK requires rest_offsets [22,3] and parents [22]")
        if parent_values[0] != -1 or any(p < 0 or p >= j for j, p in enumerate(parent_values[1:], 1)):
            raise ValueError("Parents must have root -1 followed by topologically sorted parents")
        if not torch.isfinite(offsets).all() or not torch.allclose(offsets[0], torch.zeros_like(offsets[0])):
            raise ValueError("Rest offsets must be finite and the root offset must be zero")
        self.parents = tuple(parent_values)
        self.register_buffer("rest_offsets", offsets.clone())
        self.register_buffer("region_joint_indices", torch.tensor(REGION_JOINTS, dtype=torch.long))
        if region_offsets is None:
            regions = offsets.new_zeros(NUM_REGIONS, 3)
            regions[1], regions[2] = offsets[4] / 2, offsets[5] / 2
        else:
            regions = torch.as_tensor(region_offsets).to(offsets)
        if regions.shape != (NUM_REGIONS, 3) or not torch.isfinite(regions).all():
            raise ValueError("region_offsets must be finite [8,3]")
        self.register_buffer("region_offsets", regions.clone())

    def _fk(self, motion: Tensor) -> tuple[Tensor, Tensor]:
        if motion.ndim != 3 or motion.shape[-1] != MOTION_DIM:
            raise ValueError("motion must be [B,T,135]")
        local = rotation_6d_to_matrix(motion[..., 3:].reshape(*motion.shape[:2], NUM_JOINTS, 6))
        positions = [motion[..., :3]]
        orientations = [local[..., 0, :, :]]
        offsets = self.rest_offsets.to(motion)
        for j in range(1, NUM_JOINTS):
            p = self.parents[j]
            positions.append(positions[p] + torch.matmul(orientations[p], offsets[j].unsqueeze(-1)).squeeze(-1))
            orientations.append(orientations[p] @ local[..., j, :, :])
        return torch.stack(positions, -2), torch.stack(orientations, -3)

    def forward(self, motion: Tensor) -> Tensor:
        return self._fk(motion)[0]

    def joints_world(self, motion: Tensor) -> Tensor:
        return self(motion)

    def _regions(self, joints: Tensor, rotations: Tensor) -> Tensor:
        indices = self.region_joint_indices.to(joints.device)
        attached = joints.index_select(-2, indices)
        basis = rotations.index_select(-3, indices)
        offsets = self.region_offsets.to(joints)
        return attached + (basis @ offsets[..., None]).squeeze(-1)

    def region_points(self, motion: Tensor) -> Tensor:
        return self._regions(*self._fk(motion))

    def collision_points(self, motion: Tensor) -> Tensor:
        joints, rotations = self._fk(motion)
        parent_indices = torch.tensor(self.parents[1:], device=motion.device)
        midpoints = (joints[..., 1:, :] + joints.index_select(-2, parent_indices)) / 2
        return torch.cat((joints, midpoints, self._regions(joints, rotations)), dim=-2)


@dataclass(frozen=True)
class SDFQuery:
    signed_distance_m: Tensor
    outside: Tensor

    @property
    def distances(self) -> Tensor:
        return self.signed_distance_m


class SceneGeometry(Protocol):
    def query(self, points: Tensor) -> SDFQuery:
        """Positive free space, negative solid/unknown; preserve leading shape."""


class AnalyticBoxScene(nn.Module):
    """Union of axis-aligned boxes and optional Z-up floor half-space.

    ``boxes`` has shape [N,2,3], storing lower/upper world XYZ bounds.  This
    small analytic fixture does not approximate arbitrary furniture meshes.
    """

    def __init__(self, boxes: Tensor | None = None, floor_z: float | None = 0.) -> None:
        super().__init__()
        boxes = torch.empty(0, 2, 3) if boxes is None else torch.as_tensor(boxes)
        if not boxes.is_floating_point():
            boxes = boxes.float()
        if boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or not torch.isfinite(boxes).all():
            raise ValueError("boxes must be finite [N,2,3]")
        if len(boxes) and not (boxes[:, 1] > boxes[:, 0]).all():
            raise ValueError("Every box must have positive extents")
        if floor_z is None and not len(boxes):
            raise ValueError("Scene must contain boxes or a floor")
        if floor_z is not None and not torch.isfinite(torch.tensor(floor_z)):
            raise ValueError("floor_z must be finite")
        self.register_buffer("boxes", boxes.clone())
        self.floor_z = floor_z

    def query(self, points: Tensor) -> SDFQuery:
        if points.shape[-1] != 3:
            raise ValueError("Scene points must end in XYZ")
        values = []
        if len(self.boxes):
            boxes = self.boxes.to(points)
            centres = (boxes[:, 0] + boxes[:, 1]) / 2
            half_size = (boxes[:, 1] - boxes[:, 0]) / 2
            delta = (points.unsqueeze(-2) - centres).abs() - half_size
            box_sdf = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) + delta.amax(-1).clamp_max(0)
            values.append(box_sdf.amin(-1))
        if self.floor_z is not None:
            values.append(points[..., 2] - self.floor_z)
        distance = values[0] if len(values) == 1 else torch.minimum(values[0], values[1])
        return SDFQuery(distance, torch.zeros_like(distance, dtype=torch.bool))

    sample = query


class GridSDF(nn.Module):
    """Trilinear signed field with explicit metric bounds and unknown mask.

    Grid shape is [D_z,H_y,W_x] or [B,D_z,H_y,W_x].  ``world_bounds`` is
    [2,3] (shared) or [B,2,3], lower then upper.  Bounds locate the *first and
    last grid sample centres*, not outer voxel edges (align_corners=True).
    Outside queries are unknown, never free: return a negative conservative
    distance equal to minus (distance to bounds + outside_margin).
    """

    def __init__(self, grid: Tensor, world_bounds: Tensor, outside_margin: float = .02) -> None:
        super().__init__()
        grid = torch.as_tensor(grid)
        if not grid.is_floating_point():
            grid = grid.float()
        if grid.ndim == 3:
            grid = grid[None]
        bounds = torch.as_tensor(world_bounds).to(grid)
        if bounds.ndim == 2:
            bounds = bounds[None]
        if grid.ndim != 4 or min(grid.shape[-3:]) < 2 or not torch.isfinite(grid).all():
            raise ValueError("grid must be finite [D,H,W] or [B,D,H,W], with each axis >=2")
        if bounds.ndim != 3 or bounds.shape[1:] != (2, 3) or not torch.isfinite(bounds).all():
            raise ValueError("world_bounds must be finite [2,3] or [B,2,3]")
        if not (bounds[:, 1] > bounds[:, 0]).all() or outside_margin <= 0:
            raise ValueError("Bounds need positive extent and outside_margin must be positive")
        self.register_buffer("grid", grid[:, None].clone())
        self.register_buffer("world_bounds", bounds.clone())
        self.outside_margin = float(outside_margin)

    def query(self, points: Tensor) -> SDFQuery:
        if points.ndim < 3 or points.shape[-1] != 3:
            raise ValueError("Grid query must be [B,...,3]")
        batch = points.shape[0]
        if self.grid.shape[0] not in (1, batch) or self.world_bounds.shape[0] not in (1, batch):
            raise ValueError("Grid/bounds batch must be one or match points")
        bounds = self.world_bounds.to(points).expand(batch, -1, -1)
        lower, upper = bounds[:, 0, None], bounds[:, 1, None]
        flat = points.reshape(batch, -1, 3)
        outside = ((flat < lower) | (flat > upper)).any(-1)
        normalized = 2 * (flat - lower) / (upper - lower) - 1
        volume = self.grid.to(points).expand(batch, -1, -1, -1, -1)
        distance = F.grid_sample(volume, normalized[:, None, None], mode="bilinear",
                                 padding_mode="border", align_corners=True).reshape(batch, -1)
        to_bounds = torch.relu(lower - flat) + torch.relu(flat - upper)
        unknown_distance = -torch.linalg.vector_norm(to_bounds, dim=-1) - self.outside_margin
        distance = torch.where(outside, unknown_distance, distance)
        return SDFQuery(distance.reshape(points.shape[:-1]), outside.reshape(points.shape[:-1]))

    sample = query


def _validate_relations(points: Tensor, targets: Tensor, normals: Tensor,
                        contact_active: Tensor, target_body_regions: Tensor) -> None:
    if points.ndim != 4 or points.shape[-2:] != (NUM_REGIONS, 3):
        raise ValueError("Body region points must be [B,T,8,3]")
    if targets.shape != points.shape or normals.shape != points.shape:
        raise ValueError("Targets and normals must match [B,T,8,3]")
    if contact_active.shape != points.shape[:-1] or contact_active.dtype != torch.bool:
        raise ValueError("contact_active must be bool [B,T,8]")
    if target_body_regions.shape != points.shape[:-1] or target_body_regions.dtype != torch.long:
        raise ValueError("target_body_regions must be long [B,T,8]")
    if ((target_body_regions < -1) | (target_body_regions >= NUM_REGIONS)).any():
        raise ValueError("target_body_regions must be -1 (scene) or an index in [0,8)")


def _resolve_targets(points: Tensor, targets: Tensor, target_body_regions: Tensor,
                     contact_active: Tensor, target_samples: Tensor | None = None,
                     sample_mask: Tensor | None = None) -> Tensor:
    """Choose closest target-region sample, or explicit point target fallback."""
    resolved = targets
    if sample_mask is not None and target_samples is None:
        raise ValueError("sample_mask requires target_samples")
    if target_samples is not None:
        if target_samples.ndim != 5 or target_samples.shape[:3] != points.shape[:3] or target_samples.shape[-1] != 3:
            raise ValueError("target_samples must be [B,T,8,P,3]")
        if target_samples.shape[-2] < 1:
            raise ValueError("Target sample dimension must be nonempty")
        if sample_mask is None:
            sample_mask = torch.ones(target_samples.shape[:-1], dtype=torch.bool, device=points.device)
        if sample_mask.shape != target_samples.shape[:-1] or sample_mask.dtype != torch.bool:
            raise ValueError("sample_mask must be bool [B,T,8,P]")
        has_sample = sample_mask.any(-1)
        if (contact_active & (target_body_regions == -1) & ~has_sample).any():
            raise ValueError("Active scene contact has no valid target-region sample")
        squared = (points.unsqueeze(-2) - target_samples).square().sum(-1)
        nearest = squared.masked_fill(~sample_mask, torch.inf).argmin(-1)
        selected = target_samples.gather(-2, nearest[..., None, None].expand(*nearest.shape, 1, 3)).squeeze(-2)
        resolved = torch.where(has_sample[..., None], selected, targets)
    body_target = points.gather(-2, target_body_regions.clamp_min(0)[..., None].expand_as(points))
    return torch.where((target_body_regions >= 0)[..., None], body_target, resolved)


def _velocity(points: Tensor, fps: float) -> Tensor:
    if fps <= 0 or not torch.isfinite(torch.tensor(fps)):
        raise ValueError("fps must be positive and finite")
    if points.shape[1] == 0:
        raise ValueError("Motion must contain at least one frame")
    return torch.cat((torch.zeros_like(points[:, :1]), (points[:, 1:] - points[:, :-1]) * fps), dim=1)


def relation_features(motion: Tensor, body: BodyGeometry, scene: SceneGeometry,
                      targets: Tensor, normals: Tensor, contact_active: Tensor,
                      target_body_regions: Tensor, *, fps: float,
                      target_samples: Tensor | None = None,
                      sample_mask: Tensor | None = None) -> Tensor:
    """Return [B,T,8,14], using current clean-motion geometry, never future GT.

    See RELATION_SLICES for the layout.  Normals are normalized; zero means
    unavailable, in which case tangential velocity is the full relative
    velocity.  Nearest sampled regions are discrete sample approximations,
    not triangle projections.  Without samples a scene target is a point.
    Contact_active is a supplied predicted-event gate, not an observed contact
    label; target distance remains available both inside and outside events.
    Scene geometry is static.  Its support velocity is zero, not the velocity
    of a nearest-point query (which moves when the body slides).  Self-contact
    subtracts the current target body region's velocity instead.
    """
    points = body.region_points(motion)
    _validate_relations(points, targets, normals, contact_active, target_body_regions)
    target = _resolve_targets(points, targets, target_body_regions, contact_active, target_samples, sample_mask)
    delta = target - points
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    direction = delta / distance.clamp_min(1e-6)
    normal = F.normalize(normals, dim=-1, eps=1e-6)
    body_velocity = _velocity(points, fps)
    self_velocity = body_velocity.gather(-2, target_body_regions.clamp_min(0)[..., None].expand_as(points))
    support_velocity = torch.where((target_body_regions >= 0)[..., None], self_velocity, torch.zeros_like(points))
    relative_velocity = body_velocity - support_velocity
    tangent = relative_velocity - (relative_velocity * normal).sum(-1, keepdim=True) * normal
    sdf = scene.query(points)
    return torch.cat((sdf.signed_distance_m[..., None], distance, direction, normal,
                      contact_active[..., None].to(points), sdf.outside[..., None].to(points),
                      (target_body_regions >= 0)[..., None].to(points), tangent), dim=-1)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(value)
    return (value * weights).sum() / weights.sum().clamp_min(1)


def collision_energy(motion: Tensor, body: BodyGeometry, scene: SceneGeometry,
                     *, margin: float = 0., radii: Tensor | None = None) -> Tensor:
    """All-frame squared proxy penetration; targets remain in the full SDF.

    Optional radii must broadcast to [B,T,P].  No contact event can disable
    this loss.  FK points miss surface detail, so full-mesh evaluation remains
    necessary even when the proxy loss reaches zero.
    """
    if margin < 0:
        raise ValueError("Collision margin must be nonnegative")
    points = body.collision_points(motion)
    sdf = scene.query(points).signed_distance_m
    clearance = torch.as_tensor(margin, dtype=motion.dtype, device=motion.device)
    if radii is not None:
        radii = torch.as_tensor(radii).to(motion)
        if (radii < 0).any():
            raise ValueError("Proxy radii must be nonnegative")
        clearance = clearance + radii
    return torch.relu(clearance - sdf).square().mean()


def _contact_loss(motion: Tensor, body: BodyGeometry, targets: Tensor, normals: Tensor,
                  contact_active: Tensor, target_body_regions: Tensor, *, self_only: bool,
                  tolerance: float | Tensor, target_samples: Tensor | None,
                  sample_mask: Tensor | None) -> Tensor:
    points = body.region_points(motion)
    _validate_relations(points, targets, normals, contact_active, target_body_regions)
    target = _resolve_targets(points, targets, target_body_regions, contact_active, target_samples, sample_mask)
    tolerance = torch.as_tensor(tolerance).to(motion)
    if (tolerance < 0).any():
        raise ValueError("Contact tolerance must be nonnegative")
    distance = torch.linalg.vector_norm(points - target, dim=-1)
    mask = contact_active & ((target_body_regions >= 0) if self_only else (target_body_regions == -1))
    return _masked_mean(torch.relu(distance - tolerance).square(), mask)


def contact_energy(motion: Tensor, body: BodyGeometry, targets: Tensor, normals: Tensor,
                   contact_active: Tensor, target_body_regions: Tensor, *,
                   tolerance: float | Tensor = .02, target_samples: Tensor | None = None,
                   sample_mask: Tensor | None = None) -> Tensor:
    """Event-gated *scene* contact only; use self_contact_energy separately."""
    return _contact_loss(motion, body, targets, normals, contact_active, target_body_regions,
                         self_only=False, tolerance=tolerance,
                         target_samples=target_samples, sample_mask=sample_mask)


def self_contact_energy(motion: Tensor, body: BodyGeometry, targets: Tensor, normals: Tensor,
                        contact_active: Tensor, target_body_regions: Tensor, *,
                        tolerance: float | Tensor = .02) -> Tensor:
    """Event-gated region-to-region proximity (e.g. wrist proxy to thigh).

    This is a positive self-contact requirement, not a self-collision detector.
    It does not assert that unspecified body regions can interpenetrate.
    """
    return _contact_loss(motion, body, targets, normals, contact_active, target_body_regions,
                         self_only=True, tolerance=tolerance, target_samples=None, sample_mask=None)


def foot_slip_energy(motion: Tensor, body: BodyGeometry, contact_active: Tensor,
                     normals: Tensor | None = None, *, fps: float,
                     target_positions: Tensor | None = None,
                     foot_regions: tuple[int, ...] = FOOT_REGIONS,
                     event_ids: Tensor | None = None) -> Tensor:
    """Tangential foot velocity only across consecutive active contact frames.

    Moving supports can supply [B,T,8,3] target_positions.  Optional event_ids
    [B,T,8] avoid comparing velocities across distinct consecutive events.
    A newly established contact does not punish its approach-frame velocity.
    """
    points = body.region_points(motion)
    if contact_active.shape != points.shape[:-1] or contact_active.dtype != torch.bool:
        raise ValueError("contact_active must be bool [B,T,8]")
    if normals is None:
        normals = torch.zeros_like(points)
        normals[..., 2] = 1
    if normals.shape != points.shape:
        raise ValueError("normals must be [B,T,8,3]")
    normal = F.normalize(normals, dim=-1, eps=1e-6)
    if target_positions is not None:
        if target_positions.shape != points.shape:
            raise ValueError("target_positions must be [B,T,8,3]")
        points = points - target_positions
    velocity = _velocity(points, fps)
    tangent = velocity - (velocity * normal).sum(-1, keepdim=True) * normal
    active = torch.cat((torch.zeros_like(contact_active[:, :1]),
                        contact_active[:, 1:] & contact_active[:, :-1]), dim=1)
    if event_ids is not None:
        if event_ids.shape != contact_active.shape:
            raise ValueError("event_ids must be [B,T,8]")
        same_event = torch.cat((torch.zeros_like(contact_active[:, :1]),
                                event_ids[:, 1:] == event_ids[:, :-1]), dim=1)
        active = active & same_event
    if not foot_regions or any(i < 0 or i >= NUM_REGIONS for i in foot_regions):
        raise ValueError("foot_regions must contain valid region indices")
    indices = torch.tensor(foot_regions, dtype=torch.long, device=motion.device)
    return _masked_mean(tangent.square().sum(-1).index_select(-1, indices), active.index_select(-1, indices))
