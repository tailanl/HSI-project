"""Differentiable query-time scene fields in ReMoGen's world Z-up frame.

This module deliberately contains no motion/example retrieval.  It turns the
raw occupancy attached to the current query into two purely geometric objects:

* :class:`RawSceneSDFZUp`, a metric signed-distance field (positive is free);
* :class:`QueryTimeICGF`, contact support made from either a declared target
  point, sparse joint targets, or the surface of the target occupancy component.

Raw grids use ``[X,Y,Z]`` storage and cell-bound bounds.  PyTorch's five-D
``grid_sample`` consumes ``[N,C,D,H,W]`` and coordinates in ``[W,H,D]`` order,
so a world point ``[x,y,z]`` is queried as normalized ``[z,y,x]``.  The sampler
uses ``align_corners=False`` because occupancy/SDF values live at cell centres.
Out-of-bounds points receive a differentiable negative distance that points
back toward the valid scene rather than a constant zero-gradient penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


DEFAULT_LOWER_XYZ = (-3.0, -4.0, 0.0)
DEFAULT_UPPER_XYZ = (3.0, 4.0, 2.0)


class SceneFieldError(ValueError):
    """A raw-scene or query-time field violates the metric contract."""


@dataclass(frozen=True)
class SDFSamples:
    """Metric SDF values and exact analytic scene-bounds membership."""

    signed_distance_m: Tensor
    in_bounds: Tensor


@dataclass(frozen=True)
class SDFLevelSetProjection:
    """Deterministic projection of query points onto one metric SDF level."""

    points_world_zup: Tensor
    initial_signed_distance_m: Tensor
    final_signed_distance_m: Tensor
    displacement_m: Tensor
    converged: Tensor
    iterations: int
    target_level_m: float
    tolerance_m: float


@dataclass(frozen=True)
class TargetComponent:
    """The query-selected target component and collision-only occupancy."""

    target_occupancy_xyz: np.ndarray
    collision_occupancy_xyz: np.ndarray
    seed_index_xyz: tuple[int, int, int]
    seed_distance_m: float
    search_crop_radius_m: float | None
    scene_occupied_voxel_count: int
    target_fraction_of_scene_occupied: float


def _finite_vector(name: str, value: Sequence[float], length: int = 3) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise SceneFieldError(f"{name} must be a finite length-{length} vector")
    return result


def _validate_bounds(
    lower_xyz: Sequence[float], upper_xyz: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    lower = _finite_vector("lower_xyz", lower_xyz)
    upper = _finite_vector("upper_xyz", upper_xyz)
    if not bool(np.all(lower < upper)):
        raise SceneFieldError("scene bounds must have positive XYZ extent")
    return lower, upper


def _validate_occupancy(occupancy_xyz: np.ndarray | Tensor) -> np.ndarray:
    if torch.is_tensor(occupancy_xyz):
        value = occupancy_xyz.detach().cpu().numpy()
    else:
        value = np.asarray(occupancy_xyz)
    if value.ndim != 3 or any(int(size) < 2 for size in value.shape):
        raise SceneFieldError("occupancy_xyz must be a three-D [X,Y,Z] grid")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise SceneFieldError("occupancy_xyz contains non-finite values")
    return np.ascontiguousarray(value.astype(bool, copy=False))


def _voxel_spacing(
    shape_xyz: Sequence[int], lower_xyz: np.ndarray, upper_xyz: np.ndarray
) -> np.ndarray:
    return (upper_xyz - lower_xyz) / np.asarray(shape_xyz, dtype=np.float64)


def _clear_floor(
    occupancy_xyz: np.ndarray,
    *,
    lower_xyz: np.ndarray,
    upper_xyz: np.ndarray,
    floor_ignore_height_m: float,
) -> np.ndarray:
    height = float(floor_ignore_height_m)
    if not math.isfinite(height) or height < float(lower_xyz[2]):
        raise SceneFieldError("floor_ignore_height_m is outside the scene")
    result = np.array(occupancy_xyz, dtype=bool, copy=True)
    spacing = _voxel_spacing(result.shape, lower_xyz, upper_xyz)
    centres_z = lower_xyz[2] + (np.arange(result.shape[2]) + 0.5) * spacing[2]
    result[:, :, centres_z < height] = False
    return result


def _point_to_voxel(
    point_world_zup: np.ndarray,
    *,
    shape_xyz: Sequence[int],
    lower_xyz: np.ndarray,
    upper_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spacing = _voxel_spacing(shape_xyz, lower_xyz, upper_xyz)
    raw = np.floor((point_world_zup - lower_xyz) / spacing).astype(np.int64)
    clipped = np.clip(raw, 0, np.asarray(shape_xyz, dtype=np.int64) - 1)
    return raw, clipped


def select_target_component(
    occupancy_xyz: np.ndarray | Tensor,
    target_point_world_zup: Sequence[float],
    *,
    lower_xyz: Sequence[float] = DEFAULT_LOWER_XYZ,
    upper_xyz: Sequence[float] = DEFAULT_UPPER_XYZ,
    floor_ignore_height_m: float = 0.08,
    maximum_seed_distance_m: float = 0.50,
    search_crop_radius_m: float | None = None,
    maximum_component_fraction: float = 1.0,
) -> TargetComponent:
    """Select a query-local six-connected component near a target point.

    The floor band is removed before connectivity, preventing a chair/table
    touching the rasterized floor from becoming one scene-sized component.  A
    finite ``search_crop_radius_m`` additionally clips *the connectivity
    search*, not the collision scene.  This matters for rasterized indoor
    scenes where furniture, skirting, or walls can remain connected even after
    floor removal.  Occupied voxels outside the crop always remain collision
    occupancy.

    ``maximum_component_fraction`` is a fail-closed guard against accidentally
    exempting most of a scene as the interaction target.  This is query-time
    geometry selection; it uses no pose or motion example.
    """

    occupancy = _validate_occupancy(occupancy_xyz)
    lower, upper = _validate_bounds(lower_xyz, upper_xyz)
    target = _finite_vector("target_point_world_zup", target_point_world_zup)
    maximum = float(maximum_seed_distance_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise SceneFieldError("maximum_seed_distance_m must be finite and positive")
    maximum_fraction = float(maximum_component_fraction)
    if (
        not math.isfinite(maximum_fraction)
        or maximum_fraction <= 0.0
        or maximum_fraction > 1.0
    ):
        raise SceneFieldError("maximum_component_fraction must be in (0, 1]")
    crop_radius: float | None
    if search_crop_radius_m is None:
        crop_radius = None
    else:
        crop_radius = float(search_crop_radius_m)
        if not math.isfinite(crop_radius) or crop_radius <= 0.0:
            raise SceneFieldError("search_crop_radius_m must be finite and positive")
        if crop_radius < maximum:
            raise SceneFieldError(
                "search_crop_radius_m must be at least maximum_seed_distance_m"
            )

    global_working = _clear_floor(
        occupancy,
        lower_xyz=lower,
        upper_xyz=upper,
        floor_ignore_height_m=floor_ignore_height_m,
    )
    if not bool(global_working.any()):
        raise SceneFieldError("scene has no occupied cells above the ignored floor")

    spacing = _voxel_spacing(global_working.shape, lower, upper)
    search_working = global_working
    if crop_radius is not None:
        centre_axes = tuple(
            lower[axis]
            + (np.arange(global_working.shape[axis], dtype=np.float64) + 0.5)
            * spacing[axis]
            for axis in range(3)
        )
        distance_sq = (
            np.square(centre_axes[0][:, None, None] - target[0])
            + np.square(centre_axes[1][None, :, None] - target[1])
            + np.square(centre_axes[2][None, None, :] - target[2])
        )
        search_working = global_working & (distance_sq <= crop_radius**2)
        if not bool(search_working.any()):
            raise SceneFieldError(
                "scene has no occupied target seed inside the local component crop"
            )

    _, seed = _point_to_voxel(
        target, shape_xyz=search_working.shape, lower_xyz=lower, upper_xyz=upper
    )
    seed_distance = 0.0
    if not bool(search_working[tuple(seed)]):
        occupied_indices = np.argwhere(search_working)
        occupied_centres = lower + (occupied_indices.astype(np.float64) + 0.5) * spacing
        distances_sq = np.square(occupied_centres - target[None]).sum(axis=-1)
        nearest = int(np.argmin(distances_sq))
        seed = occupied_indices[nearest]
        seed_distance = float(math.sqrt(float(distances_sq[nearest])))
    else:
        seed_centre = lower + (seed.astype(np.float64) + 0.5) * spacing
        seed_distance = float(np.linalg.norm(seed_centre - target))
    if seed_distance > maximum:
        raise SceneFieldError(
            f"nearest occupied target seed is {seed_distance:.3f}m away, "
            f"exceeding {maximum:.3f}m"
        )

    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - project environment has scipy
        raise RuntimeError("target component extraction requires scipy") from error
    labels, component_count = ndimage.label(
        search_working, structure=ndimage.generate_binary_structure(3, 1)
    )
    if int(component_count) < 1:
        raise SceneFieldError("connected-component extraction returned no component")
    label_id = int(labels[tuple(seed)])
    if label_id <= 0:
        raise SceneFieldError("selected target seed is not occupied")
    target_mask = labels == label_id
    scene_occupied_count = int(global_working.sum())
    target_fraction = float(target_mask.sum()) / float(scene_occupied_count)
    if target_fraction > maximum_fraction:
        raise SceneFieldError(
            "selected target component occupies {:.3%} of the non-floor scene, "
            "exceeding the {:.3%} safety limit".format(
                target_fraction, maximum_fraction
            )
        )
    collision = global_working & ~target_mask
    return TargetComponent(
        target_occupancy_xyz=np.ascontiguousarray(target_mask),
        collision_occupancy_xyz=np.ascontiguousarray(collision),
        seed_index_xyz=tuple(int(value) for value in seed),
        seed_distance_m=seed_distance,
        search_crop_radius_m=crop_radius,
        scene_occupied_voxel_count=scene_occupied_count,
        target_fraction_of_scene_occupied=target_fraction,
    )


def _signed_distance_from_occupancy(
    occupancy_xyz: np.ndarray,
    *,
    spacing_xyz: Sequence[float],
) -> np.ndarray:
    occupied = np.asarray(occupancy_xyz, dtype=bool)
    if not bool(occupied.any()):
        # No collision component is a valid result after target removal.  A
        # positive constant keeps every in-bounds point collision-free.
        extent = float(np.linalg.norm(np.asarray(occupied.shape) * np.asarray(spacing_xyz)))
        return np.full(occupied.shape, extent, dtype=np.float32)
    if bool(occupied.all()):
        raise SceneFieldError("collision occupancy is entirely occupied")
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("metric SDF construction requires scipy") from error
    outside = distance_transform_edt(~occupied, sampling=tuple(spacing_xyz))
    inside = distance_transform_edt(occupied, sampling=tuple(spacing_xyz))
    return np.ascontiguousarray((outside - inside).astype(np.float32))


def _occupancy_sha256(mask_xyz: np.ndarray | None) -> str | None:
    """Stable compact identity for an audited occupancy component."""

    if mask_xyz is None:
        return None
    value = np.ascontiguousarray(mask_xyz, dtype=bool)
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(value.reshape(-1), bitorder="little").tobytes())
    return digest.hexdigest()


class RawSceneSDFZUp:
    """Differentiable raw-scene SDF with optional target-component removal."""

    def __init__(
        self,
        sdf_xyz: np.ndarray | Tensor,
        *,
        lower_xyz: Sequence[float],
        upper_xyz: Sequence[float],
        target_occupancy_xyz: np.ndarray | None = None,
        collision_occupancy_xyz: np.ndarray | None = None,
        target_component_removed: bool = False,
        target_component_seed_index_xyz: Sequence[int] | None = None,
        target_component_seed_distance_m: float | None = None,
        target_component_search_crop_radius_m: float | None = None,
        scene_occupied_voxel_count: int | None = None,
        target_component_fraction_of_scene_occupied: float | None = None,
        source: str = "query_raw_scene",
    ) -> None:
        if torch.is_tensor(sdf_xyz):
            values = sdf_xyz.detach().cpu().numpy()
        else:
            values = np.asarray(sdf_xyz)
        if values.ndim != 3 or any(int(size) < 2 for size in values.shape):
            raise SceneFieldError("sdf_xyz must be [X,Y,Z]")
        if not np.isfinite(values).all():
            raise SceneFieldError("sdf_xyz contains non-finite values")
        lower, upper = _validate_bounds(lower_xyz, upper_xyz)
        self.sdf_xyz = torch.as_tensor(
            np.ascontiguousarray(values, dtype=np.float32)
        ).unsqueeze(0).unsqueeze(0)
        self._tensor_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {
            (self.sdf_xyz.device, self.sdf_xyz.dtype): self.sdf_xyz
        }
        self.lower_xyz = tuple(float(value) for value in lower)
        self.upper_xyz = tuple(float(value) for value in upper)
        self.spacing_xyz = tuple(
            float(value)
            for value in _voxel_spacing(values.shape, lower, upper)
        )
        self.source = str(source)
        self._target_component_removed = bool(target_component_removed)
        self.target_component_seed_index_xyz = (
            None
            if target_component_seed_index_xyz is None
            else tuple(int(value) for value in target_component_seed_index_xyz)
        )
        self.target_component_seed_distance_m = (
            None
            if target_component_seed_distance_m is None
            else float(target_component_seed_distance_m)
        )
        self.target_component_search_crop_radius_m = (
            None
            if target_component_search_crop_radius_m is None
            else float(target_component_search_crop_radius_m)
        )
        self.scene_occupied_voxel_count = (
            None
            if scene_occupied_voxel_count is None
            else int(scene_occupied_voxel_count)
        )
        self.target_component_fraction_of_scene_occupied = (
            None
            if target_component_fraction_of_scene_occupied is None
            else float(target_component_fraction_of_scene_occupied)
        )
        self.target_occupancy_xyz = (
            None
            if target_occupancy_xyz is None
            else np.ascontiguousarray(target_occupancy_xyz, dtype=bool)
        )
        self.collision_occupancy_xyz = (
            None
            if collision_occupancy_xyz is None
            else np.ascontiguousarray(collision_occupancy_xyz, dtype=bool)
        )
        for name, value in (
            ("target_occupancy_xyz", self.target_occupancy_xyz),
            ("collision_occupancy_xyz", self.collision_occupancy_xyz),
        ):
            if value is not None and value.shape != values.shape:
                raise SceneFieldError(f"{name} shape differs from SDF")

    @classmethod
    def from_occupancy(
        cls,
        occupancy_xyz: np.ndarray | Tensor,
        *,
        lower_xyz: Sequence[float] = DEFAULT_LOWER_XYZ,
        upper_xyz: Sequence[float] = DEFAULT_UPPER_XYZ,
        floor_ignore_height_m: float = 0.08,
        target_point_world_zup: Sequence[float] | None = None,
        target_occupancy_xyz: np.ndarray | Tensor | None = None,
        remove_target_component: bool = False,
        maximum_target_seed_distance_m: float = 0.50,
        target_component_search_crop_radius_m: float | None = None,
        maximum_target_component_fraction: float = 1.0,
        source: str = "query_raw_scene",
    ) -> "RawSceneSDFZUp":
        occupancy = _validate_occupancy(occupancy_xyz)
        lower, upper = _validate_bounds(lower_xyz, upper_xyz)
        working = _clear_floor(
            occupancy,
            lower_xyz=lower,
            upper_xyz=upper,
            floor_ignore_height_m=floor_ignore_height_m,
        )
        target_mask: np.ndarray | None = None
        selected_component: TargetComponent | None = None
        if target_occupancy_xyz is not None:
            declared_target = _validate_occupancy(target_occupancy_xyz)
            if declared_target.shape != occupancy.shape:
                raise SceneFieldError("target occupancy shape differs from raw scene")
            if bool((declared_target & ~occupancy).any()):
                raise SceneFieldError("target occupancy must be a raw-scene subset")
            target_mask = declared_target & working
            if not bool(target_mask.any()):
                raise SceneFieldError("target occupancy is empty after floor removal")
        elif target_point_world_zup is not None:
            selected_component = select_target_component(
                occupancy,
                target_point_world_zup,
                lower_xyz=lower,
                upper_xyz=upper,
                floor_ignore_height_m=floor_ignore_height_m,
                maximum_seed_distance_m=maximum_target_seed_distance_m,
                search_crop_radius_m=target_component_search_crop_radius_m,
                maximum_component_fraction=maximum_target_component_fraction,
            )
            target_mask = selected_component.target_occupancy_xyz

        scene_occupied_count = int(working.sum())
        target_fraction = (
            None
            if target_mask is None
            else float(target_mask.sum()) / float(scene_occupied_count)
        )
        if (
            target_mask is not None
            and target_fraction is not None
            and target_fraction > float(maximum_target_component_fraction)
        ):
            raise SceneFieldError(
                "target component exceeds maximum_target_component_fraction"
            )

        collision = working.copy()
        if remove_target_component and target_mask is not None:
            collision &= ~target_mask
        spacing = _voxel_spacing(collision.shape, lower, upper)
        sdf = _signed_distance_from_occupancy(collision, spacing_xyz=spacing)
        return cls(
            sdf,
            lower_xyz=lower,
            upper_xyz=upper,
            target_occupancy_xyz=target_mask,
            collision_occupancy_xyz=collision,
            target_component_removed=bool(
                remove_target_component and target_mask is not None
            ),
            target_component_seed_index_xyz=(
                None
                if selected_component is None
                else selected_component.seed_index_xyz
            ),
            target_component_seed_distance_m=(
                None
                if selected_component is None
                else selected_component.seed_distance_m
            ),
            target_component_search_crop_radius_m=(
                target_component_search_crop_radius_m
                if selected_component is None
                else selected_component.search_crop_radius_m
            ),
            scene_occupied_voxel_count=scene_occupied_count,
            target_component_fraction_of_scene_occupied=target_fraction,
            source=source,
        )

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.sdf_xyz.shape[-3:])

    @property
    def target_removed(self) -> bool:
        return self._target_component_removed

    @property
    def receipt(self) -> dict[str, object]:
        target_count = (
            None
            if self.target_occupancy_xyz is None
            else int(self.target_occupancy_xyz.sum())
        )
        collision_count = (
            None
            if self.collision_occupancy_xyz is None
            else int(self.collision_occupancy_xyz.sum())
        )
        return {
            "schema": "p360_raw_scene_sdf_zup_v1",
            "source": self.source,
            "shape_xyz": list(self.shape_xyz),
            "lower_xyz": list(self.lower_xyz),
            "upper_xyz": list(self.upper_xyz),
            "spacing_xyz": list(self.spacing_xyz),
            "coordinate_frame": "world_zup_xyz_m",
            "grid_storage_order": ["x", "y", "z"],
            "grid_sample_coordinate_order": ["z", "y", "x"],
            "align_corners": False,
            "positive_is_free": True,
            "target_component_removed": self.target_removed,
            "target_component_present": self.target_occupancy_xyz is not None,
            "target_component_voxel_count": target_count,
            "target_component_sha256": _occupancy_sha256(
                self.target_occupancy_xyz
            ),
            "target_component_seed_index_xyz": (
                None
                if self.target_component_seed_index_xyz is None
                else list(self.target_component_seed_index_xyz)
            ),
            "target_component_seed_distance_m": self.target_component_seed_distance_m,
            "target_component_search_crop_radius_m": (
                self.target_component_search_crop_radius_m
            ),
            "scene_occupied_voxel_count": self.scene_occupied_voxel_count,
            "target_component_fraction_of_scene_occupied": (
                self.target_component_fraction_of_scene_occupied
            ),
            "collision_occupancy_voxel_count": collision_count,
            "collision_occupancy_sha256": _occupancy_sha256(
                self.collision_occupancy_xyz
            ),
            "retrieval_used": False,
        }

    def clear_device_cache(self) -> None:
        """Drop device copies while retaining the authoritative CPU field."""

        self._tensor_cache = {
            (self.sdf_xyz.device, self.sdf_xyz.dtype): self.sdf_xyz
        }

    def _field_on(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (device, dtype)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = self.sdf_xyz.to(device=device, dtype=dtype)
            self._tensor_cache[key] = cached
        return cached

    def sample(self, points_world_zup: Tensor) -> SDFSamples:
        """Trilinearly sample finite ``[B,...,3]`` world Z-up points."""

        if (
            not torch.is_tensor(points_world_zup)
            or not points_world_zup.is_floating_point()
            or points_world_zup.ndim < 2
            or points_world_zup.shape[-1] != 3
            or not bool(torch.isfinite(points_world_zup).all())
        ):
            raise SceneFieldError("points_world_zup must be finite float [B,...,3]")
        batch = int(points_world_zup.shape[0])
        flat = points_world_zup.reshape(batch, -1, 3)
        lower = torch.as_tensor(
            self.lower_xyz, device=flat.device, dtype=flat.dtype
        )
        upper = torch.as_tensor(
            self.upper_xyz, device=flat.device, dtype=flat.dtype
        )
        normalized_xyz = 2.0 * (flat - lower) / (upper - lower) - 1.0
        grid = normalized_xyz[..., (2, 1, 0)].reshape(batch, 1, 1, -1, 3)
        field = self._field_on(device=flat.device, dtype=flat.dtype)
        if field.shape[0] == 1 and batch != 1:
            field = field.expand(batch, -1, -1, -1, -1)
        elif field.shape[0] != batch:
            raise SceneFieldError("SDF batch cannot broadcast to query batch")
        sampled = F.grid_sample(
            field,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0, 0, 0]
        in_bounds = ((flat >= lower) & (flat < upper)).all(dim=-1)

        # A smooth negative distance outside the cell bounds gives guidance a
        # direction back into the scene.  Include one voxel of conservative
        # margin so a point exactly on the excluded upper boundary is unsafe.
        below = F.relu(lower - flat)
        above = F.relu(flat - upper)
        outside_depth = torch.linalg.vector_norm(below + above, dim=-1)
        minimum_spacing = min(self.spacing_xyz)
        outside_sdf = -(outside_depth + float(minimum_spacing))
        sampled = torch.where(in_bounds, sampled, outside_sdf)
        output_shape = points_world_zup.shape[:-1]
        return SDFSamples(
            signed_distance_m=sampled.reshape(output_shape),
            in_bounds=in_bounds.reshape(output_shape),
        )

    def project_to_level_set(
        self,
        points_world_zup: Tensor,
        *,
        target_level_m: float = 0.01,
        tolerance_m: float = 0.003,
        maximum_iterations: int = 32,
        maximum_step_m: float = 0.08,
        maximum_displacement_m: float = 0.25,
    ) -> SDFLevelSetProjection:
        """Project explicit query points to ``SDF == target_level_m``.

        Each Newton step follows the trilinear raw-SDF gradient and is clipped
        in metric space.  Points are also constrained to remain inside the
        cell-centred scene bounds and within a fixed trust radius of their
        query-time starting point.  The method is deterministic, reads no
        motion/example memory, and reports non-convergence so callers can fail
        closed instead of silently accepting an unrelated contact point.
        """

        if (
            not torch.is_tensor(points_world_zup)
            or not points_world_zup.is_floating_point()
            or points_world_zup.ndim < 2
            or points_world_zup.shape[-1] != 3
            or not bool(torch.isfinite(points_world_zup).all())
        ):
            raise SceneFieldError(
                "level-set projection points must be finite float [...,3]"
            )
        target = float(target_level_m)
        tolerance = float(tolerance_m)
        step_limit = float(maximum_step_m)
        displacement_limit = float(maximum_displacement_m)
        if not math.isfinite(target) or target < 0.0:
            raise SceneFieldError("target_level_m must be finite and nonnegative")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise SceneFieldError("tolerance_m must be finite and positive")
        if int(maximum_iterations) < 1:
            raise SceneFieldError("maximum_iterations must be positive")
        if not math.isfinite(step_limit) or step_limit <= 0.0:
            raise SceneFieldError("maximum_step_m must be finite and positive")
        if not math.isfinite(displacement_limit) or displacement_limit <= 0.0:
            raise SceneFieldError(
                "maximum_displacement_m must be finite and positive"
            )

        initial = points_world_zup.detach().clone()
        current = initial.clone()
        lower = torch.as_tensor(
            self.lower_xyz, device=current.device, dtype=current.dtype
        )
        upper = torch.as_tensor(
            self.upper_xyz, device=current.device, dtype=current.dtype
        )
        half_spacing = 0.5 * torch.as_tensor(
            self.spacing_xyz, device=current.device, dtype=current.dtype
        )
        lower_centre = lower + half_spacing
        upper_centre = upper - half_spacing
        current = torch.maximum(torch.minimum(current, upper_centre), lower_centre)
        initial_samples = self.sample(current).signed_distance_m.detach()

        iterations = 0
        for iteration in range(int(maximum_iterations)):
            iterations = iteration + 1
            with torch.enable_grad():
                differentiable = current.detach().requires_grad_(True)
                signed_distance = self.sample(differentiable).signed_distance_m
                residual = signed_distance - target
                gradient = torch.autograd.grad(
                    signed_distance.sum(), differentiable, allow_unused=False
                )[0]
            gradient = torch.nan_to_num(
                gradient.detach(), nan=0.0, posinf=0.0, neginf=0.0
            )
            residual = torch.nan_to_num(
                residual.detach(), nan=0.0, posinf=0.0, neginf=0.0
            )
            active = residual.abs() > tolerance
            gradient_norm_sq = gradient.square().sum(dim=-1, keepdim=True)
            valid_gradient = gradient_norm_sq[..., 0] > 1.0e-10
            update_mask = active & valid_gradient
            if not bool(update_mask.any()):
                break
            step = residual.unsqueeze(-1) * gradient / gradient_norm_sq.clamp_min(
                1.0e-10
            )
            step_norm = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
            step = step * (step_limit / step_norm.clamp_min(1.0e-12)).clamp(max=1.0)
            step = torch.where(update_mask.unsqueeze(-1), step, torch.zeros_like(step))
            candidate = current - step
            displacement = candidate - initial
            displacement_norm = torch.linalg.vector_norm(
                displacement, dim=-1, keepdim=True
            )
            displacement = displacement * (
                displacement_limit / displacement_norm.clamp_min(1.0e-12)
            ).clamp(max=1.0)
            current = initial + displacement
            current = torch.maximum(
                torch.minimum(current, upper_centre), lower_centre
            ).detach()

        final_samples = self.sample(current).signed_distance_m.detach()
        in_bounds = self.sample(current).in_bounds.detach()
        converged = (final_samples.sub(target).abs() <= tolerance) & in_bounds
        displacement = torch.linalg.vector_norm(current - initial, dim=-1)
        return SDFLevelSetProjection(
            points_world_zup=current,
            initial_signed_distance_m=initial_samples,
            final_signed_distance_m=final_samples,
            displacement_m=displacement,
            converged=converged,
            iterations=iterations,
            target_level_m=target,
            tolerance_m=tolerance,
        )


def _surface_faces_zup(
    target_occupancy_xyz: np.ndarray,
    *,
    lower_xyz: Sequence[float],
    upper_xyz: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    occupancy = _validate_occupancy(target_occupancy_xyz)
    if not bool(occupancy.any()):
        raise SceneFieldError("target component is empty")
    lower, upper = _validate_bounds(lower_xyz, upper_xyz)
    spacing = _voxel_spacing(occupancy.shape, lower, upper)
    occupied_indices = np.argwhere(occupancy)
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for axis in range(3):
        for direction in (-1, 1):
            neighbour = occupied_indices.copy()
            neighbour[:, axis] += direction
            inside = (
                (neighbour >= 0)
                & (neighbour < np.asarray(occupancy.shape)[None])
            ).all(axis=-1)
            neighbour_occupied = np.zeros(len(neighbour), dtype=bool)
            valid = neighbour[inside]
            neighbour_occupied[inside] = occupancy[
                valid[:, 0], valid[:, 1], valid[:, 2]
            ]
            exposed_indices = occupied_indices[~neighbour_occupied]
            if not len(exposed_indices):
                continue
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = float(direction)
            centres = lower + (exposed_indices.astype(np.float64) + 0.5) * spacing
            face_points = centres + 0.5 * spacing[axis] * normal
            points.append(face_points)
            normals.append(np.broadcast_to(normal, face_points.shape).copy())
    if not points:
        raise SceneFieldError("target component has no exposed face")
    return (
        np.ascontiguousarray(np.concatenate(points), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(normals), dtype=np.float32),
    )


class QueryTimeICGF:
    """Analytic query-only contact affinity for explicit sparse joints.

    Support can be shared, ``[P,3]``, or joint-specific, ``[J,P,3]``.  This
    object stores geometry supplied by the current planner/scene only.
    """

    def __init__(
        self,
        support_points_world_zup: np.ndarray | Tensor,
        *,
        confidence: np.ndarray | Tensor | None = None,
        valid_support: np.ndarray | Tensor | None = None,
        kernel_sigma_m: float = 0.08,
        source: str = "query_target",
    ) -> None:
        points = torch.as_tensor(support_points_world_zup, dtype=torch.float32)
        if points.ndim not in (2, 3) or points.shape[-1] != 3 or points.shape[-2] < 1:
            raise SceneFieldError("ICGF support must be [P,3] or [J,P,3]")
        if not bool(torch.isfinite(points).all()):
            raise SceneFieldError("ICGF support contains non-finite points")
        support_shape = points.shape[:-1]
        if valid_support is None:
            valid = torch.ones(support_shape, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid_support, dtype=torch.bool)
            if valid.shape != support_shape:
                raise SceneFieldError("ICGF valid_support shape differs from support")
        if confidence is None:
            weights = valid.to(torch.float32)
        else:
            weights = torch.as_tensor(confidence, dtype=torch.float32)
            if weights.shape != support_shape:
                raise SceneFieldError("ICGF confidence shape differs from support")
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
                raise SceneFieldError("ICGF confidence must be finite and nonnegative")
            weights = weights * valid
        denominator = weights.sum(dim=-1, keepdim=True)
        group_is_declared = valid.any(dim=-1, keepdim=True)
        if bool((group_is_declared & (denominator <= 0.0)).any()):
            raise SceneFieldError("every declared ICGF group needs positive confidence")
        weights = torch.where(
            group_is_declared,
            weights / denominator.clamp_min(1.0e-30),
            torch.zeros_like(weights),
        )
        log_weights = torch.where(
            valid, torch.log(weights.clamp_min(1.0e-30)), torch.full_like(weights, -torch.inf)
        )
        sigma = float(kernel_sigma_m)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise SceneFieldError("kernel_sigma_m must be finite and positive")
        self.support_points_world_zup = points
        self.valid_support = valid
        self.log_weights = log_weights
        self.kernel_sigma_m = sigma
        self.source = str(source)

    @classmethod
    def from_target_point(
        cls,
        target_point_world_zup: Sequence[float],
        *,
        kernel_sigma_m: float = 0.08,
    ) -> "QueryTimeICGF":
        point = _finite_vector("target_point_world_zup", target_point_world_zup)
        return cls(
            point[None].astype(np.float32),
            kernel_sigma_m=kernel_sigma_m,
            source="planner_target_point",
        )

    @classmethod
    def from_sparse_joint_targets(
        cls,
        target_joint_xyz_world_zup: np.ndarray | Tensor,
        joint_mask: np.ndarray | Tensor,
        *,
        kernel_sigma_m: float = 0.08,
        source: str = "planner_sparse_joint_targets",
    ) -> "QueryTimeICGF":
        points = torch.as_tensor(target_joint_xyz_world_zup, dtype=torch.float32)
        mask = torch.as_tensor(joint_mask, dtype=torch.bool)
        if points.ndim != 2 or points.shape[-1] != 3 or mask.shape != points.shape[:1]:
            raise SceneFieldError("sparse targets must be [J,3] with joint_mask [J]")
        # Invalid joint rows remain padded; callers may query only masked IDs.
        return cls(
            points[:, None],
            valid_support=mask[:, None],
            confidence=mask[:, None].float(),
            kernel_sigma_m=kernel_sigma_m,
            source=source,
        )

    @classmethod
    def from_target_component(
        cls,
        target_occupancy_xyz: np.ndarray | Tensor,
        *,
        anchor_point_world_zup: Sequence[float],
        lower_xyz: Sequence[float] = DEFAULT_LOWER_XYZ,
        upper_xyz: Sequence[float] = DEFAULT_UPPER_XYZ,
        local_radius_m: float = 0.18,
        normal_offset_m: float = 0.0,
        maximum_points: int = 1024,
        kernel_sigma_m: float = 0.08,
    ) -> "QueryTimeICGF":
        anchor = _finite_vector("anchor_point_world_zup", anchor_point_world_zup)
        radius = float(local_radius_m)
        offset = float(normal_offset_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise SceneFieldError("local_radius_m must be finite and positive")
        if not math.isfinite(offset):
            raise SceneFieldError("normal_offset_m must be finite")
        if int(maximum_points) < 1:
            raise SceneFieldError("maximum_points must be positive")
        surface, normals = _surface_faces_zup(
            _validate_occupancy(target_occupancy_xyz),
            lower_xyz=lower_xyz,
            upper_xyz=upper_xyz,
        )
        distance = np.linalg.norm(surface.astype(np.float64) - anchor[None], axis=-1)
        minimum = float(distance.min())
        keep = distance <= minimum + radius
        ids = np.flatnonzero(keep)
        order = ids[np.lexsort((surface[ids, 2], surface[ids, 1], surface[ids, 0], distance[ids]))]
        if len(order) > int(maximum_points):
            order = order[: int(maximum_points)]
        support = surface[order] + normals[order] * offset
        confidence = np.exp(
            -0.5 * np.square((distance[order] - minimum) / max(radius, 1.0e-6))
        ).astype(np.float32)
        return cls(
            support,
            confidence=confidence,
            kernel_sigma_m=kernel_sigma_m,
            source="query_target_component_surface",
        )

    @property
    def joint_specific(self) -> bool:
        return self.support_points_world_zup.ndim == 3

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "schema": "p360_query_time_icgf_v1",
            "source": self.source,
            "joint_specific": self.joint_specific,
            "support_shape": list(self.support_points_world_zup.shape),
            "kernel_sigma_m": self.kernel_sigma_m,
            "coordinate_frame": "world_zup_xyz_m",
            "retrieval_used": False,
        }

    def soft_distance(
        self,
        query_world_zup: Tensor,
        *,
        joint_ids: Tensor | None = None,
    ) -> Tensor:
        """Return smooth metric distance for ``[B,T,N,3]`` queries."""

        if (
            not torch.is_tensor(query_world_zup)
            or not query_world_zup.is_floating_point()
            or query_world_zup.ndim != 4
            or query_world_zup.shape[-1] != 3
            or not bool(torch.isfinite(query_world_zup).all())
        ):
            raise SceneFieldError("ICGF queries must be finite float [B,T,N,3]")
        support = self.support_points_world_zup.to(
            device=query_world_zup.device, dtype=query_world_zup.dtype
        )
        log_weights = self.log_weights.to(
            device=query_world_zup.device, dtype=query_world_zup.dtype
        )
        if self.joint_specific:
            if (
                joint_ids is None
                or joint_ids.dtype != torch.long
                or joint_ids.ndim != 1
                or joint_ids.shape[0] != query_world_zup.shape[2]
                or joint_ids.device != query_world_zup.device
            ):
                raise SceneFieldError("joint-specific ICGF requires int64 joint_ids [N]")
            if bool((joint_ids < 0).any()) or bool((joint_ids >= support.shape[0]).any()):
                raise SceneFieldError("ICGF joint_ids are out of range")
            support = support.index_select(0, joint_ids)
            log_weights = log_weights.index_select(0, joint_ids)
            if bool(torch.isneginf(log_weights).all(dim=-1).any()):
                raise SceneFieldError("queried joint has no valid ICGF support")
            delta = query_world_zup.unsqueeze(-2) - support[None, None]
            scores = log_weights[None, None] - delta.square().sum(dim=-1) / (
                2.0 * self.kernel_sigma_m**2
            )
        else:
            delta = query_world_zup.unsqueeze(-2) - support
            scores = log_weights - delta.square().sum(dim=-1) / (
                2.0 * self.kernel_sigma_m**2
            )
        soft_squared = -2.0 * self.kernel_sigma_m**2 * torch.logsumexp(
            scores, dim=-1
        )
        return torch.sqrt(soft_squared.clamp_min(0.0) + 1.0e-12)


__all__ = [
    "DEFAULT_LOWER_XYZ",
    "DEFAULT_UPPER_XYZ",
    "QueryTimeICGF",
    "RawSceneSDFZUp",
    "SDFLevelSetProjection",
    "SDFSamples",
    "SceneFieldError",
    "TargetComponent",
    "select_target_component",
]

