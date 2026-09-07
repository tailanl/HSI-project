"""Actual hash-bound target/non-target distance transforms; no component guessing."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence, Protocol
from dataclasses import dataclass
import math
import hashlib
import json
import os
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from . import refine_contract as contract
from . import refine_physics as p478

@dataclass(frozen=True)
class BoundMasks:
    scene_world_xyz: np.ndarray
    target_world_xyz: np.ndarray
    collision_world_xyz: np.ndarray
    receipt: Mapping[str, Any]

def _clear_floor(
    occupancy_world_xyz: np.ndarray,
    floor_ignore_height_m: float,
) -> np.ndarray:
    value = np.array(occupancy_world_xyz, dtype=bool, copy=True, order="C")
    lower = np.asarray(contract.GRID_LOWER_XYZ, dtype=np.float64)
    upper = np.asarray(contract.GRID_UPPER_XYZ, dtype=np.float64)
    spacing = (upper - lower) / np.asarray(value.shape, dtype=np.float64)
    centres_z = lower[2] + (np.arange(value.shape[2], dtype=np.float64) + 0.5) * spacing[2]
    value[:, :, centres_z < float(floor_ignore_height_m)] = False
    return value

def load_bound_masks(
    occupancy_native_path: Path,
    target_mask_native_path: Path,
    *,
    expected_occupancy_file_sha256: str | None = None,
    expected_target_mask_file_sha256: str | None = None,
    expected_target_world_sha256: str | None = None,
    floor_ignore_height_m: float = 0.08,
    maximum_target_floor_fraction: float = 0.10,
    maximum_target_fraction: float = 0.25,
) -> BoundMasks:
    occupancy_path = Path(occupancy_native_path).resolve(strict=True)
    target_path = Path(target_mask_native_path).resolve(strict=True)
    occupancy_file_sha = contract.sha256_file(occupancy_path)
    target_file_sha = contract.sha256_file(target_path)
    if expected_occupancy_file_sha256 is not None:
        contract.require(
            occupancy_file_sha == expected_occupancy_file_sha256,
            "bound SDF scene occupancy file hash drift",
        )
    if expected_target_mask_file_sha256 is not None:
        contract.require(
            target_file_sha == expected_target_mask_file_sha256,
            "bound SDF target mask file hash drift",
        )

    occupancy_native = np.load(occupancy_path, allow_pickle=False, mmap_mode="r")
    target_native = np.load(target_path, allow_pickle=False, mmap_mode="r")
    contract.require(
        occupancy_native.shape == contract.NATIVE_OCCUPANCY_SHAPE
        and occupancy_native.dtype == np.bool_,
        "bound SDF scene occupancy native contract drift",
    )
    contract.require(
        target_native.shape == contract.NATIVE_OCCUPANCY_SHAPE
        and target_native.dtype == np.bool_,
        "bound SDF target mask native contract drift",
    )
    contract.require(bool(target_native.flags.c_contiguous), "target occupancy mask must be C-order")
    contract.require(bool(target_native.any()), "bound SDF target mask is empty")
    contract.require(
        not bool(np.logical_and(target_native, np.logical_not(occupancy_native)).any()),
        "bound SDF target mask is not a subset of current scene occupancy",
    )

    scene_world = contract.native_to_world(occupancy_native)
    target_world = contract.native_to_world(target_native)
    target_world_hash = contract.occupancy_content_sha256(target_world)
    if expected_target_world_sha256 is not None:
        contract.require(
            target_world_hash == expected_target_world_sha256,
            "bound SDF target mask world-content hash drift",
        )
    floor_height = float(floor_ignore_height_m)
    contract.require(math.isfinite(floor_height) and 0.0 <= floor_height <= 0.20, "invalid floor ignore height")
    maximum_fraction = float(maximum_target_fraction)
    maximum_floor_fraction = float(maximum_target_floor_fraction)
    contract.require(
        math.isfinite(maximum_fraction) and 0.0 < maximum_fraction <= 0.50,
        "invalid maximum target occupancy fraction",
    )
    contract.require(
        math.isfinite(maximum_floor_fraction)
        and 0.0 <= maximum_floor_fraction <= 0.25,
        "invalid maximum target floor-height voxel fraction",
    )
    scene_nonfloor = _clear_floor(scene_world, floor_height)
    contract.require(bool(scene_nonfloor.any()), "scene has no non-floor occupancy")
    target_nonfloor = _clear_floor(target_world, floor_height)
    contract.require(
        bool(target_nonfloor.any()),
        "target occupancy mask contains no non-floor object voxels",
    )
    target_floor_count = int(target_world.sum() - target_nonfloor.sum())
    target_floor_fraction = target_floor_count / float(target_world.sum())
    contract.require(
        target_floor_fraction <= maximum_floor_fraction,
        "target occupancy mask captures too many floor-height voxels",
    )
    fraction = float(target_nonfloor.sum()) / float(scene_nonfloor.sum())
    contract.require(
        fraction <= maximum_fraction,
        "target occupancy mask exempts too much of the current scene",
    )
    collision_world = np.ascontiguousarray(scene_nonfloor & ~target_world)
    receipt = {
        "selection": "explicit_current_stage1_target_occupancy_mask",
        "identity_claim": "hash_bound_stage1_instance_mask",
        "scene_occupancy_file_sha256": occupancy_file_sha,
        "target_mask_file_sha256": target_file_sha,
        "target_component_sha256": target_world_hash,
        "target_component_voxel_count": int(target_world.sum()),
        "target_nonfloor_voxel_count": int(target_nonfloor.sum()),
        "target_floor_height_voxel_count": target_floor_count,
        "target_floor_height_voxel_fraction": target_floor_fraction,
        "collision_occupancy_sha256": contract.occupancy_content_sha256(collision_world),
        "collision_occupancy_voxel_count": int(collision_world.sum()),
        "scene_nonfloor_occupied_voxel_count": int(scene_nonfloor.sum()),
        "target_fraction_of_scene_nonfloor_occupied": fraction,
        "floor_ignore_height_m": floor_height,
        "maximum_target_floor_height_voxel_fraction": maximum_floor_fraction,
        "grid_shape_world_xyz": list(contract.WORLD_OCCUPANCY_SHAPE),
        "grid_bounds_world_zup_m": [list(contract.GRID_LOWER_XYZ), list(contract.GRID_UPPER_XYZ)],
        "spacing_m": [0.02, 0.02, 0.02],
        "positive_is_free": True,
        "query_local_component_guess_used": False,
    }
    return BoundMasks(
        scene_world_xyz=scene_nonfloor,
        target_world_xyz=target_world,
        collision_world_xyz=collision_world,
        receipt=receipt,
    )

def _signed_distance(mask: np.ndarray) -> np.ndarray:
    occupied = np.asarray(mask, dtype=bool)
    contract.require(occupied.shape == contract.WORLD_OCCUPANCY_SHAPE, "SDF mask shape drift")
    if not bool(occupied.any()):
        extent = float(np.linalg.norm(np.asarray(occupied.shape, dtype=np.float64) * 0.02))
        return np.full(occupied.shape, extent, dtype=np.float32)
    contract.require(not bool(occupied.all()), "SDF mask is entirely occupied")
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:  # pragma: no cover
        raise contract.CurrentOnlyContractError("SciPy is required for bound target SDF") from error
    outside = distance_transform_edt(~occupied, sampling=(0.02, 0.02, 0.02))
    inside = distance_transform_edt(occupied, sampling=(0.02, 0.02, 0.02))
    return np.ascontiguousarray((outside - inside).astype(np.float32))

def build_fields_from_bound_mask(
    occupancy_native_path: Path,
    target_mask_native_path: Path,
    device: torch.device,
    *,
    expected_occupancy_file_sha256: str | None = None,
    expected_target_mask_file_sha256: str | None = None,
    expected_target_world_sha256: str | None = None,
    floor_ignore_height_m: float = 0.08,
    maximum_target_floor_fraction: float = 0.10,
    maximum_target_fraction: float = 0.25,
) -> p478.TargetAwareFields:
    """Construct P478-compatible fields without geometric component guessing."""

    masks = load_bound_masks(
        occupancy_native_path,
        target_mask_native_path,
        expected_occupancy_file_sha256=expected_occupancy_file_sha256,
        expected_target_mask_file_sha256=expected_target_mask_file_sha256,
        expected_target_world_sha256=expected_target_world_sha256,
        floor_ignore_height_m=floor_ignore_height_m,
        maximum_target_floor_fraction=maximum_target_floor_fraction,
        maximum_target_fraction=maximum_target_fraction,
    )
    target_sdf = _signed_distance(masks.target_world_xyz)
    collision_sdf = _signed_distance(masks.collision_world_xyz)
    lower = torch.tensor(contract.GRID_LOWER_XYZ, dtype=torch.float32, device=device)
    upper = torch.tensor(contract.GRID_UPPER_XYZ, dtype=torch.float32, device=device)
    return p478.TargetAwareFields(
        target=p478.WorldGridSDF(torch.as_tensor(target_sdf, device=device), lower, upper),
        non_target=p478.WorldGridSDF(torch.as_tensor(collision_sdf, device=device), lower, upper),
        target_mask=masks.target_world_xyz,
        collision_mask=masks.collision_world_xyz,
        component_receipt=masks.receipt,
    )
