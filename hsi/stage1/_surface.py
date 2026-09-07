"""Build one fresh, graph-target-bound sitting surface for any LINGO scene.

This producer consumes only the current P515 metric graph, its hash-bound SAM
atomic-instance receipt, the original scene OBJ, a LINGO occupancy grid and the
current query start.  It never reads a historical surface, route, keypose,
motion, memory, or future frame.  For relation-free ambiguous targets it makes
the otherwise missing decision from current-state occupancy reachability and
path cost, while retaining a complete candidate audit.
"""

from __future__ import annotations

import argparse

from dataclasses import dataclass

from datetime import datetime, timezone

import hashlib

import json

import math

import os

from pathlib import Path

from collections import deque

from typing import Any, Mapping, Sequence

import numpy as np

from scipy.ndimage import binary_dilation, distance_transform_edt, label

P523_SCHEMA = "p523.current_only_stage1_target_surface.v1"

P509_COMPAT_SCHEMA = "p508.lingo_scene_first_stage1.v1"

SAM_SCHEMA = "p515.lingo_sam31_atomic_multiview_instances.v1"

GRAPH_SCHEMA = "p515.lingo_metric_scene_graph.v1"

SURFACE_ARCHIVE_SCHEMA = "p523.target_bound_support_faces.v1"

SITTABLE_CLASSES = {"chair", "armchair", "stool", "sofa", "bench"}

FORBIDDEN_RESULT_PREFIXES = ("p373", "p480", "p517", "p520", "p522")

class SurfaceError(RuntimeError):
    pass

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise SurfaceError(message)

def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }

def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)

def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value

def verify_payload_hash(value: Mapping[str, Any], label_name: str) -> str:
    recorded = value.get("receipt_payload_sha256")
    require(isinstance(recorded, str) and len(recorded) == 64, f"{label_name} lacks payload hash")
    payload = dict(value)
    payload.pop("receipt_payload_sha256", None)
    require(canonical_hash(payload) == recorded, f"{label_name} payload hash drift")
    return recorded

def reject_historical_result(path: Path, label_name: str) -> None:
    lowered = [part.lower() for part in path.resolve().parts]
    offenders = [
        part for part in lowered
        if any(part == prefix or part.startswith(prefix + "_") for prefix in FORBIDDEN_RESULT_PREFIXES)
    ]
    require(not offenders, f"{label_name} points at a forbidden historical result: {offenders[0] if offenders else ''}")

def int64_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    array = np.asarray(np.unique(array), dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()

def bool_content_sha256(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(mask, dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(value.reshape(-1), bitorder="little").tobytes())
    return digest.hexdigest()

def finite_vector(value: Any, size: int, label_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (size,), f"{label_name} must have shape ({size},)")
    require(np.isfinite(array).all(), f"{label_name} contains non-finite values")
    return array

def world_occupancy(native: np.ndarray) -> np.ndarray:
    value = np.asarray(native)
    require(value.ndim == 3 and value.dtype == np.bool_, "LINGO occupancy must be Boolean 3-D")
    return np.ascontiguousarray(value.transpose(0, 2, 1)[:, ::-1, :])

def native_occupancy(world: np.ndarray) -> np.ndarray:
    value = np.asarray(world, dtype=np.bool_)
    require(value.ndim == 3, "world occupancy must be 3-D")
    return np.ascontiguousarray(value[:, ::-1, :].transpose(0, 2, 1))

def load_obj_world_zup(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read OBJ vertices/faces and apply LINGO [x,y-up,z] -> [x,-z,y-up]."""

    vertices_native: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if raw.startswith("v "):
                fields = raw.split()
                require(len(fields) >= 4, f"malformed OBJ vertex at line {line_number}")
                vertices_native.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif raw.startswith("f "):
                tokens = raw.split()[1:]
                require(len(tokens) >= 3, f"malformed OBJ face at line {line_number}")
                indices: list[int] = []
                for token in tokens:
                    index = int(token.split("/", 1)[0])
                    require(index != 0, f"OBJ uses invalid zero vertex index at line {line_number}")
                    if index < 0:
                        index = len(vertices_native) + index
                    else:
                        index -= 1
                    indices.append(index)
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    native = np.asarray(vertices_native, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    require(native.ndim == 2 and native.shape[1] == 3 and len(native) >= 3, "OBJ has no usable vertices")
    require(triangles.ndim == 2 and triangles.shape[1] == 3 and len(triangles) >= 1, "OBJ has no usable faces")
    require(int(triangles.min()) >= 0 and int(triangles.max()) < len(native), "OBJ face index is out of range")
    world = np.column_stack((native[:, 0], -native[:, 2], native[:, 1]))
    require(np.isfinite(world).all(), "OBJ contains non-finite coordinates")
    return np.ascontiguousarray(world), np.ascontiguousarray(triangles)

def connected_face_components(faces: np.ndarray, face_ids: np.ndarray) -> list[np.ndarray]:
    count = len(face_ids)
    parents = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        while int(parents[index]) != index:
            parents[index] = parents[int(parents[index])]
            index = int(parents[index])
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    owner: dict[int, int] = {}
    for local_index, face_id in enumerate(face_ids.tolist()):
        for vertex_id in faces[int(face_id)].tolist():
            previous = owner.setdefault(int(vertex_id), local_index)
            union(local_index, previous)
    groups: dict[int, list[int]] = {}
    for local_index, face_id in enumerate(face_ids.tolist()):
        groups.setdefault(find(local_index), []).append(int(face_id))
    return [np.asarray(values, dtype=np.int64) for _, values in sorted(groups.items())]

@dataclass(frozen=True)
class SurfaceConfig:
    grid_lower_world_xyz_m: tuple[float, float, float] = (-3.0, -4.0, 0.0)
    voxel_size_m: float = 0.02
    body_height_range_m: tuple[float, float] = (0.08, 1.75)
    navigation_clearance_m: float = 0.28
    approach_clearance_min_m: float = 0.19
    approach_offset_from_edge_m: float = 0.38
    maximum_start_snap_m: float = 1.0
    maximum_goal_snap_m: float = 1.0
    target_interaction_relaxation_radius_m: float = 0.12
    interaction_terminal_min_clearance_m: float = 0.25
    upward_normal_z_min: float = 0.62
    seat_height_min_m: float = 0.18
    seat_height_max_m: float = 0.75
    height_bin_m: float = 0.05
    height_band_half_width_m: float = 0.075
    minimum_surface_area_m2: float = 0.012
    minimum_surface_extent_m: float = 0.14
    minimum_direct_backrest_seat_area_before_topology_completion_m2: float = 0.08
    target_mask_bounds_padding_m: float = 0.06
    target_mask_seed_dilation_voxels: int = 2

    def validate(self) -> None:
        require(self.voxel_size_m > 0.0, "voxel size must be positive")
        require(0.0 < self.upward_normal_z_min <= 1.0, "normal threshold is invalid")
        require(0.0 < self.seat_height_min_m < self.seat_height_max_m, "seat height range is invalid")
        require(
            self.minimum_direct_backrest_seat_area_before_topology_completion_m2
            >= self.minimum_surface_area_m2,
            "direct-seat topology-completion threshold is below the publication minimum",
        )
        require(self.navigation_clearance_m >= self.approach_clearance_min_m > 0.0, "clearances are invalid")
        require(self.maximum_start_snap_m > 0.0 and self.maximum_goal_snap_m > 0.0, "snap radii are invalid")
        require(self.target_interaction_relaxation_radius_m > 0.0, "target relaxation radius is invalid")
        require(
            self.approach_clearance_min_m
            <= self.interaction_terminal_min_clearance_m
            <= self.navigation_clearance_m,
            "interaction terminal clearance is invalid",
        )

@dataclass
class ExtractedSurface:
    target_instance_id: str
    target_class: str
    original_class: str
    target_support_ids: np.ndarray
    target_bounds: np.ndarray
    face_ids: np.ndarray
    vertex_ids: np.ndarray
    surface: dict[str, Any]
    surface_sha256: str
    surface_id: str
    approaches: list[dict[str, Any]]
    route_prefilter: dict[str, Any]
    extraction_audit: dict[str, Any]
    semantic_override: dict[str, Any]

def build_navigation_fields(world: np.ndarray, config: SurfaceConfig) -> dict[str, np.ndarray]:
    z = config.grid_lower_world_xyz_m[2] + (np.arange(world.shape[2]) + 0.5) * config.voxel_size_m
    body = (z >= config.body_height_range_m[0]) & (z <= config.body_height_range_m[1])
    require(bool(body.any()), "occupancy has no body-height layers")
    obstacles = world[:, :, body].any(axis=2)
    clearance = distance_transform_edt(~obstacles) * config.voxel_size_m
    free = (~obstacles) & (clearance >= config.navigation_clearance_m - 1e-12)
    require(bool(free.any()), "occupancy has no human-clear navigation cells")
    return {"obstacles": obstacles, "clearance": clearance, "free": free}

def point_to_cell(point_xy: np.ndarray, shape: tuple[int, int], config: SurfaceConfig) -> tuple[int, int] | None:
    lower = np.asarray(config.grid_lower_world_xyz_m[:2], dtype=np.float64)
    cell = np.floor((point_xy - lower) / config.voxel_size_m).astype(np.int64)
    if not (0 <= int(cell[0]) < shape[0] and 0 <= int(cell[1]) < shape[1]):
        return None
    return int(cell[0]), int(cell[1])

def cell_center(cell: tuple[int, int], config: SurfaceConfig) -> np.ndarray:
    return np.asarray(config.grid_lower_world_xyz_m[:2]) + (np.asarray(cell) + 0.5) * config.voxel_size_m

def nearest_true(mask: np.ndarray, point: np.ndarray, radius_m: float, config: SurfaceConfig) -> tuple[tuple[int, int], float] | None:
    raw = point_to_cell(point, mask.shape, config)
    if raw is None:
        return None
    radius = int(math.ceil(radius_m / config.voxel_size_m)) + 1
    x0, x1 = max(0, raw[0] - radius), min(mask.shape[0], raw[0] + radius + 1)
    y0, y1 = max(0, raw[1] - radius), min(mask.shape[1], raw[1] + radius + 1)
    local = np.argwhere(mask[x0:x1, y0:y1])
    if not len(local):
        return None
    cells = local + np.asarray((x0, y0), dtype=np.int64)
    centres = np.asarray([cell_center((int(row[0]), int(row[1])), config) for row in cells])
    distances = np.linalg.norm(centres - point[None, :], axis=1)
    valid = np.flatnonzero(distances <= radius_m + 1e-12)
    if not len(valid):
        return None
    index = min(valid.tolist(), key=lambda i: (float(distances[i]), int(cells[i, 0]), int(cells[i, 1])))
    return (int(cells[index, 0]), int(cells[index, 1])), float(distances[index])

def flood_distances(free: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    distance = np.full(free.shape, -1, dtype=np.int32)
    distance[start] = 0
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        x, y = queue.popleft()
        step = int(distance[x, y]) + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < free.shape[0] and 0 <= ny < free.shape[1] and free[nx, ny] and distance[nx, ny] < 0:
                distance[nx, ny] = step
                queue.append((nx, ny))
    return distance

def extract_support_surface(
    scene_id: str,
    target_instance_id: str,
    target_class: str,
    support_ids: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_sha256: str,
    graph_payload_sha256: str,
    config: SurfaceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str, str, dict[str, Any]]:
    require(target_class in SITTABLE_CLASSES, f"unsupported sitting target class: {target_class}")
    membership = np.zeros(len(vertices), dtype=np.bool_)
    membership[support_ids] = True
    target_points = vertices[support_ids]
    target_bounds = np.stack((target_points.min(axis=0), target_points.max(axis=0)))
    target_face_membership = membership[faces].sum(axis=1)
    strict_bound_ids = np.flatnonzero(target_face_membership == 3)
    # A sparse multi-view consensus can omit one vertex of a triangle.  The
    # fallback still refuses geometry outside the exact SAM owner AABB, which
    # keeps the published surface inside the owner bounds used by Stage3.
    face_points = vertices[faces]
    inside_exact_owner = np.all(
        (face_points >= target_bounds[0][None, None, :] - 1e-9)
        & (face_points <= target_bounds[1][None, None, :] + 1e-9),
        axis=(1, 2),
    )
    relaxed_bound_ids = np.flatnonzero((target_face_membership >= 2) & inside_exact_owner)
    bound_ids = strict_bound_ids if len(strict_bound_ids) >= 2 else relaxed_bound_ids
    membership_rule = (
        "all_three_vertices_are_in_current_SAM_target_support"
        if len(strict_bound_ids) >= 2
        else "at_least_two_vertices_in_support_and_all_three_inside_exact_SAM_owner_AABB"
    )
    require(len(bound_ids) >= 2, f"{target_instance_id} has too few target-bound mesh faces")
    triangles = vertices[faces[bound_ids]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid_area = double_area > 1e-10
    normals = np.zeros_like(cross)
    normals[valid_area] = cross[valid_area] / double_area[valid_area, None]
    centroids = triangles.mean(axis=1)
    z_low = max(config.seat_height_min_m, float(target_bounds[0, 2]) + 0.05)
    z_high = min(config.seat_height_max_m, float(target_bounds[1, 2]) - 0.02)
    require(z_low < z_high, f"{target_instance_id} has no plausible sitting-height interval")
    horizontal = (
        valid_area
        & (normals[:, 2] >= config.upward_normal_z_min)
        & (centroids[:, 2] >= z_low)
        & (centroids[:, 2] <= z_high)
    )
    horizontal_ids = bound_ids[horizontal]
    require(len(horizontal_ids) >= 2, f"{target_instance_id} has no upward sitting faces")
    horizontal_centres = centroids[horizontal]
    horizontal_areas = double_area[horizontal] * 0.5
    bins = np.floor((horizontal_centres[:, 2] - z_low) / config.height_bin_m).astype(np.int64)
    bin_areas: dict[int, float] = {}
    for bin_id, area in zip(bins.tolist(), horizontal_areas.tolist()):
        bin_areas[int(bin_id)] = bin_areas.get(int(bin_id), 0.0) + float(area)
    winning_bin = min(bin_areas, key=lambda key: (-bin_areas[key], key))
    winning = bins == winning_bin
    peak_z = float(np.average(horizontal_centres[winning, 2], weights=horizontal_areas[winning]))
    band_ids = horizontal_ids[np.abs(horizontal_centres[:, 2] - peak_z) <= config.height_band_half_width_m]
    components = connected_face_components(faces, band_ids)
    audits: list[dict[str, Any]] = []
    valid_components: list[tuple[tuple[float, float, int], np.ndarray]] = []
    for component in components:
        tri = vertices[faces[component]]
        component_cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        component_areas = np.linalg.norm(component_cross, axis=1) * 0.5
        area = float(component_areas.sum())
        unique_vertices = np.unique(faces[component])
        bounds = np.stack((vertices[unique_vertices].min(axis=0), vertices[unique_vertices].max(axis=0)))
        extents = bounds[1] - bounds[0]
        passed = (
            area >= config.minimum_surface_area_m2
            and float(extents[0]) >= config.minimum_surface_extent_m
            and float(extents[1]) >= config.minimum_surface_extent_m
        )
        audits.append({
            "face_count": int(len(component)),
            "area_m2": area,
            "bounds_world_zup_m": bounds.astype(float).tolist(),
            "horizontal_extent_m": extents[:2].astype(float).tolist(),
            "gate_pass": bool(passed),
        })
        if passed:
            valid_components.append(((-area, float(np.mean(tri[:, :, 2])), int(component.min())), component))
    primary_component_audits = list(audits)
    primary_valid_components = list(valid_components)
    primary_largest_valid_area_m2 = max(
        (-float(row[0][0]) for row in primary_valid_components), default=None,
    )
    topology_recovery_used = False
    topology_recovery_audit: dict[str, Any] | None = None
    topology_recovery_reasons: list[str] = []
    if not primary_valid_components:
        topology_recovery_reasons.append("no_size_valid_direct_component")
    elif (
        target_class in {"sofa", "chair", "armchair"}
        and primary_largest_valid_area_m2 is not None
        and primary_largest_valid_area_m2
        < config.minimum_direct_backrest_seat_area_before_topology_completion_m2
    ):
        # A direct all-three-SAM surface may technically pass the very small
        # publication minimum while still being only a sparse patch of the
        # real seat.  Backrest-bearing furniture gives us an independent
        # identity cue, so allow the already guarded owner-AABB topology
        # completion to compete in this narrow case as well.
        topology_recovery_reasons.append("direct_backrest_seat_component_is_sparse")
    if topology_recovery_reasons:
        # Multi-view SAM consensus can correctly identify thousands of owner
        # vertices while leaving holes between them. Requiring every vertex of
        # every triangle to be selected then shatters a real seat into tiny
        # islands. Recover only complete mesh components which:
        #   1. remain inside the *exact* current-SAM owner AABB,
        #   2. stay in the height band established by current SAM seed faces,
        #   3. directly contain enough current SAM vertices and two-seed faces.
        # This is topology completion from current evidence, not an AABB-only
        # semantic guess; unrelated tables/cabinets inside the box fail (3).
        recovery_bound_ids = np.flatnonzero(inside_exact_owner)
        recovery_triangles = vertices[faces[recovery_bound_ids]]
        recovery_cross = np.cross(
            recovery_triangles[:, 1] - recovery_triangles[:, 0],
            recovery_triangles[:, 2] - recovery_triangles[:, 0],
        )
        recovery_double_area = np.linalg.norm(recovery_cross, axis=1)
        recovery_normals = recovery_cross / np.maximum(
            recovery_double_area[:, None], 1e-12,
        )
        recovery_centres = recovery_triangles.mean(axis=1)
        recovery_horizontal = (
            (recovery_double_area > 1e-10)
            & (recovery_normals[:, 2] >= config.upward_normal_z_min)
            & (recovery_centres[:, 2] >= z_low)
            & (recovery_centres[:, 2] <= z_high)
            & (np.abs(recovery_centres[:, 2] - peak_z) <= config.height_band_half_width_m)
        )
        recovery_band_ids = recovery_bound_ids[recovery_horizontal]
        recovery_components = connected_face_components(faces, recovery_band_ids)
        recovery_audits: list[dict[str, Any]] = []
        recovery_valid: list[tuple[tuple[float, float, int], np.ndarray]] = []
        for component in recovery_components:
            tri = vertices[faces[component]]
            component_cross = np.cross(
                tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0],
            )
            component_areas = np.linalg.norm(component_cross, axis=1) * 0.5
            area = float(component_areas.sum())
            unique_vertices = np.unique(faces[component])
            bounds = np.stack((
                vertices[unique_vertices].min(axis=0),
                vertices[unique_vertices].max(axis=0),
            ))
            extents = bounds[1] - bounds[0]
            direct_support_vertex_count = int(membership[unique_vertices].sum())
            direct_support_vertex_fraction = (
                float(direct_support_vertex_count) / float(len(unique_vertices))
            )
            component_support_counts = membership[faces[component]].sum(axis=1)
            direct_two_seed_face_count = int((component_support_counts >= 2).sum())
            surface_height = float(np.average(
                tri.mean(axis=1)[:, 2], weights=component_areas,
            ))
            local_high = target_points[
                target_points[:, 2] >= surface_height + 0.16
            ]
            if len(local_high):
                local_high = local_high[np.all(
                    (local_high[:, :2] >= bounds[0, :2][None, :] - 0.30)
                    & (local_high[:, :2] <= bounds[1, :2][None, :] + 0.30),
                    axis=1,
                )]
            component_centre_xy = np.average(
                tri.mean(axis=1)[:, :2], axis=0, weights=component_areas,
            )
            local_backrest_offset_m = (
                float(np.linalg.norm(local_high[:, :2].mean(axis=0) - component_centre_xy))
                if len(local_high)
                else None
            )
            local_backrest_required = target_class in {"sofa", "chair", "armchair"}
            local_backrest_pass = (
                not local_backrest_required
                or (
                    len(local_high) >= 8
                    and local_backrest_offset_m is not None
                    and 0.06 <= local_backrest_offset_m <= 0.80
                )
            )
            geometry_pass = (
                area >= config.minimum_surface_area_m2
                and float(extents[0]) >= config.minimum_surface_extent_m
                and float(extents[1]) >= config.minimum_surface_extent_m
            )
            identity_pass = (
                direct_support_vertex_count >= 8
                and direct_support_vertex_fraction >= 0.005
                and direct_two_seed_face_count >= 2
                and local_backrest_pass
            )
            passed = geometry_pass and identity_pass
            recovery_audits.append({
                "face_count": int(len(component)),
                "area_m2": area,
                "bounds_world_zup_m": bounds.astype(float).tolist(),
                "horizontal_extent_m": extents[:2].astype(float).tolist(),
                "geometry_gate_pass": bool(geometry_pass),
                "direct_current_sam_support_vertex_count": direct_support_vertex_count,
                "direct_current_sam_support_vertex_fraction": direct_support_vertex_fraction,
                "direct_current_sam_two_seed_face_count": direct_two_seed_face_count,
                "local_backrest_gate_required": local_backrest_required,
                "local_high_support_vertex_count": int(len(local_high)),
                "local_high_support_xy_padding_m": 0.30,
                "local_high_support_z_offset_min_m": 0.16,
                "local_backrest_offset_m": local_backrest_offset_m,
                "local_backrest_offset_range_m": [0.06, 0.80],
                "local_backrest_gate_pass": bool(local_backrest_pass),
                "identity_thresholds": {
                    "support_vertex_count_min": 8,
                    "support_vertex_fraction_min": 0.005,
                    "two_seed_face_count_min": 2,
                },
                "identity_gate_pass": bool(identity_pass),
                "gate_pass": bool(passed),
            })
            if passed:
                recovery_valid.append((
                    (-area, float(np.mean(tri[:, :, 2])), int(component.min())),
                    component,
                ))
        if recovery_valid:
            audits = recovery_audits
            topology_recovery_used = True
            membership_rule = (
                "current_SAM_seed_height_band_exact_owner_AABB_"
                "topology_completion_with_direct_seed_identity_gate"
            )
            valid_components = recovery_valid
        elif not primary_valid_components:
            audits = recovery_audits
        topology_recovery_audit = {
            "attempted": True,
            "used": topology_recovery_used,
            "trigger_reasons": topology_recovery_reasons,
            "primary_largest_valid_area_m2": primary_largest_valid_area_m2,
            "direct_backrest_seat_area_threshold_m2": (
                config.minimum_direct_backrest_seat_area_before_topology_completion_m2
            ),
            "policy": (
                "exact_current_SAM_owner_AABB_and_seed_height_band_with_"
                "direct_support_vertex_identity"
            ),
            "candidate_face_count": int(len(recovery_band_ids)),
            "component_audits": recovery_audits,
        }
    require(
        valid_components,
        f"{target_instance_id} has no size-valid sitting surface component; "
        f"dominant_height_m={peak_z:.6f}; component_audits="
        f"{json.dumps(audits, ensure_ascii=False, sort_keys=True, allow_nan=False)}",
    )
    selected_faces = min(valid_components, key=lambda row: row[0])[1]
    selected_faces = np.asarray(np.sort(selected_faces), dtype=np.int64)
    selected_vertices = np.asarray(np.unique(faces[selected_faces]), dtype=np.int64)
    tri = vertices[faces[selected_faces]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    areas = double_area * 0.5
    normals = cross / np.maximum(double_area[:, None], 1e-12)
    centre = np.average(tri.mean(axis=1), axis=0, weights=areas)
    surface_bounds = np.stack((vertices[selected_vertices].min(axis=0), vertices[selected_vertices].max(axis=0)))
    normal = np.average(normals, axis=0, weights=areas)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    face_ids_sha256 = int64_sha256(selected_faces)
    vertex_ids_sha256 = int64_sha256(selected_vertices)
    surface_sha256 = canonical_hash({
        "schema": "p523.target_bound_surface_identity.v1",
        "scene_id": scene_id,
        "target_instance_id": target_instance_id,
        "target_class": target_class,
        "mesh_sha256": mesh_sha256,
        "graph_payload_sha256": graph_payload_sha256,
        "face_ids_little_endian_int64_sha256": face_ids_sha256,
        "vertex_ids_little_endian_int64_sha256": vertex_ids_sha256,
    })
    surface_id = "INTERACTION_SURFACE_" + surface_sha256[:16].upper()
    surface = {
        "centre_world_xyz_zup_m": centre.astype(float).tolist(),
        "centroid_world_zup_m": centre.astype(float).tolist(),
        "bounds_world_zup_m": surface_bounds.astype(float).tolist(),
        "horizontal_extent_m": (surface_bounds[1, :2] - surface_bounds[0, :2]).astype(float).tolist(),
        "normal_world_zup": normal.astype(float).tolist(),
        "area_m2": float(areas.sum()),
        "surface_area_m2": float(areas.sum()),
        "face_count": int(len(selected_faces)),
        "unique_vertex_count": int(len(selected_vertices)),
    }
    audit = {
        "target_face_membership_rule": membership_rule,
        "strict_all_three_support_face_count": int(len(strict_bound_ids)),
        "relaxed_inside_exact_owner_face_count": int(len(relaxed_bound_ids)),
        "target_bound_face_count": int(len(bound_ids)),
        "upward_sitting_face_count": int(len(horizontal_ids)),
        "height_interval_world_zup_m": [z_low, z_high],
        "dominant_height_bin": int(winning_bin),
        "dominant_height_m": peak_z,
        "topology_recovery_used": topology_recovery_used,
        "topology_recovery_audit": topology_recovery_audit,
        "primary_connected_component_audits": primary_component_audits,
        "connected_component_audits": audits,
        "selected_face_ids_little_endian_int64_sha256": face_ids_sha256,
        "selected_vertex_ids_little_endian_int64_sha256": vertex_ids_sha256,
    }
    return selected_faces, selected_vertices, target_bounds, surface, surface_sha256, surface_id, audit

def infer_front_direction(target_points: np.ndarray, surface_vertices: np.ndarray, surface_z: float) -> tuple[np.ndarray, dict[str, Any]]:
    centre_xy = surface_vertices[:, :2].mean(axis=0)
    high = target_points[target_points[:, 2] >= surface_z + 0.16]
    if len(high) >= 3:
        back = high[:, :2].mean(axis=0) - centre_xy
        norm = float(np.linalg.norm(back))
        if norm >= 0.06:
            front = -back / norm
            return front, {
                "method": "opposite_high_target_support_backrest_centroid",
                "high_support_vertex_count": int(len(high)),
                "backrest_offset_m": norm,
            }
    covariance = np.cov(surface_vertices[:, :2].T)
    values, vectors = np.linalg.eigh(covariance)
    minor = vectors[:, int(np.argmin(values))]
    if minor[0] < 0.0 or (abs(float(minor[0])) < 1e-12 and minor[1] < 0.0):
        minor = -minor
    return minor / max(float(np.linalg.norm(minor)), 1e-12), {
        "method": "surface_minor_axis_two_sided_no_backrest_evidence",
        "high_support_vertex_count": int(len(high)),
        "backrest_offset_m": None,
    }

def approach_options(
    surface: Mapping[str, Any],
    target_points: np.ndarray,
    surface_points: np.ndarray,
    navigation: Mapping[str, np.ndarray],
    config: SurfaceConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    centre = np.asarray(surface["centre_world_xyz_zup_m"], dtype=np.float64)
    front, front_audit = infer_front_direction(target_points, surface_points, float(centre[2]))
    base_angle = math.atan2(float(front[1]), float(front[0]))
    if front_audit["method"].startswith("opposite_"):
        # The semantic terminal body still faces the inferred front, but a
        # cluttered interaction object can require reaching that terminal
        # from a diagonal.  Keep a front-facing half-circle of candidates so
        # Stage1 can approach a chair beside a drum kit without inventing a
        # route through the kit.  No back-side (> 90 degree) candidate is
        # published.
        angle_offsets = (
            0.0,
            -math.pi / 12, math.pi / 12,
            -math.pi / 6, math.pi / 6,
            -math.pi / 4, math.pi / 4,
            -math.pi / 3, math.pi / 3,
            -5.0 * math.pi / 12, 5.0 * math.pi / 12,
            -math.pi / 2, math.pi / 2,
        )
    else:
        angle_offsets = (0.0, math.pi, -math.pi / 8, math.pi / 8, math.pi - math.pi / 8, math.pi + math.pi / 8)
    obstacles = np.asarray(navigation["obstacles"], dtype=np.bool_)
    clearance = np.asarray(navigation["clearance"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    seen_cells: set[tuple[int, int]] = set()
    centred_surface = surface_points[:, :2] - centre[None, :2]
    for angle_offset in angle_offsets:
        angle = math.atan2(math.sin(base_angle + angle_offset), math.cos(base_angle + angle_offset))
        direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
        edge_radius = float(np.max(centred_surface @ direction))
        radius = max(0.25, edge_radius) + config.approach_offset_from_edge_m
        point = centre[:2] + radius * direction
        cell = point_to_cell(point, obstacles.shape, config)
        if cell is None or cell in seen_cells or bool(obstacles[cell]):
            continue
        cell_clearance = float(clearance[cell])
        if cell_clearance + 1e-12 < config.approach_clearance_min_m:
            continue
        seen_cells.add(cell)
        rows.append({
            "option_id": "",
            "approach_world_xy_m": point.astype(float).tolist(),
            "contact_forward_yaw_rad": angle,
            "approach_direction_from_support_rad": angle,
            "front_orientation_difference_rad": abs(math.atan2(math.sin(angle - base_angle), math.cos(angle - base_angle))),
            "radius_from_support_m": radius,
            "approach_to_support_edge_min_clearance_m": config.approach_offset_from_edge_m,
            "raw_occupancy_clearance_m": cell_clearance,
            "requires_fresh_route_planning": True,
            "source_route_copied_to_final_route": False,
            "route_planning_status": "fresh_route_required",
        })
    require(rows, "no raw-free approach option exists around the selected support surface")
    rows.sort(key=lambda row: (float(row["front_orientation_difference_rad"]), -float(row["raw_occupancy_clearance_m"]), float(row["contact_forward_yaw_rad"])))
    for index, row in enumerate(rows[:8]):
        row["option_id"] = f"APPROACH_OPTION_{index:02d}"
        row["source_option_index"] = index
    return rows[:8], {**front_audit, "front_direction_world_xy": front.astype(float).tolist(), "front_yaw_rad": base_angle}

def route_prefilter(
    approaches: Sequence[Mapping[str, Any]],
    start_xy: np.ndarray,
    navigation: Mapping[str, np.ndarray],
    config: SurfaceConfig,
) -> dict[str, Any]:
    free = np.asarray(navigation["free"], dtype=np.bool_)
    start = nearest_true(free, start_xy, config.maximum_start_snap_m, config)
    require(start is not None, "current query start has no human-clear snap")
    start_cell, start_snap = start
    obstacles = np.asarray(navigation["obstacles"], dtype=np.bool_)
    clearance = np.asarray(navigation["clearance"], dtype=np.float64)
    nx, ny = free.shape
    lower = np.asarray(config.grid_lower_world_xyz_m[:2], dtype=np.float64)
    x = lower[0] + (np.arange(nx, dtype=np.float64) + 0.5) * config.voxel_size_m
    y = lower[1] + (np.arange(ny, dtype=np.float64) + 0.5) * config.voxel_size_m
    xx, yy = np.meshgrid(x, y, indexing="ij")
    option_audits: list[dict[str, Any]] = []
    for option in approaches:
        point = np.asarray(option["approach_world_xy_m"], dtype=np.float64)
        raw_cell = point_to_cell(point, free.shape, config)
        require(raw_cell is not None, f"{option['option_id']} lies outside current occupancy")
        interaction_zone = (
            (xx - point[0]) ** 2 + (yy - point[1]) ** 2
            <= config.target_interaction_relaxation_radius_m ** 2 + 1e-12
        )
        relaxed = (
            interaction_zone
            & (~obstacles)
            & (~free)
            & (clearance >= config.interaction_terminal_min_clearance_m - 1e-12)
        )
        option_free = free | relaxed
        distances = flood_distances(option_free, start_cell)
        reachable = distances >= 0
        goal = nearest_true(reachable, point, config.maximum_goal_snap_m, config)
        if goal is None:
            option_audits.append({
                "option_id": option["option_id"],
                "reachable": False,
                "requested_cell": list(raw_cell),
                "requested_cell_clearance_m": float(clearance[raw_cell]),
                "terminal_clearance_gate_passed": False,
                "target_interaction_relaxation_used": bool(relaxed.any()),
            })
            continue
        cell, snap = goal
        grid_cost = float(distances[cell]) * config.voxel_size_m
        terminal_clearance_gate_passed = (
            float(clearance[raw_cell])
            >= config.interaction_terminal_min_clearance_m - 1e-9
            and float(clearance[cell])
            >= config.interaction_terminal_min_clearance_m - 1e-9
        )
        option_audits.append({
            "option_id": option["option_id"],
            "reachable": True,
            "requested_cell": list(raw_cell),
            "requested_cell_clearance_m": float(clearance[raw_cell]),
            "snapped_goal_cell": list(cell),
            "snapped_goal_clearance_m": float(clearance[cell]),
            "snapped_goal_uses_target_interaction_relaxation": bool(relaxed[cell]),
            "terminal_snap_distance_m": snap,
            "fresh_four_connected_path_cost_m": grid_cost,
            "ranking_path_cost_m": grid_cost + snap,
            "terminal_clearance_gate_passed": bool(terminal_clearance_gate_passed),
            "target_interaction_relaxation_used": bool(relaxed.any()),
        })
    raw_reachable = [row for row in option_audits if row["reachable"]]
    passing = [
        row for row in raw_reachable
        if row.get("terminal_clearance_gate_passed") is True
    ]
    return {
        "method": "fresh_current_occupancy_four_connected_prefilter_before_polygon_navmesh",
        "current_query_start_world_xy_m": start_xy.astype(float).tolist(),
        "snapped_start_cell": list(start_cell),
        "start_snap_distance_m": start_snap,
        "option_audits": option_audits,
        "raw_reachable_approach_count": len(raw_reachable),
        "reachable_approach_count": len(passing),
        "terminal_eligible_approach_count": len(passing),
        "minimum_path_cost_m": min((float(row["ranking_path_cost_m"]) for row in passing), default=None),
        "target_interaction_relaxation_radius_m": config.target_interaction_relaxation_radius_m,
        "interaction_terminal_min_clearance_m": config.interaction_terminal_min_clearance_m,
        "future_endpoint_or_motion_used": False,
    }

def target_mask_from_support(
    world_scene: np.ndarray,
    support_points: np.ndarray,
    target_bounds: np.ndarray,
    config: SurfaceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    lower = np.asarray(config.grid_lower_world_xyz_m, dtype=np.float64)
    shape = np.asarray(world_scene.shape, dtype=np.int64)
    raw_cells = np.floor((support_points - lower[None, :]) / config.voxel_size_m).astype(np.int64)
    in_bounds = np.all((raw_cells >= 0) & (raw_cells < shape[None, :]), axis=1)
    cells = np.unique(raw_cells[in_bounds], axis=0)
    require(len(cells) >= 1, "target support has no vertex inside the occupancy grid")
    padding = config.target_mask_bounds_padding_m
    bounds = target_bounds.copy()
    bounds[0] -= padding
    bounds[1] += padding
    bounds[0, 2] = max(bounds[0, 2], config.grid_lower_world_xyz_m[2] + 2.0 * config.voxel_size_m)
    lo = np.floor((bounds[0] - lower) / config.voxel_size_m).astype(np.int64)
    hi = np.ceil((bounds[1] - lower) / config.voxel_size_m).astype(np.int64) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape)
    require(np.all(hi > lo), "target support bounds do not overlap occupancy")
    slices = tuple(slice(int(lo[axis]), int(hi[axis])) for axis in range(3))
    local_scene = np.asarray(world_scene[slices], dtype=np.bool_)
    seed = np.zeros_like(local_scene, dtype=np.bool_)
    local_cells = cells - lo[None, :]
    local_valid = np.all((local_cells >= 0) & (local_cells < np.asarray(local_scene.shape)[None, :]), axis=1)
    local_cells = local_cells[local_valid]
    seed[tuple(local_cells.T)] = True
    dilated = binary_dilation(seed, iterations=config.target_mask_seed_dilation_voxels)
    labels, component_count = label(local_scene, structure=np.ones((3, 3, 3), dtype=np.uint8))
    touched = np.unique(labels[dilated & (labels > 0)])
    local_target = np.isin(labels, touched) if len(touched) else (local_scene & dilated)
    if not bool(local_target.any()):
        local_target = local_scene & binary_dilation(seed, iterations=max(4, config.target_mask_seed_dilation_voxels))
    require(bool(local_target.any()), "target occupancy mask is empty")
    world_target = np.zeros_like(world_scene, dtype=np.bool_)
    world_target[slices] = local_target
    require(not np.any(world_target & ~world_scene), "target occupancy mask is not a scene-occupancy subset")
    audit = {
        "method": "AABB_local_26_connected_occupancy_components_touched_by_dilated_current_SAM_support",
        "support_vertex_cells_in_grid": int(len(cells)),
        "local_component_count": int(component_count),
        "selected_component_labels": [int(value) for value in touched.tolist()],
        "world_voxel_count": int(np.count_nonzero(world_target)),
        "world_bounds_cell_min_inclusive": lo.astype(int).tolist(),
        "world_bounds_cell_max_exclusive": hi.astype(int).tolist(),
    }
    return world_target, audit

def load_and_validate_inputs(
    graph_path: Path,
    sam_path: Path,
    mesh_path: Path,
    occupancy_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    for path, name in ((graph_path, "metric graph"), (sam_path, "SAM receipt"), (mesh_path, "scene mesh"), (occupancy_path, "scene occupancy")):
        reject_historical_result(path, name)
    graph = read_json(graph_path)
    sam = read_json(sam_path)
    require(graph.get("schema") == GRAPH_SCHEMA, "unsupported metric graph schema")
    require(graph.get("status") in {"target_verified", "target_not_verified", "ambiguous_target"}, "unsupported metric graph status")
    graph_payload_sha256 = verify_payload_hash(graph, "metric graph")
    require(sam.get("schema") == SAM_SCHEMA, "unsupported SAM schema")
    require(sam.get("status") == "atomic_instances_ready", "SAM instances are not ready")
    verify_payload_hash(sam, "SAM receipt")
    require(graph.get("scene_id") == sam.get("scene_id"), "graph/SAM scene drift")
    require(graph.get("instruction") == sam.get("instruction"), "graph/SAM instruction drift")
    sam_binding = graph.get("inputs", {}).get("sam")
    require(isinstance(sam_binding, Mapping), "metric graph lacks SAM binding")
    require(Path(str(sam_binding.get("path"))).resolve(strict=True) == sam_path.resolve(strict=True), "metric graph points to a different SAM receipt")
    require(sam_binding.get("sha256") == sha256_file(sam_path), "metric graph SAM hash drift")
    mesh_binding = sam.get("source_original_scene_mesh")
    require(isinstance(mesh_binding, Mapping), "SAM receipt lacks original scene mesh")
    require(Path(str(mesh_binding.get("path"))).resolve(strict=True) == mesh_path.resolve(strict=True), "SAM receipt points to a different mesh")
    require(mesh_binding.get("sha256") == sha256_file(mesh_path), "SAM mesh hash drift")
    return graph, sam, graph_payload_sha256

def candidate_target_specs(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    query = graph.get("instruction_query")
    require(isinstance(query, Mapping), "metric graph lacks instruction query")
    require(query.get("action") == "sit", "P523 Stage1 currently supports sit only")
    requested_class = str(query.get("target_class", ""))
    require(requested_class in SITTABLE_CLASSES, "sit query lacks a supported seating target class")
    selection = graph.get("target_selection")
    require(isinstance(selection, Mapping), "metric graph lacks target selection")
    nodes = {
        str(row.get("node_id")): row
        for row in graph.get("nodes", [])
        if isinstance(row, Mapping)
    }
    if graph.get("status") == "target_verified":
        target_id = graph.get("selected_target_instance_id")
        require(isinstance(target_id, str) and target_id, "verified graph lacks selected target")
        node = nodes.get(target_id)
        require(isinstance(node, Mapping), "verified graph target node is absent")
        return [{
            "target_instance_id": target_id,
            "original_class": str(node.get("object_class")),
            "resolved_class": str(node.get("object_class")),
            "semantic_override_required": False,
            "graph_failure": None,
        }], "p515_unique_metric_graph_target"
    failures = set(selection.get("failures", []))
    require(query.get("relation") is None, "P523 only resolves relation-free target ambiguity")
    if failures == {"target_not_unique"}:
        rows = selection.get("candidates")
        require(isinstance(rows, list), "ambiguous metric graph lacks candidates")
        target_ids = sorted({str(row.get("instance_id")) for row in rows if isinstance(row, Mapping) and row.get("relation_gate_pass") is True})
        require(len(target_ids) >= 2, "target_not_unique graph lacks multiple passing candidates")
        specs: list[dict[str, Any]] = []
        for target_id in target_ids:
            node = nodes.get(target_id)
            require(isinstance(node, Mapping), f"metric graph lacks target node {target_id}")
            specs.append({
                "target_instance_id": target_id,
                "original_class": str(node.get("object_class")),
                "resolved_class": str(node.get("object_class")),
                "semantic_override_required": False,
                "graph_failure": "target_not_unique",
            })
        return specs, "p523_direct_target_current_start_fresh_occupancy_cost"

    require(
        failures in (
            {"target_missing"},
            {"target_classification_contradiction"},
            {"classification_contradiction"},
        ),
        f"metric graph failure is not safely resolvable: {sorted(failures)}",
    )
    alias_sources = {
        "sofa": {"chair", "armchair", "bench"},
        "chair": {"sofa", "bench"},
        "bench": {"sofa", "chair", "armchair"},
    }
    allowed_originals = alias_sources.get(requested_class, set())
    require(allowed_originals, f"no safe seating ontology alias is defined for {requested_class}")
    specs = []
    graph_failure = next(iter(failures))
    for target_id, node in sorted(nodes.items()):
        original_class = str(node.get("object_class", ""))
        # These gates intentionally exclude TV/cabinet/uncertain and parts.
        if original_class not in SITTABLE_CLASSES or original_class not in allowed_originals:
            continue
        if node.get("is_furniture") is not True or node.get("is_sittable") is not True:
            continue
        validation_failures = node.get("validation_failures", [])
        if not isinstance(validation_failures, list):
            continue
        expected_class_geometry_failure = f"class_geometry_xy_extent_contradiction:{original_class}"
        class_geometry_contradiction_only_exception = (
            node.get("valid_for_metric_graph") is False
            and validation_failures == [expected_class_geometry_failure]
        )
        # P515 deliberately rejects a chair whose dimensions contradict the
        # chair prior.  That exact, single rejection reason is the evidence
        # P523 needs to test a wider seating alias.  No other graph failure is
        # bypassed here: low confidence, part leakage, and arbitrary invalid
        # furniture remain ineligible.
        if node.get("valid_for_metric_graph") is not True and not class_geometry_contradiction_only_exception:
            continue
        if str(node.get("part_role", "")) != "whole_object":
            continue
        specs.append({
            "target_instance_id": target_id,
            "original_class": original_class,
            "resolved_class": requested_class,
            "semantic_override_required": True,
            "graph_failure": graph_failure,
            "semantic_confidence": node.get("semantic_confidence"),
            "source_valid_for_metric_graph": node.get("valid_for_metric_graph") is True,
            "source_validation_failures": list(validation_failures),
            "class_geometry_contradiction_only_exception": class_geometry_contradiction_only_exception,
        })
    require(specs, "no valid whole-object Qwen seating instance is eligible for ontology correction")
    return specs, "p523_direct_target_seating_ontology_shape_gate_then_current_occupancy_cost"

def seating_alias_shape_evidence(
    requested_class: str,
    original_class: str,
    target_bounds: np.ndarray,
    surface_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    owner_extent = np.asarray(target_bounds[1] - target_bounds[0], dtype=np.float64)
    horizontal = np.sort(owner_extent[:2])[::-1]
    surface_extent = np.sort(
        np.asarray(surface_geometry["horizontal_extent_m"], dtype=np.float64)
    )[::-1]
    owner_major, owner_minor = float(horizontal[0]), float(horizontal[1])
    surface_major, surface_minor = float(surface_extent[0]), float(surface_extent[1])
    owner_area = owner_major * owner_minor
    surface_area = float(surface_geometry["area_m2"])
    owner_aspect = owner_major / max(owner_minor, 1e-9)
    surface_aspect = surface_major / max(surface_minor, 1e-9)
    evidence = {
        "owner_horizontal_major_extent_m": owner_major,
        "owner_horizontal_minor_extent_m": owner_minor,
        "owner_horizontal_footprint_aabb_area_m2": owner_area,
        "owner_height_m": float(owner_extent[2]),
        "owner_horizontal_aspect_ratio": owner_aspect,
        "surface_horizontal_major_extent_m": surface_major,
        "surface_horizontal_minor_extent_m": surface_minor,
        "surface_area_m2": surface_area,
        "surface_horizontal_aspect_ratio": surface_aspect,
    }
    if requested_class == "sofa" and original_class in {"chair", "armchair", "bench"}:
        profile = "large_sofa_from_chair_or_bench_label"
        thresholds = {
            "owner_major_extent_min_m": 1.25,
            "owner_minor_extent_min_m": 0.55,
            "owner_footprint_aabb_area_min_m2": 0.90,
            "owner_height_range_m": [0.35, 1.65],
            "surface_major_extent_min_m": 0.35,
            "surface_area_min_m2": 0.06,
        }
        gates = {
            "owner_major_extent": owner_major >= thresholds["owner_major_extent_min_m"],
            "owner_minor_extent": owner_minor >= thresholds["owner_minor_extent_min_m"],
            "owner_footprint_area": owner_area >= thresholds["owner_footprint_aabb_area_min_m2"],
            "owner_height": thresholds["owner_height_range_m"][0] <= owner_extent[2] <= thresholds["owner_height_range_m"][1],
            "surface_major_extent": surface_major >= thresholds["surface_major_extent_min_m"],
            "surface_area": surface_area >= thresholds["surface_area_min_m2"],
        }
    elif requested_class == "chair" and original_class in {"sofa", "bench"}:
        profile = "compact_chair_from_sofa_or_bench_label"
        thresholds = {
            "owner_major_extent_range_m": [0.35, 1.25],
            "owner_minor_extent_range_m": [0.30, 1.15],
            "owner_footprint_aabb_area_max_m2": 1.30,
            "surface_major_extent_range_m": [0.20, 0.95],
            "surface_area_range_m2": [0.015, 0.65],
        }
        gates = {
            "owner_major_extent": thresholds["owner_major_extent_range_m"][0] <= owner_major <= thresholds["owner_major_extent_range_m"][1],
            "owner_minor_extent": thresholds["owner_minor_extent_range_m"][0] <= owner_minor <= thresholds["owner_minor_extent_range_m"][1],
            "owner_footprint_area": owner_area <= thresholds["owner_footprint_aabb_area_max_m2"],
            "surface_major_extent": thresholds["surface_major_extent_range_m"][0] <= surface_major <= thresholds["surface_major_extent_range_m"][1],
            "surface_area": thresholds["surface_area_range_m2"][0] <= surface_area <= thresholds["surface_area_range_m2"][1],
        }
    elif requested_class == "bench" and original_class in {"sofa", "chair", "armchair"}:
        profile = "elongated_bench_from_sofa_or_chair_label"
        thresholds = {
            "owner_major_extent_min_m": 1.00,
            "owner_horizontal_aspect_min": 1.45,
            "surface_major_extent_min_m": 0.65,
            "surface_horizontal_aspect_min": 1.35,
            "surface_area_min_m2": 0.06,
        }
        gates = {
            "owner_major_extent": owner_major >= thresholds["owner_major_extent_min_m"],
            "owner_horizontal_aspect": owner_aspect >= thresholds["owner_horizontal_aspect_min"],
            "surface_major_extent": surface_major >= thresholds["surface_major_extent_min_m"],
            "surface_horizontal_aspect": surface_aspect >= thresholds["surface_horizontal_aspect_min"],
            "surface_area": surface_area >= thresholds["surface_area_min_m2"],
        }
    else:
        raise SurfaceError(f"unsafe seating ontology alias: {original_class} -> {requested_class}")
    return {
        "profile": profile,
        "original_class": original_class,
        "requested_class": requested_class,
        "shape_evidence": evidence,
        "thresholds": thresholds,
        "gates": {key: bool(value) for key, value in gates.items()},
        "gate_pass": bool(all(gates.values())),
    }

def build(
    graph_path: Path,
    sam_path: Path,
    mesh_path: Path,
    occupancy_path: Path,
    start_world_xy_m: Sequence[float],
    output_dir: Path,
    config: SurfaceConfig = SurfaceConfig(),
    *, context=None,
) -> dict[str, Any]:
    config.validate()
    graph_path = graph_path.resolve(strict=True)
    sam_path = sam_path.resolve(strict=True)
    mesh_path = mesh_path.resolve(strict=True)
    occupancy_path = occupancy_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    require(not output_dir.exists() or not any(output_dir.iterdir()), f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph, sam, graph_payload_sha256 = (context.load_and_validate_inputs if context is not None else load_and_validate_inputs)(graph_path, sam_path, mesh_path, occupancy_path)
    scene_id = str(graph["scene_id"])
    instruction = str(graph["instruction"])
    query = dict(graph["instruction_query"])
    target_class_query = str(query.get("target_class"))
    require(target_class_query in SITTABLE_CLASSES or target_class_query == "chair", "query target is not sittable")
    start_xy = finite_vector(start_world_xy_m, 2, "current query start")
    vertices, faces = load_obj_world_zup(mesh_path)
    native_scene = np.load(occupancy_path, allow_pickle=False)
    world_scene = world_occupancy(native_scene)
    navigation = (context.build_navigation_fields if context is not None else build_navigation_fields)(world_scene, config)
    target_specs, selection_rule = candidate_target_specs(graph)
    nodes = {str(row.get("node_id")): row for row in graph.get("nodes", []) if isinstance(row, Mapping)}
    instances = {str(row.get("instance_id")): row for row in sam.get("instances", []) if isinstance(row, Mapping)}
    support_record = sam.get("support_vertices")
    require(isinstance(support_record, Mapping), "SAM receipt lacks support archive")
    support_path = Path(str(support_record.get("path"))).resolve(strict=True)
    reject_historical_result(support_path, "SAM support archive")
    require(support_record.get("sha256") == sha256_file(support_path), "SAM support archive hash drift")
    mesh_sha256 = sha256_file(mesh_path)
    extracted: list[ExtractedSurface] = []
    resolution_audits: list[dict[str, Any]] = []
    with np.load(support_path, allow_pickle=False) as archive:
        for target_spec in target_specs:
            target_id = str(target_spec["target_instance_id"])
            shape_evidence: dict[str, Any] | None = None
            try:
                node = nodes.get(target_id)
                instance = instances.get(target_id)
                require(isinstance(node, Mapping), f"metric graph lacks atomic target node {target_id}")
                require(isinstance(instance, Mapping), f"SAM lacks target instance {target_id}")
                original_class = str(node.get("object_class"))
                target_class = str(target_spec["resolved_class"])
                require(original_class == target_spec["original_class"], f"target node class drift for {target_id}")
                require(original_class in SITTABLE_CLASSES, f"target node is not sittable: {original_class}")
                require(target_class in SITTABLE_CLASSES, f"resolved target class is not sittable: {target_class}")
                key = str(instance.get("support_vertices_file_key", target_id))
                require(key in archive.files, f"SAM support archive lacks {key}")
                support_ids = np.asarray(archive[key], dtype=np.int64).reshape(-1)
                support_ids = np.unique(support_ids)
                require(len(support_ids) >= 3, f"{target_id} has too few support vertices")
                require(int(support_ids.min()) >= 0 and int(support_ids.max()) < len(vertices), f"{target_id} support index is out of mesh range")
                expected_count = int(instance.get("support_vertex_count", len(support_ids)))
                require(len(support_ids) == expected_count, f"{target_id} support count drift")
                selected_faces, selected_vertices, target_bounds, surface, surface_sha256, surface_id, extraction_audit = (context.extract_support_surface if context is not None else extract_support_surface)(
                    scene_id, target_id, target_class, support_ids, vertices, faces,
                    mesh_sha256, graph_payload_sha256, config,
                )
                if target_spec["semantic_override_required"]:
                    shape_evidence = seating_alias_shape_evidence(
                        target_class, original_class, target_bounds, surface,
                    )
                    require(
                        shape_evidence["gate_pass"] is True,
                        f"{target_id} failed seating ontology shape gate",
                    )
                    semantic_override = {
                        "enabled": True,
                        "policy": "P523_direct_target_seating_ontology_geometry_guard_v1",
                        "graph_failure": target_spec["graph_failure"],
                        "target_instance_id": target_id,
                        "original_class": original_class,
                        "requested_class": target_class,
                        "source_node_is_furniture": node.get("is_furniture") is True,
                        "source_node_is_sittable": node.get("is_sittable") is True,
                        "source_node_valid_for_metric_graph": target_spec["source_valid_for_metric_graph"],
                        "source_node_validation_failures": target_spec["source_validation_failures"],
                        "class_geometry_contradiction_only_exception": target_spec["class_geometry_contradiction_only_exception"],
                        "graph_validity_bypass_scope": (
                            "only_class_geometry_xy_extent_contradiction_for_same_original_sittable_class"
                            if target_spec["class_geometry_contradiction_only_exception"]
                            else None
                        ),
                        "source_node_part_role": node.get("part_role"),
                        "source_semantic_confidence": node.get("semantic_confidence"),
                        "shape_validation": shape_evidence,
                        "forbidden_source_classes": ["tv", "cabinet", "uncertain"],
                    }
                else:
                    semantic_override = {
                        "enabled": False,
                        "policy": "not_applied_graph_target_class_retained",
                        "graph_failure": target_spec["graph_failure"],
                        "target_instance_id": target_id,
                        "original_class": original_class,
                        "requested_class": target_class_query,
                        "shape_validation": None,
                    }
                semantic_override["semantic_override_sha256"] = canonical_hash(semantic_override)
                approaches, front_audit = (context.approach_options if context is not None else approach_options)(
                    surface, vertices[support_ids], vertices[selected_vertices], navigation, config,
                )
                prefilter = route_prefilter(approaches, start_xy, navigation, config)
                feasible = int(prefilter["reachable_approach_count"]) > 0
                resolution_audits.append({
                    "target_instance_id": target_id,
                    "target_class": target_class,
                    "original_class": original_class,
                    "requested_class": target_class_query,
                    "semantic_override": semantic_override,
                    "semantic_override_sha256": semantic_override["semantic_override_sha256"],
                    "support_vertex_count": int(len(support_ids)),
                    "target_bounds_world_zup_m": target_bounds.astype(float).tolist(),
                    "surface_id": surface_id,
                    "support_component_sha256": surface_sha256,
                    "surface_area_m2": float(surface["area_m2"]),
                    "surface_extraction_gate_pass": True,
                    "reachable_approach_count": int(prefilter["reachable_approach_count"]),
                    "fresh_occupancy_path_cost_m": prefilter["minimum_path_cost_m"],
                    "euclidean_start_to_surface_m": float(np.linalg.norm(start_xy - np.asarray(surface["centre_world_xyz_zup_m"])[:2])),
                    "selection_eligible": feasible,
                })
                extracted.append(ExtractedSurface(
                    target_id, target_class, original_class, support_ids, target_bounds, selected_faces,
                    selected_vertices, surface, surface_sha256, surface_id, approaches,
                    prefilter, {**extraction_audit, "front_inference": front_audit},
                    semantic_override,
                ))
            except Exception as error:  # retain a fail-closed candidate audit
                resolution_audits.append({
                    "target_instance_id": target_id,
                    "original_class": target_spec.get("original_class"),
                    "requested_class": target_class_query,
                    "semantic_override_required": target_spec.get("semantic_override_required"),
                    "shape_validation": shape_evidence,
                    "surface_extraction_gate_pass": False,
                    "selection_eligible": False,
                    "failure": f"{type(error).__name__}:{error}",
                })
    eligible = [row for row in extracted if int(row.route_prefilter["reachable_approach_count"]) > 0]
    require(
        eligible,
        "no semantic target has both a sitting surface and a reachable current-state approach; "
        f"candidate_audits={json.dumps(resolution_audits, ensure_ascii=False, sort_keys=True, allow_nan=False)}",
    )
    selected = min(
        eligible,
        key=lambda row: (
            float(row.route_prefilter["minimum_path_cost_m"]),
            float(np.linalg.norm(start_xy - np.asarray(row.surface["centre_world_xyz_zup_m"])[:2])),
            -float(row.surface["area_m2"]),
            row.target_instance_id,
        ),
    )
    if len(target_specs) == 1:
        require(selected.target_instance_id == target_specs[0]["target_instance_id"], "unique graph target became unreachable")
    selection_audit_payload = {
        "rule": selection_rule,
        "ranking_tuple": [
            "fresh_current_occupancy_four_connected_path_cost_m_ascending",
            "euclidean_current_start_to_surface_m_ascending",
            "surface_area_m2_descending",
            "target_instance_id_lexicographic",
        ],
        "current_query_start_world_xy_m": start_xy.astype(float).tolist(),
        "future_endpoint_motion_pose_or_contact_used": False,
        "candidate_audits": resolution_audits,
        "selected_target_instance_id": selected.target_instance_id,
        "selected_original_class": selected.original_class,
        "selected_resolved_class": selected.target_class,
        "selected_semantic_override": selected.semantic_override,
        "selected_semantic_override_sha256": selected.semantic_override["semantic_override_sha256"],
    }
    selection_audit_payload["selection_audit_sha256"] = canonical_hash(selection_audit_payload)
    world_target_mask, mask_audit = target_mask_from_support(
        world_scene, vertices[selected.target_support_ids], selected.target_bounds, config,
    )
    native_target_mask = native_occupancy(world_target_mask)
    require(native_target_mask.shape == native_scene.shape, "target mask/native occupancy shape drift")
    require(native_target_mask.dtype == np.bool_ and native_target_mask.flags.c_contiguous, "target mask must be C-order bool")
    require(not np.any(native_target_mask & ~native_scene), "native target mask is not a scene subset")
    mask_path = output_dir / "target_occupancy_mask.npy"
    atomic_npy(mask_path, native_target_mask)
    mask_record = artifact(mask_path)
    mask_record.update({
        "shape": list(native_target_mask.shape),
        "dtype": "bool",
        "order": "C",
        "coordinate_order": "LINGO_native_X_Yup_Z",
        "occupancy_source_sha256": sha256_file(occupancy_path),
        "world_xyz_content_sha256": bool_content_sha256(world_target_mask),
    })
    face_mask = np.zeros(len(faces), dtype=np.bool_)
    face_mask[selected.face_ids] = True
    face_mask_sha256 = bool_content_sha256(face_mask)
    support_ids_sha256 = int64_sha256(selected.target_support_ids)
    face_archive_path = output_dir / "target_support_surface_faces.npz"
    atomic_npz(
        face_archive_path,
        schema=np.asarray(SURFACE_ARCHIVE_SCHEMA),
        scene_id=np.asarray(scene_id),
        target_instance_id=np.asarray(selected.target_instance_id),
        target_class=np.asarray(selected.target_class),
        original_class=np.asarray(selected.original_class),
        semantic_override_sha256=np.asarray(selected.semantic_override["semantic_override_sha256"]),
        surface_id=np.asarray(selected.surface_id),
        support_component_sha256=np.asarray(selected.surface_sha256),
        mesh_face_mask=face_mask,
        mesh_face_indices=selected.face_ids,
        surface_vertex_ids=selected.vertex_ids,
        target_support_vertex_ids=selected.target_support_ids,
        target_bounds_world_zup_m=selected.target_bounds.astype(np.float32),
        surface_centre_world_xyz_zup_m=np.asarray(selected.surface["centre_world_xyz_zup_m"], dtype=np.float32),
        surface_bounds_world_zup_m=np.asarray(selected.surface["bounds_world_zup_m"], dtype=np.float32),
    )
    face_record = artifact(face_archive_path)
    binding_token = canonical_hash({
        "schema": "p523.current_only_target_binding_identity.v1",
        "scene_id": scene_id,
        "instruction": instruction,
        "action": "sit",
        "metric_graph_payload_sha256": graph_payload_sha256,
        "sam_receipt_sha256": sha256_file(sam_path),
        "mesh_sha256": mesh_sha256,
        "occupancy_sha256": sha256_file(occupancy_path),
        "target_instance_id": selected.target_instance_id,
        "target_class": selected.target_class,
        "original_class": selected.original_class,
        "semantic_override_sha256": selected.semantic_override["semantic_override_sha256"],
        "target_support_vertex_ids_sha256": support_ids_sha256,
        "support_component_sha256": selected.surface_sha256,
        "target_face_mask_sha256": face_mask_sha256,
        "target_occupancy_mask_world_xyz_content_sha256": mask_record["world_xyz_content_sha256"],
        "selection_audit_sha256": selection_audit_payload["selection_audit_sha256"],
    })
    references = list(graph.get("target_selection", {}).get("reference_entity_ids", []))
    relation = query.get("relation")
    if relation is None:
        references = []
        relation_evidence: list[dict[str, Any]] = []
    else:
        relation_evidence = [
            dict(row) for row in graph.get("relations", [])
            if isinstance(row, Mapping)
            and row.get("subject_id") == selected.target_instance_id
            and row.get("predicate") == relation
            and row.get("object_id") in references
        ]
        require(relation_evidence, "selected relational target lacks relation evidence")
    relation_evidence_sha256 = canonical_hash(relation_evidence)
    target_mask_contract = dict(mask_record)
    target_mask_contract["target_occupancy_mask_world_xyz_sha256"] = mask_record["world_xyz_content_sha256"]
    binding = {
        "verified": True,
        "binding_method": "current_P515_graph_target_plus_original_OBJ_support_faces_plus_current_occupancy",
        "binding_token_sha256": binding_token,
        "graph_target_instance_id": selected.target_instance_id,
        "sam_target_instance_id": selected.target_instance_id,
        "metric_graph_object_class": selected.original_class,
        "resolved_target_class": selected.target_class,
        "semantic_override": selected.semantic_override,
        "semantic_override_sha256": selected.semantic_override["semantic_override_sha256"],
        "metric_graph_relation": relation,
        "metric_graph_reference_entity_ids": references,
        "metric_scene_graph_payload_sha256": graph_payload_sha256,
        "metric_scene_graph_receipt_sha256": sha256_file(graph_path),
        "sam_receipt_sha256": sha256_file(sam_path),
        "sam_support_vertex_count": int(len(selected.target_support_ids)),
        "target_support_vertex_ids_sha256": support_ids_sha256,
        "target_bounds_world_zup_m": selected.target_bounds.astype(float).tolist(),
        "surface_inside_target_furniture_verified": True,
        "support_component_sha256": selected.surface_sha256,
        "target_face_mask_sha256": face_mask_sha256,
        "target_occupancy_mask_world_xyz_sha256": mask_record["world_xyz_content_sha256"],
        "relation_evidence_sha256": relation_evidence_sha256,
    }
    approaches = [dict(row) for row in selected.approaches]
    candidate = {
        "candidate_id": selected.surface_id,
        "support_component_sha256": selected.surface_sha256,
        "support_class": selected.target_class,
        "target_instance_id": selected.target_instance_id,
        "target_class": selected.target_class,
        "original_class": selected.original_class,
        "requested_class": target_class_query,
        "semantic_override": selected.semantic_override,
        "semantic_override_sha256": selected.semantic_override["semantic_override_sha256"],
        "action": "sit",
        "reference_relation": relation,
        "relation_evidence": relation_evidence,
        "relation_evidence_sha256": relation_evidence_sha256,
        "surface": selected.surface,
        "target_bounds_world_zup_m": selected.target_bounds.astype(float).tolist(),
        "binding_token_sha256": binding_token,
        "furniture_instance_binding": binding,
        "target_occupancy_mask": target_mask_contract,
        "target_occupancy_mask_world_xyz_sha256": mask_record["world_xyz_content_sha256"],
        "surface_face_archive": face_record,
        "target_face_mask_sha256": face_mask_sha256,
        "target_support_vertex_ids_sha256": support_ids_sha256,
        "approach": approaches[0],
        "approach_candidates": approaches,
        "approach_options": approaches,
        "feasible_approach_options": approaches,
        "route_prefilter": selected.route_prefilter,
        "surface_extraction_audit": selected.extraction_audit,
        "stage1_rank": 0,
        "stage1_score": 1.0 / (1.0 + float(selected.route_prefilter["minimum_path_cost_m"])),
        "root_chain_world_xy_m": [],
        "final_route": None,
    }
    created_at = datetime.now(timezone.utc).isoformat()
    common: dict[str, Any] = {
        "status": "published_target_verified",
        "stage_gate_pass": True,
        "publish_gate_pass": True,
        "next_action": {
            "type": "continue_to_stage2_geometry_bridge",
            "reason": "one current-scene target and interaction surface passed all planner gates",
        },
        "producer_variant_schema": P523_SCHEMA,
        "created_at_utc": created_at,
        "scene_id": scene_id,
        "instruction": instruction,
        "action": "sit",
        "action_family": "sit",
        "target_class": selected.target_class,
        "original_target_class": selected.original_class,
        "requested_target_class": target_class_query,
        "semantic_override": selected.semantic_override,
        "semantic_override_sha256": selected.semantic_override["semantic_override_sha256"],
        "target_instance_id": selected.target_instance_id,
        "unique_target_instance_id": selected.target_instance_id,
        "selected_target_instance_id": selected.target_instance_id,
        "selected_candidate_id": selected.surface_id,
        "selected_surface_id": selected.surface_id,
        "selected_surface_sha256": selected.surface_sha256,
        "support_component_sha256": selected.surface_sha256,
        "binding_token_sha256": binding_token,
        "target_bounds_world_zup_m": selected.target_bounds.astype(float).tolist(),
        "target_occupancy_mask_world_xyz_sha256": mask_record["world_xyz_content_sha256"],
        "candidate_count": 1,
        "candidate_surfaces": [candidate],
        "feasible_approach_options": approaches,
        "query": {
            "action": "sit",
            "target_class": target_class_query,
            "reference_class": query.get("reference_class"),
            "reference_instance_ids": references,
            "relation": relation,
        },
        "relation_evidence": relation_evidence,
        "relation_evidence_sha256": relation_evidence_sha256,
        "target_resolution": selection_audit_payload,
        "artifacts": {
            "metric_graph": artifact(graph_path),
            "atomic_instances": artifact(sam_path),
            "original_scene_mesh": artifact(mesh_path),
            "scene_occupancy": artifact(occupancy_path),
            "target_support_vertices": artifact(support_path),
            "target_surface_faces": face_record,
            "target_occupancy_mask": target_mask_contract,
        },
        "target_mask_audit": mask_audit,
        "route_contract": {
            "fresh_route_required": True,
            "final_route": None,
            "historical_route_read": False,
            "p480_route_polylines_copied": False,
            "p480_route_keypoints_copied": False,
            "source_route_copied_to_final_route": False,
        },
        "final_route": None,
        "current_only_contract": {
            "motion_pose_contact_keypose_or_future_frame_read": False,
            "historical_surface_or_route_read": False,
            "P373_P480_P517_P520_P522_result_read": False,
            "current_query_start_world_xy_m": start_xy.astype(float).tolist(),
            "fresh_current_occupancy_used": True,
        },
        "source_code": artifact(Path(__file__)),
    }
    primary = {"schema": P523_SCHEMA, **common}
    primary["receipt_payload_sha256"] = canonical_hash(primary)
    primary_path = output_dir / "receipt.json"
    atomic_json(primary_path, primary)
    compatibility = {"schema": P509_COMPAT_SCHEMA, **common}
    compatibility["status"] = "published_candidate_set"
    compatibility["receipt_payload_sha256"] = canonical_hash(compatibility)
    compatibility_path = output_dir / "p509_compatible_stage1_target.json"
    atomic_json(compatibility_path, compatibility)
    return primary

