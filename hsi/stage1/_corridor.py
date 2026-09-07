"""Run P508 pynavmesh and interiorize its raw polygon-corridor route.

The external ``pynavmesh/pathfinder`` result remains the route authority.  The
only post-process is a local digital trace through free polygons touched by
that raw polyline: oversample the raw corners, select a deterministic ordered
4-connected sequence of touched free cells, and join their cell centres.

There is deliberately no global grid traversal, frontier, heuristic, shortest
path recomputation, native A*, or fallback.  This fixes the raster ambiguity
that occurs when a valid polygon route lies exactly on a free/blocked cell
boundary while retaining a byte-hashed copy of the external raw route.
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

import resource

import sys

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import _navmesh as backend

RECEIPT_SCHEMA = "p523.current_only_polygon_corridor_interiorization_receipt.v1"

AUDIT_SCHEMA = "p523.current_only_polygon_corridor_interiorization.v1"

@dataclass(frozen=True)
class StackAudit:
    polygon_count: int
    recursion_limit_before: int
    recursion_limit_after: int
    stack_soft_bytes_before: int
    stack_soft_bytes_after: int
    stack_hard_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "polygon_count": self.polygon_count,
            "recursion_limit_before": self.recursion_limit_before,
            "recursion_limit_after": self.recursion_limit_after,
            "stack_soft_bytes_before": self.stack_soft_bytes_before,
            "stack_soft_bytes_after": self.stack_soft_bytes_after,
            "stack_hard_bytes": self.stack_hard_bytes,
        }

@dataclass(frozen=True)
class InteriorizedRoute:
    raw_route: np.ndarray
    dense_route: np.ndarray
    ordered_cells: tuple[tuple[int, int], ...]
    corridor_cells: frozenset[tuple[int, int]]
    oversample_count: int
    bridge_cell_count: int
    maximum_center_deviation_m: float

def canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def artifact(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved), "bytes": int(resolved.stat().st_size),
        "sha256": backend.sha256_file(resolved),
    }

def configure_scene_stack(polygon_count: int, requested_stack_mb: int) -> StackAudit:
    """Raise both OS stack and Python recursion before constructing PathFinder."""
    backend.require(polygon_count > 0, "roadmap contains no polygon")
    backend.require(64 <= requested_stack_mb <= 1024,
                    "requested stack must be in [64,1024] MiB")
    recursion_before = sys.getrecursionlimit()
    recursion_after = max(recursion_before, min(500_000, polygon_count * 6 + 10_000))
    soft_before, hard = resource.getrlimit(resource.RLIMIT_STACK)
    requested = int(requested_stack_mb) * 1024 * 1024
    if hard == resource.RLIM_INFINITY:
        soft_after = max(int(soft_before), requested)
    else:
        soft_after = min(int(hard), max(int(soft_before), requested))
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (soft_after, hard))
    except (OSError, ValueError) as error:
        raise backend.ContractError(
            "cannot raise process stack safely for pynavmesh: %s" % error
        ) from error
    sys.setrecursionlimit(recursion_after)
    actual_soft, actual_hard = resource.getrlimit(resource.RLIMIT_STACK)
    backend.require(
        actual_soft == resource.RLIM_INFINITY or int(actual_soft) >= 64 * 1024 * 1024,
        "pynavmesh process stack remains below 64 MiB",
    )
    return StackAudit(
        polygon_count=int(polygon_count),
        recursion_limit_before=int(recursion_before),
        recursion_limit_after=int(sys.getrecursionlimit()),
        stack_soft_bytes_before=int(soft_before),
        stack_soft_bytes_after=int(actual_soft),
        stack_hard_bytes=int(actual_hard),
    )

def _oversample_polyline(
    route: np.ndarray, spacing_m: float
) -> list[tuple[np.ndarray, int]]:
    points = backend.remove_consecutive_duplicates(route)
    backend.require(spacing_m > 0.0 and math.isfinite(spacing_m),
                    "oversample spacing is invalid")
    output: list[tuple[np.ndarray, int]] = []
    for segment_index, (start, finish) in enumerate(zip(points[:-1], points[1:])):
        distance = float(np.linalg.norm(finish - start))
        count = max(1, int(math.ceil(distance / spacing_m)))
        begin = 0 if segment_index == 0 else 1
        for sample_index in range(begin, count + 1):
            fraction = float(sample_index) / float(count)
            output.append((start + (finish - start) * fraction, segment_index))
    backend.require(len(output) >= 2, "raw route oversampling produced too few points")
    return output

def _point_box_distance(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    return float(np.linalg.norm(delta))

def _free_cells_near_point(
    point: np.ndarray,
    request: backend.RoadmapRequest,
    tolerance_m: float,
) -> tuple[tuple[int, int], ...]:
    coordinate = (np.asarray(point, dtype=np.float64) - request.lower_world_xy_m) / request.cell_resolution_m
    base = np.floor(coordinate).astype(np.int64)
    cells: list[tuple[int, int]] = []
    for i in range(int(base[0]) - 1, int(base[0]) + 2):
        for j in range(int(base[1]) - 1, int(base[1]) + 2):
            if not (0 <= i < request.free.shape[0] and 0 <= j < request.free.shape[1]):
                continue
            cell = (i, j)
            if not backend.cell_meets_navigation_contract(request, cell):
                continue
            lower = request.lower_world_xy_m + np.asarray(cell, dtype=np.float64) * request.cell_resolution_m
            upper = lower + request.cell_resolution_m
            if _point_box_distance(point, lower, upper) <= tolerance_m + 1e-12:
                cells.append(cell)
    return tuple(sorted(set(cells)))

def _segment_intersects_expanded_cell(
    start: np.ndarray,
    finish: np.ndarray,
    cell: tuple[int, int],
    request: backend.RoadmapRequest,
    tolerance_m: float,
) -> bool:
    lower = (
        request.lower_world_xy_m
        + np.asarray(cell, dtype=np.float64) * request.cell_resolution_m
        - tolerance_m
    )
    upper = lower + request.cell_resolution_m + 2.0 * tolerance_m
    direction = finish - start
    t_min, t_max = 0.0, 1.0
    for axis in range(2):
        if abs(float(direction[axis])) <= 1e-15:
            if float(start[axis]) < float(lower[axis]) or float(start[axis]) > float(upper[axis]):
                return False
            continue
        left = float((lower[axis] - start[axis]) / direction[axis])
        right = float((upper[axis] - start[axis]) / direction[axis])
        if left > right:
            left, right = right, left
        t_min = max(t_min, left)
        t_max = min(t_max, right)
        if t_min > t_max + 1e-12:
            return False
    return True

def _cell_touched_by_raw_route(
    cell: tuple[int, int], route: np.ndarray,
    request: backend.RoadmapRequest, tolerance_m: float,
) -> bool:
    return any(
        _segment_intersects_expanded_cell(start, finish, cell, request, tolerance_m)
        for start, finish in zip(route[:-1], route[1:])
    )

def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))

def _candidate_score(
    cell: tuple[int, int], point: np.ndarray, request: backend.RoadmapRequest
) -> tuple[float, int, int]:
    centre = backend.cell_center(cell, request.lower_world_xy_m, request.cell_resolution_m)
    return (float(np.linalg.norm(centre - point)), int(cell[0]), int(cell[1]))

def _bridge_diagonal(
    current: tuple[int, int], target: tuple[int, int], raw_route: np.ndarray,
    point: np.ndarray, request: backend.RoadmapRequest, tolerance_m: float,
) -> tuple[int, int]:
    backend.require(
        abs(current[0] - target[0]) == 1 and abs(current[1] - target[1]) == 1,
        "local raw-corridor transition is not a diagonal cell transition",
    )
    options = ((current[0], target[1]), (target[0], current[1]))
    valid = [
        cell for cell in options
        if backend.cell_meets_navigation_contract(request, cell)
        and _cell_touched_by_raw_route(cell, raw_route, request, tolerance_m)
    ]
    backend.require(bool(valid),
                    "raw polygon corridor has no free 4-connected diagonal bridge")
    return min(valid, key=lambda cell: _candidate_score(cell, point, request))

def _distance_to_polyline(point: np.ndarray, route: np.ndarray) -> float:
    best = math.inf
    for start, finish in zip(route[:-1], route[1:]):
        delta = finish - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-20:
            distance = float(np.linalg.norm(point - start))
        else:
            fraction = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
            distance = float(np.linalg.norm(point - (start + fraction * delta)))
        best = min(best, distance)
    return best

def interiorize_raw_polygon_corridor(
    raw_route: np.ndarray,
    request: backend.RoadmapRequest,
    *, oversample_spacing_m: float,
    corridor_tolerance_m: float,
    max_step_m: float,
) -> InteriorizedRoute:
    raw = backend.remove_consecutive_duplicates(raw_route)
    backend.require(len(raw) >= 2, "raw pynavmesh route contains fewer than two corners")
    samples = _oversample_polyline(raw, oversample_spacing_m)
    ordered: list[tuple[int, int]] = [request.snapped_start_cell]
    corridor: set[tuple[int, int]] = {request.snapped_start_cell}
    bridge_count = 0
    for point, _segment_index in samples:
        candidates = _free_cells_near_point(point, request, corridor_tolerance_m)
        backend.require(bool(candidates),
                        "raw pynavmesh sample has no touched free polygon")
        corridor.update(candidates)
        current = ordered[-1]
        if current in candidates:
            continue
        adjacent = [cell for cell in candidates if _manhattan(current, cell) == 1]
        if adjacent:
            selected = min(adjacent, key=lambda cell: _candidate_score(cell, point, request))
            ordered.append(selected)
            continue
        diagonal = [
            cell for cell in candidates
            if abs(cell[0] - current[0]) == 1 and abs(cell[1] - current[1]) == 1
        ]
        backend.require(bool(diagonal),
                        "oversampling skipped beyond the local raw polygon corridor")
        selected = min(diagonal, key=lambda cell: _candidate_score(cell, point, request))
        bridge = _bridge_diagonal(
            current, selected, raw, point, request, corridor_tolerance_m
        )
        corridor.add(bridge)
        ordered.extend((bridge, selected))
        bridge_count += 1

    goal = request.snapped_goal_cell
    if ordered[-1] != goal:
        distance = _manhattan(ordered[-1], goal)
        if distance == 1:
            backend.require(backend.cell_meets_navigation_contract(request, goal),
                            "snapped goal is not free")
            ordered.append(goal)
        elif distance == 2 and abs(ordered[-1][0] - goal[0]) == 1:
            bridge = _bridge_diagonal(
                ordered[-1], goal, raw, raw[-1], request, corridor_tolerance_m
            )
            ordered.extend((bridge, goal))
            corridor.add(bridge)
            bridge_count += 1
        else:
            raise backend.ContractError(
                "raw corridor trace did not terminate locally at the snapped goal"
            )

    deduplicated: list[tuple[int, int]] = []
    for cell in ordered:
        if not deduplicated or cell != deduplicated[-1]:
            deduplicated.append(cell)
    backend.require(deduplicated[0] == request.snapped_start_cell
                    and deduplicated[-1] == request.snapped_goal_cell,
                    "interior cell sequence lost a snapped endpoint")
    backend.require(all(_manhattan(left, right) == 1
                        for left, right in zip(deduplicated[:-1], deduplicated[1:])),
                    "interior cell sequence is not 4-connected")
    backend.require(all(backend.cell_meets_navigation_contract(request, cell)
                        for cell in deduplicated),
                    "interior cell sequence contains a blocked/invalid cell")
    backend.require(all(_cell_touched_by_raw_route(
        cell, raw, request, corridor_tolerance_m
    ) for cell in deduplicated),
                    "interior cell sequence escaped the raw polygon corridor")

    centres = np.asarray([
        backend.cell_center(cell, request.lower_world_xy_m, request.cell_resolution_m)
        for cell in deduplicated
    ], dtype=np.float64)
    # Snapped endpoints are cell centres by the immutable P508 request contract.
    centres[0] = request.snapped_start_world_xy_m
    centres[-1] = request.snapped_goal_world_xy_m
    dense = backend.densify_route(centres, max_step_m)
    maximum_deviation = max(_distance_to_polyline(point, raw) for point in centres)
    return InteriorizedRoute(
        raw_route=raw,
        dense_route=dense,
        ordered_cells=tuple(deduplicated),
        corridor_cells=frozenset(corridor),
        oversample_count=len(samples),
        bridge_cell_count=bridge_count,
        maximum_center_deviation_m=float(maximum_deviation),
    )

def first_strict_violation(
    route: np.ndarray,
    request: backend.RoadmapRequest,
    collision_sample_spacing_m: float,
) -> dict[str, Any] | None:
    points = np.asarray(route, dtype=np.float64)
    for segment_index, (start, finish) in enumerate(zip(points[:-1], points[1:])):
        distance = float(np.linalg.norm(finish - start))
        subdivisions = max(1, int(math.ceil(distance / collision_sample_spacing_m)))
        for sample_index in range(subdivisions + 1):
            point = start + (finish - start) * (float(sample_index) / subdivisions)
            try:
                cell = backend.world_to_cell(
                    point, request.lower_world_xy_m, request.cell_resolution_m,
                    request.free.shape, name="raw route audit",
                )
            except backend.ContractError as error:
                return {
                    "kind": "outside_roadmap", "segment_index": segment_index,
                    "sample_index": sample_index, "point_world_xy_m": point.tolist(),
                    "reason": str(error),
                }
            if not bool(request.free[cell]):
                return {
                    "kind": "blocked_cell_boundary_ambiguity",
                    "segment_index": segment_index, "sample_index": sample_index,
                    "point_world_xy_m": point.tolist(), "cell": list(cell),
                    "clearance_m": float(request.clearance_m[cell]),
                }
            if not backend.cell_meets_navigation_contract(request, cell):
                return {
                    "kind": "navigation_contract_violation",
                    "segment_index": segment_index, "sample_index": sample_index,
                    "point_world_xy_m": point.tolist(), "cell": list(cell),
                    "clearance_m": float(request.clearance_m[cell]),
                }
    return None

def _cell_sequence_sha256(cells: Sequence[tuple[int, int]]) -> str:
    array = np.ascontiguousarray(np.asarray(cells, dtype="<i8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()

def run(
    request_path: Path, output_path: Path, receipt_path: Path,
    *, stack_mb: int = 512,
) -> dict[str, Any]:
    request_path = request_path.resolve(strict=True)
    output_path = output_path.resolve()
    receipt_path = receipt_path.resolve()
    backend.require(not output_path.exists(), "refusing to overwrite route output")
    backend.require(not receipt_path.exists(), "refusing to overwrite route receipt")
    request = backend.load_request(request_path)
    stack = configure_scene_stack(int(request.free.sum()), stack_mb)
    module, identity = backend.load_pynavmesh()
    vertices, polygons = backend.build_cell_polygon_navmesh(
        request.free, request.lower_world_xy_m, request.cell_resolution_m,
    )
    raw_route = backend._pathfinder_route(  # real authenticated P508 backend
        module, vertices, polygons,
        request.snapped_start_world_xy_m, request.snapped_goal_world_xy_m,
    )
    endpoint_tolerance_backend = request.cell_resolution_m
    backend.require(float(np.linalg.norm(raw_route[0] - request.snapped_start_world_xy_m))
                    <= endpoint_tolerance_backend,
                    "pynavmesh changed the snapped start endpoint")
    backend.require(float(np.linalg.norm(raw_route[-1] - request.snapped_goal_world_xy_m))
                    <= endpoint_tolerance_backend,
                    "pynavmesh changed the snapped goal endpoint")
    raw_route[0] = request.snapped_start_world_xy_m
    raw_route[-1] = request.snapped_goal_world_xy_m

    max_step = request.cell_resolution_m * 0.5
    sample_spacing = request.cell_resolution_m * 0.25
    endpoint_tolerance = max(1e-8, request.cell_resolution_m * 1e-5)
    oversample_spacing = request.cell_resolution_m / 16.0
    corridor_tolerance = request.cell_resolution_m * 1e-4
    raw_dense = backend.densify_route(raw_route, max_step)
    raw_violation = first_strict_violation(raw_dense, request, sample_spacing)
    interior = interiorize_raw_polygon_corridor(
        raw_route, request,
        oversample_spacing_m=oversample_spacing,
        corridor_tolerance_m=corridor_tolerance,
        max_step_m=max_step,
    )
    payload = backend.make_output_payload(
        request, identity, interior.dense_route, raw_route,
        len(vertices), len(polygons),
        max_step_m=max_step,
        collision_sample_spacing_m=sample_spacing,
        endpoint_tolerance_m=endpoint_tolerance,
    )
    audit = {
        "schema": AUDIT_SCHEMA,
        "status": "verified_local_polygon_corridor_interiorization",
        "route_authority": "authenticated_p508_pynavmesh_pathfinder_raw_corner_route",
        "raw_corner_route_world_xy_float64_sha256": backend.route_sha256(raw_route),
        "raw_corner_count": int(len(raw_route)),
        "raw_dense_point_count_for_strict_audit": int(len(raw_dense)),
        "raw_strict_validation_passed": raw_violation is None,
        "raw_first_strict_violation": raw_violation,
        "oversample_spacing_m": oversample_spacing,
        "oversample_count": interior.oversample_count,
        "corridor_boundary_tolerance_m": corridor_tolerance,
        "raw_touched_free_polygon_count": len(interior.corridor_cells),
        "ordered_4_connected_cell_count": len(interior.ordered_cells),
        "ordered_4_connected_cells_sha256": _cell_sequence_sha256(interior.ordered_cells),
        "diagonal_local_bridge_cell_count": interior.bridge_cell_count,
        "maximum_cell_center_deviation_from_raw_route_m": interior.maximum_center_deviation_m,
        "all_selected_cells_free": True,
        "all_selected_cells_touch_raw_polygon_route": True,
        "cell_sequence_4_connected": True,
        "cell_centres_used": True,
        "external_pynavmesh_remains_route_authority": True,
        "global_grid_search_used": False,
        "native_grid_astar_used": False,
        "heuristic_search_used": False,
        "frontier_or_priority_queue_used": False,
        "shortest_path_recomputed": False,
        "postprocess_scope": "only_free_polygons_geometrically_touched_by_raw_route",
    }
    payload["backend"].update({
        "route_authority": audit["route_authority"],
        "polygon_corridor_interiorization_used": True,
        "global_grid_search_used": False,
        "native_grid_astar_used": False,
    })
    payload["raw_corner_route_world_xy_float64_sha256"] = backend.route_sha256(raw_route)
    payload["polygon_corridor_interiorization"] = audit
    # Re-run the unchanged P508 validator on a JSON roundtrip before publish.
    roundtrip = json.loads(json.dumps(payload, allow_nan=False))
    strict_validation = backend.validate_output_payload(roundtrip, request)
    backend.require(strict_validation == payload["validation"],
                    "serialized strict validation metrics drift")
    backend.atomic_json(output_path, payload)

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "published_current_only_polygon_corridor_route",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": artifact(request_path),
        "request_npz": artifact(request.input_npz_path),
        "output_route": artifact(output_path),
        "external_backend": {
            "distribution": identity.distribution,
            "version": identity.version,
            "import_name": identity.import_name,
            "module_file": identity.module_file,
        },
        "stack_configuration": stack.to_dict(),
        "raw_route": {
            "point_count": int(len(raw_route)),
            "sha256": backend.route_sha256(raw_route),
        },
        "interiorization_audit": audit,
        "strict_validate_route": payload["validation"],
        "route_schema_compatible_with_p508_and_p523_finalizer": True,
        "current_only": True,
        "historical_route_reused": False,
        "global_grid_search_used": False,
        "native_grid_astar_used": False,
        "gpu_used": False,
    }
    receipt["receipt_payload_sha256"] = canonical_hash(receipt)
    backend.atomic_json(receipt_path, receipt)
    return payload

