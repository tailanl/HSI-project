"""Compile a P508 scene-first Stage-1 result into one fresh P504 request.

The Stage-1 receipt intentionally contains several terminal approach options
and no final route.  This compiler rebuilds the human-clear roadmap directly
from LINGO occupancy, snaps the start, and independently audits *every*
approach against the new roadmap.  The selected request is based only on fresh
grid connectivity and distance.  No P480 route polyline or keypoint is read,
copied, or used for ranking.
"""

from __future__ import annotations

import argparse

from collections import deque

from dataclasses import dataclass

from datetime import datetime, timezone

import hashlib

import json

import math

import os

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

from scipy.ndimage import distance_transform_edt

STAGE1_SCHEMA = "p508.lingo_scene_first_stage1.v1"

P504_INPUT_SCHEMA = "p504.naomesh_human_roadmap_input.v1"

POLYGON_NAVMESH_OUTPUT_SCHEMA = "p508.pynavmesh_polygon_route_output.v1"

RECEIPT_SCHEMA = "p508.lingo_stage1_roadmap_compile.v1"

CURRENT_STATE_SCHEMA = "p366.query_current_frame0.v1"

class CompileError(RuntimeError):
    """Raised when a Stage-1 input cannot safely become a route request."""

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise CompileError(message)

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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Stage-1 JSON root must be an object")
    return value

def finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (length,), "%s must contain %d values" % (label, length))
    require(np.isfinite(array).all(), "%s contains non-finite values" % label)
    return array

def start_from_current_state(
    state_path: Path,
    receipt_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the navigation start from the current human root, never from future motion.

    The receipt is required so the state artifact is hash-bound.  Both the
    canonical query start and the SMPL-X root must agree in the state file.
    This prevents a caller from silently choosing a more convenient start.
    """

    state_resolved = state_path.expanduser().resolve(strict=True)
    receipt_resolved = receipt_path.expanduser().resolve(strict=True)
    state_receipt = read_json(receipt_resolved)
    state_record = state_receipt.get("current_frame")
    require(isinstance(state_record, Mapping), "current-state receipt lacks current_frame artifact")
    require(state_record.get("sha256") == sha256_file(state_resolved), "current-state SHA-256 drift")
    require(state_receipt.get("future_ground_truth_used") is False, "current-state receipt used future ground truth")
    require(state_receipt.get("lingo_motion_opened") is False, "current-state receipt opened LINGO motion")

    with np.load(state_resolved, allow_pickle=False) as archive:
        required = {
            "schema", "root_xyz_yaw_world_zup", "query_start_world_xz_m",
            "future_ground_truth_used", "source_kind",
        }
        require(required.issubset(set(archive.files)), "current-state NPZ lacks required arrays")
        schema = str(np.asarray(archive["schema"]).item())
        require(schema == CURRENT_STATE_SCHEMA, "unsupported current-state schema")
        root = finite_vector(archive["root_xyz_yaw_world_zup"], 4, "root_xyz_yaw_world_zup")
        query_start = finite_vector(archive["query_start_world_xz_m"], 2, "query_start_world_xz_m")
        future_used = bool(np.asarray(archive["future_ground_truth_used"]).item())
        source_kind = str(np.asarray(archive["source_kind"]).item())
    require(future_used is False, "current-state NPZ used future ground truth")
    require(
        float(np.linalg.norm(root[:2] - query_start)) <= 1e-8,
        "current human root disagrees with the query start",
    )
    receipt_root = finite_vector(
        state_receipt.get("pelvis_root_xyz_world_zup"), 3, "receipt pelvis root"
    )
    require(float(np.linalg.norm(receipt_root - root[:3])) <= 1e-7, "current-state receipt root drift")
    receipt_query_start = finite_vector(
        state_receipt.get("query_start_world_xz_m"), 2, "receipt query start"
    )
    require(float(np.linalg.norm(receipt_query_start - query_start)) <= 1e-8, "query start receipt drift")
    return root[:2].copy(), {
        "kind": "current_human_root_world_zup",
        "current_state": artifact(state_resolved),
        "current_state_receipt": artifact(receipt_resolved),
        "state_schema": schema,
        "state_source_kind": source_kind,
        "root_xyz_yaw_world_zup": root.astype(float).tolist(),
        "future_ground_truth_used": False,
        "start_learned_or_guessed": False,
    }

@dataclass(frozen=True)
class RoadmapConfig:
    lower_world_xyz_m: tuple[float, float, float] = (-3.0, -4.0, 0.0)
    cell_resolution_m: float = 0.02
    body_height_range_m: tuple[float, float] = (0.08, 1.75)
    human_radius_clearance_m: float = 0.28
    target_approach_min_clearance_m: float = 0.19
    maximum_snap_distance_m: float = 1.0
    hard_navigation_clearance_m: float | None = None
    navigation_clearance_mode: str = "human_radius_hard_buffer"
    target_interaction_relaxation_radius_m: float = 0.12
    interaction_terminal_min_clearance_m: float = 0.25

    @property
    def effective_hard_navigation_clearance_m(self) -> float:
        if self.hard_navigation_clearance_m is None:
            return float(self.human_radius_clearance_m)
        return float(self.hard_navigation_clearance_m)

    def validate(self) -> None:
        lower = finite_vector(self.lower_world_xyz_m, 3, "lower_world_xyz_m")
        require(np.isfinite(lower).all(), "roadmap lower bound is invalid")
        require(self.cell_resolution_m > 0.0, "cell resolution must be positive")
        require(
            0.0 <= self.body_height_range_m[0] < self.body_height_range_m[1],
            "body height range is invalid",
        )
        require(self.human_radius_clearance_m > 0.0, "human clearance must be positive")
        require(
            self.navigation_clearance_mode in {
                "human_radius_hard_buffer",
                "raw_free_stage3_sdf_collision",
                "target_interaction_relaxed_stage3_sdf_collision",
            },
            "unsupported navigation clearance mode",
        )
        require(
            self.effective_hard_navigation_clearance_m >= 0.0,
            "hard navigation clearance must be non-negative",
        )
        if self.navigation_clearance_mode in {
            "human_radius_hard_buffer", "target_interaction_relaxed_stage3_sdf_collision"
        }:
            require(
                abs(self.effective_hard_navigation_clearance_m - self.human_radius_clearance_m) <= 1e-12,
                "hard-buffer mode must use the nominal human radius",
            )
        else:
            require(
                self.effective_hard_navigation_clearance_m == 0.0,
                "raw-free mode cannot retain a hard clearance buffer",
            )
        require(self.target_approach_min_clearance_m > 0.0, "approach clearance must be positive")
        require(self.maximum_snap_distance_m > 0.0, "maximum snap distance must be positive")
        require(
            self.target_interaction_relaxation_radius_m > 0.0,
            "target interaction relaxation radius must be positive",
        )
        require(
            0.0 < self.interaction_terminal_min_clearance_m <= self.human_radius_clearance_m,
            "interaction terminal clearance threshold is invalid",
        )

def world_occupancy(native: np.ndarray) -> np.ndarray:
    """Convert LINGO [X,Y-up,Z] occupancy to world [X,Y,Z-up]."""

    array = np.asarray(native)
    require(array.ndim == 3 and array.dtype == np.bool_, "LINGO occupancy must be a Boolean 3-D array")
    return np.ascontiguousarray(array.transpose(0, 2, 1)[:, ::-1, :])

def build_roadmap(native_occupancy: np.ndarray, config: RoadmapConfig) -> dict[str, np.ndarray]:
    config.validate()
    occupancy = world_occupancy(native_occupancy)
    lower_z = float(config.lower_world_xyz_m[2])
    heights = lower_z + (np.arange(occupancy.shape[2]) + 0.5) * config.cell_resolution_m
    body_band = (heights >= config.body_height_range_m[0]) & (heights <= config.body_height_range_m[1])
    require(bool(np.any(body_band)), "occupancy has no voxel layer inside the body-height range")
    obstacles = occupancy[:, :, body_band].any(axis=2)
    clearance = distance_transform_edt(~obstacles) * config.cell_resolution_m
    hard_clearance = config.effective_hard_navigation_clearance_m
    free = (~obstacles) & (clearance >= hard_clearance)
    require(bool(np.any(free)), "fresh occupancy produces no human-clear roadmap cell")
    return {
        "obstacles": np.asarray(obstacles, dtype=np.bool_),
        "clearance": np.asarray(clearance, dtype=np.float64),
        "free": np.asarray(free, dtype=np.bool_),
        "target_interaction_relaxed": np.zeros_like(free, dtype=np.bool_),
    }

def relax_target_interaction_approaches(
    roadmap: Mapping[str, np.ndarray],
    options: Sequence[Mapping[str, Any]],
    config: RoadmapConfig,
) -> dict[str, np.ndarray]:
    """Allow raw-free low-clearance cells only around grounded target approaches."""

    require(
        config.navigation_clearance_mode == "target_interaction_relaxed_stage3_sdf_collision",
        "target relaxation called in the wrong clearance mode",
    )
    obstacles = np.asarray(roadmap["obstacles"], dtype=np.bool_)
    clearance = np.asarray(roadmap["clearance"], dtype=np.float64)
    base_free = np.asarray(roadmap["free"], dtype=np.bool_)
    nx, ny = obstacles.shape
    lower = np.asarray(config.lower_world_xyz_m[:2], dtype=np.float64)
    x = lower[0] + (np.arange(nx, dtype=np.float64) + 0.5) * config.cell_resolution_m
    y = lower[1] + (np.arange(ny, dtype=np.float64) + 0.5) * config.cell_resolution_m
    xx, yy = np.meshgrid(x, y, indexing="ij")
    interaction_zone = np.zeros_like(obstacles, dtype=np.bool_)
    radius_sq = config.target_interaction_relaxation_radius_m ** 2
    for option in options:
        point = finite_vector(option.get("approach_world_xy_m"), 2, "target approach")
        interaction_zone |= (xx - point[0]) ** 2 + (yy - point[1]) ** 2 <= radius_sq + 1e-12
    relaxed = (
        interaction_zone
        & (~obstacles)
        & (~base_free)
        & (clearance >= config.interaction_terminal_min_clearance_m - 1e-12)
    )
    result = {key: np.asarray(value).copy() for key, value in roadmap.items()}
    result["target_interaction_relaxed"] = relaxed
    result["free"] = base_free | relaxed
    require(not np.any(result["free"] & obstacles), "target relaxation exposed a raw obstacle")
    return result

def point_to_cell(
    point_xy: Sequence[float], shape: tuple[int, int], config: RoadmapConfig, label: str
) -> tuple[int, int]:
    point = finite_vector(point_xy, 2, label)
    lower = np.asarray(config.lower_world_xyz_m[:2], dtype=np.float64)
    cell = np.floor((point - lower) / config.cell_resolution_m).astype(np.int64)
    require(
        0 <= int(cell[0]) < shape[0] and 0 <= int(cell[1]) < shape[1],
        "%s lies outside the roadmap" % label,
    )
    return int(cell[0]), int(cell[1])

def cell_center(cell: Sequence[int], config: RoadmapConfig) -> np.ndarray:
    return np.asarray(config.lower_world_xyz_m[:2], dtype=np.float64) + (
        np.asarray(cell, dtype=np.float64) + 0.5
    ) * config.cell_resolution_m

def nearest_mask_cell(
    mask: np.ndarray,
    point_xy: np.ndarray,
    config: RoadmapConfig,
) -> tuple[tuple[int, int], float] | None:
    """Return the closest eligible cell centre within the configured snap radius."""

    raw_cell = point_to_cell(point_xy, mask.shape, config, "snap point")
    radius_cells = int(math.ceil(config.maximum_snap_distance_m / config.cell_resolution_m)) + 1
    x0, x1 = max(0, raw_cell[0] - radius_cells), min(mask.shape[0], raw_cell[0] + radius_cells + 1)
    y0, y1 = max(0, raw_cell[1] - radius_cells), min(mask.shape[1], raw_cell[1] + radius_cells + 1)
    local = np.argwhere(mask[x0:x1, y0:y1])
    if len(local) == 0:
        return None
    cells = local + np.asarray([x0, y0], dtype=np.int64)
    centres = np.asarray([cell_center(cell, config) for cell in cells], dtype=np.float64)
    distances = np.linalg.norm(centres - point_xy[None, :], axis=1)
    eligible = np.flatnonzero(distances <= config.maximum_snap_distance_m + 1e-12)
    if len(eligible) == 0:
        return None
    # Stable ordering: geometric distance, then integer X/Y cell index.
    best = min(eligible.tolist(), key=lambda index: (float(distances[index]), int(cells[index, 0]), int(cells[index, 1])))
    cell = (int(cells[best, 0]), int(cells[best, 1]))
    return cell, float(distances[best])

def four_connected_distances(free: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Fresh reachability audit; returns cell steps and never emits a route."""

    require(bool(free[start]), "snapped start is not human-clear")
    distance = np.full(free.shape, -1, dtype=np.int32)
    distance[start] = 0
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        x, y = queue.popleft()
        next_distance = int(distance[x, y]) + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < free.shape[0] and 0 <= ny < free.shape[1] and free[nx, ny] and distance[nx, ny] < 0:
                distance[nx, ny] = next_distance
                queue.append((nx, ny))
    return distance

def validate_stage1(stage1: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    require(stage1.get("schema") == STAGE1_SCHEMA, "unsupported Stage-1 schema")
    require(stage1.get("status") == "published_candidate_set", "Stage-1 did not publish a candidate set")
    require(stage1.get("final_route") is None, "Stage-1 root final_route must be null or absent")
    route_contract = stage1.get("route_contract")
    require(isinstance(route_contract, Mapping), "Stage-1 lacks its fresh-route contract")
    require(route_contract.get("fresh_route_required") is True, "Stage-1 does not require fresh routing")
    require(route_contract.get("final_route") is None, "Stage-1 route contract already contains a final route")
    require(route_contract.get("p480_route_polylines_copied") is False, "Stage-1 admits copied P480 polylines")
    candidates = stage1.get("candidate_surfaces")
    require(isinstance(candidates, list) and len(candidates) == 1, "Stage-1 must contain exactly one surface")
    candidate = candidates[0]
    require(isinstance(candidate, Mapping), "Stage-1 surface is malformed")
    require(candidate.get("final_route") is None, "candidate surface already contains a final route")
    options = candidate.get("approach_options")
    require(isinstance(options, list) and options, "candidate surface has no approach options")
    option_ids: set[str] = set()
    checked: list[Mapping[str, Any]] = []
    for index, option in enumerate(options):
        require(isinstance(option, Mapping), "approach option %d is malformed" % index)
        option_id = str(option.get("option_id", ""))
        require(option_id and option_id not in option_ids, "approach option IDs must be non-empty and unique")
        option_ids.add(option_id)
        finite_vector(option.get("approach_world_xy_m"), 2, "%s.approach_world_xy_m" % option_id)
        require(option.get("requires_fresh_route_planning") is True, "%s does not require fresh routing" % option_id)
        require(option.get("source_route_copied_to_final_route") is False, "%s permits legacy route reuse" % option_id)
        checked.append(option)
    return candidate, checked

def audit_approaches(
    options: Sequence[Mapping[str, Any]],
    roadmap: Mapping[str, np.ndarray],
    start_cell: tuple[int, int],
    config: RoadmapConfig,
) -> list[dict[str, Any]]:
    free = np.asarray(roadmap["free"], dtype=np.bool_)
    clearance = np.asarray(roadmap["clearance"], dtype=np.float64)
    relaxed = np.asarray(roadmap.get("target_interaction_relaxed", np.zeros_like(free)), dtype=np.bool_)
    distances = four_connected_distances(free, start_cell)
    reachable = distances >= 0
    audits: list[dict[str, Any]] = []
    for source_index, option in enumerate(options):
        option_id = str(option["option_id"])
        requested = finite_vector(option["approach_world_xy_m"], 2, option_id)
        raw_cell = point_to_cell(requested, free.shape, config, option_id)
        nearest_free = nearest_mask_cell(free, requested, config)
        nearest_reachable = nearest_mask_cell(reachable, requested, config)
        audit: dict[str, Any] = {
            "option_id": option_id,
            "source_option_index": source_index,
            "requested_approach_world_xy_m": requested.astype(float).tolist(),
            "requested_cell": list(raw_cell),
            "requested_cell_clearance_m": float(clearance[raw_cell]),
            "requested_cell_human_free": bool(free[raw_cell]),
            "fresh_reachability_method": "four_connected_flood_fill_from_fresh_snapped_start",
            "legacy_route_fields_read": False,
            "legacy_route_used_for_ranking": False,
        }
        if nearest_free is not None:
            free_cell, free_snap = nearest_free
            audit["nearest_human_free_cell"] = list(free_cell)
            audit["nearest_human_free_snap_distance_m"] = free_snap
            audit["nearest_human_free_cell_reachable"] = bool(reachable[free_cell])
        else:
            audit["nearest_human_free_cell"] = None
        if nearest_reachable is None:
            audit.update({
                "status": "rejected_no_reachable_human_clear_snap",
                "snapped_goal_cell": None,
                "fresh_reachable": False,
            })
        else:
            goal_cell, snap_distance = nearest_reachable
            goal_clearance = float(clearance[goal_cell])
            goal_relaxed = bool(relaxed[goal_cell])
            if goal_clearance + 1e-9 < config.effective_hard_navigation_clearance_m:
                require(
                    config.navigation_clearance_mode
                    == "target_interaction_relaxed_stage3_sdf_collision" and goal_relaxed,
                    "reachable snap violates hard navigation clearance outside target interaction relaxation",
                )
            audit.update({
                "status": "fresh_reachable",
                "snapped_goal_cell": list(goal_cell),
                "snapped_goal_world_xy_m": cell_center(goal_cell, config).astype(float).tolist(),
                "snap_distance_m": snap_distance,
                "snapped_goal_clearance_m": goal_clearance,
                "snapped_goal_uses_target_interaction_relaxation": goal_relaxed,
                "fresh_reachable": True,
                "fresh_four_connected_distance_m": float(distances[goal_cell]) * config.cell_resolution_m,
            })
        audits.append(audit)
    return audits

def compile_request(
    stage1_path: Path,
    occupancy_path: Path,
    start_world_xy_m: Sequence[float],
    request_path: Path,
    receipt_path: Path,
    config: RoadmapConfig = RoadmapConfig(),
    start_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile and publish a backend-consumable request plus an audit receipt."""

    config.validate()
    stage1_resolved = stage1_path.expanduser().resolve(strict=True)
    occupancy_resolved = occupancy_path.expanduser().resolve(strict=True)
    stage1 = read_json(stage1_resolved)
    candidate, options = validate_stage1(stage1)
    native = np.load(occupancy_resolved, allow_pickle=False)
    base_roadmap = build_roadmap(native, config)
    roadmap = base_roadmap
    start = finite_vector(start_world_xy_m, 2, "start_world_xy_m")
    resolved_start_source = dict(start_source or {
        "kind": "explicit_world_xy",
        "start_learned_or_guessed": False,
    })
    point_to_cell(start, base_roadmap["free"].shape, config, "requested start")
    start_snap = nearest_mask_cell(np.asarray(base_roadmap["free"]), start, config)
    require(start_snap is not None, "start has no human-clear snap within the allowed distance")
    start_cell, start_snap_distance = start_snap
    if config.navigation_clearance_mode == "target_interaction_relaxed_stage3_sdf_collision":
        audits = []
        for source_index, option in enumerate(options):
            option_roadmap = relax_target_interaction_approaches(
                base_roadmap, [option], config
            )
            row = audit_approaches([option], option_roadmap, start_cell, config)[0]
            row["source_option_index"] = source_index
            audits.append(row)
    else:
        audits = audit_approaches(options, roadmap, start_cell, config)
    reachable = [row for row in audits if row["fresh_reachable"] is True]
    require(reachable, "none of the Stage-1 approach options is reachable on fresh occupancy")
    if config.navigation_clearance_mode == "target_interaction_relaxed_stage3_sdf_collision":
        interaction_eligible = [
            row for row in reachable
            if float(row["snapped_goal_clearance_m"])
            >= config.interaction_terminal_min_clearance_m - 1e-9
            and float(row["requested_cell_clearance_m"])
            >= config.interaction_terminal_min_clearance_m - 1e-9
        ]
        require(interaction_eligible, "no target approach satisfies the relaxed terminal clearance gate")
        surface = candidate.get("surface")
        require(isinstance(surface, Mapping), "target candidate lacks surface geometry")
        surface_center = finite_vector(
            surface.get("centre_world_xyz_zup_m"), 3, "target surface centre"
        )[:2]
        selected = min(
            interaction_eligible,
            key=lambda row: (
                float(np.linalg.norm(
                    np.asarray(row["requested_approach_world_xy_m"], dtype=np.float64)
                    - surface_center
                )),
                float(row["fresh_four_connected_distance_m"]),
                float(row["snap_distance_m"]),
                int(row["source_option_index"]),
            ),
        )
        selection_rule = (
            "target-completion first: terminal clearance gate, minimum approach-to-surface distance, "
            "then fresh route distance and source index"
        )
        selected_option = options[int(selected["source_option_index"])]
        require(
            str(selected_option.get("option_id")) == selected["option_id"],
            "selected option index/ID drift",
        )
        roadmap = relax_target_interaction_approaches(
            base_roadmap, [selected_option], config
        )
    else:
        selected = min(
            reachable,
            key=lambda row: (
                float(row["fresh_four_connected_distance_m"]),
                float(row["snap_distance_m"]),
                int(row["source_option_index"]),
            ),
        )
        selection_rule = "minimum fresh four-connected distance, snap distance, source index"
    goal_cell = tuple(int(value) for value in selected["snapped_goal_cell"])

    request_resolved = request_path.expanduser().resolve()
    receipt_resolved = receipt_path.expanduser().resolve()
    npz_path = request_resolved.with_suffix(".npz")
    atomic_npz(
        npz_path,
        schema=np.asarray(P504_INPUT_SCHEMA),
        free_human_cells=np.asarray(roadmap["free"], dtype=np.bool_),
        obstacle_cells=np.asarray(roadmap["obstacles"], dtype=np.bool_),
        clearance_m=np.asarray(roadmap["clearance"], dtype=np.float32),
        target_interaction_relaxed_cells=np.asarray(
            roadmap["target_interaction_relaxed"], dtype=np.bool_
        ),
        lower_world_xy_m=np.asarray(config.lower_world_xyz_m[:2], dtype=np.float32),
        cell_resolution_m=np.asarray(config.cell_resolution_m, dtype=np.float32),
        start_cell=np.asarray(start_cell, dtype=np.int32),
        goal_cell=np.asarray(goal_cell, dtype=np.int32),
    )
    npz_record = artifact(npz_path)
    request = {
        "schema": P504_INPUT_SCHEMA,
        "coordinate_system": "world_zup_metric_xy",
        "human_radius_clearance_m": config.human_radius_clearance_m,
        "hard_navigation_clearance_m": config.effective_hard_navigation_clearance_m,
        "navigation_clearance_mode": config.navigation_clearance_mode,
        "target_interaction_relaxation_radius_m": (
            config.target_interaction_relaxation_radius_m
            if config.navigation_clearance_mode == "target_interaction_relaxed_stage3_sdf_collision"
            else None
        ),
        "target_approach_min_clearance_m": config.target_approach_min_clearance_m,
        "interaction_terminal_min_clearance_m": config.interaction_terminal_min_clearance_m,
        "input_npz": npz_record,
        "start_world_xy_m": start.astype(float).tolist(),
        "goal_world_xy_m": selected["requested_approach_world_xy_m"],
        "start_cell": list(start_cell),
        "goal_cell": list(goal_cell),
        "compiler_provenance": {
            "schema": RECEIPT_SCHEMA,
            "stage1_sha256": sha256_file(stage1_resolved),
            "occupancy_sha256": sha256_file(occupancy_resolved),
            "selected_candidate_id": str(candidate.get("candidate_id")),
            "selected_approach_option_id": selected["option_id"],
            "fresh_reachability_and_snapping": True,
            "legacy_p480_route_reused": False,
            "start_source": resolved_start_source,
        },
        "required_output_contract": {
            "schema": POLYGON_NAVMESH_OUTPUT_SCHEMA,
            "source_input_npz_sha256": npz_record["sha256"],
            "route_world_xy_m": "at least two finite [x,y] points",
            "route_must_remain_in_free_human_cells": True,
        },
    }
    atomic_json(request_resolved, request)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "fresh_roadmap_request_compiled",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stage1": artifact(stage1_resolved),
            "lingo_occupancy": artifact(occupancy_resolved),
            "start_world_xy_m": start.astype(float).tolist(),
            "start_source": resolved_start_source,
        },
        "roadmap": {
            "coordinate_system": "world_zup_metric_xy",
            "shape": list(np.asarray(roadmap["free"]).shape),
            "lower_world_xyz_m": list(config.lower_world_xyz_m),
            "cell_resolution_m": config.cell_resolution_m,
            "body_height_range_m": list(config.body_height_range_m),
            "human_radius_clearance_m": config.human_radius_clearance_m,
            "hard_navigation_clearance_m": config.effective_hard_navigation_clearance_m,
            "navigation_clearance_mode": config.navigation_clearance_mode,
            "target_interaction_relaxation_radius_m": (
                config.target_interaction_relaxation_radius_m
                if config.navigation_clearance_mode == "target_interaction_relaxed_stage3_sdf_collision"
                else None
            ),
            "target_approach_min_clearance_m": config.target_approach_min_clearance_m,
            "interaction_terminal_min_clearance_m": config.interaction_terminal_min_clearance_m,
            "free_human_cell_count": int(np.count_nonzero(roadmap["free"])),
            "obstacle_cell_count": int(np.count_nonzero(roadmap["obstacles"])),
            "target_interaction_relaxed_cell_count": int(
                np.count_nonzero(roadmap["target_interaction_relaxed"])
            ),
        },
        "start_audit": {
            "source": resolved_start_source,
            "requested_world_xy_m": start.astype(float).tolist(),
            "snapped_cell": list(start_cell),
            "snapped_world_xy_m": cell_center(start_cell, config).astype(float).tolist(),
            "snap_distance_m": start_snap_distance,
            "snapped_clearance_m": float(np.asarray(roadmap["clearance"])[start_cell]),
        },
        "candidate_id": str(candidate.get("candidate_id")),
        "target_surface": candidate.get("surface"),
        "approach_audits": audits,
        "selected_approach_option_id": selected["option_id"],
        "selection_rule": selection_rule,
        "route_provenance": {
            "stage1_final_route_was_null": True,
            "legacy_p480_route_polyline_read": False,
            "legacy_p480_route_keypoints_read": False,
            "legacy_p480_route_used_for_ranking": False,
            "final_route_emitted_by_compiler": False,
        },
        "p504_request": artifact(request_resolved),
    }
    atomic_json(receipt_resolved, receipt)
    return receipt

