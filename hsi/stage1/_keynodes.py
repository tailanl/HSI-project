"""Extract a small, hash-bound key-node summary *after* NavMesh planning.

The verified dense NavMesh route remains the collision-authoritative path.  A
small semantic sequence is selected from that route for motion conditioning;
each semantic transition points back to its exact dense-route slice, so a
downstream module must not replace the slice with an unchecked straight line.

For diagnostics this module also emits the smallest deterministic sequence of
farthest-visible raw NavMesh controls.  Those control-to-control chords are
independently raster checked, but they are separate from the semantic summary.
"""

from __future__ import annotations

import argparse

from datetime import datetime, timezone

import hashlib

import json

import math

import os

from pathlib import Path

import sys

from typing import Any, Mapping, Sequence

import numpy as np

from . import _roadmap as compiler

from . import _navmesh as backend

STAGE1_SCHEMA = "p508.lingo_scene_first_stage1.v1"

OUTPUT_SCHEMA = "p510.stage1_root_navmesh_key_nodes.v1"

class KeyNodeError(RuntimeError):
    """Raised when a route cannot become a trustworthy key-node handoff."""

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise KeyNodeError(message)

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

def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "%s root is not an object" % label)
    return value

def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def finite_points(value: Any, label: str, minimum: int = 2) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    require(
        points.ndim == 2 and points.shape[0] >= minimum and points.shape[1] == 2,
        "%s must be [N>=%d,2]" % (label, minimum),
    )
    require(np.isfinite(points).all(), "%s contains non-finite coordinates" % label)
    return points

def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    require(np.all(lengths > 0.0), "route contains duplicate consecutive points")
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(lengths)))

def wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)

def point_to_segment_distances(points: np.ndarray, start: np.ndarray, finish: np.ndarray) -> np.ndarray:
    chord = finish - start
    denominator = float(np.dot(chord, chord))
    if denominator <= 1e-20:
        return np.linalg.norm(points - start[None, :], axis=1)
    fractions = np.clip(((points - start[None, :]) @ chord) / denominator, 0.0, 1.0)
    projections = start[None, :] + fractions[:, None] * chord[None, :]
    return np.linalg.norm(points - projections, axis=1)

def semantic_key_indices(
    points: np.ndarray,
    node_count: int,
    *,
    human_radius_m: float,
) -> tuple[list[int], int]:
    """Return exactly ``node_count`` ordered route indices.

    Start, final-approach entry and NavMesh goal are mandatory.  Remaining
    nodes are hierarchical polyline-shape anchors selected by maximum chord
    deviation.  This is a summary only; collision safety comes from the dense
    slices retained between the selected indices.
    """

    require(5 <= node_count <= 8, "semantic node_count must be between 5 and 8")
    require(len(points) >= node_count, "route is too short for requested semantic node count")
    arc = cumulative_arclength(points)
    total = float(arc[-1])
    final_approach_distance = max(0.40, 2.0 * float(human_radius_m))
    pre_arc = max(0.0, total - final_approach_distance)
    pre_index = int(np.argmin(np.abs(arc - pre_arc)))
    pre_index = min(len(points) - 2, max(1, pre_index))
    selected = {0, pre_index, len(points) - 1}
    minimum_arc_spacing = min(0.12, total / max(2.0 * node_count, 1.0))

    while len(selected) < node_count:
        ordered = sorted(selected)
        best: tuple[float, int] | None = None
        for left, right in zip(ordered[:-1], ordered[1:]):
            if right - left <= 1:
                continue
            candidates = np.arange(left + 1, right, dtype=np.int64)
            eligible = (
                (arc[candidates] - arc[left] >= minimum_arc_spacing - 1e-12)
                & (arc[right] - arc[candidates] >= minimum_arc_spacing - 1e-12)
            )
            candidates = candidates[eligible]
            if len(candidates) == 0:
                continue
            deviations = point_to_segment_distances(points[candidates], points[left], points[right])
            local = int(np.argmax(deviations))
            proposal = (float(deviations[local]), int(candidates[local]))
            if best is None or proposal[0] > best[0] + 1e-12 or (
                abs(proposal[0] - best[0]) <= 1e-12 and proposal[1] < best[1]
            ):
                best = proposal
        if best is None:
            remaining = [index for index in range(1, len(points) - 1) if index not in selected]
            require(remaining, "cannot fill semantic key nodes")
            # Deterministic fallback for a very short/nearly straight route.
            target_arcs = np.linspace(0.0, total, node_count)
            best_index = max(
                remaining,
                key=lambda index: min(abs(float(arc[index] - target)) for target in target_arcs),
            )
            selected.add(int(best_index))
        else:
            selected.add(best[1])
    indices = sorted(selected)
    require(len(indices) == node_count, "semantic key-node count drift")
    return indices, pre_index

def rdp_key_indices(points: np.ndarray, epsilon_m: float) -> list[int]:
    """Deterministic Ramer-Douglas-Peucker indices for route-shape events."""

    require(math.isfinite(epsilon_m) and epsilon_m > 0.0, "RDP epsilon must be positive")
    selected = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        left, right = stack.pop()
        if right - left <= 1:
            continue
        candidates = np.arange(left + 1, right, dtype=np.int64)
        deviations = point_to_segment_distances(points[candidates], points[left], points[right])
        local = int(np.argmax(deviations))
        if float(deviations[local]) > epsilon_m:
            index = int(candidates[local])
            selected.add(index)
            # Reverse push order keeps the result deterministic without relying
            # on recursion depth for long routes.
            stack.append((index, right))
            stack.append((left, index))
    return sorted(selected)

def route_clearances(points: np.ndarray, request: backend.RoadmapRequest) -> np.ndarray:
    values = []
    for index, point in enumerate(points):
        cell = backend.world_to_cell(
            point,
            request.lower_world_xy_m,
            request.cell_resolution_m,
            request.free.shape,
            name="dense route point %d" % index,
        )
        require(bool(request.free[cell]), "dense route point lies outside the human-clear roadmap")
        values.append(float(request.clearance_m[cell]))
    return np.asarray(values, dtype=np.float64)

def clearance_event_indices(
    clearances: np.ndarray,
    arc: np.ndarray,
    *,
    human_radius_m: float,
    cell_resolution_m: float,
) -> list[tuple[int, str]]:
    """Detect sustained narrow-corridor entry/release events with hysteresis."""

    narrow_threshold = human_radius_m + 2.0 * cell_resolution_m
    open_threshold = human_radius_m + 4.0 * cell_resolution_m
    sustain_distance = max(0.20, 0.70 * human_radius_m)
    narrow = clearances <= narrow_threshold + 1e-12
    opened = clearances >= open_threshold - 1e-12

    def sustained(mask: np.ndarray, start: int) -> bool:
        for finish in range(start, len(mask)):
            if not bool(mask[finish]):
                return False
            if float(arc[finish] - arc[start]) >= sustain_distance - 1e-12:
                return True
        return False

    state = "open" if clearances[0] >= open_threshold else (
        "narrow" if clearances[0] <= narrow_threshold else "middle"
    )
    events: list[tuple[int, str]] = []
    for index in range(len(clearances)):
        if state != "narrow" and bool(narrow[index]) and sustained(narrow, index):
            events.append((index, "NARROW_ENTRY"))
            state = "narrow"
        elif state != "open" and bool(opened[index]) and sustained(opened, index):
            events.append((index, "CLEARANCE_RELEASE"))
            state = "open"
    return events

def adaptive_semantic_key_indices(
    points: np.ndarray,
    request: backend.RoadmapRequest,
) -> tuple[list[int], int, dict[int, list[str]], dict[str, Any]]:
    """Select an event-driven number of semantic nodes from a verified route.

    The count is intentionally not fixed.  Route-shape changes, sustained
    clearance-state changes and the final-approach boundary create nodes; long
    featureless intervals receive length anchors only when needed.
    """

    arc = cumulative_arclength(points)
    total = float(arc[-1])
    radius = float(request.human_radius_clearance_m)
    resolution = float(request.cell_resolution_m)
    final_approach_distance = max(0.50, 2.0 * radius)
    pre_arc = max(0.0, total - final_approach_distance)
    # If the whole route is already shorter than the final-approach horizon,
    # the current root itself is the honest boundary.  Do not invent a second
    # milestone one centimetre later merely to satisfy a fixed phase layout.
    pre_index = (
        0
        if total <= final_approach_distance + resolution
        else min(len(points) - 2, max(1, int(np.argmin(np.abs(arc - pre_arc)))))
    )
    rdp_epsilon = max(2.5 * resolution, 0.18 * radius)
    merge_distance = max(0.12, 0.70 * radius)
    maximum_gap = max(1.0, 3.5 * radius)

    # index -> event labels. Mandatory semantic boundaries are preferred as
    # representatives when multiple events fall in one spatial cluster.
    candidates: dict[int, set[str]] = {
        0: {"START_CURRENT_ROOT"},
        len(points) - 1: {"NAV_GOAL_APPROACH"},
    }
    candidates.setdefault(pre_index, set()).add("FINAL_APPROACH_START")
    for index in rdp_key_indices(points, rdp_epsilon)[1:-1]:
        candidates.setdefault(index, set()).add("ROUTE_SHAPE_ANCHOR")
    clearances = route_clearances(points, request)
    clearance_events = clearance_event_indices(
        clearances,
        arc,
        human_radius_m=radius,
        cell_resolution_m=resolution,
    )
    for index, label in clearance_events:
        candidates.setdefault(index, set()).add(label)

    mandatory_order = {
        "START_CURRENT_ROOT": 5,
        "NAV_GOAL_APPROACH": 5,
        "FINAL_APPROACH_START": 4,
        "CLEARANCE_RELEASE": 3,
        "NARROW_ENTRY": 3,
        "ROUTE_SHAPE_ANCHOR": 1,
    }
    clustered: list[tuple[int, set[str]]] = []
    pending: list[tuple[int, set[str]]] = []
    for index, labels in sorted(candidates.items()):
        if pending and float(arc[index] - arc[pending[0][0]]) > merge_distance + 1e-12:
            representative = max(
                pending,
                key=lambda row: (max(mandatory_order[label] for label in row[1]), -row[0]),
            )
            merged_labels = set().union(*(row[1] for row in pending))
            clustered.append((representative[0], merged_labels))
            pending = []
        pending.append((index, set(labels)))
    if pending:
        representative = max(
            pending,
            key=lambda row: (max(mandatory_order[label] for label in row[1]), -row[0]),
        )
        clustered.append((representative[0], set().union(*(row[1] for row in pending))))

    event_map = {index: set(labels) for index, labels in clustered}
    # Mandatory semantic boundaries remain at their exact route indices.
    for mandatory_index, label in ((0, "START_CURRENT_ROOT"),
            (len(points)-1, "NAV_GOAL_APPROACH"), (pre_index, "FINAL_APPROACH_START")):
        for index, labels in event_map.items():
            if index != mandatory_index:
                labels.discard(label)
        event_map.setdefault(mandatory_index, set()).add(label)
    event_map = {index: labels for index, labels in event_map.items() if labels}
    selected = sorted(event_map)
    # Long straight intervals still need timing/progress anchors.  These are
    # selected on the original dense route, not used as unchecked chords.
    length_anchors: list[int] = []
    for left, right in zip(selected[:-1], selected[1:]):
        distance = float(arc[right] - arc[left])
        segment_count = max(1, int(math.ceil(distance / maximum_gap)))
        for part in range(1, segment_count):
            target = float(arc[left] + distance * part / segment_count)
            index = int(np.argmin(np.abs(arc - target)))
            if left < index < right:
                length_anchors.append(index)
    for index in length_anchors:
        event_map.setdefault(index, set()).add("ROUTE_LENGTH_ANCHOR")
    selected = sorted(event_map)

    # Publish dynamic planning targets for audit; they are not a forced output
    # count. Complex mandatory events may legitimately exceed the preference.
    low_count = max(3, int(math.ceil(total / max(1.0, 3.5 * radius))) + 1)
    preferred_count = int(math.ceil(total / max(0.55, 2.0 * radius))) + 1
    high_count = min(12, int(math.ceil(total / max(0.35, 1.25 * radius))) + 1)
    metadata = {
        "mode": "adaptive_route_events_and_length",
        "rdp_epsilon_m": rdp_epsilon,
        "event_merge_distance_m": merge_distance,
        "maximum_unanchored_route_gap_m": maximum_gap,
        "final_approach_distance_m": final_approach_distance,
        "narrow_clearance_threshold_m": radius + 2.0 * resolution,
        "open_clearance_threshold_m": radius + 4.0 * resolution,
        "minimum_clearance_state_duration_m": max(0.20, 0.70 * radius),
        "dynamic_count_guidance": {
            "low": low_count,
            "preferred": preferred_count,
            "high": high_count,
            "actual": len(selected),
            "hard_fixed_count_used": False,
        },
        "hierarchical_segmentation_recommended": len(selected) > high_count,
    }
    labels = {index: sorted(event_map[index], key=lambda value: (-mandatory_order.get(value, 0), value)) for index in selected}
    require(selected[0] == 0 and selected[-1] == len(points) - 1, "adaptive nodes lost route endpoints")
    require(pre_index in selected, "adaptive nodes lost final-approach boundary")
    return selected, pre_index, labels, metadata

def route_index_at_distance(
    arc: np.ndarray,
    origin_index: int,
    distance_m: float,
    direction: int,
) -> int:
    target = float(arc[origin_index]) + float(direction) * distance_m
    if direction < 0:
        candidates = np.arange(0, origin_index + 1)
    else:
        candidates = np.arange(origin_index, len(arc))
    return int(candidates[np.argmin(np.abs(arc[candidates] - target))])

def tangent_yaws(points: np.ndarray, arc: np.ndarray, index: int, window_m: float = 0.10) -> tuple[float, float]:
    before = route_index_at_distance(arc, index, window_m, -1)
    after = route_index_at_distance(arc, index, window_m, +1)
    incoming_vector = points[index] - points[before] if before != index else points[min(index + 1, len(points) - 1)] - points[index]
    outgoing_vector = points[after] - points[index] if after != index else points[index] - points[max(0, index - 1)]
    incoming = math.atan2(float(incoming_vector[1]), float(incoming_vector[0]))
    outgoing = math.atan2(float(outgoing_vector[1]), float(outgoing_vector[0]))
    return float(incoming), float(outgoing)

def segment_is_human_free(
    start: np.ndarray,
    finish: np.ndarray,
    request: backend.RoadmapRequest,
    sample_spacing_m: float,
) -> tuple[bool, float, int]:
    distance = float(np.linalg.norm(finish - start))
    subdivisions = max(1, int(math.ceil(distance / sample_spacing_m)))
    minimum_clearance = math.inf
    sample_count = 0
    for sample_index in range(subdivisions + 1):
        point = start + (finish - start) * (float(sample_index) / subdivisions)
        try:
            cell = backend.world_to_cell(
                point,
                request.lower_world_xy_m,
                request.cell_resolution_m,
                request.free.shape,
                name="control chord",
            )
        except backend.ContractError:
            return False, 0.0, sample_count
        clearance = float(request.clearance_m[cell])
        if not backend.cell_meets_navigation_contract(request, cell):
            return False, clearance, sample_count + 1
        minimum_clearance = min(minimum_clearance, clearance)
        sample_count += 1
    return True, minimum_clearance, sample_count

def map_raw_to_dense(raw: np.ndarray, dense: np.ndarray, tolerance_m: float = 1e-8) -> list[int]:
    indices: list[int] = []
    cursor = 0
    for raw_index, point in enumerate(raw):
        distances = np.linalg.norm(dense[cursor:] - point[None, :], axis=1)
        local = int(np.argmin(distances))
        require(float(distances[local]) <= tolerance_m, "raw corner %d is absent from dense route" % raw_index)
        cursor += local
        indices.append(cursor)
    require(all(a < b for a, b in zip(indices[:-1], indices[1:])), "raw/dense route order drift")
    return indices

def collision_safe_control_indices(
    raw: np.ndarray,
    dense: np.ndarray,
    request: backend.RoadmapRequest,
) -> tuple[list[int], list[dict[str, Any]]]:
    raw_dense_indices = map_raw_to_dense(raw, dense)
    sample_spacing = min(0.005, 0.25 * request.cell_resolution_m)
    selected_raw = [0]
    checks: list[dict[str, Any]] = []
    current = 0
    while current < len(raw) - 1:
        chosen: tuple[int, float, int] | None = None
        for candidate in range(len(raw) - 1, current, -1):
            safe, minimum_clearance, sample_count = segment_is_human_free(
                raw[current], raw[candidate], request, sample_spacing
            )
            if safe:
                chosen = (candidate, minimum_clearance, sample_count)
                break
        require(chosen is not None, "even the next raw NavMesh edge is not human-free")
        candidate, minimum_clearance, sample_count = chosen
        checks.append({
            "from_raw_index": current,
            "to_raw_index": candidate,
            "minimum_sampled_clearance_m": minimum_clearance,
            "collision_sample_count": sample_count,
            "all_samples_human_free": True,
        })
        selected_raw.append(candidate)
        current = candidate
    return [raw_dense_indices[index] for index in selected_raw], checks

def collision_safe_dense_control_indices(
    dense: np.ndarray,
    request: backend.RoadmapRequest,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Simplify an already verified dense corridor without restoring raw corners.

    A polygon-corridor interiorizer deliberately moves boundary-ambiguous raw
    ``pynavmesh`` corners onto free cell centres.  In that case the raw corners
    are lineage evidence, not safe control vertices.  This routine greedily
    selects the farthest visible *dense-route* point and independently checks
    every emitted chord against the unchanged navigation contract.  It never
    searches outside or recomputes the authoritative dense route.
    """

    require(len(dense) >= 2, "dense corridor route must contain at least two points")
    sample_spacing = min(0.005, 0.25 * request.cell_resolution_m)
    selected = [0]
    checks: list[dict[str, Any]] = []
    current = 0
    while current < len(dense) - 1:
        chosen: tuple[int, float, int] | None = None
        for candidate in range(len(dense) - 1, current, -1):
            safe, minimum_clearance, sample_count = segment_is_human_free(
                dense[current], dense[candidate], request, sample_spacing
            )
            if safe:
                chosen = (candidate, minimum_clearance, sample_count)
                break
        require(chosen is not None, "even the next verified dense-route edge is not human-free")
        candidate, minimum_clearance, sample_count = chosen
        checks.append({
            "from_dense_route_index": current,
            "to_dense_route_index": candidate,
            "minimum_sampled_clearance_m": minimum_clearance,
            "collision_sample_count": sample_count,
            "all_samples_human_free": True,
        })
        selected.append(candidate)
        current = candidate
    return selected, checks

def validate_polygon_corridor_interiorization(
    route_json: Mapping[str, Any],
    raw: np.ndarray,
) -> Mapping[str, Any] | None:
    """Return a verified current-only corridor audit, or ``None`` for P508 routes."""

    audit = route_json.get("polygon_corridor_interiorization")
    backend_record = route_json.get("backend")
    require(isinstance(backend_record, Mapping), "route backend record is missing")
    backend_marks_corridor = backend_record.get("polygon_corridor_interiorization_used") is True
    if audit is None:
        require(not backend_marks_corridor, "route claims corridor interiorization without an audit")
        return None
    require(isinstance(audit, Mapping), "polygon corridor audit is malformed")
    require(backend_marks_corridor, "polygon corridor audit is not declared by the route backend")
    require(
        audit.get("schema") == "p523.current_only_polygon_corridor_interiorization.v1"
        and audit.get("status") == "verified_local_polygon_corridor_interiorization",
        "polygon corridor interiorization is not verified",
    )
    for field in (
        "all_selected_cells_free",
        "all_selected_cells_touch_raw_polygon_route",
        "cell_centres_used",
        "cell_sequence_4_connected",
        "external_pynavmesh_remains_route_authority",
    ):
        require(audit.get(field) is True, "polygon corridor audit failed: %s" % field)
    for field in (
        "frontier_or_priority_queue_used",
        "global_grid_search_used",
        "heuristic_search_used",
        "native_grid_astar_used",
        "shortest_path_recomputed",
    ):
        require(audit.get(field) is False, "polygon corridor audit used forbidden search: %s" % field)
    require(
        audit.get("raw_corner_route_world_xy_float64_sha256") == backend.route_sha256(raw),
        "polygon corridor raw-route lineage hash drift",
    )
    require(
        backend_record.get("route_authority")
        == "authenticated_p508_pynavmesh_pathfinder_raw_corner_route",
        "polygon corridor lost authenticated pynavmesh authority",
    )
    return audit

def selected_target_contract(stage1: Mapping[str, Any], request_json: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    candidates = stage1.get("candidate_surfaces")
    require(isinstance(candidates, list) and len(candidates) == 1, "Stage1 target surface is not unique")
    candidate = candidates[0]
    require(isinstance(candidate, Mapping), "Stage1 candidate is malformed")
    surface = candidate.get("surface")
    require(isinstance(surface, Mapping), "Stage1 candidate lacks surface")
    selected_id = str(request_json.get("compiler_provenance", {}).get("selected_approach_option_id", ""))
    matches = [row for row in candidate.get("approach_options", []) if str(row.get("option_id")) == selected_id]
    require(len(matches) == 1, "selected approach option is absent or ambiguous")
    return str(candidate.get("candidate_id", "")), surface, matches[0]

def run(
    stage1_path: Path,
    request_path: Path,
    route_path: Path,
    current_state_path: Path,
    current_state_receipt_path: Path,
    output_path: Path,
    *,
    node_count: int | None = None,
) -> dict[str, Any]:
    stage1_resolved = stage1_path.resolve(strict=True)
    request_resolved = request_path.resolve(strict=True)
    route_resolved = route_path.resolve(strict=True)
    stage1 = read_json(stage1_resolved, "Stage1")
    request_json = read_json(request_resolved, "roadmap request")
    route_json = read_json(route_resolved, "NavMesh route")
    require(stage1.get("schema") == STAGE1_SCHEMA, "wrong Stage1 schema")
    request = backend.load_request(request_resolved)
    backend.validate_output_payload(route_json, request)
    require(
        route_json.get("source", {}).get("request_json_sha256") == sha256_file(request_resolved),
        "route/request SHA-256 drift",
    )
    state_start, state_source = compiler.start_from_current_state(
        current_state_path, current_state_receipt_path
    )
    require(
        float(np.linalg.norm(state_start - request.requested_start_world_xy_m)) <= 1e-8,
        "NavMesh request was not generated from the current human root",
    )
    request_start_source = request_json.get("compiler_provenance", {}).get("start_source")
    require(isinstance(request_start_source, Mapping), "request lacks a bound start source")
    request_start_kind = str(request_start_source.get("kind", ""))
    if request_start_kind == "current_human_root_world_zup":
        require(
            request_start_source.get("current_state", {}).get("sha256")
            == state_source["current_state"]["sha256"],
            "request/current-state binding drift",
        )
        explicit_start_independently_matched = False
    else:
        # Some later planners intentionally accept the query root as an
        # explicit XY input so that their selector/route receipts remain
        # independent of the body-state producer.  This is safe for key-node
        # extraction only when the explicit value is marked as neither
        # learned nor guessed and independently equals the hash-bound current
        # SMPL-X root above.  The current state is still sealed in this output.
        require(
            request_start_kind == "explicit_world_xy",
            "unsupported roadmap start-source kind",
        )
        require(
            request_start_source.get("start_learned_or_guessed") is False,
            "explicit roadmap start was learned or guessed",
        )
        explicit_start_independently_matched = True

    dense = finite_points(route_json.get("route_world_xy_m"), "dense NavMesh route")
    raw = finite_points(route_json.get("raw_corner_route_world_xy_m"), "raw NavMesh route")
    corridor_audit = validate_polygon_corridor_interiorization(route_json, raw)
    arc = cumulative_arclength(dense)
    if node_count is None:
        semantic_indices, pre_index, event_labels, selection_metadata = adaptive_semantic_key_indices(
            dense, request
        )
    else:
        semantic_indices, pre_index = semantic_key_indices(
            dense, node_count, human_radius_m=request.human_radius_clearance_m
        )
        event_labels = {index: ["ROUTE_SHAPE_ANCHOR"] for index in semantic_indices}
        event_labels[0] = ["START_CURRENT_ROOT"]
        event_labels[pre_index] = ["FINAL_APPROACH_START"]
        event_labels[len(dense) - 1] = ["NAV_GOAL_APPROACH"]
        selection_metadata = {
            "mode": "explicit_fixed_count_compatibility",
            "requested_node_count": node_count,
            "hard_fixed_count_used": True,
        }
    if corridor_audit is None:
        control_indices, control_checks = collision_safe_control_indices(raw, dense, request)
        control_selection_method = (
            "greedy farthest-visible raw NavMesh corner with raster clearance recheck"
        )
        control_source = "exact_raw_corners_embedded_in_dense_route"
    else:
        control_indices, control_checks = collision_safe_dense_control_indices(dense, request)
        control_selection_method = (
            "greedy farthest-visible verified dense corridor point with raster clearance recheck"
        )
        control_source = "current_polygon_corridor_cell_centres"

    semantic_nodes: list[dict[str, Any]] = []
    for sequence_index, route_index in enumerate(semantic_indices):
        incoming, outgoing = tangent_yaws(dense, arc, route_index)
        labels = list(event_labels[route_index])
        if route_index == 0:
            node_type = "START_CURRENT_ROOT"
        elif route_index == len(dense) - 1:
            node_type = "NAV_GOAL_APPROACH"
        elif route_index == pre_index:
            node_type = "FINAL_APPROACH_START"
        elif "CLEARANCE_RELEASE" in labels:
            node_type = "CLEARANCE_RELEASE"
        elif "NARROW_ENTRY" in labels:
            node_type = "NARROW_ENTRY"
        else:
            turn = abs(wrap_angle(outgoing - incoming))
            node_type = "ROUTE_TURN_ANCHOR" if turn >= math.radians(12.0) else "ROUTE_SHAPE_ANCHOR"
        semantic_nodes.append({
            "sequence_index": sequence_index,
            "node_type": node_type,
            "event_labels": labels,
            "dense_route_index": route_index,
            "world_xy_m": dense[route_index].astype(float).tolist(),
            "route_arc_length_m": float(arc[route_index]),
            "route_progress_fraction": float(arc[route_index] / arc[-1]),
            "incoming_tangent_yaw_rad": incoming,
            "outgoing_tangent_yaw_rad": outgoing,
        })

    transitions = []
    for start_index, finish_index in zip(semantic_indices[:-1], semantic_indices[1:]):
        transitions.append({
            "from_dense_route_index": start_index,
            "to_dense_route_index_inclusive": finish_index,
            "dense_route_slice": [start_index, finish_index + 1],
            "path_length_m": float(arc[finish_index] - arc[start_index]),
            "must_follow_bound_navmesh_slice": True,
            "unchecked_straight_chord_forbidden": True,
        })
    require(transitions[0]["dense_route_slice"][0] == 0, "semantic slices do not start at route start")
    require(transitions[-1]["dense_route_slice"][1] == len(dense), "semantic slices do not end at route goal")
    require(
        all(a["dense_route_slice"][1] - 1 == b["dense_route_slice"][0] for a, b in zip(transitions[:-1], transitions[1:])),
        "semantic route slices are discontinuous",
    )

    candidate_id, surface, approach = selected_target_contract(stage1, request_json)
    surface_center = np.asarray(surface.get("centre_world_xyz_zup_m"), dtype=np.float64)
    require(surface_center.shape == (3,) and np.isfinite(surface_center).all(), "target surface centre is invalid")
    terminal_facing = float(approach.get("contact_forward_yaw_rad"))
    final_delta = dense[-1] - dense[-2]
    arrival_tangent = math.atan2(float(final_delta[1]), float(final_delta[0]))
    reorientation = wrap_angle(terminal_facing - arrival_tangent)
    nav_goal_to_surface = float(np.linalg.norm(dense[-1] - surface_center[:2]))

    result: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "status": "verified_root_navmesh_key_nodes_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_order": [
            "resolve_start_from_current_human_root",
            "snap_start_to_human_clear_roadmap",
            "run_verified_polygon_navmesh",
            "select_key_nodes_from_verified_route",
        ],
        "inputs": {
            "stage1": artifact(stage1_resolved),
            "roadmap_request": artifact(request_resolved),
            "navmesh_route": artifact(route_resolved),
            "current_state": artifact(current_state_path),
            "current_state_receipt": artifact(current_state_receipt_path),
        },
        "start": {
            "source": state_source,
            "requested_current_root_world_xy_m": state_start.astype(float).tolist(),
            "snapped_navmesh_start_world_xy_m": request.snapped_start_world_xy_m.astype(float).tolist(),
            "snapped_start_cell": list(request.snapped_start_cell),
            "start_snap_distance_m": float(np.linalg.norm(state_start - request.snapped_start_world_xy_m)),
        },
        "navmesh_execution_route": {
            "authoritative_source_field": "inputs.navmesh_route -> route_world_xy_m",
            "route_world_xy_float64_sha256": route_json["route_world_xy_float64_sha256"],
            "dense_route_point_count": len(dense),
            "raw_corner_count": len(raw),
            "route_length_m": float(arc[-1]),
            "all_segment_samples_human_free": True,
            "minimum_sampled_clearance_m": float(route_json["validation"]["minimum_sampled_clearance_m"]),
            "nominal_human_radius_m": request.human_radius_clearance_m,
            "hard_navigation_clearance_m": request.effective_hard_navigation_clearance_m,
            "navigation_clearance_mode": request.navigation_clearance_mode,
            "stage3_sdf_body_collision_required": (
                request.navigation_clearance_mode != "human_radius_hard_buffer"
            ),
        },
        "semantic_key_node_summary": {
            "node_count": len(semantic_nodes),
            "selection": selection_metadata,
            "nodes": semantic_nodes,
            "transitions": transitions,
            "role": "sparse motion-conditioning milestones over the authoritative NavMesh route",
            "direct_linear_interpolation_allowed": False,
        },
        "collision_safe_control_polyline": {
            "node_count": len(control_indices),
            "dense_route_indices": control_indices,
            "nodes_world_xy_m": [dense[index].astype(float).tolist() for index in control_indices],
            "segment_checks": control_checks,
            "selection_method": control_selection_method,
            "control_source": control_source,
            "raw_boundary_corners_reintroduced": False if corridor_audit is not None else None,
            "authoritative_route_recomputed": False,
            "direct_linear_interpolation_allowed": True,
            "body_volume_collision_guaranteed_by_stage1": (
                request.navigation_clearance_mode == "human_radius_hard_buffer"
            ),
            "stage3_sdf_recheck_required": (
                request.navigation_clearance_mode != "human_radius_hard_buffer"
            ),
        },
        "terminal_handoff": {
            "target_candidate_id": candidate_id,
            "selected_approach_option_id": str(approach.get("option_id")),
            "nav_goal_world_xy_m": dense[-1].astype(float).tolist(),
            "target_surface_center_world_xyz_m": surface_center.astype(float).tolist(),
            "nav_goal_to_surface_center_horizontal_m": nav_goal_to_surface,
            "arrival_tangent_yaw_rad": arrival_tangent,
            "terminal_body_facing_yaw_rad": terminal_facing,
            "orientation_only_reorientation_rad": reorientation,
            "nav_goal_is_contact_keypose": False,
            "stage2_or_stage3_contact_transition_required": True,
        },
        "verified_gates": {
            "start_read_from_hash_bound_current_root": True,
            "explicit_request_start_independently_matched_current_root": (
                explicit_start_independently_matched
            ),
            "future_ground_truth_used": False,
            "polygon_navmesh_ran_before_key_node_selection": True,
            "dense_route_remains_execution_authority": True,
            "semantic_slices_cover_full_route_without_gap": True,
            "control_chords_independently_clear_of_raw_occupancy": True,
            "polygon_corridor_interiorization_verified": corridor_audit is not None,
            "raw_boundary_corners_reintroduced": False if corridor_audit is not None else None,
            "nominal_body_collision_delegated_to_stage3_sdf": (
                request.navigation_clearance_mode != "human_radius_hard_buffer"
            ),
        },
        "source": artifact(Path(__file__)),
    }
    result["receipt_payload_sha256"] = canonical_hash(result)
    atomic_json(output_path.resolve(), result)
    return result

