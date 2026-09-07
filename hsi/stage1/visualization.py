from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
from PIL import Image,ImageDraw

from . import _visual_geometry as base

KEY_NODES_SCHEMA = "p510.stage1_root_navmesh_key_nodes.v1"

KEY_NODES_STATUS = "verified_root_navmesh_key_nodes_ready"

OUTPUT_SCHEMA = "p510.navmesh_key_nodes_visualization.v1"

class KeyNodeVisualizationError(RuntimeError):
    """Raised when the route/key-node bundle is not mutually hash bound."""

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise KeyNodeVisualizationError(message)

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
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }

def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "%s JSON root must be an object" % label)
    return value

def canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def finite_xy(value: Any, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    require(point.shape == (2,), "%s must be [2]" % label)
    require(np.isfinite(point).all(), "%s contains non-finite values" % label)
    return point

def _validate_keynode_route_artifact(
    keynodes: Mapping[str, Any], route_path: Path, route: Mapping[str, Any]
) -> None:
    inputs = keynodes.get("inputs")
    require(isinstance(inputs, Mapping), "keynodes lacks inputs")
    source_route = inputs.get("navmesh_route")
    require(isinstance(source_route, Mapping), "keynodes lacks inputs.navmesh_route artifact")
    resolved_route = route_path.resolve(strict=True)
    require(source_route.get("sha256") == sha256_file(resolved_route), "keynodes/source route SHA-256 drift")
    require(source_route.get("bytes") == resolved_route.stat().st_size, "keynodes/source route byte count drift")
    recorded_path = Path(str(source_route.get("path", ""))).expanduser()
    if not recorded_path.is_absolute():
        recorded_path = route_path.resolve().parent / recorded_path
    require(recorded_path.resolve(strict=True) == resolved_route, "keynodes points to a different NavMesh route")
    execution = keynodes.get("navmesh_execution_route")
    require(isinstance(execution, Mapping), "keynodes lacks navmesh_execution_route")
    require(
        execution.get("route_world_xy_float64_sha256")
        == route.get("route_world_xy_float64_sha256"),
        "keynodes dense-route byte hash differs from route",
    )
    require(
        execution.get("authoritative_source_field") == "inputs.navmesh_route -> route_world_xy_m",
        "keynodes does not name the dense route as execution authority",
    )

def validate_semantic_nodes(
    keynodes: Mapping[str, Any], dense: np.ndarray
) -> tuple[list[Mapping[str, Any]], list[int]]:
    summary = keynodes.get("semantic_key_node_summary")
    require(isinstance(summary, Mapping), "keynodes lacks semantic_key_node_summary")
    require(summary.get("direct_linear_interpolation_allowed") is False,
            "semantic nodes incorrectly allow direct interpolation")
    nodes = summary.get("nodes")
    require(isinstance(nodes, list), "semantic nodes must be a list")
    require(int(summary.get("node_count", -1)) == len(nodes), "semantic node count drift")
    require(2 <= len(nodes) <= 32, "adaptive semantic node count must be between two and thirty-two")
    if len(nodes) == 2:
        require("START_CURRENT_ROOT" in nodes[0].get("event_labels", [])
                and "FINAL_APPROACH_START" in nodes[0].get("event_labels", [])
                and "NAV_GOAL_APPROACH" in nodes[-1].get("event_labels", []),
                "two-node route must start within its final approach and preserve the goal")
    indices: list[int] = []
    for sequence_index, node in enumerate(nodes):
        require(isinstance(node, Mapping), "semantic node %d is malformed" % sequence_index)
        require(int(node.get("sequence_index", -1)) == sequence_index,
                "semantic sequence index drift at node %d" % sequence_index)
        dense_index = int(node.get("dense_route_index", -1))
        require(0 <= dense_index < len(dense), "semantic dense index is out of range")
        coordinate = finite_xy(node.get("world_xy_m"), "semantic node coordinate")
        require(np.array_equal(coordinate, dense[dense_index]),
                "semantic node coordinate differs from its dense-route index")
        indices.append(dense_index)
    require(indices == sorted(set(indices)), "semantic dense indices are not strictly increasing")
    require(indices[0] == 0 and indices[-1] == len(dense) - 1,
            "semantic nodes do not preserve both dense-route endpoints")

    transitions = summary.get("transitions")
    require(isinstance(transitions, list) and len(transitions) == len(nodes) - 1,
            "semantic transition count drift")
    for transition_index, transition in enumerate(transitions):
        require(isinstance(transition, Mapping), "semantic transition is malformed")
        start, finish = indices[transition_index], indices[transition_index + 1]
        require(int(transition.get("from_dense_route_index", -1)) == start,
                "semantic transition start index drift")
        require(int(transition.get("to_dense_route_index_inclusive", -1)) == finish,
                "semantic transition finish index drift")
        require(transition.get("dense_route_slice") == [start, finish + 1],
                "semantic dense-route slice drift")
        require(transition.get("must_follow_bound_navmesh_slice") is True,
                "semantic transition does not require its dense route slice")
        require(transition.get("unchecked_straight_chord_forbidden") is True,
                "semantic transition permits an unchecked shortcut")
    return nodes, indices

def validate_safe_controls(
    keynodes: Mapping[str, Any], dense: np.ndarray
) -> tuple[np.ndarray | None, list[int]]:
    controls = keynodes.get("collision_safe_control_polyline")
    if controls is None:
        return None, []
    require(isinstance(controls, Mapping), "collision_safe_control_polyline is malformed")
    indices_raw = controls.get("dense_route_indices")
    coordinates_raw = controls.get("nodes_world_xy_m")
    require(isinstance(indices_raw, list) and isinstance(coordinates_raw, list),
            "safe controls lack indices or coordinates")
    require(int(controls.get("node_count", -1)) == len(indices_raw) == len(coordinates_raw),
            "safe-control node count drift")
    require(len(indices_raw) >= 2, "safe-control polyline is too short")
    indices = [int(value) for value in indices_raw]
    require(indices == sorted(set(indices)), "safe-control dense indices are not strictly increasing")
    require(indices[0] == 0 and indices[-1] == len(dense) - 1,
            "safe controls do not preserve dense-route endpoints")
    coordinates = np.asarray(coordinates_raw, dtype=np.float64)
    require(coordinates.shape == (len(indices), 2) and np.isfinite(coordinates).all(),
            "safe-control coordinates are malformed")
    require(np.array_equal(coordinates, dense[np.asarray(indices, dtype=np.int64)]),
            "safe-control coordinate differs from its dense-route index")
    checks = controls.get("segment_checks")
    require(isinstance(checks, list) and len(checks) == len(indices) - 1,
            "safe-control segment-check count drift")
    require(all(isinstance(row, Mapping) and row.get("all_samples_human_free") is True for row in checks),
            "a safe-control chord lacks a passing collision check")
    require(controls.get("direct_linear_interpolation_allowed") is True,
            "safe controls are not certified for direct interpolation")
    return coordinates, indices

def load_and_validate(
    stage1_path: Path,
    request_path: Path,
    route_path: Path,
    keynodes_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray], list[Mapping[str, Any]], np.ndarray | None]:
    stage1, request, route, arrays = base.load_bound_inputs(stage1_path, request_path, route_path)
    keynodes = read_json(keynodes_path, "keynodes")
    require(keynodes.get("schema") == KEY_NODES_SCHEMA, "wrong keynodes schema")
    require(keynodes.get("status") == KEY_NODES_STATUS, "keynodes status is not verified")
    recorded_payload_hash = keynodes.get("receipt_payload_sha256")
    require(isinstance(recorded_payload_hash, str), "keynodes lacks canonical payload hash")
    canonical_payload = dict(keynodes)
    canonical_payload.pop("receipt_payload_sha256")
    require(canonical_hash(canonical_payload) == recorded_payload_hash, "keynodes canonical payload drift")
    gates = keynodes.get("verified_gates")
    require(isinstance(gates, Mapping), "keynodes lacks verified_gates")
    require(gates.get("dense_route_remains_execution_authority") is True,
            "dense route is not declared execution authority")
    require(gates.get("semantic_slices_cover_full_route_without_gap") is True,
            "semantic slices were not verified")
    _validate_keynode_route_artifact(keynodes, route_path, route)
    dense = np.asarray(arrays["route_world_xy_m"], dtype=np.float64)
    semantic_nodes, _semantic_indices = validate_semantic_nodes(keynodes, dense)
    controls, _control_indices = validate_safe_controls(keynodes, dense)
    return stage1, request, route, keynodes, arrays, semantic_nodes, controls

def _numbered_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    number: int,
    *,
    radius: int,
) -> None:
    x, y = xy
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(250, 204, 21, 255),
        outline=(32, 35, 40, 255),
        width=max(2, radius // 3),
    )
    label = str(number)
    box = draw.textbbox((0, 0), label)
    draw.text(
        (x - (box[2] - box[0]) / 2.0, y - (box[3] - box[1]) / 2.0 - 1.0),
        label,
        fill=(20, 23, 28, 255),
    )

def render(
    stage1_path: Path,
    request_path: Path,
    route_path: Path,
    keynodes_path: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    scale: int = 3,
) -> dict[str, Any]:
    require(1 <= scale <= 12, "scale must be between 1 and 12")
    stage1, request, route, keynodes, arrays, semantic_nodes, controls = load_and_validate(
        stage1_path, request_path, route_path, keynodes_path
    )
    candidate_id, surface = base.target_surface(stage1)
    free = np.asarray(arrays["free_human_cells"], dtype=np.bool_)
    obstacles = np.asarray(arrays["obstacle_cells"], dtype=np.bool_)
    target_relaxed = np.asarray(
        arrays.get("target_interaction_relaxed_cells", np.zeros_like(free)),
        dtype=np.bool_,
    )
    require(target_relaxed.shape == free.shape, "target-relaxed mask shape differs from roadmap")
    require(not np.any(target_relaxed & obstacles), "target-relaxed mask overlaps raw occupancy")
    lower = np.asarray(arrays["lower_world_xy_m"], dtype=np.float64)
    resolution = float(np.asarray(arrays["cell_resolution_m"]).item())
    dense = np.asarray(arrays["route_world_xy_m"], dtype=np.float64)
    nx, ny = free.shape

    clearance_buffer = (~obstacles) & (~free)
    base_rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    # Visualization shows raw occupancy only. Clearance-rejected raw-free
    # cells remain non-walkable internally but receive no buffer overlay.
    base_rgb[:, :, :] = (242, 246, 243)
    base_rgb[obstacles.T[::-1]] = (32, 38, 46)
    # The generic clearance buffer remains hidden.  Only the grounded target's
    # explicitly walkable interaction-entry cells are shown, so the picture can
    # demonstrate that the route enters (rather than avoids) this local region.
    base_rgb[target_relaxed.T[::-1]] = (166, 174, 184)
    map_image = Image.fromarray(base_rgb, mode="RGB").resize(
        (nx * scale, ny * scale), Image.Resampling.NEAREST
    )
    header_height = 136
    canvas = Image.new("RGB", (map_image.width, map_image.height + header_height), (18, 25, 34))
    canvas.paste(map_image, (0, header_height))
    draw = ImageDraw.Draw(canvas, "RGBA")

    def pixel(point: Sequence[float]) -> tuple[float, float]:
        xy = np.asarray(point, dtype=np.float64)
        return (
            float((xy[0] - lower[0]) / resolution * scale),
            float(header_height + (ny - (xy[1] - lower[1]) / resolution) * scale),
        )

    bounds = np.asarray(surface.get("bounds_world_zup_m"), dtype=np.float64)
    require(bounds.shape == (2, 3) and np.isfinite(bounds).all(), "target surface bounds are invalid")
    first, second = pixel(bounds[0, :2]), pixel(bounds[1, :2])
    draw.rectangle(
        (min(first[0], second[0]), min(first[1], second[1]),
         max(first[0], second[0]), max(first[1], second[1])),
        fill=(37, 204, 126, 75),
        outline=(17, 145, 86, 255),
        width=max(2, scale),
    )
    centre = np.asarray(surface.get("centre_world_xyz_zup_m"), dtype=np.float64)
    require(centre.shape == (3,) and np.isfinite(centre).all(), "target surface centre is invalid")
    cx, cy = pixel(centre[:2])
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(17, 145, 86, 255), outline="white")

    # Safe control chords are diagnostic; draw them first and thinner so the
    # authoritative dense execution route remains visually dominant.
    if controls is not None:
        control_pixels = [pixel(point) for point in controls]
        draw.line(control_pixels, fill=(20, 184, 202, 150), width=max(1, scale - 1), joint="curve")
        radius = max(3, scale + 1)
        for x, y in control_pixels:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(20, 184, 202, 255), outline=(225, 252, 255, 255), width=1,
            )

    draw.line(
        [pixel(point) for point in dense],
        fill=(238, 65, 73, 255),
        width=max(3, scale * 2),
        joint="curve",
    )

    semantic_radius = max(7, scale * 3)
    for node in semantic_nodes:
        _numbered_marker(
            draw,
            pixel(node["world_xy_m"]),
            int(node["sequence_index"]) + 1,
            radius=semantic_radius,
        )

    control_count = 0 if controls is None else len(controls)
    draw.text(
        (14, 10),
        "DENSE ROUTE = EXECUTION AUTHORITY | semantic node count is adaptive (%d here)" % len(semantic_nodes),
        fill=(255, 245, 238, 255),
    )
    draw.text(
        (14, 38),
        "MAP  off-white: raw free | gray: target interaction entry (walkable) | dark: raw occupied",
        fill=(203, 214, 226, 255),
    )
    draw.text(
        (14, 60),
        "OVERLAY  red: dense route (%d) | yellow 1-%d: adaptive nodes | cyan: %d safe controls | green: %s"
        % (len(dense), len(semantic_nodes), control_count, candidate_id),
        fill=(203, 214, 226, 255),
    )
    draw.text(
        (14, 82),
        "Semantic transitions follow bound dense slices; unchecked yellow-to-yellow chords are forbidden.",
        fill=(250, 204, 21, 255),
    )
    draw.text(
        (14, 106),
        "route=%.3f m   clearance>=%.3f m   target=%s"
        % (
            float(route["validation"]["route_length_m"]),
            float(route["validation"]["minimum_sampled_clearance_m"]),
            candidate_id,
        ),
        fill=(154, 166, 181, 255),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)

    receipt = {
        "schema": OUTPUT_SCHEMA,
        "status": "verified_key_nodes_visualized_over_authoritative_dense_route",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stage1": artifact(stage1_path),
            "roadmap_request": artifact(request_path),
            "navmesh_route": artifact(route_path),
            "keynodes": artifact(keynodes_path),
        },
        "target_candidate_id": candidate_id,
        "dense_route_point_count": int(len(dense)),
        "semantic_node_count": int(len(semantic_nodes)),
        "safe_control_count": int(control_count),
        "internal_map_class_counts": {
            "navigation_contract_walkable": int(np.count_nonzero(free)),
            "target_interaction_relaxed_walkable": int(np.count_nonzero(target_relaxed)),
            "clearance_rejected_raw_free_not_separately_visualized": int(np.count_nonzero(clearance_buffer)),
            "raw_occupied_body_height_footprint": int(np.count_nonzero(obstacles)),
        },
        "display_map_class_counts": {
            "off_white_raw_free_non_target_relaxed": int(
                np.count_nonzero((~obstacles) & (~target_relaxed))
            ),
            "gray_target_interaction_relaxed_walkable": int(np.count_nonzero(target_relaxed)),
            "dark_raw_occupied": int(np.count_nonzero(obstacles)),
        },
        "color_legend": {
            "off_white": "raw-free cells outside the explicitly relaxed target interaction entry",
            "gray": "grounded target-interaction cells admitted to NavMesh despite sub-nominal clearance",
            "dark": "raw occupied footprint in the body-height band",
            "red": "authoritative dense NavMesh execution route",
            "yellow": "adaptive semantic conditioning nodes",
            "cyan": "independently collision-checked control chords",
            "green": "target interaction surface",
        },
        "clearance_buffer_visualized_as_separate_class": False,
        "target_interaction_relaxation_visualized_as_separate_class": bool(
            np.any(target_relaxed)
        ),
        "bindings_verified": {
            "keynodes_schema_and_status": True,
            "keynodes_source_route_sha256": True,
            "keynodes_dense_route_byte_hash": True,
            "semantic_coordinates_equal_dense_route_indices": True,
            "semantic_slices_cover_route_without_unchecked_shortcuts": True,
            "safe_control_coordinates_equal_dense_route_indices": controls is not None,
            "dense_route_remains_execution_authority": True,
        },
        "claim_boundary": (
            "Yellow semantic nodes are conditioning summaries. The red dense NavMesh route, not "
            "straight yellow-node chords, remains the collision-authoritative execution path."
        ),
        "visualization": artifact(output_path),
    }
    atomic_json(receipt_path, receipt)
    return receipt
