from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
from PIL import Image,ImageDraw

STAGE1_SCHEMA = "p508.lingo_scene_first_stage1.v1"

P504_INPUT_SCHEMA = "p504.naomesh_human_roadmap_input.v1"

ROUTE_SCHEMA = "p508.pynavmesh_polygon_route_output.v1"

OUTPUT_SCHEMA = "p508.scene_navmesh_route_visualization.v1"

class VisualizationError(RuntimeError):
    """Raised when inputs are not a mutually bound verified route bundle."""

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise VisualizationError(message)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}

def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "%s JSON root must be an object" % label)
    return value

def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def finite_points(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] == 2, "%s must be [N>=2,2]" % label)
    require(np.isfinite(array).all(), "%s contains non-finite coordinates" % label)
    return array

def route_sha256(route: np.ndarray) -> str:
    data = np.ascontiguousarray(np.asarray(route, dtype="<f8"))
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()

def resolve_artifact(owner: Path, record: Any, label: str) -> Path:
    require(isinstance(record, Mapping), "%s is not an artifact record" % label)
    value = Path(str(record.get("path", ""))).expanduser()
    if not value.is_absolute():
        value = owner.parent / value
    resolved = value.resolve(strict=True)
    require(sha256_file(resolved) == record.get("sha256"), "%s SHA-256 drift" % label)
    return resolved

def load_bound_inputs(
    stage1_path: Path, request_path: Path, route_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    stage1_resolved = stage1_path.resolve(strict=True)
    request_resolved = request_path.resolve(strict=True)
    route_resolved = route_path.resolve(strict=True)
    stage1 = read_json(stage1_resolved, "Stage-1")
    request = read_json(request_resolved, "request")
    route = read_json(route_resolved, "route")
    require(stage1.get("schema") == STAGE1_SCHEMA, "wrong Stage-1 schema")
    require(request.get("schema") == P504_INPUT_SCHEMA, "wrong roadmap request schema")
    require(route.get("schema") == ROUTE_SCHEMA, "wrong NavMesh route schema")
    require(route.get("status") == "verified_polygon_navmesh_route", "route was not verified")
    backend = route.get("backend")
    require(isinstance(backend, Mapping), "route lacks backend identity")
    require(backend.get("external_polygon_navmesh") is True, "route is not external polygon NavMesh")
    require(backend.get("native_grid_astar_used") is False, "grid A* is not a NavMesh route")
    source = route.get("source")
    require(isinstance(source, Mapping), "route lacks source binding")
    require(source.get("request_json_sha256") == sha256_file(request_resolved), "route/request hash mismatch")
    npz_path = resolve_artifact(request_resolved, request.get("input_npz"), "request input_npz")
    require(source.get("input_npz_sha256") == sha256_file(npz_path), "route/NPZ hash mismatch")

    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "schema", "free_human_cells", "obstacle_cells", "clearance_m",
            "lower_world_xy_m", "cell_resolution_m", "start_cell", "goal_cell",
        }
        require(required.issubset(set(archive.files)), "roadmap NPZ lacks required arrays")
        require(str(np.asarray(archive["schema"]).item()) == P504_INPUT_SCHEMA, "NPZ schema drift")
        arrays = {name: np.asarray(archive[name]).copy() for name in required if name != "schema"}
        if "target_interaction_relaxed_cells" in archive.files:
            arrays["target_interaction_relaxed_cells"] = np.asarray(
                archive["target_interaction_relaxed_cells"], dtype=np.bool_
            ).copy()
    free = np.asarray(arrays["free_human_cells"], dtype=np.bool_)
    obstacles = np.asarray(arrays["obstacle_cells"], dtype=np.bool_)
    clearance = np.asarray(arrays["clearance_m"], dtype=np.float64)
    require(free.ndim == 2 and obstacles.shape == free.shape and clearance.shape == free.shape, "roadmap shapes disagree")
    require(not np.any(free & obstacles), "roadmap marks cells both free and occupied")

    points = finite_points(route.get("route_world_xy_m"), "route_world_xy_m")
    require(route_sha256(points) == route.get("route_world_xy_float64_sha256"), "route byte hash mismatch")
    lower = np.asarray(arrays["lower_world_xy_m"], dtype=np.float64)
    resolution = float(np.asarray(arrays["cell_resolution_m"]).item())
    require(lower.shape == (2,) and resolution > 0.0 and math.isfinite(resolution), "roadmap geometry is invalid")
    start_cell = tuple(int(value) for value in np.asarray(arrays["start_cell"]).tolist())
    goal_cell = tuple(int(value) for value in np.asarray(arrays["goal_cell"]).tolist())
    start_world = lower + (np.asarray(start_cell, dtype=np.float64) + 0.5) * resolution
    goal_world = lower + (np.asarray(goal_cell, dtype=np.float64) + 0.5) * resolution
    require(float(np.linalg.norm(points[0] - start_world)) <= resolution * 1e-4 + 1e-8, "route start differs from snapped start")
    require(float(np.linalg.norm(points[-1] - goal_world)) <= resolution * 1e-4 + 1e-8, "route goal differs from snapped goal")

    # Independently raster-check the serialized route before drawing it.
    for segment_index, (begin, end) in enumerate(zip(points[:-1], points[1:])):
        length = float(np.linalg.norm(end - begin))
        subdivisions = max(1, int(math.ceil(length / (0.25 * resolution))))
        for fraction in np.linspace(0.0, 1.0, subdivisions + 1):
            point = begin + (end - begin) * float(fraction)
            cell = np.floor((point - lower) / resolution).astype(np.int64)
            require(
                0 <= int(cell[0]) < free.shape[0] and 0 <= int(cell[1]) < free.shape[1],
                "route segment %d leaves roadmap" % segment_index,
            )
            require(bool(free[int(cell[0]), int(cell[1])]), "route segment %d enters a blocked cell" % segment_index)
    arrays["route_world_xy_m"] = points
    return stage1, request, route, arrays

def target_surface(stage1: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    candidates = stage1.get("candidate_surfaces")
    require(isinstance(candidates, list) and len(candidates) == 1, "Stage-1 must have exactly one target surface")
    candidate = candidates[0]
    require(isinstance(candidate, Mapping), "target candidate is malformed")
    surface = candidate.get("surface")
    require(isinstance(surface, Mapping), "target candidate lacks surface geometry")
    return str(candidate.get("candidate_id", "target_surface")), surface
