"""Strict polygon-NavMesh backend for a compiled LINGO human roadmap.

This adapter consumes the hash-bound ``p504`` roadmap request produced by the
existing Stage-1 compiler.  It deliberately supports only the real
``pynavmesh`` distribution (whose import package is ``pathfinder``).  If that
distribution is unavailable, execution fails closed; there is no grid-A*
fallback and no result is labelled as NaoMesh.

The input grid is already inflated by the human radius.  Every free cell is
therefore converted to a convex X-Z-plane polygon with *shared* lattice vertex
indices, and ``pathfinder.PathFinder`` performs polygonal NavMesh pathfinding.
The backend route is densified and then checked against the source grid.  The
checker binds the output to both source hashes and verifies snapped endpoints,
maximum step length, and every raster sample along every route segment.
"""

import argparse

from dataclasses import dataclass

import hashlib

import importlib

import importlib.metadata

import json

import math

import os

from pathlib import Path

import sys

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

P504_INPUT_SCHEMA = "p504.naomesh_human_roadmap_input.v1"

OUTPUT_SCHEMA = "p508.pynavmesh_polygon_route_output.v1"

LEGACY_P504_OUTPUT_SCHEMA = "p504.naomesh_human_roadmap_output.v1"

BACKEND_DISTRIBUTION = "pynavmesh"

BACKEND_IMPORT = "pathfinder"

class NavmeshBackendError(RuntimeError):
    """Base class for all fail-closed backend errors."""

class BackendUnavailable(NavmeshBackendError):
    """Raised when the real third-party backend cannot be authenticated."""

class ContractError(NavmeshBackendError):
    """Raised when an input or output violates the strict route contract."""

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ContractError(message)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def route_sha256(route: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(route, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()

def read_json_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON root must be an object")
    return value

def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def finite_vector(value: Any, size: int, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    require(array.shape == (size,), "%s must contain exactly %d values" % (name, size))
    require(np.isfinite(array).all(), "%s contains non-finite values" % name)
    return array

def world_to_cell(
    point_xy: Sequence[float], lower_xy: np.ndarray, resolution_m: float,
    shape: Tuple[int, int], *, name: str = "point",
) -> Tuple[int, int]:
    point = finite_vector(point_xy, 2, name)
    cell = np.floor((point - lower_xy) / resolution_m).astype(np.int64)
    require(
        0 <= int(cell[0]) < shape[0] and 0 <= int(cell[1]) < shape[1],
        "%s lies outside the roadmap: %s" % (name, point.tolist()),
    )
    return int(cell[0]), int(cell[1])

def cell_center(cell: Sequence[int], lower_xy: np.ndarray, resolution_m: float) -> np.ndarray:
    index = np.asarray(cell, dtype=np.float64)
    return lower_xy + (index + 0.5) * resolution_m

@dataclass(frozen=True)
class RoadmapRequest:
    request_path: Path
    request_json_sha256: str
    input_npz_path: Path
    input_npz_sha256: str
    free: np.ndarray
    obstacles: np.ndarray
    clearance_m: np.ndarray
    lower_world_xy_m: np.ndarray
    cell_resolution_m: float
    human_radius_clearance_m: float
    requested_start_world_xy_m: np.ndarray
    requested_goal_world_xy_m: np.ndarray
    requested_start_cell: Tuple[int, int]
    requested_goal_cell: Tuple[int, int]
    snapped_start_cell: Tuple[int, int]
    snapped_goal_cell: Tuple[int, int]
    snapped_start_world_xy_m: np.ndarray
    snapped_goal_world_xy_m: np.ndarray
    hard_navigation_clearance_m: Optional[float] = None
    navigation_clearance_mode: str = "human_radius_hard_buffer"
    target_interaction_relaxed_cells: Optional[np.ndarray] = None
    target_interaction_relaxation_radius_m: Optional[float] = None
    interaction_terminal_min_clearance_m: Optional[float] = None

    @property
    def effective_hard_navigation_clearance_m(self) -> float:
        if self.hard_navigation_clearance_m is None:
            return float(self.human_radius_clearance_m)
        return float(self.hard_navigation_clearance_m)

def _resolve_artifact_path(request_path: Path, raw_path: Any) -> Path:
    value = Path(str(raw_path)).expanduser()
    if not value.is_absolute():
        value = request_path.parent / value
    return value.resolve(strict=True)

def load_request(path: Path) -> RoadmapRequest:
    """Load and validate a P504 request without conflating requested/snapped goals."""

    request_path = path.resolve(strict=True)
    payload = read_json_object(request_path)
    require(payload.get("schema") == P504_INPUT_SCHEMA, "unsupported roadmap input schema")

    artifact = payload.get("input_npz")
    require(isinstance(artifact, Mapping), "request lacks input_npz artifact")
    npz_path = _resolve_artifact_path(request_path, artifact.get("path", ""))
    actual_npz_hash = sha256_file(npz_path)
    require(actual_npz_hash == artifact.get("sha256"), "input NPZ hash does not match request")
    required_output = payload.get("required_output_contract")
    require(isinstance(required_output, Mapping), "request lacks required_output_contract")
    require(
        required_output.get("source_input_npz_sha256") == actual_npz_hash,
        "required output hash is not bound to the input NPZ",
    )
    required_output_schema = str(required_output.get("schema", ""))

    with np.load(npz_path, allow_pickle=False) as archive:
        required_keys = {
            "schema", "free_human_cells", "obstacle_cells", "clearance_m",
            "lower_world_xy_m", "cell_resolution_m", "start_cell", "goal_cell",
        }
        require(required_keys.issubset(set(archive.files)), "input NPZ is missing roadmap arrays")
        npz_schema = str(np.asarray(archive["schema"]).item())
        require(npz_schema == P504_INPUT_SCHEMA, "NPZ schema does not match request schema")
        free = np.asarray(archive["free_human_cells"], dtype=np.bool_).copy()
        obstacles = np.asarray(archive["obstacle_cells"], dtype=np.bool_).copy()
        clearance = np.asarray(archive["clearance_m"], dtype=np.float64).copy()
        lower = np.asarray(archive["lower_world_xy_m"], dtype=np.float64).copy()
        resolution = float(np.asarray(archive["cell_resolution_m"]).item())
        snapped_start = tuple(int(v) for v in np.asarray(archive["start_cell"]).tolist())
        snapped_goal = tuple(int(v) for v in np.asarray(archive["goal_cell"]).tolist())
        if "target_interaction_relaxed_cells" in archive.files:
            target_relaxed = np.asarray(
                archive["target_interaction_relaxed_cells"], dtype=np.bool_
            ).copy()
        else:
            target_relaxed = np.zeros_like(free, dtype=np.bool_)

    require(free.ndim == 2 and free.size > 0, "free_human_cells must be a non-empty 2-D grid")
    require(obstacles.shape == free.shape, "obstacle grid shape differs from free grid")
    require(clearance.shape == free.shape, "clearance grid shape differs from free grid")
    require(target_relaxed.shape == free.shape, "target interaction relaxation grid shape differs from free grid")
    require(np.isfinite(clearance).all() and np.all(clearance >= 0.0), "clearance grid is invalid")
    require(lower.shape == (2,) and np.isfinite(lower).all(), "lower_world_xy_m is invalid")
    require(math.isfinite(resolution) and resolution > 0.0, "cell resolution must be positive")
    require(np.count_nonzero(free) > 0, "roadmap contains no free human cell")
    require(not np.any(free & obstacles), "a roadmap cell cannot be both obstacle and free")

    requested_start = finite_vector(payload.get("start_world_xy_m"), 2, "start_world_xy_m")
    requested_goal = finite_vector(payload.get("goal_world_xy_m"), 2, "goal_world_xy_m")
    requested_start_cell = world_to_cell(requested_start, lower, resolution, free.shape, name="requested start")
    requested_goal_cell = world_to_cell(requested_goal, lower, resolution, free.shape, name="requested goal")

    for name, cell in (("snapped start", snapped_start), ("snapped goal", snapped_goal)):
        require(len(cell) == 2, "%s cell is malformed" % name)
        require(0 <= cell[0] < free.shape[0] and 0 <= cell[1] < free.shape[1], "%s cell is outside grid" % name)
        require(bool(free[cell]), "%s cell is not human-walkable" % name)

    json_start_cell = tuple(int(v) for v in payload.get("start_cell", []))
    json_goal_cell = tuple(int(v) for v in payload.get("goal_cell", []))
    require(json_start_cell == snapped_start, "JSON/NPZ snapped start cells disagree")
    require(json_goal_cell == snapped_goal, "JSON/NPZ snapped goal cells disagree")

    snapped_start_world = cell_center(snapped_start, lower, resolution)
    snapped_goal_world = cell_center(snapped_goal, lower, resolution)
    # P504 searched at most 1 metre for a free cell.  Preserve that safety
    # boundary while exposing both coordinate variants explicitly.
    max_snap_distance = 1.0 + math.sqrt(2.0) * resolution
    require(
        float(np.linalg.norm(snapped_start_world - requested_start)) <= max_snap_distance,
        "snapped start is implausibly far from requested start",
    )
    require(
        float(np.linalg.norm(snapped_goal_world - requested_goal)) <= max_snap_distance,
        "snapped goal is implausibly far from requested goal",
    )

    human_clearance = float(payload.get("human_radius_clearance_m"))
    require(math.isfinite(human_clearance) and human_clearance > 0.0, "human clearance is invalid")
    hard_clearance = float(payload.get("hard_navigation_clearance_m", human_clearance))
    clearance_mode = str(payload.get("navigation_clearance_mode", "human_radius_hard_buffer"))
    require(math.isfinite(hard_clearance) and hard_clearance >= 0.0, "hard navigation clearance is invalid")
    require(
        clearance_mode in {
            "human_radius_hard_buffer",
            "raw_free_stage3_sdf_collision",
            "target_interaction_relaxed_stage3_sdf_collision",
        },
        "unsupported navigation clearance mode",
    )
    if clearance_mode == "human_radius_hard_buffer":
        require(
            required_output_schema in {OUTPUT_SCHEMA, LEGACY_P504_OUTPUT_SCHEMA},
            "unsupported required output schema",
        )
        require(abs(hard_clearance - human_clearance) <= 1e-12, "hard-buffer request radius drift")
        require(not np.any(target_relaxed), "hard-buffer request contains target relaxation cells")
        relaxation_radius = None
        terminal_min_clearance = None
    elif clearance_mode == "raw_free_stage3_sdf_collision":
        require(
            required_output_schema in {OUTPUT_SCHEMA, LEGACY_P504_OUTPUT_SCHEMA},
            "unsupported required output schema",
        )
        require(hard_clearance == 0.0, "raw-free request retains a hard clearance buffer")
        require(np.array_equal(free, ~obstacles), "raw-free request does not expose every raw-free cell")
        require(not np.any(target_relaxed), "global raw-free request contains target-only relaxation cells")
        relaxation_radius = None
        terminal_min_clearance = None
    else:
        require(
            required_output_schema == OUTPUT_SCHEMA,
            "target-relaxed request must require the polygon NavMesh output schema",
        )
        require(abs(hard_clearance - human_clearance) <= 1e-12, "target-relaxed request nominal radius drift")
        relaxation_radius = float(payload.get("target_interaction_relaxation_radius_m"))
        terminal_min_clearance = float(payload.get("interaction_terminal_min_clearance_m"))
        require(
            math.isfinite(relaxation_radius) and relaxation_radius > 0.0,
            "target interaction relaxation radius is invalid",
        )
        require(
            math.isfinite(terminal_min_clearance)
            and 0.0 < terminal_min_clearance <= hard_clearance,
            "interaction terminal minimum clearance is invalid",
        )
        require(not np.any(target_relaxed & obstacles), "target relaxation includes a raw obstacle")
        base_free = (~obstacles) & (clearance >= hard_clearance)
        require(
            not np.any(target_relaxed & base_free),
            "target relaxation redundantly marks nominal-clear cells",
        )
        require(
            not np.any(target_relaxed & (clearance + 1e-5 < terminal_min_clearance)),
            "target relaxation includes a cell below terminal minimum clearance",
        )
        cell_indices = np.argwhere(target_relaxed)
        if len(cell_indices):
            relaxed_centres = lower[None, :] + (
                cell_indices.astype(np.float64) + 0.5
            ) * resolution
            relaxed_distance = np.linalg.norm(
                relaxed_centres - requested_goal[None, :], axis=1
            )
            require(
                bool(np.all(relaxed_distance <= relaxation_radius + 1e-6)),
                "target relaxation includes a cell outside the selected interaction neighbourhood",
            )
        require(
            np.array_equal(free, base_free | target_relaxed),
            "target-relaxed free mask is not base-free union interaction relaxation",
        )
    below_hard = free & (clearance + 1e-5 < hard_clearance)
    require(
        not np.any(below_hard & ~target_relaxed),
        "free grid contains low-clearance cells outside target interaction relaxation",
    )

    return RoadmapRequest(
        request_path=request_path,
        request_json_sha256=sha256_file(request_path),
        input_npz_path=npz_path,
        input_npz_sha256=actual_npz_hash,
        free=free,
        obstacles=obstacles,
        clearance_m=clearance,
        lower_world_xy_m=lower,
        cell_resolution_m=resolution,
        human_radius_clearance_m=human_clearance,
        requested_start_world_xy_m=requested_start,
        requested_goal_world_xy_m=requested_goal,
        requested_start_cell=requested_start_cell,
        requested_goal_cell=requested_goal_cell,
        snapped_start_cell=snapped_start,
        snapped_goal_cell=snapped_goal,
        snapped_start_world_xy_m=snapped_start_world,
        snapped_goal_world_xy_m=snapped_goal_world,
        hard_navigation_clearance_m=hard_clearance,
        navigation_clearance_mode=clearance_mode,
        target_interaction_relaxed_cells=target_relaxed,
        target_interaction_relaxation_radius_m=relaxation_radius,
        interaction_terminal_min_clearance_m=terminal_min_clearance,
    )

def cell_meets_navigation_contract(request: RoadmapRequest, cell: tuple[int, int]) -> bool:
    if not bool(request.free[cell]):
        return False
    clearance = float(request.clearance_m[cell])
    if clearance + 1e-5 >= request.effective_hard_navigation_clearance_m:
        return True
    return bool(
        request.navigation_clearance_mode == "target_interaction_relaxed_stage3_sdf_collision"
        and request.target_interaction_relaxed_cells is not None
        and request.target_interaction_relaxed_cells[cell]
    )

def build_cell_polygon_navmesh(
    free: np.ndarray, lower_world_xy_m: np.ndarray, resolution_m: float,
) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    """Convert free grid cells into shared-topology convex X-Z polygons."""

    free = np.asarray(free, dtype=np.bool_)
    require(free.ndim == 2 and np.any(free), "cannot polygonize an empty free grid")
    lower = finite_vector(lower_world_xy_m, 2, "lower_world_xy_m")
    require(math.isfinite(resolution_m) and resolution_m > 0.0, "invalid polygon resolution")
    vertices: List[Tuple[float, float, float]] = []
    polygons: List[List[int]] = []
    lattice_to_vertex: Dict[Tuple[int, int], int] = {}

    def vertex(i: int, j: int) -> int:
        key = (i, j)
        found = lattice_to_vertex.get(key)
        if found is not None:
            return found
        # pynavmesh performs navigation in X-Z; Y is the vertical coordinate.
        index = len(vertices)
        vertices.append((
            float(lower[0] + i * resolution_m),
            0.0,
            float(lower[1] + j * resolution_m),
        ))
        lattice_to_vertex[key] = index
        return index

    for i, j in np.argwhere(free):
        x, y = int(i), int(j)
        # Same +Y winding as the official pynavmesh plane example.
        polygons.append([
            vertex(x, y),
            vertex(x, y + 1),
            vertex(x + 1, y + 1),
            vertex(x + 1, y),
        ])
    require(len(polygons) == int(np.count_nonzero(free)), "polygonization lost free cells")
    return vertices, polygons

@dataclass(frozen=True)
class BackendIdentity:
    distribution: str
    version: str
    import_name: str
    module_file: str

def load_pynavmesh() -> Tuple[Any, BackendIdentity]:
    """Authenticate and import the official pynavmesh distribution."""

    # P508 pins the PyPI wheel inside the workspace instead of mutating a
    # shared conda environment.  Adding this exact, auditable distribution
    # root is not a fallback implementation: the metadata and import-owner
    # checks below still have to identify the official ``pynavmesh`` wheel.
    # Runtime must install pynavmesh explicitly; no private source-path loading.

    try:
        version = importlib.metadata.version(BACKEND_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as error:
        raise BackendUnavailable(
            "pynavmesh is not installed; refusing to substitute native grid A*"
        ) from error
    owners = importlib.metadata.packages_distributions().get(BACKEND_IMPORT, [])
    if BACKEND_DISTRIBUTION not in {str(value).lower() for value in owners}:
        raise BackendUnavailable("the pathfinder import is not owned by the pynavmesh distribution")
    try:
        module = importlib.import_module(BACKEND_IMPORT)
    except (ImportError, OSError) as error:
        raise BackendUnavailable("pynavmesh is installed but pathfinder cannot be imported: %s" % error) from error
    if not callable(getattr(module, "PathFinder", None)):
        raise BackendUnavailable("pathfinder.PathFinder API is unavailable")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise BackendUnavailable("pathfinder module has no auditable source location")
    return module, BackendIdentity(
        distribution=BACKEND_DISTRIBUTION,
        version=str(version),
        import_name=BACKEND_IMPORT,
        module_file=str(Path(module_file).resolve()),
    )

def remove_consecutive_duplicates(route: np.ndarray, tolerance_m: float = 1e-10) -> np.ndarray:
    points = np.asarray(route, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2, "route must be shaped [N,2]")
    keep = [0]
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[keep[-1]])) > tolerance_m:
            keep.append(index)
    return points[np.asarray(keep, dtype=np.int64)]

def densify_route(route: np.ndarray, max_step_m: float) -> np.ndarray:
    require(math.isfinite(max_step_m) and max_step_m > 0.0, "max route step must be positive")
    points = remove_consecutive_duplicates(route)
    require(len(points) >= 2, "route must contain two distinct endpoints")
    output = [points[0]]
    for start, finish in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(finish - start))
        count = max(1, int(math.ceil(distance / max_step_m)))
        for index in range(1, count + 1):
            output.append(start + (finish - start) * (float(index) / count))
    return np.asarray(output, dtype=np.float64)

def validate_route(
    route: np.ndarray, request: RoadmapRequest, *, max_step_m: float,
    collision_sample_spacing_m: float, endpoint_tolerance_m: float,
) -> Dict[str, Any]:
    """Strictly validate endpoints, continuity and every segment against the grid."""

    points = np.asarray(route, dtype=np.float64)
    require(points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 2, "route must be [N>=2,2]")
    require(np.isfinite(points).all(), "route contains non-finite coordinates")
    require(max_step_m > 0.0 and math.isfinite(max_step_m), "max_step_m is invalid")
    require(
        collision_sample_spacing_m > 0.0 and math.isfinite(collision_sample_spacing_m),
        "collision sample spacing is invalid",
    )
    require(endpoint_tolerance_m >= 0.0 and math.isfinite(endpoint_tolerance_m), "endpoint tolerance is invalid")

    start_error = float(np.linalg.norm(points[0] - request.snapped_start_world_xy_m))
    goal_error = float(np.linalg.norm(points[-1] - request.snapped_goal_world_xy_m))
    require(start_error <= endpoint_tolerance_m, "route does not start at snapped_start_world_xy_m")
    require(goal_error <= endpoint_tolerance_m, "route does not end at snapped_goal_world_xy_m")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    require(np.all(segment_lengths > 0.0), "route contains duplicate consecutive points")
    largest_step = float(np.max(segment_lengths))
    require(largest_step <= max_step_m + 1e-9, "route contains a discontinuous step")

    sample_count = 0
    minimum_clearance = math.inf
    visited_cells = set()
    for segment_index, (start, finish) in enumerate(zip(points[:-1], points[1:])):
        distance = float(np.linalg.norm(finish - start))
        subdivisions = max(1, int(math.ceil(distance / collision_sample_spacing_m)))
        for sample_index in range(subdivisions + 1):
            point = start + (finish - start) * (float(sample_index) / subdivisions)
            try:
                cell = world_to_cell(
                    point, request.lower_world_xy_m, request.cell_resolution_m,
                    request.free.shape, name="route segment %d sample" % segment_index,
                )
            except ContractError as error:
                raise ContractError("route leaves roadmap bounds: %s" % error) from error
            require(bool(request.free[cell]), "route segment %d crosses a blocked cell %s" % (segment_index, cell))
            clearance = float(request.clearance_m[cell])
            require(
                cell_meets_navigation_contract(request, cell),
                "route segment %d violates hard clearance outside target interaction relaxation" % segment_index,
            )
            minimum_clearance = min(minimum_clearance, clearance)
            visited_cells.add(cell)
            sample_count += 1

    length = float(np.sum(segment_lengths))
    return {
        "route_point_count": int(len(points)),
        "route_segment_count": int(len(points) - 1),
        "route_length_m": length,
        "maximum_step_m": largest_step,
        "minimum_sampled_clearance_m": minimum_clearance,
        "collision_sample_count": sample_count,
        "visited_free_cell_count": len(visited_cells),
        "snapped_start_error_m": start_error,
        "snapped_goal_error_m": goal_error,
        "all_segment_samples_human_free": True,
    }

def _pathfinder_route(
    module: Any, vertices: List[Tuple[float, float, float]], polygons: List[List[int]],
    start_world_xy_m: np.ndarray, goal_world_xy_m: np.ndarray,
) -> np.ndarray:
    try:
        pathfinder = module.PathFinder(vertices, polygons)
        start_xz = (float(start_world_xy_m[0]), 0.0, float(start_world_xy_m[1]))
        goal_xz = (float(goal_world_xy_m[0]), 0.0, float(goal_world_xy_m[1]))
        raw = pathfinder.search_path(start_xz, goal_xz)
    except Exception as error:  # third-party boundary; converted to a fail-closed error
        raise NavmeshBackendError("pynavmesh path search failed: %s" % error) from error
    require(raw is not None, "pynavmesh found no route")
    route_xz = np.asarray(raw, dtype=np.float64)
    require(
        route_xz.ndim == 2 and route_xz.shape[0] >= 2 and route_xz.shape[1] == 3,
        "pynavmesh returned a malformed route",
    )
    require(np.isfinite(route_xz).all(), "pynavmesh returned non-finite coordinates")
    return remove_consecutive_duplicates(route_xz[:, (0, 2)])

def make_output_payload(
    request: RoadmapRequest, identity: BackendIdentity, route: np.ndarray,
    raw_corner_route: np.ndarray, mesh_vertex_count: int, mesh_polygon_count: int,
    *, max_step_m: float, collision_sample_spacing_m: float,
    endpoint_tolerance_m: float,
) -> Dict[str, Any]:
    validation = validate_route(
        route, request, max_step_m=max_step_m,
        collision_sample_spacing_m=collision_sample_spacing_m,
        endpoint_tolerance_m=endpoint_tolerance_m,
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "verified_polygon_navmesh_route",
        "backend": {
            "distribution": identity.distribution,
            "version": identity.version,
            "import_name": identity.import_name,
            "module_file": identity.module_file,
            "external_polygon_navmesh": True,
            "native_grid_astar_used": False,
            "backend_label": "pynavmesh/pathfinder",
        },
        "source": {
            "request_json_path": str(request.request_path),
            "request_json_sha256": request.request_json_sha256,
            "input_npz_path": str(request.input_npz_path),
            "input_npz_sha256": request.input_npz_sha256,
        },
        "coordinate_contract": {
            "public_route_coordinates": "world_zup_metric_xy",
            "pynavmesh_coordinates": "x_horizontal_y_vertical_z_horizontal",
            "requested_start_world_xy_m": request.requested_start_world_xy_m.tolist(),
            "requested_goal_world_xy_m": request.requested_goal_world_xy_m.tolist(),
            "requested_start_cell": list(request.requested_start_cell),
            "requested_goal_cell": list(request.requested_goal_cell),
            "snapped_start_cell": list(request.snapped_start_cell),
            "snapped_goal_cell": list(request.snapped_goal_cell),
            "snapped_start_world_xy_m": request.snapped_start_world_xy_m.tolist(),
            "snapped_goal_world_xy_m": request.snapped_goal_world_xy_m.tolist(),
            "backend_consumed_snapped_endpoints_only": True,
        },
        "polygon_navmesh": {
            "source": "shared-topology quads from navigation-contract walkable cells",
            "vertex_count": mesh_vertex_count,
            "polygon_count": mesh_polygon_count,
            "human_radius_clearance_m": request.human_radius_clearance_m,
            "hard_navigation_clearance_m": request.effective_hard_navigation_clearance_m,
            "navigation_clearance_mode": request.navigation_clearance_mode,
            "target_interaction_relaxed_cell_count": int(
                0 if request.target_interaction_relaxed_cells is None
                else np.count_nonzero(request.target_interaction_relaxed_cells)
            ),
            "target_interaction_relaxation_radius_m": (
                request.target_interaction_relaxation_radius_m
            ),
            "interaction_terminal_min_clearance_m": (
                request.interaction_terminal_min_clearance_m
            ),
            "cell_resolution_m": request.cell_resolution_m,
        },
        "raw_corner_route_world_xy_m": raw_corner_route.astype(float).tolist(),
        "route_world_xy_m": route.astype(float).tolist(),
        "route_world_xy_float64_sha256": route_sha256(route),
        "validation_parameters": {
            "max_step_m": max_step_m,
            "collision_sample_spacing_m": collision_sample_spacing_m,
            "endpoint_tolerance_m": endpoint_tolerance_m,
        },
        "validation": validation,
    }

def validate_output_payload(payload: Mapping[str, Any], request: RoadmapRequest) -> Dict[str, Any]:
    """Revalidate a serialized result, including hashes and coordinate metadata."""

    require(payload.get("schema") == OUTPUT_SCHEMA, "wrong polygon route output schema")
    require(payload.get("status") == "verified_polygon_navmesh_route", "route output is not verified")
    backend = payload.get("backend")
    require(isinstance(backend, Mapping), "route output lacks backend identity")
    require(backend.get("distribution") == BACKEND_DISTRIBUTION, "route did not use pynavmesh")
    require(backend.get("import_name") == BACKEND_IMPORT, "route did not use pathfinder API")
    require(backend.get("external_polygon_navmesh") is True, "route is not external polygon NavMesh")
    require(backend.get("native_grid_astar_used") is False, "grid A* cannot satisfy this contract")
    polygon_navmesh = payload.get("polygon_navmesh")
    require(isinstance(polygon_navmesh, Mapping), "route output lacks polygon NavMesh metadata")
    require(
        float(polygon_navmesh.get("human_radius_clearance_m")) == request.human_radius_clearance_m,
        "nominal human radius metadata drift",
    )
    require(
        float(polygon_navmesh.get(
            "hard_navigation_clearance_m",
            polygon_navmesh.get("human_radius_clearance_m"),
        ))
        == request.effective_hard_navigation_clearance_m,
        "hard navigation clearance metadata drift",
    )
    require(
        polygon_navmesh.get("navigation_clearance_mode", "human_radius_hard_buffer")
        == request.navigation_clearance_mode,
        "navigation clearance mode metadata drift",
    )
    expected_relaxed_count = int(
        0 if request.target_interaction_relaxed_cells is None
        else np.count_nonzero(request.target_interaction_relaxed_cells)
    )
    require(
        int(polygon_navmesh.get("target_interaction_relaxed_cell_count", 0))
        == expected_relaxed_count,
        "target interaction relaxed-cell metadata drift",
    )
    if request.navigation_clearance_mode == "target_interaction_relaxed_stage3_sdf_collision":
        require(
            float(polygon_navmesh.get("target_interaction_relaxation_radius_m"))
            == request.target_interaction_relaxation_radius_m,
            "target interaction relaxation radius metadata drift",
        )
        require(
            float(polygon_navmesh.get("interaction_terminal_min_clearance_m"))
            == request.interaction_terminal_min_clearance_m,
            "interaction terminal minimum clearance metadata drift",
        )
    source = payload.get("source")
    require(isinstance(source, Mapping), "route output lacks source binding")
    require(source.get("request_json_sha256") == request.request_json_sha256, "request JSON hash mismatch")
    require(source.get("input_npz_sha256") == request.input_npz_sha256, "roadmap NPZ hash mismatch")

    coordinates = payload.get("coordinate_contract")
    require(isinstance(coordinates, Mapping), "route output lacks coordinate contract")
    expected_arrays = {
        "requested_start_world_xy_m": request.requested_start_world_xy_m,
        "requested_goal_world_xy_m": request.requested_goal_world_xy_m,
        "snapped_start_world_xy_m": request.snapped_start_world_xy_m,
        "snapped_goal_world_xy_m": request.snapped_goal_world_xy_m,
    }
    for name, expected in expected_arrays.items():
        actual = finite_vector(coordinates.get(name), 2, name)
        require(np.array_equal(actual, expected), "%s drifted from request" % name)
    require(tuple(coordinates.get("snapped_start_cell", [])) == request.snapped_start_cell, "snapped start cell mismatch")
    require(tuple(coordinates.get("snapped_goal_cell", [])) == request.snapped_goal_cell, "snapped goal cell mismatch")
    require(coordinates.get("backend_consumed_snapped_endpoints_only") is True, "backend endpoint contract is ambiguous")

    parameters = payload.get("validation_parameters")
    require(isinstance(parameters, Mapping), "route output lacks validation parameters")
    route = np.asarray(payload.get("route_world_xy_m"), dtype=np.float64)
    require(route_sha256(route) == payload.get("route_world_xy_float64_sha256"), "route byte hash mismatch")
    validation = validate_route(
        route, request,
        max_step_m=float(parameters.get("max_step_m")),
        collision_sample_spacing_m=float(parameters.get("collision_sample_spacing_m")),
        endpoint_tolerance_m=float(parameters.get("endpoint_tolerance_m")),
    )
    return validation

def run(request_path: Path, output_path: Path) -> Dict[str, Any]:
    request = load_request(request_path)
    module, identity = load_pynavmesh()
    vertices, polygons = build_cell_polygon_navmesh(
        request.free, request.lower_world_xy_m, request.cell_resolution_m,
    )
    raw_route = _pathfinder_route(
        module, vertices, polygons,
        request.snapped_start_world_xy_m, request.snapped_goal_world_xy_m,
    )

    backend_endpoint_tolerance = request.cell_resolution_m
    require(
        float(np.linalg.norm(raw_route[0] - request.snapped_start_world_xy_m)) <= backend_endpoint_tolerance,
        "pynavmesh changed the snapped start endpoint",
    )
    require(
        float(np.linalg.norm(raw_route[-1] - request.snapped_goal_world_xy_m)) <= backend_endpoint_tolerance,
        "pynavmesh changed the snapped goal endpoint",
    )
    raw_route[0] = request.snapped_start_world_xy_m
    raw_route[-1] = request.snapped_goal_world_xy_m

    max_step = request.cell_resolution_m * 0.5
    sample_spacing = request.cell_resolution_m * 0.25
    endpoint_tolerance = max(1e-8, request.cell_resolution_m * 1e-5)
    dense_route = densify_route(raw_route, max_step)
    payload = make_output_payload(
        request, identity, dense_route, raw_route, len(vertices), len(polygons),
        max_step_m=max_step,
        collision_sample_spacing_m=sample_spacing,
        endpoint_tolerance_m=endpoint_tolerance,
    )
    # A serialized copy must pass exactly the same validator before publication.
    validate_output_payload(json.loads(json.dumps(payload, allow_nan=False)), request)
    atomic_json(output_path, payload)
    return payload

