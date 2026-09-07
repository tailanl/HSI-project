"""Exact current route-root arithmetic; no historical source loader."""

from __future__ import annotations

import math
import numpy as np

from typing import Any, Iterable, Mapping, Sequence

from hsi.common.artifacts import require, MemoryContractError as ContractError

FORBIDDEN_SCHEMA_TOKENS = (
    "p481.", "p481_", "legacy_p481",
    "p373.", "p374.p373", "p373_", "legacy_p373",
    "p516.", "p516_", "legacy_p516",
    "p520.", "p520_", "legacy_p520",
    "p522.", "p522_", "legacy_p522",
)

def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for _key, item in value.items():
            # Audit keys such as ``p373_input_used: false`` are useful proof,
            # not provenance.  Only string values can name a schema/path.
            yield from _walk_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for item in value:
            yield from _walk_strings(item)

def reject_legacy_value(value: Any, label: str) -> None:
    for text in _walk_strings(value):
        lowered = text.lower()
        if any(token in lowered for token in FORBIDDEN_SCHEMA_TOKENS):
            raise ContractError(f"{label} contains forbidden P373/P520 provenance: {text}")
        if "/p373" in lowered or "/p520" in lowered or "\\p373" in lowered or "\\p520" in lowered:
            raise ContractError(f"{label} points to a forbidden P373/P520 artifact: {text}")

def finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape, f"{label} must have shape {shape}, got {result.shape}")
    require(np.isfinite(result).all(), f"{label} contains non-finite values")
    return result

def wrapped_angle(value: float) -> float:
    return float(math.atan2(math.sin(float(value)), math.cos(float(value))))

def angle_error(left: float, right: float) -> float:
    return abs(wrapped_angle(float(left) - float(right)))

def extract_route_roots(keynodes: Mapping[str, Any], seed_root: np.ndarray) -> list[np.ndarray]:
    reject_legacy_value(keynodes, "Stage1 key nodes")
    controls = keynodes.get("collision_safe_control_polyline")
    roots: list[np.ndarray] = []
    if isinstance(controls, Mapping) and "nodes_world_xy_m" in controls:
        xy = np.asarray(controls["nodes_world_xy_m"], dtype=np.float64)
        require(xy.ndim == 2 and xy.shape[1] == 2 and len(xy) >= 2,
                "collision-safe control polyline must be [N>=2,2]")
        require(np.isfinite(xy).all(), "collision-safe controls contain non-finite values")
        for index, point in enumerate(xy):
            if index + 1 < len(xy):
                delta = xy[index + 1] - point
            else:
                delta = point - xy[index - 1]
            yaw = math.atan2(float(delta[1]), float(delta[0]))
            roots.append(np.asarray((point[0], point[1], seed_root[2], yaw), dtype=np.float64))
    else:
        raw_nodes = keynodes.get("nodes", keynodes.get("key_nodes"))
        require(isinstance(raw_nodes, list) and len(raw_nodes) >= 2,
                "Stage1 key nodes must publish at least two route roots")
        for index, row in enumerate(raw_nodes):
            require(isinstance(row, Mapping), f"key node {index} is malformed")
            raw_root = row.get("root_xyz_yaw", row.get("root_xyz_yaw_world_zup"))
            if raw_root is None:
                point = np.asarray(row.get("world_xy_m"), dtype=np.float64)
                require(point.shape == (2,), f"key node {index} lacks root/XY")
                raw_root = (point[0], point[1], seed_root[2], seed_root[3])
            roots.append(finite(raw_root, (4,), f"key node {index} root"))
    terminal_handoff = keynodes.get("terminal_handoff")
    if isinstance(terminal_handoff, Mapping):
        arrival_yaw = float(terminal_handoff.get("arrival_tangent_yaw_rad", math.nan))
        require(math.isfinite(arrival_yaw),
                "Stage1 terminal handoff arrival tangent is invalid")
        if "nav_goal_world_xy_m" in terminal_handoff:
            nav_goal = finite(
                terminal_handoff["nav_goal_world_xy_m"], (2,),
                "Stage1 terminal handoff NavMesh goal",
            )
            require(float(np.linalg.norm(nav_goal - roots[-1][:2])) <= 2.0e-4,
                    "Stage1 terminal handoff goal differs from route endpoint")
        # Control polylines may simplify the route geometry.  The final
        # heading must nevertheless remain the dense NavMesh route's measured
        # arrival tangent, not the last simplified chord direction.
        roots[-1][3] = wrapped_angle(arrival_yaw)
    start_error = float(np.linalg.norm(roots[0][:2] - seed_root[:2]))
    if start_error > 2.0e-4:
        # Polygon NavMesh starts at a human-clear snapped point.  Preserve the
        # exact current query root as node zero, but permit the tiny bridge to
        # the snapped point only when the canonical current Stage1 key-node
        # receipt proves every endpoint and the measured snap distance.
        start = keynodes.get("start")
        require(isinstance(start, Mapping),
                "Stage1 route start differs from frame0 without a snap contract")
        requested = finite(
            start.get("requested_current_root_world_xy_m"), (2,),
            "Stage1 requested current root XY",
        )
        snapped = finite(
            start.get("snapped_navmesh_start_world_xy_m"), (2,),
            "Stage1 snapped NavMesh start XY",
        )
        declared_snap = float(start.get("start_snap_distance_m", math.nan))
        source = start.get("source")
        require(isinstance(source, Mapping), "Stage1 snap contract lacks current-root source")
        source_root = finite(
            source.get("root_xyz_yaw_world_zup"), (4,),
            "Stage1 hash-bound current root",
        )
        gates = keynodes.get("verified_gates")
        require(isinstance(gates, Mapping)
                and gates.get("start_read_from_hash_bound_current_root") is True,
                "Stage1 snap was not read from the hash-bound current root")
        require(float(np.linalg.norm(requested - seed_root[:2])) <= 2.0e-4,
                "Stage1 requested root differs from the fresh query frame0")
        require(float(np.linalg.norm(source_root[:3] - seed_root[:3])) <= 2.0e-4
                and angle_error(source_root[3], seed_root[3]) <= 2.0e-4,
                "Stage1 snap source root differs from the fresh query frame0")
        require(float(np.linalg.norm(snapped - roots[0][:2])) <= 2.0e-6,
                "Stage1 snap endpoint differs from its route start")
        measured_snap = float(np.linalg.norm(snapped - requested))
        require(math.isfinite(declared_snap)
                and abs(declared_snap - measured_snap) <= 2.0e-6,
                "Stage1 declared start snap distance drift")
        require(measured_snap <= 0.05,
                "current-root to NavMesh-start bridge exceeds 5 cm")
        roots.insert(0, np.asarray(seed_root, dtype=np.float64).copy())
    require(float(np.linalg.norm(roots[0][:2] - seed_root[:2])) <= 2.0e-4,
            "Stage1 route start and fresh query frame0 differ")
    return roots

