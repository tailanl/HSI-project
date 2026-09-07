"""Unchanged scene grid, relation, search bounds and physical release gates."""
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

NATIVE_OCCUPANCY_SHAPE = (300, 100, 400)

WORLD_OCCUPANCY_SHAPE = (300, 400, 100)

GRID_LOWER_XYZ = (-3.0, -4.0, 0.0)

GRID_UPPER_XYZ = (3.0, 4.0, 2.0)

REORIENTATION_BINDING_TOLERANCE_RAD = 1.0e-6

SITTABLE_TARGET_CLASSES = {
    "chair", "armchair", "stool", "sofa", "couch", "bench", "ottoman", "seat"
}

REQUIRED_PUBLICATION_GATES = (
    "maximum_non_target_scene_penetration_m",
    "maximum_target_forbidden_body_penetration_m",
    "maximum_target_allowed_contact_penetration_m",
    "minimum_target_allowed_contact_vertices",
    "maximum_ground_penetration_m",
    "minimum_left_foot_ground_vertices",
    "minimum_right_foot_ground_vertices",
    "minimum_hips_support_inside_surface_fraction",
    "maximum_sdf_outside_fraction",
    "maximum_terminal_facing_error_rad",
)

class CurrentOnlyContractError(ValueError):
    """A current-only provenance, geometry, or physics contract failed."""

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise CurrentOnlyContractError(message)

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def occupancy_content_sha256(mask_world_xyz: np.ndarray) -> str:
    """P478-compatible compact identity for a world-layout boolean mask."""

    value = np.ascontiguousarray(mask_world_xyz, dtype=bool)
    require(value.shape == WORLD_OCCUPANCY_SHAPE, "world target mask shape drift")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(value.reshape(-1), bitorder="little").tobytes())
    return digest.hexdigest()

def native_to_world(mask_native_xyz: np.ndarray) -> np.ndarray:
    value = np.asarray(mask_native_xyz)
    require(value.shape == NATIVE_OCCUPANCY_SHAPE, "native occupancy shape drift")
    require(value.dtype == np.bool_, "native occupancy dtype must be bool")
    return np.ascontiguousarray(value.transpose(0, 2, 1)[:, ::-1, :])

def artifact(path: Path) -> dict[str, Any]:
    value = Path(path).resolve(strict=True)
    return {
        "path": str(value),
        "bytes": int(value.stat().st_size),
        "sha256": sha256_file(value),
    }

def normalize_relation(value: Any) -> str:
    """Preserve direct-target semantics without inventing a relation edge."""

    if value is None:
        return "none"
    relation = str(value).strip().casefold()
    require(relation, "Stage1 relation is empty")
    require(relation not in {"null", "n/a", "unknown"}, "ambiguous Stage1 relation")
    return relation

def angle_difference(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(float(left) - float(right)), math.cos(float(left) - float(right))))

def _validate_refine_and_gates(config: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float | int]]:
    refine = config.get("refine")
    require(isinstance(refine, Mapping), "campaign refine block is missing")
    radius = float(refine.get("same_surface_xy_search_radius_m"))
    step = float(refine.get("same_surface_xy_search_step_m"))
    clearance = float(refine.get("same_surface_max_clearance_m"))
    clearance_step = float(refine.get("same_surface_clearance_step_m", 0.005))
    edge_margin = float(refine.get("same_surface_edge_margin_m", 0.03))
    require(math.isfinite(radius) and 0.0 <= radius <= 0.30, "invalid same-surface XY radius")
    require(math.isfinite(step) and 0.001 <= step <= 0.10, "invalid same-surface XY step")
    require(radius == 0.0 or step <= 2.0 * radius + 1.0e-12, "XY search step skips configured radius")
    require(math.isfinite(clearance) and 0.0 <= clearance <= 0.15, "invalid maximum clearance")
    require(math.isfinite(clearance_step) and 0.001 <= clearance_step <= 0.05, "invalid clearance step")
    require(math.isfinite(edge_margin) and 0.0 <= edge_margin <= 0.10, "invalid surface edge margin")
    refine_values = {
        "same_surface_xy_search_radius_m": radius,
        "same_surface_xy_search_step_m": step,
        "same_surface_max_clearance_m": clearance,
        "same_surface_clearance_step_m": clearance_step,
        "same_surface_edge_margin_m": edge_margin,
    }

    gates = config.get("publication_gates")
    require(isinstance(gates, Mapping), "campaign publication gates are missing")
    require(set(REQUIRED_PUBLICATION_GATES).issubset(gates), "campaign publication gates are incomplete")
    gate_values: dict[str, float | int] = {}
    for key in REQUIRED_PUBLICATION_GATES:
        value = gates[key]
        if key.startswith("minimum_") and key.endswith("_vertices"):
            require(type(value) is int and 1 <= value <= 10000, f"invalid publication gate {key}")
            gate_values[key] = int(value)
        else:
            number = float(value)
            require(math.isfinite(number) and number >= 0.0, f"invalid publication gate {key}")
            gate_values[key] = number
    require(float(gate_values["maximum_sdf_outside_fraction"]) <= 0.10, "SDF outside gate is too permissive")
    require(
        0.0 < float(gate_values["minimum_hips_support_inside_surface_fraction"]) <= 1.0,
        "invalid hips support fraction",
    )
    require(float(gate_values["maximum_terminal_facing_error_rad"]) <= math.radians(30.0), "terminal yaw gate is too permissive")
    return refine_values, gate_values

def publication_gates(
    metrics: Mapping[str, float | int],
    configured: Mapping[str, float | int],
    *,
    facing_error_rad: float,
    selected_surface_match: bool,
    contact_lock_error_m: float = 0.0,
) -> dict[str, bool]:
    """Evaluate P509 physics metrics against campaign-owned thresholds."""

    return {
        "selected_surface_hash_bound": bool(selected_surface_match),
        "contact_anchor_locked": float(contact_lock_error_m) <= 2.0e-5,
        "target_forbidden_body_penetration": float(metrics["target_forbidden_body_penetration_m"])
        <= float(configured["maximum_target_forbidden_body_penetration_m"]),
        "target_allowed_contact_penetration": float(metrics["target_allowed_contact_penetration_m"])
        <= float(configured["maximum_target_allowed_contact_penetration_m"]),
        "target_allowed_contact_area": int(metrics["target_allowed_contact_vertex_count_in_band"])
        >= int(configured["minimum_target_allowed_contact_vertices"]),
        "non_target_scene_penetration": float(metrics["non_target_scene_penetration_m"])
        <= float(configured["maximum_non_target_scene_penetration_m"]),
        "ground_penetration": float(metrics["ground_penetration_m"])
        <= float(configured["maximum_ground_penetration_m"]),
        "sdf_grid_coverage": float(metrics["outside_fraction"])
        <= float(configured["maximum_sdf_outside_fraction"]),
        "left_foot_ground_contact_area": int(metrics["left_foot_ground_vertex_count_in_band"])
        >= int(configured["minimum_left_foot_ground_vertices"]),
        "right_foot_ground_contact_area": int(metrics["right_foot_ground_vertex_count_in_band"])
        >= int(configured["minimum_right_foot_ground_vertices"]),
        "hips_support_inside_surface": float(metrics["hips_support_inside_surface_fraction"])
        >= float(configured["minimum_hips_support_inside_surface_fraction"]),
        "terminal_facing": float(facing_error_rad)
        <= float(configured["maximum_terminal_facing_error_rad"]),
    }

def metric_rank(
    metrics: Mapping[str, float | int],
    gates: Mapping[str, bool],
    configured: Mapping[str, float | int],
) -> tuple[float, ...]:
    """Hard gates precede normalized physical quality in multistart ranking."""

    def violation(value: float, maximum: float) -> float:
        return max(0.0, value / max(maximum, 1.0e-12) - 1.0)

    failed = sum(not bool(value) for value in gates.values())
    normalized = (
        violation(
            float(metrics["target_forbidden_body_penetration_m"]),
            float(configured["maximum_target_forbidden_body_penetration_m"]),
        )
        + violation(
            float(metrics["target_allowed_contact_penetration_m"]),
            float(configured["maximum_target_allowed_contact_penetration_m"]),
        )
        + violation(
            float(metrics["non_target_scene_penetration_m"]),
            float(configured["maximum_non_target_scene_penetration_m"]),
        )
        + violation(
            float(metrics["ground_penetration_m"]),
            float(configured["maximum_ground_penetration_m"]),
        )
        + violation(
            float(metrics["outside_fraction"]),
            float(configured["maximum_sdf_outside_fraction"]),
        )
    )
    contact_deficit = max(
        0,
        int(configured["minimum_target_allowed_contact_vertices"])
        - int(metrics["target_allowed_contact_vertex_count_in_band"]),
    ) / max(1, int(configured["minimum_target_allowed_contact_vertices"]))
    return (
        float(failed),
        float(normalized + contact_deficit),
        float(metrics["non_target_scene_penetration_m"]),
        float(metrics["target_forbidden_body_penetration_m"]),
        -float(metrics["target_allowed_contact_vertex_count_in_band"]),
    )

def inclusive_offsets(radius: float, step: float) -> np.ndarray:
    """Symmetric deterministic grid that always contains zero and both limits."""

    radius, step = float(radius), float(step)
    require(radius >= 0.0 and step > 0.0, "invalid search-grid parameters")
    if radius <= 1.0e-12:
        return np.asarray((0.0,), dtype=np.float64)
    count = int(math.floor(radius / step + 1.0e-10))
    positive = [step * index for index in range(1, count + 1)]
    if not positive or radius - positive[-1] > 1.0e-10:
        positive.append(radius)
    positive[-1] = min(positive[-1], radius)
    return np.asarray(tuple(-value for value in reversed(positive)) + (0.0,) + tuple(positive), dtype=np.float64)
