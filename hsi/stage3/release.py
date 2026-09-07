"""The current 34 physical checks, without a motion-production claim."""
from __future__ import annotations
import math
from importlib.resources import files
from typing import Any, Mapping
from hsi.common.artifacts import require, read_json

class SelectionContractError(RuntimeError):
    pass

def nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def gate(actual: Any, comparison: str, threshold: Any) -> tuple[bool, float]:
    if comparison == "true":
        return actual is True, 0.0 if actual is True else 1.0
    if comparison == "false":
        return actual is False, 0.0 if actual is False else 1.0
    if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(float(actual)):
        return False, 1.0
    limit = float(threshold)
    if comparison == "max":
        passed = float(actual) <= limit + 1e-12
        return passed, max(0.0, float(actual) / max(abs(limit), 1e-12) - 1.0)
    if comparison == "min":
        passed = float(actual) >= limit - 1e-12
        return passed, max(0.0, limit / max(abs(float(actual)), 1e-12) - 1.0)
    raise SelectionContractError(f"unsupported comparison: {comparison}")

def release_gates(evaluation, policy):
    """Apply exactly the retained policy, including mandatory post-skin gates."""
    require(policy["schema"] == "p523.current_only_stage3_physical_release_policy.v2", "Unexpected release policy")
    gates = {}
    for name, spec in policy["gates"].items():
        actual = nested(evaluation, name)
        if actual is None and spec.get("optional_legacy_alias"):
            actual = nested(evaluation, spec["optional_legacy_alias"])
        if actual is None and spec.get("none_means_zero"):
            parent = nested(evaluation, name.rsplit(".", 1)[0])
            if isinstance(parent, dict) and name.rsplit(".", 1)[1] in parent:
                actual = 0.0
        passed, _ = gate(actual, spec["comparison"], spec["threshold"])
        gates[name] = {"actual": actual, "threshold": spec["threshold"], "passed": passed}
    full = evaluation.get("fullmesh_scene_collision", {})
    delivery = policy["post_render_delivery_gate"]
    gates["fullmesh_available"] = {"passed": full.get("available") is True and full.get("vertex_count") == 10475}
    for metric, threshold_key in (("forbidden_collision_fraction", "fullmesh_structural_nonfloor_collision_fraction_max"),
            ("forbidden_penetration_depth_m.maximum", "fullmesh_structural_nonfloor_penetration_depth_m_max")):
        actual = nested(full.get("structural_nonfloor", {}), metric)
        if actual is None and metric.endswith("maximum") and isinstance(full.get("structural_nonfloor", {}).get("forbidden_penetration_depth_m"), dict):
            actual = 0.0
        passed, _ = gate(actual, "max", delivery[threshold_key])
        gates["fullmesh." + metric] = {"actual": actual, "threshold": delivery[threshold_key], "passed": passed}
    # A failed scheduler/runtime cannot be published even if some numeric fields happen to be favorable.
    gates["actual_runner_success"] = {"passed": evaluation["runner_returncode"] == 0}
    return gates


def current_policy():
    """Load the original fixed policy shipped as package data."""
    return read_json(files("hsi.stage3").joinpath("physical_release_policy.json"))

