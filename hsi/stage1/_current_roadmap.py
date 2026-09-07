"""Compile a P523 current-only roadmap request with path-cost-first options.

The frozen P508 compiler remains the implementation authority for occupancy,
clearance, snapping and request serialization.  P523 adds one narrow wrapper:
after semantic/terminal gates pass, select the reachable approach with the
lowest fresh current-occupancy path cost.  This prevents a millimetre-scale
surface-distance tie from choosing a several-metre detour.
"""

from __future__ import annotations

import argparse

import copy

from datetime import datetime, timezone

import hashlib

import json

from pathlib import Path

import sys

import tempfile

from typing import Any, Mapping, Sequence

import numpy as np

from . import _roadmap as p508

P523_TARGET_SCHEMA = "p523.current_only_stage1_target_surface.v1"

P509_TARGET_SCHEMA = "p508.lingo_scene_first_stage1.v1"

SELECTOR_SCHEMA = "p523.current_only_roadmap_option_selector.v2"

FORBIDDEN_RESULT_PREFIXES = ("p373", "p480", "p517", "p520", "p522")

class CurrentRoadmapError(RuntimeError):
    pass

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise CurrentRoadmapError(message)

def canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON root must be an object: %s" % path)
    return value

def verify_payload_hash(value: Mapping[str, Any], label: str) -> None:
    recorded = value.get("receipt_payload_sha256")
    require(isinstance(recorded, str) and len(recorded) == 64, "%s lacks payload hash" % label)
    payload = dict(value)
    payload.pop("receipt_payload_sha256", None)
    require(canonical_hash(payload) == recorded, "%s payload hash drift" % label)

def reject_historical(path: Path, label: str) -> None:
    for part in (entry.lower() for entry in path.resolve().parts):
        if any(part == prefix or part.startswith(prefix + "_") for prefix in FORBIDDEN_RESULT_PREFIXES):
            raise CurrentRoadmapError("%s points at a forbidden historical result: %s" % (label, part))

def finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (length,), "%s must have shape [%d]" % (label, length))
    require(np.isfinite(array).all(), "%s contains non-finite values" % label)
    return array

def selected_candidate(target: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    candidates = target.get("candidate_surfaces")
    require(isinstance(candidates, list) and len(candidates) == 1,
            "%s must contain exactly one surface" % label)
    candidate = candidates[0]
    require(isinstance(candidate, Mapping), "%s candidate is malformed" % label)
    require(candidate.get("candidate_id") == target.get("selected_candidate_id"),
            "%s selected candidate drift" % label)
    return candidate

def validate_target_pair(
    p523_target: Mapping[str, Any], p509_target: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    require(p523_target.get("schema") == P523_TARGET_SCHEMA, "wrong P523 target schema")
    require(str(p523_target.get("status", "")).startswith("published"), "P523 target is not published")
    require(p509_target.get("schema") == P509_TARGET_SCHEMA, "wrong P509-compatible target schema")
    require(p509_target.get("status") == "published_candidate_set", "P509-compatible target is not published")
    verify_payload_hash(p523_target, "P523 target")
    verify_payload_hash(p509_target, "P509-compatible target")
    for key in (
        "scene_id", "instruction", "action", "target_instance_id", "target_class",
        "selected_candidate_id", "selected_surface_sha256", "binding_token_sha256",
        "candidate_surfaces", "target_resolution",
    ):
        require(p523_target.get(key) == p509_target.get(key), "target pair drift: %s" % key)
    primary_candidate = selected_candidate(p523_target, "P523 target")
    compatibility_candidate = selected_candidate(p509_target, "P509 target")
    options = compatibility_candidate.get("approach_options")
    require(isinstance(options, list) and len(options) >= 1, "target has no approach options")
    option_ids = [str(row.get("option_id", "")) for row in options if isinstance(row, Mapping)]
    require(len(option_ids) == len(options) and all(option_ids), "approach option is malformed")
    require(len(set(option_ids)) == len(option_ids), "approach option IDs are not unique")
    require(primary_candidate.get("approach_options") == options, "P523/P509 approach option drift")
    return primary_candidate, compatibility_candidate, list(options)

def audit_options(
    occupancy_path: Path,
    start: np.ndarray,
    options: Sequence[Mapping[str, Any]],
    config: p508.RoadmapConfig,
) -> tuple[list[dict[str, Any]], tuple[int, int], float, dict[str, np.ndarray]]:
    native = np.load(occupancy_path.resolve(strict=True), allow_pickle=False)
    base = p508.build_roadmap(native, config)
    p508.point_to_cell(start, np.asarray(base["free"]).shape, config, "requested start")
    snapped = p508.nearest_mask_cell(np.asarray(base["free"]), start, config)
    require(snapped is not None, "current root has no human-clear snap")
    start_cell, start_snap_distance = snapped
    audits: list[dict[str, Any]] = []
    for source_index, option in enumerate(options):
        roadmap = p508.relax_target_interaction_approaches(base, [option], config)
        row = p508.audit_approaches([option], roadmap, start_cell, config)[0]
        row["source_option_index"] = source_index
        if row.get("fresh_reachable") is True:
            row["current_occupancy_ranking_path_cost_m"] = (
                float(row["fresh_four_connected_distance_m"]) + float(row["snap_distance_m"])
            )
            row["terminal_clearance_gate_passed"] = (
                float(row["snapped_goal_clearance_m"])
                >= config.interaction_terminal_min_clearance_m - 1e-9
                and float(row["requested_cell_clearance_m"])
                >= config.interaction_terminal_min_clearance_m - 1e-9
            )
        else:
            row["current_occupancy_ranking_path_cost_m"] = None
            row["terminal_clearance_gate_passed"] = False
        audits.append(row)
    return audits, start_cell, float(start_snap_distance), base

def select_path_cost_first_option(
    p523_candidate: Mapping[str, Any],
    options: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    start: np.ndarray,
    start_cell: tuple[int, int],
    start_snap_distance: float,
) -> tuple[int, Mapping[str, Any], dict[str, Any]]:
    eligible = [
        row for row in audits
        if row.get("fresh_reachable") is True and row.get("terminal_clearance_gate_passed") is True
    ]
    require(eligible, "no approach passes current reachability and terminal clearance gates")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["current_occupancy_ranking_path_cost_m"]),
            float(row["fresh_four_connected_distance_m"]),
            float(row["snap_distance_m"]),
            int(row["source_option_index"]),
        ),
    )
    selected_index = int(selected["source_option_index"])
    require(str(options[selected_index].get("option_id")) == selected.get("option_id"),
            "selected option index/ID drift")

    prefilter = p523_candidate.get("route_prefilter")
    require(isinstance(prefilter, Mapping), "P523 target lacks fresh route_prefilter")
    require(
        prefilter.get("method") == "fresh_current_occupancy_four_connected_prefilter_before_polygon_navmesh",
        "P523 route_prefilter method drift",
    )
    require(prefilter.get("future_endpoint_or_motion_used") is False,
            "P523 route_prefilter used future endpoint or motion")
    prefilter_start = finite_vector(
        prefilter.get("current_query_start_world_xy_m"), 2, "prefilter current start"
    )
    require(float(np.linalg.norm(prefilter_start - start)) <= 1e-8,
            "P523 route_prefilter/current root drift")
    require(prefilter.get("snapped_start_cell") == list(start_cell),
            "P523 route_prefilter snapped-start drift")
    require(abs(float(prefilter.get("start_snap_distance_m")) - start_snap_distance) <= 1e-8,
            "P523 route_prefilter start snap-distance drift")
    target_audits = prefilter.get("option_audits")
    require(isinstance(target_audits, list) and len(target_audits) == len(options),
            "P523 route_prefilter option audit count drift")
    target_by_id = {str(row.get("option_id")): row for row in target_audits if isinstance(row, Mapping)}
    require(len(target_by_id) == len(options), "P523 route_prefilter option IDs drift")
    for row in audits:
        option_id = str(row["option_id"])
        target_row = target_by_id.get(option_id)
        require(isinstance(target_row, Mapping), "P523 route_prefilter omitted %s" % option_id)
        require(bool(target_row.get("reachable")) == (row.get("fresh_reachable") is True),
                "P523 route_prefilter reachability drift: %s" % option_id)
        require(
            bool(target_row.get("terminal_clearance_gate_passed"))
            == (row.get("terminal_clearance_gate_passed") is True),
            "P523 route_prefilter terminal-clearance drift: %s" % option_id,
        )
        if row.get("fresh_reachable") is True:
            require(target_row.get("snapped_goal_cell") == row.get("snapped_goal_cell"),
                    "P523 route_prefilter goal-cell drift: %s" % option_id)
            require(abs(float(target_row.get("fresh_four_connected_path_cost_m"))
                        - float(row["fresh_four_connected_distance_m"])) <= 1e-8,
                    "P523 route_prefilter path-cost drift: %s" % option_id)
            require(abs(float(target_row.get("terminal_snap_distance_m"))
                        - float(row["snap_distance_m"])) <= 1e-8,
                    "P523 route_prefilter terminal-snap drift: %s" % option_id)
            require(abs(float(target_row.get("ranking_path_cost_m"))
                        - float(row["current_occupancy_ranking_path_cost_m"])) <= 1e-8,
                    "P523 route_prefilter ranking-cost drift: %s" % option_id)
    target_reachable = [
        row for row in target_audits
        if row.get("reachable") is True
        and row.get("terminal_clearance_gate_passed") is True
    ]
    target_selected = min(
        target_reachable,
        key=lambda row: (float(row["ranking_path_cost_m"]), str(row["option_id"])),
    )
    require(target_selected.get("option_id") == selected.get("option_id"),
            "independent current path-cost selection disagrees with P523 route_prefilter")
    audit_payload = {
        "schema": SELECTOR_SCHEMA,
        "status": "verified_path_cost_first_approach_selected",
        "selection_rule": (
            "semantic target gate, fresh reachability and terminal-clearance gate, then "
            "fresh current-occupancy four-connected path cost plus terminal snap"
        ),
        "current_query_start_world_xy_m": start.astype(float).tolist(),
        "selected_approach_option_id": str(selected["option_id"]),
        "selected_source_option_index": selected_index,
        "selected_ranking_path_cost_m": float(selected["current_occupancy_ranking_path_cost_m"]),
        "option_audits": [dict(row) for row in audits],
        "p523_route_prefilter_verified": True,
        "future_endpoint_or_motion_used": False,
        "legacy_route_used_for_ranking": False,
        "surface_distance_used_before_path_cost": False,
    }
    audit_payload["selection_audit_sha256"] = canonical_hash(audit_payload)
    return selected_index, selected, audit_payload

def filtered_target(
    target: Mapping[str, Any], selected_option_id: str, selection_audit: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(target))
    candidate = result["candidate_surfaces"][0]
    for field in ("approach_options", "approach_candidates"):
        rows = candidate.get(field)
        if isinstance(rows, list):
            candidate[field] = [row for row in rows if row.get("option_id") == selected_option_id]
            require(len(candidate[field]) == 1, "filtered target lost selected option in %s" % field)
    result["current_only_roadmap_selection"] = dict(selection_audit)
    result.pop("receipt_payload_sha256", None)
    result["receipt_payload_sha256"] = canonical_hash(result)
    return result

def compile_current_request(
    p523_target_path: Path,
    p509_target_path: Path,
    occupancy_path: Path,
    current_state_path: Path,
    current_state_receipt_path: Path,
    request_path: Path,
    receipt_path: Path,
    config: p508.RoadmapConfig,
) -> dict[str, Any]:
    paths = {
        "P523 target": p523_target_path.resolve(strict=True),
        "P509 target": p509_target_path.resolve(strict=True),
        "occupancy": occupancy_path.resolve(strict=True),
        "current state": current_state_path.resolve(strict=True),
        "current state receipt": current_state_receipt_path.resolve(strict=True),
    }
    for label, path in paths.items():
        reject_historical(path, label)
    config.validate()
    require(
        config.navigation_clearance_mode == "target_interaction_relaxed_stage3_sdf_collision",
        "P523 current wrapper requires target-relaxed Stage3-SDF mode",
    )
    p523_target = read_json(paths["P523 target"])
    p509_target = read_json(paths["P509 target"])
    p523_candidate, _p509_candidate, options = validate_target_pair(p523_target, p509_target)
    start, start_source = p508.start_from_current_state(
        paths["current state"], paths["current state receipt"]
    )
    audits, start_cell, start_snap_distance, _base = audit_options(
        paths["occupancy"], start, options, config
    )
    selected_index, selected, selection_audit = select_path_cost_first_option(
        p523_candidate, options, audits, start, start_cell, start_snap_distance
    )
    selected_id = str(selected["option_id"])
    request_resolved = request_path.resolve()
    receipt_resolved = receipt_path.resolve()
    request_resolved.parent.mkdir(parents=True, exist_ok=True)
    receipt_resolved.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="p523_current_roadmap_") as temporary:
        filtered_path = Path(temporary) / "p509_selected_target.json"
        p508.atomic_json(filtered_path, filtered_target(p509_target, selected_id, selection_audit))
        p508.compile_request(
            filtered_path, paths["occupancy"], start,
            request_resolved, receipt_resolved, config, start_source=start_source,
        )

    request = read_json(request_resolved)
    provenance = request.get("compiler_provenance")
    require(isinstance(provenance, dict), "P508 request lacks compiler provenance")
    require(provenance.get("selected_approach_option_id") == selected_id,
            "P508 filtered compile selected a different option")
    provenance["stage1_sha256"] = p508.sha256_file(paths["P509 target"])
    provenance["current_only_option_selection"] = {
        "schema": SELECTOR_SCHEMA,
        "selection_audit_sha256": selection_audit["selection_audit_sha256"],
        "selected_approach_option_id": selected_id,
        "p523_target_sha256": p508.sha256_file(paths["P523 target"]),
        "p509_compatible_target_sha256": p508.sha256_file(paths["P509 target"]),
        "frozen_p508_compiler_reused_after_selection": True,
        "frozen_p508_selection_rule_used": False,
    }
    p508.atomic_json(request_resolved, request)

    receipt = read_json(receipt_resolved)
    receipt["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["producer_variant_schema"] = SELECTOR_SCHEMA
    receipt["inputs"]["stage1"] = p508.artifact(paths["P509 target"])
    receipt["inputs"]["p523_stage1_target"] = p508.artifact(paths["P523 target"])
    receipt["approach_audits"] = [dict(row) for row in audits]
    receipt["selected_approach_option_id"] = selected_id
    receipt["selection_rule"] = selection_audit["selection_rule"]
    receipt["current_only_option_selection"] = selection_audit
    receipt["p504_request"] = p508.artifact(request_resolved)
    receipt["verified_gates"] = {
        "P523_target_route_prefilter_hash_bound": True,
        "all_options_reaudited_on_fresh_occupancy": True,
        "semantic_and_terminal_clearance_gates_precede_path_cost": True,
        "fresh_current_path_cost_ranked_before_surface_distance": True,
        "frozen_P508_roadmap_construction_reused": True,
        "historical_route_used": False,
        "future_endpoint_or_motion_used": False,
    }
    receipt["selected_source_option_index"] = selected_index
    receipt.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = canonical_hash(receipt)
    p508.atomic_json(receipt_resolved, receipt)
    return receipt

