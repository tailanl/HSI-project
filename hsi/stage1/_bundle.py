"""Finalize the current-only P523 Stage1 handoff after fresh NavMesh/keynodes."""

from __future__ import annotations

import argparse

from datetime import datetime, timezone

import hashlib

import json

import math

import os

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

P523_TARGET_SCHEMA = "p523.current_only_stage1_target_surface.v1"

P523_BUNDLE_SCHEMA = "p523.current_only_stage1_bundle.v1"

P509_TARGET_SCHEMA = "p508.lingo_scene_first_stage1.v1"

P509_BUNDLE_SCHEMA = "p508.lingo_sam3_qwen_navmesh_stage1_bundle.v1"

GRAPH_SCHEMA = "p515.lingo_metric_scene_graph.v1"

SAM_SCHEMA = "p515.lingo_sam31_atomic_multiview_instances.v1"

REQUEST_SCHEMA = "p504.naomesh_human_roadmap_input.v1"

ROUTE_SCHEMA = "p508.pynavmesh_polygon_route_output.v1"

COMPILE_SCHEMA = "p508.lingo_stage1_roadmap_compile.v1"

KEYNODES_SCHEMA = "p510.stage1_root_navmesh_key_nodes.v1"

FORBIDDEN_RESULT_PREFIXES = ("p373", "p480", "p517", "p520", "p522")

class BundleError(RuntimeError):
    pass

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise BundleError(message)

def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}

def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value

def verify_payload_hash(value: Mapping[str, Any], label_name: str) -> str:
    recorded = value.get("receipt_payload_sha256")
    require(isinstance(recorded, str) and len(recorded) == 64, f"{label_name} lacks payload hash")
    payload = dict(value)
    payload.pop("receipt_payload_sha256", None)
    require(canonical_hash(payload) == recorded, f"{label_name} payload hash drift")
    return recorded

def reject_historical_result(path: Path, label_name: str) -> None:
    offenders = [
        part for part in (entry.lower() for entry in path.resolve().parts)
        if any(part == prefix or part.startswith(prefix + "_") for prefix in FORBIDDEN_RESULT_PREFIXES)
    ]
    require(not offenders, f"{label_name} points at a forbidden historical result: {offenders[0] if offenders else ''}")

def resolve_record(record: Mapping[str, Any], label_name: str) -> Path:
    require(isinstance(record.get("path"), str), f"{label_name} lacks artifact path")
    path = Path(str(record["path"])).resolve(strict=True)
    require(int(record.get("bytes", -1)) == path.stat().st_size, f"{label_name} byte drift")
    require(record.get("sha256") == sha256_file(path), f"{label_name} hash drift")
    return path

def finite_array(value: Any, shape: tuple[int, ...], label_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == shape, f"{label_name} must have shape {shape}")
    require(np.isfinite(array).all(), f"{label_name} contains non-finite values")
    return array

def route_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def same_target_semantics(primary: Mapping[str, Any], compatibility: Mapping[str, Any]) -> None:
    keys = (
        "scene_id", "instruction", "action", "target_class", "target_instance_id",
        "original_target_class", "requested_target_class", "semantic_override",
        "semantic_override_sha256",
        "selected_candidate_id", "selected_surface_sha256", "binding_token_sha256",
        "target_bounds_world_zup_m", "target_occupancy_mask_world_xyz_sha256",
        "candidate_surfaces", "query", "relation_evidence", "relation_evidence_sha256",
    )
    for key in keys:
        require(primary.get(key) == compatibility.get(key), f"P509 target compatibility drift: {key}")

def finalize(
    target_path: Path,
    p509_target_path: Path,
    graph_path: Path,
    sam_path: Path,
    request_path: Path,
    route_path: Path,
    compile_path: Path,
    keynodes_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    paths = {
        "stage1 target": target_path.resolve(strict=True),
        "P509 target": p509_target_path.resolve(strict=True),
        "metric graph": graph_path.resolve(strict=True),
        "SAM receipt": sam_path.resolve(strict=True),
        "roadmap request": request_path.resolve(strict=True),
        "NavMesh route": route_path.resolve(strict=True),
        "roadmap compile receipt": compile_path.resolve(strict=True),
        "keynodes": keynodes_path.resolve(strict=True),
    }
    for name, path in paths.items():
        reject_historical_result(path, name)
    output_dir = output_dir.resolve()
    require(not output_dir.exists() or not any(output_dir.iterdir()), f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    target = read_json(paths["stage1 target"])
    compatibility_target = read_json(paths["P509 target"])
    graph = read_json(paths["metric graph"])
    sam = read_json(paths["SAM receipt"])
    request = read_json(paths["roadmap request"])
    route = read_json(paths["NavMesh route"])
    compile_receipt = read_json(paths["roadmap compile receipt"])
    keynodes = read_json(paths["keynodes"])

    require(target.get("schema") == P523_TARGET_SCHEMA, "wrong P523 target schema")
    require(str(target.get("status", "")).startswith("published"), "P523 target is not published")
    verify_payload_hash(target, "P523 target")
    require(compatibility_target.get("schema") == P509_TARGET_SCHEMA, "wrong P509-compatible target schema")
    require(compatibility_target.get("status") == "published_candidate_set", "P509-compatible target is not published")
    verify_payload_hash(compatibility_target, "P509-compatible target")
    same_target_semantics(target, compatibility_target)
    require(target.get("action") == target.get("action_family") == "sit", "Stage1 action must be sit")
    candidates = target.get("candidate_surfaces")
    require(isinstance(candidates, list) and len(candidates) == 1, "Stage1 must contain one surface")
    candidate = candidates[0]
    require(isinstance(candidate, Mapping), "Stage1 candidate is malformed")
    surface_id = str(target.get("selected_candidate_id"))
    surface_sha256 = str(target.get("selected_surface_sha256"))
    target_id = str(target.get("target_instance_id"))
    target_class = str(target.get("target_class"))
    original_target_class = str(target.get("original_target_class", target_class))
    require(candidate.get("candidate_id") == surface_id, "selected surface ID drift")
    require(candidate.get("support_component_sha256") == surface_sha256, "selected surface hash drift")
    require(candidate.get("target_instance_id") == target_id, "selected target instance drift")
    require(candidate.get("target_class") == target_class, "selected target class drift")
    require(candidate.get("original_class", original_target_class) == original_target_class, "selected original class drift")
    semantic_override = target.get("semantic_override")
    require(isinstance(semantic_override, Mapping), "target semantic override audit is missing")
    semantic_override_sha256 = str(target.get("semantic_override_sha256", ""))
    override_payload = dict(semantic_override)
    recorded_nested_override_hash = override_payload.pop("semantic_override_sha256", None)
    require(
        len(semantic_override_sha256) == 64
        and recorded_nested_override_hash == semantic_override_sha256
        and canonical_hash(override_payload) == semantic_override_sha256,
        "semantic override hash drift",
    )
    require(candidate.get("semantic_override_sha256") == semantic_override_sha256, "candidate semantic override drift")
    owner_bounds = finite_array(target.get("target_bounds_world_zup_m"), (2, 3), "target owner bounds")
    surface_bounds = finite_array(candidate.get("surface", {}).get("bounds_world_zup_m"), (2, 3), "surface bounds")
    require(np.all(surface_bounds[0] >= owner_bounds[0] - 1e-6) and np.all(surface_bounds[1] <= owner_bounds[1] + 1e-6), "surface bounds escape target owner bounds")
    binding_token = str(target.get("binding_token_sha256"))
    require(len(binding_token) == 64 and candidate.get("binding_token_sha256") == binding_token, "binding token drift")

    require(graph.get("schema") == GRAPH_SCHEMA, "wrong metric graph schema")
    verify_payload_hash(graph, "metric graph")
    require(sam.get("schema") == SAM_SCHEMA and sam.get("status") == "atomic_instances_ready", "SAM instances are not ready")
    verify_payload_hash(sam, "SAM receipt")
    for value, label_name in ((graph, "graph"), (sam, "SAM")):
        require(value.get("scene_id") == target.get("scene_id"), f"{label_name}/target scene drift")
        require(value.get("instruction") == target.get("instruction"), f"{label_name}/target instruction drift")
    target_artifacts = target.get("artifacts")
    require(isinstance(target_artifacts, Mapping), "target artifacts are missing")
    require(resolve_record(target_artifacts["metric_graph"], "target graph artifact") == paths["metric graph"], "target graph path drift")
    require(resolve_record(target_artifacts["atomic_instances"], "target SAM artifact") == paths["SAM receipt"], "target SAM path drift")
    mask_record = candidate.get("target_occupancy_mask")
    require(isinstance(mask_record, Mapping), "candidate target occupancy mask is missing")
    mask_path = resolve_record(mask_record, "target occupancy mask")
    require(mask_record.get("world_xyz_content_sha256") == target.get("target_occupancy_mask_world_xyz_sha256"), "target mask content identity drift")
    face_record = candidate.get("surface_face_archive")
    require(isinstance(face_record, Mapping), "candidate surface face archive is missing")
    face_path = resolve_record(face_record, "surface face archive")

    require(request.get("schema") == REQUEST_SCHEMA, "wrong roadmap request schema")
    provenance = request.get("compiler_provenance")
    require(isinstance(provenance, Mapping), "roadmap request lacks compiler provenance")
    require(provenance.get("legacy_p480_route_reused") is False, "roadmap request reused a historical route")
    accepted_target_hashes = {sha256_file(paths["stage1 target"]), sha256_file(paths["P509 target"])}
    require(provenance.get("stage1_sha256") in accepted_target_hashes, "roadmap request is not bound to a current target receipt")
    require(provenance.get("selected_candidate_id") == surface_id, "roadmap request surface drift")
    option_id = str(provenance.get("selected_approach_option_id"))
    options = candidate.get("approach_candidates")
    require(isinstance(options, list), "candidate approaches are missing")
    matches = [row for row in options if isinstance(row, Mapping) and row.get("option_id") == option_id]
    require(len(matches) == 1, "selected approach is absent or ambiguous")
    selected_option = matches[0]
    require(selected_option.get("requires_fresh_route_planning") is True, "selected approach did not require fresh routing")
    require(selected_option.get("source_route_copied_to_final_route") is False, "selected approach admits copied routing")
    request_goal = finite_array(request.get("goal_world_xy_m"), (2,), "roadmap goal")
    approach_goal = finite_array(selected_option.get("approach_world_xy_m"), (2,), "selected approach")
    require(float(np.linalg.norm(request_goal - approach_goal)) <= 1e-8, "roadmap goal/selected approach drift")
    request_npz = request.get("input_npz")
    require(isinstance(request_npz, Mapping), "roadmap request lacks input NPZ")
    request_npz_path = resolve_record(request_npz, "roadmap input NPZ")

    require(route.get("schema") == ROUTE_SCHEMA, "wrong NavMesh route schema")
    require(route.get("status") == "verified_polygon_navmesh_route", "NavMesh route is not verified")
    require(route.get("validation", {}).get("all_segment_samples_human_free") is True, "NavMesh route collision audit failed")
    require(route.get("backend", {}).get("external_polygon_navmesh") is True, "route did not use external polygon NavMesh")
    route_source = route.get("source")
    require(isinstance(route_source, Mapping), "NavMesh route lacks source binding")
    require(route_source.get("request_json_sha256") == sha256_file(paths["roadmap request"]), "route/request hash drift")
    require(route_source.get("input_npz_sha256") == sha256_file(request_npz_path), "route/input NPZ hash drift")
    route_points = np.asarray(route.get("route_world_xy_m"), dtype=np.float64)
    require(route_points.ndim == 2 and route_points.shape[1] == 2 and len(route_points) >= 2, "route polyline is malformed")
    require(np.isfinite(route_points).all(), "route polyline contains non-finite values")

    require(compile_receipt.get("schema") == COMPILE_SCHEMA, "wrong roadmap compile schema")
    require(compile_receipt.get("status") == "fresh_roadmap_request_compiled", "roadmap compile is incomplete")
    require(compile_receipt.get("selected_approach_option_id") == option_id, "compile selected approach drift")
    compile_stage1 = compile_receipt.get("inputs", {}).get("stage1")
    compile_request = compile_receipt.get("p504_request")
    require(isinstance(compile_stage1, Mapping) and compile_stage1.get("sha256") in accepted_target_hashes, "compile receipt target drift")
    require(isinstance(compile_request, Mapping) and compile_request.get("sha256") == sha256_file(paths["roadmap request"]), "compile receipt request drift")
    require(compile_receipt.get("route_provenance", {}).get("legacy_p480_route_polyline_read") is False, "compile receipt read a historical route")

    require(keynodes.get("schema") == KEYNODES_SCHEMA, "wrong keynode schema")
    verify_payload_hash(keynodes, "keynodes")
    key_inputs = keynodes.get("inputs")
    require(isinstance(key_inputs, Mapping), "keynodes lack input bindings")
    require(key_inputs.get("stage1", {}).get("sha256") in accepted_target_hashes, "keynodes target drift")
    require(key_inputs.get("roadmap_request", {}).get("sha256") == sha256_file(paths["roadmap request"]), "keynodes request drift")
    require(key_inputs.get("navmesh_route", {}).get("sha256") == sha256_file(paths["NavMesh route"]), "keynodes route drift")
    require(keynodes.get("navmesh_execution_route", {}).get("all_segment_samples_human_free") is True, "keynode route is not collision-audited")

    relation = target.get("query", {}).get("relation")
    references = target.get("query", {}).get("reference_instance_ids")
    evidence = target.get("relation_evidence")
    require(isinstance(references, list) and isinstance(evidence, list), "relation contract is malformed")
    if relation is None:
        require(references == [] and evidence == [], "direct target must not fabricate relation evidence")
    require(canonical_hash(evidence) == target.get("relation_evidence_sha256"), "relation evidence hash drift")

    artifacts = {
        "stage1_target": artifact(paths["stage1 target"]),
        "target": artifact(paths["stage1 target"]),
        "target_surface": artifact(paths["stage1 target"]),
        "p509_compatible_stage1_target": artifact(paths["P509 target"]),
        "roadmap_request": artifact(paths["roadmap request"]),
        "request": artifact(paths["roadmap request"]),
        "navmesh_route": artifact(paths["NavMesh route"]),
        "route": artifact(paths["NavMesh route"]),
        "roadmap_compile": artifact(paths["roadmap compile receipt"]),
        "keynodes": artifact(paths["keynodes"]),
        "key_nodes": artifact(paths["keynodes"]),
        "stage1_key_nodes": artifact(paths["keynodes"]),
        "metric_graph": artifact(paths["metric graph"]),
        "atomic_instances": artifact(paths["SAM receipt"]),
        "target_surface_faces": artifact(face_path),
        "target_occupancy_mask": artifact(mask_path),
    }
    common: dict[str, Any] = {
        "status": "complete_stage1_verified",
        "producer_variant_schema": P523_BUNDLE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_id": target["scene_id"],
        "instruction": target["instruction"],
        "action": "sit",
        "action_family": "sit",
        "target_instance_id": target_id,
        "selected_target_instance_id": target_id,
        "target_class": target_class,
        "original_target_class": original_target_class,
        "requested_target_class": target.get("requested_target_class", target_class),
        "semantic_override": semantic_override,
        "semantic_override_sha256": semantic_override_sha256,
        "selected_surface_id": surface_id,
        "selected_surface_sha256": surface_sha256,
        "support_component_sha256": surface_sha256,
        "selected_approach_option_id": option_id,
        "binding_token_sha256": binding_token,
        "target_bounds_world_zup_m": owner_bounds.astype(float).tolist(),
        "surface_bounds_world_zup_m": surface_bounds.astype(float).tolist(),
        "target_occupancy_mask_world_xyz_sha256": target["target_occupancy_mask_world_xyz_sha256"],
        "query": target["query"],
        "relation_evidence": evidence,
        "relation_evidence_sha256": target["relation_evidence_sha256"],
        "target_resolution": target["target_resolution"],
        "artifacts": artifacts,
        "route_summary": {
            "backend": route.get("backend", {}).get("backend_label"),
            "external_polygon_navmesh": True,
            "route_point_count": int(len(route_points)),
            "route_length_m": route_length(route_points),
            "all_segment_samples_human_free": True,
            "minimum_sampled_clearance_m": route.get("validation", {}).get("minimum_sampled_clearance_m"),
            "navigation_clearance_mode": route.get("polygon_navmesh", {}).get("navigation_clearance_mode"),
            "semantic_key_node_count": keynodes.get("semantic_key_node_summary", {}).get("node_count"),
        },
        "metric_graph_contract": {
            "metric_graph_receipt_sha256": sha256_file(paths["metric graph"]),
            "metric_graph_payload_sha256": graph["receipt_payload_sha256"],
            "instruction_query": graph.get("instruction_query"),
            "graph_selected_target_instance_id": graph.get("selected_target_instance_id"),
            "p523_resolved_target_instance_id": target_id,
            "target_object_class": target_class,
            "metric_graph_original_object_class": original_target_class,
            "semantic_override_sha256": semantic_override_sha256,
            "required_relation": relation,
            "reference_entity_ids": references,
            "relation_evidence_sha256": target["relation_evidence_sha256"],
            "surface_binding_token_sha256": binding_token,
        },
        "verified_gates": {
            "current_p515_metric_graph_hash_bound": True,
            "current_sam_target_support_hash_bound": True,
            "semantic_override_geometry_guard_passed_or_not_required": (
                semantic_override.get("enabled") is False
                or semantic_override.get("shape_validation", {}).get("gate_pass") is True
            ),
            "unique_target_surface_bound_inside_owner": True,
            "target_occupancy_mask_is_scene_subset": True,
            "fresh_roadmap_bound_to_current_surface": True,
            "external_polygon_navmesh_route": True,
            "route_independently_raster_collision_free": True,
            "keynodes_bound_to_dense_route": True,
            "historical_surface_or_route_reused": False,
            "future_motion_endpoint_used": False,
        },
        "lineage_contract": {
            "frontend": "current P515 metric graph + current SAM original-OBJ support -> P523 target-bound surface",
            "planner": "current occupancy roadmap -> external polygon NavMesh -> current bound keynodes",
            "old_stage1_bundle_surface_or_route_reused": False,
            "P373_P480_P517_P520_P522_result_used": False,
            "stage2_handoff_allowed": True,
            "stage3_handoff_allowed": True,
        },
        "source": artifact(Path(__file__)),
    }
    primary = {"schema": P523_BUNDLE_SCHEMA, **common}
    primary["receipt_payload_sha256"] = canonical_hash(primary)
    primary_path = output_dir / "receipt.json"
    atomic_json(primary_path, primary)

    compat_artifacts = dict(artifacts)
    compat_artifacts["stage1"] = artifact(paths["P509 target"])
    compat_artifacts["request"] = artifact(paths["roadmap request"])
    compat_artifacts["route"] = artifact(paths["NavMesh route"])
    compatibility = {"schema": P509_BUNDLE_SCHEMA, **common, "artifacts": compat_artifacts}
    compatibility["receipt_payload_sha256"] = canonical_hash(compatibility)
    atomic_json(output_dir / "p509_compatible_bundle.json", compatibility)
    return primary

