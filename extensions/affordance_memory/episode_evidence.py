"""Read-only, stage-scoped admission of existing evidence, never success laundering.

No current P550/P552/P553/P554 runner is registered as a complete online
Memory producer. Consequently this version cannot issue production positive
credit. A valid local SHA seal establishes integrity, not model authenticity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from memory_common import PROJECT, artifact, verified, read_json, read_sealed, digest, require

REQUEST_SCHEMA = "p555.admission_request.v1"
ADMISSION_SCHEMA = "p555.episode_admission.v1"
FIXTURE_SCHEMA = "p555.fixture_episode_evidence.v1"
SCENE_SCHEMA = "p550.final_fixed_scene_understanding.v1"
SELECTION_SCHEMA = "p533.terminal_candidate_selection.v4_transactional"
STAGE12_SCHEMAS = {
    "p552.fixed_scene_contact_region_to_keypose_sequence.v1",
    "p553.fixed_scene_contact_region_to_single_image_keypose_sequence.v1",
    *("p550.fixed_scene_to_verified_keypose_sequence.v" + str(i) for i in range(1, 7)),
}
GATES = (
    "collision_free", "keypose_reached", "interaction_surface_valid",
    "foot_slip_within_gate", "support_valid", "temporal_quality",
    "task_completed", "single_diffusion_chain", "trained_checkpoint",
)
PHYSICAL_GATES = {
    "contact_anchor_clearance_bounded", "contact_anchor_locked",
    "contact_point_inside_bound_surface", "contact_point_on_true_support_mask",
    "current_stage1_target_mask_hash_bound", "direct_target_relation_valid",
    "ground_penetration", "hips_support_inside_surface", "known_action_family_sit",
    "left_foot_ground_contact_area", "non_target_scene_penetration",
    "right_foot_ground_contact_area", "sdf_grid_coverage", "selected_surface_hash_bound",
    "target_allowed_contact_area", "target_allowed_contact_penetration",
    "target_forbidden_body_penetration", "terminal_facing",
}
BACKGROUND_GATES = {"enough_observed_background", "background_structure_preserved",
                    "background_intensity_consistent", "localized_unexplained_change_bounded"}
POSTURE_GATES = {"ankles_near_floor", "ankles_forward_of_hips",
                 "feet_face_open_seat_side", "shins_not_strongly_folded_back"}
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}")


def _boolean(value, label):
    require(type(value) is bool, label + " must be a real boolean")
    return value


def _xy(value):
    require(isinstance(value, list) and len(value) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in value),
            "Missing finite query start XY")
    return [float(x) for x in value]


def _family(scene_id):
    # Conservative LINGO grouping: suffix variants cannot count as new families.
    match = re.match(r"^(\d+)(?:[-_]|$)", scene_id)
    return "lingo:" + str(int(match[1])).zfill(3) if match else "scene:" + scene_id


def _seal(value, field="receipt_payload_sha256"):
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}),
            "Evidence seal mismatch: " + field)
    return value


def _source_record(record):
    require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}
            and isinstance(record["path"], str) and Path(record["path"]).suffix == ".json"
            and type(record["bytes"]) is int and 0 <= record["bytes"] <= MAX_JSON_BYTES
            and isinstance(record["sha256"], str) and HEX64.fullmatch(record["sha256"]),
            "Source must be an exact bounded JSON artifact, never model weights")
    return record


class _Reader:
    def __init__(self):
        self.records = {}

    def check(self, record, *, within=None):
        require(isinstance(record, dict) and type(record.get("bytes")) is int
                and 0 <= record["bytes"] <= MAX_ARTIFACT_BYTES, "Oversized/missing evidence artifact")
        require(Path(record.get("path", "")).suffix.lower() not in
                {".safetensors", ".pt", ".pth", ".ckpt", ".bin"}, "Model weight hashing is not admission")
        key = (record.get("path"), record.get("bytes"), record.get("sha256"))
        if key not in self.records:
            path = verified(record)
            self.records[key] = dict(record)
        else:
            path = Path(record["path"])
        require(within is None or path.is_relative_to(within), "Evidence escapes its exact case: " + str(path))
        return path

    def read(self, record, *, schema=None, within=None, sealed=True):
        require(record.get("bytes", MAX_JSON_BYTES + 1) <= MAX_JSON_BYTES, "JSON evidence exceeds bound")
        path = self.check(record, within=within)
        value = read_sealed(path) if sealed else read_json(path)
        require(schema is None or value.get("schema") == schema, "Unexpected receipt schema")
        return value

    def geometry(self, record, scene_id):
        value = self.read(record)
        require(value.get("scene_id") == scene_id and isinstance(value.get("objects"), list),
                "Fixed geometry scene mismatch")
        for key in ("source_mesh", "source_occupancy"):
            self.check(value[key])
        fingerprint = digest({"mesh_sha256": value["source_mesh"]["sha256"],
                              "occupancy_sha256": value["source_occupancy"]["sha256"]})
        return value, fingerprint

    def scene(self, record, *, current=True):
        value = self.read(record, schema=SCENE_SCHEMA)
        require(value.get("human_instruction_read") is False
                and value.get("stage2_or_stage3_output_read") is False
                and (not current or value.get("semantic_backrest_review_complete") is True),
                "Not a final instruction-independent scene publication")
        require(isinstance(value.get("scene_id"), str) and value["scene_id"], "Missing scene identity")
        geometry, fingerprint = self.geometry(value["fixed_geometry"], value["scene_id"])
        self.read(value["fixed_semantics"])
        return value, geometry, fingerprint


def _verdict(reader, source, query, classification, reason, *, domain=None, checks=None):
    for record in reader.records.values():
        verified(record)  # Detect drift across the complete selected read transaction.
    identity = {k: v for k, v in query.items() if k != "scene_family"}
    episode = digest(identity)  # No output path, seed, retry count or timestamp.
    checks = dict(checks or {})
    checks["complete_query_identity_available"] = set(identity) == {
        "scene_id", "scene_fingerprint", "fixed_geometry_sha256", "instruction", "start_world_xy_m"}
    return {
        "schema": ADMISSION_SCHEMA,
        "attempt_id": digest({"source_sha256": source["sha256"], "query_sha256": episode}),
        "episode_id": episode, "query_id": episode, "classification": classification,
        "failure_domain": domain, "real_positive_credit_allowed": False,
        "critical_memory_failure": False, "affected_record_ids": [],
        "allowed_record_ids": [], "record_extraction_receipts": [],
        "scene_id": query.get("scene_id"), "scene_fingerprint": query.get("scene_fingerprint"),
        "scene_family": query.get("scene_family", _family(query.get("scene_id", "unknown"))),
        "source_bindings": list(reader.records.values()), "checks": checks, "reason": reason,
        "purpose": "production", "fixture_success_allowed": False,
    }


def _scene_query(scene, fingerprint):
    return {"scene_id": scene["scene_id"], "scene_fingerprint": fingerprint,
            "scene_family": _family(scene["scene_id"]),
            "fixed_geometry_sha256": scene["fixed_geometry"]["sha256"]}


def _same(value, key, wanted):
    require(value.get(key) == wanted, "Evidence binding mismatch: " + key)


def _stage2(reader, record, stage1, preparation, descriptions, *, root):
    """Follow only the current row's wrapper -> controller -> candidate edges."""
    value = reader.read(record, within=root)
    _same(value, "source_stage1", stage1)
    prep = reader.read(preparation, within=root)
    _same(prep, "source_stage1", stage1)
    require(prep.get("ready_for_fresh_H3") is True, "Stage2 row used an unready preparation")
    view = reader.read(prep["view"])
    _same(view, "source_stage1_execution", stage1)
    reader.read(prep["sdf"])
    if value.get("schema") == "p553.fresh_description_and_verified_image_keypose.v1":
        _same(value, "source_preparation", preparation)
        nested = reader.read(value["verified_controller"], within=root)
        _same(nested, "source_stage1", stage1)
        _same(value, "verified_stage2_keypose", nested.get("verified_stage2_keypose"))
        require(value.get("stage3_handoff_allowed") is False, "Unexpected legacy Stage3 handoff")
        value = nested
    require(value.get("schema") in {
        "p553.fresh_verified_single_image_keypose.v1",
        "p550.fresh_verified_keypose_with_visibility_recovery.v3",
        "p550.fresh_verified_keypose_with_visibility_recovery.v4",
        "p550.fresh_verified_keypose_with_visibility_recovery.v5",
    }, "Unsupported Stage2 verification protocol; do not infer its gates")
    _same(value, "source_view", prep["view"])
    _same(value, "source_sdf", prep["sdf"])
    _same(value, "keypoint_descriptions", descriptions)
    require(value.get("actual_stage3_motion_generated") is False
            and value.get("stage3_handoff_allowed") is False, "Legacy Stage2 cannot assert online motion")
    passed = _boolean(value.get("verified_stage2_keypose"), "verified_stage2_keypose")
    candidate = reader.read(value["candidate_execution"], within=root)
    for key, expected in (("source_stage1", stage1), ("source_view", prep["view"]), ("source_sdf", prep["sdf"])):
        _same(candidate, key, expected)
    _same(candidate, "keypoint_descriptions", value["keypoint_descriptions"])
    description = reader.read(value["keypoint_descriptions"])
    _same(description, "source_stage1", stage1)
    background = None
    if value.get("background_consistency"):
        _same(candidate, "background_consistency", value["background_consistency"])
        background = reader.read(value["background_consistency"], within=root)
        for key, expected in (("source_stage1", stage1), ("source_view", prep["view"]),
                              ("source_h3", candidate["h3_receipt"]),
                              ("source_recovery", candidate["hybrikx_receipt"])):
            _same(background, key, expected)
        gates = background.get("gates")
        require(isinstance(gates, dict) and set(gates) == BACKGROUND_GATES
                and all(type(x) is bool for x in gates.values()),
                "Missing actual background gate results")
        require(_boolean(background.get("passed"), "background.passed") == all(gates.values()),
                "Background aggregate disagrees with its gates")
        # Select only the actual diagnostic inputs; never walk model manifests.
        require(background.get("schema") in {
            "p550.generated_fixed_camera_background_audit.v1",
            "p553.generated_fixed_camera_single_image_background_audit.v1",
        }, "Unknown background audit protocol")
        # P553 retains the numeric P550 schema while honestly changing this
        # artifact field to image; distinguish by its bound media receipt.
        h3 = reader.read(candidate["h3_receipt"])
        generated = ("actual_generated_image" if h3.get("schema") == "p553.h3_single_image_case.v1"
                     else "actual_generated_frame")
        require(generated in background and not (
            "actual_generated_frame" in background and "actual_generated_image" in background),
            "Ambiguous generated-image/frame evidence")
        for key in ("actual_reference", generated):
            reader.check(background[key])
        reader.read(candidate["hybrikx_receipt"])
    if not passed:
        require(value.get("status") == "failed_not_publishable", "Stage2 failure/status contradiction")
        if background is not None and background["passed"] is False:
            return False, "stage2_background_preservation", str(value.get("reason", "Background gate failed"))
        return False, "stage2_verification", str(value.get("reason", "Stage2 rejected"))
    require(value.get("status") == "complete_verified_stage2_keypose", "Stage2 success/status contradiction")
    require(candidate.get("candidate_all_original_physical_gates_passed") is True,
            "Missing actual physical candidate pass")
    require(value.get("all_original_physical_gate_count") == 18
            and value.get("additional_neutral_sit_gate_count") == 4, "Unsupported physical/pose gate counts")
    if value["schema"] in {"p550.fresh_verified_keypose_with_visibility_recovery.v5",
                           "p553.fresh_verified_single_image_keypose.v1"}:
        require(background is not None and background["passed"] is True, "Current protocol lacks background pass")
    bundle = reader.read(value["refine_bundle"], within=root)
    _same(candidate, "refine_bundle", value["refine_bundle"])
    gates = bundle.get("selected_gates")
    require(isinstance(gates, dict) and set(gates) == PHYSICAL_GATES and all(x is True for x in gates.values()),
            "Actual 18 physical gates are not all true")
    _same(bundle["outputs"], "published_keypose", value["keypose"])
    reader.check(value["keypose"], within=root)
    quality_receipt = reader.read(value["neutral_sitting_quality"], within=root)
    _same(quality_receipt, "bundle", value["refine_bundle"])
    _same(quality_receipt, "keypose", value["keypose"])
    quality = quality_receipt["quality"]
    extra = quality.get("neutral_sit_quality_gates")
    require(isinstance(extra, dict) and set(extra) == POSTURE_GATES and all(x is True for x in extra.values())
            and quality.get("all_neutral_sit_quality_gates_passed") is True, "Missing four posture checks")
    post = reader.read(value["post_refine_qwen"], within=root)
    _same(post, "source_stage2", value["candidate_execution"])
    _same(post, "source_refine_bundle", value["refine_bundle"])
    require(post.get("decision", {}).get("post_refine_semantic_pass") is True
            and post["decision"].get("failed_checks") == [], "Final semantic rejection")
    _boolean(value.get("explicit_visibility_only_recovery"), "visibility recovery status")
    # This only records a historical static protocol pass, never authenticates
    # a new Qwen invocation or supplies missing motion evidence.
    return True, None, "Verified static Stage2 only; full motion and online producer evidence absent"


def _case(reader, source, value):
    root = Path(source["path"]).parent
    scene, geometry, fingerprint = reader.scene(value["source_fixed_scene"], current=False)
    _same(value, "scene_id", scene["scene_id"])
    launch = reader.read(value["launch"], within=root)
    binding = launch.get("bindings", {})
    _same(binding, "scene", value["source_fixed_scene"])
    _same(binding, "instruction", value["instruction"])
    require(isinstance(value["instruction"], str) and value["instruction"].strip(), "Empty query instruction")
    query = _scene_query(scene, fingerprint)
    query.update(instruction=value["instruction"], start_world_xy_m=_xy(binding.get("start_world_xy_m")))
    require(value.get("stage3_motion_generated") is False, "Existing Stage12 cannot claim complete online motion")
    checks = {"source_receipt_integrity": True, "query_binding": True,
              "complete_motion_evidence_present": False, "production_producer_registered": False}
    if not value.get("stage1_sequence"):
        return _verdict(reader, source, query, "unknown", "No bound Stage1 planning receipt", checks=checks)
    plan = reader.read(value["stage1_sequence"], within=root)
    for key, expected in (("scene_id", scene["scene_id"]), ("instruction", value["instruction"]),
                          ("source_fixed_geometry", scene["fixed_geometry"])):
        _same(plan, key, expected)
    require(_xy(plan.get("initial_start_world_xy_m")) == query["start_world_xy_m"], "Stage1 start drift")
    steps = plan.get("interaction_steps")
    require(isinstance(steps, list), "Missing ordered interaction plan")
    rows = value.get("stage2_steps")
    require(isinstance(rows, list) and len(rows) <= len(steps), "Invalid Stage2 row coverage")
    failures = []
    for row in rows:
        index = row.get("interaction_index")
        require(type(index) is int and 0 <= index < len(steps), "Bad interaction index")
        step = steps[index]
        require(step.get("interaction_index") == index, "Interaction plan index drift")
        _same(row, "stage1", step.get("stage1_execution"))
        stage1 = reader.read(row["stage1"], within=root)
        _same(stage1, "source_fixed_geometry", scene["fixed_geometry"])
        _same(stage1, "scene_id", scene["scene_id"])
        target = reader.read(stage1["target"], within=root)
        require(target.get("target_instance_id") == step.get("selected_target_id"), "Target ID drift")
        ok, domain, reason = _stage2(reader, row["stage2"], row["stage1"], row["preparation"],
                                    step["keynode_descriptions"], root=root)
        if not ok:
            failures.append((domain, reason))
    require(len({r["interaction_index"] for r in rows}) == len(rows), "Repeated Stage2 interaction")
    completed = _boolean(value.get("all_requested_interactions_completed"), "all_requested_interactions_completed")
    if completed:
        require(value.get("status") == "complete_verified_stage12_keypose_sequence"
                and steps and len(rows) == len(steps) and not failures, "Incomplete/rejected steps marked completed")
        sequence = reader.read(value["keypose_sequence"], within=root)
        _same(sequence, "scene_id", scene["scene_id"])
        _same(sequence, "instruction", value["instruction"])
        _same(sequence, "source_stage1", value["stage1_sequence"])
        require(len(sequence.get("keyposes", [])) == len(steps), "Published sequence coverage drift")
        for index, item in enumerate(sequence["keyposes"]):
            require(item.get("interaction_index") == index, "Published keypose order drift")
            _same(item, "stage2_verification", rows[index]["stage2"])
        classification, domain, reason = "pending_motion_validation", None, (
            "Static Stage2 completed under its recorded protocol; no same-query complete motion. "
            "Old route/geometry validity is not automatically inherited by the current pipeline.")
    elif failures:
        classification, domain, reason = "failure", failures[0][0], failures[0][1]
    elif plan.get("stage2_sequence_handoff_allowed") is False:
        classification, domain, reason = "failure", "stage1_planning", str(plan.get("reason", plan.get("status")))
    else:
        classification, domain, reason = "unknown", None, "Missing completed downstream evidence; not a fabricated failure"
    checks.update(static_stage2_complete=completed,
                  final_backrest_publication_used=scene.get("semantic_backrest_review_complete") is True,
                  original_visibility_recovery_not_rewritten=True,
                  no_gate_inferred_from_inner_handoff=True, current_route_revalidation_claimed=False)
    return _verdict(reader, source, query, classification, reason, domain=domain, checks=checks)


def _selection(reader, source, value):
    _seal(value, "selection_payload_sha256")
    audits = value.get("candidate_audits")
    require(isinstance(audits, list) and audits and len(audits) == value.get("candidate_count"),
            "Incomplete legacy candidate selection")
    evaluations = []
    for row in audits:
        _seal(row, "audit_payload_sha256")
        root = Path(row["candidate_dir"]).resolve()
        require(root.is_relative_to(Path(source["path"]).parent), "Candidate from another selection")
        evidence = row["artifacts"]
        evaluation = reader.read(evidence["fullmesh_evaluation"], within=root, sealed=False)
        for key in ("motion", "metadata", "energy_log"):
            reader.check(evidence[key], within=root)
            _same(evaluation["artifacts"], key, evidence[key])
        evaluations.append(evaluation)
    scene_id = evaluations[0]["scene_id"]
    require(all(x["scene_id"] == scene_id for x in evaluations), "Mixed-scene selection")
    packet = evaluations[0]["artifacts"]["packet"]
    require(all(x["artifacts"]["packet"] == packet for x in evaluations), "Mixed-query selection")
    reader.read(packet, sealed=False)
    occupancy = evaluations[0]["artifacts"]["occupancy"]
    reader.check(occupancy)
    query = {"scene_id": scene_id, "scene_family": _family(scene_id),
             "scene_fingerprint": digest({"legacy_occupancy_sha256": occupancy["sha256"]}),
             "legacy_packet_sha256": packet["sha256"]}
    failed = value.get("status") == "fail_closed_no_candidate_passed"
    if failed:
        require(value.get("publication_authorized") is False and value.get("selected_candidate") is None
                and all(x.get("accepted") is False for x in audits), "Rejected selection claims publication")
    return _verdict(reader, source, query, "failure" if failed else "unknown",
                    "Legacy P533 evaluation only; not a registered current online Memory producer",
                    domain="legacy_stage3_quality" if failed else None,
                    checks={"legacy_protocol_only": True, "current_scene_fingerprint_available": False,
                            "complete_online_memory_evidence_present": False})


def classify_stage12(source_path) -> dict:
    """Classify an exact source; integrity/binding violations raise, not succeed."""
    path = Path(source_path).resolve(strict=True)
    require(path.suffix == ".json" and path.stat().st_size <= MAX_JSON_BYTES, "Source receipt must be bounded JSON")
    require(path.is_relative_to(PROJECT / "agent9/runs"), "Source must be an actual project run, not copied/fixture evidence")
    require(not any(x in {"tests", "fixtures", "test_fixtures"} for x in path.parts), "Fixture source is not production")
    reader = _Reader()
    source = artifact(path)
    value = reader.read(source, sealed=False)
    require(value.get("namespace") not in {"test_fixture", "synthetic_fixture"}
            and value.get("schema") != FIXTURE_SCHEMA
            and value.get("fixture") is not True and value.get("is_fixture") is not True
            and value.get("synthetic_fixture") is not True,
            "Fixture evidence is forbidden in production")
    if value.get("schema") == SELECTION_SCHEMA:
        return _selection(reader, source, value)
    _seal(value)
    if value.get("schema") == SCENE_SCHEMA:
        scene, geometry, fingerprint = reader.scene(source)
        query = _scene_query(scene, fingerprint)
        return _verdict(reader, source, query, "observation",
                        "Instruction-independent scene facts with explicit unknowns; no interaction success",
                        checks={"scene_fact_publication_verified": True,
                                "unknown_objects_are_not_successes": True, "motion_evidence_present": False})
    if value.get("schema") in STAGE12_SCHEMAS:
        return _case(reader, source, value)
    if value.get("schema") == "p550.stage1_query_execution.v1":
        geometry, fingerprint = reader.geometry(value["source_fixed_geometry"], value["scene_id"])
        query = {"scene_id": value["scene_id"], "scene_fingerprint": fingerprint,
                 "scene_family": _family(value["scene_id"]), "instruction": value.get("instruction"),
                 "fixed_geometry_sha256": value["source_fixed_geometry"]["sha256"]}
        return _verdict(reader, source, query,
                        "failure" if value.get("status") == "failed_not_publishable" else "observation",
                        str(value.get("reason", "Stage1 evidence only; no motion")), domain="stage1_planning"
                        if value.get("status") == "failed_not_publishable" else None,
                        checks={"world_placement_or_motion_success_implied": False})
    query = {"scene_id": str(value.get("scene_id", "unknown")),
             "scene_fingerprint": digest({"unsupported_source_sha256": source["sha256"]})}
    return _verdict(reader, source, query, "unknown", "Unsupported source schema; no inferred success")


def _fixture(source):
    reader = _Reader()
    value = reader.read(source, schema=FIXTURE_SCHEMA)
    require(set(value) == {"schema", "namespace", "query_identity", "attempt_nonce", "checks",
                          "critical_memory_failure", "affected_record_ids", "receipt_payload_sha256"},
            "Unexpected fixture evidence fields")
    require(value["namespace"] == "test_fixture", "Missing fixture namespace")
    query = value["query_identity"]
    require(isinstance(query, dict) and set(query) ==
            {"scene_id", "scene_fingerprint", "scene_family", "instruction", "start_world_xy_m"},
            "Invalid fixture query")
    require(all(isinstance(query[k], str) and query[k] for k in
                ("scene_id", "scene_fingerprint", "scene_family", "instruction"))
            and HEX64.fullmatch(query["scene_fingerprint"]), "Invalid fixture identity")
    _xy(query["start_world_xy_m"])
    require(isinstance(value["attempt_nonce"], str) and value["attempt_nonce"], "Missing fixture attempt nonce")
    checks = value["checks"]
    require(isinstance(checks, dict) and set(checks) == set(GATES)
            and all(type(x) is bool for x in checks.values()), "Fixture needs exact nine boolean gates")
    passed = all(checks.values())
    critical = _boolean(value["critical_memory_failure"], "fixture critical flag")
    ids = value["affected_record_ids"]
    require(isinstance(ids, list) and len(set(ids)) == len(ids)
            and all(isinstance(x, str) and x for x in ids), "Invalid fixture affected IDs")
    require((not critical and not ids) or (critical and not passed and ids),
            "Critical attribution must be an explicit failed fixture")
    result = _verdict(reader, source, query, "success" if passed else "failure",
                      "Isolated mechanism fixture; never real positive credit",
                      domain=None if passed else "test_fixture", checks=checks)
    result.update(purpose="test_fixture", fixture_success_allowed=passed,
                  critical_memory_failure=critical, affected_record_ids=ids)
    return result


def admit_episode(request_path, *, purpose="production") -> dict:
    """Hard entry for storage. Caller-supplied success booleans are not inputs."""
    require(purpose in {"production", "test_fixture"}, "Unknown admission purpose")
    request = read_json(request_path)
    if "receipt_payload_sha256" in request:
        _seal(request)
    fields = set(request) - {"receipt_payload_sha256"}
    require(fields == {"schema", "mode", "source_receipt"}, "Unexpected admission request fields")
    require(request["schema"] == REQUEST_SCHEMA, "Unexpected admission request schema")
    source = _source_record(request["source_receipt"])
    path = verified(source)
    if request["mode"] == "test_fixture":
        require(purpose == "test_fixture", "Production forbids all fixture evidence")
        return _fixture(source)
    require(purpose == "production", "Real observations cannot enter the fixture success namespace")
    if request["mode"] == "observe_existing":
        result = classify_stage12(path)
        require(source in result["source_bindings"] and artifact(path) == source,
                "Source changed between request validation and classification")
        return result
    require(request["mode"] == "online_complete_motion", "Unsupported admission mode")
    # Deliberately no registry setter, producer callback or trust-caller switch.
    # Adding a real producer requires a reviewed new source version and tests.
    result = classify_stage12(path)
    require(source in result["source_bindings"] and artifact(path) == source,
            "Online source changed between request validation and classification")
    result.update(classification="unknown", failure_domain=None,
                  reason="No compatible complete online motion producer is registered in P555 v1",
                  real_positive_credit_allowed=False)
    result["checks"]["production_producer_registered"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--purpose", choices=("production", "test_fixture"), default="production")
    args = parser.parse_args(argv)
    print(json.dumps(admit_episode(args.request, purpose=args.purpose), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
