"""Independent actual-Qwen audit for the explicitly non-H3 geometric provider."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

from memory_common import artifact, verified, read_sealed, write_once, require
from fast_keypose import activate, load_current

CHECKS = ("target_correct", "action_complete", "one_body", "anatomically_plausible",
          "feet_grounded_not_tiptoe", "legs_naturally_forward", "requested_pose_semantics_satisfied",
          "scene_layout_preserved", "visual_evidence_sufficient")


def decide(judgement):
    require(set(judgement) == set(CHECKS) | {"confidence", "reason"}, "Audit fields differ")
    require(all(type(judgement[k]) is bool for k in CHECKS), "Audit flags must be booleans")
    score = judgement["confidence"]
    require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, "Invalid confidence")
    require(isinstance(judgement["reason"], str) and len(judgement["reason"]) <= 100, "Invalid explanation")
    failures = [k for k in CHECKS if not judgement[k]]
    if score < .85:
        failures.append("confidence_below_0.85")
    return dict(semantic_pass=not failures, failed_checks=failures)


def run(render_path, output, *, wait_for_slot=False):
    source_at_start = artifact(__file__)
    activate()
    from qwen_scheduling import LOCK_PATH
    import compact_audit_contract as compact
    from qwen_api import Qwen
    render = read_sealed(render_path)
    require(render["schema"] == "p555.actual_geometric_provider_render.v1", "Wrong renderer protocol")
    proposal_path = verified(render["source_proposal"])
    proposal = read_sealed(proposal_path)
    current = load_current(verified(proposal["inputs"]["stage1"]))
    require(render["source_keypose"] == proposal["outputs"]["candidate"], "Render is not the selected actual keypose")
    require(render["source_stage1"] == proposal["inputs"]["stage1"], "Render current Stage1 differs")
    verified(render["source_keypose"])
    verified(render["original_mesh"])
    verified(render["original_camera"])
    verified(render["source"])
    view = read_sealed(verified(render["source_view_selection"]))
    require(view["source_stage1_execution"] == proposal["inputs"]["stage1"]
            and view["target"] == current["stage1"]["target"]
            and view["selection"]["selected_camera"] == render["original_camera"]
            and view["selection"]["selected_stage2_crop"] == render["original_reference"]
            and view["selection"]["selected_stage2_crop_target_overlay"] == render["target_overlay"],
            "Actual camera/reference/target-overlay lineage differs")
    require(render["original_scene_rendered"] is True and render["original_scene_vertices_textures_or_camera_changed"] is False
            and render["actual_body_vertices_changed"] is False and render["isolated_views_have_occluders_hidden"] is True,
            "Missing unchanged original-scene/body evidence")
    require(len(render["isolated_body_views"]) == 2, "Need two actual-body diagnostic views")
    images = [verified(render["original_reference"]), verified(render["target_overlay"]),
              verified(render["images"]["original_scene_with_actual_body_crop"]),
              *[verified(row["image"]) for row in render["isolated_body_views"]]]
    extra = render.get("additional_original_scene_views", [])
    if extra:
        require(len(extra) == 2 and all(row["scene_occluders_hidden"] is False for row in extra), "Malformed additional original views")
        index = max(range(len(extra)), key=lambda i: (extra[i]["actual_body_visibility_fraction"], -i))
        require(index == render["selected_additional_view_index"], "Additional view not selected by the numeric visibility rule")
        images.append(verified(extra[index]["image"]))
    prompt = (
        "Task: " + current["stage1"]["instruction"] + "\n"
        "IMAGE_0 is the actual empty scene crop. IMAGE_1 marks the exact target furniture and interaction surface. "
        "IMAGE_2 shows the ACTUAL proposed orange untextured human mesh rendered in the original scene and camera. "
        "IMAGE_3 and IMAGE_4 show the SAME ACTUAL body with scene occluders deliberately hidden and a gray floor coordinate reference. "
        "Use IMAGE_2 for the target and scene relationship; use the two diagnostic images for body anatomy and feet. "
        "These are geometry-based pose proposals, not an H3 image. There is no H3 articulation to preserve. "
        "Check: the exact target; completed sitting with pelvis supported by the correct seat; exactly one body; plausible, "
        "untwisted and unbroken anatomy; both feet naturally grounded, not tiptoe; legs forward through the open seat side; "
        "a natural seated posture satisfying the task; preserved original furniture relationships; sufficient visible evidence. "
        "Untextured orange body and gray scene are expected; do not reject them for lacking clothing or texture. "
        "Do not count deliberately hidden furniture in diagnostic images as a layout change. Never assume an unseen foot is correct. "
        "No physical test result or previous judgement is shown. Inspect the actual images independently. "
        "Return only the requested semantic booleans, confidence and brief visual reason, never coordinates or angles."
    )
    if extra:
        prompt += (" IMAGE_5 is an additional ACTUAL side view with the ENTIRE ORIGINAL SCENE and all occluders retained. "
                   "Its camera was placed numerically relative to the current target; neither the body nor the scene was moved. "
                   "Use it to disambiguate whether the pelvis sits on the seat and whether the knees are bent; "
                   "do not infer standing solely from an ambiguous frontal projection. Inspect all provided evidence.")
    prompt += compact.instruction(CHECKS)
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite audit")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    before_reservation = time.monotonic()
    # Default: no queue; an occupied cooperating service returns immediately.
    with LOCK_PATH.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if wait_for_slot else fcntl.LOCK_NB))
        except BlockingIOError:
            print({"status": "qwen_busy_no_queue", "inference_started": False}, flush=True)
            return None
        try:
            reserved = time.monotonic()
            client = Qwen(output / "qwen_calls", max_tokens=170)
            response = client.call(images, prompt, schema=compact.schema(CHECKS), call_id="p555_actual_geometry_keypose_semantics")
            raw = json.loads(response["raw_completion"])
            judgement = compact.normalize(raw, CHECKS)
            decision = decide(judgement)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    frozen = output / "audit_geometry_keypose_source.py"
    verified(source_at_start)
    shutil.copy2(__file__, frozen)
    value = dict(schema="p555.actual_geometry_keypose_qwen_review.v1", source=artifact(frozen),
        source_proposal=artifact(proposal_path), source_render=artifact(render_path),
        actual_images=[artifact(path) for path in images], ordered_checks=list(CHECKS),
        generation=response, raw_judgement=raw, judgement=judgement, decision=decision,
        source_qwen_call=artifact(response["call_receipt"]),
        numerical_candidate_pass=all(proposal["gates"].values()),
        numeric_and_semantic_candidate_pass=all(proposal["gates"].values()) and decision["semantic_pass"],
        qwen_received_previous_audits_or_numerical_gate_results=False, qwen_computed_coordinates=False,
        provider="current_geometry_ik_not_H3", h3_background_gate_not_applicable_to_direct_mesh_provider=True,
        original_H3_branch_gates_modified=False, stage3_handoff_allowed=False, real_positive_credit=0,
        reservation_seconds_before_audit_timer=reserved-before_reservation,
        audit_elapsed_seconds_excluding_reservation=time.monotonic()-reserved)
    write_once(output / "receipt.json", value)
    print({"scene": proposal["scene_id"], "decision": decision,
           "numeric_and_semantic_candidate_pass": value["numeric_and_semantic_candidate_pass"],
           "seconds": value["audit_elapsed_seconds_excluding_reservation"]}, flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-slot", action="store_true")
    args = parser.parse_args()
    run(args.render, args.output, wait_for_slot=args.wait_for_slot)
