"""Actual H3 image review and strictly visibility-only deferred recovery."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import compact_review as compact

CHECKS = ("target_correct", "action_complete", "one_person", "body_complete", "limbs_plausible", "feet_visible", "scene_preserved")

SCHEMA = {"type": "object", "properties": {**{name: {"type": "boolean"} for name in CHECKS},
    "foot_contact_needs_refine": {"type": "boolean"}, "needs_more_evidence": {"type": "boolean"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "reason": {"type": "string", "maxLength": 140}},
    "required": [*CHECKS, "foot_contact_needs_refine", "needs_more_evidence", "confidence", "reason"], "additionalProperties": False}

TRANSPORT_CHECKS=(*CHECKS,"foot_contact_needs_refine","needs_more_evidence")

def verdict(value):
    if not isinstance(value, dict) or set(value) != set(SCHEMA["required"]):
        raise ValueError("Wrong image audit fields")
    for key in (*CHECKS, "foot_contact_needs_refine", "needs_more_evidence"):
        if type(value[key]) is not bool: raise ValueError("Non-boolean audit field: " + key)
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    if not isinstance(value["reason"], str) or len(value["reason"]) > 140:
        raise ValueError("Invalid reason")
    failures = [key for key in CHECKS if value[key] is not True]
    if value["needs_more_evidence"]: failures.append("needs_more_evidence")
    if confidence < .82: failures.append("confidence_below_0.82")
    return {"semantic_prior_eligible_for_refine": not failures, "failed_checks": failures,
        "physical_keypose_publishable": False, "foot_contact_needs_refine": value["foot_contact_needs_refine"]}

def audit(reference, marked_reference, generated, instruction, target_id, output, source_kind, *, qwen_client):
    started = time.monotonic()
    if output.exists(): raise FileExistsError(output)
    qwen = qwen_client.session(output.parent / "qwen_calls", max_tokens=150)
    prompt = f"Task: {instruction}. Target ID: {target_id}. " \
        "IMAGE_0 is the unchanged scene; IMAGE_1 highlights the exact target; IMAGE_2 is the single generated keypose. " \
        "Audit IMAGE_2 only for the human. The empty reference images intentionally contain no person. " \
        "target_correct: person uses the exact highlighted furniture, not a replacement/distractor. " \
        "action_complete: the requested body-object relationship is complete. " \
        "one_person: exactly one opaque adult. body_complete: head, torso, arms, hands, legs and feet are visible, no cropping or furniture occlusion. " \
        "limbs_plausible: no severe extra/missing/fused/broken limbs. feet_visible: both feet are separately visible, not hidden or cropped. " \
        "scene_preserved: no moved/replaced/warped furniture or changed camera; ignore the reference highlight. " \
        "foot_contact_needs_refine: mark visible floating/penetrating feet true, but do NOT use mild contact error alone to fail the semantic checks. " \
        "Actual ground/seat contact and collision will be measured by the downstream 3D geometry verifier. " \
        "Any identity/body ambiguity sets needs_more_evidence=true. Give joint confidence and one brief reason, at most 18 words. Return only the requested JSON."
    prompt += compact.instruction(TRANSPORT_CHECKS)
    result = qwen.call([reference, marked_reference, generated], prompt, schema=compact.schema(TRANSPORT_CHECKS), call_id="compact_single_keypose_image_audit_v1")
    raw = json.loads(result["raw_completion"])
    parsed = compact.normalize(raw,TRANSPORT_CHECKS)
    decision = verdict(parsed)
    record = {"schema": "p550.compact_single_keypose_image_audit.v1", "source_kind": source_kind,
        "status": "semantic_prior_ready_for_geometric_refine" if decision["semantic_prior_eligible_for_refine"] else "rejected_semantic_prior",
        "instruction": instruction, "target_instance_id": target_id,
        "inputs": {"reference": artifact(reference), "marked_reference": artifact(marked_reference), "generated_keypose": artifact(generated)},
        "model": qwen.lineage, "runtime": qwen.runtime, "judgement": parsed, "decision": decision,
        "compact_raw_judgement":raw,"ordered_boolean_checks":list(TRANSPORT_CHECKS),
        "compact_transport_only_no_checks_or_evidence_removed":True,
        "generation": result, "qwen_call_count": 1, "video_temporal_audit_performed": False,
        "physical_contact_gate_owner": "unchanged_full_mesh_geometric_refine_publication_gates",
        "protocol_changed_from_p548": ["one_single_keypose_audit_not_two_video_orders", "one_joint_confidence",
            "no_empty_first_frame_or_temporal_check", "visible_mild_contact_error_may_enter_refine_but_never_implies_publication"],
        "elapsed_seconds": time.monotonic()-started, "source": artifact(__file__)}
    record["receipt_payload_sha256"] = digest(record)
    write_once(output, record)
    print({"status": record["status"], "seconds": record["elapsed_seconds"], "failed_checks": decision["failed_checks"]}, flush=True)
    return record

HARD_SEMANTICS = ("target_correct", "action_complete", "one_person", "limbs_plausible", "scene_preserved")

VISIBILITY = {"body_complete", "feet_visible"}

def partial_prior_eligible(audit):
    value = audit["judgement"]
    if any(type(value[k]) is not bool for k in (*HARD_SEMANTICS, *VISIBILITY, "needs_more_evidence")):
        raise ValueError("Invalid semantic boolean")
    failed = set(audit["decision"]["failed_checks"])
    expected = {k for k in (*HARD_SEMANTICS, *VISIBILITY) if not value[k]}
    score = value["confidence"]
    if isinstance(score, bool) or not isinstance(score, (int,float)) or not math.isfinite(score):
        raise ValueError("Invalid semantic confidence")
    return bool(failed and failed <= VISIBILITY and failed == expected
        and all(value[k] for k in HARD_SEMANTICS) and value["needs_more_evidence"] is False
        and .85 <= score <= 1)
