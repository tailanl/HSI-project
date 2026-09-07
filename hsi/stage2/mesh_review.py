"""Nine actual-Qwen checks of exact refined mesh and isolated-body evidence."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from contextlib import nullcontext
from . import compact_review as compact

CHECKS = ("target_correct", "action_complete", "one_body", "anatomically_plausible",
    "feet_grounded_not_tiptoe", "legs_naturally_forward", "articulation_semantics_preserved", "scene_layout_preserved",
    "visual_evidence_sufficient")

def decide(value):
    if set(value) != set(CHECKS) | {"confidence", "reason"}:
        raise ValueError("Unexpected refined-pose audit fields")
    if any(type(value[k]) is not bool for k in CHECKS):
        raise ValueError("Invalid refined-pose audit flags")
    score = value["confidence"]
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
        raise ValueError("Invalid refined-pose confidence")
    if not isinstance(value["reason"],str) or len(value["reason"])>240:
        raise ValueError("Invalid refined-pose explanation")
    failed=[k for k in CHECKS if not value[k]]
    if score<.85:
        failed.append("confidence_below_0.85")
    return {"post_refine_semantic_pass":not failed, "failed_checks":failed}

def run(source_path, bundle_path, manifest_path, isolated_path, output, *, qwen_client):
    stage2, bundle, visual = map(read_sealed, (source_path,bundle_path,manifest_path))
    verify_artifact_tree(bundle)
    verify_artifact_tree(visual)
    if bundle["stage3_handoff_allowed"] is not True or len(bundle["selected_gates"])!=18 \
            or any(v is not True for v in bundle["selected_gates"].values()):
        raise ValueError("Refined candidate needs all original eighteen physical checks")
    stage1 = read_sealed(verified(stage2["source_stage1"]))
    if visual["inputs"]["p533_complete_bundle"] != artifact(bundle_path) \
            or visual["inputs"]["h3_hybrikx_articulation_receipt"] != stage2["hybrikx_receipt"] \
            or visual["inputs"]["stage1_target"] != stage1["target"] \
            or visual["inputs"]["stage1_bundle"] != stage1["bundle"]:
        raise ValueError("Rendered pose belongs to another source, target or body prior")
    audit = read_sealed(verified(stage2["h3_semantic_audit"]))
    h3 = read_sealed(verified(stage2["h3_receipt"]))
    view = read_sealed(verified(stage2["source_view"]))
    if visual["schema"]!="p550.fast_original_camera_post_refine_evidence.v1" or len(visual["views"])!=1 \
            or visual["actual_body_or_scene_world_geometry_modified"] is not False:
        raise ValueError("Need the actual unchanged body rendered in the original scene camera")
    isolated = read_sealed(isolated_path)
    if isolated["source_refine_bundle"] != artifact(bundle_path) or isolated["source_keypose"] != visual["inputs"]["p533_published_keypose"] \
            or isolated["scene_occluders_hidden_explicitly"] is not True or isolated["actual_body_world_vertices_or_pose_modified"] is not False:
        raise ValueError("Isolated-body diagnostic does not bind to the actual unchanged refined mesh")
    if len(isolated["views"]) != 2:
        raise ValueError("Need two labeled actual-body diagnostic views")
    images = [verified(view["selection"]["selected_stage2_crop"]),
        verified(view["selection"]["selected_stage2_crop_target_overlay"]),
        verified(h3["artifacts"]["h3_image"]), verified(visual["views"][0]["image"]), *[verified(r["image"]) for r in isolated["views"]]]
    output.mkdir(parents=True,exist_ok=False)
    schema=compact.schema(CHECKS)
    prompt = ("Task: "+stage1["instruction"]+"\n" \
        "IMAGE_0 is the real empty scene crop. IMAGE_1 marks the exact furniture and seat. IMAGE_2 is the actual H3 two-dimensional pose prior. " \
        "IMAGE_3 is a NEW camera view of the actual refined uniformly colored human mesh in the full original scene. "
        "IMAGE_4 and IMAGE_5 show the SAME actual body with scene occluders explicitly HIDDEN, and a gray coordinate-reference floor. "
        "The last two are diagnostic views, NOT a changed original scene or different pose. Use IMAGE_3 for furniture/scene identity; "
        "use the unobstructed last two images to inspect the actual legs, ankles, feet and body anatomy. " \
        \
        "Audit the refined colored body, NOT the empty reference. The cameras differ deliberately; an orange or cyan untextured mesh is expected. " \
        "Check the exact target and completed sitting relation; exactly one body; plausible anatomy with no twisted, folded, fused or broken limbs; " \
        "both feet naturally on the floor, not tiptoe or strongly tucked backwards; legs naturally forward through the open seat side; " \
        "and preservation of the H3 action/posture meaning. The refinement may correct mild contact error but must not turn a normal sitting pose into a contorted pose. " \
        "Scene layout means the original furniture relationship, not identical viewpoint. Do not assume an unseen foot is correct. " \
        "Use the full-scene and unobstructed diagnostic evidence together; insufficient visual evidence must be false. "
        "Do not count deliberately hidden furniture in IMAGE_4/5 as a scene change. Ignore headers and colored wireframe annotations. " \
        \
        "Return the requested booleans, confidence and concise visual reason only. Do not output coordinates, angles or distances. " \
        "No physical test result or prior audit decision is shown.")
    prompt += compact.instruction(CHECKS)
    started=time.monotonic()
    with nullcontext():
        queue=time.monotonic()-started
        response=qwen_client.session(output / "qwen_calls",max_tokens=170).call(images,prompt,schema=schema,call_id="p550_post_refine_actual_mesh_semantics")
    raw=json.loads(response["raw_completion"])
    parsed=compact.normalize(raw,CHECKS)
    decision=decide(parsed)
    value={"schema":"p550.actual_refined_mesh_semantic_review.v5", "source":artifact(__file__),
        "source_stage2":artifact(source_path), "source_refine_bundle":artifact(bundle_path), "source_rendered_mesh_manifest":artifact(manifest_path),
        "isolated_body_diagnostic":artifact(isolated_path),
        "actual_input_images":[artifact(p) for p in images], "judgement":parsed,"decision":decision,"generation":response,
        "compact_raw_judgement":raw,"ordered_boolean_checks":list(CHECKS),
        "compact_transport_only_no_checks_or_evidence_removed":True,
        "original_h3_audit_failed_checks_retained":audit["decision"]["failed_checks"],
        "original_h3_audit_rewritten_or_claimed_passed":False,"all_original_eighteen_physical_gates_passed":True,
        "post_refine_semantic_and_physical_candidate_valid":decision["post_refine_semantic_pass"],
        "stage3_handoff_allowed":False,"actual_stage3_motion_generated":False,
        "explicit_recovery_publication_contract_still_required":True,
        "queue_seconds":queue,"elapsed_seconds":time.monotonic()-started}
    write_once(output / "receipt.json",value,seal=True)
    print({"decision":decision,"seconds":value["elapsed_seconds"]},flush=True)
