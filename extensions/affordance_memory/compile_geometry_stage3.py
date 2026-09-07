"""P555 single sit adapter: current route + audited geometric IK, not H3.

Creates an experimental Stage3 entry contract after independently recomputing
the existing numerical gates. Original proposal/review handoff flags remain
false and untouched. Publication still requires an actually generated motion
and the original full-mesh/temporal/contact/support/no-skate release gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from memory_common import PROJECT, artifact, digest, read_json, read_sealed, require, verified, write_once
import neutral_seed as neutral

SCHEMA = "p555.geometry_stage3_adapter.v1"
PACKET_SCHEMA = "p555.current_geometry_stage3_packet.v1"
PROVENANCE = "p555_current_geometry_ik"
PRODUCER_HASHES = {"d029b04dfef12ddb6284969d059c29583f2274c22d64ae0556457ec0b32f4dab"}
QWEN_CHECKS = ("target_correct", "action_complete", "one_body", "anatomically_plausible",
              "feet_grounded_not_tiptoe", "legs_naturally_forward", "requested_pose_semantics_satisfied",
              "scene_layout_preserved", "visual_evidence_sufficient")
NUMERIC_GATES = ("selected_surface_hash_bound", "contact_anchor_locked", "target_forbidden_body_penetration",
                 "target_allowed_contact_penetration", "target_allowed_contact_area", "non_target_scene_penetration",
                 "ground_penetration", "sdf_grid_coverage", "left_foot_ground_contact_area", "right_foot_ground_contact_area",
                 "hips_support_inside_surface", "terminal_facing", "current_stage1_target_mask_hash_bound",
                 "direct_target_relation_valid", "known_action_family_sit", "contact_point_inside_bound_surface",
                 "contact_point_on_true_support_mask", "contact_anchor_clearance_bounded", "hips_support_inside_actual_triangles",
                 "ankles_near_floor", "ankles_forward_of_hips", "feet_face_open_seat_side", "shins_not_strongly_folded_back")
THRESHOLDS = {"maximum_ground_penetration_m": .02, "maximum_non_target_scene_penetration_m": .02,
              "maximum_sdf_outside_fraction": .01, "maximum_target_allowed_contact_penetration_m": .03,
              "maximum_target_forbidden_body_penetration_m": .02, "maximum_terminal_facing_error_rad": math.pi/36,
              "minimum_hips_support_inside_surface_fraction": .8, "minimum_left_foot_ground_vertices": 6,
              "minimum_right_foot_ground_vertices": 6, "minimum_target_allowed_contact_vertices": 16}
# Pre-registered, not tuned after motion generation. This admits a local sit
# transition, not a 1.1m walk-to-centre hidden behind the old standing route.
TRANSITION_POLICY = {"schema": "p555.local_sit_transition_admission.v1", "maximum_root_xy_distance_m": .65,
                     "maximum_distance_leg_length_ratio": .8, "distance_measurement_slack_m": .02,
                     "maximum_lateral_m": .35, "maximum_lateral_hip_width_ratio": 2.,
                     "minimum_front_projection_m": .05, "sample_spacing_m": .01,
                     "non_target_clearance_shoulder_halfwidth_extra_m": .02,
                     "maximum_non_target_proxy_radius_m": .28,
                     "geometry_connection_is_not_a_generated_motion": True, "original_route_modified": False}


def _record(value):
    return {key: value[key] for key in ("path", "bytes", "sha256")}


def _checked(record, *, sealed=True):
    path = verified(record)
    require(path.stat().st_size <= 16*1024*1024, "Oversized JSON input")
    return read_sealed(path) if sealed else read_json(path)


def _all_true(values, names, label):
    require(isinstance(values, dict) and set(values) == set(names)
            and all(values[key] is True for key in names), label+" not all passed or gate set changed")


def _payload(value):
    return {key: item for key, item in value.items() if key != "receipt_payload_sha256"}


def validate_proposal(proposal):
    require(proposal.get("schema") == "p555.geometry_guided_keypose_proposal.v1"
            and proposal.get("provider") == "declared_analytic_sit_seed_plus_current_geometry_ik"
            and proposal.get("status") == "numeric_candidate_pass", "Unsupported or failed geometric producer")
    require(proposal["source"]["sha256"] in PRODUCER_HASHES, "Unregistered geometric producer revision")
    verified(proposal["source"])
    require(proposal.get("stage3_handoff_allowed") is False and proposal.get("semantic_publication_allowed") is False
            and proposal.get("h3_model_used") is False and proposal.get("hybrikx_model_used") is False
            and proposal.get("historical_full_pose_or_motion_read") is False, "Old proposal claim/provenance changed")
    _all_true(proposal["gates"], NUMERIC_GATES, "Recorded 23 numerical gates")
    for record in proposal["inputs"].values():
        verified(record)


def validate_semantics(proposal_record, proposal, review_record):
    review = _checked(review_record)
    require(review.get("schema") == "p555.actual_geometry_keypose_qwen_review.v1"
            and review.get("source_proposal") == proposal_record, "Qwen review belongs to another candidate")
    require(review.get("ordered_checks") == list(QWEN_CHECKS), "Qwen check order drift")
    judgement = review["judgement"]
    require(set(judgement) == set(QWEN_CHECKS)|{"confidence", "reason"}, "Unregistered Qwen judgement")
    _all_true({key: judgement[key] for key in QWEN_CHECKS}, QWEN_CHECKS, "Qwen semantics")
    confidence = judgement["confidence"]
    require(type(confidence) in (int, float) and math.isfinite(confidence) and .85 <= confidence <= 1,
            "Qwen confidence below original .85 gate")
    require(review.get("decision") == {"semantic_pass": True, "failed_checks": []}
            and review.get("numeric_and_semantic_candidate_pass") is True
            and review.get("numerical_candidate_pass") is True, "Review decision contradicts actual checks")
    require(review.get("stage3_handoff_allowed") is False and review.get("real_positive_credit") == 0
            and review.get("qwen_computed_coordinates") is False
            and review.get("qwen_received_previous_audits_or_numerical_gate_results") is False,
            "Review already claims unauthorized handoff or non-independent evidence")
    render = _checked(review["source_render"])
    require(render.get("schema") == "p555.actual_geometric_provider_render.v1"
            and render.get("source_proposal") == proposal_record
            and render.get("source_keypose") == proposal["outputs"]["candidate"]
            and render.get("source_stage1") == proposal["inputs"]["stage1"], "Render/source candidate mismatch")
    require(render.get("original_scene_rendered") is True
            and render.get("original_scene_vertices_textures_or_camera_changed") is False
            and render.get("actual_body_vertices_changed") is False
            and render.get("isolated_views_have_occluders_hidden") is True, "Render geometry or camera changed")
    require(len(render["isolated_body_views"]) == 2, "Unregistered diagnostic-view count")
    images = [render["original_reference"], render["target_overlay"], render["images"]["original_scene_with_actual_body_crop"],
              *[row["image"] for row in render["isolated_body_views"]]]
    require(review["actual_images"] == images, "Reviewed images differ from actual selected render")
    for image in images:
        verified(image)
    view = _checked(render["source_view_selection"])
    stage1 = _checked(proposal["inputs"]["stage1"])
    require(view["source_stage1_execution"] == proposal["inputs"]["stage1"] and view["target"] == stage1["target"],
            "Render camera selection belongs to another query/target")
    selection = view["selection"]
    require(render["original_camera"] == selection["selected_camera"]
            and render["image_crop_xyxy"] == selection["selected_crop_xyxy"]
            and render["original_reference"] == selection["selected_stage2_crop"]
            and render["target_overlay"] == selection["selected_stage2_crop_target_overlay"], "Camera/crop binding drift")
    verified(render["original_camera"])
    call = _checked(review["source_qwen_call"], sealed=False)
    require(call.get("schema") == "p548.qwen_local_semantic_call.v1"
            and call.get("call_id") == "p555_actual_geometry_keypose_semantics", "Not an actual registered Qwen call")
    require([_record(row) for row in call["image_evidence"]] == images
            and [row["label"] for row in call["image_evidence"]] == ["IMAGE_"+str(i) for i in range(5)],
            "Qwen call image sequence differs")
    generation = review["generation"]
    require(call["result"] == generation and generation["call_receipt"] == review["source_qwen_call"]["path"],
            "Qwen generation is not its actual call result")
    require(generation["model_id"] == "Qwen/Qwen3.8-27B-FP8" and generation["finish_reason"] == "stop"
            and generation["revision"] == call["model"]["revision"], "Qwen model/revision or completion mismatch")
    raw = json.loads(generation["raw_completion"])
    require(raw == review["raw_judgement"] and set(raw) == {"checks", "confidence", "reason"}
            and raw["checks"] == [judgement[key] for key in QWEN_CHECKS]
            and raw["confidence"] == confidence and raw["reason"] == judgement["reason"], "Qwen raw completion contradicts judgement")
    response = call["response"]
    require(response["choices"][0]["message"]["content"] == generation["raw_completion"]
            and response["choices"][0]["finish_reason"] == "stop" and response["id"] == generation["response_id"],
            "Qwen raw provider response mismatch")
    require(hashlib.sha256(call["prompt"].encode()).hexdigest() == generation["prompt_sha256"]
            and hashlib.sha256(call["system_prompt"].encode()).hexdigest() == generation["system_prompt_sha256"],
            "Qwen prompt hash mismatch")
    for row in (review["source"], render["source"], render["source_scene_loader"],
                call["model"]["config"], call["model"]["weight_index"]):
        verified(row)
    return review, render


def load_candidate(record):
    with np.load(verified(record), allow_pickle=False) as data:
        require(str(data["schema"]) == "p555.current_geometry_ik_candidate.v1", "Not the new geometric candidate schema")
        require(data["h3_model_used"].item() is False and data["hybrikx_model_used"].item() is False
                and data["stage3_handoff_allowed"].item() is False, "Candidate mislabels provider/handoff")
        arrays = {key: data[key].copy() for key in ("body_pose_axis_angle", "betas", "vertices_world_zup",
                  "joints_world_zup", "root_xyz_yaw", "pose_frame_to_world_rotation", "contact_world_xyz_m", "faces", "gluteal_support_vertex_ids")}
    for key, shape in (("body_pose_axis_angle", (21, 3)), ("betas", (10,)), ("vertices_world_zup", (10475, 3)),
                       ("joints_world_zup", (22, 3)), ("root_xyz_yaw", (4,)), ("pose_frame_to_world_rotation", (3, 3)),
                       ("contact_world_xyz_m", (3,)), ("faces", (20908, 3))):
        require(arrays[key].shape == shape and np.isfinite(arrays[key]).all(), "Invalid candidate "+key)
    require(np.array_equal(arrays["betas"], np.zeros(10)), "Candidate is not the registered zero-shape carrier")
    frame = arrays["pose_frame_to_world_rotation"]
    require(np.allclose(frame.T@frame, np.eye(3), atol=2e-5, rtol=0) and abs(np.linalg.det(frame)-1) < 2e-5,
            "Candidate pose frame is not SO(3)")
    require(arrays["faces"].dtype.kind in "iu" and arrays["faces"].min() >= 0 and arrays["faces"].max() < 10475,
            "Candidate body topology invalid")
    ids = arrays["gluteal_support_vertex_ids"]
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(ids) >= 8 and len(np.unique(ids)) == len(ids)
            and ids.min() >= 0 and ids.max() < 10475, "Invalid actual gluteal subset")
    return arrays


def numerical_recheck(proposal, arrays, current):
    """Re-materialize this pose and repeat original branches + all 23 gates."""
    import torch
    import fast_keypose as fast
    engine = fast.FastKeyposeEngine(Path(proposal["inputs"]["body_model"]["path"]), Path(proposal["inputs"]["segmentation"]["path"]))
    body, contract = engine.body, engine.contract
    require(proposal["thresholds"] == THRESHOLDS == fast.THRESHOLDS, "Numerical thresholds changed")
    config = body.target.RefineConfig(steps=1, bilateral_foot_support=True, target_allowed_tail_m=.03, ground_tail_m=.02,
        minimum_allowed_contact_vertices=16, minimum_foot_ground_vertices=6,
        maximum_target_forbidden_penetration_m=.02, maximum_non_target_penetration_m=.02, maximum_outside_fraction=.01).validate()
    cache = engine.cache.BoundSDFCache(verified(proposal["inputs"]["sdf_cache"]), engine.sdf)
    require(cache.receipt["source_binding"]["stage1_execution"] == proposal["inputs"]["stage1"]
            and cache.receipt["source_binding"]["target"] == current["stage1"]["target"], "SDF belongs to another current target")
    target, surface, yaw = current["target"], current["surface"], current["binding"]["contact_forward_yaw_rad"]
    fields = cache(current["occupancy"], current["target_mask"], torch.device("cpu"),
        expected_occupancy_file_sha256=artifact(current["occupancy"])["sha256"],
        expected_target_mask_file_sha256=artifact(current["target_mask"])["sha256"],
        expected_target_world_sha256=target["target_occupancy_mask_world_xyz_sha256"])
    pose, betas, frame, contact = [torch.as_tensor(arrays[key], dtype=torch.float32) for key in
                                 ("body_pose_axis_angle", "betas", "pose_frame_to_world_rotation", "contact_world_xyz_m")]
    ids = torch.from_numpy(arrays["gluteal_support_vertex_ids"])
    with torch.no_grad():
        state = body.materialize(engine.model, pose, betas, frame, contact, ids)
        errors = {"vertices_max_abs_m": float(np.max(np.abs(state.vertices.numpy()-arrays["vertices_world_zup"]))),
                  "j22_max_abs_m": float(np.max(np.abs(state.joints[:22].numpy()-arrays["joints_world_zup"]))),
                  "root_max_abs_m": float(np.max(np.abs(state.root.numpy()-arrays["root_xyz_yaw"][:3])))}
        require(max(errors.values()) <= 2e-5 and np.array_equal(engine.model.faces, arrays["faces"]), "Candidate differs from exact same-body materialization")
        anatomy = body.target.build_anatomy_masks(engine.model, 10475, config)
        branches = body.target.branch_terms(state.vertices, fields, anatomy, config)
        metrics = {key: float(value.detach()) if isinstance(value, torch.Tensor) else int(value) for key, value in branches.diagnostics.items()}
        support = state.vertices[ids]
        n = max(8, math.ceil(len(ids)*body.SUPPORT_ENVELOPE_FRACTION))
        anchor = torch.cat((support[:, :2].mean(0), torch.topk(support[:, 2], n, largest=False).values.mean().reshape(1)))
        error = float((anchor-contact).norm())
        bounds = np.asarray(surface["surface"]["bounds_world_zup_m"])
        metrics["hips_support_inside_surface_fraction"] = body.support_inside_surface_fraction(support.numpy(), bounds)
        axis = state.joints[2, :2]-state.joints[1, :2]
        facing_error = body.angular_error(math.atan2(float(axis[0]), float(-axis[1])), yaw)
        metrics["pelvis_frame_facing_error_rad"] = facing_error
        gates = contract.publication_gates(metrics, THRESHOLDS, facing_error_rad=facing_error, selected_surface_match=True, contact_lock_error_m=error)
        _, mask = fast.triangle_heights(support.numpy()[:, :2], current["triangles"])
        heights, contact_valid = fast.triangle_heights(contact.numpy()[None, :2], current["triangles"])
        metrics["hips_support_inside_actual_triangles_fraction"] = float(mask.mean())
        gates.update(current_stage1_target_mask_hash_bound=fields.component_receipt["target_component_sha256"] == target["target_occupancy_mask_world_xyz_sha256"],
            direct_target_relation_valid=surface.get("reference_relation") in (None, "none", "beside", "associated_with_reference_setup"),
            known_action_family_sit=target["action_family"] == "sit",
            contact_point_inside_bound_surface=bool(np.all(contact.numpy() >= bounds[0]-1e-6) and np.all(contact.numpy() <= bounds[1]+1e-6)),
            contact_point_on_true_support_mask=bool(contact_valid[0]),
            contact_anchor_clearance_bounded=bool(0 <= float(contact[2])-heights[0]+1e-6 <= .04+1e-6),
            hips_support_inside_actual_triangles=bool(mask.mean() >= .8))
        posture = engine.quality.metrics(state.joints[:22].numpy(), yaw)
        gates.update(posture["neutral_sit_quality_gates"])
        metrics["neutral_sit_quality"] = posture
    _all_true(gates, NUMERIC_GATES, "Fresh 23 numerical gates")
    source_files = [body.__file__, body.body_util.__file__, body.sparse.__file__, body.target.__file__,
                    contract.__file__, engine.sdf.__file__, engine.cache.__file__, engine.quality.__file__]
    proof = {"schema": "p555.exact_body_candidate_recheck.v1", "gates": gates, "metrics": metrics,
             "input_bindings": {"candidate": proposal["outputs"]["candidate"], **proposal["inputs"]},
             "materialization_errors": errors, "thresholds": THRESHOLDS,
             "numerical_sources": [artifact(path) for path in source_files], "body_asset_sha256": neutral.NEUTRAL_SHA256,
             "fullmesh_vertex_count": 10475, "runtime_motion_not_yet_generated": True}
    return proof, fields, anatomy


def route_contact_check(route_end, standing_root_z, arrays, non_target_sample):
    """Bounded body-scale transition + original-scene non-target clearance.

    Samples are only a conservative admission diagnostic, never output motion,
    path extension or evidence that all future mesh vertices are collision-free.
    """
    import torch
    root, joints = arrays["root_xyz_yaw"], arrays["joints_world_zup"]
    delta = np.asarray(route_end)-root[:2]
    forward = np.array([math.cos(root[3]), math.sin(root[3])])
    leg = np.mean([np.linalg.norm(joints[1]-joints[4])+np.linalg.norm(joints[4]-joints[7]),
                   np.linalg.norm(joints[2]-joints[5])+np.linalg.norm(joints[5]-joints[8])])
    hips = np.linalg.norm(joints[1]-joints[2])
    shoulders = np.linalg.norm(joints[16]-joints[17])
    require(.5 <= leg <= 1.1 and .10 <= hips <= .40 and .20 <= shoulders <= .65, "Unsupported body scale for local sit transition")
    distance = float(np.linalg.norm(delta))
    max_distance = min(.65, .8*float(leg)+.02)
    lateral = abs(float(delta[0]*forward[1]-delta[1]*forward[0]))
    checks = {"local_body_scale_distance": distance <= max_distance,
              "standing_endpoint_on_open_front_side": float(delta@forward) >= .05,
              "bounded_lateral_entry": lateral <= min(.35, 2*float(hips))}
    require(all(checks.values()), "numericroute-replan_required: standing route endpoint is not a bounded local sit entry; "+str(checks))
    start = np.r_[route_end, standing_root_z]
    count = math.ceil(float(np.linalg.norm(start-root[:3]))/.01)+1
    points = np.linspace(start, root[:3], count)
    # A root-centred body-width sphere along the inferred transition corridor.
    # Target furniture remains handled by original target/fullmesh motion gates.
    radius = min(.28, float(shoulders)/2+.02)
    values, outside = non_target_sample(torch.tensor(points, dtype=torch.float32))
    values, outside = values.detach().cpu().numpy(), outside.detach().cpu().numpy()
    require(values.shape == outside.shape == (count,) and np.isfinite(values).all(), "Invalid current non-target connection samples")
    checks["non_target_connection_in_bounds"] = not bool(outside.any())
    checks["non_target_body_width_clearance"] = bool((values >= radius+.005).all())
    require(all(checks.values()), "numericroute-replan_required: current non-target connection is obstructed; "+str(checks))
    return {"policy": TRANSITION_POLICY, "checks": checks, "route_end_world_xy_m": list(map(float, route_end)),
            "seated_root_world_xy_m": root[:2].tolist(), "root_xy_distance_m": distance,
            "body_leg_length_m": float(leg), "body_hip_width_m": float(hips), "body_shoulder_width_m": float(shoulders),
            "effective_maximum_distance_m": max_distance, "front_projection_m": float(delta@forward),
            "lateral_distance_m": lateral, "sample_count": count, "body_width_proxy_radius_m": radius,
            "minimum_non_target_sdf_m": float(values.min()), "original_route_modified": False,
            "sitting_transition_still_requires_fresh_diffusion_and_fullmesh_validation": True}


def build_packet(*, scene_id, instruction, roots, arrays, source_artifacts, target_contract):
    nodes = []
    for index, root in enumerate(roots):
        nodes.append({"ordinal": index, "semantic_id": 1, "semantic_type": "pass" if index+1 < len(roots) else "approach",
                      "phase": "NAVIGATION" if index+1 < len(roots) else "APPROACH", "root_xyz_yaw": np.asarray(root).tolist(),
                      "joints": [], "contact_joint_ids": [], "confidence": 1., "source": "current_P552_verified_route",
                      "description": "Follow the current verified route to the open-side sitting approach."})
    terminal = arrays["root_xyz_yaw"]
    reorient = np.array(roots[-1], copy=True)
    reorient[3] = terminal[3]
    turn = abs(math.atan2(math.sin(terminal[3]-roots[-1][3]), math.cos(terminal[3]-roots[-1][3])))
    nodes.append({"ordinal": len(nodes), "semantic_id": 2, "semantic_type": "prepare", "phase": "REORIENT",
                  "root_xyz_yaw": reorient.tolist(), "joints": [], "contact_joint_ids": [], "confidence": 1.,
                  "orientation_only": True, "explicit_reorientation_required": turn > 1e-6,
                  "description": "At the route endpoint, face the selected furniture's audited open side."})
    nodes.append({"ordinal": len(nodes), "semantic_id": 3, "semantic_type": "contact", "phase": "CONTACT",
                  "root_xyz_yaw": terminal.tolist(), "joints": [{"id": i, "xyz_world": point.tolist(),
                      "confidence": .98 if i in (0, 1, 2) else .72, "contact_role": i in (0, 1, 2),
                      "constraint_role": "hard_contact_proxy" if i in (0, 1, 2) else "adaptive_full_pose",
                      "mutable_in_stage3": i not in (0, 1, 2)} for i, point in enumerate(arrays["joints_world_zup"])],
                  "contact_joint_ids": [0, 1, 2], "confidence": 1., "description": instruction,
                  "source": "p555_audited_current_geometry_ik_not_H3",
                  "full_smplx_keypose": {"schema": "p555.geometric_full_smplx_terminal_keypose.v1", "coordinate_system": "world_zup",
                      "root_xyz_yaw": terminal.tolist(), "joints_world_zup": arrays["joints_world_zup"].tolist(),
                      "body_pose_axis_angle": arrays["body_pose_axis_angle"].tolist(), "betas": arrays["betas"].tolist(),
                      "pose_frame_to_world_rotation": arrays["pose_frame_to_world_rotation"].tolist(),
                      "source_stage2_keypose_sha256": source_artifacts["candidate"]["sha256"]}})
    require(len(roots) >= 2, "Need current route start and endpoint")
    return {"schema": PACKET_SCHEMA, "coordinate_system": "world_zup", "seq_name": "p555_"+scene_id+"_current_sit",
            "scene_name": scene_id, "text": instruction, "provenance": PROVENANCE, "nodes": nodes,
            "source_artifacts": source_artifacts, "surface_contract": target_contract,
            "executor_contract": {"route_nodes_are_root_only": True, "condition_window_capacity": 8,
                "condition_window_is_rolling": True, "terminal_full22_keypose_count": 1, "orientation_only_node_count": 1,
                "navigation_control_node_count": len(roots), "total_plan_node_count": len(nodes)},
            "h3_model_used": False, "hybrikx_model_used": False, "legacy_P530_lineage_claimed": False,
            "motion_release_authorized": False}


def _guidance_products(output, *, current, arrays, anatomy, source_inputs, query_id, body_assets):
    import stage3_affordance_guidance as guidance
    yaw = current["binding"]["contact_forward_yaw_rad"]
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.]])
    origin = np.asarray(current["surface"]["surface"]["centre_world_xyz_zup_m"])
    patch = guidance.SurfacePatch(origin_world_zup_m=origin, local_to_world_rotation=rotation,
                                  triangles_local_m=(current["triangles"]-origin)@rotation)
    body_path = output / "body_identity.json"
    body_identity = {"schema": "p555.current_body_identity.v1", "model_type": "smplx", "gender": "neutral",
                     "joint_order": list(guidance.J22_NAMES), "body_model_sha256": neutral.NEUTRAL_SHA256,
                     "body_assets": body_assets, "betas_sha256": neutral.array_hash(arrays["betas"]),
                     "coordinate_frame": "world_zup", "units": "metres", "source_candidate": source_inputs["candidate"]}
    write_once(body_path, body_identity)
    inputs = {"stage1_execution": source_inputs["stage1_execution"], "fixed_geometry": current["stage1"]["source_fixed_geometry"],
              "contact_surface": current["option"]["p552_region_member"]["region_arrays"],
              "raw_occupancy": source_inputs["raw_occupancy"], "body_identity": artifact(body_path), "geometry_producer": artifact(__file__)}
    path = output / "guidance_input.json"
    write_once(path, {"schema": guidance.INPUT_SCHEMA, "scene_id": current["stage1"]["scene_id"], "query_id": query_id,
                      "coordinate_frame": "world_zup", "units": "metres", "source_bindings": inputs, "surface_patch": patch.payload()})
    guidance.GuidanceBinding(path)
    vertices, joints = arrays["vertices_world_zup"], arrays["joints_world_zup"]
    butt = arrays["gluteal_support_vertex_ids"]
    count = max(8, math.ceil(len(butt)*.15))
    butt_z = float(np.sort(vertices[butt, 2])[:count].mean())
    foot_ids = [np.flatnonzero(mask.detach().cpu().numpy()).tolist() for mask in (anatomy.left_foot_support, anatomy.right_foot_support)]
    foot_z = [float(np.sort(vertices[ids, 2])[:min(32, len(ids))].mean()) for ids in foot_ids]
    proxy = {"contact_ids": [0], "contact_drop_m": [float(joints[0, 2])-butt_z],
             "foot_drop_m": [float(joints[i, 2])-z for i, z in zip((10, 11), foot_z)],
             "body_identity_sha256": inputs["body_identity"]["sha256"]}
    require(all(0 <= value <= .30 for value in [*proxy["contact_drop_m"], *proxy["foot_drop_m"]]),
            "Body-derived proxy offset exceeds registered guidance bounds; do not clip")
    proxy_proof = {"schema": "p555.current_body_proxy_offsets.v1", "source_candidate": source_inputs["candidate"],
                   "source_body_identity": inputs["body_identity"], "parameters": proxy,
                   "gluteal_support_vertex_ids": butt.tolist(), "left_foot_vertex_ids": foot_ids[0],
                   "right_foot_vertex_ids": foot_ids[1], "gluteal_lower_envelope_vertex_count": count,
                   "foot_lower_envelope_count_per_side": 32, "contact_proxy_is_not_actual_mesh_contact_validation": True,
                   "original_sdf_and_release_gates_unchanged": True}
    proof_path = output / "guidance_proxy_receipt.json"
    write_once(proof_path, proxy_proof)
    return artifact(path), proxy, artifact(proof_path), body_identity, foot_ids


def compile_adapter(proposal_path, qwen_review_path, output):
    import fast_keypose as fast
    import stage3_affordance_guidance as guidance
    started = time.monotonic()
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite geometry Stage3 adapter")
    source_before = {key: artifact(path) for key, path in {"compiler": __file__, "neutral_seed": neutral.__file__,
                    "fast_numeric_import": fast.__file__, "affordance_guidance": guidance.__file__}.items()}
    proposal_record, review_record = artifact(proposal_path), artifact(qwen_review_path)
    proposal = _checked(proposal_record)
    require(proposal.get("schema") == "p555.geometry_guided_keypose_proposal.v1"
            and proposal.get("provider") == "declared_analytic_sit_seed_plus_current_geometry_ik"
            and proposal.get("status") == "numeric_candidate_pass", "Unsupported or failed geometric producer")
    require(proposal["source"]["sha256"] in PRODUCER_HASHES, "Unregistered geometric producer revision")
    verified(proposal["source"])
    require(proposal.get("stage3_handoff_allowed") is False and proposal.get("semantic_publication_allowed") is False
            and proposal.get("h3_model_used") is False and proposal.get("hybrikx_model_used") is False
            and proposal.get("historical_full_pose_or_motion_read") is False, "Old proposal claim/provenance changed")
    _all_true(proposal["gates"], NUMERIC_GATES, "Recorded 23 numerical gates")
    review, render = validate_semantics(proposal_record, proposal, review_record)
    arrays = load_candidate(proposal["outputs"]["candidate"])
    current = fast.load_current(verified(proposal["inputs"]["stage1"]))
    stage1, target = current["stage1"], current["target"]
    steps = stage1["interaction_plan"]["steps"]
    require(len(steps) == 1 and steps[0]["action"] == "sit" and len(steps[0]["target_ids"]) == 1
            and target["action_family"] == "sit", "P555 Stage3 only supports one feet-grounded sit interaction")
    require(proposal["scene_id"] == stage1["scene_id"] and proposal["target_instance_id"] == target["target_instance_id"],
            "Proposal current scene/target mismatch")
    require(render["original_mesh"] == target["artifacts"]["original_scene_mesh"], "Qwen render uses another scene mesh")
    require(np.allclose(arrays["contact_world_xyz_m"], proposal["contact_world_xyz_m"], atol=1e-6, rtol=0)
            and abs(math.atan2(math.sin(arrays["root_xyz_yaw"][3]-current["binding"]["contact_forward_yaw_rad"]),
                              math.cos(arrays["root_xyz_yaw"][3]-current["binding"]["contact_forward_yaw_rad"]))) <= 1e-6,
            "Candidate contact/semantic facing differs from current planning")
    body_assets = neutral.verify_neutral_assets(proposal["inputs"]["body_model"])
    numeric_start = time.monotonic()
    numeric, fields, anatomy = numerical_recheck(proposal, arrays, current)
    numeric_seconds = time.monotonic()-numeric_start
    bundle = _checked(stage1["bundle"])
    key_record, route_record = bundle["artifacts"]["key_nodes"], bundle["artifacts"]["navmesh_route"]
    # The retained NavMesh producer publishes hash-bound raw JSON, not a
    # sealed receipt. Its exact bytes are bound by the sealed Stage1 bundle.
    keynodes, route = _checked(key_record), _checked(route_record, sealed=False)
    require(keynodes["inputs"]["navmesh_route"] == route_record
            and keynodes["navmesh_execution_route"]["all_segment_samples_human_free"] is True
            and keynodes["collision_safe_control_polyline"]["direct_linear_interpolation_allowed"] is True,
            "Current route/control lineage invalid")
    controls = np.asarray(keynodes["collision_safe_control_polyline"]["nodes_world_xy_m"], dtype=np.float64)
    dense = np.asarray(route["route_world_xy_m"], dtype=np.float64)
    require(controls.ndim == dense.ndim == 2 and controls.shape[1:] == dense.shape[1:] == (2,)
            and len(controls) >= 2 and np.isfinite(controls).all() and np.isfinite(dense).all(), "Invalid current route")
    require(np.linalg.norm(controls[0]-dense[0]) <= 2e-4 and np.linalg.norm(controls[-1]-dense[-1]) <= 2e-4,
            "Controls changed the current route endpoints")
    source_root = np.asarray(keynodes["start"]["source"]["root_xyz_yaw_world_zup"], dtype=np.float64)
    require(source_root.shape == (4,) and np.isfinite(source_root).all(), "No explicit current start root")
    inputs = {"proposal": proposal_record, "qwen_review": review_record, "candidate": proposal["outputs"]["candidate"],
              "stage1_execution": proposal["inputs"]["stage1"], "stage1_target": stage1["target"], "stage1_bundle": stage1["bundle"],
              "key_nodes": key_record, "navmesh_route": route_record, "raw_occupancy": artifact(current["occupancy"]),
              "target_occupancy_mask": artifact(current["target_mask"]), "sdf_cache": proposal["inputs"]["sdf_cache"],
              "body_model": proposal["inputs"]["body_model"], "original_scene_mesh": target["artifacts"]["original_scene_mesh"]}
    # Before creating the output, reject obviously nonlocal centre targets using
    # the current standing height. The exact neutral height is checked again.
    route_contact_check(controls[-1], float(source_root[2]), arrays, fields.non_target.sample)
    output.mkdir(parents=True)
    seed_receipt = neutral.build_seed(output / "neutral_seed", scene_id=stage1["scene_id"], instruction=stage1["instruction"],
        start_xy=source_root[:2], yaw=source_root[3], body_assets=body_assets,
        source_bindings={"stage1_execution": inputs["stage1_execution"], "key_nodes": key_record, "navmesh_route": route_record})
    seed_path = output / "neutral_seed/receipt.json"
    seed = neutral.load_seed(seed_path)
    old_contract = neutral.module("p555_retained_route_math", neutral.P523_FRAME.parent / "p523_stage3_contract.py")
    roots = old_contract.extract_route_roots(keynodes, seed.root_xyz_yaw_world_zup)
    transition = route_contact_check(roots[-1][:2], float(seed.root_xyz_yaw_world_zup[2]), arrays, fields.non_target.sample)
    fingerprint = digest({"mesh_sha256": inputs["original_scene_mesh"]["sha256"], "occupancy_sha256": inputs["raw_occupancy"]["sha256"]})
    query_id = digest({"scene_id": stage1["scene_id"], "scene_fingerprint": fingerprint,
                       "fixed_geometry_sha256": stage1["source_fixed_geometry"]["sha256"], "instruction": stage1["instruction"],
                       "start_world_xy_m": source_root[:2].tolist()})
    surface = {"target_instance_id": target["target_instance_id"], "target_class": target["target_class"],
               "selected_surface_id": target["selected_surface_id"], "selected_surface_sha256": target["selected_surface_sha256"],
               "target_occupancy_mask_world_xyz_sha256": target["target_occupancy_mask_world_xyz_sha256"],
               "fixed_generated_butt_vertex_indices": arrays["gluteal_support_vertex_ids"].tolist(),
               "butt_vertex_source": inputs["candidate"], "selected_approach_option_id": current["binding"]["selected_approach_option_id"]}
    guidance_input, proxy, proxy_receipt, body_identity, foot_ids = _guidance_products(output, current=current, arrays=arrays,
        anatomy=anatomy, source_inputs=inputs, query_id=query_id, body_assets=body_assets)
    surface.update(left_foot_vertex_indices=foot_ids[0], right_foot_vertex_indices=foot_ids[1])
    packet = build_packet(scene_id=stage1["scene_id"], instruction=stage1["instruction"], roots=roots, arrays=arrays,
                          source_artifacts=inputs, target_contract=surface)
    packet_path = output / "packet.json"
    write_once(packet_path, packet)
    numeric_path = output / "numeric_recheck.json"
    write_once(numeric_path, numeric)
    for row in [*source_before.values(), *inputs.values(), *numeric["numerical_sources"]]:
        verified(row)
    result = {"schema": SCHEMA, "status": "compiled_p555_geometric_stage3_input_not_motion_release",
              "producer": PROVENANCE, "scope": "single_interaction_neutral_feet_grounded_sit",
              "scene_id": stage1["scene_id"], "instruction": stage1["instruction"], "query_id": query_id,
              "episode_id": query_id, "scene_fingerprint": fingerprint, "inputs": inputs,
              "packet": artifact(packet_path), "static_seed": seed_receipt["seed"], "static_seed_receipt": artifact(seed_path),
              "numeric_recheck": artifact(numeric_path), "body_identity": {**body_identity, "actual_model_sha256": neutral.NEUTRAL_SHA256, "betas": [0.]*10},
              "surface_contract": surface, "route_contact_validation": transition,
              "guidance_input": guidance_input, "guidance_proxy_map": proxy, "guidance_proxy_receipt": proxy_receipt,
              "implementation_sources": source_before, "retained_route_math": artifact(neutral.P523_FRAME.parent / "p523_stage3_contract.py"),
              "new_entry_rollout_allowed": True, "old_proposal_handoff_flag_changed": False,
              "legacy_P530_or_H3_lineage_claimed": False, "motion_generated": False,
              "motion_release_authorized": False, "real_positive_credit_allowed": False,
              "original_release_gates_required": True, "current_fullmesh_motion_recheck_required": True,
              "timings": {"numeric_revalidation_seconds": numeric_seconds, "compile_elapsed_seconds": time.monotonic()-started}}
    write_once(output / "adapter.json", result)
    return result


def load_adapter(path):
    """Hard new entry admission; refuses old schemas and rechecks bound evidence."""
    import stage3_affordance_guidance as guidance
    value = read_sealed(path)
    require(value.get("schema") == SCHEMA and value.get("status") == "compiled_p555_geometric_stage3_input_not_motion_release"
            and value.get("producer") == PROVENANCE and value.get("new_entry_rollout_allowed") is True,
            "Not a compiled P555 geometry Stage3 adapter")
    require(value.get("scope") == "single_interaction_neutral_feet_grounded_sit"
            and value.get("legacy_P530_or_H3_lineage_claimed") is False and value.get("motion_release_authorized") is False
            and value.get("motion_generated") is False and value.get("real_positive_credit_allowed") is False,
            "Geometry adapter claims old lineage or completed motion")
    require(value["implementation_sources"]["compiler"] == artifact(__file__)
            and value["implementation_sources"]["neutral_seed"] == artifact(neutral.__file__), "New compiler/seed source drift")
    for row in [*value["implementation_sources"].values(), *value["inputs"].values(), value["retained_route_math"],
                value["packet"], value["static_seed"], value["static_seed_receipt"], value["numeric_recheck"],
                value["guidance_input"], value["guidance_proxy_receipt"]]:
        verified(row)
    proposal = _checked(value["inputs"]["proposal"])
    validate_proposal(proposal)
    require(proposal["outputs"]["candidate"] == value["inputs"]["candidate"]
            and proposal["inputs"]["stage1"] == value["inputs"]["stage1_execution"],
            "Adapter proposal/candidate changed")
    validate_semantics(value["inputs"]["proposal"], proposal, value["inputs"]["qwen_review"])
    numeric = _checked(value["numeric_recheck"])
    require(numeric["schema"] == "p555.exact_body_candidate_recheck.v1" and numeric["thresholds"] == THRESHOLDS,
            "Wrong numeric revalidation proof")
    _all_true(numeric["gates"], NUMERIC_GATES, "Recomputed candidate")
    for row in numeric["numerical_sources"]:
        verified(row)
    packet = _checked(value["packet"])
    require(packet["schema"] == PACKET_SCHEMA and packet["provenance"] == PROVENANCE
            and packet["source_artifacts"] == value["inputs"] and packet["scene_name"] == value["scene_id"]
            and packet["text"] == value["instruction"], "Packet lineage/query mismatch")
    arrays = load_candidate(value["inputs"]["candidate"])
    import fast_keypose as fast
    current = fast.load_current(verified(proposal["inputs"]["stage1"]))
    stage1, target = current["stage1"], current["target"]
    steps = stage1["interaction_plan"]["steps"]
    require(len(steps) == 1 and steps[0]["action"] == "sit" and len(steps[0]["target_ids"]) == 1
            and target["action_family"] == "sit" and value["scene_id"] == stage1["scene_id"]
            and value["instruction"] == stage1["instruction"], "Adapter scope differs from current plan")
    require(value["inputs"]["stage1_target"] == stage1["target"] and value["inputs"]["stage1_bundle"] == stage1["bundle"]
            and value["inputs"]["raw_occupancy"] == artifact(current["occupancy"])
            and value["inputs"]["target_occupancy_mask"] == artifact(current["target_mask"])
            and value["inputs"]["original_scene_mesh"] == target["artifacts"]["original_scene_mesh"], "Current scene binding drift")
    measured_numeric, fields, anatomy = numerical_recheck(proposal, arrays, current)
    require(measured_numeric == _payload(numeric), "Numeric proof does not reproduce for this exact candidate")
    terminal = packet["nodes"][-1]
    require(terminal["phase"] == "CONTACT" and terminal["contact_joint_ids"] == [0, 1, 2]
            and [row["id"] for row in terminal["joints"]] == list(range(22)), "Packet lost mandatory full J22 contact")
    require(np.array_equal(np.asarray(terminal["full_smplx_keypose"]["joints_world_zup"]), arrays["joints_world_zup"])
            and np.array_equal(np.asarray(terminal["root_xyz_yaw"]), arrays["root_xyz_yaw"]), "Packet terminal keypose altered")
    transition = value["route_contact_validation"]
    _all_true(transition["checks"], ("local_body_scale_distance", "standing_endpoint_on_open_front_side", "bounded_lateral_entry",
              "non_target_connection_in_bounds", "non_target_body_width_clearance"), "Local sit transition")
    require(transition["policy"] == TRANSITION_POLICY
            and transition["original_route_modified"] is False, "Local sit transition was not admitted")
    seed = neutral.load_seed(value["static_seed_receipt"]["path"])
    require(seed.receipt["seed"] == value["static_seed"] and seed.receipt["scene_id"] == value["scene_id"]
            and seed.receipt["instruction"] == value["instruction"], "Seed differs from adapter query")
    require(seed.receipt["source_bindings"] == {key: value["inputs"][key] for key in neutral.SOURCE_KEYS}, "Seed route source drift")
    bundle = _checked(stage1["bundle"])
    require(bundle["artifacts"]["key_nodes"] == value["inputs"]["key_nodes"]
            and bundle["artifacts"]["navmesh_route"] == value["inputs"]["navmesh_route"], "Adapter substituted route")
    keynodes = _checked(value["inputs"]["key_nodes"])
    old = neutral.module("p555_retained_route_math", neutral.P523_FRAME.parent / "p523_stage3_contract.py")
    roots = old.extract_route_roots(keynodes, seed.root_xyz_yaw_world_zup)
    measured_transition = route_contact_check(roots[-1][:2], float(seed.root_xyz_yaw_world_zup[2]), arrays, fields.non_target.sample)
    require(measured_transition == transition, "Transition proof does not reproduce for this route/body")
    proxy_proof = _checked(value["guidance_proxy_receipt"])
    require(proxy_proof["schema"] == "p555.current_body_proxy_offsets.v1"
            and proxy_proof["source_candidate"] == value["inputs"]["candidate"]
            and proxy_proof["parameters"] == value["guidance_proxy_map"], "Body proxy lineage drift")
    butt = arrays["gluteal_support_vertex_ids"]
    count = max(8, math.ceil(len(butt)*.15))
    foot_ids = [np.flatnonzero(mask.numpy()).tolist() for mask in (anatomy.left_foot_support, anatomy.right_foot_support)]
    vertices, joints = arrays["vertices_world_zup"], arrays["joints_world_zup"]
    expected_proxy = {"contact_ids": [0], "contact_drop_m": [float(joints[0, 2])-float(np.sort(vertices[butt, 2])[:count].mean())],
                      "foot_drop_m": [float(joints[i, 2])-float(np.sort(vertices[ids, 2])[:min(32, len(ids))].mean())
                                      for i, ids in zip((10, 11), foot_ids)],
                      "body_identity_sha256": proxy_proof["source_body_identity"]["sha256"]}
    require(expected_proxy == value["guidance_proxy_map"], "Proxy offsets do not reproduce from actual body")
    surface = {"target_instance_id": target["target_instance_id"], "target_class": target["target_class"],
               "selected_surface_id": target["selected_surface_id"], "selected_surface_sha256": target["selected_surface_sha256"],
               "target_occupancy_mask_world_xyz_sha256": target["target_occupancy_mask_world_xyz_sha256"],
               "fixed_generated_butt_vertex_indices": butt.tolist(), "butt_vertex_source": value["inputs"]["candidate"],
               "selected_approach_option_id": current["binding"]["selected_approach_option_id"],
               "left_foot_vertex_indices": foot_ids[0], "right_foot_vertex_indices": foot_ids[1]}
    require(surface == value["surface_contract"], "Current contact surface/body subset differs")
    expected_packet = build_packet(scene_id=value["scene_id"], instruction=value["instruction"], roots=roots, arrays=arrays,
                                   source_artifacts=value["inputs"], target_contract=surface)
    require(expected_packet == _payload(packet), "Packet NAV/REORIENT/full22 differs from current route and candidate")
    fingerprint = digest({"mesh_sha256": value["inputs"]["original_scene_mesh"]["sha256"],
                          "occupancy_sha256": value["inputs"]["raw_occupancy"]["sha256"]})
    query_id = digest({"scene_id": stage1["scene_id"], "scene_fingerprint": fingerprint,
                      "fixed_geometry_sha256": stage1["source_fixed_geometry"]["sha256"], "instruction": stage1["instruction"],
                      "start_world_xy_m": keynodes["start"]["source"]["root_xyz_yaw_world_zup"][:2]})
    require(value["scene_fingerprint"] == fingerprint and value["query_id"] == value["episode_id"] == query_id,
            "Adapter query identity changed")
    binding = guidance.GuidanceBinding(value["guidance_input"]["path"])
    binding.assert_current(query_id=value["query_id"], body_identity_sha256=value["guidance_proxy_map"]["body_identity_sha256"])
    return value, packet, seed.receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--qwen-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_adapter(args.proposal, args.qwen_review, args.output)
    print(json.dumps({"status": result["status"], "scene_id": result["scene_id"], "packet": result["packet"]["path"]}), flush=True)
