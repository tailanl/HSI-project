"""Current route binding with explicit P550 direction semantics."""
import math
from hsi.common.artifacts import artifact, digest, read_sealed, verified


def checked(record):
    return read_sealed(verified(record))


def bind(stage1):
    if stage1.get("status") != "complete": raise ValueError("Stage1 did not complete")
    target, bundle, quality = [checked(stage1[k]) for k in ("target", "bundle", "planner_receipt")]
    if quality["schema"] != "p550.decoupled_stage1_geometry_quality.v1": raise ValueError("Not a P550 quality protocol")
    if bundle["status"] != "complete_stage1_verified" or quality["status"] != "sealed_quality_pass": raise ValueError("Unsealed Stage1")
    if not quality["quality_gates"] or not all(quality["quality_gates"].values()) or not quality["publish_gate_pass"] or not quality["stage2_handoff_allowed"]:
        raise ValueError("Stage1 geometry quality failure")
    if quality["inputs"]["planner_receipt"] != stage1["target"] or bundle["artifacts"]["stage1_target"] != stage1["target"]:
        raise ValueError("Quality/bundle target mismatch")
    for key in ("target_instance_id", "selected_surface_id", "selected_surface_sha256"):
        if bundle[key] != target[key]: raise ValueError("Stage1 identity mismatch: "+key)
    compile_record = bundle["artifacts"]["roadmap_compile"]
    compiled = checked(compile_record)
    if compiled["status"] != "fresh_roadmap_request_compiled" or compiled["inputs"]["p523_stage1_target"] != stage1["target"]:
        raise ValueError("Compiled route target mismatch")
    if compiled["p504_request"] != bundle["artifacts"]["roadmap_request"]: raise ValueError("Route request mismatch")
    verified(bundle["artifacts"]["roadmap_request"])
    selection = compiled["current_only_option_selection"]
    if selection["status"] != "verified_path_cost_first_approach_selected" or selection["future_endpoint_or_motion_used"] is not False or selection["legacy_route_used_for_ranking"] is not False:
        raise ValueError("Route selection is not current-only")
    option_id = bundle["selected_approach_option_id"]
    if option_id != compiled["selected_approach_option_id"] or option_id != selection["selected_approach_option_id"]:
        raise ValueError("Approach selection mismatch")
    matches = [r for r in compiled["approach_audits"] if r["option_id"] == option_id]
    surfaces = [r for r in target["candidate_surfaces"] if r["candidate_id"] == target["selected_surface_id"]]
    if len(matches) != 1 or len(surfaces) != 1: raise ValueError("Approach/surface not unique")
    selected, surface = matches[0], surfaces[0]
    options = [r for r in surface["approach_options"] if r["option_id"] == option_id]
    if len(options) != 1: raise ValueError("Approach option not unique")
    option = options[0]
    if option["approach_world_xy_m"] != selected["requested_approach_world_xy_m"]: raise ValueError("Approach coordinate drift")
    if not all(selected[k] is True for k in ("fresh_reachable", "terminal_clearance_gate_passed", "nearest_human_free_cell_reachable")):
        raise ValueError("Approach failed reachability or clearance")
    if any(selected[k] is not False for k in ("legacy_route_fields_read", "legacy_route_used_for_ranking")):
        raise ValueError("Historical route contamination")
    hypotheses = surface["surface_extraction_audit"]["front_inference"]["p550_direction_hypotheses"]
    by_id = {r["candidate_id"]: r for r in hypotheses["candidates"]}
    chosen = by_id[option["p550_direction_hypothesis_id"]]
    expected_yaw = math.atan2(chosen["outward_world_xy"][1], chosen["outward_world_xy"][0])
    difference = abs(math.atan2(math.sin(option["contact_forward_yaw_rad"]-expected_yaw), math.cos(option["contact_forward_yaw_rad"]-expected_yaw)))
    if not math.isfinite(difference) or difference > 1e-8: raise ValueError("Body facing not bound to the selected hypothesis")
    if option["front_orientation_difference_rad"] > math.pi/2+1e-8:
        raise ValueError("Approach lies behind selected interaction-side hypothesis")
    if bundle["route_summary"].get("all_segment_samples_human_free") is not True: raise ValueError("NavMesh collision check failed")
    result = {"schema": "p550.current_route_selected_approach.v1", "scene_id": target["scene_id"],
        "instruction": target["instruction"], "target_instance_id": target["target_instance_id"],
        "selected_surface_id": target["selected_surface_id"], "selected_approach_option_id": option_id,
        "selected_approach_world_xy_m": selected["requested_approach_world_xy_m"],
        "snapped_goal_world_xy_m": selected["snapped_goal_world_xy_m"], "contact_forward_yaw_rad": expected_yaw,
        "direction_hypothesis_id": chosen["candidate_id"], "orientation_resolved_from_geometry": hypotheses["orientation_resolved"],
        "terminal_clearance_gate_passed": True, "fresh_reachability_gate_passed": True, "route_collision_gate_passed": True,
        "direction_hypothesis_binding_verified": True, "physical_body_facing_requires_stage2_recheck": True,
        "qwen_generated_coordinates": False, "provisional_surface_approach_used": False,
        "protocol_change": "single_axis_sign_45deg_veto_replaced_by_explicit_hypothesis_and_independent_body_facing",
        "physical_keypose_thresholds_changed": False, "source_bundle": stage1["bundle"], "source_target": stage1["target"],
        "source_quality": stage1["planner_receipt"], "source_roadmap_compile": compile_record, "source": artifact(__file__)}
    result["receipt_payload_sha256"] = digest(result)
    return result

