"""Current local sit transition and ordered packet arithmetic."""

from __future__ import annotations

import math
import numpy as np

from hsi.common.artifacts import require

PACKET_SCHEMA = "hsi.stage3.image_keypose_packet.v1"
PROVENANCE = "hsi_verified_h3_hybrikx"

TRANSITION_POLICY = {"schema": "p555.local_sit_transition_admission.v1", "maximum_root_xy_distance_m": .65,
                     "maximum_distance_leg_length_ratio": .8, "distance_measurement_slack_m": .02,
                     "maximum_lateral_m": .35, "maximum_lateral_hip_width_ratio": 2.,
                     "minimum_front_projection_m": .05, "sample_spacing_m": .01,
                     "non_target_clearance_shoulder_halfwidth_extra_m": .02,
                     "maximum_non_target_proxy_radius_m": .28,
                     "geometry_connection_is_not_a_generated_motion": True, "original_route_modified": False}

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
                      "joints": [], "contact_joint_ids": [], "confidence": 1., "source": "current_verified_contact_region_route",
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
                  "source": "hsi_verified_h3_hybrikx",
                  "full_smplx_keypose": {"schema": "hsi.full_smplx_terminal_keypose.v1", "coordinate_system": "world_zup",
                      "root_xyz_yaw": terminal.tolist(), "joints_world_zup": arrays["joints_world_zup"].tolist(),
                      "body_pose_axis_angle": arrays["body_pose_axis_angle"].tolist(), "betas": arrays["betas"].tolist(),
                      "pose_frame_to_world_rotation": arrays["pose_frame_to_world_rotation"].tolist(),
                      "source_stage2_keypose_sha256": source_artifacts["candidate"]["sha256"]}})
    require(len(roots) >= 2, "Need current route start and endpoint")
    return {"schema": PACKET_SCHEMA, "coordinate_system": "world_zup", "seq_name": "hsi_"+scene_id+"_current_sit",
            "scene_name": scene_id, "text": instruction, "provenance": PROVENANCE, "nodes": nodes,
            "source_artifacts": source_artifacts, "surface_contract": target_contract,
            "executor_contract": {"route_nodes_are_root_only": True, "condition_window_capacity": 8,
                "condition_window_is_rolling": True, "terminal_full22_keypose_count": 1, "orientation_only_node_count": 1,
                "navigation_control_node_count": len(roots), "total_plan_node_count": len(nodes)},
            "h3_model_used": True, "hybrikx_model_used": True, "legacy_P530_lineage_claimed": False,
            "motion_release_authorized": False}

