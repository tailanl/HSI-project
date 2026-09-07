"""Task-bound qualitative descriptions for every numerical route milestone."""
import json
from pathlib import Path
from hsi.common.artifacts import artifact, digest, read_sealed, verified, write_once
from .binding import bind, checked

def run(stage1_path, output, *, qwen):
    stage1 = read_sealed(stage1_path)
    approach = bind(stage1)
    bundle = checked(stage1["bundle"])
    target = checked(stage1["target"])
    key_record = bundle["artifacts"]["key_nodes"]
    keynodes = checked(key_record)
    nodes = keynodes["semantic_key_node_summary"]["nodes"]
    qualitative = [{"node_id": f"NAV_{r['sequence_index']:03d}", "event_labels": r["event_labels"],
        "role": "navigation_only_not_contact_pose"} for r in nodes]
    qualitative.append({"node_id": "INTERACTION_000", "event_labels": ["SIT_CONTACT_KEYPOSE"],
        "role": "semantic_terminal_pose_generated_by_2D_then_refined"})
    ids = [r["node_id"] for r in qualitative]
    props = {"node_id": {"type": "string", "enum": ids}, "description": {"type": "string", "minLength": 12, "maxLength": 240}}
    schema = {"type": "object", "additionalProperties": False, "properties": {"nodes": {"type": "array",
        "items": {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)},
        "minItems": len(ids), "maxItems": len(ids)}}, "required": ["nodes"]}
    prompt = "Write one brief English action description for EVERY supplied node, in order. Do not add/remove nodes or coordinates. " \
        "NAV nodes follow the bound NavMesh route and are NOT already seated; NAV_GOAL_APPROACH means arrive near the furniture, not contact. " \
        "Only INTERACTION_000 is the final supported sitting pose, both feet on the floor. Explain turns/narrow regions only when present in event_labels. " \
        "Do not invent left/right when not supplied.\nUSER_REQUEST="+json.dumps(stage1["instruction"])+"\nTARGET_ID="+target["target_instance_id"]+"\nNODES="+json.dumps(qualitative)
    output.mkdir(parents=True, exist_ok=False)
    call = qwen.call
    response = call(prompt, schema, output / "qwen_descriptions.json", max_tokens=1000, call_id="p550_keynode_descriptions")
    if [r["node_id"] for r in response["nodes"]] != ids: raise ValueError("Missing, reordered or invented keynode")
    annotated = []
    for source, row in zip(nodes, response["nodes"]):
        if not isinstance(row["description"], str) or not 12 <= len(row["description"]) <= 240: raise ValueError("Invalid keynode description")
        annotated.append({**source, **row, "coordinates_source": "verified_navmesh_not_qwen", "keypose_role": "locomotion_condition"})
    selected = next(r for r in target["candidate_surfaces"] if r["candidate_id"] == target["selected_surface_id"])
    terminal = {**response["nodes"][-1], "target_instance_id": target["target_instance_id"], "surface_id": target["selected_surface_id"],
        "surface_centre_world_xyz_zup_m": selected["surface"]["centre_world_xyz_zup_m"],
        "contact_forward_yaw_rad": approach["contact_forward_yaw_rad"], "keypose_role": "2D_semantic_interaction_then_3D_refine",
        "coordinates_source": "verified_scene_geometry_not_qwen", "published_contact_pose": False}
    receipt = {"schema": "p550.described_stage1_keypoint_sequence.v1", "scene_id": stage1["scene_id"],
        "source_stage1": artifact(stage1_path), "source_keynodes": key_record, "source_target": stage1["target"],
        "qwen_descriptions": artifact(output / "qwen_descriptions.json"), "navigation_nodes": annotated,
        "interaction_keypoints": [terminal], "all_keypoints_have_descriptions": True,
        "qwen_received_numeric_geometry": False, "navigation_route_still_authoritative": True,
        "semantic_plan_replaces_no_numeric_geometry": True, "source_code": artifact(__file__)}
    write_once(output / "receipt.json", receipt, seal=True)
    print({"navigation_descriptions": len(annotated), "interaction_descriptions": 1}, flush=True)

