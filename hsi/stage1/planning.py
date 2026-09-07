"""Current ID-only semantic planning and strict fixed-scene compatibility."""
import copy
import json
from pathlib import Path
import numpy as np
from hsi.common.artifacts import artifact, digest, read_sealed, verified, write_once

ACTIONS = ["sit", "lie", "stand", "walk", "reach", "touch", "open", "close", "pick_up", "put_down", "other"]

STEP_PROPS = {"action": {"type": "string", "enum": ACTIONS},
    "target_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
    "reference_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    "intent_description": {"type": "string", "maxLength": 220},
    "target_selection_reason": {"type": "string", "maxLength": 220}}

PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "steps": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": STEP_PROPS, "required": list(STEP_PROPS)}, "maxItems": 8},
    "needs_clarification": {"type": "boolean"}, "reason": {"type": "string", "maxLength": 220}},
    "required": ["steps", "needs_clarification", "reason"]}

def inventory(semantics, geometry):
    geo = {r["instance_id"]: r for r in geometry["objects"]}
    rows = []
    for obj in semantics["objects"]:
        key = obj["instance_id"]
        rows.append({"instance_id": key, "category": obj["semantic"]["object_class"],
            "description": obj["semantic"]["visual_description"], "mask_state": obj["semantic"]["mask_state"],
            "category_confirmed": obj["decision"]["category_confirmed"],
            "query_usable": obj["decision"]["query_usable"], "sit_surface_available": geo[key]["query_usable_for_sit"]})
    return rows

def validate_plan(plan, rows):
    if set(plan) != set(PLAN_SCHEMA["required"]) or type(plan["needs_clarification"]) is not bool:
        raise ValueError("Query plan schema drift")
    if not isinstance(plan["steps"], list) or not 1 <= len(plan["steps"]) <= 8:
        raise ValueError("Plan needs one to eight explicit interaction steps")
    by_id = {r["instance_id"]: r for r in rows}
    for step in plan["steps"]:
        if set(step) != set(STEP_PROPS) or step["action"] not in ACTIONS:
            raise ValueError("Invalid semantic interaction step")
        for field in ("target_ids", "reference_ids"):
            values = step[field]
            if not isinstance(values, list) or len(values) != len(set(values)) or any(k not in by_id for k in values):
                raise ValueError("Qwen invented or duplicated scene IDs")
        for key in step["target_ids"]:
            if not by_id[key]["query_usable"]: raise ValueError("Qwen selected an unresolved mask/category")
            if step["action"] == "sit" and not by_id[key]["sit_surface_available"]:
                raise ValueError("Qwen selected a target lacking a current sitting surface")
    return plan

def compatibility_inputs(semantics, geometry, step, instruction, output, plan_record):
    sam = spatial_compatibility(read_sealed(verified(geometry["source_sam"])), logical_compatibility)
    query_sam = copy.deepcopy(sam)
    query_sam["instruction"] = instruction
    query_sam["p550_query_view"] = {"schema": "p550.query_bound_perception_compatibility_view.v1",
        "source_atomic_inference": geometry["source_sam"], "source_semantics": geometry["source_semantics"],
        "new_sam_inference_performed": False, "atomic_support_changed": False,
        "instruction_attached_after_scene_understanding": True}
    sam_path = output / "inputs/query_sam_compatibility.json"
    write_once(sam_path, query_sam, seal=True)
    lookup = {r["instance_id"]: r for r in semantics["objects"]}
    selected_ids = step["target_ids"]
    if not selected_ids: raise ValueError("No verified target was selected")
    nodes = []
    for key in selected_ids:
        obj = lookup[key]
        source = obj["geometry_source"]
        nodes.append({"node_id": key, "instance_id": key, "object_class": obj["semantic"]["object_class"],
            "is_furniture": True, "is_sittable": obj["semantic"]["is_sittable_object"], "part_role": "whole_object",
            "valid_for_metric_graph": True, "validation_failures": [],
            "semantic_confidence": obj["semantic"]["confidence"], "bounds_world_zup_m": source["bounds_world_zup_m"],
            "centroid_world_zup_m": source["centroid_world_zup_m"], "semantic_evidence": obj["evidence"]})
    graph = {"schema": "p515.lingo_metric_scene_graph.v1", "scene_id": geometry["scene_id"],
        "instruction": instruction, "instruction_query": {"action": "sit", "target_class": nodes[0]["object_class"], "relation": None},
        "status": "target_verified" if len(nodes) == 1 else "ambiguous_target", "nodes": nodes, "relations": [],
        "selected_target_instance_id": selected_ids[0] if len(nodes) == 1 else None,
        "target_selection": {"failures": [] if len(nodes) == 1 else ["target_not_unique"], "reference_entity_ids": [],
            "candidates": [{"instance_id": key, "relation_gate_pass": True} for key in selected_ids]},
        "inputs": {"sam": artifact(sam_path)}, "publish_gate_pass": True,
        "p550_protocol": {"schema": "p550.fixed_inventory_query_graph_adapter.v1", "source_semantics": geometry["source_semantics"],
            "semantic_plan": plan_record, "query_target_ids_from_qwen": True, "geometry_size_overrides_category": False,
            "qwen_reference_ids": step["reference_ids"], "relation_resolution": "qualitative_Qwen_target_ID_selection",
            "remaining_target_ties_resolved_by_current_geometry": True, "new_classification_performed": False}}
    graph_path = output / "inputs/query_graph_compatibility.json"
    write_once(graph_path, graph, seal=True)
    return sam_path, graph_path

def logical_compatibility(value):
    if value.get("schema") != "p550.visual_logical_scene_instances.v1": return value
    if value.get("status") != "logical_objects_ready" or value.get("new_sam_inference_performed") is not False:
        raise ValueError("Invalid logical scene composition")
    verified(value["source_atomic_instances"])
    verified(value["support_vertices"])
    for row in value["compositions"]:
        evidence = read_sealed(verified(row["evidence"]))
        if evidence["instance"]["instance_id"] != row["instance_id"]: raise ValueError("Logical object identity drift")
        read_sealed(verified(evidence["review"]))
    result = copy.deepcopy(value)
    result.update(schema="p515.lingo_sam31_atomic_multiview_instances.v1", status="atomic_instances_ready",
        p550_compatibility_extension={"schema": "p550.logical_instances_p523_readonly_adapter.v1",
            "source_schema": value["schema"], "source_payload_sha256": value["receipt_payload_sha256"],
            "source_atomic_inference": value["source_atomic_instances"], "contains_logical_non_atomic_objects": True,
            "published_atomic_schema_is_compatibility_only": True,
            "new_sam_inference_or_complete_link_claim_for_compositions": False, "source": artifact(__file__)})
    return result

def compile_navigation_prefix(plan):
    original = copy.deepcopy(plan)
    result = copy.deepcopy(plan)
    steps, mappings = [], []
    index = 0
    while index < len(original["steps"]):
        step = original["steps"][index]
        next_step = original["steps"][index+1] if index+1 < len(original["steps"]) else None
        if (step["action"] == "walk" and next_step and next_step["action"] == "sit"
                and step["target_ids"] and set(step["target_ids"]) == set(next_step["target_ids"])
                and set(step["reference_ids"]) == set(next_step["reference_ids"])):
            compiled = copy.deepcopy(next_step)
            compiled["intent_description"] = (step["intent_description"]+"; then "+next_step["intent_description"])[:240]
            mappings.append({"source_step_indices": [index,index+1], "compiled_interaction_index": len(steps),
                "walk_executor": "current_navmesh_route_to_same_target_approach",
                "sit_executor": "stage2_semantic_2D_keypose", "actions_dropped": []})
            steps.append(compiled)
            index += 2
        else:
            mappings.append({"source_step_indices": [index], "compiled_interaction_index": len(steps),
                "action": step["action"], "actions_dropped": []})
            steps.append(copy.deepcopy(step))
            index += 1
    result["steps"] = steps
    return result, {"raw_semantic_plan": original, "execution_compilation": mappings,
        "all_source_actions_preserved": True, "same_target_navigation_is_not_an_extra_contact_keypose": True}

def partition_verified(parent, main, residual):
    arrays = [np.asarray(value, dtype=np.int64).reshape(-1) for value in (parent, main, residual)]
    if any(len(a) != len(np.unique(a)) for a in arrays):
        raise ValueError("Duplicate component support indices")
    parent, main, residual = arrays
    if not len(main) or not len(residual) or np.intersect1d(main, residual).size:
        raise ValueError("Empty or overlapping component partition")
    if not np.array_equal(np.sort(parent), np.union1d(main, residual)):
        raise ValueError("Component recovery added or discarded original support")
    return True

def spatial_compatibility(value, fallback):
    if value.get("schema") != "p550.spatial_component_scene_instances.v1":
        return fallback(value)
    if value["status"] != "component_proposals_ready" or value["new_sam_inference_performed"] is not False \
            or value["original_atoms_removed"] is not False or value["original_support_modified"] is not False:
        raise ValueError("Spatial component source misstates its provenance")
    source = read_sealed(verified(value["source_atomic_instances"]))
    source_lookup = {r["instance_id"]: r for r in source["instances"]}
    lookup = {r["instance_id"]: r for r in value["instances"]}
    with np.load(verified(value["support_vertices"]), allow_pickle=False) as archive, \
            np.load(verified(source["support_vertices"]), allow_pickle=False) as original:
        for key, row in source_lookup.items():
            if lookup[key] != row:
                raise ValueError("Original source instance changed during partition proposal")
            file_key = row["support_vertices_file_key"]
            if not np.array_equal(archive[file_key], original[file_key]):
                raise ValueError("Original source support changed")
        for record in value["component_proposals"]:
            proposal = read_sealed(verified(record))
            if proposal["scene_id"] != value["scene_id"] or proposal["audit"]["proposal_eligible"] is not True \
                    or not all(v is True for v in proposal["audit"]["gates"].values()):
                raise ValueError("Unverified spatial component proposal")
            ids = [proposal[k] for k in ("parent_id", "candidate_id", "residual_id")]
            partition_verified(*(archive[lookup[k]["support_vertices_file_key"]] for k in ids))
    result = copy.deepcopy(value)
    result.update(schema="p515.lingo_sam31_atomic_multiview_instances.v1", status="atomic_instances_ready",
        p550_compatibility_extension={"schema": "p550.reviewed_spatial_component_p523_readonly_adapter.v1",
            "source_schema": value["schema"], "source_payload_sha256": value["receipt_payload_sha256"],
            "source_original_instances": value["source_atomic_instances"], "source": artifact(__file__),
            "published_atomic_schema_is_compatibility_only": True, "new_sam_inference_performed": False,
            "parent_and_residual_support_preserved": True, "component_partition_verified": True,
            "category_authority_remains_separate_fixed_qwen_semantics": True})
    return result

def compile_supported_sequence(plan, inventory):
    validate_plan(plan, inventory)
    if any(set(step["target_ids"]) & set(step["reference_ids"]) for step in plan["steps"]):
        raise ValueError("A target cannot be its own relational reference")
    compiled, mapping = compile_navigation_prefix(plan)
    unsupported = [{"compiled_index": i, "action": step["action"],
        "source_step_indices": mapping["execution_compilation"][i]["source_step_indices"]}
        for i, step in enumerate(compiled["steps"]) if step["action"] != "sit"]
    consumed = sorted(i for item in mapping["execution_compilation"] for i in item["source_step_indices"])
    if consumed != list(range(len(plan["steps"]))):
        raise ValueError("Sequence compilation lost or duplicated a source action")
    return compiled, mapping, unsupported

