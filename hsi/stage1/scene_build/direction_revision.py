from ._common import *
from .direction_contract import decision as arrow_decision
from .backrest import decision as part_decision

def direction_decision(rows):
    return part_decision(rows) if "pink_part_is_backrest" in rows[0]["parsed"] else arrow_decision(rows)

def run(geometry_path, review_paths, output):
    geometry = read_sealed(geometry_path)
    result = copy.deepcopy(geometry)
    rows = {r["instance_id"]: r for r in result["objects"]}
    changes = []
    for review_path in review_paths:
        review = read_sealed(review_path)
        if review["source_geometry"] != artifact(geometry_path) or review["scene_id"] != geometry["scene_id"]:
            raise ValueError("Direction review does not bind to this exact fixed geometry")
        if any(review[k] is not False for k in ("human_query_read", "stage2_generated_person_read", "qwen_received_numeric_geometry")):
            raise ValueError("Direction review crossed the scene-only boundary")
        decision = direction_decision(review["judgments"])
        if decision != review["decision"]:
            raise ValueError("Direction decision does not reproduce from actual Qwen outputs")
        row = rows[review["instance_id"]]
        if row != review["source_shape"]:
            raise ValueError("Object shape or direction changed since review")
        prior = copy.deepcopy(row["direction_hypotheses"])
        if decision["direction_review_accepted"]:
            candidates = []
            for index, key in enumerate(decision["selected_direction_ids"]):
                vector = review["candidate_vectors_computed_by_geometry"][key]
                norm = math.hypot(*vector)
                if abs(norm-1) > 1e-6:
                    raise ValueError("Geometric direction candidate is not a unit vector")
                candidates.append({"candidate_id": "DIRECTION_"+str(index), "outward_world_xy": vector,
                    "source_review_candidate_id": key})
            row["direction_hypotheses"] = {"orientation_resolved": len(candidates) == 1,
                "method": "actual_qwen_selection_of_geometric_arrow_ids", "candidates": candidates,
                "category_override_allowed": False, "apply_single_front_45_degree_gate": len(candidates) == 1,
                "direction_coordinates_computed_by_qwen": False, "visual_review": artifact(review_path)}
            row["front_evidence"] = {**row["front_evidence"], "method": "actual_qwen_selection_of_geometric_arrow_ids",
                "prior_geometric_hypothesis": prior, "direction_review": artifact(review_path)}
        else:
            row["query_usable_for_sit"] = False
            row["direction_recovery_required"] = True
        row["direction_review"] = artifact(review_path)
        changes.append({"instance_id": row["instance_id"], "decision": decision,
            "category_before": review["source_shape"]["category"], "category_after": row["category"],
            "prior_directions": prior, "current_directions": row["direction_hypotheses"]})
    result.update(source_geometry_before_direction_review=artifact(geometry_path), direction_revisions=changes,
        source_code=artifact(__file__), object_categories_and_surface_masks_changed=False,
        geometry_reextracted=False, old_query_plans_automatically_revalidated=False,
        qwen_direction_candidate_id_selection_not_coordinate_prediction=True)
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / "receipt.json", result, seal=True)
    print({"scene": geometry["scene_id"], "directions": [{"id": r["instance_id"], "decision": r["decision"]} for r in changes]}, flush=True)
