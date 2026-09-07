from ._common import *
from .semantic_contract import semantic_status,size_diagnostic

def composition_description(category, members):
    descriptions = list(dict.fromkeys(r["semantic"]["visual_description"] for r in members))
    return ("One complete "+category+" formed by visually confirmed complementary regions. "
        "Original visual observations: "+"; ".join(descriptions))[:240]

def composition_eligible(review, snapshot_path):
    if review.get("whole_object_composition_authorized") is not True:
        return False
    snapshot = read_sealed(snapshot_path)
    objects = {r["instance_id"]: r for r in snapshot["objects"]}
    classes = {objects[k]["semantic"]["object_class"] for k in review["member_ids"]}
    judgments = review.get("judgments", [])
    if len(judgments) != 2:
        return False
    for judgment in judgments:
        value = judgment["parsed"]
        if value.get("all_regions_same_physical_object") is not True or value.get("union_covers_whole_visible_object") is not True or value.get("union_contains_other_furniture") is not False:
            return False
        confidence = value.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int,float)) or not .85 <= confidence <= 1:
            return False
        classes.add(value.get("object_class"))
    return len(classes) == 1

def run(snapshot_path, review_paths, output):
    snapshot = read_sealed(snapshot_path)
    source_sam = read_sealed(verified(snapshot["sources"]["atomic_instances"]))
    objects = {r["instance_id"]: copy.deepcopy(r) for r in snapshot["objects"]}
    reviewed, used = [], set()
    for path in review_paths:
        review = read_sealed(path)
        evidence_snapshot = read_sealed(verified(review["source_snapshot"]))
        if review["scene_id"] != snapshot["scene_id"] or evidence_snapshot["sources"] != snapshot["sources"]:
            raise ValueError("Part-review scene source drift")
        prior = {r["instance_id"]: r for r in evidence_snapshot["objects"]}
        ids = review["member_ids"]
        if used.intersection(ids): raise ValueError("Overlapping groups require a group review, not transitive pair merging")
        for key in ids:
            if objects[key]["evidence"] != prior[key]["evidence"]: raise ValueError("Part semantics changed since visual review")
            verified(objects[key]["evidence"])
        if len(review["judgments"]) != 2: raise ValueError("Two independent view-order reviews are required")
        classes = {objects[k]["semantic"]["object_class"] for k in ids}
        for judgment in review["judgments"]:
            value = judgment["parsed"]
            if value["all_regions_same_physical_object"] is not True or value["union_covers_whole_visible_object"] is not True or value["union_contains_other_furniture"] is not False:
                raise ValueError("Visual review does not authorize a whole logical object")
            if isinstance(value["confidence"], bool) or not .85 <= value["confidence"] <= 1:
                raise ValueError("Low or invalid composition confidence")
            classes.add(value["object_class"])
            call = read(judgment["generation"]["call_receipt"])
            if call["status"] != "complete" or call["result"] != judgment["generation"]:
                raise ValueError("Composition lacks the actual matching Qwen call")
        if len(classes) != 1: raise ValueError("Category disagreement requires classification review, not a geometry override")
        used.update(ids)
        reviewed.append((artifact(path), review, next(iter(classes))))
    output.mkdir(parents=True, exist_ok=False)
    from .geometry import geometry_kernel
    vertices, _ = geometry_kernel().load_obj_world_zup(verified(snapshot["sources"]["mesh"]))
    with np.load(verified(snapshot["sources"]["support_vertices"]), allow_pickle=False) as archive:
        supports = {key: archive[key] for key in archive.files}
    instances = copy.deepcopy(source_sam["instances"])
    compositions = []
    for review_record, review, category in reviewed:
        ids = review["member_ids"]
        identity = digest({"source_sam": snapshot["sources"]["atomic_instances"], "members": sorted(ids), "review": review_record})
        key = "SCENE_OBJECT_"+identity[:12].upper()
        member_rows = [objects[k] for k in ids]
        support = np.unique(np.concatenate([supports[r["geometry_source"]["support_vertices_file_key"]] for r in member_rows])).astype(np.int64)
        supports[key] = support
        points = vertices[support]
        bounds = np.stack((points.min(0), points.max(0))).tolist()
        views = sorted(set(v for r in member_rows for v in r["geometry_source"]["visible_view_ids"]))
        instance = {"instance_id": key, "semantic_class": None, "support_vertices_file_key": key,
            "support_vertex_count": len(support), "bounds_world_zup_m": bounds, "centroid_world_zup_m": points.mean(0).tolist(),
            "visible_view_ids": views, "cross_view_gate_pass": True,
            "published_support_policy": "union_of_verified_same_object_atoms_each_vertex_observed_in_two_views",
            "minimum_distinct_views_per_published_vertex": 2,
            "classifier_images": [e["marked"] for e in review["evidence"]],
            "atomic_source_observation_ids": sorted(set(o for r in member_rows for o in r["geometry_source"]["atomic_source_observation_ids"])),
            "source_masks": [m for r in member_rows for m in r["geometry_source"]["source_masks"]],
            "logical_object_composition": {"source_atom_ids": ids, "visual_review": review_record,
                "new_sam_inference": False, "atomic_complete_link_claim_applies_to_sources_only": True}}
        semantic = {"object_class": category, "mask_state": "whole_object", "is_sittable_object": True,
            "visible_seat_in_mask": any(r["semantic"]["visible_seat_in_mask"] for r in member_rows),
            "visible_backrest_in_mask": any(r["semantic"]["visible_backrest_in_mask"] for r in member_rows),
            "views_consistent": True, "confidence": min(j["parsed"]["confidence"] for j in review["judgments"]),
            "visual_description": composition_description(category, member_rows),
            "mask_issue": "Original atoms retained; whole-object identity confirmed by two ordered multi-view Qwen reviews."}
        evidence_path = output / "compositions" / key / "receipt.json"
        write_once(evidence_path, {"schema": "p550.verified_logical_object_composition.v1", "instance": instance,
            "semantic": semantic, "review": review_record, "source_members": [r["evidence"] for r in member_rows],
            "geometry_source": snapshot["sources"], "support_union_sha256": digest(support.tolist()),
            "semantic_class_changed_by_geometry": False, "source_code": artifact(__file__)}, seal=True)
        objects[key] = {"instance_id": key, "semantic": semantic, "decision": semantic_status(semantic),
            "evidence": artifact(evidence_path), "geometry_source": instance, "size_diagnostic": size_diagnostic(bounds)}
        for member in ids:
            objects[member]["represented_by_logical_object_id"] = key
        instances.append(instance)
        compositions.append({"instance_id": key, "source_atom_ids": ids, "evidence": artifact(evidence_path)})
    support_path = output / "logical_support_vertices.npz"
    np.savez_compressed(support_path, **supports)
    logical_sam_path = output / "logical_instances.json"
    logical = {"schema": "p550.visual_logical_scene_instances.v1", "status": "logical_objects_ready",
        "scene_id": snapshot["scene_id"], "source_original_scene_mesh": snapshot["sources"]["mesh"],
        "source_render_receipt": snapshot["sources"]["render"], "support_vertices": artifact(support_path),
        "source_atomic_instances": snapshot["sources"]["atomic_instances"], "instances": instances,
        "compositions": compositions, "new_sam_inference_performed": False, "original_atoms_removed": False,
        "instruction": "", "source_code": artifact(__file__)}
    write_once(logical_sam_path, logical, seal=True)
    value = copy.deepcopy(snapshot)
    value.update(source_snapshot_before_composition=artifact(snapshot_path), objects=list(objects.values()),
        processed_instance_count=len(objects), logical_compositions=compositions, source_code=artifact(__file__),
        original_atomic_count=len(source_sam["instances"]), new_sam_inference_performed=False)
    value["sources"].update(atomic_instances=artifact(logical_sam_path), support_vertices=artifact(support_path))
    write_once(output / "receipt.json", value, seal=True)
    print({"scene": value["scene_id"], "logical_compositions": compositions}, flush=True)
