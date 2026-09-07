"""Extract immutable Stage1A facts without pose/motion or model execution.

``build_scene_facts(final_scene_path, output_dir) -> receipt Path`` creates a
new directory, retaining every object (including unknown/rejected objects).
``load_scene_facts(path)`` validates the fact/descriptor artifact bindings.
``load_descriptors(path, eligible_only=True)`` returns (object, metadata, arrays)
triples, one per *audited* direction. No descriptor is a successful interaction:
all positive_credit fields are always zero. Experience payloads belong elsewhere.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from memory_common import artifact, verified, read_sealed, read_json, write_once, digest, require
from surface_geometry import describe_surface, validate_descriptor, load_obj_world_zup, _indices

SCHEMA = "p555.scene_geometry_facts.v1"
FINAL_SCHEMA = "p550.final_fixed_scene_understanding.v1"
ANCESTOR_FIELDS = ("source_geometry_before_direction_review", "source_geometry_before_final_publication",
                   "source_geometry_before_semantic_backrest_review")
SIT_CLASSES = {"chair", "armchair", "sofa", "bench", "stool"}
ARROW_SCHEMA = "p550.scene_only_independent_single_camera_direction.v3"
PART_SCHEMA = "p550.scene_only_semantic_backrest_direction.v1"


def _ancestor(current_path, expected, cache, depth=0, seen=None):
    require(depth <= 16, "Geometry ancestry exceeds bound")
    path = Path(current_path).resolve()
    seen = set() if seen is None else set(seen)
    require(path not in seen, "Cyclic geometry ancestry")
    seen.add(path)
    if path not in cache:
        cache[path] = (artifact(path), read_sealed(path))
    record, value = cache[path]
    if record == expected:
        return value
    for key in ANCESTOR_FIELDS:
        if value.get(key):
            found = _ancestor(verified(value[key]), expected, cache, depth + 1, seen)
            if found is not None:
                return found
    return None


def _direction_proof(geometry_path, geometry, row, cache):
    """Replay only existing independent semantic verdicts; no Qwen import/call."""
    proof_path = verified(row["direction_review"])
    proof = read_sealed(proof_path)
    require(proof["schema"] in {ARROW_SCHEMA, PART_SCHEMA}, "Unsupported direction evidence schema")
    require(proof["scene_id"] == geometry["scene_id"] and proof["instance_id"] == row["instance_id"], "Direction target identity drift")
    require(all(proof.get(k) is False for k in ("human_query_read", "stage2_generated_person_read", "qwen_received_numeric_geometry")),
            "Direction evidence is not scene-only")
    ancestor = _ancestor(geometry_path, proof["source_geometry"], cache)
    require(ancestor is not None, "Direction geometry is not an exact ancestor")
    source_shape = proof["source_shape"]
    for key in ("instance_id", "category", "stable_surface_sha256", "surface_arrays"):
        require(source_shape[key] == row[key], "Direction source surface drift: " + key)
    require(next((o for o in ancestor["objects"] if o["instance_id"] == row["instance_id"]), None) == source_shape,
            "Direction source shape is not its recorded ancestor object")
    require(proof["source_snapshot"] == ancestor["source_semantics"], "Direction semantic snapshot drift")
    judgments, evidence = proof["judgments"], proof["evidence"]
    require(len(judgments) == 3 and len({j["source_view_index"] for j in judgments}) == 3, "Need three independent direction cameras")
    require({j["view_id"] for j in judgments} == {"V0", "V1", "V2"}, "Direction view IDs drift")
    require(len(evidence) == 3 and {e["view_index"] for e in evidence} == {j["source_view_index"] for j in judgments},
            "Direction camera evidence is incomplete")
    by_view = {e["view_index"]: e for e in evidence}
    calls, fronts = [], []
    is_part = proof["schema"] == PART_SCHEMA
    checks = {"pink_part_is_backrest", "green_part_is_seat", "parts_belong_to_same_target", "evidence_sufficient"}
    for judgment in judgments:
        response, parsed = judgment["generation"], judgment["parsed"]
        call_path = Path(response["call_receipt"]).resolve(strict=True)
        call = read_json(call_path)
        require(call.get("status") == "complete" and call.get("result") == response
                and json.loads(response["raw_completion"]) == parsed, "Direction differs from actual Qwen response")
        require(response.get("image_count") == 2 and "qwen" in str(response.get("model_id", "")).lower(),
                "Direction does not bind a two-image Qwen call")
        score = parsed["confidence"]
        require(type(score) in (int, float) and math.isfinite(score) and .85 <= score <= 1, "Insufficient direction confidence")
        require(isinstance(parsed["reason"], str) and len(parsed["reason"]) <= 180, "Invalid direction reason")
        if is_part:
            require(set(parsed) == checks | {"confidence", "reason"} and all(parsed[k] is True for k in checks), "Unresolved semantic backrest parts")
            fronts.append("A")
        else:
            require(set(parsed) == {"backrest_side_arrow", "seating_front_arrow", "evidence_sufficient", "confidence", "reason"},
                    "Unexpected numeric or semantic direction fields")
            back, front = parsed["backrest_side_arrow"], parsed["seating_front_arrow"]
            require(parsed["evidence_sufficient"] is True and
                    ((front in ("A", "B") and back in ("A", "B") and front != back) or (front == "both" and back == "none")),
                    "Unresolved direction semantics")
            fronts.append(front)
        ev = by_view[judgment["source_view_index"]]
        required = ("source_rgb", "source_camera", "original_crop", "part_visualization" if is_part else "arrow_visualization")
        for key in required:
            verified(ev[key])
        images = [{k: r[k] for k in ("path", "bytes", "sha256")} for r in call["image_evidence"]]
        require(images == [ev["original_crop"], ev[required[-1]]], "Actual Qwen images differ from direction evidence")
        calls.append(artifact(call_path))
    require(len({r["path"] for r in calls}) == 3 and len(set(fronts)) == 1, "Repeated calls or disagreeing directions")
    selected = ["A", "B"] if fronts[0] == "both" else [fronts[0]]
    expected_decision = {"direction_review_accepted": True, "selected_direction_ids": selected,
                         "object_class_changed": False, "coordinate_or_angle_prediction_requested": False}
    require(proof["decision"] == expected_decision, "Stored direction decision cannot be reproduced")
    hypotheses = row["direction_hypotheses"]
    require(hypotheses.get("visual_review") == row["direction_review"]
            and hypotheses.get("category_override_allowed") is False
            and hypotheses.get("direction_coordinates_computed_by_qwen") is False, "Direction authority drift")
    candidates = hypotheses["candidates"]
    require(len(candidates) == len(selected) and len({r["candidate_id"] for r in candidates}) == len(selected)
            and {r["source_review_candidate_id"] for r in candidates} == set(selected), "Direction hypotheses lost or duplicated")
    require(hypotheses.get("orientation_resolved") is (len(candidates) == 1), "Direction ambiguity status drift")
    front_evidence = row.get("front_evidence", {})
    if not is_part and "B" in selected and front_evidence.get("high_support_vertex_count", 0) >= 100 \
            and (front_evidence.get("backrest_offset_m") or 0) >= .10 \
            and front_evidence.get("prior_geometric_hypothesis", {}).get("method") == "opposite_high_target_support_backrest_centroid":
        raise ValueError("Opposed geometric backrest requires semantic part confirmation")
    for candidate in candidates:
        actual = np.asarray(candidate["outward_world_xy"], dtype=float)
        expected = np.asarray(proof["candidate_vectors_computed_by_geometry"][candidate["source_review_candidate_id"]], dtype=float)
        require(actual.shape == (2,) and expected.shape == (2,) and np.isfinite(actual).all()
                and abs(np.linalg.norm(actual) - 1) <= 1e-6 and np.allclose(actual, expected, atol=1e-10, rtol=0),
                "Audited geometry vector drift")
    return {"direction_review": artifact(proof_path), "actual_calls": calls,
            "geometry_ancestor": proof["source_geometry"], "selected_review_ids": selected,
            "all_direction_candidates_retained": True}


def _source_rows(value, label):
    rows = value["objects"]
    require(isinstance(rows, list) and all(isinstance(r.get("instance_id"), str) and r["instance_id"] for r in rows), "Invalid " + label + " rows")
    require(len({r["instance_id"] for r in rows}) == len(rows), "Duplicate " + label + " instance IDs")
    return {r["instance_id"]: r for r in rows}


def build_scene_facts(final_scene_path, output_dir):
    started = time.monotonic()
    final_scene_path, output_dir = Path(final_scene_path).resolve(strict=True), Path(output_dir).resolve()
    require(not output_dir.exists(), "Scene facts output must be a new directory")
    final = read_sealed(final_scene_path)
    require(final.get("schema") == FINAL_SCHEMA and final.get("human_instruction_read") is False
            and final.get("stage2_or_stage3_output_read") is False and final.get("semantic_backrest_review_complete") is True,
            "Use completed instruction-independent backrest-validated Stage1A")
    geometry_path = verified(final["fixed_geometry"])
    geometry = read_sealed(geometry_path)
    require(geometry.get("schema") == "p550.fixed_scene_interaction_geometry.v2"
            and geometry["scene_id"] == final["scene_id"]
            and all(geometry.get(k) is False for k in ("task_instruction_read", "start_state_read", "future_motion_read", "numeric_geometry_computed_by_qwen")),
            "Invalid fixed geometry identity or scene-only boundary")
    require(final["fixed_semantics"] == geometry["source_semantics"], "Final and geometry semantic snapshots differ")
    semantics = read_sealed(verified(geometry["source_semantics"]))
    require(semantics["scene_id"] == geometry["scene_id"], "Semantic scene mismatch")
    shapes, semantic_rows = _source_rows(geometry, "geometry"), _source_rows(semantics, "semantic")
    require(set(shapes) == set(semantic_rows), "Fixed geometry/semantic object inventory mismatch")
    usable = final["sit_usable_ids"]
    require(isinstance(usable, list) and len(set(usable)) == len(usable) and set(usable) <= set(shapes), "Invalid final sit ID inventory")
    sources = {"final_scene": artifact(final_scene_path), "fixed_geometry": artifact(geometry_path), "fixed_semantics": final["fixed_semantics"]}
    for field in ("source_mesh", "source_support", "source_occupancy", "navigation_fields", "source_sam"):
        verified(geometry[field])
        sources[field] = geometry[field]
    vertices, faces = load_obj_world_zup(geometry["source_mesh"]["path"])
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, ancestor_cache = [], {}
    with np.load(geometry["source_support"]["path"], allow_pickle=False) as supports:
        for key, shape in shapes.items():
            sem = semantic_rows[key]
            decision = sem.get("decision", {})
            semantic = sem.get("semantic", {})
            reasons = []
            if shape.get("status") != "surface_extracted": reasons.append("surface_not_extracted")
            if shape.get("semantic_query_usable") is not True or decision.get("query_usable") is not True: reasons.append("semantic_not_query_usable")
            if decision.get("category_confirmed") is not True: reasons.append("category_unconfirmed")
            if shape.get("query_usable_for_sit") is not True or key not in usable: reasons.append("not_published_for_sit")
            if shape.get("category") != semantic.get("object_class") or shape.get("category") not in SIT_CLASSES: reasons.append("unsupported_or_mismatched_sit_category")
            row = {"instance_id": key, "category": shape.get("category"), "source_geometry_status": shape.get("status"),
                   "semantic": {"object_class": semantic.get("object_class"), "mask_state": semantic.get("mask_state"),
                                "visual_description": semantic.get("visual_description"), "decision": decision},
                   "source_eligibility": {"semantic_query_usable": shape.get("semantic_query_usable"), "query_usable_for_sit": shape.get("query_usable_for_sit"), "final_sit_id_member": key in usable},
                   "surface": shape.get("surface"), "stable_surface_sha256": shape.get("stable_surface_sha256"),
                   "surface_arrays": shape.get("surface_arrays"), "direction_hypotheses": shape.get("direction_hypotheses"),
                   "generic_face_candidates": shape.get("generic_face_candidates", []),
                   "generic_faces_archive": shape.get("generic_faces_archive"),
                   "generic_faces_are_action_positive_samples": False, "descriptors": [],
                   "evidence_kind": "scene_geometry_only", "positive_credit": 0}
            if not reasons:
                publishing = False
                try:
                    proof = _direction_proof(geometry_path, geometry, shape, ancestor_cache)
                    surface_path = verified(shape["surface_arrays"])
                    with np.load(surface_path, allow_pickle=False) as archive:
                        require(set(archive.files) == {"face_ids", "vertex_ids", "support_ids", "target_bounds"}, "Surface archive keys drift")
                        face_ids = _indices(archive["face_ids"], len(faces), "surface face", unique=True)
                        vertex_ids = _indices(archive["vertex_ids"], len(vertices), "surface vertex", unique=True)
                        support_ids = _indices(archive["support_ids"], len(vertices), "surface support", unique=True)
                        require(np.array_equal(np.unique(faces[face_ids]), np.sort(vertex_ids)), "Surface vertex/triangle incidence drift")
                        support_key = sem["geometry_source"]["support_vertices_file_key"]
                        require(np.array_equal(support_ids, np.unique(supports[support_key])), "Surface support differs from fixed instance support")
                        bounds = np.asarray(archive["target_bounds"])
                        require(bounds.shape == (2, 3) and np.isfinite(bounds).all() and np.all(bounds[1] >= bounds[0]), "Invalid target bounds")
                    products = []
                    for candidate in shape["direction_hypotheses"]["candidates"]:
                        metadata, arrays = describe_surface(vertices, faces, face_ids,
                            shape["surface"]["centre_world_xyz_zup_m"], candidate["outward_world_xy"], size=32)
                        identity = {"fixed_geometry": sources["fixed_geometry"], "instance_id": key,
                                    "stable_surface_sha256": shape["stable_surface_sha256"], "direction": candidate}
                        entry_id = "SURFACE_" + digest(identity)[:24].upper()
                        metadata.update(entry_id=entry_id, scene_id=geometry["scene_id"], instance_id=key,
                            action_role="sit_support", source_binding=identity, direction_proof=proof,
                            source_surface_arrays=shape["surface_arrays"], source_mesh=geometry["source_mesh"],
                            source_code=artifact(Path(__file__).with_name("surface_geometry.py")))
                        products.append((metadata, arrays))
                    publishing = True
                    for metadata, arrays in products:
                        folder = output_dir / "descriptors" / metadata["entry_id"]
                        folder.mkdir(parents=True, exist_ok=False)
                        raster_path = folder / "raster.npz"
                        with raster_path.open("xb") as stream:
                            np.savez_compressed(stream, **arrays)
                        metadata["raster"] = artifact(raster_path)
                        path = folder / "receipt.json"
                        write_once(path, metadata, seal=True)
                        row["descriptors"].append(artifact(path))
                    row["direction_proof"] = proof
                except (ValueError, KeyError, OSError, TypeError) as error:
                    # Missing evidence never upgrades an unknown. IO errors during
                    # publication abort the scene rather than publish partial rows.
                    if publishing:
                        raise
                    reasons.append("descriptor_evidence_rejected: " + type(error).__name__ + ": " + str(error))
            row.update(descriptor_eligible=not reasons, descriptor_rejection_reasons=reasons,
                       status="eligible_scene_descriptor" if not reasons else "unknown_or_ineligible_retained")
            rows.append(row)
    for record in sources.values():
        verified(record)
    result = {"schema": SCHEMA, "scene_id": geometry["scene_id"], "sources": sources,
              "navigation_config": geometry["navigation_config"], "objects": rows,
              "object_count": len(rows), "eligible_object_count": sum(r["descriptor_eligible"] for r in rows),
              "descriptor_count": sum(len(r["descriptors"]) for r in rows),
              "all_unknown_objects_retained": True, "task_instruction_read": False, "pose_or_motion_read": False,
              "model_inference_performed": False, "evidence_kind": "scene_geometry_only", "positive_credit": 0,
              "scene_facts_are_not_interaction_experiences": True, "elapsed_seconds": time.monotonic() - started,
              "source_codes": [artifact(__file__), artifact(Path(__file__).with_name("surface_geometry.py"))]}
    path = output_dir / "receipt.json"
    write_once(path, result, seal=True)
    return path


def load_scene_facts(receipt_path):
    """Validate stored artifact identities; does not re-run models or rasterize."""
    value = read_sealed(receipt_path)
    require(value.get("schema") == SCHEMA and value.get("evidence_kind") == "scene_geometry_only"
            and type(value.get("positive_credit")) is int and value["positive_credit"] == 0, "Invalid scene facts schema or credit")
    require(value.get("pose_or_motion_read") is False and value.get("model_inference_performed") is False
            and value.get("scene_facts_are_not_interaction_experiences") is True, "Facts crossed the experience boundary")
    for record in value["sources"].values():
        verified(record)
    geometry_path = Path(value["sources"]["fixed_geometry"]["path"])
    geometry = read_sealed(geometry_path)
    source_rows = _source_rows(geometry, "source geometry")
    final = read_sealed(value["sources"]["final_scene"]["path"])
    semantic_rows = _source_rows(read_sealed(value["sources"]["fixed_semantics"]["path"]), "source semantics")
    require(value["scene_id"] == geometry["scene_id"] == final["scene_id"]
            and final["fixed_geometry"] == value["sources"]["fixed_geometry"]
            and final["fixed_semantics"] == value["sources"]["fixed_semantics"], "Fact scene lineage drift")
    rows = _source_rows(value, "facts")
    require(value["object_count"] == len(rows) and set(rows) == set(source_rows) == set(semantic_rows), "Facts inventory count drift")
    ancestry_cache = {}
    for row in rows.values():
        source_row = source_rows[row["instance_id"]]
        for field in ("category", "surface", "stable_surface_sha256", "surface_arrays", "direction_hypotheses", "generic_faces_archive"):
            require(row[field] == source_row.get(field), "Fact object differs from fixed geometry: " + field)
        require(row["generic_face_candidates"] == source_row.get("generic_face_candidates", [])
                and row["source_geometry_status"] == source_row.get("status"), "Fact geometry state drift")
        require(type(row.get("positive_credit")) is int and row["positive_credit"] == 0
                and row.get("evidence_kind") == "scene_geometry_only" and row.get("generic_faces_are_action_positive_samples") is False,
                "Object geometry cannot create positive credit")
        require(type(row.get("descriptor_eligible")) is bool and bool(row["descriptors"]) == row["descriptor_eligible"]
                and bool(row["descriptor_rejection_reasons"]) != row["descriptor_eligible"], "Descriptor eligibility contradiction")
        expected_directions = {}
        if row["descriptor_eligible"]:
            semantic = semantic_rows[row["instance_id"]]
            require(source_row.get("status") == "surface_extracted" and source_row.get("semantic_query_usable") is True
                    and source_row.get("query_usable_for_sit") is True and row["instance_id"] in final["sit_usable_ids"]
                    and semantic.get("decision", {}).get("category_confirmed") is True
                    and semantic.get("decision", {}).get("query_usable") is True
                    and source_row["category"] == semantic.get("semantic", {}).get("object_class")
                    and source_row["category"] in SIT_CLASSES, "Stored descriptor promoted an ineligible source")
            proof = _direction_proof(geometry_path, geometry, source_row, ancestry_cache)
            require(row["direction_proof"] == proof, "Stored direction proof drift")
            expected_directions = {d["candidate_id"]: d for d in source_row["direction_hypotheses"]["candidates"]}
            require(len(row["descriptors"]) == len(expected_directions), "Stored descriptor lost an audited direction")
        seen_directions = set()
        for record in row["descriptors"]:
            descriptor = read_sealed(verified(record))
            require(descriptor["scene_id"] == value["scene_id"] and descriptor["instance_id"] == row["instance_id"]
                    and descriptor["source_binding"]["fixed_geometry"] == value["sources"]["fixed_geometry"]
                    and descriptor["source_binding"]["stable_surface_sha256"] == row["stable_surface_sha256"], "Descriptor source binding drift")
            direction = descriptor["source_binding"]["direction"]
            direction_id = direction["candidate_id"]
            require(direction_id not in seen_directions and expected_directions.get(direction_id) == direction,
                    "Duplicate or unaudited stored descriptor direction")
            seen_directions.add(direction_id)
            require(descriptor["entry_id"] == "SURFACE_" + digest(descriptor["source_binding"])[:24].upper()
                    and descriptor["source_binding"]["instance_id"] == row["instance_id"]
                    and descriptor["source_surface_arrays"] == row["surface_arrays"]
                    and descriptor["source_mesh"] == value["sources"]["source_mesh"]
                    and descriptor["direction_proof"] == row["direction_proof"]
                    and descriptor["action_role"] == "sit_support", "Descriptor identity or evidence drift")
            verified(descriptor["source_surface_arrays"])
            from surface_geometry import local_frame
            require(descriptor["frame"] == local_frame(source_row["surface"]["centre_world_xyz_zup_m"], direction["outward_world_xy"]),
                    "Descriptor local frame differs from fixed geometry")
            with np.load(verified(descriptor["raster"]), allow_pickle=False) as archive:
                validate_descriptor(descriptor, {k: archive[k] for k in archive.files})
    require(value["eligible_object_count"] == sum(r["descriptor_eligible"] for r in rows.values())
            and value["descriptor_count"] == sum(len(r["descriptors"]) for r in rows.values()), "Facts descriptor count drift")
    return value


def load_descriptors(receipt_path, eligible_only=True):
    """Return (object row, descriptor JSON, NPZ arrays) per audited direction.

    Ineligible objects never carry descriptors, even when eligible_only=False;
    use load_scene_facts(...)["objects"] to inspect their explicit rejection.
    """
    result = []
    for row in load_scene_facts(receipt_path)["objects"]:
        if eligible_only and not row["descriptor_eligible"]:
            continue
        for record in row["descriptors"]:
            metadata = read_sealed(verified(record))
            with np.load(verified(metadata["raster"]), allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            result.append((row, metadata, arrays))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_scene_facts(args.final_scene, args.output))
