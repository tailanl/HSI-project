"""Synthetic sealed Stage1A/call fixtures only; never invokes a model."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from hsi.common.artifacts import artifact, read_sealed, write_once, digest
from hsi.memory import facts


def fixture_scene(tmp_path, *, directions=1, broken=None, part=False):
    root = tmp_path / "source"
    root.mkdir()
    mesh = root / "mesh.obj"
    mesh.write_text("v 0 .4 0\nv 1 .4 0\nv 1 .4 -1\nv 0 .4 -1\nf 1 2 3\nf 1 3 4\n")
    support = root / "support.npz"
    np.savez_compressed(support, SEAT=np.arange(4))
    surface = root / "surface.npz"
    np.savez_compressed(surface, face_ids=np.array([0, 1]), vertex_ids=np.arange(4), support_ids=np.arange(4),
                        target_bounds=np.array([[0, 0, .4], [1, 1, .4]]))
    generic = root / "generic.npz"
    np.savez_compressed(generic, FACE_1=np.array([0, 1]))
    nav, occupancy = root / "nav.npz", root / "occupancy.npy"
    np.savez_compressed(nav, obstacles=np.zeros((3, 4), bool), free=np.ones((3, 4), bool), clearance=np.ones((3, 4)))
    np.save(occupancy, np.zeros((3, 2, 4), bool))
    sam = root / "sam.json"
    write_once(sam, {"schema": "synthetic.sam.v1"})
    semantic_row = {"instance_id": "SEAT", "semantic": {"object_class": "chair", "mask_state": "whole_object", "visual_description": "chair"},
                    "decision": {"category_confirmed": True, "query_usable": True},
                    "geometry_source": {"support_vertices_file_key": "SEAT"}}
    unknown_sem = {"instance_id": "UNKNOWN", "semantic": {"object_class": "chair", "mask_state": "unclear"},
                   "decision": {"category_confirmed": False, "query_usable": False}}
    semantics = root / "semantics.json"
    if broken == "semantic_unknown": semantic_row["decision"]["query_usable"] = False
    write_once(semantics, {"scene_id": "TEST", "objects": [semantic_row, unknown_sem]})
    shape = {"instance_id": "SEAT", "category": "chair", "status": "surface_extracted",
             "semantic_query_usable": True, "query_usable_for_sit": True,
             "surface": {"centre_world_xyz_zup_m": [.5, .5, .4]},
             "stable_surface_sha256": "a" * 64, "surface_arrays": artifact(surface),
             "generic_face_candidates": [{"surface_candidate_id": "FACE_1", "semantic_interaction_role": "unassigned_geometric_candidate"}],
             "generic_faces_archive": artifact(generic),
             "direction_hypotheses": {"orientation_resolved": False, "candidates": []}}
    unknown = {"instance_id": "UNKNOWN", "category": "chair", "status": "surface_unresolved",
               "semantic_query_usable": False, "query_usable_for_sit": False}
    base = {"schema": "p550.fixed_scene_interaction_geometry.v2", "scene_id": "TEST",
            "task_instruction_read": False, "start_state_read": False, "future_motion_read": False,
            "numeric_geometry_computed_by_qwen": False, "source_semantics": artifact(semantics),
            "source_mesh": artifact(mesh), "source_support": artifact(support), "source_sam": artifact(sam),
            "source_occupancy": artifact(occupancy), "navigation_fields": artifact(nav),
            "navigation_config": {"voxel_size_m": .02}, "objects": [copy.deepcopy(shape), unknown]}
    ancestor_path = root / "ancestor.json"
    write_once(ancestor_path, base)
    evidence, judgments = [], []
    for i in range(3):
        image = root / f"view_{i}_original.png"; image.write_bytes(b"synthetic-original-" + str(i).encode())
        marked = root / f"view_{i}_marked.png"; marked.write_bytes(b"synthetic-marked-" + str(i).encode())
        camera = root / f"camera_{i}.json"; write_once(camera, {"synthetic": i})
        evidence.append({"view_index": i, "source_rgb": artifact(image), "source_camera": artifact(camera),
                         "original_crop": artifact(image), "part_visualization" if part else "arrow_visualization": artifact(marked)})
        if part:
            parsed = {"pink_part_is_backrest": True, "green_part_is_seat": True,
                      "parts_belong_to_same_target": True, "evidence_sufficient": True, "confidence": .95, "reason": "visible"}
        else:
            parsed = {"backrest_side_arrow": "none" if directions == 2 else "B",
                      "seating_front_arrow": "both" if directions == 2 else "A",
                      "evidence_sufficient": True, "confidence": .95, "reason": "visible"}
        if broken == "low_confidence" and i == 0: parsed["confidence"] = .5
        if broken == "numeric_response" and i == 0: parsed["x"] = 1
        call_path = root / f"call_{i}.json"
        response = {"call_receipt": str(call_path.resolve()), "raw_completion": json.dumps(parsed),
                    "image_count": 2, "model_id": "Qwen/synthetic-test-only"}
        image_evidence = [{**artifact(image), "label": "IMAGE_0"}, {**artifact(marked), "label": "IMAGE_1"}]
        if broken == "wrong_image" and i == 0: image_evidence.reverse()
        write_once(call_path, {"status": "complete", "result": response, "image_evidence": image_evidence}, seal=False)
        judgments.append({"view_id": "V" + str(i), "source_view_index": i, "generation": response, "parsed": parsed})
    if broken == "duplicate_view": judgments[1]["source_view_index"] = 0
    if broken == "duplicate_call": judgments[1]["generation"] = judgments[0]["generation"]
    if broken == "raw_result_drift": judgments[0]["parsed"]["confidence"] = .96
    selected = ["A", "B"] if directions == 2 else ["A"]
    proof = {"schema": facts.PART_SCHEMA if part else facts.ARROW_SCHEMA, "scene_id": "TEST", "instance_id": "SEAT",
             "source_geometry": artifact(ancestor_path), "source_shape": copy.deepcopy(shape), "source_snapshot": artifact(semantics),
             "human_query_read": False, "stage2_generated_person_read": False, "qwen_received_numeric_geometry": False,
             "judgments": judgments, "evidence": evidence,
             "decision": {"direction_review_accepted": True, "selected_direction_ids": selected,
                          "object_class_changed": False, "coordinate_or_angle_prediction_requested": False},
             "candidate_vectors_computed_by_geometry": {"A": [1., 0.], "B": [-1., 0.]}}
    if broken == "wrong_scene": proof["scene_id"] = "OTHER"
    if broken == "wrong_surface": proof["source_shape"]["stable_surface_sha256"] = "b" * 64
    if broken == "query_read": proof["human_query_read"] = True
    if broken == "unaccepted": proof["decision"]["direction_review_accepted"] = False
    proof_path = root / "direction.json"
    write_once(proof_path, proof)
    shape["direction_review"] = artifact(proof_path)
    shape["direction_hypotheses"] = {"orientation_resolved": directions == 1, "visual_review": artifact(proof_path),
        "category_override_allowed": False, "direction_coordinates_computed_by_qwen": False,
        "candidates": [{"candidate_id": "DIRECTION_" + str(i), "source_review_candidate_id": letter,
                        "outward_world_xy": proof["candidate_vectors_computed_by_geometry"][letter]} for i, letter in enumerate(selected)]}
    if broken == "vector_drift": shape["direction_hypotheses"]["candidates"][0]["outward_world_xy"] = [0., 1.]
    if broken == "missing_proof": shape.pop("direction_review")
    if broken == "lost_direction": shape["direction_hypotheses"]["candidates"].pop()
    if broken == "index_drift":
        np.savez_compressed(surface, face_ids=np.array([99]), vertex_ids=np.arange(4), support_ids=np.arange(4), target_bounds=np.array([[0, 0, .4], [1, 1, .4]]))
    geometry = root / "geometry.json"
    base.update(objects=[shape, unknown], source_geometry_before_direction_review=artifact(ancestor_path))
    if broken == "missing_ancestor": base.pop("source_geometry_before_direction_review")
    write_once(geometry, base)
    final = root / "receipt.json"
    write_once(final, {"schema": facts.FINAL_SCHEMA, "scene_id": "TEST", "human_instruction_read": False,
                      "stage2_or_stage3_output_read": False, "semantic_backrest_review_complete": True,
                      "fixed_geometry": artifact(geometry), "fixed_semantics": artifact(semantics),
                      "sit_usable_ids": [] if broken == "unpublished" else ["SEAT"]})
    return final


def test_complete_scene_keeps_unknown_and_credit_zero(tmp_path):
    final = fixture_scene(tmp_path)
    path = facts.build_scene_facts(final, tmp_path / "facts")
    value = facts.load_scene_facts(path)
    assert value["object_count"] == 2 and value["eligible_object_count"] == 1 and value["descriptor_count"] == 1
    assert value["positive_credit"] == 0 and value["pose_or_motion_read"] is False
    rows = {r["instance_id"]: r for r in value["objects"]}
    assert rows["UNKNOWN"]["status"] == "unknown_or_ineligible_retained"
    assert rows["UNKNOWN"]["descriptors"] == []
    assert rows["SEAT"]["generic_faces_are_action_positive_samples"] is False
    assert rows["SEAT"]["generic_face_candidates"][0]["semantic_interaction_role"] == "unassigned_geometric_candidate"
    row, descriptor, arrays = facts.load_descriptors(path)[0]
    assert descriptor["positive_credit"] == 0 and descriptor["action_role"] == "sit_support"
    assert arrays["heightmap_m"].shape == (32, 32)


def test_all_audited_directions_kept_not_candidate_zero(tmp_path):
    path = facts.build_scene_facts(fixture_scene(tmp_path, directions=2), tmp_path / "facts")
    entries = facts.load_descriptors(path)
    assert len(entries) == 2
    assert {e[1]["source_binding"]["direction"]["source_review_candidate_id"] for e in entries} == {"A", "B"}
    assert len({e[1]["entry_id"] for e in entries}) == 2
    assert entries[0][1]["frame"]["local_to_world_rotation"] != entries[1][1]["frame"]["local_to_world_rotation"]


def test_semantic_backrest_part_proof_supported(tmp_path):
    path = facts.build_scene_facts(fixture_scene(tmp_path, part=True), tmp_path / "facts")
    assert facts.load_scene_facts(path)["descriptor_count"] == 1


@pytest.mark.parametrize("broken", ["semantic_unknown", "unpublished", "low_confidence", "numeric_response", "duplicate_view", "duplicate_call",
    "raw_result_drift", "wrong_scene", "wrong_surface", "query_read", "unaccepted", "vector_drift", "missing_proof", "missing_ancestor", "wrong_image", "index_drift"])
def test_unknown_or_incomplete_evidence_never_gets_descriptor(tmp_path, broken):
    path = facts.build_scene_facts(fixture_scene(tmp_path, broken=broken), tmp_path / "facts")
    value = facts.load_scene_facts(path)
    assert value["object_count"] == 2 and value["eligible_object_count"] == 0 and value["descriptor_count"] == 0
    assert value["objects"][0]["descriptor_rejection_reasons"]
    assert facts.load_descriptors(path, eligible_only=False) == []


def test_incomplete_multiple_direction_evidence_fails_closed(tmp_path):
    path = facts.build_scene_facts(fixture_scene(tmp_path, directions=2, broken="lost_direction"), tmp_path / "facts")
    assert facts.load_scene_facts(path)["descriptor_count"] == 0


def test_reject_existing_output_without_overwrite(tmp_path):
    final = fixture_scene(tmp_path)
    output = tmp_path / "facts"; output.mkdir()
    marker = output / "marker"; marker.write_text("user data")
    with pytest.raises(ValueError, match="new directory"):
        facts.build_scene_facts(final, output)
    assert marker.read_text() == "user data"


def test_loader_rejects_changed_source(tmp_path):
    path = facts.build_scene_facts(fixture_scene(tmp_path), tmp_path / "facts")
    mesh = tmp_path / "source/mesh.obj"; mesh.write_text(mesh.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="binding drift"):
        facts.load_scene_facts(path)


def test_loader_rejects_changed_descriptor_raster(tmp_path):
    path = facts.build_scene_facts(fixture_scene(tmp_path), tmp_path / "facts")
    value = read_sealed(path)
    descriptor = read_sealed(value["objects"][0]["descriptors"][0]["path"])
    raster = Path(descriptor["raster"]["path"])
    raster.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="binding drift"):
        facts.load_scene_facts(path)


def test_no_model_or_pose_imports():
    for name in ("torch", "qwen_api", "hybrik", "comfy", "run_hybrikx_image"):
        assert name not in facts.__dict__


def test_sit_categories_do_not_expand_frozen_executor():
    assert facts.SIT_CLASSES == {"chair", "armchair", "sofa", "bench", "stool"}


def rewrite_synthetic_seal(path, value):
    """Test-owned adversarial rewrite, never used by implementation writers."""
    value = {k: v for k, v in value.items() if k != "receipt_payload_sha256"}
    value["receipt_payload_sha256"] = digest(value)
    Path(path).write_text(json.dumps(value))


@pytest.mark.parametrize("kind", ["credit", "generic_credit", "unknown_omission", "direction_duplication", "source_category"])
def test_loader_rechecks_resealed_fact_contract(tmp_path, kind):
    path = facts.build_scene_facts(fixture_scene(tmp_path), tmp_path / "facts")
    value = read_sealed(path)
    row = value["objects"][0]
    if kind == "credit": row["positive_credit"] = 1
    if kind == "generic_credit": row["generic_faces_are_action_positive_samples"] = True
    if kind == "unknown_omission": value["objects"].pop(); value["object_count"] = 1
    if kind == "direction_duplication": row["descriptors"] *= 2; value["descriptor_count"] = 2
    if kind == "source_category": row["category"] = "bench"
    rewrite_synthetic_seal(path, value)
    with pytest.raises(ValueError):
        facts.load_scene_facts(path)


def test_loader_rejects_resealed_descriptor_frame(tmp_path):
    path = facts.build_scene_facts(fixture_scene(tmp_path), tmp_path / "facts")
    value = read_sealed(path)
    desc_path = value["objects"][0]["descriptors"][0]["path"]
    descriptor = read_sealed(desc_path)
    descriptor["frame"]["origin_world_zup_m"][0] += .1
    rewrite_synthetic_seal(desc_path, descriptor)
    value["objects"][0]["descriptors"][0] = artifact(desc_path)
    rewrite_synthetic_seal(path, value)
    with pytest.raises(ValueError, match="local frame"):
        facts.load_scene_facts(path)


def test_loader_rechecks_actual_direction_calls(tmp_path):
    path = facts.build_scene_facts(fixture_scene(tmp_path), tmp_path / "facts")
    call = tmp_path / "source/call_0.json"
    value = json.loads(call.read_text()); value["status"] = "failed"; call.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="actual Qwen"):
        facts.load_scene_facts(path)


def test_publication_io_failure_does_not_publish_partial_scene(tmp_path, monkeypatch):
    final = fixture_scene(tmp_path)
    def failed_write(*args, **kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(facts.np, "savez_compressed", failed_write)
    with pytest.raises(OSError, match="disk failure"):
        facts.build_scene_facts(final, tmp_path / "facts")
    assert not (tmp_path / "facts/receipt.json").exists()
