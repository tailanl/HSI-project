"""Synthetic transport/checkpoint tests; these do not claim live Qwen accuracy."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hsi.common.artifacts import artifact, read_sealed, write_once
from hsi.stage1.qwen import QwenTextClient
from hsi.stage1.scene_build import semantics
from hsi.stage1.scene_build.backrest import CHECKS, decision as backrest_decision
from hsi.stage1.scene_build.checkpoints import phase
from hsi.stage1.scene_build.report import write_report
from hsi.stage1.scene_build.semantic_revision import review_decision
from hsi.stage1.scene_build.semantic_contract import agree, semantic_status
from hsi.stage1 import build_scene, load_final_scene


def semantic():
    return {"object_class": "table", "mask_state": "whole_object", "is_sittable_object": False,
        "visible_seat_in_mask": False, "visible_backrest_in_mask": False, "views_consistent": True,
        "confidence": .95, "visual_description": "A table.", "mask_issue": "none"}


def test_vision_transport_retains_actual_record_and_no_coordinates(tmp_path, monkeypatch):
    request = []
    def fake_json(self, suffix, payload=None):
        request.append((suffix, payload))
        if suffix == "/models":
            return {"data": [{"id": "explicit-test-qwen"}]}
        return {"id": "synthetic-response", "model": "explicit-test-qwen",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(semantic())}}]}
    monkeypatch.setattr(QwenTextClient, "_json", fake_json)
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    client = QwenTextClient("http://localhost:1234/v1", "explicit-test-qwen")
    session = client.session(tmp_path / "calls")
    response = session.call([image], "Identify the object; no task.")
    record = read_sealed(response["call_receipt"])
    assert record["status"] == "complete" and record["result"] == response
    assert record["model"]["weight_lineage_locally_verified"] is False
    assert record["image_evidence"][0]["sha256"] == artifact(image)["sha256"]
    assert request[-1][1]["messages"][1]["content"][-1]["text"] == "Identify the object; no task."
    assert record["runtime"]["numeric_geometry_in_prompt"] is False
    assert record["runtime"]["local_queue_seconds"] == 0


def test_vision_transport_model_drift_fails_and_seals_attempt(tmp_path, monkeypatch):
    def fake_json(self, suffix, payload=None):
        if suffix == "/models":
            return {"data": [{"id": "qwen-test"}]}
        return {"id": "synthetic", "model": "other-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
    monkeypatch.setattr(QwenTextClient, "_json", fake_json)
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    session = QwenTextClient("http://localhost:1234/v1", "qwen-test").session(tmp_path / "calls")
    with pytest.raises(ValueError, match="Model drift"):
        session.call([image], "Classify")
    assert read_sealed(tmp_path / "calls/call_0001.json")["status"] == "failed"


def test_immutable_checkpoint_resumes_only_complete_receipt(tmp_path):
    calls = []
    def operation(out):
        calls.append(out)
        write_once(out / "receipt.json", {"scene_id": "s"})
    first = phase(tmp_path, "classify", operation)
    assert phase(tmp_path, "classify", operation) == first
    assert len(calls) == 1
    assert read_sealed(tmp_path / "checkpoints/classify.json")["new_execution_claim"] is True


def test_failed_phase_does_not_create_success_checkpoint(tmp_path):
    def fail(out):
        raise ValueError("missing real evidence")
    with pytest.raises(ValueError, match="missing real evidence"):
        phase(tmp_path, "classify", fail)
    assert not (tmp_path / "checkpoints/classify.json").exists()
    assert read_sealed(tmp_path / "steps/classify/attempt_001/execution_001.json")["status"] == "failed"


def test_context_disagreement_remains_unknown_and_cannot_be_forced(tmp_path):
    first, second = semantic(), dict(semantic(), mask_state="partial_object")
    judgments = []
    for index, value in enumerate((first, second)):
        path = tmp_path / f"call_{index}.json"
        generation = {"call_receipt": str(path), "raw_completion": json.dumps(value)}
        write_once(path, {"status": "complete", "result": generation})
        judgments.append({"generation": generation, "parsed": value})
    status = semantic_status(first)
    status.update(category_confirmed=False, query_usable=False, needs_visual_recovery=True)
    status["reasons"].append("independent_original_context_review_disagreement")
    review = {"judgments": judgments, "order_agreement": agree(first, second), "review_decision": status}
    assert review_decision(review)[1]["query_usable"] is False
    review["review_decision"]["query_usable"] = True
    with pytest.raises(ValueError, match="reproduce"):
        review_decision(review)


def test_backrest_needs_three_actual_agreeing_part_calls(tmp_path):
    rows = []
    for index in range(3):
        value = {**{key: True for key in CHECKS}, "confidence": .95, "reason": "Visible parts."}
        path = tmp_path / f"call_{index}.json"
        generation = {"call_receipt": str(path), "raw_completion": json.dumps(value)}
        write_once(path, {"status": "complete", "result": generation})
        rows.append({"source_view_index": index, "parsed": value, "generation": generation})
    assert backrest_decision(rows)["selected_direction_ids"] == ["A"]
    assert backrest_decision(rows)["coordinate_or_angle_prediction_requested"] is False
    rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="three different"):
        backrest_decision(rows)


def test_semantic_failure_is_local_not_fabricated_as_unknown_answer(tmp_path):
    images = []
    for index, color in enumerate(("red", "blue")):
        image = tmp_path / f"view{index}.png"
        Image.new("RGB", (8, 8), color).save(image)
        images.append(artifact(image))
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("v 0 0 0\n")
    support = tmp_path / "support.npz"
    np.savez_compressed(support, a=np.array([0]), b=np.array([0]))
    render = tmp_path / "render.json"
    write_once(render, {"scene_id": "s"})
    sam = tmp_path / "sam.json"
    write_once(sam, {"scene_id": "s", "status": "atomic_instances_ready",
        "source_original_scene_mesh": artifact(mesh), "source_render_receipt": artifact(render),
        "support_vertices": artifact(support), "instances": [
            {"instance_id": key, "classifier_images": images, "bounds_world_zup_m": [[0,0,0],[1,1,1]]}
            for key in ("a", "b")]})
    class Session:
        lineage, runtime = {"synthetic_test": True}, {"synthetic_test": True}
        def call(self, images, prompt, **kwargs):
            if kwargs["call_id"].endswith("_a"):
                raise ValueError("synthetic model failure")
            return {"raw_completion": json.dumps(semantic())}
    class Client:
        def session(self, output, **kwargs):
            return Session()
    result = semantics.run(sam, tmp_path / "understand", qwen_client=Client())
    assert result["all_instances_attempted"] is True
    assert result["errors"][0]["instance_id"] == "a"
    assert [r["instance_id"] for r in result["objects"]] == ["b"]
    assert result["status"] == "scene_understanding_complete_with_unknowns"


def test_scene_report_is_png_and_json_not_html(tmp_path):
    source, geo = tmp_path / "semantic.json", tmp_path / "geo.json"
    obj = {"instance_id": "s", "semantic": semantic(), "decision": semantic_status(semantic()),
        "geometry_source": {"classifier_images": []}}
    write_once(source, {"scene_id": "s", "objects": [obj]})
    write_once(geo, {"objects": [{"instance_id": "s", "query_usable_for_sit": False}]})
    path = write_report(tmp_path / "out", source, geo)
    result = read_sealed(path)
    assert result["html_generated"] is False and len(result["pages"]) == 1
    assert not list(tmp_path.rglob("*.html"))


def test_complete_scene_publication_cpu_nonseating_fixture(tmp_path, monkeypatch):
    """All publication steps execute; only the one text completion is synthetic.

    This is deliberately not a live perception/seat/direction accuracy test.
    """
    calls = []
    def fake_json(self, suffix, payload=None):
        if suffix == "/models":
            return {"data": [{"id": "qwen-cpu-fixture"}]}
        calls.append(payload)
        return {"id": "synthetic-cpu-fixture", "model": "qwen-cpu-fixture",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(semantic())}}]}
    monkeypatch.setattr(QwenTextClient, "_json", fake_json)
    mesh = tmp_path / "s/mesh.obj"
    mesh.parent.mkdir()
    mesh.write_text("v 0 0.5 0\nv 1 0.5 0\nv 1 0.5 -1\nv 0 0.5 -1\nf 1 2 3\nf 1 3 4\n")
    occupancy = tmp_path / "s.npy"
    np.save(occupancy, np.zeros((30, 100, 30), bool))
    images = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"view_{index}.png"
        Image.new("RGB", (8, 8), color).save(path)
        images.append(artifact(path))
    render = tmp_path / "render.json"
    write_once(render, {"scene_id": "s", "views": [], "inputs": {
        "original_scene_mesh": artifact(mesh), "query_occupancy": artifact(occupancy)}})
    support = tmp_path / "support.npz"
    np.savez_compressed(support, a=np.arange(4))
    sam = tmp_path / "sam.json"
    write_once(sam, {"schema": "p515.lingo_sam31_atomic_multiview_instances.v1",
        "scene_id": "s", "status": "atomic_instances_ready", "instruction": "",
        "source_original_scene_mesh": artifact(mesh), "source_render_receipt": artifact(render),
        "support_vertices": artifact(support), "view_inference": [],
        "query_contract": {"instruction_available_to_segmentation_or_fusion": False,
            "motion_pose_contact_keypose_planner_memory_or_hsi_read": False},
        "instances": [{"instance_id": "a", "classifier_images": images,
            "bounds_world_zup_m": [[0,0,.5],[1,1,.5]], "support_vertices_file_key": "a"}]})
    client = QwenTextClient("http://localhost:1234/v1", "qwen-cpu-fixture")
    final_path = build_scene("s", mesh, occupancy, tmp_path / "build",
        qwen_client=client, atomic_receipt=sam)
    fixed = load_final_scene(final_path)
    assert fixed.publication["semantic_backrest_review_complete"] is True
    assert fixed.publication["sit_usable_ids"] == []
    assert fixed.semantics["objects"][0]["semantic"]["object_class"] == "table"
    assert len(fixed.geometry["objects"][0]["generic_face_candidates"]) == 1
    assert len(calls) == 1
    assert build_scene("s", mesh, occupancy, tmp_path / "build", qwen_client=client,
                       atomic_receipt=sam) == final_path
    assert len(calls) == 1  # Verified FINAL reuse performs no second model call.
    assert not list((tmp_path / "build").rglob("*.html"))
    assert not list((tmp_path / "build").rglob("*navmesh*"))
