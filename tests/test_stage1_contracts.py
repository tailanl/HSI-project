import copy
import json
from pathlib import Path
import numpy as np
import pytest

from hsi.common.artifacts import artifact, read_sealed, write_once
from hsi.stage1.scene import load_final_scene, build_scene
from hsi.stage1.planning import compile_supported_sequence, inventory, validate_plan
from hsi.stage1.qwen import QwenTextClient
from hsi.stage1 import _navmesh


def final_scene(tmp_path):
    semantic = tmp_path / "semantics.json"
    write_once(semantic, {"scene_id": "test", "objects": []})
    geometry = tmp_path / "geometry.json"
    write_once(geometry, {"scene_id": "test", "source_semantics": artifact(semantic),
        "task_instruction_read": False, "start_state_read": False, "objects": []})
    publication = tmp_path / "final.json"
    payload = {"schema": "p550.final_fixed_scene_understanding.v1", "scene_id": "test",
        "human_instruction_read": False, "semantic_backrest_review_complete": True,
        "fixed_semantics": artifact(semantic), "fixed_geometry": artifact(geometry)}
    write_once(publication, payload)
    return publication, payload


def test_final_snapshot_and_backrest_contract(tmp_path):
    path, payload = final_scene(tmp_path)
    fixed = load_final_scene(path)
    assert fixed.geometry["scene_id"] == "test"
    payload["semantic_backrest_review_complete"] = False
    bad = tmp_path / "incomplete.json"
    write_once(bad, payload)
    with pytest.raises(ValueError, match="final instruction-independent"):
        load_final_scene(bad)


def test_scene_drift_is_not_silently_rebuilt(tmp_path):
    path, payload = final_scene(tmp_path)
    Path(payload["fixed_geometry"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="drift"):
        load_final_scene(path)


def plan_step(action="sit", target="CHAIR_1"):
    return {"action": action, "target_ids": [target], "reference_ids": [],
        "intent_description": "Sit on the armchair", "target_selection_reason": "Verified armchair"}


ROWS = [{"instance_id": "CHAIR_1", "query_usable": True, "sit_surface_available": True}]


def test_navigation_prefix_preserves_all_requested_actions():
    plan = {"steps": [plan_step("walk"), plan_step("sit")], "needs_clarification": False, "reason": ""}
    compiled, mapping, unsupported = compile_supported_sequence(plan, ROWS)
    assert len(compiled["steps"]) == 1
    assert mapping["execution_compilation"][0]["source_step_indices"] == [0, 1]
    assert not unsupported


def test_unsupported_actions_and_unknowns_are_not_changed_into_sit():
    plan = {"steps": [plan_step("lie")], "needs_clarification": False, "reason": ""}
    compiled, _, unsupported = compile_supported_sequence(plan, ROWS)
    assert compiled["steps"][0]["action"] == "lie" and unsupported
    bad = {"steps": [plan_step(target="INVENTED")], "needs_clarification": False, "reason": ""}
    with pytest.raises(ValueError, match="invented"):
        validate_plan(bad, ROWS)
    unknown = [dict(ROWS[0], query_usable=False)]
    with pytest.raises(ValueError, match="unresolved"):
        validate_plan({"steps": [plan_step()], "needs_clarification": False, "reason": ""}, unknown)


def test_qwen_model_mismatch_fails_and_records_no_secret(tmp_path, monkeypatch):
    client = QwenTextClient("http://127.0.0.1:8158/v1", "explicit-model", api_key_env="TEST_UNUSED_SECRET")
    def fake(self, suffix, payload=None):
        if suffix == "/models":
            return {"data": [{"id": "explicit-model"}]}
        return {"model": "wrong-model", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
    monkeypatch.setattr(QwenTextClient, "_json", fake)
    output = tmp_path / "call.json"
    with pytest.raises(ValueError, match="drift"):
        client.call("Select supplied IDs only", {"type": "object"}, output)
    receipt = read_sealed(output)
    assert receipt["status"] == "failed"
    assert "Authorization" not in json.dumps(receipt)


def test_real_external_navmesh_cold_construction_is_deterministic():
    pytest.importorskip("pathfinder")
    module, identity = _navmesh.load_pynavmesh()
    free = np.ones((7, 7), bool)
    free[3, :5] = False
    vertices, polygons = _navmesh.build_cell_polygon_navmesh(free, np.zeros(2), .1)
    args = (module, vertices, polygons, np.array([.15, .15]), np.array([.55, .15]))
    first = _navmesh._pathfinder_route(*args)
    second = _navmesh._pathfinder_route(*args)
    assert identity.distribution == "pynavmesh"
    assert np.array_equal(first, second)
    np.testing.assert_array_equal(first, np.array([
        [.15, .15], [.2, .4], [.30000000000000004, .5], [.4, .5], [.55, .15]]))
