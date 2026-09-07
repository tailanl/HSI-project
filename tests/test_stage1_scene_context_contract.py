"""Small fail-closed contracts for complete-view, member and recovery boundaries."""
import numpy as np
import pytest
from hsi.common.artifacts import artifact,write_once
from hsi.stage1.scene_build.context_review import touches_border, crop_bounds, member_groups
from hsi.stage1.scene_build.context_review import insufficient_views
from hsi.stage1.scene_build.context_review import review_projection as mesh_review


def test_border_contact_is_not_an_untruncated_view():
    mask = np.zeros((30, 40), bool)
    mask[1:4, 10:20] = True
    assert touches_border(mask)
    mask[1] = False
    assert not touches_border(mask)
    mask[-1, 15] = True
    assert touches_border(mask)


def test_crop_preserves_context_and_stays_in_actual_image():
    mask = np.zeros((120, 140), bool)
    mask[40:61, 50:81] = True
    assert crop_bounds(mask) == [15, 5, 116, 96]
    mask[:10, :10] = True
    assert crop_bounds(mask)[0:2] == [0, 0]
    with pytest.raises(ValueError, match="Empty"):
        crop_bounds(np.zeros((10, 10), bool))


def test_composed_object_requires_observation_of_every_member():
    sam = {"instances": [{"instance_id": "a", "atomic_source_observation_ids": ["a1", "a2"]},
        {"instance_id": "b", "atomic_source_observation_ids": ["b1", "b2"]}]}
    row = {"geometry_source": {"logical_object_composition": {"source_atom_ids": ["a", "b"]}}}
    groups = member_groups(sam, row)
    assert not all(group & {"a1", "a2"} for group in groups)
    assert all(group & {"a1", "b2"} for group in groups)


def test_recovery_only_for_two_view_shortage_and_same_source(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    write_once(snapshot, {"scene_id": "s"}, seal=True)
    for count in (0, 1, 2, 3):
        selection = tmp_path / f"selection{count}.json"
        write_once(selection, {"source_snapshot": artifact(snapshot), "instance_id": "a",
            "untruncated_all_member_view_count": count}, seal=True)
        assert insufficient_views(selection, snapshot, "a") == (count < 2)
        with pytest.raises(ValueError, match="another"):
            insufficient_views(selection, snapshot, "b")


@pytest.mark.parametrize("flag", ("task_instruction_read", "human_start_or_future_motion_read", "stage2_outputs_read"))
def test_projected_semantic_evidence_rejects_task_leakage(tmp_path, flag):
    snapshot = tmp_path / "snapshot.json"
    write_once(snapshot, {"scene_id": "s"}, seal=True)
    value = {"source_snapshot": artifact(snapshot), "member_ids": ["a"],
        "task_instruction_read": False, "human_start_or_future_motion_read": False, "stage2_outputs_read": False}
    value[flag] = True
    views = tmp_path / "views.json"
    write_once(views, value, seal=True)
    with pytest.raises(ValueError, match="scene-only boundary"):
        mesh_review(snapshot, views, "a", tmp_path / "output", qwen_client=None)


def test_projected_review_cannot_be_claimed_as_new_segmentation(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    write_once(snapshot, {"scene_id": "s"}, seal=True)
    views = tmp_path / "views.json"
    write_once(views, {"source_snapshot": artifact(snapshot), "member_ids": ["a"],
        "task_instruction_read": False, "human_start_or_future_motion_read": False,
        "stage2_outputs_read": False, "new_sam_inference_performed": True}, seal=True)
    with pytest.raises(ValueError, match="explicitly projected"):
        mesh_review(snapshot, views, "a", tmp_path / "output", qwen_client=None)
