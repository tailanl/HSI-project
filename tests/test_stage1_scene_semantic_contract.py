import math
import pytest
from hsi.stage1.scene_build.semantic_contract import direction_hypotheses, semantic_status, size_diagnostic


def example():
    return {"object_class": "armchair", "mask_state": "whole_object", "is_sittable_object": True,
        "visible_seat_in_mask": True, "visible_backrest_in_mask": True, "views_consistent": True,
        "confidence": .95, "visual_description": "An upholstered armchair.", "mask_issue": "none"}


def test_large_aabb_does_not_change_class_or_reject_category():
    value = example()
    diagnostic = size_diagnostic([[0, 0, 0], [1.675, 1.174, .795]])
    assert diagnostic["compact_chair_profile_anomaly"]
    assert not diagnostic["category_override_allowed"]
    assert semantic_status(value)["category"] == "armchair"
    assert semantic_status(value)["query_usable"]


def test_axis_sign_flip_preserves_same_two_hypotheses():
    method = "surface_minor_axis_two_sided_no_backrest_evidence"
    left = direction_hypotheses([1, 0], method)
    right = direction_hypotheses([-1, 0], method)
    assert not left["apply_single_front_45_degree_gate"]
    assert not left["orientation_resolved"]
    assert {tuple(c["outward_world_xy"]) for c in left["candidates"]} == {tuple(c["outward_world_xy"]) for c in right["candidates"]}


def test_partial_chair_is_still_chair_but_needs_recovery():
    value = dict(example(), mask_state="partial_object")
    status = semantic_status(value)
    assert status["category"] == "armchair" and status["category_confirmed"]
    assert status["needs_visual_recovery"] and not status["query_usable"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, -1, 1.1])
def test_invalid_confidence_rejected(score):
    with pytest.raises(ValueError): semantic_status(dict(example(), confidence=score))


@pytest.mark.parametrize("vector", [[0, 0], [math.inf, 0], [math.nan, 1]])
def test_invalid_axis_rejected(vector):
    with pytest.raises(ValueError): direction_hypotheses(vector, "surface_minor_axis_two_sided_no_backrest_evidence")
