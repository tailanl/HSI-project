"""Synthetic contracts, not claims of real Qwen accuracy."""
import copy
import json
import pytest
from hsi.common.artifacts import write_once
from hsi.stage1.scene_build.direction_contract import decision


def records(tmp_path, values=None):
    result = []
    for index in range(3):
        value = {"backrest_side_arrow": "B", "seating_front_arrow": "A",
            "evidence_sufficient": True, "confidence": .95, "reason": "Backrest is on B."}
        if values:
            value.update(values[index])
        path = tmp_path / ("call_"+str(index)+".json")
        response = {"raw_completion": json.dumps(value), "call_receipt": str(path)}
        write_once(path, {"status": "complete", "result": response})
        result.append({"view_id": "V"+str(index), "source_view_index": index+10,
            "parsed": value, "generation": response})
    return result


def test_three_independent_agreements_select_only_ids(tmp_path):
    value = decision(records(tmp_path))
    assert value["direction_review_accepted"]
    assert value["selected_direction_ids"] == ["A"]
    assert not value["object_class_changed"]
    assert not value["coordinate_or_angle_prediction_requested"]


def test_disagreement_is_not_accepted_by_majority(tmp_path):
    rows = records(tmp_path, [{}, {}, {"backrest_side_arrow": "A", "seating_front_arrow": "B"}])
    assert not decision(rows)["direction_review_accepted"]


def test_backless_seat_preserves_both_candidates(tmp_path):
    rows = records(tmp_path, [{"backrest_side_arrow": "none", "seating_front_arrow": "both"}]*3)
    assert decision(rows)["selected_direction_ids"] == ["A", "B"]


def test_same_side_is_not_front_and_back(tmp_path):
    assert not decision(records(tmp_path, [{"backrest_side_arrow": "A"}]*3))["direction_review_accepted"]


@pytest.mark.parametrize("field", ["source_view_index", "generation"])
def test_duplicate_evidence_cannot_claim_independent_calls(tmp_path, field):
    rows = records(tmp_path)
    rows[1][field] = copy.deepcopy(rows[0][field])
    with pytest.raises(ValueError, match="Repeated"):
        decision(rows)


@pytest.mark.parametrize("changes", [
    {"confidence": True}, {"confidence": float("nan")}, {"seating_front_arrow": "invented"},
    {"world_xy": [1,2]}, {"reason": 17},
])
def test_invalid_or_numeric_output_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        decision(records(tmp_path, [changes, {}, {}]))


def test_modified_verdict_does_not_match_actual_call(tmp_path):
    rows = records(tmp_path)
    rows[0]["parsed"]["confidence"] = .88
    with pytest.raises(ValueError, match="actual Qwen execution"):
        decision(rows)


def test_fewer_views_not_renamed_to_three(tmp_path):
    with pytest.raises(ValueError, match="three"):
        decision(records(tmp_path)[:2])
