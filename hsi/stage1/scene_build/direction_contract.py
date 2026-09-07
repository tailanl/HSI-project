from ._common import *

def original_decision(judgments):
    if len(judgments) != 3 or {r["view_id"] for r in judgments} != {"V0", "V1", "V2"}:
        raise ValueError("Require three separately executed camera reviews")
    fronts, good = [], True
    for item in judgments:
        response, value = item["generation"], item["parsed"]
        actual = read(response["call_receipt"])
        if actual["status"] != "complete" or actual["result"] != response or json.loads(response["raw_completion"]) != value:
            raise ValueError("Single-view direction result does not match actual Qwen execution")
        back, front = value["backrest_side_arrow"], value["seating_front_arrow"]
        if back not in ("A", "B", "none", "unclear") or front not in ("A", "B", "both", "unclear"):
            raise ValueError("Invented direction ID")
        score = value["confidence"]
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1 \
                or type(value["evidence_sufficient"]) is not bool:
            raise ValueError("Invalid direction confidence or evidence flag")
        good = good and value["evidence_sufficient"] and score >= .85 \
            and ((front in ("A", "B") and back in ("A", "B") and front != back) or (front == "both" and back == "none"))
        fronts.append(front)
    accepted = good and len(set(fronts)) == 1
    return {"direction_review_accepted": bool(accepted), "selected_direction_ids":
        (["A", "B"] if fronts[0] == "both" else [fronts[0]]) if accepted else [],
        "object_class_changed": False, "coordinate_or_angle_prediction_requested": False}

FIELDS = {"backrest_side_arrow", "seating_front_arrow", "evidence_sufficient", "confidence", "reason"}


def decision(judgments):
    if len(judgments) != 3:
        raise ValueError("Need three independent direction calls")
    if len({r["generation"]["call_receipt"] for r in judgments}) != 3:
        raise ValueError("Repeated Qwen call cannot stand for independent evidence")
    if len({r["source_view_index"] for r in judgments}) != 3:
        raise ValueError("Repeated source camera cannot stand for independent evidence")
    for item in judgments:
        parsed = item["parsed"]
        if set(parsed) != FIELDS or not isinstance(parsed["reason"], str) or len(parsed["reason"]) > 180:
            raise ValueError("Unexpected direction fields, numeric predictions, or explanation")
    return original_decision(judgments)
