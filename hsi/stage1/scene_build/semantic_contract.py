"""Category/size/direction independent semantic contract, unchanged current algorithm."""
import math

CLASSES = ("chair", "armchair", "stool", "sofa", "bench", "bed", "table", "desk", "cabinet", "shelf",
    "door", "window", "floor", "wall", "plant", "lamp", "tv", "musical_instrument", "other", "unknown")


MASK_STATES = ("whole_object", "partial_object", "multiple_objects", "uncertain")


SEATING = {"chair", "armchair", "stool", "sofa", "bench", "bed"}


PROPS = {"object_class": {"type": "string", "enum": list(CLASSES)},
    "mask_state": {"type": "string", "enum": list(MASK_STATES)},
    "is_sittable_object": {"type": "boolean"}, "visible_seat_in_mask": {"type": "boolean"},
    "visible_backrest_in_mask": {"type": "boolean"}, "views_consistent": {"type": "boolean"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "visual_description": {"type": "string", "maxLength": 240},
    "mask_issue": {"type": "string", "maxLength": 160}}


SCHEMA = {"type": "object", "properties": PROPS, "required": list(PROPS), "additionalProperties": False}


def validate(value):
    if not isinstance(value, dict) or set(value) != set(PROPS): raise ValueError("Wrong scene understanding fields")
    if value["object_class"] not in CLASSES or value["mask_state"] not in MASK_STATES:
        raise ValueError("Unknown category or mask state")
    for key in ("is_sittable_object", "visible_seat_in_mask", "visible_backrest_in_mask", "views_consistent"):
        if type(value[key]) is not bool: raise ValueError("Non-boolean "+key)
    score = value["confidence"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid semantic confidence")
    for key in ("visual_description", "mask_issue"):
        if not isinstance(value[key], str) or len(value[key]) > PROPS[key]["maxLength"]:
            raise ValueError("Invalid description")
    return value


def semantic_status(value):
    validate(value)
    reasons = []
    if value["confidence"] < .82: reasons.append("low_confidence")
    if not value["views_consistent"]: reasons.append("view_disagreement")
    if value["object_class"] == "unknown": reasons.append("unknown_category")
    if value["mask_state"] in {"multiple_objects", "uncertain"}: reasons.append("mask_identity_uncertain")
    return {"category": value["object_class"], "category_confirmed": not reasons,
        "query_usable": not reasons and value["mask_state"] == "whole_object",
        "needs_visual_recovery": bool(reasons) or value["mask_state"] == "partial_object",
        "reasons": reasons, "category_not_changed_by_size_or_direction": True}


def agree(first, second):
    """A second view check cannot silently overwrite the first judgement."""
    validate(first); validate(second)
    return all(first[key] == second[key] for key in ("object_class", "mask_state", "is_sittable_object")) \
        and first["views_consistent"] and second["views_consistent"] and min(first["confidence"], second["confidence"]) >= .82


def size_diagnostic(bounds):
    """Preserve observed dimensions as a review hint, never a class veto."""
    extents = [float(bounds[1][i])-float(bounds[0][i]) for i in range(3)]
    if any(not math.isfinite(x) or x < 0 for x in extents): raise ValueError("Invalid instance bounds")
    major, minor = sorted(extents[:2], reverse=True)
    return {"aabb_extents_m": extents, "compact_chair_profile_anomaly": major > 1.25 or minor > 1.15 or major*minor > 1.30,
        "category_override_allowed": False, "diagnostic_only": True, "orientation_dependent_aabb_not_physical_object_size": True}


def direction_hypotheses(front, method):
    """An unoriented PCA axis remains two hypotheses until visual evidence resolves it."""
    x, y = map(float, front)
    length = math.hypot(x, y)
    if not math.isfinite(length) or length < 1e-8: raise ValueError("Invalid direction axis")
    axis = [x/length, y/length]
    two_sided = method == "surface_minor_axis_two_sided_no_backrest_evidence"
    vectors = [axis, [-axis[0], -axis[1]]] if two_sided else [axis]
    return {"orientation_resolved": not two_sided, "method": method,
        "candidates": [{"candidate_id": "DIRECTION_"+str(i), "outward_world_xy": v} for i, v in enumerate(vectors)],
        "category_override_allowed": False, "apply_single_front_45_degree_gate": not two_sided}
