"""Current scene-only semantic direction proof and numeric facing recheck."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hsi.stage1.binding import bind
from hsi.stage1.scene_build.direction_contract import decision as arrow_decision
from hsi.stage1.scene_build.backrest import decision as part_decision

def ancestor_matches(path,record,depth=0):
    if depth>8:
        return False
    if artifact(path)==record:
        return True
    value=read_sealed(path)
    return any(ancestor_matches(verified(value[key]),record,depth+1) for key in (
        "source_geometry_before_direction_review","source_geometry_before_final_publication",
        "source_geometry_before_semantic_backrest_review") if value.get(key))

def validate_direction(stage1_path,review_path):
    stage1,review=read_sealed(stage1_path),read_sealed(review_path)
    geometry_path=verified(stage1["source_fixed_geometry"])
    geometry=read_sealed(geometry_path)
    approach=bind(stage1)
    key=approach["target_instance_id"]
    row=next(r for r in geometry["objects"] if r["instance_id"]==key)
    if key in conflicts(geometry):
        raise ValueError("Opposed geometric backrest and arrow require semantic part confirmation")
    if row.get("direction_review")!=artifact(review_path) or not ancestor_matches(geometry_path,review["source_geometry"]):
        raise ValueError("Direction proof does not match the fixed target and its geometry ancestry")
    if any(review[k] is not False for k in ("human_query_read","stage2_generated_person_read","qwen_received_numeric_geometry")):
        raise ValueError("Direction evidence is not scene-only semantics")
    if review["scene_id"]!=stage1["scene_id"] or review["instance_id"]!=key \
            or row["stable_surface_sha256"]!=review["source_shape"]["stable_surface_sha256"] \
            or row["category"]!=review["source_shape"]["category"]:
        raise ValueError("Different target, category or support surface")
    is_part=review["schema"]=="p550.scene_only_semantic_backrest_direction.v1"
    decision=part_decision(review["judgments"]) if is_part else arrow_decision(review["judgments"])
    if decision!=review["decision"] or decision["direction_review_accepted"] is not True:
        raise ValueError("Direction semantic proof is unresolved")
    yaw=approach["contact_forward_yaw_rad"]
    errors=[]
    for candidate in decision["selected_direction_ids"]:
        vector=review["candidate_vectors_computed_by_geometry"][candidate]
        expected=math.atan2(vector[1],vector[0])
        errors.append(abs(math.atan2(math.sin(yaw-expected),math.cos(yaw-expected))))
    if not errors or min(errors)>1e-8:
        raise ValueError("Planned body facing differs from verified geometry")
    return {"schema":"p550.semantic_part_or_arrow_direction_preflight.v2","source":artifact(__file__),
        "source_stage1":artifact(stage1_path),"source_direction_review":artifact(review_path),
        "numeric_direction_match_verified":True,"semantic_backrest_part_protocol":is_part,
        "object_category_changed":False,"Qwen_coordinate_prediction":False}

def conflicts(geometry):
    result=[]
    for row in geometry["objects"]:
        front=row.get("front_evidence",{})
        candidate=row.get("direction_hypotheses",{}).get("candidates",[{}])[0]
        if row.get("query_usable_for_sit") and candidate.get("source_review_candidate_id")=="B" \
                and front.get("high_support_vertex_count",0)>=100 and (front.get("backrest_offset_m") or 0)>=.10 \
                and front.get("prior_geometric_hypothesis",{}).get("method")=="opposite_high_target_support_backrest_centroid":
            result.append(row["instance_id"])
    return result
