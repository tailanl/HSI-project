from ._common import *
from .semantic_contract import validate,semantic_status,agree

def review_decision(review):
    judgments = review["judgments"]
    if len(judgments) != 2:
        raise ValueError("Require two actual original-context review orders")
    for judgment in judgments:
        value = validate(judgment["parsed"])
        response = judgment["generation"]
        call = read(response["call_receipt"])
        if call["status"] != "complete" or call["result"] != response:
            raise ValueError("Independent review lacks matching actual Qwen call")
        if json.loads(response["raw_completion"]) != value:
            raise ValueError("Parsed review differs from actual model completion")
    first, second = [j["parsed"] for j in judgments]
    decision = semantic_status(first)
    agreement = agree(first, second)
    if not agreement:
        decision.update(category_confirmed=False, query_usable=False, needs_visual_recovery=True)
        decision["reasons"].append("independent_original_context_review_disagreement")
    if review["order_agreement"] != agreement or review["review_decision"] != decision:
        raise ValueError("Review decision does not reproduce from actual judgments")
    return copy.deepcopy(first), decision


def run(snapshot_path, reviews, output):
    snapshot = read_sealed(snapshot_path)
    result = copy.deepcopy(snapshot)
    lookup = {r["instance_id"]: r for r in result["objects"]}
    used, changes = set(), []
    for path in reviews:
        review = read_sealed(path)
        source = read_sealed(verified(review["source_snapshot"]))
        key = review["instance_id"]
        if key in used or review["scene_id"] != snapshot["scene_id"] or source["sources"] != snapshot["sources"]:
            raise ValueError("Duplicate or wrong-scene review")
        if any(review[k] is not False for k in ("human_task_instruction_read", "qwen_received_numeric_geometry", "original_label_shown_to_qwen")):
            raise ValueError("Review violates instruction-independent visual boundary")
        row = lookup[key]
        if row != review["source_object"]:
            raise ValueError("Object changed since independent visual review")
        semantic, decision = review_decision(review)
        previous = {"semantic": row["semantic"], "decision": row["decision"], "evidence": row["evidence"]}
        row.update(semantic=semantic, decision=decision, evidence=artifact(path),
            prior_semantic_evidence=previous, semantic_revision_reason="actual_original_rgb_context_review")
        used.add(key)
        changes.append({"instance_id": key, "old_class": previous["semantic"]["object_class"],
            "new_class_hypothesis": semantic["object_class"], "new_class_confirmed": decision["category_confirmed"],
            "query_usable": decision["query_usable"], "review": artifact(path)})
    output.mkdir(parents=True, exist_ok=False)
    result.update(source_snapshot_before_context_revision=artifact(snapshot_path),
        context_revisions=changes, source_code=artifact(__file__),
        status="scene_understanding_complete_with_unknowns" if result["errors"] or any(r["decision"]["needs_visual_recovery"] for r in result["objects"]) else "scene_understanding_complete",
        task_instruction_read=False, geometry_snapshot_pending=True,
        old_query_plans_automatically_revalidated=False,
        revision_authority="independent_actual_qwen_rgb_context_not_geometry_or_direction")
    write_once(output / "receipt.json", result, seal=True)
    print({"scene": result["scene_id"], "revisions": changes}, flush=True)
