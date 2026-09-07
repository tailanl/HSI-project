from ._common import *
from .semantic_contract import SCHEMA, SEATING, agree, semantic_status, size_diagnostic, validate

PROMPT = """Build a reusable scene inventory. No human action query is provided.
The separate images show one anonymous segmented region in several views. CYAN identifies that region; other furniture is context.
Identify the physical object category, independently of whether the mask is complete. A chair fragment can belong to a chair but is NOT a whole-object mask.
mask_state: whole_object if one coherent object's visible extent is covered; partial_object if only a seat/backrest/legs or fragment is covered;
multiple_objects if the region includes parts of different physical objects; uncertain if evidence does not distinguish these.
Do not infer missing seat/backrest inside the mask from nearby unmasked furniture. Report their actual visible presence separately.
is_sittable_object describes the physical object's function, not proof of safe contact or reachable approach.
If views conflict, set views_consistent=false. Give confidence, a brief qualitative visual_description of the physical object (no annotation colors),
and a short mask_issue. Do not output coordinates, dimensions, angles or paths. Return exactly the JSON schema."""


def classify_instance(qwen, instance, output, *, double_check_seating=True):
    key = instance["instance_id"]
    folder = output / "instances" / key
    final = folder / "receipt.json"
    images, seen = [], set()
    for record in instance["classifier_images"]:
        path = verified(record)
        if record["sha256"] not in seen:
            images.append(path); seen.add(record["sha256"])
    if len(images) < 2: raise ValueError("Insufficient distinct source views: "+key)
    source = {"instance_id": key, "images": [artifact(p) for p in images], "source_instance_digest": digest(instance)}
    if final.exists():
        existing = read_sealed(final)
        if existing["source"] != source: raise ValueError("Cannot resume with changed instance evidence")
        return existing
    first_path = folder / "pass_00.json"
    if first_path.exists():
        first = read_sealed(first_path)
        if first["source"] != source: raise ValueError("First-pass source drift")
    else:
        response = qwen.call(images, PROMPT, schema=SCHEMA, max_tokens=260, call_id="p550_instruction_blind_"+key)
        parsed = validate(json.loads(response["raw_completion"]))
        first = write_once(first_path, {"source": source, "parsed": parsed, "generation": response}, seal=True)
    value = first["parsed"]
    decision = semantic_status(value)
    passes = [artifact(first_path)]
    consensus = None
    if double_check_seating and value["object_class"] in SEATING and decision["query_usable"]:
        second_path = folder / "pass_01.json"
        if second_path.exists():
            second = read_sealed(second_path)
            if second["source"] != source: raise ValueError("Second-pass source drift")
        else:
            response = qwen.call(list(reversed(images)), PROMPT, schema=SCHEMA, max_tokens=260,
                call_id="p550_seating_order_check_"+key)
            parsed = validate(json.loads(response["raw_completion"]))
            second = write_once(second_path, {"source": source, "parsed": parsed, "generation": response}, seal=True)
        consensus = agree(value, second["parsed"])
        passes.append(artifact(second_path))
        if not consensus:
            decision.update(category_confirmed=False, query_usable=False, needs_visual_recovery=True)
            decision["reasons"].append("independent_view_order_disagreement")
    return write_once(final, {"schema": "p550.instruction_blind_instance_semantics.v1", "source": source,
        "semantic": value, "decision": decision, "size_diagnostic": size_diagnostic(instance["bounds_world_zup_m"]),
        "independent_seating_order_agreement": consensus, "classification_passes": passes,
        "task_instruction_read": False, "source_code": artifact(__file__)}, seal=True)


def run(sam_path, output, selected_ids=None, *, qwen_client):
    started = time.monotonic()
    sam = read_sealed(sam_path)
    if sam["status"] != "atomic_instances_ready": raise ValueError("SAM source is not ready")
    verified(sam["source_original_scene_mesh"])
    verified(sam["support_vertices"])
    render_path = verified(sam["source_render_receipt"])
    render = read_sealed(render_path)
    if render["scene_id"] != sam["scene_id"]: raise ValueError("Scene/render identity drift")
    sources = {"atomic_instances": artifact(sam_path), "render": artifact(render_path),
        "mesh": sam["source_original_scene_mesh"], "support_vertices": sam["support_vertices"]}
    output.mkdir(parents=True, exist_ok=True)
    launch = output / "launch.json"
    if launch.exists():
        if read_sealed(launch)["sources"] != sources: raise ValueError("Scene sources changed on resume")
    else:
        write_once(launch, {"schema": "p550.scene_understanding_launch.v1", "sources": sources,
            "instruction_used": False, "source": artifact(__file__), "started_epoch": time.time()}, seal=True)
    call_root = output / "qwen_calls"
    qwen = qwen_client.session(call_root, max_tokens=260)
    existing_calls = list(call_root.glob("call_*.json"))
    qwen.count = max((int(p.stem.split("_")[-1]) for p in existing_calls), default=0)
    rows, errors = [], []
    for instance in sam["instances"]:
        if selected_ids and instance["instance_id"] not in selected_ids: continue
        try:
            result = classify_instance(qwen, instance, output)
            rows.append({"instance_id": instance["instance_id"], "semantic": result["semantic"], "decision": result["decision"],
                "evidence": artifact(output / "instances" / instance["instance_id"] / "receipt.json"),
                "geometry_source": instance, "size_diagnostic": result["size_diagnostic"]})
        except Exception as error:
            errors.append({"instance_id": instance["instance_id"], "error": f"{type(error).__name__}: {error}"})
            print(json.dumps(errors[-1]), flush=True)
    snapshot_name = "pilot_snapshot.json" if selected_ids else "receipt.json"
    result = {"schema": "p550.fixed_instruction_independent_scene_understanding.v1", "scene_id": sam["scene_id"],
        "status": "partial_pilot_not_full_scene" if selected_ids else "scene_understanding_complete_with_unknowns" if errors or any(r["decision"]["needs_visual_recovery"] for r in rows) else "scene_understanding_complete",
        "sources": sources, "objects": rows, "errors": errors, "source_instance_count": len(sam["instances"]),
        "processed_instance_count": len(rows), "task_instruction_read": False, "future_motion_read": False,
        "all_instances_attempted": not selected_ids, "unknown_instances_do_not_invalidate_other_objects": True,
        "model": qwen.lineage, "runtime": qwen.runtime, "elapsed_seconds": time.monotonic()-started,
        "source": artifact(__file__), "geometry_snapshot_pending": True}
    write_once(output / snapshot_name, result, seal=True)
    print({"scene": sam["scene_id"], "status": result["status"], "objects": len(rows), "errors": len(errors)}, flush=True)
    return result
