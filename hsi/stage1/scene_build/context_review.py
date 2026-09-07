from ._common import *
from .semantic_contract import SCHEMA,validate,agree,semantic_status

PROMPT = """Independently understand one anonymous region of a scene; no action request and no prior object label are provided.
Images occur in pairs: an UNMODIFIED original RGB view, then the SAME view with the actual segmented region lightly tinted cyan and outlined.
Use original texture, structure and surrounding context to identify its physical function. Do not assume a horizontal surface is a seat: a table, display fixture, cabinet, stacked goods, cushion, shelf and a bench are different objects. Do not assume a fragment belongs to the nearest chair.
First identify the physical object category. Independently judge whether the mask covers one whole visible object, only a part, several objects, or is uncertain. Visible_seat/backrest means visibly contained in the mask, not an imagined missing part. is_sittable_object means designed or evidently usable as seating, not merely horizontal.
Check every field against the visual description and mask_issue. Do not report a whole object while describing only its top or an isolated fragment. If evidence is ambiguous, mark it uncertain and lower confidence. Do not infer the desired task. Do not output coordinates, dimensions, angles or paths. Return only the schema with concise descriptions."""
def atomic_source(snapshot):
    sam = read_sealed(verified(snapshot["sources"]["atomic_instances"]))
    seen = set()
    while sam.get("schema") == "p550.visual_logical_scene_instances.v1":
        record = sam["source_atomic_instances"]
        if record["sha256"] in seen:
            raise ValueError("Cyclic atomic provenance")
        seen.add(record["sha256"])
        sam = read_sealed(verified(record))
    if sam.get("status") != "atomic_instances_ready":
        raise ValueError("No genuine source atomic mask observations")
    return sam
def touches_border(mask):
    return bool(mask[:2].any() or mask[-2:].any() or mask[:,:2].any() or mask[:,-2:].any())


def member_groups(sam, row):
    composition = row["geometry_source"].get("logical_object_composition")
    if not composition:
        return [set(row["geometry_source"]["atomic_source_observation_ids"])]
    lookup = {r["instance_id"]: r for r in sam["instances"]}
    return [set(lookup[key]["atomic_source_observation_ids"]) for key in composition["source_atom_ids"]]


def crop_bounds(mask, padding=.28, minimum_context=35):
    ys,xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty observed mask")
    margin = max(minimum_context, int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)*padding))
    height,width = mask.shape
    return [max(0,int(xs.min())-margin),max(0,int(ys.min())-margin),
        min(width,int(xs.max())+1+margin),min(height,int(ys.max())+1+margin)]


def evidence(snapshot, row, output):
    sam = atomic_source(snapshot)
    render = read_sealed(verified(snapshot["sources"]["render"]))
    if sam["scene_id"] != snapshot["scene_id"] or render["scene_id"] != snapshot["scene_id"]:
        raise ValueError("Scene identity mismatch")
    groups = member_groups(sam, row)
    ids = set.union(*groups)
    candidates, rejected = [], []
    for view in sam["view_inference"]:
        available = {r["observation_id"] for r in view["observations"]}
        if not all(group & available for group in groups):
            continue
        keys = sorted(ids & available)
        archive_path = verified(view["mask_archive"])
        with np.load(archive_path, allow_pickle=False) as archive:
            mask = np.any([archive[key].astype(bool) for key in keys], axis=0)
        if touches_border(mask):
            rejected.append({"view_index": view["view_index"], "reason": "observed_mask_touches_image_border"})
            continue
        candidates.append((int(mask.sum()),view["view_index"],mask,artifact(archive_path),keys))
    candidates.sort(key=lambda r: (-r[0],r[1]))
    write_once(output / "evidence_view_selection.json", {
        "schema": "p550.untruncated_all_member_semantic_views.v3", "source_snapshot": artifact(snapshot["_source_path"]),
        "instance_id": row["instance_id"], "required_member_count": len(groups),
        "untruncated_all_member_view_count": len(candidates), "rejected": rejected,
        "selected_view_indices": [r[1] for r in candidates[:3]], "qwen_computed_view_or_crop_coordinates": False,
        "source": artifact(__file__)}, seal=True)
    if len(candidates) < 2:
        raise ValueError("Need at least two untruncated views containing observations of every object member")
    images,records = [],[]
    for _,index,mask,archive,keys in candidates[:3]:
        view = next(r for r in render["views"] if r["view_index"] == index)
        raw = verified(view["rgb"])
        rgb = np.asarray(Image.open(raw).convert("RGB"))
        if mask.shape != rgb.shape[:2]:
            raise ValueError("Observed mask/RGB dimensions differ")
        crop = crop_bounds(mask)
        x0,y0,x1,y1 = crop
        raw_crop = output / f"view_{index:02d}_original_texture_crop.png"
        Image.fromarray(rgb[y0:y1,x0:x1]).save(raw_crop)
        marked = rgb.copy()
        marked[mask] = (rgb[mask]*.84+np.array([0,235,235])*.16).astype(np.uint8)
        marked[mask & ~binary_erosion(mask,iterations=2)] = [0,255,255]
        canvas = Image.new("RGB",(x1-x0,y1-y0+28),"white")
        canvas.paste(Image.fromarray(marked[y0:y1,x0:x1]),(0,28))
        ImageDraw.Draw(canvas).text((6,7),"ANONYMOUS REGION | actual observed mask",fill="black")
        contour = output / f"view_{index:02d}_context_contour.png"
        canvas.save(contour)
        images.extend([raw_crop,contour])
        records.append({"view_index":index,"unmodified_rgb":artifact(raw),"original_texture_crop":artifact(raw_crop),
            "contour_visualization":artifact(contour),"mask_archive":archive,"source_observation_ids":keys,
            "actual_mask_pixels":int(mask.sum()),"crop_xyxy_computed_by_geometry":crop,
            "all_members_observed":True,"observed_mask_touches_border":False})
    return images,records

CURRENT_PROMPT = PROMPT.replace("UNMODIFIED original RGB view", "original-texture RGB crop") + " All selected regions avoid the source image boundary. Crops are computed by geometry and show the same context in each pair."

def run(snapshot_path, instance_id, output, *, qwen_client, evidence_provider=None, prompt=None):
    started = time.monotonic()
    snapshot = read_sealed(snapshot_path)
    snapshot["_source_path"] = str(Path(snapshot_path).resolve())
    prompt = prompt or CURRENT_PROMPT
    row = next(r for r in snapshot["objects"] if r["instance_id"] == instance_id)
    output.mkdir(parents=True, exist_ok=False)
    images, records = (evidence_provider or evidence)(snapshot, row, output)
    qwen = qwen_client.session(output / "qwen_calls", max_tokens=350)
    judgments = []
    for order in (images, sum([images[i:i+2] for i in range(len(images)-2, -1, -2)], [])):
        queued = time.monotonic()
        with exclusive_qwen():
            wait = time.monotonic() - queued
            response = qwen.call(order, prompt, schema=SCHEMA, max_tokens=350,
                call_id="p550_independent_original_context_review")
        parsed = validate(json.loads(response["raw_completion"]))
        judgments.append({"parsed": parsed, "generation": response, "cooperative_queue_seconds": wait})
    first, second = [r["parsed"] for r in judgments]
    agreement = agree(first, second)
    decision = semantic_status(first)
    if not agreement:
        decision.update(category_confirmed=False, query_usable=False, needs_visual_recovery=True)
        decision["reasons"].append("independent_original_context_review_disagreement")
    result = {"schema": "p550.independent_object_context_review.v1", "scene_id": snapshot["scene_id"],
        "instance_id": instance_id, "source_snapshot": artifact(snapshot_path), "source_object": copy.deepcopy(row),
        "evidence": records, "judgments": judgments, "review_semantic": first, "review_decision": decision,
        "order_agreement": agreement, "category_changed_in_review": first["object_class"] != row["semantic"]["object_class"],
        "original_snapshot_modified": False, "original_label_shown_to_qwen": False,
        "human_task_instruction_read": False, "qwen_received_numeric_geometry": False,
        "elapsed_seconds": time.monotonic()-started, "source_code": artifact(__file__)}
    write_once(output / "receipt.json", result, seal=True)
    print({"scene": result["scene_id"], "id": instance_id, "semantic": first,
        "agreement": agreement, "seconds": result["elapsed_seconds"]}, flush=True)

def review_projection(snapshot_path,views_path,instance_id,output, *, qwen_client):
    views=read_sealed(views_path)
    if views["source_snapshot"]!=artifact(snapshot_path) or views["member_ids"]!=[instance_id]:
        raise ValueError("Scene-only supplemental views have a different source object")
    for flag in ("task_instruction_read","human_start_or_future_motion_read","stage2_outputs_read"):
        if views[flag] is not False:
            raise ValueError("Supplemental evidence crosses the scene-only boundary")
    if views["new_sam_inference_performed"] is not False:
        raise ValueError("This adapter expects explicitly projected SAM-supported mesh masks")
    eligible=[r for r in views["projected_views"] if r["eligible"]]
    eligible.sort(key=lambda r:(-min(m["visible_fraction"] for m in r["members"]),
        -sum(m["visible_pixels"] for m in r["members"]),r["view_index"]))
    if len(eligible)<2:
        raise ValueError("Need two complete unoccluded object projection views")
    def evidence(snapshot,row,out):
        if row!=views["source_member_objects"][0]:
            raise ValueError("Object snapshot differs from rendered support identity")
        images=[]
        for item in eligible[:3]:
            images.extend([verified(item["original_texture_crop"]),verified(item["contour_visualization"])])
        return images,eligible[:3]
    prompt = PROMPT.replace("UNMODIFIED original RGB view","original-texture RGB crop") + " The mask is an explicit depth-tested projection of original mesh faces supported by prior SAM vertices. It is not newly inferred segmentation and missing object regions must not be imagined."
    run(snapshot_path,instance_id,output,qwen_client=qwen_client,evidence_provider=evidence,prompt=prompt)
    write_once(output / "mesh_projection_evidence_extension.json", {
        "schema":"p550.scene_only_supplemental_projection_context_review.v1","source":artifact(__file__),
        "actual_review":artifact(output / "receipt.json"),"source_object_views":artifact(views_path),
        "new_sam_inference_performed":False,"camera_and_mask_coordinates_from_geometry":True,
        "no_human_query_or_stage2_image_consumed":True},seal=True)

def insufficient_views(path, snapshot_path, instance_id):
    value = read_sealed(path)
    if value["source_snapshot"] != artifact(snapshot_path) or value["instance_id"] != instance_id:
        raise ValueError("Failed view evidence belongs to another object")
    count = value["untruncated_all_member_view_count"]
    if type(count) is not int or count < 0:
        raise ValueError("Invalid untruncated view count")
    return count < 2


def recover(snapshot_path, geometry_path, instance_id, output, gpu, *, qwen_client):
    """Recover missing original views once, never retry a negative Qwen verdict."""
    from .checkpoints import phase
    from . import object_views
    from ..scene import verify_artifact_tree
    snapshot, geometry = read_sealed(snapshot_path), read_sealed(geometry_path)
    require(snapshot["scene_id"] == geometry["scene_id"], "Supplemental scene identity drift")
    output.mkdir(parents=True, exist_ok=True)
    final = output / "receipt.json"
    if final.exists():
        result = read_sealed(final)
        require(result["source_snapshot"] == artifact(snapshot_path)
                and result["instance_id"] == instance_id, "Cannot resume another object")
        verify_artifact_tree(result)
        return result
    recovered, failure = False, None
    try:
        review = phase(output, "observed_context", lambda out:
            run(snapshot_path, instance_id, out, qwen_client=qwen_client))
    except Exception as error:
        attempts = sorted((output / "steps/observed_context").glob("attempt_*"))
        selection = attempts[-1] / "output/evidence_view_selection.json" if attempts else None
        if selection is None or not selection.exists() or not insufficient_views(selection, snapshot_path, instance_id):
            raise
        failure = {"error": str(error), "evidence_selection": artifact(selection)}
        recovered = True
        views = phase(output, "scene_only_additional_views", lambda out:
            object_views.run(geometry_path, [instance_id], out, gpu), gpu=gpu)
        review = phase(output, "projected_context", lambda out:
            review_projection(snapshot_path, views, instance_id, out, qwen_client=qwen_client))
    result = read_sealed(review)
    write_once(final, result, seal=True)
    write_once(output / "bounded_scene_recovery_extension.json", {
        "schema": "p550.bounded_scene_only_context_recovery.v1", "source": artifact(__file__),
        "source_geometry": artifact(geometry_path), "actual_review": artifact(review),
        "published_review": artifact(final), "new_views_required": recovered,
        "original_view_failure": failure, "human_query_read": False, "stage2_outputs_read": False,
        "negative_qwen_decisions_retried": False, "thresholds_changed": False}, seal=True)
    return result
