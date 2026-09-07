from ._common import *
from .direction_contract import decision
from ._common import exclusive_qwen

PROPS = {"selected_direction_ids": {"type": "array", "items": {"type": "string", "enum": ["A", "B"]}, "maxItems": 2},
    "evidence_sufficient": {"type": "boolean"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "visible_structure_explanation": {"type": "string", "maxLength": 260}}


SCHEMA = {"type": "object", "additionalProperties": False, "properties": PROPS, "required": list(PROPS)}


def direction_decision(judgments):
    if len(judgments) != 2:
        raise ValueError("Two actual direction reviews required")
    values = []
    for item in judgments:
        value = item["parsed"]
        if set(value) != set(PROPS) or not isinstance(value["selected_direction_ids"], list):
            raise ValueError("Invalid directional schema")
        ids = value["selected_direction_ids"]
        if len(ids) != len(set(ids)) or any(k not in ("A", "B") for k in ids):
            raise ValueError("Qwen invented or duplicated direction IDs")
        score = value["confidence"]
        if type(value["evidence_sufficient"]) is not bool or isinstance(score, bool) \
                or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid directional confidence or evidence status")
        response = item["generation"]
        actual = read(response["call_receipt"])
        if actual["status"] != "complete" or actual["result"] != response or json.loads(response["raw_completion"]) != value:
            raise ValueError("Direction verdict differs from actual Qwen call")
        values.append(value)
    accepted = all(v["evidence_sufficient"] is True and v["confidence"] >= .85 and v["selected_direction_ids"] for v in values) \
        and set(values[0]["selected_direction_ids"]) == set(values[1]["selected_direction_ids"])
    return {"direction_review_accepted": bool(accepted),
        "selected_direction_ids": sorted(values[0]["selected_direction_ids"]) if accepted else [],
        "object_class_changed": False, "coordinate_or_angle_prediction_requested": False}


def project(points, camera):
    points = np.asarray(points)
    xyz = np.c_[points, np.ones(len(points))] @ np.asarray(camera["world_to_camera"]).T
    uv = xyz[:, :3] @ np.asarray(camera["K"]).T
    if np.any(xyz[:, 2] <= .05):
        raise ValueError("Direction arrows project behind camera")
    return uv[:, :2] / uv[:, 2:3]


def draw_arrow(draw, first, last, color, label):
    first, last = np.asarray(first), np.asarray(last)
    delta = last-first
    if np.linalg.norm(delta) < 12:
        raise ValueError("Direction arrows overlap in this camera")
    unit = delta / np.linalg.norm(delta); side = np.array([-unit[1], unit[0]])
    draw.line([tuple(first), tuple(last)], fill=color, width=4)
    draw.polygon([tuple(last), tuple(last-unit*11+side*5), tuple(last-unit*11-side*5)], fill=color)
    position = last + unit*8
    draw.rectangle([position[0]-10, position[1]-12, position[0]+10, position[1]+12], fill="white", outline=color, width=2)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text(tuple(position-[6, 10]), label, fill=color, font=font)


def run(geometry_path, instance_id, output, object_views_path=None, *, qwen_client):
    geometry = read_sealed(geometry_path)
    if geometry["task_instruction_read"] is not False or geometry["start_state_read"] is not False:
        raise ValueError("Direction review must precede human-query geometry")
    snapshot_path = verified(geometry["source_semantics"])
    snapshot = read_sealed(snapshot_path)
    row = next(r for r in snapshot["objects"] if r["instance_id"] == instance_id)
    shape = next(r for r in geometry["objects"] if r["instance_id"] == instance_id)
    if not row["decision"]["query_usable"] or shape["status"] != "surface_extracted":
        raise ValueError("Direction review needs an independently confirmed complete object and extracted surface")
    centre = np.asarray(shape["surface"]["centre_world_xyz_zup_m"])
    forward = np.asarray(shape["direction_hypotheses"]["candidates"][0]["outward_world_xy"])
    candidate_vectors = {"A": forward.tolist(), "B": (-forward).tolist()}
    semantic_review = read_sealed(verified(row["evidence"]))
    if semantic_review.get("schema") != "p550.independent_object_context_review.v1":
        raise ValueError("Use actual independent full-context object review as direction evidence")
    evidence = semantic_review["evidence"]
    if object_views_path:
        views = read_sealed(object_views_path)
        if views["member_ids"] != [instance_id] or views["source_mesh"] != geometry["source_mesh"] \
                or views["source_support"] != geometry["source_support"]:
            raise ValueError("Supplemental direction cameras bind to a different object or support")
        for flag in ("task_instruction_read", "human_start_or_future_motion_read", "stage2_outputs_read"):
            if views[flag] is not False:
                raise ValueError("Supplemental direction views crossed the scene-only boundary")
        camera_views = views["views"]
        evidence = sorted((r for r in views["projected_views"] if r["eligible"]),
            key=lambda r: (-min(m["visible_fraction"] for m in r["members"]), r["view_index"]))
    else:
        camera_views = read_sealed(verified(snapshot["sources"]["render"]))["views"]
    by_index = {r["view_index"]: r for r in camera_views}
    output.mkdir(parents=True, exist_ok=False)
    images, records = [], []
    origin = centre + [0, 0, .04]
    arrows = np.array([origin, origin + [* (forward*.5), 0], origin + [* (-forward*.5), 0]])
    for item in evidence:
        if len(records) == 3:
            break
        index = item["view_index"]
        view = by_index[index]
        camera = read(verified(view["camera"]))
        rgb_path = verified(view["rgb"])
        # Exact source bytes must match the actual semantic evidence, not a same-index other render.
        source_rgb = item.get("unmodified_rgb", item.get("source_rgb"))
        if source_rgb is not None and source_rgb != artifact(rgb_path):
            raise ValueError("Directional evidence camera and semantic RGB differ")
        uv = None
        for length in (.50, .35, .25, .18, .12):
            arrows = np.array([origin, origin + [*(forward*length), 0], origin + [*(-forward*length), 0]])
            projected = project(arrows, camera)
            inside = np.all((projected[:, 0] >= 24) & (projected[:, 0] < camera["width"]-24)
                & (projected[:, 1] >= 24) & (projected[:, 1] < camera["height"]-24))
            separated = all(np.linalg.norm(p-projected[0]) >= 12 for p in projected[1:])
            if inside and separated:
                uv = projected
                break
        if uv is None:
            continue
        crop = item.get("crop_xyxy_computed_by_geometry", item.get("crop_xyxy"))
        if crop is None:
            # Projected-view records store their actual crop as crop_xyxy_computed_by_geometry.
            crop = [0, 0, camera["width"], camera["height"]]
        x0, y0, x1, y1 = map(int, crop)
        # Preserve enough original context for both arrow ends and their labels.
        x0 = max(0, min(x0, int(uv[:, 0].min())-35)); y0 = max(0, min(y0, int(uv[:, 1].min())-35))
        x1 = min(camera["width"], max(x1, int(uv[:, 0].max())+36)); y1 = min(camera["height"], max(y1, int(uv[:, 1].max())+36))
        if not np.all((uv[:, 0] >= x0+20) & (uv[:, 0] < x1-20) & (uv[:, 1] >= y0+20) & (uv[:, 1] < y1-20)):
            continue
        canvas = Image.open(rgb_path).convert("RGB").crop((x0, y0, x1, y1))
        raw_path = output / f"view_{index:02d}_original.png"; canvas.save(raw_path)
        draw = ImageDraw.Draw(canvas)
        points = uv - [x0, y0]
        try:
            draw_arrow(draw, points[0], points[1], "#00758c", "A")
            draw_arrow(draw, points[0], points[2], "#cf4c19", "B")
        except ValueError:
            continue
        annotated = output / f"view_{index:02d}_direction_candidates.png"; canvas.save(annotated)
        images.extend([raw_path, annotated])
        records.append({"view_index": index, "source_rgb": artifact(rgb_path), "source_camera": view["camera"],
            "original_crop": artifact(raw_path), "arrow_visualization": artifact(annotated),
            "arrow_coordinates_computed_by_geometry": True, "visual_arrow_length_m": length})
    if len(records) != 3:
        raise ValueError("Need three actual views with both direction arrows visible")
    props = {"backrest_side_arrow": {"type": "string", "enum": ["A", "B", "none", "unclear"]},
        "seating_front_arrow": {"type": "string", "enum": ["A", "B", "both", "unclear"]},
        "evidence_sufficient": {"type": "boolean"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 180}}
    schema = {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}
    prompt = "These two images show the SAME scene view: original texture, then opposite arrows drawn from the seat of the target " \
        + row["semantic"]["object_class"] + ". " \
        "Identify the actual backrest of that same object. Which arrow points toward the backrest side, and which points AWAY from it through the open front of the seat? " \
        "The open-front arrow is the direction a naturally seated person's knees and toes face. " \
        "Use the visible backrest/seat relation, not the camera position, table position, or another nearby chair. " \
        "For a clearly backless two-sided seat use none/both. If unclear, say so. " \
        "Return candidate letters and a short visual explanation only. Do not calculate coordinates, angles, dimensions or paths. " \
        "No prior direction judgment or human task is provided."
    qwen = qwen_client.session(output / "qwen_calls", max_tokens=240)
    judgments, started = [], time.monotonic()
    for index, item in enumerate(records):
        pair = [verified(item["original_crop"]), verified(item["arrow_visualization"])]
        queued = time.monotonic()
        with exclusive_qwen():
            waiting = time.monotonic()-queued
            response = qwen.call(pair, prompt, schema=schema, max_tokens=240, call_id="p550_independent_single_view_direction")
        judgments.append({"view_id": "V"+str(index), "source_view_index": item["view_index"],
            "parsed": json.loads(response["raw_completion"]), "generation": response, "queue_seconds": waiting})
    receipt = {"schema": "p550.scene_only_independent_single_camera_direction.v3", "scene_id": geometry["scene_id"],
        "instance_id": instance_id, "source_geometry": artifact(geometry_path), "source_snapshot": artifact(snapshot_path),
        "source_object": row, "source_shape": shape, "candidate_vectors_computed_by_geometry": candidate_vectors,
        "source_object_views": artifact(object_views_path) if object_views_path else None,
        "evidence": records, "judgments": judgments, "decision": decision(judgments), "source": artifact(__file__),
        "human_query_read": False, "stage2_generated_person_read": False, "qwen_received_numeric_geometry": False,
        "prior_direction_judgments_shown_to_qwen": False, "all_three_independent_camera_verdicts_must_agree": True,
        "initial_multicamera_qwen_calls_performed": False, "adaptive_visual_arrow_length_not_direction_change": True, "elapsed_seconds_including_queue": time.monotonic()-started}
    write_once(output / "receipt.json", receipt, seal=True)
    print({"scene": geometry["scene_id"], "instance": instance_id, "decision": receipt["decision"]}, flush=True)
