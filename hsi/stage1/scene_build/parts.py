from ._common import *
from scipy.spatial import cKDTree

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "all_regions_same_physical_object": {"type": "boolean"},
    "union_covers_whole_visible_object": {"type": "boolean"},
    "union_contains_other_furniture": {"type": "boolean"},
    "object_class": {"type": "string", "enum": ["chair", "armchair", "stool", "sofa", "bench", "other", "unknown"]},
    "confidence": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["all_regions_same_physical_object", "union_covers_whole_visible_object",
        "union_contains_other_furniture", "object_class", "confidence", "reason"]}


PROMPT = """Scene-only segmentation recovery; no human task is provided. Each pair of images is an unmarked view followed by the SAME view with anonymous regions colored and labelled in the header. The same color identifies the same region across views.
Decide whether ALL colored regions are complementary parts of ONE physical object, or include different objects. Do not merge adjacent chairs just because they touch or have the same category. Inspect the uncolored context too.
Separately decide whether the UNION covers the whole visible object and whether it contains other furniture. Missing seat, base, or backrest must be reported by union_covers_whole_visible_object=false.
Do not output coordinates, dimensions, angles or paths. Return only the required JSON; reason under 140 characters. If identity is ambiguous, confidence must be low."""


def run(snapshot_path, instance_ids, output, *, qwen_client):
    started = time.monotonic()
    snapshot = read_sealed(snapshot_path)
    sam = read_sealed(verified(snapshot["sources"]["atomic_instances"]))
    render = read_sealed(verified(snapshot["sources"]["render"]))
    lookup = {r["instance_id"]: r for r in snapshot["objects"]}
    if not 2 <= len(instance_ids) <= 3 or len(set(instance_ids)) != len(instance_ids):
        raise ValueError("Review exactly two or three distinct atoms")
    objects = [lookup[key] for key in instance_ids]
    if any(not r["decision"]["category_confirmed"] or r["semantic"]["mask_state"] != "partial_object" for r in objects):
        raise ValueError("Part recovery requires confirmed-category partial atoms")
    output.mkdir(parents=True, exist_ok=False)
    from .geometry import geometry_kernel
    kernel = geometry_kernel()
    vertices, _ = kernel.load_obj_world_zup(verified(snapshot["sources"]["mesh"]))
    with np.load(verified(snapshot["sources"]["support_vertices"]), allow_pickle=False) as archive:
        points = [vertices[archive[r["geometry_source"]["support_vertices_file_key"]]] for r in objects]
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distance = float(cKDTree(points[i]).query(points[j], k=1)[0].min())
            distances.append({"left": instance_ids[i], "right": instance_ids[j], "minimum_support_distance_m": distance})
    if any(r["minimum_support_distance_m"] > .25 for r in distances):
        raise ValueError("Atoms are not geometrically adjacent; no identity merge proposal")
    observation_sets = [set(r["geometry_source"]["atomic_source_observation_ids"]) for r in objects]
    candidates = []
    for view in sam["view_inference"]:
        available = {o["observation_id"] for o in view["observations"]}
        member_observations = [sorted(ids & available) for ids in observation_sets]
        if not all(member_observations): continue
        archive_path = verified(view["mask_archive"])
        with np.load(archive_path, allow_pickle=False) as archive:
            masks = [np.any([archive[key].astype(bool) for key in keys], axis=0) for keys in member_observations]
        candidates.append((sum(int(m.sum()) for m in masks), view["view_index"], masks, artifact(archive_path)))
    if len(candidates) < 2:
        raise ValueError("Fewer than two shared views; acquire another view before merging")
    candidates.sort(key=lambda r: (-r[0], r[1]))
    images, evidence = [], []
    colors = [(0, 230, 230), (245, 60, 210), (255, 205, 0)]
    names = ["CYAN", "MAGENTA", "GOLD"]
    for _, index, masks, source_mask in candidates[:3]:
        raw_path = verified(render["views"][index]["rgb"])
        pixels = np.asarray(Image.open(raw_path).convert("RGB")).copy()
        overlay = pixels.copy()
        for mask, color in zip(masks, colors):
            overlay[mask] = (pixels[mask]*.35 + np.array(color)*.65).astype(np.uint8)
        canvas = Image.new("RGB", (pixels.shape[1], pixels.shape[0]+34), "white")
        canvas.paste(Image.fromarray(overlay), (0, 34))
        ImageDraw.Draw(canvas).text((8, 9), " | ".join(f"{n}: REGION_{i}" for i,n in enumerate(names[:len(masks)])), fill="black")
        target = output / f"view_{index:02d}_regions.png"
        canvas.save(target)
        images.extend([raw_path, target])
        evidence.append({"view_index": index, "rgb": artifact(raw_path), "marked": artifact(target), "raw_mask_archive": source_mask})
    qwen = qwen_client.session(output / "qwen_calls", max_tokens=230)
    judgments = []
    for order in (images, sum([images[i:i+2] for i in range(len(images)-2, -1, -2)], [])):
        response = qwen.call(order, PROMPT, schema=SCHEMA, max_tokens=230, call_id="p550_same_object_part_recovery")
        parsed = json.loads(response["raw_completion"])
        if set(parsed) != set(SCHEMA["required"]): raise ValueError("Recovery schema drift")
        if not isinstance(parsed["confidence"], (int,float)) or not 0 <= parsed["confidence"] <= 1:
            raise ValueError("Invalid recovery confidence")
        judgments.append({"parsed": parsed, "generation": response})
    same = all(r["parsed"]["all_regions_same_physical_object"] and not r["parsed"]["union_contains_other_furniture"]
        and r["parsed"]["confidence"] >= .85 for r in judgments)
    agreement = judgments[0]["parsed"]["object_class"] == judgments[1]["parsed"]["object_class"]
    whole = all(r["parsed"]["union_covers_whole_visible_object"] for r in judgments)
    result = {"schema": "p550.visual_same_object_parts_review.v1", "scene_id": snapshot["scene_id"],
        "source_snapshot": artifact(snapshot_path), "member_ids": instance_ids, "evidence": evidence,
        "geometric_adjacency": distances, "judgments": judgments, "same_object_confirmed": bool(same and agreement),
        "whole_object_composition_authorized": bool(same and agreement and whole), "merge_performed": False,
        "task_instruction_read": False, "numeric_geometry_computed_by_qwen": False,
        "elapsed_seconds": time.monotonic()-started, "source_code": artifact(__file__)}
    write_once(output / "receipt.json", result, seal=True)
    print({k: result[k] for k in ("member_ids", "same_object_confirmed", "whole_object_composition_authorized", "elapsed_seconds")}, flush=True)
