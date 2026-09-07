from ._common import *
from scipy.spatial import cKDTree
from .semantic_contract import SEATING

def compatible_parts(left, right):
    if left["instance_id"] == right["instance_id"]:
        return False
    for obj in (left, right):
        if not obj["decision"]["category_confirmed"] or obj["semantic"]["mask_state"] != "partial_object":
            return False
    a, b = left["semantic"], right["semantic"]
    if a["object_class"] != b["object_class"] or a["object_class"] not in SEATING-{"bed"}:
        return False
    return (a["visible_seat_in_mask"] != b["visible_seat_in_mask"]
        and (a["visible_backrest_in_mask"] or b["visible_backrest_in_mask"]))


def run(snapshot_path, output, limit=4):
    snapshot = read_sealed(snapshot_path)
    objects = snapshot["objects"]
    from .geometry import geometry_kernel
    vertices, _ = geometry_kernel().load_obj_world_zup(verified(snapshot["sources"]["mesh"]))
    sam = read_sealed(verified(snapshot["sources"]["atomic_instances"]))
    observations = {v["view_index"]: {o["observation_id"] for o in v["observations"]} for v in sam["view_inference"]}
    proposals = []
    with np.load(verified(snapshot["sources"]["support_vertices"]), allow_pickle=False) as archive:
        for i, left in enumerate(objects):
            for right in objects[i+1:]:
                if not compatible_parts(left, right):
                    continue
                ids = [left["instance_id"], right["instance_id"]]
                obs = [set(r["geometry_source"]["atomic_source_observation_ids"]) for r in (left, right)]
                shared = [v for v, available in observations.items() if all(s & available for s in obs)]
                if len(shared) < 2:
                    continue
                points = [vertices[archive[r["geometry_source"]["support_vertices_file_key"]]] for r in (left, right)]
                distance = float(cKDTree(points[0]).query(points[1])[0].min())
                if distance <= .25:
                    proposals.append({"member_ids": ids, "minimum_support_distance_m": distance,
                        "shared_view_ids": shared, "merge_authorized": False})
    proposals.sort(key=lambda r: (r["minimum_support_distance_m"], -len(r["shared_view_ids"]), r["member_ids"]))
    write_once(output, {"schema": "p550.scene_only_recovery_proposals.v1", "source_snapshot": artifact(snapshot_path),
        "proposals": proposals[:limit], "all_eligible_pair_count": len(proposals), "maximum_actual_pair_reviews": limit,
        "unreviewed_proposals_not_accepted": True, "task_instruction_read": False, "source": artifact(__file__)}, seal=True)
