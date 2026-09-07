from ._common import *
from .semantic_contract import direction_hypotheses
from .. import _surface

def geometry_kernel():
    return _surface


def surface_identity(kernel, scene_id, key, category, mesh_sha, binding_sha, face_ids, vertex_ids):
    value = {"schema": "p523.target_bound_surface_identity.v1", "scene_id": scene_id,
        "target_instance_id": key, "target_class": category, "mesh_sha256": mesh_sha,
        "metric_graph_payload_sha256": binding_sha, "mesh_face_indices_sha256": kernel.int64_sha256(face_ids),
        "surface_vertex_ids_sha256": kernel.int64_sha256(vertex_ids)}
    return value


def local_front(kernel, support_points, surface_points, surface_z):
    bounds = np.stack((surface_points.min(0), surface_points.max(0)))
    local = np.all((support_points[:, :2] >= bounds[0, :2]-.30)
        & (support_points[:, :2] <= bounds[1, :2]+.30), axis=1)
    front, audit = kernel.infer_front_direction(support_points[local], surface_points, surface_z)
    audit.update(local_support_count=int(local.sum()), total_support_count=len(support_points),
        local_window_padding_m=.30, high_support_is_geometry_hypothesis_not_semantic_proof=True)
    return front, audit


def run(snapshot_path, output):
    started = time.monotonic()
    snapshot = read_sealed(snapshot_path)
    if snapshot.get("task_instruction_read") is not False or snapshot.get("future_motion_read") is not False:
        raise ValueError("Scene understanding must be instruction and future independent")
    sam_path = verified(snapshot["sources"]["atomic_instances"])
    sam = read_sealed(sam_path)
    render = read_sealed(verified(snapshot["sources"]["render"]))
    mesh_path = verified(snapshot["sources"]["mesh"])
    occupancy_path = verified(render["inputs"]["query_occupancy"])
    support_path = verified(snapshot["sources"]["support_vertices"])
    output.mkdir(parents=True, exist_ok=False)
    kernel = geometry_kernel()
    config = kernel.SurfaceConfig()
    vertices, faces = kernel.load_obj_world_zup(mesh_path)
    native = np.load(occupancy_path, allow_pickle=False)
    world = kernel.world_occupancy(native)
    fields = kernel.build_navigation_fields(world, config)
    field_path = output / "navigation_fields.npz"
    np.savez_compressed(field_path, **fields)
    rows = []
    with np.load(support_path, allow_pickle=False) as archive:
        for obj in snapshot["objects"]:
            key, category = obj["instance_id"], obj["semantic"]["object_class"]
            row = {"instance_id": key, "category": category, "semantic_evidence": obj["evidence"],
                "semantic_query_usable": obj["decision"]["query_usable"], "size_diagnostic": obj["size_diagnostic"],
                "category_changed_by_geometry": False}
            if category not in kernel.SITTABLE_CLASSES:
                row.update(status="no_sitting_surface_requested_for_this_category", query_usable_for_sit=False)
                rows.append(row)
                continue
            support_ids = np.unique(archive[obj["geometry_source"]["support_vertices_file_key"]].astype(np.int64))
            try:
                extracted = kernel.extract_support_surface(snapshot["scene_id"], key, category, support_ids,
                    vertices, faces, snapshot["sources"]["mesh"]["sha256"], snapshot["receipt_payload_sha256"], config)
                face_ids, vertex_ids, bounds, surface, surface_sha, surface_id, audit = extracted
                front, front_audit = local_front(kernel, vertices[support_ids], vertices[vertex_ids], surface["centre_world_xyz_zup_m"][2])
                hypotheses = direction_hypotheses(front.tolist(), front_audit["method"])
                folder = output / "objects" / key
                folder.mkdir(parents=True)
                array_path = folder / "surface.npz"
                np.savez_compressed(array_path, face_ids=face_ids, vertex_ids=vertex_ids, support_ids=support_ids,
                    target_bounds=bounds)
                row.update(status="surface_extracted", surface=surface, stable_surface_id=surface_id,
                    stable_surface_sha256=surface_sha, surface_arrays=artifact(array_path), extraction_audit=audit,
                    front_evidence=front_audit, direction_hypotheses=hypotheses,
                    query_usable_for_sit=bool(obj["decision"]["query_usable"]),
                    query_and_start_used=False)
            except Exception as error:
                row.update(status="surface_unresolved", query_usable_for_sit=False,
                    failure=f"{type(error).__name__}: {error}",
                    recovery="review_local_surface_and_segmentation_not_object_category")
            rows.append(row)
    result = {"schema": "p550.fixed_scene_interaction_geometry.v1", "scene_id": snapshot["scene_id"],
        "status": "scene_geometry_complete_with_per_object_states", "source_semantics": artifact(snapshot_path),
        "source_sam": artifact(sam_path), "source_mesh": artifact(mesh_path), "source_occupancy": artifact(occupancy_path),
        "source_support": artifact(support_path), "objects": rows, "navigation_fields": artifact(field_path),
        "navigation_config": dict(vars(config)), "task_instruction_read": False, "start_state_read": False,
        "all_instances_attempted": snapshot["all_instances_attempted"], "future_motion_read": False,
        "numeric_geometry_computed_by_qwen": False, "unknown_objects_do_not_abort_scene": True,
        "geometry_kernel": artifact(_surface.__file__),
        "elapsed_seconds": time.monotonic()-started, "source_code": artifact(__file__)}
    write_once(output / "receipt.json", result, seal=True)
    print(json.dumps({"scene": result["scene_id"], "seconds": result["elapsed_seconds"],
        "objects": [{"id": r["instance_id"], "status": r["status"], "usable": r["query_usable_for_sit"]} for r in rows]}), flush=True)
    return result
