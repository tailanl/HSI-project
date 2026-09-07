from ._common import *
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .semantic_contract import semantic_status,size_diagnostic
from .geometry import geometry_kernel

def connected_labels(points, radius):
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
        shape=(len(points), len(points)))
    return connected_components(graph, directed=False)[1]


def fine_component(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 250 or not np.isfinite(points).all():
        raise ValueError("Need at least 250 finite original support points")
    masks, audits = [], []
    for radius in (.05, .07):
        labels = connected_labels(points, radius)
        counts = np.bincount(labels)
        mask = labels == np.argmax(counts)
        masks.append(mask)
        audits.append({"radius_m": radius, "component_count": len(counts),
            "largest_count": int(mask.sum()), "largest_fraction": float(mask.mean()),
            "largest_bounds": np.stack((points[mask].min(0), points[mask].max(0))).tolist()})
    intersection = int(np.count_nonzero(masks[0] & masks[1]))
    union = int(np.count_nonzero(masks[0] | masks[1]))
    stability = intersection / union
    main = masks[0]
    if not (~main).any():
        raise ValueError("No disjoint support exists; no geometry correction proposed")
    gap = float(cKDTree(points[main]).query(points[~main], k=1)[0].min())
    old_extent = np.ptp(points, axis=0)
    new_extent = np.ptp(points[main], axis=0)
    volume_ratio = float(np.prod(old_extent) / max(1e-12, np.prod(new_extent)))
    gates = {"dominant_support_fraction": float(main.mean()) >= .80,
        "stable_across_two_radii": stability >= .98, "separated_residual": gap >= .10,
        "materially_inflated_parent_bounds": volume_ratio >= 1.50,
        "residual_retained": int((~main).sum()) >= 16}
    audit = {"radii": audits, "largest_component_jaccard": stability,
        "nearest_main_residual_distance_m": gap, "parent_component_aabb_volume_ratio": volume_ratio,
        "gates": gates, "proposal_eligible": all(gates.values()),
        "semantic_category_inferred_from_geometry": False}
    return main, audit

def stable_dominant(points):
    points=np.asarray(points,dtype=float)
    core,prior=fine_component(points)
    if prior["proposal_eligible"]:
        return core,dict(prior,envelope_closure_used=False)
    if not all(prior["gates"][k] for k in ("dominant_support_fraction","stable_across_two_radii","materially_inflated_parent_bounds")):
        return core,dict(prior,envelope_closure_used=False)
    lower,upper=points[core].min(0),points[core].max(0)
    masks=[np.all((points>=lower-margin)&(points<=upper+margin),axis=1) for margin in (.07,.10)]
    main=masks[1]
    if not (~main).any():
        return core,dict(prior,envelope_closure_used=False,envelope_rejected_reason="No external residual remains")
    stability=float(np.count_nonzero(masks[0]&masks[1])/np.count_nonzero(masks[0]|masks[1]))
    gap=float(cKDTree(points[main]).query(points[~main],k=1)[0].min())
    ratio=float(np.prod(np.ptp(points,axis=0))/max(1e-12,np.prod(np.ptp(points[main],axis=0))))
    gates={"dominant_support_fraction":float(main.mean())>=.80,"stable_across_two_envelope_margins":stability>=.98,
        "fine_core_preserved":bool(np.all(main[core])),"separated_residual":gap>=.10,
        "materially_inflated_parent_bounds":ratio>=1.5,"residual_retained":int((~main).sum())>=16}
    audit={"fine_scale_audit":prior,"envelope_closure_used":True,"envelope_margins_m":[.07,.10],
        "core_bounds_world_zup_m":[lower.tolist(),upper.tolist()],"largest_component_jaccard":stability,
        "nearest_main_residual_distance_m":gap,"parent_component_aabb_volume_ratio":ratio,
        "candidate_support_fraction":float(main.mean()),"in_envelope_disconnected_points_retained":int((main&~core).sum()),
        "gates":gates,"proposal_eligible":all(gates.values()),"semantic_category_inferred_from_geometry":False,
        "new_identity_must_be_independently_visually_reviewed":True}
    return main,audit

def run(snapshot_path, instance_id, output):
    snapshot = read_sealed(snapshot_path)
    if snapshot["task_instruction_read"] is not False or snapshot["future_motion_read"] is not False:
        raise ValueError("Component recovery requires scene-only inputs")
    parent = next(r for r in snapshot["objects"] if r["instance_id"] == instance_id)
    if parent["geometry_source"].get("logical_object_composition"):
        raise ValueError("A reviewed logical assembly needs a separate composition-aware component policy")
    mesh_path = verified(snapshot["sources"]["mesh"])
    vertices, _ = geometry_kernel().load_obj_world_zup(mesh_path)
    with np.load(verified(snapshot["sources"]["support_vertices"]), allow_pickle=False) as archive:
        supports = {key: archive[key] for key in archive.files}
    ids = np.unique(supports[parent["geometry_source"]["support_vertices_file_key"]].astype(np.int64))
    points = vertices[ids]
    main, audit = stable_dominant(points)
    output.mkdir(parents=True, exist_ok=False)
    proposal = output / "component_proposal.json"
    identity = digest({"source_snapshot": artifact(snapshot_path), "parent_id": instance_id,
        "main_support_ids": ids[main].tolist(), "policy_source": artifact(__file__)})[:12].upper()
    child_id, residual_id = "SCENE_COMPONENT_" + identity, "SCENE_RESIDUAL_" + identity
    value = {"schema": "p550.scene_only_stable_component_proposal.v1", "scene_id": snapshot["scene_id"],
        "source_snapshot": artifact(snapshot_path), "parent_id": instance_id, "parent_object": parent,
        "audit": audit, "candidate_id": child_id, "residual_id": residual_id,
        "parent_support_count": len(ids), "candidate_support_count": int(main.sum()),
        "residual_support_count": int((~main).sum()), "parent_geometry_and_evidence_preserved": True,
        "no_points_added_or_deleted": True, "new_sam_inference_performed": False,
        "query_or_human_coordinates_read": False, "source": artifact(__file__)}
    write_once(proposal, value, seal=True)
    visualize(points, main, output / "component_diagnostic.png", instance_id, audit)
    if not audit["proposal_eligible"]:
        print({"status": "no_stable_dominant_component_proposal", "audit": audit}, flush=True)
        return value
    objects = copy.deepcopy(snapshot["objects"])
    source_sam = read_sealed(verified(snapshot["sources"]["atomic_instances"]))
    instances = copy.deepcopy(source_sam["instances"])
    for row in objects:
        if row["instance_id"] == instance_id:
            row["prior_component_decision"] = copy.deepcopy(row["decision"])
            row["decision"].update(query_usable=False, needs_visual_recovery=True)
            row["decision"]["reasons"].append("disjoint_support_requires_new_component_identity_review")
            row["spatial_component_proposals"] = [child_id, residual_id]
    for key, mask, role in ((child_id, main, "dominant_component_proposal"), (residual_id, ~main, "unresolved_residual_collection")):
        support = ids[mask]
        supports[key] = support
        bounds = np.stack((vertices[support].min(0), vertices[support].max(0))).tolist()
        instance = {"instance_id": key, "semantic_class": None, "support_vertices_file_key": key,
            "support_vertex_count": len(support), "bounds_world_zup_m": bounds,
            "centroid_world_zup_m": vertices[support].mean(0).tolist(),
            "visible_view_ids": parent["geometry_source"]["visible_view_ids"],
            "cross_view_gate_pass": True, "minimum_distinct_views_per_published_vertex": 2,
            "published_support_policy": "strict_subset_of_parent_two_distinct_view_consensus_vertices",
            "classifier_images": [], "atomic_source_observation_ids": parent["geometry_source"]["atomic_source_observation_ids"],
            "source_masks": parent["geometry_source"]["source_masks"],
            "spatial_component_proposal": {"parent_id": instance_id, "role": role,
                "proposal": artifact(proposal), "new_independent_object_not_yet_visually_confirmed": True}}
        semantic = {"object_class": "unknown", "mask_state": "uncertain", "is_sittable_object": False,
            "visible_seat_in_mask": False, "visible_backrest_in_mask": False, "views_consistent": False,
            "confidence": 0., "visual_description": "Unclassified spatial support proposal; independent original-mesh visual review is required.",
            "mask_issue": "Geometry does not establish object category or completeness."}
        objects.append({"instance_id": key, "semantic": semantic, "decision": semantic_status(semantic),
            "evidence": artifact(proposal), "geometry_source": instance, "size_diagnostic": size_diagnostic(bounds)})
        instances.append(instance)
    archive_path = output / "component_support_vertices.npz"
    np.savez_compressed(archive_path, **supports)
    instance_path = output / "spatial_instances.json"
    write_once(instance_path, {"schema": "p550.spatial_component_scene_instances.v1", "status": "component_proposals_ready",
        "scene_id": snapshot["scene_id"], "source_original_scene_mesh": snapshot["sources"]["mesh"],
        "source_render_receipt": snapshot["sources"]["render"], "source_atomic_instances": snapshot["sources"]["atomic_instances"],
        "support_vertices": artifact(archive_path), "instances": instances, "component_proposals": [artifact(proposal)],
        "original_atoms_removed": False, "original_support_modified": False, "new_sam_inference_performed": False,
        "instruction": "", "source": artifact(__file__)}, seal=True)
    revised = copy.deepcopy(snapshot)
    revised.update(objects=objects, source_snapshot_before_spatial_recovery=artifact(snapshot_path),
        processed_instance_count=len(objects), spatial_component_proposals=[artifact(proposal)],
        new_sam_inference_performed=False, geometry_snapshot_pending=True,
        status="scene_understanding_complete_with_unknowns", source_code=artifact(__file__))
    revised["sources"].update(atomic_instances=artifact(instance_path), support_vertices=artifact(archive_path))
    write_once(output / "receipt.json", revised, seal=True)
    print({"status": "unknown_component_proposed", "candidate": child_id, "residual": residual_id, "audit": audit}, flush=True)
    return revised


def visualize(points, main, path, title, audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dims in zip(axes, ((0, 1), (0, 2))):
        ax.scatter(points[~main, dims[0]], points[~main, dims[1]], s=4, c="#ee7447", label="Preserved residual")
        ax.scatter(points[main, dims[0]], points[main, dims[1]], s=1, c="#008d95", label="Unknown component proposal")
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel(("y" if dims[1] == 1 else "z") + " (m)")
        ax.legend(fontsize=8)
    fig.suptitle(title + " | original support only; no category or success inferred")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def scan(snapshot_path, output):
    snapshot = read_sealed(snapshot_path)
    if snapshot["task_instruction_read"] is not False or snapshot["future_motion_read"] is not False:
        raise ValueError("Component scan must be scene-only")
    vertices, _ = geometry_kernel().load_obj_world_zup(verified(snapshot["sources"]["mesh"]))
    rows = []
    with np.load(verified(snapshot["sources"]["support_vertices"]), allow_pickle=False) as archive:
        for obj in snapshot["objects"]:
            if not obj["decision"]["query_usable"] or not obj["semantic"]["is_sittable_object"] \
                    or obj.get("represented_by_logical_object_id"):
                continue
            row = {"instance_id": obj["instance_id"], "candidate_proposed": False, "category_changed": False}
            if obj["geometry_source"].get("logical_object_composition"):
                row["status"] = "separate_visually_reviewed_composition_retained"
            else:
                ids = np.unique(archive[obj["geometry_source"]["support_vertices_file_key"]].astype(np.int64))
                if len(ids) > 150000:
                    row.update(status="bounded_scan_requires_larger_support_strategy", automatic_geometry_publication=False)
                else:
                    try:
                        main, audit = stable_dominant(vertices[ids])
                        row.update(status="spatial_diagnostic_complete", audit=audit, candidate_proposed=audit["proposal_eligible"])
                        row["ambiguous_disjoint_support_requires_review"] = bool(
                            not audit["gates"]["dominant_support_fraction"] and audit["gates"]["separated_residual"]
                            and audit["gates"]["materially_inflated_parent_bounds"])
                    except ValueError as error:
                        row.update(status="no_component_recovery_proposed", reason=str(error))
            rows.append(row)
    write_once(output, {"schema": "p550.scene_only_component_scan.v1", "scene_id": snapshot["scene_id"],
        "source_snapshot": artifact(snapshot_path), "objects": rows, "source": artifact(__file__),
        "human_query_or_stage2_read": False, "geometry_does_not_relabel_categories": True}, seal=True)
