"""Actual scene-only SAM inference and static strict atomic publication."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from hsi.common.artifacts import write_once, read_sealed
from . import atomic, sam_geometry
from .sam_geometry import (
    require, artifact, read_inputs, checked_artifact, load_original_obj_vertices,
    load_sam31, run_view_sam, visible_vertex_projection, Observation,
    panel_for_observation, contact_sheet, OUTPUT_SCHEMA,
)


class AtomicFusionRejected(sam_geometry.SceneInstanceError):
    def __init__(self, audit):
        super().__init__("No SAM3.1 object survived the strict atomic multi-view gates")
        self.fusion_audit = audit

def run(args: argparse.Namespace) -> dict[str, Any]:
    render_path = args.render_receipt.resolve(strict=True)
    boxes_path = args.boxes_receipt.resolve(strict=True)
    output_path = args.output.resolve()
    output_dir = output_path.parent
    require(not output_path.exists(), "SAM output receipt already exists")
    require(not output_dir.exists() or not any(output_dir.iterdir()), "SAM output directory is not empty")
    require(args.instruction == "", "Stage1A perception never accepts a task instruction")
    require(args.refine_iterations == 3 and args.minimum_support_vertices == 180,
            "Use the fixed scene-only SAM refinement/support policy")
    output_dir.mkdir(parents=True, exist_ok=True)
    render, boxes = read_inputs(render_path, boxes_path)
    source_bindings = (artifact(render_path), artifact(boxes_path))
    mesh_path = checked_artifact(render["inputs"]["original_scene_mesh"], "original scene mesh")
    vertices = load_original_obj_vertices(mesh_path, int(render["mesh"]["vertex_count"]))
    model, model_lineage = load_sam31(args.weight.resolve(strict=True), sam_root=args.sam_root)

    observations: list[Observation] = []
    view_receipts: list[dict[str, Any]] = []
    raw_mask_artifacts: list[dict[str, Any]] = []
    total_sam_seconds = 0.0
    observation_counter = 0
    for render_view, box_view in zip(render["views"], boxes["views"]):
        view_index = int(render_view["view_index"])
        require(view_index == int(box_view["view_index"]), "render/box view ordering drift")
        rgb_path = checked_artifact(render_view["rgb"], f"view {view_index} RGB")
        depth_path = checked_artifact(render_view["metric_depth_m"], f"view {view_index} depth")
        rows, elapsed = run_view_sam(model, rgb_path, box_view["proposals"], args.refine_iterations)
        total_sam_seconds += elapsed
        depth = np.load(depth_path, allow_pickle=False)
        visible_ids, visible_u, visible_v = visible_vertex_projection(vertices, render_view, depth)
        kept_masks: dict[str, np.ndarray] = {}
        local_records: list[dict[str, Any]] = []
        for row in rows:
            inside = row["mask"][visible_v, visible_u]
            support = np.unique(visible_ids[inside])
            if len(support) < args.minimum_support_vertices:
                continue
            xyz = vertices[support]
            bounds = np.stack((xyz.min(axis=0), xyz.max(axis=0)))
            extents = bounds[1] - bounds[0]
            if float(np.max(extents[:2])) > 2.8 or float(extents[2]) > 2.15:
                continue
            observation_id = f"OBSERVATION_{observation_counter:04d}"
            observation_counter += 1
            observation = Observation(
                observation_id=observation_id,
                view_index=view_index,
                view_id=str(render_view["view_id"]),
                proposal_id=row["proposal_id"],
                mask=row["mask"],
                support=support,
                bounds=bounds,
                centroid=np.median(xyz, axis=0),
                quality=float(row["quality"]),
                mask_audit=dict(row["mask_audit"]),
                prompt=dict(row["prompt"]),
            )
            observations.append(observation)
            kept_masks[observation_id] = np.asarray(row["mask"], dtype=np.uint8)
            local_records.append({
                "observation_id": observation_id,
                "proposal_id": observation.proposal_id,
                "support_vertex_count": int(len(support)),
                "bounds_world_zup_m": bounds.tolist(),
                "centroid_world_zup_m": observation.centroid.tolist(),
                "prompt_box_xywh": observation.prompt,
                "mask_audit": observation.mask_audit,
            })
        archive_path = output_dir / "raw_sam31_masks" / f"view_{view_index:02d}_masks.npz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive_path, **kept_masks)
        raw_mask_artifacts.append(artifact(archive_path))
        view_receipts.append({
            "view_index": view_index,
            "view_id": render_view["view_id"],
            "anonymous_box_count": len(box_view["proposals"]),
            "accepted_sam31_observation_count": len(local_records),
            "sam31_inference_seconds": elapsed,
            "mask_archive": artifact(archive_path),
            "observations": local_records,
        })

    require(observations, "SAM3.1 produced no valid scene observations")
    groups, positive_edges, fusion_audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    published_supports = [atomic._component_supports(tuple(group), observations)[1] for group in groups]
    if not groups:
        raise AtomicFusionRejected(fusion_audit)

    # Stable order is geometric, not a hidden semantic or historical target ID.
    group_rows: list[tuple[np.ndarray, list[int], np.ndarray]] = []
    for members in groups:
        support = np.unique(np.concatenate([observations[index].support for index in members]))
        centre = np.median(vertices[support], axis=0)
        group_rows.append((centre, members, support))
    group_rows.sort(key=lambda item: tuple(np.round(item[0], 6).tolist()))

    support_payload: dict[str, np.ndarray] = {}
    instances: list[dict[str, Any]] = []
    classifier_paths: list[Path] = []
    for instance_index, (centre, members, support) in enumerate(group_rows):
        instance_id = f"SCENE_INSTANCE_{instance_index:03d}"
        member_observations = [observations[index] for index in members]
        view_ids = sorted({row.view_index for row in member_observations})
        require(len(view_ids) >= 2, "internal cross-view gate drift")
        support_payload[instance_id] = np.asarray(support, dtype=np.int64)
        ranked = sorted(
            member_observations,
            key=lambda row: (-len(row.support), -row.quality, row.view_index, row.proposal_id),
        )
        selected: list[Observation] = []
        used_views: set[int] = set()
        for row in ranked:
            if row.view_index not in used_views:
                selected.append(row)
                used_views.add(row.view_index)
            if len(selected) == 3:
                break
        for row in ranked:
            if len(selected) == 3:
                break
            if row not in selected:
                selected.append(row)
        while len(selected) < 3:
            selected.append(selected[len(selected) % len(selected)])
        image_records: list[dict[str, Any]] = []
        source_masks: list[dict[str, Any]] = []
        for image_index, observation in enumerate(selected[:3]):
            rgb_path = checked_artifact(render["views"][observation.view_index]["rgb"], "classifier source RGB")
            image_path = output_dir / "classifier_images" / f"{instance_id}_view_{observation.view_index:02d}_{image_index}.png"
            panel_for_observation(rgb_path, image_path, instance_id, observation)
            image_records.append(artifact(image_path))
            classifier_paths.append(image_path)
            source_masks.append({
                "observation_id": observation.observation_id,
                "view_index": observation.view_index,
                "proposal_id": observation.proposal_id,
            })
        xyz = vertices[support]
        instances.append({
            "instance_id": instance_id,
            "cross_view_gate_pass": True,
            "visible_view_ids": view_ids,
            "observation_count": len(member_observations),
            "support_vertex_count": int(len(support)),
            "support_vertices_file_key": instance_id,
            "centroid_world_zup_m": centre.tolist(),
            "bounds_world_zup_m": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
            "classifier_images": image_records,
            "source_masks": source_masks,
            "semantic_class": None,
        })

    support_path = output_dir / "instance_support_vertices.npz"
    np.savez_compressed(support_path, **support_payload)
    sheet_path = output_dir / "sam31_multiview_instance_contact_sheet.png"
    contact_sheet(classifier_paths, sheet_path)
    result: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "status": "scene_instances_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_id": render["scene_id"],
        "instruction": "",
        "source_render_receipt": artifact(render_path),
        "source_anonymous_boxes_receipt": artifact(boxes_path),
        "source_original_scene_mesh": artifact(mesh_path),
        "model": model_lineage,
        "runtime": {
            "refine_iterations": args.refine_iterations,
            "total_sam31_inference_seconds": total_sam_seconds,
            "view_count": len(render["views"]),
            "anonymous_box_count": int(boxes["total_anonymous_box_count"]),
            "accepted_observation_count": len(observations),
            "positive_cross_view_edge_count": len(positive_edges),
        },
        "view_inference": view_receipts,
        "cross_view_positive_edges": positive_edges,
        "instance_count": len(instances),
        "instances": instances,
        "support_vertices": artifact(support_path),
        "raw_mask_archives": raw_mask_artifacts,
        "visualization": artifact(sheet_path),
        "query_contract": {
            "complete_camera_sweep_processed_before_semantics": True,
            "sam31_checkpoint_executed": True,
            "boxes_used_as_visual_prompts_only": True,
            "sam31_masks_define_observations": True,
            "original_obj_zero_based_vertex_ids_preserved": True,
            "two_or_more_calibrated_views_required_per_instance": True,
            "instruction_available_to_segmentation_or_fusion": False,
            "instruction_copied_only_after_instance_fusion": False,
            "semantic_class_read_or_assigned": False,
            "support_surface_candidates_read": False,
            "motion_pose_contact_keypose_planner_memory_or_hsi_read": False,
        },
        "source": artifact(Path(__file__)),
    }
    result = atomic.decorate_receipt(result, fusion_audit, published_supports)
    result["source_closure"] = [artifact(Path(__file__)), artifact(Path(atomic.__file__)), artifact(Path(sam_geometry.__file__))]
    require(read_inputs(render_path, boxes_path) == (render, boxes),
            "Render/box inputs drifted during SAM inference")
    require((artifact(render_path), artifact(boxes_path)) == source_bindings,
            "Render/box source identity drifted during SAM inference")
    require(artifact(mesh_path) == render["inputs"]["original_scene_mesh"], "scene mesh drifted during SAM inference")
    for view in render["views"]:
        checked_artifact(view["rgb"], "final RGB")
        checked_artifact(view["metric_depth_m"], "final depth")
    result.pop("receipt_payload_sha256", None)
    write_once(output_path, result, seal=True)
    return read_sealed(output_path)
