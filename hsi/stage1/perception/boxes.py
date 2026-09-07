"""Fresh instruction-blind multiscale visual prompts across every view."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from hsi.common.artifacts import artifact, read_sealed, write_once, require, verified
from . import multiscale as kernel, depth_geometry

def run(render_path, output):
    render_path = Path(render_path).resolve(strict=True)
    output = Path(output).resolve()
    require(not output.exists(), "Anonymous box receipt already exists")
    require(not output.parent.exists() or not any(output.parent.iterdir()), "Anonymous box directory is not empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    _, render = kernel.load_render(render_path)
    rows, images = [], []
    for i, view in enumerate(render["views"]):
        require(view.get("view_index") == i, "Complete calibrated render order drift")
        rgb = depth_geometry.checked_artifact(view["rgb"], "RGB")
        depth_path = depth_geometry.checked_artifact(view["depth"], "depth")
        depth = np.load(depth_path, allow_pickle=False)
        height, width = depth.shape
        proposals, audit = kernel.propose_view_multiscale(depth, np.asarray(view["K"]), np.asarray(view["world_to_camera"]),
            width=width, height=height, iou_threshold=.90, containment_threshold=.98,
            containment_area_ratio_minimum=.82, maximum_rows=48)
        overlay = output.parent / "overlays" / f"view_{i:02d}_boxes.png"
        depth_geometry.draw_boxes(rgb, proposals, overlay, i)
        images.append(overlay)
        rows.append({"view_index": i, "view_id": view.get("view_id"), "source_rgb": artifact(rgb),
                     "source_depth": artifact(depth_path), "anonymous_box_count": len(proposals), "proposals": proposals,
                     "cross_scale_deduplication": audit, "visualization": artifact(overlay)})
    sheet = output.parent / "anonymous_box_contact_sheet.png"
    depth_geometry.contact_sheet(images, sheet)
    value = {"schema": kernel.OUTPUT_SCHEMA, "extension_schema": "p548.initial_multiscale_anonymous_boxes.v1",
        "status": kernel.OUTPUT_STATUS, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_id": render["scene_id"], "source_render_receipt": artifact(render_path), "view_count": len(rows),
        "total_anonymous_box_count": sum(r["anonymous_box_count"] for r in rows), "views": rows,
        "parameters": {"scale_profiles": kernel.SCALE_PROFILES, "maximum_boxes_per_view": 48, "cross_scale_nms_iou": .90,
            "cross_scale_containment": .98, "cross_scale_containment_area_ratio": .82},
        "query_contract": {"complete_camera_sweep_processed_before_target_selection": True, "instruction_read": False,
            "semantic_class_read_or_assigned": False, "support_surface_candidates_read": False,
            "motion_pose_contact_keypose_or_hsi_read": False, "boxes_are_only_visual_prompts_and_not_final_instances": True},
        "historical_feedback_read": False, "generation_mode": "fresh_initial_multiscale_geometry",
        "kernel_source": artifact(kernel.HERE), "depth_geometry_source": artifact(Path(depth_geometry.__file__)), "source": artifact(__file__), "visualization": artifact(sheet)}
    require(read_sealed(render_path) == render, "Render changed during anonymous proposal generation")
    for row in rows:
        verified(row["source_rgb"])
        verified(row["source_depth"])
    write_once(output, value, seal=True)
    return output
