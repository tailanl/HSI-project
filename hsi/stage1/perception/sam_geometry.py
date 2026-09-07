#!/usr/bin/env python3
"""Run real SAM3.1 over every anonymous full-scene proposal and fuse in 3-D.

This is deliberately scene-first.  The image sweep and anonymous depth boxes
must have been produced without an instruction, a semantic class, a support
surface, motion, or planner memory.  SAM3.1 turns those boxes into masks.  The
masks are back-projected to *original OBJ zero-based vertex IDs* and only
objects observed by at least two calibrated views are published for Qwen.

Task instructions are not an input to this module. The native scene-only
orchestrator publishes an empty instruction for later semantic processing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree


from hsi.common.artifacts import read_sealed
HERE = Path(__file__).resolve()
RENDER_SCHEMA = "p508.lingo_fullscene_multiview_render.v1"
BOX_SCHEMA = "p508.lingo_fullscene_anonymous_sam31_box_prompts.v1"
OUTPUT_SCHEMA = "p508.lingo_sam31_multiview_instances.v1"
LINGO_YUP_TO_WORLD_ZUP = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), dtype=np.float64
)


class SceneInstanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise SceneInstanceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def checked_artifact(value: Any, label: str) -> Path:
    require(isinstance(value, Mapping), f"{label} is not an artifact record")
    path = Path(str(value.get("path", ""))).resolve(strict=True)
    require(path.stat().st_size == int(value.get("bytes", -1)), f"{label} size drift")
    require(sha256_file(path) == value.get("sha256"), f"{label} hash drift")
    return path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_inputs(render_path: Path, boxes_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    render = read_sealed(render_path)
    boxes = read_sealed(boxes_path)
    require(render.get("schema") == RENDER_SCHEMA, "wrong full-scene render schema")
    require(render.get("status") == "fullscene_multiview_ready", "full-scene render is not ready")
    require(boxes.get("schema") == BOX_SCHEMA, "wrong anonymous-box schema")
    require(boxes.get("status") == "anonymous_sam31_box_prompts_ready", "anonymous boxes are not ready")
    require(render.get("scene_id") == boxes.get("scene_id"), "render/box scene mismatch")
    source_render = checked_artifact(boxes.get("source_render_receipt"), "box source render")
    require(source_render == render_path.resolve(), "boxes are not bound to the requested render")
    require(len(render.get("views", [])) == int(render.get("view_count", -1)), "render view count drift")
    require(len(boxes.get("views", [])) == len(render["views"]), "render/box view count mismatch")
    render_contract = render.get("query_contract", {})
    box_contract = boxes.get("query_contract", {})
    for key in ("instruction_read", "semantic_target_read", "support_candidate_read"):
        require(render_contract.get(key) is False, f"render contract violates {key}")
    for key in ("instruction_read", "semantic_class_read_or_assigned", "support_surface_candidates_read"):
        require(box_contract.get(key) is False, f"box contract violates {key}")
    return render, boxes


def load_original_obj_vertices(path: Path, expected_count: int) -> np.ndarray:
    # trimesh process=False/maintain_order preserves OBJ vertex order; accepting a
    # Scene here could concatenate/reorder geometry, so fail closed instead.
    import trimesh

    mesh = trimesh.load(path, process=False, maintain_order=True)
    require(isinstance(mesh, trimesh.Trimesh), "source OBJ did not load as one ordered Trimesh")
    native = np.asarray(mesh.vertices, dtype=np.float64)
    require(native.shape == (expected_count, 3), "source OBJ vertex count/order contract drift")
    require(np.isfinite(native).all(), "source OBJ vertices contain non-finite coordinates")
    return native @ LINGO_YUP_TO_WORLD_ZUP.T


def cleanup_mask(mask: np.ndarray, box: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    binary = np.asarray(mask > 0, dtype=bool)
    height, width = binary.shape
    area_before = int(binary.sum())
    if area_before == 0:
        return binary, {"reason": "empty", "gate_pass": False}
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    x1 = max(0, int(math.floor(float(box["x"]))))
    y1 = max(0, int(math.floor(float(box["y"]))))
    x2 = min(width, int(math.ceil(float(box["x"]) + float(box["width"]))))
    y2 = min(height, int(math.ceil(float(box["y"]) + float(box["height"]))))
    ids, frequencies = np.unique(labels[y1:y2, x1:x2], return_counts=True)
    candidates = [(int(freq), int(label)) for label, freq in zip(ids, frequencies) if label != 0]
    if not candidates:
        return np.zeros_like(binary), {"reason": "no_component_inside_prompt", "gate_pass": False}
    _, chosen = max(candidates)
    cleaned = labels == chosen
    area = int(cleaned.sum())
    inside = int(cleaned[y1:y2, x1:x2].sum())
    area_fraction = area / float(height * width)
    inside_fraction = inside / float(max(area, 1))
    box_fill = inside / float(max((x2 - x1) * (y2 - y1), 1))
    gate = bool(
        area >= 300
        and area_fraction <= 0.32
        and inside_fraction >= 0.58
        and box_fill >= 0.012
    )
    return cleaned, {
        "reason": "accepted" if gate else "mask_geometry_gate_failed",
        "gate_pass": gate,
        "raw_area_px": area_before,
        "largest_prompt_component_area_px": area,
        "image_area_fraction": area_fraction,
        "mask_fraction_inside_prompt": inside_fraction,
        "prompt_box_fill_fraction": box_fill,
        "connected_component_count": int(count),
    }


def mask_nms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        audit = row["mask_audit"]
        row["quality"] = float(
            audit["mask_fraction_inside_prompt"]
            * math.sqrt(max(audit["prompt_box_fill_fraction"], 1e-8))
            * math.log1p(audit["largest_prompt_component_area_px"])
        )
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (-item["quality"], item["proposal_id"])):
        mask = row["mask"]
        duplicate = False
        for other in kept:
            intersection = int(np.count_nonzero(mask & other["mask"]))
            if intersection == 0:
                continue
            area_a, area_b = int(mask.sum()), int(other["mask"].sum())
            union = area_a + area_b - intersection
            iou = intersection / max(union, 1)
            containment = intersection / max(min(area_a, area_b), 1)
            if iou >= 0.82 or containment >= 0.94:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return sorted(kept, key=lambda item: item["proposal_id"])


def load_sam31(weight: Path, *, sam_root: Path) -> tuple[Any, dict[str, Any]]:
    require(weight.is_file(), f"SAM3.1 checkpoint missing: {weight}")
    sam_root = Path(sam_root).resolve(strict=True)
    require(sam_root.is_dir(), f"ComfyUI source missing: {sam_root}")
    sys.path.insert(0, str(sam_root))
    import torch
    import comfy.sd

    require(Path(comfy.sd.__file__).resolve().is_relative_to(sam_root), "loaded ComfyUI does not match explicit external source")
    require(torch.cuda.is_available(), "CUDA is required for real SAM3.1 inference")
    started = time.monotonic()
    model, _, _, _ = comfy.sd.load_checkpoint_guess_config(
        str(weight), output_vae=False, output_clip=False, output_clipvision=False
    )
    core = model.model.diffusion_model
    require(type(core).__name__ == "SAM3Model", f"checkpoint core is {type(core).__name__}, not SAM3Model")
    return model, {
        "checkpoint": artifact(weight),
        "loader": "comfy.sd.load_checkpoint_guess_config",
        "wrapper_class": type(model).__name__,
        "model_class": type(model.model).__name__,
        "diffusion_model_class": type(core).__name__,
        "dtype": str(model.model.get_dtype()).replace("torch.", ""),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "load_seconds": time.monotonic() - started,
        "comfy_sam3_node_source": artifact(sam_root / "comfy_extras/nodes_sam3.py"),
        "comfy_sam3_detector_source": artifact(sam_root / "comfy/ldm/sam3/detector.py"),
    }


def run_view_sam(model: Any, rgb_path: Path, proposals: Sequence[Mapping[str, Any]], refine: int) -> tuple[list[dict[str, Any]], float]:
    import torch
    from comfy_extras.nodes_sam3 import SAM3_Detect

    image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    boxes = [dict(row["box_xywh"]) for row in proposals]
    if not boxes:
        return [], 0.0
    started = time.monotonic()
    output = SAM3_Detect.execute(
        model=model, image=image_tensor, bboxes=boxes, conditioning=None,
        threshold=0.5, refine_iterations=refine, individual_masks=True,
    )
    result = getattr(output, "result", output)
    masks_tensor = result[0]
    masks = np.asarray(masks_tensor.detach().to("cpu").numpy() > 0.5, dtype=bool)
    require(masks.shape == (len(proposals), image.shape[0], image.shape[1]), "SAM3 mask/proposal count drift")
    rows: list[dict[str, Any]] = []
    for proposal, mask in zip(proposals, masks):
        cleaned, audit = cleanup_mask(mask, proposal["box_xywh"])
        if audit["gate_pass"]:
            rows.append({
                "proposal_id": str(proposal["proposal_id"]),
                "prompt": dict(proposal["box_xywh"]),
                "mask": cleaned,
                "mask_audit": audit,
                "proposal_score": float(proposal.get("anonymous_proposal_score", 0.0)),
            })
    return mask_nms(rows), time.monotonic() - started


def visible_vertex_projection(vertices: np.ndarray, view: Mapping[str, Any], depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.asarray(view["world_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(view["K"], dtype=np.float64)
    camera = vertices @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    valid_z = z > 1e-5
    u = np.rint(intrinsics[0, 0] * camera[:, 0] / np.maximum(z, 1e-6) + intrinsics[0, 2]).astype(np.int32)
    v = np.rint(intrinsics[1, 1] * camera[:, 1] / np.maximum(z, 1e-6) + intrinsics[1, 2]).astype(np.int32)
    inside = valid_z & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    ids = np.flatnonzero(inside)
    observed = depth[v[ids], u[ids]]
    tolerance = np.maximum(0.025, 0.0125 * z[ids])
    visible = (observed > 0) & (np.abs(observed - z[ids]) <= tolerance)
    chosen = ids[visible]
    return chosen.astype(np.int64), u[chosen], v[chosen]


@dataclass
class Observation:
    observation_id: str
    view_index: int
    view_id: str
    proposal_id: str
    mask: np.ndarray
    support: np.ndarray
    bounds: np.ndarray
    centroid: np.ndarray
    quality: float
    mask_audit: dict[str, Any]
    prompt: dict[str, Any]


def deterministic_sample(points: np.ndarray, maximum: int = 1200) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]


def pair_audit(left: Observation, right: Observation, vertices: np.ndarray) -> dict[str, Any]:
    exact = int(np.intersect1d(left.support, right.support, assume_unique=True).size)
    minimum_support = min(len(left.support), len(right.support))
    exact_fraction = exact / max(minimum_support, 1)
    padded_low = np.maximum(left.bounds[0] - 0.06, right.bounds[0] - 0.06)
    padded_high = np.minimum(left.bounds[1] + 0.06, right.bounds[1] + 0.06)
    intersection_extent = np.maximum(0.0, padded_high - padded_low)
    intersection_volume = float(np.prod(intersection_extent))
    left_volume = float(np.prod(np.maximum(left.bounds[1] - left.bounds[0] + 0.12, 1e-4)))
    right_volume = float(np.prod(np.maximum(right.bounds[1] - right.bounds[0] + 0.12, 1e-4)))
    aabb_fraction = intersection_volume / max(min(left_volume, right_volume), 1e-9)
    centroid_distance = float(np.linalg.norm(left.centroid - right.centroid))
    near_left = deterministic_sample(vertices[left.support])
    near_right = deterministic_sample(vertices[right.support])
    left_distance, _ = cKDTree(near_right).query(near_left, k=1)
    right_distance, _ = cKDTree(near_left).query(near_right, k=1)
    left_near = float(np.mean(left_distance <= 0.055))
    right_near = float(np.mean(right_distance <= 0.055))
    exact_gate = exact >= max(48, int(math.ceil(0.012 * minimum_support)))
    surface_gate = min(left_near, right_near) >= 0.055 and aabb_fraction >= 0.08 and centroid_distance <= 0.48
    enclosed_gate = aabb_fraction >= 0.28 and centroid_distance <= 0.30 and max(left_near, right_near) >= 0.08
    merge = bool(left.view_index != right.view_index and (exact_gate or surface_gate or enclosed_gate))
    return {
        "exact_vertex_count": exact,
        "exact_minimum_support_fraction": exact_fraction,
        "padded_aabb_intersection_fraction": aabb_fraction,
        "centroid_distance_m": centroid_distance,
        "left_near_fraction_0p055m": left_near,
        "right_near_fraction_0p055m": right_near,
        "merge_gate_pass": merge,
    }


def panel_for_observation(
    source_path: Path, output_path: Path, instance_id: str, observation: Observation
) -> None:
    source = Image.open(source_path).convert("RGB")
    rgb = np.asarray(source, dtype=np.uint8).copy()
    mask = observation.mask
    # Make the classifier attend to the mask rather than a visually salient
    # unmasked chair in the background.  Context remains visible but dim; the
    # crop contains the same cyan-marked pixels, not a raw RGB distraction.
    overlay = (0.36 * rgb).astype(np.uint8)
    overlay[mask] = (0.30 * rgb[mask] + 0.70 * np.asarray([0, 225, 255])).astype(np.uint8)
    left = Image.fromarray(overlay)
    ys, xs = np.nonzero(mask)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    pad = max(10, int(0.12 * max(x2 - x1, y2 - y1)))
    isolated = np.full_like(rgb, 18)
    isolated[mask] = overlay[mask]
    crop = Image.fromarray(isolated).crop(
        (max(0, x1 - pad), max(0, y1 - pad), min(source.width, x2 + pad), min(source.height, y2 + pad))
    )
    crop.thumbnail((300, 420), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (980, 520), (235, 238, 242))
    canvas.paste(left, (0, 40))
    canvas.paste(crop, (660 + (300 - crop.width) // 2, 70 + (420 - crop.height) // 2))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, 979, 39), fill=(20, 29, 40))
    draw.text((10, 13), f"{instance_id} | view {observation.view_index:02d} | classify ONLY CYAN SAM3.1 mask", fill="white", font=font)
    draw.rectangle((650, 60, 969, 499), outline=(0, 140, 175), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def contact_sheet(paths: Sequence[Path], output: Path, columns: int = 3) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((490, 260), Image.Resampling.LANCZOS)
        thumbs.append(image)
    rows = math.ceil(len(thumbs) / columns)
    canvas = Image.new("RGB", (columns * 490, rows * 260), (28, 32, 38))
    for index, image in enumerate(thumbs):
        canvas.paste(image, ((index % columns) * 490, (index // columns) * 260))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
