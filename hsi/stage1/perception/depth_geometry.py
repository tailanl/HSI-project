#!/usr/bin/env python3
"""Create query-clean visual prompts for scene-wide SAM3.1 segmentation.

The boxes come only from calibrated depth discontinuities over the complete
camera sweep.  They carry no category, instruction, support-surface, or target
information.  SAM3.1 remains responsible for the final object mask; these
anonymous boxes merely avoid asking an open-vocabulary detector to interpret
the untextured grey LINGO mesh as a natural photograph.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label


SOURCE_SCHEMA = "p508.lingo_fullscene_multiview_render.v1"
SCHEMA = "p508.lingo_fullscene_anonymous_sam31_box_prompts.v1"


class ProposalError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ProposalError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def checked_artifact(record: Any, label_text: str) -> Path:
    require(isinstance(record, Mapping), f"{label_text} is not an artifact")
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label_text} size drift")
    require(sha256_file(path) == record.get("sha256"), f"{label_text} hash drift")
    return path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def world_points(depth: np.ndarray, k: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    camera = np.stack(
        ((xx - k[0, 2]) * depth / k[0, 0],
         (yy - k[1, 2]) * depth / k[1, 1],
         depth, np.ones_like(depth)), axis=-1,
    )
    return (camera @ np.linalg.inv(world_to_camera).T)[..., :3]


def iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = left["box_xyxy"]
    bx0, by0, bx1, by1 = right["box_xyxy"]
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0) + 1.0)
    height = max(0.0, min(ay1, by1) - max(ay0, by0) + 1.0)
    intersection = width * height
    area_a = (ax1 - ax0 + 1.0) * (ay1 - ay0 + 1.0)
    area_b = (bx1 - bx0 + 1.0) * (by1 - by0 + 1.0)
    return intersection / max(1.0, area_a + area_b - intersection)


def propose(depth: np.ndarray, k: np.ndarray, world_to_camera: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    require(depth.ndim == 2 and np.isfinite(depth).all(), "depth must be finite metric depth")
    valid_depth = depth > 0
    require(float(np.mean(valid_depth)) >= 0.10, "depth has insufficient rendered-surface coverage")
    height, width = depth.shape
    world = world_points(depth, k, world_to_camera)
    # Pyrender uses zero depth for rays that do not hit a surface.  These
    # pixels are valid no-surface observations, not malformed metric depth.
    # Keep them outside every component and only compare adjacent 3-D points
    # when both rays hit geometry.
    edge = ~valid_depth
    horizontal_pair = valid_depth[:, 1:] & valid_depth[:, :-1]
    vertical_pair = valid_depth[1:] & valid_depth[:-1]
    horizontal = horizontal_pair & (
        np.linalg.norm(world[:, 1:] - world[:, :-1], axis=-1) > args.edge_distance_m
    )
    vertical = vertical_pair & (
        np.linalg.norm(world[1:] - world[:-1], axis=-1) > args.edge_distance_m
    )
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:] |= vertical
    edge[:-1] |= vertical
    interior = (
        valid_depth
        & (world[..., 2] >= args.minimum_object_height_m)
        & (world[..., 2] <= args.maximum_object_height_m)
        & ~edge
    )
    components, count = label(interior)
    rows: list[dict[str, Any]] = []
    image_area = height * width
    for component_id in range(1, count + 1):
        yy, xx = np.where(components == component_id)
        area = int(len(xx))
        if not (args.minimum_component_pixels <= area <= args.maximum_component_fraction * image_area):
            continue
        raw_x0, raw_x1 = int(xx.min()), int(xx.max())
        raw_y0, raw_y1 = int(yy.min()), int(yy.max())
        raw_width, raw_height = raw_x1 - raw_x0 + 1, raw_y1 - raw_y0 + 1
        if min(raw_width, raw_height) < args.minimum_component_extent_px:
            continue
        selected_world = world[yy, xx]
        bounds = np.stack((selected_world.min(axis=0), selected_world.max(axis=0)))
        if float(bounds[1, 2] - bounds[0, 2]) < args.minimum_world_height_span_m:
            continue
        padding = max(args.minimum_padding_px, int(round(args.padding_fraction * max(raw_width, raw_height))))
        x0, x1 = max(0, raw_x0 - padding), min(width - 1, raw_x1 + padding)
        y0, y1 = max(0, raw_y0 - padding), min(height - 1, raw_y1 + padding)
        expanded_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if expanded_area > args.maximum_box_fraction * image_area:
            continue
        compactness = area / max(1, raw_width * raw_height)
        score = math.log1p(area) * (0.35 + 0.65 * compactness)
        rows.append({
            "component_id": int(component_id), "box_xyxy": [x0, y0, x1, y1],
            "box_xywh": {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1},
            "raw_component_box_xyxy": [raw_x0, raw_y0, raw_x1, raw_y1],
            "component_pixels": area, "component_compactness": float(compactness),
            "component_bounds_world_zup_m": bounds.astype(float).tolist(),
            "anonymous_proposal_score": float(score),
            "prompt_kind": "calibrated_depth_discontinuity_box",
            "semantic_label": None,
        })
    rows.sort(key=lambda row: (-float(row["anonymous_proposal_score"]), row["box_xyxy"]))
    kept: list[dict[str, Any]] = []
    for row in rows:
        if any(iou(row, other) >= args.box_nms_iou for other in kept):
            continue
        kept.append(row)
        if len(kept) >= args.maximum_boxes_per_view:
            break
    for index, row in enumerate(kept):
        row["proposal_id"] = f"ANON_BOX_{index:03d}"
    return kept


def draw_boxes(image_path: Path, rows: list[Mapping[str, Any]], output_path: Path, view_index: int) -> None:
    image = Image.open(image_path).convert("RGB")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    for row in rows:
        x0, y0, x1, y1 = row["box_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=(240, 35, 115), width=2)
        draw.rectangle((x0, y0, min(x1, x0 + 104), min(y1, y0 + 17)), fill=(22, 30, 42))
        draw.text((x0 + 3, y0 + 3), str(row["proposal_id"]), fill=(255, 238, 100), font=font)
    draw.rectangle((0, 0, image.width - 1, 23), fill=(20, 29, 40))
    draw.text((7, 7), f"view {view_index:02d} | anonymous depth boxes | no instruction/classes", fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def contact_sheet(paths: list[Path], output_path: Path) -> None:
    panels = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((480, 360), Image.Resampling.LANCZOS)
        panels.append(image)
    columns = 4
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * 480, rows * 360), (232, 235, 239))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * 480, (index // columns) * 360))
    canvas.save(output_path, optimize=True)

