"""Clean current-motion image assembly; no judgement or numeric gate text.

All 3-D rendering uses the saved current meshes and full scene. This helper
only chooses numeric body+target cameras and assembles labelled time tiles.
The caller owns source binding, scene loading, EGL, and final publication.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import visualize_geometry_motion_v2 as overview
from memory_common import artifact, require

SOURCE = Path(__file__).resolve()
OVERVIEW_SHA = "8c90cbedafd939b4db12a711dea212abd116e95407b00d886b784865fce40cbb"


def close_bounds(vertices, target_bounds):
    vertices, target_bounds = np.asarray(vertices), np.asarray(target_bounds)
    require(vertices.ndim == 3 and vertices.shape[1:] == (10475, 3) and 1 <= len(vertices) <= 6
            and np.isfinite(vertices).all(), "Expected actual bounded terminal body frames")
    overview.corners(target_bounds)
    points = vertices.reshape(-1, 3)
    return np.stack((np.minimum(points.min(0), target_bounds[0]), np.maximum(points.max(0), target_bounds[1])))


def angle_distance(a, b):
    return abs((float(a)-float(b)+180.) % 360.-180.)


def select_close_views(renderer, vertices, target_bounds, frame_indices):
    require(artifact(overview.__file__)["sha256"] == OVERVIEW_SHA, "Frozen camera/mask math drift")
    indices = list(frame_indices)
    require(len(indices) == 4 and len(set(indices)) == 4 and all(type(v) is int and v >= 0 for v in indices), "Need actual four distinct terminal frames")
    bounds = close_bounds(vertices, target_bounds)
    candidates = overview.overview_candidates(bounds)
    screened = []
    for camera in candidates:
        renderer.camera(overview.screening_camera(camera))
        rows = [renderer.frame(index)[0] for index in (indices[0], indices[-1])]
        screened.append(dict(camera=camera, summary=overview.visibility_summary(rows)))
    ranked = sorted(screened, key=lambda row: (-row["summary"]["minimum_visible_fraction"],
        -row["summary"]["mean_visible_fraction"], -row["summary"]["minimum_amodal_pixels"], row["camera"]["candidate_id"]))
    selected, tested = [], []
    for row in ranked:
        if not row["summary"]["all_passed"]: continue
        camera = row["camera"]
        if selected and angle_distance(camera["azimuth_degrees"], selected[0]["camera"]["azimuth_degrees"]) < 45.:
            continue
        renderer.camera(camera)
        masks = [renderer.frame(index)[0] for index in indices]
        result = dict(camera=camera, frame_masks=masks, summary=overview.visibility_summary(masks))
        tested.append(result)
        if result["summary"]["all_passed"]:
            selected.append(result)
            if len(selected) == 2: break
    require(len(selected) == 2, "No two distinct current full-scene close views passed visibility")
    return dict(bounds_world_zup_m=bounds.tolist(), numeric_candidates=screened, full_resolution_candidates=tested,
        selected=selected, minimum_view_azimuth_separation_degrees=45., all_target_aabb_and_body_bounds_in_frame=True,
        actual_four_frame_mask_policy=dict(overview.POLICY), no_occluders_hidden=True,
        qwen_selected_coordinates=False)


def label_tile(pixels, frame_index, fps=20):
    require(type(frame_index) is int and frame_index >= 0 and type(fps) is int and fps > 0, "Invalid frame label")
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB")
    labelled = Image.new("RGB", (image.width, image.height+36), "white")
    labelled.paste(image, (0, 36))
    draw = ImageDraw.Draw(labelled)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    # No requested phase, prior verdict, gate names, or numeric measurements.
    draw.text((10, 7), "Frame %d | %.2f s" % (frame_index, frame_index/fps), fill="black", font=font)
    return labelled


def assemble_montage(tiles, columns=2):
    require(isinstance(tiles, list) and 1 <= len(tiles) <= 8 and type(columns) is int and 1 <= columns <= 4, "Invalid montage layout")
    require(all(isinstance(tile, Image.Image) and tile.mode == "RGB" for tile in tiles), "Need actual RGB time tiles")
    width, height = 640, 504
    rows = math.ceil(len(tiles)/columns)
    output = Image.new("RGB", (columns*width, rows*height), "white")
    for i, tile in enumerate(tiles):
        fitted = tile.copy()
        fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
        x, y = (i % columns)*width+(width-fitted.width)//2, (i//columns)*height+(height-fitted.height)//2
        output.paste(fitted, (x, y))
    return output
