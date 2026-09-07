"""Retained mask/depth, contact-facing and mesh projection mathematics."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap
from hsi.common.artifacts import require

LINGO_YUP_TO_WORLD_ZUP = np.asarray(((1.,0,0,0),(0,0,-1.,0),(0,1.,0,0),(0,0,0,1.)))
OPENCV_TO_OPENGL = np.diag((1.,-1.,-1.,1.))

POLICY = dict(surface_area_in_crop_min=.95, surface_mask_visible_fraction_min=.85,
    surface_mask_pixels_min=300, owner_bbox_retained_fraction_min=.98,
    owner_mask_visible_fraction_min=.90, head_probe_visible_fraction_min=.95,
    each_foot_probe_visible_fraction_min=.85, depth_tolerance_m=.03,
    surface_area_samples=5000, sample_seed=549, body_probe_is_not_a_pose_prediction=True,
    hypothetical_body_probes_are_ranking_only=True, camera_contact_facing_is_ranking_only=True,
    generated_body_visibility_checked_after_H3_by_actual_Qwen=True,
    downstream_full_mesh_physical_gates_unchanged=True)

def mask_visibility(isolated_depth, scene_depth, tolerance=.03):
    if isolated_depth.shape != scene_depth.shape:
        raise ValueError("Depth shapes differ")
    expected = np.isfinite(isolated_depth) & (isolated_depth > 0)
    visible = expected & np.isfinite(scene_depth) & (scene_depth > 0) & (np.abs(isolated_depth-scene_depth) <= tolerance)
    count = int(expected.sum())
    return float(visible.sum()/count) if count else 0.0, expected, visible

def project_probe(points, camera, scene_depth, crop, tolerance=.03):
    xyz = np.c_[points, np.ones(len(points))] @ np.asarray(camera["world_to_camera"]).T
    projected = xyz[:, :3] @ np.asarray(camera["K"]).T
    uv = projected[:, :2]/np.maximum(projected[:, 2:3], 1e-9)
    x0, y0, x1, y1 = crop
    height, width = scene_depth.shape
    inside = (xyz[:, 2] > .05) & np.isfinite(uv).all(1) & (uv[:, 0] >= max(0, x0)) & (uv[:, 1] >= max(0, y0)) \
        & (uv[:, 0] < min(width-1, x1)) & (uv[:, 1] < min(height-1, y1))
    visible = np.zeros(len(points), bool)
    ids = np.flatnonzero(inside)
    pixels = np.rint(uv[ids]).astype(int)
    observed = scene_depth[pixels[:, 1], pixels[:, 0]]
    visible[ids] = np.isfinite(observed) & (observed > 0) & (observed >= xyz[ids, 2]-tolerance)
    return uv, inside, visible

def area_samples(vertices, faces, count, seed):
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("Surface has no finite positive area")
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(area), count, p=area/area.sum())]
    u, v = rng.random((2, count))
    u = np.sqrt(u)
    return chosen[:, 0]*(1-u[:, None]) + chosen[:, 1]*(u*(1-v))[:, None] + chosen[:, 2]*(u*v)[:, None]

def _body_probes_approach(target, approach):
    chosen = next(row for row in target["candidate_surfaces"] if row["candidate_id"] == target["selected_candidate_id"])
    centre = np.asarray(chosen["surface"]["centre_world_xyz_zup_m"])
    bounds = np.asarray(chosen["surface"]["bounds_world_zup_m"])
    forward = np.asarray(approach["selected_approach_world_xy_m"])-centre[:2]
    if np.linalg.norm(forward) < 1e-6: raise ValueError("Undefined interaction side")
    forward /= np.linalg.norm(forward)
    side = np.array([-forward[1], forward[0]])
    edge_distance = max(np.dot(np.array([x, y])-centre[:2], forward) for x in bounds[:, 0] for y in bounds[:, 1])
    # Geometric room for a seated adult. This never substitutes for recovered articulation.
    head = np.array([[*(centre[:2]+forward*dx+side*dy), centre[2]+dz]
        for dx in (-.08, 0, .08) for dy in (-.10, 0, .10) for dz in (.76, .88, 1.0)])
    feet = []
    for sign in (-1, 1):
        base = centre[:2]+forward*(edge_distance+.18)+side*(sign*.13)
        feet.append(np.array([[*(base+forward*dx+side*dy), .035] for dx in (-.08, 0, .08) for dy in (-.05, 0, .05)]))
    return {"head": head, "left_foot": feet[0], "right_foot": feet[1]}

def facing_cosine(centre, yaw, camera_xy):
    vector = np.asarray(camera_xy, dtype=float) - np.asarray(centre, dtype=float)[:2]
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not math.isfinite(yaw) or norm <= 1e-8:
        return float("nan")
    return float(np.dot(vector / norm, [math.cos(yaw), math.sin(yaw)]))

def replace_arrival_gate(row, cosine, threshold):
    if "camera_on_stage1_approach_side" not in row["gates"]:
        raise ValueError("Expected legacy walking-arrival view gate is missing")
    original = row["gates"].pop("camera_on_stage1_approach_side")
    row["legacy_walking_arrival_side_gate_diagnostic_only"] = original
    row["camera_contact_facing_cosine"] = cosine if math.isfinite(cosine) else None
    row["gates"]["camera_on_contact_facing_side"] = math.isfinite(cosine) and cosine >= threshold
    row["all_geometry_gates_passed"] = all(row["gates"].values())
    row["failed_geometry_gates"] = sorted(k for k, v in row["gates"].items() if not v)
    return row

def soft_contact_gate(original, row, cosine, threshold):
    row = original(row, cosine, threshold)
    row["contact_facing_gate_ranking_only"] = row["gates"].pop("camera_on_contact_facing_side")
    row["all_geometry_gates_passed"] = all(row["gates"].values())
    row["failed_geometry_gates"] = sorted(k for k, v in row["gates"].items() if not v)
    return row

def load_scene_mesh(path: Path) -> Any:
    import trimesh

    loaded = trimesh.load(
        path.resolve(strict=True),
        process=False,
        maintain_order=True,
    )
    if isinstance(loaded, trimesh.Scene):
        parts = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            parts.append(geometry)
        require(parts, "scene mesh contains no geometry")
        loaded = trimesh.util.concatenate(parts)
    require(
        isinstance(loaded, trimesh.Trimesh)
        and len(loaded.vertices) > 0
        and len(loaded.faces) > 0,
        "scene OBJ did not load as a non-empty triangle mesh",
    )
    loaded.apply_transform(LINGO_YUP_TO_WORLD_ZUP)
    return loaded

def opencv_to_pyrender_pose(world_to_camera: np.ndarray) -> np.ndarray:
    value = np.asarray(world_to_camera, dtype=np.float64)
    require(value.shape == (4, 4), "world_to_camera must be 4x4")
    return np.linalg.inv(OPENCV_TO_OPENGL @ value)

def body_probes(target, approach):
    selected = next(r for r in target["candidate_surfaces"] if r["candidate_id"] == target["selected_candidate_id"])
    centre = np.asarray(selected["surface"]["centre_world_xyz_zup_m"])
    yaw = approach["contact_forward_yaw_rad"]
    require(math.isfinite(yaw), "Nonfinite current contact yaw")
    body_side = dict(approach, selected_approach_world_xy_m=(centre[:2] + np.array([np.cos(yaw), np.sin(yaw)])).tolist())
    return _body_probes_approach(target, body_side)


def board(items, output, title, *, columns=3, width=480, height=360):
    if not items: return None
    font = lambda size: ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size)
    canvas = Image.new("RGB", (columns*width, math.ceil(len(items)/columns)*(height+76)+96), "#f3f5f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((18,14), title, font=font(23), fill="#152739")
    draw.text((18,53), "HSI | Fixed scene understanding / current query / measured evidence", font=font(15), fill="#56677a")
    for index, (path, label) in enumerate(items):
        x,y = (index%columns)*width, 96+(index//columns)*(height+76)
        with Image.open(path) as raw:
            thumb = ImageOps.contain(raw.convert("RGB"), (width-12,height-8), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x+(width-thumb.width)//2, y+(height-thumb.height)//2))
        for row,line in enumerate(textwrap.wrap(label,width=62)[:3]):
            draw.text((x+10,y+height+row*21),line,font=font(14),fill="#243a51")
    canvas.save(output)
    return output
