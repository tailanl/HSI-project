#!/usr/bin/env python3
"""Render scene-first calibrated RGB/depth views of a LINGO room.

Camera anchors are derived only from the query occupancy grid.  The original
scene mesh is used only for rendering and visibility coverage measurement.
No support candidate, semantic target, motion, pose, contact label, planner
memory, P480 receipt, or HSI output is read.

The configured complete scene sweep places anchors across every substantial human-clear
free-space component, adds farthest-point anchors for room coverage, and
renders a complete yaw sweep at every anchor.  Every view retains RGB, metric
depth, intrinsics, OpenCV world-to-camera extrinsics, and artifact hashes.
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
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, label


SCHEMA = "p508.lingo_fullscene_multiview_render.v1"
CAMERA_SCHEMA = "p508.lingo_fullscene_calibrated_cameras.v1"
VIEW_CAMERA_SCHEMA = "p508.lingo_fullscene_camera.v1"

RESOLUTION_M = 0.02
OPENCV_TO_OPENGL = np.diag((1.0, -1.0, -1.0, 1.0)).astype(np.float64)
LINGO_YUP_TO_WORLD_ZUP = np.asarray(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


class FullSceneRenderError(RuntimeError):
    """Raised when a scene-first rendering invariant is not satisfied."""


@dataclass(frozen=True)
class Camera:
    camera_id: int
    anchor_id: int
    yaw_index: int
    yaw_degrees: float
    position_world_zup_m: np.ndarray
    target_world_zup_m: np.ndarray
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise FullSceneRenderError(message)


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


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def native_to_world_occupancy(native: np.ndarray) -> np.ndarray:
    """Convert native [X,Y-up,Z] occupancy to world [X,Y,Z-up]."""

    require(
        native.shape == (300, 100, 400) and native.dtype == np.bool_,
        f"unexpected LINGO occupancy contract: {native.shape}/{native.dtype}",
    )
    return np.ascontiguousarray(native.transpose(0, 2, 1)[:, ::-1, :])


def cell_to_world_xy(cell: np.ndarray) -> np.ndarray:
    value = np.asarray(cell, dtype=np.float64)
    require(value.shape[-1] == 2, "cell must end with XY indices")
    # The reversed world-Y index zero has native Z index 399.  Its cell centre
    # is y=-3.99 m, matching the established P498 LOWER=(-3,-4,0) contract.
    lower_world_xy = np.asarray((-3.0, -4.0), dtype=np.float64)
    return lower_world_xy + (value + 0.5) * RESOLUTION_M


def human_clear_space(
    native: np.ndarray,
    *,
    body_z_min_m: float,
    body_z_max_m: float,
    clearance_m: float,
    minimum_component_cells: int,
) -> dict[str, Any]:
    world = native_to_world_occupancy(native)
    z = (np.arange(world.shape[2], dtype=np.float64) + 0.5) * RESOLUTION_M
    band = (z >= body_z_min_m) & (z <= body_z_max_m)
    require(bool(np.any(band)), "body-height occupancy band is empty")
    obstacles = np.asarray(world[:, :, band].any(axis=2), dtype=bool)
    clearance = distance_transform_edt(~obstacles) * RESOLUTION_M
    free = (clearance >= clearance_m) & ~obstacles
    component_labels, component_count = label(
        free, structure=np.ones((3, 3), dtype=np.uint8)
    )
    counts = np.bincount(component_labels.ravel())
    kept_ids = [
        int(index)
        for index in range(1, len(counts))
        if int(counts[index]) >= minimum_component_cells
    ]
    kept_ids.sort(key=lambda index: (-int(counts[index]), index))
    require(kept_ids, "no substantial human-clear free-space component")
    retained = np.isin(component_labels, kept_ids)
    components = []
    for component_id in kept_ids:
        cells = np.argwhere(component_labels == component_id)
        bounds = cell_to_world_xy(
            np.stack((cells.min(axis=0), cells.max(axis=0)), axis=0)
        )
        components.append(
            {
                "component_id": component_id,
                "cell_count": int(len(cells)),
                "world_xy_bounds_m": bounds.astype(float).tolist(),
                "maximum_clearance_m": float(clearance[component_labels == component_id].max()),
            }
        )
    return {
        "world_occupancy": world,
        "obstacles": obstacles,
        "clearance_m": clearance,
        "free": free,
        "component_labels": component_labels,
        "retained_free": retained,
        "retained_component_ids": kept_ids,
        "components": components,
        "discarded_small_component_count": int(component_count - len(kept_ids)),
    }


def _representative_component_cell(
    cells: np.ndarray,
    clearance: np.ndarray,
) -> np.ndarray:
    values = clearance[tuple(cells.T)]
    maximum = float(values.max())
    shortlist = cells[values >= maximum - RESOLUTION_M]
    centre = np.mean(cells, axis=0)
    index = int(np.argmin(np.sum((shortlist - centre[None]) ** 2, axis=1)))
    return shortlist[index].astype(np.int64)


def select_anchor_cells(
    space: Mapping[str, Any],
    *,
    anchor_count: int,
    minimum_anchor_clearance_m: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Cover all retained components, then add deterministic farthest anchors."""

    labels = np.asarray(space["component_labels"], dtype=np.int32)
    clearance = np.asarray(space["clearance_m"], dtype=np.float64)
    retained = np.asarray(space["retained_free"], dtype=bool)
    component_ids = list(space["retained_component_ids"])
    require(
        anchor_count >= len(component_ids),
        "anchor count must cover every substantial human-clear component",
    )
    candidate_mask = retained & (clearance >= minimum_anchor_clearance_m)
    require(bool(np.any(candidate_mask)), "no cell satisfies anchor clearance")
    candidate_cells = np.argwhere(candidate_mask)

    selected: list[np.ndarray] = []
    reasons: list[str] = []
    for component_id in component_ids:
        cells = np.argwhere((labels == component_id) & candidate_mask)
        require(
            len(cells) > 0,
            f"component {component_id} has no cell at anchor clearance",
        )
        selected.append(_representative_component_cell(cells, clearance))
        reasons.append("maximum_clearance_representative_for_component")

    while len(selected) < anchor_count:
        selected_array = np.asarray(selected, dtype=np.float64)
        delta = candidate_cells[:, None, :] - selected_array[None, :, :]
        nearest_distance_cells = np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)
        candidate_clearance = clearance[tuple(candidate_cells.T)]
        score = nearest_distance_cells + 0.20 * (
            candidate_clearance / RESOLUTION_M
        )
        order = np.lexsort(
            (
                candidate_cells[:, 1],
                candidate_cells[:, 0],
                -score,
            )
        )
        chosen = None
        existing = {tuple(cell.tolist()) for cell in selected}
        for index in order:
            proposal = candidate_cells[int(index)]
            if tuple(proposal.tolist()) not in existing:
                chosen = proposal.astype(np.int64)
                break
        require(chosen is not None, "cannot find another unique anchor")
        selected.append(chosen)
        reasons.append("farthest_human_clear_cell_with_clearance_tiebreak")

    cells = np.asarray(selected, dtype=np.int64)
    rows = []
    for anchor_id, (cell, reason) in enumerate(zip(cells, reasons)):
        component_id = int(labels[tuple(cell)])
        rows.append(
            {
                "anchor_id": anchor_id,
                "component_id": component_id,
                "cell_xy": cell.astype(int).tolist(),
                "world_xy_m": cell_to_world_xy(cell).astype(float).tolist(),
                "clearance_m": float(clearance[tuple(cell)]),
                "selection_reason": reason,
            }
        )
    return cells, rows


def anchor_coverage_statistics(
    retained_free: np.ndarray,
    labels: np.ndarray,
    anchor_cells: np.ndarray,
    component_ids: Sequence[int],
) -> dict[str, Any]:
    free_cells = np.argwhere(retained_free)
    distances = np.sqrt(
        np.sum(
            (
                free_cells[:, None, :].astype(np.float64)
                - anchor_cells[None, :, :].astype(np.float64)
            )
            ** 2,
            axis=2,
        )
    ).min(axis=1) * RESOLUTION_M
    return {
        "retained_human_clear_cell_count": int(len(free_cells)),
        "nearest_anchor_distance_m": {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p90": float(np.percentile(distances, 90.0)),
            "maximum": float(np.max(distances)),
        },
        "fraction_within_0p75m": float(np.mean(distances <= 0.75)),
        "fraction_within_1p00m": float(np.mean(distances <= 1.00)),
        "fraction_within_1p50m": float(np.mean(distances <= 1.50)),
        "fraction_within_2p00m": float(np.mean(distances <= 2.00)),
        "retained_component_count": int(len(component_ids)),
        "components_with_anchor_count": int(
            sum(
                np.any(labels[tuple(anchor_cells.T)] == component_id)
                for component_id in component_ids
            )
        ),
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    require(norm > 1.0e-10, "zero-length camera direction")
    return value / norm


def make_camera(
    camera_id: int,
    anchor_id: int,
    yaw_index: int,
    yaw_degrees: float,
    anchor_xy_m: np.ndarray,
    *,
    eye_height_m: float,
    look_height_m: float,
    look_distance_m: float,
    width: int,
    height: int,
    vertical_fov_degrees: float,
) -> Camera:
    angle = math.radians(yaw_degrees)
    eye = np.asarray(
        (anchor_xy_m[0], anchor_xy_m[1], eye_height_m),
        dtype=np.float64,
    )
    target = eye + np.asarray(
        (
            look_distance_m * math.cos(angle),
            look_distance_m * math.sin(angle),
            look_height_m - eye_height_m,
        ),
        dtype=np.float64,
    )
    forward = _unit(target - eye)
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = _unit(np.cross(forward, world_up))
    down = _unit(np.cross(forward, right))
    rotation = np.stack((right, down, forward), axis=0)
    require(
        np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-8)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-8),
        "camera rotation is not a right-handed orthonormal matrix",
    )
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = -rotation @ eye
    focal = 0.5 * height / math.tan(
        math.radians(vertical_fov_degrees) * 0.5
    )
    intrinsics = np.asarray(
        (
            (focal, 0.0, 0.5 * width),
            (0.0, focal, 0.5 * height),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return Camera(
        camera_id=camera_id,
        anchor_id=anchor_id,
        yaw_index=yaw_index,
        yaw_degrees=float(yaw_degrees),
        position_world_zup_m=eye,
        target_world_zup_m=target,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        width=width,
        height=height,
    )


def build_cameras(
    anchor_cells: np.ndarray,
    *,
    yaw_count: int,
    eye_height_m: float,
    look_height_m: float,
    look_distance_m: float,
    width: int,
    height: int,
    vertical_fov_degrees: float,
) -> list[Camera]:
    require(yaw_count >= 3, "yaw count must be at least three")
    cameras: list[Camera] = []
    for anchor_id, cell in enumerate(anchor_cells):
        xy = cell_to_world_xy(cell)
        for yaw_index, yaw in enumerate(
            np.linspace(0.0, 360.0, yaw_count, endpoint=False)
        ):
            cameras.append(
                make_camera(
                    len(cameras),
                    anchor_id,
                    yaw_index,
                    float(yaw),
                    xy,
                    eye_height_m=eye_height_m,
                    look_height_m=look_height_m,
                    look_distance_m=look_distance_m,
                    width=width,
                    height=height,
                    vertical_fov_degrees=vertical_fov_degrees,
                )
            )
    return cameras


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


def image_statistics(rgb: np.ndarray, depth: np.ndarray) -> dict[str, Any]:
    image = np.asarray(rgb, dtype=np.float32)
    grey = (
        0.299 * image[..., 0]
        + 0.587 * image[..., 1]
        + 0.114 * image[..., 2]
    )
    positive = np.asarray(depth, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    require(len(positive) > 0, "rendered view contains no metric depth")
    return {
        "positive_depth_pixel_count": int(len(positive)),
        "positive_depth_fraction": float(len(positive) / depth.size),
        "positive_depth_m": {
            "minimum": float(positive.min()),
            "median": float(np.median(positive)),
            "p95": float(np.percentile(positive, 95.0)),
            "maximum": float(positive.max()),
        },
        "rgb_standard_deviation": float(image.std()),
        "grey_robust_range_p95_minus_p05": float(
            np.percentile(grey, 95.0) - np.percentile(grey, 5.0)
        ),
    }


def backproject_surface_voxels(
    depth: np.ndarray,
    camera: Camera,
    *,
    stride: int,
    voxel_m: float,
) -> set[tuple[int, int, int]]:
    yy, xx = np.mgrid[0 : camera.height : stride, 0 : camera.width : stride]
    values = np.asarray(depth[::stride, ::stride], dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.03) & (values < 20.0)
    if not np.any(valid):
        return set()
    u = xx[valid].astype(np.float64)
    v = yy[valid].astype(np.float64)
    z = values[valid]
    k = camera.intrinsics
    xyz1 = np.stack(
        (
            (u - k[0, 2]) * z / k[0, 0],
            (v - k[1, 2]) * z / k[1, 1],
            z,
            np.ones_like(z),
        ),
        axis=1,
    )
    world = (
        np.linalg.inv(camera.world_to_camera) @ xyz1.T
    ).T[:, :3]
    voxels = np.floor(world / voxel_m).astype(np.int64)
    unique = np.unique(voxels, axis=0)
    return {tuple(row.tolist()) for row in unique}


def sampled_vertex_visibility(
    vertices_world: np.ndarray,
    depth: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    vertices = np.asarray(vertices_world, dtype=np.float64)
    homogeneous = np.concatenate(
        (vertices, np.ones((len(vertices), 1), dtype=np.float64)),
        axis=1,
    )
    camera_xyz = (
        homogeneous @ camera.world_to_camera.T
    )[:, :3]
    z = camera_xyz[:, 2]
    good = np.isfinite(z) & (z > 0.03)
    u = np.rint(
        camera.intrinsics[0, 0]
        * camera_xyz[:, 0]
        / np.maximum(z, 1.0e-8)
        + camera.intrinsics[0, 2]
    ).astype(np.int64)
    v = np.rint(
        camera.intrinsics[1, 1]
        * camera_xyz[:, 1]
        / np.maximum(z, 1.0e-8)
        + camera.intrinsics[1, 2]
    ).astype(np.int64)
    good &= (
        (u >= 0)
        & (u < camera.width)
        & (v >= 0)
        & (v < camera.height)
    )
    indices = np.flatnonzero(good)
    visible = np.zeros(len(vertices), dtype=bool)
    if not len(indices):
        return visible
    rendered = depth[v[indices], u[indices]]
    tolerance = np.maximum(0.035, 0.015 * z[indices])
    passed = (
        np.isfinite(rendered)
        & (rendered > 0.0)
        & (np.abs(rendered - z[indices]) <= tolerance)
    )
    visible[indices[passed]] = True
    return visible


def render_views(
    mesh: Any,
    cameras: Sequence[Camera],
    output_dir: Path,
    *,
    scene_id: str,
    depth_stride: int,
    surface_voxel_m: float,
    maximum_vertex_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyrender

    require(cameras, "camera list is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    camera_dir = output_dir / "cameras"
    for directory in (rgb_dir, depth_dir, camera_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mesh_node = pyrender.Mesh.from_trimesh(
        mesh.copy(), smooth=False
    )
    scene = pyrender.Scene(
        bg_color=np.asarray((238, 239, 241, 255), dtype=np.uint8),
        ambient_light=np.asarray((0.45, 0.45, 0.45), dtype=np.float32),
    )
    scene.add(mesh_node)
    top_light = pyrender.DirectionalLight(
        color=np.ones(3), intensity=1.0
    )
    top_pose = np.eye(4, dtype=np.float64)
    top_pose[:3, 3] = (0.0, 0.0, 4.0)
    scene.add(top_light, pose=top_pose)
    renderer = pyrender.OffscreenRenderer(
        cameras[0].width, cameras[0].height
    )

    vertex_count = len(mesh.vertices)
    sample_count = min(vertex_count, maximum_vertex_samples)
    sample_ids = np.linspace(
        0, vertex_count - 1, sample_count, dtype=np.int64
    )
    sampled_vertices = np.asarray(mesh.vertices, dtype=np.float64)[sample_ids]
    visible_union = np.zeros(sample_count, dtype=bool)
    surface_voxel_union: set[tuple[int, int, int]] = set()
    rows: list[dict[str, Any]] = []
    try:
        for camera in cameras:
            pose = opencv_to_pyrender_pose(camera.world_to_camera)
            camera_model = pyrender.IntrinsicsCamera(
                fx=float(camera.intrinsics[0, 0]),
                fy=float(camera.intrinsics[1, 1]),
                cx=float(camera.intrinsics[0, 2]),
                cy=float(camera.intrinsics[1, 2]),
                znear=0.03,
                zfar=20.0,
            )
            camera_node = scene.add(camera_model, pose=pose)
            light_node = scene.add(
                pyrender.DirectionalLight(
                    color=np.ones(3), intensity=1.75
                ),
                pose=pose,
            )
            try:
                color, depth = renderer.render(
                    scene, flags=pyrender.RenderFlags.RGBA
                )
            finally:
                scene.remove_node(camera_node)
                scene.remove_node(light_node)
            rgb = np.asarray(color[..., :3], dtype=np.uint8)
            depth = np.asarray(depth, dtype=np.float32)
            require(
                rgb.shape == (camera.height, camera.width, 3)
                and depth.shape == (camera.height, camera.width),
                "renderer returned unexpected raster dimensions",
            )
            stats = image_statistics(rgb, depth)
            require(
                stats["positive_depth_fraction"] >= 0.10,
                f"view {camera.camera_id} has insufficient scene depth coverage",
            )

            rgb_path = rgb_dir / f"view_{camera.camera_id:02d}.png"
            depth_path = depth_dir / (
                f"view_{camera.camera_id:02d}_metric_depth_m.npy"
            )
            camera_path = camera_dir / f"view_{camera.camera_id:02d}.json"
            Image.fromarray(rgb, mode="RGB").save(
                rgb_path, optimize=True
            )
            np.save(
                depth_path,
                depth,
                allow_pickle=False,
            )
            camera_record = {
                "schema": VIEW_CAMERA_SCHEMA,
                "camera_id": camera.camera_id,
                "anchor_id": camera.anchor_id,
                "yaw_index": camera.yaw_index,
                "yaw_degrees": camera.yaw_degrees,
                "coordinate_system": "world_zup_metric",
                "extrinsic_convention": "opencv_world_to_camera",
                "intrinsics_convention": (
                    "pixels_fx_fy_cx_cy_origin_top_left"
                ),
                "width": camera.width,
                "height": camera.height,
                "K": camera.intrinsics.astype(float).tolist(),
                "intrinsics": camera.intrinsics.astype(float).tolist(),
                "world_to_camera": (
                    camera.world_to_camera.astype(float).tolist()
                ),
                "position_world_zup_m": (
                    camera.position_world_zup_m.astype(float).tolist()
                ),
                "target_world_zup_m": (
                    camera.target_world_zup_m.astype(float).tolist()
                ),
                "depth_units": "metres",
                "depth_zero_semantics": "no_rendered_surface",
            }
            atomic_json(camera_path, camera_record)

            rgb_artifact = artifact(rgb_path)
            depth_artifact = artifact(depth_path)
            camera_artifact = artifact(camera_path)

            view_voxels = backproject_surface_voxels(
                depth,
                camera,
                stride=depth_stride,
                voxel_m=surface_voxel_m,
            )
            new_voxel_count = len(view_voxels - surface_voxel_union)
            surface_voxel_union.update(view_voxels)
            visible = sampled_vertex_visibility(
                sampled_vertices, depth, camera
            )
            new_visible_count = int(
                np.count_nonzero(visible & ~visible_union)
            )
            visible_union |= visible
            rows.append(
                {
                    "view_index": camera.camera_id,
                    "view_id": (
                        f"{scene_id}_fullscene_anchor"
                        f"{camera.anchor_id:02d}_yaw"
                        f"{camera.yaw_index:02d}"
                    ),
                    "anchor_id": camera.anchor_id,
                    "yaw_index": camera.yaw_index,
                    "yaw_degrees": camera.yaw_degrees,
                    "width": camera.width,
                    "height": camera.height,
                    "K": camera.intrinsics.astype(float).tolist(),
                    "intrinsics": (
                        camera.intrinsics.astype(float).tolist()
                    ),
                    "world_to_camera": (
                        camera.world_to_camera.astype(float).tolist()
                    ),
                    "position_world_zup_m": (
                        camera.position_world_zup_m.astype(float).tolist()
                    ),
                    "target_world_zup_m": (
                        camera.target_world_zup_m.astype(float).tolist()
                    ),
                    "rgb": rgb_artifact,
                    "depth": depth_artifact,
                    "metric_depth_m": depth_artifact,
                    "camera": camera_artifact,
                    "raster_statistics": stats,
                    "coverage": {
                        "sampled_visible_mesh_vertex_count": int(
                            np.count_nonzero(visible)
                        ),
                        "new_sampled_visible_mesh_vertex_count": (
                            new_visible_count
                        ),
                        "backprojected_surface_voxel_count": len(
                            view_voxels
                        ),
                        "new_backprojected_surface_voxel_count": (
                            new_voxel_count
                        ),
                    },
                }
            )
    finally:
        renderer.delete()

    coverage = {
        "sampled_mesh_vertex_count": sample_count,
        "union_visible_sampled_mesh_vertex_count": int(
            np.count_nonzero(visible_union)
        ),
        "union_visible_sampled_mesh_vertex_fraction": float(
            np.mean(visible_union)
        ),
        "sampled_mesh_vertex_denominator_note": (
            "Deterministic samples cover the complete watertight mesh, "
            "including exterior/back surfaces that an interior camera "
            "cannot observe."
        ),
        "surface_voxel_size_m": surface_voxel_m,
        "depth_backprojection_stride_pixels": depth_stride,
        "union_backprojected_surface_voxel_count": len(
            surface_voxel_union
        ),
        "minimum_positive_depth_fraction": float(
            min(
                row["raster_statistics"][
                    "positive_depth_fraction"
                ]
                for row in rows
            )
        ),
        "maximum_positive_depth_fraction": float(
            max(
                row["raster_statistics"][
                    "positive_depth_fraction"
                ]
                for row in rows
            )
        ),
        "minimum_rgb_standard_deviation": float(
            min(
                row["raster_statistics"][
                    "rgb_standard_deviation"
                ]
                for row in rows
            )
        ),
    }
    return rows, coverage


def draw_contact_sheet(
    views: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    columns: int = 4,
) -> None:
    require(views, "cannot draw an empty contact sheet")
    panel_width, panel_height, header = 320, 240, 34
    rows = math.ceil(len(views) / columns)
    canvas = Image.new(
        "RGB",
        (columns * panel_width, rows * (panel_height + header)),
        (232, 234, 238),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, view in enumerate(views):
        path = Path(str(view["rgb"]["path"])).resolve(strict=True)
        image = Image.open(path).convert("RGB")
        image.thumbnail(
            (panel_width, panel_height),
            Image.Resampling.LANCZOS,
        )
        x = (index % columns) * panel_width
        y = (index // columns) * (panel_height + header)
        canvas.paste(
            image,
            (
                x + (panel_width - image.width) // 2,
                y + header + (panel_height - image.height) // 2,
            ),
        )
        draw.rectangle(
            (x, y, x + panel_width - 1, y + header - 1),
            fill=(19, 29, 42),
        )
        draw.text(
            (x + 8, y + 11),
            (
                f"view {int(view['view_index']):02d} | "
                f"anchor {int(view['anchor_id'])} | "
                f"yaw {float(view['yaw_degrees']):.0f} deg"
            ),
            fill=(235, 242, 250),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def draw_anchor_map(
    space: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    *,
    yaw_count: int,
    output_path: Path,
) -> None:
    retained = np.asarray(space["retained_free"], dtype=bool)
    obstacles = np.asarray(space["obstacles"], dtype=bool)
    labels = np.asarray(space["component_labels"], dtype=np.int32)
    rgb = np.full((*retained.shape, 3), 224, dtype=np.uint8)
    rgb[obstacles] = np.asarray((38, 44, 52), dtype=np.uint8)
    palette = (
        (204, 238, 255),
        (215, 255, 214),
        (255, 232, 195),
        (240, 214, 255),
    )
    for index, component_id in enumerate(
        space["retained_component_ids"]
    ):
        rgb[labels == component_id] = np.asarray(
            palette[index % len(palette)], dtype=np.uint8
        )
    image = Image.fromarray(
        np.transpose(rgb, (1, 0, 2))[::-1],
        mode="RGB",
    ).resize(
        (retained.shape[0] * 2, retained.shape[1] * 2),
        Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(image)
    for anchor in anchors:
        cell_x, cell_y = anchor["cell_xy"]
        x = int((cell_x + 0.5) * 2)
        y = int((retained.shape[1] - cell_y - 0.5) * 2)
        radius = 8
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 45, 112),
            outline=(255, 255, 255),
            width=2,
        )
        for yaw in np.linspace(
            0.0, 360.0, yaw_count, endpoint=False
        ):
            angle = math.radians(float(yaw))
            endpoint = (
                x + int(round(22 * math.cos(angle))),
                y - int(round(22 * math.sin(angle))),
            )
            draw.line((x, y, *endpoint), fill=(210, 20, 85), width=2)
        draw.text(
            (x + 10, y - 10),
            f"A{anchor['anchor_id']}",
            fill=(20, 20, 20),
            font=ImageFont.load_default(),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    scene_id = str(args.scene_id).strip()
    require(bool(scene_id), "scene ID must be non-empty")
    mesh_path = args.mesh.expanduser().resolve(strict=True)
    occupancy_path = args.occupancy.expanduser().resolve(strict=True)
    require(
        mesh_path.parent.name == scene_id,
        "scene ID does not match the original mesh parent directory",
    )
    require(
        occupancy_path.stem == scene_id,
        "scene ID does not match the occupancy filename",
    )
    output_dir = args.output_dir.expanduser().resolve()
    require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        f"output directory is not empty: {output_dir}",
    )
    require(
        args.anchor_count * args.yaw_count >= 12,
        "configuration must emit at least twelve views",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    native = np.load(occupancy_path, allow_pickle=False)
    space = human_clear_space(
        native,
        body_z_min_m=args.body_z_min_m,
        body_z_max_m=args.body_z_max_m,
        clearance_m=args.human_clearance_m,
        minimum_component_cells=args.minimum_component_cells,
    )
    anchor_cells, anchor_rows = select_anchor_cells(
        space,
        anchor_count=args.anchor_count,
        minimum_anchor_clearance_m=args.anchor_min_clearance_m,
    )
    anchor_coverage = anchor_coverage_statistics(
        np.asarray(space["retained_free"], dtype=bool),
        np.asarray(space["component_labels"], dtype=np.int32),
        anchor_cells,
        list(space["retained_component_ids"]),
    )
    cameras = build_cameras(
        anchor_cells,
        yaw_count=args.yaw_count,
        eye_height_m=args.eye_height_m,
        look_height_m=args.look_height_m,
        look_distance_m=args.look_distance_m,
        width=args.width,
        height=args.height,
        vertical_fov_degrees=args.vertical_fov_degrees,
    )
    require(len(cameras) >= 12, "fewer than twelve cameras were built")

    mesh = load_scene_mesh(mesh_path)
    views, render_coverage = render_views(
        mesh,
        cameras,
        output_dir,
        scene_id=scene_id,
        depth_stride=args.depth_coverage_stride,
        surface_voxel_m=args.surface_voxel_m,
        maximum_vertex_samples=args.maximum_vertex_samples,
    )
    require(len(views) == len(cameras), "not every camera was rendered")

    contact_sheet_path = output_dir / (
        "fullscene_rgb_contact_sheet.png"
    )
    draw_contact_sheet(views, contact_sheet_path)
    anchor_map_path = output_dir / "human_clear_anchor_map.png"
    draw_anchor_map(
        space,
        anchor_rows,
        yaw_count=args.yaw_count,
        output_path=anchor_map_path,
    )
    camera_manifest_path = output_dir / "calibrated_cameras.json"
    manifest = {
        "schema": CAMERA_SCHEMA,
        "status": "fullscene_calibrated_views_ready",
        "scene_id": scene_id,
        "coordinate_system": "world_zup_metric",
        "extrinsic_convention": "opencv_world_to_camera",
        "intrinsics_convention": (
            "pixels_fx_fy_cx_cy_origin_top_left"
        ),
        "depth_units": "metres",
        "view_count": len(views),
        "anchors": anchor_rows,
        "views": views,
    }
    manifest["payload_sha256"] = canonical_hash(manifest)
    atomic_json(camera_manifest_path, manifest)

    source_mesh = artifact(mesh_path)
    source_occupancy = artifact(occupancy_path)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "fullscene_multiview_ready",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "scene_id": scene_id,
        "inputs": {
            "original_scene_mesh": source_mesh,
            "query_occupancy": source_occupancy,
        },
        "mesh": {
            "coordinate_transform": (
                "LINGO native [x,y_up,z] to world "
                "Z-up [x,-z,y_up]"
            ),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "bounds_world_zup_m": (
                np.asarray(mesh.bounds, dtype=np.float64)
                .astype(float)
                .tolist()
            ),
        },
        "human_clear_space": {
            "resolution_m": RESOLUTION_M,
            "body_height_band_world_zup_m": [
                args.body_z_min_m,
                args.body_z_max_m,
            ],
            "human_clearance_m": args.human_clearance_m,
            "minimum_component_cells": (
                args.minimum_component_cells
            ),
            "components": space["components"],
            "discarded_small_component_count": space[
                "discarded_small_component_count"
            ],
            "anchor_coverage": anchor_coverage,
            "anchors": anchor_rows,
            "anchor_map": artifact(anchor_map_path),
        },
        "camera_sweep": {
            "anchor_count": args.anchor_count,
            "yaw_count_per_anchor": args.yaw_count,
            "view_count": len(views),
            "eye_height_m": args.eye_height_m,
            "look_height_m": args.look_height_m,
            "look_distance_m": args.look_distance_m,
            "vertical_fov_degrees": (
                args.vertical_fov_degrees
            ),
            "horizontal_fov_degrees": float(
                math.degrees(
                    2.0
                    * math.atan(
                        (args.width / args.height)
                        * math.tan(
                            math.radians(
                                args.vertical_fov_degrees
                            )
                            * 0.5
                        )
                    )
                )
            ),
            "width": args.width,
            "height": args.height,
            "complete_yaw_sweep_at_every_anchor": True,
        },
        "coverage": render_coverage,
        "calibrated_camera_manifest": artifact(
            camera_manifest_path
        ),
        "fullscene_rgb_contact_sheet": artifact(
            contact_sheet_path
        ),
        "view_count": len(views),
        "views": views,
        "query_contract": {
            "source_data_files_read": [
                "original_scene_mesh",
                "query_occupancy",
            ],
            "p480_candidate_or_receipt_read": False,
            "support_candidate_read": False,
            "semantic_target_read": False,
            "instruction_read": False,
            "motion_pose_contact_label_read": False,
            "planner_memory_or_hsi_read": False,
            "camera_anchor_source": (
                "human_clear_query_occupancy_only"
            ),
            "camera_yaw_source": (
                "uniform_complete_sweep_only"
            ),
            "mesh_use": (
                "RGB_depth_rendering_and_coverage_measurement_only"
            ),
        },
        "source_code": artifact(Path(__file__)),
    }
    result["receipt_payload_sha256"] = canonical_hash(result)
    atomic_json(output_dir / "receipt.json", result)
    return result


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-id",
        required=True,
        help="Exact LINGO scene identifier; must match mesh parent and occupancy stem.",
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument(
        "--occupancy",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--anchor-count", type=int, default=6)
    parser.add_argument("--yaw-count", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--vertical-fov-degrees",
        type=float,
        default=75.0,
    )
    parser.add_argument(
        "--eye-height-m", type=float, default=1.35
    )
    parser.add_argument(
        "--look-height-m", type=float, default=0.82
    )
    parser.add_argument(
        "--look-distance-m", type=float, default=1.80
    )
    parser.add_argument(
        "--body-z-min-m", type=float, default=0.08
    )
    parser.add_argument(
        "--body-z-max-m", type=float, default=1.75
    )
    parser.add_argument(
        "--human-clearance-m", type=float, default=0.28
    )
    parser.add_argument(
        "--anchor-min-clearance-m",
        type=float,
        default=0.34,
    )
    parser.add_argument(
        "--minimum-component-cells",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--depth-coverage-stride", type=int, default=4
    )
    parser.add_argument(
        "--surface-voxel-m", type=float, default=0.08
    )
    parser.add_argument(
        "--maximum-vertex-samples",
        type=int,
        default=200_000,
    )
    args = parser.parse_args(argv)
    require(args.anchor_count > 0, "anchor count must be positive")
    require(args.yaw_count >= 3, "yaw count must be at least three")
    require(
        args.anchor_count * args.yaw_count >= 12,
        "anchor-count times yaw-count must be at least twelve",
    )
    require(
        args.width >= 128 and args.height >= 128,
        "raster dimensions are too small",
    )
    require(
        30.0 <= args.vertical_fov_degrees <= 120.0,
        "vertical FOV is outside [30,120] degrees",
    )
    require(
        0.0 < args.body_z_min_m < args.body_z_max_m,
        "body height band is invalid",
    )
    require(
        args.anchor_min_clearance_m
        >= args.human_clearance_m,
        "anchor clearance must be at least human clearance",
    )
    require(
        args.depth_coverage_stride >= 1,
        "depth coverage stride must be positive",
    )
    require(
        args.surface_voxel_m > 0.0,
        "surface voxel size must be positive",
    )
    require(
        args.maximum_vertex_samples >= 1_000,
        "maximum vertex samples is too small",
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:  # noqa: BLE001 - CLI must fail closed
        print(f"ERROR: {type(error).__name__}: {error}")
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "view_count": result["view_count"],
                "output_dir": str(args.output_dir.resolve()),
                "union_surface_voxels": result["coverage"][
                    "union_backprojected_surface_voxel_count"
                ],
                "sampled_vertex_coverage": result["coverage"][
                    "union_visible_sampled_mesh_vertex_fraction"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
