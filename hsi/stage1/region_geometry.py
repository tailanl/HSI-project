"""Actual contact-mesh front -> directional band -> obstacle/reachability clipping.

No text/model calls, furniture-category inference, centre-radius rings, or
off-region goal snapping. All coordinates are computed in the fixed Z-up map.
"""

from dataclasses import asdict, dataclass

import math

import os

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"

import numpy as np

from PIL import Image, ImageDraw

from scipy.ndimage import label

@dataclass(frozen=True)
class RegionPolicy:
    minimum_front_gap_m: float = .28
    maximum_front_gap_m: float = .55
    frontier_tolerance_m: float = .04
    lane_width_m: float = .02
    lateral_margin_m: float = .02
    candidate_separation_m: float = .10
    maximum_candidates: int = 24

def grid_points(shape, lower, resolution):
    indices = np.indices(shape, dtype=np.float64).transpose(1, 2, 0)
    return np.asarray(lower) + (indices + .5) * resolution

def contact_footprint(triangles, shape, lower, resolution):
    # A union of actual triangles, not an AABB or convex hull across voids.
    canvas = Image.new("1", (shape[1], shape[0]))
    draw = ImageDraw.Draw(canvas)
    cells = (np.asarray(triangles)[..., :2] - np.asarray(lower)) / resolution - .5
    for triangle in cells:
        draw.polygon([tuple(point[::-1]) for point in triangle], fill=1)
    return np.asarray(canvas, dtype=bool)

def front_profile(triangles, direction, lane_width):
    """Intersect each tangent lane with mesh edges; retain the frontmost hit.

    The returned 3D anchors lie on real contact triangles, including interpolated
    height. Empty lateral intervals are never filled across missing surfaces.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    tangent = np.array([-direction[1], direction[0]])
    begin = triangles.reshape(-1, 3)
    end = triangles[:, [1, 2, 0]].reshape(-1, 3)
    t0, t1 = begin[:, :2] @ tangent, end[:, :2] @ tangent
    minimum, maximum = float(t0.min()), float(t0.max())
    origin = math.floor(minimum / lane_width) * lane_width
    count = int(math.ceil((maximum - origin) / lane_width))
    lane_t = origin + (np.arange(count) + .5) * lane_width
    anchors = np.full((count, 3), np.nan)
    source_triangle = np.full(count, -1, dtype=np.int64)
    delta = t1 - t0
    for index, value in enumerate(lane_t):
        valid = (abs(delta) > 1e-10) & (value >= np.minimum(t0, t1)) & (value <= np.maximum(t0, t1))
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        weights = (value - t0[ids]) / delta[ids]
        intersections = begin[ids] + weights[:, None] * (end[ids] - begin[ids])
        selected = int(np.argmax(intersections[:, :2] @ direction))
        anchors[index] = intersections[selected]
        source_triangle[index] = ids[selected] // 3
    return {"direction": direction, "tangent": tangent, "origin": origin, "lane_t": lane_t,
        "anchors": anchors, "source_triangle": source_triangle, "t_bounds": [minimum, maximum]}

def clear_connectors(points, anchors, other_obstacles, lower, resolution):
    """Super-sample the segment from approach to the contact-front anchor.

    This checks intervening non-target objects. It is not a generated body
    transition and does not replace Stage2/Stage3 full-mesh collision checks.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    steps = max(2, int(np.ceil(np.linalg.norm(points - anchors, axis=1).max() / (resolution * .25))) + 1)
    fractions = np.linspace(0, 1, steps)
    sampled = points[:, None, :] * (1 - fractions[None, :, None]) + anchors[:, None, :] * fractions[None, :, None]
    cells = np.floor((sampled - np.asarray(lower)) / resolution).astype(np.int64)
    valid = np.all((cells >= 0) & (cells < np.asarray(other_obstacles.shape)), axis=2)
    safe_cells = np.clip(cells, 0, np.asarray(other_obstacles.shape) - 1)
    blocked = other_obstacles[safe_cells[..., 0], safe_cells[..., 1]] | ~valid
    return ~blocked.any(axis=1)

def construct(triangles, directions, navigation, target_footprint, reachable, lower, resolution,
              surface_centre, policy=RegionPolicy()):
    if not 0 < policy.minimum_front_gap_m <= policy.maximum_front_gap_m or policy.lane_width_m <= 0:
        raise ValueError("Invalid contact-region policy")
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("Expected finite actual contact triangles")
    free = np.asarray(navigation["free"], dtype=bool)
    obstacles = np.asarray(navigation["obstacles"], dtype=bool)
    target = np.asarray(target_footprint, dtype=bool)
    reachable = np.asarray(reachable, dtype=bool)
    if not (free.shape == obstacles.shape == target.shape == reachable.shape):
        raise ValueError("Grid shapes disagree")
    if np.any(reachable & ~free):
        raise ValueError("Reachability includes non-human-free cells")
    xy = grid_points(free.shape, lower, resolution)
    other = obstacles & ~target
    masks = {"contact_footprint": contact_footprint(triangles, free.shape, lower, resolution),
        "target_footprint": target, "other_obstacles": other, "human_free": free,
        "reachable": reachable, "raw_region": np.zeros_like(free), "clipped_region": np.zeros_like(free),
        "reachable_region": np.zeros_like(free), "terminal_frontier": np.zeros_like(free)}
    profiles, candidates, counts = [], [], []
    if not directions or len({r["candidate_id"] for r in directions}) != len(directions):
        raise ValueError("Expected unique fixed direction hypotheses")
    for item in directions:
        vector = np.asarray(item["outward_world_xy"], dtype=float)
        if vector.shape != (2,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
            raise ValueError("Invalid fixed furniture direction")
        profile = front_profile(triangles, vector, policy.lane_width_m)
        direction, tangent = profile["direction"], profile["tangent"]
        t = xy @ tangent
        lane = np.floor((t - profile["origin"]) / policy.lane_width_m).astype(np.int64)
        valid_lane = (lane >= 0) & (lane < len(profile["anchors"]))
        lane = np.clip(lane, 0, len(profile["anchors"]) - 1)
        anchor = profile["anchors"][lane]
        gap = (xy - anchor[..., :2]) @ direction
        tmin, tmax = profile["t_bounds"]
        raw = valid_lane & np.isfinite(anchor).all(axis=2) & (t >= tmin + policy.lateral_margin_m) \
            & (t <= tmax - policy.lateral_margin_m) & (gap >= policy.minimum_front_gap_m - 1e-9) \
            & (gap <= policy.maximum_front_gap_m + 1e-9)
        # A directional contact face must lie on its declared interaction side.
        raw &= ((xy - np.asarray(surface_centre)[:2]) @ direction) >= -1e-9
        free_band = raw & free & ~obstacles
        cells = np.argwhere(free_band)
        unobstructed = clear_connectors(xy[free_band], anchor[free_band, :2], other, lower, resolution)
        clipped = np.zeros_like(free)
        if len(cells):
            clipped[tuple(cells[unobstructed].T)] = True
        current = clipped & reachable
        frontier = np.zeros_like(free)
        for index in np.unique(lane[current]):
            local = current & (lane == index)
            frontier |= local & (gap <= float(gap[local].min()) + policy.frontier_tolerance_m + 1e-9)
        component_map, component_count = label(current, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        centre_t = float(np.asarray(surface_centre)[:2] @ tangent)
        for cell in np.argwhere(frontier):
            key = tuple(cell)
            candidates.append({"cell": cell.tolist(), "approach_world_xy_m": xy[key].tolist(),
                "contact_anchor_world_xyz_m": anchor[key].tolist(), "front_gap_m": float(gap[key]),
                "lane_id": int(lane[key]), "contact_triangle_local_index": int(profile["source_triangle"][lane[key]]),
                "direction_id": item["candidate_id"], "outward_world_xy": direction.tolist(),
                "component_id": int(component_map[key]), "clearance_m": float(navigation["clearance"][key]),
                "lateral_centre_distance_m": abs(float(t[key]) - centre_t)})
        for name, value in (("raw_region", raw), ("clipped_region", clipped), ("reachable_region", current), ("terminal_frontier", frontier)):
            masks[name] |= value
            masks[item["candidate_id"] + "_" + name] = value
        profiles.append({"direction_id": item["candidate_id"], "outward_world_xy": direction.tolist(),
            "lane_count": len(profile["anchors"]), "nonempty_contact_lanes": int(np.isfinite(profile["anchors"]).all(axis=1).sum())})
        counts.append({"direction_id": item["candidate_id"], "raw_cells": int(raw.sum()),
            "human_free_cells": int(free_band.sum()), "unobstructed_cells": int(clipped.sum()),
            "reachable_cells": int(current.sum()), "frontier_cells": int(frontier.sum()), "components": component_count})
    ordered = sorted(candidates, key=lambda r: (r["front_gap_m"], r["lateral_centre_distance_m"], -r["clearance_m"], r["cell"]))
    selected = []
    # Keep at least one endpoint per reachable component before spacing samples.
    groups = sorted({(r["direction_id"], r["component_id"]) for r in ordered})
    if len(groups) > policy.maximum_candidates:
        raise ValueError("Too many disconnected endpoint regions for the explicit candidate budget")
    for group in groups:
        selected.append(next(r for r in ordered if (r["direction_id"], r["component_id"]) == group))
    for candidate in ordered:
        if len(selected) >= policy.maximum_candidates:
            break
        if any(np.linalg.norm(np.asarray(candidate["approach_world_xy_m"]) - r["approach_world_xy_m"]) < policy.candidate_separation_m for r in selected):
            continue
        selected.append(candidate)
    audit = {"policy": asdict(policy), "directions": profiles, "clipping_counts": counts,
        "candidate_count": len(selected), "status": "ready" if selected else "no_reachable_contact_front_region",
        "coordinates_from_geometry_only": True, "contact_anchors_on_actual_mesh_edges": True,
        "centre_radius_sampling_used": False, "nearby_non_target_obstacles_clip_connectors": True,
        "candidate_centres_are_exact_reachable_grid_cells": True, "off_region_snapping_allowed": False,
        "seated_pelvis_is_not_navigation_goal": True, "stage3_contact_transition_validated": False}
    return selected, masks, audit

