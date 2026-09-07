"""Exact measured triangle support; no bounding-box filling or model calls."""
import numpy as np
from hsi.common.artifacts import require


def triangle_heights(points_xy, triangles):
    """Exact triangle membership; holes remain unknown, no AABB filling."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3) and len(tris) > 0, "Invalid support triangles")
    require(np.isfinite(tris).all() and np.isfinite(points).all(), "Non-finite support query")
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    u, v = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    good = np.abs(det) > 1e-12
    safe = np.where(good, det, 1.)
    heights = np.zeros(len(points), dtype=np.float64)
    valid = np.zeros(len(points), dtype=bool)
    for first in range(0, len(points), 64):
        d = points[first:first + 64, None, :] - a[None, :, :2]
        beta = (d[..., 0] * v[:, 1] - d[..., 1] * v[:, 0]) / safe
        gamma = (u[:, 0] * d[..., 1] - u[:, 1] * d[..., 0]) / safe
        inside = good & (beta >= -1e-8) & (gamma >= -1e-8) & (beta + gamma <= 1 + 1e-8)
        z = a[:, 2] + beta * (b[:, 2] - a[:, 2]) + gamma * (c[:, 2] - a[:, 2])
        found = inside.any(axis=1)
        top = np.max(np.where(inside, z, -np.inf), axis=1)
        heights[first:first + len(found)] = np.where(found, top, 0.)
        valid[first:first + len(found)] = found
    return heights, valid

