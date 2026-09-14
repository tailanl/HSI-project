"""Vectorized equivalent of released UniHSI get_height_map (no mesh mutation).

Preserves original floor division, outer maximum-cell exclusion, height <1.5,
cell maxima, zero lower clamp, and linspace coordinates. No changed resolution.
"""
import numpy as np


def get_height_map(points, HEIGHT_MAP_DIM=16):
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or HEIGHT_MAP_DIM < 1):
        raise ValueError('Expected finite nonempty XYZ and positive resolution')
    minx, miny = points[:, 0].min(), points[:, 1].min()
    maxx, maxy = points[:, 0].max(), points[:, 1].max()
    if maxx <= minx or maxy <= miny:
        raise ValueError('Degenerate scene XY extent')
    interval_x = (maxx-minx)/HEIGHT_MAP_DIM
    interval_y = (maxy-miny)/HEIGHT_MAP_DIM
    ix = (points[:, 0]-minx) // interval_x
    iy = (points[:, 1]-miny) // interval_y
    valid = (ix >= 0) & (ix < HEIGHT_MAP_DIM) & (iy >= 0) & (iy < HEIGHT_MAP_DIM)
    ids = iy[valid].astype(np.int64)*HEIGHT_MAP_DIM + ix[valid].astype(np.int64)
    heights = np.where(points[:, 2] < 1.5, points[:, 2], 0.0)
    height_flat = np.zeros(HEIGHT_MAP_DIM*HEIGHT_MAP_DIM)
    np.maximum.at(height_flat, ids, heights[valid])
    xx, yy = np.meshgrid(np.linspace(minx, maxx, HEIGHT_MAP_DIM),
                         np.linspace(miny, maxy, HEIGHT_MAP_DIM))
    return np.concatenate((np.stack((xx, yy), axis=-1).reshape(-1, 2),
                           height_flat[:, None]), axis=-1)
