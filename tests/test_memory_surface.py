"""Synthetic CPU geometry checks; no scene/model execution."""
from pathlib import Path
import sys

import numpy as np
import pytest

from hsi.memory import surface as g


def square():
    return np.array([[0., 0., .4], [1., 0., .4], [1., 1., .4], [0., 1., .4]]), np.array([[0, 1, 2], [0, 2, 3]])


def descriptor():
    v, f = square()
    return g.describe_surface(v, f, [0, 1], [.5, .5, .4], [1, 0])


def test_exact_triangle_hole_is_not_filled():
    # Four rectangles surrounding an actual square hole; not a convex hull.
    rectangles = [(0, 0, 1, .3), (0, .7, 1, 1), (0, .3, .3, .7), (.7, .3, 1, .7)]
    vertices, faces = [], []
    for x0, y0, x1, y1 in rectangles:
        k = len(vertices)
        vertices.extend([[x0, y0, .4], [x1, y0, .4], [x1, y1, .4], [x0, y1, .4]])
        faces.extend([[k, k + 1, k + 2], [k, k + 2, k + 3]])
    meta, arrays = g.describe_surface(vertices, faces, np.arange(8), [.5, .5, .4], [1, 0])
    assert not arrays["contact_mask"][12:20, 12:20].any()
    assert arrays["contact_mask"][:5].all()
    assert np.array_equal(arrays["valid_mask"], arrays["contact_mask"])
    assert np.all(arrays["heightmap_m"][~arrays["valid_mask"]] == 0)
    assert meta["missing_cells_are_unknown_not_free"] is True


def test_triangle_not_its_bounding_box():
    v = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    _, arrays = g.describe_surface(v, [[0, 1, 2]], [0], [0, 0, 0], [1, 0])
    assert arrays["contact_mask"][2, 2]
    assert not arrays["contact_mask"][29, 29]


@pytest.mark.parametrize("angle", [-2.1, -.3, .0, .8, 2.9])
def test_translation_yaw_invariance(angle):
    v, f = square()
    base, a = descriptor()
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    shift = np.array([3.2, -1.7, 4.1])
    origin = np.array([.5, .5, .4]) @ rotation.T + shift
    meta, b = g.describe_surface(v @ rotation.T + shift, f, [0, 1], origin, rotation[:2, 0])
    assert np.array_equal(a["contact_mask"], b["contact_mask"])
    assert np.allclose(a["heightmap_m"], b["heightmap_m"], atol=1e-7)
    assert meta["features"]["surface_area_m2"] == pytest.approx(base["features"]["surface_area_m2"])
    assert meta["features"]["extent_local_xy_m"] == pytest.approx([1., 1.])


def test_metric_scale_is_not_discarded():
    v, f = square()
    meta, _ = g.describe_surface(v * 2, f, [0, 1], [1, 1, .8], [1, 0])
    assert meta["features"]["surface_area_m2"] == pytest.approx(4.)
    assert meta["features"]["extent_local_xy_m"] == pytest.approx([2., 2.])
    assert meta["cell_size_xy_m"] == pytest.approx([2 / 32, 2 / 32])


def test_height_is_barycentric_and_overlaps_are_topmost():
    v, f = square()
    v[:, 2] = v[:, 0] * .2 + v[:, 1] * .1 + .4
    _, arrays = g.describe_surface(v, f, [1, 0], [0, 0, .4], [1, 0])
    x = (np.arange(32) + .5) / 32
    assert np.allclose(arrays["heightmap_m"], .2 * x[:, None] + .1 * x[None, :])
    vertices = np.vstack((v, v + [0, 0, .3]))
    _, higher = g.describe_surface(vertices, np.vstack((f, f + 4)), np.arange(4), [0, 0, .4], [1, 0])
    assert np.allclose(higher["heightmap_m"], arrays["heightmap_m"] + .3)


@pytest.mark.parametrize("ids", [[-1], [2], [0, 0], [0.0], [], [True]])
def test_bad_indices_rejected(ids):
    v, f = square()
    with pytest.raises(ValueError):
        g.describe_surface(v, f, ids, [.5, .5, .4], [1, 0])


@pytest.mark.parametrize("kind", ["vertex_nan", "origin_nan", "direction_nan", "zero_direction", "float_face", "face_oob", "degenerate"])
def test_invalid_geometry_rejected(kind):
    v, f = square(); origin, direction = [.5, .5, .4], [1, 0]
    if kind == "vertex_nan": v[0, 0] = np.nan
    if kind == "origin_nan": origin[2] = np.nan
    if kind == "direction_nan": direction[0] = np.nan
    if kind == "zero_direction": direction = [0, 0]
    if kind == "float_face": f = f.astype(float)
    if kind == "face_oob": f[0, 0] = 99
    if kind == "degenerate": v[:, 1] = 0
    with pytest.raises(ValueError):
        g.describe_surface(v, f, [0, 1], origin, direction)


@pytest.mark.parametrize("kind", ["credit", "nan", "frame", "spacing", "mask", "coverage"])
def test_descriptor_validation_rejects_drift(kind):
    meta, arrays = descriptor()
    if kind == "credit": meta["positive_credit"] = 1
    if kind == "nan": arrays["heightmap_m"][0, 0] = np.nan
    if kind == "frame": meta["frame"]["local_to_world_rotation"][0][0] = -1
    if kind == "spacing": meta["cell_size_xy_m"][0] *= 2
    if kind == "mask": arrays["valid_mask"][0, 0] = False
    if kind == "coverage": meta["features"]["raster_coverage_fraction"] = .2
    with pytest.raises(ValueError):
        g.validate_descriptor(meta, arrays)


def test_obj_native_axes_and_negative_indices(tmp_path):
    path = tmp_path / "mesh.obj"
    path.write_text("v 0 2 0\nv 1 2 0\nv 0 2 1\nf -3 -2 -1\n")
    v, f = g.load_obj_world_zup(path)
    assert np.array_equal(v, [[0, 0, 2], [1, 0, 2], [0, -1, 2]])
    assert np.array_equal(f, [[0, 1, 2]])
