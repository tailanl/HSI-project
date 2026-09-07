"""Frozen numerical observations from pre-integration CPU kernels.

The references were obtained by executing both implementations on identical
synthetic inputs and comparing outputs. Tests need no historical source tree,
scene dataset, model, network, or GPU. Ten-decimal rounding only normalizes
floating-point representation for JSON checksums; no algorithm uses rounding.
"""
import copy

import numpy as np
import pytest

from hsi.common.artifacts import digest
from hsi.memory import surface, matching, extraction, geometry, store
from test_memory_extraction import data


def rounded(value):
    if isinstance(value, np.ndarray): return rounded(value.tolist())
    if isinstance(value, np.generic): return rounded(value.item())
    if isinstance(value, float): return round(value, 10)
    if isinstance(value, dict): return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [rounded(x) for x in value]
    return value


def sloped_surface():
    vertices = np.array([[0., 0., .4], [1., 0., .6], [1., 1., .7], [0., 1., .5]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return vertices, faces


def test_original_surface_metadata_masks_and_heights():
    vertices, faces = sloped_surface()
    result = surface.describe_surface(vertices, faces, [0, 1], [.5, .5, .4], [1, 0])
    assert digest(rounded(result)) == '42e04b8a96ff038c31bf230fe308f061844c2039495f72a1eeeecb874aa4c795'


def test_original_metric_match_diagnostics():
    vertices, faces = sloped_surface()
    metadata, arrays = surface.describe_surface(vertices, faces, [0, 1], [.5, .5, .4], [1, 0])
    original = matching.descriptor_to_shape(({}, metadata, arrays))
    current = copy.deepcopy(original)
    current['dimensions_m'][0] *= 1.15
    assert digest(rounded(matching.match_shapes(original, current))) == '1801cf63c469160e16216f4184c549ae322fbbfdb410c0ba43660978b41442c0'


def test_original_terminal_medoid_local_records():
    assert digest(rounded(extraction.derive_records(*data()))) == '3885b5834b1d1ad6360a507223bfe41fe146f44dc03356861f57f0d0ae8759c2'


def test_original_triangle_membership_and_topmost_heights():
    vertices, faces = sloped_surface()
    points = np.random.default_rng(991).uniform(-.25, 1.25, (137, 2))
    assert digest(rounded(geometry.triangle_heights(points, vertices[faces]))) == '346ffe07a0684ef3b67b85e823bc6ef504904a8e992813c0bdb124a67f9449a3'


@pytest.mark.parametrize('successes,failures,expected', [
    (0, 0, .05000000000000002), (8, 0, .7168711644368866),
    (8, 1, .6058366975634952), (12, 2, .6365582344585978),
    (20, 0, .8670540889734766),
])
def test_original_beta_lower_bound(successes, failures, expected):
    assert store._beta_lower(successes, failures) == pytest.approx(expected, abs=1e-14)
