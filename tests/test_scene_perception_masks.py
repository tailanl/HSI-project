"""CPU-only unit tests for P508 SAM3.1 mask-to-instance post-processing.

The real detector is intentionally not loaded here.  These tests exercise the
deterministic gates after inference: prompt-component cleanup, same-view NMS,
calibrated depth projection, and fail-closed multi-view fusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


from hsi.stage1.perception import sam_geometry as sam31


def _mask_row(proposal_id: str, mask: np.ndarray, *, quality_scale: float = 1.0):
    return {
        "proposal_id": proposal_id,
        "mask": np.asarray(mask, dtype=bool),
        "mask_audit": {
            "mask_fraction_inside_prompt": quality_scale,
            "prompt_box_fill_fraction": quality_scale,
            "largest_prompt_component_area_px": int(np.count_nonzero(mask)),
        },
    }


def _observation(
    observation_id: str,
    view_index: int,
    support: np.ndarray,
    vertices: np.ndarray,
) -> sam31.Observation:
    xyz = vertices[support]
    return sam31.Observation(
        observation_id=observation_id,
        view_index=view_index,
        view_id=f"view_{view_index:02d}",
        proposal_id="ANON_BOX_000",
        mask=np.ones((20, 20), dtype=bool),
        support=np.asarray(support, dtype=np.int64),
        bounds=np.stack((xyz.min(axis=0), xyz.max(axis=0))),
        centroid=np.median(xyz, axis=0),
        quality=1.0,
        mask_audit={},
        prompt={"x": 0, "y": 0, "width": 20, "height": 20},
    )


def test_cleanup_mask_keeps_prompt_component_not_larger_outside_component() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[55:90, 55:90] = True

    cleaned, audit = sam31.cleanup_mask(
        mask, {"x": 8, "y": 8, "width": 25, "height": 25}
    )

    assert audit["gate_pass"] is True
    assert audit["connected_component_count"] == 2
    assert audit["largest_prompt_component_area_px"] == 400
    assert np.count_nonzero(cleaned) == 400
    assert cleaned[15, 15]
    assert not cleaned[60, 60]


def test_cleanup_mask_fails_closed_for_empty_prompt_and_scene_sized_mask() -> None:
    empty_prompt_mask = np.zeros((100, 100), dtype=bool)
    empty_prompt_mask[60:90, 60:90] = True
    cleaned, audit = sam31.cleanup_mask(
        empty_prompt_mask, {"x": 5, "y": 5, "width": 20, "height": 20}
    )
    assert audit == {"reason": "no_component_inside_prompt", "gate_pass": False}
    assert not cleaned.any()

    scene_sized = np.zeros((100, 100), dtype=bool)
    scene_sized[5:95, 5:95] = True
    cleaned, audit = sam31.cleanup_mask(
        scene_sized, {"x": 20, "y": 20, "width": 40, "height": 40}
    )
    assert np.count_nonzero(cleaned) == 8100
    assert audit["image_area_fraction"] == 0.81
    assert audit["gate_pass"] is False
    assert audit["reason"] == "mask_geometry_gate_failed"


def test_mask_nms_suppresses_contained_duplicate_but_preserves_disjoint_mask() -> None:
    primary = np.zeros((100, 100), dtype=bool)
    primary[10:40, 10:40] = True
    contained = np.zeros_like(primary)
    contained[11:39, 11:39] = True
    disjoint = np.zeros_like(primary)
    disjoint[60:90, 60:90] = True

    kept = sam31.mask_nms(
        [
            _mask_row("ANON_BOX_000", primary, quality_scale=1.0),
            _mask_row("ANON_BOX_001", contained, quality_scale=0.8),
            _mask_row("ANON_BOX_002", disjoint, quality_scale=0.9),
        ]
    )

    assert [row["proposal_id"] for row in kept] == ["ANON_BOX_000", "ANON_BOX_002"]


def test_visible_vertex_projection_requires_in_frame_positive_and_depth_consistent() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 1.0],   # visible at pixel (2, 2)
            [1.0, 0.0, 1.0],   # in frame, but occluded by the depth buffer
            [3.0, 0.0, 1.0],   # out of frame
            [0.0, 0.0, -1.0],  # behind camera
        ],
        dtype=np.float64,
    )
    view = {
        "world_to_camera": np.eye(4, dtype=np.float64).tolist(),
        "K": [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
    }
    depth = np.zeros((5, 5), dtype=np.float64)
    depth[2, 2] = 1.0
    depth[2, 3] = 0.5

    ids, u, v = sam31.visible_vertex_projection(vertices, view, depth)

    np.testing.assert_array_equal(ids, np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(u, np.asarray([2], dtype=np.int32))
    np.testing.assert_array_equal(v, np.asarray([2], dtype=np.int32))


