"""Retained numerical regressions against integrated modules."""

from __future__ import annotations



from pathlib import Path

import numpy as np

import torch

from hsi.stage3.sdf import (  # noqa: E402
    QueryTimeICGF,
    RawSceneSDFZUp,
    select_target_component,
)

LOWER = (-1.0, -1.0, 0.0)

UPPER = (1.0, 1.0, 2.0)

def _centre(index_xyz, shape=(20, 20, 20)) -> np.ndarray:
    lower = np.asarray(LOWER, dtype=np.float32)
    upper = np.asarray(UPPER, dtype=np.float32)
    spacing = (upper - lower) / np.asarray(shape, dtype=np.float32)
    return lower + (np.asarray(index_xyz, dtype=np.float32) + 0.5) * spacing

def test_raw_sdf_has_metric_sign_and_point_gradient() -> None:
    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[10:13, :, :] = True
    field = RawSceneSDFZUp.from_occupancy(
        occupancy,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
    )
    points = torch.from_numpy(
        np.stack((_centre((4, 10, 10)), _centre((10, 10, 10))))[None, None]
    ).requires_grad_(True)
    sample = field.sample(points)
    assert sample.signed_distance_m.shape == (1, 1, 2)
    assert sample.signed_distance_m[0, 0, 0] > 0.0
    assert sample.signed_distance_m[0, 0, 1] < 0.0

    energy = torch.relu(0.01 - sample.signed_distance_m[0, 0, 1]).square()
    energy.backward()
    assert points.grad is not None
    # Index 10 is inside the slab near its lower-X face.  Negative energy
    # gradient must therefore move it toward decreasing X and out of the slab.
    assert float((-points.grad[0, 0, 1, 0]).item()) < 0.0
    assert field.receipt["grid_sample_coordinate_order"] == ["z", "y", "x"]
    assert field.receipt["retrieval_used"] is False

def test_target_point_selects_only_its_component_and_removes_it_from_sdf() -> None:
    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[3:6, 3:6, 5:9] = True
    occupancy[14:18, 14:18, 6:10] = True
    target_point = _centre((4, 4, 7))
    selected = select_target_component(
        occupancy,
        target_point,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
    )
    assert int(selected.target_occupancy_xyz.sum()) == 3 * 3 * 4
    assert int(selected.collision_occupancy_xyz.sum()) == 4 * 4 * 4

    field = RawSceneSDFZUp.from_occupancy(
        occupancy,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
        target_point_world_zup=target_point,
        remove_target_component=True,
    )
    query = torch.from_numpy(
        np.stack((_centre((4, 4, 7)), _centre((15, 15, 7))))[None]
    )
    sdf = field.sample(query).signed_distance_m
    assert sdf[0, 0] > 0.0
    assert sdf[0, 1] < 0.0
    assert field.target_removed
    receipt = field.receipt
    assert receipt["target_component_voxel_count"] == 3 * 3 * 4
    assert len(receipt["target_component_sha256"]) == 64
    assert receipt["collision_occupancy_voxel_count"] == 4 * 4 * 4
    assert len(receipt["collision_occupancy_sha256"]) == 64

def test_local_component_crop_preserves_connected_scene_outside_query() -> None:
    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[3:6, 3:6, 5:9] = True
    occupancy[14:18, 3:7, 5:10] = True
    # A one-voxel furniture/wall bridge makes the two objects globally
    # six-connected, mirroring the failure observed in the real LINGO raster.
    occupancy[6:14, 4, 7] = True
    target_point = _centre((4, 4, 7))

    global_component = select_target_component(
        occupancy,
        target_point,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
    )
    assert int(global_component.collision_occupancy_xyz.sum()) == 0

    local_component = select_target_component(
        occupancy,
        target_point,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
        maximum_seed_distance_m=0.20,
        search_crop_radius_m=0.45,
        maximum_component_fraction=0.65,
    )
    assert local_component.search_crop_radius_m == 0.45
    assert int(local_component.collision_occupancy_xyz.sum()) > 0
    assert bool(local_component.collision_occupancy_xyz[15, 4, 7])
    assert local_component.target_fraction_of_scene_occupied < 0.65

    field = RawSceneSDFZUp.from_occupancy(
        occupancy,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
        target_point_world_zup=target_point,
        remove_target_component=True,
        maximum_target_seed_distance_m=0.20,
        target_component_search_crop_radius_m=0.45,
        maximum_target_component_fraction=0.65,
    )
    receipt = field.receipt
    assert receipt["target_component_search_crop_radius_m"] == 0.45
    assert receipt["target_component_fraction_of_scene_occupied"] < 0.65
    assert (
        receipt["target_component_voxel_count"]
        + receipt["collision_occupancy_voxel_count"]
        == receipt["scene_occupied_voxel_count"]
    )

def test_query_icgf_supports_target_point_component_and_sparse_joints() -> None:
    point_field = QueryTimeICGF.from_target_point((0.2, -0.1, 0.8))
    query = torch.tensor([[[[0.2, -0.1, 0.8], [0.5, -0.1, 0.8]]]])
    distance = point_field.soft_distance(query)
    assert distance[0, 0, 0] < 1.0e-4
    assert distance[0, 0, 1] > 0.25

    targets = torch.zeros(22, 3)
    targets[0] = torch.tensor([0.1, 0.2, 0.9])
    targets[20] = torch.tensor([0.4, 0.2, 1.2])
    mask = torch.zeros(22, dtype=torch.bool)
    mask[[0, 20]] = True
    sparse = QueryTimeICGF.from_sparse_joint_targets(targets, mask)
    sparse_query = torch.stack((targets[0], targets[20]))[None, None]
    sparse_distance = sparse.soft_distance(
        sparse_query, joint_ids=torch.tensor([0, 20], dtype=torch.long)
    )
    assert torch.all(sparse_distance < 1.0e-4)

    component = np.zeros((20, 20, 20), dtype=bool)
    component[8:12, 8:12, 5:10] = True
    surface = QueryTimeICGF.from_target_component(
        component,
        anchor_point_world_zup=_centre((10, 10, 10)),
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        local_radius_m=0.20,
        maximum_points=32,
    )
    assert 1 <= surface.support_points_world_zup.shape[0] <= 32
    assert surface.receipt["source"] == "query_target_component_surface"
    assert surface.receipt["retrieval_used"] is False

def test_raw_sdf_level_set_projection_reaches_positive_surface_offset() -> None:
    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[8:12, 8:12, 4:8] = True
    field = RawSceneSDFZUp.from_occupancy(
        occupancy,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
    )
    # One query begins inside the object and one in free space above it.
    query = torch.from_numpy(
        np.stack((_centre((10, 10, 6)), _centre((10, 10, 10))))[None]
    )
    projection = field.project_to_level_set(
        query,
        target_level_m=0.01,
        tolerance_m=0.003,
        maximum_iterations=32,
        maximum_step_m=0.08,
        maximum_displacement_m=0.50,
    )
    assert bool(projection.converged.all())
    assert torch.allclose(
        projection.final_signed_distance_m,
        torch.full_like(projection.final_signed_distance_m, 0.01),
        atol=0.003,
        rtol=0.0,
    )
    assert bool((projection.displacement_m > 0.0).all())
    assert bool((projection.displacement_m <= 0.50).all())

