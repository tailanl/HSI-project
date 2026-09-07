"""CPU tests of real numerical operators, not a generated-motion benchmark."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hsi.common.artifacts import MemoryContractError
from hsi.stage3.guidance import (SurfacePatch, GuidanceConfig, GeometryAffordanceEnergy,
    J22ProxyMap, ContactContext, pool_occupancy, split_before_pool, bounded_policy)
from hsi.stage3.sdf import RawSceneSDFZUp
from hsi.stage3.release import current_policy, release_gates, gate
from hsi.stage3.metrics import point_polyline_distance, terminal_metrics, collision_metrics


def patch():
    return SurfacePatch(origin_world_zup_m=[0., 0., .5], local_to_world_rotation=np.eye(3),
        triangles_local_m=[[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])


def test_actual_triangle_projection_not_bounding_box():
    triangle = patch()
    points = torch.tensor([[.9, .9, .8], [.2, .2, .8]], dtype=torch.float64)
    closest = triangle.closest_world(points)
    torch.testing.assert_close(closest, torch.tensor([[.5, .5, .5], [.2, .2, .5]], dtype=torch.float64))
    assert not closest.requires_grad


def test_bad_patch_reflection_and_vertical_surface_rejected():
    with pytest.raises(MemoryContractError):
        SurfacePatch(origin_world_zup_m=[0., 0., 0.], local_to_world_rotation=np.diag([-1., 1., 1.]),
            triangles_local_m=[[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    with pytest.raises(MemoryContractError):
        SurfacePatch(origin_world_zup_m=[0., 0., 0.], local_to_world_rotation=np.eye(3),
            triangles_local_m=[[[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]]])


def test_target_removed_before_pool_preserves_shared_obstacle_voxel():
    occupancy = np.zeros((4, 4, 4), dtype=bool)
    target = np.zeros_like(occupancy)
    target[0, 0, 0] = True
    occupancy[0, 0, 0] = occupancy[1, 1, 1] = True
    parent, other = split_before_pool(occupancy, target, 2)
    assert parent[0, 0, 0] and other[0, 0, 0]
    assert not (parent & ~pool_occupancy(target, 2)).any()
    np.testing.assert_array_equal(occupancy, target | (np.indices((4, 4, 4)) == 1).all(axis=0))


def test_sdf_metric_sign_outside_gradient_and_parent_target_retained():
    occupancy = np.zeros((12, 12, 12), dtype=bool)
    occupancy[5:7, 5:7, 5:7] = True
    sdf = RawSceneSDFZUp.from_occupancy(occupancy, lower_xyz=(0., 0., 0.), upper_xyz=(1.2, 1.2, 1.2),
        floor_ignore_height_m=0., remove_target_component=False, source="synthetic_test_only")
    points = torch.tensor([[.55, .55, .55], [.15, .15, .15], [-.1, .5, .5]], requires_grad=True)
    sampled = sdf.sample(points)
    assert sampled.signed_distance_m[0] < 0 < sampled.signed_distance_m[1]
    assert sampled.in_bounds.tolist() == [True, True, False]
    sampled.signed_distance_m[-1].backward()
    assert points.grad[-1, 0] > 0
    assert sdf.receipt["target_component_removed"] is False


class FlatField:
    lower_xyz = (0., 0., 0.)
    spacing_xyz = (.02, .02, .02)
    receipt = {"spacing_xyz": [.02, .02, .02]}

    def __init__(self, value):
        self.value = value

    def sample(self, points):
        return SimpleNamespace(signed_distance_m=torch.full(points.shape[:-1], self.value,
            dtype=points.dtype, device=points.device), in_bounds=torch.ones(points.shape[:-1],
            dtype=torch.bool, device=points.device))


def energy_fixture(field_value=1., *, enabled=True):
    triangle, field = patch(), FlatField(field_value)
    proxy = J22ProxyMap(contact_ids=(0,), contact_drop_m=(.1,), foot_drop_m=(.03, .03),
        body_identity_sha256="a" * 64)
    context = ContactContext(field, field, triangle, proxy, {}, {}, None)
    energy = GeometryAffordanceEnergy(field, triangle, replace(GuidanceConfig(), enabled=enabled),
        contact_context=context)
    joints = torch.full((1, 2, 22, 3), .2, dtype=torch.float64)
    joints[..., 2] = .8
    joints[:, :, 1, 0], joints[:, :, 2, 0] = .1, .3
    joints.requires_grad_(True)
    points = proxy.points(joints, body_identity_sha256="a" * 64)
    return energy, joints, {**points, "contact_activation": torch.ones((1, 2), dtype=torch.float64),
        "support_activation": torch.zeros((1, 2, 2), dtype=torch.float64)}


def test_guidance_contact_gradient_and_blocked_attraction():
    energy, joints, args = energy_fixture()
    value, diagnostics = energy.energy(**args)
    assert value > 0 and diagnostics["p555_contact_active_fraction"] == 1
    value.backward()
    assert joints.grad[0, 0, 0, 2] > 0
    blocked, _, points = energy_fixture(-.1)
    value, diagnostics = blocked.energy(**points)
    assert value == 0 and diagnostics["p555_contact_active_fraction"] == 0


def test_no_blanket_proxy_exemption_and_disabled_identity():
    energy, _, args = energy_fixture()
    changed = dict(args, contact_points_world=args["contact_points_world"] + .01)
    with pytest.raises(MemoryContractError):
        energy.energy(**changed)
    disabled, _, args = energy_fixture(enabled=False)
    added, info = disabled.energy(**args)
    parent, diagnostic = torch.tensor(1.), {"sdf_energy": torch.tensor(.1)}
    result, result_info = disabled.combine_with_parent(parent, diagnostic, added, info, original_sdf_weight=1.)
    assert result is parent and result_info is diagnostic


def test_discretization_band_bound_and_spacing_proof():
    field = FlatField(1.)
    assert bounded_policy(field, GuidanceConfig())["effective_contact_end_band_m"] < .09
    field.spacing_xyz = (.1, .1, .1)
    with pytest.raises(MemoryContractError):
        bounded_policy(field, GuidanceConfig())
    field.receipt = {"spacing_xyz": list(field.spacing_xyz)}
    assert bounded_policy(field, GuidanceConfig())["effective_contact_end_band_m"] == .09


def test_fixed_34_gates_missing_fields_never_pass():
    policy = current_policy()
    checks = release_gates({"runner_returncode": 1}, policy)
    assert len(checks) == 34
    assert not any(check["passed"] for check in checks.values())
    for value in (None, True, float("nan"), float("inf"), "0"):
        assert gate(value, "max", .02)[0] is False
    assert gate(.02, "max", .02)[0] is True


def test_metric_polyline_and_contact_exemption_only_final_tail():
    distance = point_polyline_distance(np.array([[.5, .2], [2., 0.]]), np.array([[0., 0.], [1., 0.]]))
    np.testing.assert_allclose(distance, [.2, 1.])
    joints = np.full((8, 22, 3), .5)
    sdf = np.full((8, 8, 8), -.1)
    result = collision_metrics(joints, sdf, sdf, np.zeros(3), np.ones(3), (0, 1, 2), 4, 4, .08, .02)
    assert result["forbidden_collision_count"] == 8 * 22 - 4 * 3
    assert result["navigation"]["forbidden_collision_count"] == 4 * 22
