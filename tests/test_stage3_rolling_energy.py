"""Retained numerical regressions against integrated modules."""

from __future__ import annotations

from hsi.stage3 import rolling_energy as MODULE

from pathlib import Path

import pytest

import torch

ENERGY_NAMES = MODULE.ENERGY_NAMES

RollingEnergyConfig = MODULE.RollingEnergyConfig

e226_rolling_energy = MODULE.e226_rolling_energy

group_energy_components = MODULE.group_energy_components

grouped_pcgrad = MODULE.grouped_pcgrad

p360_world_zup_pelvis_energy = MODULE.p360_world_zup_pelvis_energy

def _shape_config(**overrides):
    values = dict(
        goal_weight=1.0,
        heading_weight=1.0,
        smooth_weight=1.0,
        length_ratio_weight=1.0,
        backtrack_weight=1.0,
        zigzag_weight=1.0,
        self_loop_weight=1.0,
        sdf_weight=1.0,
        self_loop_temporal_exclusion_frames=2,
        self_loop_sigma_m=0.10,
    )
    values.update(overrides)
    return RollingEnergyConfig(**values)

def test_straight_goal_aligned_free_path_has_low_energy() -> None:
    history = torch.tensor([[-1.50, 0.0], [-1.00, 0.0], [-0.50, 0.0]])
    current = torch.tensor([[0.0, 0.0], [0.50, 0.0], [1.00, 0.0]])

    total, values = e226_rolling_energy(
        history,
        current,
        active_goal_xy=(1.0, 0.0),
        next_goal_xy=(2.0, 0.0),
        yaw_heading_xy=(1.0, 0.0),
        sdf_callback=lambda xy: torch.full(
            (xy.shape[0],), 1.0, device=xy.device, dtype=xy.dtype
        ),
        config=_shape_config(),
    )

    assert float(total) < 1.0e-5
    for name in ("goal", "heading", "smooth", "length_ratio", "backtrack", "zigzag", "sdf"):
        assert float(values[name]) == pytest.approx(0.0, abs=1.0e-7)
    assert float(values["self_loop"]) < 1.0e-5
    assert float(values["length_ratio_value"]) == pytest.approx(1.0)

def test_return_and_zigzag_are_more_expensive_than_straight() -> None:
    history = torch.tensor([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    straight = torch.tensor([[0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]])
    bad = torch.tensor(
        [[0.5, 0.0], [0.0, 0.0], [0.5, 0.65], [0.9, -0.65], [2.0, 0.0]]
    )
    config = _shape_config(self_loop_temporal_exclusion_frames=1)
    _, good_values = e226_rolling_energy(
        history, straight, active_goal_xy=(2.0, 0.0), config=config
    )
    _, bad_values = e226_rolling_energy(
        history, bad, active_goal_xy=(2.0, 0.0), config=config
    )

    assert float(bad_values["backtrack"]) > float(good_values["backtrack"]) + 0.01
    assert float(bad_values["zigzag"]) > float(good_values["zigzag"]) + 0.1
    assert float(bad_values["length_ratio"]) > float(good_values["length_ratio"]) + 0.1
    assert float(bad_values["total"]) > float(good_values["total"]) + 1.0

def test_self_loop_detects_return_across_detached_history() -> None:
    history = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]]
    )
    continuation = torch.tensor([[2.5, 0.0], [3.0, 0.0], [3.5, 0.0]])
    returns_to_old_history = torch.tensor(
        [[2.5, 0.0], [1.6, 0.2], [1.0, 0.02], [0.5, 0.01]]
    )
    config = _shape_config(
        self_loop_temporal_exclusion_frames=2,
        self_loop_sigma_m=0.12,
    )
    _, good = e226_rolling_energy(
        history, continuation, active_goal_xy=(3.5, 0.0), config=config
    )
    _, loop = e226_rolling_energy(
        history, returns_to_old_history, active_goal_xy=(0.5, 0.0), config=config
    )

    assert float(good["self_loop"]) < 1.0e-6
    assert float(loop["self_loop"]) > 0.40

def test_current_has_finite_gradient_but_history_is_always_detached() -> None:
    history = torch.tensor(
        [[-0.8, 0.0], [-0.4, 0.0], [0.0, 0.0]], requires_grad=True
    )
    current = torch.tensor(
        [[0.35, 0.05], [0.65, -0.10], [0.90, 0.12]], requires_grad=True
    )

    def sdf_callback(xy):
        # A differentiable half-space: the first two samples violate 0.20 m.
        return xy[:, 0] - 0.30

    total, values = e226_rolling_energy(
        history,
        current,
        active_goal_xy=(1.2, 0.0),
        next_goal_xy=(1.6, 0.2),
        yaw_heading_xy=(1.0, 0.0),
        sdf_callback=sdf_callback,
        sdf_clearance_m=0.20,
        config=_shape_config(),
    )
    total.backward()

    assert history.grad is None
    assert current.grad is not None
    assert bool(torch.isfinite(current.grad).all())
    assert float(torch.linalg.vector_norm(current.grad)) > 0.0
    assert all(bool(torch.isfinite(values[name])) for name in ENERGY_NAMES)

def test_p360_adapter_uses_world_xy_and_ignores_world_z_height() -> None:
    history = torch.zeros(3, 2, 3)
    current_a = torch.zeros(3, 2, 3, requires_grad=True)
    history[:, 0, 0] = torch.tensor([-1.0, -0.5, 0.0])
    current_a.data[:, 0, 0] = torch.tensor([0.3, 0.6, 1.0])
    current_a.data[:, 0, 1] = torch.tensor([0.1, 0.1, 0.0])
    current_a.data[:, 0, 2] = torch.tensor([0.2, 3.0, -7.0])
    current_b = current_a.detach().clone()
    current_b[:, 0, 2] = torch.tensor([100.0, -50.0, 27.0])

    _, a = p360_world_zup_pelvis_energy(
        history,
        current_a,
        active_goal_world_zup=(1.0, 0.0, 999.0),
        next_goal_world_zup=(2.0, 0.0, -999.0),
        config=_shape_config(),
    )
    _, b = p360_world_zup_pelvis_energy(
        history,
        current_b,
        active_goal_world_zup=(1.0, 0.0, -123.0),
        next_goal_world_zup=(2.0, 0.0, 321.0),
        config=_shape_config(),
    )
    for name in (*ENERGY_NAMES, "total"):
        assert torch.allclose(a[name], b[name])

    # Changing world Y, by contrast, must change a Z-up ground-plane loss.
    current_c = current_b.clone()
    current_c[:, 0, 1] += 0.8
    _, c = p360_world_zup_pelvis_energy(
        history,
        current_c,
        active_goal_world_zup=(1.0, 0.0, 0.0),
        config=_shape_config(),
    )
    assert not torch.allclose(a["goal"], c["goal"])

def test_sdf_clearance_energy_uses_current_primitive_only() -> None:
    history = torch.tensor([[-5.0, 0.0], [-4.0, 0.0]])
    current = torch.tensor([[0.0, 0.0], [0.2, 0.0]], requires_grad=True)
    _, free = e226_rolling_energy(
        history,
        current,
        active_goal_xy=(0.2, 0.0),
        sdf_callback=lambda xy: xy[:, 0] + 1.0,
        sdf_clearance_m=0.1,
        config=_shape_config(),
    )
    _, collision = e226_rolling_energy(
        history,
        current,
        active_goal_xy=(0.2, 0.0),
        sdf_callback=lambda xy: xy[:, 0] - 0.2,
        sdf_clearance_m=0.1,
        config=_shape_config(),
    )
    assert float(free["sdf"]) == 0.0
    assert float(collision["sdf"]) > 0.0
    assert float(collision["sdf_min_m"]) < 0.0

def test_grouping_reconstructs_total_and_pcgrad_preserves_task() -> None:
    history = torch.tensor([[-0.5, 0.0], [0.0, 0.0]])
    current = torch.tensor([[0.4, 0.1], [0.8, 0.0]], requires_grad=True)
    config = _shape_config()
    total, values = e226_rolling_energy(
        history, current, active_goal_xy=(1.0, 0.0), config=config
    )
    groups = group_energy_components(values, config=config)
    assert torch.allclose(torch.stack(tuple(groups.values())).sum(), total)
    result = grouped_pcgrad(groups, current, primary_group="task")
    assert result.combined_gradient.shape == current.shape
    assert bool(torch.isfinite(result.combined_gradient).all())
    assert set(result.raw_gradients) == {"task", "path_shape", "scene"}

def test_pcgrad_removes_a_conflicting_secondary_component() -> None:
    current = torch.tensor([[1.0, 0.0]], requires_grad=True)
    # task grad in x is -2; scene grad in x is +6, a direct conflict.
    groups = {
        "task": (current[0, 0] - 2.0).square() + current[0, 1] * 0.0,
        "scene": (current[0, 0] + 2.0).square() + current[0, 1] * 0.0,
    }
    result = grouped_pcgrad(groups, current, primary_group="task")
    assert float(result.conflict_dots["scene"]) < 0.0
    dot_after = (
        result.projected_gradients["scene"] * result.raw_gradients["task"]
    ).sum()
    assert float(dot_after) == pytest.approx(0.0, abs=1.0e-6)
    assert torch.allclose(
        result.combined_gradient, result.raw_gradients["task"], atol=1.0e-6
    )

def test_next_goal_supplies_heading_when_explicit_yaw_is_absent() -> None:
    history = torch.tensor([[-0.5, 0.0], [0.0, 0.0]])
    current = torch.tensor([[0.5, 0.0], [1.0, 0.0]])
    _, aligned = e226_rolling_energy(
        history,
        current,
        active_goal_xy=(1.0, 0.0),
        next_goal_xy=(2.0, 0.0),
        config=_shape_config(),
    )
    _, opposed = e226_rolling_energy(
        history,
        current,
        active_goal_xy=(1.0, 0.0),
        next_goal_xy=(0.0, 0.0),
        config=_shape_config(),
    )
    assert float(aligned["heading"]) == pytest.approx(0.0)
    assert float(opposed["heading"]) == pytest.approx(4.0)

