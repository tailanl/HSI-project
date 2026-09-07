"""Retained numerical regressions against integrated modules."""

from __future__ import annotations

from hsi.stage3 import causal_bridge as MODULE

import json

from pathlib import Path

import pytest

import torch

P360E226BridgeConfig = MODULE.P360E226BridgeConfig

P360E226BridgeError = MODULE.P360E226BridgeError

P360RollingEnergyBridge = MODULE.P360RollingEnergyBridge

ReMoGenE226CondFn = MODULE.ReMoGenE226CondFn

RollingEnergyConfig = MODULE.RollingEnergyConfig

def _history(*, requires_grad: bool = False) -> torch.Tensor:
    value = torch.zeros(6, 22, 3)
    value[:, 0, 0] = torch.linspace(-1.0, 0.0, 6)
    value[:, 0, 2] = 0.92
    return value.requires_grad_(requires_grad)

def _current() -> torch.Tensor:
    value = torch.zeros(1, 8, 22, 3)
    value[0, :, 0, 0] = torch.linspace(0.15, 1.0, 8)
    value[0, :, 0, 2] = 0.92
    return value.requires_grad_(True)

def test_bridge_detaches_generated_prefix_and_backpropagates_current() -> None:
    history = _history(requires_grad=True)
    bridge = P360RollingEnergyBridge(
        generated_history_world_joints_zup=history,
        history_source="generated_prefix",
        active_goal_world_zup=(1.2, 0.0, 0.92),
        next_goal_world_zup=(2.0, 0.0, 0.92),
        active_yaw_rad=0.0,
    )
    current = _current()
    total, diagnostics = bridge.energy_from_world_joints(current)
    total.backward()

    assert history.grad is None
    assert bridge.history_world_joints_zup.requires_grad is False
    assert current.grad is not None
    assert bool(torch.isfinite(current.grad).all())
    assert float(torch.linalg.vector_norm(current.grad)) > 0.0
    assert diagnostics["total"].ndim == 0
    assert bridge.receipt["future_or_gt_query"] is False
    assert bridge.receipt["posthoc_edit"] is False

def test_zero_outer_scale_reports_every_effective_e226_energy_as_exact_zero() -> None:
    bridge = P360RollingEnergyBridge(
        generated_history_world_joints_zup=_history(),
        history_source="generated_prefix",
        active_goal_world_zup=(1.2, 0.0, 0.92),
        next_goal_world_zup=(2.0, 0.0, 0.92),
        active_yaw_rad=0.0,
        config=P360E226BridgeConfig(rolling_scale=0.0),
    )
    total, diagnostics = bridge.energy_from_world_joints(_current())
    assert float(total) == 0.0
    for name in MODULE.ENERGY_NAMES:
        assert float(diagnostics[name]) == 0.0
        assert float(diagnostics[f"weighted_{name}"]) == 0.0
    assert float(diagnostics["total"]) == 0.0
    # Geometry remains observable even though its energy contribution is off.
    assert float(diagnostics["path_length_m"]) > 0.0

def test_bridge_rejects_gt_oracle_history_source() -> None:
    with pytest.raises(P360E226BridgeError, match="GT/oracle history is forbidden"):
        P360RollingEnergyBridge(
            generated_history_world_joints_zup=_history(),
            history_source="gt_future_root",
            active_goal_world_zup=(1.0, 0.0, 0.9),
        )

def test_json_mapping_is_strict_and_defaults_to_no_duplicate_pelvis_sdf() -> None:
    config = P360E226BridgeConfig.from_mapping(
        {
            "history_tail_frames": 12,
            "rolling": {
                "goal_weight": 2.0,
                "sdf_weight": 0.0,
            },
        }
    )
    assert config.history_tail_frames == 12
    assert config.rolling.goal_weight == 2.0
    assert config.rolling.sdf_weight == 0.0
    with pytest.raises(P360E226BridgeError, match="unknown"):
        P360E226BridgeConfig.from_mapping({"future_gt": True})
    with pytest.raises(P360E226BridgeError, match="must be zero"):
        P360E226BridgeConfig(
            include_pelvis_sdf=False,
            rolling=RollingEnergyConfig(sdf_weight=1.0),
        ).validate()

class _PrimitiveUtility:
    def tensor_to_dict(self, value):
        return {"joints": value}

class _Dataset:
    primitive_utility = _PrimitiveUtility()

    def denormalize(self, value):
        return value

class _VAE(torch.nn.Module):
    def decode(self, latent, history, nfuture, scale_latent):
        return latent

class _SDF:
    class _Samples:
        def __init__(self, points):
            self.signed_distance_m = points[..., 0] * 0.0 + 1.0
            self.in_bounds = torch.ones_like(
                self.signed_distance_m, dtype=torch.bool
            )

    def sample(self, points):
        return self._Samples(points)

def _subclass_for_energy(*, base_goal=None) -> ReMoGenE226CondFn:
    return ReMoGenE226CondFn(
        generated_history_world_joints_zup=_history(),
        history_source="generated_prefix",
        active_goal_world_zup=(1.2, 0.0, 0.92),
        next_goal_world_zup=(2.0, 0.0, 0.92),
        active_yaw_rad=0.0,
        vae_model=_VAE(),
        dataset=_Dataset(),
        history_motion=torch.zeros(1, 2, 66),
        scene_sdf=_SDF(),
        current_rotmat=torch.eye(3).unsqueeze(0),
        current_transl=torch.zeros(1, 1, 3),
        global_rotmat=torch.eye(3).unsqueeze(0),
        global_transl=torch.zeros(1, 1, 3),
        future_length=8,
        scale_latent=False,
        num_diffusion_steps=10,
        collision_joint_ids=(),
        goal_world_zup=base_goal,
    )

def test_subclass_composes_base_and_rolling_energy_for_cond_fn_dispatch() -> None:
    cond_fn = _subclass_for_energy()
    current = _current()
    total, diagnostics = cond_fn.energy_from_world_joints(current)
    total.backward()

    assert "scene_energy" in diagnostics
    assert "e226_energy" in diagnostics
    assert "combined_scene_e226_energy" in diagnostics
    assert "e226_backtrack" in diagnostics
    assert torch.allclose(total, diagnostics["combined_scene_e226_energy"])
    assert current.grad is not None
    assert float(torch.linalg.vector_norm(current.grad)) > 0.0

def test_subclass_rejects_double_goal_guidance() -> None:
    with pytest.raises(P360E226BridgeError, match="double-count"):
        _subclass_for_energy(base_goal=torch.tensor((1.2, 0.0, 0.92)))

