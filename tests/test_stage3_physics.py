"""Retained numerical regressions against integrated modules."""

from __future__ import annotations

from hsi.stage3 import physics

from pathlib import Path

from types import SimpleNamespace

import pytest

import torch

class FlatSDF:
    def __init__(self, distance_m: float = 1.0, ground_z_m: float = 0.0):
        self.distance_m = float(distance_m)
        self.lower_xyz = (-3.0, -4.0, float(ground_z_m))

    def sample(self, points: torch.Tensor):
        distance = points[..., 0] * 0.0 + self.distance_m
        return SimpleNamespace(
            signed_distance_m=distance,
            in_bounds=torch.ones_like(distance, dtype=torch.bool),
        )

class DirectionalXSDF(FlatSDF):
    def sample(self, points: torch.Tensor):
        distance = points[..., 0]
        return SimpleNamespace(
            signed_distance_m=distance,
            in_bounds=torch.ones_like(distance, dtype=torch.bool),
        )

def skeleton(frames: int = 4) -> torch.Tensor:
    value = torch.zeros(1, frames, 22, 3, dtype=torch.float32)
    value[..., 0, 2] = 0.80
    value[..., 1, 1] = 0.11
    value[..., 2, 1] = -0.11
    value[..., 1:3, 2] = 0.74
    value[..., 4, 1], value[..., 5, 1] = 0.11, -0.11
    value[..., 4:6, 2] = 0.42
    value[..., 7, 1], value[..., 8, 1] = 0.11, -0.11
    value[..., 7:9, 2] = 0.10
    value[..., 10, 1], value[..., 11, 1] = 0.11, -0.11
    value[..., 10:12, 0] = 0.12
    value[..., 10:12, 2] = 0.03
    value[..., 3:, 2] += 0.20
    # Restore the four lower-leg/foot rows after the generic upper-body lift.
    value[..., 4:6, 2] = 0.42
    value[..., 7:9, 2] = 0.10
    value[..., 10:12, 2] = 0.03
    return value

def test_proxy_has_full_current_sole_and_leg_cardinality() -> None:
    sole, legs, points, sole_groups, leg_groups = (
        physics.build_current_lower_body_proxy(skeleton())
    )
    assert sole.shape == (1, 4, 24, 3)
    assert legs.shape == (1, 4, 224, 3)
    assert points.shape == (1, 4, 248, 3)
    assert sole_groups.bincount().tolist() == [12, 12]
    assert leg_groups.bincount().tolist() == [56, 56, 56, 56]

def test_predicted_stance_penalizes_slide_but_releases_fast_root_and_swing() -> None:
    config = physics.CurrentStepwisePhysicsConfig()
    planted = skeleton(5)
    sliding = planted.clone()
    sliding[:, :, physics.FOOT_IDS, 0] += torch.linspace(0.0, 0.08, 5)[None, :, None]
    planted_stance, _ = physics.inferred_stance_probability(
        planted, ground_z_m=0.0, config=config
    )
    sliding_stance, _ = physics.inferred_stance_probability(
        sliding, ground_z_m=0.0, config=config
    )
    # Horizontal foot speed is not itself used to erase evidence of skating.
    assert sliding_stance.mean().item() == pytest.approx(
        planted_stance.mean().item(), rel=1e-5
    )

    fast_root = sliding.clone()
    shift = torch.linspace(0.0, 0.40, 5)[None, :, None]
    fast_root[..., 0] += shift
    fast_stance, _ = physics.inferred_stance_probability(
        fast_root, ground_z_m=0.0, config=config
    )
    assert fast_stance.mean() < sliding_stance.mean() * 0.05

    swing = sliding.clone()
    swing[:, 2:, 10, 2] += 0.20
    swing_stance, _ = physics.inferred_stance_probability(
        swing, ground_z_m=0.0, config=config
    )
    assert swing_stance[:, 1:, 0].mean() < sliding_stance[:, 1:, 0].mean()

def test_clean_prediction_energy_distinguishes_planted_and_sliding_feet() -> None:
    history = skeleton(1)[0]
    planted = skeleton(4).requires_grad_(True)
    sliding = skeleton(4)
    sliding[:, :, physics.FOOT_IDS, 0] += torch.linspace(0.02, 0.14, 4)[
        None, :, None
    ]
    field = FlatSDF(distance_m=1.0)
    energy = physics.CurrentStepwisePhysicsEnergy(field, history)
    planted_total, planted_diag = energy.energy(planted)
    sliding_total, sliding_diag = energy.energy(sliding)
    assert sliding_diag["p523_foot_slip_raw_energy"] > (
        planted_diag["p523_foot_slip_raw_energy"] + 0.01
    )
    assert sliding_total > planted_total
    planted_total.backward()
    assert planted.grad is not None
    assert torch.isfinite(planted.grad).all()

def test_collision_and_ground_come_from_current_scene_contract() -> None:
    current = skeleton(3)
    history = skeleton(1)[0]
    safe = physics.CurrentStepwisePhysicsEnergy(
        FlatSDF(distance_m=0.20, ground_z_m=0.0), history
    )
    colliding = physics.CurrentStepwisePhysicsEnergy(
        FlatSDF(distance_m=-0.02, ground_z_m=0.0), history
    )
    _, safe_diag = safe.energy(current)
    _, collision_diag = colliding.energy(current)
    assert collision_diag["p523_lower_body_proxy_raw_energy"] > (
        safe_diag["p523_lower_body_proxy_raw_energy"]
    )
    shifted_scene = physics.CurrentStepwisePhysicsEnergy(
        FlatSDF(distance_m=0.20, ground_z_m=0.20), history
    )
    _, shifted_diag = shifted_scene.energy(current)
    assert shifted_scene.receipt["resolved_ground_z_m"] == pytest.approx(0.20)
    assert shifted_diag["p523_floor_penetration_raw_energy"] > (
        safe_diag["p523_floor_penetration_raw_energy"]
    )

def test_shallow_floor_contact_is_separated_from_sole_obstacle_sdf() -> None:
    current = skeleton(3)
    history = skeleton(1)[0]
    energy = physics.CurrentStepwisePhysicsEnergy(
        FlatSDF(distance_m=-0.02, ground_z_m=0.0), history
    )
    _, diagnostics = energy.energy(current)
    # Sole bottom is 5 mm above ground and lies inside the explicit shallow
    # ground slab, so it is handled by the floor/contact terms rather than
    # simultaneously being repelled by a thick raw floor reconstruction.
    assert diagnostics["p523_sole_obstacle_collision_active_fraction"] == 0.0
    assert diagnostics["p523_sole_proxy_scene_raw_energy"] == 0.0
    # The separation is narrow: lower-leg obstacle collision remains active.
    assert diagnostics["p523_leg_proxy_scene_raw_energy"] > 0.0

def test_floor_and_obstacle_gradients_point_toward_physical_clearance() -> None:
    history = skeleton(1)[0]
    low = skeleton(3)
    low[..., 10:12, 2] = 0.010
    low = low.requires_grad_(True)
    floor_energy = physics.CurrentStepwisePhysicsEnergy(FlatSDF(), history)
    total, diagnostics = floor_energy.energy(low)
    assert diagnostics["p523_floor_penetration_raw_energy"] > 0.0
    total.backward()
    # Gradient descent subtracts this negative dE/dz and therefore lifts the
    # predicted foot; this tests direction, not merely nonzero magnitude.
    assert low.grad[..., 10:12, 2].mean() < 0.0

    colliding = skeleton(3)
    lower_ids = (1, 2, 4, 5, 7, 8)
    colliding[..., lower_ids, 0] = -0.20
    colliding = colliding.requires_grad_(True)
    obstacle_energy = physics.CurrentStepwisePhysicsEnergy(
        DirectionalXSDF(), history
    )
    _, obstacle_diagnostics = obstacle_energy.energy(colliding)
    proxy_term = obstacle_diagnostics["p523_leg_proxy_scene_raw_energy"]
    assert proxy_term > 0.0
    proxy_term.backward()
    # For sdf(x)=x, clearance is reached by increasing x.  The negative
    # derivative below means a clean-latent gradient-descent update does that.
    assert colliding.grad[..., lower_ids, 0].mean() < 0.0

