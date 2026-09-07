"""Differentiable target, obstacle, legal-contact and bilateral-foot branches."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence, Protocol
from dataclasses import dataclass
import math
import hashlib
import json
import os
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class TargetAwareContractError(RuntimeError):
    """A fail-closed input, provenance, or geometry contract failed."""

def require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetAwareContractError(message)

@dataclass(frozen=True)
class RefineConfig:
    steps: int = 280
    pose_learning_rate: float = 0.018
    pelvis_z_learning_rate: float = 0.012
    pelvis_xy_learning_rate: float = 0.012
    xy_relaxation_m: float = 0.0
    maximum_pose_delta_rad: float = 0.55
    maximum_pelvis_z_delta_m: float = 0.15
    sit_support_joint_closure: bool = False
    component_bounded_parameterization: bool = False
    bilateral_foot_support: bool = False
    collision_reduction: str = "cvar"
    cvar_fraction: float = 0.05
    cvar_min_count: int = 32
    cvar_max_count: int = 384
    allowed_tail_topk: int = 12
    obstacle_margin_m: float = 0.005
    target_forbidden_margin_m: float = 0.005
    target_allowed_tail_m: float = 0.035
    ground_tail_m: float = 0.025
    allowed_contact_band_m: float = 0.020
    allowed_contact_topk: int = 96
    minimum_allowed_contact_vertices: int = 16
    foot_ground_band_m: float = 0.030
    minimum_foot_ground_vertices: int = 12
    pelvis_hip_lbs_threshold: float = 0.50
    foot_lbs_threshold: float = 0.50
    maximum_target_forbidden_penetration_m: float = 0.040
    maximum_non_target_penetration_m: float = 0.040
    maximum_outside_fraction: float = 0.010
    reprojection_weight: float = 1.0
    nonsemantic_reprojection_weight: float = 0.30
    pose_prior_weight: float = 0.012
    component_limit_weight: float = 0.05
    norm_limit_weight: float = 0.05
    non_target_collision_weight: float = 35.0
    target_forbidden_collision_weight: float = 45.0
    target_allowed_tail_weight: float = 20.0
    target_contact_surface_weight: float = 0.50
    ground_penetration_weight: float = 35.0
    foot_ground_surface_weight: float = 0.15
    gradient_clip_norm: float = 5.0
    history_interval: int = 20

    def validate(self) -> "RefineConfig":
        require(type(self.steps) is int and self.steps > 0, "steps must be positive")
        require(self.collision_reduction in {"mean", "cvar"}, "collision reduction must be mean or cvar")
        require(0.0 < self.cvar_fraction <= 1.0, "cvar fraction must be in (0,1]")
        require(1 <= self.cvar_min_count <= self.cvar_max_count, "invalid CVaR count bounds")
        require(self.allowed_tail_topk > 0, "allowed-tail top-k must be positive")
        require(0.0 < self.maximum_pose_delta_rad <= 1.0, "invalid pose delta bound")
        require(0.0 < self.maximum_pelvis_z_delta_m <= 0.25, "invalid pelvis Z bound")
        require(type(self.sit_support_joint_closure) is bool, "support closure flag must be bool")
        require(type(self.component_bounded_parameterization) is bool, "component parameterization flag must be bool")
        require(type(self.bilateral_foot_support) is bool, "bilateral support flag must be bool")
        require(self.foot_ground_surface_weight > 0.0, "foot support weight must be positive")
        require(self.component_limit_weight > 0.0, "component-limit weight must be positive")
        require(self.non_target_collision_weight > 0.0, "non-target collision weight must be positive")
        require(self.pose_learning_rate > 0.0 and self.pelvis_z_learning_rate > 0.0, "learning rates must be positive")
        require(
            math.isclose(self.xy_relaxation_m, 0.0, abs_tol=1.0e-12)
            or math.isclose(self.xy_relaxation_m, 0.15, abs_tol=1.0e-12),
            "XY relaxation must be either hard lock (0) or preregistered local region (0.15m)",
        )
        require(self.pelvis_hip_lbs_threshold == 0.50, "P478 sit contact semantics require pelvis/hip LBS >= 0.5")
        require(self.allowed_contact_topk > 0, "contact top-k must be positive")
        require(
            self.target_allowed_tail_weight > 0.0
            and self.target_contact_surface_weight > 0.0
            and self.target_forbidden_collision_weight > 0.0,
            "target-aware loss weights must be positive",
        )
        return self

@dataclass(frozen=True)
class WorldGridSDF:
    values_xyz: Tensor
    lower_xyz: Tensor
    upper_xyz: Tensor

    def sample(self, points_world: Tensor) -> tuple[Tensor, Tensor]:
        lower = self.lower_xyz.to(points_world)
        upper = self.upper_xyz.to(points_world)
        shape = points_world.new_tensor(self.values_xyz.shape)
        spacing = (upper - lower) / shape
        index = (points_world - lower) / spacing - 0.5
        outside = ((index < 0.0) | (index > shape - 1.0)).any(dim=-1)
        clamped = torch.minimum(torch.maximum(index, index.new_zeros(())), shape - 1.0)
        low = torch.floor(clamped).long()
        high = torch.minimum(low + 1, (shape - 1.0).long())
        frac = clamped - low.to(clamped.dtype)
        x0, y0, z0 = low.unbind(-1)
        x1, y1, z1 = high.unbind(-1)
        fx, fy, fz = frac.unbind(-1)
        values = self.values_xyz.to(points_world)
        c000, c001 = values[x0, y0, z0], values[x0, y0, z1]
        c010, c011 = values[x0, y1, z0], values[x0, y1, z1]
        c100, c101 = values[x1, y0, z0], values[x1, y0, z1]
        c110, c111 = values[x1, y1, z0], values[x1, y1, z1]
        c00, c01 = c000 * (1.0 - fx) + c100 * fx, c001 * (1.0 - fx) + c101 * fx
        c10, c11 = c010 * (1.0 - fx) + c110 * fx, c011 * (1.0 - fx) + c111 * fx
        c0, c1 = c00 * (1.0 - fy) + c10 * fy, c01 * (1.0 - fy) + c11 * fy
        return c0 * (1.0 - fz) + c1 * fz, outside

@dataclass(frozen=True)
class TargetAwareFields:
    target: WorldGridSDF
    non_target: WorldGridSDF
    target_mask: np.ndarray
    collision_mask: np.ndarray
    component_receipt: Mapping[str, Any]

def reduction(values: Tensor, config: RefineConfig) -> Tensor:
    require(values.ndim == 1 and values.numel() > 0, "collision reduction requires a nonempty vector")
    squared = values.square()
    if config.collision_reduction == "mean":
        return squared.mean()
    count = int(math.ceil(float(values.numel()) * config.cvar_fraction))
    count = min(
        int(values.numel()),
        max(config.cvar_min_count, min(config.cvar_max_count, count)),
    )
    return torch.topk(squared, k=count, largest=True).values.mean()

def allowed_tail_reduction(values: Tensor, config: RefineConfig) -> Tensor:
    """Use a tighter top-k for legal-contact tails than for large body sets."""

    require(values.ndim == 1 and values.numel() > 0, "allowed tail requires nonempty values")
    if config.collision_reduction == "mean":
        return values.square().mean()
    count = min(int(values.numel()), config.allowed_tail_topk)
    return torch.topk(values.square(), k=count, largest=True).values.mean()

@dataclass(frozen=True)
class AnatomyMasks:
    allowed_contact: Tensor
    foot_support: Tensor
    receipt: Mapping[str, Any]
    left_foot_support: Tensor | None = None
    right_foot_support: Tensor | None = None

def build_anatomy_masks(body_model: nn.Module, vertex_count: int, config: RefineConfig) -> AnatomyMasks:
    weights = getattr(body_model, "lbs_weights", None)
    require(isinstance(weights, Tensor), "SMPL-X model lacks LBS weights")
    weights = weights[:, :22]
    require(weights.shape[0] == vertex_count, "SMPL-X LBS/vertex count drift")
    allowed_influence = weights[:, (0, 1, 2)].sum(dim=1)
    left_foot_influence = weights[:, (7, 10)].sum(dim=1)
    right_foot_influence = weights[:, (8, 11)].sum(dim=1)
    foot_influence = left_foot_influence + right_foot_influence
    allowed = allowed_influence >= config.pelvis_hip_lbs_threshold
    feet = foot_influence >= config.foot_lbs_threshold
    left_feet = left_foot_influence >= config.foot_lbs_threshold
    right_feet = right_foot_influence >= config.foot_lbs_threshold
    require(
        bool(allowed.any()) and bool(feet.any()) and bool(left_feet.any()) and bool(right_feet.any()),
        "empty anatomy mask",
    )
    dominant = weights.argmax(dim=1)
    dominant_allowed = sorted(set(int(value) for value in dominant[allowed].detach().cpu().tolist()))
    require(set(dominant_allowed).issubset({0, 1, 2}), "LBS>=0.5 sit mask includes a non pelvis/hip dominant joint")
    receipt = {
        "allowed_contact_policy": "sum_SMPXLBS(joints_0_1_2)>=0.50",
        "allowed_contact_joint_ids": [0, 1, 2],
        "allowed_contact_vertex_count": int(allowed.sum().detach().cpu()),
        "allowed_contact_dominant_joint_ids": dominant_allowed,
        "foot_support_policy": "sum_SMPXLBS(joints_7_8_10_11)>=0.50",
        "foot_support_joint_ids": [7, 8, 10, 11],
        "foot_support_vertex_count": int(feet.sum().detach().cpu()),
        "left_foot_support_vertex_count": int(left_feet.sum().detach().cpu()),
        "right_foot_support_vertex_count": int(right_feet.sum().detach().cpu()),
        "bilateral_foot_support_loss": config.bilateral_foot_support,
        "spatial_radius_contact_mask_used": False,
    }
    return AnatomyMasks(
        allowed_contact=allowed,
        foot_support=feet,
        left_foot_support=left_feet,
        right_foot_support=right_feet,
        receipt=receipt,
    )

@dataclass(frozen=True)
class BranchTerms:
    loss_non_target: Tensor
    loss_target_forbidden: Tensor
    loss_target_allowed_tail: Tensor
    loss_target_contact_surface: Tensor
    loss_ground_penetration: Tensor
    loss_foot_ground_surface: Tensor
    diagnostics: Mapping[str, Tensor | int]

def branch_terms(
    vertices: Tensor,
    fields: TargetAwareFields,
    masks: AnatomyMasks,
    config: RefineConfig,
) -> BranchTerms:
    target, target_outside = fields.target.sample(vertices)
    obstacle, obstacle_outside = fields.non_target.sample(vertices)
    valid = ~(target_outside | obstacle_outside)
    require(bool(valid.any()), "every body vertex is outside the scene SDF")
    allowed = masks.allowed_contact.to(vertices.device) & valid
    forbidden = ~masks.allowed_contact.to(vertices.device) & valid
    feet = masks.foot_support.to(vertices.device)
    left_feet = (
        masks.left_foot_support.to(vertices.device)
        if masks.left_foot_support is not None else feet
    )
    right_feet = (
        masks.right_foot_support.to(vertices.device)
        if masks.right_foot_support is not None else feet
    )
    require(bool(allowed.any()) and bool(forbidden.any()) and bool(feet.any()), "empty branch mask")

    obstacle_violation = F.relu(config.obstacle_margin_m - obstacle[valid])
    forbidden_violation = F.relu(config.target_forbidden_margin_m - target[forbidden])
    allowed_tail_violation = F.relu(-target[allowed] - config.target_allowed_tail_m)
    ground_violation = F.relu(-vertices[:, 2] - config.ground_tail_m)

    contact_count = min(config.allowed_contact_topk, int(allowed.sum().detach().cpu()))
    closest_contact = torch.topk(target[allowed].abs(), k=contact_count, largest=False).values
    foot_count = min(64, int(feet.sum().detach().cpu()))
    closest_ground = torch.topk(vertices[feet, 2].abs(), k=foot_count, largest=False).values
    left_count = min(32, int(left_feet.sum().detach().cpu()))
    right_count = min(32, int(right_feet.sum().detach().cpu()))
    closest_left_ground = torch.topk(
        vertices[left_feet, 2].abs(), k=left_count, largest=False
    ).values
    closest_right_ground = torch.topk(
        vertices[right_feet, 2].abs(), k=right_count, largest=False
    ).values

    def penetration(values: Tensor) -> Tensor:
        return F.relu(-values).max() if values.numel() else vertices.new_zeros(())

    diagnostics: dict[str, Tensor | int] = {
        "queried_vertex_count": int(vertices.shape[0]),
        "outside_fraction": (target_outside | obstacle_outside).to(vertices.dtype).mean(),
        "non_target_scene_minimum_m": obstacle[valid].min(),
        "non_target_scene_penetration_m": penetration(obstacle[valid]),
        "non_target_scene_negative_vertex_count": int((obstacle[valid] < 0.0).sum().detach().cpu()),
        "target_forbidden_body_minimum_m": target[forbidden].min(),
        "target_forbidden_body_penetration_m": penetration(target[forbidden]),
        "target_forbidden_negative_vertex_count": int((target[forbidden] < 0.0).sum().detach().cpu()),
        "target_allowed_contact_minimum_m": target[allowed].min(),
        "target_allowed_contact_penetration_m": penetration(target[allowed]),
        "target_allowed_negative_vertex_count": int((target[allowed] < 0.0).sum().detach().cpu()),
        "target_allowed_surface_topk_mean_m": closest_contact.mean(),
        "target_allowed_contact_vertex_count_in_band": int(
            (target[allowed].abs() <= config.allowed_contact_band_m).sum().detach().cpu()
        ),
        "ground_penetration_m": F.relu(-vertices[:, 2]).max(),
        "foot_ground_surface_topk_mean_m": closest_ground.mean(),
        "left_foot_ground_surface_topk_mean_m": closest_left_ground.mean(),
        "right_foot_ground_surface_topk_mean_m": closest_right_ground.mean(),
        "foot_ground_vertex_count_in_band": int(
            (vertices[feet, 2].abs() <= config.foot_ground_band_m).sum().detach().cpu()
        ),
        "left_foot_ground_vertex_count_in_band": int(
            (vertices[left_feet, 2].abs() <= config.foot_ground_band_m).sum().detach().cpu()
        ),
        "right_foot_ground_vertex_count_in_band": int(
            (vertices[right_feet, 2].abs() <= config.foot_ground_band_m).sum().detach().cpu()
        ),
    }
    return BranchTerms(
        loss_non_target=reduction(obstacle_violation, config),
        loss_target_forbidden=reduction(forbidden_violation, config),
        loss_target_allowed_tail=allowed_tail_reduction(allowed_tail_violation, config),
        loss_target_contact_surface=closest_contact.square().mean(),
        loss_ground_penetration=reduction(ground_violation, config),
        loss_foot_ground_surface=(
            0.5 * (
                closest_left_ground.square().mean()
                + closest_right_ground.square().mean()
            )
            if config.bilateral_foot_support
            else closest_ground.square().mean()
        ),
        diagnostics=diagnostics,
    )
