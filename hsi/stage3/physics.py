"""Current-only foot support and lower-body collision energy for P523 Stage3.

The energy is evaluated on the differentiable clean-latent J22 decode inside
each ReMoGen DDPM step.  Its only temporal context is the detached causal
prefix already owned by P478.  It does not read a generated motion file,
dataset/future motion, retrieval, memory, or any historical experiment
artifact, and it never edits decoded frames after sampling.

The stance mask is deliberately *inferred*, not declared.  A foot is likely
to be in stance only when the current predicted-clean foot is near the floor,
its vertical speed is small, and the predicted pelvis speed is below a release
threshold.  The mask is detached before it weights the loss, which prevents a
gradient from reducing the loss merely by changing the inferred phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


POLICY = "current_predicted_stance_clean_xstart_v1"
SCHEMA = "p523.current_stepwise_physics_guidance.v1"
PELVIS_ID = 0
FOOT_IDS = (10, 11)
ANKLE_FOOT_PAIRS = ((7, 10), (8, 11))
LEG_SEGMENTS = ((1, 4, 0.085), (4, 7, 0.070),
                (2, 5, 0.085), (5, 8, 0.070))
SOLE_FORWARD_OFFSETS_M = (-0.055, 0.0, 0.065, 0.130)
SOLE_LATERAL_OFFSETS_M = (-0.040, 0.0, 0.040)
LEG_ALPHAS = (0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875)
LEG_ANGLES_RAD = tuple(math.radians(value) for value in range(0, 360, 45))


class CurrentPhysicsGuidanceError(ValueError):
    """A current-only physics input or configuration is unsafe."""


def _finite_positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    numeric = float(value)
    valid = math.isfinite(numeric) and (numeric >= 0.0 if allow_zero else numeric > 0.0)
    if not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise CurrentPhysicsGuidanceError(f"{name} must be finite and {qualifier}")


@dataclass(frozen=True)
class CurrentStepwisePhysicsConfig:
    """Fixed, auditable limits for generated-only stance guidance."""

    policy: str = POLICY
    fps: float = 20.0
    foot_joint_contact_height_m: float = 0.12
    height_temperature_m: float = 0.020
    maximum_stance_vertical_speed_mps: float = 0.35
    vertical_speed_temperature_mps: float = 0.10
    root_speed_release_mps: float = 0.85
    root_speed_temperature_mps: float = 0.15
    slip_tolerance_mps: float = 0.08
    slip_huber_beta_mps: float = 0.10
    sole_drop_m: float = 0.025
    floor_clearance_m: float = 0.005
    contact_band_half_width_m: float = 0.020
    scene_clearance_m: float = 0.008
    sole_scene_ground_separation_height_m: float = 0.050
    slip_weight: float = 1.0
    contact_band_weight: float = 0.35
    floor_penetration_weight: float = 3.0
    lower_body_collision_weight: float = 1.5
    proxy_topk_per_group: int = 4

    def validate(self) -> "CurrentStepwisePhysicsConfig":
        if self.policy != POLICY:
            raise CurrentPhysicsGuidanceError(f"unsupported policy: {self.policy}")
        for name in (
            "fps", "foot_joint_contact_height_m", "height_temperature_m",
            "maximum_stance_vertical_speed_mps", "vertical_speed_temperature_mps",
            "root_speed_release_mps", "root_speed_temperature_mps",
            "slip_tolerance_mps", "slip_huber_beta_mps", "sole_drop_m",
            "floor_clearance_m", "contact_band_half_width_m",
            "scene_clearance_m", "sole_scene_ground_separation_height_m",
        ):
            _finite_positive(float(getattr(self, name)), name)
        for name in (
            "slip_weight", "contact_band_weight", "floor_penetration_weight",
            "lower_body_collision_weight",
        ):
            _finite_positive(float(getattr(self, name)), name, allow_zero=True)
        caps = {
            "slip_weight": 4.0,
            "contact_band_weight": 2.0,
            "floor_penetration_weight": 8.0,
            "lower_body_collision_weight": 4.0,
        }
        for name, cap in caps.items():
            if float(getattr(self, name)) > cap:
                raise CurrentPhysicsGuidanceError(f"{name} must be <= {cap:g}")
        if self.root_speed_release_mps > 2.0:
            raise CurrentPhysicsGuidanceError(
                "root_speed_release_mps must be <= 2 to release fast locomotion"
            )
        if self.sole_scene_ground_separation_height_m <= self.floor_clearance_m:
            raise CurrentPhysicsGuidanceError(
                "sole/scene ground separation must exceed floor clearance"
            )
        if self.sole_scene_ground_separation_height_m > 0.10:
            raise CurrentPhysicsGuidanceError(
                "sole/scene ground separation must be <= 0.10 m"
            )
        if int(self.proxy_topk_per_group) != 4:
            raise CurrentPhysicsGuidanceError("proxy_topk_per_group must be exactly 4")
        if self.foot_joint_contact_height_m <= self.sole_drop_m:
            raise CurrentPhysicsGuidanceError(
                "foot contact height must exceed the anatomical sole drop"
            )
        return self

    def receipt(self, *, ground_z_m: float | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "policy": self.policy,
            "enabled": True,
            "config": asdict(self),
            "guidance_domain": "predicted_clean_latent_world_j22",
            "guidance_location": "inside_every_reached_ddpm_denoising_step",
            "history_source": "p478_detached_causal_prefix_only",
            "stance_source": (
                "current_pred_xstart_height_vertical_speed_and_pelvis_speed"
            ),
            "stance_mask_detached": True,
            "root_speed_or_swing_phase_releases_stance": True,
            "ground_z_source": "current_raw_scene_sdf.lower_xyz[2]",
            "foot_joint_ids": list(FOOT_IDS),
            "sole_points_per_frame": 24,
            "lower_leg_proxy_points_per_frame": 224,
            "sole_shallow_ground_collision_policy": (
                "sole_raw_sdf_disabled_only_inside_ground_separation_slab"
            ),
            "foot_and_ankle_j22_still_use_parent_full_raw_sdf": True,
            "lower_leg_proxy_still_uses_full_raw_sdf": True,
            "posthoc_motion_edit": False,
            "generated_motion_file_read": False,
            "future_or_ground_truth_motion_read": False,
            "retrieval_or_memory_read": False,
            "historical_seed_motion_or_metric_read": False,
        }
        if ground_z_m is not None:
            value["resolved_ground_z_m"] = float(ground_z_m)
        return value


def _require_j22(value: Tensor, label: str) -> Tensor:
    if (
        not torch.is_tensor(value)
        or not value.is_floating_point()
        or value.ndim != 4
        or value.shape[-2:] != (22, 3)
        or not bool(torch.isfinite(value).all())
    ):
        raise CurrentPhysicsGuidanceError(
            f"{label} must be finite floating [B,T,22,3]"
        )
    return value


def _history_j22(value: Tensor, like: Tensor) -> Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise CurrentPhysicsGuidanceError("causal history must be floating J22")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if (
        value.ndim != 4
        or value.shape[-2:] != (22, 3)
        or value.shape[0] not in (1, like.shape[0])
        or value.shape[1] < 1
        or not bool(torch.isfinite(value).all())
    ):
        raise CurrentPhysicsGuidanceError(
            "causal history must be finite [H,22,3] or [B,H,22,3]"
        )
    # One prior frame is sufficient for the only cross-boundary velocity.  Do
    # the slice before the device transfer so a long causal rollout never
    # creates per-denoise-step traffic proportional to its full duration.
    value = value[:, -1:].to(device=like.device, dtype=like.dtype)
    if value.shape[0] == 1 and like.shape[0] != 1:
        value = value.expand(like.shape[0], -1, -1, -1)
    return value.detach()


def ground_z_from_current_scene(scene_sdf: Any) -> float:
    """Resolve the floor exclusively from the current raw-scene bounds."""

    lower = getattr(scene_sdf, "lower_xyz", None)
    if not isinstance(lower, (tuple, list)) or len(lower) != 3:
        raise CurrentPhysicsGuidanceError(
            "current raw scene SDF must expose three-value lower_xyz bounds"
        )
    values = tuple(float(item) for item in lower)
    if not all(math.isfinite(item) for item in values):
        raise CurrentPhysicsGuidanceError("current scene lower bounds are non-finite")
    return values[2]


def _safe_normalize(value: Tensor, fallback: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return torch.where(
        (norm > 1.0e-8).expand_as(value),
        value / norm.clamp_min(1.0e-8),
        fallback.expand_as(value),
    )


def build_current_lower_body_proxy(
    joints_world_zup: Tensor,
    config: CurrentStepwisePhysicsConfig = CurrentStepwisePhysicsConfig(),
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return fresh 24-sole/224-leg proxy points and their group IDs."""

    joints = _require_j22(joints_world_zup, "current predicted joints")
    config.validate()
    shape = (*joints.shape[:2], 3)
    up = joints.new_tensor((0.0, 0.0, 1.0)).expand(shape)
    fallback_x = joints.new_tensor((1.0, 0.0, 0.0)).expand(shape)

    sole_groups: list[Tensor] = []
    sole_group_ids: list[int] = []
    for side, (ankle_id, foot_id) in enumerate(ANKLE_FOOT_PAIRS):
        ankle = joints[:, :, ankle_id]
        foot = joints[:, :, foot_id]
        raw = foot - ankle
        horizontal = torch.stack(
            (raw[..., 0], raw[..., 1], torch.zeros_like(raw[..., 0])), dim=-1
        )
        forward = _safe_normalize(horizontal, fallback_x)
        lateral = torch.stack(
            (-forward[..., 1], forward[..., 0], torch.zeros_like(forward[..., 0])),
            dim=-1,
        )
        points = []
        for forward_offset in SOLE_FORWARD_OFFSETS_M:
            for lateral_offset in SOLE_LATERAL_OFFSETS_M:
                xy_point = (
                    foot + float(forward_offset) * forward
                    + float(lateral_offset) * lateral
                )
                point = torch.stack(
                    (xy_point[..., 0], xy_point[..., 1],
                     foot[..., 2] - float(config.sole_drop_m)),
                    dim=-1,
                )
                points.append(point)
        group = torch.stack(points, dim=2)
        sole_groups.append(group)
        sole_group_ids.extend([side] * int(group.shape[2]))
    sole = torch.cat(sole_groups, dim=2)

    leg_groups: list[Tensor] = []
    leg_group_ids: list[int] = []
    for group_id, (parent_id, child_id, radius) in enumerate(LEG_SEGMENTS):
        parent = joints[:, :, parent_id]
        child = joints[:, :, child_id]
        axis = _safe_normalize(child - parent, up)
        use_x = torch.abs((axis * up).sum(dim=-1, keepdim=True)) > 0.90
        reference = torch.where(use_x.expand_as(axis), fallback_x, up)
        radial_a = _safe_normalize(
            torch.linalg.cross(axis, reference, dim=-1), fallback_x
        )
        radial_b = _safe_normalize(
            torch.linalg.cross(axis, radial_a, dim=-1), up
        )
        points = []
        for alpha in LEG_ALPHAS:
            center = parent + float(alpha) * (child - parent)
            for angle in LEG_ANGLES_RAD:
                radial = math.cos(angle) * radial_a + math.sin(angle) * radial_b
                points.append(center + float(radius) * radial)
        group = torch.stack(points, dim=2)
        leg_groups.append(group)
        leg_group_ids.extend([group_id] * int(group.shape[2]))
    legs = torch.cat(leg_groups, dim=2)
    if sole.shape[2] != 24 or legs.shape[2] != 224:
        raise CurrentPhysicsGuidanceError("lower-body proxy cardinality drift")
    return (
        sole,
        legs,
        torch.cat((sole, legs), dim=2),
        torch.as_tensor(sole_group_ids, device=joints.device, dtype=torch.long),
        torch.as_tensor(leg_group_ids, device=joints.device, dtype=torch.long),
    )


def _group_topk_mean(values: Tensor, ids: Tensor, groups: int, k: int) -> Tensor:
    rows = []
    for group in range(groups):
        selected = values[..., ids == group]
        if selected.shape[-1] < k:
            raise CurrentPhysicsGuidanceError("proxy group is smaller than top-k")
        rows.append(torch.topk(selected, k=k, dim=-1, sorted=False).values.mean(-1))
    return torch.stack(rows, dim=-1).mean()


def _smooth_l1_violation(value: Tensor, tolerance: float, beta: float) -> Tensor:
    violation = F.relu(value - float(tolerance))
    return F.smooth_l1_loss(
        violation, torch.zeros_like(violation), reduction="none", beta=float(beta)
    )


def inferred_stance_probability(
    chain_world_j22: Tensor,
    *,
    ground_z_m: float,
    config: CurrentStepwisePhysicsConfig = CurrentStepwisePhysicsConfig(),
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Infer a detached stance probability from the current clean prediction."""

    chain = _require_j22(chain_world_j22, "causal-prefix/current chain")
    config.validate()
    if chain.shape[1] < 2:
        raise CurrentPhysicsGuidanceError("stance inference needs at least two frames")
    feet = chain[:, :, FOOT_IDS]
    velocity = (feet[:, 1:] - feet[:, :-1]) * float(config.fps)
    pelvis_velocity = (
        chain[:, 1:, PELVIS_ID, :2] - chain[:, :-1, PELVIS_ID, :2]
    ) * float(config.fps)
    foot_height = torch.maximum(feet[:, 1:, :, 2], feet[:, :-1, :, 2])
    vertical_speed = torch.abs(velocity[..., 2])
    root_speed = torch.linalg.vector_norm(pelvis_velocity, dim=-1)
    height_probability = torch.sigmoid(
        (float(ground_z_m + config.foot_joint_contact_height_m) - foot_height)
        / float(config.height_temperature_m)
    )
    vertical_probability = torch.sigmoid(
        (float(config.maximum_stance_vertical_speed_mps) - vertical_speed)
        / float(config.vertical_speed_temperature_mps)
    )
    root_release = torch.sigmoid(
        (float(config.root_speed_release_mps) - root_speed)
        / float(config.root_speed_temperature_mps)
    ).unsqueeze(-1)
    probability = (height_probability * vertical_probability * root_release).detach()
    return probability, {
        "height_probability": height_probability.detach(),
        "vertical_probability": vertical_probability.detach(),
        "root_release_probability": root_release.detach(),
        "root_speed_mps": root_speed.detach(),
        "foot_vertical_speed_mps": vertical_speed.detach(),
    }


class CurrentStepwisePhysicsEnergy:
    """Additive clean-xstart physics energy using only current scene/prefix."""

    def __init__(
        self,
        scene_sdf: Any,
        causal_history_world_j22: Tensor,
        config: CurrentStepwisePhysicsConfig = CurrentStepwisePhysicsConfig(),
    ) -> None:
        if not callable(getattr(scene_sdf, "sample", None)):
            raise CurrentPhysicsGuidanceError("current scene SDF must provide sample")
        self.scene_sdf = scene_sdf
        self.causal_history_world_j22 = causal_history_world_j22
        self.config = config.validate()
        self.ground_z_m = ground_z_from_current_scene(scene_sdf)

    @property
    def receipt(self) -> dict[str, Any]:
        return self.config.receipt(ground_z_m=self.ground_z_m)

    def energy(
        self, current_world_j22: Tensor, *, valid_future_frames: int | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        current = _require_j22(current_world_j22, "current predicted joints")
        valid = current.shape[1] if valid_future_frames is None else int(valid_future_frames)
        if not 1 <= valid <= current.shape[1]:
            raise CurrentPhysicsGuidanceError("valid future frame count is invalid")
        future = current[:, :valid]
        history = _history_j22(self.causal_history_world_j22, future)
        chain = torch.cat((history, future), dim=1)
        stance, stance_diagnostics = inferred_stance_probability(
            chain, ground_z_m=self.ground_z_m, config=self.config
        )

        feet = chain[:, :, FOOT_IDS]
        foot_velocity = (feet[:, 1:] - feet[:, :-1]) * float(self.config.fps)
        horizontal_speed = torch.linalg.vector_norm(foot_velocity[..., :2], dim=-1)
        slip_per_edge = _smooth_l1_violation(
            horizontal_speed,
            self.config.slip_tolerance_mps,
            self.config.slip_huber_beta_mps,
        )
        stance_sum = stance.sum()
        slip = (
            (slip_per_edge * stance).sum() / stance_sum.clamp_min(1.0e-8)
            if bool(stance_sum.detach() > 1.0e-8)
            else future.sum() * 0.0
        )

        sole, legs, proxy, sole_groups, leg_groups = build_current_lower_body_proxy(
            future, self.config
        )
        sole_z = sole[..., 2]
        floor_level = float(self.ground_z_m + self.config.floor_clearance_m)
        floor_penetration = F.relu(floor_level - sole_z)
        floor_energy = floor_penetration.square().mean()

        # Twelve points belong to each foot.  Their shared Z value reduces to
        # one stance-conditioned band observation per side/frame.
        sole_z_by_foot = torch.stack(
            (sole_z[..., sole_groups == 0].mean(-1),
             sole_z[..., sole_groups == 1].mean(-1)), dim=-1
        )
        contact_band_distance = torch.abs(sole_z_by_foot - floor_level)
        contact_band = _smooth_l1_violation(
            contact_band_distance,
            self.config.contact_band_half_width_m,
            self.config.contact_band_half_width_m,
        )
        # stance has one boundary-to-first-future row followed by F-1 current
        # rows, so it aligns exactly with the F future frames.
        contact_band_energy = (
            (contact_band * stance).sum() / stance_sum.clamp_min(1.0e-8)
            if bool(stance_sum.detach() > 1.0e-8)
            else future.sum() * 0.0
        )

        sampled = self.scene_sdf.sample(proxy)
        sdf = sampled.signed_distance_m
        if sdf.shape != proxy.shape[:-1]:
            raise CurrentPhysicsGuidanceError("current SDF/proxy shape mismatch")
        sole_violation_unmasked = F.relu(
            float(self.config.scene_clearance_m) - sdf[..., : sole.shape[2]]
        ).square()
        # The raw occupancy contract clears the nominal floor layer, but a
        # thick/reconstructed floor can still leave shallow voxels above that
        # cut.  Do not let the sole's obstacle term repel the same contact that
        # the explicit ground band attracts.  This exemption is restricted to
        # bottom-surface proxy points inside a 5 cm ground slab: ankle/foot J22
        # remain in P360's parent full-SDF term, and all 224 leg points remain
        # collision-active here.
        sole_obstacle_mask = sole_z > float(
            self.ground_z_m
            + self.config.sole_scene_ground_separation_height_m
        )
        sole_violation = sole_violation_unmasked * sole_obstacle_mask.to(
            sole_violation_unmasked.dtype
        )
        leg_violation = F.relu(
            float(self.config.scene_clearance_m) - sdf[..., sole.shape[2] :]
        ).square()
        k = int(self.config.proxy_topk_per_group)
        sole_scene = _group_topk_mean(sole_violation, sole_groups, 2, k)
        leg_scene = _group_topk_mean(leg_violation, leg_groups, 4, k)
        proxy_collision = 0.5 * (sole_scene + leg_scene)

        weighted_slip = float(self.config.slip_weight) * slip
        weighted_band = float(self.config.contact_band_weight) * contact_band_energy
        weighted_floor = float(self.config.floor_penetration_weight) * floor_energy
        weighted_proxy = (
            float(self.config.lower_body_collision_weight) * proxy_collision
        )
        total = weighted_slip + weighted_band + weighted_floor + weighted_proxy
        root_release = stance_diagnostics["root_release_probability"]
        return total, {
            "p523_physics_total_energy": total,
            "p523_foot_slip_raw_energy": slip,
            "p523_foot_slip_weighted_energy": weighted_slip,
            "p523_contact_band_raw_energy": contact_band_energy,
            "p523_contact_band_weighted_energy": weighted_band,
            "p523_floor_penetration_raw_energy": floor_energy,
            "p523_floor_penetration_weighted_energy": weighted_floor,
            "p523_lower_body_proxy_raw_energy": proxy_collision,
            "p523_lower_body_proxy_weighted_energy": weighted_proxy,
            "p523_sole_proxy_scene_raw_energy": sole_scene,
            "p523_leg_proxy_scene_raw_energy": leg_scene,
            "p523_sole_obstacle_collision_active_fraction": (
                sole_obstacle_mask.to(future.dtype).mean()
            ),
            "p523_stance_probability_mean": stance.mean(),
            "p523_stance_probability_max": stance.amax(),
            "p523_root_release_probability_mean": root_release.mean(),
            "p523_predicted_root_speed_mps_mean": (
                stance_diagnostics["root_speed_mps"].mean()
            ),
            "p523_predicted_foot_vertical_speed_mps_mean": (
                stance_diagnostics["foot_vertical_speed_mps"].mean()
            ),
            "p523_predicted_foot_horizontal_speed_mps_mean": horizontal_speed.mean(),
            "p523_proxy_minimum_scene_sdf_m": sdf.amin(),
            "p523_proxy_in_bounds_fraction": sampled.in_bounds.to(sdf.dtype).mean(),
            "p523_minimum_sole_z_m": sole_z.amin(),
            "p523_ground_z_m": future.new_tensor(self.ground_z_m),
            "p523_sole_points_per_frame": future.new_tensor(24.0),
            "p523_lower_leg_proxy_points_per_frame": future.new_tensor(224.0),
            "p523_causal_history_frames_visible": future.new_tensor(
                float(history.shape[1])
            ),
        }


__all__ = [
    "CurrentPhysicsGuidanceError",
    "CurrentStepwisePhysicsConfig",
    "CurrentStepwisePhysicsEnergy",
    "POLICY",
    "SCHEMA",
    "build_current_lower_body_proxy",
    "ground_z_from_current_scene",
    "inferred_stance_probability",
]

