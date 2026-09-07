"""In-denoising raw-scene SDF and sparse-joint ICGF guidance for ReMoGen.

Both guidance modes evaluate the same differentiable path:

``pred_xstart -> frozen MVAE -> future J22 -> world Z-up -> scene energy``

``legacy_mean`` returns ``-dE/dx_t`` for ReMoGen's
``condition_mean_with_grad`` contract.  ``xstart_posterior`` instead returns a
bounded ``-dE/d(pred_xstart)`` step which this module inserts by recomputing the
exact DDPM posterior.  Neither mode retrieves a motion nor edits a decoded
frame.  The only behavioral target is the explicit query-time
:class:`QueryTimeICGF` supplied by the planner/image-keypose frontend;
collision comes from the raw scene SDF.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from hsi.stage3.sdf import QueryTimeICGF, RawSceneSDFZUp


class ReMoGenSceneGuidanceError(ValueError):
    """The ReMoGen/scene guidance contract is inconsistent."""


@dataclass(frozen=True)
class SceneGuidanceConfig:
    """Weights and schedules for one ReMoGen primitive."""

    sdf_weight: float = 1.0
    icgf_weight: float = 1.0
    collision_clearance_m: float = 0.01
    icgf_deadzone_m: float = 0.03
    icgf_huber_beta_m: float = 0.04
    contact_frames: int = 4
    exempt_contact_joints_from_sdf_tail: bool = True
    guidance_scale: float = 1.0
    max_grad_norm: float = 1.0
    denoise_start_fraction: float = 0.40
    schedule_power: float = 1.0
    pcgrad_goal: bool = True
    goal_tolerance_m: float = 0.20
    guidance_mode: str = "legacy_mean"
    normalize_scene_gradient: bool = False
    clean_latent_step_size: float = 0.10
    max_clean_latent_update_norm: float = 0.15
    gradient_norm_epsilon: float = 1.0e-10

    def validate(self) -> "SceneGuidanceConfig":
        nonnegative = (
            "sdf_weight",
            "icgf_weight",
            "collision_clearance_m",
            "icgf_deadzone_m",
            "guidance_scale",
            "max_grad_norm",
            "schedule_power",
            "goal_tolerance_m",
            "clean_latent_step_size",
            "max_clean_latent_update_norm",
            "gradient_norm_epsilon",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ReMoGenSceneGuidanceError(f"{name} must be finite and nonnegative")
        if not math.isfinite(float(self.icgf_huber_beta_m)) or self.icgf_huber_beta_m <= 0.0:
            raise ReMoGenSceneGuidanceError("icgf_huber_beta_m must be finite and positive")
        if int(self.contact_frames) < 1:
            raise ReMoGenSceneGuidanceError("contact_frames must be positive")
        if not 0.0 <= float(self.denoise_start_fraction) < 1.0:
            raise ReMoGenSceneGuidanceError(
                "denoise_start_fraction must lie in [0,1)"
            )
        if self.guidance_mode not in ("legacy_mean", "xstart_posterior"):
            raise ReMoGenSceneGuidanceError(
                "guidance_mode must be legacy_mean or xstart_posterior"
            )
        if float(self.gradient_norm_epsilon) <= 0.0:
            raise ReMoGenSceneGuidanceError(
                "gradient_norm_epsilon must be positive"
            )
        return self


def _joint_ids(
    values: Sequence[int] | Tensor | None,
    *,
    default_all: bool,
) -> tuple[int, ...]:
    if values is None:
        result = tuple(range(22)) if default_all else ()
    elif torch.is_tensor(values):
        if values.dtype == torch.bool:
            if values.ndim != 1 or values.shape[0] != 22:
                raise ReMoGenSceneGuidanceError("boolean joint mask must be [22]")
            result = tuple(int(value) for value in torch.nonzero(values).flatten().tolist())
        else:
            if values.ndim != 1:
                raise ReMoGenSceneGuidanceError("joint IDs must be one-dimensional")
            result = tuple(int(value) for value in values.tolist())
    else:
        result = tuple(int(value) for value in values)
    if len(set(result)) != len(result) or any(value < 0 or value >= 22 for value in result):
        raise ReMoGenSceneGuidanceError("joint IDs must be unique values in [0,21]")
    return result


def _translation(value: Tensor, *, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    result = value.to(device=device, dtype=dtype)
    if result.ndim == 3 and result.shape[1:] == (1, 3):
        result = result[:, 0]
    if result.ndim != 2 or result.shape[-1] != 3:
        raise ReMoGenSceneGuidanceError("translations must be [B,3] or [B,1,3]")
    if result.shape[0] == 1 and batch != 1:
        result = result.expand(batch, -1)
    if result.shape[0] != batch:
        raise ReMoGenSceneGuidanceError("translation batch differs from motion batch")
    return result


def _rotation(value: Tensor, *, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    result = value.to(device=device, dtype=dtype)
    if result.ndim != 3 or result.shape[1:] != (3, 3):
        raise ReMoGenSceneGuidanceError("rotations must be [B,3,3]")
    if result.shape[0] == 1 and batch != 1:
        result = result.expand(batch, -1, -1)
    if result.shape[0] != batch:
        raise ReMoGenSceneGuidanceError("rotation batch differs from motion batch")
    return result


def transform_future_joints_to_world_zup(
    joints_active_local_zup: Tensor,
    *,
    current_rotmat: Tensor,
    current_transl: Tensor,
    global_rotmat: Tensor,
    global_transl: Tensor,
) -> Tensor:
    """Apply ``world = Rg @ (Rc @ local + tc) + tg`` to ``[B,T,J,3]``."""

    if (
        joints_active_local_zup.ndim != 4
        or joints_active_local_zup.shape[-1] != 3
        or not bool(torch.isfinite(joints_active_local_zup).all())
    ):
        raise ReMoGenSceneGuidanceError("local joints must be finite [B,T,J,3]")
    batch = int(joints_active_local_zup.shape[0])
    device = joints_active_local_zup.device
    dtype = joints_active_local_zup.dtype
    current_r = _rotation(current_rotmat, batch=batch, device=device, dtype=dtype)
    global_r = _rotation(global_rotmat, batch=batch, device=device, dtype=dtype)
    current_t = _translation(current_transl, batch=batch, device=device, dtype=dtype)
    global_t = _translation(global_transl, batch=batch, device=device, dtype=dtype)
    initial = torch.einsum(
        "bij,btkj->btki", current_r, joints_active_local_zup
    ) + current_t[:, None, None]
    return torch.einsum("bij,btkj->btki", global_r, initial) + global_t[:, None, None]


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if mask.shape != value.shape or mask.dtype != torch.bool:
        raise ReMoGenSceneGuidanceError("masked mean requires aligned boolean mask")
    count = mask.to(value.dtype).sum()
    if bool(mask.any()):
        return value.masked_select(mask).mean()
    return value.sum() * 0.0


class ReMoGenRawSceneCondFn:
    """Return a clipped latent log-score from raw SDF and query-only ICGF."""

    def __init__(
        self,
        *,
        vae_model: torch.nn.Module,
        dataset: Any,
        history_motion: Tensor,
        scene_sdf: RawSceneSDFZUp,
        current_rotmat: Tensor,
        current_transl: Tensor,
        global_rotmat: Tensor,
        global_transl: Tensor,
        future_length: int,
        scale_latent: bool,
        num_diffusion_steps: int,
        icgf: QueryTimeICGF | None = None,
        contact_joint_ids: Sequence[int] | Tensor | None = None,
        collision_joint_ids: Sequence[int] | Tensor | None = None,
        valid_future_frames: int | None = None,
        goal_world_zup: Tensor | None = None,
        primitive_utility: Any | None = None,
        config: SceneGuidanceConfig = SceneGuidanceConfig(),
    ) -> None:
        self.vae_model = vae_model
        self.dataset = dataset
        self.primitive_utility = (
            getattr(dataset, "primitive_utility", None)
            if primitive_utility is None
            else primitive_utility
        )
        if self.primitive_utility is None or not callable(
            getattr(self.primitive_utility, "tensor_to_dict", None)
        ):
            raise ReMoGenSceneGuidanceError(
                "primitive_utility must provide tensor_to_dict"
            )
        if not callable(getattr(dataset, "denormalize", None)):
            raise ReMoGenSceneGuidanceError("dataset must provide denormalize")
        if not callable(getattr(vae_model, "decode", None)):
            raise ReMoGenSceneGuidanceError("vae_model must provide decode")
        if not torch.is_tensor(history_motion) or not history_motion.is_floating_point():
            raise ReMoGenSceneGuidanceError("history_motion must be floating point")
        self.history_motion = history_motion
        self.scene_sdf = scene_sdf
        self.icgf = icgf
        self.current_rotmat = current_rotmat
        self.current_transl = current_transl
        self.global_rotmat = global_rotmat
        self.global_transl = global_transl
        self.future_length = int(future_length)
        self.valid_future_frames = (
            self.future_length if valid_future_frames is None else int(valid_future_frames)
        )
        if self.future_length < 1 or not 1 <= self.valid_future_frames <= self.future_length:
            raise ReMoGenSceneGuidanceError("invalid future/valid frame count")
        self.scale_latent = bool(scale_latent)
        self.num_diffusion_steps = int(num_diffusion_steps)
        if self.num_diffusion_steps < 1:
            raise ReMoGenSceneGuidanceError("num_diffusion_steps must be positive")
        self.contact_joint_ids = _joint_ids(
            contact_joint_ids, default_all=False
        )
        self.collision_joint_ids = _joint_ids(
            collision_joint_ids, default_all=True
        )
        if icgf is not None and not self.contact_joint_ids:
            raise ReMoGenSceneGuidanceError(
                "an ICGF requires explicit sparse contact_joint_ids"
            )
        self.goal_world_zup = goal_world_zup
        self.config = config.validate()
        self.records: list[Dict[str, float]] = []
        self.last_diagnostics: Dict[str, float] = {}

    def decode_world_joints(self, pred_xstart: Tensor) -> Tensor:
        """Decode ReMoGen ``pred_xstart [B,1,D]`` into world J22."""

        if pred_xstart.ndim != 3 or not pred_xstart.is_floating_point():
            raise ReMoGenSceneGuidanceError("pred_xstart must be float [B,S,D]")
        latent = pred_xstart.permute(1, 0, 2)
        future_motion = self.vae_model.decode(
            latent,
            self.history_motion.to(
                device=pred_xstart.device, dtype=pred_xstart.dtype
            ),
            nfuture=self.future_length,
            scale_latent=self.scale_latent,
        )
        future_frames = self.dataset.denormalize(future_motion)
        feature = self.primitive_utility.tensor_to_dict(future_frames)
        if "joints" not in feature:
            raise ReMoGenSceneGuidanceError("decoded motion lacks joints")
        joints = feature["joints"]
        expected = pred_xstart.shape[0] * self.future_length * 22 * 3
        if not torch.is_tensor(joints) or joints.numel() != expected:
            raise ReMoGenSceneGuidanceError(
                "decoded joints do not contain B*future_length*22*3 values"
            )
        joints_local = joints.reshape(
            pred_xstart.shape[0], self.future_length, 22, 3
        )
        return transform_future_joints_to_world_zup(
            joints_local,
            current_rotmat=self.current_rotmat,
            current_transl=self.current_transl,
            global_rotmat=self.global_rotmat,
            global_transl=self.global_transl,
        )

    def energy_from_world_joints(
        self, joints_world_zup: Tensor
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Compute differentiable primitive energy and tensor diagnostics."""

        if joints_world_zup.ndim != 4 or joints_world_zup.shape[-2:] != (22, 3):
            raise ReMoGenSceneGuidanceError("world joints must be [B,T,22,3]")
        joints = joints_world_zup[:, : self.valid_future_frames]
        zero = joints.sum() * 0.0

        collision_ids = torch.as_tensor(
            self.collision_joint_ids, device=joints.device, dtype=torch.long
        )
        if collision_ids.numel():
            collision_points = joints.index_select(2, collision_ids)
            sdf_samples = self.scene_sdf.sample(collision_points)
            sdf = sdf_samples.signed_distance_m
            sdf_mask = torch.ones_like(sdf, dtype=torch.bool)
            if (
                self.config.exempt_contact_joints_from_sdf_tail
                and self.contact_joint_ids
            ):
                contact_set = set(self.contact_joint_ids)
                exemptions = torch.as_tensor(
                    [value in contact_set for value in self.collision_joint_ids],
                    device=joints.device,
                    dtype=torch.bool,
                )
                tail_count = min(
                    int(self.config.contact_frames), self.valid_future_frames
                )
                sdf_mask[:, -tail_count:, exemptions] = False
            penetration = F.relu(float(self.config.collision_clearance_m) - sdf)
            sdf_energy = _masked_mean(penetration.square(), sdf_mask)
            sdf_constraint_count = sdf_mask.to(joints.dtype).sum()
            sdf_active_count = (sdf_mask & (penetration > 0.0)).to(joints.dtype).sum()
            penetration_fraction = _masked_mean(
                (sdf < 0.0).to(sdf.dtype), sdf_mask
            )
            in_bounds_fraction = _masked_mean(
                sdf_samples.in_bounds.to(sdf.dtype), sdf_mask
            )
            if bool(sdf_mask.any()):
                masked_sdf = torch.where(
                    sdf_mask, sdf, torch.full_like(sdf, torch.inf)
                )
                minimum_sdf = masked_sdf.amin()
            else:
                minimum_sdf = zero
        else:
            sdf_energy = zero
            sdf_constraint_count = zero
            sdf_active_count = zero
            penetration_fraction = zero
            in_bounds_fraction = zero
            minimum_sdf = zero

        icgf_energy = zero
        mean_icgf_distance = zero
        final_icgf_distance = zero
        icgf_constraint_count = zero
        if self.icgf is not None and self.contact_joint_ids:
            contact_ids = torch.as_tensor(
                self.contact_joint_ids, device=joints.device, dtype=torch.long
            )
            tail_count = min(int(self.config.contact_frames), self.valid_future_frames)
            query = joints[:, -tail_count:].index_select(2, contact_ids)
            distance = self.icgf.soft_distance(query, joint_ids=contact_ids)
            violation = F.relu(distance - float(self.config.icgf_deadzone_m))
            per_value = F.smooth_l1_loss(
                violation,
                torch.zeros_like(violation),
                reduction="none",
                beta=float(self.config.icgf_huber_beta_m),
            )
            ramp = torch.linspace(
                1.0 / tail_count,
                1.0,
                tail_count,
                device=joints.device,
                dtype=joints.dtype,
            )
            weights = ramp[None, :, None].expand_as(per_value)
            icgf_energy = (per_value * weights).sum() / weights.sum().clamp_min(1.0e-8)
            icgf_constraint_count = torch.as_tensor(
                per_value.numel(), device=joints.device, dtype=joints.dtype
            )
            mean_icgf_distance = distance.mean()
            final_icgf_distance = distance[:, -1].mean()

        total = (
            float(self.config.sdf_weight) * sdf_energy
            + float(self.config.icgf_weight) * icgf_energy
        )
        return total, {
            "scene_energy": total,
            "sdf_energy": sdf_energy,
            "icgf_energy": icgf_energy,
            "sdf_constraint_count": sdf_constraint_count,
            "sdf_active_constraint_count": sdf_active_count,
            "icgf_constraint_count": icgf_constraint_count,
            "minimum_sdf_m": minimum_sdf,
            "penetration_fraction": penetration_fraction,
            "in_bounds_fraction": in_bounds_fraction,
            "mean_icgf_distance_m": mean_icgf_distance,
            "final_icgf_distance_m": final_icgf_distance,
        }

    def _goal_loss(self, joints_world_zup: Tensor) -> tuple[Tensor, Tensor]:
        if self.goal_world_zup is None:
            zero = joints_world_zup.sum() * 0.0
            return zero, zero
        goal = self.goal_world_zup.to(
            device=joints_world_zup.device, dtype=joints_world_zup.dtype
        )
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        if goal.ndim != 2 or goal.shape[-1] != 3:
            raise ReMoGenSceneGuidanceError("goal_world_zup must be [B,3]")
        if goal.shape[0] == 1 and joints_world_zup.shape[0] != 1:
            goal = goal.expand(joints_world_zup.shape[0], -1)
        if goal.shape[0] != joints_world_zup.shape[0]:
            raise ReMoGenSceneGuidanceError("goal batch differs from motion batch")
        final_pelvis_xy = joints_world_zup[:, self.valid_future_frames - 1, 0, :2]
        distance = torch.linalg.vector_norm(final_pelvis_xy - goal[:, :2], dim=-1)
        violation = F.relu(distance - float(self.config.goal_tolerance_m))
        return violation.square().mean(), distance.mean()

    def __call__(
        self,
        x: Tensor,
        t: Tensor,
        p_mean_var: Dict[str, Tensor],
        **_: Any,
    ) -> Tensor:
        """Return a scheduled negative-energy step in the configured domain."""

        if "pred_xstart" not in p_mean_var:
            raise ReMoGenSceneGuidanceError("p_mean_var lacks pred_xstart")
        with torch.enable_grad():
            pred_xstart = p_mean_var["pred_xstart"]
            joints_world = self.decode_world_joints(pred_xstart)
            energy, diagnostics = self.energy_from_world_joints(joints_world)
            # Audit the two Jacobian stages separately.  The clean-latent
            # gradient contains only the frozen MVAE decode Jacobian, whereas
            # the x_t gradient below additionally crosses the START_X
            # denoiser.  A small noisy/clean norm ratio exposes
            # denoiser-Jacobian attenuation instead of hiding it inside a
            # single scale.
            clean_latent_grad = torch.autograd.grad(
                energy,
                pred_xstart,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if clean_latent_grad is None:
                clean_latent_grad = torch.zeros_like(pred_xstart)
            clean_latent_grad = torch.nan_to_num(
                clean_latent_grad, nan=0.0, posinf=0.0, neginf=0.0
            )
            xt_scene_grad = torch.autograd.grad(
                energy,
                x,
                retain_graph=(
                    self.config.pcgrad_goal and self.goal_world_zup is not None
                ),
                allow_unused=True,
            )[0]
            if xt_scene_grad is None:
                xt_scene_grad = torch.zeros_like(x)
            xt_scene_grad = torch.nan_to_num(
                xt_scene_grad, nan=0.0, posinf=0.0, neginf=0.0
            )
            if self.config.guidance_mode == "xstart_posterior":
                scene_grad = clean_latent_grad
                gradient_target = pred_xstart
            else:
                scene_grad = xt_scene_grad
                gradient_target = x
            raw_grad = scene_grad

            goal_loss = x.new_zeros(())
            goal_distance = x.new_zeros(())
            conflict_fraction = x.new_zeros(())
            projection_fraction = x.new_zeros(())
            if self.config.pcgrad_goal and self.goal_world_zup is not None:
                goal_loss, goal_distance = self._goal_loss(joints_world)
                goal_grad = torch.autograd.grad(
                    goal_loss,
                    gradient_target,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
                if goal_grad is None:
                    goal_grad = torch.zeros_like(gradient_target)
                goal_grad = torch.nan_to_num(
                    goal_grad, nan=0.0, posinf=0.0, neginf=0.0
                )
                flat_scene = scene_grad.flatten(1)
                flat_goal = goal_grad.flatten(1)
                dot = (flat_scene * flat_goal).sum(dim=1, keepdim=True)
                goal_norm_sq = flat_goal.square().sum(dim=1, keepdim=True)
                conflict = (dot < 0.0) & (goal_norm_sq > 1.0e-12)
                coefficient = torch.where(
                    conflict,
                    dot / goal_norm_sq.clamp_min(1.0e-12),
                    torch.zeros_like(dot),
                )
                projected = flat_scene - coefficient * flat_goal
                before = flat_scene.norm(dim=1)
                change = (projected - flat_scene).norm(dim=1)
                scene_grad = projected.reshape_as(scene_grad)
                conflict_fraction = conflict.float().mean()
                projection_fraction = (
                    change / before.clamp_min(1.0e-12)
                ).mean()

            selected_norm = scene_grad.flatten(1).norm(dim=1, keepdim=True)
            if self.config.normalize_scene_gradient:
                active = selected_norm > float(self.config.gradient_norm_epsilon)
                scene_grad = torch.where(
                    active.view(-1, *([1] * (scene_grad.ndim - 1))),
                    scene_grad / selected_norm.clamp_min(
                        float(self.config.gradient_norm_epsilon)
                    ).view(-1, *([1] * (scene_grad.ndim - 1))),
                    torch.zeros_like(scene_grad),
                )
            elif self.config.max_grad_norm > 0.0:
                clip = (
                    float(self.config.max_grad_norm)
                    / selected_norm.clamp_min(1.0e-12)
                ).clamp(max=1.0)
                scene_grad = scene_grad * clip.view(
                    -1, *([1] * (scene_grad.ndim - 1))
                )
            clipped_norm = scene_grad.flatten(1).norm(dim=1)

            if self.num_diffusion_steps == 1:
                progress = torch.ones_like(t, dtype=x.dtype)
            else:
                progress = (
                    (float(self.num_diffusion_steps - 1) - t.to(x.dtype))
                    / float(self.num_diffusion_steps - 1)
                ).clamp(0.0, 1.0)
            start = float(self.config.denoise_start_fraction)
            schedule = ((progress - start).clamp_min(0.0) / (1.0 - start)).pow(
                float(self.config.schedule_power)
            )
            schedule_view = schedule.view(
                -1, *([1] * (scene_grad.ndim - 1))
            )
            magnitude = float(self.config.guidance_scale)
            if self.config.guidance_mode == "xstart_posterior":
                magnitude *= float(self.config.clean_latent_step_size)
            log_score_grad = -magnitude * schedule_view * scene_grad
            update_was_clipped = torch.zeros_like(schedule, dtype=torch.bool)
            if (
                self.config.guidance_mode == "xstart_posterior"
                and self.config.max_clean_latent_update_norm > 0.0
            ):
                update_norm = log_score_grad.flatten(1).norm(dim=1, keepdim=True)
                update_clip = (
                    float(self.config.max_clean_latent_update_norm)
                    / update_norm.clamp_min(1.0e-12)
                ).clamp(max=1.0)
                update_was_clipped = update_clip[:, 0] < 1.0
                log_score_grad = log_score_grad * update_clip.view(
                    -1, *([1] * (log_score_grad.ndim - 1))
                )
            log_score_grad = torch.nan_to_num(
                log_score_grad, nan=0.0, posinf=0.0, neginf=0.0
            )

        scalar_diagnostics = {
            name: float(value.detach().mean().cpu())
            for name, value in diagnostics.items()
        }
        scalar_diagnostics.update(
            timestep=float(t.detach().float().mean().cpu()),
            schedule=float(schedule.detach().mean().cpu()),
            goal_loss=float(goal_loss.detach().cpu()),
            goal_distance_m=float(goal_distance.detach().cpu()),
            raw_grad_norm=float(raw_grad.detach().flatten(1).norm(dim=1).mean().cpu()),
            xt_grad_norm=float(
                xt_scene_grad.detach().flatten(1).norm(dim=1).mean().cpu()
            ),
            clean_latent_grad_norm=float(
                clean_latent_grad.detach().flatten(1).norm(dim=1).mean().cpu()
            ),
            xt_to_clean_grad_norm_ratio=float(
                (
                    xt_scene_grad.detach().flatten(1).norm(dim=1)
                    / clean_latent_grad.detach().flatten(1).norm(dim=1).clamp_min(1.0e-12)
                ).mean().cpu()
            ),
            clipped_grad_norm=float(clipped_norm.detach().mean().cpu()),
            selected_gradient_normalized=float(
                bool(self.config.normalize_scene_gradient)
            ),
            clean_latent_update_clipped_fraction=float(
                update_was_clipped.detach().float().mean().cpu()
            ),
            goal_conflict_fraction=float(conflict_fraction.detach().cpu()),
            projection_fraction=float(projection_fraction.detach().cpu()),
            returned_grad_norm=float(
                log_score_grad.detach().flatten(1).norm(dim=1).mean().cpu()
            ),
        )
        self.last_diagnostics = scalar_diagnostics
        self.records.append(dict(scalar_diagnostics))
        return log_score_grad


def ddpm_sample_loop_with_scene_grad(
    diffusion: Any,
    model: torch.nn.Module,
    shape: Sequence[int],
    *,
    cond_fn: ReMoGenRawSceneCondFn,
    model_kwargs: Dict[str, Any],
    noise: Tensor | None = None,
    clip_denoised: bool = False,
) -> Tensor:
    """Run ReMoGen DDPM guidance without its ``const_noise`` signature mismatch."""

    if noise is None:
        image = torch.randn(*shape, device=next(model.parameters()).device)
    else:
        image = noise
    mode = str(cond_fn.config.guidance_mode)
    for index in range(int(diffusion.num_timesteps) - 1, -1, -1):
        timestep = torch.full(
            (int(shape[0]),), index, device=image.device, dtype=torch.long
        )
        if mode == "legacy_mean":
            with torch.no_grad():
                output = diffusion.p_sample_with_grad(
                    model,
                    image,
                    timestep,
                    clip_denoised=clip_denoised,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            conditioning_strategy = "ddpm_condition_mean"
            xstart_shift_norm = 0.0
            actual_mean_shift_norm = None
        elif mode == "xstart_posterior":
            # The scene objective is defined on the decoded clean latent.  Take
            # a bounded, optionally unit-normalized step in that same space,
            # then recompute q(x_{t-1}|x_t,x_0) exactly.  This avoids routing a
            # clean-latent loss through the often-contracting START_X denoiser
            # Jacobian and preserves ReMoGen's original DDPM variance schedule.
            with torch.enable_grad():
                x = image.detach().requires_grad_()
                base = diffusion.p_mean_variance(
                    model,
                    x,
                    timestep,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
                noise_value = torch.randn_like(x)
                clean_delta = cond_fn(
                    x, timestep, base, **model_kwargs
                ).detach()
                guided_xstart = base["pred_xstart"] + clean_delta
                guided_mean, _, _ = diffusion.q_posterior_mean_variance(
                    x_start=guided_xstart, x_t=x, t=timestep
                )
                nonzero_mask = (timestep != 0).to(x.dtype).view(
                    -1, *([1] * (x.ndim - 1))
                )
                sample = guided_mean + nonzero_mask * torch.exp(
                    0.5 * base["log_variance"]
                ) * noise_value
                actual_mean_shift_norm = float(
                    (guided_mean - base["mean"])
                    .detach()
                    .flatten(1)
                    .norm(dim=1)
                    .mean()
                    .cpu()
                )
                xstart_shift_norm = float(
                    clean_delta.flatten(1).norm(dim=1).mean().cpu()
                )
                output = {
                    "sample": sample.detach(),
                    "pred_xstart": guided_xstart.detach(),
                }
            conditioning_strategy = "xstart_step_then_exact_ddpm_posterior"
        else:  # validated by SceneGuidanceConfig; defensive for foreign cond_fn.
            raise ReMoGenSceneGuidanceError(
                f"unsupported guidance mode: {mode}"
            )
        if cond_fn.records:
            # p_sample_with_grad uses ReMoGen's fixed-small posterior variance
            # to turn the returned log-score into a mean displacement.  Save
            # that exact attenuation so an apparently nonzero condition
            # function cannot be mistaken for an effective sampler update.
            variance = torch.as_tensor(
                diffusion.posterior_variance[index],
                device=image.device,
                dtype=image.dtype,
            )
            xstart_coefficient = torch.as_tensor(
                diffusion.posterior_mean_coef1[index],
                device=image.device,
                dtype=image.dtype,
            )
            returned_norm = float(
                cond_fn.records[-1].get("returned_grad_norm", 0.0)
            )
            if mode == "xstart_posterior":
                expected_mean_shift = (
                    float(xstart_coefficient.detach().cpu()) * returned_norm
                )
            else:
                expected_mean_shift = (
                    float(variance.detach().cpu()) * returned_norm
                )
            cond_fn.records[-1].update(
                conditioning_strategy=conditioning_strategy,
                posterior_variance=float(variance.detach().cpu()),
                posterior_xstart_coefficient=float(
                    xstart_coefficient.detach().cpu()
                ),
                expected_mean_shift_norm=expected_mean_shift,
                clean_xstart_shift_norm=xstart_shift_norm,
                actual_mean_shift_norm=(
                    expected_mean_shift
                    if actual_mean_shift_norm is None
                    else actual_mean_shift_norm
                ),
            )
        image = output["sample"]
    return image.detach()


__all__ = [
    "ReMoGenRawSceneCondFn",
    "ReMoGenSceneGuidanceError",
    "SceneGuidanceConfig",
    "ddpm_sample_loop_with_scene_grad",
    "transform_future_joints_to_world_zup",
]

