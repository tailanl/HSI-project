"""Causal P478 bridge from P360 decoded joints to rolling E2.2.6 energy.

This module intentionally leaves the frozen P360 implementation untouched.
``ReMoGenE226CondFn`` subclasses P360's existing raw-scene condition function
and overrides only ``energy_from_world_joints``.  Consequently the added
energy follows exactly the same differentiable path as the existing SDF/ICGF
terms::

    pred_xstart -> frozen MVAE -> current world-Z-up J22 -> rolling energy

Only a detached prefix that has already been emitted by the rollout is kept as
history.  Planner goals/yaw are fixed conditions.  No helper in this file can
read a future frame, a ground-truth contact field, or retrieve a motion.

The bridge is batch-one by design because one P360 rollout owns one causal
history.  It is a small integration layer, not a post-processing function: it
returns an energy to the DDPM condition function and never edits joints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor







from hsi.stage3.conditioning import (  # noqa: E402
    ReMoGenRawSceneCondFn,
    ReMoGenSceneGuidanceError,
)

from hsi.stage3.rolling_energy import (  # noqa: E402
    ENERGY_NAMES,
    RollingEnergyConfig,
    RollingEnergyError,
    p360_world_zup_pelvis_energy,
)


ALLOWED_HISTORY_SOURCES: tuple[str, ...] = (
    "native_initial_history",
    "query_static_initial_history",
    "generated_prefix",
)


class P360E226BridgeError(ValueError):
    """The P478/P360 causal bridge contract is inconsistent."""


@dataclass(frozen=True)
class P360E226BridgeConfig:
    """Configuration for composing base scene and rolling pelvis energies.

    ``rolling_scale`` multiplies the already weighted total returned by
    :class:`RollingEnergyConfig`.  P360's full-body SDF remains the default
    collision term, so ``include_pelvis_sdf`` is false and the recommended
    rolling config has ``sdf_weight=0``.  The optional pelvis-only SDF path is
    retained for a controlled ablation.
    """

    enabled: bool = True
    rolling_scale: float = 1.0
    history_tail_frames: int = 64
    pelvis_joint_index: int = 0
    include_pelvis_sdf: bool = False
    forbid_duplicate_goal_guidance: bool = True
    rolling: RollingEnergyConfig = field(
        default_factory=lambda: RollingEnergyConfig(sdf_weight=0.0)
    )

    def with_memory_multiplier(self, multiplier: float) -> "P360E226BridgeConfig":
        """Return a bounded runtime copy with only the outer energy scaled.

        A three-stage memory profile may modulate the rolling energy after its
        snapshot has been frozen.  It must not rewrite goals, thresholds, or
        individual loss terms; those remain planner/query-geometry owned.
        """

        value = float(multiplier)
        if not math.isfinite(value) or value <= 0.0:
            raise P360E226BridgeError(
                "memory e226 multiplier must be finite and positive"
            )
        return replace(self, rolling_scale=float(self.rolling_scale) * value).validate()

    def validate(self) -> "P360E226BridgeConfig":
        if not math.isfinite(float(self.rolling_scale)) or self.rolling_scale < 0.0:
            raise P360E226BridgeError("rolling_scale must be finite and non-negative")
        if int(self.history_tail_frames) < 1:
            raise P360E226BridgeError("history_tail_frames must be positive")
        if int(self.pelvis_joint_index) < 0:
            raise P360E226BridgeError("pelvis_joint_index must be non-negative")
        self.rolling.validate()
        if not self.include_pelvis_sdf and float(self.rolling.sdf_weight) != 0.0:
            raise P360E226BridgeError(
                "rolling.sdf_weight must be zero when include_pelvis_sdf is false; "
                "P360 already supplies full-body SDF"
            )
        return self

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P360E226BridgeConfig":
        """Parse the JSON-friendly CLI/config contract without silent keys."""

        allowed = {
            "enabled",
            "rolling_scale",
            "history_tail_frames",
            "pelvis_joint_index",
            "include_pelvis_sdf",
            "forbid_duplicate_goal_guidance",
            "rolling",
        }
        unknown = set(value) - allowed
        if unknown:
            raise P360E226BridgeError(
                "unknown P360E226BridgeConfig keys: {}".format(sorted(unknown))
            )
        arguments = dict(value)
        rolling = arguments.get("rolling")
        if rolling is not None:
            if not isinstance(rolling, Mapping):
                raise P360E226BridgeError("rolling config must be a mapping")
            valid_rolling = set(RollingEnergyConfig.__dataclass_fields__)
            unknown_rolling = set(rolling) - valid_rolling
            if unknown_rolling:
                raise P360E226BridgeError(
                    "unknown RollingEnergyConfig keys: {}".format(
                        sorted(unknown_rolling)
                    )
                )
            arguments["rolling"] = RollingEnergyConfig(**dict(rolling))
        return cls(**arguments).validate()


def yaw_to_world_xy(yaw_rad: float, *, like: Tensor) -> Tensor:
    """Convert P142/P360 Z-up yaw to its world-XY unit heading."""

    value = float(yaw_rad)
    if not math.isfinite(value):
        raise P360E226BridgeError("active_yaw_rad must be finite")
    return torch.as_tensor(
        (math.cos(value), math.sin(value)), device=like.device, dtype=like.dtype
    )


def _world_xyz_condition(
    value: Tensor | Sequence[float] | None,
    *,
    name: str,
    like: Tensor,
) -> Tensor | None:
    if value is None:
        return None
    result = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if result.ndim == 2 and result.shape == (1, 3):
        result = result[0]
    if result.shape != (3,) or not bool(torch.isfinite(result).all()):
        raise P360E226BridgeError(f"{name} must be one finite world XYZ triplet")
    return result.detach().clone()


def _causal_generated_history(
    value: Tensor,
    *,
    source: str,
    tail_frames: int,
    pelvis_joint_index: int,
) -> Tensor:
    """Validate and freeze a generated/native prefix in world-Z-up J3 form."""

    if str(source) not in ALLOWED_HISTORY_SOURCES:
        raise P360E226BridgeError(
            "history_source must be one of {}; GT/oracle history is forbidden".format(
                ALLOWED_HISTORY_SOURCES
            )
        )
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise P360E226BridgeError("generated history must be a floating tensor")
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise P360E226BridgeError("generated history batch must be one")
        value = value[0]
    if value.ndim != 3 or value.shape[-1] != 3 or value.shape[0] < 1:
        raise P360E226BridgeError(
            "generated history must have shape [H,J,3] with H >= 1"
        )
    if int(pelvis_joint_index) >= int(value.shape[1]):
        raise P360E226BridgeError("pelvis_joint_index lies outside history joints")
    if not bool(torch.isfinite(value).all()):
        raise P360E226BridgeError("generated history must contain finite values")
    # clone prevents later rollout concatenation/mutation from changing the
    # denoising objective; detach is the explicit causal gradient boundary.
    return value[-int(tail_frames) :].detach().clone()


class P360RollingEnergyBridge:
    """Host-independent rolling-energy state used by the P360 subclass."""

    def __init__(
        self,
        *,
        generated_history_world_joints_zup: Tensor,
        history_source: str,
        active_goal_world_zup: Tensor | Sequence[float],
        next_goal_world_zup: Tensor | Sequence[float] | None = None,
        active_yaw_rad: float | None = None,
        scene_sdf: Any | None = None,
        config: P360E226BridgeConfig = P360E226BridgeConfig(),
    ) -> None:
        self.config = config.validate()
        joint = int(self.config.pelvis_joint_index)
        self.history_world_joints_zup = _causal_generated_history(
            generated_history_world_joints_zup,
            source=str(history_source),
            tail_frames=int(self.config.history_tail_frames),
            pelvis_joint_index=joint,
        )
        self.history_source = str(history_source)
        self.active_goal_world_zup = _world_xyz_condition(
            active_goal_world_zup,
            name="active_goal_world_zup",
            like=self.history_world_joints_zup,
        )
        assert self.active_goal_world_zup is not None
        self.next_goal_world_zup = _world_xyz_condition(
            next_goal_world_zup,
            name="next_goal_world_zup",
            like=self.history_world_joints_zup,
        )
        self.active_yaw_rad = (
            None if active_yaw_rad is None else float(active_yaw_rad)
        )
        if self.active_yaw_rad is not None and not math.isfinite(self.active_yaw_rad):
            raise P360E226BridgeError("active_yaw_rad must be finite")
        self.scene_sdf = scene_sdf
        if self.config.include_pelvis_sdf:
            if scene_sdf is None or not callable(getattr(scene_sdf, "sample", None)):
                raise P360E226BridgeError(
                    "include_pelvis_sdf requires scene_sdf.sample"
                )

    @property
    def receipt(self) -> dict[str, Any]:
        """Serializable proof of the bridge's causal inputs and coordinates."""

        return {
            "schema": "p478.p360_e226_bridge_receipt.v1",
            "coordinate_frame": "world_zup_xyz_m_navigation_xy",
            "history_source": self.history_source,
            "history_is_detached": not self.history_world_joints_zup.requires_grad,
            "history_frames": int(self.history_world_joints_zup.shape[0]),
            "history_tail_frames": int(self.config.history_tail_frames),
            "current_source": "differentiable_pred_xstart_decode",
            "goal_source": "planner_active_node",
            "next_goal_source": (
                "planner_next_node" if self.next_goal_world_zup is not None else None
            ),
            "yaw_source": (
                "planner_active_node" if self.active_yaw_rad is not None else None
            ),
            "posthoc_edit": False,
            "future_or_gt_query": False,
            "config": asdict(self.config),
        }

    def energy_from_world_joints(
        self, current_world_joints_zup: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Evaluate rolling energy on current decoded joints without mutation."""

        if (
            not torch.is_tensor(current_world_joints_zup)
            or not current_world_joints_zup.is_floating_point()
            or current_world_joints_zup.ndim != 4
            or current_world_joints_zup.shape[0] != 1
            or current_world_joints_zup.shape[-1] != 3
        ):
            raise P360E226BridgeError(
                "current decoded joints must be floating [1,T,J,3]"
            )
        current = current_world_joints_zup
        history = self.history_world_joints_zup.to(
            device=current.device, dtype=current.dtype
        )
        goal = self.active_goal_world_zup.to(
            device=current.device, dtype=current.dtype
        )
        next_goal = (
            None
            if self.next_goal_world_zup is None
            else self.next_goal_world_zup.to(device=current.device, dtype=current.dtype)
        )
        heading = (
            None
            if self.active_yaw_rad is None
            else yaw_to_world_xy(self.active_yaw_rad, like=current)
        )

        sdf_callback = None
        if self.config.include_pelvis_sdf:
            pelvis_z = current[0, :, int(self.config.pelvis_joint_index), 2]

            def query_pelvis_sdf(xy: Tensor) -> Any:
                xyz = torch.cat((xy, pelvis_z[:, None]), dim=-1)
                return self.scene_sdf.sample(xyz)

            sdf_callback = query_pelvis_sdf

        try:
            total, diagnostics = p360_world_zup_pelvis_energy(
                history,
                current,
                active_goal_world_zup=goal,
                next_goal_world_zup=next_goal,
                yaw_heading_xy=heading,
                pelvis_joint_index=int(self.config.pelvis_joint_index),
                sdf_callback=sdf_callback,
                config=self.config.rolling,
            )
        except RollingEnergyError as error:
            raise P360E226BridgeError(str(error)) from error
        scale = float(self.config.rolling_scale)
        # Diagnostics named as E2.2.6 energies must describe the effective
        # contribution after the outer release scale.  Geometric diagnostics
        # (path length, direct distance, ratios, SDF measurements) remain raw.
        # This is identical for the legacy/default scale=1 path, while making
        # terminal mode ``off`` auditable as exact zero rather than merely a
        # zero returned total accompanied by stale nonzero diagnostic energy.
        scaled_diagnostics = dict(diagnostics)
        for name in (*ENERGY_NAMES, *(f"weighted_{name}" for name in ENERGY_NAMES), "total"):
            scaled_diagnostics[name] = diagnostics[name] * scale
        return scale * total, scaled_diagnostics


class ReMoGenE226CondFn(ReMoGenRawSceneCondFn):
    """P360 raw-scene cond_fn augmented by causal rolling pelvis energy.

    All keyword arguments not named below are forwarded unchanged to
    :class:`ReMoGenRawSceneCondFn`.  For M3+ pass ``goal_world_zup=None`` to
    the base class: the rolling task group already owns the active root goal,
    and duplicating P360's legacy goal-PCGrad objective is rejected by default.
    """

    def __init__(
        self,
        *,
        generated_history_world_joints_zup: Tensor,
        history_source: str,
        active_goal_world_zup: Tensor | Sequence[float],
        next_goal_world_zup: Tensor | Sequence[float] | None = None,
        active_yaw_rad: float | None = None,
        e226_config: P360E226BridgeConfig = P360E226BridgeConfig(),
        **base_cond_kwargs: Any,
    ) -> None:
        scene_sdf = base_cond_kwargs.get("scene_sdf")
        super().__init__(**base_cond_kwargs)
        if (
            e226_config.forbid_duplicate_goal_guidance
            and self.goal_world_zup is not None
            and float(e226_config.rolling.goal_weight) > 0.0
        ):
            raise P360E226BridgeError(
                "base goal_world_zup and rolling goal_weight would double-count the "
                "same planner goal; pass base goal_world_zup=None for M3+"
            )
        self.e226_bridge = P360RollingEnergyBridge(
            generated_history_world_joints_zup=generated_history_world_joints_zup,
            history_source=history_source,
            active_goal_world_zup=active_goal_world_zup,
            next_goal_world_zup=next_goal_world_zup,
            active_yaw_rad=active_yaw_rad,
            scene_sdf=scene_sdf,
            config=e226_config,
        )

    @property
    def e226_receipt(self) -> dict[str, Any]:
        return self.e226_bridge.receipt

    def energy_from_world_joints(
        self, joints_world_zup: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compose P360 scene energy and e226 energy inside denoising."""

        base_total, base_diagnostics = super().energy_from_world_joints(
            joints_world_zup
        )
        rolling_total, rolling_diagnostics = self.e226_bridge.energy_from_world_joints(
            joints_world_zup[:, : self.valid_future_frames]
        )
        if not self.e226_bridge.config.enabled:
            rolling_total = rolling_total * 0.0
        total = base_total + rolling_total
        diagnostics = dict(base_diagnostics)
        diagnostics.update(
            {
                "e226_energy": rolling_total,
                "combined_scene_e226_energy": total,
                **{
                    f"e226_{name}": value
                    for name, value in rolling_diagnostics.items()
                },
            }
        )
        return total, diagnostics


__all__ = [
    "ALLOWED_HISTORY_SOURCES",
    "P360E226BridgeConfig",
    "P360E226BridgeError",
    "P360RollingEnergyBridge",
    "ReMoGenE226CondFn",
    "ReMoGenSceneGuidanceError",
    "yaw_to_world_xy",
]

