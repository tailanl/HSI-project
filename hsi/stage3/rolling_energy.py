"""Differentiable rolling root-trajectory energy for the P478/P360 bridge.

The module is deliberately an *in-denoising* objective.  It never changes a
decoded trajectory after generation.  The caller supplies an immutable,
already-generated pelvis history and the differentiable pelvis trajectory of
the current motion primitive.  All coordinates are metric world coordinates
in a Z-up scene, hence the navigation plane is ``(world X, world Y)``.

The historical E2.2.6 experiments motivated the individual path-shape terms,
while :mod:`kimodo_sceneco.guidance.anti_zigzag_energy_v9` provides the local
anti-zigzag formulas.  This implementation differs in two important ways:

* the current primitive is optimized inside the diffusion guidance hook; no
  post-hoc motion edit is exposed;
* self-loop evidence spans detached history and the current primitive, so a
  short rolling primitive can detect a long-range return.

The public API is intentionally host-agnostic.  ``p360_world_zup_pelvis_energy``
is only a thin coordinate adapter and does not modify the frozen P360 code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, NamedTuple, Sequence

import torch
from torch import Tensor


ENERGY_NAMES: tuple[str, ...] = (
    "goal",
    "heading",
    "smooth",
    "length_ratio",
    "backtrack",
    "zigzag",
    "self_loop",
    "sdf",
)

DEFAULT_ENERGY_GROUPS: Mapping[str, tuple[str, ...]] = {
    "task": ("goal", "heading"),
    "path_shape": (
        "smooth",
        "length_ratio",
        "backtrack",
        "zigzag",
        "self_loop",
    ),
    "scene": ("sdf",),
}


class RollingEnergyError(ValueError):
    """Raised when a rolling-energy input violates the coordinate contract."""


@dataclass(frozen=True)
class RollingEnergyConfig:
    """Weights and metric thresholds for one rolling primitive.

    The defaults are conservative guidance defaults rather than a claim of a
    calibrated P360 setting.  All distances are in metres.
    """

    goal_weight: float = 1.0
    heading_weight: float = 0.25
    smooth_weight: float = 4.0
    length_ratio_weight: float = 2.0
    backtrack_weight: float = 4.0
    zigzag_weight: float = 1.0
    self_loop_weight: float = 1.0
    sdf_weight: float = 30.0

    max_length_ratio: float = 1.35
    minimum_direct_distance_m: float = 0.05
    backtrack_tolerance_m: float = 0.0
    zigzag_turn_threshold_deg: float = 55.0
    minimum_step_m: float = 1.0e-5
    heading_tail_segments: int = 3
    self_loop_temporal_exclusion_frames: int = 4
    self_loop_sigma_m: float = 0.15
    sdf_clearance_m: float = 0.10

    def validate(self) -> "RollingEnergyConfig":
        weights = (
            "goal_weight",
            "heading_weight",
            "smooth_weight",
            "length_ratio_weight",
            "backtrack_weight",
            "zigzag_weight",
            "self_loop_weight",
            "sdf_weight",
        )
        nonnegative = weights + (
            "max_length_ratio",
            "minimum_direct_distance_m",
            "backtrack_tolerance_m",
            "minimum_step_m",
            "self_loop_sigma_m",
            "sdf_clearance_m",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise RollingEnergyError(f"{name} must be finite and non-negative")
        if not 0.0 < float(self.zigzag_turn_threshold_deg) < 180.0:
            raise RollingEnergyError(
                "zigzag_turn_threshold_deg must lie strictly between 0 and 180"
            )
        if int(self.heading_tail_segments) < 1:
            raise RollingEnergyError("heading_tail_segments must be positive")
        if int(self.self_loop_temporal_exclusion_frames) < 0:
            raise RollingEnergyError(
                "self_loop_temporal_exclusion_frames must be non-negative"
            )
        if float(self.minimum_direct_distance_m) <= 0.0:
            raise RollingEnergyError("minimum_direct_distance_m must be positive")
        if float(self.minimum_step_m) <= 0.0:
            raise RollingEnergyError("minimum_step_m must be positive")
        if float(self.self_loop_sigma_m) <= 0.0:
            raise RollingEnergyError("self_loop_sigma_m must be positive")
        return self

    def weights(self) -> dict[str, float]:
        """Return weights keyed by the public component names."""

        return {
            "goal": float(self.goal_weight),
            "heading": float(self.heading_weight),
            "smooth": float(self.smooth_weight),
            "length_ratio": float(self.length_ratio_weight),
            "backtrack": float(self.backtrack_weight),
            "zigzag": float(self.zigzag_weight),
            "self_loop": float(self.self_loop_weight),
            "sdf": float(self.sdf_weight),
        }


class PCGradResult(NamedTuple):
    """Audit-friendly result of asymmetric, task-preserving PCGrad."""

    raw_gradients: dict[str, Tensor]
    projected_gradients: dict[str, Tensor]
    conflict_dots: dict[str, Tensor]
    combined_gradient: Tensor


def _require_xy(value: Tensor, *, name: str, allow_empty: bool) -> Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise RollingEnergyError(f"{name} must be a floating-point tensor")
    if value.ndim != 2 or value.shape[-1] != 2:
        raise RollingEnergyError(f"{name} must have shape [frames,2]")
    if not allow_empty and value.shape[0] < 1:
        raise RollingEnergyError(f"{name} must contain at least one frame")
    if not bool(torch.isfinite(value).all()):
        raise RollingEnergyError(f"{name} must contain only finite values")
    return value


def _condition_xy(value: Tensor | Sequence[float], *, like: Tensor, name: str) -> Tensor:
    result = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if result.shape != (2,) or not bool(torch.isfinite(result).all()):
        raise RollingEnergyError(f"{name} must be a finite world-XY pair")
    # A condition is not an optimization variable of this energy function.
    return result.detach()


def _zero(current_xy: Tensor) -> Tensor:
    """A scalar zero that preserves an autograd edge to the current primitive."""

    return current_xy.sum() * 0.0


def _segments_entering_current(history_xy: Tensor, current_xy: Tensor) -> tuple[Tensor, Tensor]:
    """Return path points and segments whose endpoint belongs to ``current``."""

    if history_xy.shape[0] > 0:
        points = torch.cat((history_xy[-1:], current_xy), dim=0)
    else:
        points = current_xy
    if points.shape[0] < 2:
        return points, current_xy.new_zeros((0, 2))
    return points, points[1:] - points[:-1]


def _arrival_heading(
    *,
    anchor_xy: Tensor,
    active_goal_xy: Tensor,
    next_goal_xy: Tensor | None,
    yaw_heading_xy: Tensor | None,
    epsilon: float,
) -> Tensor | None:
    """Resolve a fixed arrival-tangent proxy in descending priority order."""

    if yaw_heading_xy is not None:
        vector = yaw_heading_xy
    elif next_goal_xy is not None:
        vector = next_goal_xy - active_goal_xy
    else:
        vector = active_goal_xy - anchor_xy
    norm = torch.linalg.vector_norm(vector)
    if float(norm.detach().cpu()) <= epsilon:
        return None
    return vector / norm.clamp_min(epsilon)


def _heading_energy(
    segments: Tensor,
    desired_heading: Tensor | None,
    *,
    tail_segments: int,
    minimum_step_m: float,
    zero: Tensor,
) -> Tensor:
    if desired_heading is None or segments.shape[0] == 0:
        return zero
    tail = segments[-int(tail_segments) :]
    speed = torch.linalg.vector_norm(tail, dim=-1)
    valid = speed > float(minimum_step_m)
    if not bool(valid.any()):
        return zero
    unit = tail / speed.clamp_min(float(minimum_step_m)).unsqueeze(-1)
    cosine = (unit * desired_heading).sum(dim=-1).clamp(-1.0, 1.0)
    # 0 for aligned, 1 for orthogonal, 2 for reversed.  Squaring gives a
    # stronger response to a backwards terminal tangent.
    return ((1.0 - cosine[valid]) ** 2).mean()


def _smooth_energy(history_xy: Tensor, current_xy: Tensor, *, zero: Tensor) -> Tensor:
    """Second-difference energy for stencils touching the current primitive."""

    # Two historical points are sufficient to evaluate both boundary
    # accelerations.  Older history cannot receive a gradient and should not
    # contribute a constant loss.
    context = torch.cat((history_xy[-2:], current_xy), dim=0)
    history_count = min(int(history_xy.shape[0]), 2)
    if context.shape[0] < 3:
        return zero
    acceleration = context[2:] - 2.0 * context[1:-1] + context[:-2]
    newest_index = torch.arange(2, context.shape[0], device=context.device)
    touches_current = newest_index >= history_count
    if not bool(touches_current.any()):
        return zero
    return acceleration[touches_current].square().sum(dim=-1).mean()


def _zigzag_energy(
    history_xy: Tensor,
    current_xy: Tensor,
    *,
    threshold_deg: float,
    minimum_step_m: float,
    zero: Tensor,
) -> Tensor:
    # One previous velocity is enough to catch a turn exactly at a primitive
    # boundary.  Every evaluated turn therefore contains a current segment.
    context = torch.cat((history_xy[-2:], current_xy), dim=0)
    history_count = min(int(history_xy.shape[0]), 2)
    if context.shape[0] < 3:
        return zero
    velocity = context[1:] - context[:-1]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    unit = velocity / speed.clamp_min(float(minimum_step_m)).unsqueeze(-1)
    cosine = (unit[1:] * unit[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
    later_endpoint = torch.arange(2, context.shape[0], device=context.device)
    valid = (
        (later_endpoint >= history_count)
        & (speed[1:] > float(minimum_step_m))
        & (speed[:-1] > float(minimum_step_m))
    )
    if not bool(valid.any()):
        return zero
    cosine_limit = math.cos(math.radians(float(threshold_deg)))
    return torch.relu(cosine_limit - cosine[valid]).square().mean()


def _self_loop_energy(
    history_xy: Tensor,
    current_xy: Tensor,
    *,
    temporal_exclusion: int,
    sigma_m: float,
    zero: Tensor,
) -> Tensor:
    """Penalize a current point returning near any sufficiently old point.

    Only pairs whose *newer* member is in the current primitive are included.
    This both prevents a constant historical penalty and guarantees that
    history-to-current loops are visible to the current gradient.
    """

    history_count = int(history_xy.shape[0])
    combined = torch.cat((history_xy, current_xy), dim=0)
    scores: list[Tensor] = []
    denominator = 2.0 * float(sigma_m) * float(sigma_m)
    for local_index in range(int(current_xy.shape[0])):
        global_index = history_count + local_index
        oldest_allowed_end = global_index - int(temporal_exclusion)
        if oldest_allowed_end <= 0:
            continue
        earlier = combined[:oldest_allowed_end]
        distance2 = (current_xy[local_index] - earlier).square().sum(dim=-1)
        # A max-like detector avoids washing out one real return in a long
        # history.  It is the differentiable analogue of a nearest-old-point
        # loop test (apart from measure-zero ties).
        scores.append(torch.exp(-distance2 / denominator).amax())
    if not scores:
        return zero
    return torch.stack(scores).mean()


def _extract_sdf_tensor(value: object, *, like: Tensor) -> Tensor:
    if torch.is_tensor(value):
        result = value
    elif isinstance(value, Mapping) and "signed_distance_m" in value:
        result = value["signed_distance_m"]
    elif hasattr(value, "signed_distance_m"):
        result = getattr(value, "signed_distance_m")
    else:
        raise RollingEnergyError(
            "sdf_callback must return a Tensor or an object/mapping with "
            "signed_distance_m"
        )
    if not torch.is_tensor(result) or not result.is_floating_point():
        raise RollingEnergyError("SDF signed distances must be a floating tensor")
    result = result.to(device=like.device, dtype=like.dtype)
    if result.numel() != like.shape[0]:
        raise RollingEnergyError(
            "sdf_callback must return exactly one signed distance per current frame"
        )
    result = result.reshape(like.shape[0])
    if not bool(torch.isfinite(result).all()):
        raise RollingEnergyError("SDF signed distances must be finite")
    return result


def e226_rolling_energy(
    history_xy: Tensor,
    current_xy: Tensor,
    *,
    active_goal_xy: Tensor | Sequence[float],
    next_goal_xy: Tensor | Sequence[float] | None = None,
    yaw_heading_xy: Tensor | Sequence[float] | None = None,
    sdf_callback: Callable[[Tensor], object] | None = None,
    sdf_clearance_m: float | None = None,
    config: RollingEnergyConfig = RollingEnergyConfig(),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute a decomposed rolling root energy.

    Parameters
    ----------
    history_xy:
        Already generated metric world-XY pelvis positions, shape ``[H,2]``.
        They are unconditionally detached inside this function.
    current_xy:
        Differentiable metric world-XY pelvis positions decoded from the
        current primitive, shape ``[T,2]``.
    active_goal_xy / next_goal_xy:
        Ordered planner nodes.  The endpoint targets the active node.  The
        next node supplies an arrival tangent when no explicit yaw vector is
        available.
    yaw_heading_xy:
        Optional world-XY unit direction derived from a Z-up yaw.  Because the
        input contains positions rather than predicted body yaw, it supervises
        the terminal trajectory tangent only; it is not a body-facing loss.
    sdf_callback:
        Optional differentiable callback queried on ``current_xy``.  It must
        return one metric signed distance per frame (positive in free space).

    Returns
    -------
    total, components:
        Scalar weighted total and a mapping containing all eight raw energies,
        their weighted values, useful scalar diagnostics, and ``total``.
    """

    config = config.validate()
    history_xy = _require_xy(history_xy, name="history_xy", allow_empty=True)
    current_xy = _require_xy(current_xy, name="current_xy", allow_empty=False)
    if history_xy.device != current_xy.device:
        raise RollingEnergyError("history_xy and current_xy must share a device")
    # This is a hard causal boundary, even if a caller accidentally passes a
    # requires_grad history tensor.
    history = history_xy.detach().to(dtype=current_xy.dtype)
    active_goal = _condition_xy(
        active_goal_xy, like=current_xy, name="active_goal_xy"
    )
    next_goal = (
        None
        if next_goal_xy is None
        else _condition_xy(next_goal_xy, like=current_xy, name="next_goal_xy")
    )
    yaw_heading = (
        None
        if yaw_heading_xy is None
        else _condition_xy(yaw_heading_xy, like=current_xy, name="yaw_heading_xy")
    )
    if yaw_heading is not None:
        norm = torch.linalg.vector_norm(yaw_heading)
        if float(norm.detach().cpu()) <= float(config.minimum_step_m):
            raise RollingEnergyError("yaw_heading_xy must be non-zero")
        yaw_heading = yaw_heading / norm

    zero = _zero(current_xy)
    anchor = history[-1] if history.shape[0] else current_xy[0]
    _, segments = _segments_entering_current(history, current_xy)

    goal = (current_xy[-1] - active_goal).square().sum()
    desired_heading = _arrival_heading(
        anchor_xy=anchor,
        active_goal_xy=active_goal,
        next_goal_xy=next_goal,
        yaw_heading_xy=yaw_heading,
        epsilon=float(config.minimum_step_m),
    )
    heading = _heading_energy(
        segments,
        desired_heading,
        tail_segments=int(config.heading_tail_segments),
        minimum_step_m=float(config.minimum_step_m),
        zero=zero,
    )
    smooth = _smooth_energy(history, current_xy, zero=zero)

    if segments.shape[0] == 0:
        path_length = zero
        direct_distance = zero
        ratio_value = zero + 1.0
        length_ratio = zero
        backtrack = zero
    else:
        path_length = torch.linalg.vector_norm(segments, dim=-1).sum()
        direct_distance = torch.linalg.vector_norm(current_xy[-1] - anchor)
        ratio_value = path_length / direct_distance.clamp_min(
            float(config.minimum_direct_distance_m)
        )
        length_ratio = torch.relu(
            ratio_value - float(config.max_length_ratio)
        ).square()
        path_points = torch.cat((anchor.reshape(1, 2), current_xy), dim=0)
        distance_to_goal = torch.linalg.vector_norm(
            path_points - active_goal.reshape(1, 2), dim=-1
        )
        distance_increase = distance_to_goal[1:] - distance_to_goal[:-1]
        backtrack = torch.relu(
            distance_increase - float(config.backtrack_tolerance_m)
        ).square().mean()

    zigzag = _zigzag_energy(
        history,
        current_xy,
        threshold_deg=float(config.zigzag_turn_threshold_deg),
        minimum_step_m=float(config.minimum_step_m),
        zero=zero,
    )
    self_loop = _self_loop_energy(
        history,
        current_xy,
        temporal_exclusion=int(config.self_loop_temporal_exclusion_frames),
        sigma_m=float(config.self_loop_sigma_m),
        zero=zero,
    )

    clearance = (
        float(config.sdf_clearance_m)
        if sdf_clearance_m is None
        else float(sdf_clearance_m)
    )
    if not math.isfinite(clearance) or clearance < 0.0:
        raise RollingEnergyError("sdf_clearance_m must be finite and non-negative")
    if sdf_callback is None:
        sdf = zero
        sdf_min = zero
        sdf_violation_fraction = zero
    else:
        signed_distance = _extract_sdf_tensor(
            sdf_callback(current_xy), like=current_xy
        )
        violation = torch.relu(clearance - signed_distance)
        sdf = violation.square().mean()
        sdf_min = signed_distance.amin()
        sdf_violation_fraction = (signed_distance < clearance).to(current_xy.dtype).mean()

    raw: dict[str, Tensor] = {
        "goal": goal,
        "heading": heading,
        "smooth": smooth,
        "length_ratio": length_ratio,
        "backtrack": backtrack,
        "zigzag": zigzag,
        "self_loop": self_loop,
        "sdf": sdf,
    }
    weights = config.weights()
    weighted = {name: raw[name] * weights[name] for name in ENERGY_NAMES}
    total = torch.stack(tuple(weighted[name] for name in ENERGY_NAMES)).sum()
    components = {
        **raw,
        **{f"weighted_{name}": value for name, value in weighted.items()},
        "path_length_m": path_length,
        "direct_distance_m": direct_distance,
        "length_ratio_value": ratio_value,
        "sdf_min_m": sdf_min,
        "sdf_violation_fraction": sdf_violation_fraction,
        "total": total,
    }
    return total, components


def group_energy_components(
    components: Mapping[str, Tensor],
    *,
    config: RollingEnergyConfig = RollingEnergyConfig(),
    groups: Mapping[str, Sequence[str]] = DEFAULT_ENERGY_GROUPS,
) -> dict[str, Tensor]:
    """Return weighted scalar group energies for PCGrad or block clipping.

    Groups must form a non-overlapping partition of the eight public energies;
    this prevents accidental double weighting in a guidance hook.
    """

    config = config.validate()
    seen: list[str] = []
    for group_name, names in groups.items():
        if not str(group_name):
            raise RollingEnergyError("energy group names must be non-empty")
        seen.extend(str(name) for name in names)
    if len(seen) != len(set(seen)) or set(seen) != set(ENERGY_NAMES):
        raise RollingEnergyError(
            "groups must contain every public energy exactly once"
        )
    weights = config.weights()
    result: dict[str, Tensor] = {}
    for group_name, names in groups.items():
        terms: list[Tensor] = []
        for name in names:
            value = components.get(name)
            if not torch.is_tensor(value) or value.ndim != 0:
                raise RollingEnergyError(
                    f"component {name!r} must be present as a scalar Tensor"
                )
            terms.append(value * weights[name])
        if not terms:
            raise RollingEnergyError(f"energy group {group_name!r} may not be empty")
        result[str(group_name)] = torch.stack(terms).sum()
    return result


def project_gradient_against_reference(
    gradient: Tensor,
    reference: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor]:
    """Asymmetrically remove only a component conflicting with ``reference``.

    This is the goal-preserving PCGrad variant used by P360's existing scene
    hook: the reference/task gradient is never changed.  The returned scalar
    is the pre-projection dot product for audit logging.
    """

    if gradient.shape != reference.shape:
        raise RollingEnergyError("PCGrad tensors must have identical shapes")
    if not gradient.is_floating_point() or not reference.is_floating_point():
        raise RollingEnergyError("PCGrad tensors must be floating point")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise RollingEnergyError("PCGrad epsilon must be finite and positive")
    dot = (gradient * reference).sum()
    if float(dot.detach().cpu()) < 0.0:
        norm2 = reference.square().sum().clamp_min(float(epsilon))
        gradient = gradient - dot / norm2 * reference
    return gradient, dot


def grouped_pcgrad(
    group_energies: Mapping[str, Tensor],
    current_xy: Tensor,
    *,
    primary_group: str = "task",
    epsilon: float = 1.0e-12,
    create_graph: bool = False,
) -> PCGradResult:
    """Differentiate groups and project conflicts against a primary group.

    The helper returns a gradient only.  Applying its negative inside a DDPM
    condition function is the caller's responsibility; no trajectory or
    decoded motion is mutated here.
    """

    _require_xy(current_xy, name="current_xy", allow_empty=False)
    if primary_group not in group_energies:
        raise RollingEnergyError(f"unknown primary PCGrad group {primary_group!r}")
    names = tuple(str(name) for name in group_energies)
    raw: dict[str, Tensor] = {}
    for index, name in enumerate(names):
        energy = group_energies[name]
        if not torch.is_tensor(energy) or energy.ndim != 0:
            raise RollingEnergyError("every PCGrad group energy must be scalar")
        if energy.requires_grad:
            gradient = torch.autograd.grad(
                energy,
                current_xy,
                retain_graph=index < len(names) - 1 or bool(create_graph),
                create_graph=bool(create_graph),
                allow_unused=True,
            )[0]
        else:
            gradient = None
        raw[name] = torch.zeros_like(current_xy) if gradient is None else gradient

    reference = raw[primary_group]
    projected: dict[str, Tensor] = {primary_group: reference}
    conflict_dots: dict[str, Tensor] = {}
    for name in names:
        if name == primary_group:
            continue
        projected[name], conflict_dots[name] = project_gradient_against_reference(
            raw[name], reference, epsilon=epsilon
        )
    combined = torch.stack(tuple(projected[name] for name in names), dim=0).sum(dim=0)
    return PCGradResult(raw, projected, conflict_dots, combined)


def p360_world_zup_pelvis_energy(
    history_world_joints_zup: Tensor,
    current_world_joints_zup: Tensor,
    *,
    active_goal_world_zup: Tensor | Sequence[float],
    next_goal_world_zup: Tensor | Sequence[float] | None = None,
    yaw_heading_xy: Tensor | Sequence[float] | None = None,
    pelvis_joint_index: int = 0,
    sdf_callback: Callable[[Tensor], object] | None = None,
    sdf_clearance_m: float | None = None,
    config: RollingEnergyConfig = RollingEnergyConfig(),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Thin P360 adapter: extract pelvis **XY** from world-Z-up J3 tensors.

    Accepted joint layouts are ``[H,J,3]`` / ``[T,J,3]`` and their batch-one
    variants ``[1,H,J,3]`` / ``[1,T,J,3]``.  Batch sizes above one must be
    split by the host so each rollout retains its own detached history.
    """

    def unbatch(value: Tensor, *, name: str) -> Tensor:
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise RollingEnergyError(f"{name} must be a floating-point tensor")
        if value.ndim == 4:
            if value.shape[0] != 1:
                raise RollingEnergyError(f"{name} batch dimension must be one")
            value = value[0]
        if value.ndim != 3 or value.shape[-1] != 3:
            raise RollingEnergyError(f"{name} must have shape [frames,joints,3]")
        if not bool(torch.isfinite(value).all()):
            raise RollingEnergyError(f"{name} must contain only finite values")
        return value

    history = unbatch(history_world_joints_zup, name="history_world_joints_zup")
    current = unbatch(current_world_joints_zup, name="current_world_joints_zup")
    joint = int(pelvis_joint_index)
    if joint < 0 or joint >= history.shape[1] or joint >= current.shape[1]:
        raise RollingEnergyError("pelvis_joint_index lies outside a joint tensor")

    def world_xy(value: Tensor | Sequence[float] | None, *, name: str):
        if value is None:
            return None
        result = torch.as_tensor(value, device=current.device, dtype=current.dtype)
        if result.shape != (3,) or not bool(torch.isfinite(result).all()):
            raise RollingEnergyError(f"{name} must be a finite world XYZ triplet")
        # Correct Z-up ground plane: X/Y, never X/Z.
        return result[:2]

    return e226_rolling_energy(
        history[:, joint, :2],
        current[:, joint, :2],
        active_goal_xy=world_xy(
            active_goal_world_zup, name="active_goal_world_zup"
        ),
        next_goal_xy=world_xy(next_goal_world_zup, name="next_goal_world_zup"),
        yaw_heading_xy=yaw_heading_xy,
        sdf_callback=sdf_callback,
        sdf_clearance_m=sdf_clearance_m,
        config=config,
    )


__all__ = [
    "DEFAULT_ENERGY_GROUPS",
    "ENERGY_NAMES",
    "PCGradResult",
    "RollingEnergyConfig",
    "RollingEnergyError",
    "e226_rolling_energy",
    "group_energy_components",
    "grouped_pcgrad",
    "p360_world_zup_pelvis_energy",
    "project_gradient_against_reference",
]

