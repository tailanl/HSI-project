"""Predict ordered event durations without accepting ground-truth timestamps.

Time is measured in future frames: the last history frame is 0 and generated
frames are 1..T. A keypose anchors the END of its declared zero-based event
slot. Optional stages must be removed by the event compiler; every active slot
has an integer minimum of at least one frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _require_integer(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must be an integer tensor")


def _require_prefix(mask: Tensor, name: str) -> None:
    if mask.shape[1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError(f"{name} must contain a valid prefix followed by padding")


def _timing_dtype(value: Tensor) -> torch.dtype:
    # Avoid losing single-frame minima when the surrounding network uses AMP.
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def _boundaries_and_keyposes(
    durations: Tensor, slot_mask: Tensor, keypose_slots: Tensor, total_frames: Tensor
) -> tuple[Tensor, Tensor]:
    ends = durations.cumsum(dim=1)
    # Floating normalization/cumsum can overshoot T by a few ulps. The final
    # boundary is fixed by the supplied horizon, not another predicted value.
    last_slot = slot_mask.sum(dim=1) - 1
    is_last = torch.arange(durations.shape[1], device=durations.device)[None] == last_slot[:, None]
    ends = torch.where(is_last, total_frames[:, None].to(ends.dtype), ends)
    ends = ends.masked_fill(~slot_mask, 0)
    times = ends.gather(1, keypose_slots.clamp_min(0).long())
    return ends, times.masked_fill(keypose_slots < 0, 0)


@dataclass(frozen=True)
class EventSchedule:
    """Continuous training schedule or integer inference schedule.

    ``durations`` and cumulative end ``boundaries`` are [B,M].
    ``keypose_times`` and ``keypose_slots`` are [B,K]. Numeric padding is zero,
    except ``keypose_slots`` uses -1. The two masks are True for valid entries.
    ``total_frames`` is [B]. Additional minima and slot indices retain the
    information needed to discretize without dropping a required event.

    Use :func:`allocate_durations` to construct a validated schedule. Training
    can supervise the continuous result, while the motion branch can consume
    ``schedule.detach()`` to prevent moving times to evade its pose losses.
    """

    durations: Tensor
    boundaries: Tensor
    keypose_times: Tensor
    slot_mask: Tensor
    keypose_mask: Tensor
    total_frames: Tensor
    minimum_frames: Tensor
    keypose_slots: Tensor

    def detach(self) -> "EventSchedule":
        """Keep the schedule values while stopping gradients into timing."""
        return replace(
            self,
            durations=self.durations.detach(),
            boundaries=self.boundaries.detach(),
            keypose_times=self.keypose_times.detach(),
        )

    @torch.no_grad()
    def integer(self) -> "EventSchedule":
        """Discretize once for inference using deterministic largest remainder.

        Integer minima are assigned first. Remaining frames are apportioned
        according to the predicted extras, with earlier slots breaking ties.
        The result has exact row sums T, zero padding and no collapsed slots.
        This operation deliberately supplies no straight-through gradient.
        """
        if not self.durations.is_floating_point():
            return self.detach()
        minima = self.minimum_frames.long()
        remaining = self.total_frames.long() - minima.sum(dim=1)
        extras = (self.durations.double() - minima.double()).clamp_min(0)
        extras = extras.masked_fill(~self.slot_mask, 0)
        extra_sum = extras.sum(dim=1, keepdim=True)
        # In a low-precision schedule a tiny spare budget can round away.
        # Uniform fallback preserves the budget and all declared minima.
        uniform = self.slot_mask.double() / self.slot_mask.sum(1, keepdim=True)
        weights = torch.where(
            extra_sum > 0,
            extras / extra_sum.clamp_min(torch.finfo(torch.float64).tiny),
            uniform,
        )
        apportioned = weights * remaining[:, None].double()
        whole = apportioned.floor().long()
        leftover = remaining - whole.sum(dim=1)
        active_count = self.slot_mask.sum(dim=1)
        if bool(((leftover < 0) | (leftover > active_count)).any()):
            raise ValueError("Frame budget exceeds reliable apportionment precision")
        fractions = (apportioned - whole).masked_fill(~self.slot_mask, -torch.inf)
        order = fractions.argsort(dim=1, descending=True, stable=True)
        bonuses_by_rank = (
            torch.arange(whole.shape[1], device=whole.device)[None] < leftover[:, None]
        ).long()
        bonuses = torch.zeros_like(whole).scatter(1, order, bonuses_by_rank)
        durations = (minima + whole + bonuses).masked_fill(~self.slot_mask, 0)
        if not torch.equal(durations.sum(dim=1), self.total_frames.long()):
            raise ValueError("Discrete event durations do not sum to total_frames")
        if bool((durations[self.slot_mask] < minima[self.slot_mask]).any()):
            raise ValueError("Discretization violated a required event minimum")
        boundaries, times = _boundaries_and_keyposes(
            durations, self.slot_mask, self.keypose_slots, self.total_frames
        )
        return replace(self, durations=durations, boundaries=boundaries, keypose_times=times)

    def frame_mask(self, max_frames: int) -> Tensor:
        """Return [B,F] validity for future frames 1..T; cropping is forbidden."""
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if bool((self.total_frames > max_frames).any()):
            raise ValueError("max_frames must cover every sample's total_frames")
        frames = torch.arange(1, max_frames + 1, device=self.durations.device)
        return frames[None] <= self.total_frames[:, None]

    def phase_features(self, max_frames: int) -> Tensor:
        """Return [B,F,M] per-slot progress, clamped to [0,1].

        Frames before a slot have progress 0 and frames after it have progress
        1. Padded frames and slots are zero. Semantic event types belong to
        upstream slot tokens, not to this timing calculation.
        """
        valid_frames = self.frame_mask(max_frames)
        dtype = _timing_dtype(self.durations)
        durations = self.durations.to(dtype)
        starts = self.boundaries.to(dtype) - durations
        frames = torch.arange(1, max_frames + 1, device=durations.device, dtype=dtype)
        progress = ((frames[None, :, None] - starts[:, None]) / durations[:, None].clamp_min(1)).clamp(0, 1)
        return progress.masked_fill(~(valid_frames[:, :, None] & self.slot_mask[:, None]), 0)

    def attention_bias(self, max_frames: int, width: float = 4.0) -> Tensor:
        """Return [B,F,K] Gaussian temporal bias with width measured in frames.

        All valid keyposes remain visible (finite bias); invalid keys have
        -inf. Padded frames have zero bias and must also use ``frame_mask``.
        """
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
            raise ValueError("width must be a finite positive number of frames")
        valid_frames = self.frame_mask(max_frames)
        dtype = _timing_dtype(self.keypose_times)
        frames = torch.arange(1, max_frames + 1, device=self.durations.device, dtype=dtype)
        distance = (frames[None, :, None] - self.keypose_times.to(dtype)[:, None]) / float(width)
        # A narrow but legal width must not turn a valid far-away goal into a
        # nonfinite mask through floating-point overflow.
        bias = -0.5 * distance.square()
        bias = torch.nan_to_num(bias, neginf=-torch.finfo(dtype).max)
        bias = bias.masked_fill(~self.keypose_mask[:, None], -torch.inf)
        return bias.masked_fill(~valid_frames[:, :, None], 0)


def allocate_durations(
    logits: Tensor,
    minimum_frames: Tensor,
    slot_mask: Tensor,
    total_frames: Tensor,
    keypose_slots: Tensor,
) -> EventSchedule:
    """Allocate a differentiable, ordered duration schedule with fixed minima.

    Inputs are logits/minima/mask [B,M], total future lengths [B], and ordered
    anchor slot indices [B,K]. Valid slots/keyposes are prefixes. Each active
    minimum is >=1, padded minima are 0, and padded keypose indices are -1.
    No absolute/relative ground-truth event time is accepted by this API.
    """
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim != 2 or min(logits.shape) < 1:
        raise ValueError("logits must have nonempty shape [B,M]")
    _require_integer(minimum_frames, "minimum_frames")
    _require_integer(total_frames, "total_frames")
    _require_integer(keypose_slots, "keypose_slots")
    if not isinstance(slot_mask, Tensor) or slot_mask.dtype != torch.bool:
        raise TypeError("slot_mask must be a boolean tensor")
    batch, slots = logits.shape
    if minimum_frames.shape != logits.shape or slot_mask.shape != logits.shape:
        raise ValueError("minimum_frames and slot_mask must match logits [B,M]")
    if total_frames.shape != (batch,):
        raise ValueError("total_frames must have shape [B]")
    if keypose_slots.ndim != 2 or keypose_slots.shape[0] != batch:
        raise ValueError("keypose_slots must have shape [B,K]")
    if any(value.device != logits.device for value in (minimum_frames, slot_mask, total_frames, keypose_slots)):
        raise ValueError("All timing tensors must be on the same device")
    if not bool(slot_mask.any(dim=1).all()):
        raise ValueError("Every sample needs at least one valid event slot")
    _require_prefix(slot_mask, "slot_mask")
    if bool((minimum_frames[slot_mask] < 1).any()):
        raise ValueError("Every valid event slot requires minimum_frames >= 1")
    if bool((minimum_frames[~slot_mask] != 0).any()):
        raise ValueError("Padded event slots require zero minimum_frames")
    if bool((total_frames < 1).any()):
        raise ValueError("total_frames must be positive")
    if not bool(torch.isfinite(logits[slot_mask]).all()):
        raise ValueError("Valid event logits must be finite")
    minimum_frames = minimum_frames.long()
    total_frames = total_frames.long()
    remaining = total_frames - minimum_frames.sum(dim=1)
    if bool((remaining < 0).any()):
        raise ValueError("Infeasible length: minimum_frames sum exceeds total_frames")
    if bool((keypose_slots < -1).any()):
        raise ValueError("Padded keypose_slots must be -1")
    keypose_slots = keypose_slots.long()
    keypose_mask = keypose_slots >= 0
    _require_prefix(keypose_mask, "keypose_slots")
    if bool((keypose_slots[keypose_mask] >= slots).any()):
        raise ValueError("A keypose slot is outside the event slot array")
    if bool((keypose_mask & ~slot_mask.gather(1, keypose_slots.clamp_min(0))).any()):
        raise ValueError("A keypose cannot anchor a padded event slot")
    if keypose_slots.shape[1] > 1 and bool(
        (keypose_mask[:, 1:] & (keypose_slots[:, 1:] <= keypose_slots[:, :-1])).any()
    ):
        raise ValueError("Valid keypose_slots must be strictly increasing")
    safe_logits = logits.to(_timing_dtype(logits)).masked_fill(~slot_mask, 0)
    positive = (F.softplus(safe_logits) + 1.0e-6).masked_fill(~slot_mask, 0)
    scaled = positive / positive.amax(dim=1, keepdim=True)
    weights = scaled / scaled.sum(dim=1, keepdim=True)
    durations = minimum_frames.to(weights.dtype) + remaining[:, None].to(weights.dtype) * weights
    durations = durations.masked_fill(~slot_mask, 0)
    boundaries, times = _boundaries_and_keyposes(durations, slot_mask, keypose_slots, total_frames)
    return EventSchedule(
        durations=durations,
        boundaries=boundaries,
        keypose_times=times,
        slot_mask=slot_mask,
        keypose_mask=keypose_mask,
        total_frames=total_frames,
        minimum_frames=minimum_frames,
        keypose_slots=keypose_slots,
    )


class DurationHead(nn.Module):
    """Trainable readout of fused history/pose/route/scene event context.

    The caller supplies [B,M,D] context tokens, including event semantics and
    permissions. Total length is embedded as log1p(T); true event timestamps
    are never an input. Invalid slot logits are masked by allocate_durations.
    """

    def __init__(self, context_dim: int) -> None:
        super().__init__()
        if isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim < 1:
            raise ValueError("context_dim must be a positive integer")
        self.context_dim = context_dim
        self.context_norm = nn.LayerNorm(context_dim)
        self.length_embedding = nn.Sequential(
            nn.Linear(1, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )
        self.readout = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.SiLU(), nn.Linear(context_dim, 1)
        )

    def forward(self, slot_context: Tensor, total_frames: Tensor) -> Tensor:
        if not isinstance(slot_context, Tensor) or not slot_context.is_floating_point():
            raise TypeError("slot_context must be floating point")
        if slot_context.ndim != 3 or slot_context.shape[-1] != self.context_dim or min(slot_context.shape[:2]) < 1:
            raise ValueError("slot_context must have nonempty shape [B,M,context_dim]")
        _require_integer(total_frames, "total_frames")
        if total_frames.shape != (slot_context.shape[0],) or total_frames.device != slot_context.device:
            raise ValueError("total_frames must match the context batch and device")
        if bool((total_frames < 1).any()) or not bool(torch.isfinite(slot_context).all()):
            raise ValueError("Duration context must be finite and total_frames positive")
        log_length = total_frames.to(_timing_dtype(slot_context)).log1p().to(slot_context.dtype)[:, None]
        length = self.length_embedding(log_length)[:, None]
        return self.readout(self.context_norm(slot_context) + length).squeeze(-1)


__all__ = ["DurationHead", "EventSchedule", "allocate_durations"]
