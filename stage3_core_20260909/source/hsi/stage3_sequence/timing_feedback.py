"""Bounded scene/contact-conditioned timing updates from denoising evidence.

This is a learned residual readout, not a new noisy diffusion variable. Static
context is cached once. Evidence is measured on legal RAW predictions before
hard keypose projection, and never includes supervised event timestamps.
Every update is relative to the initial schedule, so repeated calls cannot
accumulate unrestricted timing drift. Learned contacts are observations only;
they cannot erase the declared contact or release requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import JOINTS, expand_contacts, future_mask
from .constraints import legalize_rotation_6d
from .timing import EventSchedule


def legal_motion_for_feedback(raw: Tensor, condition) -> Tensor:
    """Legalize rotations and padding, without placing any target keyposes."""
    rotations = raw[..., 3:].reshape(*raw.shape[:2], JOINTS, 6)
    legal = torch.cat((raw[..., :3], legalize_rotation_6d(rotations).flatten(-2)), -1)
    return legal.masked_fill(~future_mask(condition, raw.shape[1])[..., None], 0)


def _motion_reference_indices(condition, slots: Tensor) -> Tensor:
    """Choose comparison poses separately from contact ownership.

    An explicit event-end keypose anchor takes priority. Without one, retain
    the relation owner's pose as a comparison reference, not a new hard goal
    or a full-body hold requirement. No new upstream intent is compiled here.
    A release can thus retain a seated *contact relation* while its explicitly
    anchored end pose is standing; an unanchored hold retains its reference.
    """
    c = condition
    valid = c.keypose_mask[:, None]
    exact = (c.keypose_slots[:, None] == slots[..., None]) & valid
    rows = torch.arange(c.batch_size, device=slots.device)[:, None]
    reference = c.slot_keyposes[rows, slots].clamp_min(0)
    return torch.where(exact.any(-1), exact.long().argmax(-1), reference)


def _with_durations(base: EventSchedule, durations: Tensor) -> EventSchedule:
    ends = durations.cumsum(1)
    last = base.slot_mask.sum(1) - 1
    is_last = torch.arange(durations.shape[1], device=durations.device)[None] == last[:, None]
    ends = torch.where(is_last, base.total_frames[:, None].to(ends), ends)
    ends = ends.masked_fill(~base.slot_mask, 0)
    times = ends.gather(1, base.keypose_slots.clamp_min(0)).masked_fill(~base.keypose_mask, 0)
    return replace(base, durations=durations, boundaries=ends, keypose_times=times)


@dataclass(frozen=True)
class TimingFeedbackResult:
    schedule: EventSchedule
    residual_logits: Tensor
    max_boundary_shift: Tensor
    adjustment_scale: Tensor
    integer_boundary_budget: Tensor

    def effective_hysteresis_frames(self, requested: float = 1.5) -> Tensor:
        """Keep sub-frame freedom when the entire allowed shift is one frame."""
        if isinstance(requested, bool) or not isinstance(requested, (int, float)) or not math.isfinite(requested) or requested < 0:
            raise ValueError("timing hysteresis must be finite and nonnegative")
        maximum = (self.integer_boundary_budget.to(self.schedule.durations) - .5).clamp_min(.5)
        return maximum.clamp_max(float(requested))

    def _project_integer_boundaries(self, initial: EventSchedule, desired: Tensor) -> EventSchedule:
        """Forward feasibility projection, identical to the original rounding cap."""
        minimum = initial.minimum_frames.long()
        future_minimum = minimum.sum(1, keepdim=True) - minimum.cumsum(1)
        previous = torch.zeros_like(initial.total_frames)
        durations = []
        for slot in range(minimum.shape[1]):
            valid = initial.slot_mask[:, slot]
            lower = torch.maximum(previous + minimum[:, slot],
                                  initial.boundaries[:, slot] - self.integer_boundary_budget)
            upper = torch.minimum(initial.total_frames - future_minimum[:, slot],
                                  initial.boundaries[:, slot] + self.integer_boundary_budget)
            final = initial.slot_mask.sum(1) - 1 == slot
            end = torch.maximum(lower, torch.minimum(upper, desired[:, slot]))
            end = torch.where(final, initial.total_frames, end)
            durations.append(torch.where(valid, end - previous, 0))
            previous = torch.where(valid, end, previous)
        return _with_durations(initial, torch.stack(durations, 1))

    def _checked_current(self, current: EventSchedule, initial: EventSchedule) -> EventSchedule:
        """Validate an externally supplied integer schedule, not just its ends.

        EventSchedule.integer() deliberately fast-paths existing integer
        schedules; it is not a validator for dataclasses.replace corruption.
        The hysteresis interface must not let malformed current state through.
        """
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if not isinstance(current, EventSchedule):
            raise ValueError("current timing schedule must be an EventSchedule")
        masks = ("slot_mask", "keypose_mask")
        integer_fields = ("durations", "boundaries", "keypose_times", "total_frames", "minimum_frames", "keypose_slots")
        for name in masks + integer_fields:
            value, reference = getattr(current, name), getattr(initial, name)
            if not isinstance(value, Tensor) or value.shape != reference.shape or value.device != reference.device:
                raise ValueError("current timing schedule tensor shape/device differs from initial schedule")
            if (name in masks and value.dtype != torch.bool) or (name in integer_fields and value.dtype not in integer_types):
                raise ValueError("current timing schedule requires boolean masks and integer time tensors")
        for name in ("slot_mask", "keypose_mask", "total_frames", "minimum_frames", "keypose_slots"):
            if not torch.equal(getattr(current, name), getattr(initial, name)):
                raise ValueError("current timing schedule metadata differs from initial schedule")
        durations = current.durations.long()
        if bool((durations[~current.slot_mask] != 0).any()) or bool((durations[current.slot_mask] < current.minimum_frames[current.slot_mask]).any()):
            raise ValueError("current timing schedule violates event minima or zero padding")
        if not torch.equal(durations.sum(1), current.total_frames.long()):
            raise ValueError("current timing schedule durations do not sum to the requested total")
        expected = _with_durations(initial, durations)
        if not torch.equal(current.boundaries.long(), expected.boundaries) or not torch.equal(current.keypose_times.long(), expected.keypose_times):
            raise ValueError("current timing schedule boundaries/keypose times disagree with durations")
        if bool(((current.boundaries-initial.boundaries).abs() > self.integer_boundary_budget[:, None]).any()):
            raise ValueError("current timing schedule exceeds the initial boundary cap")
        return expected

    @torch.no_grad()
    def integer(self, initial_schedule: EventSchedule, *, current_schedule: EventSchedule | None = None,
                hysteresis_frames: float = 1.5) -> EventSchedule:
        """Round with minima, initial absolute caps and optional hysteresis.

        No current schedule, or hysteresis=0, reproduces the original integer
        allocation exactly. Otherwise retain a current boundary while the raw
        continuous proposal is inside its deadband, then re-project the mixed
        boundaries to preserve event minima, order and the exact total. The
        deadband is capped below the available integer shift budget, so a
        short sequence with a one-frame budget is not completely frozen.
        A supplied current schedule must be a validated integer schedule. This
        changes no learned head output, contact requirement or time loss.
        """
        effective = self.effective_hysteresis_frames(hysteresis_frames)
        initial = initial_schedule.integer()
        candidate = self._project_integer_boundaries(initial, self.schedule.integer().boundaries)
        if current_schedule is None or hysteresis_frames == 0:
            return candidate
        current = self._checked_current(current_schedule, initial)
        # Strict interior: a proposal exactly on the threshold may quantize
        # away. In particular, a .5-frame tie must not consume every allowed
        # movement of a very short sequence with a one-frame budget.
        hold = (self.schedule.boundaries-current.boundaries.to(self.schedule.boundaries)).abs() < effective[:, None]
        desired = torch.where(hold, current.boundaries, candidate.boundaries)
        return self._project_integer_boundaries(initial, desired)


class TimingFeedbackHead(nn.Module):
    """Compact residual allocation from current pose/geometry/contact evidence.

    Per-region features: R[14], predicted contact probability, and a geometric
    proximity proxy (not a replacement contact label), 16 -> 16 -> 16.
    Eight region codes plus root displacement/velocity and separate root/body
    rotation errors form 136 -> bottleneck frame evidence. Event pooling and
    a reduced cached slot token produce a bounded scalar residual per event.
    The full 512-wide context is not expanded over all body sample points.
    """
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64, *,
                 maximum_log_shift: float = .5, maximum_boundary_fraction: float = .1):
        super().__init__()
        if hidden_dim < 1 or bottleneck_dim < 1:
            raise ValueError("timing feedback widths must be positive")
        if not math.isfinite(maximum_log_shift) or maximum_log_shift <= 0:
            raise ValueError("maximum_log_shift must be finite and positive")
        if not math.isfinite(maximum_boundary_fraction) or not 0 < maximum_boundary_fraction <= 1:
            raise ValueError("maximum_boundary_fraction must be in (0,1]")
        self.maximum_log_shift = float(maximum_log_shift)
        self.maximum_boundary_fraction = float(maximum_boundary_fraction)
        self.region_encoder = nn.Sequential(nn.Linear(16, 16), nn.SiLU(),
                                            nn.Linear(16, 16), nn.SiLU())
        self.region_id = nn.Embedding(8, 16)
        self.frame_encoder = nn.Sequential(nn.Linear(136, bottleneck_dim), nn.SiLU())
        self.context_encoder = nn.Linear(hidden_dim, bottleneck_dim)
        duration_layers, previous = [], 2
        for width in dict.fromkeys((min(8, bottleneck_dim), min(16, bottleneck_dim),
                                   min(32, bottleneck_dim), bottleneck_dim)):
            duration_layers.append(nn.Linear(previous, width))
            if width != bottleneck_dim:
                duration_layers.append(nn.SiLU())
            previous = width
        self.duration_encoder = nn.Sequential(*duration_layers)
        self.readout = nn.Sequential(nn.LayerNorm(bottleneck_dim),
                                     nn.Linear(bottleneck_dim, max(4, bottleneck_dim // 2)),
                                     nn.SiLU(), nn.Linear(max(4, bottleneck_dim // 2), 1))
        nn.init.normal_(self.readout[-1].weight, std=.01)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, slot_context: Tensor, condition, initial_schedule: EventSchedule,
                current_schedule: EventSchedule, raw_motion: Tensor, relations: Tensor,
                contact_logits: Tensor) -> TimingFeedbackResult:
        c = condition
        batch, frames, dimension = raw_motion.shape
        if (batch, frames, dimension) != (c.batch_size, c.max_frames, 135):
            raise ValueError("timing feedback requires the exact padded raw motion horizon")
        if relations.shape != (batch, frames, 8, 14) or contact_logits.shape != (batch, frames, 8):
            raise ValueError("timing evidence must be B,T,8,14 relations and B,T,8 contact logits")
        if slot_context.shape[:2] != c.slot_mask.shape:
            raise ValueError("timing slot context must match B,M")
        valid = future_mask(c, frames)
        for name, value in (("raw motion", raw_motion), ("relations", relations),
                            ("contact logits", contact_logits)):
            if not bool(torch.isfinite(value[valid]).all()):
                raise ValueError(f"nonfinite valid {name} for timing feedback")
        # Stop an auxiliary timing objective from teaching the pilot to invent
        # convenient geometry. Cached semantic context remains differentiable.
        raw = raw_motion.detach().masked_fill(~valid[..., None], 0)
        rel = relations.detach().masked_fill(~valid[..., None, None], 0)
        logits = contact_logits.detach().masked_fill(~valid[..., None], 0)
        fields = expand_contacts(c, current_schedule.detach(), frames)
        slots = fields["slot_indices"]
        rows = torch.arange(batch, device=raw.device)[:, None]
        motion_references = _motion_reference_indices(c, slots)
        goals = c.keyposes[rows, motion_references]
        history_end = c.history[torch.arange(batch, device=raw.device), c.history_mask.sum(1) - 1]
        velocity = torch.diff(torch.cat((history_end[:, None, :3], raw[..., :3]), 1), dim=1) * c.fps
        rotation_error = (raw[..., 3:] - goals[..., 3:]).reshape(batch, frames, JOINTS, 6).square().mean(-1)
        motion_features = torch.cat((raw[..., :3] - goals[..., :3], velocity,
                                     rotation_error[..., :1], rotation_error[..., 1:].mean(-1, keepdim=True)), -1)
        proximity = torch.exp(-rel[..., 1:2].clamp_min(0) / .05)
        region_features = torch.cat((rel, logits.sigmoid()[..., None], proximity), -1)
        region_codes = self.region_encoder(region_features)
        region_codes = region_codes + self.region_id.weight[None, None]
        frame_codes = self.frame_encoder(torch.cat((region_codes.flatten(-2), motion_features), -1))
        membership = F.one_hot(slots, num_classes=c.slot_mask.shape[1]).to(frame_codes)
        membership = membership * valid[..., None]
        event_codes = torch.bmm(membership.transpose(1, 2), frame_codes)
        event_codes = event_codes / membership.sum(1).clamp_min(1)[..., None]
        base = initial_schedule.durations
        duration_features = torch.stack((base, current_schedule.durations.detach().to(base)), -1)
        duration_features = duration_features / c.total_frames[:, None, None]
        safe_context = slot_context.masked_fill(~c.slot_mask[..., None], 0)
        fused = (event_codes + self.context_encoder(safe_context) +
                 self.duration_encoder(duration_features.to(slot_context)))
        residual = self.maximum_log_shift * torch.tanh(self.readout(fused).squeeze(-1))
        residual = residual.masked_fill(~c.slot_mask, 0)
        minimum = initial_schedule.minimum_frames.to(base)
        spare = c.total_frames.to(base) - minimum.sum(1)
        extras = (base - minimum).clamp_min(0)
        log_weights = extras.clamp_min(torch.finfo(base.dtype).eps).log() + residual.to(base)
        weights = log_weights.masked_fill(~c.slot_mask, -torch.inf).softmax(1)
        proposal = minimum + spare[:, None] * weights
        delta = (proposal - base).masked_fill(~c.slot_mask, 0)
        shift = delta.cumsum(1).masked_fill(~c.slot_mask, 0).abs().amax(1)
        budget = c.total_frames.to(base) * self.maximum_boundary_fraction
        scale = (budget / shift.clamp_min(torch.finfo(base.dtype).eps)).clamp_max(1)
        durations = (base + scale[:, None] * delta).masked_fill(~c.slot_mask, 0)
        schedule = _with_durations(initial_schedule, durations)
        max_shift = (schedule.boundaries - initial_schedule.boundaries).abs().amax(1)
        integer_budget = budget.floor().long().clamp_min(1)
        return TimingFeedbackResult(schedule, residual, max_shift, scale, integer_budget)
