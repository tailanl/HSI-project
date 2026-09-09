# 初始时间分配与循环中的有界时间反馈

内部GT帧号不是条件输入。初始时长按正值和最小时长分配；循环反馈读取当前预测动作及几何/接触证据，有界更新并保留执行顺序。integer依赖同一完整源码中的校验和边界投影辅助函数；下面不是可单独运行的替代实现。

下面是备份源码的逐字摘录，不是伪代码，也不是独立可执行模块。完整imports、辅助函数和校验仍在对应原文件中。

## DurationHead

来源：[hsi/stage3_sequence/timing.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/timing.py:254)，原文件第 254–287 行。

```python
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
```

## allocate_durations

来源：[hsi/stage3_sequence/timing.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/timing.py:174)，原文件第 174–251 行。

```python
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
```

## TimingFeedbackHead

来源：[hsi/stage3_sequence/timing_feedback.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/timing_feedback.py:153)，原文件第 153–255 行。

```python
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
```

## TimingFeedbackResult.integer

来源：[hsi/stage3_sequence/timing_feedback.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/timing_feedback.py:125)，原文件第 125–150 行。

```python
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
```
