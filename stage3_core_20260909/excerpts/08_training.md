# 联合训练损失与真实运动监督

实际损失入口是v2 joint_training_loss。旧stage3_joint/training.py仅提供下面这些配置与辅助函数；旧foot_slip_across_events和旧joint_training_loss不属于本次实际路径。速度/加速度是匹配GT变化，不是统一压到0。GT时间只进入时间监督。

下面是备份源码的逐字摘录，不是伪代码，也不是独立可执行模块。完整imports、辅助函数和校验仍在对应原文件中。

## JointLossConfig

来源：[hsi/stage3_joint/training.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/training.py:26)，原文件第 26–50 行。

```python
@dataclass(frozen=True)
class JointLossConfig:
    denoise: float = 1.0
    timing: float = 10.0
    keypose: float = .10
    root_velocity: float = .15
    root_acceleration: float = .05
    joint_velocity: float = .15
    joint_acceleration: float = .05
    projected_continuity: float = .03
    contact_classification: float = .10
    contact: float = .25
    collision: float = .25
    foot_slip: float = .02
    observed_foot_slip: float = .02
    release: float = .05
    rotation_regularization: float = .005
    acceleration_scale_m_s2: float = 10.0

    def validate(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("joint weights must be finite and nonnegative")
        if self.acceleration_scale_m_s2 <= 0:
            raise ValueError("positive acceleration scale required")
        return self
```

## masked_contact_loss

来源：[hsi/stage3_joint/training.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/training.py:78)，原文件第 78–86 行。

```python
def masked_contact_loss(logits, labels, known, frame_mask):
    if logits.shape != labels.shape or known.shape != labels.shape or known.dtype != torch.bool:
        raise ValueError("contact values and explicit boolean known mask must match")
    active = known & frame_mask[..., None]
    if not bool(torch.isfinite(labels[active]).all()) or bool(((labels[active] < 0) | (labels[active] > 1)).any()):
        raise ValueError("invalid known contact label")
    # Unknown values may contain NaN: remove BEFORE BCE, not after multiplication.
    safe = torch.where(active, labels, torch.zeros_like(labels))
    return masked_mean(F.binary_cross_entropy_with_logits(logits, safe.to(logits), reduction="none"), active)
```

## derivative_losses

来源：[hsi/stage3_joint/training.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/training.py:89)，原文件第 89–101 行。

```python
def derivative_losses(predicted, target, history, mask, fps, acceleration_scale=10.):
    """Dense derivative matching includes history→future and padded masking."""
    if predicted.shape != target.shape or history.shape[1] != 2:
        raise ValueError("matching trajectories and two history frames required")
    a, b = torch.cat((history.detach(), predicted), 1), torch.cat((history.detach(), target.detach()), 1)
    va, vb = torch.diff(a, dim=1) * fps, torch.diff(b, dim=1) * fps
    aa, ab = torch.diff(va, dim=1), torch.diff(vb, dim=1)
    extra = (None,) * (predicted.ndim - 2)
    m = mask[(...,) + extra]
    return (
        masked_mean((va[:, 1:] - vb[:, 1:]).square(), m),
        masked_mean(((aa - ab) * fps / acceleration_scale).square(), m),
    )
```

## perturb_allowed_keyposes

来源：[hsi/stage3_joint/training.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/training.py:122)，原文件第 122–144 行。

```python
def perturb_allowed_keyposes(condition, body, *, generator=None, max_degrees=3.):
    """Controlled Stage2-error augmentation, NEVER releases input permissions."""
    c = condition
    if not math.isfinite(max_degrees) or not 0 <= max_degrees <= 10:
        raise ValueError("bounded keypose perturbation required")
    with torch.no_grad():
        rotations = rotation_6d_to_matrix(c.keyposes[..., 3:].reshape(c.batch_size, -1, 22, 6))
        axes = torch.randn((*rotations.shape[:-2], 3), device=c.device, generator=generator)
        axes = F.normalize(axes, dim=-1)
        angle = torch.rand(rotations.shape[:-2], device=c.device, generator=generator)
        bound = torch.minimum(c.joint_tolerance, angle.new_full(angle.shape, math.radians(max_degrees)))
        angle = angle * bound * (~c.locked_joints & c.keypose_mask[..., None])
        x, y, z = axes.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape_as(rotations)
        eye = torch.eye(3, device=c.device, dtype=rotations.dtype)
        delta = eye + angle.sin()[..., None, None] * skew + (1-angle.cos())[..., None, None] * (skew @ skew)
        changed = matrix_to_rotation_6d(delta @ rotations)
        original = c.keyposes[..., 3:].reshape_as(changed)
        changed = torch.where(c.locked_joints[..., None], original, changed)
        keyposes = torch.cat((c.keyposes[..., :3], changed.flatten(-2)), -1)
        joints = body.joints_world(keyposes) - keyposes[..., None, :3]
    return replace(c, keyposes=keyposes, keypose_joints=joints).validate()
```

## GeometryCache

来源：[hsi/stage3_joint/training.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/training.py:53)，原文件第 53–75 行。

```python
class GeometryCache:
    """One differentiable skinning graph per candidate, never across steps."""
    def __init__(self, motion, body):
        self.motion = motion
        if callable(getattr(body, "mesh", None)):
            mesh = mesh_in_chunks(body, motion)
            self.joints = mesh.joints
            self.regions = body.region_points_from_vertices(mesh.vertices)
            self.collisions = body.collision_points_from_vertices(mesh.vertices)
        else:
            self.joints = body.joints_world(motion)
            self.regions = body.region_points(motion)
            self.collisions = body.collision_points(motion)

    def region_points(self, motion):
        if motion is not self.motion:
            raise ValueError("geometry cache motion mismatch")
        return self.regions

    def collision_points(self, motion):
        if motion is not self.motion:
            raise ValueError("geometry cache motion mismatch")
        return self.collisions
```

## mesh_in_chunks

来源：[hsi/stage3_joint/refine.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/refine.py:98)，原文件第 98–127 行。

```python
def mesh_in_chunks(body, motion, chunk_frames=24, *, checkpoint_backward=True):
    """Reusable actual-skinning API for joint training as well as U15.

    Return MeshOutput with differentiable concatenated vertices/joints. Shape
    and hand buffers are never optimized. Checkpointing bounds intermediate
    skinning activations; output tensors still cover every frame and vertex.
    """
    if type(chunk_frames) is not int or chunk_frames < 1:
        raise ValueError('chunk_frames must be a positive integer')
    def evaluate(chunk):
        result = body.mesh(chunk)
        return result.vertices, result.joints
    chunks = []
    for start in range(0, motion.shape[1], chunk_frames):
        chunk = motion[:, start:start+chunk_frames]
        if checkpoint_backward and chunk.requires_grad and torch.is_grad_enabled():
            # Recompute skinning during backward to bound activation memory;
            # no detach of the motion or learned/physical gradient is involved.
            from torch.utils.checkpoint import checkpoint
            chunks.append(checkpoint(evaluate, chunk, use_reentrant=False))
        else:
            chunks.append(evaluate(chunk))
    vertices = torch.cat([part[0] for part in chunks], 1)
    joints = torch.cat([part[1] for part in chunks], 1)
    if (vertices.ndim != 4 or vertices.shape[:2] != motion.shape[:2] or vertices.shape[-1] != 3
            or joints.shape != (*motion.shape[:2], 22, 3)):
        raise ValueError('mesh body must return world vertices and 22 world joints')
    if not bool(torch.isfinite(vertices).all() & torch.isfinite(joints).all()):
        raise ValueError('nonfinite mesh geometry')
    return MeshOutput(vertices, joints)
```

## joint_training_loss

来源：[hsi/stage3_joint_v2/training.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/training.py:23)，原文件第 23–127 行。

```python
def joint_training_loss(model, diffusion, batch, *, weights=None, timesteps=None,
                        noise=None, generator=None, physics_scale=1., feedback_rounds=1):
    """Same x0/temporal supervision, corrected event contact and stance losses.

The shared physical objective is also used by v2 sampling. Its relation channel
8 means currently geometrically established contact, NOT future mandatory
contact. Required contact is handled separately: establish at end, hold for
the interval. Observed foot labels are kept independent and unknown-safe.
"""
    cfg = (weights or JointLossConfig()).validate()
    c = batch.condition.validate()
    target, durations = batch.clean, batch.timing
    if target.shape != (c.batch_size, c.max_frames, 135) or not bool(torch.isfinite(target).all()):
        raise ValueError('finite target must cover exact padded future horizon')
    if durations.shape != c.slot_mask.shape or durations.dtype != torch.long:
        raise ValueError('integer duration supervision mismatch')
    if not torch.equal(durations.sum(1), c.total_frames) or bool((durations[~c.slot_mask] != 0).any()):
        raise ValueError('duration sum/padding mismatch')
    if bool((durations[c.slot_mask] < c.minimum_frames[c.slot_mask]).any()):
        raise ValueError('duration supervision violates minima')
    if not 0 < physics_scale <= 1 or feedback_rounds not in (1, 2, 3):
        raise ValueError('joint physical training cannot be disabled; use 1–3 feedback rounds')
    mask = future_mask(c)
    if timesteps is None:
        timesteps = torch.randint(diffusion.steps, (c.batch_size,), device=c.device, generator=generator)
    if noise is None:
        noise = torch.randn(target.shape, device=c.device, generator=generator)
    normalized = diffusion.normalize(target)
    noisy = diffusion.add_noise(normalized, timesteps, noise)
    context = model.encode(c)
    schedule = context.schedule.integer()
    relations, feedback = None, None
    feedback_losses = []
    target_fractions = durations.to(target) / c.total_frames[:, None]
    timing_initial = masked_mean(
        (context.schedule.durations / c.total_frames[:, None] - target_fractions).square(), c.slot_mask)
    if getattr(model, 'timing_feedback', None) is None:
        raise ValueError('joint protocol requires dynamic timing feedback')
    for _ in range(feedback_rounds):
        with torch.no_grad():
            pilot = model.denoise(noisy, timesteps, context, schedule=schedule, relations=relations)
            raw_pilot = legal_motion_for_feedback(diffusion.denormalize(pilot.clean_motion), c)
            pilot_cache = GeometryCache(raw_pilot, batch.body)
            evidence = geometry_relations(raw_pilot, c, schedule, pilot_cache, batch.scene).detach()
        feedback = model.timing_feedback(context.slot_context, c, context.schedule, schedule,
                                         raw_pilot, evidence, pilot.contact_logits.detach())
        feedback_losses.append(masked_mean(
            (feedback.schedule.durations / c.total_frames[:, None] - target_fractions).square(), c.slot_mask))
        schedule = feedback.integer(context.schedule, current_schedule=schedule)
        with torch.no_grad():
            actual_pilot, _ = projected_motion(raw_pilot, c, schedule)
            relations = geometry_relations(actual_pilot, c, schedule,
                GeometryCache(actual_pilot, batch.body), batch.scene).detach()
    prediction = model.denoise(noisy, timesteps, context, schedule=schedule, relations=relations)
    raw = diffusion.denormalize(prediction.clean_motion)
    legal = legal_motion_for_feedback(raw, c)
    physical, rotation_valid = projected_motion(raw, c, schedule)
    raw_cache = GeometryCache(legal, batch.body)
    physical_cache = GeometryCache(physical, batch.body)
    physics = physical_objectives(physical, c, schedule, physical_cache, batch.scene)
    with torch.no_grad():
        target_joints = batch.body.joints_world(target)
        history_joints = batch.body.joints_world(c.history)
    rv, ra = derivative_losses(legal[..., :3], target[..., :3], c.history[..., :3],
                               mask, c.fps, cfg.acceleration_scale_m_s2)
    jv, ja = derivative_losses(raw_cache.joints - legal[..., None, :3],
        target_joints - target[..., None, :3], history_joints - c.history[..., None, :3],
        mask, c.fps, cfg.acceleration_scale_m_s2)
    pv, pa = derivative_losses(physical[..., :3], target[..., :3], c.history[..., :3],
                               mask, c.fps, cfg.acceleration_scale_m_s2)
    pjv, pja = derivative_losses(physical_cache.joints - physical[..., None, :3],
        target_joints - target[..., None, :3], history_joints - c.history[..., None, :3],
        mask, c.fps, cfg.acceleration_scale_m_s2)
    # This separate dense stance supervision already excludes unknown/swing
    # frames. Do not replace it with a future keypose's contact requirements.
    observed = batch.contact_known[:, :, 3:5] & (batch.contacts[:, :, 3:5] > .5) & mask[..., None]
    pair = observed[:, 1:] & observed[:, :-1]
    feet_velocity = torch.diff(raw_cache.regions[:, :, 3:5], dim=1) * c.fps
    observed_slip = masked_mean(feet_velocity[..., :2].square().sum(-1), pair)
    rotations = raw[..., 3:].reshape(*raw.shape[:2], 22, 6)
    first, second = rotations[..., :3], rotations[..., 3:]
    rotation_reg = ((first.square().sum(-1)-1).square() + (second.square().sum(-1)-1).square()
                    + (first*second).sum(-1).square())
    losses = dict(
        denoise=masked_mean((prediction.clean_motion-normalized).square(), mask[..., None]),
        timing=(timing_initial + torch.stack(feedback_losses).mean())*.5,
        keypose=keypose_loss(raw, c, schedule), root_velocity=rv, root_acceleration=ra,
        joint_velocity=jv, joint_acceleration=ja, projected_continuity=pv+pa+pjv+pja,
        contact_classification=masked_contact_loss(prediction.contact_logits, batch.contacts, batch.contact_known, mask),
        contact=physics['contact'], collision=physics['collision'],
        foot_slip=physics['foot_slip'], observed_foot_slip=observed_slip,
        release=physics['release'], rotation_regularization=masked_mean(rotation_reg, mask[..., None]),
    )
    physical_names = {'contact', 'collision', 'foot_slip', 'release', 'projected_continuity', 'observed_foot_slip'}
    loss = sum(value * getattr(cfg, name) * (physics_scale if name in physical_names else 1.)
               for name, value in losses.items())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('nonfinite joint objective')
    return dict(loss=loss, losses=losses, context=context, schedule=schedule, raw_motion=raw,
        motion=physical, prediction=prediction, timing_feedback=feedback,
        timing_initial=timing_initial, outside_rate=physics['outside_rate'],
        contact_known_count=int((batch.contact_known & mask[..., None]).sum()),
        positive_contact_count=int((batch.contact_known & (batch.contacts > .5) & mask[..., None]).sum()),
        raw_rotation_valid=rotation_valid, feedback_rounds=feedback_rounds,
        training_policy_version=POLICY_VERSION)
```
