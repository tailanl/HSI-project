"""Joint retraining with endpoint establishment and geometric stance semantics.

The model, dense motion objectives and real data are unchanged. Unlike the
source-frozen v1 training, future foot contact is not treated as a planted foot
throughout its approach interval. This is a new objective policy, not an exact
continuation of a v1 optimizer. Ground-truth times remain supervision only.
"""
from __future__ import annotations

import torch

from hsi.stage3_sequence.contracts import future_mask
from hsi.stage3_sequence.objectives import masked_mean, projected_motion, keypose_loss
from hsi.stage3_sequence.timing_feedback import legal_motion_for_feedback
from hsi.stage3_joint.training import (
    JointLossConfig, GeometryCache, masked_contact_loss, derivative_losses,
    perturb_allowed_keyposes,
)
from hsi.stage3_joint_v2.contacts import POLICY_VERSION
from hsi.stage3_joint_v2.objectives import physical_objectives, geometry_relations


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
