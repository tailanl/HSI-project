"""Whole-sequence DDIM with predicted timing, geometry feedback and hard locks."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .contracts import future_mask
from .constraints import extract_keyposes, free_dof_mask
from .objectives import (
    projected_motion, geometry_relations, physical_objectives, keypose_loss, keypose_metrics,
)
from .timing_feedback import legal_motion_for_feedback


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 20
    seed: int = 0
    geometry_feedback: bool = True
    timing_hysteresis_frames: float = 1.5
    guidance_scale: float = .05
    guidance_start_fraction: float = .5
    maximum_update_norm: float = .1
    collision_weight: float = 1.
    contact_weight: float = 1.
    keypose_weight: float = .5
    foot_slip_weight: float = .01
    release_weight: float = .1

    def validate(self):
        if type(self.steps) is not int or self.steps < 2 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("steps/seed must be nonnegative integers and steps >=2")
        for name in ("guidance_scale","maximum_update_norm","collision_weight",
                     "contact_weight","keypose_weight","foot_slip_weight","release_weight",
                     "timing_hysteresis_frames"):
            v = getattr(self,name)
            if not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid sampling parameter {name}")
        if isinstance(self.timing_hysteresis_frames, bool):
            raise ValueError("invalid sampling parameter timing_hysteresis_frames")
        if not 0 <= self.guidance_start_fraction <= 1:
            raise ValueError("guidance_start_fraction must be in [0,1]")
        return self


def _energy(motion,c,schedule,body,scene,config):
    values = physical_objectives(motion,c,schedule,body,scene)
    values["keypose"] = keypose_loss(motion,c,schedule)
    total = (config.collision_weight*values["collision"]+
             config.contact_weight*values["contact"]+
             config.keypose_weight*values["keypose"]+
             config.foot_slip_weight*values["foot_slip"]+
             config.release_weight*values["release"])
    return total,values


def sample_motion(model,diffusion,condition,*,body,scene,config=None):
    """Generate all future frames together; result is NOT production release.

    The caller supplies a trained model (or explicitly labels an untrained
    diagnostic). Internal targets come only from condition.keyposes and its
    declared events, never the supervised future or true event times.
    """
    config = (config or SamplingConfig()).validate()
    c = condition.validate()
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=c.device).manual_seed(config.seed)
    try:
        with torch.no_grad():
            context = model.encode(c)
            initial_schedule = context.schedule
            schedule = context.schedule.integer()
            mask = future_mask(c)
            noisy = torch.randn((c.batch_size,c.max_frames,135),device=c.device,
                                dtype=c.history.dtype,generator=generator)
            noisy = noisy.masked_fill(~mask[...,None],0)
        indices = diffusion.inference_steps(config.steps)
        relations, logs, raw_invalid_total = None, [], 0
        for index, step in enumerate(indices):
            following = indices[index+1] if index+1 < len(indices) else -1
            with torch.no_grad():
                previous_schedule = schedule
                prediction = model.denoise(
                    noisy,torch.full((c.batch_size,),step,device=c.device,dtype=torch.long),
                    context,schedule=schedule,relations=relations)
                clean_normalized = prediction.clean_motion.detach()
                feedback = None
                timing_candidate = schedule
                feedback_head = getattr(model, "timing_feedback", None)
                if feedback_head is not None:
                    # Time observes the RAW legal prediction, not hard-placed
                    # target poses. Its contact evidence cannot change the
                    # declared contact/release masks in the condition.
                    raw_legal = legal_motion_for_feedback(
                        diffusion.denormalize(clean_normalized), c)
                    raw_relations = geometry_relations(raw_legal, c, schedule, body, scene)
                    feedback = feedback_head(context.slot_context, c, initial_schedule,
                                             schedule, raw_legal, raw_relations,
                                             prediction.contact_logits)
                    timing_candidate = feedback.integer(initial_schedule)
                    schedule = feedback.integer(initial_schedule, current_schedule=schedule,
                                                hysteresis_frames=config.timing_hysteresis_frames)
                candidate,raw_valid = projected_motion(
                    diffusion.denormalize(clean_normalized),c,schedule)
                raw_invalid = int((~raw_valid & mask[...,None]).sum().item())
                raw_invalid_total += raw_invalid
            guided = config.guidance_scale > 0 and index/max(len(indices)-1,1) >= config.guidance_start_fraction
            update_norm, energy_before = 0., None
            if guided:
                with torch.enable_grad():
                    variable = clean_normalized.detach().requires_grad_(True)
                    actual,_ = projected_motion(diffusion.denormalize(variable),c,schedule)
                    energy,values = _energy(actual,c,schedule,body,scene,config)
                    if not bool(torch.isfinite(energy)):
                        raise FloatingPointError("nonfinite sampling guidance energy")
                    gradient = torch.autograd.grad(energy,variable,only_inputs=True)[0]
                    if not bool(torch.isfinite(gradient).all()):
                        raise FloatingPointError("nonfinite geometry gradient")
                    free = free_dof_mask(actual,schedule.keypose_times.long(),c.keypose_mask,
                                         c.locked_root,c.locked_joints) & mask[...,None]
                    delta = -config.guidance_scale*gradient*free
                    norm = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
                    cap = (config.maximum_update_norm/norm).clamp_max(1)
                    delta = delta*cap[:,None,None]
                    update_norm = float(delta.flatten(1).norm(dim=1).max().detach())
                    energy_before = float(energy.detach())
                    clean_normalized = (variable+delta).detach()
                with torch.no_grad():
                    candidate,_ = projected_motion(diffusion.denormalize(clean_normalized),c,schedule)
            with torch.no_grad():
                # Always query the actual post-constraint body, not an earlier
                # unconstrained pose which could conceal projection collisions.
                energy_after,values = _energy(candidate,c,schedule,body,scene,config)
                relations = (values["relations"].detach() if config.geometry_feedback else None)
                encoded_clean = diffusion.normalize(candidate)
                noisy = diffusion.ddim_step(noisy,encoded_clean,step,following)
                noisy = noisy.masked_fill(~mask[...,None],0)
                metrics = keypose_metrics(candidate,c,schedule)
                logs.append({
                    "step":step,"guided":guided,"update_norm":update_norm,
                    "energy_before":energy_before,"energy_after":float(energy_after),
                    "collision":float(values["collision"]),"contact":float(values["contact"]),
                    "release":float(values["release"]),"foot_slip":float(values["foot_slip"]),
                    "max_locked_root_error":float(metrics["max_locked_root_error"]),
                    "max_locked_joint_error":float(metrics["max_locked_joint_error"]),
                    "raw_degenerate_rotation_count":raw_invalid,
                    "timing_feedback":feedback is not None,
                    "previous_durations":previous_schedule.durations.cpu().tolist(),
                    "durations":schedule.durations.cpu().tolist(),
                    "previous_keypose_times":previous_schedule.keypose_times.cpu().tolist(),
                    "keypose_times":schedule.keypose_times.cpu().tolist(),
                    "timing_shift_from_initial":(
                        schedule.boundaries - initial_schedule.integer().boundaries
                    ).abs().amax(1).cpu().tolist(),
                    "timing_boundary_budget":(feedback.integer_boundary_budget.cpu().tolist()
                                               if feedback is not None else [0] * c.batch_size),
                    "timing_continuous_shift":(feedback.max_boundary_shift.cpu().tolist()
                                                if feedback is not None else [0.] * c.batch_size),
                    "timing_adjustment_scale":(feedback.adjustment_scale.cpu().tolist()
                                                if feedback is not None else [1.] * c.batch_size),
                    "timing_effective_hysteresis_frames":(
                        feedback.effective_hysteresis_frames(config.timing_hysteresis_frames).cpu().tolist()
                        if feedback is not None else [0.] * c.batch_size),
                    "timing_candidate_changed":(timing_candidate.boundaries != previous_schedule.boundaries).any(1).cpu().tolist(),
                    "timing_update_changed":(schedule.boundaries != previous_schedule.boundaries).any(1).cpu().tolist(),
                    "timing_candidate_retained":(
                        (timing_candidate.boundaries != previous_schedule.boundaries).any(1) &
                        (schedule.boundaries == previous_schedule.boundaries).all(1)).cpu().tolist(),
                    "timing_retained_boundary_count":(
                        (timing_candidate.boundaries != previous_schedule.boundaries) &
                        (schedule.boundaries == previous_schedule.boundaries)).sum(1).cpu().tolist(),
                })
        with torch.no_grad():
            result,_ = projected_motion(diffusion.denormalize(noisy),c,schedule)
            metrics = keypose_metrics(result,c,schedule)
            _,last_physics = _energy(result,c,schedule,body,scene,config)
            actual_keyposes = extract_keyposes(result,schedule.keypose_times.long(),c.keypose_mask)
        scalar_metrics = {name:float(value) for name,value in metrics.items() if value.numel()==1}
        return {
            "motion":result,"keyposes":actual_keyposes,"schedule":schedule,
            "metrics":scalar_metrics,"trace":logs,
            "collision":float(last_physics["collision"]),
            "contact":float(last_physics["contact"]),
            "raw_degenerate_rotation_count":raw_invalid_total,
            "full_mesh_evaluation_performed":False,
            "motion_release_authorized":False,
            "status":"research_generation_requires_independent_evaluation",
        }
    finally:
        model.train(was_training)
