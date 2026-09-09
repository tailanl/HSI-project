# DDIM完整采样循环与末端refine入口

当前循环来自stage3_joint_v2，而不是stage3_sequence/sampling.py的旧循环。后者仅提供SamplingConfig。先预测当前clean动作，更新时间、执行权限投影和几何guidance，再查询约束后关系反馈下一轮；结束后sample_joint_motion调用全序列refine。

下面是备份源码的逐字摘录，不是伪代码，也不是独立可执行模块。完整imports、辅助函数和校验仍在对应原文件中。

## SamplingConfig

来源：[hsi/stage3_sequence/sampling.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/sampling.py:17)，原文件第 17–45 行。

```python
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
```

## MotionDiffusion.ddim_step

来源：[hsi/stage3_sequence/diffusion.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/diffusion.py:56)，原文件第 56–65 行。

```python
    def ddim_step(self, noisy: Tensor, predicted_clean: Tensor, current: int, following: int) -> Tensor:
        """Deterministic DDIM (eta=0); follow=-1 returns the clean endpoint."""
        if following == -1:
            return predicted_clean
        if not 0 <= following < current < self.steps:
            raise ValueError("DDIM indices must descend")
        alpha = self.alpha_bar[current].to(noisy)
        next_alpha = self.alpha_bar[following].to(noisy)
        noise = (noisy-alpha.sqrt()*predicted_clean)/(1-alpha).sqrt().clamp_min(1e-8)
        return next_alpha.sqrt()*predicted_clean+(1-next_alpha).sqrt()*noise
```

## _energy

来源：[hsi/stage3_joint_v2/sampling.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/sampling.py:26)，原文件第 26–34 行。

```python
def _energy(motion,c,schedule,body,scene,config):
    values = physical_objectives(motion,c,schedule,body,scene)
    values["keypose"] = keypose_loss(motion,c,schedule)
    total = (config.collision_weight*values["collision"]+
             config.contact_weight*values["contact"]+
             config.keypose_weight*values["keypose"]+
             config.foot_slip_weight*values["foot_slip"]+
             config.release_weight*values["release"])
    return total,values
```

## sample_motion

来源：[hsi/stage3_joint_v2/sampling.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/sampling.py:37)，原文件第 37–173 行。

```python
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
                    "training_policy_version":TRAINING_POLICY_VERSION,
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
            "training_policy_version":TRAINING_POLICY_VERSION,
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
```

## sample_joint_motion

来源：[hsi/stage3_joint_v2/sampling.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/sampling.py:218)，原文件第 218–294 行。

```python
def sample_joint_motion(model, diffusion, condition, *, body, scene,
                        config: JointSamplingConfig | None = None,
                        allowed_dofs: Tensor | None = None):
    """Return the old sample fields plus separately identified refinement.

``unrefined_motion`` is the original full-loop result, NOT raw/no-lock DDIM.
``motion`` and ``keyposes`` describe the final returned motion; keyposes are
always extracted at the unmodified predicted times. ``trace`` is exclusively
the diffusion trace; post-refinement steps live in ``refine_report['trace']``.
Ground-truth future motion or timestamps are not accepted by this interface.
"""
    if not isinstance(model, JointSequenceDenoiser):
        raise TypeError("JointSequenceDenoiser required; do not silently relabel an R3 model")
    cfg = (config or JointSamplingConfig()).validate()
    c = condition.validate()
    condition_before = _snapshot(c)
    result = sample_motion(model, diffusion, c, body=body, scene=scene, config=cfg.sampling)
    _unchanged(c, condition_before, "condition during diffusion")
    original = result['motion'].detach().clone()
    schedule = result['schedule']
    schedule_before = _snapshot(schedule)
    unrefined_keys = extract_keyposes(original, schedule.keypose_times.long(), c.keypose_mask)
    original_allowed = allowed_dofs.detach().clone() if isinstance(allowed_dofs, Tensor) else None
    if cfg.enable_refine:
        # A separate buffer prevents any downstream in-place operation from
        # replacing the saved full-loop result, even before integrity checks.
        refined = refine_motion(c, original.clone(), schedule, body=body, scene=scene,
                                config=cfg.refine, allowed_dofs=allowed_dofs)
        if not isinstance(refined, dict) or 'motion' not in refined or 'report' not in refined:
            raise ValueError("Post-refinement must return motion and an explicit report")
        final, report = refined['motion'], refined['report']
        if not isinstance(report, dict) or report.get('motion_release_authorized') is not False:
            raise ValueError("Refinement report must explicitly deny motion release")
    else:
        final = original.clone()
        report = dict(schema='hsi.stage3_joint.refine_disabled.v2', status='disabled_by_caller',
            accepted_iterations=0, trace=[], schedule_changed=False, motion_release_authorized=False)
    _unchanged(c, condition_before, "condition during refinement")
    _unchanged(schedule, schedule_before, "predicted schedule during refinement")
    if original_allowed is not None and not torch.equal(allowed_dofs, original_allowed):
        raise RuntimeError("Refinement modified the caller's allowed_dofs mask")
    if (not isinstance(final, Tensor) or final.shape != original.shape or final.device != original.device
            or final.dtype != original.dtype or not bool(torch.isfinite(final).all())):
        raise ValueError("Refinement returned malformed/nonfinite motion")
    free = free_dof_mask(original, schedule.keypose_times.long(), c.keypose_mask,
                         c.locked_root, c.locked_joints) & future_mask(c)[..., None]
    if allowed_dofs is not None:
        if not isinstance(allowed_dofs, Tensor) or allowed_dofs.shape != free.shape or allowed_dofs.dtype != torch.bool or allowed_dofs.device != free.device:
            raise ValueError("allowed_dofs must be a boolean motion-shaped permission mask")
        free &= allowed_dofs
    if not torch.equal(final[~free], original[~free]):
        raise RuntimeError("Post-refinement changed locked, disallowed or padded coordinates")
    with torch.no_grad():
        final = final.detach().clone()
        actual_keyposes = extract_keyposes(final, schedule.keypose_times.long(), c.keypose_mask)
        measured = keypose_metrics(final, c, schedule)
        physics = physical_objectives(final, c, schedule, body, scene)
    # Preserve old before-refine numbers rather than leaving stale statistics
    # under fields that now describe the final returned motion.
    output = dict(result)
    output.update(schema='hsi.stage3_joint.sampling_with_refine.v2',
        training_policy_version=TRAINING_POLICY_VERSION,
        unrefined_motion=original, unrefined_keyposes=unrefined_keys,
        unrefined_metrics=result['metrics'],
        unrefined_physics={name: result.get(name) for name in ('collision', 'contact')},
        unrefined_status=result.get('status'),
        motion=final, keyposes=actual_keyposes,
        metrics={name: float(value) for name, value in measured.items() if value.numel() == 1},
        collision=float(physics['collision']), contact=float(physics['contact']),
        final_outside_rate=float(physics['outside_rate']), refine_report=report,
        refine_status=report.get('status'), refinement_executed=cfg.enable_refine,
        refinement_accepted=bool(report.get('accepted_iterations', 0)),
        joint_sampling_config=asdict(cfg), original_diffusion_trace_retained=True,
        final_keyposes_extracted_from_motion=True, schedule_changed_by_refinement=False,
        full_mesh_evaluation_performed=False, motion_release_authorized=False,
        status='joint_sampling_refinement_diagnostic_requires_independent_evaluation')
    return output
```
