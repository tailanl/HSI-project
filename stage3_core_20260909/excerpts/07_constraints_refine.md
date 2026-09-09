# 权限投影、全序列refine与失败回退

root/朝向锁和授权关节容差不默认放开。全网格审核失败时拒绝更新或回退，不通过挪动场景、平滑或删除失败帧来补结果。这里的projected_motion是当前仍共用的基础工具，不是基础文件中已被v2替换的旧接触目标函数。

下面是备份源码的逐字摘录，不是伪代码，也不是独立可执行模块。完整imports、辅助函数和校验仍在对应原文件中。

## project_keyposes

来源：[hsi/stage3_sequence/constraints.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/constraints.py:160)，原文件第 160–173 行。

```python
def project_keyposes(motion: Tensor, keyposes: Tensor, keypose_frames: Tensor, keypose_mask: Tensor,
                     locked_root: Tensor, locked_joints: Tensor) -> Tensor:
    """Copy only specified root coordinates / complete six-value joint groups.

    No interpolation, tolerance clamping, pose legalizing, or in-place edit is
    performed. Gradients to overwritten motion coordinates are exactly zero.
    """
    _targets(motion, keyposes, keypose_frames, keypose_mask)
    locks = _locks(motion, keypose_frames, keypose_mask, locked_root, locked_joints)
    batch_indices = torch.arange(motion.shape[0], device=motion.device)[:, None].expand_as(keypose_frames)[keypose_mask]
    frames = keypose_frames[keypose_mask] - 1
    output = motion.clone()
    output[batch_indices, frames] = torch.where(locks[keypose_mask], keyposes[keypose_mask], motion[batch_indices, frames])
    return output
```

## free_dof_mask

来源：[hsi/stage3_sequence/constraints.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/constraints.py:176)，原文件第 176–183 行。

```python
def free_dof_mask(motion: Tensor, keypose_frames: Tensor, keypose_mask: Tensor,
                  locked_root: Tensor, locked_joints: Tensor) -> Tensor:
    """True for freely optimizable motion coordinates; body shape is not a DOF."""
    locks = _locks(motion, keypose_frames, keypose_mask, locked_root, locked_joints)
    batch_indices = torch.arange(motion.shape[0], device=motion.device)[:, None].expand_as(keypose_frames)[keypose_mask]
    result = torch.ones_like(motion, dtype=torch.bool)
    result[batch_indices, keypose_frames[keypose_mask] - 1] = ~locks[keypose_mask]
    return result
```

## projected_motion

来源：[hsi/stage3_sequence/objectives.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/objectives.py:41)，原文件第 41–48 行。

```python
def projected_motion(raw, condition, schedule):
    """Record raw degeneracy before legalizing, then enforce only locked DOFs."""
    rotations = raw[...,3:].reshape(*raw.shape[:2],JOINTS,6)
    valid = rotation_6d_valid_mask(rotations)
    clean = torch.cat((raw[...,:3],legalize_rotation_6d(rotations).flatten(-2)),dim=-1)
    projected = project_keyposes(clean,condition.keyposes,schedule.keypose_times.long(),
                                  condition.keypose_mask,condition.locked_root,condition.locked_joints)
    return projected.masked_fill(~future_mask(condition,raw.shape[1])[...,None],0),valid
```

## _event_terms

来源：[hsi/stage3_joint_v2/refine.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/refine.py:142)，原文件第 142–147 行。

```python
def _event_terms(regions, fields, c, schedule, cfg):
    """Single shared policy: establish endpoint, hold interval, release endpoint."""
    values = contact_event_losses(regions, c, schedule,
        contact_tolerance=cfg.contact_tolerance_m, release_margin=cfg.release_margin_m)
    return (values['contact'], values['release'], values['contact_error_m'],
            values['release_error_m'], values['resolved_targets'])
```

## _foot_slip

来源：[hsi/stage3_joint_v2/refine.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/refine.py:150)，原文件第 150–154 行。

```python
def _foot_slip(regions, history_regions, fields, c, target, cfg, schedule):
    """Slip is conditional on actual established support, never declared intent."""
    values = foot_slip_terms(regions, c, schedule, history_regions=history_regions,
                            contact_tolerance=cfg.contact_tolerance_m)
    return values['loss'], values['pair_count'], values['per_row']
```

## _evaluate

来源：[hsi/stage3_joint_v2/refine.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/refine.py:171)，原文件第 171–227 行。

```python
def _evaluate(motion, c, schedule, body, scene, cfg, reference, history_geometry, *, audit=False):
    vertices, joints, regions = _mesh(body, motion, cfg.mesh_chunk_frames)
    fields = event_contact_state(c, schedule, regions, length=motion.shape[1],
                                 contact_tolerance=cfg.contact_tolerance_m)
    valid = fields['frame_mask']
    # Audit ALWAYS uses every actual mesh vertex, even when the body adapter
    # has a configured collision subsample. Optimization uses fixed uniform IDs.
    count = vertices.shape[-2]
    ids = torch.arange(count, device=motion.device) if audit else torch.linspace(
        0, count-1, min(count, cfg.max_surface_points), device=motion.device).long()
    points = vertices.index_select(-2, ids)
    sdf = scene.query(points)
    distance, outside = sdf.signed_distance_m, sdf.outside
    if distance.shape != points.shape[:-1] or outside.shape != distance.shape or outside.dtype != torch.bool:
        raise ValueError('scene query returned invalid distance/outside shapes')
    known = valid[..., None] & ~outside
    if not bool(torch.isfinite(distance[known]).all()):
        raise ValueError('nonfinite known scene distance')
    # Unknown fallback distances are NEVER interpreted as negative labels.
    penetration = torch.relu(-torch.where(known, distance, torch.zeros_like(distance)))
    collision = _mean(penetration.square(), known)
    outside_rate = _row_mean(outside.to(motion), valid[..., None])
    unknown_distance = torch.zeros_like(distance)
    bounds = getattr(scene, 'world_bounds', None)
    if bounds is None:
        bounds = getattr(getattr(scene, 'grid', None), 'world_bounds', None)
    if isinstance(bounds, Tensor):
        bounds = bounds.to(motion)
        if bounds.ndim == 2:
            bounds = bounds[None]
        bounds = bounds.expand(c.batch_size, -1, -1)
        delta = torch.relu(bounds[:, None, 0, None]-points)+torch.relu(points-bounds[:, None, 1, None])
        unknown_distance = torch.where(outside, delta.norm(dim=-1), torch.zeros_like(distance))
    unknown_loss = outside_rate.mean()+_mean(unknown_distance.square(), valid[..., None])
    contact, release, contact_max, release_max, targets = _event_terms(regions, fields, c, schedule, cfg)
    slip, slip_pairs, slip_rows = _foot_slip(regions, history_geometry[2], fields, c, targets, cfg, schedule)
    vel, acc, boundary = _temporal(joints, reference['joints'], history_geometry[1], valid, c.fps, cfg)
    pose = _mean((motion[..., :3]-reference['motion'][..., :3]).square(), valid[..., None])
    angles = rotation_geodesic(motion[..., 3:].reshape(*motion.shape[:2], 22, 6),
                               reference['motion'][..., 3:].reshape(*motion.shape[:2], 22, 6))
    pose = pose+_mean(angles.square(), valid[..., None])
    km = keypose_metrics(motion, c, schedule)
    components = dict(collision=collision, contact=contact, release=release, outside=unknown_loss,
        foot_slip=slip, history_boundary=boundary.mean(), velocity_preservation=vel,
        acceleration_preservation=acc, pose_preservation=pose, keypose=keypose_loss(motion, c, schedule))
    energy = sum(getattr(cfg, key)*value for key, value in components.items())
    root_path = (motion[:, 1:, :3]-motion[:, :-1, :3]).norm(dim=-1)*valid[:, 1:]
    rotation_path = rotation_geodesic(motion[:, 1:, 3:].reshape(c.batch_size, -1, 22, 6),
                                      motion[:, :-1, 3:].reshape(c.batch_size, -1, 22, 6))
    rotation_path = (rotation_path*valid[:, 1:, None]).sum(1)
    return dict(energy=energy, components=components, outside_mask=outside & valid[..., None],
        outside_rate=outside_rate, unknown_distance_m=_row_max(unknown_distance, valid[..., None]),
        penetration_rms_m=_row_mean(penetration.square(), known).sqrt(),
        penetration_max_m=_row_max(penetration, known), contact_error_m=contact_max,
        release_error_m=release_max, history_boundary=boundary, foot_slip_m2_s2=slip_rows,
        foot_slip_pairs=slip_pairs, root_path_length_m=root_path.sum(1), rotation_path_length_rad=rotation_path, keypose=km,
        sampled_vertices=len(ids), total_vertices=count, joints=joints)
```

## _rejections

来源：[hsi/stage3_joint_v2/refine.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/refine.py:244)，原文件第 244–266 行。

```python
def _rejections(candidate, current, initial, cfg):
    reasons = []
    if not math.isfinite(float(candidate['energy'])) or float(candidate['energy']) >= float(current['energy'])-cfg.energy_epsilon:
        reasons.append('energy_not_decreased')
    if bool((candidate['outside_mask'] & ~current['outside_mask']).any()):
        reasons.append('new_unknown_surface_points')
    for name in ('unknown_distance_m', 'penetration_rms_m', 'penetration_max_m', 'contact_error_m', 'release_error_m'):
        if bool((candidate[name] > current[name]+cfg.max_regression_m).any()):
            reasons.append(name+'_regressed')
    for name, tolerance in (('root_violation', cfg.max_regression_m), ('joint_violation', cfg.max_regression_rad)):
        limit = torch.minimum(current['keypose'][name], initial['keypose'][name])+tolerance
        if bool((candidate['keypose'][name] > limit).any()):
            reasons.append('keypose_'+name+'_regressed')
    if bool((candidate['root_path_length_m'] < initial['root_path_length_m']*cfg.min_path_length_ratio).any()):
        reasons.append('motion_path_collapsed')
    meaningful = initial['rotation_path_length_rad'] > 1e-4
    if bool((meaningful & (candidate['rotation_path_length_rad'] < initial['rotation_path_length_rad']*cfg.min_path_length_ratio)).any()):
        reasons.append('joint_motion_path_collapsed')
    physical_names = ('penetration_rms_m', 'contact_error_m', 'release_error_m', 'unknown_distance_m',
                      'history_boundary', 'foot_slip_m2_s2')
    if not any(bool((candidate[name] < current[name]-cfg.energy_epsilon).any()) for name in physical_names):
        reasons.append('no_geometry_or_continuation_improvement')
    return reasons
```

## refine_motion

来源：[hsi/stage3_joint_v2/refine.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint_v2/refine.py:300)，原文件第 300–396 行。

```python
def refine_motion(condition, motion: Tensor, schedule, *, body, scene,
                  config: RefineConfig | None = None, allowed_dofs: Tensor | None = None):
    """Return ``{motion, report}``; rejected updates leave exact input bits intact.

    Optional allowed_dofs is bool [B,T,135]; a rotation's six coordinates must
    be allowed together. It can narrow, never widen, upstream anchor freedoms.
    Padded frames remain bitwise unchanged. No GT timestamps or motion are read.
    Baseline violating locked anchor inputs are rejected, NOT teleported.
    """
    cfg, c = (config or RefineConfig()).validate(), condition
    _validate(c, motion, schedule, body, cfg)
    original = motion.detach().clone()
    valid = future_mask(c, motion.shape[1])
    free = free_dof_mask(original, schedule.keypose_times, c.keypose_mask, c.locked_root, c.locked_joints)
    free &= valid[..., None]
    if allowed_dofs is not None:
        if allowed_dofs.shape != free.shape or allowed_dofs.dtype != torch.bool or allowed_dofs.device != free.device:
            raise ValueError('allowed_dofs must be bool [B,T,135] on the motion device')
        rotation_allowed = allowed_dofs[..., 3:].reshape(*free.shape[:2], 22, 6)
        if bool((rotation_allowed.any(-1) != rotation_allowed.all(-1)).any()):
            raise ValueError('a local rotation must be allowed or forbidden as one six-coordinate unit')
        free &= allowed_dofs
    report = dict(schema='hsi.stage3_joint.refine.v2', config=asdict(cfg),
        training_policy_version=POLICY_VERSION,
        actual_mesh_body=_actual_smplx(body), schedule_changed=False, motion_release_authorized=False,
        input_mutated=False, body_shape_or_hands_optimized=False, target_furniture_removed=False,
        unknown_is_negative_supervision=False, collision_audit='all_actual_mesh_vertices_all_valid_future_frames',
        history_foot_slip_pair_used=False,
        history_foot_slip_limitation='Condition lacks initial support target position/normal; dense history continuity remains active.',
        contact_representation=getattr(body, 'region_reduction', 'test_fixture_regions'),
        limitations=['Vertex/frame checks do not certify triangle or between-frame nonpenetration.',
                     'Declared region centroids are not guaranteed anatomical contact patches.',
                     'No self-collision or motion-quality certification; failures remain in reports.'],
        accepted_iterations=0, trace=[])
    metric = keypose_metrics(original, c, schedule)
    if float(metric['max_locked_root_error']) > cfg.max_regression_m or float(metric['max_locked_joint_error']) > cfg.max_regression_rad:
        report.update(status='rejected_input_locked_anchor_violation',
                      locked_root_error_m=float(metric['max_locked_root_error']),
                      locked_joint_error_rad=float(metric['max_locked_joint_error']))
        return {'motion': original, 'report': report}
    with torch.no_grad():
        history_geometry = _mesh(body, _history_tail(c), cfg.mesh_chunk_frames)
        reference = {'motion': original, 'joints': _mesh(body, original, cfg.mesh_chunk_frames)[1]}
        initial = _evaluate(original, c, schedule, body, scene, cfg, reference, history_geometry, audit=True)
    current, current_audit = original.clone(), initial
    report['before'] = _summary(initial)
    for iteration in range(cfg.iterations):
        with torch.enable_grad():
            variable = current.detach().requires_grad_(True)
            value = _evaluate(variable, c, schedule, body, scene, cfg, reference, history_geometry)
            gradient, = torch.autograd.grad(value['energy'], variable, allow_unused=False)
        if not bool(torch.isfinite(gradient).all()):
            report['trace'].append(dict(iteration=iteration, accepted=False, reasons=['nonfinite_gradient']))
            break
        step = -gradient*free*cfg.learning_rate
        root = step[..., :3]
        root *= (cfg.max_root_step_m/root.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1.)
        rotation = step[..., 3:].reshape(*motion.shape[:2], 22, 6)
        rotation *= (cfg.max_rotation_step_6d/rotation.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1.)
        accepted = False
        for backtrack in range(cfg.backtracking_steps):
            with torch.no_grad():
                proposal = current+step*(.5**backtrack)
                rots = proposal[..., 3:].reshape(*motion.shape[:2], 22, 6)
                # Only changed/free rotations are retracted; locked/raw/padded
                # coordinates retain their exact original representation.
                changed = free[..., 3:].reshape(*free.shape[:2], 22, 6).all(-1)
                canonical = legalize_rotation_6d(rots)
                proposal[..., 3:] = torch.where(changed[..., None], canonical, current[..., 3:].reshape_as(rots)).flatten(-2)
                proposal = torch.where(free, proposal, original)
                root_delta = (proposal[..., :3]-original[..., :3]).norm(dim=-1)
                joint_delta = rotation_geodesic(proposal[..., 3:].reshape(*motion.shape[:2], 22, 6),
                                               original[..., 3:].reshape(*motion.shape[:2], 22, 6))
                reasons = []
                if bool((root_delta[valid] > cfg.max_total_root_change_m).any()):
                    reasons.append('root_trust_region_exceeded')
                if bool((joint_delta[valid] > cfg.max_total_joint_change_rad).any()):
                    reasons.append('joint_trust_region_exceeded')
                candidate = _evaluate(proposal, c, schedule, body, scene, cfg, reference, history_geometry, audit=True)
                reasons.extend(_rejections(candidate, current_audit, initial, cfg))
            report['trace'].append(dict(iteration=iteration, backtrack=backtrack, accepted=not reasons,
                reasons=reasons, energy=float(candidate['energy']),
                gradient_surface_vertices=value['sampled_vertices'], audit_surface_vertices=candidate['sampled_vertices']))
            if not reasons:
                current, current_audit = proposal, candidate
                report['accepted_iterations'] += 1
                accepted = True
                break
        if not accepted:
            break
    report['after'] = _summary(current_audit)
    report['status'] = 'refined_diagnostic' if report['accepted_iterations'] else 'unchanged_no_accepted_update'
    report['locked_and_disallowed_coordinates_bitwise_unchanged'] = bool(torch.equal(current[~free], original[~free]))
    report['root_max_change_m'] = float((current[..., :3]-original[..., :3]).norm(dim=-1)[valid].max())
    report['joint_max_change_rad'] = float(rotation_geodesic(current[..., 3:].reshape(*motion.shape[:2], 22, 6),
        original[..., 3:].reshape(*motion.shape[:2], 22, 6))[valid].max())
    return {'motion': current.detach(), 'report': report}
```

## extract_keyposes

来源：[hsi/stage3_sequence/constraints.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/constraints.py:152)，原文件第 152–157 行。

```python
def extract_keyposes(motion: Tensor, keypose_frames: Tensor, keypose_mask: Tensor) -> Tensor:
    """Gather one-based target frames, zeroing inactive padded slots."""
    batch, _, count = _layout(motion, keypose_frames, keypose_mask)
    indices = (keypose_frames - 1).clamp_min(0).unsqueeze(-1).expand(batch, count, MOTION_DIM)
    gathered = motion.gather(1, indices)
    return torch.where(keypose_mask[..., None], gathered, torch.zeros_like(gathered))
```
