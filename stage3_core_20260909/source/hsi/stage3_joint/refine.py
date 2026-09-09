"""U15: bounded whole-sequence mesh/SDF refinement after learned sampling.

The supplied integer event schedule, body shape/hands, and locked anchor DOFs
are immutable. This is optimization of actual mesh geometry, not temporal
filtering, retiming, interpolation, or a replacement motion generator. A lower
energy is NOT a quality certificate: reports retain physical failures, unknown
space, rejection reasons, and the limits of vertex/frame sampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor

from hsi.stage3_sequence.contracts import expand_contacts, future_mask
from hsi.stage3_sequence.constraints import (
    free_dof_mask, legalize_rotation_6d, rotation_6d_valid_mask, rotation_geodesic,
)
from hsi.stage3_sequence.mesh_body import MeshBody, MeshOutput
from hsi.stage3_sequence.objectives import keypose_metrics, keypose_loss


@dataclass(frozen=True)
class RefineConfig:
    iterations: int = 8
    backtracking_steps: int = 5
    learning_rate: float = 1.
    max_root_step_m: float = .015
    max_rotation_step_6d: float = .025
    max_total_root_change_m: float = .15
    max_total_joint_change_rad: float = .25
    max_surface_points: int = 2048
    mesh_chunk_frames: int = 24
    contact_tolerance_m: float = .02
    release_margin_m: float = .05
    collision: float = 20.
    contact: float = 10.
    release: float = 5.
    outside: float = 20.
    foot_slip: float = .05
    history_boundary: float = .1
    velocity_preservation: float = .02
    acceleration_preservation: float = .01
    pose_preservation: float = .1
    keypose: float = 10.
    acceleration_scale_m_s2: float = 10.
    max_regression_m: float = 1e-5
    max_regression_rad: float = 1e-5
    energy_epsilon: float = 1e-10
    min_path_length_ratio: float = .8
    allow_test_body: bool = False

    def validate(self):
        for name in ('iterations', 'backtracking_steps', 'max_surface_points', 'mesh_chunk_frames'):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name == 'iterations' else 1):
                raise ValueError(name+' must be an integer in its valid range')
        for name, value in asdict(self).items():
            if name in ('iterations', 'backtracking_steps', 'max_surface_points', 'mesh_chunk_frames', 'allow_test_body'):
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(name+' must be finite and nonnegative')
        if self.acceleration_scale_m_s2 <= 0 or self.learning_rate <= 0:
            raise ValueError('positive acceleration scale and learning rate required')
        if not 0 <= self.min_path_length_ratio <= 1 or type(self.allow_test_body) is not bool:
            raise ValueError('invalid path ratio or test-body flag')
        return self


def _mean(value, mask):
    """Zero-safe masking BEFORE arithmetic; unknown values are not labels."""
    mask = torch.broadcast_to(mask, value.shape)
    return torch.where(mask, value, torch.zeros_like(value)).sum()/mask.sum().clamp_min(1)


def _row_mean(value, mask):
    mask = torch.broadcast_to(mask, value.shape)
    return torch.where(mask, value, torch.zeros_like(value)).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)


def _row_max(value, mask):
    mask = torch.broadcast_to(mask, value.shape)
    return torch.where(mask, value, torch.zeros_like(value)).flatten(1).amax(1)


def _history_tail(c):
    """Two valid history frames; single-frame history has zero observed velocity."""
    last = c.history_mask.sum(1)-1
    row = torch.arange(c.batch_size, device=c.history.device)
    return torch.stack((c.history[row, (last-1).clamp_min(0)], c.history[row, last]), 1).detach()


def _actual_smplx(body):
    return isinstance(body, MeshBody) and 'SMPLXLayer' in {kind.__name__ for kind in type(body.model).__mro__}


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


def _mesh(body, motion, chunk_frames=24):
    mesh = mesh_in_chunks(body, motion, chunk_frames)
    return mesh.vertices, mesh.joints, body.region_points_from_vertices(mesh.vertices)


def _resolved(regions, fields):
    body_ids = fields['target_body_regions']
    self_targets = regions.gather(2, body_ids.clamp_min(0)[..., None].expand_as(regions))
    return torch.where((body_ids >= 0)[..., None], self_targets, fields['targets'])


def _event_terms(regions, fields, c, schedule, cfg):
    """Establish at event end; hold throughout; release at end along normal."""
    target = _resolved(regions, fields)
    displacement = regions-target
    distance = displacement.norm(dim=-1)
    contact_terms, release_terms = [], []
    contact_max, release_max = regions.new_zeros(c.batch_size, c.slot_mask.shape[1]), regions.new_zeros(c.batch_size, c.slot_mask.shape[1])
    for row in range(c.batch_size):
        for slot in range(c.slot_mask.shape[1]):
            if not bool(c.slot_mask[row, slot]):
                continue
            end = int(schedule.boundaries[row, slot])-1
            active = fields['contact_active'][row] & (fields['slot_indices'][row] == slot)[:, None]
            # Unlike a static hold, an establishment interval may legitimately
            # start away from the target. Only its terminal frame is mandatory.
            if int(c.slot_types[row, slot]) == 1:
                end_only = torch.arange(regions.shape[1], device=regions.device) == end
                active = active & end_only[:, None]
            error = torch.relu(distance[row]-cfg.contact_tolerance_m)
            if bool(active.any()):
                contact_terms.append(_mean(error.square(), active))
                contact_max[row, slot] = error[active].max()
            released = fields['release_active'][row, end]
            if bool(released.any()):
                outward = (displacement[row, end]*fields['normals'][row, end]).sum(-1)
                gap = torch.where(fields['target_body_regions'][row, end] >= 0, distance[row, end], outward)
                error = torch.relu(cfg.release_margin_m-gap)
                release_terms.append(_mean(error.square(), released))
                release_max[row, slot] = error[released].max()
    zero = regions.sum()*0
    return (torch.stack(contact_terms).mean() if contact_terms else zero,
            torch.stack(release_terms).mean() if release_terms else zero,
            contact_max, release_max, target)


def _foot_slip(regions, history_regions, fields, c, target, cfg):
    """Same physical support across slot boundaries is still a planted foot.

    History-to-future pairs require explicit initial contact target identity.
    Establishment counts as planted only once the foot reaches its target.
    Self-contact velocity is relative to its moving body-region target.
    """
    feet, normals = regions[..., 3:5, :], fields['normals'][..., 3:5, :]
    targets = target[..., 3:5, :]
    ids = fields['target_ids'][..., 3:5]
    body_ids = fields['target_body_regions'][..., 3:5]
    established = (feet-targets).norm(dim=-1) <= cfg.contact_tolerance_m
    holding = fields['slot_types'][..., None] == 2
    planted = fields['contact_active'][..., 3:5] & (holding | established)
    previous_feet = torch.cat((history_regions[:, -1:, 3:5], feet[:, :-1]), 1)
    previous_planted = torch.cat((c.initial_contact_active[:, None, 3:5], planted[:, :-1]), 1)
    previous_ids = torch.cat((c.initial_contact_target_ids[:, None, 3:5], ids[:, :-1]), 1)
    previous_body_ids = torch.cat((body_ids[:, :1], body_ids[:, :-1]), 1)
    history_target = history_regions[:, -1:].gather(2, body_ids[:, :1].clamp_min(0)[..., None].expand(-1, -1, -1, 3))
    first_target = torch.where((body_ids[:, :1] >= 0)[..., None], history_target, targets[:, :1])
    previous_targets = torch.cat((first_target, targets[:, :-1]), 1)
    pair = planted & previous_planted & (ids == previous_ids) & (body_ids == previous_body_ids)
    pair &= fields['frame_mask'][..., None]
    velocity = ((feet-targets)-(previous_feet-previous_targets))*c.fps
    tangent = velocity-(velocity*normals).sum(-1, keepdim=True)*normals
    return _mean(tangent.square().sum(-1), pair), pair.sum(), _row_mean(tangent.square().sum(-1), pair)


def _temporal(joints, reference_joints, history_joints, frame_mask, fps, cfg):
    full = torch.cat((history_joints, joints), 1)
    reference = torch.cat((history_joints, reference_joints), 1)
    velocity, ref_velocity = torch.diff(full, dim=1)*fps, torch.diff(reference, dim=1)*fps
    acceleration, ref_acceleration = torch.diff(velocity, dim=1)*fps, torch.diff(ref_velocity, dim=1)*fps
    mask = frame_mask[..., None, None]
    velocity_loss = _mean((velocity[:, 1:]-ref_velocity[:, 1:]).square(), mask)
    acceleration_loss = _mean(((acceleration-ref_acceleration)/cfg.acceleration_scale_m_s2).square(), mask)
    # Explicit continuation objective, not an all-sequence zero-velocity prior.
    boundary = (velocity[:, 1]-velocity[:, 0]).square().mean((-1, -2))
    boundary += (acceleration[:, 0]/cfg.acceleration_scale_m_s2).square().mean((-1, -2))
    return velocity_loss, acceleration_loss, boundary


def _evaluate(motion, c, schedule, body, scene, cfg, reference, history_geometry, *, audit=False):
    vertices, joints, regions = _mesh(body, motion, cfg.mesh_chunk_frames)
    fields = expand_contacts(c, schedule, motion.shape[1])
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
    slip, slip_pairs, slip_rows = _foot_slip(regions, history_geometry[2], fields, c, targets, cfg)
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


def _summary(values):
    names = ('energy', 'outside_rate', 'unknown_distance_m', 'penetration_rms_m', 'penetration_max_m',
        'contact_error_m', 'release_error_m', 'history_boundary', 'foot_slip_m2_s2',
        'foot_slip_pairs', 'root_path_length_m', 'rotation_path_length_rad', 'sampled_vertices', 'total_vertices')
    def plain(value):
        return value.detach().cpu().tolist() if isinstance(value, Tensor) else value
    result = {name: plain(values[name]) for name in names}
    result['components'] = {key: plain(value) for key, value in values['components'].items()}
    result['keypose'] = {key: plain(values['keypose'][key]) for key in (
        'max_root_violation', 'max_joint_violation', 'max_locked_root_error', 'max_locked_joint_error',
        'rotation_invalid_count', 'tolerances_satisfied')}
    return result


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


def _validate(c, motion, schedule, body, cfg):
    c.validate()
    cfg.validate()
    if not _actual_smplx(body) and not cfg.allow_test_body:
        raise TypeError('actual MeshBody required; allow_test_body is CPU fixture diagnostics only')
    if not callable(getattr(body, 'mesh', None)) or not callable(getattr(body, 'region_points_from_vertices', None)):
        raise TypeError('mesh-backed body with explicit named surface regions required')
    if motion.ndim != 3 or motion.shape != (c.batch_size, int(c.total_frames.max()), 135):
        raise ValueError('motion must be [B,max(total_frames),135]')
    if motion.dtype != c.history.dtype or motion.device != c.history.device or not bool(torch.isfinite(motion).all()):
        raise ValueError('motion must be finite, metric, and on condition device/dtype')
    valid = future_mask(c, motion.shape[1])
    if not bool(rotation_6d_valid_mask(motion[..., 3:].reshape(*motion.shape[:2], 22, 6))[valid].all()):
        raise ValueError('input active-frame rotation6D is degenerate')
    for key, value in (('slot_mask', c.slot_mask), ('keypose_mask', c.keypose_mask),
                       ('total_frames', c.total_frames), ('keypose_slots', c.keypose_slots)):
        if not torch.equal(getattr(schedule, key), value):
            raise ValueError('schedule/condition mismatch: '+key)
    for name in ('durations', 'boundaries', 'keypose_times'):
        value = getattr(schedule, name)
        if value.is_floating_point() or value.device != motion.device:
            raise ValueError('post-refinement requires fixed integer schedule on motion device')
    if not torch.equal(schedule.durations.cumsum(1).masked_fill(~c.slot_mask, 0), schedule.boundaries):
        raise ValueError('schedule boundaries are inconsistent')
    if not torch.equal(schedule.durations.sum(1), c.total_frames) or bool((schedule.durations[c.slot_mask] < c.minimum_frames[c.slot_mask]).any()):
        raise ValueError('schedule does not preserve required event minima/horizon')
    expected = schedule.boundaries.gather(1, c.keypose_slots.clamp_min(0)).masked_fill(~c.keypose_mask, 0)
    if not torch.equal(expected, schedule.keypose_times):
        raise ValueError('schedule anchor times do not match its events')


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
    report = dict(schema='hsi.stage3_joint.refine.v1', config=asdict(cfg),
        actual_mesh_body=_actual_smplx(body), schedule_changed=False, motion_release_authorized=False,
        input_mutated=False, body_shape_or_hands_optimized=False, target_furniture_removed=False,
        unknown_is_negative_supervision=False, collision_audit='all_actual_mesh_vertices_all_valid_future_frames',
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
