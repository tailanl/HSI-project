"""One event/contact interpretation shared by training, sampling and refinement.

An ordered future contact is not evidence that a foot is already planted.
Establish requires contact at its terminal frame, hold throughout its interval,
and release requires outward separation only at its terminal frame. Current
geometric establishment is separately reported; it never replaces a mandatory
relation or a missing ground-truth contact label.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from hsi.stage3_sequence.contracts import expand_contacts, future_mask

POLICY_VERSION = 'event_end_establish_geometric_stance_v2'


def _tolerance(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name+' must be finite and nonnegative')


def _mean(values, mask):
    mask = torch.broadcast_to(mask, values.shape).to(values)
    return (values*mask).sum()/mask.sum().clamp_min(1)


def event_contact_state(condition, schedule, regions: Tensor | None = None, *,
                        contact_tolerance: float = .02, length: int | None = None):
    """Return immutable condition fields plus explicit [B,T,8] semantic masks.

    ``declared_contact``: the slot's ordered relation, NOT current support.
    ``required_contact`` / ``contact_active``: mandatory now (end/hold policy).
    ``release_required`` / ``release_active``: mandatory separation now.
    ``established``: declared relation geometrically within tolerance NOW.
    ``foot_support``: established, non-release feet only; not all declarations.

    An integer predicted timetable is required, never rounded silently. The
    body region ordering is the existing eight-region contract. All returned
    booleans are evidence/requirements, not interchangeable pseudo-GT labels.
    """
    _tolerance(contact_tolerance, 'contact_tolerance')
    c = condition
    length = c.max_frames if length is None else length
    if not isinstance(length, int) or length < c.max_frames:
        raise ValueError('Contact buffer must cover the complete future horizon')
    boundaries = schedule.boundaries
    if (boundaries.shape != c.slot_mask.shape or boundaries.device != c.device
            or boundaries.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Contact semantics require explicit integer event boundaries')
    if (not torch.equal(schedule.slot_mask, c.slot_mask)
            or not torch.equal(schedule.total_frames, c.total_frames)):
        raise ValueError('Schedule and contact condition disagree')
    fields = dict(expand_contacts(c, schedule, length))
    frame = torch.arange(1, length+1, device=c.device)[None]
    rows = torch.arange(c.batch_size, device=c.device)[:, None]
    end = frame == boundaries[rows, fields['slot_indices']]
    kind = fields['slot_types']
    declared = fields['contact_active']
    release_declared = fields['release_active']
    # Explicit retained relations (including carrying something during travel
    # or keeping feet planted while the seat releases) stay mandatory. Owner
    # keypose contacts alone never invent a slot declaration.
    required = declared & ((kind != 1) | end)[..., None]
    release_required = release_declared & ((kind == 3) & end)[..., None]
    fields.update(declared_contact=declared, declared_release=release_declared,
        required_contact=required, contact_active=required,
        release_required=release_required, release_active=release_required,
        event_end=end & fields['frame_mask'])
    if regions is None:
        # Missing geometry must never masquerade as observed no-contact.
        fields.update(established=None, foot_support=None, resolved_targets=None,
                      target_distance=None, establishment_geometry_known=False)
        return fields
    if (regions.shape != (c.batch_size, length, 8, 3) or regions.device != c.device
            or not regions.is_floating_point() or not bool(torch.isfinite(regions).all())):
        raise ValueError('Finite current body regions [B,T,8,3] required')
    body_ids = fields['target_body_regions']
    self_targets = regions.gather(2, body_ids.clamp_min(0)[..., None].expand_as(regions))
    target = torch.where((body_ids >= 0)[..., None], self_targets, fields['targets'])
    distance = (regions-target).norm(dim=-1)
    established = ((declared | release_declared) & (distance <= contact_tolerance)
                   & fields['frame_mask'][..., None])
    feet = torch.zeros_like(established); feet[..., 3:5] = True
    # Only the released relation may depart: releasing a seat does not erase
    # separately declared, retained support of the two feet.
    foot_support = established & feet & ~release_declared
    fields.update(established=established, foot_support=foot_support,
        resolved_targets=target, target_distance=distance, establishment_geometry_known=True)
    return fields


def contact_event_losses(regions, condition, schedule, *, contact_tolerance=.02,
                         release_margin=.05, state=None):
    """Per-event normalized contact/release losses using the shared masks."""
    _tolerance(release_margin, 'release_margin')
    f = state if state is not None else event_contact_state(condition, schedule, regions,
        contact_tolerance=contact_tolerance, length=regions.shape[1])
    if not f.get('establishment_geometry_known', False):
        raise ValueError('Contact losses require current resolved body geometry')
    c = condition
    delta = regions-f['resolved_targets']
    error = torch.relu(f['target_distance']-contact_tolerance)
    normals = F.normalize(f['normals'], dim=-1, eps=1e-6)
    normal_known = f['normals'].norm(dim=-1) > 1e-8
    self_contact = f['target_body_regions'] >= 0
    if bool((f['release_required'] & ~self_contact & ~normal_known).any()):
        raise ValueError('Static-surface release requires an explicit outward normal')
    gap = torch.where(self_contact, f['target_distance'], (delta*normals).sum(-1))
    release_error = torch.relu(release_margin-gap)
    contact_terms, release_terms = [], []
    contact_max = regions.new_zeros(c.batch_size, c.slot_mask.shape[1])
    release_max = torch.zeros_like(contact_max)
    for row in range(c.batch_size):
        for slot in range(c.slot_mask.shape[1]):
            membership = (f['slot_indices'][row] == slot)[:, None]
            active = f['required_contact'][row] & membership
            released = f['release_required'][row] & membership
            if bool(active.any()):
                contact_terms.append(_mean(error[row].square(), active))
                contact_max[row, slot] = error[row][active].max()
            if bool(released.any()):
                release_terms.append(_mean(release_error[row].square(), released))
                release_max[row, slot] = release_error[row][released].max()
    zero = regions.sum()*0
    return dict(contact=torch.stack(contact_terms).mean() if contact_terms else zero,
        release=torch.stack(release_terms).mean() if release_terms else zero,
        contact_error_m=contact_max, release_error_m=release_max,
        resolved_targets=f['resolved_targets'], contact_state=f)


def foot_support_pairs(regions, condition, schedule, *, history_regions=None,
                       history_targets=None, history_normals=None,
                       contact_tolerance=.02, same_target_tolerance=.02,
                       normal_cosine_min=.99, state=None):
    """Support pairs for each future frame (first pair optional real history).

    Both frames must be geometrically established on the same declared target.
    Fixed-world targets require matching target identity AND location; two
    different steps on floor ID 0 are not the same planted foot. A newly landed
    foot has no pair with its distant approach frame. Releases remain free.
    """
    _tolerance(same_target_tolerance, 'same_target_tolerance')
    if not isinstance(normal_cosine_min, (int, float)) or not 0 <= normal_cosine_min <= 1:
        raise ValueError('Normal agreement cosine must be in [0,1]')
    c = condition
    f = state if state is not None else event_contact_state(c, schedule, regions,
        contact_tolerance=contact_tolerance, length=regions.shape[1])
    if f['foot_support'] is None:
        raise ValueError('Foot support requires current body geometry')
    planted = f['foot_support'][..., 3:5]
    feet = regions[..., 3:5, :]
    target = f['resolved_targets'][..., 3:5, :]
    ids = f['target_ids'][..., 3:5]
    body_ids = f['target_body_regions'][..., 3:5]
    previous_feet = torch.cat((feet[:, :1], feet[:, :-1]), 1)
    previous_target = torch.cat((target[:, :1], target[:, :-1]), 1)
    previous_planted = torch.cat((torch.zeros_like(planted[:, :1]), planted[:, :-1]), 1)
    previous_ids = torch.cat((ids[:, :1], ids[:, :-1]), 1)
    previous_body_ids = torch.cat((body_ids[:, :1], body_ids[:, :-1]), 1)
    raw_normals = f['normals'][..., 3:5, :]
    normals = F.normalize(raw_normals, dim=-1, eps=1e-6)
    previous_normals = torch.cat((normals[:, :1], normals[:, :-1]), 1)
    normal_known = raw_normals.norm(dim=-1) > 1e-8
    previous_normal_known = torch.cat((normal_known[:, :1], normal_known[:, :-1]), 1)
    if (history_targets is None) != (history_normals is None):
        raise ValueError('History support targets and normals must be supplied together')
    history_pair_geometry_known = history_regions is not None and history_targets is not None
    if history_regions is not None:
        if (history_regions.ndim != 4 or history_regions.shape[0] != c.batch_size
                or history_regions.shape[2:] != (8, 3) or history_regions.shape[1] < 1
                or history_regions.device != regions.device or not bool(torch.isfinite(history_regions).all())):
            raise ValueError('Finite actual history regions [B,H,8,3] required')
        history_last = history_regions[:, -1:]
        history_self = history_last.gather(2, body_ids[:, :1].clamp_min(0)[..., None].expand(-1,-1,-1,3))
        history_target = torch.where((body_ids[:, :1] >= 0)[..., None], history_self, target[:, :1])
        if history_pair_geometry_known:
            for value in (history_targets, history_normals):
                if (value.shape != (c.batch_size, 8, 3) or value.device != regions.device
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError('Explicit history support targets/normals must be finite [B,8,3]')
            history_target = torch.where((body_ids[:, :1] >= 0)[..., None], history_self,
                                         history_targets[:, None, 3:5])
            previous_feet = torch.cat((history_last[..., 3:5, :], feet[:, :-1]), 1)
            previous_target = torch.cat((history_target, target[:, :-1]), 1)
            history_close = (previous_feet[:, :1]-history_target).norm(dim=-1) <= contact_tolerance
            history_active = c.initial_contact_active[:, None, 3:5] & history_close
            previous_planted = torch.cat((history_active, planted[:, :-1]), 1)
            previous_ids = torch.cat((c.initial_contact_target_ids[:, None, 3:5], ids[:, :-1]), 1)
            history_normal = history_normals[:, None, 3:5]
            previous_normals = torch.cat((F.normalize(history_normal, dim=-1, eps=1e-6), normals[:, :-1]), 1)
            previous_normal_known = torch.cat((history_normal.norm(dim=-1) > 1e-8, normal_known[:, :-1]), 1)
    same_kind = body_ids == previous_body_ids
    same_identity = (ids == previous_ids) | ((body_ids >= 0) & same_kind)
    same_location = ((target-previous_target).norm(dim=-1) <= same_target_tolerance) | (body_ids >= 0)
    same_normal = (normal_known & previous_normal_known
                   & ((normals*previous_normals).sum(-1) >= normal_cosine_min)) | (body_ids >= 0)
    pair = planted & previous_planted & same_identity & same_kind & same_location & same_normal
    pair &= f['frame_mask'][..., None]
    # A static scene support has zero velocity even if a nearest-point query
    # changes. Only an actual body-region target supplies support motion.
    support_delta = torch.where((body_ids >= 0)[..., None], target-previous_target, torch.zeros_like(target))
    velocity = ((feet-previous_feet)-support_delta)*c.fps
    tangent = velocity-(velocity*normals).sum(-1, keepdim=True)*normals
    return dict(mask=pair, relative_tangential_velocity=tangent, contact_state=f,
                history_pair_geometry_known=history_pair_geometry_known)


def foot_slip_terms(regions, condition, schedule, *, history_regions=None,
                    history_targets=None, history_normals=None, contact_tolerance=.02, state=None):
    result = foot_support_pairs(regions, condition, schedule, history_regions=history_regions,
                               history_targets=history_targets, history_normals=history_normals,
                               contact_tolerance=contact_tolerance, state=state)
    values = result['relative_tangential_velocity'].square().sum(-1)
    mask = result['mask']
    per_row = (values*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    return dict(loss=_mean(values, mask), per_row=per_row, pair_mask=mask,
                pair_count=mask.sum(), history_pair_geometry_known=result['history_pair_geometry_known'])


def foot_slip_loss(regions, condition, schedule, **kwargs):
    return foot_slip_terms(regions, condition, schedule, **kwargs)['loss']
