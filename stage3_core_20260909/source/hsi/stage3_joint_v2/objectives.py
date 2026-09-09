"""V2 contact semantics; same geometry/objective interfaces as frozen Stage 3.

Relation tensor layout stays [B,T,8,14]. Channel 8 explicitly means current
geometrically established contact, NOT future intent or mandatory contact.
Mandatory contact is independently exposed in ``contact_state`` and losses;
the condition's keypose contacts retain the original requested semantics.
Collision/keypose algorithms are not changed in this contact-only revision.
"""
from __future__ import annotations

import torch

from hsi.stage3_sequence.geometry import relation_features
from hsi.stage3_sequence.objectives import (
    masked_mean, projected_motion, keypose_loss, keypose_metrics, _MotionGeometrySnapshot,
)
from .contacts import POLICY_VERSION, event_contact_state, contact_event_losses, foot_slip_terms

RELATION_DIM = 14
RELATION_ESTABLISHED_INDEX = 8


def geometry_relations(motion, c, schedule, body, scene, *, contact_tolerance=.02):
    regions = body.region_points(motion)
    state = event_contact_state(c, schedule, regions, contact_tolerance=contact_tolerance,
                                length=motion.shape[1])
    # Targets remain visible before contact as geometric/semantic context;
    # channel 8 must not tell the network that an approaching foot is planted.
    relation = relation_features(motion, body, scene, state['targets'], state['normals'],
        state['established'], state['target_body_regions'], fps=c.fps)
    return relation.masked_fill(~state['frame_mask'][..., None, None], 0)


def physical_objectives(motion, c, schedule, body, scene, *, contact_tolerance=.02, release_margin=.05):
    cache = _MotionGeometrySnapshot(motion, body)
    regions = cache.region_points(motion)
    state = event_contact_state(c, schedule, regions, contact_tolerance=contact_tolerance,
                                length=motion.shape[1])
    events = contact_event_losses(regions, c, schedule, contact_tolerance=contact_tolerance,
                                  release_margin=release_margin, state=state)
    slip = foot_slip_terms(regions, c, schedule, contact_tolerance=contact_tolerance, state=state)
    relation = relation_features(motion, cache, scene, state['targets'], state['normals'],
        state['established'], state['target_body_regions'], fps=c.fps)
    relation = relation.masked_fill(~state['frame_mask'][..., None, None], 0)
    sdf = scene.query(cache.collision_points(motion))
    valid = state['frame_mask'][..., None]
    # Preserve previous all-frame collision/outside policy here. This patch
    # changes contact event semantics, not SDF values or collision thresholds.
    collision = masked_mean(torch.relu(-sdf.signed_distance_m).square(), valid)
    outside_rate = masked_mean(sdf.outside.to(motion.dtype), valid)
    return dict(collision=collision, contact=events['contact'], release=events['release'],
        foot_slip=slip['loss'], outside_rate=outside_rate, relations=relation,
        contact_state=state, contact_error_m=events['contact_error_m'],
        release_error_m=events['release_error_m'], foot_slip_per_row=slip['per_row'],
        foot_slip_pair_count=slip['pair_count'], policy_version=POLICY_VERSION)
