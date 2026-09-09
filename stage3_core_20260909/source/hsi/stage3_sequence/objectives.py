"""Shared training/sampling objectives; no objective authorizes motion release."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .contracts import JOINTS, expand_contacts, future_mask
from .constraints import (
    legalize_rotation_6d, project_keyposes, constraint_loss, constraint_metrics,
    rotation_6d_valid_mask,
)
from .geometry import relation_features, foot_slip_energy


@dataclass(frozen=True)
class LossWeights:
    denoise: float = 1.
    timing: float = 1.
    keypose: float = .1
    contact: float = .05
    collision: float = .05
    velocity: float = .01
    foot_slip: float = .001
    release: float = .01
    contact_classification: float = .01

    def validate(self):
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("loss weights must be finite and nonnegative")
        return self


def masked_mean(values, mask):
    mask = torch.broadcast_to(mask,values.shape).to(values)
    return (values*mask).sum()/mask.sum().clamp_min(1)


def projected_motion(raw, condition, schedule):
    """Record raw degeneracy before legalizing, then enforce only locked DOFs."""
    rotations = raw[...,3:].reshape(*raw.shape[:2],JOINTS,6)
    valid = rotation_6d_valid_mask(rotations)
    clean = torch.cat((raw[...,:3],legalize_rotation_6d(rotations).flatten(-2)),dim=-1)
    projected = project_keyposes(clean,condition.keyposes,schedule.keypose_times.long(),
                                  condition.keypose_mask,condition.locked_root,condition.locked_joints)
    return projected.masked_fill(~future_mask(condition,raw.shape[1])[...,None],0),valid


def keypose_loss(motion, c, schedule):
    return constraint_loss(motion,c.keyposes,schedule.keypose_times.long(),c.keypose_mask,
                           c.locked_root,c.locked_joints,c.root_tolerance,c.joint_tolerance)


def keypose_metrics(motion,c,schedule):
    return constraint_metrics(motion,c.keyposes,schedule.keypose_times.long(),c.keypose_mask,
                              c.locked_root,c.locked_joints,c.root_tolerance,c.joint_tolerance)


def geometry_relations(motion,c,schedule,body,scene):
    fields = expand_contacts(c,schedule,motion.shape[1])
    rel = relation_features(motion,body,scene,fields["targets"],fields["normals"],
                            fields["contact_active"],fields["target_body_regions"],fps=c.fps)
    return rel.masked_fill(~fields["frame_mask"][...,None,None],0)


class _MotionGeometrySnapshot:
    """Reuse one candidate's skinning graph inside one objective evaluation.

    Never stored on the model/body, detached, or reused across guidance steps.
    The identity check prevents accidentally querying a different candidate.
    """
    def __init__(self, motion, body):
        self.motion = motion
        mesh_api = ("mesh", "region_points_from_vertices", "collision_points_from_vertices")
        if all(callable(getattr(body, name, None)) for name in mesh_api):
            vertices = body.mesh(motion).vertices
            self.regions = body.region_points_from_vertices(vertices)
            self.collisions = body.collision_points_from_vertices(vertices)
        else:
            self.regions = body.region_points(motion)
            self.collisions = body.collision_points(motion)

    def region_points(self, motion):
        if motion is not self.motion:
            raise ValueError("geometry snapshot belongs to a different motion candidate")
        return self.regions

    def collision_points(self, motion):
        if motion is not self.motion:
            raise ValueError("geometry snapshot belongs to a different motion candidate")
        return self.collisions


def physical_objectives(motion,c,schedule,body,scene,*,contact_tolerance=.02,release_margin=.05):
    """Evaluate ALL valid frames, with separately normalized contact events.

    Geometry depends on the supplied body adapter: FK is a proxy diagnostic,
    whereas a mesh-backed adapter may supply all actual vertices. Neither this
    function nor learned contact logits may remove target furniture from SDF.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (contact_tolerance, release_margin)):
        raise ValueError("contact/release distances must be finite and nonnegative")
    body = _MotionGeometrySnapshot(motion, body)
    fields = expand_contacts(c,schedule,motion.shape[1])
    valid = fields["frame_mask"]
    rel = geometry_relations(motion,c,schedule,body,scene)
    points = body.collision_points(motion)
    sdf = scene.query(points)
    collision = masked_mean(torch.relu(-sdf.signed_distance_m).square(),valid[...,None])
    outside_rate = masked_mean(sdf.outside.to(motion.dtype),valid[...,None])
    # Each required event is counted once, not weighted by its predicted length.
    contact_values, release_values = [],[]
    for slot in range(c.slot_mask.shape[1]):
        active = fields["contact_active"] & (fields["slot_indices"] == slot)[...,None]
        released = fields["release_active"] & (fields["slot_indices"] == slot)[...,None]
        for row in range(c.batch_size):
            if bool(active[row].any()):
                contact_values.append(masked_mean(
                    torch.relu(rel[row,...,1]-contact_tolerance).square(),active[row]))
            if bool(released[row].any()):
                # Separation is required at the END of release, not at its
                # first frame where support may legitimately still be present.
                end = int(schedule.boundaries[row,slot].item())-1
                mask = released[row,end]
                # For static furniture, require separation along its outward
                # surface normal: sliding sideways on a seat is not release.
                # Self-contact has no static world normal; use region distance.
                distance = rel[row,end,:,1]
                # Use the signed displacement directly, not norm*unit-vector:
                # the latter loses its outward gradient at exact coincidence.
                displacement = body.region_points(motion)[row,end]-fields["targets"][row,end]
                outward_gap = (displacement*fields["normals"][row,end]).sum(-1)
                gap = torch.where(fields["target_body_regions"][row,end] >= 0,
                                  distance, outward_gap)
                release_values.append(masked_mean(
                    torch.relu(release_margin-gap).square(),mask))
    zero = motion.sum()*0
    contact = torch.stack(contact_values).mean() if contact_values else zero
    release = torch.stack(release_values).mean() if release_values else zero
    event_ids = fields["slot_indices"][...,None].expand_as(fields["contact_active"])
    slip = foot_slip_energy(motion,body,fields["contact_active"],fields["normals"],
                            fps=c.fps,event_ids=event_ids)
    return {"collision":collision,"contact":contact,"release":release,"foot_slip":slip,
            "outside_rate":outside_rate,"relations":rel}
