"""Experimental geometry guidance, not a physical controller.

Inside DiP: differentiable native-RIC joints/bone samples (NOT full mesh).
After DiP: callback and metrics on the actual, fixed-shape SMPL-X surface.
The target furniture is always retained. Contact is an event, not collision
permission. No input contains GT event frames or future motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HSI = Path(__file__).resolve().parents[3] / 'HSI-project'
if str(HSI) not in sys.path:
    sys.path.insert(0, str(HSI))
from hsi.stage3_sequence.geometry import GridSDF, SDFQuery

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


class BoundScene(nn.Module):
    """Stage2 XYZ cells with external bounds -> ZYX centre GridSDF."""
    def __init__(self, target_xyz, other_xyz, edge_bounds, floor_z=0.):
        super().__init__()
        target = torch.as_tensor(target_xyz, dtype=torch.float32)
        other = torch.as_tensor(other_xyz, dtype=torch.float32)
        bounds = torch.as_tensor(edge_bounds, dtype=torch.float32)
        if target.ndim != 3 or target.shape != other.shape or min(target.shape) < 2:
            raise ValueError('Expected equal nondegenerate XYZ SDF arrays')
        if bounds.shape != (2, 3) or not (bounds[1] > bounds[0]).all():
            raise ValueError('Invalid external world bounds')
        if not torch.isfinite(target).all() or not torch.isfinite(other).all():
            raise ValueError('Nonfinite SDF')
        spacing = (bounds[1] - bounds[0]) / torch.tensor(target.shape)
        centres = torch.stack((bounds[0] + spacing / 2, bounds[1] - spacing / 2))
        self.target = GridSDF(target.permute(2, 1, 0).contiguous(), centres)
        self.other = GridSDF(other.permute(2, 1, 0).contiguous(), centres)
        self.register_buffer('centre_bounds', centres)
        self.register_buffer('edge_bounds', bounds)
        self.floor_z = float(floor_z)
        if not math.isfinite(self.floor_z):
            raise ValueError('Floor must be finite')

    @classmethod
    def from_case(cls, case):
        record = case.sdf
        # Explicit names, never infer a grid ordering from its dimensions.
        if record.get('layout') != 'XYZ' or record.get('bounds_kind') != 'voxel_outer_edges' \
                or record.get('coordinate_system') != 'world_zup_m' or record.get('positive_is_free') is not True \
                or record.get('align_corners') is not False \
                or record.get('full_scene_requires_target_union_collision_and_floor') is not True \
                or record.get('outside_is_unknown_not_free') is not True:
            raise ValueError('Unexpected Stage2 SDF coordinate contract')
        target = np.load(record['target_sdf']['path'], allow_pickle=False)
        collision = np.load(record['collision_sdf']['path'], allow_pickle=False)
        if list(target.shape) != record.get('shape') or list(collision.shape) != record.get('shape'):
            raise ValueError('SDF dimensions disagree with the source receipt')
        return cls(target, collision,
                   record['world_bounds'], floor_z=record['ground_plane_z_m']).to(case.initial_pose.device)

    def query(self, points):
        centres, edges = self.centre_bounds.to(points), self.edge_bounds.to(points)
        clamped = points.clamp(centres[0], centres[1])
        target, other = self.target.query(clamped), self.other.query(clamped)
        outside = ((points < edges[0]) | (points > edges[1])).any(-1)
        excess = F.relu(edges[0]-points) + F.relu(points-edges[1])
        unknown = -excess.norm(dim=-1) - .02
        distance = torch.minimum(torch.minimum(target.distances, other.distances),
                                 points[..., 2] - self.floor_z)
        return SDFQuery(torch.where(outside, torch.minimum(distance, unknown), distance), outside)

    def outside_loss(self, points):
        b = self.edge_bounds.to(points)
        return (F.relu(b[0] - points).square() + F.relu(points - b[1]).square()).mean()

    def collision_loss(self, points, clearance=0., tolerance=.003):
        distances = self.query(points).distances
        return F.relu(float(clearance) - distances - tolerance).square().mean() + self.outside_loss(points)

    def mesh_energy(self, mesh, motion):
        # All vertices/all frames, no furniture omission or per-frame subsample.
        loss = self.collision_loss(mesh.vertices, tolerance=.003)
        return {'loss': 100. * loss, 'full_mesh_sdf_m2': loss.detach()}


def interpolate_route(route, count):
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or count < 2:
        raise ValueError('Polyline [N,2] and count>=2 required')
    lengths = (route[1:] - route[:-1]).norm(dim=-1)
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    samples = torch.linspace(0., 1., count, device=route.device) * cumulative[-1]
    segment = torch.searchsorted(cumulative[1:].contiguous(), samples).clamp_max(len(lengths)-1)
    alpha = (samples - cumulative[segment]) / lengths[segment].clamp_min(1e-8)
    return route[segment] + alpha[:, None] * (route[segment+1] - route[segment])


@dataclass
class EventSchedule:
    route_xy: torch.Tensor
    target_strength: torch.Tensor
    anchor_frames: list[int]
    description: dict


def make_schedule(case, frames=120, fps=20., approach_transition=False):
    """Deterministic pilot timetable, NOT a trained allocation network.

    A bounded speed allocation follows Stage1 geometry, reserving >=1.5 s
    for entering/holding the input keypose. No real keypose times are read.
    """
    if frames < 60 or fps <= 0:
        raise ValueError('Pilot needs at least 60 frames and positive FPS')
    route = case.route_xy
    root = case.initial_pose[:2]
    route = torch.cat((root[None], route), 0)
    if not approach_transition:
        route = torch.cat((route, case.target_root_xyz[None, :2]), 0)
    route = route[torch.cat((torch.ones(1, dtype=torch.bool, device=route.device),
        (route[1:] - route[:-1]).norm(dim=-1) > 1e-6))]
    if len(route) < 2:
        route = route.expand(2, 2).clone()
    length = float((route[1:] - route[:-1]).norm(dim=-1).sum())
    walk_frames = min(frames-30, max(20, round(length / .65 * fps)))
    # Approach the route terminal; the input target is a fixed spatial event.
    path = interpolate_route(route, walk_frames)
    approach = path[-1].clone()
    path = torch.cat((path, approach[None].expand(frames-walk_frames, 2)))
    hold = max(4, round(.4 * fps))
    event_frame = frames - hold
    phase = (torch.arange(frames, device=route.device) - walk_frames + 1).float()
    strength = (phase / max(1, event_frame-walk_frames)).clamp(0., 1.)
    strength = strength.square() * (3. - 2. * strength)
    if approach_transition:
        path[walk_frames:] = approach + strength[walk_frames:, None] * (case.target_root_xyz[:2]-approach)
    return EventSchedule(path, strength, list(range(event_frame, frames)), dict(
        kind='deterministic_route_length_allocation_not_learned', fps=fps,
        total_frames=frames, walk_frames=walk_frames, interaction_anchor_frame_0based=event_frame,
        route_length_m=length, nominal_speed_m_s=.65, gt_timestamps_used=False,
        walk_follows_only_stage1_approach_polyline=bool(approach_transition),
        unverified_approach_to_contact_is_walkable=False if approach_transition else None,
        keypose_root_heading_shape_permissions_relaxed=False))


def bone_samples(joints):
    parents = torch.tensor(PARENTS[1:], device=joints.device)
    child, parent = joints[..., 1:, :], joints.index_select(-2, parents)
    return torch.cat((joints, .25*child+.75*parent, .5*child+.5*parent,
                      .75*child+.25*parent), -2)


class DenoisedGuidance:
    def __init__(self, decode, history, initial_history_frames, offset, schedule,
                 target_joints, scene=None, iterations=12, learning_rate=.055, trust_radius=.8,
                 bone_lengths=None, consistency=None, temporal_weight=0., root_height_path=None):
        self.decode, self.history = decode, history.detach()
        if history.shape[1:3] != (263, 1) or history.shape[-1] != initial_history_frames + offset:
            raise ValueError('Pass complete native history: static prefix plus every preceding generated frame')
        if initial_history_frames != 20 or offset < 0 or offset + 40 > len(schedule.route_xy):
            raise ValueError('Invalid context length, offset or incomplete schedule')
        if target_joints.shape != (22, 3) or not torch.isfinite(target_joints).all():
            raise ValueError('Expected finite target J22 in world coordinates')
        if iterations < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Invalid guidance optimizer settings')
        if not math.isfinite(trust_radius) or not 0 < trust_radius <= 4:
            raise ValueError('Guidance trust radius must be in (0,4] normalized feature units')
        self.initial_history_frames, self.offset = initial_history_frames, offset
        self.schedule, self.target_joints, self.scene = schedule, target_joints, scene
        self.iterations, self.learning_rate = iterations, learning_rate
        self.trust_radius = float(trust_radius)
        self.bone_lengths, self.consistency = bone_lengths, consistency
        self.root_height_path = root_height_path
        self.temporal_weight = float(temporal_weight)
        if bone_lengths is not None and (bone_lengths.shape != (21,) or not (bone_lengths > 0).all()):
            raise ValueError('Positive reference lengths for 21 bones required')
        self.trace = []

    def __call__(self, x0):
        # The official inpainting implementation may pass suffix or prefix+suffix.
        if x0.shape[-1] != 40:
            raise ValueError('Expected the predicted 40-frame suffix only')
        original = x0.detach()
        with torch.enable_grad():
            candidate = original.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([candidate], lr=self.learning_rate)
            path = self.schedule.route_xy[self.offset:self.offset+40]
            strength = self.schedule.target_strength[self.offset:self.offset+40]
            with torch.no_grad():
                original_decoded = self.decode(torch.cat((self.history, original), -1))
                original_joints, prior = original_decoded[:, -40:], original_decoded[:, :-40]
            parents = torch.tensor(PARENTS[1:], device=x0.device)
            for iteration in range(self.iterations):
                all_features = torch.cat((self.history, candidate), -1)
                joints = self.decode(all_features)[:, -40:]
                route_loss = (joints[..., 0, :2] - path).square().mean()
                if self.root_height_path is not None:
                    height = self.root_height_path[self.offset:self.offset+40]
                    route_loss = route_loss + .5*(joints[..., 0, 2]-height).square().mean()
                target_loss = ((joints - self.target_joints).square().mean((-1, -2)) * strength).mean()
                trust = (candidate - original).square().mean()
                sdf_loss = joints.new_zeros(())
                if self.scene is not None:
                    sdf_loss = self.scene.collision_loss(bone_samples(joints), tolerance=.003)
                bone_loss, temporal_loss, consistency_loss = (joints.new_zeros(()) for _ in range(3))
                if self.bone_lengths is not None:
                    bones = (joints[..., 1:, :] - joints.index_select(-2, parents)).norm(dim=-1)
                    bone_loss = (bones-self.bone_lengths).square().mean()
                if self.temporal_weight:
                    correction = joints-original_joints
                    padded = torch.cat((torch.zeros_like(correction[:, :2]), correction), 1)
                    temporal_loss = (padded[:, 2:]-2*padded[:, 1:-1]+padded[:, :-2]).square().mean()
                    boundary = joints[:, 0] - prior[:, -1] - (prior[:, -1]-prior[:, -2])
                    temporal_loss = temporal_loss + boundary.square().mean()
                if self.consistency is not None:
                    agreement = self.consistency(all_features)
                    consistency_loss = (agreement['per_frame_fk_error_m2'][:, -40:].mean()
                        + agreement['per_frame_velocity_error_m2'][:, -41:-1].mean())
                loss = (12. * route_loss + 15. * target_loss + 100. * sdf_loss + .08 * trust
                        + 80. * bone_loss + self.temporal_weight*temporal_loss + 30.*consistency_loss)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite guidance energy')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if candidate.grad is None or not torch.isfinite(candidate.grad).all():
                    raise ValueError('Missing/nonfinite guidance gradient')
                optimizer.step()
                with torch.no_grad():
                    candidate.copy_(original + (candidate-original).clamp(-self.trust_radius, self.trust_radius))
                if iteration in (0, self.iterations-1):
                    self.trace.append(dict(call=len(self.trace)//2, iteration=iteration,
                        route_m2=float(route_loss.detach()), target_m2=float(target_loss.detach()),
                        proxy_sdf_m2=float(sdf_loss.detach()), trust=float(trust.detach()),
                        bone_m2=float(bone_loss.detach()), temporal_correction_m2=float(temporal_loss.detach()),
                        native_consistency_m2=float(consistency_loss.detach()),
                        trust_saturation_fraction=float(((candidate.detach()-original).abs() >= self.trust_radius-1e-5).float().mean())))
        return candidate.detach()


@torch.no_grad()
def evaluate_mesh(mesh, case, scene, fps=20., anchor_frames=None):
    vertices, joints = mesh.vertices[0], mesh.joints[0]
    d = scene.query(vertices)
    target_dist = (joints - case.target_joints).norm(dim=-1).mean(-1)
    root = joints[:, 0]
    feet = joints[:, [10, 11]]
    speed = (feet[1:, :, :2] - feet[:-1, :, :2]).norm(dim=-1) * fps
    near_floor = (feet[1:, :, 2] < .12) & (feet[:-1, :, 2] < .12)
    # This is a geometry-conditioned diagnostic, not a physics contact label.
    contact_slip = speed[near_floor]
    velocity = (joints[1:] - joints[:-1]) * fps
    acceleration = (velocity[1:] - velocity[:-1]) * fps
    anchors = list(anchor_frames or [])
    return dict(frames=len(vertices), fps=fps,
        all_vertices_all_frames_evaluated=True, target_furniture_in_sdf=True,
        physics_validated=False, motion_release_authorized=False,
        penetration_fraction_below_minus_3mm=float((d.distances < -.003).float().mean()),
        conservative_penetration_metric_includes_outside_unknown=True,
        known_domain_penetration_fraction=float(((d.distances < -.003) & ~d.outside).float().mean()),
        penetration_frame_fraction=float((d.distances < -.003).any(-1).float().mean()),
        deepest_penetration_m=float(F.relu(-d.distances).max()),
        outside_fraction=float(d.outside.float().mean()),
        root_path_length_m=float((root[1:,:2]-root[:-1,:2]).norm(dim=-1).sum()),
        root_final_error_m=float((root[-1]-case.target_root_xyz).norm()),
        final_keypose_mean_joint_error_m=float(target_dist[-1]),
        closest_keypose_mean_joint_error_m=float(target_dist.min()),
        max_joint_step_m=float((joints[1:]-joints[:-1]).norm(dim=-1).max()),
        mean_joint_acceleration_m_s2=float(acceleration.norm(dim=-1).mean()),
        near_floor_foot_speed_m_s=float(contact_slip.mean()) if contact_slip.numel() else None,
        near_floor_foot_sample_count=int(contact_slip.numel()),
        algorithm_anchor_frames_0based=anchors,
        anchor_keypose_mean_joint_error_m=[float(target_dist[i]) for i in anchors])
