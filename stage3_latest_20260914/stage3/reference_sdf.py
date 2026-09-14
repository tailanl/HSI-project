"""Python3.8-compatible geometry adapter for CLOSED-LOOP reference proposals.

This is a geometry constraint, not a learned controller or dynamics substitute.
The executed PhysX state must remain the next prefix. Reference joints alone
are not a full-surface collision guarantee. No past/prefix coordinates change.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def checked(record):
    path = Path(record['path']).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(data)
    if path.stat().st_size != record['bytes'] or h.hexdigest() != record['sha256']:
        raise ValueError('Geometry artifact identity mismatch: '+str(path))
    return path


class SceneSDF(nn.Module):
    def __init__(self, target_xyz, other_xyz, edge_bounds, floor_z):
        super().__init__()
        target = torch.as_tensor(target_xyz, dtype=torch.float32)
        other = torch.as_tensor(other_xyz, dtype=torch.float32)
        bounds = torch.as_tensor(edge_bounds, dtype=torch.float32)
        if (target.ndim != 3 or target.shape != other.shape or min(target.shape) < 2
                or bounds.shape != (2, 3) or not bool((bounds[1] > bounds[0]).all())
                or not bool(torch.isfinite(target).all() and torch.isfinite(other).all())
                or not bool(torch.isfinite(bounds).all()) or not np.isfinite(floor_z)):
            raise ValueError('Invalid explicit XYZ SDF grids or bounds')
        # Interpolate each source field, then union. Interpolation and min do
        # not commute where the nearest object changes inside a voxel.
        self.register_buffer('grid', torch.stack((target,other)).permute(0,3,2,1)[None].contiguous())
        self.register_buffer('bounds', bounds)
        spacing = (bounds[1]-bounds[0])/torch.tensor(target.shape)
        self.register_buffer('centres', torch.stack((bounds[0]+spacing/2, bounds[1]-spacing/2)))
        self.floor_z = float(floor_z)

    @classmethod
    def from_packet(cls, packet):
        packet = json.loads(Path(packet).read_text())
        if packet['schema'] != 'agent9.paper_structure.stage12_packet.v1':
            raise ValueError('Wrong Stage1/2 packet schema')
        r = packet['sdf']
        if not (r['layout'] == 'XYZ' and r['positive_is_free'] is True
                and r['bounds_kind'] == 'voxel_outer_edges' and r['align_corners'] is False
                and r['coordinate_system'] == 'world_zup_m'
                and r['outside_is_unknown_not_free'] is True
                and r['full_scene_requires_target_union_collision_and_floor'] is True):
            raise ValueError('Wrong SDF coordinate contract')
        a = np.load(checked(r['target_sdf']), allow_pickle=False)
        b = np.load(checked(r['collision_sdf']), allow_pickle=False)
        if list(a.shape) != r['shape'] or list(b.shape) != r['shape']:
            raise ValueError('SDF shapes do not match source')
        return cls(a, b, r['world_bounds'], r['ground_plane_z_m'])

    def query(self, points):
        if points.shape[-1] != 3 or points.device != self.grid.device or not bool(torch.isfinite(points).all()):
            raise ValueError('Expected finite XYZ on geometry device')
        clamped = points.clamp(self.centres[0], self.centres[1])
        grid = 2*(clamped-self.bounds[0])/(self.bounds[1]-self.bounds[0])-1
        fields = F.grid_sample(self.grid, grid.reshape(1,1,1,-1,3), mode='bilinear',
                               padding_mode='border', align_corners=False).reshape(2,*points.shape[:-1])
        sample = torch.minimum(fields[0],fields[1])
        known_distance = torch.minimum(sample, points[...,2]-self.floor_z)
        unknown = ((points < self.bounds[0]) | (points > self.bounds[1])).any(-1)
        excess = F.relu(self.bounds[0]-points)+F.relu(points-self.bounds[1])
        unknown_distance = -excess.norm(dim=-1)-.02
        distance = torch.where(unknown, torch.minimum(known_distance, unknown_distance), known_distance)
        return distance, unknown

    def loss(self, points, tolerance=.003):
        distance, unknown = self.query(points)
        return F.relu(-distance-float(tolerance)).square().mean()


class ReferenceSDFGuide:
    """Correct only each clean future proposal; decoder prepends fixed physics past.

    `decode_future` must return WORLD Z-UP positions for future frames ONLY,
    e.g. via CLOSD's original RepresentationHandler and current recon_data.
    No synthetic next state is returned to the controller or feedback buffer.
    """
    def __init__(self, scene, decode_future, iterations=2, learning_rate=.02,
                 trust_radius=.15, sdf_weight=100., trust_weight=1.):
        if not callable(decode_future) or not (1 <= iterations <= 8):
            raise ValueError('Bounded optimization and explicit decoder required')
        if not all(np.isfinite(v) and v > 0 for v in (learning_rate, trust_radius, sdf_weight, trust_weight)):
            raise ValueError('Finite positive guidance parameters required')
        self.scene, self.decode_future = scene, decode_future
        self.iterations, self.learning_rate, self.trust_radius = iterations, learning_rate, trust_radius
        self.sdf_weight, self.trust_weight = sdf_weight, trust_weight
        self.trace = []

    def __call__(self, clean_future):
        original = clean_future.detach().clone()
        if not bool(torch.isfinite(original).all()):
            raise ValueError('Nonfinite reference proposal')
        with torch.enable_grad():
            candidate = original.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([candidate], lr=self.learning_rate)
            for iteration in range(self.iterations):
                points = self.decode_future(candidate)
                collision = self.scene.loss(points)
                trust = (candidate-original).square().mean()
                loss = self.sdf_weight*collision+self.trust_weight*trust
                optimizer.zero_grad(); loss.backward()
                if candidate.grad is None or not bool(torch.isfinite(candidate.grad).all()):
                    raise ValueError('Missing/invalid decoder-to-reference gradient')
                optimizer.step()
                with torch.no_grad():
                    candidate.copy_(original+(candidate-original).clamp(-self.trust_radius,self.trust_radius))
                self.trace.append(dict(iteration=iteration, collision_m2=float(collision.detach()),
                    reference_change_max=float((candidate-original).abs().max().detach()),
                    full_body_surface_constraint=False, physical_state_feedback_unchanged=True))
        return candidate.detach()
