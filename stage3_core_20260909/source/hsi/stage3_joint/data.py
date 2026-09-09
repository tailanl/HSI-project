"""Observed continuous LINGO joint-training data; no changes to frozen R3.

Unknown contact labels carry an explicit mask. A positive label is a declared
geometry/velocity pseudo-label, not mocap ground truth. All spatial fields use
one common XY translation; motion Z and rotations are never fitted to furniture.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickletools
import re

import numpy as np
import torch
from torch import nn

from hsi.stage3_sequence.contracts import SequenceCondition
from hsi.stage3_sequence.geometry import GridSDF, SDFQuery, REGION_NAMES
from hsi.stage3_sequence.mesh_body import MeshBody

SCHEMA = 'hsi.stage3_joint.real_continuous_lingo.v1'
ADJUSTABLE_JOINTS = (3, 6, 9, 13, 14, 16, 17, 18, 19)
SEMANTIC_IDS = {'walk': 1, 'sit': 2, 'hold': 3}


def artifact(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def verified(record):
    actual = artifact(record['path'])
    if actual != record:
        raise ValueError('Artifact changed: ' + str(record['path']))
    return Path(actual['path'])


def read_string_lists(path, expected_count=None):
    """Interpret only primitive list/string pickle opcodes, never execute pickle."""
    stack, memo, mark = [], {}, object()
    with Path(path).open('rb') as f:
        for op, arg, _ in pickletools.genops(f):
            name = op.name
            if name in ('PROTO', 'FRAME'):
                continue
            if name == 'EMPTY_LIST':
                stack.append([])
            elif name in ('SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8'):
                stack.append(arg)
            elif name == 'MARK':
                stack.append(mark)
            elif name == 'MEMOIZE':
                memo[len(memo)] = stack[-1]
            elif name in ('BINPUT', 'LONG_BINPUT'):
                memo[arg] = stack[-1]
            elif name in ('BINGET', 'LONG_BINGET'):
                stack.append(memo[arg])
            elif name == 'APPEND':
                value = stack.pop()
                if not isinstance(stack[-1], list):
                    raise ValueError('APPEND target is not a list')
                stack[-1].append(value)
            elif name == 'APPENDS':
                index = len(stack) - 1
                while index >= 0 and stack[index] is not mark:
                    index -= 1
                if index < 1 or not isinstance(stack[index-1], list):
                    raise ValueError('Invalid primitive list stack')
                values = stack[index+1:]
                del stack[index:]
                stack[-1].extend(values)
            elif name == 'STOP':
                if f.read(1):
                    raise ValueError('Trailing pickle payload')
                break
            else:
                raise ValueError('Executable/unsupported pickle opcode: ' + name)
    if len(stack) != 1 or not isinstance(stack[0], list):
        raise ValueError('Need one primitive list')
    values = stack[0]
    if expected_count is not None and len(values) != expected_count:
        raise ValueError('Metadata count differs from raw index')
    if not all(isinstance(x, list) and len(x) and all(isinstance(t, str) for t in x) for x in values):
        raise ValueError('Expected nonempty list[str] labels per segment')
    return values


def select_candidates(raw_root, scene_ids=None):
    """Exact adjacent labeled walk/sit/hold; no gaps, mirrors or guessed joins."""
    from experiments.stage3_lingo_motion_pretrain import scene_runs
    root = Path(raw_root)
    starts, ends = (np.load(root / n, allow_pickle=False) for n in ('start_idx.npy', 'end_idx.npy'))
    if starts.ndim != 1 or starts.shape != ends.shape or not np.all(ends > starts):
        raise ValueError('Invalid segment indices')
    texts = read_string_lists(root / 'text_aug.pkl', len(starts))
    raw = np.load(root / 'transl_aligned.npy', mmap_mode='r', allow_pickle=False)
    runs = scene_runs(root / 'scene_name.pkl', len(raw))
    rstarts = [r[0] for r in runs]
    wanted = None if scene_ids is None else {int(s) for s in scene_ids}
    def scene(i):
        n = bisect_right(rstarts, int(starts[i])) - 1
        return runs[n][2] if n >= 0 and ends[i] <= runs[n][1] else None
    result = []
    for i in range(1, len(starts)-1):
        labels = [texts[j][0] for j in (i-1, i, i+1)]
        if not re.search(r'\bwalk(?:s|ed|ing)?\b', labels[0], re.I):
            continue
        if not re.search(r'\bsit(?:s|ting)?\s+down\b', labels[1], re.I):
            continue
        if not re.search(r'maintain\w*\s+sit\w*\s+posture', labels[2], re.I):
            continue
        name = scene(i)
        if (name is None or 'mirror' in name.lower() or name != scene(i-1) or name != scene(i+1)
                or ends[i-1] != starts[i] or ends[i] != starts[i+1]):
            continue
        match = re.match(r'^(\d+)', name)
        if match is None or (wanted is not None and int(match[1]) not in wanted):
            continue
        base = f'{int(match[1]):03d}'
        result.append(dict(case_id=f'scene{base}_clips{i-1}_{i}_{i+1}',
            clip_ids=[i-1, i, i+1], source_scene=name, scene_id=base,
            raw_start=int(starts[i-1]), sit_start=int(starts[i]), hold_start=int(starts[i+1]),
            raw_end_exclusive=int(ends[i+1]), labels=labels,
            split='heldout' if base == '036' else 'train'))
    return result


def contact_pseudolabels(regions_world, *, seat_distance, inside_seat, fps=15.):
    """Conservative positives/negatives; all unobserved body regions unknown.

    The supplied seat distance is distance to the geometrically verified real
    seat triangles. Ground is the declared scene Z=0, not a fitted body floor.
    Thresholds define pseudo-label confidence, not Stage2 acceptance gates.
    """
    points = np.asarray(regions_world, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (8, 3) or not np.isfinite(points).all():
        raise ValueError('Need finite [T,8,3] anatomical region centroids')
    speed = np.linalg.norm(np.diff(points, axis=0, prepend=points[:1]), axis=-1)*fps
    labels = np.zeros(points.shape[:2], np.float32)
    known = np.zeros(points.shape[:2], bool)
    confidence = np.zeros(points.shape[:2], np.float32)
    distances = np.zeros(points.shape[:2], np.float64)
    distances[:, 0] = seat_distance
    distances[:, 3:5] = np.abs(points[:, 3:5, 2])
    for region in (0, 3, 4):
        close = distances[:, region] <= (.065 if region == 0 else .035)
        slow = speed[:, region] <= (.15 if region == 0 else .18)
        positive = close & slow
        if region == 0:
            positive &= inside_seat
        # An isolated near-surface instant is not established contact.
        stable = np.convolve(positive.astype(int), np.ones(3, int), mode='same') >= 2
        positive &= stable
        negative = distances[:, region] >= (.16 if region == 0 else .09)
        known[:, region] = positive | negative
        labels[positive, region] = 1.
        confidence[positive, region] = .8
        confidence[negative, region] = .7
    return labels, known, confidence, speed


def keypose_frames(total, establish_frame, k, travel_end=None):
    """One-based internal event poses; last pose is NOT forced to the last frame."""
    if k not in (1, 2, 3) or not 4 <= establish_frame <= total-4:
        raise ValueError('Need an interior establishment event and K in 1..3')
    travel_end = max(1, establish_frame//2) if travel_end is None else int(travel_end)
    if not 1 <= travel_end < establish_frame:
        raise ValueError('Travel must precede establishment')
    if k == 1:
        return [establish_frame]
    if k == 2:
        return [travel_end, establish_frame]
    return [travel_end, establish_frame,
            min(total-1, establish_frame+max(1, (total-establish_frame)//2))]


def make_condition(arrays, record, k=1):
    """Create B=1 condition and separate GT duration/contact tensors."""
    motion = torch.as_tensor(arrays['motion'], dtype=torch.float32)
    joints = torch.as_tensor(arrays['joints'], dtype=torch.float32)
    regions = torch.as_tensor(arrays['regions'], dtype=torch.float32)
    h, total = int(record['history_frames']), len(motion)-int(record['history_frames'])
    event = int(record['establish_frame_1based'])
    travel_end = int(record['sit_frame_1based'])-1
    anchors = keypose_frames(total, event, k, travel_end)
    # Travel remains its own event even for K=1. Event order is an input;
    # actual source boundary frame numbers appear only in supervision.
    boundaries = sorted(set([travel_end]+anchors+[total]))
    durations = np.diff([0] + boundaries)
    m = len(boundaries)
    z = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
    b = lambda *shape: torch.zeros(*shape, dtype=torch.bool)
    i = lambda *shape: torch.zeros(*shape, dtype=torch.long)
    poses = motion[[h+a-1 for a in anchors]][None]
    pose_joints = joints[[h+a-1 for a in anchors]][None]-poses[..., None, :3]
    active = b(1, k, 8)
    targets = poses[..., None, :3].expand(1, k, 8, 3).clone()
    normals = z(1, k, 8, 3)
    ids = i(1, k, 8)-1
    semantics = i(1, k)
    labels = torch.as_tensor(arrays['contacts'], dtype=torch.float32)
    known = torch.as_tensor(arrays['contact_known'], dtype=torch.bool)
    seat_target = torch.as_tensor(arrays['seat_target'], dtype=torch.float32)
    for p, a in enumerate(anchors):
        idx = h+a-1
        active[0, p] = (labels[idx] > .5) & known[idx]
        if a <= travel_end:
            semantics[0, p] = SEMANTIC_IDS['walk']
        elif a == event:
            semantics[0, p] = SEMANTIC_IDS['sit']
        else:
            semantics[0, p] = SEMANTIC_IDS['hold']
        for r in (0, 3, 4):
            if active[0, p, r]:
                targets[0, p, r] = seat_target if r == 0 else regions[idx, r]
                if r in (3, 4):
                    targets[0, p, r, 2] = 0.
                normals[0, p, r] = torch.as_tensor(arrays.get('seat_normal',[0.,0.,1.])) if r == 0 else torch.tensor([0.,0.,1.])
                ids[0, p, r] = 1 if r == 0 else 0
    slot_types = i(1, m)
    owners = i(1, m)
    slot_contacts = b(1, m, 8)
    for s, boundary in enumerate(boundaries):
        owner = next((p for p,a in enumerate(anchors) if a >= boundary),k-1)
        owners[0, s] = owner
        slot_types[0, s] = 0 if boundary <= travel_end else 1 if boundary == event else 2
        if slot_types[0, s] == 2:
            trustworthy_hold=torch.as_tensor(record.get('hold_contact_requirements',[True]*8),dtype=torch.bool)
            slot_contacts[0, s] = active[0, owner] & trustworthy_hold
    locked = torch.ones(1, k, 22, dtype=torch.bool)
    locked[..., list(ADJUSTABLE_JOINTS)] = False
    tolerance = z(1, k, 22)
    tolerance[..., list(ADJUSTABLE_JOINTS)] = np.deg2rad(5.)
    initial = (labels[h-1] > .5) & known[h-1]
    initial_ids = i(1, 8)-1
    initial_ids[0, initial] = 0
    if initial[0]:
        initial_ids[0, 0] = 1
    route = z(1, m, 8)
    # Input/history-to-goal geometric displacement only; no GT trajectory or
    # per-frame velocity supplied as route features.
    delta = poses[0, -1, :2]-motion[h-1, :2]
    route[..., :2] = delta
    route[..., 2] = delta.norm()
    route[..., 6] = seat_target[2]
    route[..., 7] = (slot_types != 0).float()
    c = SequenceCondition(history=motion[:h][None], history_mask=torch.ones(1,h,dtype=torch.bool),
        history_joints=joints[:h][None]-motion[:h][None,:,None,:3],
        initial_contact_active=initial[None], initial_contact_target_ids=initial_ids,
        keyposes=poses, keypose_mask=torch.ones(1,k,dtype=torch.bool), keypose_joints=pose_joints,
        keypose_text=z(1,k,512), semantic_ids=semantics,
        scene_points=torch.as_tensor(arrays['scene_points'],dtype=torch.float32)[None],
        scene_features=torch.as_tensor(arrays['scene_features'],dtype=torch.float32)[None],
        scene_mask=torch.ones(1,len(arrays['scene_points']),dtype=torch.bool),
        contact_targets=targets, contact_normals=normals, contact_active=active,
        contact_body_regions=i(1,k,8)-1, contact_target_ids=ids,
        locked_root=torch.ones(1,k,3,dtype=torch.bool), locked_joints=locked,
        root_tolerance=z(1,k,3), joint_tolerance=tolerance,
        slot_mask=torch.ones(1,m,dtype=torch.bool), slot_types=slot_types, slot_keyposes=owners,
        slot_contact_active=slot_contacts, slot_release_active=b(1,m,8),
        minimum_frames=torch.ones(1,m,dtype=torch.long), keypose_slots=torch.tensor([[boundaries.index(a) for a in anchors]]),
        route_features=route, total_frames=torch.tensor([total]), fps=15.).validate()
    return c, motion[h:][None], torch.as_tensor(durations,dtype=torch.long)[None], labels[h:][None], known[h:][None]


class Scene(nn.Module):
    """Unmodified real occupancy SDF plus explicit world floor, commonly rebased."""
    def __init__(self, grid_zyx, centre_bounds, origin):
        super().__init__()
        origin = torch.as_tensor(origin,dtype=torch.float32)
        self.grid = GridSDF(torch.as_tensor(grid_zyx,dtype=torch.float32),
                            torch.as_tensor(centre_bounds,dtype=torch.float32)-origin)
        self.floor_z = -float(origin[2])

    def query(self, points):
        result = self.grid.query(points)
        floor=points[...,2]-self.floor_z
        # A point in the explicit infinite ground half-space is known solid,
        # even when below the voxel grid. Do not relabel floor penetration as
        # unobserved space or silently exclude it from physical supervision.
        known_floor=floor<=0
        distance=torch.minimum(result.signed_distance_m,floor)
        distance=torch.where(known_floor & result.outside,floor,distance)
        return SDFQuery(distance,result.outside & ~known_floor)

    sample = query


def make_body(asset, region_vertex_ids, collision_vertex_ids=None, model=None):
    if model is None:
        from smplx import SMPLXLayer
        model = SMPLXLayer(str(asset),gender='neutral',num_betas=10,num_expression_coeffs=10,
            use_pca=False,flat_hand_mean=True,use_face_contour=False,dtype=torch.float32)
    identity = torch.eye(3).expand(15,3,3).clone()
    return MeshBody(model,torch.zeros(10),left_hand_pose=identity,right_hand_pose=identity,
        region_vertex_ids=region_vertex_ids,collision_vertex_ids=collision_vertex_ids)


@dataclass
class JointBatch:
    condition: SequenceCondition
    clean: torch.Tensor
    timing: torch.Tensor
    contacts: torch.Tensor
    contact_known: torch.Tensor
    body: MeshBody
    scene: Scene
    metadata: list[dict]


class JointDataset:
    """Immutable, individually verified cases; pilot batches deliberately B=1."""
    def __init__(self, prepared_root, body_asset=None, device='cpu'):
        self.root = Path(prepared_root).resolve(strict=True)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['schema'] != SCHEMA or not self.manifest['cases']:
            raise ValueError('No verified real continuous cases')
        self.records = self.manifest['cases']
        self.train_indices = [n for n,r in enumerate(self.records) if r['split']=='train']
        self.heldout_indices = [n for n,r in enumerate(self.records) if r['split']=='heldout']
        self.asset = verified(self.manifest['body_asset'])
        if body_asset is not None and Path(body_asset).resolve() != self.asset:
            raise ValueError('Another body asset requested')
        self.device = torch.device(device)
        self._model = None
        self._arrays = {}
        self._scene_arrays = {}

    def __len__(self):
        return len(self.records)

    def batch(self, indices, *, keypose_counts=None, device=None):
        if len(indices) != 1:
            raise ValueError('Initial real-mesh joint training uses explicit batch size one')
        n = int(indices[0]); record = self.records[n]
        if n not in self._arrays:
            with np.load(verified(record['arrays']),allow_pickle=False) as z:
                self._arrays[n] = {name:np.array(z[name],copy=True) for name in z.files}
        arrays = self._arrays[n]
        k = int(keypose_counts[0]) if keypose_counts is not None else 1+n%3
        condition,clean,timing,contacts,known = make_condition(arrays,record,k)
        chosen = self.device if device is None else torch.device(device)
        body = make_body(self.asset,record['body_regions'],record['collision_vertex_ids'],model=self._model).to(chosen)
        self._model = body.model
        scene_record = record['scene_arrays']
        key = scene_record['sha256']
        if key not in self._scene_arrays:
            with np.load(verified(scene_record),allow_pickle=False) as z:
                self._scene_arrays[key] = {name:np.array(z[name],copy=True) for name in ('sdf_zyx','centre_bounds')}
        geometry = self._scene_arrays[key]
        scene = Scene(geometry['sdf_zyx'],geometry['centre_bounds'],record['world_origin']).to(chosen)
        return JointBatch(condition.to(chosen),clean.to(chosen),timing.to(chosen),contacts.to(chosen),known.to(chosen),
                          body,scene,[record])
