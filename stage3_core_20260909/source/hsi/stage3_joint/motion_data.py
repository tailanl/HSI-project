"""Broad real LINGO subwindows, NEVER joins between language clips.

Each row indexes compact, mesh-derived per-recording arrays. Global XYZ is
rebased once per window; shapes/hands are declared canonical, not subject GT.
Only observed sole geometry supplies contact pseudo-labels. Unverified seat,
hand, thigh and back contacts remain unknown, even when text says 'sit'.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from hsi.stage3_joint.data import Scene, JointBatch, make_body, artifact, verified, read_string_lists
from hsi.stage3_sequence.contracts import SequenceCondition

SCHEMA = 'hsi.stage3_joint.real_motion_windows.v1'
SEMANTICS = {'other': 0, 'walk': 1, 'sit': 2, 'stand': 4, 'turn': 5}
# Only shoulders/elbows may relax; root/global/legs/spine remain exact anchors.
ADJUSTABLE_JOINTS = (13, 14, 16, 17, 18, 19)


def semantic_id(text):
    label = str(text).lower()
    for kind, pattern in (('sit', r'\bsit(?:s|ting)?\b'), ('stand', r'\bstand(?:s|ing)?\b|\bget(?:s|ting)? up\b'),
                          ('turn', r'\bturn(?:s|ed|ing)?\b'), ('walk', r'\bwalk(?:s|ed|ing)?\b')):
        if re.search(pattern, label):
            return SEMANTICS[kind]
    return SEMANTICS['other']


def inherited_heldout(manifests):
    result = set()
    for path in manifests:
        data = json.loads(Path(path).read_text())
        values = list(data.get('heldout_base_scenes', []))
        for key in ('selections', 'cases'):
            values += [r.get('base_scene', r.get('scene_id')) for r in data.get(key, []) if r['split'] == 'heldout']
        for value in values:
            if value is None or not str(value).isdigit():
                raise ValueError('Heldout manifest lacks numeric base scene identity')
            result.add(f'{int(value):03d}')
    if not result:
        raise ValueError('Explicit previous heldout groups required')
    return result


def enumerate_windows(starts, ends, runs, labels, heldout, *, lengths=(60, 90, 120, 150, 180), hop=30):
    """2-frame history + T future, 30→15 FPS; strictly within ONE raw clip.

    Recording arrays are sampled at scene-run start+2*i. Start is rounded UP
    onto that grid, never before the annotation. No language-clip join exists.
    """
    starts, ends = np.asarray(starts), np.asarray(ends)
    if (starts.ndim != 1 or starts.shape != ends.shape or len(labels) != len(starts)
            or starts.dtype.kind not in 'iu' or ends.dtype.kind not in 'iu'
            or np.any(starts < 0) or np.any(ends <= starts) or np.any(starts[1:] < ends[:-1])):
        raise ValueError('Need disjoint ordered exact raw-language-clip intervals')
    if type(hop) is not int or hop < 1 or not lengths or any(type(t) is not int or not 60 <= t <= 180 for t in lengths):
        raise ValueError('Valid explicit 60..180 lengths and positive hop required')
    run_starts = [int(r[0]) for r in runs]
    records, rejected = [], []
    for clip_id, (first, stop) in enumerate(zip(starts, ends)):
        first, stop = int(first), int(stop)
        index = bisect_right(run_starts, first)-1
        if index < 0 or stop > int(runs[index][1]):
            rejected.append(dict(clip_id=clip_id, reason='crosses_scene_recording_run')); continue
        rstart, rend, source = runs[index]
        match = re.fullmatch(r'(\d+)[A-Za-z0-9_.-]*', source)
        if 'mirror' in source.lower() or match is None:
            rejected.append(dict(clip_id=clip_id, reason='mirror_or_unrecognized_scene')); continue
        base = f'{int(match[1]):03d}'
        start_idx = (first-int(rstart)+1)//2
        stop_idx = (stop-int(rstart)+1)//2
        for total in lengths:
            for start in range(start_idx, stop_idx-total-2+1, hop):
                raw_start, raw_end = int(rstart)+2*start, int(rstart)+2*(start+total+1)+1
                if raw_start < first or raw_end > stop:
                    raise AssertionError('A window escaped its original single clip')
                records.append(dict(case_id=f'motion_{source}_clip{clip_id}_raw{raw_start}_t{total}',
                    recording_id=f'run{index:03d}_{source}', recording_index=index, source_scene=source,
                    scene_id=base, clip_id=clip_id, raw_segment_start=first, raw_segment_end_exclusive=stop,
                    raw_start=raw_start, raw_end_exclusive=raw_end, array_start=start, history_frames=2,
                    future_frames=total, fps=15., stride=2, split='heldout' if base in heldout else 'train',
                    label=labels[clip_id][0], semantic_id=semantic_id(labels[clip_id][0]),
                    continuity='one_original_language_clip_no_cross_clip_join'))
    return records, rejected


def foot_pseudolabels(regions, clip_ids, *, fps=15.):
    points = np.asarray(regions, dtype=np.float32)
    clip_ids = np.asarray(clip_ids)
    if points.ndim != 3 or points.shape[1:] != (8, 3) or clip_ids.shape != (len(points),) or not np.isfinite(points).all():
        raise ValueError('Finite actual surface regions and per-frame source clip IDs required')
    labels, known = np.zeros((len(points), 8), np.float32), np.zeros((len(points), 8), bool)
    speed = np.zeros((len(points), 2), np.float32)
    speed[1:] = np.linalg.norm(np.diff(points[:, 3:5], axis=0), axis=-1)*fps
    continuous = np.zeros(len(points), bool)
    continuous[1:] = (clip_ids[1:] == clip_ids[:-1]) & (clip_ids[1:] >= 0)
    for n, region in enumerate((3, 4)):
        height = points[:, region, 2]
        close_slow = (height >= -.015) & (height <= .035) & (speed[:, n] <= .18) & continuous
        stable = np.zeros(len(points), bool)
        if len(points) > 2:
            stable[1:-1] = ((close_slow[:-2].astype(int)+close_slow[1:-1]+close_slow[2:]) >= 2)
            stable[1:-1] &= continuous[1:-1] & continuous[2:]
        positive = close_slow & stable
        negative = (height > .09) & continuous  # below-floor failures are UNKNOWN, not negative
        labels[positive, region] = 1.
        known[:, region] = positive | negative
    return labels, known


def anchor_frames(total, count, case_id):
    if count not in (1, 2, 3) or total < 60:
        raise ValueError('K must be 1..3 for a real 60+ frame window')
    seed = int(hashlib.sha256((case_id+f'/k{count}').encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    minimum_gap = max(4, total//12)
    for _ in range(1000):
        values = np.sort(rng.choice(np.arange(4, total-3), count, replace=False))
        if np.all(np.diff(values) >= minimum_gap):
            return values.tolist()
    raise ValueError('Cannot draw feasible separated interior anchors')


def make_motion_condition(arrays, record, scene_arrays, k=1):
    total, h = int(record['future_frames']), 2
    start = int(record['array_start']); stop = start+h+total
    origin = np.asarray(record['world_origin'], np.float32)
    motion = torch.from_numpy(np.array(arrays['motion_world'][start:stop], copy=True))
    motion[:, :3] -= torch.from_numpy(origin)
    joints_relative = torch.from_numpy(np.array(arrays['joints_relative'][start:stop], copy=True))
    regions = torch.from_numpy(np.array(arrays['regions_world'][start:stop], copy=True))-torch.from_numpy(origin)
    labels = torch.from_numpy(np.array(arrays['contacts'][start:stop], copy=True))
    known = torch.from_numpy(np.array(arrays['contact_known'][start:stop], copy=True))
    if len(motion) != h+total:
        raise ValueError('Indexed motion source too short')
    anchors = anchor_frames(total, k, record['case_id'])
    indices = [h+a-1 for a in anchors]
    poses = motion[indices][None]
    active = ((labels[indices] > .5) & known[indices])[None]
    targets = regions[indices][None].clone()
    # Only floor is geometrically certified here. Keep all furniture in SDF.
    targets[..., 3:5, 2] = -float(origin[2])
    normals = torch.zeros(1, k, 8, 3); normals[..., 2] = active.float()
    ids = torch.full((1, k, 8), -1, dtype=torch.long); ids[active] = 0
    count = k+1
    locked = torch.ones(1, k, 22, dtype=torch.bool); locked[..., list(ADJUSTABLE_JOINTS)] = False
    tolerance = torch.zeros(1, k, 22); tolerance[..., list(ADJUSTABLE_JOINTS)] = np.deg2rad(5.)
    slot_types = torch.zeros(1, count, dtype=torch.long)
    slot_active = torch.zeros(1, count, 8, dtype=torch.bool)
    for n in range(k):
        if active[0, n].any():
            slot_types[0, n] = 1  # only establishes at its internal anchor
            slot_active[0, n] = active[0, n]
    initial = (labels[h-1] > .5) & known[h-1]
    initial_ids = torch.full((1, 8), -1, dtype=torch.long); initial_ids[0, initial] = 0
    scene_points = torch.from_numpy(np.array(scene_arrays['mesh_points'], copy=True))-torch.from_numpy(origin)
    features = torch.zeros(len(scene_points), 4)
    features[:, :3] = torch.from_numpy(np.array(scene_arrays['mesh_normals'], copy=True))
    route = torch.zeros(1, count, 8)
    previous = torch.cat((motion[h-1:h, :3], poses[0, :-1, :3]), 0)
    delta = poses[0, :, :3]-previous
    route[0, :k, :2] = delta[:, :2]; route[0, :k, 2] = delta[:, :2].norm(dim=-1)
    c = SequenceCondition(history=motion[:h][None], history_mask=torch.ones(1, h, dtype=torch.bool),
        history_joints=joints_relative[:h][None], initial_contact_active=initial[None], initial_contact_target_ids=initial_ids,
        keyposes=poses, keypose_mask=torch.ones(1, k, dtype=torch.bool), keypose_joints=joints_relative[indices][None],
        keypose_text=torch.zeros(1, k, 512), semantic_ids=torch.full((1, k), record['semantic_id'], dtype=torch.long),
        scene_points=scene_points[None], scene_features=features[None], scene_mask=torch.ones(1, len(scene_points), dtype=torch.bool),
        contact_targets=targets, contact_normals=normals, contact_active=active,
        contact_body_regions=torch.full((1, k, 8), -1, dtype=torch.long), contact_target_ids=ids,
        locked_root=torch.ones(1, k, 3, dtype=torch.bool), locked_joints=locked,
        root_tolerance=torch.zeros(1, k, 3), joint_tolerance=tolerance,
        slot_mask=torch.ones(1, count, dtype=torch.bool), slot_types=slot_types,
        slot_keyposes=torch.tensor([list(range(k))+[k-1]]), slot_contact_active=slot_active,
        slot_release_active=torch.zeros(1, count, 8, dtype=torch.bool), minimum_frames=torch.ones(1, count, dtype=torch.long),
        keypose_slots=torch.arange(k)[None], route_features=route, total_frames=torch.tensor([total]), fps=15.).validate()
    durations = torch.tensor([np.diff([0]+anchors+[total]).tolist()], dtype=torch.long)
    return c, motion[h:][None], durations, labels[h:][None], known[h:][None]


class MotionJointDataset:
    """Compatible B=1 loader with bounded array caches and strict artifact hashes."""
    def __init__(self, prepared_root, body_asset=None, device='cpu'):
        self.root = Path(prepared_root).resolve(strict=True)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('schema') != SCHEMA or self.manifest.get('status') != 'sealed_real_motion_windows':
            raise ValueError('Need a sealed actual continuous-motion manifest')
        self.records = self.manifest['cases']
        if not self.records:
            raise ValueError('Sealed motion corpus has no actual windows')
        self.train_indices = [n for n, r in enumerate(self.records) if r['split'] == 'train']
        self.heldout_indices = [n for n, r in enumerate(self.records) if r['split'] == 'heldout']
        heldout = set(self.manifest['heldout_base_scenes'])
        if any((r['scene_id'] in heldout) != (r['split'] == 'heldout') for r in self.records):
            raise ValueError('Inherited scene split drift')
        self.asset = verified(self.manifest['body_asset'])
        if body_asset is not None and Path(body_asset).resolve() != self.asset:
            raise ValueError('Body asset differs from prepared sources')
        self.device, self._model = torch.device(device), None
        self._recording_cache, self._scene_cache, self._verified = OrderedDict(), OrderedDict(), set()

    def __len__(self):
        return len(self.records)

    def _arrays(self, record, cache, limit=None):
        limit = (16 if cache is self._scene_cache else 8) if limit is None else limit
        key = record['sha256']
        if key not in cache:
            path = verified(record)
            with np.load(path, allow_pickle=False) as z:
                cache[key] = {name: np.array(z[name], copy=True) for name in z.files}
            while len(cache) > limit:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def batch(self, indices, *, keypose_counts=None, device=None):
        if len(indices) != 1:
            raise ValueError('Actual-mesh corpus uses explicit batch size one; accumulate gradients externally')
        n = int(indices[0]); record = self.records[n]
        source = self.manifest['recordings'][record['recording_id']]
        arrays = self._arrays(source['arrays'], self._recording_cache)
        scene_source = self.manifest['scenes'][record['source_scene']]
        geometry = self._arrays(scene_source['geometry_arrays'], self._scene_cache)
        k = int(keypose_counts[0]) if keypose_counts is not None else 1+n%3
        c, clean, timing, labels, known = make_motion_condition(arrays, record, geometry, k)
        chosen = self.device if device is None else torch.device(device)
        body = make_body(self.asset, self.manifest['body_regions'], self.manifest['collision_vertex_ids'], model=self._model).to(chosen)
        self._model = body.model
        scene = Scene(geometry['sdf_zyx'], geometry['centre_bounds'], record['world_origin']).to(chosen)
        metadata = dict(record, scene_arrays=scene_source['geometry_arrays'], body_regions=self.manifest['body_regions'],
            collision_vertex_ids=self.manifest['collision_vertex_ids'], text_embedding='explicit_null_not_pretrained_text',
            contact_supervision='actual_fixed_sole_geometry_pseudolabels_not_ground_truth',
            unverified_seat_hand_contacts_unknown=True, corpus_kind='broad_real_single_clip_motion')
        return JointBatch(c.to(chosen), clean.to(chosen), timing.to(chosen), labels.to(chosen), known.to(chosen), body, scene, [metadata])
