"""Explicit mixture of registered interactions and broad observed motion.

Dataset manifests remain immutable and separately identifiable. A broad-motion
case never acquires a seat/contact annotation just because it shares a scene
with a verified interaction. Indices are local training bookkeeping only.
"""
from __future__ import annotations

from pathlib import Path
from .data import JointDataset


class JointCorpus:
    def __init__(self, interaction_root, body_asset, device="cpu", motion_root=None):
        interaction = JointDataset(interaction_root,body_asset,device=device)
        self.datasets = [interaction]
        if motion_root is not None:
            from .motion_data import MotionJointDataset
            self.datasets.append(MotionJointDataset(motion_root,body_asset,device=device))
        self.records, self.lookup = [], []
        self.train_indices, self.heldout_indices = [], []
        self.interaction_train_indices, self.motion_train_indices = [], []
        self.manifest = dict(interaction=interaction.manifest,
            motion=self.datasets[1].manifest if len(self.datasets)>1 else None)
        for source,dataset in enumerate(self.datasets):
            offset = len(self.records)
            self.records.extend(dataset.records)
            self.lookup.extend((source,i) for i in range(len(dataset)))
            train = [offset+int(i) for i in dataset.train_indices]
            self.train_indices.extend(train)
            self.heldout_indices.extend(offset+int(i) for i in dataset.heldout_indices)
            (self.interaction_train_indices if source==0 else self.motion_train_indices).extend(train)
        train_scenes = {str(self.records[i]['scene_id']) for i in self.train_indices}
        heldout_scenes = {str(self.records[i]['scene_id']) for i in self.heldout_indices}
        if train_scenes & heldout_scenes:
            raise ValueError("Scene split leakage between motion and interaction corpora: "+str(train_scenes & heldout_scenes))

    def __len__(self):
        return len(self.records)

    def batch(self, indices, *, keypose_counts=None, device=None):
        if len(indices)!=1:
            raise ValueError("Use gradient accumulation, not mismatched-anatomy batching")
        source,index=self.lookup[int(indices[0])]
        return self.datasets[source].batch([index],keypose_counts=keypose_counts,device=device)

    def choose(self,rng,interaction_probability=.2):
        if not 0<interaction_probability<=1:
            raise ValueError("Interaction sampling probability must be positive")
        pool=self.train_indices
        if self.motion_train_indices:
            pool=(self.interaction_train_indices if rng.random()<interaction_probability else self.motion_train_indices)
        return int(rng.choice(pool))
