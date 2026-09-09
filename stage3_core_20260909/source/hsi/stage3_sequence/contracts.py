"""Typed, timestamp-free input contract for the experimental sequence model."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

import torch
from torch import Tensor

MOTION_DIM = 135
JOINTS = 22
REGIONS = 8
ROUTE_DIM = 8
SCENE_FEATURE_DIM = 4
TEXT_DIM = 512
SCHEMA = "hsi.stage3_sequence.untimed.v1"


def prefix_mask(mask: Tensor, name: str) -> None:
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional bool mask")
    if not bool(mask.any(dim=1).all()):
        raise ValueError(f"{name} must contain at least one valid entry per sample")
    if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError(f"{name} must be a contiguous valid prefix")


@dataclass(frozen=True)
class SequenceCondition:
    """All masks use True=valid. World coordinates are metres, Z-up.

    Motion channels: root xyz then 22 row-convention 6D rotations.
    History is fixed context, not included in total_frames. Internal GT times
    are deliberately absent. keypose_slots references EVENT ORDER, not frames.
    Target IDs bind relations within the current scene, not learned ID classes.
    Text embeddings must come from the caller's declared frozen text encoder.
    """

    history: Tensor                        # B,H,135
    history_mask: Tensor                   # B,H
    history_joints: Tensor                 # B,H,22,3 root-relative, same body
    initial_contact_active: Tensor         # B,R at final history frame
    initial_contact_target_ids: Tensor     # B,R identity of those relations
    keyposes: Tensor                       # B,K,135
    keypose_mask: Tensor                   # B,K
    keypose_joints: Tensor                 # B,K,22,3 root-relative
    keypose_text: Tensor                   # B,K,512
    semantic_ids: Tensor                   # B,K in [0,31]
    scene_points: Tensor                   # B,S,3
    scene_features: Tensor                 # B,S,4 (caller-declared geometry)
    scene_mask: Tensor                     # B,S
    contact_targets: Tensor                # B,K,R,3 world positions
    contact_normals: Tensor                # B,K,R,3 world unit normals
    contact_active: Tensor                 # B,K,R bool at keypose
    contact_body_regions: Tensor           # B,K,R -1=scene, otherwise region ID
    contact_target_ids: Tensor             # B,K,R current-scene object/part IDs
    locked_root: Tensor                    # B,K,3 bool
    locked_joints: Tensor                  # B,K,22 bool: whole local rotation
    root_tolerance: Tensor                 # B,K,3 metres
    joint_tolerance: Tensor                # B,K,22 radians
    slot_mask: Tensor                      # B,M valid event-duration slots
    slot_types: Tensor                     # B,M 0 travel,1 establish,2 hold,3 release
    slot_keyposes: Tensor                  # B,M keypose whose relation is referenced
    slot_contact_active: Tensor            # B,M,R explicit contact requirements
    slot_release_active: Tensor            # B,M,R explicit separation requirements
    minimum_frames: Tensor                 # B,M integers, >=1 for active slots
    keypose_slots: Tensor                  # B,K event-end anchors, -1 padding
    route_features: Tensor                 # B,M,8 geometry, never GT trajectory/time
    total_frames: Tensor                   # B integer FUTURE lengths
    fps: float = 20.0

    @property
    def batch_size(self) -> int:
        return int(self.history.shape[0])

    @property
    def device(self):
        return self.history.device

    @property
    def max_frames(self) -> int:
        return int(self.total_frames.max().item())

    def to(self, device=None, dtype=None) -> "SequenceCondition":
        # Never cast boolean/index tensors to a floating dtype.
        return replace(self, **{
            field.name: getattr(self, field.name).to(
                device=device,
                dtype=dtype if getattr(self, field.name).is_floating_point() else None)
            for field in fields(self) if isinstance(getattr(self, field.name), Tensor)
        })

    def validate(self) -> "SequenceCondition":
        if self.history.ndim != 3 or self.history.shape[-1] != MOTION_DIM:
            raise ValueError("history must be B,H,135")
        if self.keyposes.ndim != 3 or self.keyposes.shape[-1] != MOTION_DIM:
            raise ValueError("keyposes must be B,K,135")
        b, h = self.history.shape[:2]
        k = self.keyposes.shape[1]
        if self.scene_points.ndim != 3 or self.slot_mask.ndim != 2:
            raise ValueError("scene_points/slot_mask rank invalid")
        s, m = self.scene_points.shape[1], self.slot_mask.shape[1]
        if min(b, h, k, s, m) < 1:
            raise ValueError("all batch, sequence and scene dimensions must be positive")
        shapes = {
            "history_mask": (b, h), "history_joints": (b,h,JOINTS,3),
            "initial_contact_active": (b,REGIONS), "initial_contact_target_ids": (b,REGIONS),
            "keyposes": (b, k, MOTION_DIM),
            "keypose_mask": (b, k), "keypose_joints": (b,k,JOINTS,3),
            "keypose_text": (b,k,TEXT_DIM), "semantic_ids": (b,k),
            "scene_points": (b,s,3), "scene_features": (b,s,SCENE_FEATURE_DIM),
            "scene_mask": (b,s), "contact_targets": (b,k,REGIONS,3),
            "contact_normals": (b,k,REGIONS,3), "contact_active": (b,k,REGIONS),
            "contact_body_regions": (b,k,REGIONS), "contact_target_ids": (b,k,REGIONS),
            "locked_root": (b,k,3), "locked_joints": (b,k,JOINTS),
            "root_tolerance": (b,k,3), "joint_tolerance": (b,k,JOINTS),
            "slot_mask": (b,m), "slot_types": (b,m), "slot_keyposes": (b,m),
            "slot_contact_active": (b,m,REGIONS), "slot_release_active": (b,m,REGIONS),
            "minimum_frames": (b,m),
            "keypose_slots": (b,k), "route_features": (b,m,ROUTE_DIM),
            "total_frames": (b,),
        }
        bools = {"history_mask","initial_contact_active","keypose_mask","scene_mask","contact_active",
                 "locked_root","locked_joints","slot_mask","slot_contact_active","slot_release_active"}
        ints = {"initial_contact_target_ids","semantic_ids","contact_body_regions","contact_target_ids","slot_types",
                "slot_keyposes","minimum_frames","keypose_slots","total_frames"}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.device != self.device:
                raise ValueError(f"{name} is on a different device")
            if name in bools and value.dtype != torch.bool:
                raise ValueError(f"{name} must be bool")
            if name in ints and value.dtype != torch.long:
                raise ValueError(f"{name} must be int64")
            if name not in bools | ints:
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{name} must be finite floating-point, including padding")
                if value.dtype != self.history.dtype:
                    raise ValueError(f"{name} dtype differs from history")
        if not self.history.is_floating_point() or not bool(torch.isfinite(self.history).all()):
            raise ValueError("history must be finite floating-point")
        for name in ("history_mask","keypose_mask","scene_mask","slot_mask"):
            prefix_mask(getattr(self, name), name)
        if not isinstance(self.fps, (float,int)) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be explicit, finite and positive")
        if bool((self.total_frames < 1).any()):
            raise ValueError("total_frames must be positive")
        if bool((self.minimum_frames[self.slot_mask] < 1).any()):
            raise ValueError("every required slot needs at least one frame")
        if bool((self.minimum_frames[~self.slot_mask] != 0).any()):
            raise ValueError("padding slot minimum must be zero")
        if bool((self.minimum_frames.sum(1) > self.total_frames).any()):
            raise ValueError("total_frames is infeasible for required event durations")
        if bool(((self.slot_types < 0) | (self.slot_types > 3))[self.slot_mask].any()):
            raise ValueError("unsupported event type")
        if bool(((self.semantic_ids < 0) | (self.semantic_ids >= 32))[self.keypose_mask].any()):
            raise ValueError("semantic_ids must be in [0,31]")
        if bool((self.root_tolerance < 0).any()) or bool((self.joint_tolerance < 0).any()):
            raise ValueError("tolerances must be nonnegative")
        if bool((self.joint_tolerance > math.pi).any()):
            raise ValueError("joint_tolerance is measured in radians, at most pi")
        if bool(((self.contact_body_regions < -1) | (self.contact_body_regions >= REGIONS)).any()):
            raise ValueError("contact_body_regions is outside [-1,R-1]")
        active = self.contact_active & self.keypose_mask[..., None]
        normal_norm = self.contact_normals.norm(dim=-1)
        if bool(((normal_norm-1).abs()[active] > 1e-3).any()):
            raise ValueError("active contact normals must be unit length")
        if bool((self.contact_target_ids[active] < 0).any()):
            raise ValueError("active relations need explicit target identity")
        if bool((self.initial_contact_target_ids[self.initial_contact_active] < 0).any()):
            raise ValueError("initial contacts need explicit target identity")
        for row in range(b):
            nk, nm = int(self.keypose_mask[row].sum()), int(self.slot_mask[row].sum())
            anchors = self.keypose_slots[row,:nk]
            if bool(((anchors < 0) | (anchors >= nm)).any()):
                raise ValueError("keypose_slots must point to valid event ends")
            if nk > 1 and bool((anchors[1:] <= anchors[:-1]).any()):
                raise ValueError("keypose anchors must be strictly ordered")
            if bool((self.keypose_slots[row,nk:] != -1).any()):
                raise ValueError("padding keypose_slots must be -1")
            owners = self.slot_keyposes[row,:nm]
            if bool(((owners < 0) | (owners >= nk)).any()):
                raise ValueError("slot_keyposes references a missing keypose")
            if bool((self.slot_keyposes[row,nm:] != -1).any()):
                raise ValueError("padding slot_keyposes must be -1")
            if bool(self.slot_contact_active[row,nm:].any()):
                raise ValueError("padding slots cannot establish contacts")
            if bool(self.slot_release_active[row,nm:].any()):
                raise ValueError("padding slots cannot release contacts")
            if bool((self.slot_contact_active[row] & self.slot_release_active[row]).any()):
                raise ValueError("a relation cannot be required held and released at once")
            available = self.contact_active[row,owners]
            if bool((self.slot_contact_active[row,:nm] & ~available).any()):
                raise ValueError("slot contact is not declared by its referenced keypose")
            if bool((self.slot_release_active[row,:nm] & ~available).any()):
                raise ValueError("slot release is not declared by its referenced keypose")
        # A locked joint refers to a valid rotation, never a zero/noisy 6D vector.
        from .constraints import rotation_6d_valid_mask
        for name, value, mask in (
            ("history",self.history,self.history_mask),
            ("keyposes",self.keyposes,self.keypose_mask),
        ):
            rotations = value[...,3:].reshape(*value.shape[:2], JOINTS, 6)
            valid = rotation_6d_valid_mask(rotations)
            if not bool(valid[mask].all()):
                raise ValueError(f"{name} contains degenerate 6D rotations")
        return self


def future_mask(condition: SequenceCondition, length: int | None = None) -> Tensor:
    length = condition.max_frames if length is None else length
    if length < condition.max_frames:
        raise ValueError("future buffer is shorter than requested generation")
    return torch.arange(length, device=condition.device)[None] < condition.total_frames[:,None]


def expand_contacts(condition: SequenceCondition, schedule, length: int | None = None) -> dict[str, Tensor]:
    """Expand declared relations with PREDICTED times, never GT contact gates."""
    length = condition.max_frames if length is None else length
    frames = torch.arange(1,length+1,device=condition.device)[None,:,None]
    contains_end = (frames <= schedule.boundaries[:,None]) & condition.slot_mask[:,None]
    slots = contains_end.to(torch.long).argmax(dim=-1)
    batch = torch.arange(condition.batch_size, device=condition.device)[:,None]
    owners = condition.slot_keyposes[batch,slots].clamp_min(0)
    valid = future_mask(condition,length)
    active = condition.slot_contact_active[batch,slots] & valid[...,None]
    return {
        "targets": condition.contact_targets[batch,owners],
        "normals": condition.contact_normals[batch,owners],
        "contact_active": active,
        "release_active": condition.slot_release_active[batch,slots] & valid[...,None],
        "target_body_regions": condition.contact_body_regions[batch,owners],
        "target_ids": condition.contact_target_ids[batch,owners],
        "slot_indices": slots,
        "slot_types": condition.slot_types[batch,slots],
        "frame_mask": valid,
    }
