"""Pose-derived root: fixed gluteal support, no free root regression."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence, Protocol
from dataclasses import dataclass
import math
import hashlib
import json
import os
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from . import body_model as body_util

class SparsePoseError(RuntimeError):
    """Fail-closed lineage, optimization, or audit contract error."""

@dataclass(frozen=True)
class State:
    pose: Tensor
    root: Tensor
    joints: Tensor
    vertices: Tensor
    contact_offset: Tensor

def materialize_pose_derived_root(
    body_model: nn.Module,
    pose: Tensor,
    betas: Tensor,
    frame: Tensor,
    contact_goal: Tensor,
    fixed_subset: Tensor,
    contact_anchor_policy: str = "fixed_subset_mean_xyz",
    support_envelope_fraction: float = 0.15,
) -> State:
    zero = contact_goal.new_zeros(3)
    joints_local, vertices_local = body_util.materialize_locked_frame(body_model, pose, betas, zero, frame)
    points = vertices_local.index_select(0, fixed_subset)
    if contact_anchor_policy == "fixed_subset_mean_xyz":
        contact_offset = points.mean(dim=0)
    elif contact_anchor_policy == "fixed_subset_xy_mean_z_lower_envelope":
        count = min(
            int(points.shape[0]),
            max(8, int(math.ceil(float(points.shape[0]) * support_envelope_fraction))),
        )
        lower_z = torch.topk(points[:, 2], k=count, largest=False).values.mean()
        contact_offset = torch.cat((points[:, :2].mean(dim=0), lower_z.reshape(1)))
    else:
        raise SparsePoseError(f"unsupported contact-anchor policy: {contact_anchor_policy}")
    root = contact_goal - contact_offset
    return State(
        pose=pose,
        root=root,
        joints=joints_local + root,
        vertices=vertices_local + root,
        contact_offset=contact_offset,
    )
