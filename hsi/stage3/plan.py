"""Ordered world-keypose parsing, local transforms and scheduler."""
from __future__ import annotations

import argparse

from dataclasses import asdict, dataclass, replace

import json

import math

import os

from pathlib import Path

import pickle

import random

import re

import sys

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import torch


def load_remogen_models(*args, **kwargs):
    from hsi.stage3.models import load_remogen_models as load
    return load(*args, **kwargs)

EXTERNAL_MULTIVIEW_PROVENANCE = frozenset(
    (
        "generated_image_alphapose",
        "external_2d_observations",
        "qwen3_planner_qwen_image_shared_smplx",
    )
)

@dataclass(frozen=True)
class WorldKeyposeNode:
    """One ordered partial pose in the LINGO/ReMoGen world Z-up frame."""

    ordinal: int
    semantic_id: int
    root_xyz_yaw: np.ndarray
    joint_xyz_world: np.ndarray
    joint_mask: np.ndarray
    confidence: float
    joint_confidence: np.ndarray

    def validate(self) -> "WorldKeyposeNode":
        if int(self.ordinal) < 0 or int(self.semantic_id) < 0:
            raise ValueError("ordinal and semantic_id must be non-negative")
        if np.asarray(self.root_xyz_yaw).shape != (4,):
            raise ValueError("root_xyz_yaw must have shape [4]")
        if np.asarray(self.joint_xyz_world).shape != (22, 3):
            raise ValueError("joint_xyz_world must have shape [22,3]")
        if np.asarray(self.joint_mask).shape != (22,):
            raise ValueError("joint_mask must have shape [22]")
        if np.asarray(self.joint_confidence).shape != (22,):
            raise ValueError("joint_confidence must have shape [22]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        mask = np.asarray(self.joint_mask, dtype=bool)
        values = np.asarray(self.joint_xyz_world, dtype=np.float64)
        conf = np.asarray(self.joint_confidence, dtype=np.float64)
        if not np.isfinite(np.asarray(self.root_xyz_yaw, dtype=np.float64)).all():
            raise ValueError("root_xyz_yaw contains non-finite values")
        if mask.any() and not np.isfinite(values[mask]).all():
            raise ValueError("a selected joint contains non-finite values")
        if mask.any() and ((conf[mask] < 0.0).any() or (conf[mask] > 1.0).any()):
            raise ValueError("selected joint confidence must be in [0,1]")
        return self

@dataclass(frozen=True)
class WorldKeyposePlan:
    schema: str
    seq_name: str
    scene_name: str
    text: str
    nodes: Tuple[WorldKeyposeNode, ...]
    provenance: str = "unspecified"

    def validate(self) -> "WorldKeyposePlan":
        if not self.seq_name or not self.scene_name or not self.text:
            raise ValueError("seq_name, scene_name and text are required")
        if not str(self.provenance).strip():
            raise ValueError("keypose provenance must be non-empty")
        if not self.nodes:
            raise ValueError("the plan must contain at least one keypose node")
        ordinals = [int(node.validate().ordinal) for node in self.nodes]
        if any(right <= left for left, right in zip(ordinals, ordinals[1:])):
            raise ValueError("nodes must be stored in strictly increasing ordinal order")
        return self

def _default_provenance_for_schema(schema: str) -> str:
    value = str(schema).lower()
    if "smoke" in value or "oracle" in value:
        return "oracle_gt_projection_geometry_smoke"
    return "unspecified"

def _is_oracle_provenance(provenance: str) -> bool:
    """Recognize GT/oracle aliases conservatively, not only one prefix.

    Historical artifacts used several labels (for example ``heldout_gt``,
    ``gt_projection`` and ``lingo_projection``).  Treating only strings that
    start with ``oracle`` as privileged made those aliases silently look like
    deployment input.
    """

    value = str(provenance).strip().lower()
    tokens = tuple(re.findall(r"[a-z0-9]+", value))
    compact = "".join(tokens)
    if "oracle" in tokens or "gt" in tokens:
        return True
    if "groundtruth" in compact or "futuregroundtruth" in compact:
        return True
    if "lingo" in tokens and "projection" in tokens:
        return True
    return False

def enforce_plan_provenance(
    plan: WorldKeyposePlan,
    *,
    allow_oracle: bool = False,
    allow_unknown: bool = False,
) -> None:
    """Fail closed unless the packet came from the external MV provider."""

    provenance = str(plan.provenance).strip().lower()
    if provenance in EXTERNAL_MULTIVIEW_PROVENANCE:
        return
    if _is_oracle_provenance(provenance):
        if allow_oracle:
            return
        raise ValueError(
            "GT/oracle keypose plan is an upper bound; rerun with "
            "--allow-oracle-plan only for an explicitly labelled diagnostic"
        )
    if allow_unknown:
        return
    if provenance == "unspecified":
        raise ValueError(
            "keypose plan provenance is unspecified; provide provenance in the plan "
            "or pass --allow-unknown-provenance for an explicitly labelled diagnostic"
        )
    raise ValueError(
        "untrusted keypose plan provenance {!r}; deployment rollout accepts only "
        "the explicit external multi-view provider labels {}. Use "
        "--allow-unknown-provenance only for a labelled diagnostic.".format(
            plan.provenance, sorted(EXTERNAL_MULTIVIEW_PROVENANCE)
        )
    )

def _node_from_mapping(item: Dict[str, Any], fallback_ordinal: int) -> WorldKeyposeNode:
    root = np.asarray(item["root_xyz_yaw"], dtype=np.float32)
    xyz = np.zeros((22, 3), dtype=np.float32)
    mask = np.zeros(22, dtype=bool)
    joint_conf = np.zeros(22, dtype=np.float32)
    if "joint_xyz_world" in item:
        xyz[...] = np.asarray(item["joint_xyz_world"], dtype=np.float32)
        mask[...] = np.asarray(item.get("joint_mask", np.ones(22)), dtype=bool)
        joint_conf[...] = np.asarray(
            item.get("joint_confidence", mask.astype(np.float32)), dtype=np.float32
        )
    else:
        # Compact GenZI/planner form: only list the constrained joints.
        for joint in item.get("joints", []):
            joint_id = int(joint["id"])
            if not 0 <= joint_id < 22:
                raise ValueError("joint id {} is outside [0,21]".format(joint_id))
            xyz[joint_id] = np.asarray(joint["xyz_world"], dtype=np.float32)
            mask[joint_id] = True
            joint_conf[joint_id] = float(joint.get("confidence", 1.0))
    return WorldKeyposeNode(
        ordinal=int(item.get("ordinal", fallback_ordinal)),
        semantic_id=int(item.get("semantic_id", 1)),
        root_xyz_yaw=root,
        joint_xyz_world=xyz,
        joint_mask=mask,
        confidence=float(item.get("confidence", 1.0)),
        joint_confidence=joint_conf,
    ).validate()

def _semantic_string_id(value: str) -> int:
    semantic = str(value).lower()
    if semantic == "release":
        return 4
    if semantic == "prepare":
        return 2
    if semantic in {
        "support_contact",
        "distributed_support",
        "manipulation_contact",
        "contact",
    }:
        return 3
    return 1

def _normalize_lingo_sequence_name(value: str) -> str:
    text = str(value)
    if text.startswith("seg_") and text[4:].isdigit():
        return "seg" + text[4:]
    return text

def load_world_keypose_plan(path: Path) -> WorldKeyposePlan:
    """Load the public JSON or NPZ world-space keypose interchange format."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("coordinate_system", "world_zup") != "world_zup":
            raise ValueError("only coordinate_system=world_zup is accepted")
        nodes = tuple(
            _node_from_mapping(item, index)
            for index, item in enumerate(payload.get("nodes", []))
        )
        schema = str(payload.get("schema", "p142_ordered_world_keypose_v1"))
        return WorldKeyposePlan(
            schema=schema,
            seq_name=str(payload["seq_name"]),
            scene_name=str(payload["scene_name"]),
            text=str(payload["text"]),
            nodes=nodes,
            provenance=str(
                payload.get("provenance", _default_provenance_for_schema(schema))
            ),
        ).validate()
    if path.suffix.lower() != ".npz":
        raise ValueError("plan must be .json or .npz")
    with np.load(path, allow_pickle=False) as data:
        # Direct bridge from ``run_lingo_multiview_smoke.py``.  Its sparse
        # joints are pelvis-relative but remain aligned to world axes, so the
        # absolute GenZI joint locations are root + relative position.
        if "partial_root_xyz_z_up" in data.files:
            root = np.asarray(data["partial_root_xyz_z_up"], dtype=np.float32).reshape(1, 3)
            heading = np.asarray(data["partial_heading_xy_z_up"], dtype=np.float32).reshape(1, 2)
            relative = np.asarray(data["partial_joint_xyz_pelvis"], dtype=np.float32).reshape(1, 22, 3)
            masks = np.asarray(data["partial_joint_mask"], dtype=bool).reshape(1, 22)
            confidence_value = float(np.asarray(data["partial_confidence"]).reshape(()))
            semantic_value = str(np.asarray(data["partial_semantic_type"]).reshape(()))
            yaw = np.arctan2(heading[:, 1], heading[:, 0]).astype(np.float32)
            roots = np.concatenate((root, yaw[:, None]), axis=-1)
            absolute = root[:, None, :] + relative
            joint_conf = masks.astype(np.float32) * confidence_value
            node = WorldKeyposeNode(
                ordinal=0,
                semantic_id=_semantic_string_id(semantic_value),
                root_xyz_yaw=roots[0],
                joint_xyz_world=absolute[0],
                joint_mask=masks[0],
                confidence=confidence_value,
                joint_confidence=joint_conf[0],
            ).validate()
            sequence_key = "seq_name" if "seq_name" in data.files else "sample_id"
            scene_key = "scene_name" if "scene_name" in data.files else "scene"
            sample_id = str(np.asarray(data[sequence_key]).reshape(()))
            schema = str(
                np.asarray(data.get("schema", "p142_genzi_multiview_keypose_v1")).reshape(())
            )
            provenance = str(
                np.asarray(
                    data.get(
                        "provenance",
                        np.asarray(_default_provenance_for_schema(schema)),
                    )
                ).reshape(())
            )
            return WorldKeyposePlan(
                schema=schema,
                seq_name=_normalize_lingo_sequence_name(sample_id),
                scene_name=str(np.asarray(data[scene_key]).reshape(())),
                text=str(np.asarray(data["text"]).reshape(())),
                nodes=(node,),
                provenance=provenance,
            ).validate()
        coordinate = str(data.get("coordinate_system", np.asarray("world_zup")).item())
        if coordinate != "world_zup":
            raise ValueError("only coordinate_system=world_zup is accepted")
        roots = np.asarray(data["root_xyz_yaw"], dtype=np.float32)
        xyz = np.asarray(data["joint_xyz_world"], dtype=np.float32)
        masks = np.asarray(data["joint_mask"], dtype=bool)
        count = int(roots.shape[0])
        semantics = np.asarray(data.get("semantic_id", np.ones(count)), dtype=np.int64)
        ordinals = np.asarray(data.get("ordinal", np.arange(count)), dtype=np.int64)
        confidence = np.asarray(data.get("confidence", np.ones(count)), dtype=np.float32)
        joint_conf = np.asarray(
            data.get("joint_confidence", masks.astype(np.float32)), dtype=np.float32
        )
        nodes = tuple(
            WorldKeyposeNode(
                ordinal=int(ordinals[index]),
                semantic_id=int(semantics[index]),
                root_xyz_yaw=roots[index],
                joint_xyz_world=xyz[index],
                joint_mask=masks[index],
                confidence=float(confidence[index]),
                joint_confidence=joint_conf[index],
            ).validate()
            for index in range(count)
        )
        schema = str(data.get("schema", np.asarray("p142_ordered_world_keypose_v1")).item())
        provenance = str(
            data.get(
                "provenance", np.asarray(_default_provenance_for_schema(schema))
            ).item()
        )
        return WorldKeyposePlan(
            schema=schema,
            seq_name=str(data["seq_name"].item()),
            scene_name=str(data["scene_name"].item()),
            text=str(data["text"].item()),
            nodes=nodes,
            provenance=provenance,
        ).validate()

def _batched_translation(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        if value.shape[1] != 1:
            raise ValueError("translation must have shape [B,1,3] or [B,3]")
        return value[:, 0]
    if value.ndim == 2:
        return value
    raise ValueError("translation must have shape [B,1,3] or [B,3]")

def world_points_to_active_local(
    points_world: torch.Tensor,
    global_rotmat: torch.Tensor,
    global_transl: torch.Tensor,
    current_rotmat: torch.Tensor,
    current_transl: torch.Tensor,
) -> torch.Tensor:
    """Invert ``world = Rg @ (Rc @ local + tc) + tg`` for arbitrary points."""

    if points_world.ndim < 3 or points_world.shape[-1] != 3:
        raise ValueError("points_world must have shape [B,...,3]")
    batch = points_world.shape[0]
    if any(value.shape[0] != batch for value in (global_rotmat, current_rotmat)):
        raise ValueError("all transforms must share the point batch")
    global_t = _batched_translation(global_transl)
    current_t = _batched_translation(current_transl)
    flat = points_world.reshape(batch, -1, 3)
    initial = torch.einsum(
        "bij,bnj->bni", global_rotmat.transpose(1, 2), flat - global_t[:, None]
    )
    local = torch.einsum(
        "bij,bnj->bni", current_rotmat.transpose(1, 2), initial - current_t[:, None]
    )
    return local.reshape_as(points_world)

def world_directions_to_active_local(
    directions_world: torch.Tensor,
    global_rotmat: torch.Tensor,
    current_rotmat: torch.Tensor,
) -> torch.Tensor:
    if directions_world.ndim < 3 or directions_world.shape[-1] != 3:
        raise ValueError("directions_world must have shape [B,...,3]")
    batch = directions_world.shape[0]
    flat = directions_world.reshape(batch, -1, 3)
    initial = torch.einsum(
        "bij,bnj->bni", global_rotmat.transpose(1, 2), flat
    )
    local = torch.einsum(
        "bij,bnj->bni", current_rotmat.transpose(1, 2), initial
    )
    return local.reshape_as(directions_world)

def active_local_points_to_world(
    points_local: torch.Tensor,
    global_rotmat: torch.Tensor,
    global_transl: torch.Tensor,
    current_rotmat: torch.Tensor,
    current_transl: torch.Tensor,
) -> torch.Tensor:
    """Forward counterpart of :func:`world_points_to_active_local`."""

    batch = points_local.shape[0]
    flat = points_local.reshape(batch, -1, 3)
    initial = torch.einsum("bij,bnj->bni", current_rotmat, flat)
    initial = initial + _batched_translation(current_transl)[:, None]
    world = torch.einsum("bij,bnj->bni", global_rotmat, initial)
    world = world + _batched_translation(global_transl)[:, None]
    return world.reshape_as(points_local)

def pending_plan_to_local_packet(
    plan: WorldKeyposePlan,
    next_node_index: int,
    maximum_nodes: int,
    global_rotmat: torch.Tensor,
    global_transl: torch.Tensor,
    current_rotmat: torch.Tensor,
    current_transl: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """Transform remaining ordered nodes to a batched ``KeyposePacket``.

    Logical ordinals are reset to ``0..K-1`` because that is the local order
    vocabulary used during adapter training.  The external ordinal is retained
    by the scheduler/metadata and is never interpreted as a frame number.
    """

    from hsi.stage3.keypose_adapter import KeyposePacket

    if maximum_nodes < 1:
        raise ValueError("maximum_nodes must be positive")
    if not 0 <= next_node_index < len(plan.nodes):
        raise IndexError(next_node_index)
    device = global_rotmat.device if device is None else device
    batch = int(global_rotmat.shape[0])
    selected = plan.nodes[next_node_index : next_node_index + maximum_nodes]
    count = len(selected)
    root_world = torch.as_tensor(
        np.stack([node.root_xyz_yaw[:3] for node in selected]), device=device, dtype=dtype
    )[None].expand(batch, -1, -1)
    joints_world = torch.as_tensor(
        np.stack([node.joint_xyz_world for node in selected]), device=device, dtype=dtype
    )[None].expand(batch, -1, -1, -1)
    yaw_world = torch.as_tensor(
        np.asarray([node.root_xyz_yaw[3] for node in selected]), device=device, dtype=dtype
    )[None].expand(batch, -1)
    directions_world = torch.stack(
        (torch.cos(yaw_world), torch.sin(yaw_world), torch.zeros_like(yaw_world)), dim=-1
    )
    root_local = world_points_to_active_local(
        root_world, global_rotmat, global_transl, current_rotmat, current_transl
    )
    joints_local_absolute = world_points_to_active_local(
        joints_world, global_rotmat, global_transl, current_rotmat, current_transl
    )
    directions_local = world_directions_to_active_local(
        directions_world, global_rotmat, current_rotmat
    )
    yaw_local = torch.atan2(directions_local[..., 1], directions_local[..., 0])

    roots = torch.zeros(batch, maximum_nodes, 4, device=device, dtype=dtype)
    joint_xyz = torch.zeros(batch, maximum_nodes, 22, 3, device=device, dtype=dtype)
    joint_mask = torch.zeros(batch, maximum_nodes, 22, device=device, dtype=torch.bool)
    semantics = torch.zeros(batch, maximum_nodes, device=device, dtype=torch.long)
    ordinals = torch.arange(maximum_nodes, device=device, dtype=torch.long)[None].expand(batch, -1).clone()
    confidence = torch.zeros(batch, maximum_nodes, device=device, dtype=dtype)
    node_mask = torch.zeros(batch, maximum_nodes, device=device, dtype=torch.bool)
    joint_confidence = torch.zeros(batch, maximum_nodes, 22, device=device, dtype=dtype)
    roots[:, :count, :3] = root_local
    roots[:, :count, 3] = yaw_local
    joint_xyz[:, :count] = joints_local_absolute - root_local[:, :, None]
    node_mask[:, :count] = True
    for local_index, node in enumerate(selected):
        mask = torch.as_tensor(node.joint_mask, device=device, dtype=torch.bool)
        joint_mask[:, local_index] = mask
        semantics[:, local_index] = int(node.semantic_id)
        confidence[:, local_index] = float(node.confidence)
        joint_confidence[:, local_index] = torch.as_tensor(
            node.joint_confidence, device=device, dtype=dtype
        )
    return KeyposePacket(
        root_xyz_yaw=roots,
        joint_xyz=joint_xyz,
        joint_mask=joint_mask,
        semantic_id=semantics,
        ordinal=ordinals,
        confidence=confidence,
        node_mask=node_mask,
        joint_confidence=joint_confidence,
    ).validate(require_sorted_ordinals=True)

def heading_yaw_from_joints(joints_world: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints_world, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError("joints_world must have shape [T,22,3]")
    right = joints[:, 2] - joints[:, 1]
    forward = np.stack((-right[:, 1], right[:, 0]), axis=-1)
    norm = np.linalg.norm(forward, axis=-1, keepdims=True)
    forward = forward / np.maximum(norm, 1.0e-8)
    return np.arctan2(forward[:, 1], forward[:, 0])

def angular_distance(left: np.ndarray, right: float) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(left - right), np.cos(left - right)))

@dataclass(frozen=True)
class CompletionThresholds:
    root_xy_m: float = 0.35
    root_z_m: float = 0.25
    yaw_rad: float = math.radians(45.0)
    joint_m: float = 0.30

class OrderedKeyposeScheduler:
    """Frame-free state machine that consumes nodes only in logical order."""

    def __init__(
        self,
        plan: WorldKeyposePlan,
        thresholds: CompletionThresholds,
        max_attempts_per_node: int = 0,
    ) -> None:
        self.plan = plan.validate()
        self.thresholds = thresholds
        self.max_attempts_per_node = int(max_attempts_per_node)
        if self.max_attempts_per_node < 0:
            raise ValueError("max_attempts_per_node must be non-negative")
        self.next_node_index = 0
        self.attempts = 0
        self.events: List[Dict[str, Any]] = []

    @property
    def complete(self) -> bool:
        return self.next_node_index >= len(self.plan.nodes)

    def _frame_metrics(self, node: WorldKeyposeNode, joints: np.ndarray, yaw: np.ndarray):
        root_delta = joints[:, 0] - node.root_xyz_yaw[:3][None]
        root_xy = np.linalg.norm(root_delta[:, :2], axis=-1)
        root_z = np.abs(root_delta[:, 2])
        yaw_error = angular_distance(yaw, float(node.root_xyz_yaw[3]))
        if node.joint_mask.any():
            selected = np.flatnonzero(node.joint_mask)
            distance = np.linalg.norm(
                joints[:, selected] - node.joint_xyz_world[selected][None], axis=-1
            )
            weights = np.maximum(node.joint_confidence[selected], 1.0e-6)
            joint_error = (distance * weights[None]).sum(axis=-1) / weights.sum()
        else:
            joint_error = np.zeros(len(joints), dtype=np.float64)
        return root_xy, root_z, yaw_error, joint_error

    def observe_primitive(self, joints_world: np.ndarray, primitive_id: int) -> List[Dict[str, Any]]:
        """Consume any sequential nodes reached by the eight generated frames."""

        joints = np.asarray(joints_world, dtype=np.float64)
        yaw = heading_yaw_from_joints(joints)
        start_frame = 0
        consumed: List[Dict[str, Any]] = []
        while not self.complete and start_frame < len(joints):
            node = self.plan.nodes[self.next_node_index]
            metrics = self._frame_metrics(node, joints[start_frame:], yaw[start_frame:])
            root_xy, root_z, yaw_error, joint_error = metrics
            valid = (
                (root_xy <= self.thresholds.root_xy_m)
                & (root_z <= self.thresholds.root_z_m)
                & (yaw_error <= self.thresholds.yaw_rad)
            )
            if node.joint_mask.any():
                valid &= joint_error <= self.thresholds.joint_m
            reached = np.flatnonzero(valid)
            if not len(reached):
                best = int(np.argmin(root_xy + root_z + 0.1 * yaw_error + joint_error))
                event = {
                    "primitive_id": int(primitive_id),
                    "external_ordinal": int(node.ordinal),
                    "status": "pending",
                    "best_frame_in_primitive": int(start_frame + best),
                    "root_xy_m": float(root_xy[best]),
                    "root_z_m": float(root_z[best]),
                    "yaw_rad": float(yaw_error[best]),
                    "joint_m": float(joint_error[best]),
                }
                self.attempts += 1
                if self.max_attempts_per_node and self.attempts >= self.max_attempts_per_node:
                    event["status"] = "forced_advance_after_attempt_budget"
                    self.next_node_index += 1
                    self.attempts = 0
                self.events.append(event)
                consumed.append(event)
                break
            local_frame = int(reached[0])
            frame = start_frame + local_frame
            event = {
                "primitive_id": int(primitive_id),
                "external_ordinal": int(node.ordinal),
                "status": "reached",
                "frame_in_primitive": int(frame),
                "root_xy_m": float(root_xy[local_frame]),
                "root_z_m": float(root_z[local_frame]),
                "yaw_rad": float(yaw_error[local_frame]),
                "joint_m": float(joint_error[local_frame]),
            }
            self.events.append(event)
            consumed.append(event)
            self.next_node_index += 1
            self.attempts = 0
            start_frame = frame + 1
        return consumed

class _ClassifierFreeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if not model.cond_mask_prob > 0:
            raise ValueError("released denoiser does not support classifier-free guidance")

    def forward(self, x, timesteps, y=None):
        conditional = dict(y or {})
        conditional["uncond"] = False
        output = self.model(x, timesteps, conditional)
        unconditional = dict(y or {})
        unconditional["uncond"] = True
        output_unconditional = self.model(x, timesteps, unconditional)
        return output_unconditional + conditional["scale"] * (output - output_unconditional)

def _build_occ_two(
    dataset,
    grid,
    scene_name,
    global_rotmat,
    current_rotmat,
    current_root_world,
    goal_world,
):
    batch = int(current_root_world.shape[0])
    current_ground = current_root_world.clone()
    current_ground[..., 2] = 0.0
    goal_ground = goal_world.clone()
    goal_ground[..., 2] = 0.0
    # active-local -> world is R_global @ (R_current @ p + t_current)
    # plus the global translation.  The query centres are already expressed
    # in world coordinates, so only the composed orientation belongs here.
    # Omitting R_current makes occupancy disagree with history/keyposes after
    # the first generated turn.
    active_world_rotmat = torch.matmul(global_rotmat, current_rotmat)
    points_current = (
        torch.matmul(grid, active_world_rotmat.transpose(1, 2))
        + current_ground[:, None]
    )
    points_goal = (
        torch.matmul(grid, active_world_rotmat.transpose(1, 2))
        + goal_ground[:, None]
    )
    names = [scene_name] * batch
    nx, ny, nz = dataset.nb_voxels
    occ_current = dataset.get_occ_for_points(points_current, names).reshape(batch, nx, ny, nz).float()
    occ_goal = dataset.get_occ_for_points(points_goal, names).reshape(batch, nx, ny, nz).float()
    return torch.cat((occ_current.permute(0, 3, 1, 2), occ_goal.permute(0, 3, 1, 2)), dim=1)

def _tensor_tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensor_tree_cpu(item) for key, item in value.items()}
    return value
