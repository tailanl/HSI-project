"""Fixed external SMPL-X carrier, exact SO(3) limits and Z-up materialization."""
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

BODY_COMPONENT_LIMITS_RAD = torch.tensor(
    (
        (1.45, 1.15, 1.15),
        (1.45, 1.15, 1.15),
        (0.60, 0.60, 0.60),
        (2.60, 0.40, 0.40),
        (2.60, 0.40, 0.40),
        (0.55, 0.55, 0.55),
        (0.85, 0.65, 0.65),
        (0.85, 0.65, 0.65),
        (0.55, 0.55, 0.55),
        (0.65, 0.45, 0.45),
        (0.65, 0.45, 0.45),
        (0.80, 0.80, 0.80),
        (0.85, 0.85, 0.85),
        (0.85, 0.85, 0.85),
        (0.85, 0.85, 0.85),
        (2.25, 1.80, 2.25),
        (2.25, 1.80, 2.25),
        (2.65, 0.55, 0.55),
        (2.65, 0.55, 0.55),
        (1.20, 1.20, 1.20),
        (1.20, 1.20, 1.20),
    ),
    dtype=torch.float32,
)

BODY_NORM_LIMITS_RAD = torch.tensor(
    (
        1.75,
        1.75,
        0.85,
        2.70,
        2.70,
        0.80,
        1.05,
        1.05,
        0.80,
        0.80,
        0.80,
        1.05,
        1.20,
        1.20,
        1.10,
        2.75,
        2.75,
        2.75,
        2.75,
        1.65,
        1.65,
    ),
    dtype=torch.float32,
)

class FitContractError(RuntimeError):
    """Raised for fail-closed input or fitting contract violations."""

def require(condition: bool, message: str) -> None:
    if not condition:
        raise FitContractError(message)

class SMPLXLike(Protocol):
    faces: np.ndarray
    lbs_weights: Tensor

    def __call__(
        self,
        *,
        body_pose: Tensor,
        global_orient: Tensor,
        betas: Tensor,
        transl: Tensor,
        return_verts: bool = True,
    ) -> Any:
        ...

def load_fixed_smplx(path: Path, device: torch.device) -> nn.Module:
    from smplx import SMPLX

    model = SMPLX(
        str(Path(path).resolve()),
        model_type="smplx",
        gender="male",
        batch_size=1,
        use_pca=False,
    ).to(device).eval()
    model.requires_grad_(False)
    return model

def axis_angle_to_matrix(value: Tensor) -> Tensor:
    require(value.shape[-1] == 3, "axis angle must end in 3")
    theta2 = value.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1.0e-16))
    x, y, z = value.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(value.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=value.dtype, device=value.device).expand(value.shape[:-1] + (3, 3))
    small = theta2 < 1.0e-8
    a = torch.where(small, 1.0 - theta2 / 6.0 + theta2.square() / 120.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0 + theta2.square() / 720.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(1.0e-16))
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)

def pose_geodesic_from_initialization(body_pose: Tensor, initial_pose: Tensor) -> Tensor:
    current = axis_angle_to_matrix(body_pose.reshape(-1, 3))
    initial = axis_angle_to_matrix(initial_pose.reshape(-1, 3))
    relative = initial.transpose(-1, -2) @ current
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0 + 1.0e-7, 1.0 - 1.0e-7
    )
    return torch.acos(cosine).square().mean()

def pose_limit_terms(body_pose: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    value = body_pose.reshape(21, 3)
    component_excess = F.relu(value.abs() - BODY_COMPONENT_LIMITS_RAD.to(value))
    norm_excess = F.relu(torch.linalg.vector_norm(value, dim=-1) - BODY_NORM_LIMITS_RAD.to(value))
    return (
        component_excess.square().mean(),
        norm_excess.square().mean(),
        component_excess.max(),
        norm_excess.max(),
    )

def _extract_body_output(output: Any) -> tuple[Tensor, Tensor]:
    joints = getattr(output, "joints", None)
    vertices = getattr(output, "vertices", None)
    if joints is None:
        joints = getattr(output, "Jtr", None)
    if vertices is None:
        vertices = getattr(output, "v", None)
    require(isinstance(joints, Tensor) and isinstance(vertices, Tensor), "body output lacks tensor joints/vertices")
    require(joints.ndim == 3 and joints.shape[0] == 1 and joints.shape[1] >= 22, "body joints must be [1,J>=22,3]")
    require(vertices.ndim == 3 and vertices.shape[0] == 1, "body vertices must be [1,V,3]")
    return joints[0, :22], vertices[0]

def materialize_locked_frame(
    body_model: SMPLXLike,
    body_pose: Tensor,
    betas: Tensor,
    root_xyz_world: Tensor,
    pose_frame_to_world_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    """Materialize SMPL-X with planner yaw fixed outside the optimizer."""

    zero = body_pose.new_zeros((1, 3))
    output = body_model(
        body_pose=body_pose.reshape(1, 63),
        global_orient=zero,
        betas=betas.reshape(1, 10),
        transl=zero,
        return_verts=True,
    )
    joints_yup, vertices_yup = _extract_body_output(output)
    joints_zup = torch.stack((joints_yup[:, 0], -joints_yup[:, 2], joints_yup[:, 1]), dim=-1)
    vertices_zup = torch.stack((vertices_yup[:, 0], -vertices_yup[:, 2], vertices_yup[:, 1]), dim=-1)
    pelvis = joints_zup[0]
    rotation = pose_frame_to_world_rotation.to(joints_zup)
    joints_world = (joints_zup - pelvis) @ rotation.T + root_xyz_world
    vertices_world = (vertices_zup - pelvis) @ rotation.T + root_xyz_world
    return joints_world, vertices_world
