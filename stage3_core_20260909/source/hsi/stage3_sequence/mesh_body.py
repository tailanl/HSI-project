"""Explicit mesh-backed Stage 3 body adapter; no automatic asset discovery.

The caller supplies an SMPLXLayer (or a matching matrix-pose nn.Module),
shape, fixed hand rotations and named surface-vertex groups.  Motion root
XYZ denotes the *pelvis joint*, not the SMPL-X transl parameter.  The global
rotation must already map the asset's rest frame into metric world Z-up;
this adapter does not silently rotate/scale a body or infer a dataset frame.

Region points are centroids of caller-declared surface vertices, not lowest
points or guaranteed contact patches.  Collision defaults to all vertices,
but vertex sampling still does not prove triangle/inter-frame nonpenetration.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .geometry import MOTION_DIM, NUM_JOINTS, NUM_REGIONS, REGION_NAMES, rotation_6d_to_matrix


@dataclass(frozen=True)
class MeshOutput:
    """World vertices [B,T,V,3] and the first 22 SMPL-X joints [B,T,22,3]."""

    vertices: Tensor
    joints: Tensor


def _fixed_float(value: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(value)
    if not value.is_floating_point():
        value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.detach().clone()


def _hand_matrices(value: Tensor, name: str) -> Tensor:
    value = _fixed_float(value, name)
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4 or value.shape[1:] != (15, 3, 3) or value.shape[0] < 1:
        raise ValueError(f"{name} must be [15,3,3] or [B,15,3,3] rotation matrices")
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    if not torch.allclose(value @ value.transpose(-1, -2), identity.expand_as(value), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} contains non-orthogonal rotation matrices")
    if not torch.allclose(torch.linalg.det(value), torch.ones(value.shape[:2], device=value.device, dtype=value.dtype),
                          atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} must contain proper rotations, not reflections")
    return value


def _vertex_ids(value: Tensor | Sequence[int], name: str) -> Tensor:
    value = torch.as_tensor(value)
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain integer vertex indices")
    if value.ndim != 1 or value.numel() < 1 or (value < 0).any():
        raise ValueError(f"{name} must be a nonempty 1D list of nonnegative vertex indices")
    if value.unique().numel() != value.numel():
        raise ValueError(f"{name} must not contain duplicate vertex indices")
    return value.long().detach().clone()


class MeshBody(nn.Module):
    """Frozen mesh model with differentiable motion inputs and fixed anatomy.

    Required model contract: matrix-valued global_orient [N,3,3], body_pose
    [N,21,3,3], hands [N,15,3,3], pose2rot=False, return_verts=True; output
    exposes ``vertices`` and unremapped ``joints`` with pelvis at index zero.
    Installed smplx.SMPLXLayer meets this contract.  Ordinary smplx.SMPLX does
    not: despite its pose2rot argument, it first reshapes axis-angle poses.

    ``betas`` is [N_beta] or [B,N_beta].  Hands and betas broadcast from one
    body or match B, and are constant across T.  Vertex groups must provide
    all eight REGION_NAMES, in that order when supplied as a sequence.
    """

    def __init__(self, model: nn.Module, betas: Tensor, *, left_hand_pose: Tensor,
                 right_hand_pose: Tensor,
                 region_vertex_ids: Mapping[str, Sequence[int] | Tensor] | Sequence[Sequence[int] | Tensor],
                 collision_vertex_ids: Sequence[int] | Tensor | None = None) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an explicitly supplied matrix-pose nn.Module")
        ancestry = {kind.__name__ for kind in type(model).__mro__}
        if "SMPLX" in ancestry and "SMPLXLayer" not in ancestry:
            raise TypeError("Use SMPLXLayer, not the ordinary axis-angle SMPLX.forward")
        if getattr(model, "joint_mapper", None) is not None:
            raise ValueError("MeshBody requires unmapped SMPL-X joints with pelvis at index zero")
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()
        shape = _fixed_float(betas, "betas")
        if shape.ndim == 1:
            shape = shape[None]
        if shape.ndim != 2 or min(shape.shape) < 1:
            raise ValueError("betas must be [N_beta] or [B,N_beta]")
        expected_betas = getattr(model, "num_betas", shape.shape[-1])
        if shape.shape[-1] != expected_betas:
            raise ValueError("betas dimension does not match the explicitly supplied model")
        self.register_buffer("betas", shape)
        self.register_buffer("left_hand_pose", _hand_matrices(left_hand_pose, "left_hand_pose"))
        self.register_buffer("right_hand_pose", _hand_matrices(right_hand_pose, "right_hand_pose"))
        if isinstance(region_vertex_ids, Mapping):
            if set(region_vertex_ids) != set(REGION_NAMES):
                raise ValueError("region_vertex_ids must declare exactly the eight REGION_NAMES")
            groups = [region_vertex_ids[name] for name in REGION_NAMES]
        else:
            groups = list(region_vertex_ids)
            if len(groups) != NUM_REGIONS:
                raise ValueError("region_vertex_ids must contain eight vertex groups")
        for i, ids in enumerate(groups):
            self.register_buffer(f"region_ids_{i}", _vertex_ids(ids, REGION_NAMES[i]))
        collision = None if collision_vertex_ids is None else _vertex_ids(collision_vertex_ids, "collision_vertex_ids")
        self.register_buffer("collision_vertex_ids", collision)
        self.uses_all_vertices = collision is None
        self.region_reduction = "declared_surface_vertex_centroid"
        template = getattr(model, "v_template", None)
        if isinstance(template, Tensor):
            self._validate_vertex_count(template.shape[-2])

    def train(self, mode: bool = True) -> "MeshBody":
        # Surrounding denoiser.train() must not enable dropout or weight updates
        # in the fixed anatomical model.  No no_grad() surrounds its forward.
        super().train(mode)
        self.model.eval()
        self.model.requires_grad_(False)
        return self

    def _validate_vertex_count(self, count: int) -> None:
        for i in range(NUM_REGIONS):
            if (getattr(self, f"region_ids_{i}") >= count).any():
                raise ValueError(f"{REGION_NAMES[i]} vertex index is outside the supplied mesh")
        if self.collision_vertex_ids is not None and (self.collision_vertex_ids >= count).any():
            raise ValueError("collision_vertex_ids contains an index outside the supplied mesh")

    def _expand_fixed(self, value: Tensor, motion: Tensor) -> Tensor:
        batch, frames = motion.shape[:2]
        if value.shape[0] not in (1, batch):
            raise ValueError("Fixed shape/hand batch must be one or match the motion batch")
        return value.to(motion).expand(batch, *value.shape[1:])[:, None].expand(
            batch, frames, *value.shape[1:]).reshape(batch * frames, *value.shape[1:])

    def _check_model_device(self, motion: Tensor) -> None:
        reference = next((value for value in self.model.parameters() if value.is_floating_point()), None)
        if reference is None:
            reference = next((value for value in self.model.buffers() if value.is_floating_point()), None)
        if reference is not None and (reference.device != motion.device or reference.dtype != motion.dtype):
            raise ValueError("Move MeshBody and motion to the same device and floating dtype before calling")

    def mesh(self, motion: Tensor) -> MeshOutput:
        """Evaluate actual mesh once; callers may reuse it for several losses."""
        if motion.ndim != 3 or motion.shape[-1] != MOTION_DIM or min(motion.shape[:2]) < 1:
            raise ValueError("motion must be nonempty [B,T,135]")
        self._check_model_device(motion)
        batch, frames = motion.shape[:2]
        flat = motion.reshape(batch * frames, MOTION_DIM)
        rotation = rotation_6d_to_matrix(flat[:, 3:].reshape(-1, NUM_JOINTS, 6))
        identity = torch.eye(3, device=motion.device, dtype=motion.dtype).expand(batch * frames, 3, 3)
        output = self.model(
            betas=self._expand_fixed(self.betas, motion),
            global_orient=rotation[:, 0], body_pose=rotation[:, 1:],
            left_hand_pose=self._expand_fixed(self.left_hand_pose, motion),
            right_hand_pose=self._expand_fixed(self.right_hand_pose, motion),
            jaw_pose=identity, leye_pose=identity, reye_pose=identity,
            expression=motion.new_zeros(batch * frames, int(getattr(self.model, "num_expression_coeffs", 10))),
            transl=motion.new_zeros(batch * frames, 3),
            pose2rot=False, return_verts=True, return_full_pose=False,
        )
        if isinstance(output, Mapping):
            vertices, joints = output.get("vertices"), output.get("joints")
        else:
            vertices, joints = getattr(output, "vertices", None), getattr(output, "joints", None)
        if not isinstance(vertices, Tensor) or vertices.ndim != 3 or vertices.shape[0] != batch * frames or vertices.shape[-1] != 3:
            raise ValueError("Matrix-pose model must return vertices [B*T,V,3]")
        if not isinstance(joints, Tensor) or joints.ndim != 3 or joints.shape[0] != batch * frames or joints.shape[1] < NUM_JOINTS or joints.shape[-1] != 3:
            raise ValueError("Model must return at least 22 unremapped joints [B*T,J,3]")
        self._validate_vertex_count(vertices.shape[1])
        # Do not detach this calibration: movement/shape/rotation derivatives
        # of the model pelvis must cancel in the requested world-root frame.
        translation = flat[:, :3] - joints[:, 0]
        vertices = vertices + translation[:, None]
        joints = joints[:, :NUM_JOINTS] + translation[:, None]
        return MeshOutput(vertices.reshape(batch, frames, -1, 3), joints.reshape(batch, frames, NUM_JOINTS, 3))

    def forward(self, motion: Tensor) -> Tensor:
        return self.joints_world(motion)

    def joints_world(self, motion: Tensor) -> Tensor:
        return self.mesh(motion).joints

    def vertices_world(self, motion: Tensor) -> Tensor:
        return self.mesh(motion).vertices

    def region_points_from_vertices(self, vertices: Tensor) -> Tensor:
        if vertices.ndim != 4 or vertices.shape[-1] != 3:
            raise ValueError("vertices must be [B,T,V,3]")
        self._validate_vertex_count(vertices.shape[-2])
        return torch.stack([vertices.index_select(-2, getattr(self, f"region_ids_{i}").to(vertices.device)).mean(-2)
                            for i in range(NUM_REGIONS)], dim=-2)

    def collision_points_from_vertices(self, vertices: Tensor) -> Tensor:
        if vertices.ndim != 4 or vertices.shape[-1] != 3:
            raise ValueError("vertices must be [B,T,V,3]")
        self._validate_vertex_count(vertices.shape[-2])
        if self.collision_vertex_ids is None:
            return vertices
        return vertices.index_select(-2, self.collision_vertex_ids.to(vertices.device))

    def region_points(self, motion: Tensor) -> Tensor:
        return self.region_points_from_vertices(self.vertices_world(motion))

    def collision_points(self, motion: Tensor) -> Tensor:
        return self.collision_points_from_vertices(self.vertices_world(motion))
