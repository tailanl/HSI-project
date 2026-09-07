"""Neutral current-frame SMPL-X carrier, measured from the actual body."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from hsi.common.artifacts import require, MemoryContractError as ContractError
from hsi.stage3.static_seed import (WORLD_CONVERSION_YUP_TO_ZUP, wrapped_angle,
    anatomical_yaw, rotation_z)

def finite(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == shape and np.isfinite(array).all(), "Invalid " + label)
    return array

def _model_forward(
    model_path: Path, betas: np.ndarray, body_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return neutral/native vertices, J22 and faces on CPU."""
    try:
        import torch
        from smplx import SMPLX
    except ImportError as error:  # pragma: no cover - depends on deployment env
        raise ContractError(
            "Neutral frame generation needs torch, smplx and the explicitly "
            "bound neutral body asset; no historical frame fallback is available"
        ) from error
    model = SMPLX(str(model_path), model_type="smplx", batch_size=1).to("cpu").eval()
    model.requires_grad_(False)
    with torch.no_grad():
        result = model(
            betas=torch.from_numpy(np.asarray(betas, dtype=np.float32)).reshape(1, 10),
            body_pose=torch.from_numpy(np.asarray(body_pose, dtype=np.float32)).reshape(1, 63),
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            transl=torch.zeros((1, 3), dtype=torch.float32),
        )
    return (
        result.vertices[0].cpu().numpy().astype(np.float64),
        result.joints[0, :22].cpu().numpy().astype(np.float64),
        np.ascontiguousarray(model.faces, dtype=np.int64),
    )


def _carrier_from_model(
    model_path: Path, betas: np.ndarray, body_pose: np.ndarray, root: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _vertices, base_joints, _faces = _model_forward(model_path, betas, body_pose)
    converted = base_joints @ WORLD_CONVERSION_YUP_TO_ZUP.T
    yaw_delta = wrapped_angle(float(root[3]) - anatomical_yaw(converted))
    global_rotation = rotation_z(yaw_delta) @ WORLD_CONVERSION_YUP_TO_ZUP
    pelvis_delta_native = base_joints[0]
    translation = root[:3] - pelvis_delta_native
    return global_rotation, translation, pelvis_delta_native


def _require_runtime_pelvis_translation(
    root: np.ndarray, translation: np.ndarray, pelvis_delta_native: np.ndarray
) -> None:
    """Enforce the exact carrier algebra used by the deployed P478 loader.

    SMPL-X applies ``global_orient`` about its shaped pelvis.  Consequently the
    native pelvis offset is not rotated around the world origin: the query
    pelvis is ``pelvis_delta_native + transl``.  Keeping this check in the
    producer prevents a geometry-consistent mesh from carrying a numerically
    incompatible runtime translation.
    """

    error = float(
        np.max(
            np.abs(
                np.asarray(root[:3], dtype=np.float64)
                - (
                    np.asarray(pelvis_delta_native, dtype=np.float64)
                    + np.asarray(translation, dtype=np.float64)
                )
            )
        )
    )
    require(
        error <= 5.0e-5,
        "stored static SMPL-X translation does not place the pelvis at the query root",
    )


def _build_at_explicit_xy(
    start_xy: Sequence[float], model_path: Path, yaw: float, floor_clearance: float
) -> dict[str, Any]:
    start = finite(start_xy, (2,), "explicit start XY")
    require(math.isfinite(yaw), "explicit yaw is non-finite")
    require(0.0 <= floor_clearance <= 0.10,
            "floor clearance must be in [0, 0.10] m")
    betas = np.zeros((10,), dtype=np.float64)
    body = np.zeros((63,), dtype=np.float64)
    native_vertices, native_joints, faces = _model_forward(model_path, betas, body)
    converted_joints = native_joints @ WORLD_CONVERSION_YUP_TO_ZUP.T
    yaw_delta = wrapped_angle(float(yaw) - anatomical_yaw(converted_joints))
    orient = rotation_z(yaw_delta) @ WORLD_CONVERSION_YUP_TO_ZUP
    # SMPL-X global orientation acts about the shaped native pelvis, not the
    # world origin.  Rotating about the origin makes the rendered J22/mesh look
    # internally consistent while storing a ``transl`` that the runtime cannot
    # use.  Preserve the native pelvis and rotate every relative offset around
    # it, matching the deployed SMPL-X forward exactly.
    pelvis_native = np.asarray(native_joints[0], dtype=np.float64)
    vertices = (native_vertices - pelvis_native[None]) @ orient.T + pelvis_native[None]
    joints = (native_joints - pelvis_native[None]) @ orient.T + pelvis_native[None]
    translation = np.asarray((
        float(start[0] - pelvis_native[0]),
        float(start[1] - pelvis_native[1]),
        float(floor_clearance - np.min(vertices[:, 2])),
    ))
    vertices += translation[None]
    joints += translation[None]
    root = np.asarray((joints[0, 0], joints[0, 1], joints[0, 2], anatomical_yaw(joints)))
    _require_runtime_pelvis_translation(root, translation, pelvis_native)
    return {
        "vertices": vertices, "joints": joints, "faces": faces, "root": root,
        "body_pose": body, "betas": betas, "global_orient": orient,
        "translation": translation, "pelvis_delta": pelvis_native,
        "source_receipt": None,
    }
