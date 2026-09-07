"""Thin joint-driven body adapter over externally installed official SMPL-X.

This module does not vendor the mixed-source historical renderer or SMPL-X
implementation. Shape blend, joint regression and rotation FK use the external
SMPL-X API. Standard LBS translations are then set from all 22 actual generated
body-joint positions; this is not rotation-only FK posing. Unmodelled face/hand
joints remain attached to their real parents. The source positions are never
edited to make a mesh pass an evaluator.
"""
from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np

from hsi.common.artifacts import artifact, require, verified, write_once


def _joint_driven_chunk(model, rotations, positions, betas, hand_rotations):
    """Standard affine skinning using exact emitted body positions.

    SMPL-X's root/body indices are 0:22, face 22:25, hands 25:55.
    For any joint, the skinning transform is [R, p - R J_rest].
    """
    import torch
    from smplx.lbs import blend_shapes, vertices2joints, batch_rigid_transform

    count = len(positions)
    parents = model.parents
    require(len(parents) == 55 and int(model.NUM_BODY_JOINTS) == 21,
            "Expected official 55-joint SMPL-X with 22 body joints")
    shaped = model.v_template.unsqueeze(0) + blend_shapes(betas, model.shapedirs[..., :10])
    rest = vertices2joints(model.J_regressor, shaped)
    local = torch.eye(3, dtype=rotations.dtype, device=rotations.device).expand(count,55,3,3).clone()
    local[:, :22] = rotations
    local[:, 25:] = hand_rotations.unsqueeze(0)
    corrective = (local[:, 1:] - torch.eye(3, dtype=local.dtype, device=local.device)).reshape(count,-1)
    posed_vertices = shaped + (corrective @ model.posedirs).reshape(count,10475,3)
    _, transforms = batch_rigid_transform(local, rest, parents, dtype=local.dtype)
    orientation = transforms[:, :, :3, :3]
    driver = torch.empty_like(rest)
    driver[:, :22] = positions
    for joint in range(22, 55):
        parent = int(parents[joint])
        require(0 <= parent < joint, "SMPL-X auxiliary joints need ordered parents")
        offset = rest[:, joint] - rest[:, parent]
        driver[:, joint] = driver[:, parent] + (orientation[:, parent] @ offset.unsqueeze(-1)).squeeze(-1)
    affine = transforms[:, :, :3, :].clone()
    affine[:, :, :, 3] = driver - (orientation @ rest.unsqueeze(-1)).squeeze(-1)
    weights = model.lbs_weights
    sums = weights.sum(dim=1, keepdim=True)
    weights = weights / sums.clamp_min(1e-8)
    empty = sums[:, 0] < 1e-8
    if bool(empty.any()):
        weights = weights.clone()
        weights[empty] = 0
        weights[empty, 0] = 1
    # Matrix products apply the ordinary LBS equation without another pose/FK pass.
    blended = torch.matmul(weights, affine.reshape(count,55,12)).reshape(count,10475,3,4)
    homogeneous = torch.cat((posed_vertices, torch.ones_like(posed_vertices[..., :1])), dim=-1)
    vertices = torch.matmul(blended, homogeneous.unsqueeze(-1)).squeeze(-1)
    return vertices, driver[:, :22], vertices2joints(model.J_regressor, vertices)[:, :22]


def skin_joint_driven(local_rot_mats, posed_joints_world, betas, *, body_model_path,
                      mean_hands_path=None, device="cpu", chunk_size=32):
    """All-frame 10,475-vertex mesh using an explicit licensed neutral model.

    No beta.npy, dataset motion, old renderer module or implicit model path is
    read. ``mean_hands_path=None`` means the original explicit zero-hand fallback.
    """
    import torch
    import smplx
    from smplx.lbs import batch_rodrigues

    require(device in ("cpu", "cuda") and type(chunk_size) is int and chunk_size > 0,
            "Explicit supported skinning device and positive chunk size required")
    rotations, joints, shape = map(np.asarray, (local_rot_mats, posed_joints_world, betas))
    frames = len(joints)
    for value, expected in ((rotations,(frames,22,3,3)), (joints,(frames,22,3)), (shape,(frames,10))):
        require(value.dtype == np.float32 and value.shape == expected and frames > 0 and np.isfinite(value).all(),
                "Skinning requires finite float32 actual motion arrays, preserving all frames")
    require(np.max(np.abs(np.swapaxes(rotations,-1,-2)@rotations-np.eye(3))) <= 2e-3
            and np.max(np.abs(np.linalg.det(rotations)-1)) <= 2e-3, "Skinning rotations are not SO(3)")
    body_model_path = Path(body_model_path).resolve(strict=True)
    source_model = artifact(body_model_path)
    hands = np.zeros((30,3), np.float32)
    hand_source = None
    if mean_hands_path is not None:
        hand_source = artifact(mean_hands_path)
        array = np.load(verified(hand_source), allow_pickle=False)
        require(array.shape == (90,) and np.isfinite(array).all(), "Expected explicit finite 90-D mean hands")
        hands = np.asarray(array, dtype=np.float32).reshape(30,3)
    model = smplx.SMPLXLayer(str(body_model_path), gender="neutral", num_betas=10,
                            use_pca=False, flat_hand_mean=True).to(device).eval()
    require(tuple(model.v_template.shape) == (10475,3)
            and tuple(model.lbs_weights.shape) == (10475,55), "External model is not complete SMPL-X")
    require(int(model.parents[0]) == -1
            and all(0 <= int(model.parents[i]) < i for i in range(1,55)), "Unexpected SMPL-X parent order")
    faces = np.asarray(model.faces, dtype=np.int64)
    require(faces.shape == (20908,3) and faces.min() >= 0 and faces.max() < 10475,
            "External SMPL-X body topology drift")
    chunks, control_error, regressed_error = [], [], []
    with torch.inference_mode():
        hand_rotations = batch_rodrigues(torch.as_tensor(hands, dtype=torch.float32, device=device))
        for begin in range(0, frames, chunk_size):
            end = min(frames, begin+chunk_size)
            r, p, b = [torch.as_tensor(value[begin:end], dtype=torch.float32, device=device)
                       for value in (rotations,joints,shape)]
            vertices, drivers, regressed = _joint_driven_chunk(model,r,p,b,hand_rotations)
            require(bool(torch.isfinite(vertices).all()), "Nonfinite full-body skinning result")
            chunks.append(vertices.cpu().numpy().astype(np.float32))
            control_error.append(float((drivers-p).abs().max()))
            regressed_error.append(float(torch.linalg.vector_norm(regressed-p,dim=-1).max()))
    require(max(control_error) == 0.0, "Skinning did not preserve every generated body-joint driver")
    verified(source_model)
    if hand_source is not None:
        verified(hand_source)
    external_modules = {name: artifact(importlib.import_module(name).__file__)
                        for name in ("smplx.body_models", "smplx.lbs")}
    return np.concatenate(chunks), faces, {
        "schema": "hsi.stage3.joint_driven_smplx_audit.v1", "frame_count": frames, "vertex_count": 10475,
        "all_generated_body_joints_are_exact_drivers": True, "body_joint_driver_max_abs_error_m": max(control_error),
        "regressed_joint_error_max_m_diagnostic_only": max(regressed_error),
        "regressed_joints_never_replace_generated_joints": True,
        "body_model": source_model, "mean_hands": hand_source, "external_implementations": external_modules,
        "mapping": {"body": [0,22], "identity_face": [22,25], "mean_hands": [25,55]},
        "shape_source": "actual_generated_motion_betas", "beta_npy_read": False,
        "fk_only_body_positions_used": False, "historical_renderer_source_vendored": False,
        "source": artifact(__file__)}


def skin_motion(motion, motion_path, metadata_path, output, *, body_model_record,
                expected_faces, mean_hands_path=None, device="cpu"):
    """Publish source-bound full mesh, checking exact Stage2 body topology."""
    output = Path(output)
    model_path = verified(body_model_record)
    vertices, faces, audit = skin_joint_driven(motion["rotations"], motion["joints"], motion["betas"],
        body_model_path=model_path, mean_hands_path=mean_hands_path, device=device)
    require(np.array_equal(faces,np.asarray(expected_faces)), "Stage2 and motion full-mesh topology differ")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "skinned_vertices.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, vertices_world=vertices, faces=faces)
    receipt_path = output / "skinning_receipt.json"
    write_once(receipt_path, {"schema": "hsi.stage3.actual_joint_driven_skinning.v1",
        "source_motion": artifact(motion_path), "source_metadata": artifact(metadata_path),
        "source_body_model": body_model_record, "outputs": {"vertices": artifact(path)}, "audit": audit,
        "all_source_frames_preserved": True, "source_motion_edited": False,
        "frame_count": len(vertices), "vertex_count": 10475, "source": artifact(__file__)})
    return path, receipt_path
