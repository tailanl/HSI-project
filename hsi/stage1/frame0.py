"""Query-owned neutral current state; external SMPL-X model is an explicit argument."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from hsi.common.artifacts import artifact, require, sha256 as sha256_file
from ._roadmap import atomic_json
FRAME0_SCHEMA = "p366.query_current_frame0.v1"

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value

def wrapped_angle(value: float) -> float:
    return float(math.atan2(math.sin(float(value)), math.cos(float(value))))

def angle_error(left: float, right: float) -> float:
    return abs(wrapped_angle(float(left) - float(right)))

def anatomical_yaw_world_zup(joints: np.ndarray) -> float:
    joints = np.asarray(joints, dtype=np.float64)
    require(joints.shape == (22, 3), "current frame0 must contain SMPL-X22 joints")
    require(bool(np.isfinite(joints).all()), "current frame0 joints are non-finite")
    hip_axis = joints[2, :2] - joints[1, :2]
    require(float(np.linalg.norm(hip_axis)) > 1.0e-6, "current frame0 hip axis is degenerate")
    return wrapped_angle(math.atan2(float(hip_axis[0]), float(-hip_axis[1])))

def world_zup(points_native_yup: np.ndarray) -> np.ndarray:
    points = np.asarray(points_native_yup, dtype=np.float64)
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)

def rotate_z(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    rotation = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return (np.asarray(points, dtype=np.float64) - pivot[None]) @ rotation.T + pivot[None]

def build(
    query_path: Path,
    smplx_model_path: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    initial_yaw_rad: float,
    floor_clearance_m: float,
) -> dict[str, Any]:
    query_path = Path(query_path).resolve()
    smplx_model_path = Path(smplx_model_path).resolve()
    output_path = Path(output_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    for path, label in ((query_path, "P127 query"), (smplx_model_path, "SMPL-X model")):
        require(path.is_file(), f"missing {label}: {path}")
    require(not output_path.exists(), f"refusing to overwrite current frame: {output_path}")
    require(not receipt_path.exists(), f"refusing to overwrite receipt: {receipt_path}")
    require(math.isfinite(initial_yaw_rad), "initial yaw is non-finite")
    require(0.0 <= floor_clearance_m <= 0.10, "floor clearance must be in [0,0.10] m")

    query = read_json(query_path)
    require(query.get("schema") == "p123_causal_scene_text_query_v1", "wrong P127 query schema")
    start_xy = np.asarray(query.get("start_world_xz_m"), dtype=np.float64)
    require(start_xy.shape == (2,) and bool(np.isfinite(start_xy).all()), "invalid query start")

    # Importing SMPL-X is intentionally deferred so contract/unit tests remain
    # lightweight and do not need to instantiate the 100+ MB body model.
    import torch
    from smplx import SMPLX

    torch.set_grad_enabled(False)
    model = SMPLX(str(smplx_model_path), model_type="smplx", batch_size=1).to("cpu").eval()
    model.requires_grad_(False)
    body_pose = torch.zeros((1, 63), dtype=torch.float32)
    global_orient = torch.zeros((1, 3), dtype=torch.float32)
    betas = torch.zeros((1, 10), dtype=torch.float32)
    with torch.no_grad():
        result = model(
            body_pose=body_pose,
            global_orient=global_orient,
            betas=betas,
            transl=torch.zeros((1, 3), dtype=torch.float32),
        )
    vertices = world_zup(result.vertices[0].cpu().numpy())
    joints = world_zup(result.joints[0, :22].cpu().numpy())
    faces = np.ascontiguousarray(model.faces, dtype=np.int64)
    require(vertices.shape == (10475, 3), "unexpected SMPL-X vertex shape")
    require(joints.shape == (22, 3), "unexpected SMPL-X22 joint shape")

    requested_yaw = wrapped_angle(initial_yaw_rad)
    canonical_yaw = anatomical_yaw_world_zup(joints)
    yaw_delta = wrapped_angle(requested_yaw - canonical_yaw)
    pelvis = joints[0].copy()
    vertices = rotate_z(vertices, yaw_delta, pelvis)
    joints = rotate_z(joints, yaw_delta, pelvis)
    translation = np.asarray(
        (
            float(start_xy[0] - joints[0, 0]),
            float(start_xy[1] - joints[0, 1]),
            float(floor_clearance_m - np.min(vertices[:, 2])),
        ),
        dtype=np.float64,
    )
    vertices += translation[None]
    joints += translation[None]
    observed_yaw = anatomical_yaw_world_zup(joints)
    root = np.asarray((joints[0, 0], joints[0, 1], joints[0, 2], observed_yaw), dtype=np.float64)
    require(float(np.linalg.norm(root[:2] - start_xy)) <= 2.0e-6, "placed root differs from query start")
    require(angle_error(observed_yaw, requested_yaw) <= 2.0e-6, "placed anatomical yaw differs from request")
    require(abs(float(np.min(vertices[:, 2])) - floor_clearance_m) <= 2.0e-6, "body was not grounded")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(FRAME0_SCHEMA),
        vertices_world=np.ascontiguousarray(vertices, dtype=np.float32),
        joints_world=np.ascontiguousarray(joints, dtype=np.float32),
        faces=faces,
        body_pose_axis_angle=np.zeros((63,), dtype=np.float32),
        global_orient_axis_angle=np.zeros((3,), dtype=np.float32),
        betas=np.zeros((10,), dtype=np.float32),
        root_xyz_yaw_world_zup=root,
        source_kind=np.asarray("canonical_query_initialization_not_observed_motion"),
        query_start_world_xz_m=start_xy,
        explicit_initial_yaw_rad=np.asarray(requested_yaw, dtype=np.float64),
        floor_clearance_m=np.asarray(floor_clearance_m, dtype=np.float64),
        future_ground_truth_used=np.asarray(False),
        interaction_memory_used=np.asarray(False),
        retrieval_used=np.asarray(False),
        pose_template_used=np.asarray(False),
    )
    payload = {
        "schema": "p366.query_current_frame0_receipt.v1",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": str(query["scene_id"]),
        "instruction": str(query["instruction"]),
        "current_frame": artifact(output_path),
        "query": artifact(query_path),
        "smplx_model": artifact(smplx_model_path),
        "source_kind": "canonical_query_initialization_not_observed_motion",
        "source_frame_indices_read": [],
        "query_start_world_xz_m": start_xy.astype(float).tolist(),
        "initial_yaw_rad": float(observed_yaw),
        "initial_yaw_source": "explicit_query_initialization_then_verified_from_anatomical_hip_axis",
        "pelvis_root_xyz_world_zup": root[:3].astype(float).tolist(),
        "minimum_vertex_height_m": float(np.min(vertices[:, 2])),
        "future_ground_truth_used": False,
        "lingo_motion_opened": False,
        "interaction_memory_used": False,
        "retrieval_used": False,
        "pose_template_used": False,
        "sensor_observation_claimed": False,
        "generated_motion_claimed": False,
        "action_specific_pose_used": False,
        "deterministic_given_query_model_and_initial_yaw": True,
    }
    atomic_json(receipt_path, payload)
    return payload

