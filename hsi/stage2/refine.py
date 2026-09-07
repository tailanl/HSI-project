"""Materialized current neutral-sit objective, strict gates and bounded 2-mm search.

This numerical receipt alone never grants the semantic Stage2 publication.
"""
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

from datetime import datetime, timezone
from . import bound_sdf
from . import refine_contract as contract
from . import refine_target as p523
from . import body_geometry as p498
from .refine_margins import strict_terms, bilateral_support_loss, violation_rank
from .posture_quality import metrics as posture_metrics, loss as posture_loss

SOURCE_SCHEMA = "p533.h3_hybrikx_stage1_conditioned_smplx_keypose.v1"

KEYPOSE_SCHEMA = "p533.h3_hybrikx_hybrid_surface_bound_smplx_keypose.v1"

RECEIPT_SCHEMA = "p533.h3_hybrikx_hybrid_stage2_refine_receipt.v1"

OBSERVATION_SCHEMA = "p533.lingo_crop_letterbox_projection_observation.v1"

ABLATION_MODES = ("p523_only", "hybrid", "hybrid_reachable")

CONTACT_SUPPORT_ROWS = p523.CONTACT_SUPPORT_ROWS

COLLISION_ESCAPE_ROWS = p523.COLLISION_ESCAPE_ROWS

ACTIVE_ROWS = p523.ACTIVE_ROWS

CONTACT_JOINT_IDS = p523.CONTACT_JOINT_IDS

P523_SUPPORT_LIMIT_RAD = p523.CONTACT_SUPPORT_DELTA_LIMIT_RAD

P523_ESCAPE_LIMIT_RAD = p523.COLLISION_ESCAPE_DELTA_LIMIT_RAD

SUPPORT_OVERRIDE_LIMIT_RAD = 0.75

COLLIDING_CHAIN_LIMIT_RAD = 0.35

NONCOLLIDING_LEG_LIMIT_RAD = 0.12

UNRELATED_LIMIT_RAD = 0.07

INITIAL_COLLISION_TOLERANCE_M = 0.005

LOWER_LIMB_JOINT_IDS = (4, 5, 7, 8, 10, 11)

LOWER_LIMB_ROWS = (3, 4, 6, 7, 9, 10)

KINEMATIC_CHAINS = (
    (0, 3, 6, 9),
    (1, 4, 7, 10),
    (2, 5, 8, 11, 14),
    (12, 15, 17, 19),
    (13, 16, 18, 20),
)

SOURCE_LINEAGE_KEYS = (
    "source_h3_model_manifest_sha256",
    "source_h3_generation_receipt_sha256",
    "source_h3_frame_sha256",
    "source_hybrikx_model_sha256",
    "source_hybrikx_raw_recovery_sha256",
    "source_hybrikx_recovery_receipt_sha256",
)

REACHABLE_SUPPORT_JOINT_IDS = (1, 2, 4, 5, 7, 8, 10, 11)

REACHABLE_BILATERAL_JOINT_PAIRS = ((1, 2), (4, 5), (7, 8), (10, 11))

REACHABLE_SHIN_JOINT_PAIRS = ((4, 7), (5, 8))

REACHABLE_MAX_SUPPORT_DELTA_MEAN_M = 0.108

REACHABLE_MAX_SUPPORT_DELTA_MAX_M = 0.220

REACHABLE_MAX_SHIN_ANGLE_DEG = 45.0

REACHABLE_MAX_BILATERAL_DELTA_ASYMMETRY_M = 0.120

REACHABLE_MAX_SUPPORT_POSE_GEODESIC_RAD = 0.600

REACHABLE_LIMIT_PROFILES = (
    {
        "name": "raw_tight",
        "quiet_hip_knee_ankle_rad": (0.18, 0.22, 0.12),
        "colliding_hip_knee_ankle_rad": (0.34, 0.38, 0.20),
        "colliding_foot_rad": 0.18,
    },
    {
        "name": "balanced",
        "quiet_hip_knee_ankle_rad": (0.22, 0.28, 0.14),
        "colliding_hip_knee_ankle_rad": (0.40, 0.46, 0.22),
        "colliding_foot_rad": 0.20,
    },
    {
        "name": "knee_reach",
        "quiet_hip_knee_ankle_rad": (0.20, 0.32, 0.16),
        "colliding_hip_knee_ankle_rad": (0.46, 0.56, 0.27),
        "colliding_foot_rad": 0.24,
    },
    {
        "name": "symmetric_medium",
        "quiet_hip_knee_ankle_rad": (0.26, 0.36, 0.18),
        "colliding_hip_knee_ankle_rad": (0.55, 0.50, 0.26),
        "colliding_foot_rad": 0.24,
    },
)

def _scalar(archive: Mapping[str, np.ndarray], key: str) -> Any:
    contract.require(key in archive, f"P533 source lacks {key}")
    value = np.asarray(archive[key])
    contract.require(value.size == 1, f"P533 source {key} must be scalar")
    return value.reshape(()).item()

def _valid_sha(value: object, label: str) -> str:
    text = str(value)
    contract.require(len(text) == 64 and all(c in "0123456789abcdef" for c in text), f"invalid {label}")
    return text

def _artifact_from_declared(path: Path, expected_sha: str | None, label: str) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    actual = contract.sha256_file(path)
    if expected_sha is not None:
        contract.require(actual == _valid_sha(expected_sha, f"{label} sha256"), f"{label} hash drift")
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}

def _load_source(source_path: Path, target_id: str, surface_hash: str, world_hash: str) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as archive:
        contract.require(str(_scalar(archive, "schema")) == SOURCE_SCHEMA, "source is not a P533 H3/HybrIK-X keypose")
        contract.require("published" in str(_scalar(archive, "status")), "P533 source is not published")
        for flag in ("h3_model_used", "hybrikx_model_used", "camera_world_placement_discarded"):
            contract.require(bool(_scalar(archive, flag)), f"P533 source requires {flag}=true")
        contract.require(not bool(_scalar(archive, "old_world_placement_consumed")), "HybrIK camera placement was consumed")
        contract.require(str(_scalar(archive, "selected_candidate_id")) == target_id, "P533 source target ID drift")
        contract.require(str(_scalar(archive, "support_component_sha256")) == surface_hash, "P533 source surface hash drift")
        contract.require(
            str(_scalar(archive, "target_occupancy_mask_world_xyz_sha256")) == world_hash,
            "P533 source target occupancy hash drift",
        )
        lineage = {key: _valid_sha(_scalar(archive, key), key) for key in SOURCE_LINEAGE_KEYS}
        pose = np.asarray(archive["body_pose_axis_angle"], dtype=np.float64).reshape(21, 3)
        betas = np.asarray(archive["betas"], dtype=np.float64).reshape(10)
        root = np.asarray(archive["root_xyz_yaw"], dtype=np.float64).reshape(4)
        frame = np.asarray(archive["pose_frame_to_world_rotation"], dtype=np.float64).reshape(3, 3)
        vertices = np.asarray(archive["vertices_world_zup"], dtype=np.float64).reshape(10475, 3)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        route_yaw = float(_scalar(archive, "route_arrival_tangent_yaw_rad"))
        terminal_yaw = float(_scalar(archive, "terminal_facing_yaw_rad"))
        source_turn = float(_scalar(archive, "arrival_to_terminal_turn_rad"))
        expected_turn = float(_scalar(archive, "expected_arrival_to_terminal_turn_rad"))
        contract.require(not bool(_scalar(archive, "route_arrival_tangent_used_as_terminal_facing")), "route tangent replaced terminal yaw")
        contract.require(not bool(_scalar(archive, "legacy_fixed_pi_expectation_used")), "legacy fixed-pi turn was used")
    contract.require(np.isfinite(pose).all() and np.isfinite(betas).all(), "non-finite P533 articulation")
    contract.require(np.isfinite(root).all() and np.isfinite(frame).all(), "non-finite P533 placement carrier")
    contract.require(faces.ndim == 2 and faces.shape[1] == 3, "P533 faces are malformed")
    return {
        "pose": pose,
        "betas": betas,
        "root": root,
        "frame": frame,
        "vertices": vertices,
        "faces": faces,
        "route_yaw": route_yaw,
        "terminal_yaw": terminal_yaw,
        "source_turn": source_turn,
        "expected_turn": expected_turn,
        "lineage": lineage,
    }

def _camera_mapping(observation_path: Path | None) -> dict[str, Any] | None:
    if observation_path is None:
        return None
    observation_path = Path(observation_path).resolve(strict=True)
    value = contract.read_json(observation_path)
    contract.require(value.get("schema") == OBSERVATION_SCHEMA, "wrong P533 projection observation schema")
    camera = value.get("camera")
    contract.require(isinstance(camera, Mapping), "projection observation lacks camera")
    contract.require(camera.get("extrinsic_convention") == "opencv_world_to_camera", "projection camera must use OpenCV world_to_camera")
    w2c = np.asarray(camera.get("world_to_camera"), dtype=np.float64)
    k_source = np.asarray(camera.get("source_intrinsics"), dtype=np.float64)
    contract.require(w2c.shape == (4, 4) and k_source.shape == (3, 3), "projection camera matrices malformed")
    contract.require(np.isfinite(w2c).all() and np.isfinite(k_source).all(), "non-finite projection camera")
    mapping = camera.get("crop_letterbox")
    contract.require(isinstance(mapping, Mapping), "projection observation lacks crop_letterbox mapping")
    crop = np.asarray(mapping.get("crop_xywh_px"), dtype=np.float64)
    output_wh = np.asarray(mapping.get("output_wh_px"), dtype=np.int64)
    pad = np.asarray(mapping.get("pad_xy_px"), dtype=np.float64)
    contract.require(crop.shape == (4,) and crop[2] > 0 and crop[3] > 0, "invalid crop_xywh_px")
    contract.require(output_wh.shape == (2,) and bool(np.all(output_wh > 0)), "invalid output_wh_px")
    contract.require(pad.shape == (2,) and np.isfinite(pad).all(), "invalid letterbox padding")
    declared_affine = mapping.get("source_to_observation_affine_3x3")
    if declared_affine is not None:
        affine = np.asarray(declared_affine, dtype=np.float64)
        contract.require(affine.shape == (3, 3) and np.isfinite(affine).all(), "invalid source-to-observation affine")
        contract.require(bool(np.allclose(affine[2], (0.0, 0.0, 1.0), atol=1e-9)), "projection mapping is not affine")
        contract.require(affine[0, 0] > 0 and affine[1, 1] > 0, "projection mapping flips an image axis")
        contract.require(abs(affine[0, 1]) <= 1e-9 and abs(affine[1, 0]) <= 1e-9,
                         "projection mapping must be crop/resize/letterbox without shear")
        scale_xy = np.asarray((affine[0, 0], affine[1, 1]), dtype=np.float64)
    else:
        if mapping.get("scale_xy") is not None:
            scale_xy = np.asarray(mapping.get("scale_xy"), dtype=np.float64)
        else:
            uniform = float(mapping.get("uniform_scale"))
            scale_xy = np.asarray((uniform, uniform), dtype=np.float64)
        contract.require(scale_xy.shape == (2,) and np.isfinite(scale_xy).all() and bool(np.all(scale_xy > 0)),
                         "invalid resize scale")
        affine = np.asarray(((scale_xy[0], 0.0, pad[0] - scale_xy[0] * crop[0]),
                             (0.0, scale_xy[1], pad[1] - scale_xy[1] * crop[1]),
                             (0.0, 0.0, 1.0)), dtype=np.float64)
    k_observation = affine @ k_source
    declared_k = np.asarray(camera.get("observation_intrinsics"), dtype=np.float64)
    contract.require(declared_k.shape == (3, 3), "observation_intrinsics missing or malformed")
    contract.require(bool(np.allclose(k_observation, declared_k, atol=1e-5, rtol=1e-7)), "crop/letterbox K derivation drift")
    depth = None
    depth_artifact = None
    if value.get("depth_m") is not None:
        depth_block = value["depth_m"]
        contract.require(isinstance(depth_block, Mapping), "depth_m block malformed")
        contract.require(depth_block.get("alignment") == "observation_pixels_after_crop_letterbox", "depth map is not aligned after crop/letterbox")
        depth_path = Path(str(depth_block.get("path")))
        if not depth_path.is_absolute():
            depth_path = observation_path.parent / depth_path
        depth_artifact = _artifact_from_declared(depth_path, str(depth_block.get("sha256")), "depth map")
        depth = np.load(depth_path, allow_pickle=False)
        contract.require(depth.shape == (int(output_wh[1]), int(output_wh[0])), "depth map/output size drift")
        depth = np.asarray(depth, dtype=np.float32)
    return {
        "path": observation_path,
        "artifact": contract.artifact(observation_path),
        "world_to_camera": w2c,
        "K": k_observation,
        "output_wh": (int(output_wh[0]), int(output_wh[1])),
        "crop": crop,
        "scale_xy": scale_xy,
        "affine": affine,
        "pad": pad,
        "depth": depth,
        "depth_artifact": depth_artifact,
    }

def _project(points: torch.Tensor, camera: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    w2c = torch.as_tensor(camera["world_to_camera"], dtype=points.dtype, device=points.device)
    k = torch.as_tensor(camera["K"], dtype=points.dtype, device=points.device)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    homogeneous = cam @ k.T
    uv = homogeneous[:, :2] / z.clamp_min(1e-4).unsqueeze(1)
    return uv, z

def _deterministic_vertex_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0, length - 1, min(length, count), device=device).round().long()

def _projection_terms(
    state: Any,
    initial_uv: torch.Tensor,
    sample_ids: torch.Tensor,
    camera: Mapping[str, Any],
    depth_margin_m: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    points = torch.cat((state.joints[:22], state.vertices.index_select(0, sample_ids)), dim=0)
    uv, z = _project(points, camera)
    width, height = camera["output_wh"]
    visible0 = (
        (initial_uv[:, 0] >= 0) & (initial_uv[:, 0] <= width - 1)
        & (initial_uv[:, 1] >= 0) & (initial_uv[:, 1] <= height - 1)
    )
    valid = visible0 & (z > 1e-4)
    diagonal = math.hypot(width, height)
    projection = ((uv[valid] - initial_uv[valid]) / diagonal).square().sum(1).mean() if bool(valid.any()) else uv.new_zeros(())
    depth_loss = uv.new_zeros(())
    behind_max = 0.0
    if camera["depth"] is not None:
        depth_map = torch.as_tensor(camera["depth"], dtype=uv.dtype, device=uv.device)[None, None]
        grid = torch.stack((2.0 * uv[:, 0] / max(width - 1, 1) - 1.0,
                            2.0 * uv[:, 1] / max(height - 1, 1) - 1.0), dim=1).reshape(1, 1, -1, 2)
        observed = F.grid_sample(depth_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(-1)
        valid_depth = valid & torch.isfinite(observed) & (observed > 0.05)
        if bool(valid_depth.any()):
            behind = F.relu(z[valid_depth] - observed[valid_depth] + float(depth_margin_m))
            k_tail = max(1, int(math.ceil(0.10 * behind.numel())))
            depth_loss = behind.square().mean() + torch.topk(behind, k_tail).values.square().mean() + behind.max().square()
            behind_max = float(behind.max().detach().cpu())
    return projection, depth_loss, {
        "projection_visible_point_count": int(valid.sum().detach().cpu()),
        "projection_rms_normalized": float(torch.sqrt(projection.detach()).cpu()),
        "depth_upper_envelope_max_violation_m": behind_max,
    }

def _lower_limb_ids(body_model: Any, vertex_count: int, device: torch.device) -> torch.Tensor:
    weights = getattr(body_model, "lbs_weights", None)
    contract.require(isinstance(weights, torch.Tensor) and weights.shape[0] == vertex_count, "SMPL-X LBS weights malformed")
    dominant = weights[:, :22].argmax(1)
    mask = torch.zeros(vertex_count, dtype=torch.bool, device=device)
    for joint in LOWER_LIMB_JOINT_IDS:
        mask |= dominant.to(device) == joint
    # ``torch.flatnonzero`` is unavailable in the project's pinned PyTorch;
    # keep the equivalent operation compatible with that runtime.
    ids = torch.nonzero(mask, as_tuple=False).flatten()
    contract.require(ids.numel() > 0, "empty lower-limb vertex set")
    return ids

def _local_lower_limb_tail(vertices: torch.Tensor, ids: torch.Tensor, fields: Any) -> tuple[torch.Tensor, dict[str, float | int]]:
    points = vertices.index_select(0, ids)
    target_sdf, target_outside = fields.target.sample(points)
    obstacle_sdf, obstacle_outside = fields.non_target.sample(points)
    valid = ~(target_outside | obstacle_outside)
    contract.require(bool(valid.any()), "all lower-limb vertices are outside SDF")
    depth = torch.maximum(F.relu(-target_sdf[valid] - INITIAL_COLLISION_TOLERANCE_M),
                          F.relu(-obstacle_sdf[valid] - INITIAL_COLLISION_TOLERANCE_M))
    count = max(1, int(math.ceil(0.10 * depth.numel())))
    top = torch.topk(depth, count, largest=True).values
    loss = depth.square().mean() + top.square().mean() + depth.max().square()
    return loss, {
        "lower_limb_sdf_vertex_count": int(depth.numel()),
        "lower_limb_penetrating_vertex_count": int((depth > 0).sum().detach().cpu()),
        "lower_limb_penetration_max_m": float(depth.max().detach().cpu()),
        "lower_limb_penetration_top10_mean_m": float(top.mean().detach().cpu()),
    }

def _adaptive_limits(body_model: Any, initial_vertices: torch.Tensor, fields: Any, anatomy: Any) -> tuple[np.ndarray, dict[str, Any]]:
    target_sdf, target_outside = fields.target.sample(initial_vertices)
    obstacle_sdf, obstacle_outside = fields.non_target.sample(initial_vertices)
    weights = body_model.lbs_weights[:, :22].to(initial_vertices.device)
    dominant = weights.argmax(1)
    forbidden = ~anatomy.allowed_contact.to(initial_vertices.device)
    collision = (
        (F.relu(-obstacle_sdf - INITIAL_COLLISION_TOLERANCE_M) > 0) & ~obstacle_outside
    ) | (
        (F.relu(-target_sdf - INITIAL_COLLISION_TOLERANCE_M) > 0) & ~target_outside & forbidden
    )
    direct_rows = {row for row in ACTIVE_ROWS if bool((collision & (dominant == row + 1)).any())}
    chain_rows: set[int] = set()
    for chain in KINEMATIC_CHAINS:
        if direct_rows.intersection(chain):
            chain_rows.update(chain)
    limits = []
    policies = []
    for row in ACTIVE_ROWS:
        if row in CONTACT_SUPPORT_ROWS:
            limit, policy = SUPPORT_OVERRIDE_LIMIT_RAD, "sit_support_override"
        elif row in chain_rows:
            limit, policy = COLLIDING_CHAIN_LIMIT_RAD, "initial_collision_chain"
        elif row in LOWER_LIMB_ROWS:
            limit, policy = NONCOLLIDING_LEG_LIMIT_RAD, "noncolliding_lower_limb"
        else:
            limit, policy = UNRELATED_LIMIT_RAD, "unrelated_joint"
        limits.append(limit)
        policies.append({"body_pose_row": row, "limit_rad": limit, "policy": policy, "direct_initial_collision": row in direct_rows})
    return np.asarray(limits, dtype=np.float32), {
        "initial_collision_tolerance_m": INITIAL_COLLISION_TOLERANCE_M,
        "initial_colliding_vertex_count": int(collision.sum().detach().cpu()),
        "direct_collision_rows": sorted(direct_rows),
        "collision_chain_rows": sorted(chain_rows),
        "per_active_row": policies,
        "sit_support_override_limit_rad": SUPPORT_OVERRIDE_LIMIT_RAD,
        "free_translation_yaw_scale": False,
    }

def _sha256_array(value: np.ndarray) -> str:
    """Hash an array with its dtype/shape made explicit in the payload."""

    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{','.join(str(x) for x in array.shape)}|".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()

def _initial_collision_rows(
    body_model: Any,
    initial_vertices: torch.Tensor,
    fields: Any,
    anatomy: Any,
) -> tuple[set[int], set[int], dict[str, Any]]:
    """Attribute initial target/scene/ground violations to SMPL-X chains."""

    target_sdf, target_outside = fields.target.sample(initial_vertices)
    obstacle_sdf, obstacle_outside = fields.non_target.sample(initial_vertices)
    weights = body_model.lbs_weights[:, :22].to(initial_vertices.device)
    dominant = weights.argmax(1)
    forbidden = ~anatomy.allowed_contact.to(initial_vertices.device)
    target_collision = (
        (F.relu(-target_sdf - INITIAL_COLLISION_TOLERANCE_M) > 0)
        & ~target_outside
        & forbidden
    )
    scene_collision = (
        (F.relu(-obstacle_sdf - INITIAL_COLLISION_TOLERANCE_M) > 0)
        & ~obstacle_outside
    )
    ground_collision = initial_vertices[:, 2] < -INITIAL_COLLISION_TOLERANCE_M
    collision = target_collision | scene_collision | ground_collision
    direct_rows = {
        row for row in ACTIVE_ROWS if bool((collision & (dominant == row + 1)).any())
    }
    chain_rows: set[int] = set()
    for chain in KINEMATIC_CHAINS:
        if direct_rows.intersection(chain):
            chain_rows.update(chain)
    return direct_rows, chain_rows, {
        "initial_collision_tolerance_m": INITIAL_COLLISION_TOLERANCE_M,
        "initial_colliding_vertex_count": int(collision.sum().detach().cpu()),
        "initial_target_forbidden_collision_vertex_count": int(target_collision.sum().detach().cpu()),
        "initial_non_target_collision_vertex_count": int(scene_collision.sum().detach().cpu()),
        "initial_ground_collision_vertex_count": int(ground_collision.sum().detach().cpu()),
    }

def _reachable_limits(
    body_model: Any,
    initial_vertices: torch.Tensor,
    fields: Any,
    anatomy: Any,
    initialization_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one of four bounded, anatomically separated IK hypotheses."""

    profile_index = int(initialization_seed) % len(REACHABLE_LIMIT_PROFILES)
    profile = REACHABLE_LIMIT_PROFILES[profile_index]
    direct_rows, chain_rows, collision_receipt = _initial_collision_rows(
        body_model, initial_vertices, fields, anatomy
    )
    left_chain = {0, 3, 6, 9}
    right_chain = {1, 4, 7, 10}
    left_collision = bool(direct_rows.intersection(left_chain))
    right_collision = bool(direct_rows.intersection(right_chain))
    side_for_row = {0: "left", 1: "right", 3: "left", 4: "right", 6: "left", 7: "right", 9: "left", 10: "right"}
    support_kind = {0: 0, 1: 0, 3: 1, 4: 1, 6: 2, 7: 2}
    quiet = tuple(float(x) for x in profile["quiet_hip_knee_ankle_rad"])
    colliding = tuple(float(x) for x in profile["colliding_hip_knee_ankle_rad"])
    limits: list[float] = []
    policies: list[dict[str, Any]] = []
    for row in ACTIVE_ROWS:
        side = side_for_row.get(row)
        side_collision = left_collision if side == "left" else right_collision if side == "right" else False
        if row in support_kind:
            kind = support_kind[row]
            limit = colliding[kind] if side_collision else quiet[kind]
            policy = f"{side}_{'colliding' if side_collision else 'quiet'}_{('hip', 'knee', 'ankle')[kind]}"
        elif row in (9, 10):
            limit = float(profile["colliding_foot_rad"]) if side_collision else 0.08
            policy = f"{side}_{'colliding' if side_collision else 'quiet'}_foot"
        elif row in chain_rows:
            limit, policy = 0.30, "initial_collision_chain_escape"
        elif row in LOWER_LIMB_ROWS:
            limit, policy = 0.10, "quiet_lower_limb"
        else:
            limit, policy = UNRELATED_LIMIT_RAD, "unrelated_joint"
        limits.append(limit)
        policies.append({
            "body_pose_row": row,
            "joint_id": row + 1,
            "side": side,
            "limit_rad": limit,
            "policy": policy,
            "direct_initial_collision": row in direct_rows,
            "collision_chain_member": row in chain_rows,
        })
    result = np.asarray(limits, dtype=np.float32)
    contract.require(float(result.max()) < SUPPORT_OVERRIDE_LIMIT_RAD, "reachable limits reused 0.75-rad override")
    return result, {
        "enabled": True,
        "policy": "four_profile_collision_attributed_norm_bounded_reachability",
        "candidate_profile_index": profile_index,
        "candidate_profile_name": str(profile["name"]),
        "candidate_profile_count": len(REACHABLE_LIMIT_PROFILES),
        "candidate_enumeration": "campaign_seed_modulo_profile_count_two_initializations_each",
        "quiet_hip_knee_ankle_rad": list(quiet),
        "colliding_hip_knee_ankle_rad": list(colliding),
        "left_leg_initial_collision": left_collision,
        "right_leg_initial_collision": right_collision,
        "direct_collision_rows": sorted(direct_rows),
        "collision_chain_rows": sorted(chain_rows),
        "per_active_row": policies,
        "maximum_active_row_norm_delta_rad": float(result.max()),
        "old_unconditional_support_override_used": False,
        "hip_knee_ankle_decoupled": True,
        "free_translation_yaw_scale": False,
        **collision_receipt,
    }

def _norm_bounded_delta(raw: torch.Tensor, row_limits: torch.Tensor) -> torch.Tensor:
    """Map unconstrained vectors to a strict per-joint axis-angle norm ball."""

    norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    direction = raw / norm.clamp_min(1.0e-8)
    return row_limits * torch.tanh(norm) * direction

def _pose_geodesic_per_row(pose: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    current = p498.body_util.axis_angle_to_matrix(pose.reshape(-1, 3))
    initial = p498.body_util.axis_angle_to_matrix(reference.reshape(-1, 3))
    relative = initial.transpose(-1, -2) @ current
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0 + 1.0e-7, 1.0 - 1.0e-7
    )
    return torch.acos(cosine)

def _reachable_terms(
    state: Any,
    reference_state: Any,
    pose: torch.Tensor,
    initial_pose: torch.Tensor,
    frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | list[float]]]:
    """H3-preservation terms and the matching reachability diagnostics."""

    current_local = (state.joints[:22] - state.joints[0]) @ frame
    reference_local = (reference_state.joints[:22] - reference_state.joints[0]) @ frame
    support_ids = torch.as_tensor(REACHABLE_SUPPORT_JOINT_IDS, dtype=torch.long, device=state.joints.device)
    support_delta = current_local.index_select(0, support_ids) - reference_local.index_select(0, support_ids)
    support_loss = support_delta.square().sum(1).mean()

    all_delta = current_local - reference_local
    mirror = all_delta.new_tensor((1.0, -1.0, 1.0))
    bilateral_vectors = torch.stack([
        all_delta[left] - all_delta[right] * mirror
        for left, right in REACHABLE_BILATERAL_JOINT_PAIRS
    ])
    bilateral_loss = bilateral_vectors.square().sum(1).mean()

    shin_cosines = []
    for knee, ankle in REACHABLE_SHIN_JOINT_PAIRS:
        current_shin = F.normalize((current_local[ankle] - current_local[knee]).reshape(1, 3), dim=1)[0]
        reference_shin = F.normalize((reference_local[ankle] - reference_local[knee]).reshape(1, 3), dim=1)[0]
        shin_cosines.append(torch.clamp(torch.dot(current_shin, reference_shin), -1.0, 1.0))
    shin_cosines_tensor = torch.stack(shin_cosines)
    shin_loss = (1.0 - shin_cosines_tensor).mean()

    with torch.no_grad():
        support_distances = torch.linalg.vector_norm(support_delta, dim=1)
        bilateral_distances = torch.linalg.vector_norm(bilateral_vectors, dim=1)
        shin_angles = torch.rad2deg(torch.acos(shin_cosines_tensor))
        pose_geodesic = _pose_geodesic_per_row(pose, initial_pose)
        support_pose_geodesic = pose_geodesic[list(CONTACT_SUPPORT_ROWS)]
        metrics: dict[str, float | list[float]] = {
            "support_root_relative_delta_mean_m": float(support_distances.mean().cpu()),
            "support_root_relative_delta_max_m": float(support_distances.max().cpu()),
            "support_root_relative_delta_per_joint_m": [float(x) for x in support_distances.cpu()],
            "bilateral_delta_asymmetry_mean_m": float(bilateral_distances.mean().cpu()),
            "bilateral_delta_asymmetry_max_m": float(bilateral_distances.max().cpu()),
            "left_shin_direction_delta_deg": float(shin_angles[0].cpu()),
            "right_shin_direction_delta_deg": float(shin_angles[1].cpu()),
            "maximum_shin_direction_delta_deg": float(shin_angles.max().cpu()),
            "support_pose_geodesic_mean_rad": float(support_pose_geodesic.mean().cpu()),
            "support_pose_geodesic_max_rad": float(support_pose_geodesic.max().cpu()),
        }
    return support_loss, bilateral_loss, shin_loss, metrics

def _gluteal_contact_term(
    state: Any,
    fixed_subset: torch.Tensor,
    fields: Any,
    *,
    contact_band_m: float = 0.020,
    required_vertices: int = 16,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Give hips contact its own pressure term, independent of knee/ankle IK."""

    values, outside = fields.target.sample(state.vertices.index_select(0, fixed_subset))
    valid = ~outside
    contract.require(bool(valid.any()), "all fixed gluteal vertices are outside target SDF")
    absolute = values[valid].abs()
    required = min(int(required_vertices), int(absolute.numel()))
    nearest_required = torch.topk(absolute, k=required, largest=False).values
    # The maximum of the nearest required points is a differentiable surrogate
    # for the publication contact-area gate.  It changes pelvis/hip pose only;
    # knees and ankles retain their own bounds and lower-body priors.
    area_hinge = F.relu(nearest_required.max() - float(contact_band_m))
    loss = nearest_required.square().mean() + 4.0 * area_hinge.square()
    return loss, {
        "fixed_gluteal_target_min_abs_sdf_m": float(absolute.min().detach().cpu()),
        "fixed_gluteal_target_required_kth_abs_sdf_m": float(nearest_required.max().detach().cpu()),
        "fixed_gluteal_target_vertex_count_in_band": int((absolute <= float(contact_band_m)).sum().detach().cpu()),
        "fixed_gluteal_target_required_vertices": required,
    }

def _grid_boundary_term(vertices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Differentiable counterpart of the strict SDF-grid coverage gate."""

    lower = vertices.new_tensor(contract.GRID_LOWER_XYZ)
    upper = vertices.new_tensor(contract.GRID_UPPER_XYZ)
    shape = vertices.new_tensor(contract.WORLD_OCCUPANCY_SHAPE)
    half_spacing = 0.5 * (upper - lower) / shape
    # WorldGridSDF stores voxel-centre samples: index=(x-lower)/spacing-0.5.
    # Therefore the interpolator's valid domain is one half voxel inside the
    # geometric box on both sides, not the raw [lower, upper] extent.
    lower = lower + half_spacing
    upper = upper - half_spacing
    coordinate_violation = F.relu(lower - vertices) + F.relu(vertices - upper)
    depth = torch.linalg.vector_norm(coordinate_violation, dim=1)
    count = max(1, int(math.ceil(0.10 * depth.numel())))
    tail = torch.topk(depth, k=count, largest=True).values
    loss = depth.square().mean() + tail.square().mean() + depth.max().square()
    return loss, {
        "grid_boundary_outside_vertex_count": int((depth > 0).sum().detach().cpu()),
        "grid_boundary_outside_fraction": float((depth > 0).to(vertices.dtype).mean().detach().cpu()),
        "grid_boundary_max_violation_m": float(depth.max().detach().cpu()),
        "grid_boundary_top10_mean_violation_m": float(tail.mean().detach().cpu()),
    }

def _reachable_gates(metrics: Mapping[str, float | int | list[float]], betas: torch.Tensor) -> dict[str, bool]:
    return {
        "stage3_male_zero_beta_carrier_exact": bool(torch.count_nonzero(betas).detach().cpu()) is False,
        "support_root_relative_delta_mean": float(metrics["support_root_relative_delta_mean_m"])
        <= REACHABLE_MAX_SUPPORT_DELTA_MEAN_M,
        "support_root_relative_delta_max": float(metrics["support_root_relative_delta_max_m"])
        <= REACHABLE_MAX_SUPPORT_DELTA_MAX_M,
        "shin_direction_prior": float(metrics["maximum_shin_direction_delta_deg"])
        <= REACHABLE_MAX_SHIN_ANGLE_DEG,
        "bilateral_lower_body_prior": float(metrics["bilateral_delta_asymmetry_mean_m"])
        <= REACHABLE_MAX_BILATERAL_DELTA_ASYMMETRY_M,
        "support_pose_deviation_bounded": float(metrics["support_pose_geodesic_max_rad"])
        <= REACHABLE_MAX_SUPPORT_POSE_GEODESIC_RAD,
    }

def _rank_with_hybrid(
    base_rank: tuple[float, ...],
    metrics: Mapping[str, float | int],
    ablation_mode: str,
) -> tuple[float, ...]:
    """Keep strict gate/violation priority, then prefer hybrid safety terms."""

    if ablation_mode == "p523_only":
        return base_rank
    if ablation_mode == "hybrid_reachable":
        return (
            base_rank[0],
            base_rank[1],
            -float(metrics.get("target_allowed_contact_vertex_count_in_band", 0)),
            float(metrics.get("target_allowed_surface_topk_mean_m", float("inf"))),
            max(
                float(metrics.get("left_foot_ground_surface_topk_mean_m", float("inf"))),
                float(metrics.get("right_foot_ground_surface_topk_mean_m", float("inf"))),
            ),
            -min(
                float(metrics.get("left_foot_ground_vertex_count_in_band", 0)),
                float(metrics.get("right_foot_ground_vertex_count_in_band", 0)),
            ),
            float(metrics.get("target_forbidden_body_penetration_m", float("inf"))),
            float(metrics.get("target_allowed_contact_penetration_m", float("inf"))),
            float(metrics.get("non_target_scene_penetration_m", float("inf"))),
            float(metrics.get("ground_penetration_m", float("inf"))),
            float(metrics.get("lower_limb_penetration_max_m", 0.0)),
            float(metrics.get("support_root_relative_delta_mean_m", float("inf"))),
            float(metrics.get("maximum_shin_direction_delta_deg", float("inf"))),
            float(metrics.get("bilateral_delta_asymmetry_mean_m", float("inf"))),
            float(metrics.get("depth_upper_envelope_max_violation_m", 0.0)),
            float(metrics.get("projection_rms_normalized", 0.0)),
        )
    return (
        base_rank[0],
        base_rank[1],
        float(metrics.get("lower_limb_penetration_max_m", 0.0)),
        float(metrics.get("depth_upper_envelope_max_violation_m", 0.0)),
        float(metrics.get("projection_rms_normalized", 0.0)),
        *base_rank[2:],
    )

def _search_receipt_summary(
    pose_candidate_ids: Sequence[str],
    selected_pose_candidate_id: str,
    search_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ids = tuple(dict.fromkeys(str(value) for value in pose_candidate_ids))
    selected = str(selected_pose_candidate_id)
    selected_base = selected.removesuffix(":micro")
    contract.require(selected_base in ids, "selected search pose is not in the enumerated pose bank")
    return {
        "pose_candidate_ids": list(ids),
        "pose_candidate_count": len(ids),
        "selected_pose_candidate_id": selected,
        "micro_search_executed": any(
            str(record.get("search_scale", "")).startswith("micro")
            for record in search_history
        ),
        "selection_order": "all_strict_gates_then_contact_and_bilateral_foot_physics_then_reachability_prior",
        "all_existing_physics_gates_retained": True,
    }

def run(
    target_path: Path,
    source_path: Path,
    occupancy_path: Path,
    target_mask_path: Path,
    smplx_path: Path,
    segmentation_path: Path,
    output_dir: Path,
    *,
    steps: int,
    device_name: str,
    initialization_seed: int,
    search: Mapping[str, float],
    publication_thresholds: Mapping[str, float | int],
    expected_occupancy_file_sha256: str,
    expected_target_mask_file_sha256: str,
    expected_target_world_sha256: str,
    ablation_mode: str = "hybrid",
    projection_observation_path: Path | None = None,
    projection_weight: float = 2.0,
    depth_weight: float = 4.0,
    lower_limb_tail_weight: float = 500.0,
    depth_margin_m: float = 0.015,
    floor_ignore_height_m: float = 0.08,
    maximum_target_floor_fraction: float = 0.10,
    maximum_target_fraction: float = 0.25,
    field_builder=None,
) -> dict[str, Any]:
    """Refine one H3/HybrIK-X articulation while preserving Stage-1 authority."""

    contract.require(ablation_mode in ABLATION_MODES, f"ablation_mode must be one of {ABLATION_MODES}")
    for value, label in (
        (projection_weight, "projection_weight"),
        (depth_weight, "depth_weight"),
        (lower_limb_tail_weight, "lower_limb_tail_weight"),
        (depth_margin_m, "depth_margin_m"),
    ):
        contract.require(math.isfinite(float(value)) and float(value) >= 0.0, f"invalid {label}")
    target_path = Path(target_path).resolve(strict=True)
    source_path = Path(source_path).resolve(strict=True)
    occupancy_path = Path(occupancy_path).resolve(strict=True)
    target_mask_path = Path(target_mask_path).resolve(strict=True)
    smplx_path = Path(smplx_path).resolve(strict=True)
    segmentation_path = Path(segmentation_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    contract.require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    target_receipt = contract.read_json(target_path)
    target_id, surface_hash, surface_centre, contact_xyz, bounds, target_yaw, resolved_contact = p523._validate_target(
        target_receipt, artifact_base_dir=target_path.parent
    )
    relation = contract.normalize_relation(target_receipt.get("reference_relation"))
    contract.require(str(target_receipt.get("target_occupancy_mask_world_xyz_sha256")) == expected_target_world_sha256,
                     "target/SDF world-mask hash drift")
    source = _load_source(source_path, target_id, surface_hash, expected_target_world_sha256)
    measured_turn = contract.angle_difference(source["route_yaw"], target_yaw)
    contract.require(contract.angle_difference(source["terminal_yaw"], target_yaw) <= contract.REORIENTATION_BINDING_TOLERANCE_RAD,
                     "P533 source terminal yaw is not Stage-1 semantic yaw")
    contract.require(abs(measured_turn - source["source_turn"]) <= 2e-6 and abs(measured_turn - source["expected_turn"]) <= 2e-6,
                     "P533 route/target reorientation drift")

    camera = _camera_mapping(projection_observation_path)
    if ablation_mode in {"hybrid", "hybrid_reachable"} and projection_weight > 0:
        contract.require(camera is not None, "hybrid projection loss requires --projection-observation")
    requested_depth_weight = float(depth_weight)
    if ablation_mode == "p523_only":
        projection_weight = depth_weight = lower_limb_tail_weight = 0.0
    elif camera is None or camera["depth"] is None:
        # Depth is explicitly optional.  Its requested and effective weights
        # remain distinct in the receipt so a missing map cannot masquerade as
        # an executed depth experiment.
        depth_weight = 0.0

    device = torch.device(device_name)
    contract.require(device.type == "cpu", "P533 refiner is CPU-only; generation remains GPU-separate")
    body_model = p498.body_util.load_fixed_smplx(smplx_path, device)
    segmentation = contract.read_json(segmentation_path)
    fixed_subset_np = p498.hips_support_subset(source["vertices"], segmentation)
    fixed_subset = torch.as_tensor(fixed_subset_np, dtype=torch.long, device=device)
    refine_config = p498.target.RefineConfig(
        steps=1,
        bilateral_foot_support=True,
        target_allowed_tail_m=float(publication_thresholds["maximum_target_allowed_contact_penetration_m"]),
        ground_tail_m=float(publication_thresholds["maximum_ground_penetration_m"]),
        minimum_allowed_contact_vertices=int(publication_thresholds["minimum_target_allowed_contact_vertices"]),
        minimum_foot_ground_vertices=min(int(publication_thresholds["minimum_left_foot_ground_vertices"]),
                                         int(publication_thresholds["minimum_right_foot_ground_vertices"])),
        maximum_target_forbidden_penetration_m=float(publication_thresholds["maximum_target_forbidden_body_penetration_m"]),
        maximum_non_target_penetration_m=float(publication_thresholds["maximum_non_target_scene_penetration_m"]),
        maximum_outside_fraction=float(publication_thresholds["maximum_sdf_outside_fraction"]),
    ).validate()
    fields = (field_builder or bound_sdf.build_fields_from_bound_mask)(
        occupancy_path, target_mask_path, device,
        expected_occupancy_file_sha256=expected_occupancy_file_sha256,
        expected_target_mask_file_sha256=expected_target_mask_file_sha256,
        expected_target_world_sha256=expected_target_world_sha256,
        floor_ignore_height_m=float(floor_ignore_height_m),
        maximum_target_floor_fraction=float(maximum_target_floor_fraction),
        maximum_target_fraction=float(maximum_target_fraction),
    )
    anatomy = p498.target.build_anatomy_masks(body_model, 10475, refine_config)
    initial_pose = torch.as_tensor(source["pose"], dtype=torch.float32, device=device)
    source_shape_betas = torch.as_tensor(source["betas"], dtype=torch.float32, device=device)
    # Stage3 materializes the male neutral model with zero betas.  Project the
    # H3 articulation onto that exact carrier before any physics optimization,
    # then re-land the same topology-stable gluteal subset on Stage1 contact.
    betas = (
        torch.zeros_like(source_shape_betas)
        if ablation_mode == "hybrid_reachable"
        else source_shape_betas
    )
    contact = torch.as_tensor(contact_xyz, dtype=torch.float32, device=device)
    delta_yaw = math.atan2(math.sin(target_yaw - source["root"][3]), math.cos(target_yaw - source["root"][3]))
    frame_np = p498.rotation_z(delta_yaw) @ source["frame"]
    frame = torch.as_tensor(frame_np, dtype=torch.float32, device=device)
    source_shape_state = p498.materialize(
        body_model, initial_pose, source_shape_betas, frame, contact, fixed_subset
    )
    initial_state = p498.materialize(body_model, initial_pose, betas, frame, contact, fixed_subset)
    lower_limb_ids = _lower_limb_ids(body_model, 10475, device)

    if ablation_mode == "hybrid":
        row_limits_np, adaptive_receipt = _adaptive_limits(body_model, initial_state.vertices, fields, anatomy)
    elif ablation_mode == "hybrid_reachable":
        row_limits_np, adaptive_receipt = _reachable_limits(
            body_model,
            initial_state.vertices,
            fields,
            anatomy,
            int(initialization_seed),
        )
    else:
        row_limits_np = np.asarray([
            P523_SUPPORT_LIMIT_RAD if row in CONTACT_SUPPORT_ROWS else P523_ESCAPE_LIMIT_RAD
            for row in ACTIVE_ROWS
        ], dtype=np.float32)
        adaptive_receipt = {
            "enabled": False,
            "policy": "exact_P523_fixed_bounds",
            "per_active_row": [{"body_pose_row": row, "limit_rad": float(limit)} for row, limit in zip(ACTIVE_ROWS, row_limits_np)],
            "free_translation_yaw_scale": False,
        }
    for index, row in enumerate(ACTIVE_ROWS):
        if row in (6, 7):
            row_limits_np[index] = max(row_limits_np[index], .70)
        elif row in (9, 10):
            row_limits_np[index] = max(row_limits_np[index], .50)
    adaptive_receipt["p550_neutral_sitting_ankle_limit_override"] = {"ankle_rad": .70, "foot_rad": .50,
        "world_contact_yaw_scale_or_translation_freed": False}
    row_limits = torch.as_tensor(row_limits_np, dtype=torch.float32, device=device).reshape(-1, 1)

    sample_ids = _deterministic_vertex_indices(10475, 512, device)
    initial_uv = None
    if camera is not None:
        initial_points = torch.cat((initial_state.joints[:22], initial_state.vertices.index_select(0, sample_ids)), dim=0)
        initial_uv, _ = _project(initial_points, camera)
        initial_uv = initial_uv.detach()

    steps = int(steps)
    seed = int(initialization_seed)
    contract.require(steps > 0 and seed >= 0, "invalid optimizer steps/seed")
    maximum_clearance = float(search["same_surface_max_clearance_m"])
    if seed == 0:
        raw_initial = np.zeros((len(ACTIVE_ROWS), 3), dtype=np.float32)
    else:
        raw_initial = np.random.default_rng(seed).normal(0.0, 0.12, size=(len(ACTIVE_ROWS), 3)).astype(np.float32)
    raw = torch.nn.Parameter(torch.as_tensor(raw_initial, dtype=torch.float32, device=device))
    raw_clearance = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam((raw, raw_clearance), lr=0.025)
    best = None
    history: list[dict[str, Any]] = []
    reachable_pose_bank: dict[str, tuple[tuple[float, ...], torch.Tensor]] = {}
    for iteration in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pose = initial_pose.clone()
        active_delta = (
            _norm_bounded_delta(raw, row_limits)
            if ablation_mode == "hybrid_reachable"
            else row_limits * torch.tanh(raw)
        )
        pose[list(ACTIVE_ROWS)] = initial_pose[list(ACTIVE_ROWS)] + active_delta
        clearance = maximum_clearance * torch.sigmoid(raw_clearance)
        optimized_contact = contact + torch.stack((clearance.new_zeros(()), clearance.new_zeros(()), clearance))
        state = p498.materialize(body_model, pose, betas, frame, optimized_contact, fixed_subset)
        branches = p498.target.branch_terms(state.vertices, fields, anatomy, refine_config)
        pose_prior = p498.body_util.pose_geodesic_from_initialization(pose, initial_pose)
        component_limit, norm_limit, _, _ = p498.body_util.pose_limit_terms(pose)
        loss = (14.0 * branches.loss_non_target + 18.0 * branches.loss_target_forbidden
                + 30.0 * branches.loss_target_allowed_tail + 12.0 * branches.loss_target_contact_surface
                + 30.0 * branches.loss_ground_penetration + 120.0 * branches.loss_foot_ground_surface
                + 0.20 * pose_prior + component_limit + norm_limit)
        local_diag: dict[str, float | int] = {}
        projection_diag: dict[str, float | int] = {}
        reachable_diag: dict[str, float | list[float]] = {}
        gluteal_diag: dict[str, float | int] = {}
        grid_diag: dict[str, float | int] = {}
        if ablation_mode in {"hybrid", "hybrid_reachable"}:
            local_loss, local_diag = _local_lower_limb_tail(state.vertices, lower_limb_ids, fields)
            loss = loss + float(lower_limb_tail_weight) * local_loss
            if camera is not None and initial_uv is not None:
                projection_loss, depth_loss, projection_diag = _projection_terms(
                    state, initial_uv, sample_ids, camera, float(depth_margin_m)
                )
                loss = loss + float(projection_weight) * projection_loss
                if camera["depth"] is not None:
                    loss = loss + float(depth_weight) * depth_loss
        if ablation_mode == "hybrid_reachable":
            support_prior, bilateral_prior, shin_prior, reachable_diag = _reachable_terms(
                state, initial_state, pose, initial_pose, frame
            )
            gluteal_contact, gluteal_diag = _gluteal_contact_term(
                state,
                fixed_subset,
                fields,
                contact_band_m=refine_config.allowed_contact_band_m,
                required_vertices=refine_config.minimum_allowed_contact_vertices,
            )
            grid_boundary, grid_diag = _grid_boundary_term(state.vertices)
            loss = (
                loss
                + 8.0 * support_prior
                + 6.0 * bilateral_prior
                + 0.25 * shin_prior
                + 0.30 * pose_prior
                + 100.0 * gluteal_contact
                + 150.0 * grid_boundary
            )
        if ablation_mode == "hybrid":
            loss = loss + strict_terms(state.vertices, fields, anatomy,
                publication_thresholds, _grid_boundary_term)
        loss = loss + posture_loss(state, target_yaw) + bilateral_support_loss(
            state.vertices, anatomy, publication_thresholds, refine_config.foot_ground_band_m)
        contract.require(bool(torch.isfinite(loss)), "non-finite P533 keypose objective")
        with torch.no_grad():
            metrics = p523._float_diagnostics(branches)
            metrics.update(local_diag)
            metrics.update(projection_diag)
            metrics.update(reachable_diag)
            metrics.update(gluteal_diag)
            metrics.update(grid_diag)
            support = state.vertices.index_select(0, fixed_subset)
            count = min(len(fixed_subset_np), max(8, int(math.ceil(len(fixed_subset_np) * p498.SUPPORT_ENVELOPE_FRACTION))))
            anchor = torch.cat((support[:, :2].mean(0), torch.topk(support[:, 2], k=count, largest=False).values.mean().reshape(1)))
            contact_error = float(torch.linalg.vector_norm(anchor - optimized_contact).detach().cpu())
            metrics["hips_support_inside_surface_fraction"] = p498.support_inside_surface_fraction(support.cpu().numpy(), bounds)
            gates = contract.publication_gates(metrics, publication_thresholds, facing_error_rad=0.0,
                                                selected_surface_match=True, contact_lock_error_m=contact_error)
            if ablation_mode == "hybrid_reachable":
                gates.update(_reachable_gates(metrics, betas))
            rank = _rank_with_hybrid(
                contract.metric_rank(metrics, gates, publication_thresholds), metrics, ablation_mode
            )
            quality = posture_metrics(state.joints[:22].detach().cpu().numpy(), target_yaw)
            metrics["p550_neutral_sit_quality"] = quality
            rank = (*violation_rank(metrics, gates, publication_thresholds),
                sum(not v for v in quality["neutral_sit_quality_gates"].values()),
                float(posture_loss(state, target_yaw).detach().cpu()), *rank)
            if best is None or rank < best[0]:
                best = (rank, pose.detach().clone(), metrics, gates, contact_error)
            if ablation_mode == "hybrid_reachable":
                contact_count = float(metrics["target_allowed_contact_vertex_count_in_band"])
                outside = float(metrics["outside_fraction"])
                ground = float(metrics["ground_penetration_m"])
                support_delta = float(metrics["support_root_relative_delta_mean_m"])
                reachability_gate_names = set(_reachable_gates(metrics, betas))
                physics_failed = float(sum(
                    not bool(value)
                    for key, value in gates.items()
                    if key not in reachability_gate_names
                ))
                candidate_scores = {
                    "contact_rich": (-contact_count, outside, ground, support_delta),
                    "coverage_rich": (outside, -contact_count, ground, support_delta),
                    "physics_contact_balance": (
                        physics_failed,
                        max(
                            0.0,
                            outside / float(publication_thresholds["maximum_sdf_outside_fraction"]) - 1.0,
                        )
                        + max(
                            0.0,
                            (
                                float(publication_thresholds["minimum_target_allowed_contact_vertices"])
                                - contact_count
                            )
                            / max(
                                1.0,
                                float(publication_thresholds["minimum_target_allowed_contact_vertices"]),
                            ),
                        ),
                        -contact_count,
                        support_delta,
                    ),
                }
                for candidate_id, candidate_score in candidate_scores.items():
                    retained = reachable_pose_bank.get(candidate_id)
                    if retained is None or candidate_score < retained[0]:
                        reachable_pose_bank[candidate_id] = (
                            tuple(float(x) for x in candidate_score),
                            pose.detach().clone(),
                        )
                if iteration == steps:
                    reachable_pose_bank["final"] = ((0.0,), pose.detach().clone())
            if iteration == 0 or iteration == steps or iteration % 20 == 0:
                history.append({"iteration": iteration, "loss": float(loss.detach().cpu()), "rank": list(rank),
                                "metrics": metrics, "gates": gates})
        if iteration == steps:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw, raw_clearance), 5.0)
        optimizer.step()

    contract.require(best is not None, "optimizer retained no keypose")
    _, best_pose, _, _, _ = best
    search_pose_candidates: list[tuple[str, torch.Tensor]] = [("rank_best", best_pose)]
    if ablation_mode == "hybrid_reachable":
        search_pose_candidates.extend(
            (candidate_id, retained[1])
            for candidate_id, retained in sorted(reachable_pose_bank.items())
        )
        unique_candidates: list[tuple[str, torch.Tensor]] = []
        seen_pose_hashes: set[str] = set()
        for candidate_id, candidate_pose in search_pose_candidates:
            pose_hash = _sha256_array(candidate_pose.detach().cpu().numpy().astype(np.float32))
            if pose_hash in seen_pose_hashes:
                continue
            seen_pose_hashes.add(pose_hash)
            unique_candidates.append((candidate_id, candidate_pose))
        search_pose_candidates = unique_candidates
    xy_offsets = contract.inclusive_offsets(float(search["same_surface_xy_search_radius_m"]),
                                             float(search["same_surface_xy_search_step_m"]))
    clearance_offsets = p523._positive_offsets(maximum_clearance, float(search["same_surface_clearance_step_m"]))
    edge_margin = float(search["same_surface_edge_margin_m"])
    search_best = None
    search_history: list[dict[str, Any]] = []

    def evaluate_search_candidate(
        search_pose: torch.Tensor,
        candidate_np: np.ndarray,
        clearance_from_seed_m: float,
    ) -> tuple[tuple[float, ...], Any, dict[str, Any], dict[str, bool], float]:
        candidate = torch.as_tensor(candidate_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            state = p498.materialize(body_model, search_pose, betas, frame, candidate, fixed_subset)
            branches = p498.target.branch_terms(state.vertices, fields, anatomy, refine_config)
            metrics: dict[str, Any] = p523._float_diagnostics(branches)
            if ablation_mode in {"hybrid", "hybrid_reachable"}:
                _, local_diag = _local_lower_limb_tail(state.vertices, lower_limb_ids, fields)
                metrics.update(local_diag)
                if camera is not None and initial_uv is not None:
                    _, _, projection_diag = _projection_terms(
                        state, initial_uv, sample_ids, camera, float(depth_margin_m)
                    )
                    metrics.update(projection_diag)
            if ablation_mode == "hybrid_reachable":
                _, _, _, reachable_diag = _reachable_terms(
                    state, initial_state, search_pose, initial_pose, frame
                )
                metrics.update(reachable_diag)
                _, gluteal_diag = _gluteal_contact_term(
                    state,
                    fixed_subset,
                    fields,
                    contact_band_m=refine_config.allowed_contact_band_m,
                    required_vertices=refine_config.minimum_allowed_contact_vertices,
                )
                metrics.update(gluteal_diag)
                _, grid_diag = _grid_boundary_term(state.vertices)
                metrics.update(grid_diag)
            support = state.vertices.index_select(0, fixed_subset)
            count = min(
                len(fixed_subset_np),
                max(8, int(math.ceil(len(fixed_subset_np) * p498.SUPPORT_ENVELOPE_FRACTION))),
            )
            anchor = torch.cat((
                support[:, :2].mean(0),
                torch.topk(support[:, 2], k=count, largest=False).values.mean().reshape(1),
            ))
            error = float(torch.linalg.vector_norm(anchor - candidate).detach().cpu())
            metrics["hips_support_inside_surface_fraction"] = p498.support_inside_surface_fraction(
                support.cpu().numpy(), bounds
            )
            gates = contract.publication_gates(
                metrics,
                publication_thresholds,
                facing_error_rad=0.0,
                selected_surface_match=True,
                contact_lock_error_m=error,
            )
            gates.update({
                "current_stage1_target_mask_hash_bound": fields.component_receipt["target_component_sha256"]
                == expected_target_world_sha256,
                "direct_target_relation_valid": relation == "none"
                or relation in {"beside", "associated_with_reference_setup"},
                "known_action_family_sit": True,
                "contact_point_inside_bound_surface": True,
                "contact_point_on_true_support_mask": resolved_contact.support_mask is None
                or resolved_contact.support_mask.contains_world_xy(candidate_np[:2]),
                "contact_anchor_clearance_bounded": 0.0
                <= float(clearance_from_seed_m)
                <= maximum_clearance + 1e-12,
            })
            if ablation_mode == "hybrid_reachable":
                gates.update(_reachable_gates(metrics, betas))
            rank = _rank_with_hybrid(
                contract.metric_rank(metrics, gates, publication_thresholds), metrics, ablation_mode
            )
        return rank, state, metrics, gates, error

    for pose_candidate_id, search_pose in search_pose_candidates:
        for dx in xy_offsets:
            for dy in xy_offsets:
                for dz in clearance_offsets:
                    candidate_np = contact_xyz + np.asarray((dx, dy, dz), dtype=np.float64)
                    if not (bounds[0, 0] + edge_margin <= candidate_np[0] <= bounds[1, 0] - edge_margin
                            and bounds[0, 1] + edge_margin <= candidate_np[1] <= bounds[1, 1] - edge_margin):
                        continue
                    if resolved_contact.support_mask is not None and not resolved_contact.support_mask.contains_world_xy(candidate_np[:2]):
                        continue
                    rank, state, metrics, gates, error = evaluate_search_candidate(
                        search_pose, candidate_np, float(dz)
                    )
                    record = {"pose_candidate_id": pose_candidate_id,
                              "search_scale": "coarse",
                              "offset_xyz_m": [float(dx), float(dy), float(dz)], "rank": list(rank),
                              "all_gates_passed": all(gates.values()),
                              "failed_gates": sorted(k for k, value in gates.items() if not value)}
                    if all(gates.values()) or len(search_history) < 16:
                        search_history.append(record)
                    if search_best is None or rank < search_best[0]:
                        search_best = (
                            rank,
                            state,
                            metrics,
                            gates,
                            error,
                            candidate_np,
                            search_pose.detach().clone(),
                            pose_candidate_id,
                        )

    # The campaign's 20-mm XY / 5-mm Z grid can straddle the exact intersection
    # of contact area and the voxel-centre SDF domain.  Only the new branch gets
    # a bounded 2-mm local search around the best coarse hypothesis.  It cannot
    # leave the registered Stage1 radius, surface mask, or clearance interval.
    if (
        ablation_mode in {"hybrid", "hybrid_reachable"}
        and search_best is not None
        and not all(search_best[3].values())
    ):
        coarse_contact = np.asarray(search_best[5], dtype=np.float64)
        micro_pose = search_best[6]
        micro_pose_id = str(search_best[7])
        micro_offsets = np.arange(-0.006, 0.006 + 1.0e-12, 0.002, dtype=np.float64)
        radius = float(search["same_surface_xy_search_radius_m"])
        micro_history_count = 0
        for mx in micro_offsets:
            for my in micro_offsets:
                for mz in micro_offsets:
                    candidate_np = coarse_contact + np.asarray((mx, my, mz), dtype=np.float64)
                    total_offset = candidate_np - contact_xyz
                    if abs(float(total_offset[0])) > radius + 1e-12 or abs(float(total_offset[1])) > radius + 1e-12:
                        continue
                    if not 0.0 <= float(total_offset[2]) <= maximum_clearance + 1e-12:
                        continue
                    if not (
                        bounds[0, 0] + edge_margin <= candidate_np[0] <= bounds[1, 0] - edge_margin
                        and bounds[0, 1] + edge_margin <= candidate_np[1] <= bounds[1, 1] - edge_margin
                    ):
                        continue
                    if (
                        resolved_contact.support_mask is not None
                        and not resolved_contact.support_mask.contains_world_xy(candidate_np[:2])
                    ):
                        continue
                    rank, state, metrics, gates, error = evaluate_search_candidate(
                        micro_pose, candidate_np, float(total_offset[2])
                    )
                    record = {
                        "pose_candidate_id": f"{micro_pose_id}:micro",
                        "search_scale": "micro_2mm_within_6mm",
                        "offset_xyz_m": [float(x) for x in total_offset],
                        "rank": list(rank),
                        "all_gates_passed": all(gates.values()),
                        "failed_gates": sorted(key for key, value in gates.items() if not value),
                    }
                    if all(gates.values()) or micro_history_count < 16:
                        search_history.append(record)
                        micro_history_count += 1
                    if search_best is None or rank < search_best[0]:
                        search_best = (
                            rank,
                            state,
                            metrics,
                            gates,
                            error,
                            candidate_np,
                            micro_pose.detach().clone(),
                            f"{micro_pose_id}:micro",
                        )
    contract.require(search_best is not None, "same-surface search retained no candidate")
    (best_rank, best_state, best_metrics, best_gates, contact_error, selected_contact,
     best_pose, selected_pose_candidate_id) = search_best
    search_history.append({"pose_candidate_id": selected_pose_candidate_id,
                           "offset_xyz_m": (selected_contact - contact_xyz).tolist(), "rank": list(best_rank),
                           "all_gates_passed": all(best_gates.values()), "selected": True})
    published = all(best_gates.values())
    keypose_path = output_dir / ("published_keypose.npz" if published else "review_only_keypose.npz")
    vertices_np = best_state.vertices.detach().cpu().numpy().astype(np.float32)
    joints_np = best_state.joints.detach().cpu().numpy().astype(np.float32)
    root_np = best_state.root.detach().cpu().numpy()
    output_betas_np = betas.detach().cpu().numpy().astype(np.float32)
    source_betas_np = source_shape_betas.detach().cpu().numpy().astype(np.float32)
    fixed_subset_sha256 = _sha256_array(np.asarray(fixed_subset_np, dtype=np.int64))
    source_betas_sha256 = _sha256_array(source_betas_np)
    carrier_betas_sha256 = _sha256_array(output_betas_np)
    carrier_projection: dict[str, Any] | None = None
    if ablation_mode == "hybrid_reachable":
        with torch.no_grad():
            source_rel = source_shape_state.joints[:22] - source_shape_state.joints[0]
            carrier_rel = initial_state.joints[:22] - initial_state.joints[0]
            joint_projection_delta = torch.linalg.vector_norm(carrier_rel - source_rel, dim=1)
            support_projection_delta = joint_projection_delta[
                list(REACHABLE_SUPPORT_JOINT_IDS)
            ]
            source_support = source_shape_state.vertices.index_select(0, fixed_subset)
            carrier_support = initial_state.vertices.index_select(0, fixed_subset)
            support_count = min(
                len(fixed_subset_np),
                max(8, int(math.ceil(len(fixed_subset_np) * p498.SUPPORT_ENVELOPE_FRACTION))),
            )

            def support_anchor(points: torch.Tensor) -> torch.Tensor:
                return torch.cat((
                    points[:, :2].mean(0),
                    torch.topk(points[:, 2], k=support_count, largest=False).values.mean().reshape(1),
                ))

            source_anchor_error = torch.linalg.vector_norm(support_anchor(source_support) - contact)
            carrier_anchor_error = torch.linalg.vector_norm(support_anchor(carrier_support) - contact)
            carrier_projection = {
                "schema": "p533.stage3_male_zero_beta_carrier_projection.v1",
                "source_h3_keypose_sha256": contract.sha256_file(source_path),
                "source_h3_betas_sha256": source_betas_sha256,
                "source_h3_betas_l2_norm": float(torch.linalg.vector_norm(source_shape_betas).cpu()),
                "target_stage3_carrier_model_sha256": contract.sha256_file(smplx_path),
                "target_stage3_carrier_gender": "male",
                "target_stage3_carrier_betas_sha256": carrier_betas_sha256,
                "target_stage3_carrier_betas_l2_norm": float(torch.linalg.vector_norm(betas).cpu()),
                "fixed_gluteal_subset_sha256": fixed_subset_sha256,
                "fixed_gluteal_subset_vertex_count": int(len(fixed_subset_np)),
                "joint_root_relative_projection_delta_mean_m": float(joint_projection_delta.mean().cpu()),
                "joint_root_relative_projection_delta_max_m": float(joint_projection_delta.max().cpu()),
                "support_joint_root_relative_projection_delta_mean_m": float(support_projection_delta.mean().cpu()),
                "support_joint_root_relative_projection_delta_max_m": float(support_projection_delta.max().cpu()),
                "source_shape_gluteal_relanding_error_m": float(source_anchor_error.cpu()),
                "zero_beta_carrier_gluteal_relanding_error_m": float(carrier_anchor_error.cpu()),
                "source_to_carrier_root_delta_m": float(torch.linalg.vector_norm(initial_state.root - source_shape_state.root).cpu()),
                "projection_precedes_physics_refine": True,
                "gluteal_relanding_after_projection": True,
            }
    publication_status = (
        f"published_strict_{ablation_mode}" if published else f"review_only_{ablation_mode}_failed_physics"
    )
    np.savez_compressed(
        keypose_path,
        schema=np.asarray(KEYPOSE_SCHEMA),
        status=np.asarray(publication_status),
        ablation_mode=np.asarray(ablation_mode),
        scene_id=np.asarray(target_receipt.get("scene_id")), instruction=np.asarray(target_receipt.get("instruction")),
        action_family=np.asarray("sit"), target_instance_id=np.asarray(target_receipt.get("target_instance_id")),
        target_class=np.asarray(target_receipt.get("target_class")), selected_candidate_id=np.asarray(target_id),
        support_component_sha256=np.asarray(surface_hash),
        target_occupancy_mask_world_xyz_sha256=np.asarray(expected_target_world_sha256),
        reference_relation=np.asarray(relation), surface_centre_world_xyz_zup_m=surface_centre.astype(np.float32),
        surface_bounds_world_zup_m=bounds.astype(np.float32), contact_seed_world_xyz_zup_m=contact_xyz.astype(np.float32),
        contact_seed_source=np.asarray(resolved_contact.contact_seed_source),
        contact_seed_surface_cell=np.asarray(
            resolved_contact.contact_surface_cell if resolved_contact.contact_surface_cell is not None else (-1, -1),
            dtype=np.int32,
        ),
        contact_seed_is_p530_validated=np.asarray(
            resolved_contact.contact_seed_source == "p530_validated_interior_action_space_contact"
        ),
        contact_goal_world_xyz_zup_m=selected_contact.astype(np.float32),
        contact_offset_within_surface_xyz_m=(selected_contact - surface_centre).astype(np.float32),
        contact_seed_to_selected_offset_xyz_m=(selected_contact - contact_xyz).astype(np.float32),
        contact_yaw_rad=np.asarray(target_yaw, dtype=np.float32), terminal_facing_yaw_rad=np.asarray(target_yaw, dtype=np.float32),
        route_arrival_tangent_yaw_rad=np.asarray(source["route_yaw"], dtype=np.float32),
        arrival_to_terminal_turn_rad=np.asarray(measured_turn, dtype=np.float32),
        expected_arrival_to_terminal_turn_rad=np.asarray(measured_turn, dtype=np.float32),
        explicit_reorientation_required=np.asarray(measured_turn > contract.REORIENTATION_BINDING_TOLERANCE_RAD),
        route_arrival_tangent_used_as_terminal_facing=np.asarray(False), legacy_fixed_pi_expectation_used=np.asarray(False),
        root_xyz_yaw=np.asarray((*root_np.tolist(), target_yaw), dtype=np.float32),
        body_pose_axis_angle=best_pose.detach().cpu().numpy().astype(np.float32), betas=output_betas_np,
        pose_frame_to_world_rotation=frame_np.astype(np.float32), vertices_world_zup=vertices_np,
        vertices_world_zup_m=vertices_np, joints_world_zup=joints_np, joints_world_zup_m=joints_np,
        faces=source["faces"], fixed_generated_butt_vertex_indices=fixed_subset_np,
        contact_segment=np.asarray("hips_gluteal_native_mesh_region"), contact_joint_ids=np.asarray(CONTACT_JOINT_IDS, dtype=np.int64),
        active_body_pose_rows=np.asarray(ACTIVE_ROWS, dtype=np.int64),
        contact_support_body_pose_rows=np.asarray(CONTACT_SUPPORT_ROWS, dtype=np.int64),
        collision_escape_body_pose_rows=np.asarray(COLLISION_ESCAPE_ROWS, dtype=np.int64),
        active_body_pose_delta_limits_rad=row_limits_np,
        adaptive_joint_bounds_used=np.asarray(ablation_mode in {"hybrid", "hybrid_reachable"}),
        h3_model_used=np.asarray(True), hybrikx_model_used=np.asarray(True), camera_world_placement_discarded=np.asarray(True),
        old_world_placement_consumed=np.asarray(False), free_translation_used=np.asarray(False), free_yaw_used=np.asarray(False),
        free_scale_used=np.asarray(False), source_keypose_sha256=np.asarray(contract.sha256_file(source_path)),
        current_only=np.asarray(True), legacy_world_placement_reused=np.asarray(False),
        p373_placement_used=np.asarray(False), p520_placement_used=np.asarray(False),
        p508_placement_used=np.asarray(False), gt_root_path_pose_motion_contact_icgf_used=np.asarray(False),
        all_publication_gates_passed=np.asarray(published), optimizer_initialization_seed=np.asarray(seed, dtype=np.int64),
        **({
            "reachable_joint_bounds_used": np.asarray(True),
            "stage3_male_zero_beta_carrier_projection_used": np.asarray(True),
            "stage3_carrier_gender": np.asarray("male"),
            "stage3_carrier_model_sha256": np.asarray(contract.sha256_file(smplx_path)),
            "source_h3_betas_sha256": np.asarray(source_betas_sha256),
            "stage3_carrier_betas_sha256": np.asarray(carrier_betas_sha256),
            "fixed_gluteal_subset_sha256": np.asarray(fixed_subset_sha256),
            "selected_pose_candidate_id": np.asarray(selected_pose_candidate_id),
        } if ablation_mode == "hybrid_reachable" else {}),
        **{key: np.asarray(value) for key, value in source["lineage"].items()},
    )

    camera_receipt = None
    if camera is not None:
        camera_receipt = {
            "observation": camera["artifact"], "extrinsic_convention": "opencv_world_to_camera",
            "world_to_camera": camera["world_to_camera"].tolist(), "observation_intrinsics": camera["K"].tolist(),
            "crop_xywh_px": camera["crop"].tolist(), "scale_xy": camera["scale_xy"].tolist(),
            "source_to_observation_affine_3x3": camera["affine"].tolist(),
            "pad_xy_px": camera["pad"].tolist(), "output_wh_px": list(camera["output_wh"]),
            "depth_m": camera["depth_artifact"],
        }
    result = {
        "schema": RECEIPT_SCHEMA,
        "status": f"published_strict_{ablation_mode}" if published else f"vetoed_strict_{ablation_mode}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_id": target_receipt.get("scene_id"), "instruction": target_receipt.get("instruction"),
        "action_family": "sit", "ablation_mode": ablation_mode,
        "target_binding": {
            "target_instance_id": target_receipt.get("target_instance_id"), "target_class": target_receipt.get("target_class"),
            "target_surface_id": target_id, "target_surface_sha256": surface_hash,
            "target_occupancy_mask_world_xyz_sha256": expected_target_world_sha256, "relation": relation,
            "contact_seed_world_xyz_zup_m": contact_xyz.tolist(), "selected_contact_world_xyz_zup_m": selected_contact.tolist(),
            "terminal_semantic_yaw_rad": target_yaw,
        },
        "contact_seed": {
            "source": resolved_contact.contact_seed_source,
            "surface_cell": list(resolved_contact.contact_surface_cell) if resolved_contact.contact_surface_cell is not None else None,
            "true_support_mask_validated": resolved_contact.support_mask is not None,
        },
        "fixed_world_variables": {"translation": "pose_derived_from_fixed_gluteal_contact", "yaw": target_yaw,
                                  "scale": "fixed_from_neutral_SMPLX_carrier", "furniture_switch_allowed": False},
        "optimizer_steps": steps, "optimizer_initialization_seed": seed,
        "objective": {
            "p523_base": {"non_target_sdf": 14.0, "target_forbidden_sdf": 18.0, "allowed_contact_tail": 30.0,
                          "target_contact_surface": 12.0, "ground_penetration": 30.0, "bilateral_foot_surface": 120.0,
                          "pose_prior": 0.20},
            "hybrid_enabled": ablation_mode in {"hybrid", "hybrid_reachable"}, "lower_limb_mean_top10_max_sdf": float(lower_limb_tail_weight),
            "initial_projection_preservation": float(projection_weight),
            "optional_depth_upper_envelope_requested": requested_depth_weight,
            "optional_depth_upper_envelope_applied": float(depth_weight),
            "depth_margin_m": float(depth_margin_m),
        },
        "adaptive_joint_bounds": adaptive_receipt,
        "projection_observation": camera_receipt,
        "metrics": best_metrics, "gates": best_gates, "best_rank": list(best_rank),
        "contact_lock_error_m": contact_error, "optimization_history": history,
        "same_surface_contact_search": search_history, "sdf_target_binding": dict(fields.component_receipt),
        "publication_thresholds": dict(publication_thresholds), "keypose": contract.artifact(keypose_path),
        "inputs": {
            "stage1_target": contract.artifact(target_path), "p533_h3_hybrikx_source": contract.artifact(source_path),
            "scene_occupancy": contract.artifact(occupancy_path), "target_occupancy_mask": contract.artifact(target_mask_path),
            "fixed_neutral_smplx": contract.artifact(smplx_path), "smplx_segmentation": contract.artifact(segmentation_path),
        },
        "lineage": {
            "articulation_prior": "MiniMax-H3_single_image_to_HybrIK-X_SMPL-X",
            "source_hashes": source["lineage"], "camera_recovery_used_as_world_placement": False,
            "stage1_target_identity_surface_contact_yaw_authoritative": True,
            "free_translation_yaw_scale": False, "furniture_switch_allowed": False,
            "neutral_smplx_carrier_allowed": True,
        },
        "stage3_handoff_allowed": published,
    }
    if ablation_mode == "hybrid_reachable":
        search_summary = _search_receipt_summary(
            [candidate_id for candidate_id, _ in search_pose_candidates],
            selected_pose_candidate_id,
            search_history,
        )
        result["selected_pose_candidate_id"] = selected_pose_candidate_id
        result["same_surface_contact_search_summary"] = search_summary
        result["carrier_projection"] = carrier_projection
        result["objective"]["reachability_prior"] = {
            "support_root_relative_weight": 8.0,
            "bilateral_delta_symmetry_weight": 6.0,
            "shin_direction_weight": 0.25,
            "additional_pose_geodesic_weight": 0.30,
            "fixed_gluteal_contact_area_weight": 100.0,
            "strict_grid_boundary_weight": 150.0,
            "reference": "H3 articulation rematerialized on Stage3 male zero-beta carrier",
        }
        result["reachability_gate"] = {
            "schema": "p533.hybrid_reachable_gate.v1",
            "fail_closed": True,
            "all_existing_physics_gates_retained": True,
            "thresholds": {
                "maximum_support_root_relative_delta_mean_m": REACHABLE_MAX_SUPPORT_DELTA_MEAN_M,
                "maximum_support_root_relative_delta_max_m": REACHABLE_MAX_SUPPORT_DELTA_MAX_M,
                "maximum_shin_direction_delta_deg": REACHABLE_MAX_SHIN_ANGLE_DEG,
                "maximum_bilateral_delta_asymmetry_mean_m": REACHABLE_MAX_BILATERAL_DELTA_ASYMMETRY_M,
                "maximum_support_pose_geodesic_rad": REACHABLE_MAX_SUPPORT_POSE_GEODESIC_RAD,
            },
            "metrics": {
                key: best_metrics[key]
                for key in (
                    "support_root_relative_delta_mean_m",
                    "support_root_relative_delta_max_m",
                    "support_root_relative_delta_per_joint_m",
                    "bilateral_delta_asymmetry_mean_m",
                    "bilateral_delta_asymmetry_max_m",
                    "left_shin_direction_delta_deg",
                    "right_shin_direction_delta_deg",
                    "maximum_shin_direction_delta_deg",
                    "support_pose_geodesic_mean_rad",
                    "support_pose_geodesic_max_rad",
                )
            },
            "gates": {
                key: best_gates[key]
                for key in (
                    "stage3_male_zero_beta_carrier_exact",
                    "support_root_relative_delta_mean",
                    "support_root_relative_delta_max",
                    "shin_direction_prior",
                    "bilateral_lower_body_prior",
                    "support_pose_deviation_bounded",
                )
            },
            "selected_pose_candidate_id": selected_pose_candidate_id,
        }
        result["lineage"].update({
            "stage3_male_zero_beta_carrier_projected_before_refine": True,
            "carrier_projection_parent_keypose_sha256": contract.sha256_file(source_path),
            "carrier_projection_model_sha256": contract.sha256_file(smplx_path),
            "fixed_gluteal_subset_sha256": fixed_subset_sha256,
            "old_unconditional_support_override_used": False,
            "hip_knee_ankle_limits_decoupled": True,
            "candidate_profile": adaptive_receipt["candidate_profile_name"],
        })
    p498.atomic_json(output_dir / "receipt.json", result)
    return result
