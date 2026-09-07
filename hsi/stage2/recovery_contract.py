"""Complete fresh single-image HybrIK prior/coordinate-chain validation."""
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

from . import image_contract

RAW_SCHEMA = "p533.hybrikx_single_image_recovery.raw.v1"

PRIOR_SCHEMA = "p533.fresh_h3_hybrikx_articulation.v1"

PRIOR_RECEIPT_SCHEMA = "p553.fresh_h3_image_hybrikx_articulation_receipt.v1"

H3_RECEIPT_SCHEMA = image_contract.SCHEMA

NATIVE_YUP_TO_MATERIALIZER_ZUP = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)

class P533ContractError(RuntimeError):
    """Raised when a P533 artifact or cross-stage binding is invalid."""

def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise P533ContractError(message)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value

def artifact(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}

def scalar(archive: Mapping[str, np.ndarray], key: str) -> Any:
    require(key in archive, f"P533 archive lacks {key}")
    value = np.asarray(archive[key])
    require(value.size == 1, f"P533 archive field {key} must be scalar")
    return value.reshape(()).item()

def _artifact_path(record: Any, label: str) -> Path:
    require(isinstance(record, Mapping), f"{label} artifact record is missing")
    raw = str(record.get("path", "")).strip()
    require(raw, f"{label}.path is missing")
    path = Path(raw).resolve(strict=True)
    require(int(record.get("bytes", -1)) == path.stat().st_size, f"{label} byte count drift")
    require(str(record.get("sha256", "")) == sha256_file(path), f"{label} SHA-256 drift")
    return path

def _receipt_payload_hash(receipt: Mapping[str, Any], label: str) -> str:
    declared = str(receipt.get("receipt_payload_sha256", ""))
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    require(len(declared) == 64 and canonical_hash(payload) == declared, f"{label} payload hash drift")
    return declared

def _receipt_artifact(
    receipt: Mapping[str, Any], containers: tuple[str, ...], aliases: tuple[str, ...], label: str
) -> Mapping[str, Any]:
    for container_name in containers:
        container = receipt.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for alias in aliases:
            value = container.get(alias)
            if isinstance(value, Mapping):
                return value
    raise P533ContractError(f"{label} artifact is missing (aliases={aliases})")

def validate_h3_hybrikx_prior(
    prior_path: Path,
    receipt_path: Path,
    *,
    scene_id: str,
    instruction: str,
    target_instance_id: str,
    target_surface_id: str,
) -> dict[str, Any]:
    """Validate a placement-free H3/HybrIK-X articulation prior and lineage."""

    prior_path = Path(prior_path).resolve(strict=True)
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == PRIOR_RECEIPT_SCHEMA, "P533 articulation receipt schema drift")
    require(
        str(receipt.get("status", "")).startswith(("complete", "published")),
        "P533 articulation receipt is incomplete",
    )
    receipt_payload_sha = _receipt_payload_hash(receipt, "P533 articulation receipt")
    for key, expected in {
        "scene_id": scene_id,
        "instruction": instruction,
        "target_instance_id": target_instance_id,
        "target_surface_id": target_surface_id,
    }.items():
        require(receipt.get(key) == expected, f"P533 articulation receipt {key} drift")
    nonce = str(receipt.get("run_nonce", ""))
    require(len(nonce) >= 16, "P533 articulation run nonce is weak")

    prior_record = _receipt_artifact(
        receipt, ("outputs", "output"), ("articulation_prior", "fresh_articulation_prior"),
        "P533 articulation prior",
    )
    raw_record = _receipt_artifact(
        receipt, ("inputs", "outputs"), ("hybrikx_raw_recovery", "raw_recovery"),
        "HybrIK-X raw recovery",
    )
    frame_record = _receipt_artifact(
        receipt, ("inputs",), ("h3_image",),
        "H3 single image",
    )
    h3_receipt_record = _receipt_artifact(
        receipt, ("inputs",), ("h3_generation_receipt",), "H3 generation receipt"
    )
    camera_record = _receipt_artifact(
        receipt, ("inputs",), ("stage1_camera", "current_stage1_camera"), "Stage1 camera"
    )
    require(_artifact_path(prior_record, "P533 articulation prior") == prior_path,
            "P533 receipt points to another prior")
    raw_path = _artifact_path(raw_record, "HybrIK-X raw recovery")
    frame_path = _artifact_path(frame_record, "H3 frame")
    h3_receipt_path = _artifact_path(h3_receipt_record, "H3 generation receipt")
    camera_path = _artifact_path(camera_record, "Stage1 camera")

    h3_receipt = image_contract.validate_receipt(h3_receipt_path, expected_case_id=receipt.get("case_id"), expected_image=frame_path)
    require(h3_receipt["inputs"]["camera"] == camera_record, "H3/HybrIK camera drift")
    require(receipt["models"]["h3_model_manifest"] == h3_receipt["model_components"]["transformer"], "HybrIK bound wrong H3 model")
    require(receipt["inputs"]["stage1_artifact"] == h3_receipt["source_binding"]["stage1_bundle"], "H3/HybrIK Stage1 drift")
    with np.load(prior_path, allow_pickle=False) as provider_arrays:
        require(str(scalar(provider_arrays, "source_h3_model_manifest_sha256")) == h3_receipt["model_components"]["transformer"]["sha256"], "NPZ bound wrong H3 model")
    require(h3_receipt.get("schema") == H3_RECEIPT_SCHEMA, "H3 generation receipt schema drift")
    require(str(h3_receipt.get("status", "")).startswith("complete"), "H3 generation is incomplete")
    h3_payload_sha = _receipt_payload_hash(h3_receipt, "H3 generation receipt")

    with np.load(raw_path, allow_pickle=False) as raw:
        require(str(scalar(raw, "schema")) == RAW_SCHEMA, "HybrIK-X raw schema drift")
        require(np.asarray(raw["pred_theta_mat"]).shape == (55, 3, 3), "HybrIK-X rotation shape drift")
        require(np.asarray(raw["pred_shape_full"]).shape == (21,), "HybrIK-X shape vector drift")

    prior_sha = sha256_file(prior_path)
    with np.load(prior_path, allow_pickle=False) as prior:
        require(str(scalar(prior, "schema")) == PRIOR_SCHEMA, "P533 prior schema drift")
        require("published" in str(scalar(prior, "status")), "P533 prior is not published")
        for key, expected in {
            "scene_id": scene_id,
            "instruction": instruction,
            "target_instance_id": target_instance_id,
            "target_surface_id": target_surface_id,
            "source_candidate_id": target_surface_id,
            "run_nonce": nonce,
            "source_h3_frame_sha256": sha256_file(frame_path),
            "source_raw_recovery_sha256": sha256_file(raw_path),
        }.items():
            require(str(scalar(prior, key)) == str(expected), f"P533 prior {key} drift")
        for key, shape in {
            "body_pose_axis_angle": (21, 3),
            "betas": (10,),
            "vertices_world_zup": (10475, 3),
            "root_xyz_yaw": (4,),
            "pose_frame_to_world_rotation": (3, 3),
        }.items():
            value = np.asarray(prior[key])
            require(value.shape == shape and np.isfinite(value).all(), f"P533 prior {key} shape/value drift")
        joints = np.asarray(prior["joints_world_zup"])
        faces = np.asarray(prior["faces"])
        require(joints.ndim == 2 and joints.shape[0] >= 22 and joints.shape[1] == 3,
                "P533 prior joint shape drift")
        require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) == 20908,
                "P533 prior topology drift")

        # HybrIK-X rotations act on native SMPL-X Y-up coordinates, while the
        # inherited P478/P509 materializer converts the body to Z-up before it
        # applies ``pose_frame_to_world_rotation``.  This boundary is easy to
        # get wrong while still producing finite, correctly shaped arrays (the
        # original failure laid a seated body sideways through the target).
        # Recompute both paths here so an old or regressed adapter cannot pass
        # merely by declaring the right coordinate-system labels.
        coordinate_fields = (
            "native_yup_to_materializer_zup_rotation",
            "native_yup_to_world_yaw0_rotation",
            "vertices_native_yup",
            "vertices_materializer_zup",
            "joints_native_yup",
            "joints_materializer_zup",
            "vertex_coordinate_chain_max_error_m",
            "joint_coordinate_chain_max_error_m",
            "pose_frame_factorization_max_error",
            "native_materializer_up_axis_error",
            "coordinate_chain",
            "pose_frame_consumes_materializer_zup",
        )
        require(all(key in prior for key in coordinate_fields),
                "P533 prior lacks the audited Y-up/Z-up coordinate-chain evidence")
        conversion = np.asarray(prior["native_yup_to_materializer_zup_rotation"], dtype=np.float64)
        native_to_world = np.asarray(prior["native_yup_to_world_yaw0_rotation"], dtype=np.float64)
        frame = np.asarray(prior["pose_frame_to_world_rotation"], dtype=np.float64)
        require(conversion.shape == (3, 3) and native_to_world.shape == (3, 3),
                "P533 coordinate rotations are malformed")
        require(np.allclose(conversion, NATIVE_YUP_TO_MATERIALIZER_ZUP, atol=1.0e-7),
                "P533 native-Y-up/materializer-Z-up conversion drift")
        for value, label in ((native_to_world, "native-to-world"), (frame, "pose-frame")):
            require(
                np.allclose(value.T @ value, np.eye(3), atol=2.0e-5)
                and np.isclose(np.linalg.det(value), 1.0, atol=2.0e-5),
                f"P533 {label} rotation is not proper",
            )
        factorization_error = float(np.abs(frame @ conversion - native_to_world).max(initial=0.0))
        require(factorization_error <= 2.0e-6,
                "P533 pose frame omits or duplicates the Y-up/Z-up basis change")

        vertices_native = np.asarray(prior["vertices_native_yup"], dtype=np.float64)
        vertices_materializer = np.asarray(prior["vertices_materializer_zup"], dtype=np.float64)
        vertices_world = np.asarray(prior["vertices_world_zup"], dtype=np.float64)
        joints_native = np.asarray(prior["joints_native_yup"], dtype=np.float64)
        joints_materializer = np.asarray(prior["joints_materializer_zup"], dtype=np.float64)
        require(vertices_native.shape == vertices_materializer.shape == vertices_world.shape,
                "P533 vertex coordinate-chain shapes drift")
        require(joints_native.shape == joints_materializer.shape == joints.shape,
                "P533 joint coordinate-chain shapes drift")
        vertex_basis_error = float(
            np.linalg.norm(vertices_materializer - vertices_native @ conversion.T, axis=1).max(initial=0.0)
        )
        vertex_world_error = float(
            np.linalg.norm(vertices_world - vertices_materializer @ frame.T, axis=1).max(initial=0.0)
        )
        vertex_direct_error = float(
            np.linalg.norm(vertices_world - vertices_native @ native_to_world.T, axis=1).max(initial=0.0)
        )
        joint_basis_error = float(
            np.linalg.norm(joints_materializer - joints_native @ conversion.T, axis=1).max(initial=0.0)
        )
        joint_world_error = float(
            np.linalg.norm(joints - joints_materializer @ frame.T, axis=1).max(initial=0.0)
        )
        require(
            max(vertex_basis_error, vertex_world_error, vertex_direct_error,
                joint_basis_error, joint_world_error) <= 2.0e-6,
            "P533 serialized coordinate-chain arrays do not round-trip",
        )
        for key in (
            "vertex_coordinate_chain_max_error_m",
            "joint_coordinate_chain_max_error_m",
            "pose_frame_factorization_max_error",
            "native_materializer_up_axis_error",
        ):
            require(float(scalar(prior, key)) <= 2.0e-6, f"P533 prior {key} exceeds tolerance")
        require(str(scalar(prior, "coordinate_chain"))
                == "smplx_native_yup__C_to_materializer_zup__pose_frame_to_world_zup",
                "P533 coordinate-chain declaration drift")
        require(bool(scalar(prior, "pose_frame_consumes_materializer_zup")),
                "P533 pose-frame input basis is not explicitly bound")
        for key in (
            "fresh_h3_used", "fresh_hybrikx_used", "articulation_only",
            "camera_world_placement_discarded", "camera_projection_evidence_retained",
            "neutral_smplx_carrier_used", "kid_shape_folded_to_ten_betas",
        ):
            require(bool(scalar(prior, key)), f"P533 prior requires {key}=true")
        for key in (
            "fresh_qwen_image_used", "fresh_rompv2_used", "old_world_placement_consumed",
            "legacy_world_placement_reused", "gt_root_path_pose_motion_contact_icgf_used",
            "lingo_pose_or_motion_donor_used", "retrieval_or_positive_memory_used",
            "handwritten_task_pose_template_used", "motion_repair_used",
        ):
            require(key in prior and not bool(scalar(prior, key)),
                    f"P533 prior requires {key}=false")

    lineage = receipt.get("lineage")
    require(isinstance(lineage, Mapping), "P533 articulation receipt lacks lineage")
    for key in (
        "fresh_h3_used", "fresh_hybrikx_used", "articulation_only",
        "camera_world_placement_discarded", "camera_projection_evidence_retained",
    ):
        require(lineage.get(key) is True, f"P533 receipt requires lineage.{key}=true")
    for key in ("fresh_qwen_image_used", "fresh_rompv2_used", "old_world_placement_consumed"):
        require(lineage.get(key) is False, f"P533 receipt requires lineage.{key}=false")

    return {
        "schema": PRIOR_SCHEMA,
        "receipt_schema": PRIOR_RECEIPT_SCHEMA,
        "recovery_schema": RAW_SCHEMA,
        "prior_sha256": prior_sha,
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_payload_sha256": receipt_payload_sha,
        "run_nonce": nonce,
        "generated_at_utc": receipt.get("generated_at_utc", receipt.get("created_at_utc")),
        "source_input_rgb_sha256": h3_receipt["artifacts"]["h3_condition_image"]["sha256"],
        "source_h3_frame_sha256": sha256_file(frame_path),
        "source_raw_recovery_sha256": sha256_file(raw_path),
        "hybrikx_recovery_receipt_sha256": sha256_file(receipt_path),
        "hybrikx_recovery_payload_sha256": receipt_payload_sha,
        "h3_generation_receipt_sha256": sha256_file(h3_receipt_path),
        "h3_generation_payload_sha256": h3_payload_sha,
        "stage1_camera_sha256": sha256_file(camera_path),
        "artifacts": {
            "h3_image": artifact(frame_path),
            "h3_generation_receipt": artifact(h3_receipt_path),
            "hybrikx_raw_recovery": artifact(raw_path),
            "stage1_camera": artifact(camera_path),
            "articulation_prior": artifact(prior_path),
            "articulation_receipt": artifact(receipt_path),
        },
        "fresh_h3_used": True,
        "fresh_hybrikx_used": True,
        "fresh_qwen_image_used": False,
        "fresh_rompv2_used": False,
        "camera_world_placement_discarded": True,
        "camera_projection_evidence_retained": True,
        "old_world_placement_consumed": False,
    }
