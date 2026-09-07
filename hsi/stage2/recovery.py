"""Static, genuine HybrIK-X single-image recovery with explicit external assets.

The detector, 55-joint SO(3), adult-shape folding and camera-frame math are
retained. No historical module is loaded or monkeypatched. The output is an
articulation prior, never a world-placed or physically approved keypose.
Run model recovery in an isolated process before torch has been imported.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
import numpy as np
from PIL import Image, ImageDraw
from . import image_contract

HERE = Path(__file__).resolve()
RAW_SCHEMA = "p533.hybrikx_single_image_recovery.raw.v1"
PRIOR_SCHEMA = "p533.fresh_h3_hybrikx_articulation.v1"
RECEIPT_SCHEMA = "p553.fresh_h3_image_hybrikx_articulation_receipt.v1"
H3_RECEIPT_SCHEMA = image_contract.SCHEMA
EXPECTED_IMAGE_SIZE = (512, 512)
BODY_JOINT_COUNT = 21
FULL_SMPLX_JOINT_COUNT = 55
NATIVE_YUP_TO_MATERIALIZER_ZUP = np.asarray(((1.,0.,0.),(0.,0.,-1.),(0.,1.,0.)),dtype=np.float64)

@dataclass(frozen=True)
class RuntimeConfig:
    hybrik_root: Path
    hybrik_config: Path
    hybrik_checkpoint: Path
    detector_checkpoint: Path
    neutral_smplx: Path
    kid_template: Path
    extra_python_paths: tuple[Path, ...] = ()

    @classmethod
    def from_mapping(cls, value):
        required = {"hybrik_root", "hybrik_config", "hybrik_checkpoint", "detector_checkpoint", "neutral_smplx", "kid_template"}
        image_contract.require(set(value) <= required | {"extra_python_paths"} and required <= set(value), "External recovery paths must be explicit")
        paths = {key: Path(value[key]).resolve(strict=True) for key in required}
        image_contract.require(paths["hybrik_root"].is_dir(), "HybrIK root must be an installed external project")
        image_contract.require(all(path.is_file() for key,path in paths.items() if key != "hybrik_root"), "Recovery asset missing")
        image_contract.require(paths["hybrik_config"].is_relative_to(paths["hybrik_root"]), "HybrIK config must be inside installed root")
        extra = tuple(Path(path).resolve(strict=True) for path in value.get("extra_python_paths", ()))
        image_contract.require(all(path.is_dir() for path in extra), "External Python dependency directory missing")
        return cls(**paths, extra_python_paths=extra)

class P533HybrIKError(RuntimeError):
    """Raised when an input or recovered artifact violates the P533 contract."""


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise P533HybrIKError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P533HybrIKError(f"invalid JSON: {path}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    require(path.is_absolute() or base is not None, "Relative artifact requires an explicit base directory")
    return (path if path.is_absolute() else Path(base) / path).resolve(strict=True)


def _path_value(value: Any, label: str, *, base: Path) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path")
    require(isinstance(value, str) and bool(value.strip()), f"{label} path is missing")
    return resolve_path(value, base=base)


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


def project_so3(matrix: np.ndarray) -> np.ndarray:
    """Project one or more 3x3 matrices to proper SO(3) with SVD."""

    value = np.asarray(matrix, dtype=np.float64)
    require(value.shape[-2:] == (3, 3), "SO(3) input must end in [3,3]")
    require(np.isfinite(value).all(), "SO(3) input contains non-finite values")
    flat = value.reshape(-1, 3, 3)
    projected = np.empty_like(flat)
    for index, item in enumerate(flat):
        left, _singular, right_t = np.linalg.svd(item)
        rotation = left @ right_t
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_t
        projected[index] = rotation
    return projected.reshape(value.shape)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation to axis-angle, including the pi branch."""

    rotation = project_so3(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1.0e-7:
        return np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        ) * 0.5
    if math.pi - angle <= 1.0e-5:
        # The usual skew/sin(theta) expression is unstable at pi.  The
        # eigenvector with eigenvalue one is the rotation axis.
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        norm = float(np.linalg.norm(axis))
        require(norm > 1.0e-8, "cannot recover pi-rotation axis")
        axis /= norm
        skew_hint = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        )
        if np.linalg.norm(skew_hint) > 1.0e-8 and np.dot(axis, skew_hint) < 0.0:
            axis *= -1.0
        return axis * angle
    axis = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def rotations_to_axis_angle(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float64)
    require(rotations.shape[-2:] == (3, 3), "rotation array must end in [3,3]")
    return np.stack([matrix_to_axis_angle(row) for row in rotations.reshape(-1, 3, 3)]).reshape(
        rotations.shape[:-2] + (3,)
    )


def rotation_z(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


@dataclass(frozen=True)
class Camera:
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int
    schema: str

    @classmethod
    def load(cls, path: Path) -> "Camera":
        raw = read_json(path)
        world_to_camera = np.asarray(raw.get("world_to_camera"), dtype=np.float64)
        require(world_to_camera.shape == (4, 4), "camera world_to_camera must be [4,4]")
        require(np.isfinite(world_to_camera).all(), "camera world_to_camera is not finite")
        require(
            np.allclose(world_to_camera[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8),
            "camera homogeneous row is invalid",
        )
        rotation = world_to_camera[:3, :3]
        require(
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5)
            and np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5),
            "camera world_to_camera rotation is not proper",
        )
        intrinsics = np.asarray(raw.get("intrinsics", raw.get("K")), dtype=np.float64)
        require(intrinsics.shape == (3, 3), "camera intrinsics must be [3,3]")
        require(np.isfinite(intrinsics).all(), "camera intrinsics are not finite")
        width, height = raw.get("width"), raw.get("height")
        require(type(width) is int and type(height) is int, "camera dimensions are invalid")
        return cls(world_to_camera, intrinsics, int(width), int(height), str(raw.get("schema", "")))

    @property
    def camera_to_world_rotation(self) -> np.ndarray:
        # Inverse of a proper rotation, deliberately excluding camera translation.
        return self.world_to_camera[:3, :3].T


@dataclass(frozen=True)
class RootDecomposition:
    root_rotation_camera: np.ndarray
    root_rotation_world: np.ndarray
    native_yup_to_world_yaw0_rotation: np.ndarray
    pose_frame_to_world_rotation: np.ndarray
    recovered_yaw_world_rad: float
    yaw_residual_rad: float
    heading_axis: str


def decompose_root_rotation(root_camera: np.ndarray, camera: Camera) -> RootDecomposition:
    """Remove world-Z yaw while retaining gravity-aware pitch and roll.

    HybrIK-X root rotation is camera-relative.  The exact calibrated
    ``world_to_camera`` rotation first carries it into LINGO world coordinates.
    SMPL-X's canonical forward direction is +Z.  Left multiplication by the
    inverse recovered world yaw first produces a yaw-zero mapping from native
    SMPL-X Y-up to world Z-up.  P509 applies a fixed Y-up-to-Z-up conversion
    before its serialized frame, so the published frame additionally includes
    the exact inverse of that fixed conversion.  Pitch/roll and gravity are
    preserved across both paths.
    """

    root_camera = project_so3(np.asarray(root_camera, dtype=np.float64).reshape(3, 3))
    root_world = project_so3(camera.camera_to_world_rotation @ root_camera)
    forward = root_world @ np.asarray((0.0, 0.0, 1.0))
    horizontal = float(np.linalg.norm(forward[:2]))
    heading_axis = "smplx_forward_plus_z"
    if horizontal <= 1.0e-7:
        # A near-vertical forward vector can occur for lying poses.  The local
        # +X axis is a deterministic secondary heading; this affects yaw only.
        forward = root_world @ np.asarray((1.0, 0.0, 0.0))
        horizontal = float(np.linalg.norm(forward[:2]))
        heading_axis = "fallback_smplx_right_plus_x"
    require(horizontal > 1.0e-7, "recovered root has no stable horizontal heading")
    recovered_yaw = math.atan2(float(forward[1]), float(forward[0]))
    native_yup_to_world_yaw0 = project_so3(rotation_z(-recovered_yaw) @ root_world)
    pose_frame = project_so3(
        native_yup_to_world_yaw0 @ NATIVE_YUP_TO_MATERIALIZER_ZUP.T
    )
    check_axis = (0.0, 0.0, 1.0) if heading_axis == "smplx_forward_plus_z" else (1.0, 0.0, 0.0)
    check_forward = native_yup_to_world_yaw0 @ np.asarray(check_axis)
    yaw_residual = math.atan2(float(check_forward[1]), float(check_forward[0]))
    require(abs(yaw_residual) <= 1.0e-6, "pose-frame yaw normalization failed")
    require(
        np.allclose(
            pose_frame @ NATIVE_YUP_TO_MATERIALIZER_ZUP,
            native_yup_to_world_yaw0,
            atol=1.0e-8,
        ),
        "materializer pose-frame factorization failed",
    )
    return RootDecomposition(
        root_rotation_camera=root_camera,
        root_rotation_world=root_world,
        native_yup_to_world_yaw0_rotation=native_yup_to_world_yaw0,
        pose_frame_to_world_rotation=pose_frame,
        recovered_yaw_world_rad=float(recovered_yaw),
        yaw_residual_rad=float(yaw_residual),
        heading_axis=heading_axis,
    )


@dataclass(frozen=True)
class ShapeFold:
    original_beta11: np.ndarray
    folded_beta10: np.ndarray
    target_displacement: np.ndarray
    reconstructed_displacement: np.ndarray
    residual_rms_m: float
    residual_max_m: float
    design_rank: int
    design_condition: float


def fold_kid_beta_to_neutral(
    beta11: np.ndarray,
    neutral_model: Path,
    kid_template: Path,
) -> ShapeFold:
    """Least-squares fold HybrIK-X's kid coefficient into ten adult betas."""

    beta11 = np.asarray(beta11, dtype=np.float64).reshape(-1)
    require(beta11.shape == (11,), "HybrIK-X beta vector must contain 11 coefficients")
    require(np.isfinite(beta11).all(), "HybrIK-X beta vector is not finite")
    with np.load(neutral_model.resolve(strict=True), allow_pickle=True) as archive:
        shapedirs = np.asarray(archive["shapedirs"], dtype=np.float64)[..., :10]
        adult_template = np.asarray(archive["v_template"], dtype=np.float64)
    kid = np.asarray(np.load(kid_template.resolve(strict=True)), dtype=np.float64)
    require(shapedirs.shape == (10475, 3, 10), "neutral SMPL-X shapedirs drift")
    require(kid.shape == adult_template.shape == (10475, 3), "SMPL-X/kid template shape drift")
    # This exactly mirrors HybrIK-X SMPLX(age='kid'): the kid template is
    # centered before the extra shape direction is appended.
    kid_centered = kid - kid.mean(axis=0, keepdims=True)
    kid_direction = kid_centered - adult_template
    design = shapedirs.reshape(-1, 10)
    target = (shapedirs * beta11[None, None, :10]).sum(axis=2)
    target += kid_direction * float(beta11[10])
    folded, _residuals, rank, singular = np.linalg.lstsq(design, target.reshape(-1), rcond=None)
    reconstructed = (shapedirs * folded[None, None, :]).sum(axis=2)
    residual = reconstructed - target
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else float("inf")
    return ShapeFold(
        original_beta11=beta11.astype(np.float32),
        folded_beta10=folded.astype(np.float32),
        target_displacement=target.astype(np.float32),
        reconstructed_displacement=reconstructed.astype(np.float32),
        residual_rms_m=float(np.sqrt(np.mean(np.square(residual)))),
        residual_max_m=float(np.linalg.norm(residual, axis=1).max(initial=0.0)),
        design_rank=int(rank),
        design_condition=condition,
    )


def _make_hybrik_transform(cfg: Any, edict: Any, transform_class: Any) -> Any:
    shape = [value * 1.0e-3 for value in cfg.MODEL.BBOX_3D_SHAPE]
    dummy = edict(
        joint_pairs_17=None,
        joint_pairs_24=None,
        joint_pairs_29=None,
        bbox_3d_shape=shape,
    )
    return transform_class(
        dummy,
        scale_factor=cfg.DATASET.SCALE_FACTOR,
        color_factor=cfg.DATASET.COLOR_FACTOR,
        occlusion=cfg.DATASET.OCCLUSION,
        input_size=cfg.MODEL.IMAGE_SIZE,
        output_size=cfg.MODEL.HEATMAP_SIZE,
        depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
        bbox_3d_shape=shape,
        rot=cfg.DATASET.ROT_FACTOR,
        sigma=cfg.MODEL.EXTRA.SIGMA,
        train=False,
        add_dpg=False,
        loss_type=cfg.LOSS.TYPE,
    )


def _tensor_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = value.detach().cpu().numpy()[0]
    return array.reshape(shape) if shape is not None else array


def _load_runtime(gpu: str, runtime: RuntimeConfig) -> tuple[Any, ...]:
    require('torch' not in sys.modules, 'HybrIK recovery requires an isolated process before torch import')
    # Set visibility before importing torch; the requested physical GPU then
    # becomes cuda:0 inside this isolated process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for path in [str(runtime.hybrik_root), *map(str, runtime.extra_python_paths)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    # HybrIK-X's preset module constructs a neutral SMPL-X layer at import
    # time using a repository-relative ``model_files/...`` path.  Import it
    # under the official repository root as well as running inference there.
    with working_directory(runtime.hybrik_root):
        import torch
        from easydict import EasyDict as edict
        from hybrik.models import builder
        from hybrik.models.layers.smplx.body_models import SMPLXLayer
        from hybrik.utils.config import update_config
        from hybrik.utils.presets import SimpleTransform3DSMPLX
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        from torchvision.transforms.functional import pil_to_tensor

    require(torch.cuda.is_available(), f"CUDA GPU is unavailable after selecting {gpu!r}")
    return (
        torch,
        edict,
        builder,
        SMPLXLayer,
        update_config,
        SimpleTransform3DSMPLX,
        keypointrcnn_resnet50_fpn,
        pil_to_tensor,
    )


def _render_reprojection(
    frame_path: Path,
    uvd: np.ndarray,
    transformed_bbox: np.ndarray,
    detector_bbox: np.ndarray,
    detector_score: float,
    output_path: Path,
) -> dict[str, Any]:
    """Retain auditable image-space evidence without using it for placement."""

    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = np.asarray(transformed_bbox, dtype=np.float64).reshape(4)
    width, height = x2 - x1, y2 - y1
    require(width > 0.0 and height > 0.0, "HybrIK-X transformed bbox is invalid")
    points = np.asarray(uvd, dtype=np.float64).reshape(-1, 3)[:, :2].copy()
    points[:, 0] = (points[:, 0] + 0.5) * width + x1
    points[:, 1] = (points[:, 1] + 0.5) * height + y1
    dx1, dy1, dx2, dy2 = np.asarray(detector_bbox, dtype=np.float64).reshape(4)
    draw.rectangle((dx1, dy1, dx2, dy2), outline=(255, 180, 0), width=3)
    for index, (x, y) in enumerate(points):
        radius = 4 if index < 22 else 2
        colour = (0, 230, 80) if index < 22 else (40, 180, 255)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
    draw.text((max(2.0, dx1), max(2.0, dy1 - 18.0)), f"person {detector_score:.3f}", fill=(255, 180, 0))
    image.save(output_path, format="PNG", optimize=True)
    canvas_width, canvas_height = image.size
    key = points[:22]
    inside = (
        (key[:, 0] >= 0.0)
        & (key[:, 0] < canvas_width)
        & (key[:, 1] >= 0.0)
        & (key[:, 1] < canvas_height)
    )
    return {
        "body22_inside_image_fraction": float(inside.mean()),
        "body22_bbox_xyxy": [
            float(key[:, 0].min()),
            float(key[:, 1].min()),
            float(key[:, 0].max()),
            float(key[:, 1].max()),
        ],
        "uvd_to_pixel_rule": "pixel=(pred_uvd_xy+0.5)*hybrik_transformed_bbox_wh+bbox_xyxy_min",
    }


def _select_person(detection: Mapping[str, Any]) -> tuple[int, float, np.ndarray]:
    scores = detection["scores"].detach().cpu().numpy().astype(np.float64)
    boxes = detection["boxes"].detach().cpu().numpy().astype(np.float64)
    labels = detection["labels"].detach().cpu().numpy().astype(np.int64)
    require(len(scores) > 0, "Keypoint R-CNN found no person")
    person_indices = np.flatnonzero(labels == 1)
    require(len(person_indices) > 0, "Keypoint R-CNN found no COCO person instance")
    index = int(person_indices[np.argmax(scores[person_indices])])
    return index, float(scores[index]), boxes[index]


def _model_artifact_contract(runtime: RuntimeConfig, h3_model_manifest: Path) -> dict[str, Any]:
    paths = {
        "hybrikx_config": runtime.hybrik_config,
        "hybrikx_checkpoint": runtime.hybrik_checkpoint,
        "keypoint_rcnn_checkpoint": runtime.detector_checkpoint,
        "neutral_smplx": runtime.neutral_smplx,
        "kid_template": runtime.kid_template,
        "h3_model_manifest": h3_model_manifest,
    }
    for label, path in paths.items():
        require(path.is_file(), f"missing {label}: {path}")
    return {label: artifact(path) for label, path in paths.items()}


@dataclass(frozen=True)
class Inputs:
    frame: Path
    camera: Path
    stage1: Path
    config: Path
    output_dir: Path
    h3_receipt: Path | None
    case_id: str | None


def _derive_identity(config: Mapping[str, Any], inputs: Inputs) -> dict[str, Any]:
    binding = config.get("stage1_binding", {})
    require(isinstance(binding, Mapping), "config stage1_binding must be an object")
    scene_id = str(config.get("scene_id", inputs.case_id or ""))
    instruction = str(config.get("instruction", ""))
    require(bool(scene_id), "config scene_id is missing")
    require(bool(instruction), "config instruction is missing")
    frame_sha = sha256_file(inputs.frame)
    run_nonce = str(config.get("run_nonce", "")).strip()
    if len(run_nonce) < 16:
        # A content-derived nonce makes standalone explicit invocations usable
        # without claiming that the H3 campaign supplied a nonce it does not
        # currently emit.
        run_nonce = f"p533-{inputs.case_id or scene_id}-{frame_sha[:20]}"
    return {
        "case_id": inputs.case_id or scene_id,
        "scene_id": scene_id,
        "instruction": instruction,
        "action_family": str(config.get("action_family", "unknown")),
        "target_instance_id": str(binding.get("target_instance_id", config.get("target_instance_id", ""))),
        "target_surface_id": str(binding.get("target_surface_id", config.get("target_surface_id", ""))),
        "run_nonce": run_nonce,
    }


def run(inputs: Inputs, *, gpu: str, runtime: RuntimeConfig) -> dict[str, Any]:
    require(not inputs.output_dir.exists(), f"refusing to overwrite output directory: {inputs.output_dir}")
    image = Image.open(inputs.frame).convert("RGB")
    require(image.size == EXPECTED_IMAGE_SIZE, f"H3 frame must be 512x512, got {image.size}")
    config = read_json(inputs.config)
    identity = _derive_identity(config, inputs)
    require(inputs.h3_receipt is not None, "--h3-receipt is required for a publishable fresh-H3 prior")
    h3_receipt = _validate_bound_image(inputs)
    if inputs.case_id is None:
        identity["case_id"] = str(h3_receipt["case_id"])
    camera = Camera.load(inputs.camera)
    h3_model_manifest = image_contract.verified(h3_receipt["model_components"]["transformer"])
    models = _model_artifact_contract(runtime, h3_model_manifest)
    inputs.output_dir.mkdir(parents=True)
    raw_path = inputs.output_dir / "hybrikx_raw_recovery.npz"
    prior_path = inputs.output_dir / "fresh_h3_hybrikx_articulation_prior.npz"
    overlay_path = inputs.output_dir / "hybrikx_reprojection_evidence.png"

    (
        torch,
        edict,
        builder,
        SMPLXLayer,
        update_config,
        transform_class,
        detector_factory,
        pil_to_tensor,
    ) = _load_runtime(gpu, runtime)

    with working_directory(runtime.hybrik_root):
        detector = detector_factory(weights=None, weights_backbone=None)
        detector_state = torch.load(runtime.detector_checkpoint, map_location="cpu", weights_only=True)
        detector.load_state_dict(detector_state)
        detector = detector.eval().cuda()
        with torch.inference_mode():
            detection = detector([pil_to_tensor(image).float().cuda() / 255.0])[0]
        person_index, detector_score, detector_bbox = _select_person(detection)
        detector_boxes = detection["boxes"].detach().cpu().numpy().astype(np.float32)
        detector_scores = detection["scores"].detach().cpu().numpy().astype(np.float32)
        detector_labels = detection["labels"].detach().cpu().numpy().astype(np.int64)
        del detector, detector_state, detection
        torch.cuda.empty_cache()

        cfg = update_config(str(runtime.hybrik_config.relative_to(runtime.hybrik_root)))
        cfg.MODEL.EXTRA.USE_KID = cfg.DATASET.get("USE_KID", False)
        transform = _make_hybrik_transform(cfg, edict, transform_class)
        model = builder.build_sppe(cfg.MODEL)
        checkpoint = torch.load(runtime.hybrik_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
        model = model.eval().cuda()

        image_array = np.asarray(image)
        pose_input, transformed_bbox, image_center = transform.test_transform(
            image_array.copy(), detector_bbox.astype(float).tolist()
        )
        pose_input = pose_input[None].cuda()
        bbox_tensor = torch.as_tensor(transformed_bbox, device="cuda")[None].float()
        center_tensor = torch.as_tensor(image_center, device="cuda")[None].float()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                pose_input,
                flip_test=True,
                bboxes=bbox_tensor,
                img_center=center_tensor,
            )
        torch.cuda.synchronize()
        inference_seconds = float(time.perf_counter() - started)

        raw_theta = _tensor_array(output.pred_theta_mat, (FULL_SMPLX_JOINT_COUNT, 3, 3)).astype(np.float64)
        projected_theta = project_so3(raw_theta)
        beta11 = _tensor_array(output.pred_beta).reshape(11).astype(np.float64)
        shape_full = _tensor_array(output.pred_shape_full).reshape(21).astype(np.float64)
        expression = _tensor_array(output.pred_expression).reshape(10).astype(np.float64)
        require(np.allclose(beta11, shape_full[:11], atol=1.0e-6), "HybrIK-X beta/shape output drift")
        require(np.allclose(expression, shape_full[11:], atol=1.0e-6), "HybrIK-X expression/shape output drift")
        fold = fold_kid_beta_to_neutral(beta11, runtime.neutral_smplx, runtime.kid_template)
        decomposition = decompose_root_rotation(projected_theta[0], camera)

        # Reconstruct the exact kid-space pose and the folded neutral pose with
        # root identity.  This makes the published mesh placement-free while
        # measuring the approximation introduced by folding beta_kid.
        full_pose = projected_theta.copy()
        full_pose[0] = np.eye(3)
        full_pose_tensor = torch.as_tensor(full_pose, dtype=torch.float32, device="cuda")[None]
        expression_tensor = torch.as_tensor(expression, dtype=torch.float32, device="cuda")[None]
        with torch.inference_mode():
            kid_output = model.smplx_layer.forward_simple(
                betas=torch.as_tensor(beta11, dtype=torch.float32, device="cuda")[None],
                expression=expression_tensor,
                full_pose=full_pose_tensor,
                return_verts=True,
                root_align=True,
            )
            neutral_layer = SMPLXLayer(
                model_path=str(runtime.neutral_smplx),
                num_betas=10,
                use_pca=False,
                age="adult",
                kid_template_path=str(runtime.kid_template),
            ).eval().cuda()
            neutral_output = neutral_layer.forward_simple(
                betas=torch.as_tensor(fold.folded_beta10, dtype=torch.float32, device="cuda")[None],
                expression=expression_tensor,
                full_pose=full_pose_tensor,
                return_verts=True,
                root_align=True,
            )
        kid_vertices = kid_output.vertices[0].detach().cpu().numpy().astype(np.float64)
        neutral_vertices = neutral_output.vertices[0].detach().cpu().numpy().astype(np.float64)
        neutral_joints = neutral_output.joints[0].detach().cpu().numpy().astype(np.float64)
        neutral_joints55 = neutral_output.joints_55[0].detach().cpu().numpy().astype(np.float64)
        faces = np.asarray(neutral_layer.faces, dtype=np.int64)
        mesh_fold_error = np.linalg.norm(neutral_vertices - kid_vertices, axis=1)
        pose_frame = decomposition.pose_frame_to_world_rotation
        native_to_world_yaw0 = decomposition.native_yup_to_world_yaw0_rotation

        # Follow the exact P478/P509 materializer path when publishing the
        # prior: native SMPL-X Y-up -> materializer Z-up -> serialized frame.
        # The direct native-to-world path is computed independently and must
        # agree vertex-for-vertex and joint-for-joint.
        vertices_materializer_zup = neutral_vertices @ NATIVE_YUP_TO_MATERIALIZER_ZUP.T
        joints_materializer_zup = neutral_joints @ NATIVE_YUP_TO_MATERIALIZER_ZUP.T
        joints55_materializer_zup = neutral_joints55 @ NATIVE_YUP_TO_MATERIALIZER_ZUP.T
        vertices_world_yaw0 = vertices_materializer_zup @ pose_frame.T
        joints_world_yaw0 = joints_materializer_zup @ pose_frame.T
        joints55_world_yaw0 = joints55_materializer_zup @ pose_frame.T
        vertices_world_direct = neutral_vertices @ native_to_world_yaw0.T
        joints_world_direct = neutral_joints @ native_to_world_yaw0.T
        vertex_coordinate_chain_error = float(
            np.linalg.norm(vertices_world_yaw0 - vertices_world_direct, axis=1).max(initial=0.0)
        )
        joint_coordinate_chain_error = float(
            np.linalg.norm(joints_world_yaw0 - joints_world_direct, axis=1).max(initial=0.0)
        )
        pose_frame_factorization_error = float(
            np.abs(
                pose_frame @ NATIVE_YUP_TO_MATERIALIZER_ZUP - native_to_world_yaw0
            ).max(initial=0.0)
        )
        native_up_world = native_to_world_yaw0 @ np.asarray((0.0, 1.0, 0.0))
        materializer_up_world = pose_frame @ np.asarray((0.0, 0.0, 1.0))
        up_axis_error = float(np.linalg.norm(native_up_world - materializer_up_world))
        require(vertex_coordinate_chain_error <= 2.0e-7, "vertex Y-up/Z-up coordinate chain drift")
        require(joint_coordinate_chain_error <= 2.0e-7, "joint Y-up/Z-up coordinate chain drift")
        require(pose_frame_factorization_error <= 2.0e-7, "pose-frame factorization drift")
        require(up_axis_error <= 2.0e-7, "native/materializer up-axis mapping drift")

        raw_arrays: dict[str, Any] = {
            "schema": np.asarray(RAW_SCHEMA),
            "source_frame_sha256": np.asarray(sha256_file(inputs.frame)),
            "source_hybrikx_config_sha256": np.asarray(sha256_file(runtime.hybrik_config)),
            "source_hybrikx_model_sha256": np.asarray(sha256_file(runtime.hybrik_checkpoint)),
            "source_detector_model_sha256": np.asarray(sha256_file(runtime.detector_checkpoint)),
            "source_h3_model_manifest_sha256": np.asarray(sha256_file(h3_model_manifest)),
            "detector_all_boxes_xyxy": detector_boxes,
            "detector_all_scores": detector_scores,
            "detector_all_labels": detector_labels,
            "selected_person_index": np.asarray(person_index, dtype=np.int64),
            "selected_person_score": np.asarray(detector_score, dtype=np.float32),
            "selected_person_bbox_xyxy": np.asarray(detector_bbox, dtype=np.float32),
            "hybrik_transformed_bbox_xyxy": np.asarray(transformed_bbox, dtype=np.float32),
            "hybrik_image_center_xy": np.asarray(image_center, dtype=np.float32),
            "pred_vertices_camera_relative_m": _tensor_array(output.pred_vertices, (-1, 3)).astype(np.float32),
            "faces": np.asarray(model.smplx_layer.faces, dtype=np.int64),
            "pred_uvd_jts": _tensor_array(output.pred_uvd_jts, (-1, 3)).astype(np.float32),
            "pred_xyz_hybrik": _tensor_array(output.pred_xyz_hybrik, (-1, 3)).astype(np.float32),
            "pred_xyz_hybrik_struct": _tensor_array(output.pred_xyz_hybrik_struct, (-1, 3)).astype(np.float32),
            "pred_xyz_full": _tensor_array(output.pred_xyz_full, (-1, 3)).astype(np.float32),
            "pred_uv_full": _tensor_array(output.pred_uv_full, (-1, 2)).astype(np.float32),
            "pred_theta_quat_55": _tensor_array(output.pred_theta_quat, (55, 4)).astype(np.float32),
            "pred_theta_mat": raw_theta.astype(np.float32),
            "pred_theta_mat_55_raw": raw_theta.astype(np.float32),
            "pred_theta_mat_55_projected_so3": projected_theta.astype(np.float32),
            "pred_theta_axis_angle_55_projected_so3": rotations_to_axis_angle(projected_theta).astype(np.float32),
            "pred_shape_full": shape_full.astype(np.float32),
            "pred_shape_full_21": shape_full.astype(np.float32),
            "pred_beta_kid_11": beta11.astype(np.float32),
            "pred_expression_10": expression.astype(np.float32),
            "pred_phi_54x2": _tensor_array(output.pred_phi, (54, 2)).astype(np.float32),
            "pred_camera": _tensor_array(output.pred_camera).astype(np.float32),
            "cam_scale": _tensor_array(output.cam_scale).astype(np.float32),
            "cam_root": _tensor_array(output.cam_root, (3,)).astype(np.float32),
            "transl": _tensor_array(output.transl, (3,)).astype(np.float32),
            # The configured RLE ``Reg`` head exposes uncertainty as
            # ``pred_sigma``; the non-Reg sibling calls the same tensor
            # ``sigma``.  Preserve one canonical raw field for either official
            # head without guessing values.
            "sigma_71": _tensor_array(
                output.sigma if hasattr(output, "sigma") else output.pred_sigma,
                (-1,),
            ).astype(np.float32),
            "maxvals": _tensor_array(output.maxvals).astype(np.float32),
            "img_feat": _tensor_array(output.img_feat).astype(np.float32),
            "source_camera_world_to_camera": camera.world_to_camera.astype(np.float64),
            "source_camera_intrinsics": camera.intrinsics.astype(np.float64),
            "source_camera_size_wh": np.asarray((camera.width, camera.height), dtype=np.int64),
            "inference_seconds": np.asarray(inference_seconds, dtype=np.float64),
            "camera_translation_used_for_world_placement": np.asarray(False),
            "hybrik_translation_used_for_world_placement": np.asarray(False),
        }
        atomic_npz(raw_path, **raw_arrays)
        reprojection = _render_reprojection(
            inputs.frame,
            raw_arrays["pred_uvd_jts"],
            np.asarray(transformed_bbox),
            detector_bbox,
            detector_score,
            overlay_path,
        )

        generated_at = datetime.now(timezone.utc).isoformat()
        body_axis_angle = rotations_to_axis_angle(projected_theta[1:22]).astype(np.float32)
        require(body_axis_angle.shape == (BODY_JOINT_COUNT, 3), "body pose extraction drift")
        atomic_npz(
            prior_path,
            schema=np.asarray(PRIOR_SCHEMA),
            status=np.asarray("published_fresh_h3_hybrikx_articulation_only"),
            generated_at_utc=np.asarray(generated_at),
            run_nonce=np.asarray(identity["run_nonce"]),
            case_id=np.asarray(identity["case_id"]),
            scene_id=np.asarray(identity["scene_id"]),
            instruction=np.asarray(identity["instruction"]),
            action_family=np.asarray(identity["action_family"]),
            target_instance_id=np.asarray(identity["target_instance_id"]),
            target_surface_id=np.asarray(identity["target_surface_id"]),
            selected_candidate_id=np.asarray(identity["target_surface_id"]),
            source_candidate_id=np.asarray(identity["target_surface_id"]),
            body_pose_axis_angle=body_axis_angle,
            body_pose_rotation_matrices=projected_theta[1:22].astype(np.float32),
            betas=fold.folded_beta10.astype(np.float32),
            hybrikx_beta_kid_11=beta11.astype(np.float32),
            hybrikx_expression_10=expression.astype(np.float32),
            root_xyz_yaw=np.zeros(4, dtype=np.float32),
            pose_frame_to_world_rotation=pose_frame.astype(np.float32),
            native_yup_to_materializer_zup_rotation=NATIVE_YUP_TO_MATERIALIZER_ZUP.astype(np.float32),
            native_yup_to_world_yaw0_rotation=native_to_world_yaw0.astype(np.float32),
            recovered_root_rotation_camera=decomposition.root_rotation_camera.astype(np.float32),
            recovered_root_rotation_world=decomposition.root_rotation_world.astype(np.float32),
            recovered_root_yaw_world_rad=np.asarray(decomposition.recovered_yaw_world_rad, dtype=np.float32),
            pose_frame_yaw_residual_rad=np.asarray(decomposition.yaw_residual_rad, dtype=np.float32),
            vertices_native_yup=np.asarray(neutral_vertices, dtype=np.float32),
            joints_native_yup=np.asarray(neutral_joints, dtype=np.float32),
            joints55_native_yup=np.asarray(neutral_joints55, dtype=np.float32),
            vertices_materializer_zup=np.asarray(vertices_materializer_zup, dtype=np.float32),
            joints_materializer_zup=np.asarray(joints_materializer_zup, dtype=np.float32),
            joints55_materializer_zup=np.asarray(joints55_materializer_zup, dtype=np.float32),
            # ``pose_frame`` acts on these materializer-Z-up arrays.  Keep the
            # historical generic names, but make their frame explicit through
            # the fields and assertions above.
            vertices_pose_frame=np.asarray(vertices_materializer_zup, dtype=np.float32),
            joints_pose_frame=np.asarray(joints_materializer_zup, dtype=np.float32),
            joints55_pose_frame=np.asarray(joints55_materializer_zup, dtype=np.float32),
            vertices_world_zup=np.asarray(vertices_world_yaw0, dtype=np.float32),
            vertices_world_zup_m=np.asarray(vertices_world_yaw0, dtype=np.float32),
            joints_world_zup=np.asarray(joints_world_yaw0, dtype=np.float32),
            joints_world_zup_m=np.asarray(joints_world_yaw0, dtype=np.float32),
            joints55_world_zup_m=np.asarray(joints55_world_yaw0, dtype=np.float32),
            vertex_coordinate_chain_max_error_m=np.asarray(vertex_coordinate_chain_error, dtype=np.float64),
            joint_coordinate_chain_max_error_m=np.asarray(joint_coordinate_chain_error, dtype=np.float64),
            pose_frame_factorization_max_error=np.asarray(pose_frame_factorization_error, dtype=np.float64),
            native_materializer_up_axis_error=np.asarray(up_axis_error, dtype=np.float64),
            native_up_world_yaw0=np.asarray(native_up_world, dtype=np.float32),
            materializer_up_world_yaw0=np.asarray(materializer_up_world, dtype=np.float32),
            coordinate_chain=np.asarray(
                "smplx_native_yup__C_to_materializer_zup__pose_frame_to_world_zup"
            ),
            faces=faces,
            source_h3_frame_sha256=np.asarray(sha256_file(inputs.frame)),
            source_raw_recovery_sha256=np.asarray(sha256_file(raw_path)),
            source_raw_hybrikx_sha256=np.asarray(sha256_file(raw_path)),
            source_hybrikx_model_sha256=np.asarray(sha256_file(runtime.hybrik_checkpoint)),
            source_h3_model_manifest_sha256=np.asarray(sha256_file(h3_model_manifest)),
            source_stage1_sha256=np.asarray(sha256_file(inputs.stage1)),
            source_config_sha256=np.asarray(sha256_file(inputs.config)),
            hybrikx_used=np.asarray(True),
            h3_frame_used=np.asarray(True),
            fresh_h3_used=np.asarray(True),
            fresh_hybrikx_used=np.asarray(True),
            romp_used=np.asarray(False),
            qwen_image_used=np.asarray(False),
            fresh_rompv2_used=np.asarray(False),
            fresh_qwen_image_used=np.asarray(False),
            articulation_only=np.asarray(True),
            body_rotations_projected_to_so3=np.asarray(True),
            kid_beta_folded_by_neutral_shape_least_squares=np.asarray(True),
            camera_rotation_used_for_gravity_aligned_pose_frame=np.asarray(True),
            camera_world_to_camera_used_exactly=np.asarray(True),
            camera_translation_used=np.asarray(False),
            camera_world_placement_discarded=np.asarray(True),
            camera_projection_evidence_retained=np.asarray(True),
            native_yup_to_materializer_zup_applied=np.asarray(True),
            pose_frame_consumes_materializer_zup=np.asarray(True),
            hybrikx_camera_translation_used=np.asarray(False),
            hybrikx_world_placement_reused=np.asarray(False),
            stage1_world_placement_reused=np.asarray(False),
            requires_stage1_contact_surface_placement=np.asarray(True),
            neutral_smplx_carrier_used=np.asarray(True),
            kid_shape_folded_to_ten_betas=np.asarray(True),
            old_world_placement_consumed=np.asarray(False),
            legacy_world_placement_reused=np.asarray(False),
            gt_root_path_pose_motion_contact_icgf_used=np.asarray(False),
            lingo_pose_or_motion_donor_used=np.asarray(False),
            retrieval_or_positive_memory_used=np.asarray(False),
            handwritten_task_pose_template_used=np.asarray(False),
            motion_repair_used=np.asarray(False),
            dataset_pose_or_motion_used=np.asarray(False),
            retrieval_or_pose_bank_used=np.asarray(False),
        )

        del output, model, neutral_layer, kid_output, neutral_output
        torch.cuda.empty_cache()

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete_fresh_h3_hybrikx_articulation_published",
        "created_at_utc": generated_at,
        **identity,
        "inputs": {
            "h3_image": artifact(inputs.frame),
            "h3_frame": artifact(inputs.frame),
            "stage1_camera": artifact(inputs.camera),
            "camera": artifact(inputs.camera),
            "stage1_artifact": artifact(inputs.stage1),
            "job_config": artifact(inputs.config),
            "hybrikx_raw_recovery": artifact(raw_path),
            **({"h3_generation_receipt": artifact(inputs.h3_receipt)} if inputs.h3_receipt else {}),
        },
        "models": models,
        "outputs": {
            "articulation_prior": artifact(prior_path),
            "hybrikx_raw_recovery": artifact(raw_path),
            "raw_hybrikx_recovery": artifact(raw_path),
            "fresh_h3_hybrikx_articulation_prior": artifact(prior_path),
            "reprojection_evidence": artifact(overlay_path),
        },
        "recovery": {
            "detector": "Keypoint R-CNN R50 FPN COCO with hash-bound local checkpoint",
            "selected_person_index": int(person_index),
            "selected_person_score": float(detector_score),
            "hybrikx_inference_seconds": inference_seconds,
            "theta_matrix_count": int(len(projected_theta)),
            "body_joint_rotation_slice": "pred_theta_mat[1:22]",
            "all_body_rotations_projected_to_so3": True,
            **reprojection,
        },
        "shape_fold": {
            "method": "unregularized_least_squares_in_neutral_smplx_vertex_shape_space",
            "equation": "argmin_b ||S10*b-(S10*beta10+kid_direction*beta_kid)||_2",
            "design_rank": fold.design_rank,
            "design_condition": fold.design_condition,
            "template_displacement_rms_m": fold.residual_rms_m,
            "template_displacement_max_vertex_m": fold.residual_max_m,
            "posed_mesh_rms_vertex_m": float(np.sqrt(np.mean(np.square(mesh_fold_error)))),
            "posed_mesh_max_vertex_m": float(mesh_fold_error.max(initial=0.0)),
        },
        "root_frame": {
            "camera_extrinsic_convention": "opencv_world_to_camera",
            "root_rotation_world_equation": "R_world_root=R_world_to_camera.T@R_camera_root",
            "yaw_removal_equation": "R_native_yup_to_world_yaw0=Rz(-yaw_world)@R_world_root",
            "materializer_conversion": "row_materializer_zup=row_native_yup@C.T; C maps [x,y,z] to [x,-z,y]",
            "pose_frame_equation": "R_pose_frame=R_native_yup_to_world_yaw0@C.T",
            "published_vertex_equation": "V_world=(V_native@C.T)@R_pose_frame.T=V_native@R_native_yup_to_world_yaw0.T",
            "heading_axis": decomposition.heading_axis,
            "recovered_yaw_world_rad": decomposition.recovered_yaw_world_rad,
            "pose_frame_yaw_residual_rad": decomposition.yaw_residual_rad,
            "pitch_roll_retained": True,
            "native_yup_to_materializer_zup_rotation": NATIVE_YUP_TO_MATERIALIZER_ZUP.tolist(),
            "native_yup_to_world_yaw0_rotation": native_to_world_yaw0.tolist(),
            "pose_frame_to_world_rotation": pose_frame.tolist(),
            "vertex_coordinate_chain_max_error_m": vertex_coordinate_chain_error,
            "joint_coordinate_chain_max_error_m": joint_coordinate_chain_error,
            "pose_frame_factorization_max_error": pose_frame_factorization_error,
            "native_materializer_up_axis_error": up_axis_error,
        },
        "placement_firewall": {
            "articulation_only": True,
            "camera_rotation_used": True,
            "camera_translation_used": False,
            "hybrikx_translation_used_for_world_placement": False,
            "image_rays_used_for_world_placement": False,
            "agent10_humanise_object_id_placement_used": False,
            "agent10_scannet_interaction_point_used": False,
            "stage1_target_identity_retained_as_lineage_only": True,
            "stage1_contact_surface_must_supply_future_world_placement": True,
            "projection_evidence_retained_for_future_hybrid_postprocess": True,
        },
        "lineage": {
            "fresh_h3_used": True,
            "fresh_hybrikx_used": True,
            "fresh_qwen_image_used": False,
            "fresh_rompv2_used": False,
            "articulation_only": True,
            "camera_world_placement_discarded": True,
            "camera_projection_evidence_retained": True,
            "neutral_smplx_carrier_used": True,
            "kid_shape_folded_to_ten_betas": True,
            "old_world_placement_consumed": False,
        },
        "schema_integrity": {
            "prior_schema": PRIOR_SCHEMA,
            "rompv2_schema_or_alias_emitted": False,
            "qwen_schema_or_alias_emitted": False,
            "source_model": "MiniMax H3 native single image followed by HybrIK-X",
        },
        "stage2_articulation_handoff_allowed": True,
    }
    require(_validate_bound_image(inputs) == h3_receipt, "H3 input bindings changed during recovery")
    require(_model_artifact_contract(runtime, h3_model_manifest) == models, "Recovery model bytes changed")
    receipt.update(h3_native_output_kind="image", h3_generation_schema=image_contract.SCHEMA,
        h3_model_components=h3_receipt["model_components"], video_generation_performed=False,
        video_frame_extraction_performed=False,
        execution_source={"recovery": artifact(HERE), "input_contract": artifact(Path(image_contract.__file__))},
        publication={"articulation_only": True, "keypose_publishable": False,
            "world_placement_refine_and_full_body_checks_still_required": True})
    image_contract._no_temporal_media_fields(receipt)
    receipt["receipt_payload_sha256"] = canonical_hash(receipt)
    receipt_path = inputs.output_dir / "receipt.json"
    atomic_json(receipt_path, receipt)
    return receipt

def validate_bound_inputs(*, image: Path, h3_receipt: Path, camera: Path,
                          stage1: Path, config: Path, case_id: str) -> dict[str, Any]:
    value = image_contract.validate_receipt(h3_receipt, expected_case_id=case_id, expected_image=image)
    image_contract.require(value["inputs"]["camera"] == image_contract.artifact(camera),
                     "HybrIK camera differs from the actual single-image generation camera")
    binding = value["source_binding"]
    image_contract.require(binding["stage1_bundle"] == image_contract.artifact(stage1),
                     "HybrIK Stage1 bundle differs from the image generation bundle")
    job = json.loads(config.read_text(encoding="utf-8"))
    image_contract.require(isinstance(job, dict), "HybrIK config must be a JSON object")
    for key in ("scene_id", "instruction"):
        image_contract.require(job.get(key) == binding[key], "HybrIK config/image " + key + " drift")
    image_contract.require(job.get("case_id", case_id) == case_id, "HybrIK config/image case ID drift")
    target = job.get("stage1_binding", {})
    image_contract.require(isinstance(target, dict), "HybrIK config stage1_binding must be an object")
    for key in ("target_instance_id", "target_surface_id"):
        image_contract.require(target.get(key, job.get(key)) == binding[key], "HybrIK config/image " + key + " drift")
    for key in ("scene_id", "target_class"):
        if key in target:
            image_contract.require(target[key] == binding[key], "HybrIK target binding " + key + " drift")
    sources = job.get("sources", {})
    image_contract.require(isinstance(sources, dict), "HybrIK config sources must be an object")
    expected_sources = {"camera": value["inputs"]["camera"], "stage1_bundle": binding["stage1_bundle"],
                        "stage1_target": binding["stage1_target"],
                        "stage1_crop": value["inputs"]["condition_source"],
                        "crop_derivation": value["inputs"]["crop_derivation"]}
    for key, expected in expected_sources.items():
        if key in sources:
            image_contract.require(sources[key] == expected, "HybrIK config source " + key + " drift")
    return value


def _validate_bound_image(inputs):
    return validate_bound_inputs(image=inputs.frame,h3_receipt=inputs.h3_receipt,
        camera=inputs.camera,stage1=inputs.stage1,config=inputs.config,case_id=inputs.case_id)


def recover(image, h3_receipt, camera, stage1_bundle, config, output, *, runtime, gpu=0, case_id):
    """Execute the actual detector/model; explicit missing assets fail closed."""
    require("torch" not in sys.modules, "Launch recovery in its own process before importing torch")
    require(type(gpu) is int and gpu >= 0, "Expected physical nonnegative GPU ID")
    if not isinstance(runtime, RuntimeConfig):
        runtime = RuntimeConfig.from_mapping(runtime)
    inputs = Inputs(frame=Path(image).resolve(strict=True),camera=Path(camera).resolve(strict=True),
        stage1=Path(stage1_bundle).resolve(strict=True),config=Path(config).resolve(strict=True),
        output_dir=Path(output).resolve(),h3_receipt=Path(h3_receipt).resolve(strict=True),case_id=case_id)
    return run(inputs,gpu=str(gpu),runtime=runtime)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("image","h3-receipt","camera","stage1-bundle","config","output","runtime"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--case-id",required=True)
    args=parser.parse_args()
    result=recover(args.image,args.h3_receipt,args.camera,args.stage1_bundle,args.config,args.output,
        runtime=read_json(args.runtime),gpu=args.gpu,case_id=args.case_id)
    print(json.dumps({"status":result["status"],"receipt":str(args.output/"receipt.json")},ensure_ascii=False))


if __name__ == "__main__":
    main()
