"""Static camera, crop and coarse geometry contracts; no Qwen view inference."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve()

CONFIG_SCHEMA = "p536.qwen3vl_stage2_view_crop_audit_config.v1"
OUTPUT_SCHEMA = "p536.qwen3vl_stage2_view_crop_audit.v1"
QWEN_RESPONSE_SCHEMA = "p536.qwen3vl_stage2_view_scores.v1"

SCORE_KEYS = (
    "target_visibility",
    "contact_surface_clarity",
    "interaction_space",
    "approach_consistency",
    "distractor_separation",
)
BOOLEAN_KEYS = (
    "target_identity_match",
    "target_fully_visible",
    "contact_region_visible",
    "interaction_space_visible",
    "low_distractor_ambiguity",
)
SCORE_WEIGHTS = {
    "target_visibility": 3.0,
    "contact_surface_clarity": 3.0,
    "interaction_space": 2.0,
    "approach_consistency": 1.0,
    "distractor_separation": 2.0,
}

DEFAULT_GEOMETRY = {
    "near_plane_m": 0.05,
    "minimum_bbox_width_px": 70.0,
    "minimum_bbox_height_px": 70.0,
    "minimum_bbox_area_fraction": 0.012,
    "maximum_bbox_area_fraction": 0.70,
    "minimum_bbox_retained_fraction": 0.95,
    "minimum_approach_side_cosine": 0.0,
    "minimum_camera_target_distance_m": 0.40,
    "maximum_camera_target_distance_m": 10.0,
}
DEFAULT_CROP = {
    "output_size_wh": [512, 512],
    "horizontal_context_bbox_ratio": 0.55,
    "top_context_bbox_ratio": 0.90,
    "bottom_context_bbox_ratio": 1.25,
    "minimum_crop_width_fraction": 0.42,
    "minimum_crop_height_fraction": 0.62,
    "minimum_target_padding_px": 6.0,
    "minimum_resized_bbox_width_px": 80.0,
    "minimum_resized_bbox_height_px": 80.0,
}
DEFAULT_QWEN = {
    "group_size": 8,
    "maximum_candidates": 48,
    "minimum_confidence": 0.70,
    "minimum_winner_margin": 1.0,
    "minimum_scores": {
        "target_visibility": 4,
        "contact_surface_clarity": 3,
        "interaction_space": 3,
        "approach_consistency": 3,
        "distractor_separation": 3,
    },
    "max_pixels": 196_608,
    "max_new_tokens": 2048,
    "panel_side_px": 384,
}


class ViewAuditError(RuntimeError):
    """Raised when an input, model response, or publication invariant fails."""


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ViewAuditError(message)


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    require(path.is_file(), f"artifact is not a file: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ViewAuditError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def resolve_path(raw: str | Path, *, base: Path | None = None) -> Path:
    path = Path(raw).expanduser()
    require(path.is_absolute() or base is not None, "Relative artifact requires an explicit base directory")
    return (path if path.is_absolute() else Path(base) / path).resolve(strict=True)


def checked_artifact(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    """Resolve and verify an explicit path/bytes/SHA256 artifact record."""

    require(isinstance(value, Mapping), f"{label} must be a hash-bound artifact record")
    require(set(("path", "bytes", "sha256")) <= set(value), f"{label} lacks path/bytes/sha256")
    path = resolve_path(str(value["path"]))
    observed = artifact(path)
    require(type(value["bytes"]) is int, f"{label}.bytes must be an integer")
    require(int(value["bytes"]) == observed["bytes"], f"{label} byte count drift")
    require(
        isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
        f"{label}.sha256 is invalid",
    )
    require(value["sha256"] == observed["sha256"], f"{label} SHA256 drift")
    return path, observed


def validate_receipt_payload(value: Mapping[str, Any], label: str) -> None:
    declared = value.get("receipt_payload_sha256")
    require(
        isinstance(declared, str) and re.fullmatch(r"[0-9a-f]{64}", declared) is not None,
        f"{label} lacks a valid receipt_payload_sha256",
    )
    payload = dict(value)
    payload.pop("receipt_payload_sha256", None)
    require(canonical_hash(payload) == declared, f"{label} payload hash drift")


def merge_settings(defaults: Mapping[str, Any], supplied: Any, label: str) -> dict[str, Any]:
    if supplied is None:
        return dict(defaults)
    require(isinstance(supplied, Mapping), f"{label} must be an object")
    unknown = set(supplied) - set(defaults)
    require(not unknown, f"unknown {label} keys: {sorted(unknown)}")
    result = dict(defaults)
    result.update(supplied)
    return result


@dataclass(frozen=True)
class Camera:
    view_index: int
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    position_world_xy: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    view_index: int
    rgb_path: Path
    rgb_artifact: dict[str, Any]
    camera_path: Path
    camera_artifact: dict[str, Any]
    camera: Camera
    supplied_overlay_path: Path | None
    supplied_overlay_artifact: dict[str, Any] | None


@dataclass(frozen=True)
class Campaign:
    config_path: Path
    config_artifact: dict[str, Any]
    raw: dict[str, Any]
    scene_id: str
    instruction: str
    target_path: Path
    target_artifact: dict[str, Any]
    target: dict[str, Any]
    target_instance_id: str
    target_surface_id: str
    target_class: str
    target_bounds: np.ndarray
    approach_world_xy: np.ndarray
    approach_source: dict[str, Any] | None
    candidates: tuple[Candidate, ...]
    geometry: dict[str, Any]
    crop: dict[str, Any]
    qwen: dict[str, Any]


def _numeric_pair(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (2,) and np.isfinite(array).all(), f"{label} must be two finite numbers")
    return array


def load_camera(path: Path, expected_view_index: int) -> Camera:
    value = read_json(path)
    camera_id = int(value.get("camera_id", -1))
    require(camera_id == expected_view_index, "camera_id does not match candidate view_index")
    world_to_camera = np.asarray(value.get("world_to_camera"), dtype=np.float64)
    require(
        world_to_camera.shape == (4, 4) and np.isfinite(world_to_camera).all(),
        f"view {camera_id}: world_to_camera must be finite [4,4]",
    )
    require(
        np.allclose(world_to_camera[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8),
        f"view {camera_id}: invalid homogeneous camera row",
    )
    rotation = world_to_camera[:3, :3]
    require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5),
        f"view {camera_id}: camera rotation is not proper",
    )
    intrinsics = np.asarray(value.get("intrinsics", value.get("K")), dtype=np.float64)
    require(
        intrinsics.shape == (3, 3)
        and np.isfinite(intrinsics).all()
        and intrinsics[0, 0] > 0.0
        and intrinsics[1, 1] > 0.0,
        f"view {camera_id}: invalid intrinsics",
    )
    width, height = value.get("width"), value.get("height")
    require(type(width) is int and type(height) is int and width > 0 and height > 0,
            f"view {camera_id}: invalid camera dimensions")
    inferred_position = -(rotation.T @ world_to_camera[:3, 3])
    declared_position = value.get("position_world_zup_m")
    if declared_position is not None:
        declared = np.asarray(declared_position, dtype=np.float64)
        require(declared.shape == (3,) and np.isfinite(declared).all(),
                f"view {camera_id}: invalid declared camera position")
        require(np.allclose(declared, inferred_position, atol=1.0e-5),
                f"view {camera_id}: declared camera position disagrees with extrinsic")
    return Camera(
        view_index=camera_id,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        position_world_xy=inferred_position[:2],
        width=int(width),
        height=int(height),
    )


def load_campaign(config_path: Path) -> Campaign:
    config_path = config_path.resolve(strict=True)
    raw = read_json(config_path)
    require(raw.get("schema") == CONFIG_SCHEMA, f"expected config schema {CONFIG_SCHEMA}")
    scene_id = str(raw.get("scene_id", "")).strip()
    instruction = str(raw.get("instruction", "")).strip()
    require(scene_id and instruction, "scene_id and instruction are required")

    target_path, target_record = checked_artifact(raw.get("stage1_target"), "stage1_target")
    target = read_json(target_path)
    validate_receipt_payload(target, "Stage1 target")
    require(str(target.get("status", "")).startswith("published"), "Stage1 target is not published")
    require(str(target.get("scene_id", "")) == scene_id, "Stage1 target scene_id drift")
    require(str(target.get("instruction", "")) == instruction, "Stage1 target instruction drift")
    target_instance_id = str(
        target.get("selected_target_instance_id", target.get("target_instance_id", ""))
    )
    target_surface_id = str(target.get("selected_surface_id", target.get("target_surface_id", "")))
    target_class = str(target.get("target_class", target.get("requested_target_class", "unknown")))
    require(target_instance_id and target_surface_id, "Stage1 target identity/surface is missing")
    if raw.get("target_instance_id") is not None:
        require(str(raw["target_instance_id"]) == target_instance_id, "configured target instance drift")
    if raw.get("target_surface_id") is not None:
        require(str(raw["target_surface_id"]) == target_surface_id, "configured target surface drift")
    bounds = np.asarray(target.get("target_bounds_world_zup_m"), dtype=np.float64)
    require(
        bounds.shape == (2, 3)
        and np.isfinite(bounds).all()
        and np.all(bounds[1] > bounds[0]),
        "Stage1 target_bounds_world_zup_m is malformed",
    )

    approach = _numeric_pair(raw.get("approach_world_xy_m"), "approach_world_xy_m")
    approach_source: dict[str, Any] | None = None
    if raw.get("stage1_approach_source") is not None:
        _approach_path, approach_source = checked_artifact(
            raw["stage1_approach_source"], "stage1_approach_source"
        )

    geometry = merge_settings(DEFAULT_GEOMETRY, raw.get("geometry_thresholds"), "geometry_thresholds")
    crop = merge_settings(DEFAULT_CROP, raw.get("crop_policy"), "crop_policy")
    qwen = merge_settings(DEFAULT_QWEN, raw.get("qwen_policy"), "qwen_policy")
    if raw.get("qwen_policy") and "minimum_scores" in raw["qwen_policy"]:
        minimum_scores = raw["qwen_policy"]["minimum_scores"]
        require(isinstance(minimum_scores, Mapping), "qwen_policy.minimum_scores must be an object")
        require(set(minimum_scores) == set(SCORE_KEYS), "qwen minimum score keys mismatch")
        qwen["minimum_scores"] = dict(minimum_scores)

    for key in (
        "near_plane_m", "minimum_bbox_width_px", "minimum_bbox_height_px",
        "minimum_bbox_area_fraction", "maximum_bbox_area_fraction",
        "minimum_bbox_retained_fraction", "minimum_approach_side_cosine",
        "minimum_camera_target_distance_m", "maximum_camera_target_distance_m",
    ):
        require(type(geometry[key]) in (int, float) and math.isfinite(float(geometry[key])),
                f"geometry threshold {key} must be finite")
    require(0.0 < float(geometry["minimum_bbox_retained_fraction"]) <= 1.0,
            "minimum_bbox_retained_fraction outside (0,1]")
    require(-1.0 <= float(geometry["minimum_approach_side_cosine"]) <= 1.0,
            "minimum_approach_side_cosine outside [-1,1]")
    require(float(geometry["minimum_bbox_area_fraction"]) < float(geometry["maximum_bbox_area_fraction"]),
            "bbox area thresholds are inverted")

    output_wh = crop["output_size_wh"]
    require(
        isinstance(output_wh, list)
        and len(output_wh) == 2
        and all(type(item) is int and item >= 256 and item % 16 == 0 for item in output_wh),
        "crop output_size_wh must contain two multiples of 16 >=256",
    )
    for key in (
        "horizontal_context_bbox_ratio", "top_context_bbox_ratio",
        "bottom_context_bbox_ratio", "minimum_crop_width_fraction",
        "minimum_crop_height_fraction", "minimum_target_padding_px",
        "minimum_resized_bbox_width_px", "minimum_resized_bbox_height_px",
    ):
        require(type(crop[key]) in (int, float) and math.isfinite(float(crop[key]))
                and float(crop[key]) >= 0.0, f"crop setting {key} is invalid")

    require(type(qwen["group_size"]) is int and 2 <= qwen["group_size"] <= 8,
            "qwen group_size must be in [2,8]")
    require(type(qwen["maximum_candidates"]) is int and 2 <= qwen["maximum_candidates"] <= 48,
            "qwen maximum_candidates must be in [2,48]")
    require(type(qwen["max_new_tokens"]) is int and qwen["max_new_tokens"] >= 512,
            "qwen max_new_tokens is too small")
    require(type(qwen["max_pixels"]) is int and qwen["max_pixels"] >= 65_536,
            "qwen max_pixels is too small")
    require(type(qwen["panel_side_px"]) is int and 256 <= qwen["panel_side_px"] <= 640,
            "qwen panel_side_px outside [256,640]")
    require(0.0 <= float(qwen["minimum_confidence"]) <= 1.0,
            "qwen minimum_confidence outside [0,1]")
    require(float(qwen["minimum_winner_margin"]) >= 0.0, "negative winner margin")
    for key in SCORE_KEYS:
        value = qwen["minimum_scores"][key]
        require(type(value) is int and 0 <= value <= 5, f"invalid minimum score {key}")

    rows = raw.get("candidate_views", raw.get("candidates"))
    require(isinstance(rows, list) and 2 <= len(rows) <= qwen["maximum_candidates"],
            "candidate_views count is outside [2, maximum_candidates]")
    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    seen_views: set[int] = set()
    seen_rgb_hashes: set[str] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"candidate {index} is not an object")
        view_index = row.get("view_index")
        require(type(view_index) is int and view_index >= 0, f"candidate {index} has invalid view_index")
        candidate_id = str(row.get("candidate_id", f"view_{view_index:02d}"))
        require(re.fullmatch(r"[A-Za-z0-9_-]+", candidate_id) is not None,
                f"unsafe candidate_id: {candidate_id}")
        require(candidate_id not in seen_ids and view_index not in seen_views,
                f"duplicate candidate id or view: {candidate_id}/{view_index}")
        rgb_path, rgb_record = checked_artifact(row.get("rgb"), f"{candidate_id}.rgb")
        camera_path, camera_record = checked_artifact(row.get("camera"), f"{candidate_id}.camera")
        require(rgb_record["sha256"] not in seen_rgb_hashes,
                f"duplicate RGB bytes across candidates: {candidate_id}")
        seen_ids.add(candidate_id)
        seen_views.add(view_index)
        seen_rgb_hashes.add(rgb_record["sha256"])
        camera = load_camera(camera_path, view_index)
        with Image.open(rgb_path) as handle:
            require(handle.size == (camera.width, camera.height),
                    f"{candidate_id}: RGB size disagrees with camera")
        overlay_path: Path | None = None
        overlay_record: dict[str, Any] | None = None
        if row.get("target_overlay") is not None:
            overlay_path, overlay_record = checked_artifact(
                row["target_overlay"], f"{candidate_id}.target_overlay"
            )
            with Image.open(overlay_path) as handle:
                require(handle.size == (camera.width, camera.height),
                        f"{candidate_id}: supplied overlay size drift")
        candidates.append(Candidate(
            candidate_id=candidate_id,
            view_index=view_index,
            rgb_path=rgb_path,
            rgb_artifact=rgb_record,
            camera_path=camera_path,
            camera_artifact=camera_record,
            camera=camera,
            supplied_overlay_path=overlay_path,
            supplied_overlay_artifact=overlay_record,
        ))
    candidates.sort(key=lambda item: (item.view_index, item.candidate_id))
    return Campaign(
        config_path=config_path,
        config_artifact=artifact(config_path),
        raw=raw,
        scene_id=scene_id,
        instruction=instruction,
        target_path=target_path,
        target_artifact=target_record,
        target=target,
        target_instance_id=target_instance_id,
        target_surface_id=target_surface_id,
        target_class=target_class,
        target_bounds=bounds,
        approach_world_xy=approach,
        approach_source=approach_source,
        candidates=tuple(candidates),
        geometry=geometry,
        crop=crop,
        qwen=qwen,
    )


def target_corners(bounds: np.ndarray) -> np.ndarray:
    low, high = np.asarray(bounds, dtype=np.float64)
    return np.asarray(
        [[x, y, z] for x in (low[0], high[0])
         for y in (low[1], high[1]) for z in (low[2], high[2])],
        dtype=np.float64,
    )


def project_target(bounds: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    corners = target_corners(bounds)
    homogeneous = np.concatenate((corners, np.ones((len(corners), 1))), axis=1)
    camera_points = homogeneous @ camera.world_to_camera.T
    depths = camera_points[:, 2]
    normalized = camera_points[:, :2] / depths[:, None]
    pixels_h = np.concatenate((normalized, np.ones((len(normalized), 1))), axis=1)
    pixels = (pixels_h @ camera.intrinsics.T)[:, :2]
    bbox = np.stack((pixels.min(axis=0), pixels.max(axis=0)), axis=0)
    return bbox, depths


def _fit_interval(low: float, high: float, span: float, limit: int) -> tuple[float, float]:
    require(span <= limit + 1.0e-6, "requested crop span exceeds image")
    centre = 0.5 * (low + high)
    left = centre - 0.5 * span
    right = centre + 0.5 * span
    if left < 0.0:
        right -= left
        left = 0.0
    if right > float(limit):
        left -= right - float(limit)
        right = float(limit)
    return max(0.0, left), min(float(limit), right)


def derive_crop(
    bbox: np.ndarray,
    image_wh: tuple[int, int],
    policy: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    """Derive one deterministic, aspect-correct crop around target and action space."""

    width, height = image_wh
    x1, y1 = np.asarray(bbox[0], dtype=np.float64)
    x2, y2 = np.asarray(bbox[1], dtype=np.float64)
    bbox_w, bbox_h = x2 - x1, y2 - y1
    require(bbox_w > 0.0 and bbox_h > 0.0, "cannot crop a degenerate bbox")
    wanted_left = x1 - float(policy["horizontal_context_bbox_ratio"]) * bbox_w
    wanted_right = x2 + float(policy["horizontal_context_bbox_ratio"]) * bbox_w
    wanted_top = y1 - float(policy["top_context_bbox_ratio"]) * bbox_h
    wanted_bottom = y2 + float(policy["bottom_context_bbox_ratio"]) * bbox_h
    wanted_w = max(wanted_right - wanted_left, float(policy["minimum_crop_width_fraction"]) * width)
    wanted_h = max(wanted_bottom - wanted_top, float(policy["minimum_crop_height_fraction"]) * height)
    output_w, output_h = (int(value) for value in policy["output_size_wh"])
    aspect = float(output_w) / float(output_h)
    if wanted_w / wanted_h < aspect:
        wanted_w = wanted_h * aspect
    else:
        wanted_h = wanted_w / aspect
    scale = min(1.0, float(width) / wanted_w, float(height) / wanted_h)
    wanted_w *= scale
    wanted_h *= scale
    centre_x = 0.5 * (wanted_left + wanted_right)
    centre_y = 0.5 * (wanted_top + wanted_bottom)
    crop_x1, crop_x2 = _fit_interval(centre_x - wanted_w / 2, centre_x + wanted_w / 2, wanted_w, width)
    crop_y1, crop_y2 = _fit_interval(centre_y - wanted_h / 2, centre_y + wanted_h / 2, wanted_h, height)
    integer = (
        max(0, int(math.floor(crop_x1))),
        max(0, int(math.floor(crop_y1))),
        min(width, int(math.ceil(crop_x2))),
        min(height, int(math.ceil(crop_y2))),
    )
    # Rounding can perturb the requested aspect slightly; downstream resize is
    # explicit and the exact integer crop is recorded.
    require(integer[0] < integer[2] and integer[1] < integer[3], "derived crop is empty")
    return integer


def geometry_audit(campaign: Campaign, candidate: Candidate) -> dict[str, Any]:
    camera = candidate.camera
    bbox, depths = project_target(campaign.target_bounds, camera)
    width, height = camera.width, camera.height
    clip_low = np.maximum(bbox[0], (0.0, 0.0))
    clip_high = np.minimum(bbox[1], (float(width), float(height)))
    full_wh = np.maximum(bbox[1] - bbox[0], 0.0)
    clipped_wh = np.maximum(clip_high - clip_low, 0.0)
    full_area = float(np.prod(full_wh))
    clipped_area = float(np.prod(clipped_wh))
    retained = clipped_area / full_area if full_area > 0.0 else 0.0
    area_fraction = clipped_area / float(width * height)
    centre = 0.5 * (bbox[0] + bbox[1])
    target_xy = campaign.target_bounds.mean(axis=0)[:2]
    approach_vector = campaign.approach_world_xy - target_xy
    camera_vector = camera.position_world_xy - target_xy
    approach_norm = float(np.linalg.norm(approach_vector))
    camera_norm = float(np.linalg.norm(camera_vector))
    approach_cosine = (
        float(np.dot(approach_vector, camera_vector) / (approach_norm * camera_norm))
        if approach_norm > 1.0e-8 and camera_norm > 1.0e-8 else float("nan")
    )
    crop: tuple[int, int, int, int] | None = None
    crop_error: str | None = None
    try:
        crop = derive_crop(bbox, (width, height), campaign.crop)
    except Exception as error:
        crop_error = f"{type(error).__name__}: {error}"

    gates = {
        "all_target_aabb_corners_in_front": bool(
            np.isfinite(depths).all() and np.all(depths >= float(campaign.geometry["near_plane_m"]))
        ),
        "target_bbox_intersects_image": bool(clipped_area > 0.0),
        "target_bbox_not_materially_clipped": bool(
            retained >= float(campaign.geometry["minimum_bbox_retained_fraction"])
        ),
        "target_bbox_width_sufficient": bool(
            clipped_wh[0] >= float(campaign.geometry["minimum_bbox_width_px"])
        ),
        "target_bbox_height_sufficient": bool(
            clipped_wh[1] >= float(campaign.geometry["minimum_bbox_height_px"])
        ),
        "target_bbox_area_sufficient": bool(
            area_fraction >= float(campaign.geometry["minimum_bbox_area_fraction"])
        ),
        "target_bbox_area_not_excessive": bool(
            area_fraction <= float(campaign.geometry["maximum_bbox_area_fraction"])
        ),
        "target_bbox_center_inside_image": bool(
            0.0 <= centre[0] < width and 0.0 <= centre[1] < height
        ),
        "camera_on_stage1_approach_side": bool(
            math.isfinite(approach_cosine)
            and approach_cosine >= float(campaign.geometry["minimum_approach_side_cosine"])
        ),
        "camera_target_distance_valid": bool(
            float(campaign.geometry["minimum_camera_target_distance_m"])
            <= camera_norm
            <= float(campaign.geometry["maximum_camera_target_distance_m"])
        ),
        "deterministic_crop_derived": crop is not None,
    }
    resized_bbox_wh = [0.0, 0.0]
    if crop is not None:
        padding = float(campaign.crop["minimum_target_padding_px"])
        gates["target_inside_crop_with_padding"] = bool(
            bbox[0, 0] >= crop[0] + padding
            and bbox[0, 1] >= crop[1] + padding
            and bbox[1, 0] <= crop[2] - padding
            and bbox[1, 1] <= crop[3] - padding
        )
        crop_wh = np.asarray((crop[2] - crop[0], crop[3] - crop[1]), dtype=np.float64)
        output_wh = np.asarray(campaign.crop["output_size_wh"], dtype=np.float64)
        resized_bbox_wh = (full_wh * output_wh / crop_wh).tolist()
        gates["target_large_enough_after_crop_resize"] = bool(
            resized_bbox_wh[0] >= float(campaign.crop["minimum_resized_bbox_width_px"])
            and resized_bbox_wh[1] >= float(campaign.crop["minimum_resized_bbox_height_px"])
        )
    else:
        gates["target_inside_crop_with_padding"] = False
        gates["target_large_enough_after_crop_resize"] = False
    return {
        "candidate_id": candidate.candidate_id,
        "view_index": candidate.view_index,
        "projected_target_bbox_xyxy": [
            float(bbox[0, 0]), float(bbox[0, 1]), float(bbox[1, 0]), float(bbox[1, 1])
        ],
        "clipped_target_bbox_xyxy": [
            float(clip_low[0]), float(clip_low[1]), float(clip_high[0]), float(clip_high[1])
        ],
        "target_corner_depth_range_m": [float(depths.min()), float(depths.max())],
        "target_bbox_retained_fraction": retained,
        "target_bbox_area_fraction": area_fraction,
        "target_bbox_clipped_wh_px": clipped_wh.astype(float).tolist(),
        "camera_target_xy_distance_m": camera_norm,
        "stage1_approach_side_cosine": approach_cosine,
        "derived_crop_xyxy": list(crop) if crop is not None else None,
        "target_bbox_resized_wh_px": [float(value) for value in resized_bbox_wh],
        "crop_derivation_error": crop_error,
        "gates": gates,
        "all_geometry_gates_passed": all(gates.values()),
        "failed_geometry_gates": sorted(key for key, passed in gates.items() if not passed),
    }
