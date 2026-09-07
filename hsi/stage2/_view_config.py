"""Static current calibrated-view configuration; no model or historical code loading."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from PIL import Image
HERE = Path(__file__).resolve()

CONFIG_SCHEMA = "p536.qwen3vl_stage2_view_crop_audit_config.v1"
BUILD_SCHEMA = "p536.stage2_view_audit_config_build.v1"
BLOCKED_SCHEMA = "p536.stage2_view_audit_config_build_blocked.v1"
FULLSCENE_SCHEMA = "p508.lingo_fullscene_multiview_render.v1"
CAMERA_MANIFEST_SCHEMA = "p508.lingo_fullscene_calibrated_cameras.v1"
FULLSCENE_STATUS = "fullscene_multiview_ready"
CAMERA_MANIFEST_STATUS = "fullscene_calibrated_views_ready"
LINEAGE_POLICIES = ("automatic-only", "oracle-isolation")
TAINT_TOKENS = ("oracle", "manual", "human_review", "review_only")
FORBIDDEN_INPUT_NAME_TOKENS = (
    "selection_manifest",
    "selected_view_crop",
    "view_crop_selection",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

GEOMETRY_THRESHOLDS = {
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
CROP_POLICY = {
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
QWEN_POLICY = {
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


class ConfigBuildError(RuntimeError):
    """An input is unverified, inconsistent, tainted, or incomplete."""


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ConfigBuildError(message)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ConfigBuildError(f"cannot read strict JSON {label}: {error}") from error
    require(isinstance(value, dict), f"{label} root is not an object")
    return value


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


def checked_artifact(value: Any, label: str, *, artifact_base_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    require(isinstance(value, Mapping), f"{label} is not a hash-bound artifact")
    require({"path", "bytes", "sha256"} <= set(value), f"{label} lacks path/bytes/sha256")
    raw_path = Path(str(value["path"])).expanduser()
    require(raw_path.is_absolute() or artifact_base_dir is not None,
            f"{label} relative artifact requires explicit artifact_base_dir")
    path = (raw_path if raw_path.is_absolute() else Path(artifact_base_dir) / raw_path).resolve(strict=True)
    observed = artifact(path)
    require(type(value["bytes"]) is int, f"{label}.bytes is not an integer")
    require(int(value["bytes"]) == observed["bytes"], f"{label} byte count drift")
    require(
        isinstance(value["sha256"], str) and SHA256_PATTERN.fullmatch(value["sha256"]) is not None,
        f"{label}.sha256 is invalid",
    )
    require(value["sha256"] == observed["sha256"], f"{label} SHA256 drift")
    return path, observed


def validate_payload_hash(value: Mapping[str, Any], key: str, label: str) -> None:
    declared = value.get(key)
    require(
        isinstance(declared, str) and SHA256_PATTERN.fullmatch(declared) is not None,
        f"{label} lacks valid {key}",
    )
    payload = dict(value)
    payload.pop(key, None)
    require(canonical_hash(payload) == declared, f"{label} payload hash drift")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def reject_old_selection_authority(path: Path, label: str) -> None:
    lowered = str(path).casefold()
    offending = [token for token in FORBIDDEN_INPUT_NAME_TOKENS if token in lowered]
    require(not offending, f"{label} points to forbidden old view/crop selection authority")


def _walk_strings(value: Any, pointer: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_pointer = f"{pointer}/{key}"
            yield next_pointer + "/@key", str(key)
            yield from _walk_strings(item, next_pointer)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{pointer}/{index}")
    elif isinstance(value, str):
        yield pointer, value


def lineage_taint(value: Mapping[str, Any], authority: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for pointer, text in _walk_strings(value):
        lowered = text.casefold()
        hits = [token for token in TAINT_TOKENS if token in lowered]
        if hits:
            evidence.append({
                "authority": authority,
                "json_pointer": pointer,
                "matched_tokens": ",".join(sorted(set(hits))),
                "value_excerpt": text[:240],
            })
    return evidence


def _finite_pair(value: Any, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 2, f"{label} must be a two-item list")
    require(
        all(type(item) in (int, float) and type(item) is not bool and math.isfinite(float(item)) for item in value),
        f"{label} must contain finite numbers",
    )
    return [float(value[0]), float(value[1])]


def _validate_target(target: Mapping[str, Any]) -> dict[str, Any]:
    validate_payload_hash(target, "receipt_payload_sha256", "Stage1 target")
    require(str(target.get("status", "")).startswith("published"), "Stage1 target is not published")
    scene_id = str(target.get("scene_id", "")).strip()
    instruction = str(target.get("instruction", "")).strip()
    target_id = str(
        target.get("selected_target_instance_id", target.get("target_instance_id", ""))
    ).strip()
    surface_id = str(target.get("selected_surface_id", target.get("target_surface_id", ""))).strip()
    target_class = str(target.get("target_class", target.get("requested_target_class", ""))).strip()
    require(scene_id and instruction, "Stage1 target lacks scene_id/instruction")
    require(target_id and surface_id and target_class, "Stage1 target lacks target/surface/class identity")
    bounds = target.get("target_bounds_world_zup_m")
    require(
        isinstance(bounds, list)
        and len(bounds) == 2
        and all(isinstance(row, list) and len(row) == 3 for row in bounds),
        "Stage1 target bounds are malformed",
    )
    low = [float(item) for item in bounds[0]]
    high = [float(item) for item in bounds[1]]
    require(all(math.isfinite(item) for item in low + high), "Stage1 target bounds are non-finite")
    require(all(high[index] > low[index] for index in range(3)), "Stage1 target bounds are inverted")
    return {
        "scene_id": scene_id,
        "instruction": instruction,
        "target_instance_id": target_id,
        "target_surface_id": surface_id,
        "target_class": target_class,
    }


def _extract_approach(
    source: Mapping[str, Any], identity: Mapping[str, Any]
) -> tuple[list[float], str]:
    if source.get("scene_id") is not None:
        require(str(source["scene_id"]) == identity["scene_id"], "approach scene_id drift")
    if source.get("instruction") is not None:
        require(str(source["instruction"]) == identity["instruction"], "approach instruction drift")
    source_target_id = source.get("selected_target_instance_id", source.get("target_instance_id"))
    if source_target_id is not None:
        require(str(source_target_id) == identity["target_instance_id"], "approach target identity drift")
    source_surface_id = source.get("selected_surface_id", source.get("target_surface_id"))
    if source_surface_id is not None:
        require(str(source_surface_id) == identity["target_surface_id"], "approach surface identity drift")

    for key in ("selected_approach_world_xy_m", "approach_world_xy_m"):
        if source.get(key) is not None:
            return _finite_pair(source[key], key), f"/{key}"

    candidates = source.get("candidate_surfaces")
    require(isinstance(candidates, list), "approach source has no candidate_surfaces")
    selected_id = str(source.get("selected_candidate_id", identity["target_surface_id"]))
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            continue
        row_ids = {
            str(row.get("candidate_id", "")),
            str(row.get("surface_id", "")),
            str(row.get("interaction_surface_id", "")),
        }
        if selected_id in row_ids or identity["target_surface_id"] in row_ids:
            matches.append((index, row))
    require(len(matches) == 1, "approach source does not uniquely bind the selected surface")
    index, selected = matches[0]
    approach = selected.get("approach")
    require(isinstance(approach, Mapping), "selected surface lacks an approach object")
    require(
        approach.get("action_space_hard_gate_passed") is True,
        "selected approach did not pass its action-space hard gate",
    )
    pointer = f"/candidate_surfaces/{index}/approach/approach_world_xy_m"
    return _finite_pair(approach.get("approach_world_xy_m"), pointer), pointer


def _automatic_lineage_evidence(
    target: Mapping[str, Any], taint: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    contract = target.get("current_only_contract")
    checks = {
        "no_oracle_or_manual_taint": not taint,
        "current_only_contract_present": isinstance(contract, Mapping),
        "fresh_current_occupancy_used": bool(
            isinstance(contract, Mapping) and contract.get("fresh_current_occupancy_used") is True
        ),
        "historical_surface_or_route_not_read": bool(
            isinstance(contract, Mapping) and contract.get("historical_surface_or_route_read") is False
        ),
        "motion_pose_contact_keypose_or_future_frame_not_read": bool(
            isinstance(contract, Mapping)
            and contract.get("motion_pose_contact_keypose_or_future_frame_read") is False
        ),
    }
    return {"checks": checks, "automatic_target_eligible": all(checks.values())}


def _validate_camera_file(path: Path, view_index: int, row: Mapping[str, Any]) -> None:
    camera = read_json(path, f"camera view {view_index}")
    require(camera.get("camera_id") == view_index, f"camera {view_index} id drift")
    require(camera.get("width") == row.get("width"), f"camera {view_index} width drift")
    require(camera.get("height") == row.get("height"), f"camera {view_index} height drift")
    require(camera.get("extrinsic_convention") == "opencv_world_to_camera", "camera extrinsic convention drift")
    matrix = camera.get("world_to_camera")
    intrinsics = camera.get("intrinsics", camera.get("K"))
    require(
        isinstance(matrix, list) and len(matrix) == 4 and all(isinstance(line, list) and len(line) == 4 for line in matrix),
        f"camera {view_index} world_to_camera malformed",
    )
    require(
        isinstance(intrinsics, list)
        and len(intrinsics) == 3
        and all(isinstance(line, list) and len(line) == 3 for line in intrinsics),
        f"camera {view_index} intrinsics malformed",
    )
    numbers = [float(item) for line in matrix for item in line]
    numbers.extend(float(item) for line in intrinsics for item in line)
    require(all(math.isfinite(item) for item in numbers), f"camera {view_index} has non-finite calibration")


def _load_fullscene_candidates(
    receipt_path: Path, expected_scene_id: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    receipt = read_json(receipt_path, "fullscene render receipt")
    validate_payload_hash(receipt, "receipt_payload_sha256", "fullscene render receipt")
    require(receipt.get("schema") == FULLSCENE_SCHEMA, "wrong fullscene receipt schema")
    require(receipt.get("status") == FULLSCENE_STATUS, "fullscene render is not ready")
    require(str(receipt.get("scene_id")) == expected_scene_id, "fullscene scene_id drift")
    query_contract = receipt.get("query_contract")
    require(isinstance(query_contract, Mapping), "fullscene receipt lacks query_contract")
    for key in (
        "semantic_target_read",
        "motion_pose_contact_label_read",
        "p480_candidate_or_receipt_read",
        "planner_memory_or_hsi_read",
        "support_candidate_read",
    ):
        require(query_contract.get(key) is False, f"fullscene renderer was not instruction/target blind: {key}")

    manifest_path, manifest_record = checked_artifact(
        receipt.get("calibrated_camera_manifest"), "calibrated camera manifest"
    )
    manifest = read_json(manifest_path, "calibrated camera manifest")
    validate_payload_hash(manifest, "payload_sha256", "calibrated camera manifest")
    require(manifest.get("schema") == CAMERA_MANIFEST_SCHEMA, "wrong calibrated camera manifest schema")
    require(manifest.get("status") == CAMERA_MANIFEST_STATUS, "calibrated cameras are not ready")
    require(str(manifest.get("scene_id")) == expected_scene_id, "camera manifest scene_id drift")
    require(manifest.get("coordinate_system") == "world_zup_metric", "camera coordinate system drift")
    require(manifest.get("extrinsic_convention") == "opencv_world_to_camera", "camera extrinsic convention drift")

    views = manifest.get("views")
    require(isinstance(views, list) and 2 <= len(views) <= 48, "candidate view count is outside [2,48]")
    require(manifest.get("view_count") == len(views), "camera manifest view_count drift")
    require(receipt.get("camera_sweep", {}).get("view_count") == len(views), "render receipt view_count drift")
    indices = [row.get("view_index") if isinstance(row, Mapping) else None for row in views]
    require(all(type(index) is int for index in indices), "candidate view_index is invalid")
    require(sorted(indices) == list(range(len(views))), "candidate views are not one complete contiguous sweep")

    candidates: list[dict[str, Any]] = []
    rgb_hashes: set[str] = set()
    for row in sorted(views, key=lambda item: item["view_index"]):
        index = int(row["view_index"])
        rgb_path, rgb_record = checked_artifact(row.get("rgb"), f"view {index} RGB")
        camera_path, camera_record = checked_artifact(row.get("camera"), f"view {index} camera")
        require(rgb_record["sha256"] not in rgb_hashes, f"duplicate RGB bytes at view {index}")
        rgb_hashes.add(rgb_record["sha256"])
        require(type(row.get("width")) is int and type(row.get("height")) is int, "invalid view dimensions")
        with Image.open(rgb_path) as image:
            require(image.size == (row["width"], row["height"]), f"view {index} RGB size drift")
            image.verify()
        _validate_camera_file(camera_path, index, row)
        candidates.append({
            "candidate_id": f"view_{index:02d}",
            "view_index": index,
            "rgb": rgb_record,
            "camera": camera_record,
        })
    return receipt, manifest_record, candidates


def build_config(
    stage1_target_path: Path,
    fullscene_render_receipt_path: Path,
    approach_source_path: Path,
    output_path: Path,
    lineage_policy: str,
) -> dict[str, Any]:
    require(lineage_policy in LINEAGE_POLICIES, f"invalid lineage policy: {lineage_policy}")
    paths = {
        "stage1_target": stage1_target_path.resolve(strict=True),
        "fullscene_render_receipt": fullscene_render_receipt_path.resolve(strict=True),
        "stage1_approach_source": approach_source_path.resolve(strict=True),
    }
    for label, path in paths.items():
        reject_old_selection_authority(path, label)

    target = read_json(paths["stage1_target"], "Stage1 target")
    identity = _validate_target(target)
    approach_source = read_json(paths["stage1_approach_source"], "Stage1 approach source")
    validate_payload_hash(approach_source, "receipt_payload_sha256", "Stage1 approach source")
    approach_xy, approach_pointer = _extract_approach(approach_source, identity)

    taint = lineage_taint(target, "stage1_target")
    if paths["stage1_approach_source"] != paths["stage1_target"]:
        taint.extend(lineage_taint(approach_source, "stage1_approach_source"))
    automatic_evidence = _automatic_lineage_evidence(target, taint)
    if lineage_policy == "automatic-only":
        require(
            automatic_evidence["automatic_target_eligible"] is True,
            "automatic-only policy rejected non-automatic or insufficiently proven Stage1 lineage",
        )

    render_receipt, camera_manifest_record, candidates = _load_fullscene_candidates(
        paths["fullscene_render_receipt"], identity["scene_id"]
    )
    isolation_only = lineage_policy == "oracle-isolation"
    config: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "status": "candidate_views_ready",
        "scene_id": identity["scene_id"],
        "instruction": identity["instruction"],
        "target_instance_id": identity["target_instance_id"],
        "target_surface_id": identity["target_surface_id"],
        "stage1_target": artifact(paths["stage1_target"]),
        "approach_world_xy_m": approach_xy,
        "stage1_approach_source": artifact(paths["stage1_approach_source"]),
        "fullscene_render_receipt": artifact(paths["fullscene_render_receipt"]),
        "calibrated_camera_manifest": camera_manifest_record,
        "candidate_views": candidates,
        "geometry_thresholds": dict(GEOMETRY_THRESHOLDS),
        "crop_policy": dict(CROP_POLICY),
        "qwen_policy": json.loads(json.dumps(QWEN_POLICY)),
        "config_build": {
            "schema": BUILD_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "lineage_policy": lineage_policy,
            "approach_json_pointer": approach_pointer,
            "source_render_receipt_payload_sha256": render_receipt["receipt_payload_sha256"],
            "lineage_audit": {
                "taint_evidence": taint,
                "automatic_evidence": automatic_evidence,
                "oracle_lineage_isolation_only": isolation_only,
                "end_to_end_automatic_claim_allowed": bool(
                    not isolation_only and automatic_evidence["automatic_target_eligible"]
                ),
                "claim_restriction": (
                    "This job isolates replacement of the former Stage2 view/crop audit; "
                    "its Stage1 target lineage is not evidence of an end-to-end automatic pipeline."
                    if isolation_only
                    else None
                ),
            },
            "authority_contract": {
                "old_selection_manifest_read": False,
                "preselected_view_or_crop_read": False,
                "all_fullscene_manifest_views_included": True,
                "candidate_subset_selection_performed": False,
                "target_overlay_supplied": False,
                "target_overlay_must_be_regenerated_by_view_auditor": True,
                "gpu_or_qwen_invoked_by_builder": False,
            },
            "source": artifact(HERE),
        },
    }
    payload = dict(config)
    config["config_payload_sha256"] = canonical_hash(payload)
    json.dumps(config, allow_nan=False)
    atomic_json(output_path, config)
    return config
