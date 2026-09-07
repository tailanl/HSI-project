"""Scene-only small/medium/large calibrated-depth candidate kernels."""
from __future__ import annotations
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import numpy as np
from hsi.common.artifacts import read_sealed
from . import depth_geometry as base
HERE = Path(__file__).resolve()
OUTPUT_SCHEMA = base.SCHEMA
OUTPUT_STATUS = "anonymous_sam31_box_prompts_ready"
P508_BASELINE: dict[str, Any] = {
    "edge_distance_m": 0.035,
    "minimum_object_height_m": 0.12,
    "maximum_object_height_m": 1.90,
    "minimum_world_height_span_m": 0.08,
    "minimum_component_pixels": 250,
    "minimum_component_extent_px": 18,
    "maximum_component_fraction": 0.16,
    "maximum_box_fraction": 0.45,
    "minimum_padding_px": 12,
    "padding_fraction": 0.30,
    "box_nms_iou": 0.74,
    "maximum_boxes_per_view": 18,
}

SCALE_PROFILES: dict[str, dict[str, Any]] = {
    "small": {
        "edge_distance_m": 0.022,
        "minimum_object_height_m": 0.08,
        "maximum_object_height_m": 2.00,
        "minimum_world_height_span_m": 0.035,
        "minimum_component_pixels": 72,
        "minimum_component_extent_px": 8,
        "maximum_component_fraction": 0.10,
        "maximum_box_fraction": 0.28,
        "minimum_padding_px": 6,
        "padding_fraction": 0.16,
        "box_nms_iou": 0.86,
        "maximum_boxes_per_view": 28,
    },
    "medium": dict(P508_BASELINE),
    "large": {
        "edge_distance_m": 0.060,
        "minimum_object_height_m": 0.06,
        "maximum_object_height_m": 2.20,
        "minimum_world_height_span_m": 0.06,
        "minimum_component_pixels": 180,
        "minimum_component_extent_px": 12,
        "maximum_component_fraction": 0.32,
        "maximum_box_fraction": 0.70,
        "minimum_padding_px": 18,
        "padding_fraction": 0.45,
        "box_nms_iou": 0.72,
        "maximum_boxes_per_view": 14,
    },
}
SCALE_ORDER = {"medium": 0, "small": 1, "large": 2}



class MultiscaleFeedbackError(RuntimeError):
    """An input or regeneration contract was violated."""


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise MultiscaleFeedbackError(message)


def load_render(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve(strict=True)
    value = read_sealed(resolved)
    require(value.get("schema") == base.SOURCE_SCHEMA, "wrong full-scene render schema")
    require(value.get("status") == "fullscene_multiview_ready", "full-scene render is not ready")
    views = value.get("views")
    require(isinstance(views, list) and bool(views), "render contains no views")
    require(len(views) == value.get("view_count"), "render view_count drift")
    contract = value.get("query_contract")
    require(isinstance(contract, Mapping), "render query contract is missing")
    for key in (
        "instruction_read",
        "semantic_target_read",
        "support_candidate_read",
        "motion_pose_contact_label_read",
        "planner_memory_or_hsi_read",
    ):
        require(contract.get(key) is False, f"render query contract violates {key}")
    return resolved, value


def parameter_changes() -> dict[str, dict[str, dict[str, Any]]]:
    changes: dict[str, dict[str, dict[str, Any]]] = {}
    for scale, profile in SCALE_PROFILES.items():
        changes[scale] = {
            key: {"from": P508_BASELINE[key], "to": value}
            for key, value in profile.items()
            if value != P508_BASELINE[key]
        }
    return changes


def _box_area(row: Mapping[str, Any]) -> int:
    x0, y0, x1, y1 = (int(value) for value in row["box_xyxy"])
    return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)


def _containment(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = (int(value) for value in left["box_xyxy"])
    bx0, by0, bx1, by1 = (int(value) for value in right["box_xyxy"])
    width = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    height = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    return float(width * height / max(1, min(_box_area(left), _box_area(right))))


def _proposal_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["anonymous_proposal_score"]),
        SCALE_ORDER[str(row["proposal_scale"])],
        _box_area(row),
        tuple(int(value) for value in row["box_xyxy"]),
        str(row["source_p508_proposal_id"]),
    )


def deduplicate_multiscale(
    rows: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
    containment_threshold: float,
    containment_area_ratio_minimum: float,
    maximum_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove only geometrically equivalent prompts and retain provenance."""

    for value, label in (
        (iou_threshold, "IoU threshold"),
        (containment_threshold, "containment threshold"),
        (containment_area_ratio_minimum, "containment area ratio"),
    ):
        require(math.isfinite(value) and 0.0 <= value <= 1.0, f"{label} outside [0,1]")
    require(type(maximum_rows) is int and maximum_rows >= 1, "maximum_rows must be positive")
    kept: list[dict[str, Any]] = []
    suppressions: list[dict[str, Any]] = []
    for source in sorted(rows, key=_proposal_rank):
        row = dict(source)
        row["equivalent_source_scales"] = [str(row["proposal_scale"])]
        row["deduplicated_source_proposals"] = [
            {
                "scale": str(row["proposal_scale"]),
                "proposal_id": str(row["source_p508_proposal_id"]),
            }
        ]
        duplicate: tuple[dict[str, Any], float, float, float] | None = None
        for other in kept:
            overlap_iou = float(base.iou(row, other))
            containment = _containment(row, other)
            area_ratio = min(_box_area(row), _box_area(other)) / max(
                1, max(_box_area(row), _box_area(other))
            )
            if overlap_iou >= iou_threshold or (
                containment >= containment_threshold
                and area_ratio >= containment_area_ratio_minimum
            ):
                duplicate = other, overlap_iou, containment, area_ratio
                break
        if duplicate is None:
            kept.append(row)
            continue
        other, overlap_iou, containment, area_ratio = duplicate
        other["equivalent_source_scales"] = sorted(
            set(other["equivalent_source_scales"]) | {str(row["proposal_scale"])}
        )
        other["deduplicated_source_proposals"].append(
            {
                "scale": str(row["proposal_scale"]),
                "proposal_id": str(row["source_p508_proposal_id"]),
            }
        )
        suppressions.append(
            {
                "reason": "cross_scale_geometric_duplicate",
                "kept_source_scale": other["proposal_scale"],
                "kept_source_proposal_id": other["source_p508_proposal_id"],
                "suppressed_source_scale": row["proposal_scale"],
                "suppressed_source_proposal_id": row["source_p508_proposal_id"],
                "box_iou": overlap_iou,
                "box_containment": containment,
                "box_area_ratio": float(area_ratio),
            }
        )
    if len(kept) > maximum_rows:
        for row in kept[maximum_rows:]:
            suppressions.append(
                {
                    "reason": "post_deduplication_view_budget",
                    "suppressed_source_scale": row["proposal_scale"],
                    "suppressed_source_proposal_id": row["source_p508_proposal_id"],
                }
            )
        kept = kept[:maximum_rows]
    for index, row in enumerate(kept):
        row["proposal_id"] = f"ANON_MS_BOX_{index:03d}"
    return kept, suppressions


def _validated_scale_rows(
    raw_rows: Sequence[Mapping[str, Any]], scale: str, width: int, height: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, source in enumerate(raw_rows):
        require(isinstance(source, Mapping), f"{scale} proposal {index} is malformed")
        require(source.get("semantic_label") is None, f"{scale} proposal carries a semantic label")
        box = source.get("box_xyxy")
        require(
            isinstance(box, list)
            and len(box) == 4
            and all(type(value) is int for value in box),
            f"{scale} proposal {index} has invalid box",
        )
        x0, y0, x1, y1 = box
        require(0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height, f"{scale} proposal box is outside image")
        score = source.get("anonymous_proposal_score")
        require(type(score) in (int, float) and math.isfinite(float(score)), f"{scale} proposal score is invalid")
        row = dict(source)
        row["source_p508_proposal_id"] = str(source.get("proposal_id", f"ANON_BOX_{index:03d}"))
        row["proposal_scale"] = scale
        row["scale_profile_id"] = f"p539_{scale}_depth_components_v1"
        output.append(row)
    return output


def propose_view_multiscale(
    depth: np.ndarray,
    k: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    width: int,
    height: int,
    iou_threshold: float,
    containment_threshold: float,
    containment_area_ratio_minimum: float,
    maximum_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}
    for scale in ("small", "medium", "large"):
        rows = base.propose(
            depth,
            k,
            world_to_camera,
            SimpleNamespace(**SCALE_PROFILES[scale]),
        )
        validated = _validated_scale_rows(rows, scale, width, height)
        raw_counts[scale] = len(validated)
        combined.extend(validated)
    kept, suppressions = deduplicate_multiscale(
        combined,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
        containment_area_ratio_minimum=containment_area_ratio_minimum,
        maximum_rows=maximum_rows,
    )
    return kept, {
        "raw_count_by_scale": raw_counts,
        "combined_raw_count": len(combined),
        "kept_count": len(kept),
        "suppressed_count": len(suppressions),
        "suppressions": suppressions,
    }
