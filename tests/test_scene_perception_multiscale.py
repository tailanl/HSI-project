"""Original numerical multiscale tests; task-feedback replay is out of Stage1A scope."""
from __future__ import annotations
from typing import Any
import numpy as np
from hsi.stage1.perception import multiscale as MODULE

def _proposal(box: list[int], score: float, proposal_id: str) -> dict[str, Any]:
    x0, y0, x1, y1 = box
    return {
        "proposal_id": proposal_id,
        "box_xyxy": box,
        "box_xywh": {
            "x": x0,
            "y": y0,
            "width": x1 - x0 + 1,
            "height": y1 - y0 + 1,
        },
        "anonymous_proposal_score": score,
        "prompt_kind": "calibrated_depth_discontinuity_box",
        "semantic_label": None,
    }


def test_deduplication_retains_cross_scale_provenance() -> None:
    rows = []
    for scale, score in (("small", 8.0), ("medium", 10.0), ("large", 7.0)):
        row = _proposal([5, 5, 20, 20], score, f"{scale}_0")
        row.update(
            {
                "proposal_scale": scale,
                "source_p508_proposal_id": f"{scale}_0",
                "scale_profile_id": f"p539_{scale}_depth_components_v1",
            }
        )
        rows.append(row)

    kept, suppressions = MODULE.deduplicate_multiscale(
        rows,
        iou_threshold=0.9,
        containment_threshold=0.98,
        containment_area_ratio_minimum=0.82,
        maximum_rows=10,
    )

    assert len(kept) == 1
    assert kept[0]["proposal_scale"] == "medium"
    assert kept[0]["equivalent_source_scales"] == ["large", "medium", "small"]
    assert len(suppressions) == 2
    assert all(row["reason"] == "cross_scale_geometric_duplicate" for row in suppressions)


def test_real_p508_depth_proposal_path_runs_on_cpu() -> None:
    depth = np.zeros((96, 96), dtype=np.float64)
    for y in range(25, 56):
        depth[y, 30:61] = 0.70 + 0.008 * (y - 25)
    k = np.asarray([[80.0, 0.0, 48.0], [0.0, 80.0, 48.0], [0.0, 0.0, 1.0]])

    rows, audit = MODULE.propose_view_multiscale(
        depth,
        k,
        np.eye(4, dtype=np.float64),
        width=96,
        height=96,
        iou_threshold=0.90,
        containment_threshold=0.98,
        containment_area_ratio_minimum=0.82,
        maximum_rows=48,
    )

    assert audit["raw_count_by_scale"].keys() == {"small", "medium", "large"}
    assert audit["combined_raw_count"] >= 1
    assert rows
    assert all(row["semantic_label"] is None for row in rows)
    assert all(row["proposal_scale"] in {"small", "medium", "large"} for row in rows)
