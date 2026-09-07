"""Static body/support numerical helpers shared by placement and refine."""
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

from . import body_model as body_util
from . import refine_physics as target
from . import contact_materialize as sparse

TARGET_SCHEMA = "p498.functional_relation_surface_target.v3"

SUPPORT_ENVELOPE_FRACTION = 0.15

class KeyposeError(ValueError):
    pass

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise KeyposeError(message)

def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def rotation_z(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)

def support_inside_surface_fraction(
    support_vertices: np.ndarray,
    surface_bounds: np.ndarray,
    *,
    edge_slack_m: float = 0.015,
) -> float:
    """Fraction of the gluteal support envelope that lands on the candidate.

    Root/pelvis distance alone can pass while most of the body is outside a
    small seat.  This body-scale gate is intentionally evaluated in Stage-2,
    after a concrete SMPL-X keypose exists.
    """

    support = np.asarray(support_vertices, dtype=np.float64)
    bounds = np.asarray(surface_bounds, dtype=np.float64)
    require(support.ndim == 2 and support.shape[1] == 3 and len(support) > 0,
            "hips support vertices are malformed")
    require(bounds.shape == (2, 3), "surface bounds are malformed")
    lower = bounds[0, :2] - float(edge_slack_m)
    upper = bounds[1, :2] + float(edge_slack_m)
    inside = np.logical_and(support[:, :2] >= lower, support[:, :2] <= upper).all(axis=1)
    return float(inside.mean())

def hips_support_subset(vertices: np.ndarray, segmentation: Mapping[str, Any]) -> np.ndarray:
    """Select a fixed lower hips/gluteal envelope, never a lower-back half."""

    hips = np.asarray(segmentation.get("hips", []), dtype=np.int64)
    require(hips.ndim == 1 and len(hips) >= 100, "SMPL-X hips segmentation is missing or too small")
    require(int(hips.min()) >= 0 and int(hips.max()) < len(vertices), "hips segmentation index out of range")
    points = np.asarray(vertices, dtype=np.float64)[hips]
    threshold = float(np.quantile(points[:, 2], 0.20))
    pool = hips[points[:, 2] <= threshold + 1e-12]
    require(len(pool) >= 24, "lower hips support pool is too small")
    pool_points = vertices[pool]
    centre = np.median(pool_points, axis=0)
    distances = np.linalg.norm(pool_points - centre, axis=1)
    subset = np.asarray(pool[distances <= np.quantile(distances, 0.80) + 1e-12], dtype=np.int64)
    require(len(subset) >= 16 and len(np.unique(subset)) == len(subset), "hips support subset is malformed")
    return np.sort(subset)

def _float_diagnostics(branches: Any) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in branches.diagnostics.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().cpu())
        else:
            result[key] = int(value)
    return result

def materialize(
    body_model: Any,
    pose: torch.Tensor,
    betas: torch.Tensor,
    frame: torch.Tensor,
    contact: torch.Tensor,
    fixed_subset: torch.Tensor,
) -> Any:
    return sparse.materialize_pose_derived_root(
        body_model,
        pose,
        betas,
        frame,
        contact,
        fixed_subset,
        "fixed_subset_xy_mean_z_lower_envelope",
        SUPPORT_ENVELOPE_FRACTION,
    )
