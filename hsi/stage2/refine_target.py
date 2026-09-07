"""Current target binding and separated contact/collision degrees of freedom."""
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

from . import refine_contract as contract
from . import body_geometry as p498
from . import contact_seed as contact_seed_contract

CONTACT_SUPPORT_ROWS = (0, 1, 3, 4, 6, 7)

COLLISION_ESCAPE_ROWS = (2, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20)

ACTIVE_ROWS = CONTACT_SUPPORT_ROWS + COLLISION_ESCAPE_ROWS

CONTACT_SUPPORT_DELTA_LIMIT_RAD = 0.75

COLLISION_ESCAPE_DELTA_LIMIT_RAD = 0.45

CONTACT_JOINT_IDS = (0, 1, 2)

def _float_diagnostics(branches: Any) -> dict[str, float | int]:
    return p498._float_diagnostics(branches)

def _validate_target(
    target_receipt: Mapping[str, Any],
    *,
    artifact_base_dir: Path | None = None,
) -> tuple[
    str,
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    contact_seed_contract.ResolvedContactSeed,
]:
    contract.require(
        target_receipt.get("schema") == p498.TARGET_SCHEMA,
        "normalized Stage1 target has the wrong P498 compatibility schema",
    )
    contract.require(
        str(target_receipt.get("status", "")).startswith("published"),
        "normalized Stage1 target is not published",
    )
    contract.require(target_receipt.get("current_only_stage1_verified") is True, "target lacks current Stage1 proof")
    contract.require(target_receipt.get("action_family") == "sit", "current retarget supports sit only")
    target_class = str(target_receipt.get("target_class", "")).strip().casefold()
    contract.require(
        target_class in contract.SITTABLE_TARGET_CLASSES,
        f"current sit target is not a supported sittable object: {target_class}",
    )
    target_id = str(target_receipt.get("selected_candidate_id", ""))
    surface_hash = str(target_receipt.get("support_component_sha256", ""))
    contract.require(target_id and len(surface_hash) == 64, "normalized target identity is invalid")
    relation = contract.normalize_relation(target_receipt.get("reference_relation"))
    contract.require(
        relation == "none" or relation in {"beside", "associated_with_reference_setup"},
        f"unsupported Stage1 target relation: {relation}",
    )
    surface = target_receipt.get("surface")
    contract.require(isinstance(surface, Mapping), "normalized target surface is missing")
    centre = np.asarray(surface.get("centre_world_xyz_zup_m"), dtype=np.float64)
    bounds = np.asarray(surface.get("bounds_world_zup_m"), dtype=np.float64)
    contract.require(centre.shape == (3,) and np.isfinite(centre).all(), "target contact centre is malformed")
    contract.require(bounds.shape == (2, 3) and np.isfinite(bounds).all(), "target bounds are malformed")
    # Horizontal support surfaces may be exactly planar in Z.  Their XY
    # footprint, however, must remain a proper positive-area rectangle.
    contract.require(bool(np.all(bounds[0] <= bounds[1])), "target bounds are inverted")
    contract.require(bool(np.all(bounds[0, :2] < bounds[1, :2])), "target XY footprint has non-positive extent")
    approach = target_receipt.get("approach")
    contract.require(isinstance(approach, Mapping), "normalized target approach is missing")
    yaw = float(approach.get("contact_forward_yaw_rad"))
    contract.require(math.isfinite(yaw), "target yaw is not finite")
    try:
        resolved_contact = contact_seed_contract.resolve_stage2_contact_seed(
            target_receipt,
            approach,
            surface,
            artifact_base_dir=artifact_base_dir,
        )
    except contact_seed_contract.ContactSeedError as error:
        raise contract.CurrentOnlyContractError(str(error)) from error
    return (
        target_id,
        surface_hash,
        centre,
        resolved_contact.contact_world_xyz_m,
        bounds,
        yaw,
        resolved_contact,
    )

def _positive_offsets(maximum: float, step: float) -> np.ndarray:
    grid = contract.inclusive_offsets(float(maximum), float(step))
    return np.ascontiguousarray(grid[grid >= -1.0e-12])
