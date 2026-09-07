#!/usr/bin/env python3
"""Fail-closed resolution of a Stage2 contact seed.

P530 separates the support-edge anchor used for approach orientation from the
interior support point used to place/refine the body.  Older Stage1 payloads do
not carry that proof, so they deliberately retain the historical surface-centre
fallback.  A partial or unverifiable P530 claim is never treated as legacy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


P530_PRODUCER_SCHEMA = "p530.action_space_aware_contact_selection.v1"
P530_PRODUCER_SCHEMA_V3 = "p530.action_space_aware_contact_selection.v3"
SUPPORTED_P530_PRODUCER_SCHEMAS = {
    P530_PRODUCER_SCHEMA,
    P530_PRODUCER_SCHEMA_V3,
}
P529_MASK_SCHEMA = "p529.filled_shape_filtered_interaction_masks.v1"
P530_MASK_SCHEMA = "p530.action_space_candidate_audit_masks.v1"


class ContactSeedError(ValueError):
    """A claimed interior Stage2 contact could not be proved."""


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ContactSeedError(message)


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == (size,) and np.isfinite(array).all(), f"{label} is malformed")
    return array


def _cell(value: Any, label: str) -> tuple[int, int]:
    array = np.asarray(value)
    _require(array.shape == (2,), f"{label} is malformed")
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ContactSeedError(f"{label} is malformed") from error
    _require(np.isfinite(numeric).all(), f"{label} is not finite")
    rounded = np.rint(numeric)
    _require(bool(np.all(np.abs(numeric - rounded) <= 1.0e-12)), f"{label} is not integral")
    return int(rounded[0]), int(rounded[1])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.casefold())


@dataclass(frozen=True)
class SupportMaskBinding:
    """Hash-checked 2-D support mask and its metric coordinate transform."""

    mask: np.ndarray
    lower_world_xy_m: np.ndarray
    cell_resolution_m: float
    selected_support_height_m: float
    artifact_path: Path
    artifact_sha256: str
    schema: str

    def contains_world_xy(self, world_xy_m: Sequence[float]) -> bool:
        xy = _vector(world_xy_m, 2, "same-surface candidate XY")
        continuous = (xy - self.lower_world_xy_m) / self.cell_resolution_m
        tolerance = 1.0e-9
        if bool(np.any(continuous < -tolerance)):
            return False
        if bool(np.any(continuous >= np.asarray(self.mask.shape, dtype=np.float64) + tolerance)):
            return False
        cell = np.floor(np.maximum(continuous, 0.0)).astype(np.int64)
        if bool(np.any(cell < 0) or np.any(cell >= np.asarray(self.mask.shape))):
            return False
        return bool(self.mask[int(cell[0]), int(cell[1])])


@dataclass(frozen=True)
class ResolvedContactSeed:
    surface_centre_world_xyz_m: np.ndarray
    contact_world_xyz_m: np.ndarray
    contact_surface_cell: tuple[int, int] | None
    contact_seed_source: str
    support_mask: SupportMaskBinding | None


def _support_mask_record(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return an explicitly named support-mask artifact, if one is carried."""

    locations: list[Any] = [
        payload.get("filled_contact_and_shape_filtered_approach_mask"),
        payload.get("p530_support_mask_artifact"),
        payload.get("stage2_contact_support_mask"),
    ]
    audit = payload.get("p530_action_space_audit")
    if isinstance(audit, Mapping):
        locations.extend((
            audit.get("source_p529_mask_artifact"),
            audit.get("support_mask_artifact"),
            audit.get("p530_action_space_mask_artifact"),
        ))
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        locations.extend((
            artifacts.get("source_p529_mask"),
            artifacts.get("p530_support_mask"),
            artifacts.get("stage2_contact_support_mask"),
        ))
    records = [value for value in locations if value is not None]
    if not records:
        return None
    _require(all(isinstance(value, Mapping) for value in records), "P530 support-mask artifact is malformed")
    hashes = {str(value.get("sha256", "")) for value in records}
    _require(len(hashes) == 1, "P530 carries ambiguous support-mask artifacts")
    return records[0]


def _load_support_mask(
    record: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    audit: Mapping[str, Any],
    artifact_base_dir: Path | None,
) -> SupportMaskBinding:
    raw_path = Path(str(record.get("path", ""))).expanduser()
    if not raw_path.is_absolute() and artifact_base_dir is not None:
        raw_path = artifact_base_dir / raw_path
    try:
        path = raw_path.resolve(strict=True)
    except OSError as error:
        raise ContactSeedError("P530 support-mask artifact path is invalid") from error
    try:
        recorded_bytes = int(record.get("bytes", -1))
    except (TypeError, ValueError) as error:
        raise ContactSeedError("P530 support-mask artifact byte count is malformed") from error
    _require(path.stat().st_size == recorded_bytes, "P530 support-mask artifact byte drift")
    recorded_sha = record.get("sha256")
    _require(_valid_sha256(recorded_sha), "P530 support-mask artifact SHA-256 is malformed")
    actual_sha = _sha256_file(path)
    _require(actual_sha == recorded_sha, "P530 support-mask artifact SHA-256 drift")
    expected_source_sha = audit.get("source_p529_mask_sha256")
    _require(_valid_sha256(expected_source_sha), "P530 audit source P529 mask SHA-256 is malformed")
    _require(actual_sha == expected_source_sha, "P530 support mask is not the audited P529 source mask")

    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "schema", "support_surface_mask", "lower_world_xy_m",
                "cell_resolution_m", "selected_support_height_m",
            }
            _require(required.issubset(set(archive.files)), "P530 support-mask archive is incomplete")
            schema = str(np.asarray(archive["schema"]).reshape(()))
            _require(schema in {P529_MASK_SCHEMA, P530_MASK_SCHEMA}, "P530 support-mask archive schema is unsupported")
            mask = np.asarray(archive["support_surface_mask"], dtype=np.bool_).copy()
            lower = _vector(archive["lower_world_xy_m"], 2, "support-mask lower XY")
            resolution = float(np.asarray(archive["cell_resolution_m"]).reshape(()))
            height = float(np.asarray(archive["selected_support_height_m"]).reshape(()))
            archive_scene_id = (
                str(np.asarray(archive["scene_id"]).reshape(()))
                if "scene_id" in archive else None
            )
            archive_target_id = (
                str(np.asarray(archive["target_instance_id"]).reshape(()))
                if "target_instance_id" in archive else None
            )
    except (OSError, ValueError) as error:
        if isinstance(error, ContactSeedError):
            raise
        raise ContactSeedError("P530 support-mask archive cannot be read safely") from error
    _require(mask.ndim == 2 and mask.size > 0, "P530 support surface mask is malformed")
    _require(math.isfinite(resolution) and resolution > 0.0, "P530 support-mask resolution is invalid")
    _require(math.isfinite(height), "P530 support height is invalid")
    scene_id = payload.get("scene_id")
    target_id = payload.get("target_instance_id", payload.get("selected_target_instance_id"))
    if archive_scene_id is not None and scene_id is not None:
        _require(archive_scene_id == str(scene_id), "P530 support-mask scene binding drift")
    if archive_target_id is not None and target_id is not None:
        _require(archive_target_id == str(target_id), "P530 support-mask target binding drift")
    return SupportMaskBinding(
        mask=mask,
        lower_world_xy_m=lower,
        cell_resolution_m=resolution,
        selected_support_height_m=height,
        artifact_path=path,
        artifact_sha256=actual_sha,
        schema=schema,
    )


def resolve_stage2_contact_seed(
    payload: Mapping[str, Any],
    approach: Mapping[str, Any],
    surface: Mapping[str, Any],
    *,
    artifact_base_dir: Path | None = None,
) -> ResolvedContactSeed:
    """Resolve legacy centre placement or a fully proved P530 interior contact."""

    centre = _vector(
        surface.get("centre_world_xyz_zup_m", surface.get("center_world_xyz_zup_m")),
        3,
        "surface centre",
    )
    bounds = np.asarray(surface.get("bounds_world_zup_m"), dtype=np.float64)
    _require(bounds.shape == (2, 3) and np.isfinite(bounds).all(), "surface bounds are malformed")
    _require(bool(np.all(bounds[0] <= bounds[1])), "surface bounds are inverted")
    _require(bool(np.all(centre >= bounds[0] - 1.0e-6) and np.all(centre <= bounds[1] + 1.0e-6)), "surface centre is outside bounds")

    proof_keys = {
        "stage2_contact_surface_cell",
        "stage2_contact_world_xyz_m",
        "stage2_contact_is_interior_action_space_validated",
    }
    producer = payload.get("producer_variant_schema")
    producer_claims_p530 = (
        isinstance(producer, str)
        and producer.startswith("p530.action_space_aware_contact_selection.")
    )
    proof_claimed = producer_claims_p530 or any(key in approach for key in proof_keys)
    if not proof_claimed:
        return ResolvedContactSeed(
            surface_centre_world_xyz_m=centre,
            contact_world_xyz_m=centre.copy(),
            contact_surface_cell=None,
            contact_seed_source="legacy_surface_centre_fallback",
            support_mask=None,
        )

    _require(
        producer in SUPPORTED_P530_PRODUCER_SCHEMAS,
        "interior Stage2 contact lacks a supported P530 producer proof",
    )
    _require(proof_keys.issubset(approach), "P530 interior Stage2 contact proof is incomplete")
    _require(
        approach.get("stage2_contact_is_interior_action_space_validated") is True,
        "P530 interior Stage2 contact validation flag is not true",
    )
    _require(approach.get("action_space_hard_gate_passed") is True, "P530 selected contact did not pass action-space hard gates")
    audit = payload.get("p530_action_space_audit")
    _require(isinstance(audit, Mapping), "P530 action-space provenance is missing")
    if producer == P530_PRODUCER_SCHEMA_V3:
        _require(
            audit.get("selected_only_after_hard_gates_and_direction_guards") is True,
            "P530 V3 selection was not made after hard gates and direction guards",
        )
        _require(
            audit.get("same_component_face_seed_normal_guard_applied") is True,
            "P530 V3 interaction-face guard is missing",
        )
        _require(
            audit.get("robust_frontier_ranked_before_original_p529_score") is True,
            "P530 V3 robust-frontier ranking proof is missing",
        )
        for key in (
            "direction_guard_passed",
            "interaction_face_guard_passed",
            "semantic_guard_passed",
            "shape_guard_passed",
            "robust_frontier_member",
        ):
            _require(approach.get(key) is True, f"P530 V3 selected contact failed {key}")
        _require(
            audit.get("selected_action_space_robustness_status")
            == approach.get("action_space_robustness_status"),
            "P530 V3 robustness-status provenance drift",
        )
        hard_pass_key = "action_space_hard_pass_candidate_count"
    else:
        _require(
            audit.get("selected_only_after_hard_gates") is True,
            "P530 selection was not made after hard gates",
        )
        hard_pass_key = "hard_pass_candidate_count"
    _require(audit.get("stage3_sdf_remains_final_physical_authority") is True, "P530 provenance weakens Stage3 SDF authority")
    try:
        enumerated_count = int(audit.get("enumerated_reachable_terminal_count", -1))
        hard_pass_count = int(audit.get(hard_pass_key, -1))
    except (TypeError, ValueError) as error:
        raise ContactSeedError("P530 action-space audit counts are malformed") from error
    _require(enumerated_count >= hard_pass_count >= 1, "P530 action-space audit counts are invalid")
    for key in ("source_p529_receipt_sha256", "source_p529_mask_sha256"):
        _require(_valid_sha256(audit.get(key)), f"P530 action-space provenance lacks valid {key}")

    contact = _vector(approach.get("stage2_contact_world_xyz_m"), 3, "P530 Stage2 contact world XYZ")
    cell = _cell(approach.get("stage2_contact_surface_cell"), "P530 Stage2 contact cell")
    _require("contact_world_xyz_m" in approach and "contact_surface_cell" in approach, "P530 legacy contact aliases are missing")
    legacy_contact = _vector(approach.get("contact_world_xyz_m"), 3, "P530 legacy contact world XYZ")
    legacy_cell = _cell(approach.get("contact_surface_cell"), "P530 legacy contact cell")
    _require(bool(np.allclose(contact, legacy_contact, rtol=0.0, atol=1.0e-10)), "P530 new/legacy contact world fields drift")
    _require(cell == legacy_cell, "P530 new/legacy contact cell fields drift")
    _require(bool(np.all(contact >= bounds[0] - 1.0e-6) and np.all(contact <= bounds[1] + 1.0e-6)), "P530 Stage2 contact lies outside surface bounds")

    support_mask = None
    record = _support_mask_record(payload)
    if record is not None:
        support_mask = _load_support_mask(
            record,
            payload=payload,
            audit=audit,
            artifact_base_dir=artifact_base_dir,
        )
        _require(0 <= cell[0] < support_mask.mask.shape[0] and 0 <= cell[1] < support_mask.mask.shape[1], "P530 Stage2 contact cell is outside support-mask bounds")
        _require(bool(support_mask.mask[cell]), "P530 Stage2 contact cell is not on the true support mask")
        expected_xy = support_mask.lower_world_xy_m + (
            np.asarray(cell, dtype=np.float64) + 0.5
        ) * support_mask.cell_resolution_m
        tolerance = max(1.0e-8, support_mask.cell_resolution_m * 1.0e-6)
        _require(bool(np.allclose(contact[:2], expected_xy, rtol=0.0, atol=tolerance)), "P530 Stage2 contact cell/world XY alignment drift")
        _require(abs(float(contact[2]) - support_mask.selected_support_height_m) <= tolerance, "P530 Stage2 contact/support height drift")

    return ResolvedContactSeed(
        surface_centre_world_xyz_m=centre,
        contact_world_xyz_m=contact,
        contact_surface_cell=cell,
        contact_seed_source="p530_validated_interior_action_space_contact",
        support_mask=support_mask,
    )
