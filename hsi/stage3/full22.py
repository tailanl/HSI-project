#!/usr/bin/env python3
"""P533-only runtime repair for legacy P360's coupled sparse ICGF field.

The integrated executor uses the current static rollout implementation.  That
implementation uses ``icgf_contact_joints`` both to build the sparse ICGF and
to exempt terminal-tail SDF rows.  P533 needs two different sets:

* all 22 packet joints are attracted to the Stage-2 keypose by ICGF;
* only the three declared contact-role joints retain contact/SDF semantics.

This module installs a narrow, process-local context manager.  It never edits
P360/P478 source files.  The legacy rollout, diffusion sampler, checkpoints,
and serializers remain unchanged.  Every patched global is restored in a
``finally`` block.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


RUNTIME_SCHEMA = "p533.full22_sparse_joint_icgf_runtime.v1"
BUILDER_SCHEMA = "p533.full22_sparse_joint_icgf_builder.v1"
IMPLEMENTATION = "p533_process_local_decoupled_sparse_icgf_v1"


class Full22ICGFAdapterError(ValueError):
    """The isolated P533 ICGF adapter is not wired exactly as declared."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Full22ICGFAdapterError(message)


def _ids(value: Sequence[int] | torch.Tensor | None) -> tuple[int, ...]:
    if value is None:
        result: tuple[int, ...] = ()
    elif torch.is_tensor(value):
        result = tuple(int(item) for item in value.detach().cpu().reshape(-1).tolist())
    else:
        result = tuple(int(item) for item in value)
    _require(
        len(result) == len(set(result)) and all(0 <= item < 22 for item in result),
        "joint IDs must be unique SMPL-X22 indices",
    )
    return result


def constraint_count(
    guidance_joint_ids: Sequence[int],
    *,
    contact_frames: int,
    valid_future_frames: int,
    batch_size: int = 1,
) -> int:
    ids = _ids(guidance_joint_ids)
    _require(int(contact_frames) >= 1, "contact_frames must be positive")
    _require(int(valid_future_frames) >= 1, "valid_future_frames must be positive")
    _require(int(batch_size) >= 1, "batch_size must be positive")
    return (
        int(batch_size)
        * min(int(contact_frames), int(valid_future_frames))
        * len(ids)
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _require(
        value.shape == mask.shape and mask.dtype == torch.bool,
        "masked mean needs an aligned boolean mask",
    )
    if bool(mask.any()):
        return value.masked_select(mask).mean()
    return value.sum() * 0.0


def _decoupled_energy_from_world_joints(
    self: Any, joints_world_zup: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact legacy scene energy with independent ICGF/SDF joint sets."""

    if joints_world_zup.ndim != 4 or joints_world_zup.shape[-2:] != (22, 3):
        raise Full22ICGFAdapterError("world joints must be [B,T,22,3]")
    joints = joints_world_zup[:, : self.valid_future_frames]
    _require(int(joints.shape[0]) == 1, "P533/P478 runtime requires batch size one")
    zero = joints.sum() * 0.0

    guidance_ids = _ids(getattr(self, "icgf_guidance_joint_ids", ()))
    exempt_ids = _ids(getattr(self, "sdf_exempt_joint_ids", ()))
    semantic_ids = _ids(getattr(self, "contact_joint_ids", ()))
    _require(
        exempt_ids == semantic_ids,
        "P533 SDF-exempt joints must equal semantic contact-role joints",
    )
    _require(
        set(exempt_ids).issubset(set(guidance_ids)),
        "P533 SDF-exempt joints must be an ICGF-guidance subset",
    )
    if self.icgf is not None:
        expected_ids = _ids(getattr(self, "p533_expected_icgf_guidance_joint_ids", ()))
        _require(guidance_ids == expected_ids, "runtime ICGF guidance IDs drifted")
    else:
        _require(not guidance_ids and not exempt_ids, "inactive ICGF retained joint IDs")

    collision_ids_tuple = _ids(self.collision_joint_ids)
    collision_ids = torch.as_tensor(
        collision_ids_tuple, device=joints.device, dtype=torch.long
    )
    if collision_ids.numel():
        collision_points = joints.index_select(2, collision_ids)
        sdf_samples = self.scene_sdf.sample(collision_points)
        sdf = sdf_samples.signed_distance_m
        sdf_mask = torch.ones_like(sdf, dtype=torch.bool)
        if self.config.exempt_contact_joints_from_sdf_tail and exempt_ids:
            exempt_set = set(exempt_ids)
            exemptions = torch.as_tensor(
                [item in exempt_set for item in collision_ids_tuple],
                device=joints.device,
                dtype=torch.bool,
            )
            tail_count = min(int(self.config.contact_frames), self.valid_future_frames)
            sdf_mask[:, -tail_count:, exemptions] = False
        penetration = F.relu(float(self.config.collision_clearance_m) - sdf)
        sdf_energy = _masked_mean(penetration.square(), sdf_mask)
        sdf_constraint_count = sdf_mask.to(joints.dtype).sum()
        sdf_active_count = (sdf_mask & (penetration > 0.0)).to(joints.dtype).sum()
        penetration_fraction = _masked_mean((sdf < 0.0).to(sdf.dtype), sdf_mask)
        in_bounds_fraction = _masked_mean(
            sdf_samples.in_bounds.to(sdf.dtype), sdf_mask
        )
        minimum_sdf = (
            torch.where(sdf_mask, sdf, torch.full_like(sdf, torch.inf)).amin()
            if bool(sdf_mask.any())
            else zero
        )
    else:
        sdf_energy = zero
        sdf_constraint_count = zero
        sdf_active_count = zero
        penetration_fraction = zero
        in_bounds_fraction = zero
        minimum_sdf = zero

    icgf_energy = zero
    mean_icgf_distance = zero
    final_icgf_distance = zero
    icgf_constraint_count = zero
    expected_icgf_count = 0
    observed_query_shape = (0, 0, 0, 3)
    if self.icgf is not None and guidance_ids:
        selected = torch.as_tensor(
            guidance_ids, device=joints.device, dtype=torch.long
        )
        tail_count = min(int(self.config.contact_frames), self.valid_future_frames)
        query = joints[:, -tail_count:].index_select(2, selected)
        observed_query_shape = tuple(int(value) for value in query.shape)
        distance = self.icgf.soft_distance(query, joint_ids=selected)
        violation = F.relu(distance - float(self.config.icgf_deadzone_m))
        per_value = F.smooth_l1_loss(
            violation,
            torch.zeros_like(violation),
            reduction="none",
            beta=float(self.config.icgf_huber_beta_m),
        )
        ramp = torch.linspace(
            1.0 / tail_count,
            1.0,
            tail_count,
            device=joints.device,
            dtype=joints.dtype,
        )
        weights = ramp[None, :, None].expand_as(per_value)
        icgf_energy = (per_value * weights).sum() / weights.sum().clamp_min(1.0e-8)
        expected_icgf_count = constraint_count(
            guidance_ids,
            contact_frames=int(self.config.contact_frames),
            valid_future_frames=int(self.valid_future_frames),
            batch_size=int(joints.shape[0]),
        )
        _require(
            int(per_value.numel()) == expected_icgf_count,
            "ICGF energy did not consume every declared joint/frame constraint",
        )
        icgf_constraint_count = torch.as_tensor(
            per_value.numel(), device=joints.device, dtype=joints.dtype
        )
        mean_icgf_distance = distance.mean()
        final_icgf_distance = distance[:, -1].mean()

    total = (
        float(self.config.sdf_weight) * sdf_energy
        + float(self.config.icgf_weight) * icgf_energy
    )
    scalar = lambda value: torch.as_tensor(  # noqa: E731
        value, device=joints.device, dtype=joints.dtype
    )
    return total, {
        "scene_energy": total,
        "sdf_energy": sdf_energy,
        "icgf_energy": icgf_energy,
        "sdf_constraint_count": sdf_constraint_count,
        "sdf_active_constraint_count": sdf_active_count,
        "icgf_constraint_count": icgf_constraint_count,
        "minimum_sdf_m": minimum_sdf,
        "penetration_fraction": penetration_fraction,
        "in_bounds_fraction": in_bounds_fraction,
        "mean_icgf_distance_m": mean_icgf_distance,
        "final_icgf_distance_m": final_icgf_distance,
        "p533_full22_icgf_adapter_active": scalar(self.icgf is not None),
        "p533_actual_icgf_constraint_count": scalar(expected_icgf_count),
        "p533_icgf_guidance_joint_count": scalar(len(guidance_ids)),
        "p533_semantic_contact_joint_count": scalar(len(semantic_ids)),
        "p533_sdf_exempt_joint_count": scalar(len(exempt_ids)),
        "p533_icgf_query_batch_count": scalar(observed_query_shape[0]),
        "p533_icgf_query_frame_count": scalar(observed_query_shape[1]),
        "p533_icgf_query_joint_count": scalar(observed_query_shape[2]),
    }


def _validate_builder_receipt(
    value: Mapping[str, Any],
    *,
    guidance_ids: tuple[int, ...],
    semantic_contact_ids: tuple[int, ...],
    sdf_exempt_ids: tuple[int, ...],
    contact_frames: int,
) -> None:
    expected = constraint_count(
        guidance_ids,
        contact_frames=contact_frames,
        valid_future_frames=contact_frames,
    )
    _require(value.get("p533_builder_schema") == BUILDER_SCHEMA, "P533 builder proof missing")
    _require(
        tuple(value.get("actual_icgf_guidance_joint_ids", ())) == guidance_ids,
        "P533 builder did not consume the requested ICGF joints",
    )
    _require(
        tuple(value.get("semantic_contact_joint_ids", ())) == semantic_contact_ids,
        "P533 builder contact-role IDs drifted",
    )
    _require(
        tuple(value.get("actual_sdf_exempt_joint_ids", ())) == sdf_exempt_ids,
        "P533 builder SDF exemptions drifted",
    )
    _require(int(value.get("valid_sparse_support_count", -1)) == len(guidance_ids),
             "P533 sparse ICGF support mask is incomplete")
    _require(int(value.get("contact_frames", -1)) == int(contact_frames),
             "P533 builder contact-frame count drifted")
    _require(int(value.get("constraints_per_batch_denoise_call", -1)) == expected,
             "P533 builder constraint count drifted")


def validate_runtime_output(
    payload: Mapping[str, Any],
    *,
    dry_run: bool,
    energy_log_records: Sequence[Mapping[str, Any]] | None = None,
    guidance_joint_ids: Sequence[int],
    semantic_contact_joint_ids: Sequence[int],
    sdf_exempt_joint_ids: Sequence[int],
    contact_frames: int,
) -> dict[str, Any]:
    """Fail closed on either a constructed dry-run field or actual energy logs."""

    guidance_ids = _ids(guidance_joint_ids)
    semantic_ids = _ids(semantic_contact_joint_ids)
    exempt_ids = _ids(sdf_exempt_joint_ids)
    expected = constraint_count(
        guidance_ids,
        contact_frames=contact_frames,
        valid_future_frames=contact_frames,
    )
    contract = payload.get("icgf_contract", payload.get("p478_icgf_contract"))
    _require(isinstance(contract, Mapping), "runtime output lacks P478 ICGF contract")
    _require(tuple(contract.get("icgf_guidance_joint_ids", ())) == guidance_ids,
             "P478 ICGF contract guidance IDs drifted")
    _require(tuple(contract.get("contact_joint_ids", ())) == semantic_ids,
             "P478 semantic contact IDs drifted")
    _require(tuple(contract.get("sdf_exempt_joint_ids", ())) == exempt_ids,
             "P478 SDF exemption IDs drifted")

    if dry_run:
        field = payload.get("first_active_icgf")
        _require(isinstance(field, Mapping), "dry-run lacks the constructed ICGF field")
        _validate_builder_receipt(
            field,
            guidance_ids=guidance_ids,
            semantic_contact_ids=semantic_ids,
            sdf_exempt_ids=exempt_ids,
            contact_frames=contact_frames,
        )
        return {
            "schema": RUNTIME_SCHEMA,
            "status": "verified_dry_run_field_construction_gpu_not_started",
            "actual_model_generation": False,
            "implementation": IMPLEMENTATION,
            "actual_icgf_guidance_joint_ids": list(guidance_ids),
            "semantic_contact_joint_ids": list(semantic_ids),
            "actual_sdf_exempt_joint_ids": list(exempt_ids),
            "contact_frames": int(contact_frames),
            "constraints_per_batch_denoise_call": int(expected),
            "valid_sparse_support_count": len(guidance_ids),
            "energy_execution_observed": False,
            "legacy_p360_p478_source_modified": False,
        }

    records = payload.get("primitive_records")
    _require(isinstance(records, list), "generation metadata lacks primitive records")
    active: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        field = record.get("icgf_receipt")
        if not isinstance(field, Mapping) or field.get("p533_builder_schema") != BUILDER_SCHEMA:
            continue
        _validate_builder_receipt(
            field,
            guidance_ids=guidance_ids,
            semantic_contact_ids=semantic_ids,
            sdf_exempt_ids=exempt_ids,
            contact_frames=contact_frames,
        )
        _require(tuple(record.get("active_contact_joint_ids", ())) == semantic_ids,
                 "legacy transport no longer carries semantic contact IDs")
        summary = record.get("energy_summary")
        _require(isinstance(summary, Mapping), "active ICGF primitive lacks energy summary")
        for key in (
            "icgf_constraint_count_last",
            "icgf_constraint_count_mean",
            "p533_actual_icgf_constraint_count_last",
            "p533_actual_icgf_constraint_count_mean",
        ):
            value = float(summary.get(key, math.nan))
            _require(math.isfinite(value) and abs(value - expected) <= 1.0e-6,
                     f"runtime {key} is not the full22 constraint count")
        for key, expected_value in (
            ("p533_icgf_query_batch_count_last", 1.0),
            ("p533_icgf_query_frame_count_last", float(contact_frames)),
            ("p533_icgf_query_joint_count_last", float(len(guidance_ids))),
        ):
            value = float(summary.get(key, math.nan))
            _require(math.isfinite(value) and abs(value - expected_value) <= 1.0e-6,
                     f"runtime {key} does not match the actual full22 query")
        _require(int(summary.get("calls", 0)) > 0, "active ICGF energy was never called")
        e226 = record.get("e226_receipt")
        proof = e226.get("p533_full22_icgf_runtime") if isinstance(e226, Mapping) else None
        _require(isinstance(proof, Mapping), "condition receipt lacks P533 runtime proof")
        _require(tuple(proof.get("actual_icgf_guidance_joint_ids", ())) == guidance_ids,
                 "condition receipt guidance IDs drifted")
        _require(int(proof.get("constraints_per_batch_denoise_call", -1)) == expected,
                 "condition receipt constraint count drifted")
        active.append({
            "primitive_id": int(record.get("primitive_id", -1)),
            "denoise_calls": int(summary["calls"]),
            "constraints_per_call": int(expected),
        })
    _require(active, "generation never executed a P533 full22 ICGF primitive")
    _require(energy_log_records is not None, "generation runtime lacks energy_log.jsonl")
    observed_rows: list[Mapping[str, Any]] = []
    for row in energy_log_records:
        if not isinstance(row, Mapping):
            raise Full22ICGFAdapterError("energy log contains a malformed row")
        marker = float(row.get("p533_full22_icgf_adapter_active", 0.0))
        if abs(marker) <= 1.0e-6:
            continue
        _require(abs(marker - 1.0) <= 1.0e-6, "energy log has an invalid P533 marker")
        for key, expected_value in (
            ("icgf_constraint_count", float(expected)),
            ("p533_actual_icgf_constraint_count", float(expected)),
            ("p533_icgf_query_batch_count", 1.0),
            ("p533_icgf_query_frame_count", float(contact_frames)),
            ("p533_icgf_query_joint_count", float(len(guidance_ids))),
        ):
            value = float(row.get(key, math.nan))
            _require(math.isfinite(value) and abs(value - expected_value) <= 1.0e-6,
                     f"energy-log {key} does not prove full22 execution")
        observed_rows.append(row)
    expected_calls = sum(item["denoise_calls"] for item in active)
    _require(
        len(observed_rows) == expected_calls and expected_calls > 0,
        "full22 energy-log rows do not match active primitive DDPM calls",
    )
    return {
        "schema": RUNTIME_SCHEMA,
        "status": "verified_actual_per_denoise_full22_icgf_execution",
        "actual_model_generation": True,
        "implementation": IMPLEMENTATION,
        "actual_icgf_guidance_joint_ids": list(guidance_ids),
        "semantic_contact_joint_ids": list(semantic_ids),
        "actual_sdf_exempt_joint_ids": list(exempt_ids),
        "contact_frames": int(contact_frames),
        "constraints_per_batch_denoise_call": int(expected),
        "valid_sparse_support_count": len(guidance_ids),
        "energy_execution_observed": True,
        "observed_icgf_query_shape": [1, int(contact_frames), len(guidance_ids), 3],
        "active_primitives": active,
        "terminal_denoise_call_count": int(len(observed_rows)),
        "all_terminal_denoise_calls_exact": True,
        "legacy_p360_p478_source_modified": False,
    }


@contextmanager
def expose_full22_sparse_icgf(
    p478: ModuleType,
    *,
    guidance_joint_ids: Sequence[int],
    semantic_contact_joint_ids: Sequence[int],
    sdf_exempt_joint_ids: Sequence[int],
    contact_frames: int,
) -> Iterator[dict[str, Any]]:
    """Temporarily decouple legacy P360's ICGF and SDF/contact joint sets."""

    guidance_ids = _ids(guidance_joint_ids)
    semantic_ids = _ids(semantic_contact_joint_ids)
    exempt_ids = _ids(sdf_exempt_joint_ids)
    _require(guidance_ids == tuple(range(22)), "P533 requires exact ordered full22 guidance")
    _require(exempt_ids == semantic_ids, "P533 SDF exemptions must equal contact semantics")
    _require(set(exempt_ids).issubset(set(guidance_ids)), "contact IDs not in guidance set")
    _require(int(contact_frames) == 4, "P533 full22 ICGF contract requires four tail frames")

    p360 = p478.p360
    original_builder = p360.build_active_icgf
    original_condition_class = p478.ReMoGenE226CondFn
    legacy_base_class = p360.ReMoGenRawSceneCondFn
    original_energy = legacy_base_class.energy_from_world_joints
    source_path = Path(inspect.getsourcefile(original_builder) or "").resolve()
    _require(
        source_path == Path(p360.__file__).resolve(),
        "P533 adapter was pointed at an unexpected P360 implementation",
    )

    def build_active_icgf(
        args: Any,
        node: Any,
        raw_scene: Any,
        scene_sdf: Any | None = None,
    ) -> tuple[Any, tuple[int, ...], dict[str, Any]]:
        del raw_scene, scene_sdf
        source = str(args.icgf_source)
        if source == "auto":
            source = "sparse_joint_targets" if bool(node.joint_mask.any()) else "none"
        _require(source == "sparse_joint_targets",
                 "P533 full22 adapter accepts sparse_joint_targets only")
        supplied_guidance = tuple(p360._parse_joint_ids(args.icgf_guidance_joints))
        supplied_exemption = tuple(p360._parse_joint_ids(args.sdf_exempt_joints))
        supplied_semantic = tuple(p360._contact_joint_ids(args, node))
        _require(supplied_guidance == guidance_ids, "builder guidance IDs differ from launch")
        _require(supplied_exemption == exempt_ids, "builder SDF exemptions differ from launch")
        _require(supplied_semantic == semantic_ids, "builder semantic contacts differ from packet")
        _require(int(args.contact_frames) == int(contact_frames),
                 "builder contact_frames differs from launch")
        declared = tuple(int(value) for value in np.flatnonzero(node.joint_mask))
        _require(declared == guidance_ids, "active packet node is not ordered full22")
        targets = np.asarray(node.joint_xyz_world, dtype=np.float32).copy()
        _require(targets.shape == (22, 3) and bool(np.isfinite(targets).all()),
                 "active packet sparse targets are not finite J22 world points")
        target_mask = np.zeros(22, dtype=bool)
        target_mask[np.asarray(guidance_ids, dtype=np.int64)] = True
        icgf = p360.QueryTimeICGF.from_sparse_joint_targets(
            targets,
            target_mask,
            kernel_sigma_m=args.icgf_kernel_sigma,
            source="p533_full22_predicted_packet_sparse_joint_targets",
        )
        count = constraint_count(
            guidance_ids,
            contact_frames=contact_frames,
            valid_future_frames=contact_frames,
        )
        receipt = {
            **icgf.receipt,
            "p533_builder_schema": BUILDER_SCHEMA,
            "resolved_source": source,
            "actual_icgf_guidance_joint_ids": list(guidance_ids),
            "semantic_contact_joint_ids": list(semantic_ids),
            "contact_joint_ids": list(semantic_ids),
            "actual_sdf_exempt_joint_ids": list(exempt_ids),
            "valid_sparse_support_count": int(target_mask.sum()),
            "contact_frames": int(contact_frames),
            "constraints_per_batch_denoise_call": int(count),
            "active_external_ordinal": int(node.ordinal),
            "active_node_policy": str(args.icgf_active_node_policy),
            "guidance_target_source": "active_predicted_packet_sparse_joint_xyz_world",
            "sdf_exemption_source": "active_predicted_packet_contact_role",
            "legacy_transport_return_joint_ids": list(semantic_ids),
            "legacy_p360_builder_bypassed_in_memory": True,
            "legacy_p360_p478_source_modified": False,
            "retrieval_used": False,
            "future_motion_read": False,
        }
        # P360's second return value remains semantic contact3.  The patched
        # condition class below obtains full22 from the sealed closure.
        return icgf, semantic_ids, receipt

    class P533Full22ICGFCondFn(original_condition_class):
        def __init__(self, **kwargs: Any) -> None:
            active = kwargs.get("icgf") is not None
            incoming = _ids(kwargs.get("contact_joint_ids"))
            _require(
                incoming == (semantic_ids if active else ()),
                "legacy condition transport contact IDs drifted",
            )
            super().__init__(**kwargs)
            self.icgf_guidance_joint_ids = guidance_ids if active else ()
            self.sdf_exempt_joint_ids = exempt_ids if active else ()
            self.p533_expected_icgf_guidance_joint_ids = guidance_ids if active else ()
            self.p533_full22_icgf_active = bool(active)

        @property
        def e226_receipt(self) -> dict[str, Any]:
            receipt = dict(super().e226_receipt)
            active = bool(self.p533_full22_icgf_active)
            count = (
                constraint_count(
                    self.icgf_guidance_joint_ids,
                    contact_frames=int(self.config.contact_frames),
                    valid_future_frames=int(self.valid_future_frames),
                )
                if active
                else 0
            )
            receipt["p533_full22_icgf_runtime"] = {
                "schema": RUNTIME_SCHEMA,
                "implementation": IMPLEMENTATION,
                "active": active,
                "actual_icgf_guidance_joint_ids": list(self.icgf_guidance_joint_ids),
                "semantic_contact_joint_ids": list(self.contact_joint_ids),
                "actual_sdf_exempt_joint_ids": list(self.sdf_exempt_joint_ids),
                "contact_frames": int(self.config.contact_frames),
                "valid_future_frames": int(self.valid_future_frames),
                "constraints_per_batch_denoise_call": int(count),
                "legacy_p360_p478_source_modified": False,
            }
            return receipt

    P533Full22ICGFCondFn.__name__ = "P533Full22SparseICGFCondFn"
    audit = {
        "schema": RUNTIME_SCHEMA,
        "implementation": IMPLEMENTATION,
        "legacy_builder_source": str(source_path),
        "legacy_builder_replaced_in_process": True,
        "legacy_energy_replaced_in_process": True,
        "legacy_p360_p478_source_modified": False,
        "actual_icgf_guidance_joint_ids": list(guidance_ids),
        "semantic_contact_joint_ids": list(semantic_ids),
        "actual_sdf_exempt_joint_ids": list(exempt_ids),
        "contact_frames": int(contact_frames),
        "constraints_per_batch_denoise_call": constraint_count(
            guidance_ids,
            contact_frames=contact_frames,
            valid_future_frames=contact_frames,
        ),
    }
    try:
        p360.build_active_icgf = build_active_icgf
        legacy_base_class.energy_from_world_joints = _decoupled_energy_from_world_joints
        p478.ReMoGenE226CondFn = P533Full22ICGFCondFn
        yield audit
    finally:
        p478.ReMoGenE226CondFn = original_condition_class
        legacy_base_class.energy_from_world_joints = original_energy
        p360.build_active_icgf = original_builder


__all__ = [
    "BUILDER_SCHEMA",
    "Full22ICGFAdapterError",
    "IMPLEMENTATION",
    "RUNTIME_SCHEMA",
    "constraint_count",
    "expose_full22_sparse_icgf",
    "validate_runtime_output",
]

