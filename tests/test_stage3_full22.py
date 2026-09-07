"""Retained numerical regressions against integrated modules."""

from __future__ import annotations

from hsi.stage3 import full22

from pathlib import Path

from types import SimpleNamespace

import numpy as np

import pytest

import torch

GUIDANCE = tuple(range(22))

CONTACT = (0, 1, 2)

class _SceneSample:
    def __init__(self, points: torch.Tensor) -> None:
        shape = points.shape[:-1]
        self.signed_distance_m = torch.ones(
            shape, device=points.device, dtype=points.dtype
        )
        self.in_bounds = torch.ones(shape, device=points.device, dtype=torch.bool)

class _Scene:
    def sample(self, points: torch.Tensor) -> _SceneSample:
        return _SceneSample(points)

class _ICGF:
    def __init__(self) -> None:
        self.queried_joint_ids: tuple[int, ...] | None = None
        self.query_shape: tuple[int, ...] | None = None

    def soft_distance(
        self, query: torch.Tensor, *, joint_ids: torch.Tensor
    ) -> torch.Tensor:
        self.queried_joint_ids = tuple(int(value) for value in joint_ids.tolist())
        self.query_shape = tuple(int(value) for value in query.shape)
        return torch.linalg.vector_norm(query, dim=-1)

def _energy_host() -> SimpleNamespace:
    icgf = _ICGF()
    config = SimpleNamespace(
        exempt_contact_joints_from_sdf_tail=True,
        contact_frames=4,
        collision_clearance_m=0.01,
        icgf_deadzone_m=0.03,
        icgf_huber_beta_m=0.04,
        sdf_weight=1.0,
        icgf_weight=0.4,
    )
    return SimpleNamespace(
        valid_future_frames=8,
        collision_joint_ids=GUIDANCE,
        scene_sdf=_Scene(),
        config=config,
        icgf=icgf,
        contact_joint_ids=CONTACT,
        sdf_exempt_joint_ids=CONTACT,
        icgf_guidance_joint_ids=GUIDANCE,
        p533_expected_icgf_guidance_joint_ids=GUIDANCE,
    )

def test_exact_full22_four_frame_energy_consumes_88_constraints() -> None:
    host = _energy_host()
    joints = torch.zeros((1, 8, 22, 3), dtype=torch.float32, requires_grad=True)
    total, diagnostics = full22._decoupled_energy_from_world_joints(host, joints)
    assert host.icgf.queried_joint_ids == GUIDANCE
    assert host.icgf.query_shape == (1, 4, 22, 3)
    assert int(diagnostics["icgf_constraint_count"].item()) == 88
    assert int(diagnostics["p533_actual_icgf_constraint_count"].item()) == 88
    # 8*22 SDF rows minus contact3 exemptions over the terminal four frames.
    assert int(diagnostics["sdf_constraint_count"].item()) == 164
    total.backward()
    assert joints.grad is not None

class _SelectiveScene:
    def __init__(self, penetrating_joint: int) -> None:
        self.penetrating_joint = int(penetrating_joint)

    def sample(self, points: torch.Tensor) -> _SceneSample:
        sample = _SceneSample(points)
        joint = points[..., 0].round().to(torch.long) == self.penetrating_joint
        terminal_tail = points[..., 2] >= 4.0
        sample.signed_distance_m = torch.where(
            joint & terminal_tail,
            torch.full_like(sample.signed_distance_m, -0.1),
            sample.signed_distance_m,
        )
        return sample

@pytest.mark.parametrize(
    ("joint_id", "expected_active"),
    ((0, 0), (3, 4)),
)
def test_sdf_exemption_is_numerically_exact_contact3(
    joint_id: int, expected_active: int
) -> None:
    host = _energy_host()
    host.scene_sdf = _SelectiveScene(joint_id)
    joints = torch.zeros((1, 8, 22, 3), dtype=torch.float32)
    joints[..., 0] = torch.arange(22, dtype=torch.float32)[None, None]
    joints[..., 2] = torch.arange(8, dtype=torch.float32)[None, :, None]
    _, diagnostics = full22._decoupled_energy_from_world_joints(host, joints)
    assert int(diagnostics["sdf_active_constraint_count"].item()) == expected_active

def test_energy_fails_closed_if_runtime_guidance_shrinks_to_contact3() -> None:
    host = _energy_host()
    host.icgf_guidance_joint_ids = CONTACT
    joints = torch.zeros((1, 8, 22, 3), dtype=torch.float32)
    with pytest.raises(full22.Full22ICGFAdapterError, match="guidance IDs drifted"):
        full22._decoupled_energy_from_world_joints(host, joints)

def test_energy_fails_closed_if_sdf_exemption_expands_to_full22() -> None:
    host = _energy_host()
    host.sdf_exempt_joint_ids = GUIDANCE
    joints = torch.zeros((1, 8, 22, 3), dtype=torch.float32)
    with pytest.raises(full22.Full22ICGFAdapterError, match="must equal semantic"):
        full22._decoupled_energy_from_world_joints(host, joints)

def test_process_local_builder_constructs_full22_mask_and_restores_legacy() -> None:
    from hsi.stage3 import executor as runner
    original_builder = runner.p360.build_active_icgf
    original_energy = runner.p360.ReMoGenRawSceneCondFn.energy_from_world_joints
    original_condition = runner.ReMoGenE226CondFn
    args = SimpleNamespace(
        icgf_source="sparse_joint_targets",
        icgf_guidance_joints=",".join(map(str, GUIDANCE)),
        sdf_exempt_joints="0,1,2",
        icgf_contact_joints="0,1,2",
        contact_frames=4,
        icgf_kernel_sigma=0.08,
        icgf_active_node_policy="terminal",
    )
    node = SimpleNamespace(
        ordinal=4,
        joint_mask=np.ones(22, dtype=bool),
        joint_xyz_world=np.arange(66, dtype=np.float32).reshape(22, 3) / 100.0,
    )
    with full22.expose_full22_sparse_icgf(
        runner,
        guidance_joint_ids=GUIDANCE,
        semantic_contact_joint_ids=CONTACT,
        sdf_exempt_joint_ids=CONTACT,
        contact_frames=4,
    ):
        field, transport_ids, receipt = runner.p360.build_active_icgf(
            args, node, None, None
        )
        assert transport_ids == CONTACT
        assert tuple(receipt["actual_icgf_guidance_joint_ids"]) == GUIDANCE
        assert receipt["valid_sparse_support_count"] == 22
        assert receipt["constraints_per_batch_denoise_call"] == 88
        assert int(field.valid_support.sum().item()) == 22
        assert runner.p360.build_active_icgf is not original_builder
        assert runner.p360.ReMoGenRawSceneCondFn.energy_from_world_joints is not original_energy
        assert runner.ReMoGenE226CondFn is not original_condition
    assert runner.p360.build_active_icgf is original_builder
    assert runner.p360.ReMoGenRawSceneCondFn.energy_from_world_joints is original_energy
    assert runner.ReMoGenE226CondFn is original_condition

def test_constraint_arithmetic_is_22_times_4() -> None:
    assert full22.constraint_count(
        GUIDANCE, contact_frames=4, valid_future_frames=8, batch_size=1
    ) == 88

def _runtime_payload(count: float = 88.0) -> tuple[dict, list[dict]]:
    contract = {
        "icgf_guidance_joint_ids": list(GUIDANCE),
        "contact_joint_ids": list(CONTACT),
        "sdf_exempt_joint_ids": list(CONTACT),
    }
    field = {
        "p533_builder_schema": full22.BUILDER_SCHEMA,
        "actual_icgf_guidance_joint_ids": list(GUIDANCE),
        "semantic_contact_joint_ids": list(CONTACT),
        "actual_sdf_exempt_joint_ids": list(CONTACT),
        "valid_sparse_support_count": 22,
        "contact_frames": 4,
        "constraints_per_batch_denoise_call": 88,
    }
    condition = {
        "p533_full22_icgf_runtime": {
            "actual_icgf_guidance_joint_ids": list(GUIDANCE),
            "constraints_per_batch_denoise_call": 88,
        }
    }
    summary = {
        "calls": 1,
        "icgf_constraint_count_last": count,
        "icgf_constraint_count_mean": count,
        "p533_actual_icgf_constraint_count_last": count,
        "p533_actual_icgf_constraint_count_mean": count,
        "p533_icgf_query_batch_count_last": 1.0,
        "p533_icgf_query_frame_count_last": 4.0,
        "p533_icgf_query_joint_count_last": 22.0,
    }
    payload = {
        "p478_icgf_contract": contract,
        "primitive_records": [{
            "primitive_id": 7,
            "active_contact_joint_ids": list(CONTACT),
            "icgf_receipt": field,
            "energy_summary": summary,
            "e226_receipt": condition,
        }],
    }
    row = {
        "p533_full22_icgf_adapter_active": 1.0,
        "icgf_constraint_count": count,
        "p533_actual_icgf_constraint_count": count,
        "p533_icgf_query_batch_count": 1.0,
        "p533_icgf_query_frame_count": 4.0,
        "p533_icgf_query_joint_count": 22.0,
    }
    return payload, [row]

def test_runtime_attestation_checks_every_energy_row() -> None:
    payload, rows = _runtime_payload()
    proof = full22.validate_runtime_output(
        payload,
        dry_run=False,
        energy_log_records=rows,
        guidance_joint_ids=GUIDANCE,
        semantic_contact_joint_ids=CONTACT,
        sdf_exempt_joint_ids=CONTACT,
        contact_frames=4,
    )
    assert proof["constraints_per_batch_denoise_call"] == 88
    assert proof["terminal_denoise_call_count"] == 1
    assert proof["all_terminal_denoise_calls_exact"] is True

def test_runtime_attestation_rejects_legacy_twelve_constraint_path() -> None:
    payload, rows = _runtime_payload(12.0)
    with pytest.raises(full22.Full22ICGFAdapterError, match="full22 constraint count"):
        full22.validate_runtime_output(
            payload,
            dry_run=False,
            energy_log_records=rows,
            guidance_joint_ids=GUIDANCE,
            semantic_contact_joint_ids=CONTACT,
            sdf_exempt_joint_ids=CONTACT,
            contact_frames=4,
        )

