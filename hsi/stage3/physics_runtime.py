"""Current clean-xstart physics integration, no historical loaders."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Iterator, Sequence
from hsi.common.artifacts import require, MemoryContractError as ContractError
from hsi.stage3.physics import (CurrentPhysicsGuidanceError, CurrentStepwisePhysicsConfig,
    CurrentStepwisePhysicsEnergy, POLICY as CURRENT_PHYSICS_POLICY)

ROLE_AWARE_FULL22_GUIDANCE_IDS = tuple(range(22))
PHYSICS_RECEIPT_SCHEMA = "p533.current_stepwise_physics_guidance.v1"

@contextmanager
def expose_decoupled_joint_fields(p478: ModuleType) -> Iterator[None]:
    """Expose the two audited P478 fields without changing its public parser."""

    original = p478.p360.parse_args

    def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--icgf-guidance-joints", required=True)
        parser.add_argument("--sdf-exempt-joints", required=True)
        own, remaining = parser.parse_known_args(argv)
        parsed = original(remaining)
        parsed.icgf_guidance_joints = own.icgf_guidance_joints
        parsed.sdf_exempt_joints = own.sdf_exempt_joints
        return parsed

    p478.p360.parse_args = parse_args
    try:
        yield
    finally:
        p478.p360.parse_args = original


@contextmanager
def expose_adaptive_guidance_policy(p478: ModuleType) -> Iterator[None]:
    """Run P478's original full-packet check, then restore the audited subset."""

    original = p478._bind_packet_contact_ids

    def bind(args: argparse.Namespace, plan: Any) -> tuple[int, ...]:
        supplied_text = getattr(args, "icgf_guidance_joints", None)
        supplied = (
            p478.p360._parse_joint_ids(supplied_text)
            if supplied_text is not None else ()
        )
        # The role-aware P523 policy now supplies full J22, so frozen P478
        # executes its original exact-packet gate directly.  Retain the
        # context manager for compatibility with already imported callers,
        # but never rewrite the P523 full22 request.
        if tuple(supplied) != ROLE_AWARE_FULL22_GUIDANCE_IDS:
            return original(args, plan)
        return original(args, plan)

    p478._bind_packet_contact_ids = bind
    try:
        yield
    finally:
        p478._bind_packet_contact_ids = original

def _parse_physics_options(
    argv: Sequence[str],
) -> tuple[CurrentStepwisePhysicsConfig | None, list[str]]:
    names = (
        "--p533-current-physics-policy",
        "--p533-physics-slip-weight",
        "--p533-physics-contact-band-weight",
        "--p533-physics-floor-weight",
        "--p533-physics-proxy-collision-weight",
        "--p533-physics-root-speed-release-mps",
    )
    for name in names:
        require(list(argv).count(name) <= 1, f"{name} may appear at most once")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--p533-current-physics-policy")
    parser.add_argument("--p533-physics-slip-weight", type=float, default=4.0)
    parser.add_argument("--p533-physics-contact-band-weight", type=float, default=1.0)
    parser.add_argument("--p533-physics-floor-weight", type=float, default=8.0)
    parser.add_argument("--p533-physics-proxy-collision-weight", type=float, default=2.0)
    parser.add_argument("--p533-physics-root-speed-release-mps", type=float, default=0.85)
    own, remaining = parser.parse_known_args(list(argv))
    if own.p533_current_physics_policy is None:
        require(not any(name in argv for name in names[1:]),
                "P533 physics weights require --p533-current-physics-policy")
        return None, list(remaining)
    require(own.p533_current_physics_policy == CURRENT_PHYSICS_POLICY,
            "unsupported P533 current physics policy")
    try:
        config = CurrentStepwisePhysicsConfig(
            policy=own.p533_current_physics_policy,
            slip_weight=own.p533_physics_slip_weight,
            contact_band_weight=own.p533_physics_contact_band_weight,
            floor_penetration_weight=own.p533_physics_floor_weight,
            lower_body_collision_weight=own.p533_physics_proxy_collision_weight,
            root_speed_release_mps=own.p533_physics_root_speed_release_mps,
        ).validate()
    except CurrentPhysicsGuidanceError as error:
        raise ContractError(str(error)) from error
    return config, list(remaining)


def _public_physics_receipt(
    config: CurrentStepwisePhysicsConfig, *, ground_z_m: float | None = None
) -> dict[str, Any]:
    receipt = dict(config.receipt(ground_z_m=ground_z_m))
    receipt["schema"] = PHYSICS_RECEIPT_SCHEMA
    receipt["implementation_reused_without_parameter_change"] = True
    receipt["provider_or_stage2_identity_inferred_by_numeric_energy"] = False
    return receipt


@contextmanager
def expose_p533_current_stepwise_physics_guidance(
    p478: ModuleType, config: CurrentStepwisePhysicsConfig | None
) -> Iterator[None]:
    if config is None:
        yield
        return
    config.validate()
    original = p478.ReMoGenE226CondFn

    class P533CurrentPhysicsCondFn(original):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.p533_current_physics = CurrentStepwisePhysicsEnergy(
                self.scene_sdf, self.e226_bridge.history_world_joints_zup, config
            )

        @property
        def e226_receipt(self) -> dict[str, Any]:
            receipt = dict(super().e226_receipt)
            receipt["p533_current_stepwise_physics_guidance"] = _public_physics_receipt(
                config, ground_z_m=self.p533_current_physics.ground_z_m
            )
            return receipt

        def energy_from_world_joints(self, joints_world_zup: Any):
            parent_total, parent_diagnostics = super().energy_from_world_joints(joints_world_zup)
            physics_total, raw_diagnostics = self.p533_current_physics.energy(
                joints_world_zup, valid_future_frames=self.valid_future_frames
            )
            physics_diagnostics = {
                ("p533_" + key[len("p523_"):]) if key.startswith("p523_") else key: value
                for key, value in raw_diagnostics.items()
            }
            overlap = sorted(set(parent_diagnostics) & set(physics_diagnostics))
            if overlap:
                raise CurrentPhysicsGuidanceError(
                    "P533 physics diagnostics overwrite parent fields: " + ", ".join(overlap)
                )
            total = parent_total + physics_total
            diagnostics = dict(parent_diagnostics)
            diagnostics.update(physics_diagnostics)
            diagnostics["p533_combined_parent_and_physics_energy"] = total
            return total, diagnostics

    P533CurrentPhysicsCondFn.__name__ = "P533CurrentStepwisePhysicsCondFn"
    p478.ReMoGenE226CondFn = P533CurrentPhysicsCondFn
    try:
        yield
    finally:
        p478.ReMoGenE226CondFn = original

