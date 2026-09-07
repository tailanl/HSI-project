#!/usr/bin/env python3
"""Run the frozen P360 executor with the causal P478 M1--M5 protocol.

This file is deliberately a *wrapper*, not a fork of P360's rollout.  During
one call to :func:`p360.rollout` it temporarily replaces three narrow runtime
interfaces:

* the ablation table (M1--M5);
* the raw-scene condition-function constructor (M3--M5 only);
* the ordered scheduler (to capture emitted J22 and, for M5, request a newly
  sampled feedback primitive).

The frozen P360 source remains read-only.  The rolling E2.2.6 energy receives
the two native observed frames for primitive zero and the exact, detached
world-J22 prefix emitted by earlier primitives thereafter.  It never receives
future or ground-truth frames.  M4/M5 ICGF is accepted only when it is built
from sparse contact joints in a trusted externally predicted keypose packet.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch














from hsi.stage3.model_runtime import current as current_motion_runtime
DEFAULT_BRIDGE_CONFIG = Path(__file__).with_name("causal_bridge_config.json")
DEFAULT_P360_ADAPTER = None

def _load_memory_profile(session_path, profile_path):
    if session_path is not None or profile_path is not None:
        raise ValueError("Historical coefficient profiles are disabled; use current typed upstream Memory")
    return None, None

from hsi.stage3 import rollout_core as p360
from hsi.stage3.bridge_contract import (  # noqa: E402
    EXPECTED_ABLATIONS,
    load_contract,
)
from hsi.stage3.causal_bridge import (  # noqa: E402
    P360E226BridgeConfig,
    P360E226BridgeError,
    ReMoGenE226CondFn,
)

from hsi.stage3.static_seed import (  # noqa: E402
    HISTORY_SOURCE as QUERY_STATIC_HISTORY_SOURCE,
    QueryStaticSeed,
    generated_only_sequence,
)


TRUSTED_PREDICTED_PROVENANCE = frozenset(
    (
        "generated_image_alphapose",
        "external_2d_observations",
        "qwen3_planner_qwen_image_shared_smplx",
    )
)
PREDICTED_PACKET_ICGF_SOURCES = frozenset(
    ("sparse_joint_targets", "surface_projected_sparse_joint_targets")
)
TERMINAL_E226_MODES = ("inherit", "stop_terms_off", "off")
TERMINAL_E226_RECEIPT_SCHEMA = "p478.terminal_e226_release_receipt.v1"


class P478HSIContractError(ValueError):
    """A requested rollout violates the P478 causal experiment contract."""





@dataclass(frozen=True)
class FeedbackConfig:
    hold_frames: int = 3
    maximum_extra_primitives: int = 1
    threshold_scale: float = 1.0

    def validate(self) -> "FeedbackConfig":
        if self.hold_frames < 1 or self.maximum_extra_primitives < 0:
            raise P478HSIContractError(
                "feedback hold frames must be positive and extra primitive count nonnegative"
            )
        if not math.isfinite(self.threshold_scale) or self.threshold_scale <= 0.0:
            raise P478HSIContractError(
                "feedback threshold scale must be finite and positive"
            )
        return self


ROLEWISE_SUPPORT_CHAIN_IDS = frozenset((1, 2, 4, 5, 7, 8, 10, 11))
ROLEWISE_AXIAL_POSTURE_IDS = frozenset((0, 3, 6, 9, 12, 15))
ROLEWISE_UPPER_LIMB_IDS = frozenset((13, 14, 16, 17, 18, 19, 20, 21))


@dataclass(frozen=True)
class RolewiseTerminalFeedbackConfig:
    """Opt-in, causal completion gate for a sparse terminal body node.

    This is intentionally separate from the legacy M5 mean-error hold.  The
    default is disabled, so M1--M5 retain their existing scheduler and replay
    behavior.  When enabled, every declared sparse joint must remain inside a
    fixed anatomical-role threshold for the tail hold before terminal
    completion is accepted.
    """

    enabled: bool = False
    hold_frames: int = 3
    maximum_extra_primitives: int = 3
    contact_max_m: float = 0.15
    support_chain_max_m: float = 0.25
    axial_posture_max_m: float = 0.20
    upper_limb_max_m: float = 0.25
    other_sparse_max_m: float = 0.25
    global_per_joint_max_m: float = 0.30

    def validate(self) -> "RolewiseTerminalFeedbackConfig":
        if self.hold_frames < 1 or self.maximum_extra_primitives < 0:
            raise P478HSIContractError(
                "role-wise terminal hold frames must be positive and extra "
                "primitive count nonnegative"
            )
        values = (
            self.contact_max_m,
            self.support_chain_max_m,
            self.axial_posture_max_m,
            self.upper_limb_max_m,
            self.other_sparse_max_m,
            self.global_per_joint_max_m,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise P478HSIContractError(
                "role-wise terminal thresholds must be finite and positive"
            )
        return self

    @property
    def role_thresholds_m(self) -> Dict[str, float]:
        return {
            "declared_contact": float(self.contact_max_m),
            "support_chain": float(self.support_chain_max_m),
            "axial_posture": float(self.axial_posture_max_m),
            "upper_limb": float(self.upper_limb_max_m),
            "other_sparse": float(self.other_sparse_max_m),
        }


P478_ABLATIONS: Dict[str, p360.AblationSpec] = {
    "M1_KEYPOSE": p360.AblationSpec("M1_KEYPOSE", True, False, False),
    "M2_KEYPOSE_SDF": p360.AblationSpec("M2_KEYPOSE_SDF", True, True, False),
    "M3_KEYPOSE_SDF_E226": p360.AblationSpec(
        "M3_KEYPOSE_SDF_E226", True, True, False
    ),
    "M4_KEYPOSE_SDF_E226_ICGF": p360.AblationSpec(
        "M4_KEYPOSE_SDF_E226_ICGF", True, True, True
    ),
    "M5_M4_FEEDBACK_HOLD": p360.AblationSpec(
        "M5_M4_FEEDBACK_HOLD", True, True, True
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _trusted_predicted_plan(plan: Any) -> bool:
    return str(plan.provenance).strip().lower() in TRUSTED_PREDICTED_PROVENANCE


def _packet_contact_joint_ids(path: Path, *, terminal_ordinal: int) -> tuple[int, ...]:
    """Read explicit contact semantics from the predicted packet itself.

    P142's legacy interchange loader retains sparse joints but does not retain
    their ``contact_role`` bit.  M4/M5 therefore audit the original packet in
    parallel and refuse to reinterpret every active/keypose joint as contact.
    """

    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise P478HSIContractError("predicted JSON packet needs a nodes list")
        selected = next(
            (
                node
                for index, node in enumerate(nodes)
                if int(node.get("ordinal", index)) == int(terminal_ordinal)
            ),
            None,
        )
        if not isinstance(selected, Mapping):
            raise P478HSIContractError(
                "predicted packet does not contain terminal ordinal {}".format(
                    terminal_ordinal
                )
            )
        if "contact_joint_ids" in selected:
            values = selected["contact_joint_ids"]
        elif "joint_contact_mask" in selected:
            mask = np.asarray(selected["joint_contact_mask"], dtype=bool)
            if mask.shape != (22,):
                raise P478HSIContractError("joint_contact_mask must have shape [22]")
            values = np.flatnonzero(mask).tolist()
        else:
            joints = selected.get("joints")
            if not isinstance(joints, list) or not all(
                isinstance(item, Mapping) for item in joints
            ):
                raise P478HSIContractError(
                    "M4/M5 packet must declare contact_joint_ids, "
                    "joint_contact_mask, or per-joint contact_role"
                )
            if not all("contact_role" in item for item in joints):
                raise P478HSIContractError(
                    "every sparse joint needs an explicit contact_role for M4/M5"
                )
            values = [int(item["id"]) for item in joints if bool(item["contact_role"])]
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "contact_joint_mask" in payload.files:
                masks = np.asarray(payload["contact_joint_mask"], dtype=bool)
                mask = masks[-1] if masks.ndim == 2 else masks
                if mask.shape != (22,):
                    raise P478HSIContractError(
                        "contact_joint_mask must be [22] or [N,22]"
                    )
                values = np.flatnonzero(mask).tolist()
            elif "contact_joint_ids" in payload.files:
                raw = np.asarray(payload["contact_joint_ids"])
                values = raw[-1].tolist() if raw.ndim == 2 else raw.tolist()
            else:
                raise P478HSIContractError(
                    "M4/M5 NPZ packet needs contact_joint_mask/contact_joint_ids"
                )
    else:
        raise P478HSIContractError("predicted packet must be JSON or NPZ")
    result = tuple(int(value) for value in values)
    if not result:
        raise P478HSIContractError(
            "M4/M5 terminal predicted packet declares no contact joints"
        )
    if len(set(result)) != len(result) or any(value < 0 or value >= 22 for value in result):
        raise P478HSIContractError(
            "predicted contact joint IDs must be unique integers in [0,21]"
        )
    return result


def _bind_packet_contact_ids(args: argparse.Namespace, plan: Any) -> tuple[int, ...]:
    if str(args.ablation) not in (
        "M4_KEYPOSE_SDF_E226_ICGF",
        "M5_M4_FEEDBACK_HOLD",
    ):
        return ()
    if str(args.icgf_active_node_policy) != "terminal":
        raise P478HSIContractError(
            "M4/M5 require terminal-only ICGF because one contact set belongs to "
            "the terminal predicted packet"
        )
    predicted = _packet_contact_joint_ids(
        Path(args.plan), terminal_ordinal=int(plan.nodes[-1].ordinal)
    )
    supplied = p360._parse_joint_ids(args.icgf_contact_joints)
    if supplied and supplied != predicted:
        raise P478HSIContractError(
            "--icgf-contact-joints {} does not exactly match packet-declared {}"
            .format(list(supplied), list(predicted))
        )
    args.icgf_contact_joints = ",".join(str(value) for value in predicted)
    guidance_arg = getattr(args, "icgf_guidance_joints", None)
    exemption_arg = getattr(args, "sdf_exempt_joints", None)
    if (guidance_arg is None) != (exemption_arg is None):
        raise P478HSIContractError(
            "decoupled field requires both --icgf-guidance-joints and --sdf-exempt-joints"
        )
    if guidance_arg is not None:
        packet_active = tuple(
            int(value) for value in np.flatnonzero(plan.nodes[-1].joint_mask)
        )
        supplied_guidance = p360._parse_joint_ids(guidance_arg)
        supplied_exemption = p360._parse_joint_ids(exemption_arg)
        if supplied_guidance != packet_active:
            raise P478HSIContractError(
                "ICGF guidance IDs {} do not exactly match packet active sparse IDs {}"
                .format(list(supplied_guidance), list(packet_active))
            )
        if supplied_exemption != predicted:
            raise P478HSIContractError(
                "SDF exemption IDs {} do not exactly match packet contact-role IDs {}"
                .format(list(supplied_exemption), list(predicted))
            )
        args.icgf_guidance_joints = ",".join(str(value) for value in packet_active)
        args.sdf_exempt_joints = ",".join(str(value) for value in predicted)
    return predicted


def _strict_icgf_contract(args: argparse.Namespace, plan: Any) -> Dict[str, Any]:
    enabled = str(args.ablation) in (
        "M4_KEYPOSE_SDF_E226_ICGF",
        "M5_M4_FEEDBACK_HOLD",
    )
    receipt: Dict[str, Any] = {
        "enabled": enabled,
        "schema": "p478.predicted_packet_icgf_contract.v1",
        "builder": "P360.build_active_icgf",
        "source": str(args.icgf_source),
        "plan_provenance": str(plan.provenance),
        "trusted_predicted_plan": _trusted_predicted_plan(plan),
        "contact_target_source": (
            "predicted_keypose_packet_joint_xyz_world" if enabled else None
        ),
        "ground_truth_contact_read": False,
        "ground_truth_icgf_read": False,
        "future_motion_read": False,
        "memory_or_retrieval_read": False,
    }
    if not enabled:
        return receipt
    if bool(args.allow_oracle_plan) or p360.p142._is_oracle_provenance(
        plan.provenance
    ):
        raise P478HSIContractError(
            "M4/M5 forbid oracle/GT plans because their ICGF must come from a "
            "predicted keypose packet"
        )
    if not _trusted_predicted_plan(plan):
        raise P478HSIContractError(
            "M4/M5 require trusted predicted plan provenance {}; got {!r}".format(
                sorted(TRUSTED_PREDICTED_PROVENANCE), plan.provenance
            )
        )
    if str(args.icgf_source) not in PREDICTED_PACKET_ICGF_SOURCES:
        raise P478HSIContractError(
            "M4/M5 ICGF source must be one of {}; target/component/GT fields are "
            "forbidden".format(sorted(PREDICTED_PACKET_ICGF_SOURCES))
        )
    policy = str(args.icgf_active_node_policy)
    if policy != "terminal":
        raise P478HSIContractError("M4/M5 ICGF must use terminal node policy")
    selected = plan.nodes if policy == "all" else (plan.nodes[-1],)
    missing = [int(node.ordinal) for node in selected if not node.joint_mask.any()]
    if missing:
        raise P478HSIContractError(
            "ICGF-active predicted nodes have no sparse contact joints: {}".format(
                missing
            )
        )
    explicit = p360._parse_joint_ids(args.icgf_contact_joints)
    if not explicit:
        raise P478HSIContractError(
            "M4/M5 require explicit packet-derived contact joint IDs"
        )
    guidance = (
        p360._parse_joint_ids(args.icgf_guidance_joints)
        if getattr(args, "icgf_guidance_joints", None) is not None
        else explicit
    )
    exemption = (
        p360._parse_joint_ids(args.sdf_exempt_joints)
        if getattr(args, "sdf_exempt_joints", None) is not None
        else explicit
    )
    if not set(exemption).issubset(set(guidance)):
        raise P478HSIContractError("SDF exemption IDs are not an ICGF-guidance subset")
    for node in selected:
        declared = set(int(value) for value in np.flatnonzero(node.joint_mask))
        undeclared = sorted(set(guidance) - declared)
        if undeclared:
            raise P478HSIContractError(
                "packet ICGF-guidance joints {} are absent from sparse node ordinal {}"
                .format(undeclared, node.ordinal)
            )
    receipt["active_node_policy"] = policy
    receipt["active_external_ordinals"] = [int(node.ordinal) for node in selected]
    receipt["contact_joint_id_source"] = "predicted_packet_explicit_contact_semantics"
    receipt["contact_joint_ids"] = list(explicit)
    receipt["icgf_guidance_joint_ids"] = list(guidance)
    receipt["sdf_exempt_joint_ids"] = list(exemption)
    receipt["decoupled_sparse_keypose_field"] = bool(
        guidance != explicit or exemption != explicit
    )
    receipt["guidance_target_source"] = "predicted_packet_sparse_joint_xyz_world"
    receipt["sdf_exemption_source"] = "predicted_packet_contact_role"
    receipt["declared_joint_ids_by_ordinal"] = {
        str(node.ordinal): [int(value) for value in np.flatnonzero(node.joint_mask)]
        for node in selected
    }
    return receipt


def _contact_selective_receipt(
    args: argparse.Namespace, icgf_contract: Mapping[str, Any]
) -> Dict[str, Any]:
    enabled = bool(icgf_contract.get("enabled", False))
    collision = p360._parse_joint_ids(args.collision_joint_ids)
    return {
        "schema": "p478.predicted_packet_contact_selective_sdf.v1",
        "enabled": enabled,
        "full_raw_sdf_keeps_target_component": bool(
            enabled and not p360._exclude_target_component(
                args, P478_ABLATIONS[str(args.ablation)]
            )
        ),
        "collision_joint_ids": list(collision) if collision else list(range(22)),
        "predicted_contact_joint_ids": list(
            icgf_contract.get("contact_joint_ids", ())
        ),
        "icgf_guidance_joint_ids": list(
            icgf_contract.get(
                "icgf_guidance_joint_ids",
                icgf_contract.get("contact_joint_ids", ()),
            )
        ),
        "sdf_exempt_joint_ids": list(
            icgf_contract.get(
                "sdf_exempt_joint_ids",
                icgf_contract.get("contact_joint_ids", ()),
            )
        ),
        "decoupled_sparse_keypose_field": bool(
            icgf_contract.get("decoupled_sparse_keypose_field", False)
        ),
        "guidance_target_source": "predicted_packet_sparse_joint_xyz_world",
        "sdf_exemption_source": "predicted_packet_contact_role",
        "contact_joint_sdf_exemption": (
            "terminal_tail_only" if enabled else "not_applicable"
        ),
        "contact_frames": int(args.contact_frames),
        "noncontact_joint_target_component_exemption": False,
        "ground_truth_contact_used": False,
        "ground_truth_or_dataset_pose_motion_read": False,
        "positive_memory_or_retrieval_used": False,
        "prior_rollout_motion_or_metrics_read": False,
    }


def _terminal_sparse_joint_roles(
    selected_joint_ids: Sequence[int],
    contact_joint_ids: Sequence[int],
) -> Dict[str, tuple[int, ...]]:
    """Partition each terminal sparse joint into exactly one fixed role."""

    selected = tuple(int(value) for value in selected_joint_ids)
    contacts = frozenset(int(value) for value in contact_joint_ids)
    if not selected:
        raise P478HSIContractError(
            "role-wise terminal completion requires sparse body constraints"
        )
    if len(set(selected)) != len(selected) or any(
        value < 0 or value >= 22 for value in selected
    ):
        raise P478HSIContractError(
            "role-wise selected joints must be unique IDs in [0,21]"
        )
    if not contacts.issubset(set(selected)):
        raise P478HSIContractError(
            "role-wise contact joints must be a subset of terminal sparse joints"
        )
    roles: Dict[str, list[int]] = {
        "declared_contact": [],
        "support_chain": [],
        "axial_posture": [],
        "upper_limb": [],
        "other_sparse": [],
    }
    for joint_id in selected:
        if joint_id in contacts:
            role = "declared_contact"
        elif joint_id in ROLEWISE_SUPPORT_CHAIN_IDS:
            role = "support_chain"
        elif joint_id in ROLEWISE_AXIAL_POSTURE_IDS:
            role = "axial_posture"
        elif joint_id in ROLEWISE_UPPER_LIMB_IDS:
            role = "upper_limb"
        else:
            role = "other_sparse"
        roles[role].append(joint_id)
    return {
        role: tuple(joint_ids)
        for role, joint_ids in roles.items()
        if joint_ids
    }


def _rolewise_feedback_config(
    own: argparse.Namespace,
    args: argparse.Namespace,
    plan: Any,
    contact_joint_ids: Sequence[int],
) -> RolewiseTerminalFeedbackConfig:
    config = RolewiseTerminalFeedbackConfig(
        enabled=bool(own.terminal_rolewise_feedback),
        hold_frames=int(own.terminal_rolewise_hold_frames),
        maximum_extra_primitives=int(
            own.terminal_rolewise_maximum_extra_primitives
        ),
        contact_max_m=float(own.terminal_rolewise_contact_max_m),
        support_chain_max_m=float(
            own.terminal_rolewise_support_chain_max_m
        ),
        axial_posture_max_m=float(
            own.terminal_rolewise_axial_posture_max_m
        ),
        upper_limb_max_m=float(own.terminal_rolewise_upper_limb_max_m),
        other_sparse_max_m=float(own.terminal_rolewise_other_sparse_max_m),
        global_per_joint_max_m=float(
            own.terminal_rolewise_global_per_joint_max_m
        ),
    ).validate()
    defaults = RolewiseTerminalFeedbackConfig()
    if not config.enabled:
        # Treat non-default strict options without the enable flag as a typo;
        # the ordinary/default execution path itself remains unchanged.
        if config != defaults:
            raise P478HSIContractError(
                "role-wise terminal options require --terminal-rolewise-feedback"
            )
        return config
    if str(args.ablation) not in (
        "M4_KEYPOSE_SDF_E226_ICGF",
        "M5_M4_FEEDBACK_HOLD",
    ):
        raise P478HSIContractError(
            "role-wise terminal replay is supported only for M4/M5"
        )
    terminal = plan.nodes[-1]
    selected = tuple(int(value) for value in np.flatnonzero(terminal.joint_mask))
    _terminal_sparse_joint_roles(selected, contact_joint_ids)
    return config


class _RuntimeContext:
    """Per-rollout causal state shared by the three temporary wrappers."""

    def __init__(
        self,
        *,
        plan: Any,
        ablation: str,
        bridge_config: P360E226BridgeConfig,
        feedback_config: FeedbackConfig,
        rolewise_feedback_config: RolewiseTerminalFeedbackConfig = (
            RolewiseTerminalFeedbackConfig()
        ),
        terminal_contact_joint_ids: Sequence[int] = (),
        initial_history_source: str = "native_initial_history",
        terminal_e226_mode: str = "inherit",
    ) -> None:
        self.plan = plan
        self.ablation = str(ablation)
        self.bridge_config = bridge_config
        self.feedback_config = feedback_config.validate()
        self.rolewise_feedback_config = rolewise_feedback_config.validate()
        self.terminal_contact_joint_ids = tuple(
            int(value) for value in terminal_contact_joint_ids
        )
        if self.rolewise_feedback_config.enabled:
            terminal_selected = tuple(
                int(value)
                for value in np.flatnonzero(self.plan.nodes[-1].joint_mask)
            )
            self.terminal_joint_roles = _terminal_sparse_joint_roles(
                terminal_selected, self.terminal_contact_joint_ids
            )
        else:
            self.terminal_joint_roles = {}
        if str(terminal_e226_mode) not in TERMINAL_E226_MODES:
            raise P478HSIContractError(
                "terminal_e226_mode must be one of {}".format(TERMINAL_E226_MODES)
            )
        self.terminal_e226_mode = str(terminal_e226_mode)
        if initial_history_source not in (
            "native_initial_history",
            QUERY_STATIC_HISTORY_SOURCE,
        ):
            raise P478HSIContractError("unsupported initial history source")
        self.initial_history_source = str(initial_history_source)
        self.scheduler: Any | None = None
        self.causal_world_j22: torch.Tensor | None = None
        self.native_history_frames = 0
        self.generated_frames = 0
        self.cond_receipts: list[Dict[str, Any]] = []
        self.feedback_events: list[Dict[str, Any]] = []
        self.feedback_pending = False
        self.feedback_primitives_generated = 0
        self.packet_replays = 0
        self.rolewise_terminal_complete: Optional[bool] = None
        self.rolewise_fail_closed = False
        self.rolewise_failure_reason: Optional[str] = None

    def terminal_e226_config(
        self,
        *,
        active_index: int,
        history_world_j22: torch.Tensor,
    ) -> tuple[P360E226BridgeConfig, Dict[str, Any]]:
        """Resolve an opt-in terminal release from strictly causal state.

        The release gate sees only the active packet node and the last pelvis
        in the already-emitted/generated prefix.  It never looks at the
        differentiable future being sampled.  Before the gate, or for the
        default ``inherit`` mode, the exact original config object is returned.
        """

        if self.scheduler is None:
            raise P478HSIContractError("terminal E2.2.6 gate has no scheduler")
        if (
            not torch.is_tensor(history_world_j22)
            or history_world_j22.ndim != 3
            or tuple(history_world_j22.shape[1:]) != (22, 3)
            or history_world_j22.shape[0] < 1
            or not bool(torch.isfinite(history_world_j22).all())
        ):
            raise P478HSIContractError(
                "terminal E2.2.6 gate needs finite causal history [H,22,3]"
            )
        active = self.plan.nodes[int(active_index)]
        is_terminal = int(active_index) == len(self.plan.nodes) - 1
        has_body_constraints = bool(np.asarray(active.joint_mask, dtype=bool).any())
        pelvis_xy = history_world_j22[-1, 0, :2].detach().cpu().to(torch.float64)
        target_xy = torch.as_tensor(
            active.root_xyz_yaw[:2], dtype=torch.float64
        )
        distance = float(torch.linalg.vector_norm(pelvis_xy - target_xy))
        threshold = float(self.scheduler.thresholds.root_xy_m)
        if not math.isfinite(distance) or not math.isfinite(threshold) or threshold <= 0.0:
            raise P478HSIContractError("terminal E2.2.6 gate has invalid metric values")
        # A feedback replay exists only after an already-emitted primitive
        # caused the nominal terminal scheduler to complete.  Preserve the
        # requested terminal E2.2.6 release (notably T2 ``off``) throughout
        # that replay even if the last tail pelvis subsequently drifted beyond
        # the root threshold.  This is causal state; no sample being denoised
        # is queried here.
        terminal_packet_replay = bool(
            is_terminal and has_body_constraints and self.feedback_pending
        )
        gate_passed = bool(
            is_terminal
            and has_body_constraints
            and (distance <= threshold or terminal_packet_replay)
        )
        applied = bool(self.terminal_e226_mode != "inherit" and gate_passed)
        effective = self.bridge_config
        disabled_terms: list[str] = []
        if applied and self.terminal_e226_mode == "stop_terms_off":
            rolling = replace(
                self.bridge_config.rolling,
                heading_weight=0.0,
                length_ratio_weight=0.0,
                backtrack_weight=0.0,
                zigzag_weight=0.0,
                self_loop_weight=0.0,
                sdf_weight=0.0,
            )
            effective = replace(self.bridge_config, rolling=rolling).validate()
            disabled_terms = [
                "heading", "length_ratio", "backtrack", "zigzag",
                "self_loop", "rolling_pelvis_sdf",
            ]
        elif applied and self.terminal_e226_mode == "off":
            effective = replace(self.bridge_config, rolling_scale=0.0).validate()
            disabled_terms = ["all_e226_terms"]
        # Identity, rather than equality alone, is a tested compatibility
        # invariant for inherit and every pre-gate call.
        if not applied and effective is not self.bridge_config:
            raise AssertionError("pre-gate terminal E2.2.6 changed config identity")
        receipt = {
            "schema": TERMINAL_E226_RECEIPT_SCHEMA,
            "requested_mode": self.terminal_e226_mode,
            "effective_mode": self.terminal_e226_mode if applied else "inherit",
            "release_gate_passed": gate_passed,
            "release_applied": applied,
            "active_node_index": int(active_index),
            "active_external_ordinal": int(active.ordinal),
            "active_is_terminal": is_terminal,
            "terminal_has_body_constraints": has_body_constraints,
            "causal_pelvis_world_xy": [float(value) for value in pelvis_xy.tolist()],
            "active_target_world_xy": [float(value) for value in target_xy.tolist()],
            "causal_pelvis_to_target_xy_m": distance,
            "scheduler_root_xy_threshold_m": threshold,
            "release_comparator": "distance_le_scheduler_root_xy_threshold",
            "causal_prefix_only": True,
            "future_or_current_sample_read": False,
            "disabled_terms": disabled_terms,
            "retained_terms": (
                ["goal", "smooth"]
                if applied and self.terminal_e226_mode == "stop_terms_off"
                else ([] if applied else ["inherit_all"])
            ),
            "effective_rolling_scale": float(effective.rolling_scale),
            "effective_rolling_weights": effective.rolling.weights(),
            "parent_sdf_unchanged": True,
            "parent_icgf_unchanged": True,
            "posthoc_motion_edit": False,
        }
        if terminal_packet_replay:
            receipt.update(
                {
                    "terminal_packet_replay_after_nominal_completion": True,
                    "release_gate_reason": "causal_terminal_packet_replay",
                    "release_comparator": (
                        "distance_le_scheduler_root_xy_threshold_or_"
                        "causal_terminal_packet_replay"
                    ),
                }
            )
        return effective, receipt

    @property
    def use_e226(self) -> bool:
        return self.ablation in EXPECTED_ABLATIONS[2:]

    @property
    def use_feedback(self) -> bool:
        return bool(
            self.rolewise_feedback_config.enabled
            or self.ablation == "M5_M4_FEEDBACK_HOLD"
        )

    @property
    def use_rolewise_feedback(self) -> bool:
        return bool(self.rolewise_feedback_config.enabled)

    @property
    def maximum_feedback_primitives(self) -> int:
        if self.use_rolewise_feedback:
            return int(self.rolewise_feedback_config.maximum_extra_primitives)
        return int(self.feedback_config.maximum_extra_primitives)

    def _native_history_from_kwargs(self, kwargs: Mapping[str, Any]) -> torch.Tensor:
        history_motion = kwargs["history_motion"]
        dataset = kwargs["dataset"]
        utility = getattr(dataset, "primitive_utility")
        frames = dataset.denormalize(history_motion)
        feature = utility.tensor_to_dict(frames)
        joints = feature.get("joints")
        if not torch.is_tensor(joints) or joints.numel() % (22 * 3):
            raise P478HSIContractError(
                "P360 native history does not decode to an integral J22 sequence"
            )
        joints_local = joints.reshape(int(joints.shape[0]), -1, 22, 3)
        world = p360.p142.active_local_points_to_world(
            joints_local,
            kwargs["global_rotmat"],
            kwargs["global_transl"],
            kwargs["current_rotmat"],
            kwargs["current_transl"],
        )
        if world.shape[0] != 1:
            raise P478HSIContractError("P478 causal rollout supports batch size one")
        return world[0].detach().cpu().clone()

    def causal_history_for_cond(
        self, kwargs: Mapping[str, Any]
    ) -> tuple[torch.Tensor, str]:
        if self.causal_world_j22 is None:
            native = self._native_history_from_kwargs(kwargs)
            if native.shape[0] != 2:
                raise P478HSIContractError(
                    "primitive zero must expose exactly P360's two native frames"
                )
            self.causal_world_j22 = native
            self.native_history_frames = int(native.shape[0])
            return native, self.initial_history_source
        if self.generated_frames < 1:
            raise P478HSIContractError(
                "a later condition call occurred without an emitted generated prefix"
            )
        return self.causal_world_j22, "generated_prefix"

    def append_generated(self, joints_world: np.ndarray, primitive_id: int) -> None:
        value = torch.as_tensor(joints_world, dtype=torch.float32)
        if value.ndim != 3 or value.shape[1:] != (22, 3):
            raise P478HSIContractError(
                "scheduler emitted joints must have shape [T,22,3]"
            )
        if self.use_e226 and self.causal_world_j22 is None:
            raise P478HSIContractError(
                "generated frames appeared before native causal history was captured"
            )
        if self.causal_world_j22 is not None:
            self.causal_world_j22 = torch.cat(
                (self.causal_world_j22, value.detach().cpu().clone()), dim=0
            )
        self.generated_frames += int(value.shape[0])

    def condition_factory(self, **kwargs: Any) -> ReMoGenE226CondFn:
        if self.scheduler is None:
            raise P478HSIContractError(
                "condition function constructed before the ordered scheduler"
            )
        history, source = self.causal_history_for_cond(kwargs)
        active_index = min(
            int(self.scheduler.next_node_index), len(self.plan.nodes) - 1
        )
        active = self.plan.nodes[active_index]
        next_node = (
            self.plan.nodes[active_index + 1]
            if active_index + 1 < len(self.plan.nodes)
            else None
        )
        supplied_goal = kwargs.pop("goal_world_zup", None)
        if supplied_goal is None:
            raise P478HSIContractError("P360 did not supply the active planner goal")
        supplied = torch.as_tensor(supplied_goal).detach().cpu().reshape(-1, 3)[0]
        expected = torch.as_tensor(active.root_xyz_yaw[:3], dtype=supplied.dtype)
        if not torch.allclose(supplied, expected, atol=1.0e-4, rtol=0.0):
            raise P478HSIContractError(
                "P360 active goal and P478 scheduler node disagree"
            )
        primitive_id = len(self.cond_receipts)
        effective_e226_config, terminal_e226_receipt = self.terminal_e226_config(
            active_index=active_index,
            history_world_j22=history,
        )
        cond = ReMoGenE226CondFn(
            generated_history_world_joints_zup=history,
            history_source=source,
            active_goal_world_zup=active.root_xyz_yaw[:3],
            next_goal_world_zup=(
                None if next_node is None else next_node.root_xyz_yaw[:3]
            ),
            active_yaw_rad=float(active.root_xyz_yaw[3]),
            e226_config=effective_e226_config,
            goal_world_zup=None,
            **kwargs,
        )
        receipt = {
            **cond.e226_receipt,
            "primitive_id": int(primitive_id),
            "active_node_index": int(active_index),
            "active_external_ordinal": int(active.ordinal),
            "active_goal_world_xyz": active.root_xyz_yaw[:3].tolist(),
            "active_yaw_rad": float(active.root_xyz_yaw[3]),
            "next_goal_world_xyz": (
                None if next_node is None else next_node.root_xyz_yaw[:3].tolist()
            ),
            "causal_prefix_frames_total": int(history.shape[0]),
            "native_initial_frames_in_prefix": int(self.native_history_frames),
            "generated_frames_in_prefix": int(self.generated_frames),
            "feedback_terminal_packet_replay": bool(
                self.use_feedback and self.feedback_pending
            ),
            "terminal_e226": terminal_e226_receipt,
        }
        self.cond_receipts.append(receipt)
        return cond

    def evaluate_feedback_hold(
        self, scheduler: Any, joints_world: np.ndarray, primitive_id: int
    ) -> Dict[str, Any]:
        if self.use_rolewise_feedback:
            return self.evaluate_rolewise_terminal_hold(
                scheduler, joints_world, primitive_id
            )
        joints = np.asarray(joints_world, dtype=np.float64)
        node = self.plan.nodes[-1]
        yaw = p360.p142.heading_yaw_from_joints(joints)
        root_xy, root_z, yaw_error, joint_error = scheduler._frame_metrics(
            node, joints, yaw
        )
        tail = min(int(self.feedback_config.hold_frames), len(joints))
        scale = float(self.feedback_config.threshold_scale)
        valid = (
            (root_xy <= scheduler.thresholds.root_xy_m * scale)
            & (root_z <= scheduler.thresholds.root_z_m * scale)
            & (yaw_error <= scheduler.thresholds.yaw_rad * scale)
        )
        if node.joint_mask.any():
            valid &= joint_error <= scheduler.thresholds.joint_m * scale
        passed = bool(valid[-tail:].all())
        return {
            "schema": "p478.feedback_hold_observation.v1",
            "primitive_id": int(primitive_id),
            "external_ordinal": int(node.ordinal),
            "tail_frames_required": int(tail),
            "tail_valid_count": int(valid[-tail:].sum()),
            "passed": passed,
            "tail_max_root_xy_m": float(root_xy[-tail:].max()),
            "tail_max_root_z_m": float(root_z[-tail:].max()),
            "tail_max_yaw_rad": float(yaw_error[-tail:].max()),
            "tail_max_selected_joint_m": float(joint_error[-tail:].max()),
            "threshold_scale": scale,
            "response": "no_motion_edit",
        }

    def evaluate_rolewise_terminal_hold(
        self, scheduler: Any, joints_world: np.ndarray, primitive_id: int
    ) -> Dict[str, Any]:
        """Check an emitted primitive without averaging away joint outliers."""

        config = self.rolewise_feedback_config
        if not config.enabled:
            raise P478HSIContractError(
                "role-wise terminal hold called while the opt-in is disabled"
            )
        joints = np.asarray(joints_world, dtype=np.float64)
        if (
            joints.ndim != 3
            or joints.shape[1:] != (22, 3)
            or len(joints) < int(config.hold_frames)
            or not np.isfinite(joints).all()
        ):
            raise P478HSIContractError(
                "role-wise terminal hold needs finite emitted joints [T,22,3] "
                "with at least hold_frames observations"
            )
        node = self.plan.nodes[-1]
        selected = tuple(int(value) for value in np.flatnonzero(node.joint_mask))
        if not selected or set(selected) != {
            joint_id
            for role_ids in self.terminal_joint_roles.values()
            for joint_id in role_ids
        }:
            raise P478HSIContractError(
                "terminal sparse joint roles drifted after context construction"
            )
        yaw = p360.p142.heading_yaw_from_joints(joints)
        root_xy, root_z, yaw_error, legacy_joint_mean = scheduler._frame_metrics(
            node, joints, yaw
        )
        selected_array = np.asarray(selected, dtype=np.int64)
        target = np.asarray(node.joint_xyz_world, dtype=np.float64)[selected_array]
        per_joint = np.linalg.norm(
            joints[:, selected_array] - target[None], axis=-1
        )
        tail = int(config.hold_frames)
        tail_slice = slice(len(joints) - tail, None)
        root_valid = (
            (root_xy <= scheduler.thresholds.root_xy_m)
            & (root_z <= scheduler.thresholds.root_z_m)
            & (yaw_error <= scheduler.thresholds.yaw_rad)
        )
        global_max = per_joint.max(axis=-1)
        combined_valid = root_valid & (
            global_max <= float(config.global_per_joint_max_m)
        )
        id_to_column = {
            joint_id: index for index, joint_id in enumerate(selected)
        }
        role_receipts: Dict[str, Any] = {}
        for role, role_ids in self.terminal_joint_roles.items():
            columns = [id_to_column[joint_id] for joint_id in role_ids]
            values = per_joint[:, columns]
            maximum = values.max(axis=-1)
            threshold = float(config.role_thresholds_m[role])
            role_valid = maximum <= threshold
            combined_valid &= role_valid
            role_receipts[role] = {
                "joint_ids": list(role_ids),
                "maximum_threshold_m": threshold,
                "tail_maximum_m": float(maximum[tail_slice].max()),
                "tail_mean_m": float(values[tail_slice].mean()),
                "tail_all_frames_pass": bool(role_valid[tail_slice].all()),
                "tail_per_frame_maximum_m": maximum[tail_slice].tolist(),
            }
        passed = bool(combined_valid[tail_slice].all())
        tail_per_joint = per_joint[tail_slice].max(axis=0)
        return {
            "schema": "p478.rolewise_terminal_hold_observation.v1",
            "primitive_id": int(primitive_id),
            "external_ordinal": int(node.ordinal),
            "tail_frames_required": tail,
            "tail_valid_count": int(combined_valid[tail_slice].sum()),
            "passed": passed,
            "tail_max_root_xy_m": float(root_xy[tail_slice].max()),
            "tail_max_root_z_m": float(root_z[tail_slice].max()),
            "tail_max_yaw_rad": float(yaw_error[tail_slice].max()),
            "legacy_tail_max_selected_joint_mean_m": float(
                legacy_joint_mean[tail_slice].max()
            ),
            "legacy_mean_hold_would_pass": bool(
                (
                    root_valid[tail_slice]
                    & (
                        legacy_joint_mean[tail_slice]
                        <= scheduler.thresholds.joint_m
                    )
                ).all()
            ),
            "tail_global_per_joint_max_m": float(global_max[tail_slice].max()),
            "global_per_joint_max_threshold_m": float(
                config.global_per_joint_max_m
            ),
            "tail_per_joint_maximum_m": {
                str(joint_id): float(tail_per_joint[column])
                for column, joint_id in enumerate(selected)
            },
            "role_gates": role_receipts,
            "thresholds": asdict(config),
            "causal_prefix_only": True,
            "future_or_current_denoising_sample_read": False,
            "ground_truth_or_dataset_motion_read": False,
            "response": "sample_new_terminal_conditioned_primitive_or_stop",
            "existing_frames_modified": False,
            "completion_only_not_precompletion_physics_repair": True,
            "cannot_remove_failed_approach_frames_already_in_output": True,
            "transactional_extension_not_implemented": (
                "reject_failed_provisional_terminal_segment_before_publish_and_"
                "regenerate_from_terminal_start_prefix"
            ),
        }


def _make_scheduler_class(original: type, context: _RuntimeContext) -> type:
    class P478OrderedScheduler(original):
        _inside_base_observe = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            context.scheduler = self

        @property
        def complete(self) -> bool:
            base_complete = self.next_node_index >= len(self.plan.nodes)
            if self._inside_base_observe:
                return base_complete
            return base_complete and not context.feedback_pending

        def observe_primitive(
            self, joints_world: np.ndarray, primitive_id: int
        ) -> list[Dict[str, Any]]:
            context.append_generated(joints_world, primitive_id)
            if self.next_node_index < len(self.plan.nodes):
                self._inside_base_observe = True
                try:
                    events = list(
                        super().observe_primitive(joints_world, primitive_id)
                    )
                finally:
                    self._inside_base_observe = False
                just_completed = self.next_node_index >= len(self.plan.nodes)
                if context.use_feedback and just_completed:
                    audit = context.evaluate_feedback_hold(
                        self, joints_world, primitive_id
                    )
                    if (
                        not audit["passed"]
                        and context.maximum_feedback_primitives > 0
                    ):
                        context.feedback_pending = True
                        audit["status"] = "feedback_extra_primitive_requested"
                        audit["next_action"] = (
                            "sample_new_terminal_conditioned_primitive"
                        )
                        if context.use_rolewise_feedback:
                            context.rolewise_terminal_complete = False
                    else:
                        if context.use_rolewise_feedback and not audit["passed"]:
                            audit["status"] = (
                                "rolewise_feedback_budget_zero_fail_closed"
                            )
                            audit["next_action"] = "stop_incomplete_without_frame_edit"
                            context.rolewise_terminal_complete = False
                            context.rolewise_fail_closed = True
                            context.rolewise_failure_reason = (
                                "zero_extra_terminal_primitive_budget"
                            )
                        else:
                            audit["status"] = "feedback_hold_already_satisfied"
                            audit["next_action"] = "stop"
                            if context.use_rolewise_feedback:
                                context.rolewise_terminal_complete = True
                    self.events.append(audit)
                    events.append(audit)
                    context.feedback_events.append(dict(audit))
                return events

            if not (context.use_feedback and context.feedback_pending):
                raise P478HSIContractError(
                    "scheduler received a primitive after terminal completion"
                )
            audit = context.evaluate_feedback_hold(self, joints_world, primitive_id)
            context.feedback_primitives_generated += 1
            exhausted = (
                context.feedback_primitives_generated
                >= context.maximum_feedback_primitives
            )
            if audit["passed"]:
                audit["status"] = "feedback_hold_satisfied"
                audit["next_action"] = "stop"
                context.feedback_pending = False
                if context.use_rolewise_feedback:
                    context.rolewise_terminal_complete = True
            elif exhausted:
                audit["status"] = (
                    "rolewise_feedback_budget_exhausted_fail_closed"
                    if context.use_rolewise_feedback
                    else "feedback_budget_exhausted"
                )
                audit["next_action"] = (
                    "stop_incomplete_without_frame_edit"
                    if context.use_rolewise_feedback
                    else "stop_without_frame_edit"
                )
                context.feedback_pending = False
                if context.use_rolewise_feedback:
                    context.rolewise_terminal_complete = False
                    context.rolewise_fail_closed = True
                    context.rolewise_failure_reason = (
                        "maximum_extra_terminal_primitives_exhausted"
                    )
            else:
                audit["status"] = "feedback_extra_primitive_requested"
                audit["next_action"] = "sample_new_terminal_conditioned_primitive"
            audit["feedback_primitive_index"] = int(
                context.feedback_primitives_generated
            )
            self.events.append(audit)
            context.feedback_events.append(dict(audit))
            return [audit]

    P478OrderedScheduler.__name__ = "P478OrderedKeyposeScheduler"
    return P478OrderedScheduler


@contextmanager
def _patched_p360(context: _RuntimeContext) -> Iterator[None]:
    """Install the minimal P478 hooks and always restore frozen P360 globals."""

    old_ablations = p360.ABLATIONS
    old_cond = p360.ReMoGenRawSceneCondFn
    old_scheduler = p360.p142.OrderedKeyposeScheduler
    old_packet = p360.p142.pending_plan_to_local_packet

    scheduler_cls = _make_scheduler_class(old_scheduler, context)

    def packet_wrapper(
        plan: Any, next_node_index: int, maximum_nodes: int, *args: Any, **kwargs: Any
    ) -> Any:
        resolved = int(next_node_index)
        if resolved == len(plan.nodes) and context.use_feedback and context.feedback_pending:
            resolved = len(plan.nodes) - 1
            context.packet_replays += 1
        return old_packet(plan, resolved, maximum_nodes, *args, **kwargs)

    try:
        p360.ABLATIONS = P478_ABLATIONS
        p360.p142.OrderedKeyposeScheduler = scheduler_cls
        p360.p142.pending_plan_to_local_packet = packet_wrapper
        if context.use_e226:
            p360.ReMoGenRawSceneCondFn = context.condition_factory
        yield
    finally:
        p360.ABLATIONS = old_ablations
        p360.ReMoGenRawSceneCondFn = old_cond
        p360.p142.OrderedKeyposeScheduler = old_scheduler
        p360.p142.pending_plan_to_local_packet = old_packet


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[
    argparse.Namespace,
    argparse.Namespace,
    Dict[str, Any],
    P360E226BridgeConfig,
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--bridge-config", type=Path, default=DEFAULT_BRIDGE_CONFIG)
    pre.add_argument("--feedback-hold-frames", type=int, default=3)
    pre.add_argument("--maximum-feedback-primitives", type=int, default=1)
    pre.add_argument("--feedback-threshold-scale", type=float, default=1.0)
    pre.add_argument(
        "--terminal-rolewise-feedback",
        action="store_true",
        help=(
            "Opt in to causal per-joint/role-wise terminal completion and "
            "new-primitive replay for M4/M5. Disabled by default."
        ),
    )
    pre.add_argument("--terminal-rolewise-hold-frames", type=int, default=3)
    pre.add_argument(
        "--terminal-rolewise-maximum-extra-primitives", type=int, default=3
    )
    pre.add_argument("--terminal-rolewise-contact-max-m", type=float, default=0.15)
    pre.add_argument(
        "--terminal-rolewise-support-chain-max-m", type=float, default=0.25
    )
    pre.add_argument(
        "--terminal-rolewise-axial-posture-max-m", type=float, default=0.20
    )
    pre.add_argument(
        "--terminal-rolewise-upper-limb-max-m", type=float, default=0.25
    )
    pre.add_argument(
        "--terminal-rolewise-other-sparse-max-m", type=float, default=0.25
    )
    pre.add_argument(
        "--terminal-rolewise-global-per-joint-max-m", type=float, default=0.30
    )
    pre.add_argument("--memory-session", type=Path, default=None)
    pre.add_argument("--memory-profile", type=Path, default=None)
    pre.add_argument("--query-static-seed", type=Path, default=None)
    pre.add_argument(
        "--terminal-e226-mode",
        choices=TERMINAL_E226_MODES,
        default="inherit",
        help=(
            "Opt-in causal release after the terminal body node and root-XY gate. "
            "Default inherit preserves retry3 exactly."
        ),
    )
    own, remaining = pre.parse_known_args(argv)
    remaining = list(remaining)
    if "--ablation" not in remaining:
        remaining.extend(("--ablation", "M3_KEYPOSE_SDF_E226"))
    contract, bridge_config = load_contract(Path(own.bridge_config).resolve())

    old = p360.ABLATIONS
    try:
        p360.ABLATIONS = P478_ABLATIONS
        args = p360.parse_args(remaining)
    finally:
        p360.ABLATIONS = old

    # P478 is evaluated with the selected scene-disjoint P360 adapter unless
    # the caller deliberately provides another audited checkpoint.
    if "--adapter-checkpoint" not in remaining:
        args.adapter_checkpoint = DEFAULT_P360_ADAPTER
    if "--checkpoint-policy-label" not in remaining:
        args.checkpoint_policy_label = "p360_scene_disjoint_v3_step36000"
    if "--post-completion-primitives" not in remaining:
        args.post_completion_primitives = 0
    if "--output-dir" not in remaining:
        args.output_dir = Path("outputs/motion")
    if str(args.ablation) in (
        "M4_KEYPOSE_SDF_E226_ICGF",
        "M5_M4_FEEDBACK_HOLD",
    ):
        if "--icgf-source" not in remaining:
            args.icgf_source = "sparse_joint_targets"
        if "--icgf-active-node-policy" not in remaining:
            args.icgf_active_node_policy = "terminal"
    memory_session, memory_profile = _load_memory_profile(
        own.memory_session, own.memory_profile
    )
    if memory_profile is not None:
        multipliers = memory_profile["multipliers"]
        args.sdf_weight = float(args.sdf_weight) * float(multipliers["sdf_weight"])
        args.icgf_weight = float(args.icgf_weight) * float(multipliers["icgf_weight"])
        bridge_config = bridge_config.with_memory_multiplier(
            float(multipliers["e226_rolling_scale"])
        )
    return (
        args,
        own,
        contract,
        bridge_config,
        memory_session,
        memory_profile,
    )


def _validate_p478_args(
    args: argparse.Namespace,
    own: argparse.Namespace,
    plan: Any,
    contract: Mapping[str, Any],
) -> tuple[FeedbackConfig, Dict[str, Any]]:
    if str(args.ablation) not in EXPECTED_ABLATIONS:
        raise P478HSIContractError("unknown P478 ablation")
    if int(args.post_completion_primitives) != 0:
        raise P478HSIContractError(
            "P478 requires --post-completion-primitives 0; M5 alone may add a "
            "causally requested feedback primitive"
        )
    if own.query_static_seed is not None and str(args.seq_name_override):
        raise P478HSIContractError(
            "query-static mode forbids --seq-name-override; no LINGO sequence may seed it"
        )
    expected_sources = (
        [QUERY_STATIC_HISTORY_SOURCE, "generated_prefix"]
        if own.query_static_seed is not None
        else ["native_initial_history", "generated_prefix"]
    )
    declared_sources = contract.get("causal_contract", {}).get(
        "allowed_history_sources"
    )
    if declared_sources != expected_sources:
        raise P478HSIContractError(
            "bridge config history sources must be exactly {} for this seed mode"
            .format(expected_sources)
        )
    feedback = FeedbackConfig(
        hold_frames=int(own.feedback_hold_frames),
        maximum_extra_primitives=int(own.maximum_feedback_primitives),
        threshold_scale=float(own.feedback_threshold_scale),
    ).validate()
    if str(args.ablation) != "M5_M4_FEEDBACK_HOLD" and (
        feedback.maximum_extra_primitives != 1
        or feedback.hold_frames != 3
        or feedback.threshold_scale != 1.0
    ):
        raise P478HSIContractError(
            "feedback options may only be changed for M5"
        )
    if str(args.ablation) == "M5_M4_FEEDBACK_HOLD" and (
        feedback.maximum_extra_primitives < 1
    ):
        raise P478HSIContractError("M5 needs at least one feedback primitive budget")
    declared = contract["ablations"][str(args.ablation)]
    runtime = P478_ABLATIONS[str(args.ablation)]
    if bool(declared["keypose_packet"]) != bool(runtime.use_keypose_packet):
        raise P478HSIContractError("ablation keypose switch differs from contract")
    if bool(declared["sdf"]) != bool(runtime.use_sdf):
        raise P478HSIContractError("ablation SDF switch differs from contract")
    if bool(declared["predicted_icgf"]) != bool(runtime.use_icgf):
        raise P478HSIContractError("ablation ICGF switch differs from contract")
    return feedback, _strict_icgf_contract(args, plan)


def _input_receipt(
    args: argparse.Namespace,
    plan: Any,
    query_static_seed: Optional[QueryStaticSeed] = None,
) -> Dict[str, Any]:
    paths = {
        "plan": Path(args.plan),
        "raw_occupancy": Path(args.raw_occupancy),
        "adapter_checkpoint": Path(args.adapter_checkpoint),
        "base_checkpoint": Path(args.base_checkpoint),
        "mvae_checkpoint": Path(args.mvae_checkpoint),
    }
    artifacts: Dict[str, Any] = {}
    for role, path in paths.items():
        artifacts[role] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    if query_static_seed is not None:
        artifacts["query_static_seed"] = {
            "path": str(query_static_seed.path),
            "exists": True,
            "sha256": query_static_seed.seed_hash,
        }
    receipt = {
        "schema": "p478.hsi_input_provenance.v1",
        "plan_provenance": str(plan.provenance),
        "plan_is_trusted_predicted": _trusted_predicted_plan(plan),
        "plan_is_oracle": bool(p360.p142._is_oracle_provenance(plan.provenance)),
        "generation_motion_seed_frames": [0, 1],
        "future_motion_conditioned": False,
        "gt_contact_conditioned": False,
        "gt_icgf_conditioned": False,
        "memory_or_retrieval_conditioned": False,
        "artifacts": artifacts,
    }
    if query_static_seed is not None:
        receipt.update(
            {
                "generation_motion_seed_frames": [],
                "history_source": QUERY_STATIC_HISTORY_SOURCE,
                "query_static_seed_enabled": True,
                "lingo_sequence_lookup": False,
                "lingo_motion_opened": False,
                "source_frame_indices": [],
                "same_programmatic_frame_repeated": True,
            }
        )
    return receipt


def _base_manifest(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    plan: Any,
    icgf_contract: Mapping[str, Any],
    feedback: FeedbackConfig,
    memory_session: Optional[Mapping[str, Any]] = None,
    memory_profile: Optional[Mapping[str, Any]] = None,
    query_static_seed: Optional[QueryStaticSeed] = None,
    terminal_e226_mode: str = "inherit",
    rolewise_feedback: Optional[RolewiseTerminalFeedbackConfig] = None,
) -> Dict[str, Any]:
    result = {
        "schema": "p478.hsi_m1_m5_run_manifest.v1",
        "status": "planned",
        "selected_ablation": str(args.ablation),
        "selected_switches": contract["ablations"][str(args.ablation)],
        "all_ablations": contract["ablations"],
        "coordinate_frame": "world_zup_xyz_m_navigation_xy",
        "plan": {
            "path": str(Path(args.plan).resolve()),
            "seq_name": plan.seq_name,
            "scene_name": plan.scene_name,
            "text": plan.text,
            "node_count": len(plan.nodes),
            "provenance": plan.provenance,
        },
        "causal_contract": contract["causal_contract"],
        "input_provenance": _input_receipt(args, plan, query_static_seed),
        "icgf_contract": dict(icgf_contract),
        "contact_selective_target_policy": _contact_selective_receipt(
            args, icgf_contract
        ),
        "feedback_contract": {
            **asdict(feedback),
            "enabled": str(args.ablation) == "M5_M4_FEEDBACK_HOLD",
            "mechanism": "newly_sampled_primitive_only",
            "edits_or_duplicates_existing_frames": False,
        },
        "terminal_e226_contract": {
            "schema": "p478.terminal_e226_release_contract.v1",
            "requested_mode": str(terminal_e226_mode),
            "default_mode": "inherit",
            "release_requires_terminal_node": True,
            "release_requires_body_constraints": True,
            "release_requires_causal_pelvis_within_scheduler_root_xy": True,
            "uses_current_or_future_sample_for_release": False,
            "stop_terms_off_retains": ["goal", "smooth"],
            "off_sets_rolling_scale_to_zero": True,
            "parent_sdf_unchanged": True,
            "parent_icgf_unchanged": True,
            "posthoc_motion_edit": False,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "p360_runner": str(Path(p360.__file__).resolve()),
            "integration": "temporary_constructor_scheduler_wrappers",
            "p360_source_modified": False,
        },
    }
    result["memory"] = {
        "enabled": memory_profile is not None,
        "session": None if memory_session is None else dict(memory_session),
        "profile": None if memory_profile is None else dict(memory_profile),
        "changes_targets": False,
        "metric_sdf_has_veto": True,
        "retrieval_used": False,
    }
    result["input_provenance"]["memory_or_retrieval_conditioned"] = bool(
        memory_profile is not None
    )
    result["input_provenance"]["retrieval_conditioned"] = False
    if rolewise_feedback is not None and rolewise_feedback.enabled:
        result["terminal_rolewise_feedback_contract"] = {
            "schema": "p478.rolewise_terminal_feedback_contract.v1",
            **asdict(rolewise_feedback),
            "supported_ablations": [
                "M4_KEYPOSE_SDF_E226_ICGF",
                "M5_M4_FEEDBACK_HOLD",
            ],
            "trigger": "after_nominal_terminal_completion_only",
            "observation": "already_emitted_current_primitive_world_j22",
            "mechanism": "replay_terminal_packet_and_sample_new_primitive",
            "existing_frames_modified_or_duplicated": False,
            "budget_exhaustion": "fail_closed_incomplete",
            "terminal_e226_requested_mode_persists_during_replay": True,
            "completion_only_not_precompletion_physics_repair": True,
            "known_limitation": (
                "append-only replay cannot remove penetration already emitted "
                "before nominal terminal completion"
            ),
            "transactional_future_design": (
                "treat terminal segment as provisional; reject a failed segment "
                "before publication, restore terminal-start prefix, then regenerate "
                "under refined conditions"
            ),
            "transactional_future_design_implemented": False,
        }
    return result


def _augment_result(
    metadata: Dict[str, Any],
    context: _RuntimeContext,
    manifest: Dict[str, Any],
    icgf_contract: Mapping[str, Any],
    contact_selective: Mapping[str, Any],
    memory_profile: Optional[Mapping[str, Any]] = None,
    query_static_runtime_audit: Optional[Mapping[str, Any]] = None,
) -> None:
    if context.use_rolewise_feedback and context.feedback_pending:
        # The outer P360 primitive cap ended the rollout before the strict
        # replay budget resolved.  Preserve scheduler_complete=false and emit
        # an explicit fail-closed terminal state.
        context.rolewise_terminal_complete = False
        context.rolewise_fail_closed = True
        context.rolewise_failure_reason = (
            "rollout_max_primitives_before_rolewise_completion"
        )
    elif (
        context.use_rolewise_feedback
        and context.rolewise_terminal_complete is None
    ):
        context.rolewise_terminal_complete = False
        context.rolewise_fail_closed = True
        context.rolewise_failure_reason = "nominal_terminal_never_completed"
    records = metadata.get("primitive_records", [])
    if context.use_e226 and len(context.cond_receipts) != len(records):
        raise P478HSIContractError(
            "expected one E2.2.6 receipt per generated primitive, got {} for {}"
            .format(len(context.cond_receipts), len(records))
        )
    for index, record in enumerate(records):
        record["e226_receipt"] = (
            context.cond_receipts[index] if context.use_e226 else None
        )
        record["history_source"] = (
            context.cond_receipts[index]["history_source"]
            if context.use_e226
            else None
        )
        record["contact_selective_target_policy"] = dict(contact_selective)
    metadata["schema"] = "p478_hsi_m1_m5_rollout_v1"
    metadata["p478_selected_ablation"] = context.ablation
    metadata["p478_icgf_contract"] = dict(icgf_contract)
    metadata["contact_selective_target_policy"] = dict(contact_selective)
    metadata["p478_e226_receipts"] = context.cond_receipts
    metadata["p478_terminal_e226_receipts"] = [
        receipt["terminal_e226"] for receipt in context.cond_receipts
    ]
    metadata["p478_terminal_e226_mode"] = context.terminal_e226_mode
    metadata["p478_feedback_events"] = context.feedback_events
    metadata["p478_feedback_primitives_generated"] = int(
        context.feedback_primitives_generated
    )
    metadata["p478_terminal_packet_replays"] = int(context.packet_replays)
    if context.use_rolewise_feedback:
        metadata["p478_terminal_rolewise_completion"] = {
            "schema": "p478.rolewise_terminal_completion_result.v1",
            "enabled": True,
            "terminal_complete": bool(context.rolewise_terminal_complete),
            "fail_closed_incomplete": bool(context.rolewise_fail_closed),
            "failure_reason": context.rolewise_failure_reason,
            "extra_primitives_generated": int(
                context.feedback_primitives_generated
            ),
            "maximum_extra_primitives": int(
                context.rolewise_feedback_config.maximum_extra_primitives
            ),
            "terminal_packet_replays": int(context.packet_replays),
            "events": context.feedback_events,
            "thresholds": asdict(context.rolewise_feedback_config),
            "causal_current_primitive_only": True,
            "existing_frames_modified": False,
            "completion_only_not_precompletion_physics_repair": True,
            "cannot_remove_failed_approach_frames_already_emitted": True,
            "transactional_reject_before_publish_implemented": False,
        }
    if query_static_runtime_audit is None:
        metadata["p478_causal_prefix"] = {
            "native_initial_frames": int(context.native_history_frames),
            "generated_frames_observed": int(context.generated_frames),
            "history_sources_allowed": [
                "native_initial_history",
                "generated_prefix",
            ],
            "future_or_gt_read": False,
        }
    else:
        metadata["p478_causal_prefix"] = {
            "initial_history_frames": int(context.native_history_frames),
            "native_initial_frames": 0,
            "query_static_initial_frames": int(context.native_history_frames),
            "generated_frames_observed": int(context.generated_frames),
            "initial_history_source": QUERY_STATIC_HISTORY_SOURCE,
            "history_sources_allowed": [
                QUERY_STATIC_HISTORY_SOURCE,
                "generated_prefix",
            ],
            "future_or_gt_read": False,
        }
    metadata["posthoc_motion_edit_used"] = False
    metadata["feedback_modifies_emitted_frames"] = False
    metadata["gt_contact_or_icgf_used"] = False
    metadata["memory_used"] = bool(memory_profile is not None)
    metadata["memory_profile"] = (
        None if memory_profile is None else dict(memory_profile)
    )
    metadata["memory_changes_targets"] = False
    metadata["retrieval_used"] = False
    if query_static_runtime_audit is not None:
        metadata["query_static_runtime_audit"] = dict(query_static_runtime_audit)
        metadata["generated_only_output"] = True
    manifest["status"] = "complete"
    manifest["result"] = {
        "generated_frames": metadata.get("generated_frames_full"),
        "primitives_generated": metadata.get("primitives_generated"),
        "scheduler_complete": metadata.get("scheduler_complete"),
        "nodes_consumed": metadata.get("nodes_consumed"),
        "feedback_primitives_generated": context.feedback_primitives_generated,
        "e226_receipts": len(context.cond_receipts),
        "terminal_e226_mode": context.terminal_e226_mode,
        "terminal_e226_release_count": sum(
            bool(receipt["terminal_e226"]["release_applied"])
            for receipt in context.cond_receipts
        ),
    }
    if context.use_rolewise_feedback:
        manifest["result"]["terminal_rolewise_complete"] = bool(
            context.rolewise_terminal_complete
        )
        manifest["result"]["terminal_rolewise_fail_closed"] = bool(
            context.rolewise_fail_closed
        )
        manifest["result"]["terminal_rolewise_failure_reason"] = (
            context.rolewise_failure_reason
        )
    if memory_profile is not None:
        manifest["result"]["memory_snapshot_id"] = int(
            memory_profile["snapshot_id"]
        )
        manifest["result"]["memory_snapshot_hash"] = str(
            memory_profile["snapshot_hash"]
        )
    if query_static_runtime_audit is not None:
        manifest["query_static_runtime_audit"] = dict(query_static_runtime_audit)
        manifest["result"]["generated_only_output"] = True
        manifest["result"]["gt_fields_written"] = False


def _query_static_generated_only_sequence(
    generated: Mapping[str, Any],
    *,
    plan: Any,
    seed: QueryStaticSeed,
    history_length: int = 2,
    future_length: int = 8,
) -> Dict[str, Any]:
    """Rebuild eval.pkl from the full generated arrays, never a static pseudo-GT.

    The audited frozen P360 snapshot predates its generated-only serialization
    hook.  Its query-static compatibility branch can therefore label the
    programmatic repeated seed as ``gt_*`` and truncate ``sequence`` to that
    seed's synthetic length.  The returned ``generated`` mapping is the
    unchanged full rollout and is the authoritative serialization source.
    """

    required = ("betas", "transl", "global_orient", "body_pose", "joints")
    arrays: Dict[str, torch.Tensor] = {}
    frame_count: Optional[int] = None
    for name in required:
        if name not in generated:
            raise P478HSIContractError(
                "query-static generated output misses {}".format(name)
            )
        value = np.asarray(generated[name])
        if value.ndim < 1 or not np.isfinite(value).all():
            raise P478HSIContractError(
                "query-static generated {} is empty/non-finite".format(name)
            )
        if frame_count is None:
            frame_count = int(value.shape[0])
        elif int(value.shape[0]) != frame_count:
            raise P478HSIContractError(
                "query-static generated arrays disagree on frame count"
            )
        arrays[name] = torch.as_tensor(np.ascontiguousarray(value))[None]
    if frame_count is None or frame_count < 1:
        raise P478HSIContractError("query-static generated rollout has no frames")
    generated_world: Dict[str, Any] = {**arrays, "gender": str(seed.gender)}
    sequence = generated_only_sequence(
        generated_world,
        plan=plan,
        history_length=int(history_length),
        future_length=int(future_length),
    )
    if any(str(key).startswith("gt_") for key in sequence):
        raise P478HSIContractError("generated-only sequence contains gt_* fields")
    for name in required:
        value = sequence[name]
        if not torch.is_tensor(value) or value.shape[0] != 1 or value.shape[1] != frame_count:
            raise P478HSIContractError(
                "generated-only sequence {} did not preserve all frames".format(name)
            )
    if sequence.get("generated_only") is not True:
        raise P478HSIContractError("generated-only sequence flag is false")
    if sequence.get("reference_motion_available") is not False:
        raise P478HSIContractError("generated-only sequence exposes a reference motion")
    return sequence


def _normalize_query_static_metadata(
    metadata: Dict[str, Any], *, frame_count: int, history_length: int = 2
) -> None:
    """Correct frozen-P360 historical labels without touching motion arrays."""

    metadata.update(
        {
            "evaluator_overlap_frames": None,
            "evaluation_generated_frames": int(frame_count),
            "gt_access": "none; query-static builder never loads split",
            "runtime_opens_source_pickle": False,
            "runtime_calls_dataset_get_seq": True,
            "runtime_dataset_get_seq_source": "synthetic_static_only",
            "future_frame_2_plus_available_to_runtime": False,
            "generation_reads_motion_frames": [],
            "query_static_initial_history_frames": int(history_length),
            "generated_only_output": True,
        }
    )
    if metadata["runtime_opens_source_pickle"] is not False:
        raise P478HSIContractError("query-static metadata claims source-pickle access")
    if metadata["future_frame_2_plus_available_to_runtime"] is not False:
        raise P478HSIContractError("query-static metadata claims future-frame access")
    if metadata["generation_reads_motion_frames"] != []:
        raise P478HSIContractError("query-static metadata claims dataset motion frames")


@contextmanager
def _patched_query_static_runtime(
    args: argparse.Namespace,
    seed: Optional[QueryStaticSeed],
) -> Iterator[Dict[str, Any]]:
    """Opt-in P360 dataset boundary that never loads a LINGO motion split."""

    state: Dict[str, Any] = {}
    if seed is None:
        yield state
        return
    old_runtime_dataset = p360._runtime_dataset
    old_builder = getattr(args, "_generated_only_sequence_builder", None)
    had_builder = hasattr(args, "_generated_only_sequence_builder")

    def runtime_dataset(
        args: argparse.Namespace,
        device: torch.device,
        plan: Any,
        raw_scene: Any,
        *,
        load_data: bool = True,
    ) -> Any:
        # P360 retains a historical ``load_data`` switch for its native
        # dataset path.  In query-static mode this wrapper always routes to the
        # sealed builder below, which instantiates LINGODataset with
        # load_data=False and guards the split file against opening.  Accepting
        # the keyword is API compatibility, never permission to load motion.
        dataset = seed.build_runtime_dataset(args, device, plan, raw_scene)
        audit = getattr(dataset, "_p478_query_static_runtime_audit", None)
        if not isinstance(audit, Mapping):
            raise P478HSIContractError("query static dataset omitted its runtime audit")
        state["audit"] = {
            **dict(audit),
            "p360_load_data_argument_received": bool(load_data),
            "p360_load_data_request_ignored_due_query_static_seed": True,
            "query_static_builder_forced_load_data_false": True,
        }
        return dataset

    try:
        p360._runtime_dataset = runtime_dataset
        args._generated_only_sequence_builder = generated_only_sequence
        yield state
    finally:
        p360._runtime_dataset = old_runtime_dataset
        if had_builder:
            args._generated_only_sequence_builder = old_builder
        else:
            delattr(args, "_generated_only_sequence_builder")


def main(argv: Optional[Sequence[str]] = None) -> int:
    (
        args,
        own,
        contract,
        bridge_config,
        memory_session,
        memory_profile,
    ) = _parse_args(argv)
    query_static_seed = (
        None
        if own.query_static_seed is None
        else QueryStaticSeed.load(Path(own.query_static_seed).resolve())
    )
    p360._resolve_paths(args)
    old_ablations = p360.ABLATIONS
    try:
        p360.ABLATIONS = P478_ABLATIONS
        p360._validate_args(args)
    finally:
        p360.ABLATIONS = old_ablations
    plan = p360.p142.load_world_keypose_plan(args.plan)
    plan_seq_name_from_file = plan.seq_name
    if args.seq_name_override:
        from dataclasses import replace

        plan = replace(plan, seq_name=str(args.seq_name_override)).validate()
    p360.p142.enforce_plan_provenance(
        plan,
        allow_oracle=bool(args.allow_oracle_plan),
        allow_unknown=bool(args.allow_unknown_provenance),
    )
    packet_contact_ids = _bind_packet_contact_ids(args, plan)
    feedback, icgf_contract = _validate_p478_args(
        args, own, plan, contract
    )
    rolewise_feedback = _rolewise_feedback_config(
        own, args, plan, packet_contact_ids
    )
    contact_selective = _contact_selective_receipt(args, icgf_contract)
    p360._print_status(
        "load_and_hash_query_inputs",
        ablation=str(args.ablation),
        plan=str(Path(args.plan)),
        raw_occupancy=str(Path(args.raw_occupancy)),
    )
    raw_scene = p360.load_raw_occupancy(args, plan)
    manifest = _base_manifest(
        args,
        contract,
        plan,
        icgf_contract,
        feedback,
        memory_session,
        memory_profile,
        query_static_seed,
        own.terminal_e226_mode,
        rolewise_feedback,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "run_manifest.json", manifest)
    p360._print_status(
        "query_inputs_ready",
        ablation=str(args.ablation),
        occupancy_shape=list(raw_scene.occupancy_xyz.shape),
        plan_nodes=len(plan.nodes),
    )

    if args.dry_run:
        old_ablations = p360.ABLATIONS
        try:
            p360.ABLATIONS = P478_ABLATIONS
            receipt = p360.dry_run_receipt(args, plan, raw_scene)
        finally:
            p360.ABLATIONS = old_ablations
        receipt.update(
            {
                "schema": "p478_hsi_m1_m5_dry_run_v1",
                "bridge_config": asdict(bridge_config),
                "icgf_contract": icgf_contract,
                "contact_selective_target_policy": contact_selective,
                "feedback_contract": manifest["feedback_contract"],
                "terminal_e226_contract": manifest["terminal_e226_contract"],
                "input_provenance": manifest["input_provenance"],
                "actual_model_generation": False,
                "query_static_seed": (
                    None
                    if query_static_seed is None
                    else {
                        "path": str(query_static_seed.path),
                        "sha256": query_static_seed.seed_hash,
                        "history_source": QUERY_STATIC_HISTORY_SOURCE,
                        "lingo_sequence_lookup": False,
                        "lingo_motion_opened": False,
                        "future_or_gt_motion_used": False,
                        "generated_only_output_required": True,
                    }
                ),
            }
        )
        if rolewise_feedback.enabled:
            receipt["terminal_rolewise_feedback_contract"] = manifest[
                "terminal_rolewise_feedback_contract"
            ]
        manifest["status"] = "dry_run_complete"
        manifest["result"] = {"actual_model_generation": False}
        _write_json(args.output_dir / "dry_run_receipt.json", receipt)
        _write_json(args.output_dir / "run_manifest.json", manifest)
        print(json.dumps(_jsonable(receipt), indent=2, ensure_ascii=False))
        return 0

    for name in ("adapter_checkpoint", "base_checkpoint", "mvae_checkpoint"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(
                "--{} missing: {}".format(name.replace("_", "-"), path)
            )

    os.chdir(str(current_motion_runtime().remogen_root))
    if str(current_motion_runtime().remogen_root) not in sys.path:
        sys.path.insert(0, str(current_motion_runtime().remogen_root))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    context = _RuntimeContext(
        plan=plan,
        ablation=str(args.ablation),
        bridge_config=bridge_config,
        feedback_config=feedback,
        rolewise_feedback_config=rolewise_feedback,
        terminal_contact_joint_ids=packet_contact_ids,
        initial_history_source=(
            "native_initial_history"
            if query_static_seed is None
            else QUERY_STATIC_HISTORY_SOURCE
        ),
        terminal_e226_mode=own.terminal_e226_mode,
    )
    with _patched_query_static_runtime(args, query_static_seed) as static_state:
        with _patched_p360(context):
            sequence, metadata, generated, energy_log = p360.rollout(
                args, plan, raw_scene
            )
    query_static_runtime_audit = static_state.get("audit")
    if query_static_seed is not None and not isinstance(
        query_static_runtime_audit, Mapping
    ):
        raise P478HSIContractError(
            "query-static rollout completed without a runtime seed audit"
        )
    if query_static_seed is not None:
        sequence = _query_static_generated_only_sequence(
            generated,
            plan=plan,
            seed=query_static_seed,
            history_length=2,
            future_length=8,
        )
        generated_frame_count = int(np.asarray(generated["joints"]).shape[0])
        _normalize_query_static_metadata(
            metadata, frame_count=generated_frame_count, history_length=2
        )
        if int(sequence["joints"].shape[1]) != generated_frame_count:
            raise P478HSIContractError(
                "generated-only eval sequence is shorter than generated motion"
            )
    metadata["plan_seq_name_from_file"] = plan_seq_name_from_file
    metadata["seq_name_override"] = str(args.seq_name_override) or None
    metadata["checkpoint_policy_label"] = (
        str(args.checkpoint_policy_label) or "unspecified"
    )
    _augment_result(
        metadata,
        context,
        manifest,
        icgf_contract,
        contact_selective,
        memory_profile,
        query_static_runtime_audit,
    )

    with (args.output_dir / "eval.pkl").open("wb") as handle:
        pickle.dump([sequence], handle)
    np.savez_compressed(args.output_dir / "generated_motion.npz", **generated)
    _write_json(args.output_dir / "generation_metadata.json", metadata)
    with (args.output_dir / "energy_log.jsonl").open("w", encoding="utf-8") as handle:
        for record in energy_log:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
    _write_json(args.output_dir / "e226_receipts.json", {
        "schema": "p478.e226_receipts.v1",
        "receipts": context.cond_receipts,
    })
    feedback_receipt = {
        "schema": "p478.feedback_receipt.v1",
        "events": context.feedback_events,
        "extra_primitives_generated": context.feedback_primitives_generated,
        "existing_frames_modified": False,
    }
    if context.use_rolewise_feedback:
        feedback_receipt["terminal_rolewise_completion"] = metadata[
            "p478_terminal_rolewise_completion"
        ]
    _write_json(args.output_dir / "feedback_receipt.json", feedback_receipt)
    _write_json(args.output_dir / "run_manifest.json", manifest)
    terminal_summary = {
        "status": "complete",
        "output_dir": str(args.output_dir.resolve()),
        "ablation": args.ablation,
        "scheduler_complete": metadata["scheduler_complete"],
        "nodes_consumed": metadata["nodes_consumed"],
        "generated_frames_full": metadata["generated_frames_full"],
        "primitives_generated": metadata["primitives_generated"],
        "feedback_primitives_generated": context.feedback_primitives_generated,
        "e226_receipts": len(context.cond_receipts),
    }
    if context.use_rolewise_feedback:
        terminal_summary.update(
            {
                "terminal_rolewise_complete": bool(
                    context.rolewise_terminal_complete
                ),
                "terminal_rolewise_fail_closed": bool(
                    context.rolewise_fail_closed
                ),
                "terminal_rolewise_failure_reason": (
                    context.rolewise_failure_reason
                ),
            }
        )
    print(
        json.dumps(
            terminal_summary,
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0




