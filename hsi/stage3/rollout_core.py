#!/usr/bin/env python3
"""Run the P142 ordered rollout with query-time raw-scene guidance.

This is the executable P360 bridge.  It deliberately keeps the released
ReMoGen/P142 primitive loop and changes only the sampling call for the guided
ablations:

``M0``
    Native ReMoGen text, history, local occupancy and root-goal conditions.
``M1``
    M0 plus the P142 ordered sparse-keypose adapter.
``M2``
    M1 plus an in-denoising SDF score built from the query raw occupancy.
``M3``
    M2 plus an in-denoising query-time ICGF score for the active sparse pose.
``M4_contact_selective``
    M3 with the complete/raw SDF retained for every non-contact body joint;
    only explicitly text-routed contact joints are exempted in the terminal
    contact tail and attracted by query-time ICGF.

No memory bank, retrieved motion, post-hoc pose edit, or future ground-truth
frame is read by this module.  An oracle P142 plan is accepted only behind the
same explicit diagnostic flag used by P142.  Ground truth is attached after
generation solely for compatibility with the existing ReMoGen evaluator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch



















from hsi.stage3.model_runtime import current as current_motion_runtime
DEFAULT_BASE = DEFAULT_MVAE = DEFAULT_ADAPTER = DEFAULT_DATA = DEFAULT_SCENE = DEFAULT_CFG = None

from hsi.stage3 import plan as p142
from hsi.stage3.sdf import (  # noqa: E402
    DEFAULT_LOWER_XYZ,
    DEFAULT_UPPER_XYZ,
    QueryTimeICGF,
    RawSceneSDFZUp,
    SceneFieldError,
    select_target_component,
)
from hsi.stage3.conditioning import (  # noqa: E402
    ReMoGenRawSceneCondFn,
    SceneGuidanceConfig,
    ddpm_sample_loop_with_scene_grad,
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    use_keypose_packet: bool
    use_sdf: bool
    use_icgf: bool


ABLATIONS: Dict[str, AblationSpec] = {
    "M0": AblationSpec("M0", False, False, False),
    "M1": AblationSpec("M1", True, False, False),
    "M2": AblationSpec("M2", True, True, False),
    "M3": AblationSpec("M3", True, True, True),
    "M4_contact_selective": AblationSpec(
        "M4_contact_selective", True, True, True
    ),
}


@dataclass(frozen=True)
class RawOccupancyInput:
    native_occupancy_xyz: np.ndarray
    occupancy_xyz: np.ndarray
    original_shape_xyz: Tuple[int, int, int]
    sdf_downsample: int
    lower_xyz: Tuple[float, float, float]
    upper_xyz: Tuple[float, float, float]
    source_path: str
    source_key: str
    input_layout: str
    conversion: str
    declared_scene_name: Optional[str]

    @property
    def receipt(self) -> Dict[str, Any]:
        return {
            "schema": "p360_raw_occupancy_input_v1",
            "source_path": self.source_path,
            "source_key": self.source_key,
            "input_layout": self.input_layout,
            "conversion": self.conversion,
            "original_shape_xyz": list(self.original_shape_xyz),
            "shape_xyz": list(self.occupancy_xyz.shape),
            "sdf_downsample": int(self.sdf_downsample),
            "occupied_fraction": float(self.occupancy_xyz.mean()),
            "lower_xyz": list(self.lower_xyz),
            "upper_xyz": list(self.upper_xyz),
            "declared_scene_name": self.declared_scene_name,
            "coordinate_frame": "world_zup_xyz_m",
        }


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar string")
    return str(array.reshape(()).item())


def _load_np_occupancy(path: Path, requested_key: str) -> tuple[np.ndarray, str, Optional[str]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False), "npy", None
    if suffix != ".npz":
        raise ValueError("--raw-occupancy must be .npy or .npz")
    with np.load(path, allow_pickle=False) as payload:
        candidates = (
            (requested_key,) if requested_key else ()
        ) + ("occupancy_xyz", "occupancy", "occ", "scene_occ")
        key = next((name for name in candidates if name and name in payload.files), None)
        if key is None:
            raise KeyError(
                "occupancy NPZ needs one of {} (available: {})".format(
                    list(dict.fromkeys(candidates)), payload.files
                )
            )
        scene_name = None
        for scene_key in ("scene_name", "scene"):
            if scene_key in payload.files:
                scene_name = _scalar_string(payload[scene_key])
                break
        return np.asarray(payload[key]), str(key), scene_name


def _to_zup_xyz(
    occupancy: np.ndarray,
    *,
    layout: str,
) -> tuple[np.ndarray, str, str]:
    value = np.asarray(occupancy)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError("raw occupancy must be [X,Y,Z] or singleton [1,X,Y,Z]")
    resolved = str(layout)
    if resolved == "auto":
        if tuple(value.shape) == (300, 100, 400):
            resolved = "lingo_yup"
        elif tuple(value.shape) == (300, 400, 100):
            resolved = "xyz_zup"
        else:
            # The scene convention has a substantially shorter vertical axis.
            # Refuse ambiguous shapes rather than silently swapping a valid map.
            smallest = int(np.argmin(value.shape))
            if smallest == 2:
                resolved = "xyz_zup"
            elif smallest == 1:
                resolved = "lingo_yup"
            else:
                raise ValueError(
                    "cannot infer occupancy layout from shape {}; pass "
                    "--occupancy-layout explicitly".format(tuple(value.shape))
                )
    if resolved == "xyz_zup":
        converted = value
        conversion = "identity_xyz_zup"
    elif resolved == "lingo_yup":
        # ReMoGen's official conversion: [X,Y(vertical),Z] -> [X,-Z,Y].
        converted = np.transpose(value, (0, 2, 1))[:, ::-1, :]
        conversion = "transpose_0_2_1_then_flip_new_y"
    else:
        raise ValueError("occupancy layout must be auto, xyz_zup, or lingo_yup")
    if np.issubdtype(converted.dtype, np.floating):
        if not np.isfinite(converted).all():
            raise ValueError("raw occupancy contains non-finite values")
        converted = converted > 0.5
    else:
        converted = converted.astype(bool, copy=False)
    return np.ascontiguousarray(converted), resolved, conversion


def _conservative_downsample(occupancy_xyz: np.ndarray, factor: int) -> np.ndarray:
    """Max-pool occupancy while preserving the world bounds.

    The official LINGO field is 2 cm and expensive to transform repeatedly.
    A factor of two produces a conservative 4 cm collision field: a pooled
    cell is occupied whenever any source cell is occupied.
    """

    factor = int(factor)
    if factor < 1:
        raise ValueError("sdf-downsample must be positive")
    value = np.asarray(occupancy_xyz, dtype=bool)
    if factor == 1:
        return np.ascontiguousarray(value)
    shape = np.asarray(value.shape, dtype=np.int64)
    pooled_shape = (shape + factor - 1) // factor
    padded_shape = pooled_shape * factor
    pad = tuple((0, int(padded_shape[axis] - shape[axis])) for axis in range(3))
    padded = np.pad(value, pad, mode="constant", constant_values=False)
    reshaped = padded.reshape(
        int(pooled_shape[0]), factor,
        int(pooled_shape[1]), factor,
        int(pooled_shape[2]), factor,
    )
    return np.ascontiguousarray(reshaped.any(axis=(1, 3, 5)))


def load_raw_occupancy(args: argparse.Namespace, plan: p142.WorldKeyposePlan) -> RawOccupancyInput:
    path = Path(args.raw_occupancy)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw, key, declared_scene = _load_np_occupancy(path, str(args.occupancy_key))
    occupancy, layout, conversion = _to_zup_xyz(
        raw, layout=str(args.occupancy_layout)
    )
    original_shape = tuple(int(value) for value in occupancy.shape)
    native_occupancy = occupancy
    occupancy = _conservative_downsample(
        native_occupancy, int(args.sdf_downsample)
    )
    if (
        declared_scene is not None
        and declared_scene != plan.scene_name
        and not args.allow_occupancy_scene_mismatch
    ):
        raise ValueError(
            "occupancy declares scene {!r}, but plan uses {!r}".format(
                declared_scene, plan.scene_name
            )
        )
    return RawOccupancyInput(
        native_occupancy_xyz=native_occupancy,
        occupancy_xyz=occupancy,
        original_shape_xyz=original_shape,
        sdf_downsample=int(args.sdf_downsample),
        lower_xyz=tuple(float(value) for value in args.scene_lower),
        upper_xyz=tuple(float(value) for value in args.scene_upper),
        source_path=str(path.resolve()),
        source_key=key,
        input_layout=layout,
        conversion=conversion,
        declared_scene_name=declared_scene,
    )


def _parse_joint_ids(value: str) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return ()
    result = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if len(set(result)) != len(result) or any(item < 0 or item >= 22 for item in result):
        raise ValueError("joint IDs must be unique comma-separated integers in [0,21]")
    return result


def _contact_joint_ids(args: argparse.Namespace, node: p142.WorldKeyposeNode) -> tuple[int, ...]:
    explicit = _parse_joint_ids(args.icgf_contact_joints)
    if explicit:
        return explicit
    return tuple(int(value) for value in np.flatnonzero(node.joint_mask))


def build_active_icgf(
    args: argparse.Namespace,
    node: p142.WorldKeyposeNode,
    raw_scene: RawOccupancyInput,
    scene_sdf: RawSceneSDFZUp | None = None,
) -> tuple[Optional[QueryTimeICGF], tuple[int, ...], Dict[str, Any]]:
    """Build the active node field without consulting a memory or motion bank."""

    source = str(args.icgf_source)
    contact_ids = _contact_joint_ids(args, node)
    if source == "auto":
        source = "sparse_joint_targets" if bool(node.joint_mask.any()) else "none"
    if source == "none":
        return None, (), {
            "source": "none",
            "reason": "active node has no declared sparse contact joints",
            "retrieval_used": False,
        }
    if not contact_ids:
        raise ValueError(
            "ICGF source {} needs node joints or --icgf-contact-joints".format(source)
        )
    projection_receipt: Dict[str, Any] = {}
    if source in ("sparse_joint_targets", "surface_projected_sparse_joint_targets"):
        # A joint-specific field can be queried only for rows declared in the
        # plan.  An explicit list may safely select a subset, never add targets.
        declared = set(int(value) for value in np.flatnonzero(node.joint_mask))
        missing = [value for value in contact_ids if value not in declared]
        if missing:
            raise ValueError(
                "sparse ICGF contact joints are absent from active node: {}".format(missing)
            )
        targets = np.asarray(node.joint_xyz_world, dtype=np.float32).copy()
        target_mask = np.zeros_like(node.joint_mask, dtype=bool)
        target_mask[np.asarray(contact_ids, dtype=np.int64)] = True
        if source == "surface_projected_sparse_joint_targets":
            if scene_sdf is None:
                raise ValueError("surface-projected ICGF requires the raw scene SDF")
            if scene_sdf.target_removed:
                raise ValueError(
                    "surface-projected ICGF requires the complete raw scene SDF"
                )
            source_points = torch.as_tensor(
                targets[np.asarray(contact_ids, dtype=np.int64)], dtype=torch.float32
            ).unsqueeze(0)
            projection = scene_sdf.project_to_level_set(
                source_points,
                target_level_m=float(args.icgf_surface_level),
                tolerance_m=float(args.icgf_surface_projection_tolerance),
                maximum_iterations=int(args.icgf_surface_projection_max_iterations),
                maximum_step_m=float(args.icgf_surface_projection_max_step),
                maximum_displacement_m=float(
                    args.icgf_surface_projection_max_displacement
                ),
            )
            if not bool(projection.converged.all()):
                raise SceneFieldError(
                    "one or more sparse contact targets did not converge to the "
                    "requested raw-SDF level set"
                )
            projected = projection.points_world_zup[0].detach().cpu().numpy()
            targets[np.asarray(contact_ids, dtype=np.int64)] = projected
            projection_receipt = {
                "projection_schema": "p362_query_raw_sdf_level_set_projection_v1",
                "target_level_m": float(projection.target_level_m),
                "tolerance_m": float(projection.tolerance_m),
                "iterations": int(projection.iterations),
                "maximum_iterations": int(args.icgf_surface_projection_max_iterations),
                "maximum_step_m": float(args.icgf_surface_projection_max_step),
                "maximum_displacement_m": float(
                    args.icgf_surface_projection_max_displacement
                ),
                "initial_points_world_zup": source_points[0].tolist(),
                "projected_points_world_zup": projected.tolist(),
                "initial_signed_distance_m": projection.initial_signed_distance_m[
                    0
                ].tolist(),
                "final_signed_distance_m": projection.final_signed_distance_m[
                    0
                ].tolist(),
                "displacement_m": projection.displacement_m[0].tolist(),
                "converged": projection.converged[0].tolist(),
                "raw_scene_sdf_target_component_removed": False,
                "memory_used": False,
                "future_motion_read": False,
            }
        icgf = QueryTimeICGF.from_sparse_joint_targets(
            targets,
            target_mask,
            kernel_sigma_m=args.icgf_kernel_sigma,
            source=(
                "query_raw_sdf_surface_projected_sparse_targets"
                if source == "surface_projected_sparse_joint_targets"
                else "planner_sparse_joint_targets"
            ),
        )
    elif source == "target_point":
        icgf = QueryTimeICGF.from_target_point(
            node.root_xyz_yaw[:3], kernel_sigma_m=args.icgf_kernel_sigma
        )
    elif source == "target_component":
        component = select_target_component(
            raw_scene.occupancy_xyz,
            node.root_xyz_yaw[:3],
            lower_xyz=raw_scene.lower_xyz,
            upper_xyz=raw_scene.upper_xyz,
            floor_ignore_height_m=args.floor_ignore_height,
            maximum_seed_distance_m=args.maximum_target_seed_distance,
            search_crop_radius_m=getattr(args, "target_component_crop_radius", 0.75),
            maximum_component_fraction=getattr(
                args, "maximum_target_component_fraction", 0.25
            ),
        )
        icgf = QueryTimeICGF.from_target_component(
            component.target_occupancy_xyz,
            anchor_point_world_zup=node.root_xyz_yaw[:3],
            lower_xyz=raw_scene.lower_xyz,
            upper_xyz=raw_scene.upper_xyz,
            local_radius_m=args.icgf_component_radius,
            normal_offset_m=args.icgf_normal_offset,
            maximum_points=args.icgf_maximum_points,
            kernel_sigma_m=args.icgf_kernel_sigma,
        )
    else:
        raise ValueError(
            "icgf source must be auto, none, sparse_joint_targets, target_point, "
            "surface_projected_sparse_joint_targets, or target_component"
        )
    return icgf, contact_ids, {
        **icgf.receipt,
        "resolved_source": source,
        "contact_joint_ids": list(contact_ids),
        "active_external_ordinal": int(node.ordinal),
        "active_node_policy": str(args.icgf_active_node_policy),
        **projection_receipt,
    }


def _tensor_tree_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensor_tree_cpu(item) for key, item in value.items()}
    return value


def _print_status(stage: str, **payload: Any) -> None:
    print(
        json.dumps({"stage": stage, **payload}, ensure_ascii=False),
        flush=True,
    )


def _aggregate_energy(records: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    if not records:
        return {"calls": 0, "guidance_applied": False}
    keys = sorted(
        set.intersection(*(set(record.keys()) for record in records))
    )
    summary: Dict[str, Any] = {
        "calls": len(records),
        "guidance_applied": True,
    }
    for key in keys:
        try:
            values = np.asarray(
                [record[key] for record in records], dtype=np.float64
            )
        except (TypeError, ValueError):
            # Audit records may also contain categorical sampler provenance
            # such as ``conditioning_strategy``; only scalar numeric fields
            # belong in the aggregate statistics.
            continue
        if np.isfinite(values).all():
            summary[key + "_mean"] = float(values.mean())
            summary[key + "_last"] = float(values[-1])
    return summary


def _runtime_dataset(
    args: argparse.Namespace,
    device: torch.device,
    plan: p142.WorldKeyposePlan,
    raw_scene: RawOccupancyInput,
):
    from data_loaders.humanml.data.dataset_lingo import LINGODataset

    dataset = LINGODataset(
        dataset_path=str(args.data_dir),
        dataset_name="lingo",
        cfg_path=str(args.cfg_path),
        enforce_gender="neutral",
        enforce_zero_beta=1,
        body_type="smplx",
        split=args.split,
        device=device,
        scene_root=str(args.scene_root),
        # Loading a split would cache every scene on the GPU.  P360 has an
        # explicit query scene, so inject only that same raw field below.
        load_scene=False,
    )
    expected_shape = tuple(int(value) for value in dataset.scene_grid[6:].tolist())
    if tuple(raw_scene.native_occupancy_xyz.shape) != expected_shape:
        raise ValueError(
            "native ReMoGen occupancy expects {}, got {}; downsampling is only "
            "allowed for the auxiliary SDF".format(
                expected_shape, tuple(raw_scene.native_occupancy_xyz.shape)
            )
        )
    dataset.scene_embedding_dict[plan.scene_name] = {
        "occ": torch.from_numpy(raw_scene.native_occupancy_xyz).to(device=device),
        "source": "p360_cli_raw_occupancy",
        "basename": Path(raw_scene.source_path).name,
    }
    return dataset


def _make_scene_config(args: argparse.Namespace, spec: AblationSpec) -> SceneGuidanceConfig:
    return SceneGuidanceConfig(
        sdf_weight=float(args.sdf_weight) if spec.use_sdf else 0.0,
        icgf_weight=float(args.icgf_weight) if spec.use_icgf else 0.0,
        collision_clearance_m=float(args.collision_clearance),
        icgf_deadzone_m=float(args.icgf_deadzone),
        icgf_huber_beta_m=float(args.icgf_huber_beta),
        contact_frames=int(args.contact_frames),
        exempt_contact_joints_from_sdf_tail=bool(args.exempt_contact_sdf_tail),
        guidance_scale=float(args.scene_guidance_scale),
        max_grad_norm=float(args.max_grad_norm),
        denoise_start_fraction=float(args.denoise_start_fraction),
        schedule_power=float(args.schedule_power),
        pcgrad_goal=bool(args.pcgrad_goal),
        goal_tolerance_m=float(args.goal_tolerance),
        guidance_mode=str(args.scene_guidance_mode),
        normalize_scene_gradient=bool(args.normalize_scene_gradient),
        clean_latent_step_size=float(args.clean_latent_step_size),
        max_clean_latent_update_norm=float(
            args.max_clean_latent_update_norm
        ),
    ).validate()


def _exclude_target_component(args: argparse.Namespace, spec: AblationSpec) -> bool:
    explicit = getattr(args, "exclude_target_component_from_sdf", None)
    if explicit is None:
        # M3 has an explicit contact field.  Its declared target component must
        # not simultaneously repel the body as a collision obstacle.
        return bool(spec.name == "M3")
    return bool(explicit)


def _contact_selective_contract(
    args: argparse.Namespace, spec: AblationSpec
) -> Dict[str, Any]:
    """Receipt for M4's joint-selective target-contact policy."""

    enabled = spec.name == "M4_contact_selective"
    contact_ids = _parse_joint_ids(args.icgf_contact_joints) if enabled else ()
    collision_ids = _parse_joint_ids(args.collision_joint_ids) if enabled else ()
    return {
        "enabled": enabled,
        "schema": "p360_m4_contact_selective_contract_v1",
        "full_raw_sdf_keeps_target_component": bool(
            enabled and not _exclude_target_component(args, spec)
        ),
        "collision_joint_ids": (
            list(collision_ids) if collision_ids else list(range(22))
        ),
        "text_routed_contact_joint_ids": list(contact_ids),
        "contact_joint_sdf_exemption": (
            "terminal contact tail only" if enabled else "not_applicable"
        ),
        "contact_frames": int(getattr(args, "contact_frames", 4)),
        "noncontact_joint_target_component_exemption": False,
        "icgf_active_node_policy": str(args.icgf_active_node_policy),
        "memory_used": False,
        "retrieval_used": False,
    }


@torch.no_grad()
def rollout(
    args: argparse.Namespace,
    plan: p142.WorldKeyposePlan,
    raw_scene: RawOccupancyInput,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray], list[Dict[str, Any]]]:
    """Generate a causal ordered rollout and record every denoising energy."""

    from mld.train_hsi_module_adapter import create_gaussian_diffusion

    spec = ABLATIONS[str(args.ablation)]
    requested = str(args.device)
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")

    _print_status("load_models", ablation=spec.name, device=str(device))
    denoiser_args, denoiser, vae = p142.load_remogen_models(
        args.base_checkpoint, args.adapter_checkpoint, args.mvae_checkpoint, device
    )
    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = str(args.respacing)
    diffusion = create_gaussian_diffusion(diffusion_args)
    if spec.use_sdf and str(args.respacing):
        raise ValueError("P360 guided rollout currently supports DDPM; leave --respacing empty")

    _print_status("load_single_scene_dataset", scene_name=plan.scene_name)
    dataset = _runtime_dataset(args, device, plan, raw_scene)
    index_by_name = {
        item["seq_name"]: index for index, item in enumerate(dataset.dataset)
    }
    if plan.seq_name not in index_by_name:
        raise KeyError("{} is not in LINGO {} split".format(plan.seq_name, args.split))
    sample = dataset.get_seq(index_by_name[plan.seq_name])
    sample_scene = str(sample.get("scene_names", ""))
    if sample_scene != plan.scene_name:
        raise ValueError(
            "plan scene {} != seed scene {}".format(plan.scene_name, sample_scene)
        )
    if plan.text not in dataset.text_embedding_dict:
        dataset.update_text_embedding_dict([plan.text])
    text_embedding = dataset.text_embedding_dict[plan.text].reshape(1, -1)

    history_length = int(dataset.history_length)
    future_length = int(dataset.future_length)
    if history_length != 2 or future_length != 8:
        raise RuntimeError(
            "P142 was trained for H=2,F=8, got H={},F={}".format(
                history_length, future_length
            )
        )
    # Generation boundary: retain only the two native history frames.
    normalized = sample.pop("motion_tensor_normalized")
    history_motion = (
        normalized[..., :history_length].squeeze(2).permute(0, 2, 1).clone()
    )
    del normalized
    global_rotmat = sample["transf_rotmat"]
    global_transl = sample["transf_transl"]
    gender = sample["gender"][0]
    primitive_length = history_length + future_length
    first_beta = sample["betas"][:, :1].to(device)
    betas = first_beta.expand(1, primitive_length, -1).clone()
    utility = dataset.primitive_utility
    pelvis_delta = utility.calc_calibrate_offset(
        {"betas": betas[:, 0], "gender": gender}
    )

    exclude_target_component = _exclude_target_component(args, spec)
    target_component_point = (
        plan.nodes[-1].root_xyz_yaw[:3]
        if (spec.name == "M3" or exclude_target_component)
        else None
    )
    _print_status(
        "build_raw_scene_sdf",
        shape=list(raw_scene.occupancy_xyz.shape),
        exclude_target_component=exclude_target_component,
    )
    scene_sdf = RawSceneSDFZUp.from_occupancy(
        raw_scene.occupancy_xyz,
        lower_xyz=raw_scene.lower_xyz,
        upper_xyz=raw_scene.upper_xyz,
        floor_ignore_height_m=args.floor_ignore_height,
        target_point_world_zup=target_component_point,
        remove_target_component=exclude_target_component,
        maximum_target_seed_distance_m=args.maximum_target_seed_distance,
        target_component_search_crop_radius_m=getattr(
            args, "target_component_crop_radius", 0.75
        ),
        maximum_target_component_fraction=getattr(
            args, "maximum_target_component_fraction", 0.25
        ),
        source="p360_cli_raw_occupancy",
    )
    current_rotmat = torch.eye(3, device=device).unsqueeze(0)
    current_transl = torch.zeros(1, 1, 3, device=device)
    grid = dataset.create_meshgrid(batch_size=1).to(device)
    scheduler = p142.OrderedKeyposeScheduler(
        plan,
        p142.CompletionThresholds(
            root_xy_m=args.root_xy_threshold,
            root_z_m=args.root_z_threshold,
            yaw_rad=math.radians(args.yaw_threshold_deg),
            joint_m=args.joint_threshold,
        ),
        max_attempts_per_node=args.max_attempts_per_node,
    )
    motion_sequences = None
    primitive_records: list[Dict[str, Any]] = []
    energy_log: list[Dict[str, Any]] = []
    completion_primitive = None
    post_remaining = int(args.post_completion_primitives)
    scene_config = _make_scene_config(args, spec)
    collision_joint_ids = _parse_joint_ids(args.collision_joint_ids)
    if not collision_joint_ids:
        collision_joint_ids = tuple(range(22))

    for primitive_id in range(int(args.max_primitives)):
        if scheduler.complete:
            if completion_primitive is None:
                completion_primitive = primitive_id
            # A permissive completion threshold can consume nearby sparse
            # nodes in the first primitive.  Keep the historical default
            # (minimum=0), while allowing qualitative full-action runs to
            # require a minimum causal rollout length without duplicating,
            # holding or retiming any generated frame.
            if primitive_id >= int(args.minimum_primitives):
                if post_remaining <= 0:
                    break
                post_remaining -= 1

        _print_status(
            "primitive_start",
            primitive_id=int(primitive_id),
            next_node_index=int(scheduler.next_node_index),
        )
        history_denorm = dataset.denormalize(history_motion)
        history_features = utility.tensor_to_dict(history_denorm)
        root_local = history_features["joints"][:, -1, :3]
        current_root_world = p142.active_local_points_to_world(
            root_local[:, None],
            global_rotmat,
            global_transl,
            current_rotmat,
            current_transl,
        )[:, 0]
        active_index = min(scheduler.next_node_index, len(plan.nodes) - 1)
        active_node = plan.nodes[active_index]
        goal_world = torch.as_tensor(
            active_node.root_xyz_yaw[:3],
            device=device,
            dtype=history_motion.dtype,
        ).reshape(1, 3)
        target_local = p142.world_points_to_active_local(
            goal_world[:, None],
            global_rotmat,
            global_transl,
            current_rotmat,
            current_transl,
        )[:, 0]
        occ = p142._build_occ_two(
            dataset,
            grid,
            plan.scene_name,
            global_rotmat,
            current_rotmat,
            current_root_world,
            goal_world,
        )
        packet = None
        if spec.use_keypose_packet and not scheduler.complete:
            packet = p142.pending_plan_to_local_packet(
                plan,
                scheduler.next_node_index,
                args.maximum_nodes,
                global_rotmat,
                global_transl,
                current_rotmat,
                current_transl,
                dtype=history_motion.dtype,
            )
        guidance = torch.ones(
            1, *denoiser_args.model_args.noise_shape, device=device
        ) * float(args.guidance_param)
        control_input = {
            "text_embedding": text_embedding,
            "history_motion_normalized": history_motion,
            "occ": occ,
            "target_pelvis": target_local,
            "keypose_packet": packet,
        }
        y = {
            "text_embedding": text_embedding,
            "history_motion_normalized": history_motion,
            "scale": guidance,
            "control_input": control_input,
        }
        noise = torch.zeros_like(guidance) if args.zero_noise else None

        cond_fn = None
        icgf_receipt: Dict[str, Any] = {
            "source": "disabled_by_ablation",
            "retrieval_used": False,
        }
        contact_ids: tuple[int, ...] = ()
        if spec.use_sdf:
            icgf = None
            icgf_active_for_node = bool(
                spec.use_icgf
                and (
                    str(args.icgf_active_node_policy) == "all"
                    or active_index == len(plan.nodes) - 1
                )
            )
            if icgf_active_for_node:
                icgf, contact_ids, icgf_receipt = build_active_icgf(
                    args, active_node, raw_scene, scene_sdf
                )
            elif spec.use_icgf:
                icgf_receipt = {
                    "source": "inactive_by_active_node_policy",
                    "active_node_policy": str(args.icgf_active_node_policy),
                    "active_node_index": int(active_index),
                    "terminal_node_index": int(len(plan.nodes) - 1),
                    "retrieval_used": False,
                }
            cond_fn = ReMoGenRawSceneCondFn(
                vae_model=vae,
                dataset=dataset,
                history_motion=history_motion,
                scene_sdf=scene_sdf,
                icgf=icgf,
                current_rotmat=current_rotmat,
                current_transl=current_transl,
                global_rotmat=global_rotmat,
                global_transl=global_transl,
                future_length=future_length,
                scale_latent=denoiser_args.rescale_latent,
                num_diffusion_steps=int(diffusion.num_timesteps),
                contact_joint_ids=contact_ids,
                collision_joint_ids=collision_joint_ids,
                goal_world_zup=goal_world,
                config=scene_config,
            )
            sampled = ddpm_sample_loop_with_scene_grad(
                diffusion,
                denoiser,
                (1, *denoiser_args.model_args.noise_shape),
                cond_fn=cond_fn,
                model_kwargs={"y": y},
                noise=noise,
                clip_denoised=False,
            )
        else:
            sample_fn = (
                diffusion.p_sample_loop
                if not args.respacing
                else diffusion.ddim_sample_loop
            )
            sampled = sample_fn(
                denoiser,
                (1, *denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={"y": y},
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=noise,
                const_noise=False,
            )
        latent = sampled.permute(1, 0, 2)
        future_normalized = vae.decode(
            latent,
            history_motion,
            nfuture=future_length,
            scale_latent=denoiser_args.rescale_latent,
        )
        future_only = dataset.denormalize(future_normalized)
        all_frames = torch.cat((history_denorm, future_only), dim=1)
        output_frames = all_frames if primitive_id == 0 else future_only
        output_features = utility.tensor_to_dict(output_frames)
        output_features.update(
            {
                "transf_rotmat": current_rotmat,
                "transf_transl": current_transl,
                "gender": gender,
                "betas": betas[:, : output_frames.shape[1]],
                "pelvis_delta": pelvis_delta,
            }
        )
        primitive_initial = utility.feature_dict_to_smpl_dict(output_features)
        primitive_initial = utility.transform_primitive_to_world(primitive_initial)
        if motion_sequences is None:
            motion_sequences = primitive_initial
        else:
            for key in ("transl", "global_orient", "body_pose", "betas", "joints"):
                motion_sequences[key] = torch.cat(
                    (motion_sequences[key], primitive_initial[key]), dim=1
                )

        future_features = utility.tensor_to_dict(future_only)
        future_joints_local = future_features["joints"].reshape(
            1, future_length, 22, 3
        )
        future_joints_world_tensor = p142.active_local_points_to_world(
            future_joints_local,
            global_rotmat,
            global_transl,
            current_rotmat,
            current_transl,
        )
        future_joints_world = (
            future_joints_world_tensor[0].detach().cpu().numpy()
        )
        events = scheduler.observe_primitive(future_joints_world, primitive_id)
        if scheduler.complete and completion_primitive is None:
            completion_primitive = primitive_id

        step_records: list[Dict[str, Any]] = []
        if cond_fn is not None:
            if len(cond_fn.records) != int(diffusion.num_timesteps):
                raise RuntimeError(
                    "scene cond_fn must run once per DDPM step: expected {}, got {}"
                    .format(diffusion.num_timesteps, len(cond_fn.records))
                )
            for call_index, record in enumerate(cond_fn.records):
                item = {
                    "record_type": "denoise_energy",
                    "ablation": spec.name,
                    "primitive_id": int(primitive_id),
                    "active_node_index": int(active_index),
                    "active_external_ordinal": int(active_node.ordinal),
                    "call_index": int(call_index),
                    **record,
                }
                step_records.append(item)
                energy_log.append(item)
        else:
            # Baselines have no scene energy by construction.  Keep an honest
            # per-step audit trail rather than inventing a post-hoc energy.
            for call_index, timestep in enumerate(
                range(int(diffusion.num_timesteps) - 1, -1, -1)
            ):
                item = {
                    "record_type": "denoise_energy_disabled",
                    "ablation": spec.name,
                    "primitive_id": int(primitive_id),
                    "active_node_index": int(active_index),
                    "active_external_ordinal": int(active_node.ordinal),
                    "call_index": int(call_index),
                    "timestep": int(timestep),
                    "condition_fn_invoked": False,
                    "scene_energy": None,
                }
                step_records.append(item)
                energy_log.append(item)
        energy_summary = _aggregate_energy(cond_fn.records if cond_fn is not None else [])
        primitive_records.append(
            {
                "primitive_id": primitive_id,
                "active_node_index": int(active_index),
                "active_external_ordinal": int(active_node.ordinal),
                "active_goal_world_xyz": active_node.root_xyz_yaw[:3].tolist(),
                "current_root_world_xyz": current_root_world[0].detach().cpu().tolist(),
                "keypose_packet_enabled": bool(packet is not None),
                "sdf_guidance_enabled": bool(spec.use_sdf),
                "icgf_guidance_enabled": bool(spec.use_icgf and contact_ids),
                "collision_joint_ids": list(collision_joint_ids),
                "active_contact_joint_ids": list(contact_ids),
                "contact_selective_target_policy": _contact_selective_contract(
                    args, spec
                ),
                "condition_fn_invocations": int(
                    len(cond_fn.records) if cond_fn is not None else 0
                ),
                "ddpm_steps": int(diffusion.num_timesteps),
                "energy_log_records": len(step_records),
                "icgf_receipt": icgf_receipt,
                "energy_summary": energy_summary,
                "scheduler_events": events,
                "pending_after": int(len(plan.nodes) - scheduler.next_node_index),
            }
        )
        _print_status(
            "primitive_complete",
            primitive_id=int(primitive_id),
            condition_fn_invocations=int(
                len(cond_fn.records) if cond_fn is not None else 0
            ),
            nodes_consumed=int(scheduler.next_node_index),
        )

        new_history = all_frames[:, -history_length:]
        history_feature = utility.tensor_to_dict(new_history)
        history_feature.update(
            {
                "transf_rotmat": current_rotmat,
                "transf_transl": current_transl,
                "gender": gender,
                "betas": betas[:, :history_length],
                "pelvis_delta": pelvis_delta,
            }
        )
        canonical_history, blended = utility.get_blended_feature(
            history_feature, use_predicted_joints=args.use_predicted_joints
        )
        current_rotmat = canonical_history["transf_rotmat"]
        current_transl = canonical_history["transf_transl"]
        history_motion = dataset.normalize(utility.dict_to_tensor(blended))

    if motion_sequences is None:
        raise RuntimeError("rollout produced no primitive")
    generated_initial = dict(motion_sequences)
    generated_initial.update(
        {
            "transf_rotmat": global_rotmat,
            "transf_transl": global_transl,
            "pelvis_delta": pelvis_delta,
        }
    )
    generated_world = utility.transform_primitive_to_world(generated_initial)
    generated_frames = int(generated_world["joints"].shape[1])

    # Evaluation-only boundary.  Historical runs retain the exact old GT
    # serialization path.  A query-owned static-history wrapper may instead
    # inject a generated-only builder; in that opt-in mode this function never
    # indexes ``sample['motion_canonicalized']`` or any future motion field.
    generated_only_builder = getattr(args, "_generated_only_sequence_builder", None)
    if generated_only_builder is None:
        gt_canonical = dict(sample["motion_canonicalized"])
        gt_canonical.update(
            {
                "transf_rotmat": global_rotmat,
                "transf_transl": global_transl,
                "gender": gender,
                "pelvis_delta": pelvis_delta,
            }
        )
        gt_world = utility.transform_primitive_to_world(gt_canonical)
        gt_frames_available = int(gt_world["joints"].shape[1] - 1)
        eval_frames = min(generated_frames, gt_frames_available)
        sequence = {
            "texts": plan.text,
            "gender": generated_world["gender"],
            "betas": generated_world["betas"][:, :eval_frames],
            "transl": generated_world["transl"][:, :eval_frames],
            "global_orient": generated_world["global_orient"][:, :eval_frames],
            "body_pose": generated_world["body_pose"][:, :eval_frames],
            "joints": generated_world["joints"][:, :eval_frames],
            "history_length": history_length,
            "future_length": future_length,
            "replicate_times": 1,
            "scene_names": plan.scene_name,
            "seq_name": plan.seq_name,
            "gt_transl": gt_world["transl"][0, :eval_frames],
            "gt_betas": gt_world["betas"][0, :eval_frames],
            "gt_global_orient": gt_world["global_orient"][0, :eval_frames],
            "gt_body_pose": gt_world["body_pose"][0, :eval_frames],
            "gt_joints": gt_world["joints"][0, :eval_frames].reshape(-1, 22, 3),
        }
        gt_access = "serialization_and_metrics_only_after_generation"
    else:
        if not callable(generated_only_builder):
            raise TypeError("_generated_only_sequence_builder must be callable")
        eval_frames = generated_frames
        sequence = generated_only_builder(
            generated_world,
            plan=plan,
            history_length=history_length,
            future_length=future_length,
        )
        if any(str(key).startswith("gt_") for key in sequence):
            raise RuntimeError("generated-only sequence builder emitted a gt_* field")
        gt_access = "none_query_static_generated_only"
    sequence = _tensor_tree_cpu(sequence)
    full_npz = {
        key: generated_world[key][0].detach().cpu().numpy()
        for key in ("transl", "global_orient", "body_pose", "betas", "joints")
    }
    metadata = {
        "schema": "p360_guided_ordered_rollout_v1",
        "ablation": asdict(spec),
        "seq_name": plan.seq_name,
        "scene_name": plan.scene_name,
        "text": plan.text,
        "plan_provenance": plan.provenance,
        "oracle_keypose_upper_bound": p142._is_oracle_provenance(plan.provenance),
        "plan_nodes": len(plan.nodes),
        "nodes_consumed": scheduler.next_node_index,
        "scheduler_complete": scheduler.complete,
        "generated_frames_full": generated_frames,
        "seed": int(args.seed),
        "minimum_primitives": int(args.minimum_primitives),
        "evaluator_overlap_frames": eval_frames,
        "primitives_generated": len(primitive_records),
        "completion_primitive": completion_primitive,
        "primitive_records": primitive_records,
        "scheduler_events": scheduler.events,
        "raw_occupancy": raw_scene.receipt,
        "raw_scene_sdf": scene_sdf.receipt,
        "exclude_target_component_from_sdf": exclude_target_component,
        "target_component_anchor_world_xyz": (
            None
            if target_component_point is None
            else [float(value) for value in target_component_point]
        ),
        "target_component_selection_policy": {
            "scope": "euclidean_query_local_connected_component",
            "search_crop_radius_m": float(
                getattr(args, "target_component_crop_radius", 0.75)
            ),
            "maximum_fraction_of_nonfloor_scene": float(
                getattr(args, "maximum_target_component_fraction", 0.25)
            ),
        },
        "contact_selective_target_policy": _contact_selective_contract(args, spec),
        "scene_guidance_config": asdict(scene_config),
        "energy_log_records": len(energy_log),
        "ddpm_steps_per_primitive": int(diffusion.num_timesteps),
        "generation_reads_motion_frames": [0, 1],
        "generator_reads_future_root_directly_from_dataset": False,
        "external_plan_contains_future_gt": p142._is_oracle_provenance(plan.provenance),
        "generator_reads_gt_duration": False,
        "gt_access": gt_access,
        "length_policy": "scheduler_completion_or_cli_max_primitives",
        "scheduler_thresholds": {
            "root_xy_m": float(args.root_xy_threshold),
            "root_z_m": float(args.root_z_threshold),
            "yaw_deg": float(args.yaw_threshold_deg),
            "selected_joint_m": float(args.joint_threshold),
        },
        "minimum_primitives": int(args.minimum_primitives),
        "memory_used": False,
        "retrieval_used": False,
        "posthoc_motion_edit_used": False,
        "guidance_location": "inside_each_primitive_ddpm_denoising",
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "adapter_checkpoint": str(Path(args.adapter_checkpoint).resolve()),
        "mvae_checkpoint": str(Path(args.mvae_checkpoint).resolve()),
    }
    if generated_only_builder is not None:
        metadata["generation_reads_motion_frames"] = []
        metadata["query_static_initial_history"] = True
        metadata["lingo_sequence_lookup"] = False
        metadata["lingo_motion_opened"] = False
    return sequence, metadata, full_npz, energy_log


def dry_run_receipt(
    args: argparse.Namespace,
    plan: p142.WorldKeyposePlan,
    raw_scene: RawOccupancyInput,
) -> Dict[str, Any]:
    spec = ABLATIONS[str(args.ablation)]
    identity = torch.eye(3).unsqueeze(0)
    zero = torch.zeros(1, 1, 3)
    packet = p142.pending_plan_to_local_packet(
        plan, 0, args.maximum_nodes, identity, zero, identity, zero
    )
    exclude_target_component = _exclude_target_component(args, spec)
    target_component_point = (
        plan.nodes[-1].root_xyz_yaw[:3]
        if (spec.name == "M3" or exclude_target_component)
        else None
    )
    field = RawSceneSDFZUp.from_occupancy(
        raw_scene.occupancy_xyz,
        lower_xyz=raw_scene.lower_xyz,
        upper_xyz=raw_scene.upper_xyz,
        floor_ignore_height_m=args.floor_ignore_height,
        target_point_world_zup=target_component_point,
        remove_target_component=exclude_target_component,
        maximum_target_seed_distance_m=args.maximum_target_seed_distance,
        target_component_search_crop_radius_m=getattr(
            args, "target_component_crop_radius", 0.75
        ),
        maximum_target_component_fraction=getattr(
            args, "maximum_target_component_fraction", 0.25
        ),
        source="p360_cli_raw_occupancy_dry_run",
    )
    icgf_receipt: Dict[str, Any] = {
        "source": "disabled_by_ablation",
        "retrieval_used": False,
    }
    icgf_node_index = None
    if spec.use_icgf:
        icgf_node_index = (
            0
            if str(args.icgf_active_node_policy) == "all"
            else len(plan.nodes) - 1
        )
        _, _, icgf_receipt = build_active_icgf(
            args, plan.nodes[icgf_node_index], raw_scene, field
        )
    return {
        "status": "dry_run_ok",
        "schema": "p360_guided_ordered_rollout_dry_run_v1",
        "ablation": asdict(spec),
        "plan": {
            "seq_name": plan.seq_name,
            "scene_name": plan.scene_name,
            "text": plan.text,
            "provenance": plan.provenance,
            "nodes": len(plan.nodes),
            "packet_valid_nodes": int(packet.node_mask.sum()),
            "packet_valid_joint_tokens": int(packet.joint_mask.sum()),
        },
        "raw_occupancy": raw_scene.receipt,
        "raw_scene_sdf": field.receipt,
        "exclude_target_component_from_sdf": exclude_target_component,
        "target_component_selection_policy": {
            "scope": "euclidean_query_local_connected_component",
            "search_crop_radius_m": float(
                getattr(args, "target_component_crop_radius", 0.75)
            ),
            "maximum_fraction_of_nonfloor_scene": float(
                getattr(args, "maximum_target_component_fraction", 0.25)
            ),
        },
        "contact_selective_target_policy": _contact_selective_contract(args, spec),
        "first_active_icgf": icgf_receipt,
        "dry_run_icgf_node_index": icgf_node_index,
        "memory_used": False,
        "retrieval_used": False,
        "posthoc_motion_edit_used": False,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P142 ordered keyposes + query raw-scene SDF/ICGF DDPM rollout"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw-occupancy", type=Path, required=True)
    parser.add_argument("--occupancy-key", default="")
    parser.add_argument(
        "--occupancy-layout",
        choices=("auto", "xyz_zup", "lingo_yup"),
        default="auto",
    )
    parser.add_argument("--scene-lower", type=float, nargs=3, default=DEFAULT_LOWER_XYZ)
    parser.add_argument("--scene-upper", type=float, nargs=3, default=DEFAULT_UPPER_XYZ)
    parser.add_argument("--allow-occupancy-scene-mismatch", action="store_true")
    parser.add_argument(
        "--sdf-downsample",
        type=int,
        default=2,
        help="Conservative max-pool factor; 2 converts LINGO's 2 cm field to 4 cm.",
    )
    parser.add_argument("--ablation", choices=tuple(ABLATIONS), default="M3")
    parser.add_argument("--adapter-checkpoint", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--mvae-checkpoint", type=Path, default=DEFAULT_MVAE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--cfg-path", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/motion"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--guidance-param", type=float, default=5.0)
    parser.add_argument("--respacing", default="")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--maximum-nodes", type=int, default=4)
    parser.add_argument("--max-primitives", type=int, default=24)
    parser.add_argument(
        "--minimum-primitives",
        type=int,
        default=0,
        help="Generate at least this many primitives even if all nodes complete early.",
    )
    parser.add_argument("--post-completion-primitives", type=int, default=1)
    parser.add_argument("--max-attempts-per-node", type=int, default=0)
    parser.add_argument("--root-xy-threshold", type=float, default=0.35)
    parser.add_argument("--root-z-threshold", type=float, default=0.25)
    parser.add_argument("--yaw-threshold-deg", type=float, default=45.0)
    parser.add_argument("--joint-threshold", type=float, default=0.30)
    parser.add_argument("--use-predicted-joints", type=int, choices=(0, 1), default=1)

    parser.add_argument("--sdf-weight", type=float, default=1.0)
    parser.add_argument("--icgf-weight", type=float, default=1.0)
    parser.add_argument("--scene-guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--scene-guidance-mode",
        choices=("legacy_mean", "xstart_posterior"),
        default="xstart_posterior",
        help=(
            "legacy_mean uses variance*score through the START_X denoiser; "
            "xstart_posterior takes a bounded clean-latent step and recomputes "
            "the exact DDPM posterior mean"
        ),
    )
    parser.add_argument(
        "--clean-latent-step-size",
        type=float,
        default=0.10,
        help="Native clean-latent L2 step for guidance scale 1.",
    )
    parser.add_argument(
        "--max-clean-latent-update-norm",
        type=float,
        default=0.15,
        help="Per-step clean-latent trust-region cap; zero disables the cap.",
    )
    gradient_normalization_group = parser.add_mutually_exclusive_group()
    gradient_normalization_group.add_argument(
        "--normalize-scene-gradient",
        dest="normalize_scene_gradient",
        action="store_true",
        help="Use the per-sample clean-latent gradient direction, independent of loss units.",
    )
    gradient_normalization_group.add_argument(
        "--no-normalize-scene-gradient",
        dest="normalize_scene_gradient",
        action="store_false",
    )
    parser.add_argument("--collision-clearance", type=float, default=0.01)
    parser.add_argument("--collision-joint-ids", default="")
    parser.add_argument("--contact-frames", type=int, default=4)
    parser.add_argument("--icgf-deadzone", type=float, default=0.03)
    parser.add_argument("--icgf-huber-beta", type=float, default=0.04)
    parser.add_argument("--icgf-kernel-sigma", type=float, default=0.08)
    parser.add_argument(
        "--icgf-source",
        choices=(
            "auto",
            "none",
            "sparse_joint_targets",
            "surface_projected_sparse_joint_targets",
            "target_point",
            "target_component",
        ),
        default="auto",
    )
    parser.add_argument(
        "--icgf-active-node-policy",
        choices=("all", "terminal"),
        default="all",
        help="Apply ICGF to every active node or only the terminal node.",
    )
    parser.add_argument("--icgf-contact-joints", default="")
    parser.add_argument("--icgf-component-radius", type=float, default=0.18)
    parser.add_argument("--icgf-normal-offset", type=float, default=0.0)
    parser.add_argument("--icgf-maximum-points", type=int, default=1024)
    parser.add_argument("--icgf-surface-level", type=float, default=0.01)
    parser.add_argument(
        "--icgf-surface-projection-tolerance", type=float, default=0.003
    )
    parser.add_argument(
        "--icgf-surface-projection-max-iterations", type=int, default=32
    )
    parser.add_argument(
        "--icgf-surface-projection-max-step", type=float, default=0.08
    )
    parser.add_argument(
        "--icgf-surface-projection-max-displacement", type=float, default=0.25
    )
    parser.add_argument("--floor-ignore-height", type=float, default=0.08)
    parser.add_argument("--maximum-target-seed-distance", type=float, default=0.50)
    parser.add_argument(
        "--target-component-crop-radius",
        type=float,
        default=0.75,
        help=(
            "Metric radius used only for target-component connectivity. "
            "Collision occupancy outside this crop is always preserved."
        ),
    )
    parser.add_argument(
        "--maximum-target-component-fraction",
        type=float,
        default=0.25,
        help=(
            "Fail closed if the selected target exemption exceeds this "
            "fraction of non-floor occupied voxels."
        ),
    )
    target_sdf_group = parser.add_mutually_exclusive_group()
    target_sdf_group.add_argument(
        "--exclude-target-component-from-sdf",
        dest="exclude_target_component_from_sdf",
        action="store_true",
        help="Remove the final-node target connected component from collision SDF.",
    )
    target_sdf_group.add_argument(
        "--include-target-component-in-sdf",
        dest="exclude_target_component_from_sdf",
        action="store_false",
        help="Keep the target component in collision SDF as a negative control.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--denoise-start-fraction", type=float, default=0.40)
    parser.add_argument("--schedule-power", type=float, default=1.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.20)
    pcgrad_group = parser.add_mutually_exclusive_group()
    pcgrad_group.add_argument("--pcgrad-goal", dest="pcgrad_goal", action="store_true")
    pcgrad_group.add_argument("--no-pcgrad-goal", dest="pcgrad_goal", action="store_false")
    exemption_group = parser.add_mutually_exclusive_group()
    exemption_group.add_argument(
        "--exempt-contact-sdf-tail",
        dest="exempt_contact_sdf_tail",
        action="store_true",
    )
    exemption_group.add_argument(
        "--no-exempt-contact-sdf-tail",
        dest="exempt_contact_sdf_tail",
        action="store_false",
    )
    parser.set_defaults(
        pcgrad_goal=True,
        exempt_contact_sdf_tail=True,
        exclude_target_component_from_sdf=None,
        normalize_scene_gradient=True,
    )

    parser.add_argument("--seq-name-override", default="")
    parser.add_argument("--checkpoint-policy-label", default="")
    parser.add_argument("--allow-oracle-plan", action="store_true")
    parser.add_argument("--allow-unknown-provenance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "plan",
        "raw_occupancy",
        "adapter_checkpoint",
        "base_checkpoint",
        "mvae_checkpoint",
        "data_dir",
        "scene_root",
        "cfg_path",
        "output_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).resolve())


def _validate_args(args: argparse.Namespace) -> None:
    spec = ABLATIONS[str(args.ablation)]
    if args.maximum_nodes < 1 or args.max_primitives < 1:
        raise ValueError("maximum-nodes and max-primitives must be positive")
    if not 0 <= args.minimum_primitives <= args.max_primitives:
        raise ValueError("minimum-primitives must be in [0,max-primitives]")
    if args.post_completion_primitives < 0 or args.max_attempts_per_node < 0:
        raise ValueError("primitive/attempt counts cannot be negative")
    if args.icgf_maximum_points < 1:
        raise ValueError("icgf-maximum-points must be positive")
    if args.sdf_downsample < 1:
        raise ValueError("sdf-downsample must be positive")
    if args.icgf_surface_projection_max_iterations < 1:
        raise ValueError("icgf-surface-projection-max-iterations must be positive")
    for name in (
        "icgf_surface_projection_tolerance",
        "icgf_surface_projection_max_step",
        "icgf_surface_projection_max_displacement",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(name.replace("_", "-") + " must be finite and positive")
    if (
        not math.isfinite(float(args.icgf_surface_level))
        or args.icgf_surface_level < 0.0
    ):
        raise ValueError("icgf-surface-level must be finite and nonnegative")
    if (
        args.icgf_source == "surface_projected_sparse_joint_targets"
        and _exclude_target_component(args, ABLATIONS[str(args.ablation)])
    ):
        raise ValueError(
            "surface-projected sparse ICGF requires "
            "--include-target-component-in-sdf"
        )
    if (
        not math.isfinite(float(args.clean_latent_step_size))
        or args.clean_latent_step_size < 0.0
    ):
        raise ValueError("clean-latent-step-size must be finite and nonnegative")
    if (
        not math.isfinite(float(args.max_clean_latent_update_norm))
        or args.max_clean_latent_update_norm < 0.0
    ):
        raise ValueError(
            "max-clean-latent-update-norm must be finite and nonnegative"
        )
    if (
        not math.isfinite(float(args.target_component_crop_radius))
        or args.target_component_crop_radius <= 0.0
        or args.target_component_crop_radius < args.maximum_target_seed_distance
    ):
        raise ValueError(
            "target-component-crop-radius must be finite, positive, and at "
            "least maximum-target-seed-distance"
        )
    if (
        not math.isfinite(float(args.maximum_target_component_fraction))
        or not 0.0 < args.maximum_target_component_fraction <= 1.0
    ):
        raise ValueError("maximum-target-component-fraction must be in (0,1]")
    _parse_joint_ids(args.collision_joint_ids)
    _parse_joint_ids(args.icgf_contact_joints)
    if spec.name == "M4_contact_selective":
        contact_ids = _parse_joint_ids(args.icgf_contact_joints)
        collision_ids = _parse_joint_ids(args.collision_joint_ids)
        if not contact_ids:
            raise ValueError(
                "M4_contact_selective requires explicit text-routed "
                "--icgf-contact-joints"
            )
        if _exclude_target_component(args, spec):
            raise ValueError(
                "M4_contact_selective must keep the target component in the full SDF"
            )
        if not bool(args.exempt_contact_sdf_tail):
            raise ValueError(
                "M4_contact_selective requires terminal-tail contact-joint exemption"
            )
        if str(args.icgf_active_node_policy) != "terminal":
            raise ValueError(
                "M4_contact_selective requires --icgf-active-node-policy terminal"
            )
        if collision_ids and set(collision_ids) != set(range(22)):
            raise ValueError(
                "M4_contact_selective collision IDs must be empty (all 22) or exactly 0..21"
            )







