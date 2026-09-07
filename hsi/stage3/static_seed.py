#!/usr/bin/env python3
"""Static numerical history for the current strictly verified neutral seed.

The artifact represented here is not a motion clip.  It contains one
query-owned, action-agnostic SMPL-X frame and an explicit contract asking the
executor to repeat that exact frame to form its numerical H=2 history.  The
source LINGO split and any declared ``current_pose`` sidecar are deliberately
outside this module's input surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import builtins
import hashlib
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    import torch
    from .seed import NeutralSeed


SEED_SCHEMA = "p478.query_static_history_seed.v1"
RECEIPT_SCHEMA = "p478.query_static_history_seed_receipt.v1"
FRAME_SCHEMA = "p366.query_current_frame0.v1"
FRAME_RECEIPT_SCHEMA = "p366.query_current_frame0_receipt.v1"
GROUNDING_SCHEMA = "p366.qwen_chain_registry_grounding.v1"
HISTORY_SOURCE = "query_static_initial_history"
WORLD_CONVERSION_YUP_TO_ZUP = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


class QueryStaticSeedError(ValueError):
    """The static seed or its immutable lineage violates the query contract."""


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise QueryStaticSeedError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def verify_artifact(record: Mapping[str, Any], label: str) -> Path:
    require(isinstance(record, Mapping), f"{label} must be an artifact record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    require(path.is_file(), f"missing {label}: {path}")
    require(int(record.get("bytes", -1)) == path.stat().st_size, f"{label} byte drift")
    require(str(record.get("sha256", "")) == sha256_file(path), f"{label} SHA-256 drift")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _scalar(payload: Mapping[str, np.ndarray], key: str) -> Any:
    require(key in payload, f"static seed is missing {key}")
    value = np.asarray(payload[key])
    require(value.size == 1, f"static seed {key} must be scalar")
    return value.reshape(()).item()


def wrapped_angle(value: float) -> float:
    return float(math.atan2(math.sin(float(value)), math.cos(float(value))))


def anatomical_yaw(joints_world: np.ndarray) -> float:
    joints = np.asarray(joints_world, dtype=np.float64)
    require(joints.shape == (22, 3), "static source needs J22 world joints")
    require(bool(np.isfinite(joints).all()), "static source joints are non-finite")
    hip_axis = joints[2, :2] - joints[1, :2]
    require(float(np.linalg.norm(hip_axis)) > 1.0e-7, "static source hip axis is degenerate")
    return wrapped_angle(math.atan2(float(hip_axis[0]), float(-hip_axis[1])))


def rotation_z(yaw_rad: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    rotation_z = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rotation_z


def world_global_orientation(yaw_delta_rad: float) -> np.ndarray:
    return np.ascontiguousarray(
        rotation_z(yaw_delta_rad) @ WORLD_CONVERSION_YUP_TO_ZUP
    )











@dataclass(frozen=True)
class QueryStaticSeed:
    path: Path
    seed_hash: str
    gender: str
    betas: np.ndarray
    body_pose_axis_angle: np.ndarray
    global_orient_world_rotmat: np.ndarray
    smplx_transl_world_zup: np.ndarray
    pelvis_delta_native: np.ndarray
    joints_world_zup: np.ndarray
    root_xyz_yaw_world_zup: np.ndarray
    lineage: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "NeutralSeed":
        """Load only the current neutral seed through its complete authority.

        Accept the current receipt JSON or its hash-bound ``static_seed.npz``
        (the executor CLI's original argument convention). There is no old
        male-frame loader, metadata-only fallback, or implicit lineage repair.
        """
        from .seed import load_seed

        path = Path(path).expanduser().resolve(strict=True)
        receipt_path = path.parent / "receipt.json" if path.suffix == ".npz" else path
        seed = load_seed(receipt_path)
        if path.suffix == ".npz":
            require(seed.path.resolve(strict=True) == path,
                    "Static seed NPZ is not the file bound by its current receipt")
        return seed

    def build_p360_sample(self, dataset: Any, device: torch.device, plan: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create the exact P360 H=2 normalized history without a motion row."""
        import torch

        require(int(dataset.history_length) == 2 and int(dataset.future_length) == 8,
                "query static adapter requires P360 H=2,F=8")
        utility = dataset.primitive_utility
        dtype = torch.float32
        repeat = 3
        betas_one = torch.as_tensor(self.betas, device=device, dtype=dtype).reshape(1, 10)
        betas = betas_one[:, None].expand(1, repeat, 10).clone()
        body_aa = torch.as_tensor(self.body_pose_axis_angle, device=device, dtype=dtype).reshape(21, 3)
        # Use PyTorch3D's convention already used by ReMoGen when available;
        # an explicit Rodrigues implementation would risk a convention drift.
        from pytorch3d import transforms
        body_rot = transforms.axis_angle_to_matrix(body_aa).reshape(1, 1, 21, 3, 3).expand(1, repeat, 21, 3, 3).clone()
        global_rot = torch.as_tensor(self.global_orient_world_rotmat, device=device, dtype=dtype)
        global_rot = global_rot.reshape(1, 1, 3, 3).expand(1, repeat, 3, 3).clone()
        joints = torch.as_tensor(self.joints_world_zup, device=device, dtype=dtype)
        joints = joints.reshape(1, 1, 22, 3).expand(1, repeat, 22, 3).clone()
        pelvis_delta = utility.calc_calibrate_offset({"betas": betas_one, "gender": self.gender})
        expected_pelvis_delta = torch.as_tensor(
            self.pelvis_delta_native, device=device, dtype=dtype
        ).reshape(1, 3)
        require(float((pelvis_delta - expected_pelvis_delta).abs().max().item()) <= 5.0e-6,
                "runtime ReMoGen SMPL-X pelvis delta differs from the hash-bound neutral model")
        root = joints[:, 0, 0]
        # SMPL-X translation that places its shaped pelvis at the immutable
        # query root under the reconstructed world global orientation.
        transl_one = torch.as_tensor(
            self.smplx_transl_world_zup, device=device, dtype=dtype
        ).reshape(1, 3)
        require(float((root - (pelvis_delta + transl_one)).abs().max().item()) <= 5.0e-5,
                "stored static SMPL-X translation does not place the pelvis at the query root")
        transl = transl_one[:, None].expand(1, repeat, 3).clone()
        body_model = utility.get_smpl_model(self.gender)
        with torch.no_grad():
            reconstructed_body = body_model(
                betas=betas_one,
                global_orient=global_rot[:, 0],
                body_pose=body_rot[:, 0],
                transl=transl_one,
            )
        reconstructed_j22 = reconstructed_body.joints[:, :22]
        carrier_j22_error = float(
            (reconstructed_j22 - joints[:, 0]).abs().max().item()
        )
        require(
            carrier_j22_error <= 5.0e-5,
            f"reconstructed neutral SMPL-X carrier differs from seed J22 by {carrier_j22_error:.8f} m",
        )
        primitive = {
            "gender": self.gender,
            "betas": betas,
            "transl": transl,
            "global_orient": global_rot,
            "body_pose": body_rot,
            "pelvis_delta": pelvis_delta,
            "joints": joints.reshape(1, repeat, 66),
            "transf_rotmat": torch.eye(3, device=device, dtype=dtype).unsqueeze(0),
            "transf_transl": torch.zeros(1, 1, 3, device=device, dtype=dtype),
        }
        _, _, canonical = utility.canonicalize(primitive, use_predicted_joints=True)
        feature = utility.calc_features(canonical, use_predicted_joints=True)
        transl_delta = feature["transl_delta"]
        joints_delta = feature["joints_delta"]
        orient_delta = feature["global_orient_delta_6d"]
        require(float(transl_delta.abs().max().item()) <= 1.0e-7, "static transl delta is nonzero")
        require(float(joints_delta.abs().max().item()) <= 1.0e-7, "static joint delta is nonzero")
        identity6 = transforms.matrix_to_rotation_6d(torch.eye(3, device=device, dtype=dtype)).reshape(1, 1, 6)
        require(float((orient_delta - identity6).abs().max().item()) <= 1.0e-7,
                "static global-orientation delta is not identity")
        for key in ("transl", "poses_6d", "joints"):
            feature[key] = feature[key][:, :-1]
        tensor = utility.dict_to_tensor(feature)
        require(tuple(tensor.shape[:2]) == (1, 2), "static feature tensor is not H=2")
        history = dataset.normalize(tensor).permute(0, 2, 1).unsqueeze(2)

        # Denormalize -> predicted-SMPL dictionary -> world J22 roundtrip.  The
        # roundtrip checks the actual tensors P360 will consume, not just the
        # unnormalized source arrays.
        decoded = dataset.denormalize(history.squeeze(2).permute(0, 2, 1))
        decoded_feature = utility.tensor_to_dict(decoded)
        decoded_feature.update({
            "gender": self.gender,
            "betas": betas[:, :2],
            "pelvis_delta": pelvis_delta,
            "transf_rotmat": canonical["transf_rotmat"],
            "transf_transl": canonical["transf_transl"],
        })
        decoded_smpl = utility.feature_dict_to_smpl_dict(decoded_feature)
        decoded_world = utility.transform_primitive_to_world(decoded_smpl)
        roundtrip = decoded_world["joints"].reshape(1, 2, 22, 3)
        expected = joints[:, :2]
        max_roundtrip = float((roundtrip - expected).abs().max().item())
        require(max_roundtrip <= 5.0e-5, f"static world-J22 roundtrip drift {max_roundtrip:.8f} m")
        sample = {
            "gender": [self.gender],
            "betas": betas,
            "motion_tensor_normalized": history,
            "transf_rotmat": canonical["transf_rotmat"],
            "transf_transl": canonical["transf_transl"],
            "scene_names": str(plan.scene_name),
        }
        audit = {
            "schema": "p478.query_static_runtime_audit.v1",
            "history_source": HISTORY_SOURCE,
            "seed_path": str(self.path),
            "seed_sha256": self.seed_hash,
            "lingo_sequence_lookup": False,
            "lingo_motion_opened": False,
            "future_or_gt_motion_used": False,
            "source_frame_indices": [],
            "physical_frame_repeat_count": 3,
            "feature_history_length": 2,
            "all_physical_frames_identical": True,
            "transl_delta_max_abs_m": float(transl_delta.abs().max().item()),
            "joint_delta_max_abs_m": float(joints_delta.abs().max().item()),
            "global_orientation_delta_identity6_max_abs": float((orient_delta - identity6).abs().max().item()),
            "global_orientation_delta_angle_max_rad": 0.0,
            "world_j22_roundtrip_max_abs_m": max_roundtrip,
            "smplx_carrier_j22_max_abs_m": carrier_j22_error,
            "gender": self.gender,
            "gender_policy": "current_hash_verified_neutral_carrier",
            "generated_only_output_required": True,
        }
        return sample, audit

    def build_runtime_dataset(self, args: Any, device: torch.device, plan: Any, raw_scene: Any) -> Any:
        """Instantiate P360 statistics/utility/text only, never a split row."""
        import torch

        from data_loaders.humanml.data.dataset_lingo import LINGODataset

        forbidden_split = (Path(args.data_dir) / f"{args.split}.pkl").resolve()
        original_open = builtins.open

        def guarded_open(file: Any, *open_args: Any, **open_kwargs: Any) -> Any:
            try:
                candidate = Path(file).expanduser().resolve()
            except (TypeError, ValueError):
                candidate = None
            if candidate == forbidden_split:
                raise QueryStaticSeedError(
                    f"query-static mode forbids opening the LINGO motion split: {candidate}"
                )
            return original_open(file, *open_args, **open_kwargs)

        try:
            builtins.open = guarded_open
            dataset = LINGODataset(
                dataset_path=str(args.data_dir), dataset_name="lingo", cfg_path=str(args.cfg_path),
                enforce_gender=None, enforce_zero_beta=None, body_type="smplx", split=args.split,
                device=device, scene_root=str(args.scene_root), load_scene=False, load_data=False,
            )
        finally:
            builtins.open = original_open
        expected_shape = tuple(int(value) for value in dataset.scene_grid[6:].tolist())
        require(tuple(raw_scene.native_occupancy_xyz.shape) == expected_shape,
                f"native ReMoGen occupancy expects {expected_shape}, got {tuple(raw_scene.native_occupancy_xyz.shape)}")
        dataset.scene_embedding_dict[plan.scene_name] = {
            "occ": torch.from_numpy(raw_scene.native_occupancy_xyz).to(device=device),
            "source": "p478_query_static_cli_raw_occupancy",
            "basename": Path(raw_scene.source_path).name,
        }
        sample, audit = self.build_p360_sample(dataset, device, plan)
        # Preserve P360's old lookup/get_seq statements while binding them to
        # one synthetic metadata row.  No dataset split was loaded.
        dataset.dataset = [{"seq_name": str(plan.seq_name), "scene": str(plan.scene_name)}]

        def get_static_seq(_dataset: Any, index: int) -> dict[str, Any]:
            require(int(index) == 0, "query static dataset exposes exactly one synthetic row")
            # Return fresh tensors because P360 pops motion_tensor_normalized.
            copied = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in sample.items()}
            return copied

        dataset.get_seq = MethodType(get_static_seq, dataset)
        dataset._p478_query_static_runtime_audit = audit
        return dataset


def generated_only_sequence(generated_world: Mapping[str, Any], *, plan: Any,
                            history_length: int, future_length: int) -> dict[str, Any]:
    """Create an evaluator container containing generated tensors and no GT keys."""

    generated_frames = int(generated_world["joints"].shape[1])
    sequence = {
        "texts": plan.text,
        "gender": generated_world["gender"],
        "betas": generated_world["betas"][:, :generated_frames],
        "transl": generated_world["transl"][:, :generated_frames],
        "global_orient": generated_world["global_orient"][:, :generated_frames],
        "body_pose": generated_world["body_pose"][:, :generated_frames],
        "joints": generated_world["joints"][:, :generated_frames],
        "history_length": int(history_length),
        "future_length": int(future_length),
        "replicate_times": 1,
        "scene_names": plan.scene_name,
        "seq_name": plan.seq_name,
        "generated_only": True,
        "reference_motion_available": False,
    }
    require(not any(str(key).startswith("gt_") for key in sequence), "generated-only sequence leaked GT")
    return sequence


__all__ = [
    "HISTORY_SOURCE", "QueryStaticSeed", "QueryStaticSeedError", "artifact",
    "generated_only_sequence", "sha256_file",
]
