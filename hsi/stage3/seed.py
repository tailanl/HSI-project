"""New P555 neutral, query-owned static history; never a P366/P530 seed.

Only a declared current start XY/yaw and the exact neutral body asset create
frame0. Three identical physical frames produce P360's two feature frames;
no dataset row, historical pose, future frame or motion clip is read.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from hsi.common.artifacts import artifact, digest, read_sealed, require, verified, write_once

from hsi.stage3 import frame, static_seed
BUFFER_KEYS = frozenset(("v_template", "J_regressor", "lbs_weights", "shapedirs", "posedirs"))

SEED_SCHEMA = "hsi.neutral_query_static_seed.v1"
RECEIPT_SCHEMA = "hsi.neutral_query_static_seed_receipt.v1"
NEUTRAL_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"





YUP_TO_ZUP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
ARRAY_SHAPES = {"betas": (10,), "body_pose_axis_angle": (63,), "global_orient_world_rotmat": (3, 3),
                "smplx_transl_world_zup": (3,), "pelvis_delta_native": (3,), "joints_world_zup": (22, 3),
                "root_xyz_yaw_world_zup": (4,), "vertices_world_zup": (10475, 3), "faces": (20908, 3)}
SOURCE_KEYS = {"stage1_execution", "key_nodes", "navmesh_route"}
ASSET_KEYS = {"candidate_body", "runtime_body", "render_body"}





def array_hash(value):
    array = np.ascontiguousarray(value)
    require(array.dtype.kind in "fiub" and np.isfinite(array).all(), "Only finite numeric arrays may be fingerprinted")
    return digest({"dtype": array.dtype.str, "shape": list(array.shape),
                   "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()})


def model_array_hashes(path):
    """Canonical runtime buffers, not gender-string inference or pickle loading."""
    with np.load(path, allow_pickle=False) as data:
        convert = lambda value: np.ascontiguousarray(value, dtype=np.float32)
        result = {"v_template": convert(data["v_template"]),
                  "J_regressor": convert(data["J_regressor"]),
                  "lbs_weights": convert(data["weights"]),
                  "shapedirs": convert(data["shapedirs"][..., :10]),
                  "posedirs": convert(data["posedirs"].reshape(-1, data["posedirs"].shape[-1]).T)}
    return {key: array_hash(value) for key, value in result.items()}





def validate_arrays(arrays):
    require(set(arrays) == set(ARRAY_SHAPES), "Unexpected/missing neutral seed arrays")
    for key, shape in ARRAY_SHAPES.items():
        value = np.asarray(arrays[key])
        require(value.shape == shape and value.dtype.kind in "fi" and np.isfinite(value).all(), "Invalid neutral seed array: "+key)
    require(np.array_equal(arrays["betas"], np.zeros(10)) and np.array_equal(arrays["body_pose_axis_angle"], np.zeros(63)),
            "Fresh neutral seed must be zero-shape and zero-pose, never an old full pose")
    rotation = arrays["global_orient_world_rotmat"]
    require(np.allclose(rotation.T@rotation, np.eye(3), atol=2e-5, rtol=0)
            and abs(np.linalg.det(rotation)-1) <= 2e-5, "Neutral seed orientation is not SO(3)")
    root, joints = arrays["root_xyz_yaw_world_zup"], arrays["joints_world_zup"]
    require(np.max(np.abs(joints[0]-root[:3])) <= 2e-5, "Seed root/J22 mismatch")
    require(np.max(np.abs(arrays["smplx_transl_world_zup"]+arrays["pelvis_delta_native"]-root[:3])) <= 5e-5,
            "Seed pelvis calibration/translation mismatch")
    axis = joints[2, :2]-joints[1, :2]
    require(np.linalg.norm(axis) > 1e-7, "Degenerate seed hip axis")
    yaw = np.arctan2(axis[0], -axis[1])
    require(abs(np.arctan2(np.sin(yaw-root[3]), np.cos(yaw-root[3]))) <= 2e-5, "Seed yaw/J22 mismatch")
    require(arrays["faces"].dtype.kind in "iu" and arrays["faces"].min() >= 0 and arrays["faces"].max() < 10475,
            "Seed body topology mismatch")
    return arrays


def validate_scalar(data, key, expected):
    scalar = data[key]
    require(scalar.shape == () and type(scalar.item()) is type(expected) and scalar.item() == expected,
            "Seed scalar contract mismatch: "+key)


def build_seed(output, *, scene_id, instruction, start_xy, yaw, source_bindings, body_assets=None):
    """Compile exact neutral frame0; body CPU only, no ReMoGen checkpoint loaded."""
    output = Path(output)
    require(not output.exists(), "Refuse to overwrite static seed directory")
    require(isinstance(scene_id, str) and bool(scene_id) and isinstance(instruction, str) and bool(instruction), "Invalid current query")
    start_xy = np.asarray(start_xy, dtype=np.float64)
    require(start_xy.shape == (2,) and np.isfinite(start_xy).all() and math_isfinite(yaw), "Invalid current start/yaw")
    require(set(source_bindings) == SOURCE_KEYS, "Fresh seed needs exact current route bindings")
    for row in source_bindings.values():
        verified(row)
    require(body_assets is not None, "Explicit candidate/runtime/render body assets are required")
    assets = body_assets
    require(set(assets) == ASSET_KEYS, "Incomplete neutral asset identity")
    for row in assets.values():
        verified(row)
        require(row["sha256"] == NEUTRAL_SHA256, "Non-neutral body asset")
    carrier = frame._build_at_explicit_xy(start_xy, Path(assets["candidate_body"]["path"]), float(yaw), .01)
    arrays = {"betas": np.asarray(carrier["betas"], dtype=np.float32),
              "body_pose_axis_angle": np.asarray(carrier["body_pose"], dtype=np.float32).reshape(63),
              "global_orient_world_rotmat": np.asarray(carrier["global_orient"], dtype=np.float32),
              "smplx_transl_world_zup": np.asarray(carrier["translation"], dtype=np.float32),
              "pelvis_delta_native": np.asarray(carrier["pelvis_delta"], dtype=np.float32),
              "joints_world_zup": np.asarray(carrier["joints"], dtype=np.float32),
              "root_xyz_yaw_world_zup": np.asarray(carrier["root"], dtype=np.float64),
              "vertices_world_zup": np.asarray(carrier["vertices"], dtype=np.float32),
              "faces": np.asarray(carrier["faces"], dtype=np.int64)}
    validate_arrays(arrays)
    output.mkdir(parents=True)
    path = output / "static_seed.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays, schema=np.array(SEED_SCHEMA), gender=np.array("neutral"),
                            coordinate_system=np.array("world_zup_metric"), static_history=np.array(True),
                            history_length=np.array(2), source_frame_repeat_count=np.array(3),
                            source_frame_indices=np.array([], dtype=np.int64))
    receipt = {"schema": RECEIPT_SCHEMA, "status": "fresh_neutral_static_seed_cpu_compiled",
               "seed": artifact(path), "scene_id": scene_id, "instruction": instruction,
               "start_world_xy_m": list(map(float, start_xy)), "initial_yaw_rad": float(yaw),
               "gender": "neutral", "body_asset_identity_not_inferred_from_gender": True,
               "body_assets": assets, "runtime_model_array_hashes": model_array_hashes(assets["candidate_body"]["path"]),
               "array_hashes": {key: array_hash(value) for key, value in arrays.items()},
               "source_bindings": source_bindings, "source": artifact(__file__), "retained_frame_math": artifact(frame.__file__),
               "retained_static_feature_math": artifact(static_seed.__file__),
               "current_start_only": True, "physical_identical_repeats": 3, "feature_history_frames": 2,
               "old_pose_or_motion_read": False, "future_or_gt_motion_used": False,
               "lingo_split_read": False, "runtime_roundtrip_not_yet_executed": True,
               "calibration_tolerance_m": 5e-6, "world_j22_roundtrip_tolerance_m": 5e-5,
               "motion_generated": False, "motion_release_authorized": False}
    validate_buffer_identity(receipt)
    receipt["runtime_buffer_identity_policy"] = "exact_five_keys_recomputed_from_bound_neutral_asset"
    write_once(output / "receipt.json", receipt)
    return receipt


@dataclass
class NeutralSeed:
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
    lineage: dict
    receipt: dict

    def build_p360_sample(self, dataset, device, plan):
        validate_buffer_identity(self.receipt)
        require(self.gender == "neutral", "Current seed must use exact neutral body")
        body = dataset.primitive_utility.get_smpl_model("neutral")
        actual = {}
        for key in self.receipt["runtime_model_array_hashes"]:
            value = getattr(body, key).detach().cpu().numpy()
            actual[key] = array_hash(np.ascontiguousarray(value, dtype=np.float32))
        require(actual == self.receipt["runtime_model_array_hashes"], "Runtime neutral model arrays differ from candidate body")
        # Unbound numerical method receives THIS P555 neutral object, never a
        # historical male seed. The old loader and old source receipts are not used.
        sample, measured = static_seed.QueryStaticSeed.build_p360_sample(self, dataset, device, plan)
        require(measured["gender"] == "neutral" and measured["smplx_carrier_j22_max_abs_m"] <= 5e-5
                and measured["world_j22_roundtrip_max_abs_m"] <= 5e-5, "Neutral runtime roundtrip failed")
        audit = {**measured, "schema": "p555.neutral_static_runtime_audit.v1",
                 "gender_policy": "p555_exact_neutral_asset_zero_shape_current_start",
                 "retained_function_historical_gender_label_not_used": True,
                 "source": artifact(__file__), "retained_numeric_source": artifact(static_seed.__file__),
                 "runtime_body_array_hashes": actual, "legacy_P366_male_lineage_claimed": False,
                 "p555_seed_receipt": self.lineage["p555_seed_receipt"]}
        return sample, audit

    def build_runtime_dataset(self, args, device, plan, raw_scene):
        require(plan.scene_name == self.receipt["scene_id"] and plan.text == self.receipt["instruction"],
                "Runtime plan and static seed belong to different queries")
        # Retains load_data=False, guarded split opening, and synthetic get_seq;
        # this calls the overridden P555 build_p360_sample above.
        return static_seed.QueryStaticSeed.build_runtime_dataset(self, args, device, plan, raw_scene)


def load_seed(receipt_path):
    receipt_path = Path(receipt_path)
    value = read_sealed(receipt_path)
    require(value.get("schema") == RECEIPT_SCHEMA and value.get("gender") == "neutral"
            and value.get("status") == "fresh_neutral_static_seed_cpu_compiled", "Wrong current neutral seed receipt")
    require(value.get("runtime_buffer_identity_policy") == "exact_five_keys_recomputed_from_bound_neutral_asset", "Missing current neutral identity policy")
    validate_buffer_identity(value)
    require(value.get("current_start_only") is True and value.get("old_pose_or_motion_read") is False
            and value.get("future_or_gt_motion_used") is False and value.get("lingo_split_read") is False,
            "Historical/future pose is not a current static seed")
    require(set(value["body_assets"]) == ASSET_KEYS and set(value["source_bindings"]) == SOURCE_KEYS,
            "Incomplete neutral seed asset/source identity")
    require(type(value.get("physical_identical_repeats")) is int and value["physical_identical_repeats"] == 3
            and type(value.get("feature_history_frames")) is int and value["feature_history_frames"] == 2
            and value.get("motion_generated") is False and value.get("motion_release_authorized") is False,
            "Static history count or release declaration changed")
    for row in [*value["body_assets"].values(), *value["source_bindings"].values(), value["source"],
                value["retained_frame_math"], value["retained_static_feature_math"]]:
        verified(row)
    require(value["source"] == artifact(__file__), "P555 neutral seed implementation drift")
    require(value["retained_frame_math"] == artifact(frame.__file__)
            and value["retained_static_feature_math"] == artifact(static_seed.__file__), "Static numerical source substituted")
    require(all(row["sha256"] == NEUTRAL_SHA256 for row in value["body_assets"].values()), "Seed neutral asset hash mismatch")
    path = verified(value["seed"])
    with np.load(path, allow_pickle=False) as data:
        for key, expected in (("schema", SEED_SCHEMA), ("gender", "neutral"), ("coordinate_system", "world_zup_metric"),
                              ("static_history", True), ("history_length", 2), ("source_frame_repeat_count", 3)):
            validate_scalar(data, key, expected)
        require(data["source_frame_indices"].shape == (0,) and data["source_frame_indices"].dtype.kind in "iu",
                "Static seed must not import temporal frames")
        require(set(value["array_hashes"]) == set(ARRAY_SHAPES), "Unexpected/missing seed numeric hashes")
        require(set(data.files) == set(ARRAY_SHAPES)|{"schema", "gender", "coordinate_system", "static_history", "history_length",
                                                     "source_frame_repeat_count", "source_frame_indices"}, "Unregistered seed fields")
        arrays = {key: data[key].copy() for key in value["array_hashes"]}
    validate_arrays(arrays)
    require({key: array_hash(array) for key, array in arrays.items()} == value["array_hashes"], "Seed numeric hash drift")
    require(np.linalg.norm(arrays["root_xyz_yaw_world_zup"][:2]-value["start_world_xy_m"]) <= 2e-5, "Seed current start drift")
    require(abs(np.arctan2(np.sin(arrays["root_xyz_yaw_world_zup"][3]-value["initial_yaw_rad"]),
                          np.cos(arrays["root_xyz_yaw_world_zup"][3]-value["initial_yaw_rad"]))) <= 2e-5, "Seed initial yaw drift")
    return NeutralSeed(path, value["seed"]["sha256"], "neutral", arrays["betas"], arrays["body_pose_axis_angle"],
                          arrays["global_orient_world_rotmat"], arrays["smplx_transl_world_zup"], arrays["pelvis_delta_native"],
                          arrays["joints_world_zup"], arrays["root_xyz_yaw_world_zup"],
                          {"p555_seed_receipt": artifact(receipt_path), "body_model": value["body_assets"]["candidate_body"]}, value)


def math_isfinite(value):
    return type(value) in (int, float, np.float32, np.float64) and np.isfinite(value)

def validate_buffer_identity(receipt):
    fingerprints = receipt.get("runtime_model_array_hashes")
    require(isinstance(fingerprints, dict) and set(fingerprints) == BUFFER_KEYS,
            "Neutral runtime buffer fingerprint set must contain exactly five keys")
    body = receipt["body_assets"]["candidate_body"]
    verified(body)
    require(body["sha256"] == NEUTRAL_SHA256, "Runtime fingerprints need the registered neutral asset")
    expected = model_array_hashes(body["path"])
    require(fingerprints == expected, "Runtime fingerprints do not reproduce from the bound neutral asset")
    return expected


def verify_neutral_assets(*, candidate_body, runtime_body, render_body):
    """All three users of the body must bind the same actual neutral asset."""
    rows = {name: artifact(path) for name, path in {
        "candidate_body": candidate_body, "runtime_body": runtime_body, "render_body": render_body}.items()}
    require(all(row["sha256"] == NEUTRAL_SHA256 for row in rows.values()), "Neutral body asset identity differs")
    return rows
