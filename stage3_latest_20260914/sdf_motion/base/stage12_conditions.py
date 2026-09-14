"""Bound, read-only Stage 1/2 conditions for an independent Stage 3 experiment.

This is NOT ``stage2.pipeline.validate_success``. Historical H3/HybrIK/Qwen
and physical decisions are retained and a finite set of direct artifacts is
SHA-checked; their expensive validators are not rerun. No model inference,
future motion, reference pose, GT event times, or coordinate fitting is used.
The same-shape initial standing pose is newly materialized from Stage1 XY/yaw.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import torch

HSI_ROOT = Path(__file__).resolve().parents[3] / "HSI-project"
if str(HSI_ROOT) not in sys.path:
    sys.path.insert(0, str(HSI_ROOT))
from hsi.common.artifacts import read_json, read_sealed

SCHEMA = "agent9.closd_unihsi_sdf.stage12_conditions.v1"
BODY_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
WORLD_BOUNDS = ((-3., -4., 0.), (3., 4., 2.))
_UPSTREAM = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Stage12Case:
    schema: str
    scene_id: str
    instruction: str
    action_family: str
    target_id: str
    target_class: str
    surface_id: str
    initial_pose: torch.Tensor             # [135], pelvis XYZ + 22 row-6D rotations
    target_pose: torch.Tensor              # [135], no timestamp
    betas: torch.Tensor                    # [10], fixed exact Stage2 shape
    body: object                          # differentiable MeshBody, frozen parameters
    initial_joints: torch.Tensor
    initial_vertices: torch.Tensor
    target_joints: torch.Tensor            # [22,3], world Z-up metres
    target_vertices: torch.Tensor          # [10475,3]
    faces: torch.Tensor                    # [20908,3]
    route_xy: torch.Tensor                 # Stage1 control polyline, not GT trajectory
    contact_triangles: torch.Tensor
    contact_goal: torch.Tensor
    contact_normal: torch.Tensor
    contact_joint_ids: torch.Tensor
    contact_vertex_ids: torch.Tensor
    target_root_xyz: torch.Tensor
    terminal_yaw: float
    arrival_yaw: float
    permissions: dict
    sdf: dict
    scene_mesh: dict
    keypoint_descriptions: dict
    provenance: dict


def _record(value):
    require(isinstance(value, dict), "Missing artifact record")
    result = {k: value.get(k) for k in ("path", "bytes", "sha256")}
    require(isinstance(result["path"], str) and Path(result["path"]).is_absolute(), "Absolute artifact path required")
    require(type(result["bytes"]) is int and result["bytes"] >= 0, "Invalid artifact byte count")
    require(isinstance(result["sha256"], str) and re.fullmatch("[0-9a-f]{64}", result["sha256"]), "Invalid artifact SHA")
    return result


class _Audit:
    def __init__(self):
        self.checked = {}

    def file(self, record, limit=256 * 1024 * 1024):
        record = _record(record)
        p = Path(record["path"])
        require(p.is_file() and not p.is_symlink(), "Missing or symlink artifact: " + str(p))
        require(record["bytes"] <= limit, "Artifact exceeds bounded reader size")
        if str(p) in self.checked:
            require(self.checked[str(p)] == record, "Conflicting identity for one artifact")
            return p
        before = p.stat()
        require(before.st_size == record["bytes"], "Artifact size mismatch: " + str(p))
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        after = p.stat()
        require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Artifact changed during read")
        require(h.hexdigest() == record["sha256"], "Artifact SHA mismatch: " + str(p))
        self.checked[str(p)] = record
        return p

    def json(self, record, sealed=True):
        path = self.file(record, 8 * 1024 * 1024)
        return (read_sealed if sealed else read_json)(path)


def _archive(path):
    with zipfile.ZipFile(path) as archive:
        require(sum(v.file_size for v in archive.infolist()) <= 64 * 1024 * 1024, "Oversized expanded NPZ")
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    for key, value in values.items():
        require(value.dtype.kind != "O", "Object arrays are forbidden")
        if value.dtype.kind in "fci":
            require(np.isfinite(value).all(), "Nonfinite NPZ: " + key)
    return values


def _array(values, name, shape, integer=False):
    require(name in values, "Missing array: " + name)
    value = values[name]
    require(value.shape == shape and np.isfinite(value).all(), "Invalid array shape/value: " + name)
    require(value.dtype.kind in ("iu" if integer else "f"), "Invalid numeric dtype: " + name)
    return value


def _scalar(values, name):
    require(name in values and values[name].ndim == 0, "Missing scalar: " + name)
    return values[name].item()


def _upstream():
    global _UPSTREAM
    if _UPSTREAM is None:
        path = HSI_ROOT / "experiments/stage3_upstream_adapter.py"
        name = "_agent9_closd_historical_upstream"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _UPSTREAM = module
    return _UPSTREAM


def _make_body(model_path, arrays):
    """One fixed matrix-pose SMPL-X load; preserve its non-flat hand means."""
    from smplx import SMPLXLayer
    from hsi.stage3_sequence.data import axis_angle_to_matrix
    from hsi.stage3_sequence.mesh_body import MeshBody
    from hsi.stage2.refine_physics import build_anatomy_masks, RefineConfig
    layer = SMPLXLayer(str(model_path), gender="neutral", num_betas=10,
        use_pca=False, flat_hand_mean=False, use_face_contour=False).cpu().eval()
    require(tuple(layer.v_template.shape) == (10475, 3), "Wrong SMPL-X body topology")
    require(np.array_equal(layer.faces, arrays["faces"]), "Source body / target face topology mismatch")
    vertices = arrays["vertices_world_zup"]
    butt = arrays["fixed_generated_butt_vertex_indices"]
    count = min(len(butt), max(8, math.ceil(len(butt) * .15)))
    butt = butt[np.argsort(vertices[butt, 2], kind="stable")[:count]]
    masks = build_anatomy_masks(layer, 10475, RefineConfig(steps=1, bilateral_foot_support=True).validate())
    dominant = layer.lbs_weights[:, :22].argmax(1).numpy()
    feet = []
    for mask in (masks.left_foot_support, masks.right_foot_support):
        ids = np.flatnonzero(mask.numpy())
        feet.append(ids[np.argsort(vertices[ids, 2], kind="stable")[:32]].tolist())
    regions = dict(pelvis=butt.tolist(), left_thigh=np.flatnonzero(dominant == 1).tolist(),
        right_thigh=np.flatnonzero(dominant == 2).tolist(), left_foot=feet[0], right_foot=feet[1],
        left_hand=np.flatnonzero(dominant == 20).tolist(), right_hand=np.flatnonzero(dominant == 21).tolist(),
        back=np.flatnonzero(dominant == 9).tolist())
    return MeshBody(layer, torch.as_tensor(arrays["betas"], dtype=torch.float32),
        left_hand_pose=axis_angle_to_matrix(layer.left_hand_mean.reshape(15, 3)),
        right_hand_pose=axis_angle_to_matrix(layer.right_hand_mean.reshape(15, 3)), region_vertex_ids=regions)


def _make_initial(body, xy, yaw):
    return _upstream().shaped_static_history(body, xy, yaw)


def _target_motion(arrays):
    # Stage2's frame rotates its already-Z-up carrier. SMPL-X's native rest
    # frame is Y-up; retain the original adapter's explicit right-composition.
    upstream = _upstream()
    return _upstream().pose_to_motion(arrays["root_xyz_yaw"][:3], arrays["body_pose_axis_angle"],
        torch.as_tensor(arrays["pose_frame_to_world_rotation"], dtype=torch.float32) @ upstream.Q_YUP_TO_ZUP)


def load_case(receipt_path, device="cpu"):
    """Return one historical verified single-sit case without re-running gates.

    Geometry is built on CPU then explicitly moved to ``device``. This does
    not choose a GPU, load a motion model, or accept future-reference arrays.
    Default permissions are strict; Stage2's static-refine allowances are
    recorded separately and never silently authorize Stage3 pose changes.
    """
    audit = _Audit()
    p = Path(receipt_path).absolute()
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= 8 * 1024 * 1024, "Invalid Stage2 receipt path")
    raw = p.read_bytes()
    receipt_record = {"path": str(p), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    stage2 = audit.json(receipt_record)
    require(stage2.get("schema") == "hsi.stage2.verified_keypose.v1" and
        stage2.get("status") == "complete_verified_stage2_keypose" and
        stage2.get("verified_stage2_keypose") is True and stage2.get("stage3_handoff_allowed") is True,
        "Historical Stage2 did not authorize handoff")
    require(stage2.get("qwen_generated_coordinates") is False, "Qwen coordinate provenance not accepted")
    require(stage2.get("all_original_physical_gate_count") == 18 and stage2.get("additional_neutral_sit_gate_count") == 4,
        "Historical publication gate set differs")
    stage1 = audit.json(stage2["source_stage1"])
    require(stage1.get("status") == "complete" and stage1.get("qwen_computed_coordinates") is False,
        "Incomplete or coordinate-generated Stage1")
    target, bundle = audit.json(stage1["target"]), audit.json(stage1["bundle"])
    steps = stage1["interaction_plan"]["steps"]
    require(len(steps) == 1 and steps[0]["action"] == "sit" and target["action_family"] == "sit",
        "This adapter explicitly supports only one upstream sit; direction cannot change action type")
    target_id, surface_id = target["target_instance_id"], target["selected_surface_id"]
    require(steps[0]["target_ids"] == [target_id], "Plan/target identity mismatch")
    require(target["scene_id"] == stage1["scene_id"], "Target belongs to another scene")
    keys = audit.json(bundle["artifacts"]["key_nodes"])
    descriptions = audit.json(stage2["keypoint_descriptions"])
    require(descriptions.get("source_stage1") == stage2["source_stage1"] and
        descriptions.get("all_keypoints_have_descriptions") is True, "Keypoint description binding missing")
    require(keys["inputs"]["navmesh_route"] == bundle["artifacts"]["navmesh_route"], "Route/keypoint binding mismatch")
    route_record = bundle["artifacts"]["navmesh_route"]
    audit.file(route_record)
    controls = keys["collision_safe_control_polyline"]
    require(controls.get("direct_linear_interpolation_allowed") is True and
        keys["navmesh_execution_route"].get("all_segment_samples_human_free") is True,
        "Unverified historical route controls")
    route = np.asarray(controls["nodes_world_xy_m"], dtype=np.float32)
    require(route.ndim == 2 and route.shape[1] == 2 and 2 <= len(route) <= 256 and np.isfinite(route).all(), "Invalid route")
    initial_source = keys["start"]["source"]
    require(initial_source.get("future_ground_truth_used") is False and
        initial_source.get("state_source_kind") == "canonical_query_initialization_not_observed_motion",
        "Initial condition must be query initialization, not future GT")
    initial_record = audit.json(initial_source["current_state_receipt"], sealed=False)
    require(initial_record.get("future_ground_truth_used") is False and initial_record.get("lingo_motion_opened") is False
        and initial_record.get("source_frame_indices_read") == [] and initial_record.get("action_specific_pose_used") is False,
        "Initial-state provenance contains future/action-specific motion")
    require(initial_record.get("current_frame") == initial_source["current_state"], "Initial state identity mismatch")
    audit.file(initial_source["current_state"])
    initial_root = np.asarray(initial_source["root_xyz_yaw_world_zup"], dtype=np.float64)
    require(initial_root.shape == (4,) and np.isfinite(initial_root).all(), "Invalid initial root")
    require(np.linalg.norm(route[0] - initial_root[:2]) <= 2e-4, "Route start differs from initial root")
    require(np.allclose(initial_record["pelvis_root_xyz_world_zup"], initial_root[:3], atol=1e-6, rtol=0)
        and abs(float(initial_record["initial_yaw_rad"]) - initial_root[3]) <= 1e-6, "Initial root receipt mismatch")

    physical = audit.json(stage2["physical_refine"])
    require(physical.get("stage3_handoff_allowed") is True and physical["source_stage1"] == stage2["source_stage1"]
        and physical["outputs"]["published_keypose"] == stage2["keypose"], "Physical/keypose provenance mismatch")
    selection = audit.json(physical["outputs"]["selection"])
    trial = audit.json(selection["selected_refine_receipt"], sealed=False)
    require(selection["published_keypose"] == stage2["keypose"] and
        trial["keypose"]["sha256"] == stage2["keypose"]["sha256"], "Selected pose identity mismatch")
    body_record = _record(trial["inputs"]["fixed_neutral_smplx"])
    require(body_record["sha256"] == BODY_SHA256, "Unregistered fixed neutral SMPL-X identity")
    body_path = audit.file(body_record)
    post = audit.json(stage2["post_refine_qwen"])
    require(post.get("decision", {}).get("post_refine_semantic_pass") is True and
        post.get("source_refine_bundle") == stage2["physical_refine"], "Historical final Qwen decision/binding mismatch")
    h3 = audit.json(stage2["h3_receipt"])
    require(h3.get("schema") == "p553.h3_single_image_case.v1" and
        h3.get("status") == "complete_native_single_image" and h3.get("native_output_kind") == "image"
        and h3["inputs"]["stage1_execution"] == stage2["source_stage1"]
        and h3["source_binding"]["target_instance_id"] == target_id
        and h3["source_binding"]["target_surface_id"] == surface_id, "Historical H3 target/source mismatch")
    recovery = audit.json(stage2["recovery_receipt"])
    require(recovery.get("schema") == "p553.fresh_h3_image_hybrikx_articulation_receipt.v1"
        and recovery.get("stage2_articulation_handoff_allowed") is True
        and recovery["inputs"]["h3_generation_receipt"] == stage2["h3_receipt"]
        and recovery.get("target_instance_id") == target_id and recovery.get("target_surface_id") == surface_id,
        "Recovery belongs to another target")
    arrays = _archive(audit.file(stage2["keypose"], 8 * 1024 * 1024))
    for name, expected in (("scene_id", stage1["scene_id"]), ("target_instance_id", target_id),
        ("target_class", target["target_class"]), ("selected_candidate_id", surface_id), ("action_family", "sit"),
        ("support_component_sha256", target["selected_surface_sha256"])):
        require(str(_scalar(arrays, name)) == str(expected), "Keypose identity mismatch: " + name)
    for name in ("h3_model_used", "hybrikx_model_used", "all_publication_gates_passed"):
        require(_scalar(arrays, name) is True, "Missing historical keypose provenance: " + name)
    for name in ("gt_root_path_pose_motion_contact_icgf_used", "free_translation_used", "free_yaw_used", "free_scale_used"):
        require(_scalar(arrays, name) is False, "Forbidden keypose provenance: " + name)
    require(not any(re.search(r"(^gt_|timestamp|keypose_frame|future_motion)", name) for name in arrays
        if name != "gt_root_path_pose_motion_contact_icgf_used"), "Future/timestamp arrays are not condition inputs")
    for name, shape in (("root_xyz_yaw", (4,)), ("body_pose_axis_angle", (21, 3)), ("betas", (10,)),
        ("pose_frame_to_world_rotation", (3, 3)), ("joints_world_zup", (22, 3)),
        ("vertices_world_zup", (10475, 3)), ("contact_goal_world_xyz_zup_m", (3,))):
        _array(arrays, name, shape)
    faces = _array(arrays, "faces", (20908, 3), integer=True)
    require(faces.min() >= 0 and faces.max() < 10475, "Invalid face indices")
    butt = arrays["fixed_generated_butt_vertex_indices"]
    require(butt.ndim == 1 and butt.dtype.kind in "iu" and len(butt) >= 8 and len(np.unique(butt)) == len(butt)
        and butt.min() >= 0 and butt.max() < 10475, "Invalid contact vertex IDs")
    require(np.array_equal(arrays["contact_joint_ids"], [0, 1, 2]), "Unexpected sit contact joint mapping")
    terminal = float(_scalar(arrays, "terminal_facing_yaw_rad"))
    arrival = float(_scalar(arrays, "route_arrival_tangent_yaw_rad"))
    require(abs(math.remainder(terminal - float(arrays["root_xyz_yaw"][3]), 2 * math.pi)) < 1e-5,
        "Target root/facing mismatch")
    require(abs(math.remainder(terminal - keys["terminal_handoff"]["terminal_body_facing_yaw_rad"], 2 * math.pi)) < 1e-5,
        "Stage1/Stage2 terminal facing mismatch")
    require(abs(math.remainder(arrival - keys["terminal_handoff"]["arrival_tangent_yaw_rad"], 2 * math.pi)) < 1e-5,
        "Stage1/Stage2 arrival tangent mismatch")
    surface = next((s for s in target["candidate_surfaces"] if s["candidate_id"] == surface_id), None)
    require(surface is not None, "Selected surface absent")
    option = next((o for o in surface["approach_options"]
        if o["option_id"] == keys["terminal_handoff"]["selected_approach_option_id"]), None)
    require(option is not None, "Selected approach option absent")
    regions = _archive(audit.file(option["p552_region_member"]["region_arrays"], 64 * 1024 * 1024))
    triangles = regions["contact_triangles"]
    require(triangles.ndim == 3 and triangles.shape[1:] == (3, 3) and 0 < len(triangles) <= 500000
        and triangles.dtype.kind == "f", "Invalid actual contact triangles")
    normal = np.asarray(surface["surface"]["normal_world_zup"], dtype=np.float32)
    require(normal.shape == (3,) and np.isfinite(normal).all() and abs(np.linalg.norm(normal)-1) < .01,
        "Invalid contact normal")
    sdf_record = _record(stage2["source_sdf"])
    sdf = audit.json(sdf_record)
    require(sdf.get("schema") == "hsi.stage2.bound_sdf_cache.v1" and sdf.get("pose_image_or_future_motion_used") is False
        and sdf.get("source_kind") == "current_new_scene"
        and sdf["source_binding"] == {"stage1_execution": stage2["source_stage1"], "target": stage1["target"]},
        "SDF belongs to another target/plan or used future conditions")
    require(sdf.get("settings") == {"floor_ignore_height_m": .08, "maximum_target_floor_fraction": .10,
        "maximum_target_fraction": .25}, "Historical SDF settings differ")
    for key, name in (("occupancy", "scene_occupancy"), ("target_mask", "target_occupancy_mask")):
        require(sdf["inputs"][key] == _record(target["artifacts"][name]), "SDF raw geometry identity mismatch")
        audit.file(sdf["inputs"][key])
    require(set(sdf["arrays"]) == {"target_sdf", "collision_sdf"}, "Incomplete SDF pair")
    for record in sdf["arrays"].values():
        audit.file(record, 64 * 1024 * 1024)

    body = _make_body(body_path, arrays)
    require(torch.equal(body.betas.cpu().reshape(-1), torch.as_tensor(arrays["betas"])), "Body shape changed")
    target_motion = _target_motion(arrays)
    with torch.no_grad():
        mesh = body.mesh(target_motion[None, None])
    vertex_error = float((mesh.vertices[0, 0] - torch.as_tensor(arrays["vertices_world_zup"])).abs().max())
    joint_error = float((mesh.joints[0, 0] - torch.as_tensor(arrays["joints_world_zup"])).abs().max())
    require(max(vertex_error, joint_error) <= 5e-5, "Fixed-shape target mesh roundtrip mismatch")
    initial = _make_initial(body, initial_root[:2], float(initial_root[3]))
    require(np.array_equal(initial["betas"], arrays["betas"]), "Initial body shape changed")
    initial_pose = np.asarray(initial["motion"])
    require(initial_pose.shape == (135,) and np.isfinite(initial_pose).all()
        and np.allclose(initial_pose[:2], initial_root[:2], atol=2e-6, rtol=0), "Initial pose/start mismatch")
    permissions = {"root_xyz_locked": True, "global_orientation_locked": True, "shape_locked": True,
        "locked_joint_ids": list(range(22)), "root_tolerance_m": [0., 0., 0.], "joint_tolerance_rad": [0.] * 22,
        "stage2_static_refine_allowances_are_not_stage3_permission": True,
        "stage2_static_refine": {k: arrays[k].tolist() for k in (
            "active_body_pose_rows", "active_body_pose_delta_limits_rad", "contact_support_body_pose_rows", "collision_escape_body_pose_rows")}}
    tf = lambda x: torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32, device=device)
    ti = lambda x: torch.as_tensor(np.asarray(x).copy(), dtype=torch.long, device=device)
    proof = {"existing_historical_checks_not_reexecuted": True,
        "historical_handoff_allowed": True, "fresh_qwen_called": False, "full_physical_gates_reexecuted": False,
        "future_gt_pose_motion_or_times_used": False, "initialization_kind": "new_same_shape_neutral_standing_from_stage1_xy_yaw",
        "initial_pose_is_observed_motion": False, "target_pose_source": stage2["keypose"],
        "source_stage2": receipt_record, "source_stage1": stage2["source_stage1"], "source_body": body_record,
        "direct_artifacts_sha256_checked": list(audit.checked.values()),
        "transitive_source_closure_revalidated": False, "body_vertex_roundtrip_max_abs_m": vertex_error,
        "body_joint_roundtrip_max_abs_m": joint_error, "motion_generated": False, "motion_quality_passed": False,
        "sdf_arrays_hashed_but_not_loaded_or_recomputed": True,
        "scope": "trusted local artifact consistency, not cryptographic execution attestation"}
    return Stage12Case(SCHEMA, str(stage1["scene_id"]), stage1["instruction"], "sit", target_id,
        target["target_class"], surface_id, tf(initial_pose), target_motion.to(device), tf(arrays["betas"]), body.to(device),
        tf(initial["joints"]), tf(initial["vertices"]), tf(arrays["joints_world_zup"]), tf(arrays["vertices_world_zup"]),
        ti(faces), tf(route), tf(triangles), tf(arrays["contact_goal_world_xyz_zup_m"]), tf(normal),
        ti(arrays["contact_joint_ids"]), ti(butt), tf(arrays["root_xyz_yaw"][:3]), terminal, arrival, permissions,
        {"receipt": sdf_record, **sdf["arrays"], "layout": "XYZ", "shape": [300, 400, 100],
         "world_bounds": [list(x) for x in WORLD_BOUNDS], "bounds_kind": "voxel_outer_edges",
         "align_corners": False, "positive_is_free": True, "coordinate_system": "world_zup_m",
         "floor_ignore_height_m": .08, "ground_plane_z_m": 0., "full_scene_requires_target_union_collision_and_floor": True,
         "outside_is_unknown_not_free": True}, _record(target["artifacts"]["original_scene_mesh"]), descriptions, proof)
