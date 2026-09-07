"""Current full-mesh motion metrics and strict source-bound release evaluation.

The numerical combination is the unchanged P555/P523 evaluator: 20 Hz,
four-frame contact tail, factor-two conservative occupancy pooling, original
world bounds, 8 cm floor split, and all 34 current release gates. Full-mesh
data must come from an explicitly admitted joint-driven backend, never from
22-joint interpolation or a favorable caller-supplied pass flag.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any
import time

import numpy as np

from hsi.common.artifacts import artifact, read_json, read_sealed, require, verified, write_once
from . import metrics as old
from .release import current_policy, release_gates


def validate_fresh_metadata(metadata, packet, raw_occupancy_path):
    """Retained current-only provenance checks, independent of favorable metrics."""
    require(metadata.get("scene_name") == packet.get("scene_name"), "metadata/packet scene drift")
    require(metadata.get("plan_seq_name_from_file") == packet.get("seq_name"), "metadata/packet sequence drift")
    require(metadata.get("plan_provenance") == packet.get("provenance"), "metadata/packet provenance drift")
    for key in ("lingo_motion_opened", "posthoc_motion_edit_used", "future_frame_2_plus_available_to_runtime",
                "generator_reads_future_root_directly_from_dataset", "gt_contact_or_icgf_used"):
        require(metadata.get(key) is False, "Current-only generation boundary failed: " + key)
    require(metadata.get("generated_only_output") is True, "Motion is not a generated-only output")
    runtime = metadata.get("query_static_runtime_audit")
    require(isinstance(runtime, Mapping), "Fresh query-static runtime audit is missing")
    require(runtime.get("future_or_gt_motion_used") is False and runtime.get("lingo_motion_opened") is False,
            "Query-static runtime is not fresh/current-only")
    occ = metadata.get("raw_occupancy")
    require(isinstance(occ, Mapping), "Metadata raw occupancy contract is missing")
    require(Path(str(occ.get("source_path", ""))).resolve(strict=True) == Path(raw_occupancy_path).resolve(strict=True),
            "Metadata raw occupancy path drift")


def load_motion(motion_path, metadata_path, packet):
    """Read original ReMoGen generated arrays without FK replacement or edits."""
    motion_path, metadata_path = Path(motion_path), Path(metadata_path)
    metadata = read_json(metadata_path)
    with np.load(motion_path, allow_pickle=False) as archive:
        require({"joints", "global_orient", "body_pose", "betas"} <= set(archive.files),
                "Generated motion lacks joints, local rotations or explicit betas")
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    joints = arrays["joints"]
    require(joints.dtype == np.float32 and joints.ndim == 3 and joints.shape[1:] == (22, 3)
            and len(joints) >= 4 and np.isfinite(joints).all(),
            "Current metrics need float32 [T>=4,22,3] generated joints")
    frames = len(joints)
    for name, shape in (("global_orient", (frames,3,3)), ("body_pose", (frames,21,3,3))):
        require(arrays[name].dtype == np.float32 and arrays[name].shape == shape and np.isfinite(arrays[name]).all(),
                "Invalid generated local rotation array: " + name)
    rotations = np.ascontiguousarray(np.concatenate((arrays["global_orient"][:, None], arrays["body_pose"]), axis=1))
    require(np.max(np.abs(np.swapaxes(rotations,-1,-2) @ rotations - np.eye(3))) <= 2e-3
            and np.max(np.abs(np.linalg.det(rotations)-1)) <= 2e-3, "Generated rotations are not SO(3)")
    betas = arrays["betas"]
    require(betas.dtype == np.float32 and betas.shape in ((10,), (1,10), (frames,10))
            and np.isfinite(betas).all(), "Invalid generated body-shape layout")
    expanded_betas = np.broadcast_to(betas.reshape(-1,10), (frames,10)).copy()
    require(np.array_equal(expanded_betas, np.zeros_like(expanded_betas)),
            "Generated body differs from the current neutral zero-shape seed")
    for key in ("scene_name", "text"):
        expected = packet["scene_name" if key == "scene_name" else "text"]
        if key in arrays:
            require(arrays[key].shape == () and arrays[key].item() == expected, "Motion/packet scalar drift: " + key)
    require(metadata.get("scene_name") == packet["scene_name"], "Motion metadata/packet scene drift")
    for key in ("retrieval_used", "posthoc_motion_edit_used"):
        require(metadata.get(key) is False, "Runtime must explicitly exclude " + key)
        if key in arrays:
            require(arrays[key].shape == () and type(arrays[key].item()) is bool and arrays[key].item() is False,
                    "Motion archive declares retrieval or posthoc editing")
    return {"joints": joints, "rotations": rotations, "betas": expanded_betas,
            "arrays": arrays, "metadata": metadata}


def admit_stage2(stage2_path, packet):
    """Revalidate actual H3/HybrIK-X/full-SMPL-X evidence and exact terminal body."""
    from hsi.stage2.pipeline import validate_success
    stage2_path = Path(stage2_path).resolve(strict=True)
    stage2 = validate_success(stage2_path)
    require(stage2.get("stage3_handoff_allowed") is True, "Stage2 did not authorize Stage3")
    keypose_path = verified(stage2["keypose"])
    with np.load(keypose_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    shapes = {"vertices_world_zup": (10475,3), "faces": (20908,3), "joints_world_zup": (22,3),
              "root_xyz_yaw": (4,), "body_pose_axis_angle": (21,3), "betas": (10,)}
    for key, shape in shapes.items():
        require(key in arrays and arrays[key].shape == shape and arrays[key].dtype.kind in "fiub"
                and np.isfinite(arrays[key]).all(), "Stage2 full body field missing/invalid: " + key)
    require(arrays["faces"].dtype.kind in "iu" and arrays["faces"].min() >= 0
            and arrays["faces"].max() < 10475, "Stage2 topology is not the full current SMPL-X body")
    ids = arrays.get("fixed_generated_butt_vertex_indices", np.array([]))
    require(ids.dtype.kind in "iu" and ids.ndim == 1 and len(ids)>0 and len(np.unique(ids))==len(ids)
            and np.all((ids>=0)&(ids<10475)), "Invalid actual gluteal contact vertex IDs")
    require(packet.get("schema") == "hsi.stage3.image_keypose_packet.v1"
            and packet.get("provenance") == "hsi_verified_h3_hybrikx"
            and packet.get("coordinate_system") == "world_zup", "Wrong integrated Stage3 packet")
    require(packet["source_artifacts"]["stage2"] == artifact(stage2_path)
            and packet["source_artifacts"]["candidate"] == stage2["keypose"],
            "Packet provenance does not bind this exact admitted Stage2 publication")
    terminal = packet["nodes"][-1]["full_smplx_keypose"]
    require(terminal["source_stage2_keypose_sha256"] == stage2["keypose"]["sha256"],
            "Packet terminal body is not the admitted Stage2 body")
    for key in ("joints_world_zup", "root_xyz_yaw", "body_pose_axis_angle", "betas", "pose_frame_to_world_rotation"):
        value = np.asarray(terminal[key])
        require(value.shape == arrays[key].shape and np.allclose(value, arrays[key], atol=2e-6, rtol=0),
                "Packet/Stage2 terminal field differs: " + key)
    require(np.array_equal(arrays["betas"], np.zeros(10)), "Current runtime requires the admitted neutral body")
    require(str(arrays["scene_id"].item()) == packet["scene_name"], "Stage2/packet scene differs")
    return stage2, arrays


def numeric_evaluation(joints, packet, metadata, occupancy_path, skin_path, butt_ids, *, official_sources):
    """Unchanged P523 metrics and fixed world-bounds/20 Hz policy."""
    require(np.asarray(joints).ndim == 3 and np.asarray(joints).shape[1:] == (22,3)
            and len(joints) >= 4 and np.isfinite(joints).all(), "Incomplete/nonfinite generated joint sequence")
    require(skin_path is not None, "Actual full-mesh skinning is required, not a joints-only evaluation")
    with np.load(skin_path, allow_pickle=False) as archive:
        require("vertices_world" in archive.files, "Full-mesh archive lacks vertices_world")
        vertices = archive["vertices_world"]
        require(vertices.shape == (len(joints),10475,3) and vertices.dtype.kind == "f"
                and np.isfinite(vertices).all(), "Full-mesh must cover every frame with 10475 finite vertices")
    raw = np.load(occupancy_path, allow_pickle=False)
    occupancy, layout, conversion = old.occupancy_to_zup(raw, "auto")
    occupancy = old.conservative_pool(occupancy, 2)
    lower, upper = np.array([-3., -4., 0.]), np.array([3., 4., 2.])
    sdf, spacing = old.build_signed_sdf(occupancy, lower, upper)
    structural = occupancy.copy()
    floor = lower[2] + (np.arange(occupancy.shape[2]) + .5)*spacing[2] <= .08
    structural[:, :, floor] = False
    structural_sdf, structural_spacing = old.build_signed_sdf(structural, lower, upper)
    require(np.allclose(spacing, structural_spacing), "Structural SDF spacing drift")
    terminal = old.terminal_metrics(joints, packet, tail_frames=4)
    route = old.route_metrics(joints, packet, metadata)
    terminal_start = int(route["phase_split"]["terminal_start_frame_inclusive"])
    quality = old.motion_quality(joints, fps=20., foot_height=.10, official_sources=official_sources)
    collision = old.collision_metrics(joints, sdf, structural_sdf, lower, upper,
        tuple(packet["nodes"][-1]["contact_joint_ids"]), 4, terminal_start, .08, .02)
    fullmesh, _ = old.fullmesh_metrics(skin_path, sdf, structural_sdf, lower, upper,
        {"fixed_generated_butt_vertex_indices": butt_ids}, 4, terminal_start)
    require(fullmesh["available"] is True and fullmesh["frame_count"] == len(joints), "Full-mesh/frame coverage incomplete")
    rolewise = metadata.get("p478_terminal_rolewise_completion", {})
    return {"execution": {"frames": int(len(joints)), "scheduler_complete": bool(metadata.get("scheduler_complete", False)),
        "nodes_consumed": int(metadata.get("nodes_consumed", 0)), "plan_node_count": len(packet["nodes"]),
        "all_nodes_consumed": int(metadata.get("nodes_consumed", 0)) == len(packet["nodes"]),
        "rolewise_terminal_complete": rolewise.get("terminal_complete"),
        "rolewise_fail_closed_incomplete": rolewise.get("fail_closed_incomplete"), "posthoc_motion_edit_used": False},
        "terminal": terminal, "route": route, "motion_quality": quality, "scene_collision": collision,
        "fullmesh_scene_collision": fullmesh, "occupancy_contract": {"input_layout": layout, "conversion": conversion,
            "sdf_shape_xyz": list(sdf.shape), "voxel_spacing_m": spacing.tolist(),
            "scene_lower_xyz_m": lower.tolist(), "scene_upper_xyz_m": upper.tolist(),
            "target_component_removed": False, "floor_exclusion_height_world_z_m": .08}}


def evaluate(execution_path, output, *, official_sources, mean_hands_path=None, device="cpu"):
    """Reconstruct the actual full mesh and apply every current physical gate.

    The runtime validator owns process/weight/source attestation. This evaluator
    additionally revalidates Stage2, the packet body, generated arrays and skin
    identity; it never promotes a stored caller-provided metric or success flag.
    """
    from .runtime import validate_execution
    from . import skin
    from .seed import NEUTRAL_SHA256

    require(device in ("cpu", "cuda"), "Unsupported full-mesh evaluation device")
    require(set(official_sources) == {"metrics_unified", "lingo_evaluation_driver"},
            "Exact official ReMoGen metric source paths are required")
    official_sources = {key: Path(path).resolve(strict=True) for key,path in official_sources.items()}
    started = time.monotonic()
    execution_path = Path(execution_path).resolve(strict=True)
    state = validate_execution(execution_path)
    required = {"receipt", "packet_path", "metadata_path", "motion_path", "stage2_path", "occupancy_path",
                "body_model_record", "runner_returncode"}
    require(isinstance(state,dict) and required <= set(state), "Incomplete typed runtime evaluation handoff")
    require(state["receipt"].get("schema") == "hsi.stage3.execution.v1", "Wrong execution lineage")
    require(type(state["runner_returncode"]) is int, "Runtime return code must be actual integer process status")
    paths = {key: Path(state[key]).resolve(strict=True) for key in
             ("packet_path", "metadata_path", "motion_path", "stage2_path", "occupancy_path")}
    before = {key: artifact(path) for key,path in paths.items()}
    before["execution"] = artifact(execution_path)
    packet = read_sealed(paths["packet_path"])
    stage2, body = admit_stage2(paths["stage2_path"], packet)
    motion = load_motion(paths["motion_path"], paths["metadata_path"], packet)
    metadata = motion["metadata"]
    validate_fresh_metadata(metadata, packet, paths["occupancy_path"])
    require(type(metadata.get("scheduler_complete")) is bool
            and type(metadata.get("nodes_consumed")) is int, "Scheduler metadata must retain actual typed values")
    body_model_record = state["body_model_record"]
    verified(body_model_record)
    require(body_model_record["sha256"] == NEUTRAL_SHA256, "Full-mesh evaluator requires the pinned current neutral body")
    physical = read_sealed(verified(stage2["physical_refine"]))
    selection = read_sealed(verified(physical["outputs"]["selection"]))
    trial = read_json(verified(selection["selected_refine_receipt"]))
    require(trial["inputs"]["fixed_neutral_smplx"]["sha256"] == body_model_record["sha256"],
            "Motion skinning body differs from the actually admitted Stage2 body")
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite evaluation")
    output.mkdir(parents=True)
    sources = [artifact(Path(__file__)), artifact(Path(old.__file__)), artifact(Path(skin.__file__)),
               *[artifact(path) for path in official_sources.values()]]
    skin_start = time.monotonic()
    skin_path, skin_receipt = skin.skin_motion(motion, paths["motion_path"], paths["metadata_path"], output,
        body_model_record=body_model_record, expected_faces=body["faces"],
        mean_hands_path=mean_hands_path, device=device)
    skin_seconds = time.monotonic()-skin_start
    metric_start = time.monotonic()
    numbers = numeric_evaluation(motion["joints"], packet, metadata, paths["occupancy_path"],
        skin_path, body["fixed_generated_butt_vertex_indices"], official_sources=official_sources)
    result = {"schema": "hsi.stage3.fullmesh_motion_evaluation.v1",
        "status": "complete_current_fullmesh_motion_evaluation", "evaluation_complete": True,
        "input_contract_valid": True, "scene_id": packet["scene_name"], "instruction": packet["text"],
        "source_execution": before["execution"], "source_packet": before["packet_path"],
        "source_motion": before["motion_path"], "source_metadata": before["metadata_path"],
        "source_stage2": before["stage2_path"], "source_occupancy": before["occupancy_path"],
        "sources": sources, "skinning": artifact(skin_receipt), "skinned_vertices": artifact(skin_path),
        "runner_returncode": state["runner_returncode"], "generated_motion_modified": False,
        "positive_credit": 0, "credit_registration_performed": False, **numbers}
    gates = release_gates(result, current_policy())
    require(len(gates) == 34, "Current full-mesh release must retain all 34 gates")
    failed = [name for name,value in gates.items() if value["passed"] is not True]
    result.update(release_gates=gates, failed_release_gates=failed, motion_publishable=not failed,
        thresholds_relaxed=False, timings={"actual_fullmesh_skinning_seconds": skin_seconds,
            "original_metrics_seconds": time.monotonic()-metric_start, "total_seconds": time.monotonic()-started},
        successful_motion_is_only_pending_producer_registration=True)
    for record in [*before.values(), *sources, body_model_record]:
        verified(record)
    write_once(output / "receipt.json", result)
    return result
