"""Evaluate fresh P555/ReMoGen motion with original P523 + P519 numerics.

No H3-labelled evaluator or caller-supplied passed flag is accepted. This
entry closes the new compiler/launcher lineage, reconstructs all 10,475 body
vertices with the original corrected P519 joint-driven renderer, evaluates
the original P523 terminal/route/motion/collision metrics, and applies its
unchanged physical_release_policy_v2. It never edits generated motion and
never grants Memory credit; producer registration remains a separate task.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from memory_common import artifact, read_json, read_sealed, require, verified, write_once
from run_geometry_stage3 import (HERE, PROJECT, P523, P533, load_inputs, load_module,
                                 runtime_proof, validate_affordance_runtime)

EVALUATOR = P523 / "evaluation/code/evaluate_current_stage3_motion_v1.py"
RANKER = P523 / "evaluation/code/rank_and_select_current_stage3_v1.py"
POLICY = P523 / "evaluation/configs/current_physical_release_policy_v2.json"
P519 = PROJECT / "agent9/methods/p519_correct_lingo_motion_renderer_20260901/code/render_remogen_original_scene_corrected_v2.py"
ASSETS = PROJECT / "agent4/runs/kimodo_smplx_wave_20260721/runtime_assets/skeletons/smplx22"
SCHEMA = "p555.geometry_stage3_motion_evaluation.v1"


def load_execution(path):
    path = Path(path).resolve(strict=True)
    receipt = read_sealed(path)
    require(receipt.get("schema") == "p555.geometry_stage3_execution.v1"
            and receipt.get("dry_run") is False and receipt.get("error") is None
            and receipt.get("full22_runtime_proof") is not None, "No complete typed Stage3 execution evidence")
    launch_path = verified(receipt["launch"])
    require(launch_path == path.parent / "launch.json", "Execution launch path drift")
    launch = read_sealed(launch_path)
    require(launch["schema"] == "p555.geometry_stage3_launch.v1" and launch["dry_run"] is False,
            "Wrong current geometry launch")
    for source in launch["sources"]:
        verified(source)
    inputs = load_inputs(verified(launch["adapter"]))
    require(launch["packet"] == artifact(inputs["packet_path"])
            and launch["neutral_seed"] == artifact(inputs["seed_path"]), "Generation input binding drift")
    runtime = path.parent / "runtime"
    require(Path(launch["runtime_output"]).resolve() == runtime, "Runtime directory mismatch")
    required = ("generated_motion.npz", "generation_metadata.json", "energy_log.jsonl", "run_manifest.json")
    outputs = {}
    for name in required:
        outputs[name] = verified(receipt["outputs"][name])
        require(outputs[name] == runtime / name, "Output is not the actual fresh runtime artifact")
    physics = load_module("p555_evaluation_retained_full22", P533 / "stage3/code/run_p533_p478_v1.py")
    proof, metadata, rows = runtime_proof(physics, runtime, dry_run=False)
    require(proof == receipt["full22_runtime_proof"], "Full22 runtime attestation drift")
    require(launch["affordance"] in ("on", "off"), "Unknown affordance mode")
    affordance_proof = validate_affordance_runtime(metadata, rows, receipt["affordance_runtime"], inputs["packet"],
        enabled=launch["affordance"] == "on")
    require(affordance_proof == receipt["affordance_runtime_proof"], "Affordance runtime proof drift")
    return {"receipt": receipt, "launch": launch, "inputs": inputs, "outputs": outputs, "metadata": metadata}


def release_gates(evaluation, policy, ranker):
    """Apply exactly the retained policy, including mandatory post-skin gates."""
    require(policy["schema"] == "p523.current_only_stage3_physical_release_policy.v2", "Unexpected release policy")
    gates = {}
    for name, spec in policy["gates"].items():
        actual = ranker.nested(evaluation, name)
        if actual is None and spec.get("optional_legacy_alias"):
            actual = ranker.nested(evaluation, spec["optional_legacy_alias"])
        if actual is None and spec.get("none_means_zero"):
            parent = ranker.nested(evaluation, name.rsplit(".", 1)[0])
            if isinstance(parent, dict) and name.rsplit(".", 1)[1] in parent:
                actual = 0.0
        passed, _ = ranker.gate(actual, spec["comparison"], spec["threshold"])
        gates[name] = {"actual": actual, "threshold": spec["threshold"], "passed": passed}
    full = evaluation.get("fullmesh_scene_collision", {})
    delivery = policy["post_render_delivery_gate"]
    gates["fullmesh_available"] = {"passed": full.get("available") is True and full.get("vertex_count") == 10475}
    for metric, threshold_key in (("forbidden_collision_fraction", "fullmesh_structural_nonfloor_collision_fraction_max"),
            ("forbidden_penetration_depth_m.maximum", "fullmesh_structural_nonfloor_penetration_depth_m_max")):
        actual = ranker.nested(full.get("structural_nonfloor", {}), metric)
        if actual is None and metric.endswith("maximum") and isinstance(full.get("structural_nonfloor", {}).get("forbidden_penetration_depth_m"), dict):
            actual = 0.0
        passed, _ = ranker.gate(actual, "max", delivery[threshold_key])
        gates["fullmesh." + metric] = {"actual": actual, "threshold": delivery[threshold_key], "passed": passed}
    # A failed scheduler/runtime cannot be published even if some numeric fields happen to be favorable.
    gates["actual_runner_success"] = {"passed": evaluation["runner_returncode"] == 0}
    return gates


def skin_motion(renderer, motion_path, metadata_path, output, *, body_model_record, device):
    """Use the original P519 implementation, not a joints-only substitute."""
    import torch
    actual_model = artifact(ASSETS / "SMPLX_NEUTRAL.npz")
    require(actual_model["sha256"] == body_model_record["sha256"], "P519 and actual neutral body model differ")
    motion = renderer.load_motion(motion_path, metadata_path)
    require(np.array_equal(motion["betas"], np.zeros_like(motion["betas"])), "Generated body shape differs from neutral zero-shape seed")
    vertices, faces, audit = renderer.skin_exact_motion_joint_driven(
        local_rot_mats=motion["rotations"], posed_joints_world=motion["joints"], betas=motion["betas"],
        asset_dir=ASSETS, device=torch.device(device), chunk_size=32)
    require(vertices.shape == (len(motion["joints"]), 10475, 3) and np.isfinite(vertices).all(), "Incomplete P519 mesh sequence")
    path = output / "skinned_vertices.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, vertices_world=vertices, faces=faces)
    sources = {"neutral_asset": actual_model,
        "mean_hands": artifact(ASSETS / "mean_hands.npy") if (ASSETS / "mean_hands.npy").exists() else None}
    receipt = write_once(output / "skinning_receipt.json", {"schema": "p555.actual_p519_motion_skinning.v1",
        "source": artifact(P519), "source_motion": artifact(motion_path), "source_metadata": artifact(metadata_path),
        "body_sources": sources, "outputs": {"vertices": artifact(path)}, "audit": audit,
        "all_source_frames_preserved": True, "frame_count": len(vertices), "vertex_count": 10475,
        "source_motion_edited": False, "renderer_original_numeric_function_used": True})
    return path, motion, receipt


def numeric_evaluation(old, joints, packet, metadata, occupancy_path, skin_path, butt_ids):
    """Unchanged P523 metrics and fixed P548/P523 bounds/20 Hz policy."""
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
    quality = old.motion_quality(joints, fps=20., foot_height=.10)
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


def evaluate(execution_path, output, *, device="cuda"):
    require(device in ("cpu", "cuda"), "Invalid skinning device")
    started = time.monotonic()
    state = load_execution(execution_path)
    inputs, paths, metadata = state["inputs"], state["outputs"], state["metadata"]
    packet, adapter = inputs["packet"], inputs["adapter"]
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite evaluation")
    output.mkdir(parents=True)
    sources = [artifact(path) for path in (Path(__file__), EVALUATOR, RANKER, POLICY, P519,
        HERE / "run_geometry_stage3.py", HERE / "compile_geometry_stage3.py", HERE / "neutral_seed.py")]
    old = load_module("p555_retained_original_motion_metrics", EVALUATOR)
    ranker = load_module("p555_retained_original_release_gates", RANKER)
    old.validate_fresh_metadata(metadata, packet, inputs["occupancy"])
    renderer = load_module("p555_retained_p519_skinning", P519)
    before = artifact(paths["generated_motion.npz"])
    skin_start = time.monotonic()
    skin_path, motion, skin_receipt = skin_motion(renderer, paths["generated_motion.npz"], paths["generation_metadata.json"], output,
        body_model_record=adapter["inputs"]["body_model"], device=device)
    skin_seconds = time.monotonic()-skin_start
    ids = np.asarray(adapter["surface_contract"]["fixed_generated_butt_vertex_indices"], dtype=np.int64)
    require(ids.ndim == 1 and len(ids) > 0 and len(np.unique(ids)) == len(ids)
            and np.all((ids >= 0) & (ids < 10475)), "Invalid current body's actual gluteal vertex IDs")
    metrics_start = time.monotonic()
    metrics = numeric_evaluation(old, motion["joints"], packet, metadata, inputs["occupancy"], skin_path, ids)
    result = {"schema": SCHEMA, "status": "complete_current_geometry_motion_evaluation",
        "evaluation_complete": True, "input_contract_valid": True, "scene_id": packet["scene_name"],
        "instruction": packet["text"], "source_execution": artifact(execution_path),
        "source_adapter": artifact(inputs["adapter_path"]), "source_packet": artifact(inputs["packet_path"]),
        "source_motion": before, "source_metadata": artifact(paths["generation_metadata.json"]),
        "source_energy_log": artifact(paths["energy_log.jsonl"]), "sources": sources,
        "skinning": artifact(output / "skinning_receipt.json"), "skinned_vertices": artifact(skin_path),
        "runner_returncode": state["receipt"]["runner_returncode"], "affordance": state["launch"]["affordance"],
        "actual_provider": "p555_current_geometry_ik", "generated_motion_modified": False,
        "h3_provider_claimed": False, "positive_credit": 0, **metrics}
    gates = release_gates(result, read_json(POLICY), ranker)
    failures = [name for name, value in gates.items() if value["passed"] is not True]
    result.update(release_gates=gates, failed_release_gates=failures, motion_publishable=not failures,
        thresholds_relaxed=False, timings={"actual_fullmesh_skinning_seconds": skin_seconds,
            "original_metrics_seconds": time.monotonic()-metrics_start, "total_seconds": time.monotonic()-started},
        credit_registration_performed=False, successful_motion_is_only_pending_producer_registration=True)
    require(artifact(paths["generated_motion.npz"]) == before, "Motion changed during evaluation")
    for record in sources:
        verified(record)
    write_once(output / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True, help="P555 actual Stage3 receipt.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = evaluate(args.execution, args.output, device=args.device)
    print({"motion_publishable": result["motion_publishable"], "failed_gates": result["failed_release_gates"]}, flush=True)
    return 0 if result["motion_publishable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
