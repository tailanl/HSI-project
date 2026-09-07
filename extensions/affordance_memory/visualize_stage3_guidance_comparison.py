"""Read-only, CPU-only comparison of one actual P555 v2 OFF/ON pair.

This is a receipt/log report, not a new evaluation, render, model execution,
or positive-experience producer. Network hashes are compared in the existing
independently rehashed pre/post receipts; checkpoint files are NOT read again.
No GPU/model modules are imported. All plots distinguish proxy-mask activation
from physical contact and publication. Output directories must be new.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any, Mapping

from memory_common import (PROJECT, MemoryContractError, artifact, canonical_bytes,
                           digest, read_json, read_sealed, require, verified, write_once,
                           _unique_pairs)

SOURCE = Path(__file__).resolve()
HERE = SOURCE.parent
P523 = PROJECT / "agent9/methods/p523_current_only_multiscene_20260902/evaluation"
POLICY = P523 / "configs/current_physical_release_policy_v2.json"
RANKER = P523 / "code/rank_and_select_current_stage3_v1.py"
RETAINED_EVALUATOR = HERE / "evaluate_geometry_stage3.py"
PINS = {
    POLICY: "47baa1fc9b5b0107e74bceee1d44eba7e908cabd5b0828b7efaa1faa180dc9d6",
    RANKER: "877f2eb5a5ff8c7e2cf6ded84b16825afb9f19bb6df13ad80a9a553cdc4c2201",
    RETAINED_EVALUATOR: "b3e49c5c0ae3751982b37f3d5b177fdbd104227de3eee9436ace69e9b37393c3",
    HERE / "run_geometry_stage3_v2.py": "7391c95364449afa87c2164a0ee558a9841f4eb7fd817224d80d34341ca4e9fd",
    HERE / "evaluate_geometry_stage3_v2.py": "f399fb0782cc4e706d9b0741eb49151f37fee1e6e7ae8f2aa4cb80df14997d3b",
    HERE / "stage3_affordance_guidance_v2.py": "a8aaa132544a3cdeef5265299559921a42651aeb3a8a60d1b80ae2f009beb27b",
}
CHECKPOINTS = {
    "adapter_checkpoint": "519de22df9d7f3e2451c80e3029bbf3cba6cd86c0d54ae972fcb2291f75b7a99",
    "base_checkpoint": "373fcc0682d095e67d083f5976b014c8562732f6e104f797273a656d5a10089e",
    "mvae_checkpoint": "e1c103cc9d5a4916adb3261bdf79e69d599a3aa4960b669b4969c1c786a58773",
    "clip_vit_b32": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}
ADDED_FIELDS = (
    "p555_affordance_energy", "p555_contact_weighted_energy", "p555_support_weighted_energy",
    "p555_facing_weighted_energy", "p555_frame_safety_fraction", "p555_contact_active_fraction",
    "p555_support_active_fraction", "p555_original_obstacle_min_sdf_m",
)
PARENT_FIELDS = ("sdf_energy", "p533_physics_total_energy", "p533_combined_parent_and_physics_energy")


def artifact_shape(record):
    require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}
            and isinstance(record["path"], str) and Path(record["path"]).is_absolute()
            and type(record["bytes"]) is int and record["bytes"] >= 0
            and isinstance(record["sha256"], str) and re.fullmatch("[0-9a-f]{64}", record["sha256"]),
            "Malformed recorded artifact")


class Reader:
    """Only explicitly requested artifacts are read; never recurse into weights."""
    def __init__(self):
        self.records = {}

    def path(self, record):
        artifact_shape(record)
        key = record["path"]
        if key in self.records:
            require(self.records[key] == record, "Conflicting artifact bindings")
            return Path(key)
        path = verified(record)
        self.records[key] = dict(record)
        return path

    def json(self, record, *, sealed=True):
        path = self.path(record)
        return read_sealed(path) if sealed else read_json(path)

    def recheck(self):
        for record in self.records.values():
            verified(record)


def retained_gate_function(reader):
    """Compile only the frozen, original three pure gate functions, no imports."""
    for path, expected in PINS.items():
        row = artifact(path)
        require(row["sha256"] == expected, "Unregistered report source revision: " + str(path))
        reader.path(row)
    env = {"require": require, "math": math, "Mapping": Mapping, "Any": Any,
           "SelectionContractError": MemoryContractError}
    for path, names in ((RANKER, {"nested", "gate"}), (RETAINED_EVALUATOR, {"release_gates"})):
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        require({node.name for node in selected} == names, "Frozen gate functions missing")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    ranker = SimpleNamespace(nested=env["nested"], gate=env["gate"])
    policy = read_json(POLICY)
    return lambda value: env["release_gates"](value, policy, ranker)


def validate_gates(evaluation, gate_function):
    actual = gate_function(evaluation)
    require(len(actual) == 34 and actual == evaluation["release_gates"], "Original 34 release gates drift")
    failures = [name for name, row in actual.items() if row["passed"] is not True]
    require(failures == evaluation["failed_release_gates"]
            and evaluation["motion_publishable"] is (not failures), "Publication/failure summary differs from gates")
    require(evaluation["thresholds_relaxed"] is False and evaluation["credit_registration_performed"] is False
            and type(evaluation["positive_credit"]) is int and evaluation["positive_credit"] == 0,
            "Comparison cannot grant credit or relax gates")


def normalized_argv(launch):
    argv = launch["actual_runner_argv"]
    require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), "Invalid actual argv")
    flags = [x for x in argv if x.startswith("--")]
    require(len(flags) == len(set(flags)), "Duplicate argv option")
    require(argv.count("--output-dir") == 1, "Missing unique runtime output option")
    index = argv.index("--output-dir") + 1
    require(index < len(argv) and argv[index] == launch["runtime_output"], "Runtime output flag mismatch")
    result = list(argv)
    result[index] = "<PER_RUN_RUNTIME_OUTPUT>"
    for flag, expected in (("--seed", str(launch["seed"])), ("--plan", launch["packet"]["path"]),
                           ("--query-static-seed", launch["neutral_seed"]["path"]),
                           ("--raw-occupancy", launch["guidance_binding"]["source_bindings"]["raw_occupancy"]["path"])):
        require(flag in argv and argv.index(flag)+1 < len(argv) and argv[argv.index(flag)+1] == expected,
                "Actual command input differs from launch: " + flag)
    require("--affordance" not in argv, "Unregistered internal affordance option")
    return result


def network_proof(reader, execution, launch, manifest):
    require(execution["network_assets_verified_pre_and_post"] is True, "Missing network pre/post proof")
    pre = reader.json(execution["network_preflight"])
    post = reader.json(execution["network_postflight"])
    source = artifact(HERE / "run_geometry_stage3_v2.py")
    require(pre["schema"] == "p555.actual_network_asset_preflight.v2"
            and post["schema"] == "p555.actual_network_asset_postflight.v2"
            and post["preflight"] == execution["network_preflight"]
            and pre["source"] == post["source"] == source
            and source in launch["sources"], "Network receipt lineage/source mismatch")
    require(pre["actual_runner_argv"] == launch["actual_runner_argv"]
            and pre["checkpoint_hash_policy"] == CHECKPOINTS
            and pre["checkpoint_and_config_actual_hashes_read_before_runtime"] is True
            and pre["historical_or_future_motion_read"] is False
            and pre["model_loaded_by_preflight"] is False
            and post["all_actual_files_rehashed_unchanged"] is True
            and post["actual_checkpoint_paths_and_sha_match_independent_preflight"] is True
            and post["runtime_manifest"] == execution["outputs"]["run_manifest.json"], "Incomplete actual network proof")
    records = pre["actual_file_artifacts"]
    required = set(CHECKPOINTS) | {"primitive_config", "base_args", "mvae_args", "normalization_statistics", "text_embedding_cache"}
    require(required <= set(records), "Network freeze lacks model/config/statistics")
    require(any(key.startswith("source:") for key in records)
            and all("clip_source:"+name in records for name in
                    ("clip.py", "model.py", "simple_tokenizer.py", "__init__.py", "bpe_simple_vocab_16e6.txt.gz")),
            "Network freeze lacks actual source/tokenizer bindings")
    for row in records.values():
        artifact_shape(row)  # Intentionally NOT verified(row): no checkpoint rehash.
    for role, checksum in CHECKPOINTS.items():
        require(records[role]["sha256"] == checksum, "Unregistered recorded checkpoint")
    imported = post["actual_imported_network_sources"]
    require(imported and len(set(imported)) == len(imported)
            and set(imported) <= {row["path"] for row in records.values()}, "Imported source outside preflight closure")
    actual = manifest["input_provenance"]["artifacts"]
    for role, row in [(name, records[name]) for name in ("adapter_checkpoint", "base_checkpoint", "mvae_checkpoint")]:
        require(actual[role] == {"exists": True, "path": row["path"], "sha256": row["sha256"]},
                "Runtime manifest differs from independent checkpoint proof")
    return pre, post


def read_energy(path):
    require(path.stat().st_size < 64*1024*1024, "Energy log too large for this bounded report")
    rows = []
    for line in path.read_text().splitlines():
        require(line.strip(), "Unexpected blank energy row")
        value = json.loads(line, object_pairs_hook=_unique_pairs)
        canonical_bytes(value)  # Reject NaN/Infinity, including unused numeric fields.
        require(isinstance(value, dict), "Energy row must be an object")
        rows.append(value)
    require(rows, "Empty actual energy log")
    return rows


def validate_energy(rows, metadata, execution, packet, mode):
    enabled = mode == "on"
    primitives = metadata["primitive_records"]
    pids = [p["primitive_id"] for p in primitives]
    require(pids == list(range(len(pids))), "Noncontiguous actual primitive sequence")
    terminal = packet["nodes"][-1]["ordinal"]
    ordinals = {node["ordinal"] for node in packet["nodes"]}
    expected = [(p["primitive_id"], i) for p in primitives for i in range(p["energy_log_records"])]
    require([(r["primitive_id"], r["call_index"]) for r in rows] == expected,
            "Energy rows missing, reordered, duplicated or from another primitive")
    for row in rows:
        require(row.get("record_type") == "denoise_energy" and type(row["primitive_id"]) is int
                and type(row["call_index"]) is int, "Not a typed actual denoise row")
        primitive = primitives[row["primitive_id"]]
        ordinal = row["active_external_ordinal"]
        require(type(ordinal) is int and ordinal in ordinals and ordinal == primitive["active_external_ordinal"],
                "Energy scheduler phase mismatch")
        for key in PARENT_FIELDS:
            require(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0,
                    "Missing/nonfinite original parent energy: " + key)
        if not enabled:
            require(not any(key.startswith("p555_") for key in row), "Disabled parent path contains added guidance")
            continue
        require(all(key in row for key in ADDED_FIELDS), "Missing actual v2 energy/activation fields")
        for key in ADDED_FIELDS:
            require(type(row[key]) in (int, float) and math.isfinite(row[key]), "Nonfinite actual guidance field")
        for key in ("p555_contact_active_fraction", "p555_support_active_fraction", "p555_frame_safety_fraction"):
            require(0 <= row[key] <= 1+1e-6, "Invalid active/safety fraction")
        require(row["p555_guidance_enabled"] == 1.0 and row["p555_v2_contact_discretization_guard"] == 1.0
                and row["p555_current_external_ordinal"] == ordinal
                and row["p555_current_node_is_terminal"] == float(ordinal == terminal)
                and row["p555_original_sdf_additive_delta"] == 0.0, "V2 phase or retained SDF guard drift")
        total = sum(row[k] for k in ("p555_contact_weighted_energy", "p555_support_weighted_energy", "p555_facing_weighted_energy"))
        require(total >= 0 and math.isclose(total, row["p555_affordance_energy"], rel_tol=2e-5, abs_tol=1e-7),
                "Added energy components do not match total")
        require(math.isclose(row["p555_combined_parent_and_affordance_energy"],
                            row["p533_combined_parent_and_physics_energy"]+total, rel_tol=2e-5, abs_tol=1e-6),
                "Added energy did not retain parent energy")
        if ordinal != terminal:
            require(row["p555_contact_active_fraction"] == row["p555_contact_raw_energy_m2"]
                    == row["p555_facing_raw_energy"] == 0, "Contact attraction leaked into navigation")
    terminal_rows = [r for r in rows if r["active_external_ordinal"] == terminal]
    full22 = execution["full22_runtime_proof"]
    require(full22["actual_model_generation"] is True and full22["all_terminal_denoise_calls_exact"] is True
            and full22["terminal_denoise_call_count"] == len(terminal_rows), "Full22 terminal runtime count drift")
    for row in terminal_rows:
        require(row["p533_actual_icgf_constraint_count"] == 88
                and row["p533_icgf_query_joint_count"] == 22
                and row["p533_icgf_query_frame_count"] == 4, "Original full22 constraints changed")
    stats, proof = execution["affordance_runtime"], execution["affordance_runtime_proof"]
    require(stats["enabled"] is enabled and proof["enabled"] is enabled
            and proof["schema"] == "p555.contact_aware_runtime_proof.v2", "Actual runtime mode/schema differs")
    if enabled:
        require(stats["energy_calls"] == proof["observed_denoise_rows"] == len(rows)
                and stats["terminal_energy_calls"] == proof["terminal_denoise_rows"] == len(terminal_rows)
                and stats["nonterminal_energy_calls"] == len(rows)-len(terminal_rows)
                and proof["original_parent_sdf_and_physics_retained"] is True
                and proof["all_observed_primitives_used_v2_context"] is True,
                "Actual guidance counters/retained parents mismatch")
    else:
        require(stats["energy_calls"] == stats["condition_instances"] == proof["observed_denoise_rows"] == 0
                and proof["exact_parent_path"] is True, "OFF is not the retained parent-only path")
    active = [r for r in rows if enabled and r["p555_contact_active_fraction"] > 0]
    summary = {"actual_denoise_rows": len(rows), "terminal_denoise_rows": len(terminal_rows),
               "added_guidance_enabled": enabled, "added_fields_observed": enabled,
               "contact_activation_field": "p555_contact_active_fraction" if enabled else None,
               "contact_active_rows": len(active) if enabled else None,
               "terminal_contact_active_rows": sum(r["active_external_ordinal"] == terminal for r in active) if enabled else None,
               "contact_active_rows_with_nonzero_schedule": sum(r["schedule"] > 0 for r in active) if enabled else None,
               "contact_active_rows_with_nonzero_weighted_energy": sum(r["p555_contact_weighted_energy"] > 0 for r in active) if enabled else None,
               "activation_is_proxy_mask_not_physical_contact_success": True,
               "off_absent_fields_not_imputed_as_observed_zeros": True}
    timeline = [{"denoise_row": i, "primitive_id": r["primitive_id"], "call_index": r["call_index"],
                 "active_external_ordinal": r["active_external_ordinal"],
                 "terminal": r["active_external_ordinal"] == terminal, "schedule": r["schedule"],
                 **{key: r[key] for key in PARENT_FIELDS},
                 **{key: r[key] if enabled else None for key in ADDED_FIELDS}}
                for i, r in enumerate(rows)]
    return summary, timeline


def load_case(execution_path, evaluation_path, mode, reader, gate_function):
    source_execution, source_evaluation = artifact(execution_path), artifact(evaluation_path)
    execution, evaluation = reader.json(source_execution), reader.json(source_evaluation)
    require(execution["schema"] == "p555.geometry_stage3_execution.v2"
            and evaluation["schema"] == "p555.geometry_stage3_motion_evaluation.v2", "Only actual v2 pair is supported")
    require(execution["status"] == "generated_pending_fullmesh_evaluation"
            and execution["dry_run"] is False and "error" in execution and execution["error"] is None
            and type(execution["runner_returncode"]) is int and execution["runner_returncode"] == 0,
            "No successful actual generation execution")
    require(evaluation["source_execution"] == source_execution
            and evaluation["status"] == "complete_current_geometry_motion_evaluation"
            and evaluation["evaluation_complete"] is True and evaluation["input_contract_valid"] is True
            and evaluation["generated_motion_modified"] is False and evaluation["h3_provider_claimed"] is False
            and type(evaluation["runner_returncode"]) is int and evaluation["runner_returncode"] == 0,
            "Incomplete/cross-execution evaluation")
    launch = reader.json(execution["launch"])
    require(launch["schema"] == "p555.geometry_stage3_launch.v2" and launch["affordance"] == evaluation["affordance"] == mode
            and launch["provider"] == evaluation["actual_provider"] == "p555_current_geometry_ik_v2"
            and launch["dry_run"] is False and type(launch["seed"]) is int,
            "Wrong mode, seed or provider")
    for key in ("full22_icgf_retained", "parent_raw_sdf_retained"):
        require(launch[key] is True, "Original guidance not retained")
    for key in ("network_and_checkpoint_parameters_changed", "unknown_provenance_bypass", "old_provider_admission_used"):
        require(launch[key] is False, "Unregistered changed/bypassed input")
    require(type(execution["positive_credit"]) is int and execution["positive_credit"] == launch["positive_credit"] == 0,
            "Report cannot admit positive credit")
    for row in launch["sources"]+evaluation["sources"]:
        reader.path(row)
    for required in (POLICY, RANKER, RETAINED_EVALUATOR, HERE / "evaluate_geometry_stage3_v2.py"):
        require(artifact(required) in evaluation["sources"], "Evaluation omitted actual retained source")
    runtime = Path(execution_path).resolve().parent / "runtime"
    require(execution["launch"]["path"] == str(runtime.parent / "launch.json")
            and launch["runtime_output"] == str(runtime), "Cross-runtime launch")
    files = execution["outputs"]
    for name in ("generated_motion.npz", "generation_metadata.json", "run_manifest.json", "energy_log.jsonl"):
        require(reader.path(files[name]) == runtime / name, "Output is not actual bound runtime file")
    for key, expected in (("source_motion", files["generated_motion.npz"]), ("source_metadata", files["generation_metadata.json"]),
                          ("source_energy_log", files["energy_log.jsonl"]), ("source_adapter", launch["adapter"]),
                          ("source_packet", launch["packet"])):
        require(evaluation[key] == expected, "Cross-motion/packet evaluation binding")
    adapter, packet = reader.json(launch["adapter"]), reader.json(launch["packet"])
    require(adapter["schema"] == "p555.geometry_stage3_adapter.v2" and packet["schema"] == "p555.current_geometry_stage3_packet.v2"
            and adapter["packet"] == launch["packet"] and adapter["static_seed"] == launch["neutral_seed"]
            and packet["source_artifacts"] == adapter["inputs"]
            and packet["surface_contract"] == adapter["surface_contract"], "Adapter/packet geometry mismatch")
    require(packet["scene_name"] == adapter["scene_id"] == evaluation["scene_id"]
            and packet["text"] == adapter["instruction"] == evaluation["instruction"]
            and adapter["query_id"] == adapter["episode_id"] == launch["guidance_binding"]["query_id"], "Current query drift")
    inputs = adapter["inputs"]
    for name, row in inputs.items():
        if name == "body_model":
            artifact_shape(row)  # Body identity is bound, not reloaded/rehashed here.
        else:
            reader.path(row)
    reader.path(launch["neutral_seed"])
    require(adapter["scene_fingerprint"] == digest({"mesh_sha256": inputs["original_scene_mesh"]["sha256"],
            "occupancy_sha256": inputs["raw_occupancy"]["sha256"]}), "Scene fingerprint differs")
    stage1, keynodes = reader.json(inputs["stage1_execution"]), reader.json(inputs["key_nodes"])
    require(stage1["scene_id"] == adapter["scene_id"] and stage1["instruction"] == adapter["instruction"]
            and stage1["target"] == inputs["stage1_target"] and stage1["bundle"] == inputs["stage1_bundle"]
            and keynodes["inputs"]["navmesh_route"] == inputs["navmesh_route"], "Current Stage1/query/route lineage differs")
    expected_query = digest({"scene_id": stage1["scene_id"], "scene_fingerprint": adapter["scene_fingerprint"],
        "fixed_geometry_sha256": stage1["source_fixed_geometry"]["sha256"], "instruction": stage1["instruction"],
        "start_world_xy_m": keynodes["start"]["source"]["root_xyz_yaw_world_zup"][:2]})
    require(expected_query == adapter["query_id"], "Current query identity formula mismatch")
    body = adapter["body_identity"]
    require(body["source_candidate"] == inputs["candidate"]
            and body["body_model_sha256"] == body["actual_model_sha256"] == inputs["body_model"]["sha256"]
            and body["betas"] == [0.]*10, "Actual candidate/body identity differs")
    skin = reader.json(evaluation["skinning"])
    require(skin["schema"] == "p555.actual_p519_motion_skinning.v1"
            and skin["source_motion"] == files["generated_motion.npz"]
            and skin["source_metadata"] == files["generation_metadata.json"]
            and skin["outputs"]["vertices"] == evaluation["skinned_vertices"]
            and skin["vertex_count"] == 10475 and skin["all_source_frames_preserved"] is True
            and skin["source_motion_edited"] is False
            and skin["body_sources"]["neutral_asset"]["sha256"] == inputs["body_model"]["sha256"],
            "Actual fullmesh skinning lineage differs")
    require(launch["guidance_binding"]["input"] == adapter["guidance_input"]
            and launch["guidance_proxy_map"] == adapter["guidance_proxy_map"], "Guidance input differs from adapter")
    for name in ("stage1_execution", "raw_occupancy"):
        require(launch["guidance_binding"]["source_bindings"][name] == inputs[name], "Current geometry/SDF binding drift")
    sdf = reader.json(inputs["sdf_cache"], sealed=False)
    require(sdf["inputs"]["occupancy"] == inputs["raw_occupancy"]
            and sdf["source_binding"]["stage1_execution"] == inputs["stage1_execution"], "Stage2 SDF/source mismatch")
    review = reader.json(inputs["qwen_review"])
    render = reader.json(review["source_render"])
    require(render["source_keypose"] == inputs["candidate"] and render["source_stage1"] == inputs["stage1_execution"]
            and render["original_mesh"] == inputs["original_scene_mesh"], "Original candidate/scene rendering differs")
    reader.path(render["original_camera"])
    metadata, manifest = reader.json(files["generation_metadata.json"], sealed=False), reader.json(files["run_manifest.json"], sealed=False)
    require(metadata["seed"] == launch["seed"] and metadata["raw_occupancy"]["source_path"] == inputs["raw_occupancy"]["path"],
            "Runtime seed/SDF differs")
    require(metadata["generated_frames_full"] == metadata["evaluation_generated_frames"] == evaluation["execution"]["frames"],
            "Evaluation frame coverage mismatch")
    require(skin["frame_count"] == evaluation["execution"]["frames"], "Skinning frame coverage mismatch")
    for role, row in (("plan", launch["packet"]), ("query_static_seed", launch["neutral_seed"]), ("raw_occupancy", inputs["raw_occupancy"])):
        require(manifest["input_provenance"]["artifacts"][role] == {"exists": True, "path": row["path"], "sha256": row["sha256"]},
                "Actual runtime manifest input mismatch")
    require(manifest["input_provenance"]["memory_or_retrieval_conditioned"] is False,
            "Unregistered learned-experience condition")
    pre, post = network_proof(reader, execution, launch, manifest)
    normalized = normalized_argv(launch)
    validate_gates(evaluation, gate_function)
    rows = read_energy(reader.path(files["energy_log.jsonl"]))
    energy, timeline = validate_energy(rows, metadata, execution, packet, mode)
    return dict(mode=mode, execution=execution, evaluation=evaluation, launch=launch, adapter=adapter,
                packet=packet, metadata=metadata, manifest=manifest, pre=pre, post=post,
                original_camera=render["original_camera"], normalized_argv=normalized,
                source_execution=source_execution, source_evaluation=source_evaluation,
                energy=energy, timeline=timeline)


def bind_pair(off, on):
    require(off["mode"] == "off" and on["mode"] == "on", "Comparison order must be OFF then ON")
    for key in ("adapter", "packet", "neutral_seed", "sources", "seed", "parent_current_physics",
                "guidance_binding", "guidance_proxy_map", "provider"):
        require(off["launch"][key] == on["launch"][key], "OFF/ON launch mismatch: " + key)
    # Reject any other launch difference, not only the currently known fields.
    ignored = {"affordance", "runtime_output", "actual_runner_argv", "receipt_payload_sha256"}
    require({k:v for k,v in off["launch"].items() if k not in ignored}
            == {k:v for k,v in on["launch"].items() if k not in ignored}, "Unexpected launch difference")
    require(off["normalized_argv"] == on["normalized_argv"], "Runner argv differs beyond exact output destination")
    require(off["pre"]["actual_file_artifacts"] == on["pre"]["actual_file_artifacts"], "Network/source/config input mismatch")
    require(off["evaluation"]["sources"] == on["evaluation"]["sources"], "Evaluation sources differ")
    require(off["original_camera"] == on["original_camera"], "Original source camera differs")
    require(off["metadata"]["raw_occupancy"] == on["metadata"]["raw_occupancy"]
            and off["evaluation"]["occupancy_contract"] == on["evaluation"]["occupancy_contract"], "Actual SDF policy differs")
    require(off["source_execution"]["path"] != on["source_execution"]["path"], "OFF/ON cannot be the same execution")
    return {"same_adapter_current_query": True, "same_random_and_neutral_seed": True,
            "same_actual_network_asset_set": True,
            "recorded_network_asset_count": len(off["pre"]["actual_file_artifacts"]),
            "pre_and_post_receipt_lineage_and_manifest_verified": True,
            "checkpoint_files_rehashed_by_this_report": False,
            "same_original_sdf_and_full22_physics_and_gate_policy": True,
            "same_actual_runner_argv_except_bound_output_dir": True,
            "affordance_switch_is_in_outer_launcher_not_inner_argv": True,
            "source_camera_bound_but_no_new_scene_render_performed": True,
            "same_uid_malicious_execution_authentication_claimed": False}


def summarize(off, on, checks):
    gate_names = off["evaluation"]["release_gates"]
    gates = []
    for name in gate_names:
        a, b = off["evaluation"]["release_gates"][name], on["evaluation"]["release_gates"][name]
        require(a.get("threshold") == b.get("threshold"), "OFF/ON release threshold mismatch")
        x, y = a.get("actual"), b.get("actual")
        numeric = type(x) in (int,float) and type(y) in (int,float)
        gates.append({"gate": name, "threshold": a.get("threshold"), "off": a, "on": b,
                      "on_minus_off": y-x if numeric else None,
                      "relative_change_percent": (y-x)/abs(x)*100 if numeric and x != 0 else None})
    cases = {}
    for case in (off, on):
        e, v = case["execution"], case["evaluation"]
        cases[case["mode"]] = {"execution": case["source_execution"], "evaluation": case["source_evaluation"],
            "network_preflight": e["network_preflight"], "network_postflight": e["network_postflight"],
            "energy_log": e["outputs"]["energy_log.jsonl"], "actual_frames": v["execution"]["frames"],
            "motion_publishable": v["motion_publishable"], "failed_release_gates": v["failed_release_gates"],
            "failed_gate_count": len(v["failed_release_gates"]), "release_gate_count": len(v["release_gates"]),
            "positive_credit": 0, "energy": case["energy"],
            "timings": {"execution_receipt_wall_seconds": e["elapsed_seconds"],
                        "network_preflight_seconds_included": case["pre"]["seconds"],
                        "network_postflight_seconds_included": case["post"]["seconds"],
                        "evaluation_wall_seconds_separate": v["timings"]["total_seconds"],
                        "scope": "execution timer includes source/adapter checks, model setup, sampling and postflight; not sampling-only; not Stage1/2; queue not measured here"}}
    return {"schema": "p555.actual_stage3_guidance_comparison.v1", "scene_id": off["adapter"]["scene_id"],
        "instruction": off["adapter"]["instruction"], "query_id": off["adapter"]["query_id"],
        "seed": off["launch"]["seed"], "adapter": off["launch"]["adapter"], "packet": off["launch"]["packet"],
        "same_source_checks": checks, "cases": cases, "release_gate_comparison": gates,
        "actual_energy_timeline": {"off": off["timeline"], "on": on["timeline"]},
        "timeline_axis": "actual denoise-call order, not emitted motion time; each primitive may have multiple denoise calls",
        "current_geometry_prior_not_learned_memory": True, "real_experience_count": 0, "positive_credit": 0,
        "motion_body_root_scene_or_existing_receipts_modified": False,
        "gpu_or_model_used_by_report": False, "this_report_reruns_no_fullmesh_metrics": True,
        "interpretation": "One same-seed paired observation, not a statistical benefit claim. Activated proxy contact guidance is not successful physical contact or motion publication."}


def plot_report(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    palette = {"off": "#4775a2", "on": "#e59138"}
    by_gate = {row["gate"]: row for row in summary["release_gate_comparison"]}
    panels = [
        ("Terminal full22 error", "m", ["terminal.full22_tail_error_m.mean", "terminal.full22_tail_error_m.p95", "terminal.full22_terminal_error_m.maximum"], ["Tail mean", "Tail p95", "Terminal max"], 1),
        ("Navigation root-to-route distance", "m", ["route.root_to_route_distance_m.p95", "route.root_to_route_distance_m.maximum"], ["p95", "Maximum"], 1),
        ("Contact-foot slip", "m / s", ["motion_quality.contact_foot_slip_m_per_s.mean", "motion_quality.contact_foot_slip_m_per_s.p95", "motion_quality.contact_foot_slip_m_per_s.maximum"], ["Mean", "p95", "Maximum"], 1),
        ("Official-compatible jerk", "m / frame^3", ["motion_quality.official_remogen_compatible.jerk_sequence_max_m_per_frame3"], ["Sequence maximum"], 1),
        ("Structural penetration depth", "m", ["scene_collision.decomposition.structural_nonfloor_collision.forbidden_penetration_depth_m.maximum", "fullmesh.forbidden_penetration_depth_m.maximum"], ["J22 maximum", "Fullmesh maximum"], 1),
        ("Structural collision fraction", "%", ["scene_collision.decomposition.structural_nonfloor_collision.forbidden_collision_fraction", "fullmesh.forbidden_collision_fraction"], ["J22", "Fullmesh"], 100),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (title, unit, names, labels, scale) in zip(axes.flat, panels):
        x = np.arange(len(names))
        maximum = 0.
        for mode, dx in (("off", -.19), ("on", .19)):
            values = [by_gate[key][mode]["actual"]*scale for key in names]
            maximum = max(maximum, *values)
            bars = ax.bar(x+dx, values, width=.34, color=palette[mode], label=mode.upper())
            ax.bar_label(bars, labels=[f"{value:.4g}" for value in values], padding=4, fontsize=9)
        thresholds = [by_gate[key]["threshold"]*scale for key in names]
        maximum = max(maximum, *thresholds)
        for i, threshold in enumerate(thresholds):
            ax.hlines(threshold, i-.44, i+.44, color="#b5454b", linestyle="--", linewidth=1.5,
                      label="Original limit" if i == 0 else None)
        ax.set(title=title, ylabel=unit, xticks=x, xticklabels=labels, ylim=(0, maximum*1.3 or 1))
        ax.grid(axis="y", alpha=.15)
        if ax is axes.flat[0]: ax.legend(frameon=False, fontsize=9)
    on_energy = summary["cases"]["on"]["energy"]
    status = " | ".join(f"{mode.upper()}: {case['failed_gate_count']}/34 gates failed" for mode,case in summary["cases"].items())
    fig.suptitle(f"Scene {summary['scene_id']} | same query / network / seed {summary['seed']} | Stage3 v2", fontsize=17, y=.98)
    fig.text(.5, .93, f"{status}   |   Positive credit = 0", ha="center", color="#a23239", fontsize=12)
    fig.text(.5, .035, f"ON proxy-contact activation: {on_energy['contact_active_rows']}/{on_energy['actual_denoise_rows']} denoise calls; "
             f"{on_energy['terminal_contact_active_rows']}/{on_energy['terminal_denoise_rows']} terminal calls. "
             "Activation is not a physical-contact pass.", ha="center", fontsize=11)
    fig.text(.5, .014, "One paired observation only. Small numeric changes may be visually indistinguishable; exact values and all 34 gates are in comparison.json.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.90))
    fig.savefig(output / "metric_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16,9))
    on_rows = summary["actual_energy_timeline"]["on"]
    x = np.arange(len(on_rows))
    axes[0,0].plot(x, [r["p555_contact_weighted_energy"] for r in on_rows], color=palette["on"], linewidth=1)
    axes[0,0].set(title="ON: actual weighted proxy-contact energy", ylabel="weighted energy (m²)")
    axes[0,1].plot(x, [r["p555_support_weighted_energy"] for r in on_rows], color="#358b83", linewidth=1)
    axes[0,1].set(title="ON: actual weighted support energy", ylabel="weighted energy (m²)")
    for key, label, color, style in (("p555_frame_safety_fraction", "Proxy frame safety mask", "#687481", "--"),
            ("p555_support_active_fraction", "Support active fraction", "#358b83", "-"),
            ("p555_contact_active_fraction", "Contact active fraction", palette["on"], "-")):
        axes[1,0].plot(x,[r[key] for r in on_rows],label=label,color=color,linestyle=style,linewidth=1.25)
    axes[1,0].set(title="ON: actual masks (not full-body safety proof)", ylabel="fraction", ylim=(-.05,1.09))
    axes[1,0].legend(frameon=False, fontsize=9, loc="lower right")
    for mode in ("off", "on"):
        rows = summary["actual_energy_timeline"][mode]
        xx = np.arange(len(rows))
        axes[1,1].plot(xx, [r["p533_physics_total_energy"] for r in rows], color=palette[mode],
                       label=mode.upper()+" retained physics", linewidth=1, alpha=.8)
        axes[1,1].plot(xx, [r["sdf_energy"] for r in rows], color=palette[mode],
                       label=mode.upper()+" retained SDF", linestyle="--", linewidth=1)
    axes[1,1].set(title="Actual retained parent energies", ylabel="energy (different weighted terms)")
    axes[1,1].set_yscale("symlog",linthresh=1e-4)
    axes[1,1].legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel("Actual denoise-call index (not motion frame / seconds)")
        ax.grid(alpha=.15)
        first = next(i for i,r in enumerate(on_rows) if r["terminal"])
        ax.axvline(first,color="#7e63a8",linestyle=":",linewidth=1)
    fig.suptitle(f"Scene {summary['scene_id']} | measured v2 guidance activity",fontsize=17,y=.98)
    fig.text(.5,.935,"Dotted line: first ON terminal denoise call. Repeated within-primitive changes are actual denoising, not a motion-time curve.",ha="center",fontsize=10)
    fig.text(.5,.025,"OFF has no p555 energy fields: intentionally not drawn as fabricated observed zeros. Safety masks are J22/proxy filters, not fullmesh publication gates.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.915))
    fig.savefig(output / "energy_timeline.png",dpi=150)
    plt.close(fig)


def markdown(summary):
    cases = summary["cases"]
    lines = [f"# Scene {summary['scene_id']}：Stage3 v2 OFF / ON 真实对照", "",
             "同一 adapter、当前 query、seed、network/source/config、原始 SDF 和原 34 个验收门；仅开启几何 affordance 附加引导。内部 argv 仅绑定的输出目录不同，affordance 开关在外层运行时包装。", "",
             "![关键指标](metric_comparison.png)", "", "![实际能量与 mask 激活](energy_timeline.png)", "",
             "## 结果与边界", ""]
    for mode,c in cases.items():
        lines.append(f"- {mode.upper()}：{c['actual_frames']} 帧；{c['failed_gate_count']}/34 门失败；motion_publishable={c['motion_publishable']}；positive_credit=0。")
    energy = cases["on"]["energy"]
    lines += [f"- ON `p555_contact_active_fraction>0`：{energy['contact_active_rows']}/{energy['actual_denoise_rows']} 次，"
              f"其中 terminal {energy['terminal_contact_active_rows']}/{energy['terminal_denoise_rows']} 次；"
              f"schedule>0 的激活行 {energy['contact_active_rows_with_nonzero_schedule']} 次。激活的是代理接触 mask，不是全身接触/动作成功。",
              "- OFF 没有 p555 能量字段；图中不将缺失值伪造为观测到的 0。横轴是实际去噪调用顺序，不是动作时间或动作帧。",
              "- 当前为冷启动几何先验，无 learned experience、无完整成功 credit。单一同 seed 对照不能证明普遍改进。",
              "- 这里只核对已绑定评价与原 gate 函数、读取实际日志；不重新评估全身碰撞、不加载模型、不渲染或修改动作/人体/场景。",
              "- 网络完整性使用本次实际 pre/post 全量校验收据与 runtime manifest；本报告不再次读取模型大权重，也不将 SHA 当作防同 UID 恶意伪造的认证。", "",
              "## 耗时口径", "", "| 模式 | execution 收据 wall (s) | 其中网络 pre / post (s) | 独立 evaluation wall (s) |", "|---|---:|---:|---:|"]
    for mode,c in cases.items():
        t=c["timings"]
        lines.append(f"| {mode.upper()} | {t['execution_receipt_wall_seconds']:.3f} | {t['network_preflight_seconds_included']:.3f} / {t['network_postflight_seconds_included']:.3f} | {t['evaluation_wall_seconds_separate']:.3f} |")
    lines += ["", "execution 包含输入/源码复核、模型启动、采样与 postflight，不是纯采样时间；不含 Stage1/2，不在此推断排队时间。评价单独列出，不重复相加网络 pre/post。", "",
              "## 全部原始 release gates", "", "| Gate | 原阈值 | OFF actual / pass | ON actual / pass | ON−OFF |", "|---|---:|---|---|---:|"]
    for row in summary["release_gate_comparison"]:
        fmt = lambda value: f"{value:.9g}" if type(value) in (int,float) else str(value)
        lines.append(f"| `{row['gate']}` | {fmt(row['threshold'])} | {fmt(row['off'].get('actual'))} / {row['off']['passed']} | "
                     f"{fmt(row['on'].get('actual'))} / {row['on']['passed']} | {fmt(row['on_minus_off'])} |")
    lines += ["", "## 原始证据", ""]
    for mode,c in cases.items():
        lines.append(f"- {mode.upper()} [execution]({c['execution']['path']}) / [evaluation]({c['evaluation']['path']}) / [实际 energy log]({c['energy_log']['path']})")
    lines += ["", "完整 artifact SHA、验证范围、实际每行能量与精确 delta 见 `comparison.json`；图与报告来源封存见 `receipt.json`。", ""]
    return "\n".join(lines)


def generate(off_execution, off_evaluation, on_execution, on_evaluation, output):
    started = time.monotonic()
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite any comparison output")
    reader = Reader()
    source = artifact(SOURCE)
    gate_function = retained_gate_function(reader)
    off = load_case(off_execution, off_evaluation, "off", reader, gate_function)
    on = load_case(on_execution, on_evaluation, "on", reader, gate_function)
    summary = summarize(off, on, bind_pair(off, on))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "visualize_stage3_guidance_comparison_source.py").open("xb") as stream:
        stream.write(SOURCE.read_bytes())
    plot_report(summary,output)
    write_once(output / "comparison.json",summary)
    with (output / "README.md").open("x",encoding="utf-8") as stream:
        stream.write(markdown(summary))
    reader.recheck()
    require(artifact(SOURCE) == source, "Report source changed while generating")
    result = {"schema": "p555.actual_stage3_guidance_comparison_report.v1", "source": source,
              "inputs": {"off_execution": off["source_execution"], "off_evaluation": off["source_evaluation"],
                         "on_execution": on["source_execution"], "on_evaluation": on["source_evaluation"]},
              "read_and_reverified_artifacts": list(reader.records.values()),
              "outputs": {name:artifact(output/name) for name in ("comparison.json", "metric_comparison.png", "energy_timeline.png", "README.md", "visualize_stage3_guidance_comparison_source.py")},
              "report_elapsed_seconds": time.monotonic()-started, "positive_credit": 0,
              "network_weights_rehashed": False, "gpu_or_model_used": False, "old_outputs_modified": False}
    write_once(output / "receipt.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for mode in ("off","on"):
        for kind in ("execution","evaluation"):
            parser.add_argument(f"--{mode}-{kind}",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=generate(args.off_execution,args.off_evaluation,args.on_execution,args.on_evaluation,args.output)
    print(json.dumps({"output":str(args.output.resolve()),"report_seconds":result["report_elapsed_seconds"],"positive_credit":0}))


if __name__ == "__main__":
    main()
