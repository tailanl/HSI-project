"""Read-only evidence for current P555 v1 motions; never a Memory writer.

Process success and scheduler completion are not task success. All existing
motion gates are reproduced from saved current arrays. Additional terminal
mesh diagnostics retain Stage2 thresholds, but cannot replace the original
full-motion policy or authorize positive experience credit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import time
import zipfile

import numpy as np

from memory_common import PROJECT, MemoryContractError, artifact, digest, read_json, read_sealed, require, verified, write_once
from neutral_seed import module

SCHEMA = "p555.current_motion_evidence.v1"
HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "evaluate_geometry_stage3.py"
RUNNER = HERE / "run_geometry_stage3.py"
P523 = PROJECT / "agent9/methods/p523_current_only_multiscene_20260902"
KERNEL = P523 / "evaluation/code/evaluate_current_stage3_motion_v1.py"
RANKER = P523 / "evaluation/code/rank_and_select_current_stage3_v1.py"
POLICY = P523 / "evaluation/configs/current_physical_release_policy_v2.json"
REGISTERED_SOURCES = {
    str(EVALUATOR): "b3e49c5c0ae3751982b37f3d5b177fdbd104227de3eee9436ace69e9b37393c3",
    str(RUNNER): "e223f31ecf2265151bbb7bbbc2b6ad6a294d2852591bd6139903a442491a30d7",
    str(KERNEL): "9bd5e9c5a1ddbd5465b61fa950858f58fb4e98e8aab2ce4902823ee3daaf6af4",
    str(RANKER): "877f2eb5a5ff8c7e2cf6ded84b16825afb9f19bb6df13ad80a9a553cdc4c2201",
    str(POLICY): "47baa1fc9b5b0107e74bceee1d44eba7e908cabd5b0828b7efaa1faa180dc9d6",
}
MODEL_HASHES = {
    "adapter_checkpoint": "519de22df9d7f3e2451c80e3029bbf3cba6cd86c0d54ae972fcb2291f75b7a99",
    "base_checkpoint": "373fcc0682d095e67d083f5976b014c8562732f6e104f797273a656d5a10089e",
    "mvae_checkpoint": "e1c103cc9d5a4916adb3261bdf79e69d599a3aa4960b669b4969c1c786a58773",
}
MAX_ARTIFACT_BYTES = 512*1024*1024
MAX_UNCOMPRESSED_BYTES = 768*1024*1024


class Reader:
    def __init__(self):
        self.bindings = {}

    def check(self, record, within=None):
        require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}, "Malformed evidence artifact")
        require(type(record.get("bytes")) is int and 0 <= record["bytes"] <= MAX_ARTIFACT_BYTES,
                "Evidence artifact exceeds its size bound")
        require(Path(record["path"]).suffix not in (".pt", ".pth", ".ckpt", ".safetensors", ".bin"),
                "Checkpoint content is not hashed or loaded by this read-only motion parser")
        key = record["path"]
        if key not in self.bindings:
            path = verified(record)
            self.bindings[key] = dict(record)
        else:
            require(self.bindings[key] == record, "One path has conflicting source identities")
            path = Path(key)
        require(within is None or path.is_relative_to(Path(within).resolve()), "Cross-case output binding")
        return path

    def read(self, record, *, schema=None, sealed=True, within=None):
        path = self.check(record, within)
        require(record["bytes"] <= 16*1024*1024, "Oversized evidence JSON")
        result = read_sealed(path) if sealed else read_json(path)
        require(schema is None or result.get("schema") == schema, "Unsupported current evidence schema")
        return result

    def finish(self):
        for record in self.bindings.values():
            verified(record)


def _record(record):
    return {key: record[key] for key in ("path", "bytes", "sha256")}


def _same(actual, expected, label):
    def equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    require(equal(actual, expected), "Current motion binding mismatch: "+label)


def _true(value, label):
    require(value is True, label+" is not explicitly true")


def _false(value, label):
    require(value is False, label+" is not explicitly false")


def _integer(value, minimum, label):
    require(type(value) is int and value >= minimum, "Invalid integer "+label)
    return value


def load_arrays(path, *, mesh=False):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 12 and sum(x.file_size for x in archive.infolist()) <= MAX_UNCOMPRESSED_BYTES,
                "Array archive decompression bound exceeded")
    with np.load(path, allow_pickle=False) as archive:
        names = {"vertices_world", "faces"} if mesh else {"transl", "global_orient", "body_pose", "betas", "joints"}
        require(set(archive.files) == names and len(archive.files) == len(names),
                "Unexpected motion/mesh arrays (including forbidden GT fields)")
        arrays = {key: archive[key].copy() for key in names}
    main = arrays["vertices_world" if mesh else "joints"]
    require(main.ndim == 3 and 4 <= len(main) <= 1000, "Invalid bounded generated frame count")
    count = len(main)
    shapes = ({"vertices_world": (count, 10475, 3), "faces": (20908, 3)} if mesh else
              {"joints": (count, 22, 3), "transl": (count, 3), "global_orient": (count, 3, 3),
               "body_pose": (count, 21, 3, 3), "betas": (count, 10)})
    for key, shape in shapes.items():
        value = arrays[key]
        require(value.shape == shape and value.dtype.kind in "fi" and np.isfinite(value).all(), "Invalid actual array: "+key)
    if mesh:
        faces = arrays["faces"]
        require(faces.dtype.kind in "iu" and faces.min() >= 0 and faces.max() < 10475, "Invalid actual mesh topology")
    else:
        require(np.array_equal(arrays["betas"], np.zeros_like(arrays["betas"])), "Motion body shape differs from current neutral seed")
        for key in ("global_orient", "body_pose"):
            rotations = arrays[key]
            require(np.max(np.abs(np.swapaxes(rotations, -1, -2)@rotations-np.eye(3))) <= 2e-5
                    and np.max(np.abs(np.linalg.det(rotations)-1)) <= 2e-5,
                    "Generated rotations are not SO(3): "+key)
    return arrays


def validate_chain(metadata, manifest, rows, frame_count):
    """One fresh *causal primitive rollout*, not one global DDPM trajectory."""
    prohibited = ("lingo_motion_opened", "lingo_sequence_lookup", "runtime_opens_source_pickle", "posthoc_motion_edit_used",
                  "feedback_modifies_emitted_frames", "future_frame_2_plus_available_to_runtime", "external_plan_contains_future_gt",
                  "generator_reads_future_root_directly_from_dataset", "generator_reads_gt_duration", "gt_contact_or_icgf_used",
                  "oracle_keypose_upper_bound", "retrieval_used", "memory_used")
    for key in prohibited:
        _false(metadata.get(key), key)
    _true(metadata.get("generated_only_output"), "generated-only output")
    _true(metadata.get("query_static_initial_history"), "query static history")
    _same(metadata.get("generation_reads_motion_frames"), [], "no imported temporal frames")
    _same(metadata.get("runtime_dataset_get_seq_source"), "synthetic_static_only", "synthetic getter")
    initial = _integer(metadata.get("query_static_initial_history_frames"), 0, "initial history")
    _same(initial, 2, "two static history features")
    causal = metadata["p478_causal_prefix"]
    _false(causal["future_or_gt_read"], "causal future/GT access")
    _same(causal["initial_history_source"], "query_static_initial_history", "initial source")
    _same(causal["history_sources_allowed"], ["query_static_initial_history", "generated_prefix"], "history allowlist")
    _same(causal["native_initial_frames"], 0, "no dataset frames")
    _same(causal["query_static_initial_frames"], initial, "query history count")
    records = metadata["primitive_records"]
    require(isinstance(records, list) and len(records) <= 125, "Invalid bounded primitive records")
    _same([row["primitive_id"] for row in records], list(range(len(records))), "contiguous primitive IDs")
    _same(len(records), _integer(metadata["primitives_generated"], 1, "primitives generated"), "primitive record count")
    for index, row in enumerate(records):
        _integer(row["primitive_id"], 0, "primitive ID")
        _same(row["history_source"], "query_static_initial_history" if index == 0 else "generated_prefix", "causal current prefix")
        _same(row["ddpm_steps"], _integer(metadata["ddpm_steps_per_primitive"], 1, "DDPM steps"), "denoising count")
        _same(row["condition_fn_invocations"], row["ddpm_steps"], "actual condition calls")
        _same(row["energy_log_records"], row["ddpm_steps"], "actual denoise log coverage")
    require(isinstance(rows, list) and len(rows) <= 100000, "Invalid bounded denoise log")
    for row in rows:
        _integer(row["primitive_id"], 0, "denoise primitive ID")
        digest(row)  # Forbid NaN/Inf even in non-gate diagnostic log columns.
    grouped = Counter(row["primitive_id"] for row in rows)
    _same(dict(grouped), {i: row["ddpm_steps"] for i, row in enumerate(records)}, "denoise log per-primitive coverage")
    _same(len(rows), metadata["energy_log_records"], "denoise total count")
    _same(frame_count, initial+8*len(records), "H2 F8 exact frame coverage")
    _same(causal["generated_frames_observed"], frame_count-initial, "causal appended frame count")
    for key in ("generated_frames_full", "evaluation_generated_frames"):
        _same(metadata[key], frame_count, key)
    result = manifest["result"]
    _same(result["generated_frames"], frame_count, "manifest frame count")
    _same(result["primitives_generated"], len(records), "manifest primitive count")
    _false(result["gt_fields_written"], "no GT output fields")
    provenance = manifest["input_provenance"]
    for key in ("future_motion_conditioned", "gt_contact_conditioned", "gt_icgf_conditioned", "lingo_motion_opened",
                "lingo_sequence_lookup", "memory_or_retrieval_conditioned", "retrieval_conditioned", "plan_is_oracle"):
        _false(provenance[key], "manifest "+key)
    _same(provenance["source_frame_indices"], [], "no source temporal indices")
    _same(provenance["generation_motion_seed_frames"], [], "no temporal seed clip")
    runtime = metadata["query_static_runtime_audit"]
    _same(runtime, manifest["query_static_runtime_audit"], "same static runtime audit")
    _same(runtime["schema"], "p555.neutral_static_runtime_audit.v1", "typed neutral runtime")
    _true(runtime["all_physical_frames_identical"], "static physical history")
    _same(runtime["source_frame_indices"], [], "static source frame indices")
    _same(runtime["physical_frame_repeat_count"], 3, "physical history repetition")
    _false(runtime["future_or_gt_motion_used"], "static builder no future")
    _true(runtime["query_static_builder_forced_load_data_false"], "static load_data false")
    for key in ("joint_delta_max_abs_m", "transl_delta_max_abs_m", "global_orientation_delta_angle_max_rad",
                "global_orientation_delta_identity6_max_abs"):
        require(type(runtime[key]) in (int, float) and abs(runtime[key]) <= 1e-8, "Static history changed: "+key)
    for key, limit in (("smplx_carrier_j22_max_abs_m", 5e-5), ("world_j22_roundtrip_max_abs_m", 5e-5)):
        require(type(runtime[key]) in (int, float) and math.isfinite(runtime[key]) and 0 <= runtime[key] <= limit,
                "Static numerical roundtrip failed")
    return {"fresh_causal_primitive_rollout": True, "single_global_ddpm_trajectory_claimed": False,
            "primitive_count": len(records), "ddpm_steps_per_primitive": metadata["ddpm_steps_per_primitive"],
            "generated_frames": frame_count-initial, "static_initial_frames": initial, "total_saved_frames": frame_count,
            "historical_motion_read": False, "trajectory_reconstruction_or_pose_splicing_used": False}


def logged_models(metadata, manifest):
    result = {}
    for key, wanted in MODEL_HASHES.items():
        row = manifest["input_provenance"]["artifacts"][key]
        _true(row.get("exists"), "logged model exists")
        _same(row["path"], metadata[key], "metadata/manifest model path")
        _same(row["sha256"], wanted, "registered checkpoint digest")
        path = Path(row["path"])
        require(path.is_absolute() and path.is_file(), "Recorded model no longer exists")
        result[key] = {"path": str(path), "logged_sha256": wanted, "current_size_bytes_unhashed": path.stat().st_size,
                       "actual_runtime_reported_hash": True, "content_rehashed_by_this_reader": False}
    return result


def _check_state(passed, evidence, reason):
    return {"state": "unknown" if passed is None else "verified_pass" if passed else "verified_fail",
            "evidence": evidence, "reason": reason}


def decide(release_gates):
    """Mandatory gaps remain unknown even for a hypothetical 34/34 pass."""
    require(isinstance(release_gates, dict) and len(release_gates) == 34
            and all(isinstance(row, dict) and type(row.get("passed")) is bool for row in release_gates.values()),
            "Incomplete original 34 gates")
    failed = [name for name, row in release_gates.items() if row["passed"] is False]
    return {"classification": "failure" if failed else "unknown", "failure_domain": "stage3_motion_validation" if failed else None,
            "failed_release_gates": failed, "real_positive_credit_allowed": False,
            "critical_memory_failure": False, "affected_record_ids": [], "allowed_record_ids": [],
            "record_extraction_receipts": [], "furniture_experience_posterior_update_allowed": False,
            "reason": "Actual same-query full motion fails the unchanged release policy" if failed else
                      "Physical gates alone do not fill missing final semantics, network freeze and registered Memory extraction",
            "memory_admission_registered": False, "final_task_semantic_evidence_available": False}


def lower_envelope_anchor(vertices, ids, fraction):
    """Same measured gluteal envelope as the frozen Stage2 body kernel."""
    vertices, ids = np.asarray(vertices), np.asarray(ids)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all(), "Invalid actual vertices")
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(ids) >= 8
            and len(np.unique(ids)) == len(ids) and ids.min() >= 0 and ids.max() < len(vertices), "Invalid gluteal vertex IDs")
    require(type(fraction) is float and 0 < fraction <= 1, "Invalid lower-envelope policy")
    support = vertices[ids]
    count = max(8, math.ceil(len(ids)*fraction))
    return np.r_[support[:, :2].mean(0), np.sort(support[:, 2])[:count].mean()]


def tail_geometry_diagnostics(reader, adapter, motion, meshes):
    """Read saved last four meshes; do not regenerate, optimize, or move them.

    These are diagnostic remeasurements, not new/relaxed Stage3 release gates.
    In particular the 20-micrometre Stage2 anchor-construction lock is kept
    verbatim and labelled construction-specific, not promoted to a motion gate.
    """
    import torch
    proposal = reader.read(adapter["inputs"]["proposal"], schema="p555.geometry_guided_keypose_proposal.v1")
    numeric = reader.read(adapter["numeric_recheck"], schema="p555.exact_body_candidate_recheck.v1")
    require(proposal["source"]["sha256"] == "d029b04dfef12ddb6284969d059c29583f2274c22d64ae0556457ec0b32f4dab",
            "Unregistered terminal numerical producer")
    compiler_path = reader.check(adapter["implementation_sources"]["compiler"])
    require(compiler_path == HERE/"compile_geometry_stage3.py", "Unexpected registered v1 compiler path")
    compiler = module("p555_evidence_tail_compiler", compiler_path)
    fast = module("p555_evidence_tail_archived_numerics", reader.check(proposal["source"]))
    _same(proposal["thresholds"], compiler.THRESHOLDS, "original Stage2 diagnostic thresholds")
    _same(numeric["thresholds"], compiler.THRESHOLDS, "recheck diagnostic thresholds")
    for record in numeric["numerical_sources"]:
        reader.check(record)
    _same(proposal["inputs"]["stage1"], adapter["inputs"]["stage1_execution"], "tail same Stage1")
    current = fast.load_current(reader.check(adapter["inputs"]["stage1_execution"]))
    _same(current["stage1"]["target"], adapter["inputs"]["stage1_target"], "tail same target")
    target, surface = current["target"], current["surface"]
    _same(current["binding"]["selected_surface_id"], adapter["surface_contract"]["selected_surface_id"], "tail same actual surface")
    _same(current["binding"]["selected_approach_option_id"], adapter["surface_contract"]["selected_approach_option_id"], "tail same approach")
    region = current["option"]["p552_region_member"]["region_arrays"]
    reader.check(region)
    engine = fast.FastKeyposeEngine(reader.check(proposal["inputs"]["body_model"]), reader.check(proposal["inputs"]["segmentation"]))
    body = engine.body
    require(np.array_equal(meshes["faces"], engine.model.faces), "Saved mesh differs from the exact neutral body topology")
    config = body.target.RefineConfig(steps=1, bilateral_foot_support=True, target_allowed_tail_m=.03, ground_tail_m=.02,
        minimum_allowed_contact_vertices=16, minimum_foot_ground_vertices=6, maximum_target_forbidden_penetration_m=.02,
        maximum_non_target_penetration_m=.02, maximum_outside_fraction=.01).validate()
    cache = engine.cache.BoundSDFCache(reader.check(proposal["inputs"]["sdf_cache"]), engine.sdf)
    _same(cache.receipt["source_binding"]["stage1_execution"], adapter["inputs"]["stage1_execution"], "tail SDF Stage1")
    _same(cache.receipt["source_binding"]["target"], adapter["inputs"]["stage1_target"], "tail SDF target")
    for source in [*cache.receipt["arrays"].values(), *cache.receipt["inputs"].values(), cache.receipt["kernel"], cache.receipt["source"]]:
        reader.check(source)
    fields = cache(current["occupancy"], current["target_mask"], torch.device("cpu"),
        expected_occupancy_file_sha256=adapter["inputs"]["raw_occupancy"]["sha256"],
        expected_target_mask_file_sha256=adapter["inputs"]["target_occupancy_mask"]["sha256"],
        expected_target_world_sha256=target["target_occupancy_mask_world_xyz_sha256"])
    anatomy = body.target.build_anatomy_masks(engine.model, 10475, config)
    ids = np.asarray(adapter["surface_contract"]["fixed_generated_butt_vertex_indices"], dtype=np.int64)
    bounds = np.asarray(surface["surface"]["bounds_world_zup_m"])
    yaw = current["binding"]["contact_forward_yaw_rad"]
    desired_contact = np.asarray(proposal["contact_world_xyz_m"])
    tail = []
    with torch.no_grad():
        for frame in range(len(motion["joints"])-4, len(motion["joints"])):
            vertices, joints = meshes["vertices_world"][frame], motion["joints"][frame]
            branches = body.target.branch_terms(torch.from_numpy(vertices), fields, anatomy, config)
            metrics = {key: float(value.detach()) if isinstance(value, torch.Tensor) else int(value)
                       for key, value in branches.diagnostics.items()}
            support = vertices[ids]
            anchor = lower_envelope_anchor(vertices, ids, float(body.SUPPORT_ENVELOPE_FRACTION))
            lock_error = float(np.linalg.norm(anchor-desired_contact))
            metrics["hips_support_inside_surface_fraction"] = body.support_inside_surface_fraction(support, bounds)
            axis = joints[2, :2]-joints[1, :2]
            facing_error = body.angular_error(math.atan2(float(axis[0]), float(-axis[1])), yaw)
            metrics["pelvis_frame_facing_error_rad"] = facing_error
            gates = engine.contract.publication_gates(metrics, compiler.THRESHOLDS,
                facing_error_rad=facing_error, selected_surface_match=True, contact_lock_error_m=lock_error)
            _, membership = fast.triangle_heights(support[:, :2], current["triangles"])
            heights, contact_valid = fast.triangle_heights(anchor[None, :2], current["triangles"])
            metrics["hips_support_inside_actual_triangles_fraction"] = float(membership.mean())
            metrics["measured_gluteal_anchor_world_xyz_m"] = anchor.tolist()
            metrics["desired_stage2_contact_world_xyz_m"] = desired_contact.tolist()
            metrics["stage2_construction_anchor_lock_error_m"] = lock_error
            metrics["anchor_minus_selected_triangle_height_m"] = float(anchor[2]-heights[0]) if contact_valid[0] else None
            gates.update(current_stage1_target_mask_hash_bound=fields.component_receipt["target_component_sha256"] == target["target_occupancy_mask_world_xyz_sha256"],
                direct_target_relation_valid=surface.get("reference_relation") in (None, "none", "beside", "associated_with_reference_setup"),
                known_action_family_sit=target["action_family"] == "sit",
                contact_point_inside_bound_surface=bool(np.all(anchor >= bounds[0]-1e-6) and np.all(anchor <= bounds[1]+1e-6)),
                contact_point_on_true_support_mask=bool(contact_valid[0]),
                contact_anchor_clearance_bounded=bool(contact_valid[0] and 0 <= float(anchor[2])-heights[0]+1e-6 <= .04+1e-6),
                hips_support_inside_actual_triangles=bool(membership.mean() >= .8))
            posture = engine.quality.metrics(joints, yaw)
            gates.update(posture["neutral_sit_quality_gates"])
            metrics["neutral_sit_quality"] = posture
            require(set(gates) == set(compiler.NUMERIC_GATES) and all(type(value) is bool for value in gates.values()),
                    "Terminal diagnostic gate set changed")
            tail.append({"frame_index": frame, "metrics": metrics, "stage2_diagnostic_23_gates": gates,
                         "failed_diagnostic_gates": [key for key, value in gates.items() if not value]})
    return {"schema": "p555.actual_motion_tail_geometry_diagnostic.v1", "source_region_arrays": region,
            "actual_source_frame_indices": [row["frame_index"] for row in tail], "frames": tail,
            "thresholds": compiler.THRESHOLDS, "units": "metres_radians_world_zup",
            "body_asset_sha256": proposal["inputs"]["body_model"]["sha256"], "same_saved_fullmesh_used": True,
            "new_body_pose_materialized": False, "motion_modified": False, "tail_only_not_full_motion_acceptance": True,
            "stage2_construction_anchor_lock_is_not_a_stage3_release_gate": True,
            "ground_reference": "retained Stage2 nominal world z=0; no original-scene floor triangle inference",
            "current_physical_release_34_gates_unchanged": True, "adds_positive_release_or_memory_credit": False}


def _classify(execution_path, evaluation_path, reader):
    start = time.monotonic()
    reader.check(artifact(__file__))
    execution_record, evaluation_record = artifact(execution_path), artifact(evaluation_path)
    execution = reader.read(execution_record, schema="p555.geometry_stage3_execution.v1")
    evaluation = reader.read(evaluation_record, schema="p555.geometry_stage3_motion_evaluation.v1")
    _false(execution["dry_run"], "real model execution")
    require(execution["error"] is None and execution["status"] == "generated_pending_fullmesh_evaluation", "No complete model execution")
    _true(evaluation["evaluation_complete"], "complete evaluation")
    _true(evaluation["input_contract_valid"], "evaluation input contract")
    _same(evaluation["source_execution"], execution_record, "evaluation/execution")
    for flag in ("generated_motion_modified", "h3_provider_claimed", "thresholds_relaxed", "credit_registration_performed"):
        _false(evaluation[flag], flag)
    launch = reader.read(execution["launch"], schema="p555.geometry_stage3_launch.v1", within=Path(execution_path).resolve().parent)
    _false(launch["dry_run"], "launch dry run")
    _same(launch["provider"], "p555_current_geometry_ik", "new actual provider")
    _same(evaluation["actual_provider"], launch["provider"], "evaluation provider")
    _same(evaluation["affordance"], launch["affordance"], "affordance mode")
    for flag in ("old_provider_admission_used", "unknown_provenance_bypass", "network_and_checkpoint_parameters_changed"):
        _false(launch[flag], flag)
    for flag in ("full22_icgf_retained", "parent_raw_sdf_retained"):
        _true(launch[flag], flag)
    sources = {}
    for row in [*launch["sources"], *evaluation["sources"]]:
        if row["path"] in sources:
            _same(row, sources[row["path"]], "same source declared by execution and evaluation")
        sources[row["path"]] = row
    for path, checksum in REGISTERED_SOURCES.items():
        require(path in sources and sources[path]["sha256"] == checksum, "Unregistered/missing producer implementation: "+path)
    for source in sources.values():
        reader.check(source)
    adapter = reader.read(launch["adapter"], schema="p555.geometry_stage3_adapter.v1")
    packet = reader.read(launch["packet"], schema="p555.current_geometry_stage3_packet.v1")
    for name, expected in (("source_adapter", launch["adapter"]), ("source_packet", launch["packet"])):
        _same(evaluation[name], expected, name)
    _same(adapter["packet"], launch["packet"], "adapter packet")
    _same(packet["source_artifacts"], adapter["inputs"], "packet original inputs")
    _same(launch["neutral_seed"], adapter["static_seed"], "current static seed")
    for source in [*adapter["inputs"].values(), adapter["numeric_recheck"], adapter["guidance_input"], adapter["guidance_proxy_receipt"],
                   *adapter["implementation_sources"].values()]:
        reader.check(source)
    seed_receipt = reader.read(adapter["static_seed_receipt"], schema="p555.neutral_query_static_seed_receipt.v1")
    _same(seed_receipt["seed"], launch["neutral_seed"], "neutral seed bytes")
    with np.load(reader.check(launch["neutral_seed"]), allow_pickle=False) as archive:
        seed_joints = archive["joints_world_zup"].copy()
    stage1 = reader.read(adapter["inputs"]["stage1_execution"], schema="p550.stage1_query_execution.v1")
    keynodes = reader.read(adapter["inputs"]["key_nodes"])
    fingerprint = digest({"mesh_sha256": adapter["inputs"]["original_scene_mesh"]["sha256"],
                          "occupancy_sha256": adapter["inputs"]["raw_occupancy"]["sha256"]})
    query = {"scene_id": stage1["scene_id"], "scene_fingerprint": fingerprint,
             "fixed_geometry_sha256": stage1["source_fixed_geometry"]["sha256"], "instruction": stage1["instruction"],
             "start_world_xy_m": keynodes["start"]["source"]["root_xyz_yaw_world_zup"][:2]}
    query_id = digest(query)
    _same(adapter["query_id"], query_id, "query identity")
    _same(adapter["episode_id"], query_id, "episode identity")
    _same(packet["scene_name"], query["scene_id"], "packet scene")
    _same(packet["text"], query["instruction"], "packet instruction")
    _same(evaluation["scene_id"], query["scene_id"], "evaluation scene")
    _same(evaluation["instruction"], query["instruction"], "evaluation instruction")
    runtime = Path(execution_path).resolve().parent/"runtime"
    _same(Path(launch["runtime_output"]).resolve(), runtime, "exact fresh runtime directory")
    output_paths = {name: reader.check(execution["outputs"][name], runtime) for name in
                    ("generated_motion.npz", "generation_metadata.json", "energy_log.jsonl", "run_manifest.json")}
    for name, path in output_paths.items():
        _same(path, runtime/name, "runtime artifact filename")
    for field, name in (("source_motion", "generated_motion.npz"), ("source_metadata", "generation_metadata.json"), ("source_energy_log", "energy_log.jsonl")):
        _same(evaluation[field], execution["outputs"][name], "same full motion "+field)
    metadata = reader.read(execution["outputs"]["generation_metadata.json"], sealed=False)
    manifest = reader.read(execution["outputs"]["run_manifest.json"], schema="p478.hsi_m1_m5_run_manifest.v1", sealed=False)
    rows = [json.loads(line) for line in output_paths["energy_log.jsonl"].read_text().splitlines() if line.strip()]
    _same(metadata["scene_name"], query["scene_id"], "metadata scene")
    _same(metadata["text"], query["instruction"], "metadata task")
    _same(metadata["plan_provenance"], launch["provider"], "runtime provider")
    _same(metadata["seed"], _integer(launch["seed"], 0, "generation RNG seed"), "same RNG seed")
    for key, record in (("plan", launch["packet"]), ("query_static_seed", launch["neutral_seed"]), ("raw_occupancy", adapter["inputs"]["raw_occupancy"])):
        logged = manifest["input_provenance"]["artifacts"][key]
        _true(logged["exists"], "actual logged input "+key)
        _same((logged["path"], logged["sha256"]), (record["path"], record["sha256"]), "manifest input "+key)
    motion = load_arrays(output_paths["generated_motion.npz"])
    frames = len(motion["joints"])
    require(np.max(np.abs(motion["joints"][:2]-seed_joints[None])) <= 5e-5, "Output initial frames do not match current neutral static seed")
    chain = validate_chain(metadata, manifest, rows, frames)
    runtime_audit = metadata["query_static_runtime_audit"]
    _same(runtime_audit["p555_seed_receipt"], adapter["static_seed_receipt"], "same runtime static seed receipt")
    _same(runtime_audit["seed_sha256"], launch["neutral_seed"]["sha256"], "same runtime seed bytes")
    require(set(runtime_audit["runtime_body_array_hashes"]) == {"v_template", "J_regressor", "lbs_weights", "shapedirs", "posedirs"},
            "Incomplete actual runtime body identity")
    _same(runtime_audit["runtime_body_array_hashes"], seed_receipt["runtime_model_array_hashes"], "runtime/seed body fingerprints")
    skin = reader.read(evaluation["skinning"], schema="p555.actual_p519_motion_skinning.v1", within=Path(evaluation_path).resolve().parent)
    for name in ("source_motion", "source_metadata"):
        _same(skin[name], evaluation[name], "skinning same motion "+name)
    _same(skin["outputs"]["vertices"], evaluation["skinned_vertices"], "same skinned mesh")
    _true(skin["all_source_frames_preserved"], "full frame skinning")
    _false(skin["source_motion_edited"], "motion unchanged during skinning")
    _true(skin["renderer_original_numeric_function_used"], "original actual skinning kernel")
    _same(skin["vertex_count"], 10475, "full mesh vertex count")
    _same(skin["frame_count"], frames, "all source mesh frames")
    _same(skin["body_sources"]["neutral_asset"]["sha256"], adapter["inputs"]["body_model"]["sha256"], "skinning body model")
    reader.check(skin["source"])
    mesh_path = reader.check(evaluation["skinned_vertices"], Path(evaluation_path).resolve().parent)
    meshes = load_arrays(mesh_path, mesh=True)
    _same(len(meshes["vertices_world"]), frames, "all actual 10475-mesh frames")
    numeric = module("p555_current_motion_retained_evaluator", EVALUATOR)
    old = module("p555_current_motion_original_metrics", KERNEL)
    ranker = module("p555_current_motion_original_ranker", RANKER)
    old.validate_fresh_metadata(metadata, packet, Path(adapter["inputs"]["raw_occupancy"]["path"]))
    expected = numeric.numeric_evaluation(old, motion["joints"], packet, metadata,
        Path(adapter["inputs"]["raw_occupancy"]["path"]), mesh_path,
        np.asarray(adapter["surface_contract"]["fixed_generated_butt_vertex_indices"], dtype=np.int64))
    for key, measured in expected.items():
        require(digest(measured) == digest(evaluation[key]), "Original full-motion metric does not reproduce: "+key)
    _same(evaluation["runner_returncode"], execution["runner_returncode"], "reported process status")
    gates = numeric.release_gates(evaluation, read_json(POLICY), ranker)
    _same(gates, evaluation["release_gates"], "exact unchanged original 34 release gates")
    decision = decide(gates)
    _same(decision["failed_release_gates"], evaluation["failed_release_gates"], "actual release failures")
    _same(evaluation["motion_publishable"], not decision["failed_release_gates"], "motion release declaration")
    tail_diagnostic = tail_geometry_diagnostics(reader, adapter, motion, meshes)
    models = logged_models(metadata, manifest)
    reader.finish()
    scene_match = re.match(r"^(\d+)(?:[-_]|$)", query["scene_id"])
    family = "lingo:"+str(int(scene_match[1])).zfill(3) if scene_match else "scene:"+query["scene_id"]
    checks = {"same_query_motion_seed": _check_state(True, ["launch", "adapter", "packet", "metadata", "motion"], "All exact current bindings agree"),
              "all_frames_fullmesh": _check_state(True, ["motion", "skinning", "skinned_vertices"], "Actual finite [T,10475,3], every frame preserved"),
              "fresh_causal_primitive_chain": _check_state(True, ["metadata.primitive_records", "energy_log"], "Sequential H2/F8 generated-prefix rollout, not a claimed global DDPM chain"),
              "full_motion_original_release": _check_state(not decision["failed_release_gates"], ["original_34_release_gates"], "Recomputed from actual saved current full motion/mesh"),
              "task_completed": _check_state(False if expected["execution"]["rolewise_terminal_complete"] is not True else None,
                  ["execution.rolewise_terminal_complete"], "Process0/scheduler complete cannot establish final task semantics"),
              "trained_checkpoint_load_identity": _check_state(None, ["run_manifest.input_provenance.artifacts"], "Registered runtime-logged hashes; complete same-session pre/post checkpoint/config/network-source freeze absent"),
              "final_task_semantics": _check_state(None, [], "No registered actual terminal static multi-view task review in this v1 execution"),
              "dynamic_mesh_contact_support": _check_state(None, ["terminal_geometry_diagnostic"],
                  "Last four actual meshes additionally measured with unchanged Stage2 thresholds; tail-only measurements are not a full-motion contact/support admission"),
              "memory_producer_and_extractor_registered": _check_state(False, [], "Frozen episode_evidence/store have not registered this evidence producer")}
    return {"schema": SCHEMA, **decision, "execution": execution_record, "evaluation": evaluation_record,
            "query_identity": query, "query_id": query_id, "episode_id": query_id, "scene_id": query["scene_id"],
            "scene_fingerprint": fingerprint, "scene_family": family,
            "attempt_id": digest({"query_id": query_id, "source_execution_sha256": execution_record["sha256"],
                                  "source_motion_sha256": evaluation["source_motion"]["sha256"], "seed": launch["seed"]}),
            "seed": launch["seed"], "affordance": launch["affordance"], "checks": checks,
            "original_34_release_gates": gates, "fresh_chain": chain, "logged_checkpoint_bindings": models,
            "terminal_geometry_diagnostic": tail_diagnostic,
            "actual_array_coverage": {"frames": frames, "joints_per_frame": 22, "vertices_per_frame": 10475,
                                      "full_mesh_value_count": frames*10475*3, "all_arrays_finite": True},
            "source_bindings": list(reader.bindings.values()), "source": artifact(__file__),
            "motion_changed": False, "gpu_or_model_inference_performed": False, "checkpoint_weights_loaded": False,
            "smplx_body_asset_loaded_for_cpu_numerics": True,
            "no_memory_write_performed": True, "elapsed_seconds": time.monotonic()-start}


def classify_current_motion(execution, evaluation):
    """Only registered complete v1 pairs; malformed/missing evidence is unknown."""
    reader = Reader()
    try:
        result = _classify(Path(execution), Path(evaluation), reader)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, AssertionError,
            zipfile.BadZipFile, RuntimeError, ImportError) as error:
        result = {"schema": SCHEMA, "classification": "unknown", "failure_domain": "evidence_contract",
                  "reason": f"{type(error).__name__}: {error}", "real_positive_credit_allowed": False,
                  "critical_memory_failure": False, "affected_record_ids": [], "allowed_record_ids": [],
                  "record_extraction_receipts": [], "furniture_experience_posterior_update_allowed": False,
                  "memory_admission_registered": False, "checks": {}, "source_bindings": list(reader.bindings.values()),
                  "source": artifact(__file__), "motion_changed": False, "gpu_or_model_inference_performed": False,
                  "checkpoint_weights_loaded": False, "no_memory_write_performed": True}
    result["receipt_payload_sha256"] = digest(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = classify_current_motion(args.execution, args.evaluation)
    if args.output:
        write_once(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
