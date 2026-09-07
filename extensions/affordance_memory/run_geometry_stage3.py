"""Strict new geometry-IK provider -> unchanged P478/ReMoGen network.

Run in the ReMoGen environment, for example with CUDA_VISIBLE_DEVICES=5. This
is a new P555 input admission boundary, never an H3/P530/P533 receipt converter.
Network/checkpoint defaults and numerical options come from P548's actual
generation_command function. Existing P523/P533 clean-xstart physics, full22
ICGF and SDF constraints stay on in BOTH affordance ablations. ``off`` installs
no added condition class and preserves exact parent energy/diagnostic objects.
No output here is publishable until the independent full-mesh evaluator passes.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import contextvars
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from memory_common import PROJECT, artifact, read_json, read_sealed, require, verified, write_once

HERE = Path(__file__).resolve().parent
P523 = PROJECT / "agent9/methods/p523_current_only_multiscene_20260902"
P533 = PROJECT / "agent9/methods/p533_h3_hybrikx_hybrid_stage2_20260903"
P548_COMMAND = PROJECT / "agent9/methods/p548_qwen38_full_pipeline_20260905/code/run_stage3.py"
P478_RUNNER = PROJECT / "agent6/runs/p478_skeleton_edit_e226_hsi_memory_20260814/code/run_p478_hsi.py"
PROVENANCE = "p555_current_geometry_ik"
PACKET_SCHEMA = "p555.current_geometry_stage3_packet.v1"
ADAPTER_SCHEMA = "p555.geometry_stage3_adapter.v1"
_ACTIVE_NODE = contextvars.ContextVar("p555_actual_stage3_node", default=None)


def load_module(name, path):
    path = Path(path).resolve(strict=True)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load_inputs(adapter_path):
    from compile_geometry_stage3 import load_adapter
    from neutral_seed import load_seed
    path = Path(adapter_path).resolve(strict=True)
    adapter, packet, seed_receipt = load_adapter(path)
    require(adapter["schema"] == ADAPTER_SCHEMA and packet["schema"] == PACKET_SCHEMA
            and packet["provenance"] == PROVENANCE, "Unregistered geometry Stage3 provider")
    packet_path = verified(adapter["packet"])
    seed_path = verified(adapter["static_seed"])
    seed_receipt_path = verified(adapter["static_seed_receipt"])
    seed = load_seed(seed_receipt_path)
    require(seed.gender == "neutral" and np.array_equal(seed.betas, np.zeros(10)),
            "Only actual neutral zero-shape fresh seed is admitted")
    require(Path(seed.path).resolve() == seed_path, "Neutral loader opened another seed")
    return {"adapter_path": path, "adapter": adapter, "packet": packet, "packet_path": packet_path,
            "seed_path": seed_path, "seed_receipt": seed_receipt, "seed_receipt_path": seed_receipt_path,
            "seed": seed, "occupancy": verified(adapter["inputs"]["raw_occupancy"])}


def retained_generation_command(stage3, occupancy, seed):
    """Compile only the real P548 pure argv builder; avoid unrelated common imports."""
    tree = ast.parse(P548_COMMAND.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generation_command"]
    require(len(definitions) == 1, "Missing exact P548 generation command")
    environment = {"PROJECT": PROJECT, "P533": P533}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(P548_COMMAND), "exec"), environment)
    return [str(value) for value in environment["generation_command"](Path(stage3), Path(occupancy), seed)]


def _replace_value(argv, flag, value):
    require(argv.count(flag) == 1, "Expected exactly one numerical flag: " + flag)
    argv[argv.index(flag) + 1] = str(value)


def generation_arguments(inputs, output, seed, physics_module, *, dry_run=False):
    require(type(seed) is int and 0 <= seed < 2**32, "Invalid generation seed")
    argv = retained_generation_command(Path(output), inputs["occupancy"], seed)[2:]
    physics, argv = physics_module._parse_physics_options(argv)
    for name in ("--p533-seed-receipt", "--p533-adapter-receipt", "--p533-preflight-receipt", "--p533-postprocess-ablation"):
        _, argv = physics_module._pop_option(argv, name)
    _replace_value(argv, "--plan", inputs["packet_path"])
    _replace_value(argv, "--query-static-seed", inputs["seed_path"])
    _replace_value(argv, "--output-dir", output)
    require(not any(value.startswith("--p533-") for value in argv), "Unhandled old-provider option")
    require("--allow-unknown-provenance" not in argv and "--allow-oracle-plan" not in argv,
            "Unknown/oracle provenance bypass forbidden")
    if dry_run:
        argv.append("--dry-run")
    return argv, physics


@contextlib.contextmanager
def admitted_provider(runner, inputs):
    """Allow only this compiler-validated packet object and this fresh seed.

    Do not broaden a global provenance set. The new exact provider is accepted
    solely after the path/hash-bound original P142 parser validated its nodes.
    Existing unknown/oracle flags remain false. Restore every runtime hook.
    """
    p142 = runner.p360.p142
    old_load, old_enforce = p142.load_world_keypose_plan, p142.enforce_plan_provenance
    old_trusted, old_seed = runner._trusted_predicted_plan, runner.QueryStaticSeed
    packet_record = artifact(inputs["packet_path"])
    admitted = []
    def load(path):
        require(Path(path).resolve() == inputs["packet_path"] and artifact(path) == packet_record,
                "P478 attempted another or changed packet")
        plan = old_load(path)
        require(plan.provenance == PROVENANCE and plan.schema == PACKET_SCHEMA, "Parsed provider drift")
        admitted.append(plan)
        return plan
    def enforce(plan, *, allow_oracle=False, allow_unknown=False):
        require(allow_oracle is False and allow_unknown is False, "Provenance bypass forbidden")
        require(any(plan is value for value in admitted) and plan.provenance == PROVENANCE,
                "Plan was not admitted by the exact current P555 loader")
        verified(packet_record)
    def trusted(plan):
        if plan.provenance == PROVENANCE:
            enforce(plan)
            return True
        return old_trusted(plan)
    class CurrentNeutralSeed:
        @classmethod
        def load(cls, path):
            require(Path(path).resolve() == inputs["seed_path"], "P478 requested another static seed")
            verified(inputs["adapter"]["static_seed"])
            verified(inputs["adapter"]["static_seed_receipt"])
            return inputs["seed"]
    p142.load_world_keypose_plan, p142.enforce_plan_provenance = load, enforce
    runner._trusted_predicted_plan, runner.QueryStaticSeed = trusted, CurrentNeutralSeed
    try:
        yield
    finally:
        p142.load_world_keypose_plan, p142.enforce_plan_provenance = old_load, old_enforce
        runner._trusted_predicted_plan, runner.QueryStaticSeed = old_trusted, old_seed


@contextlib.contextmanager
def expose_affordance(runner, *, binding, proxy, physics_module, enabled):
    """Add current triangle/support energy after exactly one complete parent call."""
    require(type(enabled) is bool, "Affordance mode must be boolean")
    stats = {"enabled": enabled, "condition_instances": 0, "energy_calls": 0,
             "terminal_energy_calls": 0, "nonterminal_energy_calls": 0}
    if not enabled:
        yield stats
        return
    import torch
    from stage3_affordance_guidance import GeometryAffordanceEnergy, GuidanceConfig
    from current_stepwise_physics_guidance_v1 import inferred_stance_probability
    original_class = runner.ReMoGenE226CondFn
    original_factory = runner._RuntimeContext.condition_factory
    def factory(context, **kwargs):
        require(context.scheduler is not None, "Missing actual ordered scheduler")
        index = min(int(context.scheduler.next_node_index), len(context.plan.nodes)-1)
        require(index >= 0 and context.plan.provenance == PROVENANCE, "Wrong current scheduler/provider")
        token = _ACTIVE_NODE.set({"index": index, "ordinal": int(context.plan.nodes[index].ordinal),
                                  "terminal": index == len(context.plan.nodes)-1})
        try:
            return original_factory(context, **kwargs)
        finally:
            _ACTIVE_NODE.reset(token)
    class CurrentGeometryAffordance(original_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            node = _ACTIVE_NODE.get()
            require(node is not None, "Affordance constructed outside the current scheduler")
            self.p555_node = dict(node)
            self.p555_energy = GeometryAffordanceEnergy(self.scene_sdf, binding.patch,
                GuidanceConfig(enabled=True), binding=binding)
            stats["condition_instances"] += 1
        @property
        def e226_receipt(self):
            result = dict(super().e226_receipt)
            result["p555_geometry_affordance"] = {"node": self.p555_node, **self.p555_energy.receipt(),
                "proxy_scope": "body-bound J22 contact proxies, NOT a full-mesh collision proof"}
            return result
        def energy_from_world_joints(self, joints):
            parent_total, parent_diagnostics = super().energy_from_world_joints(joints)
            future = joints[:, :self.valid_future_frames]
            require(future.shape[1] > 0, "No valid current frames")
            points = proxy.points(future, body_identity_sha256=binding.body_identity_sha256)
            history = self.e226_bridge.history_world_joints_zup.to(future).detach()
            if history.ndim == 3:
                history = history.unsqueeze(0)
            require(history.ndim == 4 and history.shape[1] > 0, "Missing detached causal stance context")
            history = history[:, -1:].expand(future.shape[0], -1, -1, -1)
            stance, _ = inferred_stance_probability(torch.cat((history, future), dim=1),
                ground_z_m=self.p555_energy.ground_z_m, config=self.p533_current_physics.config)
            activation = future.new_full(future.shape[:2], float(self.p555_node["terminal"]))
            added, diagnostics = self.p555_energy.energy(**points, contact_activation=activation,
                support_activation=stance, query_id=binding.value["query_id"],
                body_identity_sha256=binding.body_identity_sha256)
            diagnostics["p555_current_external_ordinal"] = future.new_tensor(float(self.p555_node["ordinal"]))
            diagnostics["p555_current_node_is_terminal"] = future.new_tensor(float(self.p555_node["terminal"]))
            stats["energy_calls"] += 1
            stats["terminal_energy_calls" if self.p555_node["terminal"] else "nonterminal_energy_calls"] += 1
            return self.p555_energy.combine_with_parent(parent_total, parent_diagnostics, added, diagnostics,
                original_sdf_weight=self.config.sdf_weight)
    runner.ReMoGenE226CondFn = CurrentGeometryAffordance
    runner._RuntimeContext.condition_factory = factory
    try:
        yield stats
    finally:
        runner.ReMoGenE226CondFn = original_class
        runner._RuntimeContext.condition_factory = original_factory


def runtime_proof(physics, output, *, dry_run):
    name = "dry_run_receipt.json" if dry_run else "generation_metadata.json"
    metadata = read_json(output / name)
    rows = None if dry_run else [json.loads(line) for line in (output / "energy_log.jsonl").read_text().splitlines() if line.strip()]
    proof = physics.validate_runtime_output(metadata, dry_run=dry_run, energy_log_records=rows,
        guidance_joint_ids=tuple(range(22)), semantic_contact_joint_ids=(0, 1, 2),
        sdf_exempt_joint_ids=(0, 1, 2), contact_frames=4)
    return proof, metadata, rows


def validate_affordance_runtime(metadata, rows, stats, packet, *, enabled):
    require(stats["enabled"] is enabled, "Affordance runtime mode mismatch")
    observed = [row for row in rows if "p555_affordance_energy" in row]
    if not enabled:
        require(not observed and stats["energy_calls"] == stats["condition_instances"] == 0,
                "Disabled affordance changed the parent runtime")
        return {"enabled": False, "exact_parent_path": True, "observed_denoise_rows": 0}
    require(observed and stats["energy_calls"] >= len(observed), "Missing or inconsistent affordance energy execution")
    terminal = int(packet["nodes"][-1]["ordinal"])
    valid_ordinals = {int(node["ordinal"]) for node in packet["nodes"]}
    for row in observed:
        ordinal = row.get("active_external_ordinal")
        require(type(ordinal) is int and ordinal in valid_ordinals
                and row.get("p555_current_external_ordinal") == ordinal
                and row.get("p555_current_node_is_terminal") == float(ordinal == terminal),
                "Affordance energy does not match the actual scheduler node")
        require("sdf_energy" in row and "p533_combined_parent_and_physics_energy" in row,
                "Affordance log lacks complete parent physics/SDF")
        require(np.isfinite(row["p555_affordance_energy"]) and row["p555_affordance_energy"] >= 0,
                "Invalid added runtime energy")
        if ordinal != terminal:
            require(row["p555_contact_raw_energy_m2"] == 0 and row["p555_facing_raw_energy"] == 0,
                    "Terminal attraction leaked into navigation")
    primitive_proofs = [row.get("e226_receipt", {}).get("p555_geometry_affordance")
        for row in metadata.get("primitive_records", [])]
    require(any(isinstance(row, dict) and row.get("node", {}).get("terminal") is True for row in primitive_proofs),
            "No terminal condition creation receipt")
    return {"enabled": True, "exact_parent_path": False, "observed_denoise_rows": len(observed),
            "terminal_denoise_rows": sum(row["active_external_ordinal"] == terminal for row in observed),
            "original_parent_sdf_and_physics_retained": True, "actual_scheduler_phase_verified": True,
            "added_energy_scope": "current_geometry_J22_proxies_not_fullmesh_proof"}


def run(adapter_path, output, seed, *, affordance="on", dry_run=False):
    require(affordance in ("on", "off"), "Unknown affordance ablation")
    started = time.monotonic()
    inputs = load_inputs(adapter_path)
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite Stage3 generation")
    output.mkdir(parents=True)
    runtime = output / "runtime"
    physics = load_module("p555_retained_stage3_physics", P533 / "stage3/code/run_p533_p478_v1.py")
    numeric = load_module("p555_retained_stage3_fields", P523 / "stage3/code/run_current_only_p478_v1.py")
    runner = load_module("p555_actual_p478_geometry_runner", P478_RUNNER)
    argv, config = generation_arguments(inputs, runtime, seed, physics, dry_run=dry_run)
    from stage3_affordance_guidance import GuidanceBinding, J22ProxyMap
    binding = GuidanceBinding(verified(inputs["adapter"]["guidance_input"]))
    proxy_payload = dict(inputs["adapter"]["guidance_proxy_map"])
    for key in ("contact_ids", "contact_drop_m", "foot_drop_m"):
        proxy_payload[key] = tuple(proxy_payload[key])
    proxy = J22ProxyMap(**proxy_payload)
    sources = [artifact(path) for path in (Path(__file__), HERE / "stage3_affordance_guidance.py",
        HERE / "compile_geometry_stage3.py", HERE / "neutral_seed.py", HERE / "memory_common.py", P548_COMMAND,
        P478_RUNNER, Path(physics.__file__), Path(numeric.__file__),
        P533 / "stage3/code/p533_full22_icgf_runtime_v1.py",
        P523 / "stage3/code/current_stepwise_physics_guidance_v1.py")]
    launch = {"schema": "p555.geometry_stage3_launch.v1", "adapter": artifact(inputs["adapter_path"]),
        "packet": artifact(inputs["packet_path"]), "neutral_seed": artifact(inputs["seed_path"]),
        "sources": sources, "actual_runner_argv": argv, "runtime_output": str(runtime), "seed": seed,
        "affordance": affordance, "dry_run": dry_run, "provider": PROVENANCE,
        "old_provider_admission_used": False, "unknown_provenance_bypass": False,
        "parent_current_physics": config.receipt(), "full22_icgf_retained": True,
        "parent_raw_sdf_retained": True, "network_and_checkpoint_parameters_changed": False,
        "guidance_binding": binding.receipt(), "guidance_proxy_map": inputs["adapter"]["guidance_proxy_map"],
        "positive_credit": 0, "motion_publishable": False}
    write_once(output / "launch.json", launch)
    status, error, proof, stats, affordance_proof = None, None, None, None, None
    before_cwd = Path.cwd()
    try:
        with admitted_provider(runner, inputs), numeric.expose_decoupled_joint_fields(runner), \
                numeric.expose_adaptive_guidance_policy(runner), \
                physics.expose_p533_current_stepwise_physics_guidance(runner, config), \
                physics.expose_full22_sparse_icgf(runner, guidance_joint_ids=tuple(range(22)),
                    semantic_contact_joint_ids=(0, 1, 2), sdf_exempt_joint_ids=(0, 1, 2), contact_frames=4), \
                expose_affordance(runner, binding=binding, proxy=proxy, physics_module=physics, enabled=affordance == "on") as stats:
            status = int(runner.main(argv))
        proof, metadata, rows = runtime_proof(physics, runtime, dry_run=dry_run)
        if not dry_run:
            require((runtime / "generated_motion.npz").is_file(), "No actual generated motion")
            affordance_proof = validate_affordance_runtime(metadata, rows, stats, inputs["packet"], enabled=affordance == "on")
        binding.verify_sources()
        for record in [launch["adapter"], launch["packet"], launch["neutral_seed"], *sources]:
            verified(record)
        return status
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        os.chdir(before_cwd)
        artifacts = {name: artifact(runtime / name) for name in ("generated_motion.npz", "generation_metadata.json",
            "run_manifest.json", "energy_log.jsonl", "dry_run_receipt.json") if (runtime / name).is_file()}
        write_once(output / "receipt.json", {"schema": "p555.geometry_stage3_execution.v1",
            "status": "generated_pending_fullmesh_evaluation" if not dry_run and error is None
                else "dry_run_completed" if dry_run and error is None else "failed_not_publishable",
            "launch": artifact(output / "launch.json"), "runner_returncode": status, "error": error,
            "dry_run": dry_run, "outputs": artifacts, "full22_runtime_proof": proof,
            "affordance_runtime": stats, "affordance_runtime_proof": affordance_proof,
            "elapsed_seconds": time.monotonic()-started,
            "full_motion_evaluation_performed": False, "motion_publishable": False, "positive_credit": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--affordance", choices=("on", "off"), default="on")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args.adapter, args.output, args.seed, affordance=args.affordance, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
