"""Strict v2 provider + contact-aware extra energy; frozen v1 stays untouched.

The network, scheduler, original RawSDF, physics and full22 guidance are retained.
Per-rollout caching stores only exact bound static geometry, never poses/motion.
"""
import argparse
import contextvars
import inspect
from pathlib import Path
import sys
import time
import types

import run_geometry_stage3 as retained
import compile_geometry_stage3_v2 as compiler
from memory_common import artifact, read_json, read_sealed, require, verified, write_once

SOURCE_PATH = Path(__file__).resolve(strict=True)
RETAINED_PATH = Path(retained.__file__).resolve(strict=True)
HERE, PROJECT, P523, P533 = retained.HERE, retained.PROJECT, retained.P523, retained.P533
RETAINED_SHA256 = "e223f31ecf2265151bbb7bbbc2b6ad6a294d2852591bd6139903a442491a30d7"
ENERGY_SHA256 = "a8aaa132544a3cdeef5265299559921a42651aeb3a8a60d1b80ae2f009beb27b"
SCHEMA = "p555.geometry_stage3_execution.v2"
LAUNCH_SCHEMA = "p555.geometry_stage3_launch.v2"
PROVENANCE = compiler.PROVENANCE
PACKET_SCHEMA, ADAPTER_SCHEMA = compiler.PACKET_SCHEMA, compiler.SCHEMA
CHECKPOINT_HASHES = {
    "adapter_checkpoint": "519de22df9d7f3e2451c80e3029bbf3cba6cd86c0d54ae972fcb2291f75b7a99",
    "base_checkpoint": "373fcc0682d095e67d083f5976b014c8562732f6e104f797273a656d5a10089e",
    "mvae_checkpoint": "e1c103cc9d5a4916adb3261bdf79e69d599a3aa4960b669b4969c1c786a58773",
    "clip_vit_b32": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}


def network_input_paths(runner, argv):
    """Exact retained CLI/default inputs; never read dataset split motions."""
    environment = PROJECT / "agent4/projects/remogen_20260716/envs/remogen"
    require(Path(sys.prefix).resolve() == environment.resolve(), "V2 network must use its bound ReMoGen environment")
    def option(flag, default):
        require(argv.count(flag) <= 1, "Duplicate network input flag")
        value = argv[argv.index(flag)+1] if flag in argv else default
        return Path(value).resolve(strict=True)
    p360 = runner.p360
    paths = {"adapter_checkpoint": option("--adapter-checkpoint", runner.DEFAULT_P360_ADAPTER),
             "base_checkpoint": option("--base-checkpoint", p360.DEFAULT_BASE),
             "mvae_checkpoint": option("--mvae-checkpoint", p360.DEFAULT_MVAE),
             "primitive_config": option("--cfg-path", p360.DEFAULT_CFG),
             "clip_vit_b32": Path.home() / ".cache/clip/ViT-B-32.pt"}
    paths["base_args"] = paths["base_checkpoint"].parent / "args.yaml"
    paths["mvae_args"] = paths["mvae_checkpoint"].parent / "args.yaml"
    data = option("--data-dir", p360.DEFAULT_DATA)
    require("--split" not in argv or argv[argv.index("--split")+1] == "val", "Only retained val text cache policy registered")
    paths.update(normalization_statistics=data / "mean_std_h2_f8.pkl", text_embedding_cache=data / "val_text_embedding_dict.pkl")
    remogen = PROJECT / "agent4/projects/remogen_20260716/ReMoGen"
    source_roots = [Path(p360.__file__).parent, Path(p360.p142.__file__).parent,
                    *[remogen / name for name in ("model", "mld", "data_loaders", "utils", "diffusion")]]
    for root in source_roots:
        require(root.is_dir(), "Network source root absent: " + str(root))
        for path in sorted(root.rglob("*.py")):
            paths["source:"+str(path.relative_to(PROJECT))] = path
    clip = PROJECT / "agent4/projects/remogen_20260716/envs/remogen/lib/python3.8/site-packages/clip"
    for name in ("clip.py", "model.py", "simple_tokenizer.py", "__init__.py", "bpe_simple_vocab_16e6.txt.gz"):
        paths["clip_source:"+name] = clip / name
    source_roots.append(clip)
    require(all(path.name not in {"train.pkl", "val.pkl", "test.pkl"} for path in paths.values()), "Motion split cannot be a model input")
    return paths, source_roots


class NetworkFreeze:
    """Independent actual file hashes before and after sampling, no model load."""
    def __init__(self, runner, argv, output):
        started = time.monotonic()
        paths, self.source_roots = network_input_paths(runner, argv)
        self.records = {role: artifact(path) for role, path in paths.items()}
        for name, expected in CHECKPOINT_HASHES.items():
            require(self.records[name]["sha256"] == expected, "Unregistered actual model checkpoint: " + name)
        self.output, self.completed, self.postflight_record = Path(output), False, None
        path = self.output / "network_preflight.json"
        write_once(path, {"schema": "p555.actual_network_asset_preflight.v2", "source": artifact(SOURCE_PATH),
            "actual_file_artifacts": self.records, "actual_runner_argv": list(argv),
            "checkpoint_hash_policy": CHECKPOINT_HASHES, "source_roots": [str(p) for p in self.source_roots],
            "checkpoint_and_config_actual_hashes_read_before_runtime": True,
            "historical_or_future_motion_read": False, "dataset_statistics_and_text_embeddings_only": True,
            "model_loaded_by_preflight": False, "seconds": time.monotonic()-started,
            "positive_credit": 0, "this_is_file_integrity_not_same_uid_execution_authentication": True})
        self.preflight_record = artifact(path)

    def finish(self, runtime):
        started = time.monotonic()
        for record in self.records.values():
            verified(record)
        manifest = read_json(Path(runtime) / "run_manifest.json")
        actual = manifest["input_provenance"]["artifacts"]
        for role in ("adapter_checkpoint", "base_checkpoint", "mvae_checkpoint"):
            require(actual[role]["path"] == self.records[role]["path"]
                    and actual[role]["sha256"] == self.records[role]["sha256"]
                    and actual[role]["exists"] is True, "Actual runtime checkpoint differs from independent preflight")
        frozen_paths = {row["path"] for row in self.records.values()}
        loaded_sources = []
        for module in list(sys.modules.values()):
            location = getattr(module, "__file__", None)
            if not location:
                continue
            path = Path(location).resolve()
            if path.suffix == ".py" and any(root == path.parent or root in path.parents for root in self.source_roots):
                require(str(path) in frozen_paths, "Actual imported numerical source was not frozen before sampling")
                loaded_sources.append(str(path))
        path = self.output / "network_postflight.json"
        write_once(path, {"schema": "p555.actual_network_asset_postflight.v2", "source": artifact(SOURCE_PATH),
            "preflight": self.preflight_record, "all_actual_files_rehashed_unchanged": True,
            "runtime_manifest": artifact(Path(runtime) / "run_manifest.json"),
            "actual_checkpoint_paths_and_sha_match_independent_preflight": True,
            "actual_imported_network_sources": sorted(set(loaded_sources)),
            "seconds": time.monotonic()-started, "positive_credit": 0})
        self.postflight_record = artifact(path)
        self.completed = True


def validate_network_proof(execution, launch):
    require(execution.get("network_assets_verified_pre_and_post") is True, "Missing independent network pre/post verification")
    pre = read_sealed(verified(execution["network_preflight"]))
    post = read_sealed(verified(execution["network_postflight"]))
    require(pre.get("schema") == "p555.actual_network_asset_preflight.v2"
            and post.get("schema") == "p555.actual_network_asset_postflight.v2"
            and post.get("preflight") == execution["network_preflight"], "Network pre/post lineage differs")
    require(pre["source"] == post["source"] == artifact(SOURCE_PATH)
            and pre["actual_runner_argv"] == launch["actual_runner_argv"]
            and pre["checkpoint_hash_policy"] == CHECKPOINT_HASHES, "Network model/source/command policy differs")
    require(post["all_actual_files_rehashed_unchanged"] is True
            and post["actual_checkpoint_paths_and_sha_match_independent_preflight"] is True, "Network postflight did not pass")
    records = pre["actual_file_artifacts"]
    required = set(CHECKPOINT_HASHES) | {"primitive_config", "base_args", "mvae_args", "normalization_statistics", "text_embedding_cache"}
    require(required <= set(records), "Independent network freeze omitted required model/config/statistics inputs")
    require(any(role.startswith("source:") for role in records)
            and all("clip_source:"+name in records for name in ("clip.py", "model.py", "simple_tokenizer.py", "__init__.py", "bpe_simple_vocab_16e6.txt.gz")),
            "Independent network freeze omitted implementation/tokenizer sources")
    for name, expected in CHECKPOINT_HASHES.items():
        require(records[name]["sha256"] == expected, "Postflight model hash differs from registered model")
    for record in records.values():
        verified(record)
    require(post["runtime_manifest"] == execution["outputs"]["run_manifest.json"], "Network postflight evaluated another runtime")
    actual = read_json(verified(post["runtime_manifest"]))["input_provenance"]["artifacts"]
    for name in ("adapter_checkpoint", "base_checkpoint", "mvae_checkpoint"):
        require(actual[name]["path"] == records[name]["path"] and actual[name]["sha256"] == records[name]["sha256"]
                and actual[name]["exists"] is True, "Runtime model differs from preflight model")
    return pre, post


def require_sources():
    require(artifact(RETAINED_PATH)["sha256"] == RETAINED_SHA256, "Frozen v1 launcher drift")
    require(artifact(HERE / "stage3_affordance_guidance_v2.py")["sha256"] == ENERGY_SHA256,
            "Unregistered contact-aware energy revision")


class RolloutContactContexts:
    """Strong identity cache: one actual parent SDF + exact binding/proxy sources."""
    def __init__(self, binding, proxy, proxy_receipt_path, stats):
        self.binding, self.proxy = binding, proxy
        self.proxy_record = artifact(proxy_receipt_path)
        self.binding_record = dict(binding.input_artifact)
        verified(self.binding_record)
        self.rows, self.stats = {}, stats
        stats["static_contact_context_cache"] = {"schema": "p555.rollout_static_contact_context_cache.v2",
            "build_count": 0, "hits": 0, "build_seconds": 0., "lookup_seconds": 0.,
            "scope": "current_rollout_parent_sdf_identity_and_exact_geometry_artifacts",
            "pose_motion_or_energy_cached": False, "source_proxy": self.proxy_record,
            "source_guidance": self.binding_record, "contexts": []}

    def get(self, parent_sdf):
        from stage3_affordance_guidance_v2 import build_bound_context
        start = time.monotonic()
        stats = self.stats["static_contact_context_cache"]
        key = id(parent_sdf)
        if key in self.rows:
            context = self.rows[key]
            require(context.parent_sdf is parent_sdf and context.binding is self.binding,
                    "Cached contact context identity changed")
            stats["hits"] += 1
        else:
            context = build_bound_context(parent_sdf, self.binding, self.proxy_record["path"])
            require(context.parent_sdf is parent_sdf and context.binding is self.binding
                    and context.proxy_map == self.proxy, "Built contact context differs from actual body/SDF binding")
            self.rows[key] = context  # Strong reference prevents Python id reuse.
            stats["build_count"] += 1
            stats["build_seconds"] += time.monotonic()-start
            stats["contexts"].append(context.receipt())
        stats["lookup_seconds"] += time.monotonic()-start
        return context

    def finish(self):
        verified(self.proxy_record)
        verified(self.binding_record)
        self.binding.verify_sources()
        for context in self.rows.values():
            context.verify_sources()


def validate_affordance_runtime(metadata, rows, stats, packet, *, enabled):
    result = retained.validate_affordance_runtime(metadata, rows, stats, packet, enabled=enabled)
    cache = stats.get("static_contact_context_cache")
    if not enabled:
        require(cache is None, "Disabled path unexpectedly built extra contact geometry")
        return {**result, "schema": "p555.contact_aware_runtime_proof.v2", "contact_contexts_built": 0}
    require(isinstance(cache, dict) and cache.get("schema") == "p555.rollout_static_contact_context_cache.v2"
            and cache.get("pose_motion_or_energy_cached") is False, "Missing current-only static context cache proof")
    contexts = cache["contexts"]
    require(type(cache["build_count"]) is int and cache["build_count"] == len(contexts) > 0
            and type(cache["hits"]) is int and cache["hits"] >= 0
            and cache["build_count"]+cache["hits"] == stats["condition_instances"], "Contact cache count mismatch")
    for row in (cache["source_proxy"], cache["source_guidance"]):
        verified(row)
    for context in contexts:
        require(context.get("schema") == "p555.current_target_contact_discretization_context.v2"
                and context.get("bound_current_geometry") is True
                and context.get("parent_raw_sdf_target_still_included") is True
                and context.get("original_release_gates_modified") is False, "Unbound/relaxed contact context")
        require(context["source"]["sha256"] == ENERGY_SHA256, "Wrong actual contact energy source")
        for row in [context["source"], context["retained_v1"], context["retained_raw_sdf"], *context["source_bindings"].values()]:
            verified(row)
    observed = [row.get("e226_receipt", {}).get("p555_geometry_affordance")
                for row in metadata.get("primitive_records", [])]
    actual = [row for row in observed if isinstance(row, dict)]
    require(actual, "No actual contact-aware primitive receipts")
    for row in actual:
        require(row.get("schema") == "p555.contact_discretization_affordance_guidance.v2"
                and row.get("contact_context") in contexts, "Primitive did not use a registered cached v2 context")
        policy = row["contact_discretization_policy"]
        require(policy["whole_body_or_whole_target_exemption"] is False
                and policy["parent_sdf_modified"] is False
                and policy["frame_raw_j22_guard_retained"] is True, "Original parent/body guard was relaxed")
    return {**result, "schema": "p555.contact_aware_runtime_proof.v2", "contact_contexts_built": len(contexts),
            "contact_context_cache_hits": cache["hits"], "all_observed_primitives_used_v2_context": True}


def _replace(source, old, new):
    require(source.count(old) == 1, "Frozen v1 adaptation anchor drift: " + old[:80])
    return source.replace(old, new)


def private_launcher():
    require_sources()
    scope = dict(vars(retained))
    scope.update(__file__=str(SOURCE_PATH), PROVENANCE=PROVENANCE, PACKET_SCHEMA=PACKET_SCHEMA,
                 ADAPTER_SCHEMA=ADAPTER_SCHEMA, RolloutContactContexts=RolloutContactContexts,
                 NetworkFreeze=NetworkFreeze,
                 _ACTIVE_NODE=contextvars.ContextVar("p555_v2_actual_stage3_node", default=None))
    for name, value in vars(retained).items():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(retained):
            fn = types.FunctionType(value.__code__, scope, value.__name__, value.__defaults__, value.__closure__)
            fn.__kwdefaults__ = value.__kwdefaults__
            scope[name] = fn
    # Decorated contexts need their owned function reconstructed in private
    # globals too; cloning contextlib's wrapper would retain a v1 closure.
    exec(compile(inspect.getsource(retained.admitted_provider), retained.__file__, "exec"), scope)
    source = inspect.getsource(retained.load_inputs)
    source = _replace(source, "from compile_geometry_stage3 import", "from compile_geometry_stage3_v2 import")
    source = _replace(source, "from neutral_seed import", "from neutral_seed_v2 import")
    exec(compile(source, retained.__file__, "exec"), scope)
    source = inspect.getsource(retained.expose_affordance)
    source = _replace(source, "physics_module, enabled):", "physics_module, enabled, proxy_receipt_path):")
    source = _replace(source, "from stage3_affordance_guidance import GeometryAffordanceEnergy, GuidanceConfig",
        "from stage3_affordance_guidance import GuidanceConfig\n    from stage3_affordance_guidance_v2 import GeometryAffordanceEnergyV2")
    source = _replace(source, "original_class = runner.ReMoGenE226CondFn",
        "contexts = RolloutContactContexts(binding, proxy, proxy_receipt_path, stats)\n    original_class = runner.ReMoGenE226CondFn")
    source = _replace(source, "self.p555_energy = GeometryAffordanceEnergy(self.scene_sdf, binding.patch,\n                GuidanceConfig(enabled=True), binding=binding)",
        "context = contexts.get(self.scene_sdf)\n            self.p555_energy = GeometryAffordanceEnergyV2(self.scene_sdf, binding.patch,\n                GuidanceConfig(enabled=True), binding=binding, contact_context=context)")
    source = _replace(source, "runner._RuntimeContext.condition_factory = original_factory",
        "runner._RuntimeContext.condition_factory = original_factory\n        contexts.finish()")
    exec(compile(source, retained.__file__, "exec"), scope)
    source = inspect.getsource(retained.run)
    source = _replace(source, 'enabled=affordance == "on") as stats:',
        'enabled=affordance == "on", proxy_receipt_path=verified(inputs["adapter"]["guidance_proxy_receipt"])) as stats:')
    source = source.replace('"p555.geometry_stage3_launch.v1"', repr(LAUNCH_SCHEMA)).replace('"p555.geometry_stage3_execution.v1"', repr(SCHEMA))
    source = _replace(source, 'Path(__file__), HERE / "stage3_affordance_guidance.py",',
        'Path(__file__), HERE / "run_geometry_stage3.py", HERE / "stage3_affordance_guidance_v2.py", HERE / "stage3_affordance_guidance.py",')
    source = _replace(source, 'HERE / "compile_geometry_stage3.py"', 'HERE / "compile_geometry_stage3_v2.py"')
    source = _replace(source, 'HERE / "neutral_seed.py"', 'HERE / "neutral_seed_v2.py"')
    source = _replace(source, 'launch = {"schema":', 'network = NetworkFreeze(runner, argv, output)\n    launch = {"schema":')
    source = _replace(source, "binding.verify_sources()", "binding.verify_sources()\n        network.finish(runtime)")
    source = _replace(source, '"full_motion_evaluation_performed": False,',
        '"network_preflight": network.preflight_record, "network_postflight": network.postflight_record,\n            "network_assets_verified_pre_and_post": network.completed, "full_motion_evaluation_performed": False,')
    exec(compile(source, retained.__file__, "exec"), scope)
    scope["validate_affordance_runtime"] = validate_affordance_runtime
    return scope


def load_inputs(path):
    return private_launcher()["load_inputs"](path)


load_module = retained.load_module
runtime_proof = retained.runtime_proof


def run(adapter_path, output, seed, *, affordance="on", dry_run=False):
    return private_launcher()["run"](adapter_path, output, seed, affordance=affordance, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--affordance", choices=("on", "off"), default="on")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.adapter, args.output, args.seed, affordance=args.affordance, dry_run=args.dry_run))
