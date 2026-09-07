"""Current process-local provider, contact context and energy hooks."""

from __future__ import annotations

import contextlib
import contextvars
import time
from pathlib import Path
import numpy as np

from hsi.common.artifacts import artifact, read_sealed, require, verified

from .packet import PROVENANCE, PACKET_SCHEMA

_ACTIVE_NODE = contextvars.ContextVar("hsi_actual_stage3_node", default=None)

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
        from .guidance import build_bound_context
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
def expose_affordance(runner, *, binding, proxy, physics_module, enabled, proxy_receipt_path):
    """Add current triangle/support energy after exactly one complete parent call."""
    require(type(enabled) is bool, "Affordance mode must be boolean")
    stats = {"enabled": enabled, "condition_instances": 0, "energy_calls": 0,
             "terminal_energy_calls": 0, "nonterminal_energy_calls": 0}
    if not enabled:
        yield stats
        return
    import torch
    from .guidance import GeometryAffordanceEnergy, GuidanceConfig
    from .physics import inferred_stance_probability
    contexts = RolloutContactContexts(binding, proxy, proxy_receipt_path, stats)
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
            context = contexts.get(self.scene_sdf)
            self.p555_energy = GeometryAffordanceEnergy(self.scene_sdf, binding.patch,
                GuidanceConfig(enabled=True), binding=binding, contact_context=context)
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
        contexts.finish()

def _validate_parent_affordance_runtime(metadata, rows, stats, packet, *, enabled):
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

def validate_affordance_runtime(metadata, rows, stats, packet, *, enabled):
    from . import guidance
    result = _validate_parent_affordance_runtime(metadata, rows, stats, packet, enabled=enabled)
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
        require(context.get("schema") == "hsi.current_target_contact_context.v1"
                and context.get("bound_current_geometry") is True
                and context.get("parent_raw_sdf_target_still_included") is True
                and context.get("original_release_gates_modified") is False, "Unbound/relaxed contact context")
        require(context["source"] == artifact(guidance.__file__), "Wrong actual contact energy source")
        for row in [context["source"], context["retained_v1"], context["retained_raw_sdf"], *context["source_bindings"].values()]:
            verified(row)
    observed = [row.get("e226_receipt", {}).get("p555_geometry_affordance")
                for row in metadata.get("primitive_records", [])]
    actual = [row for row in observed if isinstance(row, dict)]
    require(actual, "No actual contact-aware primitive receipts")
    for row in actual:
        require(row.get("schema") == guidance.SCHEMA
                and row.get("contact_context") in contexts, "Primitive did not use a registered cached v2 context")
        policy = row["contact_discretization_policy"]
        require(policy["whole_body_or_whole_target_exemption"] is False
                and policy["parent_sdf_modified"] is False
                and policy["frame_raw_j22_guard_retained"] is True, "Original parent/body guard was relaxed")
    return {**result, "schema": "p555.contact_aware_runtime_proof.v2", "contact_contexts_built": len(contexts),
            "contact_context_cache_hits": cache["hits"], "all_observed_primitives_used_v2_context": True}
