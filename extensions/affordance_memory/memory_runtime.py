"""Production M0 bridge: fixed facts -> cycle -> actual current retrieval.

This version has NO registered complete-motion producer. Cache reuse and a
current geometric prior are not learned experience. Observation receipts keep
the authority's partial identity separate from the full runtime query identity;
they never enter the positive-credit journal. No caller booleans enable credit.
The SDF callback below is a conservative two-point veto, NOT full-body physics.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import time

import numpy as np

from memory_common import (PROJECT, artifact, digest, read_json, read_sealed,
                           require, verified, write_once)
from memory_retrieval import (FRAME_CONVENTION, descriptor_to_shape,
                              load_current_surface, retrieve_current)
from memory_schema import shape_family
from memory_store import MemoryStore
from scene_facts import load_descriptors
from episode_evidence import admit_episode

HERE = Path(__file__).resolve().parent
RUNS = PROJECT / "agent9/runs"
BIND_ROUTE = PROJECT / "agent9/methods/p550_scene_understanding_decoupled_20260905/code/bind_route.py"
BODY = PROJECT / "agent10/hybrikx_three_methods_20260903/HybrIK/model_files/smplx/SMPLX_NEUTRAL.npz"
BODY_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
DEPENDENCIES = ("memory_runtime.py", "memory_common.py", "memory_retrieval.py",
                "memory_schema.py", "memory_store.py", "scene_facts.py",
                "surface_geometry.py", "episode_evidence.py")


def _sources():
    return [artifact(HERE / name) for name in DEPENDENCIES] + [artifact(BIND_ROUTE)]


def _production_source(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(RUNS), "Production evidence must be inside agent9/runs")
    require(not any("fixture" in part.lower() or part.lower() == "tests" for part in path.parts),
            "Fixture/test namespace is not production evidence")
    value = read_sealed(path)
    require(value.get("purpose", "production") == "production"
            and value.get("namespace", "production") != "test_fixture", "Fixture evidence rejected")
    return artifact(path), value


def _production_store(root):
    store = MemoryStore(root, purpose="production")
    state = store.recover()
    require(state["positive_episode_count"] == 0 and state["real_positive_credit"] == 0,
            "This runtime has no registered positive producer; nonzero production credit rejected")
    return store, state


def _retained_bind(stage1):
    # Execute the exact two retained function bodies in a private namespace.
    # This neither imports the old planner/GPU dependency tree nor mutates its
    # globals. Strong current artifact readers replace only filesystem helpers.
    source = BIND_ROUTE.read_text()
    tree = ast.parse(source, filename=str(BIND_ROUTE))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"checked", "bind"}]
    require([n.name for n in nodes] == ["checked", "bind"], "Retained route API changed")
    env = dict(math=math, read_sealed=read_sealed, verified=verified, artifact=artifact,
               digest=digest, __file__=str(BIND_ROUTE))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BIND_ROUTE), "exec"), env)
    return env["bind"](stage1)


def _current(stage1_path, facts_path, body_model, sdf_path, stage, body_shape_bin):
    require(stage == "stage2_contact", "Only Stage2 typed sit has a current verified role")
    require(body_shape_bin == "neutral", "Only the current zero-beta neutral body is registered")
    verified(body_model)
    require(body_model["path"] == str(BODY) and body_model["sha256"] == BODY_SHA256,
            "Current body model identity mismatch")
    stage_binding, stage1 = _production_source(stage1_path)
    facts_binding, facts = _production_source(facts_path)
    sdf_binding, sdf = _production_source(sdf_path)
    require(stage1.get("schema") == "p550.stage1_query_execution.v1", "Unsupported actual Stage1 execution")
    bound = _retained_bind(stage1)
    target = read_sealed(verified(stage1["target"]))
    require(target["action_family"] == "sit", "Only actual sit actions are registered")
    require(stage1["instruction"] == target["instruction"] == bound["instruction"], "Actual instruction drift")
    require(stage1["source_fixed_geometry"] == facts["sources"]["fixed_geometry"], "Stale facts/fixed geometry")
    require(stage1["scene_id"] == facts["scene_id"] == bound["scene_id"], "Current scene mismatch")
    guard = read_sealed(verified(stage1["p552_contact_region_guard"]))
    require(guard["all_gates_passed"] is True and guard["quality_gates"]
            and all(x is True for x in guard["quality_gates"].values()), "Current route region gates failed")
    require(guard["source_target"] == stage1["target"]
            and guard["target_instance_id"] == bound["target_instance_id"]
            and guard["direction_id"] == bound["direction_hypothesis_id"]
            and guard["selected_approach_option_id"] == bound["selected_approach_option_id"]
            and np.allclose(guard["snapped_goal_world_xy_m"], bound["snapped_goal_world_xy_m"], atol=1e-8, rtol=0),
            "Current route/owner/direction region binding mismatch")
    verified(guard["source_region"])
    verified(guard["source_actual_navmesh_route"])
    triples = load_descriptors(facts_path)
    selected = [t for t in triples if t[0]["instance_id"] == bound["target_instance_id"]
                and t[1]["source_binding"]["direction"]["candidate_id"] == bound["direction_hypothesis_id"]]
    require(len(selected) == 1, "Current selected owner/direction has no unique typed descriptor")
    row, descriptor, _ = selected[0]
    surface = next(x for x in target["candidate_surfaces"] if x["candidate_id"] == bound["selected_surface_id"])
    owner = surface["furniture_instance_binding"]
    require(owner["verified"] is True and owner["surface_inside_target_furniture_verified"] is True
            and owner["graph_target_instance_id"] == row["instance_id"]
            and owner["resolved_target_class"] == row["category"], "Current semantic furniture owner mismatch")
    require(surface["surface"] == row["surface"], "Selected surface geometry differs from current facts")
    # Stable facts and selected target use different hash namespaces. Compare
    # the actual shared mesh face/vertex/support IDs, not those two SHA strings.
    with np.load(verified(surface["surface_face_archive"]), allow_pickle=False) as a, \
            np.load(verified(row["surface_arrays"]), allow_pickle=False) as b:
        for left, right in (("mesh_face_indices", "face_ids"), ("surface_vertex_ids", "vertex_ids"),
                            ("target_support_vertex_ids", "support_ids")):
            require(np.array_equal(a[left], b[right]), "Current surface mesh ownership drift: " + left)
        require(str(a["target_instance_id"].item()) == row["instance_id"], "Surface archive owner mismatch")
    chosen = descriptor["source_binding"]["direction"]
    expected_direction = [math.cos(bound["contact_forward_yaw_rad"]), math.sin(bound["contact_forward_yaw_rad"])]
    require(np.allclose(chosen["outward_world_xy"], expected_direction, atol=1e-8, rtol=0), "Direction evidence drift")
    shape = descriptor_to_shape(selected[0])
    invariant = dict(action="sit", target_semantic=row["category"], surface_role="interaction_contact",
                     effector="pelvis_glute", motion_phase="terminal_contact", body_model_sha256=body_model["sha256"],
                     body_shape_bin=body_shape_bin, shape_family=shape_family(shape))
    context = load_current_surface(facts_path, row["instance_id"], bound["direction_hypothesis_id"],
                                   stage=stage, invariant=invariant, body_model=body_model)
    require(sdf["schema"] == "p549.pose_independent_bound_sdf_cache.v1"
            and sdf["source_kind"] == "current_new_scene", "Unsupported current SDF cache")
    require(sdf["source_binding"]["stage1_execution"] == stage_binding
            and sdf["source_binding"]["target"] == stage1["target"], "SDF actual Stage1/target drift")
    require(sdf["inputs"]["occupancy"] == facts["sources"]["source_occupancy"]
            and sdf["mask_receipt"]["target_component_sha256"] == owner["target_occupancy_mask_world_xyz_sha256"],
            "Current SDF scene/owner mismatch")
    expected_mask = {k: target["artifacts"]["target_occupancy_mask"][k] for k in ("path", "bytes", "sha256")}
    require(sdf["inputs"]["target_mask"] == expected_mask, "Current SDF target mask drift")
    require(sdf["mask_receipt"]["positive_is_free"] is True
            and sdf["mask_receipt"]["query_local_component_guess_used"] is False
            and sdf["pose_image_or_future_motion_used"] is False
            and sdf["target_mask_or_grid_resolution_changed"] is False, "SDF provenance/semantics mismatch")
    for rec in [*sdf["inputs"].values(), *sdf["arrays"].values(), sdf["kernel"], sdf["source"], sdf["source_binding"]["p550_source"]]:
        verified(rec)
    bundle = read_sealed(verified(stage1["bundle"]))
    require(guard["source_compile"] == bundle["artifacts"]["roadmap_compile"]
            and guard["source_actual_navmesh_route"] == bundle["artifacts"]["navmesh_route"],
            "Region guard differs from actual compiled/NavMesh route")
    request_binding = bundle["artifacts"]["roadmap_request"]
    request = read_json(verified(request_binding))  # P504 input is intentionally unsealed, artifact-bound.
    start = request["start_world_xy_m"]
    require(isinstance(start, list) and len(start) == 2
            and all(type(x) in (float, int) and math.isfinite(x) for x in start), "Invalid actual query start")
    actions = []
    for step in stage1["interaction_plan"]["steps"]:
        require(step["action"] == "sit" and isinstance(step["target_ids"], list) and step["target_ids"], "Unsupported ordered action plan")
        actions.append({k: step[k] for k in ("action", "target_ids", "reference_ids")})
    require(len(actions) == 1 and actions[0]["target_ids"] == [row["instance_id"]], "Current Stage2 is one selected sit interaction only")
    identity = dict(schema="p555.runtime_full_query_identity.v1", scene_id=context["scene_id"],
        scene_fingerprint_sha256=context["scene_fingerprint_sha256"], fixed_geometry_sha256=context["fixed_geometry"]["sha256"],
        instruction=stage1["instruction"], start_world_xy_m=[float(x) for x in start], ordered_actions=actions,
        body_model_sha256=body_model["sha256"], body_shape_bin=body_shape_bin, stage=stage)
    return dict(current_stage1_execution=stage_binding, facts=facts_binding, body_model=body_model,
                sdf_receipt=sdf_binding, sdf=sdf, context=context, identity=identity,
                route_binding=bound, route_guard=stage1["p552_contact_region_guard"], roadmap_request=request_binding)


def prepare(current_stage1_execution, facts_receipt, body_model, sdf_receipt, store_root, output,
            *, stage="stage2_contact", body_shape_bin="neutral"):
    """Create one immutable production query cycle; returns receipt.json Path."""
    started = time.monotonic()
    sources = _sources()
    require(isinstance(body_model, dict), "body_model must be an exact current artifact, not a model or flag")
    current = _current(current_stage1_execution, facts_receipt, body_model, sdf_receipt, stage, body_shape_bin)
    store, state = _production_store(store_root)
    identity = current["identity"]
    query_id = digest(identity)
    binding = dict(query_id=query_id, episode_identity=query_id,
        scene_fingerprint_sha256=identity["scene_fingerprint_sha256"], scene_id=identity["scene_id"],
        scene_family=identity["scene_id"].split("-", 1)[0], full_runtime_identity=identity)
    cycle = store.open_cycle(binding)
    require(_sources() == sources, "Runtime source changed during preparation")
    destination = Path(output) / "receipt.json"
    write_once(destination, dict(schema="p555.memory_runtime_preparation.v1", purpose="production",
        current=current, store_root=str(Path(store_root).resolve()), store_state=state, cycle=cycle,
        runtime_full_query_id=query_id, source_codes=sources, stage=stage, body_shape_bin=body_shape_bin,
        positive_credit=0, seed_and_output_excluded_from_identity=True,
        elapsed_seconds=time.monotonic() - started))
    return destination


def _validated_preparation(path):
    value = read_sealed(path)
    require(value["schema"] == "p555.memory_runtime_preparation.v1" and value["purpose"] == "production", "Invalid production preparation")
    require(value["source_codes"] == _sources(), "Runtime source closure drift")
    old = value["current"]
    current = _current(verified(old["current_stage1_execution"]), verified(old["facts"]), old["body_model"],
                       verified(old["sdf_receipt"]), value["stage"], value["body_shape_bin"])
    require(current == old and digest(current["identity"]) == value["runtime_full_query_id"], "Current runtime identity/binding drift")
    require(value["cycle"]["query_binding"]["full_runtime_identity"] == current["identity"]
            and value["cycle"]["query_binding"]["episode_identity"] == value["runtime_full_query_id"], "Cycle complete identity drift")
    store, _ = _production_store(value["store_root"])
    return value, store


class CurrentSDFPointFilter:
    """Conservative minimum of eight neighbouring current SDF voxel centres.

    No interpolation can conceal a negative corner. Outside the centre domain
    is rejected. This is ONLY a two-point filter and does not validate a body.
    """
    def __init__(self, current):
        self.current, self.calls = current, 0

    def __call__(self, projection):
        current, sdf = self.current, self.current["sdf"]
        require(projection["fixed_geometry"] == current["context"]["fixed_geometry"]
                and projection["owner"] == current["context"]["owner"], "SDF projection owner drift")
        points = np.asarray([projection["contact_world_xyz_m"], projection["root_world_xyz_m"]], dtype=float)
        require(points.shape == (2, 3) and np.isfinite(points).all(), "Invalid SDF points")
        grid = np.load(verified(sdf["arrays"]["collision_sdf"]), mmap_mode="r", allow_pickle=False)
        meta = sdf["mask_receipt"]
        require(list(grid.shape) == meta["grid_shape_world_xyz"] and grid.dtype == np.float32, "SDF grid identity/shape mismatch")
        lower, upper = np.asarray(meta["grid_bounds_world_zup_m"], dtype=float)
        spacing = np.asarray(meta["spacing_m"], dtype=float)
        require(np.allclose((upper - lower) / np.asarray(grid.shape), spacing, atol=1e-12, rtol=0), "SDF metric spacing mismatch")
        coords = (points - lower) / spacing - .5
        base = np.floor(coords).astype(int)
        outside = np.any((base < 0) | (base + 1 >= np.asarray(grid.shape)), axis=1)
        values = []
        for index, out in zip(base, outside):
            values.append(-1. if out else float(min(grid[tuple(index + [x, y, z])]
                for x in (0, 1) for y in (0, 1) for z in (0, 1))))
        require(all(math.isfinite(x) for x in values), "Nonfinite current SDF sample")
        self.calls += 1
        return dict(accepted=not outside.any() and all(x >= 0 for x in values), projection_sha256=digest(projection),
            fixed_geometry_sha256=projection["fixed_geometry"]["sha256"], owner_instance_id=projection["owner"]["instance_id"],
            non_target_sdf_m=values, outside=outside.tolist(), sdf_receipt=current["sdf_receipt"],
            sampling="conservative_min_eight_voxel_centres", full_body_validated=False)


def query(preparation_path, output, *, scopes=("scene_local", "invariant")):
    """Actual retrieve_current call at snapshot N, with live quarantine barrier."""
    started = time.monotonic()
    preparation_binding = artifact(preparation_path)
    prepared, store = _validated_preparation(preparation_path)
    callback = CurrentSDFPointFilter(prepared["current"])
    result = retrieve_current(store, prepared["cycle"], prepared["current"]["context"], sdf_check=callback, scopes=scopes)
    require(result["purpose"] == "production" and not result["records"] and result["m0_bypass"] is True,
            "Current unregistered producer cannot supply learned experiences")
    _production_store(prepared["store_root"])
    require(artifact(preparation_path) == preparation_binding and _sources() == prepared["source_codes"], "Lookup source/preparation drift")
    current = prepared["current"]
    destination = Path(output) / "receipt.json"
    write_once(destination, dict(schema="p555.memory_runtime_lookup.v1", purpose="production",
        preparation=preparation_binding, cycle=prepared["cycle"], runtime_full_query_id=prepared["runtime_full_query_id"],
        retrieval=result, experience_count=0, positive_credit=0, m0_bypass=True,
        provenance={
            "cache": dict(kind="verified_precomputed_scene_facts", artifact=current["facts"], learned_credit=0),
            "geometry_prior": dict(kind="current_typed_metric_surface", descriptor=current["context"]["descriptor"],
                fixed_geometry=current["context"]["fixed_geometry"], frame_convention=FRAME_CONVENTION,
                available=True, applied_to_keypose=False, learned_credit=0),
            "learned_experience": dict(kind="full_motion_success_only", count=0, reason="no_registered_complete_motion_producer")},
        current_sdf_filter=dict(source=current["sdf_receipt"], callback_count=callback.calls,
            point_filter_performed=callback.calls > 0, scope="contact_and_root_points_not_full_body", full_body_validated=False),
        full_body_refine_and_qwen_and_complete_motion_gates_still_required=True,
        elapsed_seconds=time.monotonic() - started))
    return destination


def record_observation(lookup_path, source_receipt, output):
    """Record actual admission only, never journal a second/partial credit ID.

    Stage1 observation is partial; a direct geometry proposal is unregistered.
    Both are useful diagnostics but neither may update furniture posteriors.
    """
    lookup_binding = artifact(lookup_path)
    lookup = read_sealed(lookup_path)
    require(lookup["schema"] == "p555.memory_runtime_lookup.v1" and lookup["purpose"] == "production"
            and lookup["experience_count"] == 0 and lookup["positive_credit"] == 0, "Invalid production M0 lookup")
    prepared, store = _validated_preparation(verified(lookup["preparation"]))
    require(lookup["cycle"] == prepared["cycle"] and lookup["runtime_full_query_id"] == prepared["runtime_full_query_id"], "Lookup/cycle identity mismatch")
    source_binding, source = _production_source(source_receipt)
    current = prepared["current"]
    if source["schema"] == "p550.stage1_query_execution.v1":
        require(source_binding == current["current_stage1_execution"], "Observation belongs to another Stage1")
    elif source["schema"] == "p555.geometry_guided_keypose_proposal.v1":
        require(source["inputs"]["stage1"] == current["current_stage1_execution"]
                and source["inputs"]["body_model"] == current["body_model"]
                and source["inputs"]["sdf_cache"] == current["sdf_receipt"], "Observation proposal Stage1/body/SDF mismatch")
    else:
        raise ValueError("Observation schema is not currently bound by this runtime")
    before = store.recover()
    request_path = Path(output) / "admission_request.json"
    write_once(request_path, dict(schema="p555.admission_request.v1", mode="observe_existing", source_receipt=source_binding))
    admission = admit_episode(request_path, purpose="production")
    require(admission["purpose"] == "production" and admission["real_positive_credit_allowed"] is False
            and admission["fixture_success_allowed"] is False and admission["critical_memory_failure"] is False
            and admission["affected_record_ids"] == [], "Observation unexpectedly requests credit/quarantine authority")
    after = store.recover()
    require(before == after and artifact(lookup_path) == lookup_binding, "Observation mutated the memory store or lookup")
    for binding in admission["source_bindings"]:
        verified(binding)
    require(_sources() == prepared["source_codes"], "Runtime source closure drift during observation")
    destination = Path(output) / "receipt.json"
    write_once(destination, dict(schema="p555.memory_runtime_observation.v1", purpose="production",
        lookup=lookup_binding, cycle=prepared["cycle"], source=source_binding, admission_request=artifact(request_path),
        runtime_full_query_id=prepared["runtime_full_query_id"],
        authority_observation_episode_id=admission["episode_id"], authority_admission=admission,
        identities_are_not_equated=True, positive_credit=0, positive_journal_modified=False,
        furniture_posterior_modified=False, quarantine_modified=False, store_before=before, store_after=after))
    return destination
