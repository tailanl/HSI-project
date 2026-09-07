"""Current contact-region endpoints and fresh real polygon-NavMesh routes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from hsi.common.artifacts import artifact, read_sealed, verified, write_once
from . import _surface, _roadmap, _current_roadmap, _corridor, _keynodes, _bundle, frame0, visualization
from .binding import bind
from .context import SceneGeometryContext
from .planning import compatibility_inputs
from .region_guard import validate_files


@dataclass(frozen=True)
class Stage1Runtime:
    smplx_model_path: Path
    initial_yaw_rad: float = 0.0
    floor_clearance_m: float = 0.02
    stack_mb: int = 512


def check_stage1_handoff(stage1_path):
    """The original final target/region/route identity gate."""
    stage1 = read_sealed(stage1_path)
    bound = bind(stage1)
    guard = read_sealed(verified(stage1["p552_contact_region_guard"]))
    if guard["all_gates_passed"] is not True or not all(guard["quality_gates"].values()):
        raise ValueError("Region endpoint handoff failed")
    if guard["source_target"] != stage1["target"] or guard["target_instance_id"] != bound["target_instance_id"]:
        raise ValueError("Stage2 target/region handoff drift")
    if not np.allclose(bound["snapped_goal_world_xy_m"], guard["snapped_goal_world_xy_m"], atol=1e-8, rtol=0):
        raise ValueError("Stage2 received another route endpoint")
    verified(guard["source_region"])
    verified(guard["source_actual_navmesh_route"])
    return guard


def execute_route(semantics, geometry, step, instruction, start, output, semantic_plan_path, *, runtime):
    """Retained numerical sequence, now direct calls with explicit dependencies.

    Every call constructs a new PathFinder. No graph cache or historical route
    is accepted; persistent Stage1A navigation *fields* are not a NavMesh graph.
    """
    output = Path(output)
    records = []

    def phase(label, function, *args, **kwargs):
        begun = time.monotonic()
        try:
            value = function(*args, **kwargs)
        except Exception:
            records.append({"step": label, "elapsed_seconds": time.monotonic()-begun, "status": "failed"})
            raise
        records.append({"step": label, "elapsed_seconds": time.monotonic()-begun, "status": "complete"})
        return value

    sam_path, graph_path = compatibility_inputs(semantics, geometry, step, instruction, output,
                                                artifact(semantic_plan_path))
    planner = output / "planner"
    context = SceneGeometryContext(geometry, start, output)
    phase("contact_region_and_target", _surface.build, graph_path, sam_path,
          verified(geometry["source_mesh"]), verified(geometry["source_occupancy"]),
          start, planner, context=context)
    target = planner / "receipt.json"
    if not target.exists():
        target = planner / "stage1_target_receipt.json"
    compat = planner / "p509_compatible_stage1_target.json"
    target_value = read_sealed(target)
    selected = next(r for r in target_value["candidate_surfaces"] if r["candidate_id"] == target_value["selected_surface_id"])
    geo = next(r for r in geometry["objects"] if r["instance_id"] == target_value["target_instance_id"])
    gates = {"category_retained": target_value["target_class"] == geo["category"],
        "semantic_whole_object_verified": geo["semantic_query_usable"],
        "scene_surface_binding_verified": selected["surface_extraction_audit"].get("p550_cached_scene_geometry") is True,
        "surface_inside_owner": selected["furniture_instance_binding"]["surface_inside_target_furniture_verified"],
        "current_reachable_approach_exists": selected["route_prefilter"]["reachable_approach_count"] > 0,
        "no_semantic_geometry_override": target_value["semantic_override"]["enabled"] is False}
    quality = {"schema": "p550.decoupled_stage1_geometry_quality.v1",
        "status": "sealed_quality_pass" if all(gates.values()) else "failed_quality",
        "quality_gates": gates, "publish_gate_pass": all(gates.values()),
        "stage2_handoff_allowed": all(gates.values()),
        "inputs": {"planner_receipt": artifact(target), "fixed_semantics": geometry["source_semantics"],
            "fixed_navigation_fields": geometry["navigation_fields"]},
        "size_gate_policy": "diagnostic_not_class_veto",
        "direction_policy": "hypothesis_relative_and_body_facing_separate",
        "physical_keypose_not_yet_verified": True, "source_code": artifact(__file__)}
    quality_path = output / "quality.json"
    write_once(quality_path, quality, seal=True)
    if not all(gates.values()):
        raise ValueError("Fixed scene geometry quality failed")
    route = output / "route"
    query = output / "inputs/query.json"
    write_once(query, {"schema": "p123_causal_scene_text_query_v1", "scene_id": geometry["scene_id"],
        "instruction": instruction, "start_world_xz_m": list(start)})
    state = route / "inputs/current_frame0.npz"
    state_receipt = route / "inputs/current_frame0_receipt.json"
    phase("current_frame0", frame0.build, query, Path(runtime.smplx_model_path), state, state_receipt,
          initial_yaw_rad=runtime.initial_yaw_rad, floor_clearance_m=runtime.floor_clearance_m)
    request = route / "roadmap_request.json"
    compiled = route / "roadmap_compile_receipt.json"
    cfg = _roadmap.RoadmapConfig(navigation_clearance_mode="target_interaction_relaxed_stage3_sdf_collision")
    phase("roadmap_compile", _current_roadmap.compile_current_request, target, compat,
          verified(geometry["source_occupancy"]), state, state_receipt, request, compiled, cfg)
    phase("compiled_goal_guard", validate_files, output, final=False)
    actual_route = route / "navmesh_route.json"
    phase("navmesh_cold_build_and_route", _corridor.run, request, actual_route,
          route / "navmesh_route_receipt.json", stack_mb=runtime.stack_mb)
    guard_path = phase("final_route_guard", validate_files, output, final=True)
    keynodes = route / "key_nodes.json"
    phase("keynodes", _keynodes.run, compat, request, actual_route, state, state_receipt, keynodes)
    route_image = route / "navmesh_keynodes.png"
    route_visual_receipt = route / "navmesh_keynodes_visualization.json"
    phase("route_visualization", visualization.render, compat, request, actual_route,
          keynodes, route_image, route_visual_receipt)
    phase("stage1_bundle", _bundle.finalize, target, compat, graph_path, sam_path, request,
          actual_route, compiled, keynodes, output / "stage1_bundle")
    write_once(output / "numerical_execution.json", {
        "schema": "hsi.stage1.numeric_execution.v1", "steps": records,
        "navigation_graph_built_fresh": True, "navigation_graph_persisted": False,
        "historical_route_reused": False, "external_backend": "pynavmesh/pathfinder",
        "dynamic_source_patching": False, "source": artifact(__file__)}, seal=True)
    return {"status": "complete", "bundle": artifact(output / "stage1_bundle/receipt.json"),
        "target": artifact(target), "compatible_target": artifact(compat),
        "planner_receipt": artifact(quality_path), "graph": artifact(graph_path),
        "qwen": artifact(semantic_plan_path), "p552_contact_region_guard": artifact(guard_path),
        "route_visualization": artifact(route_visual_receipt), "route_image": artifact(route_image),
        "p552_route_source_code": artifact(__file__),
        "p552_endpoint_contract": "actual_contact_front_clipped_region_no_goal_snap"}
