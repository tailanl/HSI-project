"""Hard terminal-region contract, checked before NavMesh and before publication."""

import numpy as np

from .region_scene import artifact, read_sealed, verified, write_once

from .region_geometry import clear_connectors

def check_points(points, mask, lower, resolution, name):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or len(points) == 0:
        raise ValueError(name + ": invalid XY coordinates")
    cells = np.floor((points - lower) / resolution).astype(np.int64)
    if not np.all((cells >= 0) & (cells < np.asarray(mask.shape))):
        raise ValueError(name + ": outside scene map")
    if not mask[tuple(cells.T)].all():
        raise ValueError(name + ": outside required region")
    return cells

def validate_endpoint(option, selected, region, arrays, route=None):
    member = option["p552_region_member"]
    canonical = next(r for r in region["approach_options"] if r["option_id"] == option["option_id"])
    if option != canonical:
        raise ValueError("Region candidate changed after geometry construction")
    if member["region_arrays"] != region["region_arrays"]:
        raise ValueError("Endpoint is bound to another region array")
    if option["p550_direction_hypothesis_id"] != member["direction_id"]:
        raise ValueError("Endpoint direction changed")
    requested = np.asarray(selected["requested_approach_world_xy_m"])
    goal = np.asarray(selected["snapped_goal_world_xy_m"])
    if not np.allclose(requested,option["approach_world_xy_m"],atol=1e-8,rtol=0):
        raise ValueError("Compiler changed requested goal")
    snap = float(np.linalg.norm(goal-requested))
    if not np.isfinite(snap) or snap > 1e-5:
        raise ValueError("P552 forbids off-candidate goal snapping")
    lower,res = arrays["grid_lower_world_xy_m"],float(arrays["voxel_size_m"])
    frontier = arrays[member["direction_id"] + "_terminal_frontier"]
    cells = check_points(np.array([requested,goal]),frontier,lower,res,"Terminal goal")
    if any(list(cell) != member["cell"] for cell in cells):
        raise ValueError("Compiler changed endpoint cell")
    anchor = np.asarray(member["contact_anchor_world_xyz_m"])
    direction = np.asarray(member["outward_world_xy"])
    gap = float((goal-anchor[:2]) @ direction)
    policy=region["audit"]["policy"]
    if not policy["minimum_front_gap_m"] - 1e-5 <= gap <= policy["maximum_front_gap_m"] + 1e-5:
        raise ValueError("Goal no longer adjacent to actual contact-front edge")
    if not clear_connectors(goal[None,:],anchor[None,:2],arrays["other_obstacles"],lower,res)[0]:
        raise ValueError("Another object separates endpoint from contact surface")
    gates = {"selected_candidate_unmodified":True,"goal_inside_direction_specific_frontier":True,
        "goal_inside_clipped_reachable_interaction_region":True,"no_off_region_snap":True,
        "no_intervening_non_target_obstacle":True,"original_human_clearance_preserved":True}
    if route is not None:
        if route["status"] != "verified_polygon_navmesh_route" or route["validation"]["all_segment_samples_human_free"] is not True:
            raise ValueError("Original NavMesh validation failed")
        points=np.asarray(route["route_world_xy_m"],dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points)<1 or not np.isfinite(points).all():
            raise ValueError("Invalid actual route")
        if np.linalg.norm(points[-1]-goal) > 1e-5:
            raise ValueError("Actual NavMesh route stopped before its interaction-region goal")
        check_points(points[-1:],frontier,lower,res,"Actual route endpoint")
        # Legacy local relaxation must not authorize cells below the original
        # 0.28m human clearance in this new endpoint protocol.
        samples=[points[:1]]
        for begin,end in zip(points[:-1],points[1:]):
            count=max(1,int(np.ceil(np.linalg.norm(end-begin)/(res*.25))))
            samples.append(begin+(end-begin)*np.linspace(0,1,count+1)[:,None])
        check_points(np.concatenate(samples),arrays["human_free"],lower,res,"Actual route human clearance")
        gates.update(actual_navmesh_endpoint_inside_region=True,actual_route_all_samples_fixed_human_free=True)
    return {"quality_gates":gates,"all_gates_passed":all(gates.values()),
        "snapped_goal_world_xy_m":goal.tolist(),"requested_goal_world_xy_m":requested.tolist(),
        "contact_anchor_world_xyz_m":anchor.tolist(),"contact_anchor_mesh_triangle_index":member["contact_triangle_local_index"],
        "front_gap_m":gap,"goal_snap_distance_m":snap,"direction_id":member["direction_id"],
        "stage3_sit_transition_validated":False,"standing_route_goal_is_not_seated_pelvis":True}

def validate_files(root, *, final=False):
    root = __import__("pathlib").Path(root)
    target_path=root / "planner/receipt.json"
    if not target_path.exists():target_path=root / "planner/stage1_target_receipt.json"
    target=read_sealed(target_path)
    compiled_path=root / "route/roadmap_compile_receipt.json"
    compiled=read_sealed(compiled_path)
    if compiled["inputs"]["p523_stage1_target"] != artifact(target_path):
        raise ValueError("Compiler consumed another target")
    surface=next(r for r in target["candidate_surfaces"] if r["candidate_id"]==target["selected_surface_id"])
    option=next(r for r in surface["approach_options"] if r["option_id"]==compiled["selected_approach_option_id"])
    selected=next(r for r in compiled["approach_audits"] if r["option_id"]==option["option_id"])
    region_binding=surface["surface_extraction_audit"]["front_inference"]["p552_contact_region"]
    region=read_sealed(verified(region_binding))
    if region["target_instance_id"]!=target["target_instance_id"] or region["scene_id"]!=target["scene_id"]:
        raise ValueError("Region/target scene identity drift")
    with np.load(verified(region["region_arrays"]),allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    route_path=root / "route/navmesh_route.json"
    # P508 route JSON is hash-bound by its receipt, but is not payload-sealed.
    import json
    route=json.loads(route_path.read_text()) if final else None
    result={"schema":"p552.contact_region_route_endpoint_guard.v1","phase":"final_navmesh" if final else "compiled_goal",
        "scene_id":target["scene_id"],"target_instance_id":target["target_instance_id"],
        "selected_approach_option_id":option["option_id"],"source_region":region_binding,
        "source_target":artifact(target_path),"source_compile":artifact(compiled_path),
        **validate_endpoint(option,selected,region,arrays,route),"source_code":artifact(__file__)}
    if final:result["source_actual_navmesh_route"]=artifact(route_path)
    path=root / "contact_regions" / ("final_route_guard.json" if final else "compiled_goal_guard.json")
    write_once(path,result,seal=True)
    return path
