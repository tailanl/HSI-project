import copy
import numpy as np
import pytest
from test_stage1_regions import rectangle, run_region, LOWER, RES
from hsi.stage1.region_guard import validate_endpoint


def fixture():
    rows,masks,audit=run_region(rectangle())
    candidate=rows[0]
    member={**candidate,"region_arrays":{"sha256":"unit-test-bound-array"}}
    option={"option_id":"APPROACH_OPTION_00","approach_world_xy_m":candidate["approach_world_xy_m"],
        "p552_region_member":member,"p550_direction_hypothesis_id":candidate["direction_id"]}
    selected={"requested_approach_world_xy_m":candidate["approach_world_xy_m"],"snapped_goal_world_xy_m":candidate["approach_world_xy_m"]}
    region={"approach_options":[copy.deepcopy(option)],"region_arrays":member["region_arrays"],"audit":audit}
    arrays={**masks,"grid_lower_world_xy_m":np.array(LOWER),"voxel_size_m":np.array(RES)}
    route={"status":"verified_polygon_navmesh_route","validation":{"all_segment_samples_human_free":True},
        "route_world_xy_m":[candidate["approach_world_xy_m"]]}
    return option,selected,region,arrays,route


def test_exact_region_goal_and_actual_navmesh_endpoint_pass():
    result=validate_endpoint(*fixture())
    assert result["all_gates_passed"] and result["goal_snap_distance_m"]==0
    assert result["stage3_sit_transition_validated"] is False


def test_remote_snap_is_rejected_even_if_planner_claims_success():
    option,selected,region,arrays,route=fixture()
    selected["snapped_goal_world_xy_m"]=[1.5,1.5]
    with pytest.raises(ValueError,match="snapping"):validate_endpoint(option,selected,region,arrays,route)


def test_route_stopping_short_of_contact_region_is_rejected():
    option,selected,region,arrays,route=fixture()
    route["route_world_xy_m"]=[[1.5,1.5]]
    with pytest.raises(ValueError,match="stopped before"):validate_endpoint(option,selected,region,arrays,route)


def test_obstacle_crossing_cannot_be_hidden_by_endpoint_success():
    option,selected,region,arrays,route=fixture()
    route["route_world_xy_m"].insert(0,[-1.5,0.])
    with pytest.raises(ValueError,match="human clearance"):validate_endpoint(option,selected,region,arrays,route)


def test_changing_contact_anchor_breaks_candidate_binding():
    option,selected,region,arrays,route=fixture()
    option["p552_region_member"]["contact_anchor_world_xyz_m"]=[-1,0,.45]
    with pytest.raises(ValueError,match="changed"):validate_endpoint(option,selected,region,arrays,route)


def test_contact_connector_new_obstacle_is_rejected():
    option,selected,region,arrays,route=fixture()
    member=option["p552_region_member"]
    point=(np.array(member["approach_world_xy_m"])+np.array(member["contact_anchor_world_xyz_m"][:2]))/2
    cell=np.floor((point-LOWER)/RES).astype(int)
    arrays["other_obstacles"][tuple(cell)]=True
    with pytest.raises(ValueError,match="separates"):validate_endpoint(option,selected,region,arrays,route)


def test_wrong_direction_region_is_rejected():
    option,selected,region,arrays,route=fixture()
    arrays["DIRECTION_0_terminal_frontier"][:]=False
    with pytest.raises(ValueError,match="required region"):validate_endpoint(option,selected,region,arrays,route)

