"""CPU numerical/admission checks, never a live generated-motion success claim."""
import copy
from pathlib import Path

import numpy as np
import pytest

from hsi.common.artifacts import write_once
from hsi.stage3.evaluation import load_motion, validate_fresh_metadata, numeric_evaluation
from hsi.stage3.release import release_gates, current_policy


def packet():
    joints = np.zeros((22,3),np.float32)
    joints[:,2] = .5
    joints[1,1],joints[2,1] = .1,-.1
    joints[[10,11],2] = .02
    return {"scene_name":"test", "seq_name":"test_seq", "text":"sit", "provenance":"hsi_verified_h3_hybrikx",
        "nodes":[{"ordinal":0,"root_xyz_yaw":[-.2,0,.5,0]},
                 {"ordinal":1,"root_xyz_yaw":[0,0,.5,0]},
                 {"ordinal":2,"root_xyz_yaw":[0,0,.5,0],"contact_joint_ids":[0,1,2],
                  "full_smplx_keypose":{"joints_world_zup":joints.tolist(),"root_xyz_yaw":[0,0,.5,0]}}]}


def metadata(occupancy_path):
    return {"scene_name":"test", "plan_seq_name_from_file":"test_seq", "plan_provenance":"hsi_verified_h3_hybrikx",
        "text":"sit", "lingo_motion_opened":False,"posthoc_motion_edit_used":False,"retrieval_used":False,
        "future_frame_2_plus_available_to_runtime":False,"generator_reads_future_root_directly_from_dataset":False,
        "gt_contact_or_icgf_used":False,"generated_only_output":True,
        "query_static_runtime_audit":{"future_or_gt_motion_used":False,"lingo_motion_opened":False},
        "raw_occupancy":{"source_path":str(occupancy_path)},"scheduler_complete":True,"nodes_consumed":3,
        "p478_terminal_rolewise_completion":{"terminal_complete":True,"fail_closed_incomplete":False},
        "p478_causal_prefix":{"initial_history_frames":0},
        "primitive_records":[{"active_external_ordinal":2,"primitive_id":1,
                              "e226_receipt":{"generated_frames_in_prefix":2}}]}


def motion_arrays():
    return {"joints":np.broadcast_to(np.asarray(packet()["nodes"][-1]["full_smplx_keypose"]["joints_world_zup"],np.float32),(6,22,3)).copy(),
        "global_orient":np.broadcast_to(np.eye(3,dtype=np.float32),(6,3,3)).copy(),
        "body_pose":np.broadcast_to(np.eye(3,dtype=np.float32),(6,21,3,3)).copy(),
        "betas":np.zeros(10,np.float32),"scene_name":np.array("test"),"text":np.array("sit")}


def test_motion_preserves_every_generated_joint_and_expands_neutral_shape(tmp_path):
    source=tmp_path/"motion.npz";meta=tmp_path/"metadata.json"
    arrays=motion_arrays();np.savez_compressed(source,**arrays)
    write_once(meta,metadata(tmp_path))
    loaded=load_motion(source,meta,packet())
    assert np.array_equal(loaded["joints"],arrays["joints"])
    assert loaded["betas"].shape==(6,10)
    assert np.array_equal(loaded["rotations"][:,0],arrays["global_orient"])


@pytest.mark.parametrize("problem",["missing_shape","wrong_dtype","nonfinite","bad_rotation","nonneutral","short_motion","posthoc"])
def test_malformed_or_edited_motion_fails_closed(tmp_path,problem):
    arrays=motion_arrays();meta_value=metadata(tmp_path)
    if problem=="missing_shape":arrays.pop("betas")
    elif problem=="wrong_dtype":arrays["joints"]=arrays["joints"].astype(np.float64)
    elif problem=="nonfinite":arrays["joints"][0,0,0]=np.nan
    elif problem=="bad_rotation":arrays["global_orient"][0,0,0]=2
    elif problem=="nonneutral":arrays["betas"][0]=.1
    elif problem=="short_motion":arrays["joints"]=arrays["joints"][:3]
    elif problem=="posthoc":meta_value["posthoc_motion_edit_used"]=True
    source=tmp_path/"motion.npz";meta=tmp_path/"metadata.json"
    np.savez_compressed(source,**arrays);write_once(meta,meta_value)
    with pytest.raises(ValueError):load_motion(source,meta,packet())


@pytest.mark.parametrize("field",["lingo_motion_opened","posthoc_motion_edit_used","future_frame_2_plus_available_to_runtime",
                                "generator_reads_future_root_directly_from_dataset","gt_contact_or_icgf_used"])
def test_future_or_oracle_metadata_is_rejected(tmp_path,field):
    occupancy=tmp_path/"occupancy.npy";np.save(occupancy,np.zeros((20,30,10),bool))
    value=metadata(occupancy);value[field]=True
    with pytest.raises(ValueError,match="boundary"):validate_fresh_metadata(value,packet(),occupancy)


def numeric_fixture(tmp_path):
    occupancy=tmp_path/"occupancy.npy"
    grid=np.zeros((20,30,10),bool);grid[0,0,0]=True;np.save(occupancy,grid)
    sources={}
    for key in ("metrics_unified","lingo_evaluation_driver"):
        path=tmp_path/(key+".py");path.write_text("# Synthetic source identity fixture, not official execution.\n")
        sources[key]=path
    skin=tmp_path/"skin.npz"
    vertices=np.broadcast_to(np.array([0,0,.5],np.float32),(6,10475,3)).copy()
    np.savez_compressed(skin,vertices_world=vertices)
    return occupancy,sources,skin,vertices


def test_exact_34_numeric_release_checks_include_actual_fullmesh(tmp_path):
    occupancy,sources,skin,_=numeric_fixture(tmp_path)
    result=numeric_evaluation(motion_arrays()["joints"],packet(),metadata(occupancy),occupancy,skin,
                              np.arange(16),official_sources=sources)
    result["runner_returncode"]=0
    gates=release_gates(result,current_policy())
    assert len(gates)==34
    assert result["fullmesh_scene_collision"]["vertex_count"]==10475
    assert result["fullmesh_scene_collision"]["frame_count"]==6
    assert result["fullmesh_scene_collision"]["target_component_kept_in_scene_sdf"] is True
    assert result["occupancy_contract"]["target_component_removed"] is False
    assert result["motion_quality"]["fps"]==20
    assert all(row["passed"] for row in gates.values())  # Numeric fixture only, no source admission.
    result["runner_returncode"]=9
    assert not release_gates(result,current_policy())["actual_runner_success"]["passed"]


@pytest.mark.parametrize("problem",["absent","joints_only","missing_frames","nan"])
def test_fullmesh_cannot_be_replaced_by_joints_or_partial_frames(tmp_path,problem):
    occupancy,sources,skin,vertices=numeric_fixture(tmp_path)
    if problem=="absent":skin=None
    else:
        if problem=="joints_only":vertices=vertices[:,:22]
        elif problem=="missing_frames":vertices=vertices[:5]
        else:vertices[0,0,0]=np.nan
        skin=tmp_path/"malformed.npz";np.savez_compressed(skin,vertices_world=vertices)
    with pytest.raises(ValueError,match="[Ff]ull-mesh"):
        numeric_evaluation(motion_arrays()["joints"],packet(),metadata(occupancy),occupancy,skin,
                           np.arange(16),official_sources=sources)
