import numpy as np
import pytest
from hsi.stage1.scene_build.components import stable_dominant
from hsi.stage1.scene_build.publication import review_envelope as run
from hsi.common.artifacts import write_once

def cloud(fringe_count=5):
    axis=np.arange(8)*.025
    main=np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),-1).reshape(-1,3)
    rng=np.random.default_rng(550)
    fringe=np.array([.260,.08,.08])+rng.uniform(-.001,.001,(fringe_count,3))
    far=np.array([2.,0.,0.])+rng.uniform(0,.08,(40,3))
    return np.concatenate([main,fringe,far])

def test_envelope_preserves_internal_parts_and_far_residual():
    points=cloud()
    main,audit=stable_dominant(points)
    assert audit["proposal_eligible"] and audit["envelope_closure_used"]
    assert main[:517].all() and not main[517:].any()
    assert len(points[main])+len(points[~main])==len(points)
    assert not audit["semantic_category_inferred_from_geometry"]

def test_envelope_does_not_merge_two_equal_objects():
    axis=np.arange(8)*.025
    first=np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),-1).reshape(-1,3)
    points=np.concatenate([first,first+[2,0,0]])
    _,audit=stable_dominant(points)
    assert not audit["proposal_eligible"]

def test_no_disjoint_points_no_recovery():
    axis=np.arange(8)*.025
    points=np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),-1).reshape(-1,3)
    with pytest.raises(ValueError,match="No disjoint"):
        stable_dominant(points)

@pytest.mark.parametrize("field",["human_instruction_read","stage2_or_stage3_output_read"])
def test_final_scene_rejects_human_or_motion_conditioned_input(tmp_path,field):
    value={"schema":"p550.spatial_and_direction_reviewed_fixed_scene.v1",
        "human_instruction_read":False,"stage2_or_stage3_output_read":False}
    value[field]=True
    path=tmp_path / "source.json"
    write_once(path,value,seal=True)
    with pytest.raises(ValueError,match="scene-only"):
        run(path,tmp_path / "output",0,qwen_client=None)
