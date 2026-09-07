"""Independent synthetic rig checks; licensed body data are not test fixtures."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

pytest.importorskip("smplx")
from hsi.stage3.skin import _joint_driven_chunk, skin_joint_driven


def rig():
    template=torch.zeros(10475,3)
    template[:55,0]=torch.arange(55)*.01
    regressor=torch.zeros(55,10475)
    regressor[torch.arange(55),torch.arange(55)]=1
    weights=torch.zeros(10475,55);weights[:,0]=1
    weights[500,0]=0;weights[500,21]=1
    return SimpleNamespace(parents=torch.tensor([-1]+[0]*54),NUM_BODY_JOINTS=21,
        v_template=template,shapedirs=torch.zeros(1,1,1).expand(10475,3,10),
        J_regressor=regressor,posedirs=torch.zeros(1,1).expand(486,10475*3),lbs_weights=weights)


def test_generated_wrist_position_drives_skin_not_just_pelvis_fk():
    model=rig();rotations=torch.eye(3).expand(2,22,3,3).clone()
    positions=model.v_template[:22].expand(2,22,3).clone()
    positions[1,21,1]=.3  # Only emitted wrist moves; all rotation matrices unchanged.
    vertices,drivers,_=_joint_driven_chunk(model,rotations,positions,torch.zeros(2,10),torch.eye(3).expand(30,3,3))
    assert torch.equal(drivers,positions)
    assert torch.allclose(vertices[1,500]-vertices[0,500],torch.tensor([0.,.3,0.]))
    assert torch.equal(vertices[1,700],vertices[0,700])


def test_missing_real_model_cannot_yield_successful_fullmesh(tmp_path):
    with pytest.raises(FileNotFoundError):
        skin_joint_driven(np.broadcast_to(np.eye(3,dtype=np.float32),(4,22,3,3)).copy(),
            np.zeros((4,22,3),np.float32),np.zeros((4,10),np.float32),
            body_model_path=tmp_path/"missing_licensed_body.npz")


def test_skin_rejects_invented_rotation_before_loading_body(tmp_path):
    rotations=np.broadcast_to(np.eye(3,dtype=np.float32),(4,22,3,3)).copy()
    rotations[0,0,0,0]=2
    with pytest.raises(ValueError,match="SO\\(3\\)"):
        skin_joint_driven(rotations,np.zeros((4,22,3),np.float32),np.zeros((4,10),np.float32),
                          body_model_path=tmp_path/"never_opened.npz")
