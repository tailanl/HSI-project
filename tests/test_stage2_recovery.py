"""CPU recovery mathematics and external-dependency boundaries; no model run."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from hsi.stage2 import recovery


@pytest.mark.parametrize('angle',[0.,1e-9,.4,math.pi-1e-7,math.pi])
def test_rotation_roundtrip_including_pi(angle):
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
    matrix=recovery.axis_angle_to_matrix(axis*angle)
    restored=recovery.axis_angle_to_matrix(recovery.matrix_to_axis_angle(matrix))
    assert np.allclose(restored,matrix,atol=1e-7)


def test_all_55_joint_matrices_recover_proper_so3():
    rng=np.random.default_rng(314)
    matrices=rng.normal(size=(55,3,3))
    rotations=recovery.project_so3(matrices)
    assert np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-12)
    assert np.allclose(np.linalg.det(rotations),1,atol=1e-12)
    vectors=recovery.rotations_to_axis_angle(rotations)
    assert vectors.shape==(55,3)
    assert np.allclose(np.stack([recovery.axis_angle_to_matrix(v) for v in vectors]),rotations,atol=1e-7)


def test_nonfinite_so3_rejected():
    with pytest.raises(recovery.P533HybrIKError):recovery.project_so3(np.full((3,3),np.nan))


def test_root_frame_retains_full_pitch_roll_without_camera_translation():
    w=np.eye(4);w[:3,:3]=recovery.axis_angle_to_matrix([.4,.2,-.1]);w[:3,3]=[90,-20,4]
    camera=recovery.Camera(w,np.eye(3),512,512,'test')
    root=recovery.axis_angle_to_matrix([.2,.7,.3])
    result=recovery.decompose_root_rotation(root,camera)
    altered=w.copy();altered[:3,3]=[-900,45,2]
    other=recovery.decompose_root_rotation(root,recovery.Camera(altered,np.eye(3),512,512,'test'))
    assert np.array_equal(result.pose_frame_to_world_rotation,other.pose_frame_to_world_rotation)
    assert np.allclose(result.pose_frame_to_world_rotation@recovery.NATIVE_YUP_TO_MATERIALIZER_ZUP,
        recovery.rotation_z(-result.recovered_yaw_world_rad)@w[:3,:3].T@root,atol=1e-8)
    assert abs(result.yaw_residual_rad)<1e-7


def test_neutral_beta_fold_retains_least_squares_exact_geometry(tmp_path):
    rng=np.random.default_rng(810)
    shape=rng.normal(size=(10475,3,10))*.005
    shape-=shape.mean(axis=0,keepdims=True)
    adult=np.zeros((10475,3));kid=shape[:,:,0]
    model=tmp_path/'neutral.npz';np.savez(model,shapedirs=shape,v_template=adult)
    template=tmp_path/'kid.npy';np.save(template,kid)
    beta=np.linspace(-.1,.1,11);beta[-1]=.2
    result=recovery.fold_kid_beta_to_neutral(beta,model,template)
    expected=beta[:10].copy();expected[0]+=.2
    assert np.allclose(result.folded_beta10,expected,atol=1e-7)
    assert result.design_rank==10 and result.residual_max_m<1e-14


class TensorFixture:
    def __init__(self,value):self.value=np.asarray(value)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


def test_detector_selects_highest_scored_actual_person_not_other_class():
    index,score,box=recovery._select_person(dict(scores=TensorFixture([.99,.6,.8]),
        labels=TensorFixture([2,1,1]),boxes=TensorFixture([[0,0,2,2],[1,1,3,3],[4,4,8,8]])))
    assert index==2 and score==.8 and np.array_equal(box,[4,4,8,8])


def test_reprojection_writes_actual_png_and_measures_points(tmp_path):
    source=tmp_path/'input.png';Image.new('RGB',(512,512),'grey').save(source)
    output=tmp_path/'overlay.png'
    result=recovery._render_reprojection(source,np.zeros((55,3)),np.array([100,80,300,400]),
        np.array([95,75,305,405]),.91,output)
    assert result['body22_inside_image_fraction']==1
    assert result['body22_bbox_xyxy']==[200.,240.,200.,240.]
    with Image.open(output) as image:assert image.format=='PNG' and image.size==(512,512)


def test_external_models_are_not_optional_or_implicitly_downloaded(tmp_path):
    with pytest.raises(ValueError,match='explicit'):
        recovery.RuntimeConfig.from_mapping({})
    assert not list(tmp_path.iterdir())


def test_recovery_contains_real_detector_and_model_calls_no_historical_loader():
    import ast
    source=Path(recovery.__file__).read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'exec','eval','compile','__import__'} for n in ast.walk(tree))
    assert 'agent9/methods/' not in source and 'agent10/' not in source
    assert 'detector([pil_to_tensor(image)' in source and 'builder.build_sppe(cfg.MODEL)' in source
    assert 'projected_theta = project_so3(raw_theta)' in source and 'model.smplx_layer.forward_simple(' in source
