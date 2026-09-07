"""Full CPU optimizer/search on a synthetic differentiable 10,475-vertex body.

This test does not execute a real SMPL-X checkpoint or claim model success.
"""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from hsi.common.artifacts import write_once, artifact
from hsi.stage2 import refine, placement, refine_physics, refine_margins, posture_quality, body_model


def test_world_grid_trilinear_gradient_and_voxel_center_boundary():
    field = refine_physics.WorldGridSDF(torch.arange(27, dtype=torch.float64).reshape(3,3,3), torch.zeros(3), torch.full((3,),3.))
    points = torch.tensor([[.5,.5,.5],[1.,1.,1.],[2.5,2.5,2.5],[.49,1.,1.]], requires_grad=True, dtype=torch.float64)
    sampled, outside = field.sample(points)
    assert sampled.tolist()[:3] == [0.,6.5,26.]
    assert outside.tolist() == [False,False,False,True]
    sampled[1].backward()
    assert torch.equal(points.grad[1], torch.tensor([9.,3.,1.], dtype=torch.float64))


def test_strict_inner_margins_and_bilateral_kth_area():
    values = torch.tensor([-.04,-.02,.03], requires_grad=True)
    loss = refine_margins.tail_hinge(values,.025)
    assert loss.item() == pytest.approx(.015**2*(1+1/3))
    loss.backward()
    assert values.grad[0] < 0 and values.grad[1] == 0
    v = torch.zeros((12,3)); v[:6,2]=.04; v[6:,2]=.015
    anatomy = SimpleNamespace(left_foot_support=torch.arange(12)<6,right_foot_support=torch.arange(12)>=6)
    assert refine_margins.bilateral_support_loss(v,anatomy,placement.GATES,.03).item() == pytest.approx(1500*.018**2+200*(.04**2+.015**2))


def test_so3_and_pose_caps():
    pose=torch.zeros((21,3),requires_grad=True)
    assert torch.equal(body_model.axis_angle_to_matrix(pose),torch.eye(3).expand(21,3,3))
    assert all(float(v)==0 for v in body_model.pose_limit_terms(pose))
    changed=pose.detach().clone(); changed[3,0]=3.
    assert body_model.pose_limit_terms(changed)[0]>0


class SyntheticSMPLX:
    def __init__(self):
        v=np.zeros((10475,3),np.float32)
        v[:,0]=np.linspace(-.15,.15,10475); v[:,1]=np.sin(np.arange(10475))*.08
        v[2500:4500,2]=-.45; v[2500:4500,0]+=.25; v[4500:,2]=.45
        self.vertices=torch.as_tensor(v); self.lbs_weights=torch.zeros((10475,55))
        for lo,hi,joint in ((0,2500,0),(2500,3500,7),(3500,4500,8),(4500,10475,16)):
            self.lbs_weights[lo:hi,joint]=1
        self.faces=np.tile([[0,1,2]],(20908,1)); self.joints=torch.zeros((22,3))
        self.joints[4:6,:]=torch.tensor([.25,0.,-.18]); self.joints[7:9,:]=torch.tensor([.25,0.,-.365])
        self.joints[10:12,:]=torch.tensor([.37,0.,-.365]); self.joints[15,:]=torch.tensor([0.,0.,.9])
    def __call__(self,*,body_pose,**kwargs):
        shift=body_pose.reshape(21,3).sum(0)*.0001
        v=self.vertices+shift; j=self.joints+shift
        return SimpleNamespace(vertices=torch.stack((v[:,0],v[:,2],-v[:,1]),-1)[None],
            joints=torch.stack((j[:,0],j[:,2],-j[:,1]),-1)[None])


class AnalyticField:
    def __init__(self,target=True): self.target=target
    def sample(self,points):
        values=(points[:,2]-.45).abs() if self.target else points[:,0]*0+.5
        return values,torch.zeros(len(points),dtype=torch.bool)


@pytest.fixture
def refine_case(tmp_path,monkeypatch):
    body=SyntheticSMPLX()
    monkeypatch.setattr(refine.p498.body_util,'load_fixed_smplx',lambda *args:body)
    target={'schema':refine.p498.TARGET_SCHEMA,'status':'published_synthetic_test_only','current_only_stage1_verified':True,
        'action_family':'sit','target_class':'chair','scene_id':'synthetic','instruction':'Sit in CPU fixture',
        'target_instance_id':'chair','selected_candidate_id':'seat','support_component_sha256':'a'*64,
        'target_occupancy_mask_world_xyz_sha256':'b'*64,'reference_relation':'none',
        'surface':{'centre_world_xyz_zup_m':[0.,0.,.45],'bounds_world_zup_m':[[-.5,-.5,.45],[.5,.5,.45]]},
        'approach':{'contact_forward_yaw_rad':0.}}
    target_path=tmp_path/'target.json';write_once(target_path,target)
    source=tmp_path/'source.npz'
    np.savez_compressed(source,schema=refine.SOURCE_SCHEMA,status='published_synthetic_test_only',
        h3_model_used=True,hybrikx_model_used=True,camera_world_placement_discarded=True,old_world_placement_consumed=False,
        selected_candidate_id='seat',support_component_sha256='a'*64,target_occupancy_mask_world_xyz_sha256='b'*64,
        body_pose_axis_angle=np.zeros((21,3)),betas=np.zeros(10),root_xyz_yaw=np.zeros(4),
        pose_frame_to_world_rotation=np.eye(3),vertices_world_zup=body.vertices.numpy()+[0,0,.45],faces=body.faces,
        route_arrival_tangent_yaw_rad=0.,terminal_facing_yaw_rad=0.,arrival_to_terminal_turn_rad=0.,expected_arrival_to_terminal_turn_rad=0.,
        route_arrival_tangent_used_as_terminal_facing=False,legacy_fixed_pi_expectation_used=False,
        **{key:'f'*64 for key in refine.SOURCE_LINEAGE_KEYS})
    occupancy=tmp_path/'occupancy.npy';np.save(occupancy,np.zeros((2,2,2),bool))
    segments=tmp_path/'segments.json';write_once(segments,{'hips':list(range(2500))})
    model=tmp_path/'synthetic.bin';model.write_bytes(b'synthetic test body, not a checkpoint')
    fields=SimpleNamespace(target=AnalyticField(),non_target=AnalyticField(False),component_receipt={'target_component_sha256':'b'*64})
    return dict(target_path=target_path,source_path=source,occupancy_path=occupancy,target_mask_path=occupancy,smplx_path=model,
        segmentation_path=segments,output_dir=tmp_path/'refined',steps=1,device_name='cpu',initialization_seed=0,
        search=dict(placement.SEARCH,same_surface_xy_search_radius_m=0.,same_surface_max_clearance_m=0.),
        publication_thresholds=placement.GATES,expected_occupancy_file_sha256=artifact(occupancy)['sha256'],
        expected_target_mask_file_sha256=artifact(occupancy)['sha256'],expected_target_world_sha256='b'*64,
        projection_weight=0.,field_builder=lambda *args,**kwargs:fields)


def test_full_current_optimizer_search_archive_eighteen_gates(refine_case):
    torch.set_num_threads(2)
    result=refine.run(**refine_case)
    assert result['stage3_handoff_allowed'] is True
    assert len(result['gates'])==18 and all(result['gates'].values())
    assert len(result['optimization_history'])==2
    assert len(result['same_surface_contact_search'])>=2
    assert result['adaptive_joint_bounds']['p550_neutral_sitting_ankle_limit_override']=={
        'ankle_rad':.70,'foot_rad':.50,'world_contact_yaw_scale_or_translation_freed':False}
    with np.load(result['keypose']['path'],allow_pickle=False) as data:
        limits=dict(zip(data['active_body_pose_rows'],data['active_body_pose_delta_limits_rad']))
        assert limits[6]>=.70 and limits[7]>=.70 and limits[9]==.50 and limits[10]==.50
        assert data['vertices_world_zup'].shape==(10475,3)
        assert not any(bool(data[k]) for k in ('free_translation_used','free_yaw_used','free_scale_used'))


def test_neutral_posture_checks_forward_direction_only_not_object_class():
    joints=SyntheticSMPLX().joints.numpy()+[0,0,.45]
    assert posture_quality.metrics(joints,0.)['all_neutral_sit_quality_gates_passed']
    assert not posture_quality.metrics(joints,math.pi)['neutral_sit_quality_gates']['feet_face_open_seat_side']


def test_failed_physics_runs_bounded_micro_search_and_does_not_publish(refine_case):
    class Occupied:
        def sample(self, points):
            return points[:,0]*0-.1,torch.zeros(len(points),dtype=torch.bool)
    fields=SimpleNamespace(target=AnalyticField(),non_target=Occupied(),component_receipt={'target_component_sha256':'b'*64})
    refine_case['field_builder']=lambda *a,**kw:fields
    result=refine.run(**refine_case)
    assert result['stage3_handoff_allowed'] is False
    assert not result['gates']['non_target_scene_penetration']
    assert any(row.get('search_scale')=='micro_2mm_within_6mm' for row in result['same_surface_contact_search'])
    assert result['keypose']['path'].endswith('review_only_keypose.npz')


def test_physics_violation_ranks_before_posture():
    values={'ground_penetration_m':.01,'non_target_scene_penetration_m':0.,'target_allowed_contact_penetration_m':.04,
        'target_forbidden_body_penetration_m':0.,'left_foot_ground_vertex_count_in_band':6,'right_foot_ground_vertex_count_in_band':6,
        'left_foot_ground_surface_topk_mean_m':.01,'right_foot_ground_surface_topk_mean_m':.01}
    first=refine_margins.violation_rank(values,{'a':False},placement.GATES)
    worse=refine_margins.violation_rank(dict(values,target_allowed_contact_penetration_m=.09),{'a':False},placement.GATES)
    assert first<worse
