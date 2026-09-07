"""Portable CPU tests: execute the whole selector with synthetic depth rendering.

Only the EGL device renderer is replaced. Camera/config/route contracts, crop,
surface sampling, owner/surface gates, ranking, artifact writes and fallback run.
These fixtures are not evidence of GPU rendering or a successfully generated pose.
"""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
from PIL import Image
import pytest

from hsi.common.artifacts import artifact, write_once, read_sealed
from hsi.stage2 import views, view_numeric, view_geometry


def test_relative_artifact_needs_explicit_base(tmp_path):
    from hsi.stage2 import _view_config
    path=tmp_path/'artifact.json';write_once(path,{'test':True})
    record=dict(artifact(path),path='artifact.json')
    with pytest.raises(_view_config.ConfigBuildError,match='explicit artifact_base_dir'):
        _view_config.checked_artifact(record,'relative')
    assert _view_config.checked_artifact(record,'relative',artifact_base_dir=tmp_path)[0]==path


@pytest.fixture
def scene(tmp_path, monkeypatch):
    import trimesh
    vertices = np.array([[-.3,-.3,.4], [.3,-.3,.4], [-.3,.3,.4], [.3,.3,.4]])
    faces = np.array([[0,1,2], [1,3,2]])
    source_vertices = vertices @ view_numeric.LINGO_YUP_TO_WORLD_ZUP[:3,:3]
    mesh = tmp_path / 'scene.obj'
    trimesh.Trimesh(source_vertices, faces, process=False).export(mesh)
    occupancy = tmp_path / 'occupancy.npy'; np.save(occupancy, np.ones((2,2,2), dtype=bool))
    depth = tmp_path / 'depth.npy'; np.save(depth, np.full((512,512), 3.3, dtype=np.float32))
    face_archive = tmp_path / 'surface.npz'
    np.savez(face_archive, mesh_face_mask=np.ones(2, dtype=bool), target_support_vertex_ids=np.arange(4), target_instance_id='chair_1')
    calibration = []
    render_views = []
    eye = np.array([0.,-3.,1.5]); look = np.array([0.,0.,.45])
    z = look-eye; z /= np.linalg.norm(z)
    x = np.cross(z, [0.,0.,1.]); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    w = np.eye(4); w[:3,:3] = [x,y,z]; w[:3,3] = -w[:3,:3] @ eye
    for index in range(48):
        camera_path = tmp_path / f'camera_{index:02d}.json'
        intrinsics = [[400.+index*.01,0,256.],[0,400.,256.],[0,0,1.]]
        camera = dict(schema='p508.lingo_fullscene_camera.v1', camera_id=index, width=512,height=512,
            coordinate_system='world_zup_metric', extrinsic_convention='opencv_world_to_camera',
            intrinsics_convention='pixels_fx_fy_cx_cy_origin_top_left', K=intrinsics, intrinsics=intrinsics,
            world_to_camera=w.tolist(), position_world_zup_m=eye.tolist(), target_world_zup_m=look.tolist())
        write_once(camera_path, camera, seal=False)
        rgb = tmp_path / f'rgb_{index:02d}.png'; Image.new('RGB',(512,512),(index,45,80)).save(rgb)
        row = dict(view_index=index, width=512,height=512,camera=artifact(camera_path),rgb=artifact(rgb))
        calibration.append(row); render_views.append(dict(row,depth=artifact(depth)))
    manifest_path = tmp_path/'cameras.json'
    manifest = dict(schema='p508.lingo_fullscene_calibrated_cameras.v1',status='fullscene_calibrated_views_ready',
        scene_id='fixture',coordinate_system='world_zup_metric',extrinsic_convention='opencv_world_to_camera',
        views=calibration,view_count=48)
    manifest['payload_sha256'] = views.digest(manifest); write_once(manifest_path,manifest,seal=False)
    render_path = tmp_path/'render.json'
    write_once(render_path,dict(schema='p508.lingo_fullscene_multiview_render.v1',status='fullscene_multiview_ready',
        scene_id='fixture',view_count=48,views=render_views,camera_sweep=dict(view_count=48),
        calibrated_camera_manifest=artifact(manifest_path),inputs=dict(original_scene_mesh=artifact(mesh),query_occupancy=artifact(occupancy)),
        query_contract={k:False for k in ('semantic_target_read','motion_pose_contact_label_read','p480_candidate_or_receipt_read','planner_memory_or_hsi_read','support_candidate_read')}))
    option = dict(option_id='front',approach_world_xy_m=[0.,-1.],contact_forward_yaw_rad=-math.pi/2,
        p550_direction_hypothesis_id='front',front_orientation_difference_rad=0.)
    surface = dict(centre_world_xyz_zup_m=[0.,0.,.4],bounds_world_zup_m=[[-.3,-.3,.4],[.3,.3,.4]])
    target_path=tmp_path/'target.json'
    target = dict(status='published_fixture_target',scene_id='fixture',instruction='Sit on the chair',
        target_instance_id='chair_1',target_class='chair',selected_surface_id='seat',selected_candidate_id='seat',
        selected_surface_sha256='1'*64,target_bounds_world_zup_m=[[-.4,-.4,.1],[.4,.4,.8]],
        candidate_surfaces=[dict(candidate_id='seat',surface=surface,approach_options=[option],
            surface_extraction_audit=dict(front_inference=dict(p550_direction_hypotheses=dict(orientation_resolved=True,
                candidates=[dict(candidate_id='front',outward_world_xy=[0.,-1.])]))))],
        artifacts=dict(original_scene_mesh=artifact(mesh),scene_occupancy=artifact(occupancy),target_surface_faces=artifact(face_archive)),
        current_only_contract=dict(fresh_current_occupancy_used=True,historical_surface_or_route_read=False,
            motion_pose_contact_keypose_or_future_frame_read=False))
    write_once(target_path,target)
    request=tmp_path/'request.json';write_once(request,dict(test_fixture=True))
    compile_path=tmp_path/'route_compile.json'
    write_once(compile_path,dict(status='fresh_roadmap_request_compiled',inputs=dict(p523_stage1_target=artifact(target_path)),
        p504_request=artifact(request),selected_approach_option_id='front',
        current_only_option_selection=dict(status='verified_path_cost_first_approach_selected',future_endpoint_or_motion_used=False,
            legacy_route_used_for_ranking=False,selected_approach_option_id='front'),
        approach_audits=[dict(option_id='front',requested_approach_world_xy_m=[0.,-1.],snapped_goal_world_xy_m=[0.,-1.],
            fresh_reachable=True,terminal_clearance_gate_passed=True,nearest_human_free_cell_reachable=True,
            legacy_route_fields_read=False,legacy_route_used_for_ranking=False)]))
    bundle_path=tmp_path/'bundle.json'
    write_once(bundle_path,dict(status='complete_stage1_verified',target_instance_id='chair_1',selected_surface_id='seat',
        selected_surface_sha256='1'*64,selected_approach_option_id='front',route_summary=dict(all_segment_samples_human_free=True),
        artifacts=dict(stage1_target=artifact(target_path),roadmap_compile=artifact(compile_path),roadmap_request=artifact(request))))
    quality_path=tmp_path/'quality.json'
    write_once(quality_path,dict(schema='p550.decoupled_stage1_geometry_quality.v1',status='sealed_quality_pass',
        quality_gates=dict(synthetic_route=True),publish_gate_pass=True,stage2_handoff_allowed=True,
        inputs=dict(planner_receipt=artifact(target_path))))
    stage1_path=tmp_path/'stage1.json'
    write_once(stage1_path,dict(status='complete',scene_id='fixture',instruction='Sit on the chair',
        target=artifact(target_path),bundle=artifact(bundle_path),planner_receipt=artifact(quality_path)))
    state=SimpleNamespace(calls=[], fail_indices=set())
    module=ModuleType('pyrender')
    class Mesh:
        @staticmethod
        def from_trimesh(*args,**kwargs):return SimpleNamespace(is_visible=True)
    class Scene:
        def __init__(self):self.nodes=[]
        def add(self,value,pose=None):
            node=SimpleNamespace(mesh=value,camera=value if hasattr(value,'fx') else None)
            self.nodes.append(node);return node
        def remove_node(self,node):self.nodes.remove(node)
    class Renderer:
        def __init__(self,width,height):assert (width,height)==(512,512)
        def render(self,scene,flags):
            index=round((scene.nodes[-1].camera.fx-400.)*100)
            state.calls.append(index)
            result=np.zeros((512,512),dtype=np.float32)
            if index not in state.fail_indices:result[210:250,210:250]=3.3
            return result
        def delete(self):pass
    module.Scene=Scene;module.Mesh=Mesh;module.OffscreenRenderer=Renderer
    module.MetallicRoughnessMaterial=lambda **kw:kw
    module.IntrinsicsCamera=lambda fx,fy,cx,cy:SimpleNamespace(fx=fx,fy=fy,cx=cx,cy=cy)
    module.RenderFlags=SimpleNamespace(DEPTH_ONLY=1)
    monkeypatch.setitem(sys.modules,'pyrender',module)
    return SimpleNamespace(stage1=stage1_path,render=render_path,target=target_path,root=tmp_path,state=state,render_views=render_views)


def test_complete_full_selector_runs_numeric_masks_and_writes_real_crops(scene):
    path=views.select_views(scene.stage1,scene.render,scene.root/'full')
    result=views.validate_receipt(path,scene.stage1)
    assert len(result['geometry_audit'])==48
    assert result['actual_depth_render_count']==len(scene.state.calls)==96
    assert result['selection']['selected_view_index']==0
    assert result['publication']==dict(stage2_generation_handoff_allowed=True,keypose_publishable=False)
    row=result['geometry_audit'][0]
    assert row['mask_metrics']['surface_visible_mask_pixels']==1600
    assert row['mask_metrics']['surface_mask_visible_fraction']==1
    assert len(row['mask_gates'])==5
    assert 'camera_on_stage1_approach_side' not in row['gates']
    assert 'camera_on_contact_facing_side' not in row['gates']
    selected=result['selection']
    with Image.open(selected['selected_source_rgb']['path']) as original, Image.open(selected['selected_stage2_crop']['path']) as crop:
        assert np.array_equal(np.asarray(original.crop(selected['selected_crop_xyxy']).resize((512,512),Image.Resampling.LANCZOS)),np.asarray(crop))
    assert len(read_sealed(scene.render)['views'])==48


def memory_fixture(scene,monkeypatch,index=0,stale=False):
    from hsi.memory import views as bank
    facts=scene.root/'facts.json';write_once(facts,dict(synthetic_fixture=True))
    context=dict(key=dict(scene_id='fixture'))
    monkeypatch.setattr(bank,'_context',lambda *a:context)
    row=scene.render_views[index]
    camera=views.read(row['camera']['path'])
    lookup=dict(geometry_identity=context['key'],role='pre_keypose_generation',grants_new_query_pass=False,
        current_stage1_execution=artifact(scene.stage1),source_facts=artifact(facts),candidates=[dict(camera=row['camera'],
            camera_parameters=dict(camera,camera_id=99) if stale else camera,
            original_selected_candidate_id=f'view_{index:02d}',historical_only=True,
            crop_hint_xyxy=['not','consumed'],historical_pass=True)])
    monkeypatch.setattr(bank,'query_views',lambda *a:lookup)
    return facts


def test_memory_fresh_one_has_same_numeric_row_and_pixels_as_full(scene,monkeypatch):
    full=views.validate_receipt(views.select_views(scene.stage1,scene.render,scene.root/'full'),scene.stage1)
    facts=memory_fixture(scene,monkeypatch)
    hint=views.validate_receipt(views.select_views(scene.stage1,scene.render,scene.root/'hint',mode='memory',facts_path=facts,store_root=scene.root/'bank'),scene.stage1)
    assert hint['audited_candidate_count']==hint['mask_depth_candidate_count']==1
    assert hint['actual_depth_render_count']==2
    left,right=copy.deepcopy(full['geometry_audit'][0]),copy.deepcopy(hint['geometry_audit'][0])
    left.pop('mask_visualization');right.pop('mask_visualization')
    assert left==right
    for key in ('selected_stage2_crop','selected_stage2_crop_target_overlay'):
        assert full['selection'][key]['sha256']==hint['selection'][key]['sha256']
    assert not hint['global_48_camera_optimum_recomputed']
    assert not (scene.root/'bank').exists()


def test_failed_hint_rechecks_full_and_preserves_original_index(scene,monkeypatch):
    facts=memory_fixture(scene,monkeypatch,42);scene.state.fail_indices.add(42)
    result=views.validate_receipt(views.select_views(scene.stage1,scene.render,scene.root/'fallback',mode='memory',facts_path=facts,store_root=scene.root/'bank'),scene.stage1)
    assert result['fallback_reason']=='fresh_hint_mask_or_geometry_checks_failed'
    assert result['audited_candidate_count']==49
    assert result['actual_depth_render_count']==98
    assert result['selection']['selected_view_index']==0
    assert scene.state.calls[:2]==[42,42]
    assert len(read_sealed(scene.render)['views'])==48


def test_stale_camera_hint_is_rejected_not_a_hit(scene,monkeypatch):
    facts=memory_fixture(scene,monkeypatch,stale=True)
    result=read_sealed(views.select_views(scene.stage1,scene.render,scene.root/'stale',mode='memory',facts_path=facts,store_root=scene.root/'bank'))
    assert result['lookup_error'] and result['memory_hint_count']==0
    assert result['audited_candidate_count']==48 and result['lookup'] is None


def test_surface_occlusion_cannot_be_overridden_by_hint(scene,monkeypatch):
    facts=memory_fixture(scene,monkeypatch);scene.state.fail_indices.update(range(48))
    result=views.validate_receipt(views.select_views(scene.stage1,scene.render,scene.root/'blocked',mode='memory',facts_path=facts,store_root=scene.root/'bank'),scene.stage1)
    assert result['selection'] is None
    assert not result['publication']['stage2_generation_handoff_allowed']
    assert result['status']=='rejected_no_complete_visible_view'


def test_partial_camera_inventory_rejected_before_render(scene):
    render=read_sealed(scene.render);render['views']=render['views'][:1];render['view_count']=1
    path=scene.root/'partial.json';write_once(path,render)
    with pytest.raises(ValueError,match='48-camera'):
        views.select_views(scene.stage1,path,scene.root/'bad')
    assert not scene.state.calls


def test_body_probes_use_contact_yaw_independent_of_route_position(scene):
    target=read_sealed(scene.target)
    first=view_numeric.body_probes(target,dict(contact_forward_yaw_rad=.4,selected_approach_world_xy_m=[1,0]))
    second=view_numeric.body_probes(target,dict(contact_forward_yaw_rad=.4,selected_approach_world_xy_m=[-1,0]))
    assert all(np.array_equal(first[key],second[key]) for key in first)
    assert first['head'].shape==(27,3)
    assert first['left_foot'].shape==first['right_foot'].shape==(9,3)


def test_mask_depth_tolerance_and_crop_exact_values():
    isolated=np.array([[0.,1.,2.],[np.nan,3.,4.]])
    scene=np.array([[0.,1.02,2.04],[2.,2.99,0.]])
    fraction,expected,visible=view_numeric.mask_visibility(isolated,scene,.03)
    assert fraction==.5 and expected.sum()==4 and visible.sum()==2
    crop=view_geometry.derive_crop(np.array([[180.,170.],[280.,290.]]),(512,512),view_geometry.DEFAULT_CROP)
    assert crop==(41,62,419,440)


def test_no_runtime_historical_code_loader_in_stage2():
    import ast
    for name in ('views.py','view_numeric.py','view_geometry.py','_view_config.py'):
        path=Path(views.__file__).with_name(name)
        tree=ast.parse(path.read_text())
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'exec','eval','compile','__import__'} for n in ast.walk(tree))
        assert 'agent9/methods/' not in path.read_text()
