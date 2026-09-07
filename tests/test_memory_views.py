"""Synthetic calibrated receipts test persistence, never actual GPU inference."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hsi.common.artifacts import artifact, digest, read_json, read_sealed, write_once
from hsi.memory import facts, views as memory
from hsi.stage2 import views as selector
from test_memory_facts import fixture_scene


def put(path, value, *, seal=True):
    write_once(path, value, seal=seal)
    return artifact(path)


@pytest.fixture
def scene(tmp_path):
    final = fixture_scene(tmp_path)
    fact_path = facts.build_scene_facts(final, tmp_path / 'facts')
    data = facts.load_scene_facts(fact_path)
    row = data['objects'][0]
    root = tmp_path / 'synthetic_task'
    root.mkdir()
    face_path = root / 'surface.npz'
    np.savez_compressed(face_path, target_instance_id='SEAT', mesh_face_indices=np.array([0, 1]),
        surface_vertex_ids=np.arange(4), target_support_vertex_ids=np.arange(4),
        mesh_face_mask=np.array([True, True]))
    option = dict(option_id='APPROACH_0', approach_world_xy_m=[2., 0.],
        p550_direction_hypothesis_id='DIRECTION_0', contact_forward_yaw_rad=0.,
        front_orientation_difference_rad=0.)
    surface = dict(candidate_id='SURFACE_0', surface=row['surface'], surface_face_archive=artifact(face_path),
        approach_options=[option], furniture_instance_binding=dict(verified=True,
            surface_inside_target_furniture_verified=True, graph_target_instance_id='SEAT',
            resolved_target_class='chair', target_occupancy_mask_world_xyz_sha256='e'*64),
        surface_extraction_audit=dict(front_inference=dict(p550_direction_hypotheses=row['direction_hypotheses'])))
    target = dict(scene_id='TEST', instruction='sit on the fixture chair', action_family='sit',
        target_instance_id='SEAT', selected_surface_id='SURFACE_0', selected_surface_sha256=row['stable_surface_sha256'],
        candidate_surfaces=[surface], artifacts=dict(original_scene_mesh=data['sources']['source_mesh'],
            scene_occupancy=data['sources']['source_occupancy'], target_surface_faces=artifact(face_path)))
    target_binding = put(root / 'target.json', target)
    request = put(root / 'request.json', {'synthetic': 'current route request'})
    compiled = dict(status='fresh_roadmap_request_compiled', inputs=dict(p523_stage1_target=target_binding),
        p504_request=request, selected_approach_option_id='APPROACH_0',
        current_only_option_selection=dict(status='verified_path_cost_first_approach_selected',
            future_endpoint_or_motion_used=False, legacy_route_used_for_ranking=False,
            selected_approach_option_id='APPROACH_0'),
        approach_audits=[dict(option_id='APPROACH_0', requested_approach_world_xy_m=[2., 0.],
            snapped_goal_world_xy_m=[2., 0.], fresh_reachable=True, terminal_clearance_gate_passed=True,
            nearest_human_free_cell_reachable=True, legacy_route_fields_read=False,
            legacy_route_used_for_ranking=False)])
    compilation = put(root / 'compile.json', compiled)
    bundle = put(root / 'bundle.json', dict(status='complete_stage1_verified',
        **{k: target[k] for k in ('target_instance_id', 'selected_surface_id', 'selected_surface_sha256')},
        selected_approach_option_id='APPROACH_0', route_summary=dict(all_segment_samples_human_free=True),
        artifacts=dict(stage1_target=target_binding, roadmap_compile=compilation, roadmap_request=request)))
    quality = put(root / 'quality.json', dict(schema='p550.decoupled_stage1_geometry_quality.v1',
        status='sealed_quality_pass', quality_gates=dict(synthetic_geometry=True), publish_gate_pass=True,
        stage2_handoff_allowed=True, inputs=dict(planner_receipt=target_binding)))
    stage_path = root / 'stage1.json'
    put(stage_path, dict(schema='p550.stage1_query_execution.v1', status='complete', scene_id='TEST',
        instruction=target['instruction'], target=target_binding, bundle=bundle, planner_receipt=quality,
        source_fixed_geometry=data['sources']['fixed_geometry']))
    rgb_path = root / 'rgb.png'
    Image.new('RGB', (640, 480), (30, 60, 90)).save(rgb_path)
    depth_path = root / 'depth.npy'
    np.save(depth_path, np.ones((480, 640), dtype=np.float32))
    rendered = []
    for index in range(48):
        camera = dict(schema='p508.lingo_fullscene_camera.v1', camera_id=index,
            coordinate_system='world_zup_metric', extrinsic_convention='opencv_world_to_camera',
            intrinsics_convention='pixels_fx_fy_cx_cy_origin_top_left', width=640, height=480,
            K=[[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]],
            world_to_camera=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 3.], [0., 0., 0., 1.]],
            position_world_zup_m=[0., 0., -3.], target_world_zup_m=[0., 0., 0.])
        camera_binding = put(root / ('camera_%02d.json' % index), camera, seal=False)
        rendered.append(dict(view_index=index, camera=camera_binding, rgb=artifact(rgb_path), depth=artifact(depth_path)))
    camera_manifest = dict(schema='p508.lingo_fullscene_calibrated_cameras.v1',
        status='fullscene_calibrated_views_ready', scene_id='TEST',
        coordinate_system='world_zup_metric', extrinsic_convention='opencv_world_to_camera',
        view_count=48, views=[dict(view_index=view['view_index'], camera=view['camera'], rgb=view['rgb'])
            for view in rendered])
    camera_manifest['payload_sha256'] = digest(camera_manifest)
    manifest_binding = put(root / 'camera_manifest.json', camera_manifest, seal=False)
    render = put(root / 'render.json', dict(schema='p508.lingo_fullscene_multiview_render.v1',
        status='fullscene_multiview_ready', scene_id='TEST', view_count=48, views=rendered,
        calibrated_camera_manifest=manifest_binding,
        inputs=dict(original_scene_mesh=data['sources']['source_mesh'], query_occupancy=data['sources']['source_occupancy'])))
    crop = [0, 0, 480, 480]
    audits = [dict(candidate_id='view_%02d' % index, view_index=index,
        mask_depth_checks_performed=index == 0, all_geometry_gates_passed=index == 0) for index in range(48)]
    audits[0].update(derived_crop_xyxy=crop, gates=dict(synthetic_actual_mask=True),
        mask_gates=dict(surface_area_in_crop=True, surface_mostly_unoccluded=True,
            surface_mask_large_enough=True, owner_not_cropped=True, owner_mostly_unoccluded=True),
        mask_metrics=dict(surface_mask_visible_fraction=1., owner_mask_visible_fraction=1.,
            surface_area_in_crop_fraction=1., surface_visible_mask_pixels=2000,
            probe_visibility=dict(head=1., left_foot=1., right_foot=1.)),
        camera_contact_facing_cosine=1.)
    selected = dict(selected_candidate_id='view_00', selected_view_index=0,
        selected_camera=rendered[0]['camera'], selected_source_rgb=rendered[0]['rgb'],
        selected_crop_xyxy=crop, selected_stage2_crop=artifact(rgb_path),
        selected_stage2_crop_target_overlay=artifact(rgb_path))
    value = dict(schema=selector.SCHEMA, mode='full', source=artifact(selector.__file__),
        status='geometry_view_ready', source_stage1_execution=artifact(stage_path), target=target_binding,
        source_render=render, surface_faces=artifact(face_path), policy=copy.deepcopy(selector.POLICY),
        geometry_audit=audits, selection=selected, qwen_calls=0, numeric_geometry_computed_by_qwen=False,
        publication=dict(stage2_generation_handoff_allowed=True, keypose_publishable=False))
    attempt = put(root / 'attempt.json', value)
    value.update(source_closure=selector.source_closure(), attempts=[dict(receipt=attempt)],
        audited_candidate_count=48, mask_depth_candidate_count=1, actual_depth_render_count=2,
        fresh_check_contract=dict(historical_gate_pass_reused=False, historical_crop_used=False,
            full_render_inventory_preserved=True), current_geometry_identity=None)
    view_path = root / 'view.json'
    put(view_path, value)
    return dict(facts=fact_path, stage1=stage_path, view=view_path, root=root)


def test_synthetic_complete_save_and_query_copies_camera_not_route(scene, tmp_path):
    bank = tmp_path / 'bank'
    receipt = memory.save_selection(scene['facts'], scene['view'], bank)
    saved = read_sealed(receipt)
    assert saved['camera_parameters'] == read_json(saved['camera_copy']['path'])
    assert saved['historical_candidate_count'] == 48
    assert saved['positive_credit'] == 0 and saved['grants_new_query_pass'] is False
    assert saved['navmesh_or_route_objects_saved'] is False
    assert saved['pose_motion_or_qwen_answers_saved'] is False
    assert {p.name for p in receipt.parent.iterdir()} == {'camera.json', 'historical_audit.json', 'receipt.json'}
    lookup = memory.query_views(scene['facts'], scene['stage1'], bank)
    assert lookup['candidate_count'] == 1 and all(lookup['must_revalidate'].values())
    assert lookup['grants_new_query_pass'] is False
    assert memory.validate_lookup(lookup, scene['facts'], scene['stage1'], bank) == lookup


def test_readonly_miss_no_bank_creation(scene, tmp_path):
    bank = tmp_path / 'absent'
    assert memory.query_views(scene['facts'], scene['stage1'], bank)['reason'] == 'store_absent'
    assert not bank.exists()


def test_idempotent_and_concurrent_save(scene, tmp_path):
    bank = tmp_path / 'bank'
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: memory.save_selection(scene['facts'], scene['view'], bank), range(2)))
    assert paths[0] == paths[1]
    before = paths[0].stat().st_mtime_ns
    assert memory.save_selection(scene['facts'], scene['view'], bank) == paths[0]
    assert paths[0].stat().st_mtime_ns == before


@pytest.mark.parametrize('kind', ['camera_bytes', 'resealed_pass', 'policy', 'subset', 'old_selector'])
def test_invalid_evidence_is_not_a_hint(scene, tmp_path, kind):
    bank = tmp_path / 'bank'
    if kind in {'camera_bytes', 'resealed_pass'}:
        path = memory.save_selection(scene['facts'], scene['view'], bank)
        if kind == 'camera_bytes':
            (path.parent / 'camera.json').write_text('{}')
        else:
            value = read_sealed(path)
            value['grants_new_query_pass'] = True
            value['receipt_payload_sha256'] = digest({k:v for k,v in value.items() if k != 'receipt_payload_sha256'})
            import json
            path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            memory.query_views(scene['facts'], scene['stage1'], bank)
    else:
        value = read_sealed(scene['view'])
        if kind == 'policy': value['policy']['surface_mask_visible_fraction_min'] = 0.
        if kind == 'subset': value['mode'] = 'memory'
        if kind == 'old_selector': value['schema'] = 'p549.geometric_stage2_view_selection.v1'
        path = scene['root'] / ('bad_' + kind + '.json')
        put(path, value)
        with pytest.raises(ValueError): memory.save_selection(scene['facts'], path, bank)
        assert not bank.exists()


@pytest.mark.parametrize('change', ['skew', 'nonrigid', 'position', 'world_pose', 'crop', 'camera_id'])
def test_camera_contract_rejects_bad_geometry(scene, change):
    selected = read_sealed(scene['view'])['selection']
    camera = read_json(selected['selected_camera']['path'])
    crop = selected['selected_crop_xyxy']
    if change == 'skew': camera['K'][0][1] = .1
    if change == 'nonrigid': camera['world_to_camera'][0][0] = 2.
    if change == 'position': camera['position_world_zup_m'][0] = 1.
    if change == 'world_pose': camera['body_pose'] = [0.] * 63
    if change == 'crop': crop[2] = 641
    if change == 'camera_id': camera['camera_id'] = True
    with pytest.raises(ValueError): memory.validate_camera(camera, crop)


def test_context_child_change_during_lookup_rejected(scene, tmp_path, monkeypatch):
    original = memory._context
    count = 0
    def changing(*args):
        nonlocal count
        result = original(*args)
        count += 1
        if count == 2: result['target']['synthetic_child_drift'] = True
        return result
    monkeypatch.setattr(memory, '_context', changing)
    with pytest.raises(ValueError, match='dependencies changed'):
        memory.query_views(scene['facts'], scene['stage1'], tmp_path / 'absent')


def test_source_code_policy_rejects_foreign_store(scene, tmp_path):
    bank = tmp_path / 'bank'
    bank.mkdir()
    marker = put(bank / 'manifest.json', dict(schema='not-a-camera-bank'))
    with pytest.raises(ValueError, match='Foreign'):
        memory.save_selection(scene['facts'], scene['view'], bank)
    assert artifact(bank / 'manifest.json') == marker
