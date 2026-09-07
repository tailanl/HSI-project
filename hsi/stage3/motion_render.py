"""Actual evaluated-mesh motion pictures; integration is not GPU-tested.

This reader binds current runtime/evaluation/skin sources. It does not independently
reskin the saved mesh or replace full-motion physical admission. No renderer pass
can grant Memory credit. All actual masks retain original scene occluders.
"""
from pathlib import Path
import time
import numpy as np
from PIL import Image
from hsi.common.artifacts import artifact, read_json, read_sealed, require, verified, write_once
from hsi.stage1.scene import verify_artifact_tree
from . import motion_contract as contract, motion_camera as overview, motion_pictures as pictures


def sources():
    from hsi.stage2 import view_numeric
    from hsi.memory import runtime as identity_runtime
    from . import runtime, evaluation, skin
    return [artifact(path) for path in (__file__, contract.__file__, overview.__file__, pictures.__file__,
        view_numeric.__file__, identity_runtime.__file__, runtime.__file__, evaluation.__file__, skin.__file__)]


def load_inputs(execution_path, evaluation_path, sequence_path):
    from .runtime import validate_execution
    from . import evaluation as evaluator, metrics, skin as skin_module
    from hsi.memory.runtime import original_identity_from_stage1
    runtime = validate_execution(execution_path)
    ev = read_sealed(evaluation_path)
    require(ev['schema'] == 'hsi.stage3.fullmesh_motion_evaluation.v1'
        and ev['status'] == 'complete_current_fullmesh_motion_evaluation'
        and ev['evaluation_complete'] is True and ev['input_contract_valid'] is True
        and ev['source_execution'] == artifact(execution_path)
        and ev['generated_motion_modified'] is False and ev['thresholds_relaxed'] is False,
        'Expected unedited current full-mesh evaluation')
    verify_artifact_tree(ev)
    require(ev['sources'][:3] == [artifact(evaluator.__file__),artifact(metrics.__file__),artifact(skin_module.__file__)],
        'Evaluation producer source changed')
    for ev_key, state_key in [('source_motion','motion_path'),('source_metadata','metadata_path'),
        ('source_packet','packet_path'),('source_stage2','stage2_path'),('source_occupancy','occupancy_path')]:
        require(ev[ev_key] == artifact(runtime[state_key]), 'Cross-execution evaluation: '+ev_key)
    packet = read_sealed(runtime['packet_path'])
    stage2 = read_sealed(runtime['stage2_path'])
    stage1_path = verified(stage2['source_stage1'])
    identity, sequence = original_identity_from_stage1(stage1_path, sequence_path, runtime['body_model_record'])
    seq = read_sealed(verified(sequence))
    target = read_sealed(verified(read_sealed(stage1_path)['target']))
    keynodes = read_sealed(verified(packet['source_artifacts']['key_nodes']))
    route = np.asarray(keynodes['collision_safe_control_polyline']['nodes_world_xy_m'],dtype=float)
    with np.load(verified(stage2['keypose']),allow_pickle=False) as body:
        contact = np.asarray(body['contact_goal_world_xyz_zup_m']).copy()
        expected_faces = body['faces'].copy()
    skin = read_sealed(verified(ev['skinning']))
    require(skin['schema'] == 'hsi.stage3.actual_joint_driven_skinning.v1'
        and skin['source_motion'] == ev['source_motion'] and skin['source_metadata'] == ev['source_metadata']
        and skin['outputs']['vertices'] == ev['skinned_vertices']
        and skin['source_body_model'] == runtime['body_model_record']
        and skin['all_source_frames_preserved'] is True and skin['source_motion_edited'] is False,
        'Actual evaluated mesh lineage differs')
    with np.load(verified(ev['skinned_vertices']),allow_pickle=False) as data:
        vertices, faces = data['vertices_world'].copy(), data['faces'].copy()
    require(vertices.shape == (ev['execution']['frames'],10475,3) and np.isfinite(vertices).all()
        and faces.shape == (20908,3) and np.array_equal(faces,expected_faces), 'Incomplete actual evaluated body')
    metadata = read_json(runtime['metadata_path'])
    policy = contract.frame_policy(metadata,packet,len(vertices),ev)
    view = read_sealed(verified(stage2['source_view']))
    bindings = dict(execution=artifact(execution_path),evaluation=artifact(evaluation_path),
        source_motion=ev['source_motion'],source_metadata=ev['source_metadata'],vertices=ev['skinned_vertices'],
        packet=ev['source_packet'],stage2=ev['source_stage2'],stage1=stage2['source_stage1'],
        mesh=packet['source_artifacts']['original_scene_mesh'])
    state = dict(bindings=bindings,vertices=vertices,faces=faces,route=route,contact=contact,
        packet=packet,evaluation=ev,metadata=metadata)
    return dict(state=state,source_original_stage1_sequence=sequence,
        source_semantic_compilation=seq['semantic_compilation'],canonical_episode_identity=identity,
        target_id=target['target_instance_id'],target_bounds=target['target_bounds_world_zup_m'],
        frame_policy=policy,original_reference=view['selection']['selected_stage2_crop'],
        target_overlay=view['selection']['selected_stage2_crop_target_overlay'])


def select_overview(renderer,state):
    bounds = overview.motion_bounds(state['vertices'],state['route'],state['contact'])
    candidates = overview.overview_candidates(bounds)
    policy = contract.frame_policy(state['metadata'],state['packet'],len(state['vertices']),state['evaluation'])
    anchors = {0,len(state['vertices'])-1,policy['first_contact_frame'],policy['first_approach_frame']}
    while len(anchors)<8:
        candidates_to_add=[i for i in range(len(state['vertices'])) if i not in anchors]
        anchors.add(max(candidates_to_add,key=lambda i:(min(abs(i-j) for j in anchors),-i)))
    indices = sorted(anchors)
    screened, tested, selected = [], [], None
    for camera in candidates:
        screen = overview.screening_camera(camera)
        renderer.camera(screen)
        masks = [renderer.frame(i)[0] for i in indices]
        screened.append(dict(camera=camera,screening_camera=screen,frame_masks=masks,
            summary=overview.visibility_summary(masks)))
    ranked = sorted(screened,key=lambda row:(-row['summary']['minimum_visible_fraction'],
        -row['summary']['mean_visible_fraction'],-row['summary']['minimum_amodal_pixels'],row['camera']['candidate_id']))
    for row in [r for r in ranked if r['summary']['all_passed']][:overview.POLICY['maximum_full_frame_candidates']]:
        renderer.camera(row['camera'])
        masks = [renderer.frame(i)[0] for i in range(len(state['vertices']))]
        result = dict(camera=row['camera'],frame_masks=masks,summary=overview.visibility_summary(masks))
        tested.append(result)
        if result['summary']['all_passed']:
            selected = result
            break
    return dict(schema='hsi.stage3.actual_motion_overview_camera_audit.v1',inputs=state['bindings'],
        bounds_world_zup_m=bounds.tolist(),policy=overview.POLICY,screening_frames=indices,
        sampled_candidates=screened,full_frame_candidates=tested,
        selected_candidate_id=selected['camera']['candidate_id'] if selected else None,
        overview_camera_pass=selected is not None,no_occluders_hidden=True,qwen_camera_selection_used=False)


def validate_overview(render,state):
    audit = read_sealed(verified(render['overview_camera_audit']))
    require(audit['inputs'] == state['bindings'] and audit['policy'] == overview.POLICY
        and audit['overview_camera_pass'] is True and audit['no_occluders_hidden'] is True
        and audit['qwen_camera_selection_used'] is False,'Wrong actual full-frame overview audit')
    bounds = overview.motion_bounds(state['vertices'],state['route'],state['contact'])
    require(np.array_equal(bounds,audit['bounds_world_zup_m'])
        and [r['camera'] for r in audit['sampled_candidates']] == overview.overview_candidates(bounds),
        'Overview camera not fitted to actual motion bounds')
    selected = [row for row in audit['full_frame_candidates'] if row['camera']['candidate_id'] == audit['selected_candidate_id']]
    require(len(selected)==1 and selected[0]['camera'] == render['overview_camera'], 'Ambiguous overview camera')
    row=selected[0]
    require([m['frame'] for m in row['frame_masks']] == list(range(len(state['vertices'])))
        and row['summary'] == overview.visibility_summary(row['frame_masks']) and row['summary']['all_passed'] is True,
        'Incomplete full-frame overview visibility')
    from .motion_review import mask_check
    for mask in row['frame_masks']:mask_check(mask,mask['frame'])
    return row['camera']


def save_frame(renderer,output,role,camera,index):
    renderer.camera(camera)
    evidence,pixels=renderer.frame(index,masks=True,rgb=True)
    require(evidence['passed'] is True,'Actual semantic frame has insufficient true visibility')
    raw=output/(role+'_frame_%06d.png'%index)
    Image.fromarray(pixels).convert('RGB').save(raw)
    tile=pictures.label_tile(pixels,index)
    labelled=output/(role+'_frame_%06d_time_tile.png'%index)
    tile.save(labelled)
    return dict(role=role,frame_index=index,camera_id=camera['candidate_id'],image=artifact(raw),
        time_tile=artifact(labelled),visibility=evidence),tile


def render(execution_path,evaluation_path,sequence_path,output,*,gpu=0):
    started=time.monotonic(); source_records=sources()
    value=load_inputs(execution_path,evaluation_path,sequence_path)
    state,policy=value['state'],value['frame_policy']
    output=Path(output).resolve()
    require(type(gpu) is int and gpu>=0,'Explicit nonnegative GPU/EGL index required')
    output.mkdir(parents=True,exist_ok=False)
    renderer=overview._SceneRenderer(state,gpu,gpu)
    frames,timeline_tiles,tail_tiles,last_views=[],[],[],[]
    try:
        audit=select_overview(renderer,state)
        write_once(output/'overview_camera_audit.json',audit)
        require(audit['overview_camera_pass'],'No actual all-frame overview camera passed')
        value['overview_camera']=next(r['camera'] for r in audit['full_frame_candidates']
            if r['camera']['candidate_id']==audit['selected_candidate_id'])
        close=pictures.select_close_views(renderer,state['vertices'][policy['terminal_frame_indices']],
            value['target_bounds'],policy['terminal_frame_indices'])
        write_once(output/'close_camera_audit.json',dict(schema='hsi.stage3.actual_motion_semantic_close_camera_audit.v1',
            inputs=state['bindings'],sources=source_records,frame_policy=policy,**close))
        for index in policy['timeline_frame_indices']:
            row,tile=save_frame(renderer,output,'overview',value['overview_camera'],index)
            frames.append(row);timeline_tiles.append(tile)
        for number,camera_row in enumerate(close['selected']):
            indices=policy['terminal_frame_indices'] if number==0 else [policy['close_frame_index']]
            for index in indices:
                row,tile=save_frame(renderer,output,'close%d'%number,camera_row['camera'],index)
                frames.append(row)
                if number==0:tail_tiles.append(tile)
                if index==policy['close_frame_index']:last_views.append(row['image'])
    finally:
        renderer.close()
    timeline,terminal=output/'02_actual_timeline.png',output/'05_actual_terminal_four.png'
    pictures.assemble_montage(timeline_tiles).save(timeline)
    pictures.assemble_montage(tail_tiles).save(terminal)
    timeline_record=dict(image=artifact(timeline),source_tiles=[r['time_tile'] for r in frames if r['role']=='overview'])
    terminal_record=dict(image=artifact(terminal),source_tiles=[r['time_tile'] for r in frames if r['role']=='close0'])
    actual_images=[value['original_reference'],value['target_overlay'],timeline_record['image'],*last_views,terminal_record['image']]
    require(len(actual_images)==6,'Exactly six blind image slots required')
    for rec in [*source_records,*state['bindings'].values(),value['source_original_stage1_sequence'],
        value['source_semantic_compilation'],*actual_images]:verified(rec)
    return write_once(output/'receipt.json',dict(schema=contract.RENDER_SCHEMA,sources=source_records,inputs=state['bindings'],
        **{k:v for k,v in value.items() if k not in {'state','target_bounds'}},source_motion=state['bindings']['source_motion'],
        skinned_vertices=state['bindings']['vertices'],original_mesh=state['bindings']['mesh'],
        frame_count=len(state['vertices']),vertex_count=10475,fps=20,
        overview_camera_audit=artifact(output/'overview_camera_audit.json'),
        close_camera_audit=artifact(output/'close_camera_audit.json'),frames=frames,
        timeline_montage=timeline_record,terminal_montage=terminal_record,actual_images=actual_images,
        original_scene_geometry_unchanged=True,scene_occluders_hidden=False,actual_body_vertices_changed=False,
        evaluated_vertices_reused_verbatim=True,body_reskinned=False,motion_or_root_modified=False,
        qwen_previous_judgements_or_gate_text_rendered=False,qwen_selected_coordinates=False,
        only_frame_index_and_time_text_added=True,original_annotated_overview_pixels_used=False,
        renderer_grants_no_publication=True,positive_credit=0,render_device=renderer.device,
        independent_fullmesh_metrics_recomputed_by_renderer=False,integration_model_execution_tested=False,
        elapsed_seconds=time.monotonic()-started))
