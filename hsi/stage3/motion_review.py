"""Actual-motion six-image Qwen review, unchanged nine checks and .85 floor.

New source integration is not GPU/model end-to-end tested. This review validates
clean frame/montage assembly and actual raw Qwen provenance, not complete physical
motion correctness; full-body evaluation remains a separate required authority.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from hsi.common.artifacts import artifact,read_sealed,require,verified,write_once,_unique_pairs
from hsi.stage1 import qwen
from . import motion_render as renderer,motion_contract as contract,motion_pictures as pictures,motion_camera as camera_math


def source_records():
    return [artifact(__file__),artifact(qwen.__file__),*renderer.sources()]


def image_pixels(record):
    with Image.open(verified(record)) as im:
        require(im.format == 'PNG' and im.mode == 'RGB', 'Semantic evidence must be actual RGB PNG')
        return np.asarray(im).copy()


def mask_check(row, index):
    require(type(row['frame']) is int and row['frame'] == index
        and type(row['amodal_body_pixels']) is type(row['in_scene_visible_body_pixels']) is int
        and 0 <= row['in_scene_visible_body_pixels'] <= row['amodal_body_pixels']
        and row['amodal_body_pixels'] >= camera_math.POLICY['minimum_amodal_body_pixels']
        and type(row['in_scene_visible_fraction']) in (float,int)
        and row['in_scene_visible_fraction'] == row['in_scene_visible_body_pixels']/row['amodal_body_pixels']
        and row['in_scene_visible_fraction'] >= camera_math.POLICY['minimum_in_scene_visible_fraction']
        and row['scene_occluders_retained_in_depth'] is True and row['passed'] is True,
        'Missing/failed actual true scene-occluded body visibility')


def validate_render(render_path):
    source_records()
    render = read_sealed(render_path)
    require(render['schema'] == contract.RENDER_SCHEMA and render['sources'] == renderer.sources(),
        'Unregistered actual motion semantic renderer')
    context = renderer.load_inputs(verified(render['inputs']['execution']), verified(render['inputs']['evaluation']), verified(render['source_original_stage1_sequence']))
    context['overview_camera'] = renderer.validate_overview(render, context['state'])
    state, policy = context['state'], context['frame_policy']
    require(render['inputs'] == state['bindings'] and render['source_motion'] == state['bindings']['source_motion']
        and render['skinned_vertices'] == state['bindings']['vertices'] and render['original_mesh'] == state['bindings']['mesh']
        and render['frame_count'] == len(state['vertices']) and render['vertex_count'] == 10475 and render['fps'] == 20,
        'Cross motion/mesh/fullframe semantic render')
    for key in ('source_original_stage1_sequence','source_semantic_compilation','canonical_episode_identity',
            'target_id','frame_policy','overview_camera','original_reference','target_overlay'):
        require(render[key] == context[key], 'Actual semantic binding drift: '+key)
    for key in ('original_scene_geometry_unchanged','evaluated_vertices_reused_verbatim',
            'only_frame_index_and_time_text_added','renderer_grants_no_publication'):
        require(render[key] is True, 'Missing unchanged renderer contract: '+key)
    for key in ('scene_occluders_hidden','actual_body_vertices_changed','body_reskinned','motion_or_root_modified',
            'qwen_previous_judgements_or_gate_text_rendered','qwen_selected_coordinates','original_annotated_overview_pixels_used'):
        require(render[key] is False, 'Edited or judgement-contaminated semantic pixels: '+key)
    close = read_sealed(verified(render['close_camera_audit']))
    require(close['schema'] == 'hsi.stage3.actual_motion_semantic_close_camera_audit.v1'
        and close['inputs'] == state['bindings'] and close['sources'] == render['sources']
        and close['frame_policy'] == policy and close['actual_four_frame_mask_policy'] == camera_math.POLICY
        and close['no_occluders_hidden'] is True and close['qwen_selected_coordinates'] is False
        and close['all_target_aabb_and_body_bounds_in_frame'] is True
        and close['minimum_view_azimuth_separation_degrees'] == 45., 'Wrong current close-camera protocol')
    bounds = pictures.close_bounds(state['vertices'][policy['terminal_frame_indices']], context['target_bounds'])
    candidates = camera_math.overview_candidates(bounds)
    require(np.array_equal(bounds,close['bounds_world_zup_m'])
        and [r['camera'] for r in close['numeric_candidates']] == candidates
        and len(close['selected']) == 2, 'Close view is not current whole-body/target numerical fit')
    selected = close['selected']
    require(pictures.angle_distance(selected[0]['camera']['azimuth_degrees'],selected[1]['camera']['azimuth_degrees'])>=45.,
        'Two close views are not distinct')
    for row in selected:
        require(row in close['full_resolution_candidates'] and row['camera'] in candidates
            and [r['frame'] for r in row['frame_masks']] == policy['terminal_frame_indices'], 'Incomplete actual four-frame close audit')
        for mask in row['frame_masks']: mask_check(mask,mask['frame'])
        require(row['summary'] == camera_math.visibility_summary(row['frame_masks']), 'Close camera visibility summary drift')
    expected = [('overview',i,render['overview_camera']) for i in policy['timeline_frame_indices']]
    expected += [('close0',i,selected[0]['camera']) for i in policy['terminal_frame_indices']]
    expected += [('close1',policy['close_frame_index'],selected[1]['camera'])]
    require(len(render['frames']) == len(expected), 'Semantic frame inventory truncated/extended')
    tiles = {'overview':[],'close0':[],'close1':[]}
    for row,(role,index,camera) in zip(render['frames'],expected):
        require(row['role'] == role and row['frame_index'] == index and row['camera_id'] == camera['candidate_id'],
            'Wrong semantic time/role/camera')
        mask_check(row['visibility'],index)
        if role.startswith('close'):
            matching = next(m for m in selected[int(role[-1])]['frame_masks'] if m['frame']==index)
            require(row['visibility'] == matching, 'Saved close image visibility differs from camera audit')
        pixels = image_pixels(row['image'])
        require(pixels.shape == (camera['height'],camera['width'],3), 'Clean frame resolution drift')
        rebuilt = pictures.label_tile(pixels,index)
        require(np.array_equal(np.asarray(rebuilt),image_pixels(row['time_tile'])), 'Time tile includes altered pixels/text')
        tiles[role].append(rebuilt)
    for key,role in (('timeline_montage','overview'),('terminal_montage','close0')):
        record = render[key]
        require(record['source_tiles'] == [r['time_tile'] for r in render['frames'] if r['role']==role],
            'Montage source frames are not the actual policy frames')
        require(np.array_equal(np.asarray(pictures.assemble_montage(tiles[role])),image_pixels(record['image'])),
            'Montage differs from exact clean time tile assembly')
    last = [next(r['image'] for r in render['frames'] if r['role']==role and r['frame_index']==policy['close_frame_index'])
        for role in ('close0','close1')]
    require(render['actual_images'] == [render['original_reference'],render['target_overlay'],
        render['timeline_montage']['image'],*last,render['terminal_montage']['image']], 'Wrong actual six-image semantic request')
    for record in render['actual_images']: verified(record)
    return render, context


def run(render_path, output, *, qwen_client):
    """Call the explicit current Qwen service on six actual motion images."""
    started = time.monotonic()
    sources = source_records()
    render, context = validate_render(render_path)
    output = Path(output).resolve()
    require(not output.exists(), 'Refuse to overwrite actual motion review')
    prompt = contract.motion_semantic_prompt(context['canonical_episode_identity'],context['target_id'])
    response = qwen_client.session(output/'qwen_calls',max_tokens=170,max_images=6).call(
        [verified(row) for row in render['actual_images']],prompt,
        schema=contract.semantic_schema(),call_id='actual_motion_semantics')
    raw = json.loads(response['raw_completion'],object_pairs_hook=_unique_pairs)
    decision = contract.semantic_decision(raw)
    for row in [*sources,artifact(render_path),*render['actual_images']]:verified(row)
    return write_once(output/'receipt.json',dict(schema=contract.REVIEW_SCHEMA,
        source=artifact(__file__),sources=sources,source_contract=artifact(contract.__file__),
        source_render=artifact(render_path),source_original_stage1_sequence=render['source_original_stage1_sequence'],
        source_motion=render['source_motion'],canonical_episode_identity=render['canonical_episode_identity'],
        target_id=render['target_id'],actual_images=render['actual_images'],ordered_checks=list(contract.SEMANTIC_CHECKS),
        generation=response,raw_judgement=raw,decision=decision,source_qwen_call=artifact(response['call_receipt']),
        qwen_received_previous_audits_or_numerical_gate_results=False,qwen_computed_coordinates=False,
        source_frame_and_montage_pixels_verified=True,full_motion_numerical_gates_not_replaced=True,
        semantic_review_alone_grants_no_credit=True,real_positive_credit=0,
        integration_model_execution_tested=False,elapsed_seconds=time.monotonic()-started))


def validate_review(review_path, *, execution_path, evaluation_path, original_sequence, identity):
    """Re-read clean current frames and actual raw response, never caller pass flags.

The caller still owns independent full-motion physical validation/admission.
This method alone cannot register positive experience or publish motion.
"""
    value = read_sealed(review_path)
    require(value['schema'] == contract.REVIEW_SCHEMA and value['source'] == artifact(__file__)
        and value['sources'] == source_records() and value['source_contract'] == artifact(contract.__file__),
        'Motion semantic source/schema drift')
    render, context = validate_render(verified(value['source_render']))
    sequence_record = original_sequence if isinstance(original_sequence,dict) else artifact(original_sequence)
    verified(sequence_record)
    require(render['inputs']['execution'] == artifact(execution_path)
        and render['inputs']['evaluation'] == artifact(evaluation_path)
        and value['source_original_stage1_sequence'] == render['source_original_stage1_sequence'] == sequence_record
        and value['canonical_episode_identity'] == context['canonical_episode_identity'] == identity
        and value['target_id'] == render['target_id']
        and value['source_motion'] == render['source_motion']
        and value['actual_images'] == render['actual_images'], 'Cross-motion/original-task semantic evidence')
    require(value['ordered_checks'] == list(contract.SEMANTIC_CHECKS)
        and value['qwen_received_previous_audits_or_numerical_gate_results'] is False
        and value['qwen_computed_coordinates'] is False
        and value['semantic_review_alone_grants_no_credit'] is True and value['real_positive_credit']==0,
        'Semantic authority or nine-check policy changed')
    call = read_sealed(verified(value['source_qwen_call']))
    require(call['schema']=='p548.qwen_local_semantic_call.v1' and call['status']=='complete'
        and call['call_id']=='actual_motion_semantics' and call['source_code']==artifact(qwen.__file__),
        'Not an actual current Qwen motion request')
    prompt = contract.motion_semantic_prompt(identity,render['target_id'])
    require(call['prompt']==prompt and call['system_prompt']==qwen.SEMANTIC_ONLY+'\n'
        and call['response_format']['json_schema']['schema']==contract.semantic_schema()
        and call['image_evidence']==[dict(label='IMAGE_'+str(i),**row) for i,row in enumerate(render['actual_images'])],
        'Actual Qwen prompt/schema/image order differs')
    runtime=call['runtime']; model=call['model']; result=call['result']; response=call['response']
    require(runtime['inference']=='local_vllm_http_real_model' and runtime['gpt_called'] is False
        and runtime['numeric_geometry_in_prompt'] is False and runtime['local_queue_seconds']==0.,
        'Wrong semantic transport')
    choice=response['choices'][0]
    require(choice['finish_reason']=='stop' and response['model']==model['model_id']==model['served_model_name']
        and result['response_model']==response['model'] and result['model_id']==model['model_id']
        and result['revision']==model['revision'] and result['response_id']==response['id']
        and result['raw_completion']==choice['message']['content'] and result['image_count']==6
        and result['finish_reason']=='stop' and result['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
        and result['system_prompt_sha256']==hashlib.sha256(call['system_prompt'].encode()).hexdigest(),
        'Actual raw Qwen response/model binding differs')
    raw=json.loads(choice['message']['content'],object_pairs_hook=_unique_pairs)
    decision=contract.semantic_decision(raw)
    require(value['generation']==result and value['raw_judgement']==raw and value['decision']==decision,
        'Motion review changed the raw nine-check judgement')
    for record in value['sources']:verified(record)
    return decision
