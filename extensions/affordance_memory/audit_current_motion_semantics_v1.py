"""Source-bound blind Qwen review of actual motion; no coordinates or credit.

validate_render is a pure CPU reader shared by the fixed Memory v2 authority.
It reconstructs every time tile and montage from bound clean frame PNGs. It
does not re-render 3-D pixels, load model weights, or trust caller pass flags.
"""
import argparse
import fcntl
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

from memory_common import PROJECT, artifact, verified, read_json, read_sealed, write_once, require
import render_current_motion_semantics_v1 as renderer
import motion_semantic_contract_v1 as contract
import motion_semantic_render_helpers as pictures
import visualize_geometry_motion_v2 as camera_math

SOURCE = Path(__file__).resolve()
P548 = PROJECT/'agent9/methods/p548_qwen38_full_pipeline_20260905/code'
LOCK_PATH = PROJECT/'agent9/runs/p550_scene_understanding_decoupled_20260905/runtime/qwen_batch_and_stage2.lock'
PINS = {
    Path(renderer.__file__): '3dc5ead58d1e2d93300b43d67ea5e0bb9be87ad73525630f7c0d252123a373ca',
    Path(pictures.__file__): 'e2196f25ffa719405f39b3153e232b8acfdad51cde8a6dc6b008c8f022d401cb',
    Path(contract.__file__): '7b3f2c25a3b9abe444f16d56de51710e78000c0f0c644db9dc58256b3cdf96dc',
    P548/'qwen_api.py': '416021a370aefea00ace569fc22c5f40f7e59c6c4c5d16cba151ebeecff10a8b',
    P548/'common.py': 'f76b8acfd586d4f6e2ef3af633d223e28b34bdcb380c2aff11b1626b957014ab',
}


def source_records():
    rows = [artifact(SOURCE)]
    for path, sha in PINS.items():
        row = artifact(path)
        require(row['sha256'] == sha, 'Actual semantic auditor dependency drift: '+str(path))
        rows.append(row)
    return rows


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
    context = renderer.load_inputs(verified(render['source_overview']), verified(render['source_original_stage1_sequence']))
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
    require(close['schema'] == 'p555.actual_motion_semantic_close_camera_audit.v1'
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


def run(render_path, output, *, wait_for_slot=False):
    started = time.monotonic()
    sources = source_records()
    source_render = artifact(render_path)
    render, context = validate_render(render_path)
    validation_seconds = time.monotonic()-started
    prompt = contract.motion_semantic_prompt(context['canonical_episode_identity'],context['target_id'])
    images = [verified(row) for row in render['actual_images']]
    output = Path(output).resolve()
    require(not output.exists(), 'Refuse to overwrite actual motion Qwen review')
    # Match the existing shared-service lease, without changing its service or
    # background jobs. Default returns busy immediately, not an inference queue.
    LOCK_PATH.parent.mkdir(parents=True,exist_ok=True)
    before_reservation = time.monotonic()
    with LOCK_PATH.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX | (0 if wait_for_slot else fcntl.LOCK_NB))
        except BlockingIOError:
            print({'status':'qwen_busy_no_queue','inference_started':False},flush=True)
            return None
        try:
            reserved = time.monotonic()
            if str(P548) not in sys.path: sys.path.insert(0,str(P548))
            qwen_api = importlib.import_module('qwen_api')
            require(Path(qwen_api.__file__).resolve() == P548/'qwen_api.py', 'Foreign Qwen adapter import')
            require(qwen_api.Qwen.__init__.__globals__['CONFIG']['model_id']=='Qwen/Qwen3.8-27B-FP8'
                and qwen_api.CONFIG['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
                'Unexpected Qwen model/revision')
            client = qwen_api.Qwen(output/'qwen_calls',max_tokens=170)
            response = client.call(images,prompt,schema=contract.semantic_schema(),call_id='p555_actual_motion_semantics')
            raw = json.loads(response['raw_completion'])
            decision = contract.semantic_decision(raw)
        finally:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    for row in [*sources,source_render,*render['actual_images']]: verified(row)
    frozen = output/'audit_current_motion_semantics_source.py'
    shutil.copy2(SOURCE,frozen)
    result = dict(schema=contract.REVIEW_SCHEMA,source=artifact(frozen),sources=sources,
        source_contract=artifact(contract.__file__),source_render=source_render,
        source_original_stage1_sequence=render['source_original_stage1_sequence'],source_motion=render['source_motion'],
        canonical_episode_identity=render['canonical_episode_identity'],target_id=render['target_id'],
        actual_images=render['actual_images'],ordered_checks=list(contract.SEMANTIC_CHECKS),
        generation=response,raw_judgement=raw,decision=decision,source_qwen_call=artifact(response['call_receipt']),
        qwen_received_previous_audits_or_numerical_gate_results=False,qwen_computed_coordinates=False,
        source_frame_and_montage_pixels_verified=True,full_motion_numerical_gates_not_replaced=True,
        semantic_review_alone_grants_no_credit=True,real_positive_credit=0,
        render_validation_cpu_seconds=validation_seconds,
        reservation_seconds_before_audit_timer=reserved-before_reservation,
        audit_elapsed_seconds_excluding_reservation=time.monotonic()-reserved,
        total_elapsed_seconds_including_validation_and_reservation=time.monotonic()-started)
    write_once(output/'receipt.json',result)
    print({'scene':render['canonical_episode_identity']['scene_id'],'decision':decision,
        'seconds':result['audit_elapsed_seconds_excluding_reservation']},flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-for-slot',action='store_true')
    args=parser.parse_args()
    run(args.render,args.output,wait_for_slot=args.wait_for_slot)
