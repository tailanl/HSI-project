"""Blind semantic evidence from this query's actual saved full-body motion.

The old annotated overview is used only for its independently validated numeric
camera. None of its pixels, gate labels or judgements are sent to Qwen. This
producer does not grant memory credit or change any motion, pose or scene.
"""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image

from memory_common import artifact, verified, read_json, read_sealed, write_once, require
import visualize_geometry_motion_v3 as binding
import visualize_geometry_motion_v2 as overview
import motion_semantic_contract_v1 as contract
import motion_semantic_render_helpers as pictures

SOURCE = Path(__file__).resolve()
PINS = {
    Path(pictures.__file__): 'e2196f25ffa719405f39b3153e232b8acfdad51cde8a6dc6b008c8f022d401cb',
    Path(binding.__file__): '52145a6fd6109cb8f74b80bf66b4c8a28077c865ded61b7cb96471e2d13880cd',
    Path(overview.__file__): '8c90cbedafd939b4db12a711dea212abd116e95407b00d886b784865fce40cbb',
    Path(contract.__file__): '7b3f2c25a3b9abe444f16d56de51710e78000c0f0c644db9dc58256b3cdf96dc',
}


def sources():
    rows = [artifact(SOURCE)]
    for path, sha in PINS.items():
        row = artifact(path)
        require(row['sha256'] == sha, 'Frozen semantic render dependency drift: '+str(path))
        rows.append(row)
    return rows


def original_identity(state, sequence_path):
    record = artifact(sequence_path)
    sequence = read_sealed(verified(record))
    require(sequence['schema'] == 'p550.stage1_interaction_sequence_execution.v1'
        and sequence['status'] == 'complete_ordered_stage1_sequence'
        and sequence['interaction_count'] == len(sequence['interaction_steps']) == 1
        and sequence['stage2_sequence_handoff_allowed'] is True
        and sequence['all_source_actions_retained'] is True
        and sequence['unsupported_actions'] == [], 'Incomplete original sequence')
    step = sequence['interaction_steps'][0]
    stage1 = read_sealed(verified(state['bindings']['stage1']))
    keynodes = read_sealed(verified(state['bindings']['key_nodes']))
    adapter = read_sealed(verified(state['bindings']['adapter']))
    target = read_sealed(verified(state['bindings']['target']))
    target_id = target['selected_target_instance_id']
    require(step['stage1_execution'] == state['bindings']['stage1'] and step['action'] == 'sit'
        and step['status'] == 'complete' and step['selected_target_id'] == target_id
        and step['requested_target_ids'] == [target_id], 'Original task points to another motion target/query')
    compilation = read_sealed(verified(sequence['semantic_compilation']))
    verified(sequence['semantic_plan'])
    require(compilation['schema'] == 'p550.ordered_interaction_semantic_compilation.v1'
        and compilation['source_actual_qwen_call'] == sequence['semantic_plan']
        and compilation['fixed_geometry'] == sequence['source_fixed_geometry'] == stage1['source_fixed_geometry']
        and compilation['mapping']['all_source_actions_preserved'] is True
        and compilation['unsupported_actions'] == []
        and compilation['mapping']['raw_semantic_plan'] == compilation['raw_plan'], 'Original semantic compilation drift')
    mappings = compilation['mapping']['execution_compilation']
    require(len(mappings) == 1 and mappings[0]['source_step_indices'] == step['source_step_indices']
        == list(range(len(compilation['raw_plan']['steps']))) and mappings[0]['actions_dropped'] == [],
        'Original source actions omitted')
    identity = contract.canonical_episode_identity(stage1, keynodes, adapter['body_identity'],
        adapter['scene_fingerprint'], sequence, compilation['raw_plan'])
    require(identity['start_world_xy_m'] == sequence['initial_start_world_xy_m']
        == step['route_start_world_xy_m'] and identity['ordered_actions'][-1]['target_ids'] == [target_id],
        'Original start/target identity drift')
    return identity, target_id, target['target_bounds_world_zup_m'], record, sequence['semantic_compilation']


def load_inputs(overview_path, sequence_path):
    sources()
    source_overview = artifact(overview_path)
    old = read_sealed(verified(source_overview))
    require(old['schema'] == binding.SCHEMA, 'Only actual Stage3 v2 overview is registered')
    state = binding.load_inputs(verified(old['inputs']['execution']), verified(old['inputs']['evaluation']))
    require(old['inputs'] == state['bindings'], 'Overview is not this actual saved motion')
    for row in old['sources']: verified(row)
    require(old['sources'][0]['sha256'] == PINS[Path(binding.__file__)]
        and old['original_scene_geometry_unchanged'] is True and old['scene_occluders_hidden'] is False
        and old['evaluated_vertices_reused_verbatim'] is True and old['body_reskinned'] is False
        and old['motion_or_root_modified'] is False and old['qwen_camera_selection_used'] is False,
        'Unsupported edited or semantic-selected source overview')
    audit = read_sealed(verified(old['source_camera_audit']))
    require(audit['inputs'] == state['bindings'] and audit['policy'] == overview.POLICY
        and audit['overview_camera_pass'] is True and audit['no_occluders_hidden'] is True
        and audit['qwen_camera_selection_used'] is False, 'Overview camera audit drift')
    selected = [row for row in audit['full_frame_candidates']
        if row['camera']['candidate_id'] == audit['selected_candidate_id']]
    require(len(selected) == 1, 'Ambiguous selected overview camera')
    selected = selected[0]
    require(selected['camera'] == old['overview_camera']
        and [r['frame'] for r in selected['frame_masks']] == list(range(len(state['vertices'])))
        and overview.visibility_summary(selected['frame_masks']) == selected['summary'] == old['overview_visibility']
        and selected['summary']['all_passed'] is True, 'Incomplete original all-frame true SEG evidence')
    bounds = overview.motion_bounds(state['vertices'], state['route'], state['contact'])
    require(np.array_equal(bounds, audit['bounds_world_zup_m'])
        and any(camera == selected['camera'] for camera in overview.overview_candidates(bounds)),
        'Overview camera is not a current numerical candidate')
    identity, target_id, target_bounds, sequence, compilation = original_identity(state, sequence_path)
    metadata = read_json(verified(state['bindings']['source_metadata']))
    policy = contract.frame_policy(metadata, state['packet'], len(state['vertices']), state['evaluation'])
    old_keypose = read_sealed(verified(state['bindings']['keypose_render']))
    for key in ('original_reference', 'target_overlay'): verified(old_keypose[key])
    return dict(state=state, source_overview=source_overview, source_original_stage1_sequence=sequence,
        source_semantic_compilation=compilation, canonical_episode_identity=identity, target_id=target_id,
        target_bounds=target_bounds, frame_policy=policy, overview_camera=selected['camera'],
        original_reference=old_keypose['original_reference'], target_overlay=old_keypose['target_overlay'])


def save_frame(renderer, output, role, camera, index):
    renderer.camera(camera)
    evidence, pixels = renderer.frame(index, masks=True, rgb=True)
    require(evidence['passed'] is True, 'Actual semantic frame has insufficient true visibility')
    raw = output / (role+'_frame_%06d.png' % index)
    Image.fromarray(pixels).convert('RGB').save(raw)
    tile = pictures.label_tile(pixels, index)
    labelled = output / (role+'_frame_%06d_time_tile.png' % index)
    tile.save(labelled)
    return dict(role=role, frame_index=index, camera_id=camera['candidate_id'], image=artifact(raw),
        time_tile=artifact(labelled), visibility=evidence), tile


def render(overview_path, sequence_path, output, *, gpu=5, egl_device=5):
    started = time.monotonic()
    source_records = sources()
    value = load_inputs(overview_path, sequence_path)
    state, policy = value['state'], value['frame_policy']
    output = Path(output).resolve()
    require(not output.exists(), 'Refuse to overwrite actual motion semantic images')
    require(type(gpu) is type(egl_device) is int and min(gpu, egl_device) >= 0, 'Explicit nonnegative GPU/EGL indices required')
    output.mkdir(parents=True)
    renderer = overview._SceneRenderer(state, gpu, egl_device)
    frames, timeline_tiles, tail_tiles, last_views = [], [], [], []
    try:
        close = pictures.select_close_views(renderer, state['vertices'][policy['terminal_frame_indices']],
            value['target_bounds'], policy['terminal_frame_indices'])
        write_once(output/'close_camera_audit.json', dict(schema='p555.actual_motion_semantic_close_camera_audit.v1',
            inputs=state['bindings'], sources=source_records, frame_policy=policy, **close))
        for index in policy['timeline_frame_indices']:
            row, tile = save_frame(renderer, output, 'overview', value['overview_camera'], index)
            frames.append(row); timeline_tiles.append(tile)
        for number, camera_row in enumerate(close['selected']):
            camera = camera_row['camera']
            indices = policy['terminal_frame_indices'] if number == 0 else [policy['close_frame_index']]
            for index in indices:
                row, tile = save_frame(renderer, output, 'close%d' % number, camera, index)
                frames.append(row)
                if number == 0: tail_tiles.append(tile)
                if index == policy['close_frame_index']: last_views.append(row['image'])
    finally:
        renderer.close()
    timeline = output/'02_actual_timeline.png'
    terminal = output/'05_actual_terminal_four.png'
    pictures.assemble_montage(timeline_tiles).save(timeline)
    pictures.assemble_montage(tail_tiles).save(terminal)
    timeline_record = dict(image=artifact(timeline), source_tiles=[r['time_tile'] for r in frames if r['role']=='overview'])
    terminal_record = dict(image=artifact(terminal), source_tiles=[r['time_tile'] for r in frames if r['role']=='close0'])
    actual_images = [value['original_reference'], value['target_overlay'], timeline_record['image'],
        *last_views, terminal_record['image']]
    require(len(actual_images) == 6, 'Exactly six blind image slots are required')
    for rec in [*source_records, *state['bindings'].values(), value['source_overview'],
            value['source_original_stage1_sequence'], value['source_semantic_compilation'], *actual_images]: verified(rec)
    result = dict(schema=contract.RENDER_SCHEMA, sources=source_records, inputs=state['bindings'],
        **{k:v for k,v in value.items() if k not in {'state','target_bounds'}},
        source_motion=state['bindings']['source_motion'], skinned_vertices=state['bindings']['vertices'],
        original_mesh=state['bindings']['mesh'], frame_count=len(state['vertices']), vertex_count=10475, fps=20,
        close_camera_audit=artifact(output/'close_camera_audit.json'), frames=frames,
        timeline_montage=timeline_record, terminal_montage=terminal_record, actual_images=actual_images,
        original_scene_geometry_unchanged=True, scene_occluders_hidden=False,
        actual_body_vertices_changed=False, evaluated_vertices_reused_verbatim=True,
        body_reskinned=False, motion_or_root_modified=False,
        qwen_previous_judgements_or_gate_text_rendered=False, qwen_selected_coordinates=False,
        only_frame_index_and_time_text_added=True, original_annotated_overview_pixels_used=False,
        renderer_grants_no_publication=True, positive_credit=0,
        render_device=renderer.device, elapsed_seconds=time.monotonic()-started)
    write_once(output/'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overview',type=Path,required=True)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=5)
    parser.add_argument('--egl-device',type=int,default=5)
    args = parser.parse_args()
    result = render(args.overview,args.sequence,args.output,gpu=args.gpu,egl_device=args.egl_device)
    print({'scene':result['canonical_episode_identity']['scene_id'],'frames':result['frame_policy'],
        'images':len(result['actual_images']),'seconds':result['elapsed_seconds']},flush=True)
