"""Shared pure contract for actual-motion rendering, blind Qwen and admission.

This module does not read files, call a model, grant credit, or import an
authority/renderer. Callers must first verify every supplied source artifact.
"""
import math

from memory_common import require

SEMANTIC_CHECKS=('target_correct','ordered_actions_complete','terminal_sitting_complete',
    'one_anatomically_plausible_body','feet_grounded_not_tiptoe','legs_naturally_forward',
    'trajectory_semantics_satisfied','scene_layout_preserved','visual_evidence_sufficient')
REVIEW_SCHEMA='p555.actual_motion_semantic_review.v1'
RENDER_SCHEMA='p555.actual_motion_semantic_render.v1'


def canonical_episode(*,scene_id,scene_fingerprint,fixed_geometry_sha256,instruction,
        start_world_xy_m,ordered_actions,body_model_sha256,betas):
    require(isinstance(scene_id,str) and scene_id and isinstance(instruction,str) and instruction.strip(),'Missing episode text')
    for value in (scene_fingerprint,fixed_geometry_sha256,body_model_sha256):
        require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Invalid identity SHA')
    require(isinstance(start_world_xy_m,list) and len(start_world_xy_m)==2
        and all(type(x) in (int,float) and math.isfinite(x) for x in start_world_xy_m),'Invalid current start')
    require(isinstance(betas,list) and len(betas)==10 and all(type(x) in (int,float) and x==0 for x in betas),'Only exact neutral body')
    require(isinstance(ordered_actions,list) and ordered_actions,'Missing ordered actions')
    for action in ordered_actions:
        require(set(action)=={'action','target_ids','reference_ids'} and action['action'] in {'walk','sit'}
            and isinstance(action['target_ids'],list) and len(action['target_ids'])==1
            and isinstance(action['target_ids'][0],str) and action['target_ids'][0]
            and action['reference_ids']==[],'Only single-target walk/sit identity is registered')
    require([x['action'] for x in ordered_actions] in (['sit'],['walk','sit'])
        and len({x['target_ids'][0] for x in ordered_actions})==1,'Unregistered complete task')
    return dict(schema='p555.canonical_episode_identity.v2',scene_id=scene_id,
        scene_fingerprint=scene_fingerprint,fixed_geometry_sha256=fixed_geometry_sha256,
        instruction=instruction,start_world_xy_m=[float(x) for x in start_world_xy_m],
        ordered_actions=[{k:list(v) if isinstance(v,list) else v for k,v in a.items()} for a in ordered_actions],
        body_model_sha256=body_model_sha256,betas=[float(x) for x in betas])


def frame_policy(metadata,packet,frame_count,evaluation=None):
    """Actual emitted indices, not planned contact times or selected best frames."""
    require(type(frame_count) is int and frame_count>=4,'Invalid semantic frame count')
    initial=metadata['query_static_initial_history_frames'];primitives=metadata['primitive_records']
    require(type(initial) is int and initial==2 and frame_count==initial+8*len(primitives),'Expected actual H2/F8 rollout')
    first={}
    for index,row in enumerate(primitives):
        require(type(row['primitive_id']) is int and row['primitive_id']==index,'Primitive order drift')
        active=row['active_node_index']
        require(type(active) is int and 0<=active<len(packet['nodes']),'Invalid actual active node')
        phase=packet['nodes'][active]['phase']
        if phase not in first:first[phase]=initial+8*index
    require('APPROACH' in first and 'CONTACT' in first and first['APPROACH']<=first['CONTACT'],'Unreached/out-of-order approach/contact')
    if evaluation is not None:
        require(evaluation['execution']['frames']==frame_count
            and evaluation['route']['phase_split']['terminal_start_frame_inclusive']==first['CONTACT'],
            'Actual evaluation contact/frame split differs')
    tail=list(range(frame_count-4,frame_count))
    return dict(schema='p555.actual_motion_semantic_frame_policy.v1',
        timeline_frame_indices=sorted({0,first['APPROACH'],first['CONTACT'],*tail}),
        terminal_frame_indices=tail,close_frame_index=frame_count-1,
        first_approach_frame=first['APPROACH'],first_contact_frame=first['CONTACT'],
        source='actual_primitive_active_node_phase_H2_F8')


def canonical_episode_identity(stage1,keynodes,body_identity,scene_fingerprint,original_sequence,raw_plan=None):
    """Complete original task/body identity; excludes stage, RNG and output IDs.

raw_plan must be read from the artifact-bound original semantic compilation,
not a second model call. Source linkage is validated by each I/O-owning caller.
"""
    require(original_sequence is not None,'Actual original Stage1 sequence is required')
    require(original_sequence['scene_id']==stage1['scene_id'] and original_sequence['interaction_count']==1
        and original_sequence['all_source_actions_retained'] is True
        and original_sequence['unsupported_actions']==[],'Incomplete original task')
    if raw_plan is None:
        raw_plan={'steps':original_sequence['verified_raw_ordered_actions']}
    steps=raw_plan['steps'];actions=[{k:s[k] for k in ('action','target_ids','reference_ids')} for s in steps]
    require(body_identity['body_model_sha256']==body_identity['actual_model_sha256'],'Actual body model identity differs')
    return canonical_episode(scene_id=stage1['scene_id'],scene_fingerprint=scene_fingerprint,
        fixed_geometry_sha256=stage1['source_fixed_geometry']['sha256'],instruction=original_sequence['instruction'],
        start_world_xy_m=keynodes['start']['source']['root_xyz_yaw_world_zup'][:2],
        ordered_actions=actions,body_model_sha256=body_identity['actual_model_sha256'],betas=body_identity['betas'])


def motion_semantic_prompt(identity,target_id):
    require(target_id==identity['ordered_actions'][-1]['target_ids'][0],'Semantic target identity drift')
    return ('Judge this ACTUAL generated human motion in the original scene.\n'
        'Original user task: '+identity['instruction']+'\nExact target ID: '+target_id+'\n'
        'IMAGE_0 is the empty original scene reference; IMAGE_1 marks the exact target. '
        'IMAGE_2 is a chronological montage of actual motion frames from start, approach, contact and the final four frames. '
        'IMAGE_3 and IMAGE_4 are two numerical close views of the same actual last frame. '
        'IMAGE_5 shows the final four actual frames from one close camera. '
        'All original furniture and occluders are retained. The orange untextured body is expected. '
        'Check the requested ordered actions and trajectory, actual final sitting on the correct surface, '
        'anatomy, both feet naturally grounded, legs forward through the seat opening, and preserved scene. '
        'Never infer an unseen foot or contact is correct. No prior judgement or numeric gate results are supplied. '
        'Do not calculate or output coordinates, distances or angles. '
        'Return JSON with checks (exactly nine booleans in this order): '+', '.join(SEMANTIC_CHECKS)+
        '; confidence (0 to 1); reason (at most 100 characters).')


def semantic_schema():
    return dict(type='object',additionalProperties=False,properties={
        'checks':dict(type='array',items={'type':'boolean'},minItems=9,maxItems=9),
        'confidence':dict(type='number',minimum=0,maximum=1),
        'reason':dict(type='string',maxLength=100)},required=['checks','confidence','reason'])


def semantic_decision(raw):
    require(isinstance(raw,dict) and set(raw)=={'checks','confidence','reason'}
        and isinstance(raw['checks'],list) and len(raw['checks'])==9
        and all(type(v) is bool for v in raw['checks']),'Exact nine semantic booleans required')
    score=raw['confidence']
    require(type(score) in (int,float) and math.isfinite(score) and 0<=score<=1,'Invalid semantic confidence')
    require(isinstance(raw['reason'],str) and len(raw['reason'])<=100,'Invalid semantic reason')
    failed=[name for name,passed in zip(SEMANTIC_CHECKS,raw['checks']) if not passed]
    if score<.85:failed.append('confidence_below_0.85')
    return dict(passed=not failed,failed_checks=failed,checks=dict(zip(SEMANTIC_CHECKS,raw['checks'])),confidence=score)
