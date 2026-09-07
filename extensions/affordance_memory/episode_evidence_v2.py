"""Fixed production authority for actual P555 v2 motion bundles.

No caller success booleans, callback registry or inherited v1 upgrade. Existing
failed motions receive zero credit. Missing final semantics/extraction remains
unknown, while a complete future registered bundle has a real success branch.
"""
from __future__ import annotations
import inspect
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np
from memory_common import PROJECT, artifact, digest, read_json, read_sealed, require, verified
from extract_local_experience_v2 import canonical_episode
from motion_semantic_contract_v1 import (SEMANTIC_CHECKS, frame_policy,
    canonical_episode_identity, motion_semantic_prompt, semantic_schema, semantic_decision)

HERE=Path(__file__).resolve().parent
REQUEST_SCHEMA='p555.admission_request.v2'
BUNDLE_SCHEMA='p555.online_episode_bundle.v2'
SEMANTIC_SCHEMA='p555.actual_motion_semantic_review.v1'
RENDER_SCHEMA='p555.actual_motion_semantic_render.v1'
PINS={
    'current_motion_evidence.py':'eeff8333eaa93f0560cebaa5737655f3521f4e5307f5409b8c090ee4a88b6c14',
    'visualize_stage3_guidance_comparison.py':'aa3417532d7bc104284e7c48660808574233584b475bfef13fe08ec4f2288d93',
    'compile_geometry_stage3_v2.py':'4313f72760bd6ee3d2bcb2b934f194750f5605b5e5eeb1c1729dd1be4c60dfb4',
    'fast_keypose.py':'1809c04b02ebcbc4c93213e859d2d82455101e1bba0d41007ce8886126165883',
    'motion_semantic_contract_v1.py':'7b3f2c25a3b9abe444f16d56de51710e78000c0f0c644db9dc58256b3cdf96dc',
}
# Reviewed frozen kernels; no runtime setter or caller-supplied hash.
SEMANTIC_SOURCE_PINS={
    'render_current_motion_semantics_v1.py':'3dc5ead58d1e2d93300b43d67ea5e0bb9be87ad73525630f7c0d252123a373ca',
    'audit_current_motion_semantics_v1.py':'ab719f647a3fef788dbaf84ff794589c9232ebd540848f0de70ac80e35109411',
    'motion_semantic_render_helpers.py':'e2196f25ffa719405f39b3153e232b8acfdad51cde8a6dc6b008c8f022d401cb',
}


class MissingRegisteredEvidence(ValueError):
    pass


def canonical_identity(stage1, keynodes, body_identity, scene_fingerprint, original_sequence=None):
    """Shared complete identity; partial fallback is negative-only, never credit."""
    if original_sequence is not None:
        return canonical_episode_identity(stage1,keynodes,body_identity,scene_fingerprint,original_sequence)
    steps=stage1['interaction_plan']['steps']
    actions=[{k:s[k] for k in ('action','target_ids','reference_ids')} for s in steps]
    return canonical_episode(scene_id=stage1['scene_id'],scene_fingerprint=scene_fingerprint,
        fixed_geometry_sha256=stage1['source_fixed_geometry']['sha256'],instruction=stage1['instruction'],
        start_world_xy_m=keynodes['start']['source']['root_xyz_yaw_world_zup'][:2],
        ordered_actions=actions,body_model_sha256=body_identity['actual_model_sha256'],betas=body_identity['betas'])


def _sources():
    result=[]
    for name,sha in PINS.items():
        row=artifact(HERE/name);require(row['sha256']==sha,'Registered source changed: '+name);result.append(row)
    return result+[artifact(__file__),artifact(HERE/'extract_local_experience_v2.py')]


def _tail_v2(reader,adapter,motion,meshes):
    import current_motion_evidence as tail
    scope=dict(vars(tail)); source=inspect.getsource(tail.tail_geometry_diagnostics)
    # Each actual archived proposal has its own path even when its bytes match.
    # Retained loader checks path identity; do not reuse an earlier case module.
    scope['module']=lambda name,path:tail.module(name+'_'+digest(str(Path(path).resolve()))[:16],path)
    replacements={
        '"d029b04dfef12ddb6284969d059c29583f2274c22d64ae0556457ec0b32f4dab"':repr(PINS['fast_keypose.py']),
        'HERE/"compile_geometry_stage3.py"':'HERE/"compile_geometry_stage3_v2.py"',
    }
    for old,new in replacements.items():
        require(source.count(old)==1,'Expected exact terminal adaptation: '+old);source=source.replace(old,new)
    require(source.count('compiler.THRESHOLDS')==4 and source.count('compiler.NUMERIC_GATES')==1,
        'Unexpected v1 numeric constant references')
    source=source.replace('compiler.THRESHOLDS','compiler.retained.THRESHOLDS').replace('compiler.NUMERIC_GATES','compiler.retained.NUMERIC_GATES')
    exec(compile(source,__file__,'exec'),scope)
    return scope['tail_geometry_diagnostics'](reader,adapter,motion,meshes)


def tail_release(tail,frame_count):
    from episode_evidence import PHYSICAL_GATES,POSTURE_GATES
    expected=PHYSICAL_GATES|POSTURE_GATES|{'hips_support_inside_actual_triangles'}
    require(type(frame_count) is int and frame_count>=4
        and tail['actual_source_frame_indices']==list(range(frame_count-4,frame_count))
        and all(type(i) is int for i in tail['actual_source_frame_indices']),'Wrong actual terminal frames')
    rows=tail['frames'];require(len(rows)==4,'Incomplete tail')
    failed=[]
    for index,row in zip(tail['actual_source_frame_indices'],rows):
        require(row['frame_index']==index,'Tail frame mismatch');gates=row['stage2_diagnostic_23_gates']
        require(set(gates)==expected and all(type(v) is bool for v in gates.values()),'Tail diagnostic gate schema drift')
        failed += [{'frame_index':index,'gate':k} for k,v in gates.items() if k!='contact_anchor_locked' and not v]
    return dict(schema='p555.additional_motion_tail_release.v1',failed=failed,passed=not failed,
        original_34_gates_unchanged=True,static_candidate_23_still_required=True,
        construction_only_diagnostic_not_dynamic_release=['contact_anchor_locked'],
        measured_anchor_triangle_height_band_m=[0.,.04],actual_frames=tail['actual_source_frame_indices'])


def _original(reader,binding,state):
    if binding is None:return None
    sequence=reader.read(binding,schema='p550.stage1_interaction_sequence_execution.v1')
    require(sequence['status']=='complete_ordered_stage1_sequence' and sequence['interaction_count']==1
        and sequence['unsupported_actions']==[] and sequence['all_source_actions_retained'] is True,'Unsupported original full task')
    step=sequence['interaction_steps'][0]
    require(step['stage1_execution']==state['adapter']['inputs']['stage1_execution'],'Original task did not produce this Stage1')
    compilation=reader.read(sequence['semantic_compilation'])
    raw=reader.read(compilation['source_actual_qwen_call'])
    require(raw['parsed']==compilation['raw_plan'],'Original raw plan binding drift')
    return {**sequence,'verified_raw_ordered_actions':compilation['raw_plan']['steps']}


def verify_motion(execution_path,evaluation_path):
    """Recompute actual full-frame metrics plus actual terminal meshes on CPU."""
    import current_motion_evidence as numeric
    import visualize_stage3_guidance_comparison as comparison
    class Reader(numeric.Reader):
        def path(self,record):return self.check(record)
        def json(self,record,*,sealed=True):return self.read(record,sealed=sealed)
    reader=Reader()
    for row in _sources():reader.check(row)
    evaluation=read_sealed(evaluation_path);gate_function=comparison.retained_gate_function(reader)
    state=comparison.load_case(execution_path,evaluation_path,evaluation['affordance'],reader,gate_function)
    motion=numeric.load_arrays(reader.check(evaluation['source_motion']))
    mesh_path=reader.check(evaluation['skinned_vertices']);meshes=numeric.load_arrays(mesh_path,mesh=True)
    require(len(motion['joints'])==len(meshes['vertices_world']),'Incomplete actual full-mesh frames')
    old=numeric.module('p555_v2_original_metrics',numeric.KERNEL)
    evaluator=numeric.module('p555_v2_original_fullmesh_evaluator',numeric.EVALUATOR)
    inputs=state['adapter']['inputs'];reader.check(inputs['raw_occupancy'])
    expected=evaluator.numeric_evaluation(old,motion['joints'],state['packet'],state['metadata'],
        Path(inputs['raw_occupancy']['path']),mesh_path,np.asarray(state['adapter']['surface_contract']['fixed_generated_butt_vertex_indices']))
    for key,value in expected.items():require(digest(value)==digest(evaluation[key]),'Actual full-motion metrics mismatch: '+key)
    gates=gate_function(evaluation);require(gates==evaluation['release_gates'] and len(gates)==34,'Original gates changed')
    tail=_tail_v2(reader,state['adapter'],motion,meshes);extra=tail_release(tail,len(motion['joints']))
    candidate=reader.read(state['adapter']['numeric_recheck'])
    candidate_gates=candidate.get('gates',candidate.get('numeric_gates'))
    require(isinstance(candidate_gates,dict) and len(candidate_gates)==23
        and all(type(x) is bool for x in candidate_gates.values()),'Missing actual static 23 recheck')
    state.update(motion=motion,meshes=meshes,tail=tail,tail_release=extra,
        original_34_release_gates=gates,static_candidate_23=candidate_gates)
    reader.finish()
    return state,reader


def _admission(source,identity,reader,classification,reason,*,checks=None,records=(),extractions=()):
    from memory_store_v2 import require_authority_sources
    require_authority_sources()
    from episode_evidence import _family
    reader.finish();episode=digest(identity)
    return dict(schema='p555.episode_admission.v1',purpose='production',fixture_success_allowed=False,
        query_id=episode,episode_id=episode,attempt_id=digest({'episode':episode,'source_sha256':source['sha256']}),
        scene_id=identity['scene_id'],scene_fingerprint=identity['scene_fingerprint'],scene_family=_family(identity['scene_id']),
        classification=classification,failure_domain='stage3_motion_validation' if classification=='failure' else None,
        real_positive_credit_allowed=classification=='success',critical_memory_failure=False,affected_record_ids=[],
        allowed_record_ids=[r['record_id'] for r in records] if classification=='success' else [],
        record_extraction_receipts=list(extractions) if classification=='success' else [],source_bindings=list(reader.bindings.values()),
        checks=checks or {},reason=reason,canonical_episode_identity=identity,authority='episode_evidence_v2')


def admit_episode(request_path,*,purpose='production'):
    from memory_store_v2 import require_authority_sources
    require_authority_sources()
    require(purpose in ('production','test_fixture'),'Unknown purpose')
    request=read_json(request_path)
    if 'receipt_payload_sha256' in request:read_sealed(request_path)
    require(set(request)-{'receipt_payload_sha256'}=={'schema','mode','source_receipt'},'Unexpected request fields')
    if request['mode']=='test_fixture':
        require(purpose=='test_fixture','Production forbids fixture evidence')
        from episode_evidence import admit_episode as fixture_authority
        require(request['schema']=='p555.admission_request.v1','Fixture uses isolated existing v1 ABI')
        return fixture_authority(request_path,purpose='test_fixture')
    require(purpose=='production' and request['schema']==REQUEST_SCHEMA
        and request['mode']=='online_complete_motion','Unregistered v2 production request')
    source=request['source_receipt'];bundle=read_sealed(verified(source))
    require(Path(source['path']).is_relative_to(PROJECT/'agent9/runs'),'Production bundle must be in agent9/runs')
    require(bundle['schema']==BUNDLE_SCHEMA and set(bundle)-{'receipt_payload_sha256'}=={
        'schema','purpose','execution','evaluation','original_stage1_sequence','facts','semantic_evidence','record_extraction_receipts'}
        and bundle['purpose']=='production','Malformed actual episode bundle')
    require(not any('fixture' in part.lower() or part.lower()=='tests' for part in Path(source['path']).parts),'Fixture namespace is not production')
    state,reader=verify_motion(verified(bundle['execution']),verified(bundle['evaluation']))
    reader.check(source);adapter=state['adapter']
    stage1=reader.read(adapter['inputs']['stage1_execution']);keynodes=reader.read(adapter['inputs']['key_nodes'])
    original=_original(reader,bundle['original_stage1_sequence'],state)
    identity=canonical_identity(stage1,keynodes,adapter['body_identity'],adapter['scene_fingerprint'],original)
    failed34=[name for name,row in state['original_34_release_gates'].items() if row['passed'] is not True]
    checks=dict(original_34_failed=failed34,actual_fullmesh_frames=len(state['motion']['joints']),vertices_per_frame=10475,
        original_34_recomputed_from_actual_arrays=True,static_candidate_23=state['static_candidate_23'],
        additional_actual_tail_release=state['tail_release'],network_pre_and_post_bound=True,
        canonical_episode_excludes_stage_seed_output_response_ids=True)
    if failed34 or not all(state['static_candidate_23'].values()) or not state['tail_release']['passed']:
        return _admission(source,identity,reader,'failure','Actual complete motion fails unchanged physical gates',checks=checks)
    if original is None or bundle['facts'] is None or bundle['semantic_evidence'] is None:
        return _admission(source,identity,reader,'unknown','Missing original task, current facts or actual final-motion semantic evidence',checks=checks)
    # A fixed strict semantic verifier is linked below; no all-true caller flag.
    try:semantic=verify_semantics(reader,bundle,state,identity)
    except MissingRegisteredEvidence as error:
        return _admission(source,identity,reader,'unknown',str(error),checks=checks)
    checks['actual_motion_semantics']=semantic
    if semantic['passed'] is not True:
        return _admission(source,identity,reader,'failure','Actual final-motion task semantics failed',checks=checks)
    records,extraction=extract_verified_records(reader,bundle,state,identity)
    if not bundle['record_extraction_receipts']:
        return _admission(source,identity,reader,'unknown','Successful motion has no exact verified local extraction receipts',checks=checks)
    from extract_local_experience_v2 import check_receipt
    require(len(bundle['record_extraction_receipts'])==len(records),'Incomplete/extra local extraction')
    for binding,record in zip(bundle['record_extraction_receipts'],records):
        reader.check(binding)
        check_receipt(binding,episode_id=digest(identity),record=record,source_episode=bundle['execution'],
            source_geometry=bundle['facts'],extraction=extraction)
    # Store independently checks every allowed ID and re-admits at CAS commit.
    return _admission(source,identity,reader,'success','Complete actual motion and measured typed-local extraction verified',
        checks=checks,records=records,extractions=bundle['record_extraction_receipts'])


def verify_semantics(reader,bundle,state,identity):
    required={'render_current_motion_semantics_v1.py','audit_current_motion_semantics_v1.py','motion_semantic_render_helpers.py'}
    if set(SEMANTIC_SOURCE_PINS)!=required:
        raise MissingRegisteredEvidence('Actual motion semantic renderer/auditor source registration not yet finalized')
    pins=[]
    for name,expected in SEMANTIC_SOURCE_PINS.items():
        rec=artifact(HERE/name);require(rec['sha256']==expected,'Registered actual-motion semantic source drift')
        reader.check(rec);pins.append(rec)
    review=reader.read(bundle['semantic_evidence'],schema=SEMANTIC_SCHEMA)
    render=reader.read(review['source_render'],schema=RENDER_SCHEMA)
    from neutral_seed import module
    auditor=module('p555_registered_actual_motion_semantics_v1',HERE/'audit_current_motion_semantics_v1.py')
    audited_render,context=auditor.validate_render(reader.check(review['source_render']))
    require(audited_render==render and context['canonical_episode_identity']==identity,
        'Independent clean image/camera/montage/original-query recheck differs')
    require(review['source_contract']==artifact(HERE/'motion_semantic_contract_v1.py'),'Review used a different semantic contract')
    review_source=reader.check(review['source'])
    require(artifact(review_source)['sha256']==SEMANTIC_SOURCE_PINS['audit_current_motion_semantics_v1.py'],'Unregistered actual audit source')
    require(any(r['sha256']==SEMANTIC_SOURCE_PINS['render_current_motion_semantics_v1.py']
        and Path(r['path']).name=='render_current_motion_semantics_v1.py' for r in render['sources']),
        'Missing actual renderer source')
    for rec in render['sources']:reader.check(rec)
    inputs=state['adapter']['inputs'];ev=state['evaluation']
    for obj in (review,render):
        require(obj['source_motion']==ev['source_motion']
            and obj['source_original_stage1_sequence']==bundle['original_stage1_sequence']
            and obj['canonical_episode_identity']==identity
            and obj['target_id']==state['adapter']['surface_contract']['target_instance_id'],'Motion/original query/target drift')
    require(review['ordered_checks']==list(SEMANTIC_CHECKS),'Semantic gate order drift')
    require(render['skinned_vertices']==ev['skinned_vertices'] and render['original_mesh']==inputs['original_scene_mesh'],
        'Semantic render does not use the actual fullmesh/raw scene')
    overview=reader.read(render['source_overview'],schema='p555.actual_geometry_motion_overview_visualization.v3')
    for key,expected in [('execution',state['source_execution']),('evaluation',state['source_evaluation'])]:
        require(overview['inputs'][key]==expected and render['inputs'][key]==expected,'Cross-motion overview/render binding')
    require(overview['inputs']==render['inputs'],'Renderer changed bound overview inputs')
    require(overview['overview_camera']==render['overview_camera'],'Wrong reused numeric overview camera')
    for key in ('original_scene_geometry_unchanged','evaluated_vertices_reused_verbatim'):
        require(render[key] is True,'Semantic renderer changed actual scene/body')
    for key in ('scene_occluders_hidden','body_reskinned','motion_or_root_modified','qwen_previous_judgements_or_gate_text_rendered'):
        require(render[key] is False,'Semantic evidence is altered/occluder-hidden/or verdict-contaminated')
    policy=frame_policy(state['metadata'],state['packet'],len(state['motion']['joints']),ev)
    require(render['frame_policy']==policy,'Semantic sampling is not actual deterministic phase/tail selection')
    frames=render['frames'];require(isinstance(frames,list) and frames,'Missing clean actual frames')
    observed=set()
    for row in frames:
        require(type(row['frame_index']) is int and row['frame_index'] in policy['timeline_frame_indices']
            and type(row['camera_id']) is int and 0<=row['camera_id']<32,'Wrong actual clean frame/camera')
        reader.check(row['image']);observed.add(row['frame_index'])
    require(observed==set(policy['timeline_frame_indices']),'Incomplete actual trajectory/tail frames')
    images=render['actual_images']
    require(isinstance(images,list) and len(images)==6 and review['actual_images']==images,'Actual Qwen must receive exactly these six bound images')
    require(images[:2]==[render['original_reference'],render['target_overlay']]
        and images[2]==render['timeline_montage']['image'] and images[5]==render['terminal_montage']['image'],
        'Wrong original reference/target/timeline/tail montage order')
    # Original reference/overlay come from the exact current Stage2 owner/view.
    old_review=reader.read(inputs['qwen_review']);old_render=reader.read(old_review['source_render'])
    require(images[:2]==[old_render['original_reference'],old_render['target_overlay']],'Semantic target images differ from current owner')
    available=[row['image'] for row in frames]
    require(images[3] in available and images[4] in available and images[3]!=images[4],'Two actual last-frame close views are missing')
    for image in images:
        path=reader.check(image)
        with path.open('rb') as stream:require(stream.read(8)==b'\x89PNG\r\n\x1a\n','Semantic image is not PNG')
    for role,montage,expected_indices in (('overview',render['timeline_montage'],policy['timeline_frame_indices']),
                                      ('close0',render['terminal_montage'],policy['terminal_frame_indices'])):
        tiles=montage['source_tiles']
        original_rows=[r for r in frames if r['role']==role]
        require([r['frame_index'] for r in original_rows]==expected_indices
            and tiles==[r['time_tile'] for r in original_rows],'Montage source tiles do not match actual deterministic frames')
        for tile in tiles:reader.check(tile)
    reader.check(render['close_camera_audit'])
    call=reader.read(review['source_qwen_call'],schema='p548.qwen_local_semantic_call.v1',sealed=False)
    require(call['status']=='complete' and call['call_id']=='p555_actual_motion_semantics', 'Not a fresh actual-motion Qwen call')
    require(call['prompt']==motion_semantic_prompt(identity,render['target_id']),'Qwen did not receive exact original-task prompt')
    require(call['response_format']['json_schema']['schema']==semantic_schema(),'Wrong exact semantic schema')
    require(call['image_evidence']==[{'label':'IMAGE_'+str(i),**image} for i,image in enumerate(images)],'Actual Qwen image ordering/hash drift')
    runtime=call['runtime']
    require(runtime['inference']=='local_vllm_http_real_model' and runtime['gpt_called'] is False
        and runtime['numeric_geometry_in_prompt'] is False and runtime['frozen'] is True,'Wrong actual semantic model runtime')
    require(call['model']['model_id']=='Qwen/Qwen3.8-27B-FP8'
        and call['model']['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','Unregistered Qwen model')
    for name,sha in {'config':'74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc',
                     'weight_index':'f0838c766951bdfe76d6afbdb2771a8f67aaa2231dedb3d33cebd817729843a2'}.items():
        require(call['model'][name]['sha256']==sha,'Qwen model metadata drift');reader.check(call['model'][name])
    result=call['result'];content=call['response']['choices'][0]['message']['content']
    require(result['raw_completion']==content and result['response_id']==call['response']['id']
        and result['image_count']==6 and result['finish_reason']=='stop'
        and result['prompt_sha256']==hashlib.sha256(call['prompt'].encode()).hexdigest(),'Actual Qwen response drift')
    from memory_common import _unique_pairs
    raw=json.loads(content,object_pairs_hook=_unique_pairs)
    decision=semantic_decision(raw)
    require(review['raw_judgement']==raw and review['decision']==decision,'Review changed raw model judgement')
    require(review['generation']==result,'Review generation is not the actual Qwen call result')
    return decision


def extract_verified_records(reader,bundle,state,identity):
    from memory_runtime import _current
    from extract_local_experience_v2 import derive_records
    inputs=state['adapter']['inputs']
    current=_current(reader.check(inputs['stage1_execution']),reader.check(bundle['facts']),inputs['body_model'],
        reader.check(inputs['sdf_cache']),'stage2_contact','neutral')
    from fast_keypose import load_current
    geometry=load_current(reader.check(inputs['stage1_execution']))
    return derive_records(current['context'],state['motion'],state['tail'],geometry['triangles'])
