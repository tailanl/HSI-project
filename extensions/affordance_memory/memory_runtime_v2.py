"""Versioned live retrieval of admitted local experience; cold M0 is ordinary.

Preparation/query grant no credit. Production accepts only the fixed v2 store;
isolated test_fixture stores exercise mechanics, never production evidence.
Only current typed Stage2 sit is registered. SDF samples veto two points, not
the full body. A consumer must call validate_lookup again immediately before use.
"""
from pathlib import Path
import time
from memory_common import artifact,digest,read_json,read_sealed,require,verified,write_once
import memory_runtime as retained
from memory_retrieval import retrieve_current,FRAME_CONVENTION
from memory_store_v2 import MemoryStore,require_authority_sources
from motion_semantic_contract_v1 import canonical_episode
from episode_evidence import _family

HERE=Path(__file__).resolve().parent
PREPARATION='p555.memory_runtime_preparation.v2'
LOOKUP='p555.memory_runtime_lookup.v2'


def sources():
    require_authority_sources()
    return [artifact(__file__),*retained._sources(),*[artifact(HERE/name) for name in
        ('memory_store_v2.py','episode_evidence_v2.py','extract_local_experience_v2.py','motion_semantic_contract_v1.py')]]


def _purpose(purpose,store_root,output):
    require(purpose in ('production','test_fixture'),'Unknown runtime purpose')
    if purpose=='test_fixture':
        for path in (store_root,output):require('test_fixture' in Path(path).resolve().parts,
            'Fixture runtime requires explicitly isolated test_fixture paths')


def original_identity(current,sequence_path):
    binding,sequence=retained._production_source(sequence_path)
    stage1=read_sealed(verified(current['current_stage1_execution']))
    require(sequence['schema']=='p550.stage1_interaction_sequence_execution.v1'
        and sequence['status']=='complete_ordered_stage1_sequence' and sequence['interaction_count']==1
        and sequence['stage2_sequence_handoff_allowed'] is True and sequence['all_source_actions_retained'] is True
        and sequence['unsupported_actions']==[] and sequence['scene_id']==stage1['scene_id'],'Incomplete original task')
    step=sequence['interaction_steps'][0];target=current['context']['owner']['instance_id']
    require(step['stage1_execution']==current['current_stage1_execution'] and step['action']=='sit'
        and step['status']=='complete' and step['selected_target_id']==target
        and step['requested_target_ids']==[target],'Original/current target mismatch')
    compilation=read_sealed(verified(sequence['semantic_compilation']))
    raw=read_sealed(verified(sequence['semantic_plan']))
    require(compilation['schema']=='p550.ordered_interaction_semantic_compilation.v1'
        and compilation['source_actual_qwen_call']==sequence['semantic_plan']
        and raw['parsed']==compilation['raw_plan']==compilation['mapping']['raw_semantic_plan']
        and compilation['fixed_geometry']==sequence['source_fixed_geometry']==stage1['source_fixed_geometry']
        and compilation['mapping']['all_source_actions_preserved'] is True and compilation['unsupported_actions']==[],
        'Original semantic compilation mismatch')
    mapping=compilation['mapping']['execution_compilation'];steps=compilation['raw_plan']['steps']
    require(len(mapping)==1 and mapping[0]['source_step_indices']==step['source_step_indices']==list(range(len(steps)))
        and mapping[0]['actions_dropped']==[],'Original actions dropped')
    start=current['identity']['start_world_xy_m']
    require(start==sequence['initial_start_world_xy_m']==step['route_start_world_xy_m'],'Original start mismatch')
    identity=canonical_episode(scene_id=stage1['scene_id'],scene_fingerprint=current['context']['scene_fingerprint_sha256'],
        fixed_geometry_sha256=current['context']['fixed_geometry']['sha256'],instruction=sequence['instruction'],
        start_world_xy_m=start,ordered_actions=[{k:r[k] for k in ('action','target_ids','reference_ids')} for r in steps],
        body_model_sha256=current['body_model']['sha256'],betas=[0.]*10)
    require(identity['ordered_actions'][-1]['target_ids']==[target],'Original terminal owner mismatch')
    return identity,binding


def prepare(current_stage1_execution,facts_receipt,body_model,sdf_receipt,store_root,output,*,
            original_sequence,stage='stage2_contact',body_shape_bin='neutral',purpose='production'):
    started=time.monotonic();_purpose(purpose,store_root,output);closure=sources()
    current=retained._current(current_stage1_execution,facts_receipt,body_model,sdf_receipt,stage,body_shape_bin)
    identity,sequence=original_identity(current,original_sequence);key=digest(identity)
    store=MemoryStore(store_root,purpose=purpose);state=store.recover()
    cycle=store.open_cycle(dict(query_id=key,episode_identity=key,scene_id=identity['scene_id'],
        scene_fingerprint_sha256=identity['scene_fingerprint'],scene_family=_family(identity['scene_id']),
        canonical_episode_identity=identity))
    require(sources()==closure,'Source changed during preparation')
    result=Path(output)/'receipt.json'
    write_once(result,dict(schema=PREPARATION,purpose=purpose,current=current,original_sequence=sequence,
        canonical_episode_identity=identity,episode_id=key,cycle=cycle,store_root=str(Path(store_root).resolve()),
        store_state=state,stage=stage,body_shape_bin=body_shape_bin,source_codes=closure,
        credit_granted_by_runtime=0,seed_output_and_stage_excluded_from_episode_identity=True,
        elapsed_seconds=time.monotonic()-started))
    return result


def validate_preparation(path,*,purpose='production'):
    value=read_sealed(path);require(value['schema']==PREPARATION and value['purpose']==purpose,'Runtime purpose/schema mismatch')
    _purpose(purpose,value['store_root'],path);require(value['source_codes']==sources(),'Runtime source drift')
    old=value['current'];current=retained._current(verified(old['current_stage1_execution']),verified(old['facts']),
        old['body_model'],verified(old['sdf_receipt']),value['stage'],value['body_shape_bin'])
    identity,sequence=original_identity(current,verified(value['original_sequence']))
    require(current==old and sequence==value['original_sequence'] and identity==value['canonical_episode_identity']
        and digest(identity)==value['episode_id'],'Current facts/body/original identity drift')
    cycle=value['cycle'];require(cycle['query_binding']==dict(query_id=value['episode_id'],episode_identity=value['episode_id'],
        scene_id=identity['scene_id'],scene_fingerprint_sha256=identity['scene_fingerprint'],
        scene_family=_family(identity['scene_id']),canonical_episode_identity=identity),'Cycle identity drift')
    store=MemoryStore(value['store_root'],purpose=purpose)
    store._cycle(cycle)
    return value,store


def _retrieve(prepared,store,scopes):
    callback=retained.CurrentSDFPointFilter(prepared['current'])
    result=retrieve_current(store,prepared['cycle'],prepared['current']['context'],sdf_check=callback,scopes=scopes)
    require(result['purpose']==store.purpose,'Cross-purpose retrieval')
    return result,callback


def query(preparation_path,output,*,scopes=('scene_local','invariant'),purpose='production'):
    started=time.monotonic();binding=artifact(preparation_path);prepared,store=validate_preparation(preparation_path,purpose=purpose)
    result,callback=_retrieve(prepared,store,scopes);current=prepared['current'];count=len(result['records'])
    require(artifact(preparation_path)==binding and sources()==prepared['source_codes'],'Lookup source changed')
    _purpose(purpose,prepared['store_root'],output);destination=Path(output)/'receipt.json'
    write_once(destination,dict(schema=LOOKUP,purpose=purpose,preparation=binding,cycle=prepared['cycle'],
        canonical_episode_identity=prepared['canonical_episode_identity'],episode_id=prepared['episode_id'],
        retrieval=result,experience_count=count,m0_bypass=not count,credit_granted_by_runtime=0,
        real_experience_count=count if purpose=='production' else 0,
        provenance={'cache':dict(kind='verified_precomputed_scene_facts',artifact=current['facts'],learned_credit=0),
            'geometry_prior':dict(kind='current_typed_metric_surface',descriptor=current['context']['descriptor'],
                frame_convention=FRAME_CONVENTION,applied_to_keypose=False,learned_credit=0),
            'learned_experience':dict(kind='fixed_v2_full_motion_authority' if purpose=='production' else 'isolated_test_fixture_only',
                count=count,applied_to_keypose=False,reason=result['m0_reason'])},
        current_sdf_filter=dict(source=current['sdf_receipt'],callback_count=callback.calls,
            scope='contact_and_root_points_not_full_body',full_body_validated=False),
        consumption_requires_live_revalidation=True,original_23_and_qwen_gates_still_required=True,
        stage3_handoff_registered=False,elapsed_seconds=time.monotonic()-started))
    return destination


def validate_lookup(path,*,purpose='production'):
    """Re-run active-only fixed-snapshot query and live quarantine/current veto."""
    binding=artifact(path);lookup=read_sealed(path)
    require(lookup['schema']==LOOKUP and lookup['purpose']==purpose,'Lookup schema/purpose mismatch')
    prepared,store=validate_preparation(verified(lookup['preparation']),purpose=purpose)
    require(lookup['cycle']==prepared['cycle'] and lookup['episode_id']==prepared['episode_id']
        and lookup['canonical_episode_identity']==prepared['canonical_episode_identity'],'Lookup identity mismatch')
    result,_=_retrieve(prepared,store,lookup['retrieval']['scope_filter'])
    require(result==lookup['retrieval'],'Live quarantine, current geometry or retrieval changed since lookup')
    count=len(result['records']);require(lookup['experience_count']==count and lookup['m0_bypass'] is (count==0)
        and lookup['real_experience_count']==(count if purpose=='production' else 0)
        and lookup['credit_granted_by_runtime']==0,'Lookup count/credit tampering')
    require(artifact(path)==binding,'Lookup changed during validation')
    return lookup,prepared,store
