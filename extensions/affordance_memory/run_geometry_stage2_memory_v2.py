"""Opt-in complete Stage2 Memory v2 child; no Stage3/positive-credit claim."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from memory_common import PROJECT,artifact,verified,read_sealed,write_once,require
import geometry_memory_contract_v2 as contract
import memory_runtime_v2 as memory
import fast_keypose_memory_v2 as provider
import render_geometry_keypose_memory_v2 as renderer
import audit_geometry_keypose_memory_v2 as auditor

HERE=Path(__file__).resolve().parent
SOURCE=Path(__file__).resolve()
PREPARE=PROJECT/'agent9/methods/p550_scene_understanding_decoupled_20260905/code/prepare_keypose_inputs_v3.py'
SCHEMA='p555.geometry_memory_stage2_execution.v2'


def sources():
    require(artifact(PREPARE)['sha256']=='495c95fea09bd019f4cf60bc93e979c808da36acdb5fd43350568f3c38efd7a4','Frozen view/SDF preparation changed')
    return contract.unique_sources([artifact(SOURCE),artifact(PREPARE),artifact(renderer.__file__),artifact(auditor.__file__),
        artifact(HERE/'reserved_lease_gl/sitecustomize.py'),*contract.sources()])


def require_gl_hook(gpu):
    state=getattr(sys,'_p555_reserved_lease_gl_state_v1',None)
    require(type(gpu) is int and gpu>=0 and state is not None,'Launch through the explicit reserved physical EGL wrapper')
    config=state['identity']['config']
    require(config and config['requested_physical_index']==gpu and os.environ.get('P555_PHYSICAL_EGL_DEVICE')==str(gpu)
        and config['trace']==os.environ.get('P555_EGL_TRACE'),'Physical EGL hook/device configuration mismatch')
    require(artifact(HERE/'reserved_lease_gl/sitecustomize.py')['sha256']=='f2bee040c7b4fc678b3b6de41ca595c47c73e2e179ca88500c22fbc3de46de62',
        'Frozen EGL/lease hook source changed')


def run(stage1,facts,memory_store,output,*,original_sequence,gpu=5,steps=100):
    begun=time.monotonic();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    closure=sources();bindings=dict(stage1=artifact(stage1),facts=artifact(facts),body_model=artifact(provider.retained.BODY),
        original_sequence=artifact(original_sequence))
    write_once(output/'launch.json',dict(schema='p555.geometry_memory_stage2_launch.v2',sources=closure,bindings=bindings,
        memory_store=str(Path(memory_store).resolve()),gpu=gpu,optimizer_steps=steps,selected_provider='typed_local_memory_current_geometry_ik_v2',
        original_H3_branch_unchanged=True,stage3_registered=False,real_positive_credit=0))
    timings={};outputs={};status='started';error=None;review=None;proposal=None;store_before=None
    try:
        require_gl_hook(gpu);require(type(steps) is int and 0<=steps<=400,'Invalid optimization steps')
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),PYOPENGL_PLATFORM='egl',EGL_DEVICE_ID='0')
        start=time.monotonic();prep=output/'preparation'
        argv=[sys.executable,str(PREPARE),'--stage1',str(Path(stage1).resolve()),'--output',str(prep),'--gpu',str(gpu)]
        with (output/'preparation.log').open('x') as stream:
            subprocess.run(argv,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
        timings['preparation_subprocess_wall_seconds']=time.monotonic()-start
        outputs['preparation']=artifact(prep/'receipt.json');prepared=read_sealed(verified(outputs['preparation']))
        require(prepared['schema']=='p550.verified_keypose_preparation.v3' and prepared['source_stage1']==bindings['stage1']
            and prepared['status']=='ready','Preparation belongs to another/unready current Stage1')
        start=time.monotonic()
        p=memory.prepare(stage1,facts,bindings['body_model'],verified(prepared['sdf']),memory_store,output/'memory_preparation',
            original_sequence=original_sequence)
        q=memory.query(p,output/'memory_lookup');lookup=read_sealed(q)
        outputs.update(memory_preparation=artifact(p),memory_lookup=artifact(q))
        store_before=memory.MemoryStore(memory_store).recover()
        timings['production_memory_prepare_and_lookup_seconds']=time.monotonic()-start
        start=time.monotonic();engine=provider.FastKeyposeEngine()
        proposal=engine.propose(stage1,verified(prepared['sdf']),output/'proposal',lookup_receipt=q,steps=steps)
        outputs['proposal']=artifact(output/'proposal/receipt.json')
        timings['body_setup_and_geometry_proposal_seconds']=time.monotonic()-start
        start=time.monotonic();numeric=contract.numerical_recheck(verified(outputs['proposal']),output/'numeric_recheck.json')
        outputs['numeric_recheck']=artifact(output/'numeric_recheck.json')
        timings['independent_actual_23_gate_recheck_seconds']=time.monotonic()-start
        start=time.monotonic()
        renderer.run(verified(outputs['proposal']),verified(prepared['view']),output/'render',numeric_recheck=verified(outputs['numeric_recheck']),gpu=gpu)
        outputs['render']=artifact(output/'render/receipt.json')
        contract.validate_render(verified(outputs['render']))
        timings['actual_original_scene_render_and_binding_seconds']=time.monotonic()-start
        if not contract.gates(proposal['gates']):status='rejected_numeric_candidate_no_semantic_publication'
        else:
            start=time.monotonic();review=auditor.run(verified(outputs['render']),output/'qwen_review',wait_for_slot=False)
            timings['actual_qwen_audit_seconds']=time.monotonic()-start
            if review is None:status='qwen_busy_no_queue_not_a_completed_stage2'
            else:
                outputs['qwen_review']=artifact(output/'qwen_review/receipt.json')
                start=time.monotonic();auditor.validate_review(verified(outputs['qwen_review']))
                timings['actual_review_binding_validation_seconds']=time.monotonic()-start
                status='numeric_and_semantic_keypose_candidate_pass' if review['numeric_and_semantic_candidate_pass'] else 'rejected_semantic_candidate'
        # Stage2 diagnostics are never rebranded as a complete-motion episode.
        start=time.monotonic();memory.validate_lookup(q)
        after=memory.MemoryStore(memory_store).recover()
        require(after==store_before,'Stage2 query/proposal mutated production memory')
        write_once(output/'memory_diagnostic.json',dict(schema='p555.stage2_memory_diagnostic.v2',purpose='production',
            lookup=artifact(q),proposal=outputs['proposal'],canonical_episode_identity=lookup['canonical_episode_identity'],
            record_ids_selected=proposal['memory_records_consumed'],status=status,store_before=store_before,store_after=after,
            episode_admission_not_attempted=True,positive_journal_modified=False,stage3_generated=False,real_positive_credit=0))
        outputs['memory_diagnostic']=artifact(output/'memory_diagnostic.json')
        timings['memory_live_recheck_and_diagnostic_seconds']=time.monotonic()-start
        require(sources()==closure,'Stage2 source closure changed')
        for row in [*closure,*bindings.values()]:verified(row)
        frozen=output/'run_geometry_stage2_memory_v2_source.py';shutil.copy2(SOURCE,frozen);outputs['source_snapshot']=artifact(frozen)
    except BaseException as exception:
        error=f'{type(exception).__name__}: {exception}';status='failed_not_publishable';raise
    finally:
        success=bool(error is None and status=='numeric_and_semantic_keypose_candidate_pass'
            and review and review['numeric_and_semantic_candidate_pass'] is True)
        consumed=[] if proposal is None else proposal['memory_records_consumed']
        result=dict(schema=SCHEMA,status=status,error=error,source_launch=artifact(output/'launch.json'),outputs=outputs,timings=timings,
            total_stage2_child_wall_seconds=time.monotonic()-begun,numeric_and_semantic_keypose_candidate_pass=success,
            selected_provider='typed_local_memory_current_geometry_ik_v2_not_H3',production_memory_queried='memory_lookup' in outputs,
            learned_record_selected=bool(consumed),learned_record_ids_selected=consumed,
            learned_ranking_used=bool(proposal and proposal['memory_consumption']['selection']['root_and_facing_used_for_initialization_ranking_only']),
            geometry_prior_applied=proposal is not None,stage1a_or_stage1b_recomputed=False,full_motion_generated=False,
            original_H3_pipeline_modified=False,stage3_handoff_allowed=False,real_positive_credit=0,completion_contract_version=2,
            actual_scene_mask_contract_version=2,original_physical_and_qwen_gates_unchanged=True)
        write_once(output/'receipt.json',result);print({'status':status,'seconds':result['total_stage2_child_wall_seconds'],'timings':timings},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('stage1','facts','original-sequence','memory-store','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5);p.add_argument('--steps',type=int,default=100)
    a=p.parse_args();run(a.stage1,a.facts,a.memory_store,a.output,original_sequence=a.original_sequence,gpu=a.gpu,steps=a.steps)
