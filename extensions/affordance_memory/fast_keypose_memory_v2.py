"""Opt-in typed local-experience initialization of the frozen geometric IK.

No historical pose/motion is read. The original analytic pose, optimizer/loss,
contact lock and all 23 numerical gates are retained verbatim. Admitted local
contact hypotheses are bounded against the current triangle candidate set;
their root/facing offsets only rank initial states AFTER physical violations.
This new provider is not registered for old memory=False Stage3 handoff.
"""
import inspect
import copy
import math
from pathlib import Path
import textwrap
import types
import numpy as np
from memory_common import artifact,digest,require
import fast_keypose as retained
import memory_runtime_v2 as runtime

RETAINED_SHA256='1809c04b02ebcbc4c93213e859d2d82455101e1bba0d41007ce8886126165883'
POLICY={'maximum_contact_distance_from_current_candidate_m':.08,
    'maximum_projected_to_actual_triangle_height_difference_m':.025,
    'maximum_facing_error_rad':math.pi/36,
    'maximum_root_offset_m':1.0,'root_distance_ranking_scale_m':.1}


def bounded_candidates(base,rows,triangles,yaw):
    """Pure final current-triangle/direction veto; never authorizes a record."""
    require(base,'Missing original current hypotheses');base=[np.asarray(x,dtype=float) for x in base]
    selected=[];rejected=[];forward=np.array([math.cos(yaw),math.sin(yaw)])
    for row in rows:
        p=row['projection'];contact=np.asarray(p['contact_world_xyz_m'],dtype=float)
        root=np.asarray(p['root_world_xyz_m'],dtype=float);facing=np.asarray(p['facing_world_xy'],dtype=float)
        require(contact.shape==root.shape==(3,) and facing.shape==(2,)
            and np.isfinite(np.r_[contact,root,facing]).all(),'Invalid projected coordinates')
        height,valid=retained.triangle_heights(contact[None,:2],triangles)
        distance=min(float(np.linalg.norm(contact[:2]-b[:2])) for b in base)
        error=abs(math.atan2(forward[0]*facing[1]-forward[1]*facing[0],float(forward@facing)))
        checks=dict(actual_triangle=bool(valid[0]),bounded_local_contact=distance<=POLICY['maximum_contact_distance_from_current_candidate_m'],
            actual_height=bool(valid[0] and abs(contact[2]-height[0])<=POLICY['maximum_projected_to_actual_triangle_height_difference_m']),
            current_direction=error<=POLICY['maximum_facing_error_rad']+1e-12,
            metric_root_offset=float(np.linalg.norm(root-contact))<=POLICY['maximum_root_offset_m'])
        if not all(checks.values()):
            rejected.append(dict(record_id=row['record_id'],checks=checks));continue
        candidate=np.r_[contact[:2],height[0]]
        selected.append(dict(record_id=row['record_id'],projection=p,candidate=candidate.tolist(),
            projection_sha256=digest(p),checks=checks,actual_triangle_height_correction_m=float(height[0]-contact[2])))
    # Stable lookup order; original hypotheses remain available unchanged.
    return [np.asarray(r['candidate']) for r in selected]+base,selected,rejected


def initialization_rank(original_rank,state,contact,yaw,selected):
    """Only a tie-break after original physical violation rank, before quality."""
    if not selected:return original_rank
    point=contact.detach().numpy();root=state.root.detach().numpy()
    distances=[]
    for row in selected:
        p=row['projection'];facing=np.asarray(p['facing_world_xy'])
        error=abs(math.atan2(math.sin(yaw-math.atan2(facing[1],facing[0])),math.cos(yaw-math.atan2(facing[1],facing[0]))))
        distances.append(float(np.linalg.norm(point-np.asarray(row['candidate']))) / .08
            +float(np.linalg.norm(root-np.asarray(p['root_world_xyz_m']))) / POLICY['root_distance_ranking_scale_m']+error)
    return (*original_rank[:-1],min(distances),original_rank[-1])


def selection(contact,state,yaw,selected):
    point=contact.detach().numpy();root=state.root.detach().numpy()
    used=[r for r in selected if np.allclose(point,r['candidate'],atol=1e-6,rtol=0)]
    return dict(records_selected=[r['record_id'] for r in used],
        initial_contact_world_xyz_m=point.tolist(),initial_root_world_xyz_m=root.tolist(),initial_facing_yaw_rad=yaw,
        projected_root_errors_m=[float(np.linalg.norm(root-np.asarray(r['projection']['root_world_xyz_m']))) for r in used],
        root_and_facing_used_for_initialization_ranking_only=bool(selected),
        original_analytic_articulation_unchanged=True,arbitrary_root_translation_used=False)


def _compiled(hooks):
    """Exact narrow seam; original objective/evaluate/optimizer remain verbatim."""
    require(artifact(retained.__file__)['sha256']==RETAINED_SHA256,'Frozen geometric provider changed')
    source=textwrap.dedent(inspect.getsource(retained.FastKeyposeEngine.propose))
    replacements={
        'contact_policy)\n':'contact_policy)\n    candidates = memory_candidates(candidates, current, yaw)\n',
        '_, values, gates, rank = evaluate(state, contact)\n':'_, values, gates, rank = evaluate(state, contact)\n            rank = memory_rank(rank, state, contact, yaw)\n',
        '_, contact, initial, initial_values, initial_gates = best_seed\n':'_, contact, initial, initial_values, initial_gates = best_seed\n    memory_selection = memory_selected(contact, initial, yaw)\n',
        'output.mkdir(parents=True)':'memory_final_revalidate()\n    output.mkdir(parents=True)',
        'p555.current_geometry_ik_candidate.v1':'p555.current_geometry_ik_candidate.v2',
        'p555.geometry_guided_keypose_proposal.v1':'p555.geometry_guided_keypose_proposal.v2',
        '"declared_analytic_sit_seed_plus_current_geometry_ik"':'"declared_analytic_sit_seed_plus_typed_local_memory_current_geometry_ik_v2"',
        '"memory_records_consumed": []':'"memory_records_consumed": memory_selection["records_selected"], "memory_consumption": memory_receipt(memory_selection)',
    }
    for old,new in replacements.items():
        require(source.count(old)==1,'Unexpected frozen seam: '+old);source=source.replace(old,new)
    scope={**vars(retained),**hooks,'__file__':__file__}
    exec(compile(source,__file__,'exec'),scope)
    scope['propose'].__adapted_source__=source
    return scope['propose']


class FastKeyposeEngine(retained.FastKeyposeEngine):
    def propose(self,stage1_path,sdf_path,output,*,lookup_receipt,steps=100,contact_policy='front_support',use_sdf=True,purpose='production'):
        lookup_binding=artifact(lookup_receipt);lookup,prepared,store=runtime.validate_lookup(lookup_receipt,purpose=purpose)
        require(prepared['current']['current_stage1_execution']==artifact(stage1_path)
            and prepared['current']['sdf_receipt']==artifact(sdf_path)
            and prepared['current']['body_model']==self.model_binding,'Lookup/provider actual current inputs mismatch')
        source=artifact(__file__);closure=runtime.sources();state={}
        def candidates(base,current,yaw):
            context=prepared['current']['context']
            require(current['target']['target_instance_id']==context['owner']['instance_id']
                and current['binding']['direction_hypothesis_id']==context['direction_id'],'Current owner/direction changed')
            result,selected,rejected=bounded_candidates(base,lookup['retrieval']['records'],current['triangles'],yaw)
            # The exact triangle may differ slightly from a raster height.
            # Recheck the snapped contact AND translated suggested root against
            # the actual current SDF instead of recycling the pre-snap verdict.
            point_filter=runtime.retained.CurrentSDFPointFilter(prepared['current'])
            safe=[]
            for row in selected:
                p=copy.deepcopy(row['projection']);delta=np.array(row['candidate'])-np.array(p['contact_world_xyz_m'])
                p['contact_world_xyz_m']=row['candidate']
                p['root_world_xyz_m']=(np.array(p['root_world_xyz_m'])+delta).tolist()
                evidence=point_filter(p)
                if evidence['accepted'] is True:
                    safe.append({**row,'triangle_snapped_point_sdf_evidence':evidence})
                else:rejected.append(dict(record_id=row['record_id'],reason='actual_triangle_snapped_current_sdf_veto',evidence=evidence))
            selected=safe;result=[np.array(r['candidate']) for r in selected]+base
            state.update(selected=selected,rejected=rejected,original_candidate_count=len(base),actual_candidate_count=len(result))
            return result
        def final_revalidate():
            runtime.validate_lookup(lookup_receipt,purpose=purpose)
            require(artifact(lookup_receipt)==lookup_binding and artifact(__file__)==source and runtime.sources()==closure,
                'Experience/source changed before saving candidate')
        def receipt(chosen):
            return dict(schema='p555.typed_local_memory_consumption.v2',purpose=purpose,source=source,
                retained_provider=artifact(retained.__file__),source_codes=closure,lookup=lookup_binding,
                canonical_episode_identity=prepared['canonical_episode_identity'],episode_id=prepared['episode_id'],
                accepted_projections=state['selected'],rejected_projections=state['rejected'],selection=chosen,
                original_candidate_count=state['original_candidate_count'],actual_candidate_count=state['actual_candidate_count'],
                m0_bypass=not state['selected'],fixture_not_production_learning=purpose=='test_fixture',
                real_records_selected=len(chosen['records_selected']) if purpose=='production' else 0,
                real_positive_credit_granted=0,policy=POLICY,live_barrier_revalidated_before_save=True,
                original_23_numeric_gates_unchanged=True,original_optimizer_and_loss_unchanged=True,
                qwen_review_still_required=True,stage3_handoff_registered=False)
        function=_compiled(dict(memory_candidates=candidates,
            memory_rank=lambda rank,body,contact,yaw:initialization_rank(rank,body,contact,yaw,state['selected']),
            memory_selected=lambda contact,body,yaw:selection(contact,body,yaw,state['selected']),
            memory_final_revalidate=final_revalidate,memory_receipt=receipt))
        return function(self,stage1_path,sdf_path,output,steps=steps,contact_policy=contact_policy,use_sdf=use_sdf)
