"""One bounded target-DiP comparison with a genuinely shared generated prefix.

Both branches correct anatomical target heading and middle-chunk text. Only
the second branch adds the original pose MSE at offset80. The two initial
40-frame predictions are generated once and copied with explicit provenance;
there are FOUR actual model.sample calls, not six independent fresh calls.
Native history is shared exactly; full-sequence SMPL-X fits may differ earlier.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import run_target_pilot as frozen
from run_target_pilot import (AGENT,BASE,LightSDFGuide,GUIDANCE,TargetDiPBackend,
    constructed_relaxed_stand,build_target_conditions,RECEIPTS,planned_text,
    tensor_array,npz_once,load_case,build_static_prefix,decode_history,
    NativeConsistency,BoundScene,make_schedule,evaluate_mesh,bone_samples,
    PARENTS,RealizationConfig,realize_motion)
from hsi.common.artifacts import artifact,verified,require,write_once

SCHEMA='agent9.stage3.corrected_target_keypose_shared_prefix.v1'
VARIANTS=('corrected_sdf','corrected_sdf_late_keypose')
POSE_WEIGHT=15.
POSE_LOSS_DEFINITION='mean_frames(strength_t * mean_joints_and_xyz((J_world_t - fixed_target_J22_world)^2))'
MIDDLE_TEXT='A person walks slowly towards a chair and begins to sit down.'


def anatomical_target_heading(case,frame):
    """Official bug-fixed hip+shoulder forward, never alter the fixed pose."""
    from hml_codec import _motion_modules,RigidCanonicalFrame
    joints=case.target_joints
    require(isinstance(frame,RigidCanonicalFrame) and torch.is_tensor(joints)
        and joints.shape==(22,3) and joints.is_floating_point() and torch.isfinite(joints).all(),
        'One finite fixed Stage2 J22 and the original rigid frame required')
    require(math.isfinite(float(case.terminal_yaw)),'Original fixed terminal yaw must be finite')
    across=(joints[2]-joints[1])+(joints[17]-joints[16])
    direct=torch.cross(joints.new_tensor([0.,0.,1.]),across,dim=-1)
    require(float(direct[:2].norm())>1e-6,'Degenerate horizontal anatomical heading')
    motion=_motion_modules(str(BASE/'vendor/mdm'))[-1]
    local=frame.to_local(joints)
    heading=motion.recover_root_rot_heading_ang(local[None,:,:,None])[:,0,0]
    vector=torch.stack((heading.sin(),torch.zeros_like(heading),heading.cos()),-1)
    rotation=torch.tensor(frame.world_from_local.copy(),dtype=joints.dtype,device=joints.device)
    forward=vector@rotation.T
    angle=torch.atan2(forward[:,1],forward[:,0])
    direct_angle=torch.atan2(direct[1],direct[0])
    delta=torch.atan2(torch.sin(angle-direct_angle),torch.cos(angle-direct_angle))
    require(torch.isfinite(angle).all() and float(delta.abs().max())<2e-5,'Official/direct heading parity failed')
    original=angle.new_tensor(float(case.terminal_yaw))
    correction=torch.atan2(torch.sin(angle-original),torch.cos(angle-original))
    return angle.detach(),dict(kind='fixed_Stage2_J22_official_anatomical_forward_not_rootR_permission',
        official_function='motion_process.recover_root_rot_heading_ang',indices=[2,1,17,16],
        corrected_world_heading_rad=float(angle[0]),original_terminal_yaw_rad=float(case.terminal_yaw),
        wrapped_difference_rad=float(correction[0]),wrapped_difference_degrees=float(correction[0]*180/torch.pi),
        world_up='+Z',official_local_up='+Y',target_pose_root_shape_unchanged=True,
        source='existing_fixed_Stage2_target_J22_not_future_motion',gt_motion_or_times_used=False)


def corrected_endpoint(case,schedule,frame,offset):
    world,original_heading,planned=frozen.endpoint_goal(case,schedule,offset)
    corrected,report=anatomical_target_heading(case,frame)
    heading=None if original_heading is None else corrected
    planned=dict(planned,heading_world_rad=None if heading is None else float(heading[0]),
        anatomical_heading_correction=report,original_target_pose_was_not_changed=True)
    return world,heading,planned


def corrected_text(offset,schedule,frame):
    require(type(offset) is int and offset in (0,40,80),'Three fixed40 chunks only')
    return MIDDLE_TEXT if offset==40 else planned_text(offset,schedule,frame,'anticipatory')


class LateKeyposeSDFGuide(LightSDFGuide):
    """Same four-step SDF protocol, one added original pose-MSE term at offset80."""
    def __init__(self,decode,history,scene,bone_lengths,consistency,*,target_joints,strength,offset):
        require(type(offset) is int and offset==80 and history.shape[-1]==100,
                'Late keypose guidance is restricted to the final offset80 suffix')
        super().__init__(decode,history,scene,bone_lengths,consistency)
        require(target_joints.shape==(22,3) and target_joints.dtype==torch.float32
            and target_joints.device==history.device and torch.isfinite(target_joints).all(),'Fixed finite target J22 required')
        require(strength.shape==(40,) and strength.dtype==torch.float32 and strength.device==history.device
            and torch.isfinite(strength).all() and bool(((strength>=0)&(strength<=1)).all())
            and bool((strength>0).any()),'Original final40 target strengths required')
        self.target_joints=target_joints.detach().clone();self.strength=strength.detach().clone()

    def __call__(self,x0):
        require(torch.is_tensor(x0) and x0.shape==(1,263,1,40) and x0.dtype==torch.float32
            and torch.isfinite(x0).all() and x0.device==self.history.device,'Finite clean40 suffix required')
        original=x0.detach().clone();call=self.calls;self.calls+=1
        with torch.enable_grad():
            candidate=original.clone().requires_grad_(True)
            optimizer=torch.optim.Adam([candidate],lr=.05)
            parents=torch.tensor(PARENTS[1:],device=x0.device)
            for iteration in range(4):
                full=torch.cat((self.history,candidate),-1);decoded=self.decode(full)
                require(decoded.shape==(1,140,22,3) and torch.isfinite(decoded).all(),'Invalid final complete-history decode')
                joints=decoded[:,-40:]
                sdf=self.scene.collision_loss(bone_samples(joints),tolerance=.003)
                lengths=(joints[...,1:,:]-joints.index_select(-2,parents)).norm(dim=-1)
                bone=(lengths-self.bone_lengths.to(lengths)).square().mean()
                agreement=self.consistency(full)
                fk,velocity=agreement['per_frame_fk_error_m2'],agreement['per_frame_velocity_error_m2']
                require(fk.shape==velocity.shape==(1,140),'Consistency must cover entire committed history')
                consistent=fk[:,-40:].mean()+velocity[:,-41:-1].mean()
                trust=(candidate-original).square().mean()
                pose=((joints-self.target_joints).square().mean((-1,-2))*self.strength).mean()
                loss=100.*sdf+80.*bone+30.*consistent+.08*trust+POSE_WEIGHT*pose
                require(loss.numel()==1 and torch.isfinite(loss).all(),'Nonfinite late keypose SDF energy')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                require(candidate.grad is not None and torch.isfinite(candidate.grad).all(),'Missing/nonfinite late guidance gradient')
                grad=float(candidate.grad.norm());optimizer.step()
                with torch.no_grad():candidate.copy_(original+(candidate-original).clamp(-.8,.8))
                self.trace.append(dict(call=call,iteration=iteration,loss=float(loss.detach()),proxy_sdf_m2=float(sdf.detach()),
                    bone_m2=float(bone.detach()),native_consistency_m2=float(consistent.detach()),trust_m2=float(trust.detach()),
                    keypose_world_mse_m2=float(pose.detach()),keypose_loss_definition=POSE_LOSS_DEFINITION,
                    gradient_l2=grad,gradient_finite=True,maximum_normalized_correction=float((candidate.detach()-original).abs().max()),
                    route_gradient_weight=0.,static_keypose_gradient_weight=POSE_WEIGHT,
                    keypose_strength_source='unchanged_make_schedule.target_strength[80:120]',
                    sdf_contains_target_other_floor_and_unknown=True))
        require(torch.isfinite(candidate).all(),'Nonfinite late-guided suffix')
        return candidate.detach()


def guide_for(variant,offset,decode,history,scene,lengths,consistency,case,schedule):
    require(variant in VARIANTS and type(offset) is int and offset in (0,40,80),'Invalid branch/offset')
    require(history.shape[-1]==20+offset,'Guide must receive only complete committed history')
    if variant==VARIANTS[1] and offset==80:
        return LateKeyposeSDFGuide(decode,history,scene,lengths,consistency,
            target_joints=case.target_joints,strength=schedule.target_strength[80:120],offset=80)
    return LightSDFGuide(decode,history,scene,lengths,consistency)


def require_completed_guide(guide,pose_weight):
    require(type(guide.calls) is int and guide.calls==10 and len(guide.trace)==40,
            'All ten clean callbacks with four guidance iterations must complete')
    for index,row in enumerate(guide.trace):
        require(type(row['call']) is int and row['call']==index//4
            and type(row['iteration']) is int and row['iteration']==index%4
            and row['gradient_finite'] is True and row['route_gradient_weight']==0.
            and row['static_keypose_gradient_weight']==pose_weight,
            'Incomplete/mismatched clean guidance trace')
        require(math.isfinite(row['loss']) and math.isfinite(row['maximum_normalized_correction'])
            and 0.<=row['maximum_normalized_correction']<=.800001,'Invalid guidance energy or trust bound')


def generate_chunk(model,case,scene,schedule,frame,decode,consistency,history,variant,seed,offset,folder,kind):
    require(type(offset) is int and offset in (0,40,80) and history.shape==(1,263,1,20+offset),'Only committed history allowed')
    require((kind=='fresh_shared_prefix' and offset in (0,40) and variant==VARIANTS[0])
        or (kind=='fresh_branch_suffix' and offset==80),'Explicit actual fresh-prefix/branch sampling kind required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);number=offset//40
    world,heading,planned=corrected_endpoint(case,schedule,frame,offset)
    target=build_target_conditions(history,model.mean,model.std,frame,world,
        vendor_root=BASE/'vendor/mdm',heading_world=heading)
    require(set(target)=={'target_cond','target_joint_names','is_heading','audit'}
        and target['audit']['target_frame']=='next_suffix_frame','Wrong official goal contract')
    goal_file=folder/f'target_{number:03d}.npz'
    npz_once(goal_file,target_cond=tensor_array(target['target_cond']),is_heading=tensor_array(target['is_heading']))
    target_json=folder/f'target_{number:03d}.json'
    write_once(target_json,dict(schema=SCHEMA+'.target',planned=planned,audit=target['audit'],
        target_joint_names=target['target_joint_names'],model_inputs=artifact(goal_file),
        learned_target_disabled=False,generation_source=dict(kind=kind,original_binding=None)))
    parents=torch.tensor(PARENTS[1:],device=case.initial_joints.device)
    lengths=(case.initial_joints[1:]-case.initial_joints.index_select(0,parents)).norm(dim=-1)
    guide=guide_for(variant,offset,decode,history,scene,lengths,consistency,case,schedule)
    prompt=corrected_text(offset,schedule,frame);tick=time.monotonic()
    suffix=model.sample(history[...,-20:],text=[prompt],seed=seed+number,
        **{k:target[k] for k in frozen.MODEL_GOAL_KEYS},target_uncond=False,denoised_fn=guide)
    require(suffix.shape==(1,263,1,40) and suffix.dtype==torch.float32 and torch.isfinite(suffix).all(),'Invalid native40')
    require_completed_guide(guide,POSE_WEIGHT if isinstance(guide,LateKeyposeSDFGuide) else 0.)
    updated=torch.cat((history,suffix),-1)
    with torch.no_grad():joints=decode(updated)[0,-40:]
    proposal=folder/f'proposal_{number:03d}.npz'
    npz_once(proposal,native_suffix=tensor_array(suffix),generated_world_joints=tensor_array(joints),
        start_frame_0based=np.asarray(offset),committed_frames=np.asarray(40))
    point=next(iter(world.values()))[0];components=2 if 'traj' in world else 3
    row=dict(chunk=number,seed=seed+number,text=prompt,output_start_0based=offset,predicted_frames=40,
        committed_frames=40,discarded_predicted_frames=0,seconds=time.monotonic()-tick,
        generation_source=dict(kind=kind,original_binding=None,actual_model_sample_calls=1),
        proposal=artifact(proposal),target=artifact(target_json),target_model_inputs=artifact(goal_file),
        raw_suffix_endpoint_goal_error_m=float((joints[-1,0,:components]-point[:components]).norm()),
        raw_goal_error_components='world_XY' if components==2 else 'world_XYZ',
        clean_x0_callback_calls=guide.calls,guidance_trace=guide.trace,
        static_keypose_gradient_weight=POSE_WEIGHT if isinstance(guide,LateKeyposeSDFGuide) else 0.)
    print(dict(variant=variant,kind=kind,offset=offset,raw_endpoint_error=row['raw_suffix_endpoint_goal_error_m']),flush=True)
    return updated,row


def copy_shared_chunk(row,folder):
    """Actual shared proposal copies; never represented as new sampling calls."""
    from hsi.common.artifacts import read_sealed
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    result=dict(row);originals={k:row[k] for k in ('proposal','target','target_model_inputs')}
    for key in ('proposal','target_model_inputs'):
        destination=folder/Path(row[key]['path']).name
        require(not destination.exists(),'Do not overwrite shared proposal copy')
        shutil.copy2(verified(row[key]),destination)
        result[key]=artifact(destination)
        require(result[key]['sha256']==row[key]['sha256'],'Shared copy bytes changed')
    target=read_sealed(verified(row['target']));target=dict(target)
    target['model_inputs']=result['target_model_inputs']
    target['generation_source']=dict(kind='shared_prefix_reused',original_binding=originals['target'])
    destination=folder/Path(row['target']['path']).name
    write_once(destination,target);result['target']=artifact(destination)
    result['seconds']=0.
    result['generation_source']=dict(kind='shared_prefix_reused',original_binding=originals,actual_model_sample_calls=0,
        generation_seconds_accounted_in_shared_prefix_only=True)
    return result


def enforce_locks(motion,case,anchors):
    require(motion.shape==(120,135) and torch.isfinite(motion).all(),'Full finite realized motion required')
    require(torch.equal(motion[0],case.initial_pose),'Realization changed shared initial pose')
    require(anchors==list(range(112,120)) and torch.equal(motion[anchors],case.target_pose.expand(8,135)),
        'Realization changed fixed full135 event poses')


def source_records():
    return [artifact(Path(__file__)),*frozen.source_records()]


def run(output,*,scene_id='scene006',seed=390917,device='cpu'):
    require(scene_id=='scene006' and type(seed) is int and seed==390917,'This is one fixed Scene006/seed390917 experiment')
    output=Path(output).resolve();require(not output.exists(),'Use a new experiment directory')
    sources=source_records();output.mkdir(parents=True,exist_ok=False);snapshot=output/'source_snapshot';snapshot.mkdir()
    for index,record in enumerate(sources):shutil.copy2(record['path'],snapshot/f'{index:02d}_{Path(record["path"]).name}')
    write_once(output/'request.json',dict(schema=SCHEMA+'.request',sources=sources,scene_id=scene_id,seed=seed,
        variants=list(VARIANTS),frames=120,fps=20.,commit_frames=40,cfg=2.5,device=device,
        shared_prefix_frames=80,total_actual_model_sample_calls=4,independent_whole_sequence_samples=False,
        base_sdf_guidance=GUIDANCE,late_keypose_weight=POSE_WEIGHT,late_keypose_loss_definition=POSE_LOSS_DEFINITION,
        late_strength_source='unchanged_make_schedule.target_strength[80:120]',middle_text=MIDDLE_TEXT,
        common_anatomical_heading_correction=True,post_mesh_sdf_in_both=True,
        new_training=False,physics_simulation=False))
    started=time.monotonic()
    try:
        model=TargetDiPBackend(device=device,guidance_scale=2.5)
        case0=load_case(AGENT/'runs/stage12_ablation_20260909/fresh_chain/full'/RECEIPTS[scene_id],device=device)
        case,initial_report=constructed_relaxed_stand(case0,source_is_adapter_constructed=True)
        scene=BoundScene.from_case(case)
        prefix,frame,codec=build_static_prefix(case.initial_joints,model.mean,model.std,vendor_root=BASE/'vendor/mdm')
        prefix=prefix.to(device=device,dtype=torch.float32)
        decode=lambda x:decode_history(x,model.mean,model.std,frame,vendor_root=BASE/'vendor/mdm')
        consistency=NativeConsistency(case.initial_joints,model.mean,model.std,frame,BASE/'vendor/mdm')
        schedule=make_schedule(case,120,20.,approach_transition=True)
        _,heading_report=anatomical_target_heading(case,frame)
        npz_once(output/'initial_state.npz',motion=tensor_array(case.initial_pose),vertices_world=tensor_array(case.initial_vertices),
            joints_world=tensor_array(case.initial_joints),betas=tensor_array(case.betas),
            original_constructed_tpose=tensor_array(case0.initial_pose),native_prefix=tensor_array(prefix))
        write_once(output/'conditions.json',dict(schema=SCHEMA+'.conditions',provenance=case.provenance,permissions=case.permissions,
            initial_adaptation=initial_report,initial_state=artifact(output/'initial_state.npz'),codec=codec,schedule=schedule.description,
            anatomical_heading=heading_report,instruction=case.instruction,target_id=case.target_id,target_class=case.target_class,
            surface_id=case.surface_id,betas=tensor_array(case.betas).tolist(),sdf=case.sdf,
            same_relaxed_initial_all_variants=True,no_future_gt_or_input_event_timestamps=True,
            only_branch_difference='offset80 additive fixed-keypose MSE; common heading/text corrections and all other conditions fixed'))
        shared=output/'shared_prefix';shared.mkdir();history=prefix.clone();shared_rows=[];tick=time.monotonic()
        for offset in (0,40):
            history,row=generate_chunk(model,case,scene,schedule,frame,decode,consistency,history,VARIANTS[0],seed,offset,
                shared/'proposals','fresh_shared_prefix')
            shared_rows.append(row)
        shared_seconds=time.monotonic()-tick
        npz_once(shared/'native.npz',native_features=tensor_array(history),world_joints=tensor_array(decode(history)[0,20:]),
            initial_history_frames=np.asarray(20),fps=np.asarray(20.))
        shared_receipt=write_once(shared/'execution.json',dict(schema=SCHEMA+'.shared',chunks=shared_rows,
            native=artifact(shared/'native.npz'),actual_model_sample_calls=2,generated_frames=80,
            initial_prefix_frames=20,seconds=shared_seconds,model=model.metadata))
        shared_history=history.detach().clone();results={};branch_natives=[]
        for variant in VARIANTS:
            folder=output/variant;folder.mkdir();branch_started=time.monotonic()
            chunks=[copy_shared_chunk(row,folder/'proposals') for row in shared_rows]
            native,last=generate_chunk(model,case,scene,schedule,frame,decode,consistency,shared_history.clone(),variant,seed,80,
                folder/'proposals','fresh_branch_suffix')
            chunks.append(last)
            require(torch.equal(native[...,:100],shared_history),'Branch rewrote shared80 or initial20')
            branch_natives.append(native.detach())
            with torch.no_grad():joints=decode(native)[0,20:]
            npz_once(folder/'native.npz',native_features=tensor_array(native),world_joints=tensor_array(joints),
                initial_history_frames=np.asarray(20),fps=np.asarray(20.),world_from_local=frame.world_from_local,
                world_translation=frame.translation,interpretation=np.asarray('shared80 plus fresh branch40, before mesh fitting'))
            motion,mesh,fit=realize_motion(joints,case,RealizationConfig(steps=80,fps=20.),scene_energy=scene.mesh_energy,
                anchor_frames=schedule.anchor_frames)
            enforce_locks(motion,case,schedule.anchor_frames)
            metrics=evaluate_mesh(mesh,case,scene,fps=20.,anchor_frames=schedule.anchor_frames)
            metrics.update(raw_native_final_root_error_m=float((joints[-1,0]-case.target_joints[0]).norm()),
                raw_native_final_keypose_mean_joint_error_m=float((joints[-1]-case.target_joints).norm(dim=-1).mean()),
                raw_metrics_are_before_any_terminal_hard_lock=True,final_keypose_enforced_in_kinematic_fitting=True,
                shared_prefix_generation_seconds=shared_seconds,branch_suffix_sampling_seconds=last['seconds'],
                mesh_realization_seconds=fit['elapsed_seconds'],physical_motion_accepted=False)
            route=torch.cat((case.route_xy,case.route_xy.new_zeros(len(case.route_xy),1)),-1)
            npz_once(folder/'sample.npz',motion=tensor_array(motion),vertices_world=tensor_array(mesh.vertices[0]),
                joints_world=tensor_array(mesh.joints[0]),faces=tensor_array(case.faces),betas=tensor_array(case.betas),
                fps=np.asarray(20.),coordinate_system=np.asarray('world_zup_m'),route_world=tensor_array(route),
                keypose_vertices_world=tensor_array(case.target_vertices[None]),keypose_joints_world=tensor_array(case.target_joints[None]),
                keypose_frames_1based=np.asarray([i+1 for i in schedule.anchor_frames],dtype=np.int64))
            write_once(folder/'sampling.json',dict(schema=SCHEMA+'.sampling',variant=variant,chunks=chunks,model=model.metadata,
                native=artifact(folder/'native.npz'),shared_prefix=artifact(shared/'execution.json'),learned_goal_enabled=True,
                target_uncond=False,actual_model_sample_calls_in_branch=1,shared_model_calls_accounted_once=2,
                initial20_and_generated80_reused_bit_exact=True,base_sdf_guidance=GUIDANCE,
                late_keypose_weight=POSE_WEIGHT if variant==VARIANTS[1] else 0.,late_keypose_loss_definition=POSE_LOSS_DEFINITION,
                no_route_gradient=True,new_training=False,commit_frames=40,discarded_future_frames=0,
                branch_suffix_sampling_seconds=last['seconds'],shared_prefix_generation_seconds=shared_seconds))
            write_once(folder/'mesh_realization.json',fit);write_once(folder/'metrics.json',metrics)
            results[variant]=dict(sample=artifact(folder/'sample.npz'),native=artifact(folder/'native.npz'),
                sampling=artifact(folder/'sampling.json'),mesh_realization=artifact(folder/'mesh_realization.json'),
                metrics_file=artifact(folder/'metrics.json'),metrics=metrics,elapsed_seconds=time.monotonic()-branch_started,
                final_keypose_hard_locked=True,post_mesh_full_surface_sdf=True,motion_release_authorized=False)
            write_once(folder/'execution.json',dict(schema=SCHEMA+'.variant',status='completed_diagnostic_not_quality_acceptance',
                scene_id=scene_id,seed=seed,variant=variant,sample=results[variant],conditions=artifact(output/'conditions.json'),
                shared_prefix=artifact(shared/'execution.json'),scene_mesh=case.scene_mesh,sources=sources))
        require(torch.equal(branch_natives[0][...,:100],branch_natives[1][...,:100]),'Native shared prefix differs across branches')
        for record in sources:verified(record)
        for key in ('checkpoint','configuration','bert','mean','std'):verified(model.metadata[key])
        result=dict(schema=SCHEMA,status='completed_diagnostic_not_quality_acceptance',scene_id=scene_id,seed=seed,samples=results,
            conditions=artifact(output/'conditions.json'),shared_prefix=artifact(shared/'execution.json'),scene_mesh=case.scene_mesh,
            backend=model.metadata,sources=sources,frames=120,fps=20.,commit_frames=40,cfg=2.5,
            total_actual_model_sample_calls=4,shared_prefix_generated_frames=80,shared_prefix_native_bit_exact=True,
            shared_prefix_generation_seconds=shared_seconds,shared_native_does_not_imply_same_early_realized_mesh=True,
            final_keypose_anchor_frames_0based=list(schedule.anchor_frames),physics_simulation=False,new_training=False,
            no_future_gt_or_input_event_timestamps=True,motion_release_authorized=False,elapsed_seconds=time.monotonic()-started)
        write_once(output/'execution.json',result);return result
    except BaseException as error:
        write_once(output/'run_failure.json',dict(schema=SCHEMA,status='failed_incomplete_pilot',sources=sources,
            error_type=type(error).__name__,reason=str(error),motion_release_authorized=False));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--device',default='cpu')
    args=parser.parse_args(argv);torch.set_num_threads(4)
    result=run(args.output,device=args.device)
    print(dict(output=str(args.output.resolve()),status=result['status']),flush=True)


if __name__=='__main__':main()
