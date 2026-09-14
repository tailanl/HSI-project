"""Fresh same-checkpoint multi-target DiP, 120 frames / three complete suffixes.

target_off masks the learned goals of THIS target checkpoint; it is NOT the
old no-target model. target_sdf adds a bounded clean-x0 J22/bone SDF proxy and
post-generation complete-SMPL-X SDF fitting: a two-level combination, not an
isolated x0 ablation. Raw outputs and hard-keypose fitting remain separate.
No training, GT history, Qwen coordinates, physics or quality release.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from generation_conditions import AGENT, constructed_relaxed_stand
from hsi.common.artifacts import artifact, verified, require, write_once
from target_backend import TargetDiPBackend, BASE
from target_coordinates import build_target_conditions
from run_pilot import RECEIPTS, planned_text, tensor_array, npz_once
from stage12_conditions import load_case
from hml_codec import build_static_prefix, decode_history
from native_consistency import NativeConsistency
from sdf_guidance import BoundScene, make_schedule, evaluate_mesh, bone_samples, PARENTS
from mesh_realization import RealizationConfig, realize_motion

SCHEMA = 'agent9.stage3.multi_target_pilot.v1'
VARIANTS = ('target_off', 'target_on', 'target_sdf')
FRAMES, FPS, COMMIT = 120, 20., 40
MODEL_GOAL_KEYS = ('target_cond', 'target_joint_names', 'is_heading')
GUIDANCE = dict(iterations=4, learning_rate=.05, trust_radius=.8,
    sdf_weight=100., bone_weight=80., native_consistency_weight=30., trust_weight=.08,
    route_gradient_weight=0., static_keypose_gradient_weight=0., temporal_correction_weight=0.,
    clean_x0_proxy_is_full_mesh=False, full_history_decode=True)


class LightSDFGuide:
    """Only the current clean 40-frame suffix changes; full past stays fixed."""
    def __init__(self, decode, history, scene, bone_lengths, consistency):
        require(torch.is_tensor(history) and history.shape[:3] == (1,263,1)
                and history.shape[-1] in (20,60,100) and history.dtype == torch.float32 and torch.isfinite(history).all(),
                'Pass complete initial20 plus every committed40 history')
        require(callable(decode) and callable(consistency) and scene is not None,
                'Explicit current scene/decode/native consistency required')
        require(bone_lengths.shape == (21,) and torch.isfinite(bone_lengths).all()
                and (bone_lengths > 0).all(), 'Actual initial-body 21 bone lengths required')
        self.history = history.detach().clone()
        self.decode, self.scene, self.consistency = decode, scene, consistency
        self.bone_lengths = bone_lengths.detach().clone()
        self.trace, self.calls = [], 0

    def __call__(self, x0):
        require(torch.is_tensor(x0) and x0.shape == (1,263,1,40) and x0.dtype == torch.float32
                and torch.isfinite(x0).all() and x0.device == self.history.device, 'Finite clean40 suffix required')
        original = x0.detach().clone(); call = self.calls; self.calls += 1
        with torch.enable_grad():
            candidate = original.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([candidate], lr=GUIDANCE['learning_rate'])
            parents = torch.tensor(PARENTS[1:],device=x0.device)
            for iteration in range(GUIDANCE['iterations']):
                full = torch.cat((self.history,candidate),-1)
                decoded = self.decode(full)
                require(decoded.shape == (1,full.shape[-1],22,3) and torch.isfinite(decoded).all(), 'Invalid full-history decode')
                joints = decoded[:,-40:]
                sdf = self.scene.collision_loss(bone_samples(joints), tolerance=.003)
                lengths = (joints[...,1:,:]-joints.index_select(-2,parents)).norm(dim=-1)
                bone = (lengths-self.bone_lengths.to(lengths)).square().mean()
                agreement = self.consistency(full)
                fk, velocity = agreement['per_frame_fk_error_m2'], agreement['per_frame_velocity_error_m2']
                require(fk.shape == velocity.shape == (1,full.shape[-1]), 'Consistency must cover the complete native history')
                # Native velocity row t describes t->t+1; include last-history
                # boundary, exclude last suffix row which has no future frame.
                consistent = fk[:,-40:].mean()+velocity[:,-41:-1].mean()
                trust = (candidate-original).square().mean()
                loss = 100.*sdf+80.*bone+30.*consistent+.08*trust
                require(loss.numel() == 1 and torch.isfinite(loss).all(), 'Nonfinite lightweight SDF energy')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                require(candidate.grad is not None and torch.isfinite(candidate.grad).all(), 'Missing/nonfinite clean-x0 gradient')
                grad_norm = float(candidate.grad.norm())
                optimizer.step()
                with torch.no_grad():
                    candidate.copy_(original+(candidate-original).clamp(-.8,.8))
                self.trace.append(dict(call=call,iteration=iteration,loss=float(loss.detach()),
                    proxy_sdf_m2=float(sdf.detach()),bone_m2=float(bone.detach()),
                    native_consistency_m2=float(consistent.detach()),trust_m2=float(trust.detach()),
                    gradient_l2=grad_norm,gradient_finite=True,
                    maximum_normalized_correction=float((candidate.detach()-original).abs().max()),
                    route_gradient_weight=0.,static_keypose_gradient_weight=0.,
                    sdf_contains_target_other_floor_and_unknown=True))
        require(torch.isfinite(candidate).all(), 'Nonfinite guided clean suffix')
        return candidate.detach()


def endpoint_goal(case, schedule, offset):
    require(type(offset) is int and offset in (0,40,80), 'Three complete40 commitments only')
    require(schedule.route_xy.shape == (120,2) and schedule.target_strength.shape == (120,), 'Fixed120 plan required')
    index = offset+39
    strength = schedule.target_strength[index]
    height = case.initial_pose[2]+strength*(case.target_pose[2]-case.initial_pose[2])
    point = torch.cat((schedule.route_xy[index],height.reshape(1)))
    require(torch.isfinite(point).all(), 'Invalid planned world endpoint')
    walking = index < schedule.description['walk_frames']
    name = 'traj' if walking else 'pelvis'
    heading = None if walking else point.new_tensor([case.terminal_yaw])
    return {name:point[None]}, heading, dict(frame_0based=index,offset=offset,target_name=name,
        source='current_make_schedule_route_xy_and_initial_to_target_root_height',
        world_goal_xyz_zup_m=point.detach().cpu().tolist(),heading_world_rad=None if walking else float(case.terminal_yaw),
        qwen_coordinates_used=False,gt_event_times_used=False)


def sample_variant(model, case, scene, schedule, initial_prefix, frame, decode, consistency,
                   variant, seed, folder):
    require(variant in VARIANTS, 'Unknown target pilot variant')
    folder = Path(folder); proposals=folder/'proposals'; proposals.mkdir(parents=True,exist_ok=False)
    history=initial_prefix.detach().clone()
    require(history.shape == (1,263,1,20), 'Original20 prefix required')
    parents=torch.tensor(PARENTS[1:],device=case.initial_joints.device)
    lengths=(case.initial_joints[1:]-case.initial_joints.index_select(0,parents)).norm(dim=-1)
    chunks=[]; started=time.monotonic()
    for number,offset in enumerate((0,40,80)):
        world,heading,planned=endpoint_goal(case,schedule,offset)
        target=build_target_conditions(history,model.mean,model.std,frame,world,
            vendor_root=BASE/'vendor/mdm',heading_world=heading)
        require(set(target) == {*MODEL_GOAL_KEYS,'audit'}, 'Goal helper returned an unregistered model contract')
        require(target['audit']['target_frame'] == 'next_suffix_frame', 'Targets must not use prefix-first coordinates')
        target_kwargs={k:target[k] for k in MODEL_GOAL_KEYS}
        target_file=proposals/f'target_{number:03d}.npz'
        npz_once(target_file,target_cond=tensor_array(target['target_cond']),
                 is_heading=tensor_array(target['is_heading']))
        coordinate_file=proposals/f'target_{number:03d}.json'
        write_once(coordinate_file,dict(schema=SCHEMA+'.target',planned=planned,audit=target['audit'],
            target_joint_names=target['target_joint_names'],model_inputs=artifact(target_file),
            learned_target_disabled=variant=='target_off'))
        callback=LightSDFGuide(decode,history,scene,lengths,consistency) if variant=='target_sdf' else None
        prompt=planned_text(offset,schedule,frame,'anticipatory')
        tick=time.monotonic()
        suffix=model.sample(history[...,-20:],text=[prompt],seed=seed+number,**target_kwargs,
                            target_uncond=variant=='target_off',denoised_fn=callback)
        require(suffix.shape == (1,263,1,40) and torch.isfinite(suffix).all(), 'Incomplete native suffix')
        if callback is not None:
            require(callback.calls > 0 and len(callback.trace) == callback.calls*4,
                    'Requested clean-x0 guidance was not actually executed')
        history=torch.cat((history,suffix),-1)
        with torch.no_grad(): joints=decode(history)[0,-40:]
        proposal=proposals/f'proposal_{number:03d}.npz'
        npz_once(proposal,native_suffix=tensor_array(suffix),generated_world_joints=tensor_array(joints),
            start_frame_0based=np.asarray(offset),committed_frames=np.asarray(40))
        goal_point=next(iter(world.values()))[0]
        components=slice(0,2) if planned['target_name']=='traj' else slice(0,3)
        error=float((joints[-1,0,components]-goal_point[components]).norm())
        chunks.append(dict(chunk=number,seed=seed+number,text=prompt,output_start_0based=offset,
            predicted_frames=40,committed_frames=40,discarded_predicted_frames=0,
            seconds=time.monotonic()-tick,proposal=artifact(proposal),target=artifact(coordinate_file),
            target_model_inputs=artifact(target_file),raw_suffix_endpoint_goal_error_m=error,
            raw_goal_error_components='world_XY' if planned['target_name']=='traj' else 'world_XYZ',
            clean_x0_callback_calls=callback.calls if callback else 0,guidance_trace=callback.trace if callback else []))
        print(dict(variant=variant,chunk=number,committed_frames=40,raw_goal_error_m=error),flush=True)
    require(history.shape == (1,263,1,140) and torch.equal(history[...,:20],initial_prefix), 'Native full history/prefix changed')
    with torch.no_grad(): joints=decode(history)[0,20:]
    if history.is_cuda: torch.cuda.synchronize(history.device)
    return history,joints,chunks,time.monotonic()-started


def source_records():
    current=Path(__file__).resolve().parent
    paths=[current/n for n in ('run_target_pilot.py','target_backend.py','target_coordinates.py',
        'fetch_target_weights.py','generation_conditions.py','guarded_refine.py')]
    paths += [BASE/'code'/n for n in ('run_pilot.py','dip_backend.py','fetch_official_weights.py',
        'hml_codec.py','stage12_conditions.py','native_consistency.py','sdf_guidance.py','mesh_realization.py')]
    paths += [AGENT/'HSI-project/hsi/stage3_sequence'/n for n in ('mesh_body.py','geometry.py','constraints.py','data.py')]
    paths += [BASE/'vendor/mdm'/n for n in ('model/mdm.py','utils/sampler_util.py','diffusion/gaussian_diffusion.py',
        'data_loaders/tensors.py','data_loaders/humanml/scripts/motion_process.py')]
    return [artifact(p) for p in paths]


def run(output, *, scene_id='scene006', seed=390917, device='cpu'):
    require(scene_id == 'scene006' and type(seed) is int and 0 <= seed < 2**32-3, 'One bounded Scene006 pilot required')
    output=Path(output).resolve(); require(not output.exists(),'Use a new target pilot output folder')
    sources=source_records(); output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot'; snapshot.mkdir()
    for number,row in enumerate(sources): shutil.copy2(row['path'],snapshot/f'{number:02d}_{Path(row["path"]).name}')
    write_once(output/'request.json',dict(schema=SCHEMA+'.request',sources=sources,scene_id=scene_id,seed=seed,
        variants=list(VARIANTS),frames=120,fps=20.,commit_frames=40,cfg=2.5,device=device,
        clean_x0_guidance=GUIDANCE,sdf_variant_is_two_level_combination=True,
        post_mesh_sdf_weight=dict(target_off=0.,target_on=0.,target_sdf=100.),
        target_off_is_original_no_target_checkpoint=False,new_training=False,physics_simulation=False))
    started=time.monotonic()
    try:
        model=TargetDiPBackend(device=device,guidance_scale=2.5)
        case0=load_case(AGENT/'runs/stage12_ablation_20260909/fresh_chain/full'/RECEIPTS[scene_id],device=device)
        case,initial_report=constructed_relaxed_stand(case0,source_is_adapter_constructed=True)
        scene=BoundScene.from_case(case)
        prefix,frame,codec=build_static_prefix(case.initial_joints,model.mean,model.std,vendor_root=BASE/'vendor/mdm')
        prefix=prefix.to(device=device,dtype=torch.float32)
        decode=lambda features:decode_history(features,model.mean,model.std,frame,vendor_root=BASE/'vendor/mdm')
        consistency=NativeConsistency(case.initial_joints,model.mean,model.std,frame,BASE/'vendor/mdm')
        schedule=make_schedule(case,frames=120,fps=20.,approach_transition=True)
        require(schedule.anchor_frames == list(range(112,120)), 'Current final keypose lock interval changed')
        npz_once(output/'initial_state.npz',motion=tensor_array(case.initial_pose),
            vertices_world=tensor_array(case.initial_vertices),joints_world=tensor_array(case.initial_joints),
            betas=tensor_array(case.betas),original_constructed_tpose=tensor_array(case0.initial_pose),native_prefix=tensor_array(prefix))
        write_once(output/'conditions.json',dict(schema=SCHEMA+'.conditions',provenance=case.provenance,
            permissions=case.permissions,initial_adaptation=initial_report,initial_state=artifact(output/'initial_state.npz'),
            codec=codec,schedule=schedule.description,instruction=case.instruction,target_id=case.target_id,
            target_class=case.target_class,surface_id=case.surface_id,betas=tensor_array(case.betas).tolist(),
            sdf=case.sdf,same_initial_pose_all_variants=True,initial_history_is_observed=False,
            no_future_gt_or_input_event_timestamps=True,goal_coordinate_contract='next_suffix_frame_NOT_prefix_first',
            target_off_not_original_no_target_model=True))
        executions={}
        for variant in VARIANTS:
            where=output/variant; where.mkdir(); tick=time.monotonic()
            native,joints,chunks,sampling_seconds=sample_variant(model,case,scene,schedule,prefix,frame,decode,
                consistency,variant,seed,where)
            npz_once(where/'native.npz',native_features=tensor_array(native),world_joints=tensor_array(joints),
                initial_history_frames=np.asarray(20),fps=np.asarray(20.),world_from_local=frame.world_from_local,
                world_translation=frame.translation,interpretation=np.asarray('fresh official target DiP before mesh fitting'))
            anchors=[] if variant=='target_off' else list(schedule.anchor_frames)
            fit_config=RealizationConfig(steps=80,fps=20.)
            motion,mesh,fit=realize_motion(joints,case,fit_config,
                scene_energy=scene.mesh_energy if variant=='target_sdf' else None,anchor_frames=anchors)
            metrics=evaluate_mesh(mesh,case,scene,fps=20.,anchor_frames=anchors)
            metrics.update(raw_native_final_keypose_mean_joint_error_m=float((joints[-1]-case.target_joints).norm(dim=-1).mean()),
                raw_native_final_root_error_m=float((joints[-1,0]-case.target_joints[0]).norm()),
                sampling_seconds=sampling_seconds,mesh_realization_seconds=fit['elapsed_seconds'],
                raw_to_fixed_shape_mesh_fit_error_m=fit['raw_native_joint_fit_error_m'],
                final_keypose_enforced_in_kinematic_fitting=bool(anchors),
                raw_metrics_are_before_any_terminal_hard_lock=True,physical_motion_accepted=False)
            route=torch.cat((case.route_xy,case.route_xy.new_zeros(len(case.route_xy),1)),-1)
            npz_once(where/'sample.npz',motion=tensor_array(motion),vertices_world=tensor_array(mesh.vertices[0]),
                joints_world=tensor_array(mesh.joints[0]),faces=tensor_array(case.faces),betas=tensor_array(case.betas),
                fps=np.asarray(20.),coordinate_system=np.asarray('world_zup_m'),route_world=tensor_array(route),
                keypose_vertices_world=tensor_array(case.target_vertices[None]),keypose_joints_world=tensor_array(case.target_joints[None]),
                keypose_frames_1based=np.asarray([i+1 for i in anchors],dtype=np.int64))
            write_once(where/'sampling.json',dict(schema=SCHEMA+'.sampling',variant=variant,chunks=chunks,model=model.metadata,
                native=artifact(where/'native.npz'),learned_goal_enabled=variant!='target_off',target_uncond=variant=='target_off',
                gradient_guidance=GUIDANCE if variant=='target_sdf' else None,
                native_consistency=consistency.metadata if variant=='target_sdf' else None,
                fresh_diffusion_generation=True,new_training=False,commit_frames=40,discarded_future_frames=0,
                model_checkpoint_shared_all_variants=True,target_off_is_original_non_target_model=False))
            write_once(where/'mesh_realization.json',fit); write_once(where/'metrics.json',metrics)
            executions[variant]=dict(sample=artifact(where/'sample.npz'),native=artifact(where/'native.npz'),
                sampling=artifact(where/'sampling.json'),mesh_realization=artifact(where/'mesh_realization.json'),
                metrics_file=artifact(where/'metrics.json'),metrics=metrics,elapsed_seconds=time.monotonic()-tick,
                final_keypose_hard_locked=bool(anchors),post_mesh_full_surface_sdf=variant=='target_sdf',motion_release_authorized=False)
            write_once(where/'execution.json',dict(schema=SCHEMA+'.variant',status='completed_diagnostic_not_quality_acceptance',
                scene_id=scene_id,seed=seed,variant=variant,sample=executions[variant],
                conditions=artifact(output/'conditions.json'),scene_mesh=case.scene_mesh,sources=sources))
        for row in sources: verified(row)
        for key in ('checkpoint','configuration','bert','mean','std'): verified(model.metadata[key])
        result=dict(schema=SCHEMA,status='completed_diagnostic_not_quality_acceptance',scene_id=scene_id,seed=seed,
            samples=executions,conditions=artifact(output/'conditions.json'),scene_mesh=case.scene_mesh,backend=model.metadata,
            sources=sources,frames=120,fps=20.,commit_frames=40,cfg=2.5,same_initial_pose_all_variants=True,
            target_off_is_original_non_target_model=False,sdf_variant_is_two_level_combination=True,
            final_keypose_anchor_frames_0based=list(schedule.anchor_frames),
            unihsi_network_in_generation=False,physics_simulation=False,motion_release_authorized=False,
            no_future_gt_or_input_event_timestamps=True,elapsed_seconds=time.monotonic()-started)
        write_once(output/'execution.json',result); return result
    except BaseException as error:
        write_once(output/'run_failure.json',dict(schema=SCHEMA,status='failed_incomplete_pilot',sources=sources,
            error_type=type(error).__name__,reason=str(error),motion_release_authorized=False)); raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--scene',choices=['scene006'],default='scene006')
    p.add_argument('--seed',type=int,default=390917); p.add_argument('--device',default='cpu')
    a=p.parse_args(argv); torch.set_num_threads(4)
    result=run(a.output,scene_id=a.scene,seed=a.seed,device=a.device)
    print(dict(status=result['status'],output=str(a.output)),flush=True)


if __name__=='__main__': main()
