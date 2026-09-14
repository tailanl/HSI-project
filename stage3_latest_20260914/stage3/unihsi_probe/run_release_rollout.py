"""Bounded original UniHSI release actor + Task + real PhysX, not paper CNN.

No training, no keypose snapping, no retargeting, no rendering. The published
439->1024->1024->512->28 MLP is kept exactly; actor uses official rl-games1.1.4
normalization. GPU execution requires --execute. Runtime/cache/output are private.
UniHSI CC BY-NC-SA4.0 and inherited NVIDIA/IsaacGym license notices apply.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import time
import types

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
ORIGINAL=Path('/home/lzsh2025/workspace/UniHSI')
SDIST=PROJECT/'methods/closd_unihsi_sdf_stage3_20260912_v2/vendor/rl-games-1.1.4.tar.gz'
SDIST_SHA='c9b7b17e5be83d185420697670c83b519ea1f8667356a84cd98d3b7d9d8e08c9'
WEIGHT_SHA='41ca3f0c9ce9fc0d69a488cdced33f104bfbc572dbc9ab141d057196708fb24f'
SCHEMA='agent9.paper_structure.unihsi_release_rollout.v1'


def require(condition,message):
    if not condition:raise ValueError(message)


def artifact(path):
    path=Path(path).resolve(strict=True);before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Artifact changed during read')
    return dict(path=str(path),bytes=after.st_size,sha256=h.hexdigest())


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    sealed=dict(value,receipt_payload_sha256=hashlib.sha256(payload).hexdigest())
    with path.open('x',encoding='utf8') as stream:json.dump(sealed,stream,indent=2,allow_nan=False)
    return artifact(path)


def unpack_rl_games():
    require(artifact(SDIST)['sha256']==SDIST_SHA,'Official rl-games1.1.4 archive SHA changed')
    dest=HERE/'vendor_rl_games_1_1_4';dest.mkdir(exist_ok=True)
    with tarfile.open(str(SDIST),'r:gz') as archive:
        members=archive.getmembers()
        require(len(members)<2000 and sum(m.size for m in members)<10_000_000,'Unexpected archive size')
        for member in members:
            relative=Path(member.name)
            require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='rl-games-1.1.4'
                and (member.isfile() or member.isdir()),'Unsafe archive member')
            target=dest/relative
            require(not target.is_symlink(),'Private vendor symlink forbidden')
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            data=archive.extractfile(member).read()
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():require(target.read_bytes()==data,'Private rl-games source drift')
            else:
                with target.open('xb') as stream:stream.write(data)
    return dest/'rl-games-1.1.4'


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def install_import_paths(vendor):
    # Original task uses poselib.poselib.*, not poselib.*. Keep its outer
    # namespace visible through unihsi only; adding inner poselib masks it.
    for path in (ORIGINAL/'unihsi',ORIGINAL/'isaacgym/python',vendor):
        sys.path.insert(0,str(path))


def numpy_compatibility(np):
    result=[]
    for name,value in dict(float=float,int=int,bool=bool,complex=complex,object=object,str=str).items():
        if name not in np.__dict__:setattr(np,name,value);result.append(name)
    return result


def block_telemetry():
    def forbidden(*args,**kwargs):raise RuntimeError('Telemetry calls forbidden in isolated rollout')
    module=types.ModuleType('wandb');module.__file__=str(Path(__file__).resolve())
    module.init=module.log=module.login=module.finish=forbidden
    sys.modules['wandb']=module


def build_actor(vendor,device='cpu'):
    import numpy as np
    import torch
    import yaml
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    rms_module=sys.modules[RunningMeanStd.__module__]
    require(str(Path(rms_module.__file__).resolve()).startswith(str(vendor.resolve())+os.sep),'Wrong rl-games normalizer version')
    checkpoint=ORIGINAL/'checkpoints/Humanoid.pth'
    require(artifact(checkpoint)['sha256']==WEIGHT_SHA,'Unrecognized released actor checkpoint')
    values=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
    params=yaml.safe_load((ORIGINAL/'unihsi/data/cfg/train/rlg/amp_humanoid_task_deep_layer.yaml').read_text())['params']
    require(params['network']['name']=='amp' and params['network']['mlp']['units']==[1024,1024,512]
        and params['network']['mlp']['activation']=='relu' and 'heightmap_enc' not in params['network'],'Release config changed')
    builder_module=import_file('_private_unihsi_release_network',ORIGINAL/'unihsi/learning/amp_network_builder.py')
    builder=builder_module.AMPBuilder();builder.load(params['network'])
    network=builder.build('amp',actions_num=28,input_shape=(439,),value_size=1,num_seqs=1,amp_input_shape=(1250,))
    state={k[len('a2c_network.'):]:v for k,v in values['model'].items() if k.startswith('a2c_network.')}
    require(len(state)==len(values['model']),'Unexpected checkpoint wrapper state')
    require(state['actor_mlp.0.weight'].shape==(1024,439) and state['mu.weight'].shape==(28,512)
        and not any('cnn' in k or 'heightmap' in k for k in state),'Not the published MLP actor')
    network.load_state_dict(state,strict=True);network.eval().requires_grad_(False).to(device)
    normalizer=RunningMeanStd((439,));normalizer.load_state_dict(values['running_mean_std'],strict=True)
    normalizer.eval().requires_grad_(False).to(device)
    require(torch.isfinite(normalizer.running_mean).all() and torch.isfinite(normalizer.running_var).all()
        and (normalizer.running_var>=0).all(),'Invalid actor normalization')
    # Actual released builder is checked against direct weight mathematics on CPU/device.
    probe=torch.linspace(-2,2,439,device=device)[None]
    with torch.no_grad():
        z=normalizer(probe);mu,_=network.eval_actor(z);direct=z
        for index in (0,2,4):
            direct=torch.relu(torch.nn.functional.linear(direct,state[f'actor_mlp.{index}.weight'].to(device),state[f'actor_mlp.{index}.bias'].to(device)))
        direct=torch.nn.functional.linear(direct,state['mu.weight'].to(device),state['mu.bias'].to(device))
    require(torch.allclose(mu,direct,atol=2e-6,rtol=2e-6),'Original actor / direct released weights parity failed')
    return network,normalizer,dict(checkpoint=artifact(checkpoint),configuration=artifact(ORIGINAL/'unihsi/data/cfg/train/rlg/amp_humanoid_task_deep_layer.yaml'),
        original_network=artifact(ORIGINAL/'unihsi/learning/amp_network_builder.py'),rl_games_sdist=artifact(SDIST),
        rms_source=artifact(rms_module.__file__),architecture='published_MLP_439_1024_1024_512_28',
        paper_cnn_reproduced=False,body_observation_dim=223,task_observation_dim=216,heightmap_grid=[9,9],
        discriminator_in_action_selection=False,normalization_statistics_updated=False,actor_math_parity_max_abs=float((mu-direct).abs().max()),
        deterministic_mu=True,action_clamp=[-1.,1.],new_training=False)


def source_closure(vendor):
    roots=(ORIGINAL/'unihsi',ORIGINAL/'isaacgym/python',vendor)
    paths={Path(__file__).resolve(),ORIGINAL/'unihsi/learning/amp_network_builder.py'}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        if path.suffix=='.py' and any(str(path).startswith(str(root)+os.sep) for root in roots):paths.add(path)
    return [artifact(p) for p in sorted(paths)]


def npz_once(path,**arrays):
    import numpy as np
    with Path(path).open('xb') as stream:np.savez_compressed(stream,**arrays)
    return artifact(path)


def snapshot(task):
    values=dict(body_pos=task._rigid_body_pos,body_rot_xyzw=task._rigid_body_rot,body_linvel=task._rigid_body_vel,
        body_angvel=task._rigid_body_ang_vel,contact_forces=task._contact_forces,root_state=task._humanoid_root_states,
        dof_pos=task._dof_pos,dof_vel=task._dof_vel,dof_force=task.dof_force_tensor,force_sensors=task.vec_sensor_tensor,
        obs=task.obs_buf,step_mode=task.step_mode,progress=task.progress_buf,reset=task.reset_buf,terminate=task._terminate_buf,
        reward=task.rew_buf,contact_valid=task.contact_valid,contact_type=task.contact_type,contact_direction=task.contact_direction,
        target_stand_point=task.stand_point,local_heightmap=task.local_height_map,location_error=task.location_diff_buf,
        joint_contact_errors=task.joint_diff_buff)
    return {k:v.detach().cpu().numpy()[0].copy() for k,v in values.items()}


def make_scene_geometry(output,plan):
    import numpy as np
    import open3d as o3d
    original=ORIGINAL/'data/scannet/scene0000_00_vh_clean_2.ply'
    mesh=o3d.io.read_triangle_mesh(str(original))
    for rotation in plan['rotate']:mesh.rotate(mesh.get_rotation_matrix_from_xyz(rotation),center=(0,0,0))
    mesh.scale(plan['scale'],center=mesh.get_center())
    floor_origin=float(np.asarray(mesh.vertices).astype(np.float32)[:,2].min())
    mesh.translate((0,0,-floor_origin));mesh.translate(plan['transfer'])
    return npz_once(output/'scene_geometry.npz',vertices_world=np.asarray(mesh.vertices).astype(np.float32),
        faces=np.asarray(mesh.triangles).astype(np.int64),vertex_colors=np.asarray(mesh.vertex_colors).astype(np.float32),
        floor_plane_z=np.asarray(0.,np.float32)),dict(
        source=artifact(original),same_original_task_transform=True,rotation_xyz=plan['rotate'],
        scale_about_mesh_center=plan['scale'],post_scale_min_z=floor_origin,transfer=plan['transfer'],env_grid_offset_xyz=[0.,0.,0.],
        separate_infinite_floor_z=0.,task_heightmap_mutated_pcd_not_used_for_render=True)


def run(output,frames,seed,gpu,sceneplan=None,plan_key='0000',control_mode='trained_actor'):
    require(type(frames) is int and 1<=frames<=900,'Bounded 1..900 frames only')
    require(control_mode in ('trained_actor','zero_action'),'Only explicit released actor or zero-action physics baseline')
    require(type(gpu) is int and 0<=gpu<8,'Invalid visible GPU index')
    output=Path(output).resolve();require(not output.exists(),'Use a new isolated output')
    output.mkdir(parents=True);started=time.monotonic();task=None;sources=[];old_cwd=Path.cwd()
    try:
        vendor=unpack_rl_games();install_import_paths(vendor);block_telemetry()
        import numpy as np
        aliases=numpy_compatibility(np)
        os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'runtime_cache/torch_extensions')
        os.environ['MAX_JOBS']='2';os.environ['WANDB_MODE']='disabled'
        # IsaacGym must be imported before torch in this fresh runtime process.
        from isaacgym import gymapi,gymutil
        from isaacgym import gymtorch
        import torch
        import yaml
        torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.cuda.set_device(gpu);torch.cuda.manual_seed_all(seed)
        from env.tasks.unihsi_scannet import UniHSI_ScanNet
        cfg_path=ORIGINAL/'unihsi/data/cfg/humanoid_unified_interaction_scene_1.yaml'
        plan_path=Path(sceneplan or ORIGINAL/'sceneplan_demo/scannet_example.json').resolve(strict=True)
        plans=json.loads(plan_path.read_text());require(plan_key in plans,'Unknown plan key')
        plan_root={plan_key:plans[plan_key]};plan=plan_root[plan_key]
        require(plan['scene_id']=='0000_00','Only real original demo Scene0000_00 is in scope for this ABI')
        selected_plan=output/'selected_sceneplan.json'
        with selected_plan.open('x') as stream:json.dump(plan_root,stream,indent=2,allow_nan=False)
        cfg=yaml.safe_load(cfg_path.read_text());cfg['env']['numEnvs']=1;cfg['env']['numScenes']=1
        cfg['env']['asset']['assetRoot']=str(ORIGINAL/'unihsi/data/assets')
        cfg['env']['motion_file']=str(ORIGINAL/'motion_clips/chair_mo014.npy')
        cfg.update(headless=True,record_video=False,objFile=str(selected_plan),frames_dir=str(output/'unused_frames'),seed=seed)
        cfg['task']={'randomize':False}
        sim=gymapi.SimParams();sim.dt=1./60;gymutil.parse_sim_config(cfg['sim'],sim)
        sim.use_gpu_pipeline=True;sim.physx.use_gpu=True;sim.physx.num_threads=4
        sim.physx.max_gpu_contact_pairs=8*1024*1024
        network,rms,model=build_actor(vendor,device='cuda:'+str(gpu));sources=source_closure(vendor)
        inputs=dict(configuration=artifact(cfg_path),sceneplan_source=artifact(plan_path),sceneplan=artifact(selected_plan),motion_initializer=artifact(cfg['env']['motion_file']),
            humanoid_xml=artifact(ORIGINAL/'unihsi/data/assets/mjcf/amp_humanoid.xml'),
            scene_segmentation=artifact(ORIGINAL/'data/scannet/scene0000_00_vh_clean_2.0.010000.segs.json'),
            scene_aggregation=artifact(ORIGINAL/'data/scannet/scene0000_00_vh_clean.aggregation.json'))
        write_once(output/'request.json',dict(schema=SCHEMA+'.request',requested_frames=frames,fps=30.,num_envs=1,seed=seed,
            original_root=str(ORIGINAL),config=cfg,model=model,inputs=inputs,sources=sources,
            numpy_compat_aliases=aliases,telemetry='explicit_disabled_stub_raises_on_use',headless=True,graphics_device_id=-1,
            simulator='IsaacGym_PhysX',gpu_index=gpu,stage12_connected=False,paper_cnn_reproduced=False,
            network_training=False,hard_keypose_snap=False,physics_state_is_not_kinematic_replay=True,plan_key=plan_key,control_mode=control_mode))
        scene_artifact,scene_info=make_scene_geometry(output,plan)
        os.chdir(str(ORIGINAL)) # Original task's relative asset READS only; recordings explicitly disabled.
        task=UniHSI_ScanNet(cfg,sim,gymapi.SIM_PHYSX,'cuda',gpu,True)
        require(task.num_envs==1 and task.num_obs==439 and task.num_actions==28 and task.graphics_device_id==-1
            and not task.record_video and not task.record_sim_data and task.viewer is None,'Unexpected task/runtime contract')
        task.reset();initial=snapshot(task)
        names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0])
        require(len(names)==15,'Original AMP15 rigid body required')
        initial_artifact=npz_once(output/'initial.npz',**initial,body_names=np.asarray(names),fps=np.asarray(30.))
        rows=[];events=[];termination_reason='frame_budget_reached'
        for index in range(frames):
            with torch.no_grad():
                require(torch.isfinite(task.obs_buf).all(),'Nonfinite actual physics observation')
                if control_mode=='trained_actor':
                    mu,_=network.eval_actor(rms(task.obs_buf));actions=mu.clamp(-1,1)
                else:actions=torch.zeros((1,28),device=task.obs_buf.device,dtype=torch.float32)
                require(actions.shape==(1,28) and torch.isfinite(actions).all(),'Invalid actual actor action')
                task.step(actions)
                task.gym.fetch_results(task.sim,True)
                task._refresh_sim_tensors()
                row=snapshot(task);row['actions']=actions.detach().cpu().numpy()[0].copy()
                row['time_seconds']=np.asarray((index+1)/30.)
                require(all(not np.issubdtype(v.dtype,np.number) or np.isfinite(v).all() for v in row.values()),'Nonfinite post-physics state')
                rows.append(row)
                if index%30==0:print(json.dumps(dict(event='physics_frame',frame=index+1,root=row['root_state'][:3].tolist(),task_step=int(row['step_mode']))),flush=True)
                if bool(row['terminate']):termination_reason='original_task_physical_termination';break
                if bool(row['reset']):
                    if int(row['progress'])>=cfg['env']['episodeLength']-1:termination_reason='original_task_timeout';break
                    if int(row['step_mode'])+1>=len(plan['contact_pairs']):termination_reason='original_plan_finished';break
                    before=row['root_state'].copy();task.reset();after=snapshot(task)
                    events.append(dict(after_frame_0based=index,previous_task_step=int(row['step_mode']),next_task_step=int(after['step_mode']),
                        original_task_reset_called=True,root_state_max_abs_change=float(np.abs(after['root_state']-before).max())))
                    if not np.array_equal(before,after['root_state']):termination_reason='original_task_reset_relocated_body';break
        require(len(rows)>0,'No real physics frames')
        arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]};arrays['fps']=np.asarray(30.);arrays['body_names']=np.asarray(names)
        sample=npz_once(output/'sample.npz',**arrays)
        for record in [*sources,*inputs.values(),*list(model[k] for k in ('checkpoint','configuration','original_network','rl_games_sdist','rms_source'))]:
            require(artifact(record['path'])==record,'Source/input changed while executing')
        force=arrays['contact_forces'];speed=np.linalg.norm(arrays['body_linvel'],axis=-1)
        metrics=dict(actual_physics_frames=len(rows),duration_seconds=len(rows)/30.,raw_root_xy_displacement_m=float(np.linalg.norm(arrays['root_state'][-1,:2]-initial['root_state'][:2])),
            max_body_speed_m_s=float(speed.max()),contact_force_nonzero_fraction=float((np.linalg.norm(force,axis=-1)>.1).mean()),
            max_contact_force_N=float(np.linalg.norm(force,axis=-1).max()),task_steps_completed=int(arrays['step_mode'].max()),
            physical_termination_observed=bool(arrays['terminate'].any()),motion_quality_passed=False)
        metrics_artifact=write_once(output/'metrics.json',metrics)
        result=dict(schema=SCHEMA,status='completed_bounded_physics_diagnostic_not_quality_acceptance',frames=len(rows),requested_frames=frames,fps=30.,
            scene_id='scene0000_00',seed=seed,variant=control_mode,control_mode=control_mode,executed_steps=len(rows),dt=1./30.,
            episode_end_reason=termination_reason,plan_key=plan_key,method='UniHSI_published_MLP_actor_original_Task_PhysX',sample=sample,initial=initial_artifact,
            scene_geometry=scene_artifact,scene_transform=scene_info,body_names=names,assets=inputs,model=model,sources=sources,
            metrics=metrics,metrics_file=metrics_artifact,task_reset_events=events,termination_reason=termination_reason,
            no_implicit_episode_restart=True,stage12_connected=False,new_training=False,paper_cnn_reproduced=False,
            ground_truth_motion_displayed=False,hard_keypose_snap=False,sdf_guidance_added=False,simulator='IsaacGym_PhysX',
            physics_parameters=dict(sim_dt=sim.dt,substeps=sim.substeps,control_frequency_inv=task.control_freq_inv,
                gravity=[sim.gravity.x,sim.gravity.y,sim.gravity.z],solver_type=sim.physx.solver_type,
                use_gpu_pipeline=sim.use_gpu_pipeline,physx_use_gpu=sim.physx.use_gpu,scene_config=cfg['sim']),
            actual_actor_used=control_mode=='trained_actor',baseline_is_zero_action_not_random_policy=control_mode=='zero_action',
            physical_state_feedback=True,all_saved_frames_post_physics=True,initial_state_saved_separately=True,
            rendering_executed=False,motion_quality_passed=False,elapsed_seconds=time.monotonic()-started)
        write_once(output/'execution.json',result);return result
    except BaseException as error:
        write_once(output/'failure.json',dict(schema=SCHEMA,status='failed_incomplete_rollout',error_type=type(error).__name__,
            error=str(error),sources=sources,physics_success_not_asserted=True,elapsed_seconds=time.monotonic()-started));raise
    finally:
        if task is not None:task.gym.destroy_sim(task.sim)
        os.chdir(str(old_cwd))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=180);p.add_argument('--seed',type=int,default=390918)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--execute',action='store_true');p.add_argument('--check-actor',action='store_true')
    p.add_argument('--sceneplan',type=Path);p.add_argument('--plan-key',default='0000')
    p.add_argument('--control-mode',choices=('trained_actor','zero_action'),default='trained_actor')
    args=p.parse_args(argv)
    if args.check_actor:
        vendor=unpack_rl_games();install_import_paths(vendor)
        import numpy as np
        numpy_compatibility(np)
        _,_,report=build_actor(vendor,'cpu');print(json.dumps(dict(status='cpu_actor_weights_and_math_checked_not_physics',model=report)));return
    if not args.execute:
        print(json.dumps(dict(status='not_executed',requires_explicit_execute=True,method='published_MLP_release_not_paper_CNN')));return
    result=run(args.output,args.frames,args.seed,args.gpu,args.sceneplan,args.plan_key,args.control_mode)
    print(json.dumps(dict(status=result['status'],frames=result['frames'])))


if __name__=='__main__':main()
