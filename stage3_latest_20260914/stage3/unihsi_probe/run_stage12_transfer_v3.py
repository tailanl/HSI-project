"""V3 partial transfer: same world geometry plus stale-body SDF action gate.

Not full SMPL-X keypose/shape consumption; no retargeting or snap. Optional
experimental SDF action filtering is distinct from the original paper policy.
Original UniHSI / NVIDIA licensing applies. No simulation without --execute.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import run_release_rollout as release

SCHEMA='agent9.paper_structure.unihsi_stage12_partial_transfer.v3'
HSI_ROOT=release.PROJECT/'HSI-project'
require=release.require


def checked(record):
    require(isinstance(record,dict) and release.artifact(record['path'])==record,'Artifact identity/hash mismatch')
    return Path(record['path'])


def read_packet(path):
    import numpy as np
    path=Path(path).resolve(strict=True);manifest=json.loads(path.read_text())
    require(manifest.get('schema')=='agent9.paper_structure.stage12_packet.v1','Wrong Stage12 packet schema')
    if 'receipt_payload_sha256' in manifest:
        payload={k:v for k,v in manifest.items() if k!='receipt_payload_sha256'}
        require(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==manifest['receipt_payload_sha256'],'Packet seal mismatch')
    require(manifest.get('action_family')=='sit','Only explicit sitting contact transfer implemented')
    refs=[manifest['arrays'],manifest['scene_mesh'],manifest['keypoint_descriptions']['source_keynodes'],
        *[manifest['sdf'][k] for k in ('receipt','target_sdf','collision_sdf')]]
    for record in refs:checked(record)
    require(manifest['sdf']['coordinate_system']=='world_zup_m' and manifest['sdf']['ground_plane_z_m']==0
        and manifest['sdf']['outside_is_unknown_not_free'] is True,'Unsupported scene/floor coordinate contract')
    with np.load(str(checked(manifest['arrays'])),allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    for key,shape in dict(initial_pose=(135,),target_pose=(135,),betas=(10,),target_root_xyz=(3,),contact_goal=(3,),contact_normal=(3,)).items():
        require(key in arrays and arrays[key].shape==shape and np.isfinite(arrays[key]).all(),'Invalid '+key)
    require(arrays['route_xy'].ndim==2 and arrays['route_xy'].shape[1]==2 and len(arrays['route_xy'])>0
        and np.isfinite(arrays['route_xy']).all(),'Invalid route')
    tri=arrays['contact_triangles']
    require(tri.ndim==3 and tri.shape[1:]==(3,3) and len(tri)>0 and np.isfinite(tri).all(),'Invalid surface triangles')
    require(np.array_equal(arrays['contact_joint_ids'],np.array([0,1,2])),'Only explicit SMPL-X pelvis / left-hip / right-hip support map implemented')
    require(np.allclose(arrays['target_pose'][:3],arrays['target_root_xyz'],rtol=0,atol=1e-7),'Target root identity mismatch')
    require(np.linalg.norm(arrays['route_xy'][0]-arrays['initial_pose'][:2])<1e-4,'Route start / initial XY mismatch')
    require(arrays['initial_joints'].shape==(22,3) and np.isfinite(arrays['initial_joints']).all(),'Invalid original initial joints')
    nodes=json.loads(checked(manifest['keypoint_descriptions']['source_keynodes']).read_text())
    stage1=nodes['start']['source']['root_xyz_yaw_world_zup']
    require(len(stage1)==4 and np.isfinite(stage1).all()
        and np.allclose(stage1[:2],arrays['initial_pose'][:2],atol=2e-6,rtol=0),'Stage1 initial binding failed')
    arrays['_validated_stage1_initial_yaw']=np.asarray(stage1[3],np.float64)
    return manifest,arrays,[release.artifact(path),*refs]


def load_world_scene(raw_path):
    """Use the exact retained Stage2 loader, including OBJ graph/order handling."""
    import numpy as np
    sys.path.insert(0,str(HSI_ROOT))
    from hsi.stage2 import view_numeric
    expected=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],np.float64)
    require(np.array_equal(view_numeric.LINGO_YUP_TO_WORLD_ZUP,expected),'Authoritative LINGO conversion changed')
    mesh=view_numeric.load_scene_mesh(Path(raw_path))
    sources=[release.artifact(view_numeric.__file__),release.artifact(HSI_ROOT/'hsi/common/artifacts.py')]
    transform=dict(raw_asset_coordinate_system='lingo_yup_m',simulation_and_saved_geometry_coordinate_system='world_zup_m',
        raw_to_world_4x4=expected.tolist(),conversion_rule='world=(raw_x,-raw_z,raw_y)',
        loader_source=sources[0],unit_scale=1.,translation=[0.,0.,0.],recentered=False,
        source_packet_global_coordinate_label_does_not_describe_raw_mesh=True,
        original_mesh_world_geometry_preserved=True,raw_mesh_coordinates_unchanged=False,
        floor_and_route_and_contact_already_world_zup_not_transformed_again=True,
        faces_order_from_authoritative_loader=True)
    return mesh,transform,sources


def world_scene_arrays(mesh,transform):
    import numpy as np
    vertices=np.asarray(mesh.vertices).astype(np.float32);faces=np.asarray(mesh.faces).astype(np.int64)
    require(vertices.ndim==2 and vertices.shape[1]==3 and len(vertices)>0 and np.isfinite(vertices).all(),'Invalid real scene vertices')
    require(faces.ndim==2 and faces.shape[1]==3 and len(faces)>0 and faces.min()>=0 and faces.max()<len(vertices),'Invalid real scene face indices')
    return dict(vertices_world=vertices,faces=faces,
        vertex_colors=(np.asarray(mesh.visual.vertex_colors[:,:3],np.float32)/255.
            if getattr(mesh.visual,'kind',None)=='vertex' else np.empty((0,3),np.float32)),
        floor_plane_z=np.asarray(0.,np.float32),raw_to_world_4x4=np.asarray(transform['raw_to_world_4x4'],np.float64))


def heading_from_smplx(pose):
    import numpy as np
    x,y=np.asarray(pose[3:9],dtype=np.float64).reshape(2,3)
    require(np.linalg.norm(x)>1e-6,'Degenerate initial root rotation');x=x/np.linalg.norm(x)
    y=y-np.dot(x,y)*x;require(np.linalg.norm(y)>1e-6,'Degenerate initial root rotation');y=y/np.linalg.norm(y)
    rotation=np.stack((x,y,np.cross(x,y)));front=rotation[:2,2]
    require(np.linalg.norm(front)>1e-6,'No initial XY heading')
    yaw=float(np.arctan2(front[1],front[0]))
    # The released task has a division by qz in its heightmap rotation. Do not
    # silently perturb a locked upstream heading to work around this defect.
    require(abs(math.sin(yaw/2))>1e-7,'Exact zero initial yaw unsupported by released heightmap code; no implicit perturbation')
    return yaw


def heading_contract(arrays):
    """Validate row-6D parity, separating global-rotation forward from hip yaw."""
    import numpy as np
    import torch
    sys.path.insert(0,str(HSI_ROOT))
    from hsi.stage3_sequence.constraints import rotation_6d_to_matrix
    pose=arrays['initial_pose'];yaw=heading_from_smplx(pose)
    rotation=rotation_6d_to_matrix(torch.as_tensor(pose[3:9],dtype=torch.float64)).numpy()
    front=rotation@np.array([0.,0.,1.]);official=float(np.arctan2(front[1],front[0]))
    require(abs(math.remainder(yaw-official,2*math.pi))<1e-7,'Heading differs from authoritative row-6D decoder')
    hip=arrays['initial_joints'][2,:2]-arrays['initial_joints'][1,:2]
    require(np.linalg.norm(hip)>1e-7,'No initial hip yaw')
    hip_yaw=float(np.arctan2(hip[0],-hip[1]));stage1=float(arrays['_validated_stage1_initial_yaw'])
    require(abs(math.remainder(hip_yaw-stage1,2*math.pi))<2e-5,'Original shape-calibrated hip / Stage1 yaw mismatch')
    return dict(smplx_forward_axis='+Z',native_AMP_forward_axis='+X',root_rotation_6d_convention='first_two_rows',
        heading_source='saved_packet_root_rotation_anatomical_forward',heading_world_rad=yaw,
        authoritative_decoder_heading_world_rad=official,decoder_parity_error_rad=abs(math.remainder(yaw-official,2*math.pi)),
        stage1_hip_axis_yaw_rad=stage1,actual_initial_smplx_hip_axis_yaw_rad=hip_yaw,
        chosen_heading_minus_stage1_hip_yaw_rad=math.remainder(yaw-stage1,2*math.pi),
        same_full_orientation_or_bodyshape_not_claimed=True,heading_perturbed=False,
        reason_for_small_difference='upstream shaped_static_history compensates shape-dependent hip axis',
        decoder_source=release.artifact(HSI_ROOT/'hsi/stage3_sequence/constraints.py'),
        initialization_source=release.artifact(HSI_ROOT/'experiments/stage3_upstream_adapter.py'))


def make_transfer_plan(manifest,arrays,seed=0):
    import numpy as np
    normal=np.asarray(arrays['contact_normal'],dtype=np.float64);normal/=np.linalg.norm(normal)
    angle=float(np.degrees(np.arccos(np.clip(normal[2],-1,1))))
    require(np.isfinite(angle) and angle<=15,'Seat normal incompatible with released discrete up-contact convention')
    tri=np.asarray(arrays['contact_triangles'],dtype=np.float64)
    areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    require(np.isfinite(areas).all() and areas.sum()>1e-10,'Degenerate contact surface')
    rng=np.random.default_rng(seed);indices=rng.choice(len(tri),200,replace=True,p=areas/areas.sum())
    uv=rng.random((200,2));s=np.sqrt(uv[:,0]);bary=np.stack((1-s,s*(1-uv[:,1]),s*uv[:,1]),axis=1)
    points=np.einsum('bi,bij->bj',bary,tri[indices]).astype(np.float32)
    root=arrays['initial_pose'][:3];waypoints=[]
    for xy in arrays['route_xy']:
        if len(waypoints)==0 and np.linalg.norm(xy-root[:2])<1e-4:continue
        waypoints.append([float(xy[0]),float(xy[1]),float(root[2])])
    require(len(waypoints)+1<=30,'Released CoC supports at most 30 events')
    goals=waypoints+[arrays['target_root_xyz'].tolist()]
    pairs=[[['upstream_route','none','none','none','none']] for _ in waypoints]
    pairs.append([['upstream_seat','seat_surface',joint,'contact','up'] for joint in ('pelvis','left_hip','right_hip')])
    return dict(scene_id=manifest['scene_id'],init_pos=root[:2].tolist(),contact_pairs=pairs,
        standpoints=goals,initial_heading_world_rad=heading_from_smplx(arrays['initial_pose']),
        initial_heading_contract=heading_contract(arrays),
        target_heading_world_rad=float(arrays['terminal_yaw']),target_heading_consumed=False,
        smplx_to_amp_contact_map={'pelvis':'pelvis','left_hip':'left_thigh','right_hip':'right_thigh'},
        contact_slot_indices=[0,1,4],surface_sampling='area_weighted_barycentric_200',surface_seed=seed,
        normal_world=normal.tolist(),released_contact_direction=[0,0,1],normal_quantization_error_deg=angle,
        route_initial_duplicate_removed=True,route_terminal_is_not_target_root=True,
        waypoint_height_source='upstream_initial_root_z',contact_goal_source='unchanged_upstream_target_root_xyz',
        fullbody_keypose_consumed=False,shape_preserved=False),points,indices,bary.astype(np.float32)


def task_class(original_cls,packet,arrays,plan,vertices,faces,surface_points):
    import numpy as np
    import torch
    from isaacgym import gymapi
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from exact_heightmap import get_height_map

    class Stage12TransferTask(original_cls):
        def __init__(self,*args,**kwargs):
            self.transfer_initial_pending=True
            super().__init__(*args,**kwargs)

        def _load_mesh(self):
            require(self.num_envs==1,'Single-env partial transfer only')
            self.pcds={'0000':vertices.copy()}
            self.init_pos=torch.tensor([plan['init_pos']],device=self.device,dtype=torch.float32)
            self.scene_idx=torch.zeros((1,1),device=self.device,dtype=torch.long)
            params=gymapi.TriangleMeshParams();params.nb_vertices=len(vertices);params.nb_triangles=len(faces)
            self.gym.add_triangle_mesh(self.sim,vertices.flatten(order='C'),faces.astype(np.uint32).flatten(order='C'),params)

        def _get_pcd_parts(self,pcd_list):
            device=self.device;steps=len(plan['standpoints'])
            self.obj_pcd_buffer=torch.zeros((1,30,15,200,3),device=device)
            self.contact_type_step=torch.zeros((1,30,15),device=device,dtype=torch.bool)
            self.contact_valid_step=torch.zeros_like(self.contact_type_step)
            self.contact_direction_step=torch.zeros((1,30,15,3),device=device,dtype=torch.long)
            self.joint_pairs=torch.zeros((1,30,15),device=device,dtype=torch.long)
            self.joint_pairs_valid=torch.zeros((1,30,15),device=device,dtype=torch.bool)
            self.scene_stand_point=torch.zeros((1,30,3),device=device)
            self.scene_stand_point[0,:steps]=torch.tensor(plan['standpoints'],device=device)
            for slot in plan['contact_slot_indices']:
                self.obj_pcd_buffer[0,steps-1,slot]=torch.tensor(surface_points,device=device)
                self.contact_type_step[0,steps-1,slot]=True;self.contact_valid_step[0,steps-1,slot]=True
                self.contact_direction_step[0,steps-1,slot,2]=1
            self.max_steps=torch.tensor([steps],device=device,dtype=torch.int32)
            self.contact_pairs=[plan['contact_pairs']]
            # Original 100x100 heightmap algorithm gets a copy because it clips
            # tall vertices in-place; collision / render mesh stays unchanged.
            h=get_height_map(vertices.copy(),HEIGHT_MAP_DIM=100)
            self.height_map=torch.tensor(h[None],device=device,dtype=torch.float32)

        def _reset_actors(self,env_ids):
            super()._reset_actors(env_ids)
            if self.transfer_initial_pending and len(env_ids):
                require(len(env_ids)==1 and int(env_ids[0])==0,'Unexpected initialization indices')
                yaw=plan['initial_heading_world_rad']
                self._humanoid_root_states[env_ids,0]=float(arrays['initial_pose'][0])
                self._humanoid_root_states[env_ids,1]=float(arrays['initial_pose'][1])
                self._humanoid_root_states[env_ids,3:7]=torch.tensor([0.,0.,math.sin(yaw/2),math.cos(yaw/2)],device=self.device)
                self.transfer_initial_pending=False
    return Stage12TransferTask


def run(packet_path,output,frames=900,seed=390919,gpu=0,control_mode='trained_actor',sdf_filter=False):
    require(type(frames) is int and 1<=frames<=900,'Bounded 1..900 frames only')
    require(control_mode in ('trained_actor','zero_action'),'Unsupported control mode')
    require(type(sdf_filter) is bool and not(sdf_filter and control_mode=='zero_action'),'SDF filter only for explicitly trained actor')
    require(type(gpu) is int and 0<=gpu<8,'Invalid visible GPU index')
    output=Path(output).resolve();require(not output.exists(),'Use a new isolated output')
    output.mkdir(parents=True);started=time.monotonic();task=None;sources=[];old_cwd=Path.cwd()
    try:
        vendor=release.unpack_rl_games();release.install_import_paths(vendor);release.block_telemetry()
        import numpy as np
        aliases=release.numpy_compatibility(np)
        os.environ['TORCH_EXTENSIONS_DIR']=str(release.HERE/'runtime_cache/torch_extensions');os.environ['MAX_JOBS']='2';os.environ['WANDB_MODE']='disabled'
        from isaacgym import gymapi,gymutil,gymtorch
        import torch
        import yaml
        torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.cuda.set_device(gpu);torch.cuda.manual_seed_all(seed)
        from env.tasks.unihsi_scannet import UniHSI_ScanNet
        packet,upstream,packet_refs=read_packet(packet_path)
        plan,points,indices,bary=make_transfer_plan(packet,upstream,seed=0)
        plan_path=output/'selected_sceneplan.json'
        with plan_path.open('x') as stream:json.dump({'0000':plan},stream,indent=2,allow_nan=False)
        surface=release.npz_once(output/'contact_adapter.npz',points_world=points,triangle_indices=indices,barycentric=bary)
        mesh,world_transform,geometry_sources=load_world_scene(checked(packet['scene_mesh']))
        geometry=world_scene_arrays(mesh,world_transform);vertices=geometry['vertices_world'];faces=geometry['faces']
        scene=release.npz_once(output/'scene_geometry.npz',**geometry)
        coordinate_addendum=release.write_once(output/'coordinate_contract.json',dict(schema=SCHEMA+'.coordinates',
            raw_asset=packet['scene_mesh'],saved_world_scene=scene,transform=world_transform,
            source_packet=packet_refs[0],target_root_xyz_world_zup=upstream['target_root_xyz'].tolist(),
            initial_heading=plan['initial_heading_contract'],sdf_already_world_zup=True))
        cfg_path=release.ORIGINAL/'unihsi/data/cfg/humanoid_unified_interaction_scene_1.yaml'
        cfg=yaml.safe_load(cfg_path.read_text());cfg['env'].update(numEnvs=1,numScenes=1)
        cfg['env']['asset']['assetRoot']=str(release.ORIGINAL/'unihsi/data/assets');cfg['env']['motion_file']=str(release.ORIGINAL/'motion_clips/chair_mo014.npy')
        cfg.update(headless=True,record_video=False,objFile=str(plan_path),frames_dir=str(output/'unused_frames'),seed=seed);cfg['task']={'randomize':False}
        sim=gymapi.SimParams();sim.dt=1./60;gymutil.parse_sim_config(cfg['sim'],sim)
        sim.use_gpu_pipeline=True;sim.physx.use_gpu=True;sim.physx.num_threads=4;sim.physx.max_gpu_contact_pairs=8*1024*1024
        actor,rms,model=release.build_actor(vendor,'cuda:'+str(gpu))
        sources=release.source_closure(vendor)+[release.artifact(__file__),*geometry_sources,
            release.artifact(Path(__file__).resolve().parent.parent/'exact_heightmap.py'),
            plan['initial_heading_contract']['decoder_source'],plan['initial_heading_contract']['initialization_source']]
        if sdf_filter:
            sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
            from pd_sdf_filter_v2 import PDSDFFilterV2 as PDSDFFilter
            sources += [release.artifact(Path(__file__).resolve().parent.parent/name)
                for name in ('pd_sdf_filter_v2.py','pd_sdf_filter.py','reference_sdf.py','comparison/physics_rollout_tools.py')]
        inputs=dict(configuration=release.artifact(cfg_path),sceneplan=release.artifact(plan_path),stage12_manifest=packet_refs[0],
            humanoid_xml=release.artifact(release.ORIGINAL/'unihsi/data/assets/mjcf/amp_humanoid.xml'),
            motion_initializer=release.artifact(cfg['env']['motion_file']),source_mesh=packet['scene_mesh'])
        release.write_once(output/'request.json',dict(schema=SCHEMA+'.request',packet=packet,config=cfg,plan=plan,
            inputs=inputs,sources=sources,model=model,coordinate_contract=coordinate_addendum,num_envs=1,requested_frames=frames,fps=30.,seed=seed,
            control_mode=control_mode,numpy_compat_aliases=aliases,telemetry='disabled_stub_raises',
            stage12_connected_partially=True,fullbody_keypose_consumed=False,sdf_filter_requested=sdf_filter,paper_cnn_reproduced=False))
        os.chdir(str(release.ORIGINAL))
        Task=task_class(UniHSI_ScanNet,packet,upstream,plan,vertices,faces,points)
        task=Task(cfg,sim,gymapi.SIM_PHYSX,'cuda',gpu,True)
        require(task.num_obs==439 and task.num_actions==28 and task.graphics_device_id==-1 and task.viewer is None
            and not task.record_video and not task.record_sim_data,'Unexpected original runtime contract')
        task.reset();initial=release.snapshot(task)
        require(np.allclose(initial['root_state'][:2],upstream['initial_pose'][:2],atol=1e-7,rtol=0),'Initial upstream XY changed')
        names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0]);require(len(names)==15,'Expected AMP15 body')
        action_filter=PDSDFFilter(task,packet_path,inputs['humanoid_xml']['path'],names) if sdf_filter else None
        initial_artifact=release.npz_once(output/'initial.npz',**initial,body_names=np.asarray(names),fps=np.asarray(30.))
        rows=[];events=[];reason='frame_budget_reached'
        for index in range(frames):
            with torch.no_grad():
                require(torch.isfinite(task.obs_buf).all(),'Nonfinite physical observation')
                if control_mode=='trained_actor':mu,_=actor.eval_actor(rms(task.obs_buf));actions=mu.clamp(-1,1)
                else:actions=torch.zeros((1,28),device=task.obs_buf.device)
                require(actions.shape==(1,28) and torch.isfinite(actions).all(),'Invalid actor actions')
                nominal=actions.detach().clone()
                if action_filter is not None:actions=action_filter(actions)
                require(actions.shape==(1,28) and torch.isfinite(actions).all(),'Invalid filtered actor actions')
                task.step(actions);task.gym.fetch_results(task.sim,True);task._refresh_sim_tensors()
                row=release.snapshot(task);row['actions']=actions.detach().cpu().numpy()[0].copy();row['time_seconds']=np.asarray((index+1)/30.)
                row['nominal_actor_actions']=nominal.detach().cpu().numpy()[0].copy()
                require(all(not np.issubdtype(v.dtype,np.number) or np.isfinite(v).all() for v in row.values()),'Nonfinite post-physics state')
                rows.append(row)
                if index%30==0:print(json.dumps(dict(event='physics_frame',frame=index+1,root=row['root_state'][:3].tolist(),task_step=int(row['step_mode']))),flush=True)
                if bool(row['terminate']):reason='original_task_physical_termination';break
                if bool(row['reset']):
                    if int(row['progress'])>=cfg['env']['episodeLength']-1:reason='original_task_timeout';break
                    if int(row['step_mode'])+1>=len(plan['contact_pairs']):reason='contact_chain_finished_not_full_keypose';break
                    before=row['root_state'].copy();task.reset();after=release.snapshot(task)
                    events.append(dict(after_frame_0based=index,previous_task_step=int(row['step_mode']),next_task_step=int(after['step_mode']),
                        original_task_reset_called=True,root_state_max_abs_change=float(np.abs(after['root_state']-before).max())))
                    if not np.array_equal(before,after['root_state']):reason='original_task_reset_relocated_body';break
        require(len(rows)>0,'No actual physics frames')
        saved={k:np.stack([r[k] for r in rows]) for k in rows[0]};saved.update(body_names=np.asarray(names),fps=np.asarray(30.))
        sample=release.npz_once(output/'sample.npz',**saved)
        force=np.linalg.norm(saved['contact_forces'],axis=-1)
        metrics=dict(actual_physics_frames=len(rows),duration_seconds=len(rows)/30.,
            raw_root_xy_displacement_m=float(np.linalg.norm(saved['root_state'][-1,:2]-initial['root_state'][:2])),
            raw_final_root_to_stage2_xyz_m=float(np.linalg.norm(saved['root_state'][-1,:3]-upstream['target_root_xyz'])),
            raw_final_root_to_stage2_xy_m=float(np.linalg.norm(saved['root_state'][-1,:2]-upstream['target_root_xyz'][:2])),
            max_body_speed_m_s=float(np.linalg.norm(saved['body_linvel'],axis=-1).max()),max_contact_force_N=float(force.max()),
            task_steps_completed=int(saved['step_mode'].max())+(reason=='contact_chain_finished_not_full_keypose'),
            physical_termination_observed=bool(saved['terminate'].any()),motion_quality_passed=False)
        metrics_file=release.write_once(output/'metrics.json',metrics)
        filter_artifact=None
        if action_filter is not None:
            require(len(action_filter.trace)==len(rows),'SDF trace / executed physics frame mismatch')
            filter_artifact=release.write_once(output/'sdf_action_filter.json',dict(
                schema=SCHEMA+'.experimental_linear_jacobian_sdf_filter',frames=len(rows),trace=action_filter.trace,
                applied_frames=sum(bool(x['applied']) for x in action_filter.trace),
                not_original_paper_component=True,not_dynamics_prediction=True,full_surface_guarantee=False,
                trust_radius_normalized_action=action_filter.trust_radius,iterations=action_filter.iterations,
                actual_actions_are_saved_and_executed_by_PhysX=True))
        for record in [*sources,*inputs.values(),*packet_refs,*[model[k] for k in ('checkpoint','configuration','original_network','rl_games_sdist','rms_source')]]:checked(record)
        result=dict(schema=SCHEMA,status='completed_bounded_partial_transfer_not_quality_acceptance',method='UniHSI_released_MLP_original_feedback_partial_LINGO_contact_transfer',
            frames=len(rows),executed_steps=len(rows),requested_frames=frames,fps=30.,dt=1./30.,scene_id='scene'+packet['scene_id'],seed=seed,
            variant=control_mode+('_sdf_filter' if sdf_filter else ''),control_mode=control_mode,sample=sample,initial=initial_artifact,scene_geometry=scene,body_names=names,
            scene_transform=dict(source=packet['scene_mesh'],**world_transform,
                env_grid_offset_xyz=[0.,0.,0.],separate_infinite_floor_z=0.),coordinate_contract=coordinate_addendum,
            source_packet=packet_refs[0],packet_artifacts=packet_refs,contact_adapter=surface,contact_plan=plan,assets=inputs,model=model,sources=sources,
            metrics=metrics,metrics_file=metrics_file,termination_reason=reason,episode_end_reason=reason,task_reset_events=events,
            initialization=dict(kind='original_native_AMP_default_pose_with_upstream_XY_and_packet_root_orientation_heading',
                upstream_smplx_root_xyz=upstream['initial_pose'][:3].tolist(),actual_native_root_xyz=initial['root_state'][:3].tolist(),
                native_minus_upstream_root_z_m=float(initial['root_state'][2]-upstream['initial_pose'][2]),
                initial_heading_world_rad=plan['initial_heading_world_rad'],upstream_betas=upstream['betas'].tolist(),
                heading_contract=plan['initial_heading_contract'],
                upstream_betas_applied=False,same_initial_fullbody_pose=False,same_body_shape=False),
            stage12_connected_partially=True,fullbody_keypose_consumed=False,shape_preserved=False,target_heading_consumed=False,
            upstream_target_root_modified=False,upstream_permissions_relaxed=False,full_stage3_contract_satisfied=False,
            sdf_packet_retained=packet['sdf'],sdf_guidance_added=sdf_filter,sdf_filter_trace=filter_artifact,
            sdf_filter_applied_frames=sum(bool(x['applied']) for x in action_filter.trace) if action_filter is not None else 0,
            sdf_filter_is_experimental_linear_action_proxy_not_paper_component=sdf_filter,sdf_metrics_computed=False,
            simulator='IsaacGym_PhysX',physics_parameters=dict(sim_dt=sim.dt,substeps=sim.substeps,control_frequency_inv=task.control_freq_inv,
                gravity=[sim.gravity.x,sim.gravity.y,sim.gravity.z],solver_type=sim.physx.solver_type,use_gpu_pipeline=sim.use_gpu_pipeline,
                physx_use_gpu=sim.physx.use_gpu,scene_config=cfg['sim']),physical_state_feedback=True,
            actual_actor_used=control_mode=='trained_actor',baseline_is_zero_action_not_random_policy=control_mode=='zero_action',
            all_saved_frames_post_physics=True,initial_state_saved_separately=True,no_implicit_episode_restart=True,
            new_training=False,paper_cnn_reproduced=False,hard_keypose_snap=False,retargeting_executed=False,
            ground_truth_motion_displayed=False,rendering_executed=False,motion_quality_passed=False,elapsed_seconds=time.monotonic()-started)
        release.write_once(output/'execution.json',result);return result
    except BaseException as error:
        release.write_once(output/'failure.json',dict(schema=SCHEMA,status='failed_incomplete_partial_transfer',
            error_type=type(error).__name__,error=str(error),sources=sources,physics_success_not_asserted=True,elapsed_seconds=time.monotonic()-started));raise
    finally:
        if task is not None:task.gym.destroy_sim(task.sim)
        os.chdir(str(old_cwd))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=900);p.add_argument('--seed',type=int,default=390919);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--control-mode',choices=('trained_actor','zero_action'),default='trained_actor')
    p.add_argument('--sdf-filter',action='store_true',help='Experimental bounded Jacobian/SDF action filter; not original UniHSI')
    p.add_argument('--execute',action='store_true');p.add_argument('--validate-only',action='store_true');a=p.parse_args(argv)
    if a.validate_only:
        packet,arrays,refs=read_packet(a.packet);plan,points,indices,bary=make_transfer_plan(packet,arrays)
        mesh,transform,sources=load_world_scene(checked(packet['scene_mesh']))
        print(json.dumps(dict(status='cpu_packet_coordinate_and_heading_validated_not_physics',sources=refs+sources,plan=plan,
            world_scene_vertices=len(mesh.vertices),world_scene_faces=len(mesh.faces),world_bounds=mesh.bounds.tolist(),
            raw_scene_coordinate_contract=transform)));return
    if not a.execute:print(json.dumps(dict(status='not_executed',requires_explicit_execute=True)));return
    r=run(a.packet,a.output,a.frames,a.seed,a.gpu,a.control_mode,a.sdf_filter);print(json.dumps(dict(status=r['status'],frames=r['frames'])))


if __name__=='__main__':main()
