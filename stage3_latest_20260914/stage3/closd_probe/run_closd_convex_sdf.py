"""One bounded original-loop ON intervention; OFF is frozen attempt03.

Only the official clean-future denoised_fn is added. Physical stepping,
learned models, static initial state, original task and actual feedback remain.
"""
import argparse,json,os,shutil,sys,traceback
from pathlib import Path
from types import SimpleNamespace

import run_closd_smoke as frozen
ROOT=frozen.ROOT;VENDOR=frozen.VENDOR


def execute(args,report):
    # Original import order and process-only compatibility, never vendor edits.
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        TORCH_EXTENSIONS_DIR=str(ROOT/'runtime/torch_extensions'),MPLCONFIGDIR=str(args.out/'matplotlib_cache'),MAX_JOBS='2')
    sys.path[:0]=[str(ROOT/'runtime/packages'),str(VENDOR),str(VENDOR/'closd')]
    import numpy as np
    for name,value in (('float',float),('int',int),('bool',bool),('complex',complex),('object',object),('str',str),('unicode',str)):
        if name not in np.__dict__:setattr(np,name,value)
    from isaacgym import gymapi
    import torch
    from closd.env.tasks.closd_multitask import CLoSDMultiTask
    from convex_reference_sdf import ConvexSceneProxy,BoundedGuide,decode_world_future
    baseline=json.loads((args.baseline/'execution.json').read_text())
    if baseline['status']!='completed_physics_loop_not_quality_acceptance' or baseline['steps_recorded']!=180 or baseline['seed']!=390918:
        raise ValueError('Exact frozen OFF baseline required')
    probe=json.loads((args.geometry.parent/'execution.json').read_text())
    if frozen.binding(args.geometry)!=probe['geometry']:raise ValueError('Geometry not CPU-verified')
    geometry=np.load(args.geometry,allow_pickle=False)
    planes=[geometry['planes_%d'%i].copy() for i in range(6)]
    original_get=CLoSDMultiTask.get_mdm_next_planning_horizon
    calls=[]
    def guided_get(task):
        original_build=task.build_completion_input;original_sample=task.sample_fn;context={}
        def record_context(*pos,**kw):
            result=original_build(*pos,**kw)
            aux,recon=result
            context['prefix']=aux['prefix'].detach().clone()
            context['recon']={k:v.detach().clone() for k,v in recon.items()}
            return result
        def guided_sample(*pos,**kw):
            if kw.get('denoised_fn') is not None:raise ValueError('Do not overwrite another guide')
            prefix=context['prefix'];before=prefix.clone();pose=task._rigid_body_state[24,:7].detach().clone()
            scene=ConvexSceneProxy(planes,pose[:3],pose[3:7])
            guide=BoundedGuide(scene,lambda future:decode_world_future(task.rep,prefix,future,context['recon']))
            actual_input=kw['model_kwargs']['y']['prefix'].detach().clone()
            kw['denoised_fn']=guide
            result=original_sample(*pos,**kw)
            if guide.calls!=10:raise ValueError('Official ten denoising callback calls required')
            if not torch.equal(prefix,before) or not torch.equal(actual_input,kw['model_kwargs']['y']['prefix']):
                raise ValueError('Executed prefix was modified')
            if not torch.equal(result[...,:20],actual_input):raise ValueError('Returned original prefix changed')
            calls.append(dict(control_frame=int(task.frame_idx),callbacks=guide.calls,target_pose_xyzw=pose.cpu().tolist(),
                prefix_bit_exact=True,trace=guide.trace))
            return result
        task.build_completion_input=record_context;task.sample_fn=guided_sample
        try:return original_get(task)
        finally:task.build_completion_input=original_build;task.sample_fn=original_sample
    CLoSDMultiTask.get_mdm_next_planning_horizon=guided_get
    report['intervention']='CLOSD+convex-proxy SDF; only official clean-future callback'
    try:
        base_args=SimpleNamespace(out=args.out,steps=180,seed=390918,task='bench',smpl_root=args.smpl_root,validate_only=False)
        frozen.run(base_args,report)
    finally:
        CLoSDMultiTask.get_mdm_next_planning_horizon=original_get
        frozen.write_json(args.out/'guidance_trace.json',dict(calls=calls,total_callbacks=sum(c['callbacks'] for c in calls),
            fixed_parameters=dict(iterations=2,learning_rate=.01,sdf_weight=100.,trust_weight=1.,trust_radius_normalized_features=.08,
                                  nonincrease_fallback=True,penetration_tolerance_m=.003),not_fullsurface=True,not_sdk_vhacd=True))
        report['guidance_trace']=frozen.binding(args.out/'guidance_trace.json')
    off=np.load(args.baseline/'physical_rollout.npz',allow_pickle=False)
    on=np.load(args.out/'physical_rollout.npz',allow_pickle=False)
    initial_equal=np.array_equal(off['initial_rigid_body_state'],on['initial_rigid_body_state'])
    off_cfg=json.loads((args.baseline/'resolved_config.json').read_text());on_cfg=json.loads((args.out/'resolved_config.json').read_text())
    off_cfg.pop('dependencies_path');on_cfg.pop('dependencies_path')
    if not initial_equal or off_cfg!=on_cfg:raise ValueError('OFF/ON initial physical state or original configuration differs')
    snapshot=args.out/'intervention_source_snapshot.py';shutil.copyfile(__file__,snapshot)
    report['sources']+= [frozen.binding(p) for p in [snapshot,ROOT/'convex_reference_sdf.py',args.geometry]]
    report.update(schema='agent9.closd.convex_proxy_sdf_loop.v1',sdf_connected=True,
        sdf_kind='six_convex_original_sofa_asset_halfspace_proxy_plus_ground',full_surface_constraint=False,
        matched_off=frozen.binding(args.baseline/'execution.json'),initial_physics_bit_exact=initial_equal,
        original_config_equal_except_private_dependencies_path=True,proxy_metadata=probe['metadata'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--smpl-root',type=Path,default=Path('/home/lzsh2025/kimodo-viser/TSTMotion/datasets/smpl'))
    args=parser.parse_args();args.out=args.out.resolve();args.baseline=args.baseline.resolve();args.geometry=args.geometry.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='agent9.closd.convex_proxy_sdf_loop.v1',task='bench',seed=390918,training=False,
                quality_accepted=False,stage12_connected=False,sdf_connected=True);rc=0
    try:execute(args,report)
    except Exception as error:report.update(status='failed_attempt',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());rc=1
    frozen.write_json(args.out/'execution.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','proxy_metadata','traceback')},indent=2),flush=True)
    raise SystemExit(rc)
