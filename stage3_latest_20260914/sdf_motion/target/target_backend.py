"""Strict offline adapter for the author's multi-target DiP checkpoint.

Separate from the frozen no-target backend: no config/weight guard is patched.
The learned goal encoder, concat embedding and all checkpoint parameters must
load exactly. Goal coordinates are prepared explicitly by the caller.
"""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from guarded_refine import AGENT
from hsi.common.artifacts import artifact,require
from dip_backend import (DiPBackend,_LOCK,_vendor_modules,_construction_bindings,
                         _load_learned_state,_read_norm)
from fetch_target_weights import SHA,SIZE,ARGS_BLOB

METHOD=Path(__file__).resolve().parents[1]
BASE=AGENT/'methods/closd_unihsi_sdf_stage3_20260912'


def read_config(path):
    raw=Path(path).read_bytes()
    require(len(raw)==1686 and hashlib.sha1(b'blob 1686\0'+raw).hexdigest()==ARGS_BLOB,
            'Only pinned official target configuration accepted')
    value=json.loads(raw)
    expected=dict(dataset='humanml',arch='trans_dec',text_encoder_type='bert',context_len=20,pred_len=40,
        diffusion_steps=10,latent_dim=512,layers=8,multi_target_cond=True,multi_encoder_type='single',
        target_enc_layers=1,emb_policy='concat',noise_schedule='cosine',mask_frames=True,
        emb_trans_dec=False,lambda_target_loc=1.,sigma_small=True,unconstrained=False)
    require(all(value.get(k)==v for k,v in expected.items()),'Unregistered multi-target architecture')
    return SimpleNamespace(**value),value


def validate_goal(value,names,heading,all_names,batch,device):
    extended=list(all_names)+['traj','heading']
    require(isinstance(value,torch.Tensor) and value.shape==(batch,len(extended),3)
            and value.is_floating_point() and torch.isfinite(value).all(),'Finite [B,goal_slots,3] goals required')
    require(isinstance(names,list) and len(names)==batch and all(isinstance(x,list) and 1<=len(x)<=2
            and len(x)==len(set(x)) and all(k in ('traj','pelvis') for k in x) for x in names),
            'Only explicit traj/pelvis goals supported in this first adapter')
    require(isinstance(heading,torch.Tensor) and heading.shape==(batch,) and heading.dtype==torch.bool,
            'Explicit bool heading validity required')
    allowed=torch.zeros_like(value,dtype=torch.bool)
    for i,selected in enumerate(names):
        for name in selected:allowed[i,extended.index(name)]=True
        if bool(heading[i]):allowed[i,-1,0]=True
        if 'traj' in selected:require(float(value[i,-2,1])==0.,'Trajectory slot must have zero local Y')
    require(torch.count_nonzero(value[~allowed])==0,'Inactive goal slots must be exactly zero')
    require(float(value.abs().max())<=20.,'Scene-local metre/radian target exceeds bounded adapter domain')
    return value.detach().to(device=device,dtype=torch.float32).clone(),heading.to(device=device).clone()


class TargetDiPBackend(DiPBackend):
    def __init__(self,*,device='cpu',guidance_scale=2.5):
        self.vendor_root=BASE/'vendor/mdm';self.bert_path=BASE/'weights/distilbert'
        root=METHOD/'weights/multi_target'
        self.args,self.config=read_config(root/'args.json')
        model_record=artifact(root/'model000300000.pt')
        require(model_record['bytes']==SIZE and model_record['sha256']==SHA,'Official multi-target checkpoint mismatch')
        bert_record=artifact(self.bert_path/'model.safetensors')
        require(bert_record['sha256']=='5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063',
                'Frozen official text encoder changed')
        self.mean=_read_norm(BASE/'weights/dip/Mean.npy','mean')
        self.std=_read_norm(BASE/'weights/dip/Std.npy','std')
        self.device=torch.device(device)
        require(self.device.type in ('cpu','cuda'),'Explicit CPU/CUDA device required')
        require(type(guidance_scale) in (int,float) and math.isfinite(guidance_scale)
                and 1<=guidance_scale<=10,'Bounded CFG required')
        self.guidance_scale=float(guidance_scale)
        self._text_cache_size=32;self._text_cache=OrderedDict()
        self.text_cache_hits=self.text_cache_misses=0
        with _LOCK,_vendor_modules(self.vendor_root) as (mdm,model_util,samplers):
            with torch.random.fork_rng(devices=[]),_construction_bindings(mdm,self.bert_path):
                self.model,self.diffusion=model_util.create_model_and_diffusion(self.args,
                    SimpleNamespace(dataset=SimpleNamespace(num_actions=1)))
            self.load_report=_load_learned_state(self.model,root/'model000300000.pt',True)
            require((self.model.njoints,self.model.nfeats,self.model.context_len,self.model.pred_len)==(263,1,20,40)
                    and self.diffusion.num_timesteps==10 and self.model.multi_target_cond is True,
                    'Native multi-target DiP contract mismatch')
            require(hasattr(self.model,'embed_target_cond'),'Missing learned spatial goal encoder')
            self.model.to(self.device);self.model.eval();self.model.requires_grad_(False)
            self._uncached_encode_text=self.model.encode_text
            self.model.encode_text=self._encode_text_cached
            self.sampling_model=(samplers.ClassifierFreeSampleModel(self.model)
                                 if self.guidance_scale!=1 else self.model)
            self.sampling_model.eval()
        self.metadata=dict(method='official_CLOSD_multi_target_DiP_not_physics',checkpoint=model_record,
            configuration=artifact(root/'args.json'),bert=bert_record,
            mean=artifact(BASE/'weights/dip/Mean.npy'),std=artifact(BASE/'weights/dip/Std.npy'),
            model_load=self.load_report,all_goal_joint_names=list(self.model.all_goal_joint_names),
            motion_parameter_count=sum(p.numel() for n,p in self.model.named_parameters() if not n.startswith('clip_model.')),
            target_encoder_parameter_count=sum(p.numel() for p in self.model.embed_target_cond.parameters()),
            cfg=self.guidance_scale,diffusion_steps=10,prefix_frames=20,prediction_frames=40,
            pretrained_target_encoder_loaded=True,embedding_policy='concat',training=False,physics_simulation=False)

    def sample(self,prefix,text,seed,*,target_cond,target_joint_names,is_heading,denoised_fn=None,target_uncond=False):
        require(isinstance(prefix,torch.Tensor) and prefix.ndim==4 and prefix.shape[1:]==(263,1,20)
                and prefix.shape[0]>0 and prefix.is_floating_point() and torch.isfinite(prefix).all(),
                'Finite normalized 20-frame HumanML prefix required')
        batch=prefix.shape[0]
        require(isinstance(text,list) and len(text)==batch and all(isinstance(x,str) and x.strip() for x in text),
                'One explicit text per sample required')
        require(type(seed) is int and 0<=seed<2**63 and type(target_uncond) is bool,'Invalid seed/target mask')
        require(denoised_fn is None or callable(denoised_fn),'Invalid clean-sample callback')
        goal,heading=validate_goal(target_cond,target_joint_names,is_heading,list(self.model.all_goal_joint_names),batch,self.device)
        devices=[] if self.device.type=='cpu' else [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        shape=(batch,263,1,40)
        with _LOCK,torch.inference_mode(False),torch.random.fork_rng(devices=devices),torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            for index in devices:torch.cuda.default_generators[index].manual_seed(seed)
            kwargs={'y':dict(prefix=prefix.detach().to(device=self.device,dtype=torch.float32).clone(),
                text=list(text),mask=torch.ones((batch,1,1,40),dtype=torch.bool,device=self.device),
                lengths=torch.full((batch,),40,dtype=torch.long,device=self.device),
                scale=torch.full((batch,),self.guidance_scale,dtype=torch.float32,device=self.device),
                target_cond=goal,target_joint_names=[list(x) for x in target_joint_names],
                is_heading=heading,target_uncond=target_uncond)}
            suffix=self.diffusion.p_sample_loop(self.sampling_model,shape,noise=torch.randn(shape,device=self.device),
                clip_denoised=False,model_kwargs=kwargs,device=self.device,denoised_fn=denoised_fn,
                cond_fn=None,cond_fn_with_grad=False,skip_timesteps=0,init_image=None,progress=False,
                dump_steps=None,const_noise=False)
        require(suffix.shape==shape and torch.isfinite(suffix).all(),'Invalid actual target-conditioned output')
        return suffix.detach()
