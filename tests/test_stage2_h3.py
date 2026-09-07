"""Real H3 caller code against explicit CPU Comfy interface doubles, no model run."""
from contextlib import nullcontext
from types import SimpleNamespace
import copy
import sys
import numpy as np
from PIL import Image
import pytest
import torch
from hsi.stage2 import h3


def fake_modules(monkeypatch, *, latent_shape=h3.VISUAL_SHAPE, decoded_shape=h3.DECODED_SHAPE, patch=True):
    calls=[]
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:calls.append('synchronize'))
    class Encoder:
        def tokenize(self,prompt,*,minimax_ref_items):
            assert minimax_ref_items[0]['type']=='image'
            assert tuple(minimax_ref_items[0]['data'].shape)==(1,512,512,3)
            calls.append(('encode_prompt',prompt));return {'tokens':'actual-call-fixture'}
        def encode_from_tokens_scheduled(self,tokens):return [[torch.ones(1),{}]]
    class VAE:
        def __init__(self,sd):pass
        def encode(self,pixels):return torch.zeros(latent_shape)
        def decode(self,latent):return torch.full(decoded_shape,.4)
    class Model:
        def __init__(self):self.patches={}
        def get_model_object(self,name):assert name=='model_sampling';return self
    def load_lora(model,clip,state,strength,textstrength):
        calls.append(('lora',state,strength,textstrength))
        if patch:model.patches[str(state)]=[strength]
        return model,None
    class Nested:
        def __init__(self,values):self.values=values
        def unbind(self):return self.values
    class Guide:
        def __init__(self,model):pass
        def set_conds(self,values):calls.append(('condition',values[0][1]))
    class Sampler:
        @staticmethod
        def execute(noise,guide,sampler,sigma,latent):
            assert sampler=='er_sde' and len(sigma)==9
            calls.append(('sample_seed',noise.seed))
            return ({'samples':Nested(tuple(value+.1 for value in latent['samples'].values))},)
    def set_values(condition,values):return [[condition[0][0],{**condition[0][1],**values}]]
    def shift(model,visual,audio):
        calls.append(('shift',visual,audio));return (model,)
    modules={'torch':torch,'numpy':np,
        'comfy.sd':SimpleNamespace(load_clip=lambda **kw:Encoder(),CLIPType=SimpleNamespace(MINIMAX='minimax'),
            VAE=VAE,load_diffusion_model=lambda *a,**kw:Model(),load_lora_for_models=load_lora),
        'comfy.utils':SimpleNamespace(load_torch_file=lambda path,**kw:path),
        'comfy.model_management':SimpleNamespace(unload_all_models=lambda:calls.append('unload'),intermediate_device=lambda:'cpu'),
        'comfy.samplers':SimpleNamespace(calculate_sigmas=lambda model,schedule,steps:torch.linspace(1,0,steps+1),sampler_object=lambda name:name),
        'comfy.nested_tensor':SimpleNamespace(NestedTensor=Nested),
        'node_helpers':SimpleNamespace(conditioning_set_values=set_values),
        'comfy_extras.nodes_minimax_h3':SimpleNamespace(MiniMaxH3SigmaShift=SimpleNamespace(execute=shift)),
        'comfy_extras.nodes_custom_sampler':SimpleNamespace(Guider_Basic=Guide,SamplerCustomAdvanced=Sampler,
            Noise_RandomNoise=lambda seed:SimpleNamespace(seed=seed))}
    return modules,calls


def run_fixture(tmp_path,monkeypatch,**options):
    modules,calls=fake_modules(monkeypatch,**options)
    condition=tmp_path/'condition.png';Image.new('RGB',(512,512),(140,140,140)).save(condition)
    output=tmp_path/'image.png';clock=h3.InferenceClock()
    actual=h3._infer(modules,{key:key for key in ('text_encoder','vae','transformer','distill_lora','image_lora')},
        'exact fixture prompt',77,condition,output,clock)
    return actual,calls,output,clock


def test_entire_direct_caller_reference_encode_adapters_sampling_and_decode(tmp_path,monkeypatch):
    actual,calls,output,clock=run_fixture(tmp_path,monkeypatch)
    assert actual['latent_shape']==[1,24,1,32,32]
    assert actual['auxiliary_audio_latent_shape']==[1,32,2,2]
    assert actual['decoded_tensor_shape']==[1,1,512,512,3]
    assert actual['adapter_applied_weight_patch_counts']=={'distill_lora':1,'image_lora':1}
    assert ('lora','distill_lora',.75,0) in calls and ('lora','image_lora',.5,0) in calls
    assert ('shift',12.,3.) in calls and ('sample_seed',77) in calls
    with Image.open(output) as image:
        assert image.size==(512,512) and image.mode=='RGB' and np.asarray(image).mean()==102
    assert {'sampling_seconds','reference_encode_seconds','decode_and_save_seconds'}<=clock.stages.keys()


@pytest.mark.parametrize('options',[{'latent_shape':(1,24,5,32,32)}, {'decoded_shape':(1,2,512,512,3)}, {'patch':False}])
def test_no_multiframe_latent_or_decoder_or_silent_adapter_failure(tmp_path,monkeypatch,options):
    with pytest.raises(ValueError):run_fixture(tmp_path,monkeypatch,**options)


def test_h3_fresh_worker_refuses_preimported_torch(tmp_path):
    assert 'torch' in sys.modules
    with pytest.raises(ValueError,match='fresh worker'):
        h3._external_modules(tmp_path,0)
