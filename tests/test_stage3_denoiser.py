"""Injection boundaries against an explicit tiny external-provider double."""
import pytest
import torch
from torch import nn
from hsi.stage3.denoiser import conditioned_class, load_current_adapter_state
from hsi.stage3.keypose_adapter import KeyposePacket


class ExternalDouble(nn.Module):
    def __init__(self, use_control):
        super().__init__()
        self.h_dim, self.num_heads, self.num_layers, self.dropout = 16, 4, 4, 0.
        self.use_control = use_control
        self.trans = nn.Module()
        self.trans.layers = nn.ModuleList([nn.Linear(16,16) for _ in range(4)])
        self.inter_scene = nn.ModuleList([nn.Identity(),nn.Identity()])
        self.forward_calls = 0

    def forward(self, x, timesteps, y):
        self.forward_calls += 1
        if y.get('fail'):
            raise ValueError('external forward failure')
        for i, layer in enumerate(self.trans.layers):
            x = layer(x)
            if (i+1)%2 == 0 and self.use_control and 'control_input' in y:
                x = self.inter_scene[i//2](x.transpose(0,1)).transpose(0,1)
        return x


def packet(batch=2):
    return KeyposePacket(root_xyz_yaw=torch.zeros(batch,2,4),joint_xyz=torch.zeros(batch,2,22,3),
        joint_mask=torch.ones(batch,2,22,dtype=torch.bool),semantic_id=torch.ones(batch,2,dtype=torch.long),
        ordinal=torch.tensor([[0,1]]).expand(batch,-1),confidence=torch.ones(batch,2),
        node_mask=torch.ones(batch,2,dtype=torch.bool))


@pytest.mark.parametrize('control',[False,True])
@pytest.mark.parametrize('active',[False,True])
def test_calls_actual_base_once_and_each_project_adapter_once(control,active):
    model=conditioned_class(ExternalDouble)(use_control=control,keypose_backend='native').eval()
    assert all(not p.requires_grad for name,p in model.named_parameters() if not name.startswith('keypose_adapters.'))
    calls=[]
    for i, adapter in enumerate(model.keypose_adapters):
        adapter.register_forward_hook(lambda m,x,y,index=i: calls.append(index))
    model.set_runtime_keypose(packet() if active else None)
    y={'control_input':{}} if control else {}
    result=model(torch.randn(2,2,16),torch.ones(2),y)
    assert result.shape==(2,2,16) and torch.isfinite(result).all()
    assert model.forward_calls==1 and calls==([0,1] if active else [])
    assert model._injection.get() is None


def test_context_restored_after_external_failure():
    model=conditioned_class(ExternalDouble)(use_control=False,keypose_backend='native')
    model.set_runtime_keypose(packet())
    with pytest.raises(ValueError,match='external forward'):
        model(torch.zeros(2,2,16),torch.ones(2),{'fail':True})
    assert model._injection.get() is None


def test_adapter_state_requires_exact_current_keys():
    model=conditioned_class(ExternalDouble)(use_control=False,keypose_backend='native')
    state={k:v for k,v in model.state_dict().items() if k.startswith('keypose_adapters.')}
    load_current_adapter_state(model,state)
    with pytest.raises(ValueError,match='key mismatch'):
        load_current_adapter_state(model,{})
    with pytest.raises(ValueError,match='key mismatch'):
        load_current_adapter_state(model,{**state,'unregistered':torch.zeros(1)})
