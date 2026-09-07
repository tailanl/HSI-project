"""Project keypose injection through public PyTorch hooks.

The externally installed ReMoGen class owns and executes its forward method.
We do not vendor or transcribe that third-party forward implementation. The
same project adapters act after each alternating scene-control block; when
scene control is absent, they act directly after the corresponding layer.
Checkpoint parameter names are unchanged.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache

import torch
from torch import nn

from hsi.stage3.keypose_adapter import KeyposePacket, ReMoGenOrderedKeyposeAdapter


@lru_cache(maxsize=4)
def conditioned_class(base_class):
    """Return a subclass of the real external denoiser, without copying it."""
    if not issubclass(base_class, nn.Module):
        raise TypeError("The external denoiser must be a torch Module")

    class OrderedKeyposeControlDenoiser(base_class):
        def __init__(self, *args, keypose_num_semantics=16, keypose_max_ordinals=8,
                     keypose_max_joints=22, keypose_backend="remogen",
                     keypose_initial_scale=.05, freeze_released_model=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.keypose_adapters = nn.ModuleList([
                ReMoGenOrderedKeyposeAdapter(hidden_dim=self.h_dim, keypose_dim=self.h_dim,
                    num_semantics=int(keypose_num_semantics), max_ordinals=int(keypose_max_ordinals),
                    max_joints=int(keypose_max_joints), num_heads=self.num_heads,
                    dropout=self.dropout, backend=keypose_backend,
                    initial_residual_scale=float(keypose_initial_scale))
                for _ in range(max(1, self.num_layers // 2))])
            self._runtime_keypose_packet = None
            self._injection = ContextVar("hsi_keypose_injection", default=None)
            self._keypose_hooks = []
            for layer_index, layer in enumerate(self.trans.layers):
                if (layer_index + 1) % 2:
                    continue
                slot = layer_index // 2
                self._keypose_hooks.append(layer.register_forward_hook(self._layer_hook(slot)))
                if self.use_control:
                    self._keypose_hooks.append(self.inter_scene[slot].register_forward_hook(self._scene_hook(slot)))
            if freeze_released_model:
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.keypose_adapters.parameters():
                    parameter.requires_grad_(True)

        def _layer_hook(self, slot):
            def inject(module, inputs, output):
                current = self._injection.get()
                if current is None or current["packet"] is None or current["scene"]:
                    return output
                return self.keypose_adapters[slot](output.transpose(0, 1), current["packet"]).transpose(0, 1)
            return inject

        def _scene_hook(self, slot):
            def inject(module, inputs, output):
                current = self._injection.get()
                if current is None or current["packet"] is None:
                    return output
                return self.keypose_adapters[slot](output, current["packet"])
            return inject

        def set_runtime_keypose(self, packet: KeyposePacket | None):
            if packet is not None:
                packet.validate(require_sorted_ordinals=True)
            self._runtime_keypose_packet = packet

        def trainable_parameter_report(self):
            return {"total": sum(p.numel() for p in self.parameters()),
                    "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                    "keypose": sum(p.numel() for p in self.keypose_adapters.parameters())}

        def forward(self, x_t, timesteps, y=None):
            conditioning = {} if y is None else y
            control = conditioning.get("control_input", {})
            packet = control.get("keypose_packet", self._runtime_keypose_packet)
            if packet is not None and packet.batch_size != x_t.shape[0]:
                raise ValueError("Keypose and latent batch sizes differ")
            token = self._injection.set({"packet": packet,
                "scene": bool(self.use_control and "control_input" in conditioning)})
            try:
                return super().forward(x_t, timesteps, conditioning)
            finally:
                self._injection.reset(token)

    return OrderedKeyposeControlDenoiser


def make_denoiser(*args, **kwargs):
    """Construct the installed ReMoGen base and attach project-owned adapters."""
    from model.mld_adapter_hsi import DenoiserControlTransformer
    return conditioned_class(DenoiserControlTransformer)(*args, **kwargs)


def load_current_adapter_state(model, state):
    """Strictly load the current adapter; historical un-gated weights are not accepted."""
    expected = {key for key in model.state_dict() if key.startswith("keypose_adapters.")}
    if set(state) != expected:
        raise ValueError(f"Adapter checkpoint key mismatch: missing={len(expected-set(state))}, "
                         f"extra={len(set(state)-expected)}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(key.startswith("keypose_adapters.") for key in result.missing_keys):
        raise ValueError("Current adapter checkpoint failed strict adapter-only validation")
