"""Load frozen external weights and project-owned keypose controls."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from hsi.stage3.plan import _ClassifierFreeWrapper

def load_remogen_models(base_checkpoint: Path, adapter_checkpoint: Path, mvae_checkpoint: Path, device):
    """Load the released HSI base, compact P142 adapter, and released MVAE."""

    import yaml
    import tyro
    from model.mld_vae import AutoMldVae
    from mld.train_hsi_module_adapter import MLDArgs
    from mld.train_mvae import Args as MVAEArgs
    from hsi.stage3.denoiser import make_denoiser, load_current_adapter_state

    with (Path(base_checkpoint).parent / "args.yaml").open("r") as handle:
        args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(handle))
    denoiser_args = args.denoiser_args
    model = make_denoiser(
        **asdict(denoiser_args.model_args),
        use_control=True,
        nb_voxels=[32, 32, 32],
        scene_type="occ_two",
        freeze_released_model=True,
    ).to(device)
    base_payload = torch.load(base_checkpoint, map_location=device)
    incompatible = model.load_state_dict(base_payload["model_state_dict"], strict=False)
    # The released HSI checkpoint contains a legacy optional progress embedding
    # which the published rollout class itself loads with ``strict=False``.
    allowed_legacy = {"embed_timeprogress.weight", "embed_timeprogress.bias"}
    unexpected = [
        name
        for name in incompatible.unexpected_keys
        if not name.startswith("keypose_adapters.") and name not in allowed_legacy
    ]
    if unexpected:
        raise RuntimeError("unexpected released keys: {}".format(unexpected[:8]))
    adapter_payload = torch.load(adapter_checkpoint, map_location=device)
    state = adapter_payload.get("adapter_state_dict", adapter_payload)
    load_current_adapter_state(model, state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    with (Path(mvae_checkpoint).parent / "args.yaml").open("r") as handle:
        vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(handle))
    vae = AutoMldVae(**asdict(vae_args.model_args)).to(device)
    vae_payload = torch.load(mvae_checkpoint, map_location=device)
    vae_state = vae_payload["model_state_dict"]
    if "latent_mean" not in vae_state:
        vae_state["latent_mean"] = torch.tensor(0.0, device=device)
    if "latent_std" not in vae_state:
        vae_state["latent_std"] = torch.tensor(1.0, device=device)
    vae.load_state_dict(vae_state)
    vae.latent_mean = vae_state["latent_mean"]
    vae.latent_std = vae_state["latent_std"]
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return denoiser_args, _ClassifierFreeWrapper(model), vae

