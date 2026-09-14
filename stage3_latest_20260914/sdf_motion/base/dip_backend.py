"""Strict, offline, inference-only wrapper around the official standalone DiP.

Use in an isolated process: the upstream repository uses generic top-level
``model``, ``utils``, ``diffusion`` and ``data_loaders`` module names.  No vendor
file is changed.  This is a kinematic HumanML3D backend, not a physics simulator
or a SMPL-X decoder.  SDF callbacks operate on the official predicted x0.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn


UPSTREAM_COMMIT = "ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778"
_LOCK = threading.RLock()
_NAMESPACES = ("model", "utils", "diffusion", "data_loaders")
_POSITION_BUFFERS = frozenset((
    "sequence_pos_encoder.pe", "embed_timestep.sequence_pos_encoder.pe",
))


def _file(path, label):
    result = Path(path).expanduser().resolve(strict=True)
    if not result.is_file():
        raise ValueError(f"{label} must be an existing file: {result}")
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_norm(path, label):
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.shape != (263,):
        raise ValueError(f"{label} must have shape (263,)")
    if value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"{label} must contain finite floating-point numbers")
    if label == "std" and not (value > 0).all():
        raise ValueError("std must be strictly positive")
    return torch.from_numpy(value.astype(np.float32, copy=True))


def _read_args(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("args.json must be an object")
    required = {
        "dataset": "humanml", "arch": "trans_dec", "text_encoder_type": "bert",
        "context_len": 20, "pred_len": 40, "diffusion_steps": 10,
        "latent_dim": 512, "layers": 8, "noise_schedule": "cosine",
        "multi_target_cond": False, "keyframe_cond_type": "",
        "emb_policy": "add", "emb_trans_dec": False, "mask_frames": True,
        "unconstrained": False, "sigma_small": True, "lambda_target_loc": 0.0,
    }
    for key, expected in required.items():
        if key not in value or value[key] != expected:
            raise ValueError(f"Unsupported DiP configuration {key}: expected {expected!r}")
    for key in ("cond_mask_prob", "pos_embed_max_len", "lambda_vel", "lambda_rcxyz", "lambda_fc"):
        if key not in value:
            raise ValueError(f"Missing official model argument: {key}")
    return SimpleNamespace(**value), value


class XYZOnlyRotation2xyz:
    """Avoid constructing unused SMPL; never pretend to perform mesh/FK decoding."""

    def __init__(self, device="cpu", dataset="humanml"):
        if dataset != "humanml":
            raise ValueError("XYZ-only compatibility is restricted to HumanML3D")
        self.device = device
        self.dataset = dataset
        # Official MDM.train/_apply access this even for hml_vec.
        self.smpl_model = nn.Identity()

    def __call__(self, x, mask=None, pose_rep=None, *args, **kwargs):
        if pose_rep != "xyz":
            raise RuntimeError("XYZ-only adapter does not implement rotation, SMPL or mesh decoding")
        return x


class _LocalBERT(nn.Module):
    """Same tokenizer/AutoModel forward and parameter names as official BERT."""

    def __init__(self, path):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.text_model = AutoModel.from_pretrained(str(path), local_files_only=True)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**inputs.to(self.text_model.device)).last_hidden_state
        return output, inputs.attention_mask.to(dtype=torch.bool)


class _UnavailableCLIP(ModuleType):
    """Import-only placeholder for a BERT-only model; CLIP execution is forbidden."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError("CLIP is unavailable and forbidden in this BERT-only DiP backend")


def _check_vendor_namespaces(vendor_root):
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in _NAMESPACES:
            continue
        paths = list(getattr(module, "__path__", ()))
        if getattr(module, "__file__", None):
            paths.append(module.__file__)
        if not paths or any(not Path(p).resolve().is_relative_to(vendor_root) for p in paths):
            raise RuntimeError(f"Foreign upstream namespace {name!r}; use an isolated DiP process")


@contextmanager
def _vendor_modules(vendor_root):
    """Import only the selected vendor, and suppress writes of vendor bytecode."""
    _check_vendor_namespaces(vendor_root)
    original_path = list(sys.path)
    original_bytecode = sys.dont_write_bytecode
    clip_proxy = None
    if "clip" not in sys.modules and importlib.util.find_spec("clip") is None:
        clip_proxy = _UnavailableCLIP("clip")
        sys.modules["clip"] = clip_proxy
    sys.path.insert(0, str(vendor_root))
    sys.dont_write_bytecode = True
    try:
        mdm = importlib.import_module("model.mdm")
        model_util = importlib.import_module("utils.model_util")
        samplers = importlib.import_module("utils.sampler_util")
        _check_vendor_namespaces(vendor_root)
        yield mdm, model_util, samplers
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
        if clip_proxy is not None and sys.modules.get("clip") is clip_proxy:
            del sys.modules["clip"]


@contextmanager
def _construction_bindings(mdm_module, bert_path):
    original_rot = mdm_module.Rotation2xyz
    original_bert = mdm_module.load_bert

    def load_local_bert(_ignored_upstream_path):
        bert = _LocalBERT(bert_path)
        bert.eval()
        bert.requires_grad_(False)
        return bert

    mdm_module.Rotation2xyz = XYZOnlyRotation2xyz
    mdm_module.load_bert = load_local_bert
    try:
        yield
    finally:
        mdm_module.Rotation2xyz = original_rot
        mdm_module.load_bert = original_bert


def _load_learned_state(model, checkpoint, use_ema):
    # No unsafe pickle fallback. These official checkpoints are tensor mappings.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a tensor mapping or model/model_avg mapping")
    if use_ema:
        if "model_avg" not in payload:
            raise ValueError("EMA requested, but checkpoint has no model_avg; no silent fallback")
        selected = "model_avg"
        state = payload[selected]
    elif "model" in payload:
        selected, state = "model", payload["model"]
    else:
        selected, state = "raw", payload
    if not isinstance(state, dict) or not state:
        raise ValueError("Selected checkpoint state must be a non-empty mapping")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("Selected checkpoint state must contain only named tensors")
    expected = model.state_dict()
    unexpected = sorted(set(state) - set(expected))
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
    filtered = {key: value for key, value in state.items() if key not in _POSITION_BUFFERS}
    missing = sorted(set(expected) - set(filtered))
    illegal = [key for key in missing if key not in _POSITION_BUFFERS and not key.startswith("clip_model.")]
    if illegal:
        raise ValueError(f"Missing learned checkpoint keys: {illegal}")
    for key, value in filtered.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"Checkpoint tensor shape mismatch: {key}")
        if value.dtype != expected[key].dtype:
            raise ValueError(f"Checkpoint tensor dtype mismatch: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
    report = model.load_state_dict(filtered, strict=False)
    if sorted(report.missing_keys) != missing or report.unexpected_keys:
        raise RuntimeError("State-load report disagrees with preflight validation")
    return {
        "selected": selected, "ema_used": selected == "model_avg",
        "loaded_tensor_count": len(filtered), "allowed_missing": missing,
        "regenerated_position_buffers": sorted(_POSITION_BUFFERS & set(expected)),
        "unexpected": [], "all_non_bert_learned_keys_loaded": True,
    }


class DiPBackend:
    """Official no-target DiP plus an optional x0 callback; no motion rewriting.

    ``mean`` and ``std`` are CPU float32 vectors of length 263. The caller must
    supply normalized HumanML3D prefix features, not absolute joints. No model
    is constructed successfully without validated pretrained learned weights.
    """

    def __init__(self, *, vendor_root, checkpoint, args_path, mean_path, std_path,
                 bert_path, device="cpu", guidance_scale=7.5, use_ema=True,
                 text_cache_size=32):
        self.vendor_root = Path(vendor_root).expanduser().resolve(strict=True)
        self.bert_path = Path(bert_path).expanduser().resolve(strict=True)
        if not self.vendor_root.is_dir() or not self.bert_path.is_dir():
            raise ValueError("vendor_root and bert_path must be existing directories")
        paths = {key: _file(value, key) for key, value in {
            "checkpoint": checkpoint, "args": args_path, "mean": mean_path, "std": std_path,
        }.items()}
        self.args, self.config = _read_args(paths["args"])
        self.mean, self.std = _read_norm(paths["mean"], "mean"), _read_norm(paths["std"], "std")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only explicit CPU or CUDA devices are supported")
        self.guidance_scale = float(guidance_scale)
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if self.guidance_scale != 1.0 and self.args.cond_mask_prob <= 0:
            raise ValueError("CFG requires a checkpoint trained with conditional masking")
        if isinstance(text_cache_size, bool) or not isinstance(text_cache_size, int) or text_cache_size < 1:
            raise ValueError("text_cache_size must be a positive integer")
        if not isinstance(use_ema, bool):
            raise ValueError("use_ema must be an explicit boolean")
        self._text_cache_size = text_cache_size
        self._text_cache = OrderedDict()
        self.text_cache_hits = self.text_cache_misses = 0
        required_sources = ("model/mdm.py", "model/BERT/BERT_encoder.py", "utils/model_util.py",
                            "utils/sampler_util.py", "diffusion/gaussian_diffusion.py", "diffusion/respace.py")
        source_hashes = {name: _sha256(_file(self.vendor_root / name, name)) for name in required_sources}
        with _LOCK, _vendor_modules(self.vendor_root) as (mdm, model_util, samplers):
            # get_model_args only needs dataset.num_actions; no dataset is loaded.
            data_contract = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))
            with torch.random.fork_rng(devices=[]), _construction_bindings(mdm, self.bert_path):
                self.model, self.diffusion = model_util.create_model_and_diffusion(self.args, data_contract)
            self.load_report = _load_learned_state(self.model, paths["checkpoint"], bool(use_ema))
            if (self.model.njoints, self.model.nfeats, self.model.context_len, self.model.pred_len) != (263, 1, 20, 40):
                raise RuntimeError("Vendor model does not implement the required DiP tensor contract")
            if self.model.data_rep != "hml_vec" or self.diffusion.num_timesteps != 10:
                raise RuntimeError("Vendor representation or diffusion schedule does not match DiP")
            self.model.to(self.device)  # Official _apply/train do not return self.
            self.model.eval()
            self.model.requires_grad_(False)
            self._uncached_encode_text = self.model.encode_text
            self.model.encode_text = self._encode_text_cached
            self.sampling_model = (samplers.ClassifierFreeSampleModel(self.model)
                                   if self.guidance_scale != 1.0 else self.model)
            self.sampling_model.eval()
        self.metadata = {
            "method": "official_standalone_DiP_no_target_with_optional_x0_guidance",
            "upstream_expected_commit": UPSTREAM_COMMIT, "vendor_root": str(self.vendor_root),
            "vendor_source_sha256": source_hashes, "assets": {
                key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()
            }, "bert_path": str(self.bert_path), "device": str(self.device),
            "fps": 20, "prefix_frames": 20, "prediction_frames": 40,
            "features": 263, "diffusion_steps": 10, "guidance_scale": self.guidance_scale,
            "load_report": self.load_report, "physics_simulation": False,
            "compatibility": ["construction-only XYZ-only Rotation2xyz, no SMPL/FK",
                              "import-only unavailable-CLIP proxy when CLIP is absent; CLIP execution forbidden",
                              "local-files-only DistilBERT, same parameter names and forward",
                              "cached text encoding; pretrained learned network unchanged"],
        }

    def _encode_text_cached(self, texts):
        key = tuple(texts)
        if key in self._text_cache:
            self.text_cache_hits += 1
            self._text_cache.move_to_end(key)
        else:
            self.text_cache_misses += 1
            with torch.no_grad():
                result = self._uncached_encode_text(list(texts))
            if not isinstance(result, tuple) or len(result) != 2 or not all(torch.is_tensor(t) for t in result):
                raise RuntimeError("Official BERT encode_text must return (token_embeddings, padding_mask)")
            self._text_cache[key] = tuple(t.detach().clone() for t in result)
            if len(self._text_cache) > self._text_cache_size:
                self._text_cache.popitem(last=False)
        # The official model/callback cannot mutate another sample's cache entry.
        return tuple(t.clone() for t in self._text_cache[key])

    def sample(self, prefix, text, seed, denoised_fn=None):
        """Return only the 40-frame native suffix; no alignment, smoothing or FPS change."""
        if not torch.is_tensor(prefix) or prefix.ndim != 4 or tuple(prefix.shape[1:]) != (263, 1, 20) or prefix.shape[0] < 1:
            raise ValueError("prefix must have shape [B,263,1,20] with B > 0")
        if not prefix.is_floating_point() or not torch.isfinite(prefix).all().item():
            raise ValueError("prefix must be finite floating-point normalized HumanML3D features")
        if not isinstance(text, (list, tuple)) or len(text) != prefix.shape[0] or not all(isinstance(t, str) and t.strip() for t in text):
            raise ValueError("text must contain one non-empty string per prefix")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if denoised_fn is not None and not callable(denoised_fn):
            raise ValueError("denoised_fn must be callable or None")
        batch = prefix.shape[0]
        shape = (batch, 263, 1, 40)
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with _LOCK, torch.inference_mode(False), torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            # Do not manual_seed_all(): unrelated GPU RNGs must not be touched.
            torch.random.default_generator.manual_seed(seed)
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            kwargs = {"y": {
                "prefix": prefix.detach().to(device=self.device, dtype=torch.float32).clone(),
                "text": list(text), "mask": torch.ones((batch, 1, 1, 40), dtype=torch.bool, device=self.device),
                "lengths": torch.full((batch,), 40, dtype=torch.long, device=self.device),
                "scale": torch.full((batch,), self.guidance_scale, dtype=torch.float32, device=self.device),
            }}
            noise = torch.randn(shape, device=self.device)
            suffix = self.diffusion.p_sample_loop(
                self.sampling_model, shape, noise=noise, clip_denoised=False,
                model_kwargs=kwargs, device=self.device, denoised_fn=denoised_fn,
                cond_fn=None, cond_fn_with_grad=False, skip_timesteps=0,
                init_image=None, progress=False, dump_steps=None, const_noise=False,
            )
            if not torch.is_tensor(suffix) or tuple(suffix.shape) != shape or not torch.isfinite(suffix).all().item():
                raise RuntimeError("Official sampler returned invalid/non-finite native suffix")
            return suffix.detach()
