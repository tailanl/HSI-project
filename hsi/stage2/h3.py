"""Direct, queue-free H3 image inference using an external ComfyUI install.

This project-owned caller does not ship ComfyUI, model implementations, model
weights, or a third-party workflow. The current recipe uses reference-image
conditioning, a single visual temporal latent and eight denoising steps.
It records actual tensors; a successful image is not a verified keypose.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image

from hsi.common.artifacts import artifact, require, verified, write_once


VISUAL_SHAPE = (1, 24, 1, 32, 32)
AUDIO_SHAPE = (1, 32, 2, 2)
DECODED_SHAPE = (1, 1, 512, 512, 3)
COMFY_FILES = (
    "comfy_extras/nodes_minimax_h3.py", "comfy_extras/nodes_custom_sampler.py",
    "comfy/ldm/minimax/model.py", "comfy/ldm/minimax/vae.py",
    "comfy/text_encoders/minimax.py",
)


@dataclass
class InferenceClock:
    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, label):
        begin = time.monotonic()
        try:
            yield
        finally:
            self.stages[label] = time.monotonic() - begin


def check_shape(tensor, expected, label):
    require(tuple(tensor.shape) == tuple(expected),
            f"{label} changed shape: {tuple(tensor.shape)}; expected {expected}")


def _external_modules(comfy_root: Path, gpu: int):
    """Configure one isolated worker; never start a server or a task queue."""
    require(type(gpu) is int and gpu >= 0, "GPU must be a nonnegative physical index")
    require("comfy" not in sys.modules and "torch" not in sys.modules,
            "Run H3 in a fresh worker process, before importing torch/ComfyUI")
    root = comfy_root.resolve(strict=True)
    require((root / "comfy" / "sd.py").is_file(), "ComfyUI install lacks comfy/sd.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for key, value in {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "NUMPY_MADVISE_HUGEPAGE": "0"}.items():
        os.environ.setdefault(key, value)
    # Only the explicitly supplied third-party installation is added; there
    # are no implicit workspace roots or loads from historical method trees.
    sys.path.insert(0, str(root))
    options = importlib.import_module("comfy.options")
    require(Path(options.__file__).resolve().is_relative_to(root),
            "A different ComfyUI installation shadowed the configured runtime")
    options.enable_args_parsing(False)
    cli = importlib.import_module("comfy.cli_args")
    settings = cli.args
    settings.enable_dynamic_vram = False
    settings.disable_dynamic_vram = True
    settings.enable_triton_backend = True
    settings.use_pytorch_cross_attention = True
    settings.preview_method = cli.LatentPreviewMethod.NoPreviews
    require(not cli.enables_dynamic_vram(), "Current H3 policy requires standard CPU weight offload")
    modules = {name: importlib.import_module(name) for name in (
        "torch", "numpy", "comfy.sd", "comfy.utils", "comfy.nested_tensor",
        "comfy.samplers", "comfy.model_management", "node_helpers",
        "comfy_extras.nodes_minimax_h3", "comfy_extras.nodes_custom_sampler",
    )}
    modules["torch"].set_num_threads(4)
    require(modules["torch"].cuda.is_available(), "H3 image inference requires the configured CUDA device")
    return modules


def _infer(modules, checkpoints, prompt, seed, condition, destination, clock):
    torch, np = modules["torch"], modules["numpy"]
    sd, utils = modules["comfy.sd"], modules["comfy.utils"]
    memory = modules["comfy.model_management"]
    sampling = modules["comfy.samplers"]
    custom = modules["comfy_extras.nodes_custom_sampler"]
    with Image.open(condition) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (512, 512),
                "H3 requires the exact selected RGB 512-square PNG, without implicit resizing")
        pixels = torch.as_tensor(np.array(image, copy=True)).to(dtype=torch.float32)[None] / 255
    with torch.inference_mode():
        with clock.measure("text_encoder_load_seconds"):
            encoder = sd.load_clip(ckpt_paths=[checkpoints["text_encoder"]],
                                   clip_type=sd.CLIPType.MINIMAX, embedding_directory=[])
        with clock.measure("text_vision_encode_seconds"):
            tokens = encoder.tokenize(prompt, minimax_ref_items=[{"type": "image", "data": pixels}])
            conditioning = encoder.encode_from_tokens_scheduled(tokens)
        del encoder, tokens
        gc.collect()
        memory.unload_all_models()
        with clock.measure("vae_load_seconds"):
            decoder = sd.VAE(sd=utils.load_torch_file(checkpoints["vae"], safe_load=True))
        with clock.measure("reference_encode_seconds"):
            reference = decoder.encode(pixels)
            check_shape(reference, VISUAL_SHAPE, "reference latent")
            conditioning = modules["node_helpers"].conditioning_set_values(conditioning, {
                "minimax_refs": [{"kind": "image", "latent": reference, "latent_h": 32, "latent_w": 32}]})
            require(all("minimax_keyframes" not in row[1] for row in conditioning),
                    "Keyframe locking is not permitted in native reference-image mode")
        patch_counts = {}
        with clock.measure("transformer_and_adapters_load_seconds"):
            transformer = sd.load_diffusion_model(checkpoints["transformer"], model_options={})
            require(transformer is not None, "Unrecognized H3 transformer checkpoint")
            for role, strength in (("distill_lora", .75), ("image_lora", .5)):
                state = utils.load_torch_file(checkpoints[role], safe_load=True)
                count_before = sum(map(len, transformer.patches.values()))
                transformer, _ = sd.load_lora_for_models(transformer, None, state, strength, 0)
                patch_counts[role] = sum(map(len, transformer.patches.values())) - count_before
                require(patch_counts[role] > 0, f"No actual transformer patches were applied for {role}")
                del state
            transformer = modules["comfy_extras.nodes_minimax_h3"].MiniMaxH3SigmaShift.execute(
                transformer, 12.0, 3.0)[0]
        intermediate = memory.intermediate_device()
        visual = torch.zeros(VISUAL_SHAPE, device=intermediate)
        auxiliary = torch.zeros(AUDIO_SHAPE, device=intermediate)
        latent = {"samples": modules["comfy.nested_tensor"].NestedTensor((visual, auxiliary))}
        guide = custom.Guider_Basic(transformer)
        guide.set_conds(conditioning)
        sigma = sampling.calculate_sigmas(transformer.get_model_object("model_sampling"), "sgm_uniform", 8).cpu()
        with clock.measure("sampling_seconds"):
            output = custom.SamplerCustomAdvanced.execute(custom.Noise_RandomNoise(seed), guide,
                sampling.sampler_object("er_sde"), sigma, latent)[0]
            torch.cuda.synchronize()
        visual_result, audio_result = output["samples"].unbind()
        check_shape(visual_result, VISUAL_SHAPE, "sampled visual latent")
        check_shape(audio_result, AUDIO_SHAPE, "sampled auxiliary audio latent")
        with clock.measure("decode_and_save_seconds"):
            decoded = decoder.decode(visual_result)
            check_shape(decoded, DECODED_SHAPE, "decoded image")
            require(bool(torch.isfinite(decoded).all()), "H3 decoder returned nonfinite pixels")
            rgb = (decoded[0, 0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(destination, format="PNG")
    return {"latent_shape": list(visual_result.shape),
            "auxiliary_audio_latent_shape": list(audio_result.shape),
            "decoded_tensor_shape": list(decoded.shape),
            "adapter_applied_weight_patch_counts": patch_counts}


def generate(job_path, output, *, comfy_root, models_root, gpu=0):
    """Generate exactly one image and validate the bound native-image receipt.

    Run this entry in its own process. No model download, video extraction,
    background/keypose pass, or Memory success credit is performed here.
    """
    from hsi.stage2 import image_contract as contract, image_job

    begin = time.monotonic()
    clock = InferenceClock()
    job_path, output = Path(job_path).resolve(strict=True), Path(output).resolve()
    comfy_root, models_root = Path(comfy_root).resolve(strict=True), Path(models_root).resolve(strict=True)
    job = image_job.validate_job(job_path)
    sources = {"sampler": artifact(__file__), "input_contract": artifact(contract.__file__),
               "job_builder": artifact(image_job.__file__), "job": artifact(job_path),
               **{name: artifact(comfy_root / name) for name in COMFY_FILES}}
    manifests, checkpoints = {}, {}
    with clock.measure("model_integrity_preflight_seconds"):
        for role in contract.MODEL_COMPONENTS:
            manifest = models_root / "manifests" / (role + ".json")
            value = contract.read_sealed(manifest)
            require(len(value["weights"]) == 1, f"One checkpoint is required for {role}")
            manifests[role] = artifact(manifest)
            checkpoints[role] = value["weights"][0]["path"]
        contract._models(manifests)
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_once(output / "source_preflight.json", sources)
        write_once(output / "job.json", job)
        condition = output / "h3_condition_image.png"
        result_image = output / "h3_image.png"
        shutil.copyfile(job["condition_source"], condition)
        modules = _external_modules(comfy_root, gpu)
        tensors = _infer(modules, checkpoints, job["prompt"], job["seed"], condition, result_image, clock)
        for record in sources.values():
            verified(record)
        require(image_job.validate_job(job_path) == job, "Job changed during H3 inference")
        contract._models(manifests)
        torch = modules["torch"]
        trace = {"schema": contract.TRACE_SCHEMA, "status": "complete_single_image_inference",
            "mode": "reference_conditioned_single_image", **contract.SINGLE_IMAGE_FLAGS, **tensors,
            "latent_layout": "BCTHW", "decoded_tensor_layout": "BTHWC",
            "audio_branch": {"role": "joint_av_latent_sampling_only", "waveform_decoded": False,
                             "audio_artifact_published": False},
            "condition_image": artifact(condition), "output_image": artifact(result_image),
            "model_components": manifests, "sampler_source": sources["sampler"],
            "source_preflight": artifact(output / "source_preflight.json"),
            "pipeline_class": "hsi.stage2.h3.ComfyUINativeSingleImage",
            "seed": job["seed"], "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
            "sampler": "er_sde", "scheduler": "sgm_uniform", "steps": 8,
            "modality_sigma_shifts": {"visual": 12., "auxiliary_audio": 3.},
            "adapter_strengths": {"distill_lora": .75, "image_lora": .5},
            "memory_policy": {"dynamic_vram_enabled": False, "cpu_weight_offload_allowed": True,
                              "triton_backend_enabled": True, "gpu_count": 1},
            "frame_zero_conditioning_used": False, "reference_conditioning_used": True,
            "comfy_commit": subprocess.check_output(["git", "-C", str(comfy_root), "rev-parse", "HEAD"], text=True).strip(),
            "comfy_sources": {name: sources[name] for name in COMFY_FILES},
            "torch_version": torch.__version__, "gpu": gpu, "gpu_name": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "timing": {**clock.stages, "total_through_image_seconds": time.monotonic() - begin}}
        trace_path = output / "sampler_trace.json"
        write_once(trace_path, trace)
        receipt = contract.make_receipt(case_id=job["case_id"], stage1_execution=job["stage1_execution"],
            camera=job["camera"], crop_derivation=job["crop_derivation"], condition_source=job["condition_source"],
            condition_image=condition, image=result_image, sampler_trace=trace_path,
            prompt=job["prompt"], seed=job["seed"],
            **{role + "_manifest": row["path"] for role, row in manifests.items()})
        write_once(output / "receipt.json", receipt)
        return contract.validate_receipt(output / "receipt.json")
    except Exception as error:
        write_once(output / "failure.json", {"schema": "hsi.h3_inference_failure.v1",
            "error_type": type(error).__name__, "error": str(error),
            "stage3_handoff_allowed": False, "positive_credit": 0})
        raise
