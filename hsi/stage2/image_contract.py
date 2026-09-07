"""Hash-bound contracts for reference-conditioned H3 single-image inference.

This module validates recorded execution evidence; it is not an attestation of
unobserved model execution.  A sampler must record its actual tensor shapes,
component manifests and outputs before the caller builds the case receipt.
There is deliberately no adapter from a selected video frame to this schema.
Only an exact 512-square condition/output mapping is currently registered.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


SCHEMA = "p553.h3_single_image_case.v1"
TRACE_SCHEMA = "p553.h3_single_image_sampler_trace.v1"
MODEL_MANIFEST_SCHEMA = "p553.h3_image_model_component.v1"
MODEL_COMPONENTS = ("transformer", "vae", "text_encoder", "image_lora", "distill_lora")
SIZE_WH = (512, 512)
FORBIDDEN_MEDIA_FIELDS = frozenset({
    "fps", "duration", "duration_s", "extract_time_s", "extract_frame",
    "frame_time_s", "frame_timestamp_s", "frame_index", "selected_frame_index",
    "num_frames", "video_path", "h3_video", "h3_frame_3s", "h3_first_frame",
    "h3_comparison_frame", "audio_path", "audio_generation_performed",
})
SINGLE_IMAGE_FLAGS = {
    "temporal_latent_count": 1,
    "decoded_image_count": 1,
    "output_image_count": 1,
    "video_generation_performed": False,
    "video_frame_extraction_performed": False,
    # H3's joint AV sampler may still update audio latents in image mode.
    # No audio waveform is decoded or published by this registered protocol.
    "audio_latent_sampling_performed": True,
    "audio_decoding_performed": False,
}
_WEIGHT_HASH_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


class ImageContractError(ValueError):
    """A native single-image claim or its lineage is not supported."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ImageContractError(message)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"Artifact is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verified(record: Mapping[str, Any], label: str = "artifact") -> Path:
    require(isinstance(record, Mapping), f"Missing {label}")
    require(set(record) == {"path", "bytes", "sha256"}, f"Unexpected {label} artifact fields")
    require(isinstance(record["path"], str) and Path(record["path"]).is_absolute(),
            f"{label} needs an absolute path")
    require(type(record["bytes"]) is int and record["bytes"] >= 0, f"Invalid {label} byte count")
    observed = artifact(record["path"])
    require(dict(record) == observed, f"Hash/path/byte drift in {label}")
    return Path(observed["path"])


def _weight_artifact(path: str | Path) -> dict[str, Any]:
    """Hash actual weights once per unchanged file identity in this process.

    This cache is intentionally not persisted as a trusted hash declaration.
    A new process must perform its own first verification.  Both timestamps
    are checked so changing bytes and restoring mtime does not hit the cache.
    """
    path = Path(path).resolve(strict=True)
    before = path.stat()
    identity = (str(path), before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns)
    if identity not in _WEIGHT_HASH_CACHE:
        observed = artifact(path)
        after = path.stat()
        require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                == identity[1:], f"Weights changed during hash verification: {path}")
        _WEIGHT_HASH_CACHE[identity] = observed
    return dict(_WEIGHT_HASH_CACHE[identity])


def make_model_manifest(*, component: str, source_directory: str | Path,
                        weight_paths: list[str | Path],
                        configuration: str | Path | None = None) -> dict[str, Any]:
    """Build a sealed inventory by reading actual local checkpoint bytes."""
    require(component in MODEL_COMPONENTS, "Unknown image model component")
    root = Path(source_directory).resolve(strict=True)
    require(root.is_dir(), "Model source directory must exist")
    require(bool(weight_paths), "Model component needs actual checkpoint files")
    weights = [_weight_artifact(path) for path in weight_paths]
    require(all(Path(row["path"]).is_relative_to(root) for row in weights),
            "Model checkpoint is outside its declared source directory")
    return seal({"schema": MODEL_MANIFEST_SCHEMA, "component": component,
        "source_directory": str(root), "weights": weights,
        "configuration": artifact(configuration) if configuration is not None else None})


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("receipt_payload_sha256", None)
    result["receipt_payload_sha256"] = canonical_hash(result)
    return result


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_sealed(path: str | Path) -> dict[str, Any]:
    value = _json(path)
    require(value == seal(value), f"Receipt payload hash drift: {path}")
    return value


def write_receipt(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Write once, sealed; validate case receipts before publication separately."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal(value)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(sealed, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sealed


def _no_temporal_media_fields(value: Any, location: str = "receipt") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            require(key not in FORBIDDEN_MEDIA_FIELDS, f"Video/time-specific field {location}.{key}")
            _no_temporal_media_fields(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _no_temporal_media_fields(item, f"{location}[{index}]")


def _png(record: Mapping[str, Any], label: str) -> Path:
    path = verified(record, label)
    with Image.open(path) as image:
        require(image.format == "PNG", f"{label} must be an actual PNG")
        require(image.size == SIZE_WH, f"{label} must be 512x512")
        require(image.mode == "RGB", f"{label} must be RGB")
        require(getattr(image, "n_frames", 1) == 1 and not getattr(image, "is_animated", False),
                f"{label} must contain exactly one static image")
        image.load()
    return path


def _models(value: Mapping[str, Any]) -> None:
    require(isinstance(value, Mapping) and set(value) == set(MODEL_COMPONENTS),
            "Need actual transformer/VAE/text-encoder/image-LoRA/distill-LoRA manifests")
    paths = []
    checkpoint_paths: set[Path] = set()
    for component in MODEL_COMPONENTS:
        path = verified(value[component], f"{component} manifest")
        manifest = read_sealed(path)
        require(manifest.get("schema") == MODEL_MANIFEST_SCHEMA
                and manifest.get("component") == component, f"Wrong {component} manifest schema or role")
        raw_root = manifest.get("source_directory")
        require(isinstance(raw_root, str) and Path(raw_root).is_absolute(), "Missing actual model source directory")
        root = Path(raw_root).resolve(strict=True)
        require(root.is_dir() and str(root) == raw_root, "Model source directory is not canonical")
        weights = manifest.get("weights")
        require(isinstance(weights, list) and bool(weights), f"No actual {component} checkpoint inventory")
        for record in weights:
            require(isinstance(record, Mapping) and set(record) == {"path", "bytes", "sha256"},
                    f"Malformed {component} checkpoint artifact")
            require(isinstance(record.get("path"), str) and Path(record["path"]).is_absolute(),
                    f"Missing {component} checkpoint path")
            require(type(record.get("bytes")) is int and record["bytes"] > 0,
                    f"Empty {component} checkpoint")
            weight_path = Path(record["path"]).resolve(strict=True)
            require(weight_path.is_relative_to(root), f"{component} checkpoint source-directory drift")
            require(weight_path not in checkpoint_paths, "Distinct components cannot reuse one checkpoint artifact")
            checkpoint_paths.add(weight_path)
            require(dict(record) == _weight_artifact(weight_path), f"{component} checkpoint hash/path/byte drift")
        configuration = manifest.get("configuration")
        if configuration is not None:
            _json(verified(configuration, f"{component} configuration"))
        paths.append(path)
    require(len(set(paths)) == len(MODEL_COMPONENTS), "Components must bind their own manifests")


def _single_flags(value: Mapping[str, Any], label: str) -> None:
    for key, expected in SINGLE_IMAGE_FLAGS.items():
        require(type(value.get(key)) is type(expected) and value[key] == expected,
                f"{label}.{key} must be {expected!r}")


def _shape(value: Any, layout: str, allowed: set[str], label: str) -> dict[str, int]:
    require(layout in allowed, f"Unregistered {label} tensor layout")
    require(isinstance(value, list) and len(value) == len(layout)
            and all(type(x) is int and x > 0 for x in value), f"Malformed {label} shape")
    shape = dict(zip(layout, value))
    require(shape["B"] == 1 and shape.get("T", 1) == 1,
            f"{label} must have batch=1 and temporal length=1")
    return shape


def validate_trace(path: str | Path, *, condition: Mapping[str, Any],
                   image: Mapping[str, Any], models: Mapping[str, Any]) -> dict[str, Any]:
    trace = read_sealed(path)
    _no_temporal_media_fields(trace, "sampler_trace")
    require(trace.get("schema") == TRACE_SCHEMA, "Wrong sampler trace schema")
    require(trace.get("status") == "complete_single_image_inference", "Incomplete image sampler")
    require(trace.get("mode") == "reference_conditioned_single_image", "Wrong sampler mode")
    _single_flags(trace, "sampler_trace")
    require(trace.get("audio_branch") == {
        "role": "joint_av_latent_sampling_only", "waveform_decoded": False,
        "audio_artifact_published": False,
    }, "Joint AV latent sampling must be distinguished from decoded audio generation")
    _shape(trace.get("latent_shape"), trace.get("latent_layout"), {"BCTHW"}, "latent")
    decoded = _shape(trace.get("decoded_tensor_shape"), trace.get("decoded_tensor_layout"),
                     {"BCTHW", "BTCHW", "BTHWC", "BCHW", "BHWC"}, "decoded")
    require((decoded["W"], decoded["H"], decoded["C"]) == (512, 512, 3),
            "Decoded tensor must be one RGB 512-square image")
    require(trace.get("condition_image") == condition and trace.get("output_image") == image,
            "Sampler trace points to a different condition or output")
    require(trace.get("model_components") == models, "Sampler/model component manifest drift")
    verified(trace.get("sampler_source"), "actual sampler source")
    require(isinstance(trace.get("pipeline_class"), str) and bool(trace["pipeline_class"].strip()),
            "Actual pipeline class must be recorded")
    _models(models)
    _png(condition, "sampler condition")
    _png(image, "sampler output")
    return trace


def validate_receipt(path: str | Path, *, expected_case_id: str | None = None,
                     expected_image: str | Path | None = None) -> dict[str, Any]:
    value = read_sealed(path)
    _no_temporal_media_fields(value)
    require(value.get("schema") == SCHEMA, "Wrong H3 single-image receipt schema")
    require(value.get("status") == "complete_native_single_image", "Incomplete native image case")
    require(value.get("native_output_kind") == "image", "Generation output must be image")
    require(isinstance(value.get("case_id"), str) and bool(value["case_id"].strip()), "Missing case ID")
    if expected_case_id is not None:
        require(value["case_id"] == expected_case_id, "H3 image case ID drift")
    _single_flags(value, "case")
    require(type(value.get("seed")) is int and value["seed"] >= 0, "Invalid image generation seed")
    require(isinstance(value.get("prompt"), str) and bool(value["prompt"].strip()), "Missing actual image prompt")
    require(value.get("spatial_mapping") == {
        "condition_size_wh": [512, 512], "output_size_wh": [512, 512],
        "condition_to_output_affine_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "transform": "identity_512square", "crop_resize_or_letterbox_performed": False,
        "scene_camera_preservation_is_model_intent_not_verified_fact": True,
    }, "Missing exact condition/output mapping or truthful camera-preservation qualification")
    outputs = value.get("artifacts")
    require(isinstance(outputs, Mapping) and set(outputs) == {"h3_condition_image", "h3_image"},
            "Case must publish only condition image and one generated image")
    condition = outputs["h3_condition_image"]
    image = outputs["h3_image"]
    _png(condition, "H3 condition")
    image_path = _png(image, "H3 output")
    require(condition["sha256"] != image["sha256"], "Output merely copies the empty condition")
    if expected_image is not None:
        require(image_path == Path(expected_image).resolve(strict=True), "H3 output image path drift")

    inputs = value.get("inputs")
    require(isinstance(inputs, Mapping) and set(inputs) == {
        "stage1_execution", "camera", "crop_derivation", "condition_source"
    }, "Missing exact Stage1/camera/crop source bindings")
    resolved = {key: verified(record, key) for key, record in inputs.items()}
    stage1 = read_sealed(resolved["stage1_execution"])
    target_path = verified(stage1.get("target"), "Stage1 target")
    verified(stage1.get("bundle"), "Stage1 bundle")
    target = _json(target_path)
    expected_binding = {
        "scene_id": stage1.get("scene_id"), "instruction": stage1.get("instruction"),
        "target_instance_id": target.get("target_instance_id"),
        "target_class": target.get("target_class"), "target_surface_id": target.get("selected_surface_id"),
        "stage1_target": stage1["target"], "stage1_bundle": stage1["bundle"],
    }
    require(all(isinstance(expected_binding[key], str) and bool(expected_binding[key]) for key in (
        "scene_id", "instruction", "target_instance_id", "target_class", "target_surface_id"
    )), "Incomplete Stage1 target identity")
    require(value.get("source_binding") == expected_binding, "H3 image/Stage1 target binding drift")
    require(str(target.get("scene_id")) == stage1["scene_id"], "Target belongs to a different scene")
    derivation = read_sealed(resolved["crop_derivation"])
    require(derivation.get("source_stage1_target") == stage1["target"], "Crop has a different target")
    require(derivation.get("source_camera") == inputs["camera"], "Crop has a different camera")
    require(derivation.get("qwen_input_image") == inputs["condition_source"], "Crop source hash drift")
    require(derivation.get("output_size_wh") == [512, 512], "Crop derivation is not 512 square")
    require(derivation.get("scene_id") == stage1["scene_id"]
            and derivation.get("instruction") == stage1["instruction"]
            and derivation.get("target_instance_id") == target["target_instance_id"], "Crop/query binding drift")
    _png(inputs["condition_source"], "Exact selected Stage1 crop")
    require(all(condition[key] == inputs["condition_source"][key] for key in ("sha256", "bytes")),
            "Condition bytes differ from the exact selected Stage1 crop")
    camera = _json(resolved["camera"])
    require(camera.get("extrinsic_convention") == "opencv_world_to_camera", "Wrong Stage1 camera convention")
    for field, rows, cols in (("world_to_camera", 4, 4), ("intrinsics", 3, 3)):
        matrix = camera.get(field, camera.get("K") if field == "intrinsics" else None)
        require(isinstance(matrix, list) and len(matrix) == rows and all(
            isinstance(row, list) and len(row) == cols and all(
                type(x) in (float, int) and math.isfinite(x) for x in row
            ) for row in matrix), f"Invalid {field} camera matrix")
    trace_path = verified(value.get("sampler_trace"), "single-image sampler trace")
    trace = validate_trace(trace_path, condition=condition, image=image, models=value.get("model_components"))
    require(trace.get("seed") == value["seed"] and type(trace.get("seed")) is int,
            "Sampler/case seed drift")
    require(trace.get("prompt_sha256") == hashlib.sha256(value["prompt"].encode("utf-8")).hexdigest(),
            "Sampler/case prompt drift")
    return value


def make_receipt(*, case_id: str, stage1_execution: str | Path, camera: str | Path,
                 crop_derivation: str | Path, condition_source: str | Path,
                 condition_image: str | Path, image: str | Path,
                 transformer_manifest: str | Path, vae_manifest: str | Path,
                 text_encoder_manifest: str | Path, image_lora_manifest: str | Path,
                 distill_lora_manifest: str | Path,
                 sampler_trace: str | Path, prompt: str, seed: int) -> dict[str, Any]:
    """Build a sealed payload; use validate_receipt after writing it once."""
    stage1 = read_sealed(stage1_execution)
    target = _json(verified(stage1["target"], "Stage1 target"))
    return seal({
        "schema": SCHEMA, "status": "complete_native_single_image", "case_id": case_id,
        "native_output_kind": "image", **SINGLE_IMAGE_FLAGS, "prompt": prompt, "seed": seed,
        "inputs": {"stage1_execution": artifact(stage1_execution), "camera": artifact(camera),
                   "crop_derivation": artifact(crop_derivation), "condition_source": artifact(condition_source)},
        "artifacts": {"h3_condition_image": artifact(condition_image), "h3_image": artifact(image)},
        "source_binding": {"scene_id": stage1["scene_id"], "instruction": stage1["instruction"],
            "target_instance_id": target["target_instance_id"], "target_class": target["target_class"],
            "target_surface_id": target["selected_surface_id"], "stage1_target": stage1["target"],
            "stage1_bundle": stage1["bundle"]},
        "model_components": {"transformer": artifact(transformer_manifest), "vae": artifact(vae_manifest),
            "text_encoder": artifact(text_encoder_manifest), "image_lora": artifact(image_lora_manifest),
            "distill_lora": artifact(distill_lora_manifest)},
        "sampler_trace": artifact(sampler_trace),
        "spatial_mapping": {"condition_size_wh": [512, 512], "output_size_wh": [512, 512],
            "condition_to_output_affine_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "transform": "identity_512square", "crop_resize_or_letterbox_performed": False,
            "scene_camera_preservation_is_model_intent_not_verified_fact": True},
    })
