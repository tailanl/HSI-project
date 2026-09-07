"""CPU-only adversarial checks: no model, service, GPU or old-source writes."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from hsi.stage2 import image_contract as contract


def json_file(path, value, sealed=False):
    if sealed:
        return contract.write_receipt(path, value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


@pytest.fixture
def case(tmp_path):
    condition = tmp_path / "condition.png"
    image = tmp_path / "h3_image.png"
    Image.new("RGB", (512, 512), (140, 140, 140)).save(condition)
    Image.new("RGB", (512, 512), (120, 130, 140)).save(image)
    camera = tmp_path / "camera.json"
    json_file(camera, {"extrinsic_convention": "opencv_world_to_camera", "width": 1024, "height": 768,
        "intrinsics": [[500, 0, 512], [0, 500, 384], [0, 0, 1]],
        "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    target = tmp_path / "target.json"
    json_file(target, {"scene_id": "079", "target_instance_id": "SEAT_7", "target_class": "bench",
                      "selected_surface_id": "SEAT_SURFACE_7"})
    bundle = tmp_path / "bundle.json"
    json_file(bundle, {"fixture_only": True})
    stage1_path = tmp_path / "stage1_execution.json"
    stage1 = json_file(stage1_path, {"scene_id": "079", "instruction": "Walk to the bench and sit down.",
        "target": contract.artifact(target), "bundle": contract.artifact(bundle)}, sealed=True)
    derivation = tmp_path / "input_derivation.json"
    json_file(derivation, {"source_stage1_target": stage1["target"], "source_camera": contract.artifact(camera),
        "qwen_input_image": contract.artifact(condition), "output_size_wh": [512, 512],
        "scene_id": "079", "instruction": stage1["instruction"], "target_instance_id": "SEAT_7"}, sealed=True)
    components = {}
    for component in contract.MODEL_COMPONENTS:
        weight = tmp_path / (component + ".bin")
        weight.write_bytes((component + "_actual_test_checkpoint").encode())
        manifest_path = tmp_path / (component + "_manifest.json")
        json_file(manifest_path, contract.make_model_manifest(component=component,
            source_directory=tmp_path, weight_paths=[weight]), sealed=True)
        components[component] = manifest_path
    prompt = "Add exactly one seated adult; keep the scene unchanged."
    trace_path = tmp_path / "sampler_trace.json"
    trace = {"schema": contract.TRACE_SCHEMA, "status": "complete_single_image_inference",
        "mode": "reference_conditioned_single_image", **contract.SINGLE_IMAGE_FLAGS,
        "audio_branch": {"role": "joint_av_latent_sampling_only", "waveform_decoded": False,
                         "audio_artifact_published": False},
        "latent_shape": [1, 16, 1, 64, 64], "latent_layout": "BCTHW",
        "decoded_tensor_shape": [1, 3, 1, 512, 512], "decoded_tensor_layout": "BCTHW",
        "condition_image": contract.artifact(condition), "output_image": contract.artifact(image),
        "model_components": {key: contract.artifact(path) for key, path in components.items()},
        "sampler_source": contract.artifact(__file__), "pipeline_class": "ActualReferenceConditionedH3",
        "seed": 55379, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    json_file(trace_path, trace, sealed=True)
    value = contract.make_receipt(case_id="scene079", stage1_execution=stage1_path, camera=camera,
        crop_derivation=derivation, condition_source=condition, condition_image=condition, image=image,
        transformer_manifest=components["transformer"], vae_manifest=components["vae"],
        text_encoder_manifest=components["text_encoder"], image_lora_manifest=components["image_lora"],
        distill_lora_manifest=components["distill_lora"], sampler_trace=trace_path, prompt=prompt, seed=55379)
    return tmp_path, value, trace


def publish(case, transform=None, *, trace_transform=None):
    root, original, original_trace = case
    value = copy.deepcopy(original)
    if trace_transform:
        trace = copy.deepcopy(original_trace)
        trace_transform(trace)
        trace_path = root / "altered_sampler_trace.json"
        contract.write_receipt(trace_path, trace)
        value["sampler_trace"] = contract.artifact(trace_path)
    if transform:
        transform(value)
    path = root / "receipt.json"
    contract.write_receipt(path, value)
    return path


def test_valid_single_image(case):
    path = publish(case)
    result = contract.validate_receipt(path, expected_case_id="scene079", expected_image=case[0] / "h3_image.png")
    assert result["native_output_kind"] == "image"
    assert result["temporal_latent_count"] == result["decoded_image_count"] == 1


@pytest.mark.parametrize("layout,shape", [
    ("BCHW", [1, 3, 512, 512]), ("BHWC", [1, 512, 512, 3]),
    ("BTCHW", [1, 1, 3, 512, 512]), ("BTHWC", [1, 1, 512, 512, 3]),
])
def test_registered_single_image_decoder_layouts(case, layout, shape):
    path = publish(case, trace_transform=lambda row: row.update(decoded_tensor_layout=layout, decoded_tensor_shape=shape))
    contract.validate_receipt(path)


@pytest.mark.parametrize("count", [5, 20, 97])
@pytest.mark.parametrize("where", ["latent", "decoded", "case_count", "trace_count"])
def test_reject_multiframe_packages_even_with_one_png(case, count, where):
    def change_trace(row):
        if where == "latent": row["latent_shape"][2] = count
        if where == "decoded": row["decoded_tensor_shape"][2] = count
        if where == "trace_count": row["decoded_image_count"] = count
    path = publish(case, lambda row: row.update(decoded_image_count=count) if where == "case_count" else None,
                   trace_transform=change_trace)
    with pytest.raises(contract.ImageContractError): contract.validate_receipt(path)


@pytest.mark.parametrize("field", sorted(contract.FORBIDDEN_MEDIA_FIELDS))
def test_reject_video_or_media_time_fields(case, field):
    path = publish(case, lambda row: row.update(extra={field: 3}))
    with pytest.raises(contract.ImageContractError, match="Video/time-specific"):
        contract.validate_receipt(path)


@pytest.mark.parametrize("field", ["video_generation_performed", "video_frame_extraction_performed", "audio_decoding_performed"])
@pytest.mark.parametrize("location", ["case", "trace"])
def test_reject_video_extraction_flags(case, field, location):
    path = publish(case,
        (lambda row: row.update({field: True})) if location == "case" else None,
        trace_transform=(lambda row: row.update({field: True})) if location == "trace" else None)
    with pytest.raises(contract.ImageContractError): contract.validate_receipt(path)


@pytest.mark.parametrize("field,value", [("temporal_latent_count", True), ("decoded_image_count", 1.0),
    ("output_image_count", "1"), ("video_generation_performed", 0)])
def test_strict_flag_types(case, field, value):
    path = publish(case, lambda row: row.update({field: value}))
    with pytest.raises(contract.ImageContractError): contract.validate_receipt(path)


def test_reject_payload_tampering(case):
    path = publish(case)
    value = json.loads(path.read_text())
    value["seed"] = 42
    path.write_text(json.dumps(value))
    with pytest.raises(contract.ImageContractError, match="payload hash drift"): contract.validate_receipt(path)


def test_reject_output_hash_drift(case):
    path = publish(case)
    Image.new("RGB", (512, 512), "red").save(case[0] / "h3_image.png")
    with pytest.raises(contract.ImageContractError, match="drift"): contract.validate_receipt(path)


def test_reject_wrong_expected_case_or_image(case):
    path = publish(case)
    with pytest.raises(contract.ImageContractError, match="case ID drift"):
        contract.validate_receipt(path, expected_case_id="scene042")
    with pytest.raises(contract.ImageContractError, match="path drift"):
        contract.validate_receipt(path, expected_image=case[0] / "condition.png")


def test_reject_same_manifest_for_transformer_and_vae(case):
    def mutate(row): row["model_components"]["vae"] = row["model_components"]["transformer"]
    path = publish(case, mutate, trace_transform=mutate)
    with pytest.raises(contract.ImageContractError, match="manifest schema or role"):
        contract.validate_receipt(path)


def test_reject_changed_manifest(case):
    path = publish(case)
    (case[0] / "transformer_manifest.json").write_text('{"wrong_model":true}')
    with pytest.raises(contract.ImageContractError, match="drift"): contract.validate_receipt(path)


@pytest.mark.parametrize("field,value", [("seed", 77), ("prompt_sha256", "0" * 64),
    ("mode", "video_to_selected_image"), ("pipeline_class", "")])
def test_reject_sampler_provenance_drift(case, field, value):
    path = publish(case, trace_transform=lambda row: row.update({field: value}))
    with pytest.raises(contract.ImageContractError): contract.validate_receipt(path)


def test_reject_wrong_target(case):
    path = publish(case, lambda row: row["source_binding"].update(target_instance_id="OTHER_SEAT"))
    with pytest.raises(contract.ImageContractError, match="target binding drift"):
        contract.validate_receipt(path)


def test_reject_wrong_camera(case):
    wrong = case[0] / "another_camera.json"
    wrong.write_bytes((case[0] / "camera.json").read_bytes())
    path = publish(case, lambda row: row["inputs"].update(camera=contract.artifact(wrong)))
    with pytest.raises(contract.ImageContractError, match="different camera"):
        contract.validate_receipt(path)


def test_reject_unregistered_spatial_mapping(case):
    path = publish(case, lambda row: row["spatial_mapping"].update(transform="video_center_crop"))
    with pytest.raises(contract.ImageContractError, match="mapping"): contract.validate_receipt(path)


def test_reject_claim_of_verified_camera_preservation(case):
    path = publish(case, lambda row: row["spatial_mapping"].update(scene_camera_preservation_is_model_intent_not_verified_fact=False))
    with pytest.raises(contract.ImageContractError, match="camera-preservation"):
        contract.validate_receipt(path)


@pytest.mark.parametrize("kind", ["jpeg", "apng", "small", "rgba", "copy"])
def test_reject_nonconforming_output_with_updated_hashes(case, kind):
    image = case[0] / "bad_image.png"
    if kind == "jpeg": Image.new("RGB", (512, 512), "red").save(image, format="JPEG")
    elif kind == "apng":
        Image.new("RGB", (512, 512), "red").save(image, save_all=True,
            append_images=[Image.new("RGB", (512, 512), "blue")], duration=100, loop=0)
    elif kind == "small": Image.new("RGB", (256, 256), "red").save(image)
    elif kind == "rgba": Image.new("RGBA", (512, 512), "red").save(image)
    else: image.write_bytes((case[0] / "condition.png").read_bytes())
    record = contract.artifact(image)
    path = publish(case, lambda row: row["artifacts"].update(h3_image=record),
        trace_transform=lambda row: row.update(output_image=record))
    with pytest.raises(contract.ImageContractError): contract.validate_receipt(path)


def test_write_once_protects_existing_receipt(case):
    path = publish(case)
    with pytest.raises(FileExistsError): contract.write_receipt(path, case[1])


def test_reject_claim_that_joint_av_audio_latents_were_not_sampled(case):
    path = publish(case, lambda row: row.update(audio_latent_sampling_performed=False))
    with pytest.raises(contract.ImageContractError): contract.validate_receipt(path)


def test_reject_decoded_audio_waveform(case):
    path = publish(case, trace_transform=lambda row: row["audio_branch"].update(waveform_decoded=True))
    with pytest.raises(contract.ImageContractError, match="Joint AV"):
        contract.validate_receipt(path)


def test_reject_nonempty_but_untyped_component_manifest(case):
    bad = case[0] / "fake_manifest.json"
    json_file(bad, {"fake": "nonempty"}, sealed=True)
    record = contract.artifact(bad)
    def mutate(row): row["model_components"]["text_encoder"] = record
    path = publish(case, mutate, trace_transform=mutate)
    with pytest.raises(contract.ImageContractError, match="manifest schema or role"):
        contract.validate_receipt(path)


@pytest.mark.parametrize("component", contract.MODEL_COMPONENTS)
def test_reject_changed_actual_weights_in_every_component(case, component):
    path = publish(case)
    weight = case[0] / (component + ".bin")
    weight.write_bytes(b"changed_checkpoint_data")
    with pytest.raises(contract.ImageContractError, match="checkpoint hash/path/byte drift"):
        contract.validate_receipt(path)


def test_hash_cache_avoids_rehashing_unchanged_large_weights(case, monkeypatch):
    path = publish(case)
    contract._WEIGHT_HASH_CACHE.clear()
    original = contract.artifact
    reads = []
    def counted(path):
        if Path(path).suffix == ".bin": reads.append(str(path))
        return original(path)
    monkeypatch.setattr(contract, "artifact", counted)
    contract.validate_receipt(path)
    contract.validate_receipt(path)
    assert len(reads) == len(contract.MODEL_COMPONENTS)


def test_hash_cache_rechecks_replacement_with_same_length_and_restored_mtime(case):
    import os
    path = publish(case)
    contract.validate_receipt(path)
    weight = case[0] / "image_lora.bin"
    old = weight.stat()
    weight.write_bytes(b"x" * old.st_size)
    os.utime(weight, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(contract.ImageContractError, match="checkpoint hash/path/byte drift"):
        contract.validate_receipt(path)
