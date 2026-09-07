"""Exact calibrated crop/depth projection into the H3 single-image pixels."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence, Protocol
from dataclasses import dataclass
import math
import hashlib
import json
import os
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import cv2
from . import recovery_contract as contract
from hsi.common.artifacts import write_once

SCHEMA = "p533.lingo_crop_letterbox_projection_observation.v1"

def _resolve(value: Path) -> Path:
    return Path(value).resolve(strict=True)

def run(
    camera_path: Path,
    derivation_path: Path,
    source_depth_path: Path,
    h3_frame_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    camera_path = _resolve(camera_path)
    derivation_path = _resolve(derivation_path)
    source_depth_path = _resolve(source_depth_path)
    h3_frame_path = _resolve(h3_frame_path)
    output_dir = Path(output_dir).resolve()
    contract.require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    camera = contract.read_json(camera_path)
    derivation = contract.read_json(derivation_path)
    contract.require(
        camera.get("extrinsic_convention") == "opencv_world_to_camera",
        "LINGO camera must be OpenCV world_to_camera",
    )
    source_wh = (int(camera["width"]), int(camera["height"]))
    K = np.asarray(camera.get("intrinsics", camera.get("K")), dtype=np.float64)
    world_to_camera = np.asarray(camera["world_to_camera"], dtype=np.float64)
    contract.require(K.shape == (3, 3) and world_to_camera.shape == (4, 4), "camera matrix drift")
    crop_xyxy = np.asarray(derivation["crop_xyxy"], dtype=np.int64)
    qwen_wh = np.asarray(derivation["output_size_wh"], dtype=np.int64)
    contract.require(crop_xyxy.shape == (4,) and qwen_wh.shape == (2,), "crop derivation shape drift")
    x0, y0, x1, y1 = [int(x) for x in crop_xyxy]
    crop_w, crop_h = x1 - x0, y1 - y0
    contract.require(
        0 <= x0 < x1 <= source_wh[0] and 0 <= y0 < y1 <= source_wh[1],
        "crop is outside source camera image",
    )
    contract.require(bool(np.all(qwen_wh > 0)), "invalid generated-image input dimensions")
    contract.require(qwen_wh.tolist() == [512, 512], "Image mode requires identity 512-square generation mapping")

    # Current selected 512-square condition maps identically to the image.
    # Retain source-camera crop affine and nearest-neighbour metric depth.
    h3_fit = min(512.0 / float(qwen_wh[0]), 512.0 / float(qwen_wh[1]))
    content_wh = np.rint(qwen_wh.astype(np.float64) * h3_fit).astype(np.int64)
    pad_xy = (np.asarray((512, 512), dtype=np.int64) - content_wh) // 2
    crop_to_qwen = np.asarray((qwen_wh[0] / crop_w, qwen_wh[1] / crop_h), dtype=np.float64)
    scale_xy = crop_to_qwen * h3_fit
    affine = np.asarray(
        (
            (scale_xy[0], 0.0, float(pad_xy[0]) - scale_xy[0] * x0),
            (0.0, scale_xy[1], float(pad_xy[1]) - scale_xy[1] * y0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    observation_K = affine @ K

    raw_depth = np.load(source_depth_path, allow_pickle=False)
    contract.require(raw_depth.shape == (source_wh[1], source_wh[0]), "source depth/camera size drift")
    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    cropped = raw_depth[y0:y1, x0:x1]
    qwen_depth = cv2.resize(
        cropped, (int(qwen_wh[0]), int(qwen_wh[1])), interpolation=cv2.INTER_NEAREST
    )
    content_depth = cv2.resize(
        qwen_depth, (int(content_wh[0]), int(content_wh[1])), interpolation=cv2.INTER_NEAREST
    )
    aligned = np.zeros((512, 512), dtype=np.float32)
    px, py = int(pad_xy[0]), int(pad_xy[1])
    aligned[py : py + int(content_wh[1]), px : px + int(content_wh[0])] = content_depth
    depth_output = output_dir / "depth_m_h3_image_512.npy"
    np.save(depth_output, np.ascontiguousarray(aligned))

    from PIL import Image

    with Image.open(h3_frame_path) as frame:
        contract.require(frame.size == (512, 512), "H3 comparison frame is not 512x512")
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_exact_lingo_camera_crop_letterbox_depth",
        "camera": {
            "extrinsic_convention": "opencv_world_to_camera",
            "world_to_camera": world_to_camera.tolist(),
            "source_intrinsics": K.tolist(),
            "source_wh_px": list(source_wh),
            "crop_letterbox": {
                "crop_xywh_px": [x0, y0, crop_w, crop_h],
                "qwen_input_wh_px": qwen_wh.astype(int).tolist(),
                "h3_content_wh_px": content_wh.astype(int).tolist(),
                "output_wh_px": [512, 512],
                "scale_xy": scale_xy.tolist(),
                "pad_xy_px": pad_xy.astype(float).tolist(),
                "source_to_observation_affine_3x3": affine.tolist(),
                "no_shear_or_flip": True,
            },
            "observation_intrinsics": observation_K.tolist(),
        },
        "depth_m": {
            **contract.artifact(depth_output),
            "alignment": "observation_pixels_after_crop_letterbox",
            "invalid_value": 0.0,
            "resize_filter": "nearest",
        },
        "inputs": {
            "camera": contract.artifact(camera_path),
            "crop_derivation": contract.artifact(derivation_path),
            "source_depth_m": contract.artifact(source_depth_path),
            "h3_image": contract.artifact(h3_frame_path),
        },
        "transform_contract": {
            "crop_to_qwen_may_be_anisotropic": True,
            "qwen_to_h3_frame_is_aspect_preserving": True,
            "equation": "K_observation = A_source_to_observation @ K_source",
            "condition_to_image_transform_is_identity": True,
            "scene_camera_preservation_requires_separate_background_audit": True,
        },
    }
    value["receipt_payload_sha256"] = contract.canonical_hash(value)
    write_once(output_dir / "projection_observation.json", value)
    return value
