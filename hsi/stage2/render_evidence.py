"""Original-camera plus two isolated actual-body views; no geometry edits."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render_scene import run as render_scene
from .render_body import run as render_body

def run(source_path,bundle_path,output,gpu):
    source,bundle=read_sealed(source_path),read_sealed(bundle_path)
    selection=read(verified(bundle["outputs"]["selection"]))
    trial=verified(selection["selected_refine_receipt"])
    stage1=read_sealed(verified(source["source_stage1"]))
    render_scene(source_path,trial,output / "original_camera",gpu)
    actual=read_sealed(output / "original_camera/receipt.json")
    if actual["keypose"]["sha256"]!=bundle["outputs"]["published_keypose"]["sha256"]:
        raise ValueError("Original-camera render is not the exact selected published geometry")
    render_body(bundle_path,output / "isolated_body",gpu)
    value={"schema":"p550.fast_original_camera_post_refine_evidence.v1","source":artifact(__file__),
        "inputs":{"p533_complete_bundle":artifact(bundle_path),
            "h3_hybrikx_articulation_receipt":source["hybrikx_receipt"],
            "stage1_target":stage1["target"],"stage1_bundle":stage1["bundle"],
            "p533_published_keypose":bundle["outputs"]["published_keypose"]},
        "views":[{"image":actual["full_original_camera"]}],
        "actual_original_camera_render":artifact(output / "original_camera/receipt.json"),
        "isolated_body_diagnostic":artifact(output / "isolated_body/receipt.json"),
        "actual_body_or_scene_world_geometry_modified":False,"stage3_handoff_allowed":False}
    write_once(output / "manifest.json",value,seal=True)
