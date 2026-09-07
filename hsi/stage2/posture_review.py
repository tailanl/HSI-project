"""Independent four-gate neutral-sitting numerical check."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .posture_quality import metrics

def run(bundle_path,output):
    bundle=read_sealed(bundle_path)
    if bundle["stage3_handoff_allowed"] is not True:
        raise ValueError("Use an actual legacy-physical-pass bundle for this additional check")
    keypose=verified(bundle["outputs"]["published_keypose"])
    with np.load(keypose,allow_pickle=False) as data:
        if str(data["target_instance_id"])!=bundle["target_binding"]["target_instance_id"]:
            raise ValueError("Different furniture")
        quality=metrics(data["joints_world_zup"],float(data["terminal_facing_yaw_rad"]))
    value={"schema":"p550.additional_neutral_sitting_quality_audit.v1","source":artifact(__file__),
        "quality_source":artifact(HERE / "posture_quality.py"),"bundle":artifact(bundle_path),
        "keypose":artifact(keypose),"original_eighteen_physical_gates_passed":True,"quality":quality,
        "old_receipts_or_Qwen_judgments_modified":False,"stage3_handoff_allowed":False,
        "additional_gate_not_implicit_retroactive_publication":True}
    write_once(output,value,seal=True)
    print({"scene":bundle["scene_id"],"quality":quality},flush=True)
