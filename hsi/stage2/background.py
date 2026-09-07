"""Single-image background audit, with actual detector masking."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .background_numeric import POLICY, measure

def run(stage1_path,view_path,h3_path,recovery_path,output):
    started=time.monotonic()
    stage1,view,h3,recovery=map(read_sealed,(stage1_path,view_path,h3_path,recovery_path))
    if view["source_stage1_execution"]!=artifact(stage1_path):
        raise ValueError("Background reference belongs to a different query")
    if recovery["inputs"]["h3_generation_receipt"]!=artifact(h3_path) \
            or recovery["inputs"]["h3_frame"]!=h3["artifacts"]["h3_image"] \
            or recovery["inputs"]["stage1_artifact"]!=stage1["bundle"]:
        raise ValueError("Background mask recovery belongs to a different H3 or Stage1")
    reference=verified(view["selection"]["selected_stage2_crop"])
    generated=verified(h3["artifacts"]["h3_image"])
    isolated=verified(h3["artifacts"]["h3_condition_image"])
    if sha256(reference)!=sha256(isolated):
        raise ValueError("H3 condition image differs from exact selected crop")
    raw_path=verified(recovery["outputs"]["raw_hybrikx_recovery"])
    with np.load(raw_path,allow_pickle=False) as data:
        bbox=data["selected_person_bbox_xyxy"].copy()
    result,mask,difference=measure(reference,generated,bbox)
    output.mkdir(parents=True,exist_ok=False)
    # Original reference / generated panels are copied without editing; only the third is a diagnostic.
    canvas=Image.new("RGB",(1536,548),"white")
    canvas.paste(Image.open(reference).convert("RGB"),(0,36))
    canvas.paste(Image.open(generated).convert("RGB"),(512,36))
    diagnostic=np.zeros((512,512,3),dtype=np.uint8)
    intensity=np.minimum(1.,difference/POLICY["large_difference_threshold"])
    diagnostic[:,:,0]=(255*intensity*mask).astype(np.uint8)
    diagnostic[:,:,1]=(100*mask).astype(np.uint8)
    canvas.paste(Image.fromarray(diagnostic),(1024,36))
    draw=ImageDraw.Draw(canvas)
    draw.text((10,10),"EXACT ORIGINAL CROP",fill="black")
    draw.text((522,10),"ACTUAL H3 SINGLE IMAGE",fill="black")
    draw.text((1034,10),"DIAGNOSTIC: black=excluded person; red=background difference",fill="black")
    picture=output / "background_consistency.png"
    canvas.save(picture)
    value={"schema":"p550.generated_fixed_camera_background_audit.v1","source":artifact(__file__),
        "source_stage1":artifact(stage1_path),"source_view":artifact(view_path),"source_h3":artifact(h3_path),
        "source_recovery":artifact(recovery_path),"actual_reference":artifact(reference),"actual_generated_image":artifact(generated),
        "person_bbox_from_actual_detector":bbox.tolist(),"policy":POLICY,**result,"visualization":artifact(picture),
        "Qwen_called":False,"original_H3_Qwen_judgement_overwritten":False,
        "local_target_contact_or_pose_correctness_not_implied":True,"elapsed_seconds":time.monotonic()-started}
    write_once(output / "receipt.json",value,seal=True)
    return value
