"""Current exact-camera background gates, excluding measured person box."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cv2

POLICY={"minimum_background_fraction":.25,"minimum_blurred_background_correlation":.90,
    "maximum_background_mean_absolute_difference":.075,"maximum_large_difference_fraction":.10,
    "large_difference_threshold":.12,"person_box_padding_pixels":24,"blur_sigma_pixels":1.2,
    "scope":"exact fixed camera H3 scene preservation; not proof of local target contact"}

def measure(reference,generated,person_box):
    ref=np.asarray(Image.open(reference).convert("RGB"),dtype=np.float32)/255.
    gen=np.asarray(Image.open(generated).convert("RGB"),dtype=np.float32)/255.
    if ref.shape!=gen.shape or ref.shape!=(512,512,3):
        raise ValueError("Only exactly aligned native 512-square images are supported")
    bbox=np.asarray(person_box,dtype=float)
    if bbox.shape!=(4,) or not np.isfinite(bbox).all() or np.any(bbox[2:]<=bbox[:2]):
        raise ValueError("Invalid actual detector person box")
    pad=POLICY["person_box_padding_pixels"]
    x0,y0=np.maximum(0,np.floor(bbox[:2]-pad)).astype(int)
    x1,y1=np.minimum(512,np.ceil(bbox[2:]+pad)).astype(int)
    background=np.ones((512,512),dtype=bool)
    background[y0:y1,x0:x1]=False
    background[:8]=background[-8:]=False
    background[:,:8]=background[:,-8:]=False
    a=cv2.GaussianBlur(cv2.cvtColor(ref,cv2.COLOR_RGB2GRAY),(0,0),POLICY["blur_sigma_pixels"])
    b=cv2.GaussianBlur(cv2.cvtColor(gen,cv2.COLOR_RGB2GRAY),(0,0),POLICY["blur_sigma_pixels"])
    valid=int(background.sum())
    if valid==0:
        raise ValueError("No background remains outside detected person")
    first,second=a[background].astype(float),b[background].astype(float)
    denominator=np.linalg.norm(first-first.mean())*np.linalg.norm(second-second.mean())
    correlation=float(np.dot(first-first.mean(),second-second.mean())/denominator) if denominator>1e-8 else None
    difference=np.abs(a-b)
    metrics={"background_fraction":float(background.mean()),"background_pixel_count":valid,
        "blurred_background_correlation":correlation,"background_mean_absolute_difference":float(difference[background].mean()),
        "large_difference_fraction":float((difference[background]>POLICY["large_difference_threshold"]).mean())}
    gates={"enough_observed_background":metrics["background_fraction"]>=POLICY["minimum_background_fraction"],
        "background_structure_preserved":correlation is not None and correlation>=POLICY["minimum_blurred_background_correlation"],
        "background_intensity_consistent":metrics["background_mean_absolute_difference"]<=POLICY["maximum_background_mean_absolute_difference"],
        "localized_unexplained_change_bounded":metrics["large_difference_fraction"]<=POLICY["maximum_large_difference_fraction"]}
    return {"metrics":metrics,"gates":gates,"passed":all(gates.values())},background,difference
