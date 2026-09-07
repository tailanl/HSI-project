"""Render exact refined full mesh in current original scene camera."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import view_numeric as loader

def run(stage2_path, refine_path, output, gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYOPENGL_PLATFORM="egl", EGL_DEVICE_ID=str(gpu))
    stage2, refine = read_sealed(stage2_path), read(refine_path)
    bundle = read_sealed(Path(refine_path).parent.parent / "receipt.json")
    selection = read_sealed(verified(bundle["outputs"]["selection"]))
    matches = [record for record in selection["trials"] if verified(record) == Path(refine_path).resolve()]
    if len(matches) != 1:
        raise ValueError("Trial is not bound by its physical bundle")
    stage1 = read_sealed(verified(stage2["source_stage1"]))
    target = read_sealed(verified(stage1["target"]))
    view = read_sealed(verified(stage2["source_view"]))
    keypose_path = verified(refine["keypose"])
    with np.load(keypose_path, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices_world_zup"], np.float64)
        faces = np.asarray(data["faces"], np.int64)
        if str(data["target_instance_id"]) != target["target_instance_id"]:
            raise ValueError("Diagnostic keypose belongs to another target")
    if vertices.shape != (10475, 3) or faces.shape != (20908, 3) or not np.isfinite(vertices).all():
        raise ValueError("Invalid actual SMPL-X topology")
    if stage1["scene_id"] != refine["scene_id"] or refine["instruction"] != stage1["instruction"]:
        raise ValueError("Diagnostic query mismatch")
    output.mkdir(parents=True, exist_ok=False)
    import pyrender
    import trimesh
    scene_mesh_path = verified(target["artifacts"]["original_scene_mesh"])
    mesh = loader.load_scene_mesh(scene_mesh_path)
    camera_record = view["selection"]["selected_camera"]
    camera = read(verified(camera_record))
    rgb_path = verified(view["selection"]["selected_source_rgb"])
    with Image.open(rgb_path) as img:
        width, height = img.size
    scene = pyrender.Scene(bg_color=[.87,.88,.9,1], ambient_light=[.55,.55,.55])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.95,.44,.12,1.], roughnessFactor=.85)
    scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, faces, process=False), material=material, smooth=False))
    K = np.asarray(camera["K"])
    pose = loader.opencv_to_pyrender_pose(np.asarray(camera["world_to_camera"]))
    scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2]), pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8), pose=pose)
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        rgb, _ = renderer.render(scene)
    finally:
        renderer.delete()
    full = output / "actual_refined_mesh_original_camera.png"
    Image.fromarray(rgb).save(full)
    crop = view["selection"]["selected_crop_xyxy"]
    cropped = Image.fromarray(rgb).crop(crop).resize((512,512), Image.Resampling.LANCZOS)
    failed = [k for k,v in refine["gates"].items() if v is not True]
    panel = Image.new("RGB", (768,620), "white")
    panel.paste(cropped, (128,88))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((15,8), "P550 / ACTUAL REFINED MESH / DIAGNOSTIC ONLY", font=font, fill="#8f220d")
    draw.text((15,35), "Scene "+stage1["scene_id"]+" | publication="+str(refine["stage3_handoff_allowed"]), font=font, fill="black")
    draw.text((15,61), "Failed: "+", ".join(failed), font=font, fill="#8f220d")
    panel_path = output / "diagnostic_keypose.png"
    panel.save(panel_path)
    write_once(output / "receipt.json", {"schema": "p550.actual_refine_diagnostic_visualization.v1",
        "source_stage2": artifact(stage2_path), "source_refine": artifact(refine_path), "keypose": artifact(keypose_path),
        "original_mesh": artifact(scene_mesh_path), "original_camera": camera_record,
        "full_original_camera": artifact(full), "panel": artifact(panel_path),
        "body_or_furniture_world_geometry_modified": False, "image_crop_xyxy": crop,
        "failed_gates": failed, "diagnostic_only": True, "publication_granted_by_renderer": False,
        "source": artifact(__file__)}, seal=True)
    print({"scene": stage1["scene_id"], "diagnostic": str(panel_path), "failed": failed}, flush=True)
