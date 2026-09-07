"""V2: actual scene-aware masks for the current geometric provider.

No P533/H3 receipt is fabricated. Two additional views hide scene occluders
explicitly; they are anatomical diagnostics, not evidence of scene layout.
"""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

from memory_common import PROJECT, artifact, verified, read_sealed, read_json, write_once, require
from fast_keypose import load_current


SOURCE = Path(__file__).resolve()


def scene_body_visibility(renderer, scene, body_node, segmentation_flag):
    """Keep every original mesh as a black occluder in the visible pass.

    pyrender SEG skips any mesh absent from seg_node_map. Omitting scene nodes
    is correct only for the explicitly amodal pass, never the visible pass.
    """
    require(body_node in scene.mesh_nodes, "Body is not a mesh node in this scene")
    amodal, _ = renderer.render(scene, flags=segmentation_flag,
        seg_node_map={body_node: [255, 255, 255]})
    node_map = {node: [0, 0, 0] for node in scene.mesh_nodes}
    node_map[body_node] = [255, 255, 255]
    visible, _ = renderer.render(scene, flags=segmentation_flag, seg_node_map=node_map)
    amodal_mask, visible_mask = amodal[..., 0] > 0, visible[..., 0] > 0
    require(amodal_mask.shape == visible_mask.shape and amodal_mask.any(), "Missing actual amodal body mask")
    require(not np.any(visible_mask & ~amodal_mask), "Visible body pixels outside paired amodal silhouette")
    return amodal_mask, visible_mask


def camera_body_coverage(vertices, world_to_camera, K, width, height):
    camera_xyz = np.asarray(vertices) @ np.asarray(world_to_camera)[:3, :3].T + np.asarray(world_to_camera)[:3, 3]
    require(np.isfinite(camera_xyz).all(), "Nonfinite body camera coordinates")
    projected = camera_xyz @ np.asarray(K).T
    positive = camera_xyz[:, 2] > .05
    xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
    within = positive & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return dict(minimum_body_camera_depth_m=float(camera_xyz[:, 2].min()),
                body_vertex_in_frame_fraction=float(within.mean()),
                entire_actual_body_in_frame=bool(within.all()))


def run(proposal_path, view_path, output, gpu=5):
    start = time.monotonic()
    source_at_start = artifact(SOURCE)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYOPENGL_PLATFORM="egl", EGL_DEVICE_ID="0")
    proposal, view = read_sealed(proposal_path), read_sealed(view_path)
    require(proposal["schema"] == "p555.geometry_guided_keypose_proposal.v1", "Not a geometric-provider proposal")
    current = load_current(verified(proposal["inputs"]["stage1"]))
    require(view["source_stage1_execution"] == proposal["inputs"]["stage1"], "Camera selection belongs to another Stage1 query")
    require(view["target"] == current["stage1"]["target"], "Camera selection target mismatch")
    candidate_path = verified(proposal["outputs"]["candidate"])
    with np.load(candidate_path, allow_pickle=False) as data:
        require(str(data["schema"]) == "p555.current_geometry_ik_candidate.v1", "Wrong candidate provider")
        vertices, faces = data["vertices_world_zup"].copy(), data["faces"].copy()
    require(vertices.shape == (10475, 3) and faces.shape == (20908, 3) and np.isfinite(vertices).all(), "Invalid full SMPL-X geometry")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import pyrender
    import trimesh
    from PIL import Image, ImageDraw
    renderer_source = PROJECT / "agent9/methods/p508_lingo_sam3_scene_first_stage1_20260831/code/render_lingo_fullscene_multiview_v1.py"
    spec = importlib.util.spec_from_file_location("p555_original_scene_renderer", renderer_source)
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    mesh_path = verified(current["target"]["artifacts"]["original_scene_mesh"])
    mesh = loader.load_scene_mesh(mesh_path)
    camera_record = view["selection"]["selected_camera"]
    camera = read_json(verified(camera_record))
    rgb_path = verified(view["selection"]["selected_source_rgb"])
    with Image.open(rgb_path) as reference:
        width, height = reference.size
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.94, .43, .13, 1.], roughnessFactor=.85)
    body_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, faces, process=False), material=material, smooth=False)
    scene = pyrender.Scene(bg_color=[.87, .88, .9, 1], ambient_light=[.55, .55, .55])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    body_node = scene.add(body_mesh)
    K = np.asarray(camera["K"])
    pose = loader.opencv_to_pyrender_pose(np.asarray(camera["world_to_camera"]))
    original_camera_node = scene.add(pyrender.IntrinsicsCamera(K[0,0], K[1,1], K[0,2], K[1,2]), pose=pose)
    original_light_node = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8), pose=pose)
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        rgb, _ = renderer.render(scene)
        scene.remove_node(body_node)
        empty, _ = renderer.render(scene)
    finally:
        renderer.delete()
    scene.remove_node(original_camera_node)
    scene.remove_node(original_light_node)
    body_node = scene.add(body_mesh)
    crop = view["selection"]["selected_crop_xyxy"]
    paths = {}
    for name, pixels in (("original_scene_with_actual_body", rgb), ("original_scene_empty_same_renderer", empty)):
        path = output / (name + ".png")
        Image.fromarray(pixels).save(path)
        paths[name] = artifact(path)
        path = output / (name + "_crop.png")
        Image.fromarray(pixels).crop(crop).resize((512, 512), Image.Resampling.LANCZOS).save(path)
        paths[name + "_crop"] = artifact(path)
    # The body-free paired render is an exact current geometry comparison,
    # while the original Stage1 reference also retains its actual source.
    centre = np.asarray(proposal["contact_world_xyz_m"])
    yaw = proposal["terminal_facing_yaw_rad"]
    diagnostic = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[.5, .5, .5])
    diagnostic_body_node = diagnostic.add(body_mesh)
    x, y = centre[:2]
    floor = trimesh.Trimesh(vertices=[[x-1.1, y-1.1, 0], [x+1.1, y-1.1, 0], [x+1.1, y+1.1, 0], [x-1.1, y+1.1, 0]],
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
    floor_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.84, .86, .88, 1], doubleSided=True)
    diagnostic.add(pyrender.Mesh.from_trimesh(floor, material=floor_material, smooth=False))
    renderer = pyrender.OffscreenRenderer(640, 640)
    isolated, additional_original = [], []
    try:
        for index, offset in enumerate((-50, 50)):
            azimuth = yaw + math.radians(offset)
            point = centre[:2] + 2.7 * np.array([math.cos(azimuth), math.sin(azimuth)])
            cam = loader.make_camera(index, index, 0, math.degrees(azimuth+math.pi), point,
                eye_height_m=1.10, look_height_m=.67, look_distance_m=2.7,
                width=640, height=640, vertical_fov_degrees=44)
            k = cam.intrinsics
            camera_pose = loader.opencv_to_pyrender_pose(cam.world_to_camera)
            cam_node = diagnostic.add(pyrender.IntrinsicsCamera(k[0,0], k[1,1], k[0,2], k[1,2]), pose=camera_pose)
            light = diagnostic.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8), pose=camera_pose)
            pixels, _ = renderer.render(diagnostic)
            full_cam_node = scene.add(pyrender.IntrinsicsCamera(k[0,0], k[1,1], k[0,2], k[1,2]), pose=camera_pose)
            full_light_node = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8), pose=camera_pose)
            full_pixels, _ = renderer.render(scene)
            amodal_mask, visible_mask = scene_body_visibility(renderer, scene, body_node, pyrender.RenderFlags.SEG)
            scene.remove_node(full_cam_node); scene.remove_node(full_light_node)
            full_path = output / f"additional_original_scene_{index}.png"
            Image.fromarray(full_pixels).save(full_path)
            total_pixels, visible_pixels = int(amodal_mask.sum()), int(visible_mask.sum())
            mask_paths = {}
            for name, mask_pixels in (("amodal", amodal_mask), ("scene_visible", visible_mask)):
                path = output / f"additional_original_scene_{index}_{name}_mask.png"
                Image.fromarray(mask_pixels.astype(np.uint8)*255).save(path)
                mask_paths[name] = artifact(path)
            additional_original.append(dict(image=artifact(full_path), world_to_camera=cam.world_to_camera.tolist(), K=k.tolist(),
                camera_placement_source="current_contact_facing_numeric_side_view",
                unoccluded_body_mask_pixels=total_pixels, visible_body_mask_pixels=visible_pixels,
                actual_body_visibility_fraction=visible_pixels / max(1, total_pixels),
                scene_occluders_hidden=False, masks=mask_paths,
                camera_body_coverage=camera_body_coverage(vertices, cam.world_to_camera, k, 640, 640),
                visibility_contract="all_original_scene_mesh_nodes_black_body_white_v2"))
            diagnostic.remove_node(cam_node); diagnostic.remove_node(light)
            im = Image.fromarray(pixels)
            draw = ImageDraw.Draw(im)
            draw.rectangle([0, 0, 640, 44], fill="white")
            draw.text((10, 6), "ACTUAL BODY | SCENE OCCLUDERS HIDDEN | DIAGNOSTIC", fill="black")
            draw.text((10, 24), "Gray plane = world z=0 reference, NOT the original scene", fill="black")
            path = output / f"isolated_actual_body_{index}.png"
            im.save(path)
            isolated.append(dict(image=artifact(path), world_to_camera=cam.world_to_camera.tolist(), K=k.tolist()))
    finally:
        renderer.delete()
    frozen = output / "render_geometry_keypose_v2_source.py"
    verified(source_at_start)
    shutil.copy2(SOURCE, frozen)
    value = dict(schema="p555.actual_geometric_provider_render.v1", source=artifact(frozen),
        source_proposal=artifact(proposal_path), source_stage1=proposal["inputs"]["stage1"],
        source_view_selection=artifact(view_path), source_keypose=artifact(candidate_path),
        original_mesh=artifact(mesh_path), original_camera=camera_record,
        original_reference=view["selection"]["selected_stage2_crop"],
        target_overlay=view["selection"]["selected_stage2_crop_target_overlay"],
        images=paths, isolated_body_views=isolated, additional_original_scene_views=additional_original, image_crop_xyxy=crop,
        additional_view_selection_rule="highest_actual_visible_body_mask_fraction_then_lower_index",
        selected_additional_view_index=max(range(len(additional_original)), key=lambda i: (additional_original[i]["actual_body_visibility_fraction"], -i)),
        source_scene_loader=artifact(renderer_source), original_scene_rendered=True,
        original_scene_vertices_textures_or_camera_changed=False, actual_body_vertices_changed=False,
        isolated_views_have_occluders_hidden=True, renderer_grants_no_semantic_or_physical_publication=True,
        mask_contract_version=2, masks_count_all_scene_mesh_occluders=True,
        visible_and_amodal_masks_saved=True, qwen_checked=False, elapsed_seconds=time.monotonic()-start)
    write_once(output / "receipt.json", value)
    print({"scene": proposal["scene_id"], "seconds": value["elapsed_seconds"], "render": str(output)}, flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()
    run(args.proposal, args.view, args.output, args.gpu)
