"""New numeric overview camera, without editing any evaluated motion/scene.

The original Stage2 camera is retained ONLY as a source-camera diagnostic.
Overview candidates use current all-frame body/route bounds, never Qwen or an
isolated-body main view. Real scene occluders participate in SEG depth tests.
V1 source/outputs remain untouched. No renderer grants motion or Memory credit.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

import visualize_geometry_motion as base
from memory_common import artifact, read_sealed, require, verified, write_once

BASE_SHA256 = "7f09899b933842ed705bec9c7e2fbbb6a37afb86020640a9613cc1854a95fa25"
SCHEMA = "p555.actual_geometry_motion_overview_visualization.v2"
POLICY = dict(schema="p555.current_motion_overview_camera_policy.v1", width=960, height=720,
    screening_width=384, screening_height=288,
    margin_pixels=32, minimum_camera_depth_m=.20, minimum_amodal_body_pixels=128,
    minimum_in_scene_visible_fraction=.45, maximum_full_frame_candidates=4,
    candidate_azimuth_degrees=list(range(0, 360, 45)), candidate_elevation_degrees=[8., 25.],
    candidate_vertical_fov_degrees=[60., 80.], scene_occluders_must_write_depth=True)


def corners(bounds):
    b = np.asarray(bounds, dtype=float)
    require(b.shape == (2, 3) and np.isfinite(b).all() and np.all(b[1] >= b[0]), "Invalid metric bounds")
    return np.array([[b[x, 0], b[y, 1], b[z, 2]] for x in (0, 1) for y in (0, 1) for z in (0, 1)])


def motion_bounds(vertices, route, contact):
    points = np.asarray(vertices)
    route, contact = np.asarray(route), np.asarray(contact)
    require(points.ndim == 3 and points.shape[-1] == 3 and points.size > 0 and np.isfinite(points).all(), "Invalid current motion vertices")
    require(route.ndim == 2 and route.shape[1] == 2 and len(route) >= 2 and np.isfinite(route).all()
            and contact.shape == (3,) and np.isfinite(contact).all(), "Invalid current route/contact")
    all_bounds = np.vstack((points.min(axis=(0,1)), points.max(axis=(0,1)), np.c_[route, np.zeros(len(route))], contact[None]))
    return np.stack((all_bounds.min(0), all_bounds.max(0)))


def projection_check(points, camera):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    pose, k = np.asarray(camera["world_to_camera"]), np.asarray(camera["K"])
    current = points @ pose[:3, :3].T + pose[:3, 3]
    require(np.isfinite(current).all(), "Invalid camera projection")
    depth = current[:, 2]
    positive = bool(np.all(depth >= POLICY["minimum_camera_depth_m"]))
    xy = current[:, :2] / np.maximum(depth[:, None], 1e-12)
    uv = xy * [k[0,0], k[1,1]] + [k[0,2], k[1,2]]
    margin = POLICY["margin_pixels"]
    inside = positive and bool(np.all(uv >= margin - 1e-7)
        and np.all(uv <= [camera["width"]-margin+1e-7, camera["height"]-margin+1e-7]))
    return dict(all_camera_depth_positive=positive, minimum_camera_depth_m=float(depth.min()),
        all_vertices_inside_safe_image_bounds=inside, projected_uv_bounds=[uv.min(0).tolist(), uv.max(0).tolist()],
        passed=positive and inside)


def overview_candidates(bounds):
    """Fit ALL body-bound corners; convexity covers every saved mesh vertex."""
    points = corners(bounds)
    centre = np.asarray(bounds).mean(0)
    width, height, margin = POLICY["width"], POLICY["height"], POLICY["margin_pixels"]
    output = []
    for elevation in POLICY["candidate_elevation_degrees"]:
        for fov in POLICY["candidate_vertical_fov_degrees"]:
            for azimuth in POLICY["candidate_azimuth_degrees"]:
                az, el = math.radians(azimuth), math.radians(elevation)
                outward = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
                forward = -outward
                right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
                down = np.cross(forward, right)
                rotation = np.stack((right, down, forward))
                focal = .5*height/math.tan(math.radians(fov)/2)
                local = (points-centre) @ rotation.T
                tx, ty = (width/2-margin)/focal, (height/2-margin)/focal
                distance = max(.5, float(np.max(np.abs(local[:,0])/tx-local[:,2])),
                    float(np.max(np.abs(local[:,1])/ty-local[:,2])),
                    float(POLICY["minimum_camera_depth_m"]-local[:,2].min())) + .10
                position = centre + outward*distance
                pose = np.eye(4); pose[:3,:3] = rotation; pose[:3,3] = -rotation@position
                camera = dict(schema="p555.numeric_current_motion_overview_camera.v1", candidate_id=len(output),
                    width=width, height=height, K=[[focal,0.,width/2],[0.,focal,height/2],[0.,0.,1.]],
                    world_to_camera=pose.tolist(), position_world_zup_m=position.tolist(), look_at_world_zup_m=centre.tolist(),
                    azimuth_degrees=azimuth, elevation_degrees=elevation, vertical_fov_degrees=fov,
                    coordinate_system="world_zup_metric", extrinsic_convention="opencv_world_to_camera",
                    numerical_source="actual_all_frame_10475_body_vertices_plus_current_route_contact_bounds",
                    original_stage2_camera=False, qwen_used=False, body_scene_or_motion_modified=False)
                camera["all_frame_bounds_projection"] = projection_check(points, camera)
                require(camera["all_frame_bounds_projection"]["passed"], "Numeric overview fit failed")
                output.append(camera)
    return output


def screening_camera(camera):
    """Same extrinsics/FOV, explicit smaller raster for eight-time screening."""
    result = dict(camera)
    factor = POLICY["screening_width"]/camera["width"]
    require(abs(POLICY["screening_height"]/camera["height"]-factor)<1e-12, "Screening aspect ratio differs")
    k = np.asarray(camera["K"], dtype=float).copy(); k[:2] *= factor
    result.update(K=k.tolist(), width=POLICY["screening_width"], height=POLICY["screening_height"],
        purpose="eight_time_mask_screening_only_not_main_render", source_main_candidate_id=camera["candidate_id"],
        source_main_resolution=[camera["width"],camera["height"]], uniform_raster_scale=factor)
    # Bounds proof is for the main camera's 32-pixel margin; it must not be
    # relabelled as the same pixel margin in this smaller screening raster.
    result.pop("all_frame_bounds_projection",None)
    return result


def segmentation_map(mesh_nodes, body_node):
    """CRITICAL: unmapped nodes are skipped, not blackened, by pyrender SEG."""
    nodes = list(mesh_nodes)
    require(body_node in nodes, "Body absent from actual scene mask nodes")
    return {node: [255,255,255] if node is body_node else [0,0,0] for node in nodes}


def mask_evidence(amodal, visible, frame):
    amodal, visible = np.asarray(amodal, bool), np.asarray(visible, bool)
    require(amodal.ndim == 2 and visible.shape == amodal.shape and np.all(~visible | amodal), "In-scene mask is not an occluded subset of the amodal body")
    total, observed = int(amodal.sum()), int(visible.sum())
    fraction = observed/max(1, total)
    passed = total >= POLICY["minimum_amodal_body_pixels"] and fraction >= POLICY["minimum_in_scene_visible_fraction"]
    return dict(frame=int(frame), amodal_body_pixels=total, in_scene_visible_body_pixels=observed,
                in_scene_visible_fraction=fraction, scene_occluders_retained_in_depth=True, passed=passed)


def visibility_summary(rows):
    require(rows, "No actual mask observations")
    return dict(frame_count=len(rows), minimum_visible_fraction=min(r["in_scene_visible_fraction"] for r in rows),
        mean_visible_fraction=float(np.mean([r["in_scene_visible_fraction"] for r in rows])),
        minimum_amodal_pixels=min(r["amodal_body_pixels"] for r in rows), all_passed=all(r["passed"] is True for r in rows))


class _SceneRenderer:
    def __init__(self, state, gpu, egl_device):
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYOPENGL_PLATFORM="egl", EGL_DEVICE_ID=str(egl_device))
        import pyrender
        import trimesh
        spec = importlib.util.spec_from_file_location("p555_overview_original_scene_loader", base.LOADER)
        loader = importlib.util.module_from_spec(spec); sys.modules[spec.name] = loader; spec.loader.exec_module(loader)
        self.p, self.t, self.loader, self.state = pyrender, trimesh, loader, state
        mesh = loader.load_scene_mesh(verified(state["bindings"]["mesh"]))
        self.scene = pyrender.Scene(bg_color=[.87,.88,.9,1.], ambient_light=[.55,.55,.55])
        self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        self.renderer = pyrender.OffscreenRenderer(POLICY["width"], POLICY["height"])
        from OpenGL.GL import glGetString, GL_RENDERER, GL_VERSION
        self.device = dict(requested_cuda_visibility=str(gpu), requested_egl_device_index=egl_device,
            egl_device_name=self.renderer._platform._egl_device.name, gl_renderer=glGetString(GL_RENDERER).decode(),
            gl_version=glGetString(GL_VERSION).decode(), physical_gpu_uuid_verified=False)
        self.camera_nodes = []
        self.material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.94,.43,.13,1.], roughnessFactor=.85)

    def camera(self, camera):
        for scene, node in self.camera_nodes: scene.remove_node(node)
        self.camera_nodes = []
        self.renderer.viewport_width, self.renderer.viewport_height = camera["width"], camera["height"]
        k = np.asarray(camera["K"])
        pose = self.loader.opencv_to_pyrender_pose(np.asarray(camera["world_to_camera"]))
        for scene in (self.scene,):
            node = scene.add(self.p.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2], znear=.05, zfar=100.), pose=pose)
            self.camera_nodes.append((scene,node))
        node = self.scene.add(self.p.DirectionalLight(color=np.ones(3), intensity=1.8), pose=pose)
        self.camera_nodes.append((self.scene,node))

    def frame(self, index, *, masks=True, rgb=False):
        vertices, faces = self.state["vertices"][index], self.state["faces"]
        mesh = self.t.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
        require(np.array_equal(mesh.vertices, vertices) and np.array_equal(mesh.faces, faces), "Renderer altered saved evaluated mesh")
        body = self.p.Mesh.from_trimesh(mesh, material=self.material, smooth=False)
        full_node = self.scene.add(body)
        try:
            evidence, pixels = None, None
            if masks:
                # Deliberately omit occluders ONLY in this amodal measurement.
                # Keep the same scene object so large scene VBOs stay cached.
                unoccluded, _ = self.renderer.render(self.scene, flags=self.p.RenderFlags.SEG,
                    seg_node_map={full_node: [255,255,255]})
                visible, _ = self.renderer.render(self.scene, flags=self.p.RenderFlags.SEG,
                    seg_node_map=segmentation_map(self.scene.mesh_nodes, full_node))
                evidence = mask_evidence(unoccluded[...,0] > 127, visible[...,0] > 127, index)
            if rgb: pixels, _ = self.renderer.render(self.scene)
            return evidence, pixels
        finally:
            self.scene.remove_node(full_node)

    def close(self):
        self.renderer.delete()


def _sheet(images, state, path, *, diagnostic=False):
    from PIL import Image, ImageDraw
    width, height = images[0].size
    sheet = Image.new("RGB", (4*width+60, 2*height+140), "white")
    draw = ImageDraw.Draw(sheet)
    title = "ORIGINAL STAGE2 CAMERA: SOURCE DIAGNOSTIC ONLY (MAY CROSS THE BODY)" if diagnostic else "NEW NUMERIC OVERVIEW CAMERA | FULL ORIGINAL SCENE | ALL-FRAME VISIBILITY CHECKED"
    draw.text((12,9), title, font=base._font(22), fill="#a52424" if diagnostic else "#16223a")
    draw.text((12,43), f"Scene {state['evaluation']['scene_id']} | affordance {state['launch']['affordance']} | "+base.status_label(state["evaluation"]), font=base._font(21), fill="#a52424")
    draw.text((12,74), "Actual evaluated vertices, no reskinning/edit. Occluders retained. Requested CONTACT phase does not mean task completed.", font=base._font(18), fill="#374459")
    for i, image in enumerate(images): sheet.paste(image, (12+(i%4)*(width+12), 110+(i//4)*(height+12)))
    sheet.save(path)


def render(execution, evaluation, output, *, animation="gif", gpu=5, egl_device=5):
    require(animation in {"none", "gif"}, "V2 CPU encoder supports none or GIF; no optional package installation")
    require(type(gpu) is type(egl_device) is int and min(gpu, egl_device) >= 0, "Explicit GPU/EGL index required")
    started = time.monotonic()
    sources = [artifact(__file__), artifact(base.__file__)]
    require(sources[1]["sha256"] == BASE_SHA256, "Frozen v1 evidence binder changed")
    state = base.load_inputs(execution, evaluation)
    output = Path(output).resolve(); require(not output.exists(), "Refuse to overwrite visualization"); output.mkdir(parents=True)
    bounds = motion_bounds(state["vertices"], state["route"], state["contact"])
    candidates = overview_candidates(bounds)
    renderer = _SceneRenderer(state, gpu, egl_device)
    sampled, full_checks, selected, frames, gif_images = [], [], None, [], []
    selected_times = [row["frame"] for row in state["selected"]]
    try:
        for camera in candidates:
            screen = screening_camera(camera)
            renderer.camera(screen)
            rows = [renderer.frame(frame)[0] for frame in selected_times]
            sampled.append(dict(camera=camera, screening_camera=screen, frame_masks=rows, summary=visibility_summary(rows)))
        ranked = sorted(sampled, key=lambda row: (-row["summary"]["minimum_visible_fraction"],
            -row["summary"]["mean_visible_fraction"], -row["summary"]["minimum_amodal_pixels"], row["camera"]["candidate_id"]))
        for row in [r for r in ranked if r["summary"]["all_passed"]][:POLICY["maximum_full_frame_candidates"]]:
            renderer.camera(row["camera"])
            masks = [renderer.frame(i)[0] for i in range(len(state["vertices"]))]
            tested = dict(camera=row["camera"], frame_masks=masks, summary=visibility_summary(masks))
            full_checks.append(tested)
            if tested["summary"]["all_passed"]:
                selected = tested; break
        audit = dict(schema="p555.actual_full_motion_overview_camera_audit.v1", sources=sources, inputs=state["bindings"],
            bounds_world_zup_m=bounds.tolist(), policy=POLICY, sampled_candidates=sampled, full_frame_candidates=full_checks,
            selected_candidate_id=selected["camera"]["candidate_id"] if selected else None,
            overview_camera_pass=selected is not None, no_occluders_hidden=True, qwen_camera_selection_used=False)
        write_once(output / "camera_audit.json", audit)
        require(selected is not None, "No full-motion overview passed visibility; camera_audit retained, no main GIF/image published")
        renderer.camera(selected["camera"])
        by_frame = {row["frame"]: row for row in state["selected"]}
        images = {}
        indices = range(len(state["vertices"])) if animation == "gif" else selected_times
        for index in indices:
            _, pixels = renderer.frame(index, masks=False, rgb=True)
            label = by_frame[index]["label"] if index in by_frame else "ACTUAL GENERATED FRAME"
            image = base._annotated(pixels, base.status_label(state["evaluation"]),
                f"NEW OVERVIEW | frame {index} | {index/base.FPS:.2f}s | {label}")
            if index in by_frame:
                path = output/f"overview_frame_{index:06d}.png"; image.save(path)
                frames.append({**by_frame[index], "image": artifact(path)}); images[index] = image
            if animation == "gif": gif_images.append(image)
        sheet_path = output/"overview_contact_sheet.png"
        _sheet([images[index] for index in selected_times], state, sheet_path)
        # Original camera still has full provenance, but cannot be mistaken for
        # the primary full-motion visual inspection after crossing the body.
        renderer.camera(state["camera"])
        diagnostic = []
        for index in selected_times:
            _, pixels = renderer.frame(index, masks=False, rgb=True)
            diagnostic.append(base._annotated(pixels, "SOURCE CAMERA DIAGNOSTIC ONLY - MAY CLIP/CROSS BODY",
                f"Original Stage2 camera | frame {index} | {index/base.FPS:.2f}s"))
        diagnostic_path = output/"original_camera_diagnostic.png"; _sheet(diagnostic,state,diagnostic_path,diagnostic=True)
    finally:
        renderer.close()
    animation_path = output/"overview_motion.gif" if animation == "gif" else None
    if gif_images:
        gif_images[0].save(animation_path, save_all=True, append_images=gif_images[1:], duration=50, loop=0, optimize=False, disposal=2)
    xy_path = output/"trajectory_xy.png"; base.trajectory_plot(state,xy_path)
    for rec in [*sources, *state["bindings"].values()]: verified(rec)
    result = dict(schema=SCHEMA, sources=sources, inputs=state["bindings"], source_camera_audit=artifact(output/"camera_audit.json"),
        scene_id=state["evaluation"]["scene_id"], affordance=state["launch"]["affordance"], seed=state["launch"]["seed"],
        instruction=state["evaluation"]["instruction"], frame_count=len(state["vertices"]), vertex_count=10475, fps=base.FPS,
        overview_camera=selected["camera"], overview_visibility=selected["summary"], original_camera_preserved_for_main_view=False,
        main_view_kind="new_numeric_current_motion_overview_not_original_stage2_camera", original_camera_diagnostic_only=True,
        overview_full_body_bounds_positive_depth_and_in_frame=True, all_frames_mask_visibility_checked=True,
        original_scene_geometry_unchanged=True, scene_occluders_hidden=False, evaluated_vertices_reused_verbatim=True,
        body_reskinned=False, motion_or_root_modified=False, qwen_camera_selection_used=False,
        frames=frames, contact_sheet=artifact(sheet_path), original_camera_diagnostic=artifact(diagnostic_path), trajectory=artifact(xy_path),
        animation=artifact(animation_path) if animation_path else None, animation_all_source_frames_preserved=bool(animation_path),
        motion_publishable=state["evaluation"]["motion_publishable"], failed_release_gates=state["failed"],
        positive_credit=0, h3_claimed=False, renderer_grants_no_publication=True, render_device=renderer.device,
        elapsed_seconds=time.monotonic()-started)
    write_once(output/"receipt.json",result)
    with (output/"README.md").open("x") as stream:
        stream.write("# Actual motion: new numeric overview\n\n"+base.status_label(state["evaluation"])+"\n\n"
            "[Main eight times](overview_contact_sheet.png) · [Root/route/contact XY](trajectory_xy.png)\n\n"
            + ("[All-frame 20 Hz GIF](overview_motion.gif)\n\n" if animation_path else "")
            + "The MAIN camera is newly computed from this actual motion's complete body bounds and current route, not Qwen and not the original Stage2 camera. "
            "All frames pass positive camera depth, safe image bounds, and the recorded amodal/in-scene mask visibility policy. "
            "Scene mesh and body frames are unchanged; furniture occlusion is retained, so visibility does not mean every body pixel is exposed.\n\n"
            "[Original Stage2 camera diagnostic](original_camera_diagnostic.png) may cross the body near the starting point; it is NOT the full-motion inspection view. "
            "Mask tests explicitly include black scene nodes in depth rendering; unmapped pyrender SEG nodes would incorrectly hide occluders. "
            "The unchanged evaluator failed gates remain failed; geometry-guidance visuals grant no motion publication or learned Memory credit.\n")
        stream.flush(); os.fsync(stream.fileno())
    return output/"receipt.json"


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution",type=Path,required=True); parser.add_argument("--evaluation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--animation",choices=("none","gif"),default="gif")
    parser.add_argument("--gpu",type=int,default=5); parser.add_argument("--egl-device",type=int,default=5)
    args=parser.parse_args()
    print(render(args.execution,args.evaluation,args.output,animation=args.animation,gpu=args.gpu,egl_device=args.egl_device),flush=True)
