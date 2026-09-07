"""Current full-motion camera fit and real scene-occluded SEG rendering.

All masks use the original scene; only the separate amodal measurement omits
occluders. Saved evaluated vertices are reused verbatim, never reskinned.
"""
from __future__ import annotations
import math
import os
import numpy as np
from hsi.common.artifacts import require, verified

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
        from hsi.stage2 import view_numeric as loader
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
