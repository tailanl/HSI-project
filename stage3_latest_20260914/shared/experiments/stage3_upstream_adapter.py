"""Strict, condition-only Stage 1/2 -> experimental Stage 3 adapter.

This does not load a denoiser, generate motion, invent supervision, or release
motion. Only the ordinary successful Stage 2 receipt is admitted. The original
research runtime and old ReMoGen execution entry points are not modified.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from hsi.common.artifacts import artifact, read_sealed, require, verified, write_once
from hsi.stage3_sequence.contracts import SequenceCondition
from hsi.stage3_sequence.constraints import matrix_to_rotation_6d
from hsi.stage3_sequence.data import axis_angle_to_matrix
from hsi.stage3_sequence.geometry import GridSDF, REGION_NAMES, SDFQuery
from hsi.stage3_sequence.mesh_body import MeshBody


SCHEMA = "hsi.stage3_sequence.verified_upstream_condition.v1"
Q_YUP_TO_ZUP = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
ROUNDTRIP_TOLERANCE_M = 5e-5
ROUTE_CHANNELS = ["delta_x_m", "delta_y_m", "polyline_segment_length_m",
                  "arrival_sin_yaw", "arrival_cos_yaw", "turn_from_previous_rad",
                  "target_surface_height_m", "is_contact_transition"]


def runtime_sources():
    directory = Path(__file__).resolve().parents[1] / "hsi/stage3_sequence"
    return {p.name: artifact(p) for p in sorted(directory.glob("*.py"))}


def admission_source():
    from hsi.stage3 import compiler
    return artifact(compiler.__file__)


def pose_to_motion(root_xyz, body_axis_angle, global_rotation):
    """Decode the explicit pelvis/root convention, without fitting or retargeting."""
    root = torch.as_tensor(root_xyz, dtype=torch.float32)
    pose = torch.as_tensor(body_axis_angle, dtype=torch.float32).reshape(21, 3)
    rotation = torch.as_tensor(global_rotation, dtype=torch.float32)
    require(root.shape == (3,) and rotation.shape == (3, 3), "Malformed pose transform")
    require(bool(torch.isfinite(root).all()) and bool(torch.isfinite(rotation).all()), "Nonfinite transform")
    require(torch.allclose(rotation.T @ rotation, torch.eye(3), atol=2e-5, rtol=0)
            and abs(float(torch.linalg.det(rotation))-1) < 2e-5, "Global transform is not SO(3)")
    rotations = torch.cat((rotation[None], axis_angle_to_matrix(pose)), 0)
    return torch.cat((root, matrix_to_rotation_6d(rotations).flatten()))


def shaped_static_history(body, start_xy, yaw, *, floor_clearance=.01):
    """Same fixed upstream shape/hands; one newly materialized static frame.

    Only the upstream initial XY and yaw are used. Its former zero-shape pelvis
    height is not transplanted to a different body. Grounding is recomputed
    from all actual vertices and no temporal or target pose is borrowed.
    """
    start_xy = np.asarray(start_xy, dtype=np.float64)
    require(start_xy.shape == (2,) and np.isfinite(start_xy).all(), "Invalid static start XY")
    require(math.isfinite(float(yaw)) and 0 <= floor_clearance <= .1, "Invalid static yaw/clearance")
    zero_pose = torch.zeros(21, 3)
    initial = pose_to_motion([0., 0., 0.], zero_pose, Q_YUP_TO_ZUP)
    with torch.no_grad():
        native = body.mesh(initial[None, None])
    hip = native.joints[0, 0, 2, :2]-native.joints[0, 0, 1, :2]
    require(float(hip.norm()) > 1e-7, "Shaped initial hips cannot determine yaw")
    initial_yaw = math.atan2(float(hip[0]), -float(hip[1]))
    delta = float(yaw)-initial_yaw
    rz = torch.tensor([[math.cos(delta), -math.sin(delta), 0.],
                       [math.sin(delta), math.cos(delta), 0.], [0., 0., 1.]])
    motion = pose_to_motion([0., 0., 0.], zero_pose, rz @ Q_YUP_TO_ZUP)
    with torch.no_grad():
        unplaced = body.mesh(motion[None, None])
        root = torch.tensor([start_xy[0], start_xy[1], floor_clearance-float(unplaced.vertices[..., 2].min())], dtype=torch.float32)
        motion[:3] = root
        result = body.mesh(motion[None, None])
    require(abs(float(result.vertices[..., 2].min())-floor_clearance) <= 2e-6,
            "Same-shape static full mesh does not touch the declared ground clearance")
    hip = result.joints[0, 0, 2, :2]-result.joints[0, 0, 1, :2]
    measured_yaw = math.atan2(float(hip[0]), -float(hip[1]))
    require(abs(math.atan2(math.sin(measured_yaw-yaw), math.cos(measured_yaw-yaw))) <= 2e-5,
            "Same-shape static yaw does not match current Stage1 start")
    return dict(motion=motion.numpy(), vertices=result.vertices[0, 0].numpy(), joints=result.joints[0, 0].numpy(),
        betas=body.betas[0].cpu().numpy(), left_hand_pose=body.left_hand_pose.cpu().numpy(),
        right_hand_pose=body.right_hand_pose.cpu().numpy(),
        root_xyz_yaw_world_zup=np.r_[motion[:3].numpy(), measured_yaw])


def save_shaped_history(output, body, current):
    output = Path(output)
    root = current['source_root']
    arrays = shaped_static_history(body, root[:2], float(root[3]))
    output.mkdir(parents=True, exist_ok=False)
    with (output/'same_shape_static_seed.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    receipt = write_once(output/'receipt.json', dict(schema=SCHEMA+'.same_shape_static_history',
        source=artifact(__file__), source_stage1=current['records']['stage1_execution'],
        source_keypose=current['records']['candidate'], source_body=current['records']['body_model'],
        seed=artifact(output/'same_shape_static_seed.npz'), original_stage1_root_xyz_yaw=np.asarray(root).tolist(),
        actual_same_shape_root_xyz_yaw=arrays['root_xyz_yaw_world_zup'].tolist(),
        fixed_upstream_betas=arrays['betas'].tolist(), history_frames=2, unique_physical_frames=1,
        floor_clearance_m=.01, body_local_pose='zero_neutral_static_NOT_target_articulation',
        history_rule='repeat_exact_same_new_frame_twice', source_future_or_dataset_motion=False,
        shape_projection_or_change=False, temporal_motion_generated=False, motion_release_authorized=False))
    return arrays, receipt


class RegisteredScene(nn.Module):
    """Union of target, other original solids, and a declared world-Z ground.

    No target geometry is removed. Cell-centre bounds are deliberately different
    from the old Stage 2 voxel-edge bounds. Domain-outside remains unknown.
    """
    def __init__(self, target_xyz, other_xyz, edge_bounds, floor_z=0.):
        super().__init__()
        target, other = torch.as_tensor(target_xyz), torch.as_tensor(other_xyz)
        bounds = torch.as_tensor(edge_bounds, dtype=torch.float32)
        require(target.shape == other.shape and target.ndim == 3, "SDF shape mismatch")
        require(bounds.shape == (2, 3), "Malformed SDF edge bounds")
        spacing = (bounds[1]-bounds[0]) / target.new_tensor(target.shape)
        centres = torch.stack((bounds[0]+spacing/2, bounds[1]-spacing/2))
        self.target = GridSDF(target.permute(2, 1, 0), centres)
        self.other = GridSDF(other.permute(2, 1, 0), centres)
        self.floor_z = float(floor_z)
        require(math.isfinite(self.floor_z), "Nonfinite floor")
        self.register_buffer("edge_bounds", bounds.clone())

    def query(self, points):
        target, other = self.target.query(points), self.other.query(points)
        distances = torch.minimum(torch.minimum(target.distances, other.distances), points[..., 2]-self.floor_z)
        return SDFQuery(distances, target.outside | other.outside)

    sample = query


def mesh_body_from_current(current, model_path):
    """Keep actual neutral anatomy and the Stage 2 non-flat default hand mean."""
    from hsi.stage2.body_model import load_fixed_smplx
    from smplx import SMPLXLayer
    ordinary = load_fixed_smplx(Path(model_path), torch.device("cpu"))
    left = axis_angle_to_matrix(ordinary.left_hand_mean.reshape(15, 3).detach())
    right = axis_angle_to_matrix(ordinary.right_hand_mean.reshape(15, 3).detach())
    require(torch.count_nonzero(ordinary.left_hand_pose).item() == 0
            and torch.count_nonzero(ordinary.right_hand_pose).item() == 0,
            "Stage 2 default hand articulation changed")
    layer = SMPLXLayer(str(model_path), gender="neutral", num_betas=10,
        num_expression_coeffs=10, use_pca=False, flat_hand_mean=True,
        use_face_contour=False, dtype=torch.float32).cpu().eval()
    for key in ("v_template", "J_regressor", "lbs_weights", "shapedirs", "posedirs"):
        a, b = getattr(ordinary, key), getattr(layer, key)
        require(a.shape == b.shape and torch.equal(a, b), "Stage 2 / matrix-body anatomy mismatch: "+key)
    arrays = current["arrays"]
    vertices = np.asarray(arrays["vertices_world_zup"])
    subset = np.asarray(arrays["fixed_generated_butt_vertex_indices"], dtype=np.int64)
    count = min(len(subset), max(8, math.ceil(len(subset)*.15)))
    # This fixed vertex set reproduces the admitted terminal lower-envelope Z.
    # It is not re-selected per generated frame and is not an arbitrary joint.
    gluteal = subset[np.argsort(vertices[subset, 2], kind="stable")[:count]]
    dominant = ordinary.lbs_weights[:, :22].argmax(1).cpu().numpy()
    anatomical = current["anatomy"]
    foot_groups = []
    for mask in (anatomical.left_foot_support, anatomical.right_foot_support):
        ids = np.flatnonzero(mask.cpu().numpy())
        foot_groups.append(ids[np.argsort(vertices[ids, 2], kind="stable")[:min(32, len(ids))]])
    ids = dict(pelvis=gluteal.tolist(), left_thigh=np.flatnonzero(dominant == 1).tolist(),
        right_thigh=np.flatnonzero(dominant == 2).tolist(), left_foot=foot_groups[0].tolist(),
        right_foot=foot_groups[1].tolist(), left_hand=np.flatnonzero(dominant == 20).tolist(),
        right_hand=np.flatnonzero(dominant == 21).tolist(), back=np.flatnonzero(dominant == 9).tolist())
    body = MeshBody(layer, torch.as_tensor(arrays["betas"], dtype=torch.float32),
        left_hand_pose=left, right_hand_pose=right, region_vertex_ids=ids)
    return body, dict(region_vertex_ids=ids,
        region_policy="fixed_terminal_gluteal_lower_envelope_and_foot_low32; inactive_other_regions_dominant_LBS",
        left_hand_matrices=left.tolist(), right_hand_matrices=right.tolist(),
        hand_policy="Rodrigues(Stage2_SMPLX_nonflat_default_hand_mean); fixed_across_time",
        body_betas=arrays["betas"].tolist(), full_vertex_count=10475,
        contact_centroid_is_not_support_force_or_polygon=True)


def roundtrip(body, motion, vertices, joints, label):
    with torch.no_grad():
        actual = body.mesh(motion.reshape(1, 1, 135))
    verr = float((actual.vertices[0, 0]-torch.as_tensor(vertices)).abs().max())
    jerr = float((actual.joints[0, 0]-torch.as_tensor(joints)).abs().max())
    require(verr <= ROUNDTRIP_TOLERANCE_M and jerr <= ROUNDTRIP_TOLERANCE_M,
            f"{label} full mesh/J22 roundtrip failed: V={verr}, J={jerr}")
    return dict(vertices_max_abs_error_m=verr, joints22_max_abs_error_m=jerr,
                tolerance_m=ROUNDTRIP_TOLERANCE_M, vertex_count=10475,
                no_fit_retarget_rotation_or_translation_correction=True)


def surface_points(current, count):
    """Deterministic samples of actual occupancy boundary faces plus actual floor."""
    from hsi.stage3.sdf import _surface_faces_zup
    from hsi.stage2.refine_contract import GRID_LOWER_XYZ, GRID_UPPER_XYZ
    require(type(count) is int and 64 <= count <= 8192, "Use 64..8192 scene points")
    rows = []
    for mask, number, owner in ((current["fields"].target_mask, count//4, 1.),
                                (current["fields"].collision_mask, count//2, 0.)):
        p, n = _surface_faces_zup(mask, lower_xyz=GRID_LOWER_XYZ, upper_xyz=GRID_UPPER_XYZ)
        ids = np.linspace(0, len(p)-1, number, dtype=np.int64)
        rows.append((p[ids], np.column_stack((n[ids], np.full(number, owner, np.float32)))))
    floor_count = count-sum(len(x[0]) for x in rows)
    side = math.ceil(math.sqrt(floor_count))
    x, y = np.meshgrid(np.linspace(-2.99, 2.99, side), np.linspace(-3.99, 3.99, side))
    p = np.column_stack((x.ravel(), y.ravel(), np.zeros(side*side)))[:floor_count]
    f = np.tile([0., 0., 1., 0.], (floor_count, 1))
    rows.append((p, f))
    return (torch.tensor(np.concatenate([r[0] for r in rows]), dtype=torch.float32),
            torch.tensor(np.concatenate([r[1] for r in rows]), dtype=torch.float32))


def _inside_triangles_xy(point, triangles):
    tri = np.asarray(triangles, dtype=np.float64)[..., :2]
    a, u, v = tri[:, 0], tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    p = np.asarray(point, dtype=np.float64)-a
    cross = lambda x, y: x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
    det = cross(u, v)
    valid = np.abs(det) > 1e-12
    denominator = np.where(valid, det, 1.)
    b, c = cross(p, v)/denominator, cross(u, p)/denominator
    return bool(np.any(valid & (b >= -1e-6) & (c >= -1e-6) & (b+c <= 1+1e-6)))


def build_condition(*, history, keypose, body, route_xy, terminal_yaw,
                    scene_points, scene_features, contact_goal, triangles, surface_normal,
                    total_frames, fps, minimum_hold_frames=8):
    """One admitted full-body target; route vertices remain geometry, not poses."""
    require(type(total_frames) is int and 8 <= total_frames <= 240, "Future length outside diagnostic bounds")
    require(type(minimum_hold_frames) is int and minimum_hold_frames >= 1, "Bad minimum hold")
    require(type(fps) in (float, int) and not isinstance(fps, bool) and math.isfinite(fps) and fps > 0, "Bad FPS")
    route = np.asarray(route_xy, dtype=np.float64)
    require(route.ndim == 2 and route.shape[1] == 2 and len(route) >= 2 and np.isfinite(route).all(), "Bad real route")
    delta = np.diff(route, axis=0)
    require(np.all(np.linalg.norm(delta, axis=1) > 1e-8), "Repeated route node")
    # Multiple travel events reference the same target, not fabricated full poses.
    segments, m = len(delta), len(delta)+2
    minimum = torch.ones(1, m, dtype=torch.long)
    minimum[0, -1] = minimum_hold_frames
    require(int(minimum.sum()) <= total_frames, "Total horizon too short for all route events and hold")
    h, k = history.reshape(1, 2, 135), keypose.reshape(1, 1, 135)
    z = lambda *s: torch.zeros(*s, dtype=torch.float32)
    t = lambda *s: torch.ones(*s, dtype=torch.bool)
    f = lambda *s: torch.zeros(*s, dtype=torch.bool)
    i = lambda *s: torch.zeros(*s, dtype=torch.long)
    with torch.no_grad():
        hj, kj = body.joints_world(h), body.joints_world(k)
        targets = body.region_points(k).clone()
    require(_inside_triangles_xy(targets[0, 0, 0, :2].numpy(), triangles),
            "Terminal gluteal centroid is outside the actual selected contact triangles")
    targets[0, 0, 0, 2] = float(contact_goal[2])
    targets[0, 0, 3:5, 2] = 0.
    active = f(1, 1, 8); active[..., [0, 3, 4]] = True
    normals = z(1, 1, 8, 3); normals[..., 2] = 1.
    surface_normal = torch.as_tensor(surface_normal, dtype=torch.float32)
    require(surface_normal.shape == (3,) and bool(torch.isfinite(surface_normal).all())
            and float(surface_normal.norm()) > 1e-7, "Selected surface normal is malformed or degenerate")
    require(float(surface_normal[2]) > 0, "Selected sit support normal must point upward; never silently flip")
    normals[0, 0, 0] = surface_normal/surface_normal.norm()
    target_ids = i(1, 1, 8)-1; target_ids[..., 0] = 1; target_ids[..., 3:5] = 0
    initial = f(1, 8); initial[:, 3:5] = True
    initial_ids = i(1, 8)-1; initial_ids[:, 3:5] = 0
    contact_slots = f(1, m, 8); contact_slots[:, -1] = active[:, 0]
    slot_types = i(1, m); slot_types[:, -2:] = torch.tensor([1, 2])
    route_features = z(1, m, 8)
    hip_axis = hj[0, -1, 2, :2]-hj[0, -1, 1, :2]
    require(float(hip_axis.norm()) > 1e-7, "Initial anatomical yaw is degenerate")
    yaw_before = math.atan2(float(hip_axis[0]), -float(hip_axis[1]))
    for index, dxy in enumerate(delta):
        yaw = math.atan2(float(dxy[1]), float(dxy[0]))
        turn = math.atan2(math.sin(yaw-yaw_before), math.cos(yaw-yaw_before))
        route_features[0, index] = torch.tensor([*dxy, np.linalg.norm(dxy), math.sin(yaw), math.cos(yaw), turn, contact_goal[2], 0.])
        yaw_before = yaw
    dxy = k[0, 0, :2].numpy()-route[-1]
    turn = math.atan2(math.sin(terminal_yaw-yaw_before), math.cos(terminal_yaw-yaw_before))
    route_features[0, -2] = torch.tensor([*dxy, np.linalg.norm(dxy), math.sin(terminal_yaw), math.cos(terminal_yaw), turn, contact_goal[2], 1.])
    route_features[0, -1, 6:] = torch.tensor([float(contact_goal[2]), 1.])
    return SequenceCondition(history=h, history_mask=t(1, 2), history_joints=hj-h[..., None, :3],
        initial_contact_active=initial, initial_contact_target_ids=initial_ids,
        keyposes=k, keypose_mask=t(1, 1), keypose_joints=kj-k[..., None, :3],
        keypose_text=z(1, 1, 512), semantic_ids=i(1, 1),
        scene_points=scene_points[None], scene_features=scene_features[None], scene_mask=t(1, len(scene_points)),
        contact_targets=targets, contact_normals=normals, contact_active=active,
        contact_body_regions=i(1, 1, 8)-1, contact_target_ids=target_ids,
        locked_root=t(1, 1, 3), locked_joints=t(1, 1, 22), root_tolerance=z(1, 1, 3), joint_tolerance=z(1, 1, 22),
        slot_mask=t(1, m), slot_types=slot_types, slot_keyposes=i(1, m),
        slot_contact_active=contact_slots, slot_release_active=f(1, m, 8),
        minimum_frames=minimum, keypose_slots=torch.tensor([[m-2]]), route_features=route_features,
        total_frames=torch.tensor([total_frames]), fps=float(fps)).validate()


def save_condition(path, condition):
    """Own condition-only schema: never fabricate SequenceExample supervision."""
    condition.validate()
    arrays = {"schema": np.array(SCHEMA)}
    for field in fields(condition):
        value = getattr(condition, field.name)
        arrays["condition__"+field.name] = np.asarray(value) if field.name == "fps" else value.detach().cpu().numpy()
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_condition(path, device="cpu"):
    expected = {"schema", *("condition__"+f.name for f in fields(SequenceCondition))}
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == expected, "Condition archive has unknown/missing fields or supervision")
        require(archive["schema"].shape == () and str(archive["schema"]) == SCHEMA, "Wrong condition schema")
        values = {f.name: float(archive["condition__fps"]) if f.name == "fps"
                  else torch.from_numpy(np.array(archive["condition__"+f.name], copy=True))
                  for f in fields(SequenceCondition)}
    return SequenceCondition(**values).to(device).validate()


@dataclass
class PreparedUpstream:
    condition: SequenceCondition
    body: MeshBody
    scene: RegisteredScene
    receipt: dict
    route_xy: np.ndarray
    keypose_vertices: np.ndarray
    keypose_joints: np.ndarray
    scene_mesh_path: Path


def _admit(stage2_path, body_asset):
    from hsi.stage3.compiler import admit
    from hsi.stage3.seed import verify_neutral_assets
    assets = verify_neutral_assets(candidate_body=body_asset, runtime_body=body_asset, render_body=body_asset)
    return admit(Path(stage2_path), assets, require_zero_shape=False), assets


def prepare(stage2_path, output, *, body_asset, total_frames=120, fps=20.,
            scene_point_count=512, minimum_hold_frames=8):
    """Prepare a new explicit directory; failed Stage 2 is never bypassed."""
    from hsi.stage3.packet import route_contact_check
    from hsi.stage2.sdf import BoundSDFCache
    from hsi.stage2.refine_contract import GRID_LOWER_XYZ, GRID_UPPER_XYZ
    output, started = Path(output).resolve(), time.monotonic()
    require(not output.exists(), "Refuse to overwrite adapter output")
    source = artifact(__file__); sources = runtime_sources(); admitted_source = admission_source()
    current, assets = _admit(stage2_path, body_asset)
    arrays, records = current["arrays"], current["records"]
    body, body_receipt = mesh_body_from_current(current, body_asset)
    keypose = pose_to_motion(arrays["root_xyz_yaw"][:3], arrays["body_pose_axis_angle"],
                            torch.as_tensor(arrays["pose_frame_to_world_rotation"]) @ Q_YUP_TO_ZUP)
    keypose_check = roundtrip(body, keypose, arrays["vertices_world_zup"], arrays["joints_world_zup"], "Stage2")
    output.mkdir(parents=True, exist_ok=False)
    seed, seed_receipt = save_shaped_history(output/"static_history", body, current)
    history_pose = torch.from_numpy(seed['motion'])
    history_check = roundtrip(body, history_pose, seed['vertices'], seed['joints'], "same_shape_static_history")
    route = np.asarray(current["keys"]["collision_safe_control_polyline"]["nodes_world_xy_m"], dtype=np.float64)
    same_shape_route_check = route_contact_check(route[-1], float(history_pose[2]), arrays, current['fields'].non_target.sample)
    points, features = surface_points(current, scene_point_count)
    condition = build_condition(history=history_pose.repeat(2, 1), keypose=keypose, body=body,
        route_xy=route, terminal_yaw=float(arrays["root_xyz_yaw"][3]), scene_points=points, scene_features=features,
        contact_goal=arrays["contact_goal_world_xyz_zup_m"], triangles=current["triangles"],
        surface_normal=current['surface']['surface']['normal_world_zup'],
        total_frames=total_frames, fps=fps, minimum_hold_frames=minimum_hold_frames)
    cache = BoundSDFCache(verified(records["sdf_cache"]), verified(records["stage1_execution"]))
    scene = RegisteredScene(cache.arrays["target_sdf"], cache.arrays["collision_sdf"], [GRID_LOWER_XYZ, GRID_UPPER_XYZ])
    save_condition(output/"condition.npz", condition)
    with (output/"geometry.npz").open("xb") as stream:
        np.savez_compressed(stream, route_xy=route, stage2_vertices=arrays["vertices_world_zup"],
            stage2_joints=arrays["joints_world_zup"], faces=arrays["faces"], contact_triangles=current["triangles"])
    require(source == artifact(__file__) and sources == runtime_sources() and admitted_source == admission_source(), "Adapter/runtime/admission source drift")
    for row in records.values(): verified(row)
    receipt = write_once(output/"receipt.json", dict(schema=SCHEMA, status="verified_upstream_condition_not_motion_release",
        source=source, runtime_sources=sources, admission_source=admitted_source, inputs=records, body_assets=assets, body=body_receipt,
        condition=artifact(output/"condition.npz"), geometry=artifact(output/"geometry.npz"),
        static_seed_receipt=artifact(output/"static_history/receipt.json"),
        roundtrip=dict(stage2=keypose_check, history=history_check),
        same_shape_route_contact_validation=same_shape_route_check,
        contact_normal_binding=dict(source_target=records['stage1_target'],
            selected_surface_id=current['target']['selected_surface_id'],
            selected_surface_sha256=current['target']['selected_surface_sha256'],
            source_surface_normal_world_zup=current['surface']['surface']['normal_world_zup'],
            normalized_gluteal_normal_world_zup=condition.contact_normals[0, 0, 0].tolist(),
            foot_normal_world_zup=[0., 0., 1.],
            policy='normalize_selected_real_support_surface_normal; preserve_sign; feet_world_ground_plus_Z',
            stage2_body_root_betas_or_contact_target_modified=False),
        settings=dict(total_frames=total_frames, fps=float(fps), scene_point_count=scene_point_count, minimum_hold_frames=minimum_hold_frames),
        condition_policy=dict(ordered_full_body_keyposes=1, keypose_source="exact_admitted_stage2_not_route_nodes",
            keypose_internal_frame_numbers_supplied=False, event_order="one_travel_per_original_route_segment_then_establish_then_hold",
            reference_or_future_motion_read=False, text="explicit_zero_unknown_encoder_NOT_text_conditioning",
            semantic_id="zero_placeholder_NOT_trained_sit_class", original_instruction=current["stage1"]["instruction"],
            original_scene_id=current["stage1"]["scene_id"], route_channels=ROUTE_CHANNELS,
            route_is_summary_not_hard_polyline_constraint=True, route_modified=False,
            scene_channels=["boundary_normal_x", "boundary_normal_y", "boundary_normal_z", "target_furniture"],
            scene_points="actual_current_hash_bound_occupancy_boundary_faces_plus_world_ground",
            sdf="target_union_original_non_target_plus_floor; XYZ_to_ZYX_and_sample_centre_bounds",
            ground_z_m=0., region_target_ids={"floor":0, "current_target_furniture":1},
            contact_requirements="gluteal_and_bilateral_foot_support_only_during_hold; no_invented_hand_back_relations",
            contact_normal_source="gluteal_selected_real_surface_normal_normalized; bilateral_feet_ground_plus_Z",
            keypose_permissions="root_and_all_22_rotations_locked; upstream_static_refine_permissions_not_reinterpreted",
            body="exact_upstream_fixed_actual_betas; no_zero_shape_projection; nonflat_hand_mean_retained",
            initial_history_body="same_actual_betas_and_hands_as_Stage2; static_grounding_recomputed_from_actual_vertices",
            training_zero_shape_distribution_shift=bool(np.any(arrays['betas'] != 0))),
        physical_or_semantic_motion_quality_verified=False, scene_condition_training_claimed=False,
        trained_model_loaded=False, motion_generated=False, motion_release_authorized=False,
        elapsed_seconds=time.monotonic()-started))
    return PreparedUpstream(condition, body, scene, receipt, route, arrays["vertices_world_zup"],
                            arrays["joints_world_zup"], verified(records["original_scene_mesh"]))


def load_prepared(receipt_path, *, device="cpu"):
    """Re-admit original success evidence and exact body instead of trusting labels."""
    from hsi.stage2.sdf import BoundSDFCache
    from hsi.stage2.refine_contract import GRID_LOWER_XYZ, GRID_UPPER_XYZ
    from hsi.stage3.packet import route_contact_check
    receipt = read_sealed(receipt_path)
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "verified_upstream_condition_not_motion_release", "Wrong adapter receipt")
    require(receipt["source"] == artifact(__file__) and receipt["runtime_sources"] == runtime_sources(), "Adapter/runtime drift")
    require(receipt['admission_source'] == admission_source(), 'Admission implementation drift')
    current, assets = _admit(verified(receipt["inputs"]["stage2"]), verified(receipt["body_assets"]["candidate_body"]))
    require(current["records"] == receipt["inputs"] and assets == receipt["body_assets"], "Upstream lineage drift")
    body, body_receipt = mesh_body_from_current(current, assets["candidate_body"]["path"])
    require(body_receipt == receipt["body"], "Body identity/group/hand drift")
    condition = load_condition(verified(receipt["condition"]))
    arrays = current["arrays"]
    expected = pose_to_motion(arrays["root_xyz_yaw"][:3], arrays["body_pose_axis_angle"],
                             torch.as_tensor(arrays["pose_frame_to_world_rotation"]) @ Q_YUP_TO_ZUP)
    require(torch.equal(condition.keyposes[0, 0], expected), "Condition replaced original Stage2 keypose")
    roundtrip(body, expected, arrays["vertices_world_zup"], arrays["joints_world_zup"], "reload_Stage2")
    with np.load(verified(receipt["geometry"]), allow_pickle=False) as z:
        route = z["route_xy"].copy()
        require(np.array_equal(route, current["keys"]["collision_safe_control_polyline"]["nodes_world_xy_m"]), "Route drift")
    seed_receipt = read_sealed(verified(receipt['static_seed_receipt']))
    require(seed_receipt['schema'] == SCHEMA+'.same_shape_static_history' and seed_receipt['source'] == artifact(__file__),
            'Wrong same-shape static seed or changed implementation')
    for name, key in [('source_stage1','stage1_execution'), ('source_keypose','candidate'), ('source_body','body_model')]:
        require(seed_receipt[name] == current['records'][key], 'Same-shape static seed belongs to different upstream: '+name)
    regenerated = shaped_static_history(body, current['source_root'][:2], float(current['source_root'][3]))
    with np.load(verified(seed_receipt['seed']), allow_pickle=False) as z:
        require(set(z.files) == set(regenerated), 'Same-shape history array keys changed')
        for name, expected_array in regenerated.items():
            require(np.array_equal(z[name], expected_array), 'Same-shape static history does not reproduce: '+name)
        history_pose = torch.from_numpy(z['motion'].copy())
        roundtrip(body, history_pose, z['vertices'], z['joints'], 'reload_same_shape_history')
    same_shape_route = route_contact_check(route[-1], float(history_pose[2]), arrays, current['fields'].non_target.sample)
    require(same_shape_route == receipt['same_shape_route_contact_validation'], 'Actual shape route transition admission changed')
    points, features = surface_points(current, receipt["settings"]["scene_point_count"])
    rebuilt = build_condition(history=history_pose.repeat(2, 1), keypose=expected, body=body, route_xy=route,
        terminal_yaw=float(arrays["root_xyz_yaw"][3]), scene_points=points, scene_features=features,
        contact_goal=arrays["contact_goal_world_xyz_zup_m"], triangles=current["triangles"],
        surface_normal=current['surface']['surface']['normal_world_zup'],
        total_frames=receipt["settings"]["total_frames"], fps=receipt["settings"]["fps"],
        minimum_hold_frames=receipt["settings"]["minimum_hold_frames"])
    for field in fields(condition):
        saved, fresh = getattr(condition, field.name), getattr(rebuilt, field.name)
        require(saved == fresh if field.name == "fps" else torch.equal(saved, fresh),
                "Condition field does not reproduce from verified upstream: "+field.name)
    normal_binding = receipt['contact_normal_binding']
    require(normal_binding['source_target'] == current['records']['stage1_target']
            and normal_binding['selected_surface_id'] == current['target']['selected_surface_id']
            and normal_binding['selected_surface_sha256'] == current['target']['selected_surface_sha256']
            and normal_binding['source_surface_normal_world_zup'] == current['surface']['surface']['normal_world_zup']
            and normal_binding['normalized_gluteal_normal_world_zup'] == rebuilt.contact_normals[0, 0, 0].tolist()
            and normal_binding['foot_normal_world_zup'] == [0., 0., 1.],
            'Selected real contact normal receipt drift')
    cache = BoundSDFCache(verified(current["records"]["sdf_cache"]), verified(current["records"]["stage1_execution"]))
    scene = RegisteredScene(cache.arrays["target_sdf"], cache.arrays["collision_sdf"], [GRID_LOWER_XYZ, GRID_UPPER_XYZ])
    return PreparedUpstream(condition.to(device), body.to(device), scene.to(device), receipt, route,
        arrays["vertices_world_zup"], arrays["joints_world_zup"], verified(current["records"]["original_scene_mesh"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--body-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=20.)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    require(1 <= args.threads <= 8, "Use 1..8 CPU threads")
    torch.set_num_threads(args.threads)
    result = prepare(args.stage2, args.output, body_asset=args.body_asset, total_frames=args.frames, fps=args.fps)
    print(result.receipt["status"], args.output/"receipt.json", flush=True)


if __name__ == "__main__":
    main()
