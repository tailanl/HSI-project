"""Kinematic realization of native joints using the *same* upstream mesh body.

No reference motion, timestamp labels, IK checkpoint or render-time smoothing
is used. Native per-frame bones initialize SO(3); complete-sequence optimization
matches native joint positions and changes. Optional geometry energy receives
the actual differentiable full mesh. A good fit is not a dynamics certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral
from pathlib import Path
import sys
import time

import torch
from torch import Tensor

HSI_ROOT = Path(__file__).resolve().parents[3] / "HSI-project"
if str(HSI_ROOT) not in sys.path:
    sys.path.insert(0, str(HSI_ROOT))
from hsi.stage3_sequence.geometry import rotation_6d_to_matrix
from hsi.stage3_sequence.constraints import matrix_to_rotation_6d
from hsi.stage3_sequence.mesh_body import MeshOutput

SCHEMA = "agent9.closd_unihsi_sdf.mesh_realization.v1"
SMPLX_PARENTS22 = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RealizationConfig:
    steps: int = 80
    lr: float = .01
    fps: float = 30.
    position_weight: float = 1.
    velocity_weight: float = .02
    acceleration_weight: float = .0002
    source_pose_weight: float = .005
    scene_weight: float = 1.
    gradient_clip: float = 10.
    trace_every: int = 10

    def validate(self):
        require(type(self.steps) is int and 0 <= self.steps <= 120, "Use 0..120 realization steps")
        require(type(self.trace_every) is int and self.trace_every >= 1, "Positive trace interval required")
        for name in ("lr", "fps", "gradient_clip"):
            v = getattr(self, name)
            require(isinstance(v, (float, int)) and math.isfinite(v) and v > 0, "Invalid positive config: " + name)
        require(self.fps <= 240, "Invalid FPS")
        for name in ("position_weight", "velocity_weight", "acceleration_weight", "source_pose_weight", "scene_weight"):
            v = getattr(self, name)
            require(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0, "Invalid energy weight: " + name)
        require(self.position_weight > 0, "Native joint tracking may not be disabled")
        return self


def _parents(body):
    value = getattr(getattr(body, "model", None), "parents", SMPLX_PARENTS22)
    parents = tuple(int(x) for x in torch.as_tensor(value).tolist()[:22])
    require(len(parents) == 22 and parents[0] == -1 and all(0 <= parents[j] < j for j in range(1, 22)),
        "Body must use ordered SMPL-X22 parents")
    require(parents == SMPLX_PARENTS22, "Unregistered joint ordering; retargeting must be explicit")
    return parents


def _align_one(source, destination):
    """Minimum SO(3) rotation; finite identity/antiparallel branches."""
    eye = torch.eye(3, device=source.device, dtype=source.dtype)
    sn, dn = source.norm(), destination.norm()
    if float(sn) < 1e-8 or float(dn) < 1e-8:
        return eye
    a, b = source / sn, destination / dn
    cosine = torch.dot(a, b).clamp(-1, 1)
    if float(cosine) < -.9999:
        coordinate = eye[int(a.abs().argmin())]
        axis = torch.linalg.cross(a, coordinate)
        axis = axis / axis.norm().clamp_min(1e-8)
        return 2 * axis[:, None] @ axis[None] - eye
    v = torch.linalg.cross(a, b)
    x, y, z = v.unbind()
    zero = v.new_zeros(())
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return eye + skew + skew @ skew / (1 + cosine).clamp_min(1e-8)


def initialize_motion(world_joints: Tensor, initial_pose: Tensor, body) -> Tensor:
    """Use no later static target pose; leaf twist stays at initial articulation."""
    require(world_joints.ndim == 3 and world_joints.shape[1:] == (22, 3), "Expected native [T,22,3]")
    parents = _parents(body)
    with torch.no_grad():
        initial_joints = body.mesh(initial_pose[None, None]).joints[0, 0]
        initial_local = rotation_6d_to_matrix(initial_pose[3:].reshape(22, 6))
        initial_global = []
        for j, parent in enumerate(parents):
            initial_global.append(initial_local[j] if j == 0 else initial_global[parent] @ initial_local[j])
        children = [[j for j, p in enumerate(parents) if p == i] for i in range(22)]
        result = initial_pose[None].repeat(len(world_joints), 1)
        result[:, :3] = world_joints[:, 0]
        for frame, joints in enumerate(world_joints):
            global_rotations, local_rotations = [], []
            for j, kids in enumerate(children):
                if kids:
                    source = torch.stack([initial_joints[k] - initial_joints[j] for k in kids])
                    target = torch.stack([joints[k] - joints[j] for k in kids])
                    valid = (source.norm(dim=-1) > 1e-8) & (target.norm(dim=-1) > 1e-8)
                    source, target = source[valid], target[valid]
                    if len(source) >= 2:
                        a = source / source.norm(dim=-1, keepdim=True)
                        b = target / target.norm(dim=-1, keepdim=True)
                        u, _, vh = torch.linalg.svd(a.T @ b)
                        correction = torch.eye(3, device=joints.device, dtype=joints.dtype)
                        correction[2, 2] = torch.linalg.det(vh.T @ u.T)
                        delta = vh.T @ correction @ u.T
                    elif len(source) == 1:
                        delta = _align_one(source[0], target[0])
                    else:
                        delta = torch.eye(3, device=joints.device, dtype=joints.dtype)
                    global_rotation = delta @ initial_global[j]
                else:
                    global_rotation = global_rotations[parents[j]] @ initial_local[j]
                local = global_rotation if j == 0 else global_rotations[parents[j]].T @ global_rotation
                global_rotations.append(global_rotation)
                local_rotations.append(local)
            result[frame, 3:] = matrix_to_rotation_6d(torch.stack(local_rotations)).flatten()
        result[0] = initial_pose
    return result


def _anchors(anchor_frames, length):
    if anchor_frames is None:
        return []
    if isinstance(anchor_frames, Tensor):
        require(anchor_frames.dtype in (torch.int32, torch.int64) and anchor_frames.ndim == 1, "Integer 1D anchor frames required")
        anchor_frames = anchor_frames.detach().cpu().tolist()
    values = list(anchor_frames)
    require(all(isinstance(x, Integral) and not isinstance(x, bool) for x in values), "Integer anchor frames required")
    values = [int(x) for x in values]
    require(values == sorted(set(values)) and all(1 <= x < length for x in values),
        "Anchors must be unique ordered 0-based frames after the locked first frame")
    return values


def _differences(value, order, fps):
    for _ in range(order):
        value = value[1:] - value[:-1]
    return value * fps ** order


def _mean_square(value):
    return value.square().mean() if value.numel() else value.sum() * 0


def _energy(motion, mesh, native, source, config, scene_energy):
    predicted = mesh.joints[0]
    terms = dict(position=_mean_square(predicted - native),
        velocity=_mean_square(_differences(predicted, 1, config.fps) - _differences(native, 1, config.fps)),
        acceleration=_mean_square(_differences(predicted, 2, config.fps) - _differences(native, 2, config.fps)),
        source_pose=_mean_square(rotation_6d_to_matrix(motion[:, 3:].reshape(-1,22,6)) -
            rotation_6d_to_matrix(source[:, 3:].reshape(-1,22,6))) + _mean_square(motion[:, :3] - source[:, :3]))
    scene = motion.sum() * 0
    if scene_energy is not None:
        supplied = scene_energy(mesh, motion[None])
        scene = supplied["loss"] if isinstance(supplied, dict) else supplied
        require(isinstance(scene, Tensor) and scene.numel() == 1 and scene.device == motion.device
            and bool(torch.isfinite(scene).all()), "Scene callback must return one finite differentiable mesh energy tensor")
        scene = scene.reshape(())
    terms["scene"] = scene
    total = sum(terms[name] * getattr(config, name + "_weight") for name in terms)
    require(bool(torch.isfinite(total)), "Nonfinite realization objective")
    return total, terms


def _diagnostics(joints, native, fps, anchors):
    error = torch.linalg.vector_norm(joints - native, dim=-1)
    transitions = sorted({i for frame in [0, *anchors] for i in (frame-1, frame) if 0 <= i < len(joints)-1})
    root_delta = (joints[1:,0] - joints[:-1,0]).norm(dim=-1)
    joint_delta = (joints[1:] - joints[:-1]).norm(dim=-1)
    return {"raw_native_joint_fit_error_m": float(error.mean()), "maximum_joint_fit_error_m": float(error.max()),
        "root_trajectory_length_m": float(root_delta.sum()), "maximum_root_step_m": float(root_delta.max()),
        "native_root_trajectory_length_m": float((native[1:,0]-native[:-1,0]).norm(dim=-1).sum()),
        "velocity_rmse_m_s": float(_mean_square(_differences(joints,1,fps)-_differences(native,1,fps)).sqrt()),
        "acceleration_rmse_m_s2": float(_mean_square(_differences(joints,2,fps)-_differences(native,2,fps)).sqrt()),
        "locked_frame_adjacent_transitions": [{"from_frame_0based": i, "to_frame_0based": i+1,
            "root_step_m": float(root_delta[i]), "root_speed_m_s": float(root_delta[i]*fps),
            "mean_joint_step_m": float(joint_delta[i].mean()), "max_joint_step_m": float(joint_delta[i].max()),
            "max_joint_speed_m_s": float(joint_delta[i].max()*fps)} for i in transitions]}


def realize_motion(world_joints, case, config=None, scene_energy=None, anchor_frames=None):
    """Return (motion[T,135], full MeshOutput[1,T,...], diagnostic trace).

    ``anchor_frames`` are caller-arranged algorithmic event frames, 0-based,
    not reference/GT labels. All such frames lock the entire target pose.
    ``scene_energy(mesh, motion[1,T,135])`` may return a scalar tensor or a
    mapping containing ``loss``. It must use this mesh, not another body.
    The best finite configured objective is returned without any post-smoothing.
    """
    started = time.monotonic()
    config = (config or RealizationConfig()).validate()
    initial = case.initial_pose.detach()
    target_pose = case.target_pose.detach().to(initial)
    require(initial.shape == target_pose.shape == (135,) and initial.is_floating_point()
        and bool(torch.isfinite(initial).all()) and bool(torch.isfinite(target_pose).all()), "Invalid fixed poses")
    native = torch.as_tensor(world_joints, device=initial.device, dtype=initial.dtype).detach().clone()
    require(native.ndim == 3 and native.shape[1:] == (22,3) and 2 <= len(native) <= 1200
        and bool(torch.isfinite(native).all()), "Finite native world joints [2..1200,22,3] required")
    anchors = _anchors(anchor_frames, len(native))
    body = case.body
    fixed_state = {name: value.detach().clone() for name, value in (
        ("betas", body.betas), ("left_hand_pose", body.left_hand_pose), ("right_hand_pose", body.right_hand_pose))}
    require(torch.equal(body.betas.reshape(-1).to(case.betas), case.betas.reshape(-1)), "Case/body shape mismatch")
    source = initialize_motion(native, initial, body)
    source[0] = initial
    lock_mask = torch.zeros(len(native), device=initial.device, dtype=torch.bool)
    lock_mask[0] = True
    fixed = source.clone()
    for frame in anchors:
        lock_mask[frame] = True
        fixed[frame] = target_pose
    parameters = torch.nn.Parameter(source.clone())
    optimizer = torch.optim.Adam([parameters], lr=config.lr)
    best_loss, best_motion, best_step = math.inf, None, None
    entries, initialization = [], None
    for step in range(config.steps + 1):
        motion = torch.where(lock_mask[:,None], fixed, parameters)
        mesh = body.mesh(motion[None])
        require(mesh.joints.shape == (1,len(native),22,3) and mesh.vertices.ndim == 4
            and mesh.vertices.shape[:2] == (1,len(native)) and mesh.vertices.shape[-1] == 3
            and bool(torch.isfinite(mesh.joints).all()) and bool(torch.isfinite(mesh.vertices).all()), "Invalid realized mesh")
        total, terms = _energy(motion, mesh, native, source, config, scene_energy)
        value = float(total.detach())
        if value < best_loss:
            best_loss, best_motion, best_step = value, motion.detach().clone(), step
        if step == 0:
            initialization = _diagnostics(mesh.joints[0].detach(), native, config.fps, anchors)
        if step % config.trace_every == 0 or step == config.steps:
            entries.append({"step": step, "objective": value, **{k:float(v.detach()) for k,v in terms.items()}})
        if step == config.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        require(parameters.grad is not None and bool(torch.isfinite(parameters.grad).all()), "Nonfinite/missing fitting gradient")
        torch.nn.utils.clip_grad_norm_([parameters], config.gradient_clip)
        optimizer.step()
    require(torch.equal(best_motion[0], initial) and all(torch.equal(best_motion[i], target_pose) for i in anchors),
        "Locked full pose changed")
    for name, value in fixed_state.items():
        require(torch.equal(getattr(body, name), value), "Fixed body shape/hands changed during realization")
    with torch.no_grad():
        final_mesh = body.mesh(best_motion[None])
    final_mesh = MeshOutput(final_mesh.vertices.detach(), final_mesh.joints.detach())
    require(bool(torch.isfinite(final_mesh.vertices).all()) and bool(torch.isfinite(final_mesh.joints).all()), "Nonfinite final mesh")
    final_fit = _diagnostics(final_mesh.joints[0], native, config.fps, anchors)
    trace = {"schema": SCHEMA, "status": "kinematic_fit_diagnostic_not_dynamics_or_collision_certification",
        "configuration": asdict(config),
        "initialization": "native_root_and_current_initial_body_bone_vector_SO3; no_future_target_initialization",
        "world_coordinate_system": "world_zup_m", "fps": config.fps, "frames": len(native),
        "vertex_count": final_mesh.vertices.shape[-2], "joint_count": 22,
        "raw_native_joint_fit_error_m": final_fit["raw_native_joint_fit_error_m"],
        "raw_native_joint_fit_error_definition": "mean Euclidean distance in metres to raw native joints, not a quality gate",
        "initial_fit": initialization, "final_fit": final_fit,
        "optimization_trace": entries, "optimizer_steps_executed": config.steps, "selected_step": best_step,
        "selected_objective": best_loss, "objective_decreased": best_loss < entries[0]["objective"],
        "anchor_frames_0based": anchors, "anchor_frame_source": "caller_algorithmic_event_schedule_not_GT",
        "first_frame_exactly_locked": True, "anchor_full_pose_exactly_locked": bool(anchors),
        "fixed_betas_preserved": True, "fixed_hands_preserved": True,
        "native_data_used": "world_joints_only_no_reference_motion_or_timestamps",
        "scene_energy_used": scene_energy is not None, "scene_callback_receives_actual_complete_mesh": True,
        "render_time_root_edit_smoothing_or_resampling": False,
        "dynamics_verified": False, "collision_verified": False, "foot_slip_verified": False,
        "motion_quality_passed": False, "elapsed_seconds": time.monotonic()-started}
    return best_motion, final_mesh, trace
