"""Retained numerical regressions against integrated modules."""

from __future__ import annotations



from pathlib import Path

import numpy as np

import torch

from hsi.stage3.sdf import QueryTimeICGF, RawSceneSDFZUp  # noqa: E402

from hsi.stage3.conditioning import (  # noqa: E402
    ReMoGenRawSceneCondFn,
    SceneGuidanceConfig,
    ddpm_sample_loop_with_scene_grad,
    transform_future_joints_to_world_zup,
)

LOWER = (-1.0, -1.0, 0.0)

UPPER = (1.0, 1.0, 2.0)

class _FakeVAE(torch.nn.Module):
    """Map the first three latent values to every decoded joint."""

    def decode(self, latent, history_motion, *, nfuture, scale_latent):
        del history_motion, scale_latent
        root = latent[0, :, :3]
        joints = root[:, None, None].expand(-1, int(nfuture), 22, -1)
        return joints.reshape(root.shape[0], int(nfuture), 66)

class _FakeUtility:
    def tensor_to_dict(self, motion):
        return {"joints": motion}

class _FakeDataset:
    primitive_utility = _FakeUtility()

    def denormalize(self, motion):
        return motion

def _identity(batch=1):
    return torch.eye(3).unsqueeze(0).expand(batch, -1, -1).clone()

def _zero_translation(batch=1):
    return torch.zeros(batch, 1, 3)

def _free_scene() -> RawSceneSDFZUp:
    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[1:3, 1:3, 8:12] = True
    return RawSceneSDFZUp.from_occupancy(
        occupancy,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
    )

def _condition(
    *,
    scene,
    icgf=None,
    contact_joint_ids=(),
    collision_joint_ids=(),
    config,
):
    return ReMoGenRawSceneCondFn(
        vae_model=_FakeVAE(),
        dataset=_FakeDataset(),
        history_motion=torch.zeros(1, 2, 1),
        scene_sdf=scene,
        current_rotmat=_identity(),
        current_transl=_zero_translation(),
        global_rotmat=_identity(),
        global_transl=_zero_translation(),
        future_length=2,
        valid_future_frames=2,
        scale_latent=True,
        num_diffusion_steps=10,
        icgf=icgf,
        contact_joint_ids=contact_joint_ids,
        collision_joint_ids=collision_joint_ids,
        config=config,
    )

def test_pred_xstart_decode_world_and_icgf_negative_energy_gradient() -> None:
    target = QueryTimeICGF.from_target_point((0.4, 0.0, 1.0))
    condition = _condition(
        scene=_free_scene(),
        icgf=target,
        contact_joint_ids=(0,),
        collision_joint_ids=(),
        config=SceneGuidanceConfig(
            sdf_weight=0.0,
            icgf_weight=1.0,
            icgf_deadzone_m=0.0,
            contact_frames=2,
            guidance_scale=1.0,
            max_grad_norm=10.0,
            denoise_start_fraction=0.0,
            pcgrad_goal=False,
        ),
    )
    x = torch.tensor([[[0.0, 0.0, 1.0]]], requires_grad=True)
    gradient = condition(
        x,
        torch.tensor([0], dtype=torch.long),
        {"pred_xstart": x * 1.0},
    )
    assert gradient.shape == x.shape
    assert torch.isfinite(gradient).all()
    # Returned value is -dE/dx and therefore moves the decoded joint toward +X.
    assert float(gradient[0, 0, 0]) > 0.0
    assert abs(float(gradient[0, 0, 1])) < 1.0e-6
    assert condition.last_diagnostics["icgf_energy"] > 0.0
    assert condition.last_diagnostics["returned_grad_norm"] > 0.0

def test_xt_to_clean_gradient_ratio_has_documented_order() -> None:
    target = QueryTimeICGF.from_target_point((0.4, 0.0, 1.0))
    condition = _condition(
        scene=_free_scene(),
        icgf=target,
        contact_joint_ids=(0,),
        collision_joint_ids=(),
        config=SceneGuidanceConfig(
            sdf_weight=0.0,
            icgf_weight=1.0,
            icgf_deadzone_m=0.0,
            contact_frames=2,
            guidance_scale=1.0,
            denoise_start_fraction=0.0,
            pcgrad_goal=False,
        ),
    )
    x = torch.tensor([[[0.0, 0.0, 0.5]]], requires_grad=True)
    condition(
        x,
        torch.tensor([0], dtype=torch.long),
        {"pred_xstart": x * 2.0},
    )
    # dE/dx_t crosses a factor-two START_X Jacobian, whereas
    # dE/d(pred_xstart) does not.  The logged ratio is noisy/clean, not its
    # reciprocal.
    assert abs(condition.last_diagnostics["xt_to_clean_grad_norm_ratio"] - 2.0) < 1.0e-5

def test_raw_sdf_gradient_moves_joint_out_of_occupied_slab() -> None:
    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[10:13, :, :] = True
    scene = RawSceneSDFZUp.from_occupancy(
        occupancy,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
    )
    condition = _condition(
        scene=scene,
        collision_joint_ids=(0,),
        config=SceneGuidanceConfig(
            sdf_weight=1.0,
            icgf_weight=0.0,
            collision_clearance_m=0.01,
            guidance_scale=1.0,
            max_grad_norm=10.0,
            denoise_start_fraction=0.0,
            pcgrad_goal=False,
        ),
    )
    # X=0.05 is the first occupied cell centre, closest to the slab's -X face.
    x = torch.tensor([[[0.05, 0.0, 1.0]]], requires_grad=True)
    gradient = condition(
        x,
        torch.tensor([0], dtype=torch.long),
        {"pred_xstart": x * 1.0},
    )
    assert float(gradient[0, 0, 0]) < 0.0
    assert condition.last_diagnostics["minimum_sdf_m"] < 0.0
    assert condition.last_diagnostics["penetration_fraction"] == 1.0

def test_m4_contact_tail_exempts_only_declared_joint_from_full_sdf() -> None:
    """The target stays solid for all non-contact joints and earlier frames."""

    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[9:12, :, :] = True
    scene = RawSceneSDFZUp.from_occupancy(
        occupancy,
        lower_xyz=LOWER,
        upper_xyz=UPPER,
        floor_ignore_height_m=0.0,
        remove_target_component=False,
    )
    condition = _condition(
        scene=scene,
        icgf=QueryTimeICGF.from_target_point((0.0, 0.0, 1.0)),
        contact_joint_ids=(0,),
        collision_joint_ids=(0, 1),
        config=SceneGuidanceConfig(
            sdf_weight=1.0,
            icgf_weight=0.25,
            collision_clearance_m=0.01,
            contact_frames=1,
            exempt_contact_joints_from_sdf_tail=True,
            icgf_deadzone_m=0.0,
            guidance_scale=1.0,
            denoise_start_fraction=0.0,
            pcgrad_goal=False,
        ),
    )
    joints = torch.zeros(1, 2, 22, 3)
    joints[..., 2] = 1.0
    total, values = condition.energy_from_world_joints(joints)
    # 2 frames x 2 collision joints minus only joint-0 in the last frame.
    assert float(values["sdf_constraint_count"]) == 3.0
    assert float(values["sdf_active_constraint_count"]) == 3.0
    # ICGF applies only to the same declared joint in the one-frame tail.
    assert float(values["icgf_constraint_count"]) == 1.0
    assert float(values["minimum_sdf_m"]) < 0.0
    assert float(total) > 0.0
    assert scene.receipt["target_component_removed"] is False

def test_late_schedule_and_composed_world_transform_contract() -> None:
    condition = _condition(
        scene=_free_scene(),
        icgf=QueryTimeICGF.from_target_point((0.4, 0.0, 1.0)),
        contact_joint_ids=(0,),
        config=SceneGuidanceConfig(
            sdf_weight=0.0,
            icgf_weight=1.0,
            icgf_deadzone_m=0.0,
            denoise_start_fraction=0.5,
            pcgrad_goal=False,
        ),
    )
    x = torch.tensor([[[0.0, 0.0, 1.0]]], requires_grad=True)
    early = condition(
        x,
        torch.tensor([9], dtype=torch.long),
        {"pred_xstart": x * 1.0},
    )
    assert torch.equal(early, torch.zeros_like(early))

    local = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    yaw_90 = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    world = transform_future_joints_to_world_zup(
        local,
        current_rotmat=yaw_90,
        current_transl=torch.tensor([[[1.0, 0.0, 0.0]]]),
        global_rotmat=_identity(),
        global_transl=torch.tensor([[[0.0, 2.0, 0.0]]]),
    )
    assert torch.allclose(world[0, 0, 0], torch.tensor([1.0, 3.0, 0.0]))

class _OneStepDiffusion:
    num_timesteps = 1
    posterior_variance = np.asarray([0.0], dtype=np.float64)
    posterior_mean_coef1 = np.asarray([1.0], dtype=np.float64)

    def p_mean_variance(self, model, x, t, *, clip_denoised, model_kwargs):
        del model, t, clip_denoised, model_kwargs
        return {
            "mean": x,
            "variance": torch.zeros_like(x),
            "log_variance": torch.zeros_like(x),
            "pred_xstart": x,
        }

    def q_posterior_mean_variance(self, x_start, x_t, t):
        del x_t, t
        zero = torch.zeros_like(x_start)
        return x_start, zero, zero

def test_clean_latent_update_survives_zero_final_posterior_variance() -> None:
    target = QueryTimeICGF.from_target_point((0.4, 0.0, 1.0))
    condition = _condition(
        scene=_free_scene(),
        icgf=target,
        contact_joint_ids=(0,),
        collision_joint_ids=(),
        config=SceneGuidanceConfig(
            sdf_weight=0.0,
            icgf_weight=1.0,
            icgf_deadzone_m=0.0,
            contact_frames=2,
            guidance_scale=1.0,
            max_grad_norm=1.0,
            denoise_start_fraction=0.0,
            pcgrad_goal=False,
            guidance_mode="xstart_posterior",
            normalize_scene_gradient=True,
            clean_latent_step_size=0.1,
            max_clean_latent_update_norm=0.1,
        ),
    )
    noise = torch.tensor([[[0.0, 0.0, 1.0]]])
    sampled = ddpm_sample_loop_with_scene_grad(
        _OneStepDiffusion(),
        torch.nn.Linear(1, 1),
        noise.shape,
        cond_fn=condition,
        model_kwargs={},
        noise=noise,
    )
    # At t=0 legacy variance*score guidance is identically zero.  The repaired
    # path changes clean x0 and then recomputes q(x_{t-1}|x_t,x0), so the final
    # sample moves toward the target inside denoising.
    assert torch.allclose(sampled[0, 0, 0], torch.tensor(0.1), atol=1.0e-6)
    assert condition.records[-1]["posterior_variance"] == 0.0
    assert condition.records[-1]["posterior_xstart_coefficient"] == 1.0
    assert abs(condition.records[-1]["clean_xstart_shift_norm"] - 0.1) < 1.0e-6
    assert abs(condition.records[-1]["actual_mean_shift_norm"] - 0.1) < 1.0e-6
    assert abs(condition.records[-1]["expected_mean_shift_norm"] - 0.1) < 1.0e-6
    assert (
        condition.records[-1]["conditioning_strategy"]
        == "xstart_step_then_exact_ddpm_posterior"
    )

