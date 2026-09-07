"""Experimental current-geometry sit proposal, NOT an H3/HybrIK result.

Zero learned memory is required. Articulation is a declared kinematic seed,
optimized against current contact/support and the retained full-mesh SDF.
Numerical acceptance never grants Qwen, Stage3, or successful-experience credit.
"""
from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from memory_common import PROJECT, artifact, verified, read_sealed, read_json, write_once, require

METHODS = PROJECT / "agent9/methods"
P550 = METHODS / "p550_scene_understanding_decoupled_20260905/code"
P552 = METHODS / "p552_contact_region_route_20260906/code"
P549 = METHODS / "p549_fast_image_stage2_20260905/code"
P523 = METHODS / "p523_current_only_multiscene_20260902/stage2/code"
P498 = METHODS / "p498_semantic_surface_keypose_fix_20260830/code"
BODY = PROJECT / "agent10/hybrikx_three_methods_20260903/HybrIK/model_files/smplx/SMPLX_NEUTRAL.npz"
SEGMENTATION = PROJECT / "agent6/official/DPoser-X/lib/data/smplx_vert_segmentation.json"

# Exact current P552/P550 campaign numerical thresholds, not the older P498 defaults.
THRESHOLDS = {
    "maximum_ground_penetration_m": .02,
    "maximum_non_target_scene_penetration_m": .02,
    "maximum_sdf_outside_fraction": .01,
    "maximum_target_allowed_contact_penetration_m": .03,
    "maximum_target_forbidden_body_penetration_m": .02,
    "maximum_terminal_facing_error_rad": math.pi / 36,
    "minimum_hips_support_inside_surface_fraction": .8,
    "minimum_left_foot_ground_vertices": 6,
    "minimum_right_foot_ground_vertices": 6,
    "minimum_target_allowed_contact_vertices": 16,
}


def activate():
    for path in (P498, P523, P549, P550, P552):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def triangle_heights(points_xy, triangles):
    """Exact triangle membership; holes remain unknown, no AABB filling."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3) and len(tris) > 0, "Invalid support triangles")
    require(np.isfinite(tris).all() and np.isfinite(points).all(), "Non-finite support query")
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    u, v = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    good = np.abs(det) > 1e-12
    safe = np.where(good, det, 1.)
    heights = np.zeros(len(points), dtype=np.float64)
    valid = np.zeros(len(points), dtype=bool)
    for first in range(0, len(points), 64):
        d = points[first:first + 64, None, :] - a[None, :, :2]
        beta = (d[..., 0] * v[:, 1] - d[..., 1] * v[:, 0]) / safe
        gamma = (u[:, 0] * d[..., 1] - u[:, 1] * d[..., 0]) / safe
        inside = good & (beta >= -1e-8) & (gamma >= -1e-8) & (beta + gamma <= 1 + 1e-8)
        z = a[:, 2] + beta * (b[:, 2] - a[:, 2]) + gamma * (c[:, 2] - a[:, 2])
        found = inside.any(axis=1)
        top = np.max(np.where(inside, z, -np.inf), axis=1)
        heights[first:first + len(found)] = np.where(found, top, 0.)
        valid[first:first + len(found)] = found
    return heights, valid


def sit_seed():
    """Declared analytic SMPL-X angles, no dataset/old pose/articulation input."""
    pose = np.zeros((21, 3), dtype=np.float32)
    pose[[0, 1], 0] = -math.pi / 2
    pose[[3, 4], 0] = math.pi / 2
    pose[15, 2], pose[16, 2] = -1.25, 1.25
    pose[[17, 18], 0] = -.15
    return pose


def contact_candidates(anchor, centre, forward, triangles, policy):
    require(policy in {"surface_centre", "front_support"}, "Unknown contact policy")
    anchor, centre, forward = map(lambda x: np.asarray(x, dtype=np.float64), (anchor, centre, forward))
    require(anchor.shape == centre.shape == (3,) and forward.shape == (2,), "Malformed contact axes")
    require(abs(np.linalg.norm(forward) - 1) < 1e-6, "Forward must be a unit vector")
    if policy == "surface_centre":
        queries = [centre[:2]]
    else:
        # Metric, body-scale, current surface hypotheses. Never scale a stored pose.
        left = np.array([-forward[1], forward[0]])
        queries = [anchor[:2] - distance * forward + lateral * left
                   for lateral in (0., -.10, .10, -.20, .20, -.30, .30)
                   for distance in (.12, .16, .08, .20)]
    heights, valid = triangle_heights(queries, triangles)
    return [np.r_[xy, z] for xy, z, ok in zip(queries, heights, valid) if ok]


def load_current(stage1_path):
    activate()
    from region_pipeline import check_stage1_handoff
    from bind_route import bind
    stage1 = read_sealed(stage1_path)
    guard = check_stage1_handoff(stage1_path)
    binding = bind(stage1)
    target = read_sealed(verified(stage1["target"]))
    require(target["action_family"] == "sit", "Fast provider currently supports sit only")
    surface = next(row for row in target["candidate_surfaces"] if row["candidate_id"] == binding["selected_surface_id"])
    option = next(row for row in surface["approach_options"] if row["option_id"] == binding["selected_approach_option_id"])
    member = option["p552_region_member"]
    with np.load(verified(member["region_arrays"]), allow_pickle=False) as data:
        triangles = data["contact_triangles"].copy()
    occupancy_record = target["artifacts"]["scene_occupancy"]
    mask_record = target["artifacts"]["target_occupancy_mask"]
    return dict(stage1=stage1, target=target, surface=surface, option=option, binding=binding,
                guard=guard, triangles=triangles,
                occupancy=verified({k: occupancy_record[k] for k in ("path", "bytes", "sha256")}),
                target_mask=verified({k: mask_record[k] for k in ("path", "bytes", "sha256")}))


class FastKeyposeEngine:
    """One CPU session reuses only the fixed body model, never a prior full pose."""
    def __init__(self, model_path=BODY, segmentation_path=SEGMENTATION, threads=4):
        activate()
        import torch
        import retarget_keypose_cpu_v1 as body
        import current_only_contract_v1 as contract
        import current_bound_sdf_v1 as sdf
        import precompute_bound_sdf as cache
        import neutral_sit_quality as quality
        import physical_margin_terms as margin
        self.torch, self.body, self.contract, self.sdf, self.cache = torch, body, contract, sdf, cache
        self.quality, self.margin = quality, margin
        require(type(threads) is int and 1 <= threads <= 8, "Invalid CPU thread count")
        torch.set_num_threads(threads)
        start = time.monotonic()
        self.model_binding = artifact(model_path)
        self.segmentation_binding = artifact(segmentation_path)
        self.model = body.body_util.load_fixed_smplx(Path(model_path), torch.device("cpu"))
        self.segmentation = read_json(segmentation_path)
        self.setup_seconds = time.monotonic() - start

    def propose(self, stage1_path, sdf_path, output, *, steps=100, contact_policy="front_support", use_sdf=True):
        torch, body, contract = self.torch, self.body, self.contract
        start = time.monotonic()
        source_at_start = artifact(__file__)
        source_dependencies = [artifact(module.__file__) for module in
            (body, body.body_util, body.sparse, body.target, contract, self.sdf, self.cache, self.quality, self.margin)]
        require(type(steps) is int and 0 <= steps <= 400, "Invalid optimization budget")
        require(type(use_sdf) is bool, "Invalid SDF switch")
        output = Path(output).resolve()
        require(not output.exists(), "Refuse to overwrite proposal")
        current = load_current(stage1_path)
        timings = {"stage1_revalidation_seconds": time.monotonic() - start}
        target, surface, option = current["target"], current["surface"], current["option"]
        yaw = current["binding"]["contact_forward_yaw_rad"]
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        candidates = contact_candidates(option["p552_region_member"]["contact_anchor_world_xyz_m"],
            surface["surface"]["centre_world_xyz_zup_m"], forward, current["triangles"], contact_policy)
        require(candidates, "No current contact hypothesis is inside the actual support triangles")
        config = body.target.RefineConfig(steps=max(1, steps), bilateral_foot_support=True,
            target_allowed_tail_m=.03, ground_tail_m=.02, minimum_allowed_contact_vertices=16,
            minimum_foot_ground_vertices=6, maximum_target_forbidden_penetration_m=.02,
            maximum_non_target_penetration_m=.02, maximum_outside_fraction=.01).validate()
        before = time.monotonic()
        cache = self.cache.BoundSDFCache(Path(sdf_path), self.sdf)
        require(cache.receipt["source_kind"] == "current_new_scene", "SDF is not current-scene geometry")
        require(cache.receipt["source_binding"]["stage1_execution"] == artifact(stage1_path), "SDF current Stage1 binding mismatch")
        fields = cache(current["occupancy"], current["target_mask"], torch.device("cpu"),
            expected_occupancy_file_sha256=artifact(current["occupancy"])["sha256"],
            expected_target_mask_file_sha256=artifact(current["target_mask"])["sha256"],
            expected_target_world_sha256=target["target_occupancy_mask_world_xyz_sha256"])
        anatomy = body.target.build_anatomy_masks(self.model, 10475, config)
        timings["bound_sdf_load_revalidation_seconds"] = time.monotonic() - before
        pose0 = torch.from_numpy(sit_seed())
        betas = torch.zeros(10)
        frame = torch.tensor(body.rotation_z(yaw + math.pi / 2), dtype=torch.float32)
        with torch.no_grad():
            _, verts = body.body_util.materialize_locked_frame(self.model, pose0, betas, torch.zeros(3), frame)
        subset_np = body.hips_support_subset(verts.numpy(), self.segmentation)
        subset = torch.from_numpy(subset_np)
        bounds = np.array(surface["surface"]["bounds_world_zup_m"])

        def evaluate(state, contact):
            branches = body.target.branch_terms(state.vertices, fields, anatomy, config)
            values = {k: float(v.detach()) if isinstance(v, torch.Tensor) else int(v)
                      for k, v in branches.diagnostics.items()}
            support = state.vertices[subset]
            n = max(8, math.ceil(len(subset_np) * body.SUPPORT_ENVELOPE_FRACTION))
            anchor = torch.cat((support[:, :2].mean(0), torch.topk(support[:, 2], n, largest=False).values.mean().reshape(1)))
            error = float(torch.linalg.vector_norm(anchor - contact).detach())
            values["hips_support_inside_surface_fraction"] = body.support_inside_surface_fraction(support.detach().numpy(), bounds)
            right = (state.joints[2, :2] - state.joints[1, :2]).detach().numpy()
            body_forward = np.array([-right[1], right[0]])
            facing_error = body.angular_error(math.atan2(body_forward[1], body_forward[0]), yaw)
            values["pelvis_frame_facing_error_rad"] = facing_error
            gates = contract.publication_gates(values, THRESHOLDS, facing_error_rad=facing_error,
                selected_surface_match=True, contact_lock_error_m=error)
            z, mask = triangle_heights(support.detach().numpy()[:, :2], current["triangles"])
            # This stricter true-mesh gate is in addition to the original AABB gate.
            values["hips_support_inside_actual_triangles_fraction"] = float(mask.mean())
            contact_height, contact_on_mesh = triangle_heights(contact.detach().numpy()[None, :2], current["triangles"])
            gates.update(current_stage1_target_mask_hash_bound=fields.component_receipt["target_component_sha256"] == target["target_occupancy_mask_world_xyz_sha256"],
                direct_target_relation_valid=surface.get("reference_relation") in (None, "none", "beside", "associated_with_reference_setup"),
                known_action_family_sit=target["action_family"] == "sit",
                contact_point_inside_bound_surface=bool(np.all(contact.detach().numpy() >= bounds[0] - 1e-6) and np.all(contact.detach().numpy() <= bounds[1] + 1e-6)),
                contact_point_on_true_support_mask=bool(contact_on_mesh[0]),
                contact_anchor_clearance_bounded=bool(0 <= float(contact[2]) - contact_height[0] + 1e-6 <= .04 + 1e-6),
                hips_support_inside_actual_triangles=bool(mask.mean() >= .8))
            posture = self.quality.metrics(state.joints[:22].detach().numpy(), yaw)
            gates.update(posture["neutral_sit_quality_gates"])
            values["neutral_sit_quality"] = posture
            rank = (*self.margin.violation_rank(values, gates, THRESHOLDS),
                    float(self.quality.loss(state, yaw).detach()))
            return branches, values, gates, rank

        # Evaluate a few deterministic current-surface contacts before optimizing.
        before = time.monotonic()
        best_seed = None
        with torch.no_grad():
            for candidate in candidates:
                contact = torch.tensor(candidate, dtype=torch.float32)
                state = body.materialize(self.model, pose0, betas, frame, contact, subset)
                _, values, gates, rank = evaluate(state, contact)
                if best_seed is None or rank < best_seed[0]:
                    best_seed = (rank, contact, state, values, gates)
        _, contact, initial, initial_values, initial_gates = best_seed
        timings["contact_hypothesis_seconds"] = time.monotonic() - before
        # Contact and yaw stay fixed. All residuals are bounded around analytic angles.
        limits = torch.full((21, 3), .20)
        limits[[0, 1], 0] = .55
        limits[[3, 4], 0] = .70
        limits[[6, 7], 0] = .80
        limits[[9, 10], 0] = .50
        limits[[15, 16, 17, 18], :] = .45
        raw = torch.nn.Parameter(torch.zeros(21, 3))
        optimizer = torch.optim.Adam([raw], lr=.065)
        history, best = [], None
        before = time.monotonic()
        for step in range(steps + 1):
            pose = pose0 + limits * torch.tanh(raw)
            state = body.materialize(self.model, pose, betas, frame, contact, subset)
            branches, values, gates, rank = evaluate(state, contact)
            if best is None or rank < best[0]:
                best = (rank, pose.detach().clone(), values, gates, step)
            support_loss = self.margin.bilateral_support_loss(state.vertices, anatomy, THRESHOLDS, config.foot_ground_band_m)
            loss = self.quality.loss(state, yaw) + support_loss + 30 * branches.loss_ground_penetration
            loss = loss + .15 * (pose - pose0).square().mean()
            # Keep the feet inside the *voxel-centre* valid SDF domain as well
            # as the unchanged 30-mm support band; sparse CVaR alone misses it.
            lower = state.vertices.new_tensor(contract.GRID_LOWER_XYZ)
            upper = state.vertices.new_tensor(contract.GRID_UPPER_XYZ)
            half = .5 * (upper - lower) / state.vertices.new_tensor(contract.WORLD_OCCUPANCY_SHAPE)
            violation = torch.relu(lower + half + .002 - state.vertices) + torch.relu(state.vertices - upper + half + .002)
            depth = torch.linalg.vector_norm(violation, dim=1)
            loss = loss + 2000 * (depth.square().mean() + torch.topk(depth, max(1, len(depth)//10)).values.square().mean() + depth.max().square())
            if use_sdf:
                # Current target contact has a limited allowance, not whole-object exemption.
                loss = loss + 35 * branches.loss_non_target + 45 * branches.loss_target_forbidden
                loss = loss + 30 * branches.loss_target_allowed_tail + 12 * branches.loss_target_contact_surface
                target_sdf, outside = fields.target.sample(state.vertices)
                other_sdf, other_outside = fields.non_target.sample(state.vertices)
                from gate_aligned_refine import tail_hinge
                allowed = anatomy.allowed_contact.bool()
                loss = loss + 3500 * tail_hinge(target_sdf[allowed & ~outside], .025)
                loss = loss + 1500 * tail_hinge(target_sdf[~allowed & ~outside], .015)
                loss = loss + 1500 * tail_hinge(other_sdf[~other_outside], .015)
            require(bool(torch.isfinite(loss)), "Non-finite geometric IK objective")
            if step % 10 == 0 or step == steps:
                history.append({"step": step, "objective": float(loss.detach()), "failed_gates": [k for k, v in gates.items() if not v],
                    "target_forbidden_m": values["target_forbidden_body_penetration_m"],
                    "non_target_m": values["non_target_scene_penetration_m"], "ground_m": values["ground_penetration_m"]})
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require(raw.grad is not None and bool(torch.isfinite(raw.grad).all()), "Non-finite IK gradient")
            torch.nn.utils.clip_grad_norm_([raw], 5.)
            optimizer.step()
        timings["optimize_seconds"] = time.monotonic() - before
        _, pose, values, gates, best_step = best
        with torch.no_grad():
            final = body.materialize(self.model, pose, betas, frame, contact, subset)
        output.mkdir(parents=True)
        frozen_source = output / "fast_keypose_source.py"
        for binding in (source_at_start, self.model_binding, self.segmentation_binding, *source_dependencies):
            verified(binding)
        shutil.copy2(__file__, frozen_source)
        def save(name, state):
            path = output / (name + ".npz")
            with path.open("xb") as stream:
                np.savez_compressed(stream, schema=np.array("p555.current_geometry_ik_candidate.v1"),
                    body_pose_axis_angle=state.pose.detach().numpy(), betas=betas.numpy(),
                    vertices_world_zup=state.vertices.detach().numpy(), joints_world_zup=state.joints[:22].detach().numpy(),
                    root_xyz_yaw=np.r_[state.root.detach().numpy(), yaw], pose_frame_to_world_rotation=frame.numpy(),
                    contact_world_xyz_m=contact.numpy(), faces=np.asarray(self.model.faces),
                    gluteal_support_vertex_ids=subset_np, h3_model_used=np.array(False),
                    hybrikx_model_used=np.array(False), stage3_handoff_allowed=np.array(False))
            return artifact(path)
        outputs = {"initial": save("initial_keypose", initial), "candidate": save("candidate_keypose", final)}
        receipt = {"schema": "p555.geometry_guided_keypose_proposal.v1", "status": "numeric_candidate_pass" if all(gates.values()) else "numeric_candidate_rejected",
            "provider": "declared_analytic_sit_seed_plus_current_geometry_ik", "source": artifact(frozen_source),
            "source_dependencies": source_dependencies,
            "source_hashes_rechecked_after_optimization": True,
            "inputs": {"stage1": artifact(stage1_path), "sdf_cache": artifact(sdf_path), "body_model": self.model_binding,
                       "segmentation": self.segmentation_binding},
            "scene_id": current["stage1"]["scene_id"], "target_instance_id": target["target_instance_id"],
            "contact_policy": contact_policy, "contact_world_xyz_m": contact.tolist(),
            "contact_hypotheses_tested": len(candidates), "terminal_facing_yaw_rad": yaw,
            "sdf_objective_enabled": use_sdf, "fullmesh_sdf_evaluated_even_if_objective_off": True,
            "thresholds": THRESHOLDS, "gates": gates, "metrics": values,
            "initial_gates": initial_gates, "initial_metrics": initial_values,
            "best_step": best_step, "optimizer_steps": steps, "history": history, "outputs": outputs,
            "timings": {**timings, "body_model_session_setup_seconds": self.setup_seconds, "proposal_elapsed_seconds": time.monotonic() - start},
            "body_model_identity": "exact_bound_file_with_zero_betas_not_inferred_from_gender_argument",
            "pose_semantics_scope": "neutral_sitting_both_feet_grounded_not_all_possible_sit_instructions",
            "extra_triangle_gate_scope": "gluteal_subset_XY_coverage_only_3D_contact_still_requires_original_SDF_gates",
            "h3_model_used": False, "hybrikx_model_used": False, "qwen_checked": False,
            "memory_records_consumed": [], "real_positive_credit": 0, "historical_full_pose_or_motion_read": False,
            "stage3_handoff_allowed": False, "semantic_publication_allowed": False,
            "protocol_change": "independent_geometric_proposal_with_current_triangle_contact_not_H3_lineage"}
        write_once(output / "receipt.json", receipt)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--contact-policy", choices=("surface_centre", "front_support"), default="front_support")
    parser.add_argument("--without-sdf-objective", action="store_true")
    args = parser.parse_args()
    result = FastKeyposeEngine().propose(args.stage1, args.sdf, args.output,
        steps=args.steps, contact_policy=args.contact_policy, use_sdf=not args.without_sdf_objective)
    print({"status": result["status"], "timings": result["timings"],
           "failed_gates": [k for k, value in result["gates"].items() if not value]}, flush=True)
