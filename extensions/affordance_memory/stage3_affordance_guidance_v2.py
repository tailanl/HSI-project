"""Contact-discretization-only extra guidance; the parent full raw SDF stays intact.

Only a body-bound pelvis/hip contact proxy may use a small terminal segment on
the actual selected support triangles. That segment must still be clear in a
numerically consistent non-target SDF. No body-wide or target-wide exemption,
no change to the network, original SDF energy, original physics or release gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import torch

import stage3_affordance_guidance as v1
from memory_common import PROJECT, artifact, digest, read_sealed, require, verified

SCHEMA = "p555.contact_discretization_affordance_guidance.v2"
CONTEXT_SCHEMA = "p555.current_target_contact_discretization_context.v2"
P360_RAW = PROJECT / "agent6/runs/p360_scene_keypose_motion_generator_20260810/code/raw_scene_field_zup.py"
MAXIMUM_END_BAND_M = .09
TRIANGLE_ENDPOINT_TOLERANCE_M = 2e-6
FIXED_FLOOR_IGNORE_HEIGHT_M = .08


def _artifact_only(value):
    return {key: value[key] for key in ("path", "bytes", "sha256")}


def _array_digest(array):
    value = np.ascontiguousarray(array)
    return digest({"shape": list(value.shape), "dtype": value.dtype.str,
                   "bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest()})


def pool_occupancy(value, factor):
    """Same conservative max-pool mathematics as the retained P360 loader."""
    require(type(factor) is int and factor in (1, 2, 4), "Unsupported current SDF pooling factor")
    value = np.asarray(value)
    require(value.ndim == 3 and value.dtype == np.bool_, "Expected boolean native world occupancy")
    require(all(size % factor == 0 for size in value.shape), "Ambiguous partial pooled cell")
    shape = tuple(size//factor for size in value.shape)
    return np.ascontiguousarray(value.reshape(shape[0], factor, shape[1], factor, shape[2], factor).any(axis=(1, 3, 5)))


def native_to_world(value):
    require(value.dtype == np.bool_ and value.shape == (300, 100, 400) and value.flags.c_contiguous,
            "Only the current bound LINGO native 300x100x400 boolean mask is registered")
    return np.ascontiguousarray(np.transpose(value, (0, 2, 1))[:, ::-1, :])


def split_before_pool(occupancy, target, factor):
    require(occupancy.dtype == target.dtype == np.bool_ and occupancy.shape == target.shape,
            "Target and occupancy grid mismatch")
    require(bool(target.any()) and not bool((target & ~occupancy).any()), "Target must be a nonempty exact raw-scene subset")
    # Removing the pooled target mask would erase non-target objects sharing a
    # coarse cell with the target. Subtract at native resolution FIRST.
    non_target = pool_occupancy(occupancy & ~target, factor)
    return pool_occupancy(occupancy, factor), non_target


def bounded_policy(parent_sdf, config):
    config.validate()
    receipt = parent_sdf.receipt
    spacing = tuple(float(x) for x in parent_sdf.spacing_xyz)
    require(len(spacing) == 3 and all(math.isfinite(x) and x > 0 for x in spacing), "Invalid actual parent SDF spacing")
    require(np.array_equal(np.asarray(receipt["spacing_xyz"]), np.asarray(spacing)), "Parent receipt spacing differs")
    diagonal = float(np.linalg.norm(spacing))
    requested = diagonal + config.segment_sample_spacing_m
    return {"schema": "p555.bounded_selected_triangle_alias_policy.v2", "actual_parent_spacing_xyz_m": list(spacing),
            "actual_parent_voxel_diagonal_m": diagonal, "segment_sample_spacing_m": config.segment_sample_spacing_m,
            "uncapped_discretization_band_m": requested, "hard_maximum_end_band_m": MAXIMUM_END_BAND_M,
            "effective_contact_end_band_m": min(requested, MAXIMUM_END_BAND_M),
            "maximum_raw_target_negative_depth_m": min(requested, MAXIMUM_END_BAND_M),
            "capped_below_discretization_estimate": requested > MAXIMUM_END_BAND_M,
            "cap_may_underactivate": requested > MAXIMUM_END_BAND_M,
            "raw_prefix_clearance_m": config.obstacle_clearance_m+config.segment_sample_spacing_m/2,
            "non_target_entire_segment_clearance_m": config.obstacle_clearance_m+config.segment_sample_spacing_m/2,
            "triangle_endpoint_tolerance_m": TRIANGLE_ENDPOINT_TOLERANCE_M,
            "frame_raw_j22_guard_retained": True, "support_v1_rules_retained": True,
            "whole_body_or_whole_target_exemption": False, "parent_sdf_modified": False,
            "this_is_not_a_physical_penetration_release_threshold": True}


@dataclass
class ContactContext:
    parent_sdf: object
    non_target_sdf: object
    patch: v1.SurfacePatch
    proxy_map: v1.J22ProxyMap
    source_bindings: dict
    proof: dict
    binding: v1.GuidanceBinding | None = None

    def verify_sources(self):
        if self.binding is not None:
            self.binding.verify_sources()
        for record in self.source_bindings.values():
            verified(record)

    def receipt(self):
        return {"schema": CONTEXT_SCHEMA, "source": artifact(__file__), "retained_v1": artifact(v1.__file__),
                "retained_raw_sdf": artifact(P360_RAW), "source_bindings": self.source_bindings, "proof": self.proof,
                "bound_current_geometry": self.binding is not None, "unbound_cpu_numerical_probe": self.binding is None,
                "target_discretization_workaround_only": True, "parent_raw_sdf_target_still_included": True,
                "original_release_gates_modified": False, "real_positive_credit_allowed": False}


def _validate_proxy(proof, binding):
    require(proof.get("schema") == "p555.current_body_proxy_offsets.v1", "Unregistered body proxy proof")
    body_record = binding.value["source_bindings"]["body_identity"]
    require(proof["source_body_identity"] == body_record, "Contact proxy uses another body")
    body = read_sealed(verified(body_record))
    candidate_record = proof["source_candidate"]
    require(body["source_candidate"] == candidate_record, "Proxy candidate differs from bound current body")
    candidate_path = verified(candidate_record)
    with np.load(candidate_path, allow_pickle=False) as candidate:
        require(candidate["schema"].shape == () and candidate["schema"].item() == "p555.current_geometry_ik_candidate.v1",
                "Unregistered contact body provider")
        vertices = candidate["vertices_world_zup"].copy()
        joints = candidate["joints_world_zup"].copy()
        butt = candidate["gluteal_support_vertex_ids"].copy()
    require(vertices.shape == (10475, 3) and joints.shape == (22, 3) and np.isfinite(vertices).all()
            and np.isfinite(joints).all(), "Invalid source body geometry")
    require(proof["gluteal_support_vertex_ids"] == butt.tolist(), "Proxy gluteal subset differs from actual candidate")
    count = max(8, math.ceil(len(butt)*.15))
    require(type(proof["gluteal_lower_envelope_vertex_count"]) is int and proof["gluteal_lower_envelope_vertex_count"] == count
            and proof["foot_lower_envelope_count_per_side"] == 32, "Proxy envelope policy drift")
    feet = [proof["left_foot_vertex_ids"], proof["right_foot_vertex_ids"]]
    for ids in [butt.tolist(), *feet]:
        require(len(ids) >= 8 and all(type(i) is int and 0 <= i < 10475 for i in ids)
                and len(ids) == len(set(ids)), "Invalid body surface proxy indices")
    expected = {"contact_ids": [0], "contact_drop_m": [float(joints[0, 2])-float(np.sort(vertices[butt, 2])[:count].mean())],
                "foot_drop_m": [float(joints[i, 2])-float(np.sort(vertices[ids, 2])[:32].mean()) for i, ids in zip((10, 11), feet)],
                "body_identity_sha256": body_record["sha256"]}
    require(proof["parameters"] == expected, "Contact proxy is not the actual body-derived offset")
    args = {key: tuple(value) if isinstance(value, list) else value for key, value in expected.items()}
    proxy = v1.J22ProxyMap(**args)
    proxy.points(torch.tensor(joints)[None, None], body_identity_sha256=body_record["sha256"])
    return proxy, candidate_record


def build_bound_context(parent_sdf, guidance_binding, proxy_receipt_path):
    """Launch-time CPU construction; no GPU, model weights or historical poses.

    Derives target mask from the same current Stage1 receipt as the triangles;
    callers cannot provide a replacement mask. Parent spacing/factor are read,
    never guessed from a log field called `rawoccupancy_shape`.
    """
    require(type(guidance_binding) is v1.GuidanceBinding, "A verified current guidance binding is required")
    guidance_binding.verify_sources()
    raw_module = sys.modules.get(type(parent_sdf).__module__)
    require(raw_module is not None and Path(raw_module.__file__).resolve() == P360_RAW.resolve()
            and type(parent_sdf).__name__ == "RawSceneSDFZUp", "Only the original P360 RawSceneSDFZUp is registered")
    require(parent_sdf.receipt["target_component_removed"] is False, "Parent raw SDF must still include target furniture")
    stage1_target_record = guidance_binding.current_contact_bindings["stage1_target"]
    target = read_sealed(verified(stage1_target_record))
    mask_description = target["artifacts"]["target_occupancy_mask"]
    mask_record = _artifact_only(mask_description)
    occupancy_record = guidance_binding.value["source_bindings"]["raw_occupancy"]
    require(mask_description["occupancy_source_sha256"] == occupancy_record["sha256"]
            and mask_description["coordinate_order"] == "LINGO_native_X_Yup_Z", "Target mask scene/frame mismatch")
    native = np.load(verified(occupancy_record), allow_pickle=False)
    target_native = np.load(verified(mask_record), allow_pickle=False)
    occupancy, target_world = native_to_world(native), native_to_world(target_native)
    require(raw_module._occupancy_sha256(target_world) == mask_description["world_xyz_content_sha256"]
            == target["target_occupancy_mask_world_xyz_sha256"], "Current target-mask numeric content drift")
    shape = tuple(parent_sdf.shape_xyz)
    require(len(shape) == 3 and all(type(x) is int and x >= 2 for x in shape), "Invalid parent SDF shape")
    factors = tuple(n//p for n, p in zip(occupancy.shape, shape))
    require(len(set(factors)) == 1 and all(n % p == 0 for n, p in zip(occupancy.shape, shape)), "Parent grid cannot be derived from current occupancy")
    factor = factors[0]
    require(tuple(parent_sdf.lower_xyz) == tuple(raw_module.DEFAULT_LOWER_XYZ)
            and tuple(parent_sdf.upper_xyz) == tuple(raw_module.DEFAULT_UPPER_XYZ), "Parent bounds differ from current LINGO scene")
    expected_spacing = (np.asarray(parent_sdf.upper_xyz)-np.asarray(parent_sdf.lower_xyz))/np.asarray(shape)
    require(np.array_equal(np.asarray(parent_sdf.spacing_xyz), expected_spacing), "Parent actual spacing/bounds/shape drift")
    pooled, non_target_pooled = split_before_pool(occupancy, target_world, factor)
    lower, upper = np.asarray(parent_sdf.lower_xyz), np.asarray(parent_sdf.upper_xyz)
    original_collision = raw_module._clear_floor(pooled, lower_xyz=lower, upper_xyz=upper,
                                                 floor_ignore_height_m=FIXED_FLOOR_IGNORE_HEIGHT_M)
    require(parent_sdf.collision_occupancy_xyz is not None and np.array_equal(parent_sdf.collision_occupancy_xyz, original_collision),
            "Parent collision field differs from exact current raw occupancy/floor policy")
    expected_sdf = raw_module._signed_distance_from_occupancy(original_collision, spacing_xyz=expected_spacing)
    require(np.array_equal(parent_sdf.sdf_xyz.detach().cpu().numpy()[0, 0], expected_sdf),
            "Parent SDF does not reproduce from its authoritative collision occupancy")
    non_target_sdf = raw_module.RawSceneSDFZUp.from_occupancy(non_target_pooled,
        lower_xyz=parent_sdf.lower_xyz, upper_xyz=parent_sdf.upper_xyz,
        floor_ignore_height_m=FIXED_FLOOR_IGNORE_HEIGHT_M, remove_target_component=False,
        source="p555_extra_contact_native_target_subtraction_only_not_parent_collision_field")
    proxy_record = artifact(proxy_receipt_path)
    proxy, candidate_record = _validate_proxy(read_sealed(proxy_receipt_path), guidance_binding)
    sources = {"stage1_target": stage1_target_record, "target_mask": mask_record, "raw_occupancy": occupancy_record,
               "guidance_input": guidance_binding.input_artifact, "body_proxy": proxy_record, "candidate": candidate_record}
    proof = {"parent_shape_xyz": list(shape), "native_world_shape_xyz": list(occupancy.shape), "actual_downsample_factor": factor,
             "actual_spacing_xyz_m": list(parent_sdf.spacing_xyz), "floor_ignore_height_m": FIXED_FLOOR_IGNORE_HEIGHT_M,
             "parent_sdf_recomputed_exactly": True, "parent_sdf_array_sha256": _array_digest(expected_sdf),
             "target_native_subtraction_before_pooling": True, "mixed_pooled_voxel_non_target_occupancy_preserved": True,
             "target_world_content_sha256": raw_module._occupancy_sha256(target_world),
             "non_target_collision_sha256": raw_module._occupancy_sha256(non_target_sdf.collision_occupancy_xyz),
             "non_target_sdf_array_sha256": _array_digest(non_target_sdf.sdf_xyz.numpy()),
             "target_instance_id": target["target_instance_id"], "selected_surface_id": target["selected_surface_id"],
             "semantic_contact_joint_ids": list(proxy.contact_ids), "not_an_entire_target_sdf_exemption": True}
    context = ContactContext(parent_sdf, non_target_sdf, guidance_binding.patch, proxy, sources, proof, guidance_binding)
    context.verify_sources()
    return context


class GeometryAffordanceEnergyV2(v1.GeometryAffordanceEnergy):
    def __init__(self, scene_sdf, patch, config=v1.GuidanceConfig(), *, contact_context,
                 multipliers=v1.MemoryMultipliers(), binding=None):
        super().__init__(scene_sdf, patch, config, multipliers=multipliers, binding=binding)
        require(type(contact_context) is ContactContext and contact_context.parent_sdf is scene_sdf,
                "Contact context belongs to another parent SDF")
        require(contact_context.patch.payload() == patch.payload(), "Contact context uses another selected triangle patch")
        require((binding is None and contact_context.binding is None) or binding is contact_context.binding,
                "Bound/unbound contact context mismatch")
        self.contact_context = contact_context
        self.alias_policy = bounded_policy(scene_sdf, config)
        contact_context.verify_sources()

    def receipt(self):
        base = super().receipt()
        return {**base, "schema": SCHEMA, "source": artifact(__file__), "retained_v1": artifact(v1.__file__),
                "contact_discretization_policy": self.alias_policy, "contact_context": self.contact_context.receipt(),
                "target_discretization_workaround_only": True, "parent_full_raw_sdf_unchanged": True,
                "original_release_gates_modified": False, "real_positive_credit_allowed": False}

    def _contact_segment_clear(self, points, targets):
        distance = (points-targets).norm(dim=-1)
        count = math.ceil(self.config.contact_capture_radius_m/self.config.segment_sample_spacing_m)+1
        alpha = torch.linspace(0, 1, count, device=points.device, dtype=points.dtype)
        samples = points.detach().unsqueeze(-2)+(targets-points.detach()).unsqueeze(-2)*alpha[..., None]
        raw_sdf, raw_inside = self._sample(samples)
        non_target = self.contact_context.non_target_sdf.sample(samples)
        other_sdf, other_inside = non_target.signed_distance_m, non_target.in_bounds
        require(other_sdf.shape == raw_sdf.shape and other_inside.shape == raw_inside.shape
                and bool(torch.isfinite(other_sdf).all()), "Non-target SDF sample contract mismatch")
        endpoint_distance = (samples-targets.unsqueeze(-2)).norm(dim=-1)
        band = endpoint_distance <= self.alias_policy["effective_contact_end_band_m"]
        raw_clear = raw_sdf >= self.alias_policy["raw_prefix_clearance_m"]
        shallow_target_alias = band & (raw_sdf >= -self.alias_policy["maximum_raw_target_negative_depth_m"])
        actual_endpoint = (self.patch.closest_world(targets)-targets).norm(dim=-1) <= TRIANGLE_ENDPOINT_TOLERANCE_M
        non_target_clear = other_sdf >= self.alias_policy["non_target_entire_segment_clearance_m"]
        clear = ((raw_clear | shallow_target_alias) & non_target_clear & raw_inside & other_inside).all(-1)
        return clear & actual_endpoint & (distance.detach() <= self.config.contact_capture_radius_m)

    def energy(self, *, contact_points_world, support_points_world, forward_world_xy, obstacle_points_world,
               contact_activation, support_activation, query_id=None, body_identity_sha256=None):
        if self.binding:
            self.binding.assert_current(query_id=query_id, body_identity_sha256=body_identity_sha256)
        # Only actual body-bound pelvis/hip J22 proxies; callers cannot use the
        # exceptional segment rule for hands, arbitrary points or full meshes.
        proxy = self.contact_context.proxy_map
        expected = proxy.points(obstacle_points_world, body_identity_sha256=proxy.body_identity_sha256)
        require(contact_points_world.shape == expected["contact_points_world"].shape
                and torch.allclose(contact_points_world.detach(), expected["contact_points_world"].detach(), atol=1e-6, rtol=0),
                "Extra contact point is not the registered current-body pelvis/hip proxy")
        values = [contact_points_world, support_points_world, obstacle_points_world]
        for point in values:
            v1._finite_tensor(point, (3,), "world points")
            require(point.ndim == 4 and min(point.shape[:3]) > 0, "Points must be nonempty [B,T,K,3]")
        contact, support, obstacle = values
        require(all(x.shape[:2] == contact.shape[:2] and x.device == contact.device and x.dtype == contact.dtype for x in values),
                "Point batch/device/dtype mismatch")
        v1._finite_tensor(forward_world_xy, (2,), "forward XY")
        require(forward_world_xy.shape == (*contact.shape[:2], 2) and forward_world_xy.device == contact.device
                and forward_world_xy.dtype == contact.dtype, "Facing shape/device/dtype mismatch")
        ca = v1._mask(contact_activation, tuple(contact.shape[:2]), contact, "contact activation")
        sa = v1._mask(support_activation, tuple(support.shape[:-1]), support, "support activation")
        zero = sum(x.sum()*0 for x in [*values, forward_world_xy])
        if not self.config.enabled:
            return zero, {"p555_affordance_energy": zero, "p555_guidance_enabled": zero.detach()}
        target = self.patch.closest_world(contact)
        contact_distance = (contact-target).norm(dim=-1)
        support_target = support.detach().clone()
        support_target[..., 2] = self.ground_z_m+self.config.support_clearance_m
        support_distance = (support[..., 2]-support_target[..., 2]).abs()
        current_sdf, inside = self._sample(obstacle)
        frame_safe = ((current_sdf >= self.config.obstacle_clearance_m) & inside).all(-1)
        contact_safe = self._contact_segment_clear(contact, target)
        # EXACT retained v1 support segment rule; no target exceptions here.
        support_safe = super()._segment_clear(support, support_target, self.config.support_capture_height_m)
        local = contact_distance.detach() <= self.config.contact_capture_radius_m
        cm = ca[..., None]*frame_safe[..., None]*contact_safe*local
        sm = sa*frame_safe[..., None]*support_safe*(support_distance.detach() <= self.config.support_capture_height_m)
        contact_raw = (.5*torch.relu(contact_distance-self.config.contact_deadzone_m).square()*cm).mean()
        support_raw = (.5*torch.relu(support_distance-self.config.support_deadzone_m).square()*sm).mean()
        desired = self.patch.rotation[:2, :2].to(contact) @ self.patch.facing.to(contact)
        norm = forward_world_xy.norm(dim=-1)
        unit = forward_world_xy/norm[..., None].clamp_min(.1)
        facing_raw = ((1-(unit*desired).sum(-1)).clamp(0, 2)*ca*frame_safe*contact_safe.all(-1)
                      *local.all(-1)*(norm.detach() >= .1)).mean()
        wc = self.config.contact_weight*self.multipliers.contact_weight_multiplier*contact_raw
        ws = self.config.support_weight*self.multipliers.stance_weight_multiplier*support_raw
        wf = self.config.facing_weight*facing_raw
        total = wc+ws+wf+zero
        return total, {"p555_affordance_energy": total, "p555_contact_raw_energy_m2": contact_raw,
                       "p555_support_raw_energy_m2": support_raw, "p555_facing_raw_energy": facing_raw,
                       "p555_contact_weighted_energy": wc, "p555_support_weighted_energy": ws,
                       "p555_facing_weighted_energy": wf, "p555_frame_safety_fraction": frame_safe.to(contact).mean(),
                       "p555_contact_active_fraction": (cm > 0).to(contact).mean(),
                       "p555_support_active_fraction": (sm > 0).to(contact).mean(),
                       "p555_original_obstacle_min_sdf_m": current_sdf.amin(), "p555_guidance_enabled": contact.new_tensor(1.),
                       "p555_v2_contact_discretization_guard": contact.new_tensor(1.),
                       "p555_v2_effective_end_band_m": contact.new_tensor(self.alias_policy["effective_contact_end_band_m"])}
