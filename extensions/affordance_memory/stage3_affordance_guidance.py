"""Opt-in current-geometry energy; not a Stage3 lineage or release adapter.

The numerical API also accepts Stage2 IK's differentiable mesh contact points.
Stage3 may explicitly supply J22 proxies, but these are NOT verified mesh
contacts. No motion, pose library or model is read here. Geometry is measured
in metres, world Z-up; local frames are proper rotations with Z = world up.

Injection point: P533 run_p533_p478_v1.expose_p533_current_stepwise_physics_guidance
adds P523 physics in energy_from_world_joints, on top of P478 and P360's raw SDF.
A new process-local wrapper can call that complete parent exactly once, compute
this energy from the same clean-xstart J22, and call combine_with_parent. P360's
existing latent norm / step caps and all evaluation / release gates stay intact.
Do not insert terminal attraction on every primitive: activation must come from
the CURRENT scheduler node, support activation from current-predicted stance.
No CLI or automatic P552/P554 -> legacy P533 schema conversion is provided.

The SDF guard is conservative sampled evidence, not a collision-free trajectory
proof: it stops added attraction near current obstacles, outside SDF bounds or
across an obstructed effector-to-target segment. It does not remove any parent
SDF constraint. Latent coupling can still move other body vertices; original
full-mesh / temporal / support / no-skate release checks remain mandatory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
import re
import zipfile
from typing import Any

import torch
from torch import Tensor

from memory_common import artifact, digest, read_sealed, require, verified

SCHEMA = "p555.current_geometry_affordance_guidance.v1"
INPUT_SCHEMA = "p555.current_geometry_affordance_input.v1"
J22_NAMES = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
             "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
             "neck", "left_collar", "right_collar", "head", "left_shoulder",
             "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
SOURCE_ROLES = {"stage1_execution", "fixed_geometry", "contact_surface", "raw_occupancy",
                "body_identity", "geometry_producer"}


def _sha(value, label):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None,
            "Invalid " + label)
    return value


def _number(value, low, high, label):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            "Invalid bounded " + label)
    return float(value)


def _finite_tensor(value, shape_tail, label):
    require(isinstance(value, Tensor) and value.is_floating_point()
            and tuple(value.shape[-len(shape_tail):]) == shape_tail and bool(torch.isfinite(value).all()),
            "Invalid finite " + label)
    return value


@dataclass(frozen=True)
class GuidanceConfig:
    enabled: bool = True
    contact_weight: float = 1.0
    support_weight: float = 0.35
    facing_weight: float = 0.025
    contact_deadzone_m: float = 0.01
    support_deadzone_m: float = 0.01
    support_clearance_m: float = 0.005
    contact_capture_radius_m: float = 0.60
    support_capture_height_m: float = 0.20
    obstacle_clearance_m: float = 0.02
    segment_sample_spacing_m: float = 0.01
    authorized_contact_end_band_m: float = 0.03

    def validate(self):
        require(type(self.enabled) is bool, "enabled must be boolean")
        for key, upper in (("contact_weight", 4), ("support_weight", 2), ("facing_weight", .2)):
            _number(getattr(self, key), 0, upper, key)
        for key, low, upper in (("contact_deadzone_m", 0, .05), ("support_deadzone_m", 0, .05),
                                ("support_clearance_m", 0, .03), ("contact_capture_radius_m", .05, 1),
                                ("support_capture_height_m", .02, .30), ("obstacle_clearance_m", .005, .10),
                                ("segment_sample_spacing_m", .005, .025),
                                ("authorized_contact_end_band_m", .005, .04)):
            _number(getattr(self, key), low, upper, key)
        require(self.contact_deadzone_m < self.contact_capture_radius_m
                and self.support_deadzone_m < self.support_capture_height_m, "Deadzone exceeds capture region")
        return self


@dataclass(frozen=True)
class MemoryMultipliers:
    """Numerical coefficients only; construction is NOT evidence of active memory.

    Production wrappers obtain these via multipliers_from_store, retain that
    query receipt and use its fixed snapshot. Direct non-unit values are manual
    coefficient ablations, never earned memory or positive credit.
    """
    sdf_weight_multiplier: float = 1.0
    contact_weight_multiplier: float = 1.0
    stance_weight_multiplier: float = 1.0

    def validate(self):
        for key, low, high in (("sdf_weight_multiplier", 1, 1.25),
                               ("contact_weight_multiplier", .8, 1.2),
                               ("stance_weight_multiplier", 1, 1.25)):
            _number(getattr(self, key), low, high, key)
        return self


def multipliers_from_store(store, cycle, *, invariant, record_id=None):
    """Hard-query production MemoryStore; no caller-authored active=true input.

    No active record -> M0, unit coefficients. Multiple active records require an
    explicit ID; no averaging or guessing. Selection still needs current local
    geometric validation. Invariant includes actual body model / shape bin.
    """
    from memory_store import MemoryStore
    from memory_schema import validate_record, validate_invariant
    require(type(store) is MemoryStore and store.purpose == "production", "A production MemoryStore is required")
    require(isinstance(invariant, dict), "Exact typed invariant required")
    validate_invariant("stage3_guidance", invariant)
    query = store.query_active(cycle, "stage3_guidance", invariant_key=invariant)
    require(query["schema"] == "p555.active_memory_query.v1" and query["purpose"] == "production",
            "Invalid active query")
    require(query["receipt_payload_sha256"] == digest({k: v for k, v in query.items() if k != "receipt_payload_sha256"}),
            "Active query seal mismatch")
    rows = query["records"]
    if not rows:
        require(record_id is None and query["m0_bypass"] is True, "Requested memory is not active")
        return MemoryMultipliers(), query
    selected = [row for row in rows if record_id is None or row["record"]["record_id"] == record_id]
    require(len(selected) == 1, "Select exactly one currently active record")
    row = selected[0]
    validate_record(row["record"])
    require(row["status"] == "active" and row["record"]["invariant"] == invariant
            and row["record"]["stage"] == "stage3_guidance", "Inactive or mismatched memory record")
    return MemoryMultipliers(**row["record"]["payload"]).validate(), query


class SurfacePatch:
    """Actual local triangles; nearest targets never fill holes with an AABB."""
    def __init__(self, *, origin_world_zup_m, local_to_world_rotation, triangles_local_m,
                 facing_local_xy=(1., 0.)):
        self.origin = torch.as_tensor(origin_world_zup_m, dtype=torch.float64).detach().clone()
        self.rotation = torch.as_tensor(local_to_world_rotation, dtype=torch.float64).detach().clone()
        self.triangles = torch.as_tensor(triangles_local_m, dtype=torch.float64).detach().clone()
        self.facing = torch.as_tensor(facing_local_xy, dtype=torch.float64).detach().clone()
        require(self.origin.shape == (3,) and self.rotation.shape == (3, 3)
                and self.triangles.ndim == 3 and self.triangles.shape[1:] == (3, 3)
                and 1 <= len(self.triangles) <= 16384 and self.facing.shape == (2,), "Invalid patch dimensions")
        require(all(bool(torch.isfinite(x).all()) for x in (self.origin, self.rotation, self.triangles, self.facing)),
                "Nonfinite patch")
        require(torch.allclose(self.rotation.T @ self.rotation, torch.eye(3, dtype=torch.float64), atol=1e-7, rtol=0)
                and abs(float(torch.det(self.rotation)) - 1) < 1e-7
                and torch.allclose(self.rotation[:, 2], torch.tensor([0., 0., 1.], dtype=torch.float64), atol=1e-7, rtol=0),
                "Patch rotation must be right-handed world Z-up")
        normals = torch.linalg.cross(self.triangles[:, 1] - self.triangles[:, 0],
                                     self.triangles[:, 2] - self.triangles[:, 0], dim=-1)
        length = torch.linalg.vector_norm(normals, dim=-1)
        require(bool((length > 1e-10).all()) and bool((normals[:, 2].abs() / length >= .5).all()),
                "Degenerate or non-support contact triangle")
        require(abs(float(self.facing.norm()) - 1) < 1e-7, "Facing must be a unit local XY vector")

    def payload(self):
        return {"origin_world_zup_m": self.origin.tolist(), "local_to_world_rotation": self.rotation.tolist(),
                "triangles_local_m": self.triangles.tolist(), "facing_local_xy": self.facing.tolist()}

    def closest_world(self, points: Tensor):
        """Piecewise exact triangle projection, bounded chunks, detached targets.

        Detaching the nearest point is the exact squared-distance gradient away
        from projection ties. This also prevents gradients through geometry.
        """
        with torch.no_grad():
            rotation, origin = self.rotation.to(points), self.origin.to(points)
            local = (points.detach() - origin) @ rotation
            flat = local.reshape(-1, 3)
            best_d = torch.full((len(flat),), float("inf"), device=points.device, dtype=points.dtype)
            best = torch.zeros_like(flat)
            for triangle in self.triangles.to(points).split(256):
                a, b, c = triangle.unbind(1)
                ab, ac = b-a, c-a
                normal = torch.linalg.cross(ab, ac, dim=-1)
                normal = normal / normal.norm(dim=-1, keepdim=True)
                p = flat[:, None, :]
                projected = p - ((p-a) * normal).sum(-1, keepdim=True) * normal
                relative = projected-a
                d00, d01, d11 = (ab*ab).sum(-1), (ab*ac).sum(-1), (ac*ac).sum(-1)
                d20, d21 = (relative*ab).sum(-1), (relative*ac).sum(-1)
                denominator = d00*d11-d01*d01
                u, v = (d11*d20-d01*d21)/denominator, (d00*d21-d01*d20)/denominator
                inside = (u >= 0) & (v >= 0) & (u+v <= 1)
                candidates = [projected]
                distances = [torch.where(inside, (p-projected).square().sum(-1), float("inf"))]
                for start, end in ((a, b), (b, c), (c, a)):
                    edge = end-start
                    alpha = (((p-start)*edge).sum(-1)/(edge*edge).sum(-1)).clamp(0, 1)
                    nearest = start + alpha[..., None]*edge
                    candidates.append(nearest)
                    distances.append((p-nearest).square().sum(-1))
                values = torch.stack(distances, -1)
                which = values.argmin(-1)
                coords = torch.stack(candidates, -2)
                coords = coords.gather(-2, which[..., None, None].expand(*which.shape, 1, 3)).squeeze(-2)
                nearest_d, which_triangle = values.amin(-1).min(-1)
                nearest = coords[torch.arange(len(flat), device=points.device), which_triangle]
                improve = nearest_d < best_d
                best, best_d = torch.where(improve[:, None], nearest, best), torch.minimum(best_d, nearest_d)
            return (best.reshape_as(points) @ rotation.T + origin).detach()


class GuidanceBinding:
    """Small, explicit source binding; no claim that legacy Stage3 accepts it.

    Input schema is new, never a disguised P530/P533 receipt. The producing
    geometry code is itself bound. Validating source hashes detects drift, not
    whether a numerical producer or complete motion passed any release gate.
    """
    def __init__(self, path):
        path = Path(path)
        require(path.stat().st_size <= 16*1024*1024, "Guidance input exceeds size bound")
        self.input_artifact = artifact(path)
        value = read_sealed(path)
        require(value.get("schema") == INPUT_SCHEMA and value.get("coordinate_frame") == "world_zup"
                and value.get("units") == "metres", "Unknown guidance schema or coordinate units")
        require(set(value["source_bindings"]) == SOURCE_ROLES, "Incomplete current source bindings")
        for record in value["source_bindings"].values():
            require(type(record.get("bytes")) is int and record["bytes"] <= 512*1024*1024,
                    "Source must be a bounded artifact, not a large model")
            verified(record)
        require(isinstance(value.get("scene_id"), str) and value["scene_id"], "Missing scene ID")
        _sha(value["query_id"], "query identity")
        body = read_sealed(value["source_bindings"]["body_identity"]["path"])
        require(body.get("schema") == "p555.current_body_identity.v1"
                and body.get("model_type") == "smplx" and body.get("joint_order") == list(J22_NAMES),
                "Explicit SMPL-X J22 body identity required")
        _sha(body["body_model_sha256"], "body model hash")
        _sha(body["betas_sha256"], "body shape hash")
        require(body.get("coordinate_frame") == "world_zup" and body.get("units") == "metres",
                "Body identity coordinate contract mismatch")
        self.body_identity_sha256 = value["source_bindings"]["body_identity"]["sha256"]
        self.patch = SurfacePatch(**value["surface_patch"])
        self.value = copy.deepcopy(value)
        self.current_contact_bindings = self._verify_current_contact()
        require(artifact(path) == self.input_artifact, "Input changed during binding")
        self.verify_sources()

    def _verify_current_contact(self):
        """Read only the registered P552 target/region chain, not arbitrary true flags.

        This validates current geometry for a numerical ablation. Even a passed
        P552 route does not manufacture a P530 Stage3 adapter or passed keypose.
        NumPy is used only to decode the existing bounded contact NPZ here;
        all runtime energy/gradient operations remain torch-only.
        """
        import numpy as np
        sources = self.value["source_bindings"]
        stage1 = read_sealed(sources["stage1_execution"]["path"])
        geometry = read_sealed(sources["fixed_geometry"]["path"])
        require(stage1.get("schema") == "p550.stage1_query_execution.v1" and stage1.get("status") == "complete"
                and stage1.get("source_fixed_geometry") == sources["fixed_geometry"], "Unbound or incomplete current Stage1")
        require(stage1.get("scene_id") == geometry.get("scene_id") == self.value["scene_id"]
                and geometry.get("schema") in {"p550.fixed_scene_interaction_geometry.v1", "p550.fixed_scene_interaction_geometry.v2"}
                and geometry.get("source_occupancy") == sources["raw_occupancy"], "Current scene/occupancy binding mismatch")
        selected = {"stage1_target": stage1["target"], "p552_route_guard": stage1["p552_contact_region_guard"]}
        for record in selected.values():
            require(type(record.get("bytes")) is int and record["bytes"] <= 16*1024*1024, "Oversized target/route JSON")
            verified(record)
        target = read_sealed(selected["stage1_target"]["path"])
        guard = read_sealed(selected["p552_route_guard"]["path"])
        require(target.get("schema") == "p523.current_only_stage1_target_surface.v1"
                and target.get("scene_id") == self.value["scene_id"] and target.get("action_family") == "sit",
                "Only current sit contact geometry is registered")
        require(guard.get("schema") == "p552.contact_region_route_endpoint_guard.v1"
                and guard.get("phase") == "final_navmesh" and guard.get("all_gates_passed") is True
                and guard.get("scene_id") == self.value["scene_id"] and guard.get("source_target") == selected["stage1_target"],
                "P552 route guard is absent or bound to another target")
        required_gates = {"selected_candidate_unmodified", "goal_inside_direction_specific_frontier",
                          "goal_inside_clipped_reachable_interaction_region", "no_off_region_snap",
                          "no_intervening_non_target_obstacle", "original_human_clearance_preserved",
                          "actual_navmesh_endpoint_inside_region", "actual_route_all_samples_fixed_human_free"}
        require(set(guard.get("quality_gates", {})) == required_gates
                and all(guard["quality_gates"][key] is True for key in required_gates), "P552 route guard was not passed intact")
        surfaces = [row for row in target["candidate_surfaces"] if row["candidate_id"] == target["selected_surface_id"]]
        require(len(surfaces) == 1, "Selected contact surface is ambiguous")
        options = [row for row in surfaces[0]["approach_options"] if row["option_id"] == guard["selected_approach_option_id"]]
        require(len(options) == 1, "Selected current approach is ambiguous")
        option = options[0]
        require(option["p552_region_member"]["region_arrays"] == sources["contact_surface"], "Contact triangles belong to another target/approach")
        occupancy = target["artifacts"]["scene_occupancy"]
        require({key: occupancy[key] for key in ("path", "bytes", "sha256")} == sources["raw_occupancy"],
                "Target uses a different raw occupancy")
        yaw = _number(option["contact_forward_yaw_rad"], -2*math.pi, 2*math.pi, "current contact yaw")
        expected = torch.tensor([math.cos(yaw), math.sin(yaw)], dtype=torch.float64)
        actual = self.patch.rotation[:2, :2] @ self.patch.facing
        require(torch.allclose(actual, expected, atol=1e-6, rtol=0), "Facing does not match current contact option")
        path = sources["contact_surface"]["path"]
        with zipfile.ZipFile(path) as archive:
            require(len(archive.infolist()) <= 64 and sum(i.file_size for i in archive.infolist()) <= 32*1024*1024,
                    "Contact archive exceeds bounded decompressed size")
        with np.load(path, allow_pickle=False) as archive:
            require("contact_triangles" in archive.files, "No actual current contact triangles")
            raw = archive["contact_triangles"]
            require(raw.dtype.kind == "f" and np.isfinite(raw).all(), "Invalid measured contact triangles")
            world = torch.from_numpy(raw.copy()).to(dtype=torch.float64)
        declared = self.patch.triangles @ self.patch.rotation.T + self.patch.origin
        require(world.shape == declared.shape and torch.allclose(world, declared, atol=1e-6, rtol=0),
                "Declared patch is not the bound current contact triangles")
        return copy.deepcopy(selected)

    def assert_current(self, *, query_id, body_identity_sha256):
        require(query_id == self.value["query_id"] and body_identity_sha256 == self.body_identity_sha256,
                "Runtime query/body identity differs from bound geometry")

    def verify_sources(self):
        """Call at launch/end boundaries, not each denoising step."""
        verified(self.input_artifact)
        for record in self.value["source_bindings"].values():
            verified(record)
        for record in self.current_contact_bindings.values():
            verified(record)

    def receipt(self):
        return {"input": copy.deepcopy(self.input_artifact), "source_bindings": copy.deepcopy(self.value["source_bindings"]),
                "query_id": self.value["query_id"], "scene_id": self.value["scene_id"],
                "body_identity_sha256": self.body_identity_sha256,
                "current_contact_bindings": copy.deepcopy(self.current_contact_bindings),
                "surface_patch_payload_sha256": digest(self.patch.payload()),
                "legacy_stage3_lineage_compatibility_verified": False,
                "motion_release_authorized": False, "real_positive_credit_allowed": False}


def _mask(value, shape, reference, label):
    require(isinstance(value, Tensor) and tuple(value.shape) == shape and value.device == reference.device,
            "Invalid " + label + " shape/device")
    require(bool(torch.isfinite(value).all()) and bool(((value >= 0) & (value <= 1)).all()), "Invalid activation values")
    require(not value.requires_grad, "Activation must be detached current phase/stance evidence")
    return value.to(reference)


class GeometryAffordanceEnergy:
    def __init__(self, scene_sdf, patch: SurfacePatch, config=GuidanceConfig(), *,
                 multipliers=MemoryMultipliers(), binding: GuidanceBinding | None = None):
        require(callable(getattr(scene_sdf, "sample", None)), "Original raw SDF.sample is required")
        lower = getattr(scene_sdf, "lower_xyz", None)
        require(isinstance(lower, (list, tuple)) and len(lower) == 3
                and all(type(x) in (int, float) and math.isfinite(x) for x in lower), "SDF lacks nominal ground Z")
        self.scene_sdf, self.patch = scene_sdf, patch
        self.config, self.multipliers = config.validate(), multipliers.validate()
        self.binding, self.ground_z_m = binding, float(lower[2])
        require(binding is None or binding.patch.payload() == patch.payload(), "Bound contact geometry mismatch")

    def receipt(self):
        return {"schema": SCHEMA, "source": artifact(__file__), "config": asdict(self.config),
                "multipliers": asdict(self.multipliers), "coordinate_frame": "world_zup", "units": "metres",
                "nominal_support_plane_source": "original_raw_sdf.lower_xyz[2]", "ground_z_m": self.ground_z_m,
                "geometry_kind": "current_actual_triangles_not_successful_experience",
                "memory_coefficients_alone_prove_active_memory": False,
                "binding": self.binding.receipt() if self.binding else None,
                "unbound_cpu_numerical_probe": self.binding is None,
                "original_sdf_constraints_removed": False, "original_release_gates_modified": False,
                "sdf_guard_is_fullmesh_or_trajectory_proof": False, "real_positive_credit_allowed": False,
                "coefficient_mapping": {"contact": "new_triangle_contact_energy",
                    "stance": "new_nominal_floor_support_band_energy_not_p523_slip",
                    "sdf": "original_parent_sdf_energy_times_original_sdf_weight_additive_nonnegative_delta",
                    "facing": "fixed_new_facing_weight_no_memory_multiplier"}}

    def _sample(self, points):
        with torch.no_grad():
            sampled = self.scene_sdf.sample(points.detach())
            sdf, inside = sampled.signed_distance_m, sampled.in_bounds
            require(sdf.shape == points.shape[:-1] and inside.shape == sdf.shape
                    and inside.dtype == torch.bool and sdf.device == points.device
                    and bool(torch.isfinite(sdf).all()), "Malformed or nonfinite raw SDF sample")
            return sdf.detach(), inside.detach()

    def _segment_clear(self, points, targets, max_length):
        count = math.ceil(max_length / self.config.segment_sample_spacing_m) + 1
        alpha = torch.linspace(0, 1, count, device=points.device, dtype=points.dtype)
        samples = points.detach().unsqueeze(-2) + alpha[:, None]*(targets-points.detach()).unsqueeze(-2)
        sdf, inside = self._sample(samples)
        # Half spacing adds a conservative buffer between sampled locations.
        threshold = self.config.obstacle_clearance_m + self.config.segment_sample_spacing_m/2
        # The intended true surface endpoint has SDF ~= 0, not clearance > 0.
        # Only a small terminal band may approach zero; negative SDF is never
        # accepted. This changes this *extra attraction guard*, never the raw
        # parent SDF or its pre-existing, explicitly registered exemptions.
        endpoint_distance = (samples-targets.unsqueeze(-2)).norm(dim=-1)
        terminal_band = endpoint_distance <= self.config.authorized_contact_end_band_m
        clear = (sdf >= threshold) | (terminal_band & (sdf >= 0))
        return (clear & inside).all(-1)

    def energy(self, *, contact_points_world: Tensor, support_points_world: Tensor,
               forward_world_xy: Tensor, obstacle_points_world: Tensor,
               contact_activation: Tensor, support_activation: Tensor,
               query_id=None, body_identity_sha256=None):
        """Pure-torch differentiable added energy, scalar + tensor diagnostics.

        points = [B,T,K,3], forward = [B,T,2], contact activation = [B,T],
        support activation = [B,T,F] (detached current stance, NOT future labels).
        Contact points should lie on the body surface; explicit J22 proxy users
        must subtract their body-bound geometric offsets before calling this.
        Obstacle points are the current whole-body joint/proxy set, not targets.
        """
        if self.binding:
            self.binding.assert_current(query_id=query_id, body_identity_sha256=body_identity_sha256)
        values = [contact_points_world, support_points_world, obstacle_points_world]
        for x in values:
            _finite_tensor(x, (3,), "world points")
            require(x.ndim == 4 and min(x.shape[:3]) > 0, "Points must be nonempty [B,T,K,3]")
        contact, support, obstacle = values
        require(all(x.shape[:2] == contact.shape[:2] and x.device == contact.device and x.dtype == contact.dtype for x in values),
                "Point batch/device/dtype mismatch")
        _finite_tensor(forward_world_xy, (2,), "forward XY")
        require(forward_world_xy.shape == (*contact.shape[:2], 2) and forward_world_xy.device == contact.device
                and forward_world_xy.dtype == contact.dtype, "Facing shape/device/dtype mismatch")
        ca = _mask(contact_activation, tuple(contact.shape[:2]), contact, "contact activation")
        sa = _mask(support_activation, tuple(support.shape[:-1]), support, "support activation")
        zero = sum(x.sum()*0 for x in [*values, forward_world_xy])
        if not self.config.enabled:
            return zero, {"p555_affordance_energy": zero, "p555_guidance_enabled": zero.detach()}
        target = self.patch.closest_world(contact)
        contact_distance = (contact-target).norm(dim=-1)
        support_target = support.detach().clone()
        support_target[..., 2] = self.ground_z_m + self.config.support_clearance_m
        support_distance = (support[..., 2]-support_target[..., 2]).abs()
        current_sdf, inside = self._sample(obstacle)
        frame_safe = ((current_sdf >= self.config.obstacle_clearance_m) & inside).all(-1)
        contact_safe = self._segment_clear(contact, target, self.config.contact_capture_radius_m)
        support_safe = self._segment_clear(support, support_target, self.config.support_capture_height_m)
        local = (contact_distance.detach() <= self.config.contact_capture_radius_m)
        cm = ca[..., None]*frame_safe[..., None]*contact_safe*local
        sm = sa*frame_safe[..., None]*support_safe*(support_distance.detach() <= self.config.support_capture_height_m)
        # Fixed denominators: rejected constraints cannot inflate remaining gradients.
        contact_raw = (.5*torch.relu(contact_distance-self.config.contact_deadzone_m).square()*cm).mean()
        support_raw = (.5*torch.relu(support_distance-self.config.support_deadzone_m).square()*sm).mean()
        desired = self.patch.rotation[:2, :2].to(contact) @ self.patch.facing.to(contact)
        norm = forward_world_xy.norm(dim=-1)
        heading_valid = norm.detach() >= .1
        unit = forward_world_xy/norm[..., None].clamp_min(.1)
        facing_raw = ((1-(unit*desired).sum(-1)).clamp(0, 2)*ca*frame_safe
                      *contact_safe.all(-1)*local.all(-1)*heading_valid).mean()
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
                       "p555_original_obstacle_min_sdf_m": current_sdf.amin(),
                       "p555_guidance_enabled": contact.new_tensor(1.)}

    def combine_with_parent(self, parent_total, parent_diagnostics, added, added_diagnostics, *, original_sdf_weight):
        """Keep parent raw SDF / ICGF / P478 / P523 energies and diagnostics.

        The disabled path returns the exact original objects. The only memory
        SDF mapping is its actual P360 raw sdf_energy, with nonnegative gain.
        No legacy six-coefficient relabeling or replacement safety term.
        """
        if not self.config.enabled:
            return parent_total, parent_diagnostics
        weight = _number(original_sdf_weight, 1e-12, 100, "original nonzero SDF weight")
        require("sdf_energy" in parent_diagnostics, "Parent lacks actual P360 raw SDF energy")
        for label, value in (("parent", parent_total), ("added", added), ("raw SDF", parent_diagnostics["sdf_energy"])):
            require(isinstance(value, Tensor) and value.ndim == 0 and bool(torch.isfinite(value)), "Invalid scalar " + label)
        require(bool(parent_diagnostics["sdf_energy"] >= 0) and bool(added >= 0), "Added/SDF energy must be nonnegative")
        require(not (set(parent_diagnostics) & set(added_diagnostics)), "Cannot overwrite parent diagnostics")
        delta = (self.multipliers.sdf_weight_multiplier-1)*weight*parent_diagnostics["sdf_energy"]
        total = parent_total+added+delta
        return total, {**parent_diagnostics, **added_diagnostics,
                       "p555_original_sdf_additive_delta": delta,
                       "p555_combined_parent_and_affordance_energy": total}


@dataclass(frozen=True)
class J22ProxyMap:
    """Explicit proxy offsets, not anatomical glute/sole contact validation.

    Offsets are metres in world Z, calibrated for the bound body identity. There
    are no universal hip/foot offsets: caller must provide them. Facing uses
    Z cross (right hip - left hip), exactly P523 anatomical_yaw's J22 convention.
    """
    contact_ids: tuple[int, ...]
    contact_drop_m: tuple[float, ...]
    foot_drop_m: tuple[float, float]
    body_identity_sha256: str

    def points(self, joints, *, body_identity_sha256):
        _finite_tensor(joints, (22, 3), "SMPL-X J22")
        require(joints.ndim == 4 and body_identity_sha256 == self.body_identity_sha256, "J22 runtime body identity mismatch")
        _sha(self.body_identity_sha256, "proxy body identity")
        require(isinstance(self.contact_ids, tuple) and 1 <= len(self.contact_ids) <= 3
                and all(type(i) is int and i in (0, 1, 2) for i in self.contact_ids)
                and len(set(self.contact_ids)) == len(self.contact_ids), "Only explicit pelvis/hip contact proxies")
        require(len(self.contact_drop_m) == len(self.contact_ids) and len(self.foot_drop_m) == 2, "Proxy offset count mismatch")
        for value in (*self.contact_drop_m, *self.foot_drop_m):
            _number(value, 0, .30, "body-calibrated proxy drop")
        contact = joints[:, :, self.contact_ids].clone()
        contact[..., 2] -= joints.new_tensor(self.contact_drop_m)
        support = joints[:, :, (10, 11)].clone()
        support[..., 2] -= joints.new_tensor(self.foot_drop_m)
        hip_axis = joints[:, :, 2, :2]-joints[:, :, 1, :2]
        forward = torch.stack((-hip_axis[..., 1], hip_axis[..., 0]), -1)
        return {"contact_points_world": contact, "support_points_world": support,
                "forward_world_xy": forward, "obstacle_points_world": joints}
