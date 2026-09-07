"""Strict metric local-experience retrieval and current-scene projection.

V1 has evidence for Stage2 sit support only. Stage1 route and Stage3 guidance
surface roles are deliberately unsupported until a typed producer exists;
an upward normal alone never makes a surface a seat or walkable floor.

Experience XY coordinates are centred on descriptor raster bounds; local Z
retains the measured surface origin. Root offset is relative to local contact.
No pose/route/motion is scaled, mirrored, rotated in a matching search, or read.
The ONLY world transform used at projection is the current audited direction.
SDF callbacks veto proposal points; they do not replace full-body/refine gates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from memory_common import (SCHEMA_HASH_FIELD, artifact, canonical_bytes, digest,
                           read_sealed, require, verified)
from memory_schema import validate_shape, validate_record, validate_invariant, SCOPES
from memory_store import MemoryStore
from scene_facts import load_descriptors
from surface_geometry import validate_descriptor

FRAME_CONVENTION = "XY=raster_bounds_center;Z=surface_origin;X=audited_outward;Y=Z_cross_X;root_offset=from_contact"
POLICY = {"maximum_xy_dimension_relative_difference": .15,
          "maximum_area_relative_difference": .35,
          "maximum_height_span_difference_m": .025,
          "minimum_observed_coverage": .5, "minimum_bidirectional_overlap": .8,
          "maximum_height_rmse_m": .02, "maximum_height_p95_m": .04,
          "maximum_contact_height_difference_m": .025,
          "maximum_normal_angle_degrees": 15.}


def _copy(value):
    return json.loads(canonical_bytes(value))


def _seal(value):
    value = _copy({k: v for k, v in value.items() if k != SCHEMA_HASH_FIELD})
    return {**value, SCHEMA_HASH_FIELD: digest(value)}


def descriptor_to_shape(triple):
    """Convert a validated load_descriptors triple, retaining metres and holes.

    Shape has no world frame. Its XY origin convention is FRAME_CONVENTION;
    this convention must also be used by any future experience extractor.
    """
    require(isinstance(triple, (tuple, list)) and len(triple) == 3, "Expected descriptor triple")
    row, metadata, arrays = triple
    validate_descriptor(metadata, arrays)
    require(metadata["raster_size"] == 32, "Only actual 32x32 descriptors are registered")
    features = metadata["features"]
    shape = {"schema": "p555.local_surface_shape.v1", "resolution": 32,
             "dimensions_m": [*features["extent_local_xy_m"], features["height_span_m"]],
             "area_m2": features["surface_area_m2"],
             "support_mask": arrays["contact_mask"].ravel().tolist(),
             "height_valid_mask": arrays["valid_mask"].ravel().tolist(),
             "heightmap_m": arrays["heightmap_m"].astype(float).ravel().tolist()}
    return validate_shape(shape)


def load_current_surface(facts_path, instance_id, direction_id, *, stage, invariant, body_model):
    """Make a sealed, source-bound current surface; requires explicit typing.

    body_model is an exact path/bytes/SHA artifact, not a model object. Facts
    remain queryable without active experiences; this function grants no credit.
    """
    require(stage == "stage2_contact", "No verified current typed role producer for Stage1/Stage3")
    facts_binding = artifact(facts_path)
    triples = load_descriptors(verified(facts_binding))
    choices = [t for t in triples if t[0]["instance_id"] == instance_id
               and t[1]["source_binding"]["direction"]["candidate_id"] == direction_id]
    require(len(choices) == 1, "Current owner/direction must select exactly one eligible audited descriptor")
    row, metadata, arrays = choices[0]
    shape = descriptor_to_shape(choices[0])
    validate_invariant(stage, invariant, shape)
    require(metadata["action_role"] == "sit_support" and invariant["action"] == "sit"
            and invariant["surface_role"] == "interaction_contact" and invariant["effector"] == "pelvis_glute"
            and invariant["motion_phase"] in {"terminal_contact", "contact_hold"}
            and invariant["target_semantic"] == row["category"], "Current typed action/owner/effector mismatch")
    verified(body_model)
    require(body_model["sha256"] == invariant["body_model_sha256"], "Current body identity mismatch")
    facts = read_sealed(facts_binding["path"])
    descriptor_binding = next(r for r in row["descriptors"] if read_sealed(r["path"])["entry_id"] == metadata["entry_id"])
    bounds = np.asarray(metadata["raster_bounds_local_xy_m"], dtype=float)
    rotation = np.asarray(metadata["frame"]["local_to_world_rotation"], dtype=float)
    centre = bounds.mean(axis=0)
    origin = np.asarray(metadata["frame"]["origin_world_zup_m"]) + rotation @ np.r_[centre, 0.]
    result = {"schema": "p555.current_memory_surface.v1", "stage": stage, "invariant": invariant,
        "frame_convention": FRAME_CONVENTION, "shape": shape,
        "facts": facts_binding, "descriptor": descriptor_binding, "body_model": body_model,
        "scene_id": facts["scene_id"],
        "scene_fingerprint_sha256": digest({"mesh_sha256": facts["sources"]["source_mesh"]["sha256"],
                                             "occupancy_sha256": facts["sources"]["source_occupancy"]["sha256"]}),
        "fixed_geometry": facts["sources"]["fixed_geometry"],
        "owner": {"instance_id": instance_id, "stable_surface_sha256": row["stable_surface_sha256"],
                  "source_surface_arrays": row["surface_arrays"]},
        "direction_id": direction_id, "direction_proof": metadata["direction_proof"],
        "frame": {"origin_world_zup_m": origin.tolist(), "local_to_world_rotation": rotation.tolist()},
        "facts_are_not_positive_experience": True, "positive_credit": 0}
    require(artifact(facts_path) == facts_binding, "Facts changed during current surface load")
    return _seal(result)


def validate_current_surface(context):
    require(isinstance(context, dict) and context.get("schema") == "p555.current_memory_surface.v1", "Invalid current context")
    expected = load_current_surface(verified(context["facts"]), context["owner"]["instance_id"], context["direction_id"],
        stage=context["stage"], invariant=context["invariant"], body_model=context["body_model"])
    require(context == expected, "Current context/frame/owner binding drift")
    return context


def _arrays(shape):
    return (np.asarray(shape["dimensions_m"][:2], dtype=float),
            np.asarray(shape["support_mask"], dtype=bool).reshape(32, 32),
            np.asarray(shape["heightmap_m"], dtype=float).reshape(32, 32))


def _directed_overlap(source, target):
    """Project metric cell centres with no normalization/scale fitting."""
    sd, sm, sh = _arrays(source)
    td, tm, th = _arrays(target)
    coords = np.argwhere(sm)
    xy = ((coords + .5) / 32. - .5) * sd
    indices = np.floor((xy / td + .5) * 32).astype(int)
    inside = np.all((indices >= 0) & (indices < 32), axis=1)
    safe = np.clip(indices, 0, 31)
    valid = inside & tm[safe[:, 0], safe[:, 1]]
    differences = np.abs(sh[coords[valid, 0], coords[valid, 1]] - th[safe[valid, 0], safe[valid, 1]])
    return float(valid.mean()), differences


def match_shapes(source, current):
    """Pure geometry diagnostic, not admission or authority to apply memory."""
    validate_shape(source)
    validate_shape(current)
    sd, sm, _ = _arrays(source)
    cd, cm, _ = _arrays(current)
    relative = np.maximum(sd, cd) / np.minimum(sd, cd) - 1.
    forward, diff1 = _directed_overlap(source, current)
    backward, diff2 = _directed_overlap(current, source)
    differences = np.r_[diff1, diff2]
    rmse = float(np.sqrt(np.mean(differences**2))) if len(differences) else None
    p95 = float(np.percentile(differences, 95)) if len(differences) else None
    checks = {"metric_dimensions": bool(np.all(relative <= POLICY["maximum_xy_dimension_relative_difference"] + 1e-12)),
              "metric_surface_area": max(source["area_m2"], current["area_m2"]) / min(source["area_m2"], current["area_m2"]) - 1 <= POLICY["maximum_area_relative_difference"],
              "metric_height_span": abs(source["dimensions_m"][2] - current["dimensions_m"][2]) <= POLICY["maximum_height_span_difference_m"],
              "observed_coverage": min(float(sm.mean()), float(cm.mean())) >= POLICY["minimum_observed_coverage"],
              "bidirectional_metric_overlap": min(forward, backward) >= POLICY["minimum_bidirectional_overlap"],
              "local_heights": rmse is not None and rmse <= POLICY["maximum_height_rmse_m"] and p95 <= POLICY["maximum_height_p95_m"]}
    score = None if rmse is None else float(relative.max() + (2 - forward - backward) + rmse / .02)
    return {"accepted": all(checks.values()), "checks": checks, "xy_relative_difference": relative.tolist(),
            "source_to_current_overlap": forward, "current_to_source_overlap": backward,
            "height_rmse_m": rmse, "height_p95_m": p95, "score": score,
            "metric_scale_preserved": True, "mirror_search_performed": False, "rotation_search_performed": False}


def _height_normal(shape, xy):
    dims, mask, heights = _arrays(shape)
    index = np.floor((np.asarray(xy) / dims + .5) * 32).astype(int)
    require(np.all(index >= 0) and np.all(index < 32), "Contact outside current metric bounds")
    ix, iy = index
    require(mask[ix, iy], "Contact is in an unknown/hole cell")
    lo, hi = np.maximum(index - 1, 0), np.minimum(index + 1, 31)
    positions = np.array([(x, y) for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1) if mask[x, y]])
    require(len(positions) >= 3, "Insufficient actual current samples for a normal")
    xy_metric = ((positions + .5) / 32 - .5) * dims
    design = np.column_stack((xy_metric, np.ones(len(positions))))
    require(np.linalg.matrix_rank(design) == 3, "Current normal samples are collinear")
    coefficients = np.linalg.lstsq(design, heights[positions[:, 0], positions[:, 1]], rcond=None)[0]
    normal = np.r_[-coefficients[:2], 1.]
    normal /= np.linalg.norm(normal)
    return float(heights[ix, iy]), normal


def project_candidate(record, context, sdf_check):
    """Geometry-only candidate check; retrieve_current enforces active memory.

    sdf_check(projection) returns fixed_geometry_sha256, owner_instance_id,
    projection_sha256, non_target_sdf_m (contact/root), outside (two booleans),
    and accepted (strict bool). It must use current bound scene geometry.
    """
    validate_record(record)
    require(context["frame_convention"] == FRAME_CONVENTION and context["stage"] == "stage2_contact", "Unsupported projection frame/role")
    require(record["stage"] == context["stage"] and record["invariant"] == context["invariant"], "Hard typed invariant mismatch")
    require(record["scope"] != "scene_local" or record["source_scene_fingerprint_sha256"] == context["scene_fingerprint_sha256"], "Scene-local source mismatch")
    match = match_shapes(record["shape"], context["shape"])
    require(match["accepted"], "Metric shape/height/coverage veto: " + str(match["checks"]))
    contact = np.asarray(record["payload"]["contact_point_local_xyz_m"], dtype=float)
    source_height, _ = _height_normal(record["shape"], contact[:2])
    require(abs(contact[2] - source_height) <= POLICY["maximum_contact_height_difference_m"], "Source local contact differs from observed support")
    height, normal = _height_normal(context["shape"], contact[:2])
    require(abs(contact[2] - height) <= POLICY["maximum_contact_height_difference_m"], "Current contact height veto")
    stored_normal = np.asarray(record["payload"]["surface_normal_local_xyz"], dtype=float)
    require(float(stored_normal @ normal) >= math.cos(math.radians(POLICY["maximum_normal_angle_degrees"])), "Current measured surface normal veto")
    rotation = np.asarray(context["frame"]["local_to_world_rotation"], dtype=float)
    origin = np.asarray(context["frame"]["origin_world_zup_m"], dtype=float)
    contact_world = origin + rotation @ np.r_[contact[:2], height]
    root_world = contact_world + rotation @ np.asarray(record["payload"]["root_offset_local_xyz_m"])
    facing_world = rotation @ np.r_[record["payload"]["facing_local_xy"], 0.]
    projection = {"schema": "p555.projected_local_contact.v1", "record_id": record["record_id"],
        "scope": record["scope"], "stage": record["stage"], "context_sha256": context[SCHEMA_HASH_FIELD],
        "frame_convention": FRAME_CONVENTION, "fixed_geometry": context["fixed_geometry"], "owner": context["owner"],
        "contact_world_xyz_m": contact_world.tolist(), "root_world_xyz_m": root_world.tolist(),
        "surface_normal_world_xyz": (rotation @ normal).tolist(), "facing_world_xy": facing_world[:2].tolist(),
        "contact_phase": record["payload"]["contact_phase"], "shape_match": match,
        "pose_or_motion_scaled": False, "source_world_transform_read": False,
        "current_full_body_physics_validation_still_required": True}
    require(callable(sdf_check), "Current bound SDF callback is required")
    evidence = sdf_check(_copy(projection))
    require(isinstance(evidence, dict) and evidence.get("projection_sha256") == digest(projection)
            and evidence.get("fixed_geometry_sha256") == context["fixed_geometry"]["sha256"]
            and evidence.get("owner_instance_id") == context["owner"]["instance_id"], "Current SDF/owner/projection binding mismatch")
    require(type(evidence.get("accepted")) is bool and evidence["accepted"], "Current SDF veto")
    values, outside = evidence.get("non_target_sdf_m"), evidence.get("outside")
    require(isinstance(values, list) and len(values) == 2 and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values), "Current non-target SDF invalid/penetrating")
    require(isinstance(outside, list) and len(outside) == 2 and all(x is False for x in outside), "Current SDF outside domain")
    return _seal({**projection, "sdf_point_evidence": evidence, "sdf_scope": "contact_and_root_points_not_full_body"})


def retrieve_current(store, cycle, context, *, sdf_check, scopes=SCOPES):
    """Active-only, live-barrier read; geometry facts are available even at M0."""
    require(isinstance(store, MemoryStore), "A verified MemoryStore, not caller candidate flags, is required")
    validate_current_surface(context)
    require(isinstance(scopes, (tuple, list)) and scopes and len(set(scopes)) == len(scopes)
            and set(scopes) <= set(SCOPES), "Invalid exact scope filter")
    binding = cycle["query_binding"]
    require(binding["scene_fingerprint_sha256"] == context["scene_fingerprint_sha256"]
            and binding["scene_id"] == context["scene_id"], "Current cycle/scene identity drift")
    query = store.query_active(cycle, context["stage"], context["invariant"])
    selected, rejected = [], []
    for wrapper in query["records"]:
        record = wrapper["record"]
        if record["scope"] not in scopes:
            continue
        try:
            projection = project_candidate(record, context, sdf_check)
            selected.append({"record_id": record["record_id"], "group_key": wrapper["group_key"],
                             "representative_attempt_id": wrapper["representative_attempt_id"], "projection": projection})
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"record_id": record["record_id"], "reason": str(error)})
    # Re-check live critical evidence after callback work; an old N cannot revive
    # a record quarantined while geometry/SDF validation was in progress.
    validate_current_surface(context)
    latest = store.query_active(cycle, context["stage"], context["invariant"])
    live = {w["record"]["record_id"] for w in latest["records"]}
    for row in selected:
        if row["record_id"] not in live:
            rejected.append({"record_id": row["record_id"], "reason": "live_critical_barrier_changed_during_projection"})
    selected = [r for r in selected if r["record_id"] in live]
    selected.sort(key=lambda r: (r["projection"]["shape_match"]["score"], r["record_id"]))
    return _seal({"schema": "p555.local_memory_retrieval.v1", "purpose": store.purpose,
        "current_surface": context, "active_query": latest, "scope_filter": list(scopes),
        "records": selected, "rejected": rejected, "m0_bypass": not selected,
        "m0_reason": None if selected else "no_active_experiences" if not latest["records"] else "scope_or_current_geometry_veto",
        "scene_facts_available": True, "facts_positive_credit": 0,
        "frame_convention": FRAME_CONVENTION, "policy": POLICY,
        "full_body_refine_and_final_physics_gates_still_required": True,
        "revalidate_live_barrier_at_consumption": True})
