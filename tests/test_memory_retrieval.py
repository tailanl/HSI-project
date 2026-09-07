"""Metric projection and authority integration using isolated fixtures only."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

from hsi.common.artifacts import artifact, digest, write_once
from hsi.memory.schema import make_record, shape_family, validate_shape
from hsi.memory.store import MemoryStore
from hsi.memory.authority import GATES, admit_episode
from hsi.memory.facts import build_scene_facts, load_descriptors
from hsi.memory.matching import (FRAME_CONVENTION, descriptor_to_shape, load_current_surface,
    match_shapes, project_candidate, retrieve_current, validate_current_surface)
from test_memory_facts import fixture_scene


def current(tmp_path, *, directions=1):
    final = fixture_scene(tmp_path, directions=directions)
    facts = build_scene_facts(final, tmp_path / "facts")
    triple = load_descriptors(facts)[0]
    shape = descriptor_to_shape(triple)
    body = tmp_path / "test_body_identity.bin"
    body.write_bytes(b"synthetic identity, no real body model or positive credit")
    invariant = {"action": "sit", "target_semantic": "chair", "surface_role": "interaction_contact",
        "effector": "pelvis_glute", "motion_phase": "terminal_contact", "body_model_sha256": artifact(body)["sha256"],
        "body_shape_bin": "neutral", "shape_family": shape_family(shape)}
    context = load_current_surface(facts, "SEAT", "DIRECTION_0", stage="stage2_contact",
        invariant=invariant, body_model=artifact(body))
    return context, triple


def prototype(context, *, scope="invariant", scene=None, payload_changes=None):
    payload = {"contact_point_local_xyz_m": [0., 0., 0.], "surface_normal_local_xyz": [0., 0., 1.],
        "root_offset_local_xyz_m": [0., 0., .3], "facing_local_xy": [1., 0.], "contact_phase": 1.}
    payload.update(payload_changes or {})
    return make_record(stage="stage2_contact", scope=scope,
        source_scene_fingerprint_sha256=scene or context["scene_fingerprint_sha256"],
        invariant=context["invariant"], shape=context["shape"], payload=payload)


def good_sdf(projection):
    return {"accepted": True, "projection_sha256": digest(projection),
        "fixed_geometry_sha256": projection["fixed_geometry"]["sha256"],
        "owner_instance_id": projection["owner"]["instance_id"],
        "non_target_sdf_m": [.2, .3], "outside": [False, False]}


def source_episode(tmp_path, context, index, *, success=True, affected=()):
    # Four synthetic families, with final family matching the actual fixture facts.
    family = index % 4
    fingerprint = context["scene_fingerprint_sha256"] if family == 3 else digest({"fixture_scene": family})
    scene_id = context["scene_id"] if family == 3 else "FIXTURE%d" % family
    checks = {k: True for k in GATES}
    if not success:
        checks["collision_free"] = False
    path = tmp_path / ("episode_%d.json" % index)
    write_once(path, {"schema": "p555.fixture_episode_evidence.v1", "namespace": "test_fixture",
        "query_identity": {"scene_id": scene_id, "scene_fingerprint": fingerprint, "scene_family": "family%d" % family,
                           "instruction": "sit fixture %d" % index, "start_world_xy_m": [0., 0.]},
        "attempt_nonce": "seed%d" % index, "checks": checks,
        "critical_memory_failure": bool(affected), "affected_record_ids": list(affected)})
    request = tmp_path / ("request_%d.json" % index)
    write_once(request, {"schema": "p555.admission_request.v1", "mode": "test_fixture", "source_receipt": artifact(path)})
    admission = admit_episode(request, purpose="test_fixture")
    binding = {"query_id": admission["query_id"], "episode_identity": admission["episode_id"],
        "scene_fingerprint_sha256": fingerprint, "scene_family": admission["scene_family"], "scene_id": scene_id}
    return request, binding


def active_store(tmp_path, context):
    store = MemoryStore(tmp_path / "fixture_memory", purpose="test_fixture")
    cycles = []
    for i in range(8):
        request, binding = source_episode(tmp_path, context, i)
        cycle = store.open_cycle(binding)
        records = [prototype(context, scope=scope, scene=binding["scene_fingerprint_sha256"])
                   for scope in ("invariant", "scene_local")]
        store.record_attempt(cycle, request, records)
        cycles.append(cycle)
    store.commit_cycle(cycles[0], additional_cycles=cycles[1:])
    return store, store.open_cycle(binding)


def test_descriptor_conversion_keeps_metric_masks_and_canonical_frame(tmp_path):
    context, triple = current(tmp_path)
    shape = descriptor_to_shape(triple)
    assert shape["dimensions_m"] == [1., 1., 0.]
    assert shape["area_m2"] == 1. and len(shape["heightmap_m"]) == 1024
    assert all(type(v) is bool for v in shape["support_mask"])
    assert context["frame_convention"] == FRAME_CONVENTION
    assert context["frame"]["origin_world_zup_m"] == [.5, .5, .4]
    assert context["positive_credit"] == 0


@pytest.mark.parametrize("change", ["scale", "area", "sparse", "height", "mirror"])
def test_metric_match_rejects_size_coverage_height_and_unsearched_mirror(tmp_path, change):
    context, _ = current(tmp_path)
    source, target = copy.deepcopy(context["shape"]), copy.deepcopy(context["shape"])
    if change == "scale":
        target["dimensions_m"][0] *= 1.151
    elif change == "area":
        target["area_m2"] *= 1.351
    elif change == "height":
        target["heightmap_m"] = [.05] * 1024
    elif change == "sparse":
        target["support_mask"] = target["height_valid_mask"] = [i < 100 for i in range(1024)]
    else:
        source["support_mask"] = source["height_valid_mask"] = [i < 512 for i in range(1024)]
        target["support_mask"] = target["height_valid_mask"] = [i >= 512 for i in range(1024)]
    result = match_shapes(source, target)
    assert result["accepted"] is False
    assert result["metric_scale_preserved"] is True
    assert result["mirror_search_performed"] is result["rotation_search_performed"] is False


def test_fifteen_percent_bound_and_no_pose_scale(tmp_path):
    context, _ = current(tmp_path)
    other = copy.deepcopy(context["shape"])
    other["dimensions_m"][:2] = [1.15, 1.15]
    assert match_shapes(context["shape"], other)["checks"]["metric_dimensions"] is True
    projected = project_candidate(prototype(context), context, good_sdf)
    assert projected["contact_world_xyz_m"] == [.5, .5, .4]
    assert np.allclose(projected["root_world_xyz_m"], [.5, .5, .7])
    assert projected["pose_or_motion_scaled"] is projected["source_world_transform_read"] is False


def test_only_explicit_current_audited_direction_controls_projection(tmp_path):
    context, _ = current(tmp_path, directions=2)
    reverse = load_current_surface(context["facts"]["path"], "SEAT", "DIRECTION_1", stage="stage2_contact",
        invariant=context["invariant"], body_model=context["body_model"])
    record = prototype(context, payload_changes={"contact_point_local_xyz_m": [.1, 0., 0.]})
    a, b = project_candidate(record, context, good_sdf), project_candidate(record, reverse, good_sdf)
    assert np.allclose(a["contact_world_xyz_m"], [.6, .5, .4])
    assert np.allclose(b["contact_world_xyz_m"], [.4, .5, .4])
    assert a["facing_world_xy"] == [1., 0.] and b["facing_world_xy"] == [-1., 0.]


@pytest.mark.parametrize("kind", ["owner", "geometry", "projection", "negative", "outside", "false", "boolean_sdf"])
def test_sdf_is_bound_and_fail_closed(tmp_path, kind):
    context, _ = current(tmp_path)
    def failed(projection):
        evidence = good_sdf(projection)
        if kind == "owner": evidence["owner_instance_id"] = "OTHER"
        if kind == "geometry": evidence["fixed_geometry_sha256"] = "f" * 64
        if kind == "projection": evidence["projection_sha256"] = "f" * 64
        if kind == "negative": evidence["non_target_sdf_m"][1] = -.001
        if kind == "outside": evidence["outside"][0] = True
        if kind == "false": evidence["accepted"] = False
        if kind == "boolean_sdf": evidence["non_target_sdf_m"][0] = True
        return evidence
    with pytest.raises(ValueError):
        project_candidate(prototype(context), context, failed)


def test_contact_normal_hole_and_hard_key_veto(tmp_path):
    context, _ = current(tmp_path)
    tilted = prototype(context, payload_changes={"surface_normal_local_xyz": [math_sqrt_3_over_2(), 0., .5]})
    with pytest.raises(ValueError, match="normal veto"):
        project_candidate(tilted, context, good_sdf)
    outside = prototype(context, payload_changes={"contact_point_local_xyz_m": [.6, 0., 0.]})
    with pytest.raises(ValueError, match="bounds"):
        project_candidate(outside, context, good_sdf)
    other = copy.deepcopy(context)
    other["invariant"]["body_shape_bin"] = "different"
    with pytest.raises(ValueError, match="Hard typed"):
        project_candidate(prototype(context), other, good_sdf)
    hole = copy.deepcopy(context)
    hole["shape"]["support_mask"][16 * 32 + 16] = False
    hole["shape"]["height_valid_mask"][16 * 32 + 16] = False
    with pytest.raises(ValueError, match="unknown/hole"):
        project_candidate(prototype(context), hole, good_sdf)


def math_sqrt_3_over_2():
    return float(np.sqrt(3.) / 2.)


@pytest.mark.parametrize("stage", ["stage1_route", "stage3_guidance"])
def test_upward_seat_never_implies_route_or_guidance_role(tmp_path, stage):
    context, _ = current(tmp_path)
    with pytest.raises(ValueError, match="No verified current typed role"):
        load_current_surface(context["facts"]["path"], "SEAT", "DIRECTION_0", stage=stage,
            invariant=context["invariant"], body_model=context["body_model"])


def test_current_context_frame_and_owner_drift_rejected(tmp_path):
    context, _ = current(tmp_path)
    context["frame"]["origin_world_zup_m"][0] += .1
    with pytest.raises(ValueError, match="context/frame/owner"):
        validate_current_surface(context)


def test_action_and_body_cannot_be_inferred_from_upward_normal(tmp_path):
    context, _ = current(tmp_path)
    wrong = dict(context["invariant"], target_semantic="bench")
    with pytest.raises(ValueError, match="typed action/owner"):
        load_current_surface(context["facts"]["path"], "SEAT", "DIRECTION_0", stage="stage2_contact",
            invariant=wrong, body_model=context["body_model"])
    wrong = dict(context["invariant"], body_model_sha256="a" * 64)
    with pytest.raises(ValueError, match="body identity"):
        load_current_surface(context["facts"]["path"], "SEAT", "DIRECTION_0", stage="stage2_contact",
            invariant=wrong, body_model=context["body_model"])


def test_zero_experience_m0_still_has_geometry_facts(tmp_path):
    context, _ = current(tmp_path)
    request, binding = source_episode(tmp_path, context, 3)
    store = MemoryStore(tmp_path / "empty_fixture", purpose="test_fixture")
    result = retrieve_current(store, store.open_cycle(binding), context, sdf_check=None)
    assert result["m0_bypass"] and result["m0_reason"] == "no_active_experiences"
    assert result["scene_facts_available"] and result["facts_positive_credit"] == 0
    assert store.recover()["positive_episode_count"] == 0


def test_active_wrapper_scope_filter_and_live_callback_quarantine(tmp_path):
    context, _ = current(tmp_path)
    store, cycle = active_store(tmp_path, context)
    both = retrieve_current(store, cycle, context, sdf_check=good_sdf)
    assert len(both["records"]) == 2 and not both["m0_bypass"]
    local = retrieve_current(store, cycle, context, sdf_check=good_sdf, scopes=["scene_local"])
    assert len(local["records"]) == 1 and local["records"][0]["projection"]["scope"] == "scene_local"
    rid = both["records"][0]["record_id"]
    request, binding = source_episode(tmp_path, context, 8, success=False, affected=[rid])
    failed_cycle = store.open_cycle(binding)
    called = []
    def quarantine_during_callback(projection):
        if not called:
            store.record_attempt(failed_cycle, request)
            called.append(True)
        return good_sdf(projection)
    blocked = retrieve_current(store, cycle, context, sdf_check=quarantine_during_callback)
    assert blocked["records"] == [] and blocked["m0_bypass"]
    assert any("live_critical_barrier" in r["reason"] for r in blocked["rejected"])
    assert store.recover()["real_positive_credit"] == 0
