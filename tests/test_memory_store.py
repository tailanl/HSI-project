"""CPU/temp-store tests using the real, purpose-limited admission entry."""
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys

import pytest


from hsi.memory import store as storage
from hsi.memory.store import MemoryStore, group_key
from hsi.common.artifacts import artifact, digest, read_sealed, write_once, SCHEMA_HASH_FIELD
from hsi.memory.schema import make_record, shape_family
from hsi.memory.authority import admit_episode, GATES, ProductionAdmissionUnavailable


def fixture_request(tmp_path, index=0, *, success=True, scene=None,
                    episode=None, affected=(), purpose="test_fixture"):
    scene = index % 4 if scene is None else scene
    query = {"scene_id": "scene%d" % scene,
             "scene_fingerprint": digest({"mesh_sha256": digest(scene), "occupancy_sha256": digest([scene])}),
             "scene_family": "family%d" % scene, "instruction": "sit test %s" % (index if episode is None else episode),
             "start_world_xy_m": [0., 0.]}
    checks = {name: True for name in GATES}
    if not success:
        checks["collision_free"] = False
    source = tmp_path / ("evidence_%d.json" % index)
    write_once(source, {"schema": "p555.fixture_episode_evidence.v1", "namespace": "test_fixture",
        "query_identity": query, "attempt_nonce": "nonce_%d" % index, "checks": checks,
        "critical_memory_failure": bool(affected), "affected_record_ids": list(affected)})
    request = tmp_path / ("request_%d.json" % index)
    write_once(request, {"schema": "p555.admission_request.v1", "mode": "test_fixture",
                         "source_receipt": artifact(source)})
    admission = admit_episode(request, purpose="test_fixture")
    binding = {"query_id": admission["query_id"], "episode_identity": admission["episode_id"],
               "scene_fingerprint_sha256": admission["scene_fingerprint"],
               "scene_family": admission["scene_family"], "scene_id": admission["scene_id"]}
    return request, binding, admission


def record(binding, *, scope="invariant", stage="stage2_contact", offset=0.):
    mask = [True] * 1024
    shape = {"schema": "p555.local_surface_shape.v1", "resolution": 32,
             "dimensions_m": [.5, .5, 0.], "area_m2": .25,
             "support_mask": mask, "height_valid_mask": list(mask), "heightmap_m": [0.] * 1024}
    roles = {"stage1_route": ("free_space_approach", "root", "approach"),
             "stage2_contact": ("interaction_contact", "pelvis_glute", "terminal_contact"),
             "stage3_guidance": ("walkable_support", "feet", "locomotion")}
    role, effector, phase = roles[stage]
    invariant = {"action": "sit", "target_semantic": "chair", "surface_role": role,
                 "effector": effector, "motion_phase": phase,
                 "body_model_sha256": "b" * 64, "body_shape_bin": "neutral",
                 "shape_family": shape_family(shape)}
    payloads = {"stage1_route": {"approach_offset_local_xy_m": [0., 0.],
                    "approach_direction_local_xy": [1., 0.], "clearance_target_m": .3},
                "stage2_contact": {"contact_point_local_xyz_m": [offset, 0., 0.],
                    "surface_normal_local_xyz": [0., 0., 1.], "root_offset_local_xyz_m": [0., 0., .3],
                    "facing_local_xy": [1., 0.], "contact_phase": 1.},
                "stage3_guidance": {"sdf_weight_multiplier": 1.1,
                    "contact_weight_multiplier": 1., "stance_weight_multiplier": 1.1}}
    return make_record(stage=stage, scope=scope,
        source_scene_fingerprint_sha256=binding["scene_fingerprint_sha256"],
        invariant=invariant, shape=shape, payload=payloads[stage])


def add_success(store, tmp_path, index, *, mirrored=False, scene=None, episode=None):
    request, binding, admission = fixture_request(tmp_path, index, scene=scene, episode=episode)
    cycle = store.open_cycle(binding)
    records = [record(binding)]
    if mirrored:
        records.append(record(binding, scope="scene_local"))
    attempt = store.record_attempt(cycle, request, records)
    commit = store.commit_cycle(cycle)
    return cycle, records, attempt, commit, binding


def positive_bytes(store):
    return {str(p.relative_to(store.root)): p.read_bytes()
            for p in [store.root / "CURRENT.json", *sorted((store.root / "snapshots").glob("*.json"))]}


def test_genesis_fixture_is_never_real_credit(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    state = store.recover()
    assert state["generation"] == state["positive_episode_count"] == state["real_positive_credit"] == 0
    _, binding, _ = fixture_request(tmp_path)
    cycle = store.open_cycle(binding)
    assert store.query_active(cycle, "stage2_contact")["m0_bypass"] is True
    assert store.open_cycle(binding) == cycle


def test_failure_journal_does_not_change_positive_bytes_or_furniture_state(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    request, binding, _ = fixture_request(tmp_path, success=False)
    cycle = store.open_cycle(binding)
    before = positive_bytes(store)
    event = store.record_attempt(cycle, request, [record(binding)])
    store.commit_cycle(cycle)
    assert event["payload"]["admission"]["critical_memory_failure"] is False
    assert positive_bytes(store) == before
    query = store.query_active(cycle, "stage2_contact")
    assert query["records"] == query["groups"] == []
    assert store.recover()["journal_length"] == 2


def test_first_success_candidate_frozen_cycle_and_recovery(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    cycle, records, _, commit, binding = add_success(store, tmp_path, 0)
    assert store.query_active(cycle, "stage2_contact")["groups"] == []
    new_cycle = store.open_cycle(binding)
    query = store.query_active(new_cycle, "stage2_contact")
    assert query["groups"][0]["status"] == "candidate"
    assert query["records"] == []
    assert store.commit_cycle(cycle) == commit
    restored = MemoryStore(store.root, purpose="test_fixture")
    assert restored.recover()["positive_episode_count"] == 1
    assert restored.query_active(new_cycle, "stage2_contact") == query


def test_multiseed_episode_credit_and_scope_mirrors_count_once(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    request, binding, _ = fixture_request(tmp_path, 0, scene=0, episode="same")
    second, binding2, _ = fixture_request(tmp_path, 1, scene=0, episode="same")
    assert binding == binding2
    cycle = store.open_cycle(binding)
    records = [record(binding), record(binding, scope="scene_local")]
    first = store.record_attempt(cycle, request, records)
    assert store.record_attempt(cycle, request, records) == first
    store.record_attempt(cycle, second, records)
    store.commit_cycle(cycle)
    assert store.recover()["positive_episode_count"] == 1
    new_cycle = store.open_cycle(binding)
    assert {r["positive_episode_count"] for r in store.query_active(new_cycle, "stage2_contact")["groups"]} == {1}
    before = positive_bytes(store)
    assert store.record_attempt(new_cycle, request, records) == first
    store.commit_cycle(new_cycle)
    assert positive_bytes(store) == before


def test_replay_changed_prototype_rejected(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    request, binding, _ = fixture_request(tmp_path)
    cycle = store.open_cycle(binding)
    store.record_attempt(cycle, request, [record(binding)])
    with pytest.raises(ValueError, match="replay conflicts"):
        store.record_attempt(cycle, request, [record(binding, offset=.05)])


def test_noop_commit_allows_fresh_seed_same_query_same_snapshot(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    first, binding, _ = fixture_request(tmp_path, 0, scene=0, episode="same", success=False)
    second, binding2, _ = fixture_request(tmp_path, 1, scene=0, episode="same", success=False)
    cycle = store.open_cycle(binding)
    store.record_attempt(cycle, first)
    store.commit_cycle(cycle)
    retry = store.open_cycle(binding2)
    assert retry["cycle_id"] != cycle["cycle_id"] and retry["generation"] == cycle["generation"] == 0
    assert retry["cycle_ordinal"] == 1
    store.record_attempt(retry, second)
    store.commit_cycle(retry)
    assert store.recover()["journal_length"] == 4


def test_batch_multiple_queries_one_atomic_generation(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    cycles = []
    for index in range(4):
        request, binding, _ = fixture_request(tmp_path, index)
        cycle = store.open_cycle(binding)
        store.record_attempt(cycle, request, [record(binding)])
        cycles.append(cycle)
    commit = store.commit_cycle(cycles[0], additional_cycles=cycles[1:])
    assert commit["payload"]["snapshot"]["generation"] == 1
    assert store.recover()["positive_episode_count"] == 4
    assert all(store.query_active(c, "stage2_contact")["groups"] == [] for c in cycles)
    assert store.commit_cycle(cycles[2]) == commit
    assert store.commit_cycle(cycles[0], additional_cycles=cycles[1:]) == commit


def test_eight_successes_four_families_promote_not_same_scene_only(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    for index in range(8):
        *_, binding = add_success(store, tmp_path, index, mirrored=True)
    cycle = store.open_cycle(binding)
    query = store.query_active(cycle, "stage2_contact")
    assert len(query["records"]) == 2  # active invariant + this-scene observed child
    invariant = next(r for r in query["records"] if r["record"]["scope"] == "invariant")
    local = next(r for r in query["records"] if r["record"]["scope"] == "scene_local")
    assert invariant["scene_family_count"] == 4 and invariant["positive_episode_count"] == 8
    assert invariant["beta_lower_05"] > .7
    assert local["active_basis"] == "inherited_current_active_invariant"
    assert all(r["record"]["stage"] == "stage2_contact" for r in query["records"])
    assert store.query_active(cycle, "stage3_guidance")["records"] == []


def test_single_scene_cannot_self_promote(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    for index in range(9):
        *_, binding = add_success(store, tmp_path, index, mirrored=True, scene=0)
    assert store.query_active(store.open_cycle(binding), "stage2_contact")["records"] == []


def test_live_critical_barrier_prevents_old_snapshot_revival(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    prior_ids = set()
    for index in range(8):
        _, records, attempt, _, binding = add_success(store, tmp_path, index)
        prior_ids.add(attempt["payload"]["admission"]["attempt_id"])
    reader = store.open_cycle(binding)
    assert store.query_active(reader, "stage2_contact")["records"]
    request, failed_binding, _ = fixture_request(tmp_path, 8, success=False, affected=[records[0]["record_id"]])
    failure_cycle = store.open_cycle(failed_binding)
    before = positive_bytes(store)
    store.record_attempt(failure_cycle, request)
    assert positive_bytes(store) == before
    assert store.query_active(reader, "stage2_contact")["records"] == []  # before failure commit
    store.commit_cycle(failure_cycle)
    for index in range(9, 23):
        *_, binding = add_success(store, tmp_path, index)
    latest = store.query_active(store.open_cycle(binding), "stage2_contact")
    assert latest["records"] and latest["records"][0]["representative_attempt_id"] not in prior_ids
    assert store.query_active(reader, "stage2_contact")["records"] == []


def test_unknown_critical_id_rejected_without_append(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    request, binding, _ = fixture_request(tmp_path, success=False, affected=["a" * 64])
    cycle = store.open_cycle(binding)
    before = store.recover()
    with pytest.raises(ValueError, match="Unknown critical"):
        store.record_attempt(cycle, request)
    assert store.recover() == before


def test_new_record_critical_blocks_old_snapshot_parent_and_local_children(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    cycles, bindings = [], []
    for index in range(8):
        request, binding, _ = fixture_request(tmp_path, index)
        cycle = store.open_cycle(binding)
        store.record_attempt(cycle, request, [record(binding), record(binding, scope="scene_local")])
        cycles.append(cycle)
        bindings.append(binding)
    store.commit_cycle(cycles[0], additional_cycles=cycles[1:])
    reader = store.open_cycle(bindings[3])
    assert len(store.query_active(reader, "stage2_contact")["records"]) == 2
    # Newly observed local payload has a new ID, absent from reader's snapshot.
    request, binding, _ = fixture_request(tmp_path, 8, scene=0)
    cycle = store.open_cycle(binding)
    new_record = record(binding, scope="scene_local", offset=.1)
    store.record_attempt(cycle, request, [record(binding, offset=.1), new_record])
    store.commit_cycle(cycle)
    request, binding, _ = fixture_request(tmp_path, 9, success=False, affected=[new_record["record_id"]])
    failure_cycle = store.open_cycle(binding)
    store.record_attempt(failure_cycle, request)
    blocked = store.query_active(reader, "stage2_contact")
    assert blocked["records"] == []
    assert all(g["status"] == "quarantined" for g in blocked["groups"])
    store.commit_cycle(failure_cycle)
    # Revalidate parent with other scenes. Scene 3 has no new local evidence.
    cycles = []
    for index in range(10, 18):
        request, binding, _ = fixture_request(tmp_path, index, scene=index % 2)
        cycle = store.open_cycle(binding)
        store.record_attempt(cycle, request, [record(binding), record(binding, scope="scene_local")])
        cycles.append(cycle)
    store.commit_cycle(cycles[0], additional_cycles=cycles[1:])
    current = store.query_active(store.open_cycle(bindings[3]), "stage2_contact")
    assert {r["record"]["scope"] for r in current["records"]} == {"invariant"}
    assert store.query_active(reader, "stage2_contact")["records"] == []
    # One actual postcritical observation in scene 3 may inherit the active parent.
    cycle, _, attempt, _, binding = add_success(store, tmp_path, 18, mirrored=True, scene=3)
    restored = store.query_active(store.open_cycle(binding), "stage2_contact")
    local = next(r for r in restored["records"] if r["record"]["scope"] == "scene_local")
    assert local["representative_attempt_id"] == attempt["payload"]["admission"]["attempt_id"]
    assert local["active_basis"] == "inherited_current_active_invariant"


def test_production_rejects_fixture_evidence_and_fixture_store(tmp_path):
    request, binding, _ = fixture_request(tmp_path)
    store = MemoryStore(tmp_path / "production")
    cycle = store.open_cycle(binding)
    before = store.recover()
    with pytest.raises(ProductionAdmissionUnavailable, match="no production credit"):
        store.record_attempt(cycle, request, [record(binding)])
    assert store.recover() == before
    with pytest.raises(ValueError, match="mixing"):
        MemoryStore(store.root, purpose="test_fixture")


def test_record_type_checker_rejects_world_pose_fields(tmp_path):
    request, binding, _ = fixture_request(tmp_path)
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    cycle = store.open_cycle(binding)
    invalid = record(binding)
    invalid["payload"]["world_pose"] = [1., 2., 3.]
    with pytest.raises(ValueError, match="Payload"):
        store.record_attempt(cycle, request, [invalid])
    assert store.recover()["journal_length"] == 0


def test_record_cannot_claim_another_source_scene(tmp_path):
    request, binding, _ = fixture_request(tmp_path)
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    wrong = dict(binding, scene_fingerprint_sha256="f" * 64)
    with pytest.raises(ValueError, match="source scene differs"):
        store.record_attempt(store.open_cycle(binding), request, [record(wrong)])
    assert store.recover()["journal_length"] == 0


def test_changed_source_before_commit_never_publishes(tmp_path):
    request, binding, _ = fixture_request(tmp_path)
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    cycle = store.open_cycle(binding)
    store.record_attempt(cycle, request, [record(binding)])
    source = Path(read_sealed(request)["source_receipt"]["path"])
    source.write_text(source.read_text() + " ")
    before = positive_bytes(store)
    with pytest.raises(ValueError, match="binding drift"):
        store.commit_cycle(cycle)
    assert positive_bytes(store) == before


def _commit_worker(root, cycle):
    try:
        return MemoryStore(root, purpose="test_fixture").commit_cycle(cycle)["kind"]
    except ValueError as error:
        return str(error)


def test_concurrent_commit_cas(tmp_path):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    cycles = []
    for index in range(2):
        request, binding, _ = fixture_request(tmp_path, index)
        cycle = store.open_cycle(binding)
        store.record_attempt(cycle, request, [record(binding)])
        cycles.append(cycle)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork")) as pool:
        results = list(pool.map(_commit_worker, [store.root, store.root], cycles))
    assert results.count("commit") == 1
    assert sum("CAS conflict" in r for r in results) == 1
    assert store.recover()["positive_episode_count"] == 1


@pytest.mark.parametrize("failure_point", ["snapshot", "pointer"])
def test_recover_committed_transaction_after_interruption(tmp_path, monkeypatch, failure_point):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    request, binding, _ = fixture_request(tmp_path)
    cycle = store.open_cycle(binding)
    store.record_attempt(cycle, request, [record(binding)])
    with monkeypatch.context() as patch:
        if failure_point == "pointer":
            patch.setattr(storage, "_replace_pointer", lambda *a: (_ for _ in ()).throw(RuntimeError("crash")))
        else:
            original = storage.write_once
            def interrupted(path, value, **kwargs):
                if path.parent.name == "snapshots" and path.name == "00000001.json":
                    raise RuntimeError("crash")
                return original(path, value, **kwargs)
            patch.setattr(storage, "write_once", interrupted)
        with pytest.raises(RuntimeError, match="crash"):
            store.commit_cycle(cycle)
    recovered = MemoryStore(store.root, purpose="test_fixture")
    assert recovered.recover()["generation"] == 1
    assert recovered.recover()["positive_episode_count"] == 1
    assert recovered.commit_cycle(cycle)["kind"] == "commit"


@pytest.mark.parametrize("target", ["journal", "snapshot", "current", "cycle"])
def test_tampered_receipt_fails_closed(tmp_path, target):
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    cycle, _, _, _, _ = add_success(store, tmp_path, 0)
    paths = {"journal": store.root / "journal/00000000.json",
             "snapshot": store.root / "snapshots/00000001.json", "current": store.root / "CURRENT.json",
             "cycle": store.root / "cycles" / (cycle["cycle_id"] + ".json")}
    path = paths[target]
    data = json.loads(path.read_bytes())
    data["tampered"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        MemoryStore(store.root, purpose="test_fixture")


def test_symlink_store_and_query_identity_drift_rejected(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    link = tmp_path / "link"
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        MemoryStore(link)
    request, binding, _ = fixture_request(tmp_path)
    store = MemoryStore(tmp_path / "fixture", purpose="test_fixture")
    binding["episode_identity"] = "c" * 64
    with pytest.raises(ValueError, match="episode differs"):
        store.record_attempt(store.open_cycle(binding), request)
