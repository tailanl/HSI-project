"""P555 local, success-only snapshots with a live negative-evidence barrier.

The evidence authority is episode_evidence.admit_episode, never a caller gate.
Production and synthetic fixture stores cannot cross-read. One episode earns
at most one credit, even across seeds and both scene_local/invariant mirrors.
All reads in a cycle use snapshot N; a newer verified critical failure is still
visible immediately. A scene_local group cannot independently promote: it is
queryable only as an observed subset of a currently active invariant parent.
Critical attribution isolates the invariant parent and all scene_local children.
The parent needs postcritical evidence in two families; a local child must also
have its own scene's postcritical observation, so old local representatives do
not revive when another scene revalidates the parent.

This is integrity in a trusted local workspace, not protection against a same
UID attacker rewriting code, every receipt, and the entire store. No model or
GPU is imported here. Persistence uses immutable journal transactions followed
by immutable snapshots and an atomic CURRENT pointer. Recovery replays complete
transactions; an incomplete/unsealed or inconsistent transaction fails closed.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid

from memory_common import (SCHEMA_HASH_FIELD, artifact, canonical_bytes, digest,
                           fsync_directory, read_sealed, require, verified, write_once)

STORE_SCHEMA = "p555.memory_store.v1"
SNAPSHOT_SCHEMA = "p555.positive_memory_snapshot.v1"
CYCLE_SCHEMA = "p555.frozen_memory_cycle.v1"
EVENT_SCHEMA = "p555.memory_journal_event.v1"
POLICY = {"minimum_successes": 8, "minimum_scene_families": 4,
          "beta_prior_alpha": 1, "beta_prior_beta": 1,
          "lower_quantile": .05, "minimum_lower_bound": .7,
          "postcritical_successes": 2, "postcritical_scene_families": 2,
          "scene_local_active_basis": "inherited_current_active_invariant",
          "episode_credit_limit": 1}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_STAGES = {"stage1_route", "stage2_contact", "stage3_guidance"}


def _clone(value):
    return json.loads(canonical_bytes(value))


def _body(value):
    return {k: v for k, v in value.items() if k != SCHEMA_HASH_FIELD}


def _seal(value):
    value = _clone(_body(value))
    value[SCHEMA_HASH_FIELD] = digest(value)
    return value


def _checksum(value, name):
    require(isinstance(value, str) and _SHA.fullmatch(value), name + " must be SHA256")


def _regular(path):
    require(not path.is_symlink(), "Store symlink rejected: " + str(path))
    require(stat.S_ISREG(path.stat().st_mode), "Expected regular store file: " + str(path))


def _read(path):
    _regular(path)
    return read_sealed(path)


def _replace_pointer(path, value):
    if path.exists() or path.is_symlink():
        _regular(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".CURRENT.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_bytes(_seal(value)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _beta_lower(successes, failures):
    """Regularized-beta inversion; same continued-fraction method as P495 V3."""
    a, b = float(1 + successes), float(1 + failures)

    def fraction(x, aa, bb):
        qab, qap, qam = aa + bb, aa + 1, aa - 1
        c, d = 1., 1. - qab * x / qap
        d = 1. / (d if abs(d) > 1e-300 else 1e-300)
        h = d
        for m in range(1, 301):
            m2 = 2 * m
            for term in (m * (bb - m) * x / ((qam + m2) * (aa + m2)),
                         -(aa + m) * (qab + m) * x / ((aa + m2) * (qap + m2))):
                d = 1. + term * d
                c = 1. + term / c
                d = d if abs(d) > 1e-300 else 1e-300
                c = c if abs(c) > 1e-300 else 1e-300
                d = 1. / d
                delta = d * c
                h *= delta
            if abs(delta - 1.) < 3e-14:
                return h
        raise ValueError("Beta continued fraction failed to converge")

    def cdf(x):
        bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
        if x < (a + 1.) / (a + b + 2.):
            return bt * fraction(x, a, b) / a
        return 1. - bt * fraction(1. - x, b, a) / b

    lo, hi = 0., 1.
    for _ in range(70):
        mid = (lo + hi) / 2
        if cdf(mid) < POLICY["lower_quantile"]:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def group_key(record, *, invariant_parent=False):
    scope = "invariant" if invariant_parent else record["scope"]
    value = {"stage": record["stage"], "scope": scope, "invariant": record["invariant"]}
    if scope == "scene_local":
        value["source_scene_fingerprint_sha256"] = record["source_scene_fingerprint_sha256"]
    return digest(value)


def _records(values):
    # Hard, versioned type checker: no production callback or schema override.
    from memory_schema import validate_record
    require(isinstance(values, (list, tuple)), "proposed_records must be a sequence")
    require(len(values) <= 256, "Too many proposed records")
    result = []
    for value in values:
        checked = validate_record(_clone(value))
        require(isinstance(checked, dict), "Record validator must return an object")
        result.append(_clone(checked))
    require(len({r["record_id"] for r in result}) == len(result), "Duplicate record ID in attempt")
    return sorted(result, key=lambda r: r["record_id"])


def _positive(admission, purpose):
    require(admission.get("schema") == "p555.episode_admission.v1", "Unknown admission schema")
    require(admission.get("purpose") == purpose, "Admission/store purpose mismatch")
    require(admission.get("classification") in {"observation", "pending_motion_validation", "failure", "success", "unknown"},
            "Unknown admission classification")
    for name in ("real_positive_credit_allowed", "fixture_success_allowed", "critical_memory_failure"):
        require(type(admission.get(name)) is bool, "Admission boolean missing: " + name)
    if purpose == "production":
        require(not admission["fixture_success_allowed"], "Fixture credit in production rejected")
        permitted = admission["real_positive_credit_allowed"]
    else:
        require(not admission["real_positive_credit_allowed"], "Real credit in fixture rejected")
        permitted = admission["fixture_success_allowed"]
    require(not permitted or admission.get("classification") == "success", "Non-success grants credit")
    affected = admission.get("affected_record_ids")
    require(isinstance(affected, list) and len(set(affected)) == len(affected), "Invalid affected record IDs")
    for rid in affected:
        _checksum(rid, "affected record ID")
    require(not affected or admission["critical_memory_failure"], "Unauthorised noncritical attribution")
    if admission["critical_memory_failure"]:
        require(admission.get("classification") == "failure" and affected and not permitted,
                "Critical failure must be a non-success with exact affected records")
    return permitted


class MemoryStore:
    def __init__(self, root, purpose="production"):
        require(purpose in {"production", "test_fixture"}, "Invalid store purpose")
        supplied = Path(root).absolute()
        for path in (supplied, *supplied.parents):
            require(not path.is_symlink(), "Store path may not traverse a symlink")
        supplied.mkdir(parents=True, exist_ok=True)
        self.root, self.purpose = supplied, purpose
        with self._locked():
            for name in ("snapshots", "cycles", "journal"):
                path = self.root / name
                require(not path.is_symlink(), "Store directory symlink")
                path.mkdir(exist_ok=True)
                require(path.is_dir(), "Store child is not a directory")
            descriptor = self.root / "store.json"
            if not descriptor.exists():
                require(not list((self.root / "journal").iterdir())
                        and not list((self.root / "snapshots").iterdir())
                        and not list((self.root / "cycles").iterdir())
                        and not (self.root / "CURRENT.json").exists(),
                        "Existing data without store descriptor")
                write_once(descriptor, {"schema": STORE_SCHEMA, "store_id": uuid.uuid4().hex,
                    "purpose": purpose, "policy": POLICY,
                    "real_credit_counted": purpose == "production"})
            self.descriptor = _read(descriptor)
            require(set(_body(self.descriptor)) == {"schema", "store_id", "purpose", "policy", "real_credit_counted"},
                    "Unexpected store descriptor fields")
            require(self.descriptor.get("schema") == STORE_SCHEMA, "Wrong store schema")
            require(self.descriptor.get("purpose") == purpose, "Production/fixture store mixing rejected")
            require(self.descriptor.get("policy") == POLICY, "Store policy drift")
            self.store_id = self.descriptor["store_id"]
            zero = self._empty_snapshot()
            path = self._snapshot_path(0)
            if not path.exists():
                require(not list((self.root / "journal").iterdir()), "Missing genesis snapshot")
                write_once(path, zero)
            self._recover_locked()

    @contextmanager
    def _locked(self):
        path = self.root / ".lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Lock is not regular")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _snapshot_path(self, generation):
        return self.root / "snapshots" / ("%08d.json" % generation)

    def _empty_snapshot(self):
        return _seal({"schema": SNAPSHOT_SCHEMA, "store_id": self.store_id,
                      "purpose": self.purpose, "generation": 0, "parent_sha256": None,
                      "credited_episodes": {}, "observations": []})

    def _cycle(self, value):
        require(isinstance(value, dict), "Cycle must be a sealed receipt")
        _checksum(value.get("cycle_id"), "cycle ID")
        disk = _read(self.root / "cycles" / (value["cycle_id"] + ".json"))
        require(disk == value and disk.get("schema") == CYCLE_SCHEMA, "Cycle seal/identity drift")
        require(disk["cycle_id"] == digest({k: v for k, v in _body(disk).items() if k != "cycle_id"}),
                "Cycle content does not match its ID")
        require(disk.get("store_id") == self.store_id and disk.get("purpose") == self.purpose,
                "Cycle belongs to another store")
        frozen = _read(self._snapshot_path(disk["generation"]))
        require(frozen[SCHEMA_HASH_FIELD] == disk["snapshot_sha256"], "Frozen snapshot drift")
        return disk, frozen

    def _append_event(self, events, kind, payload):
        event = {"schema": EVENT_SCHEMA, "store_id": self.store_id, "purpose": self.purpose,
                 "sequence": len(events), "previous_sha256": events[-1][SCHEMA_HASH_FIELD] if events else None,
                 "kind": kind, "payload": payload}
        return write_once(self.root / "journal" / ("%08d.json" % len(events)), event)

    def _transition(self, previous, attempts):
        result = _clone(previous)
        changed = False
        known = {o["record"]["record_id"]: o["record"] for o in result["observations"]}
        for event in attempts:
            payload = event["payload"]
            admission = payload["admission"]
            if not _positive(admission, self.purpose) or not payload["records"]:
                continue
            episode = admission["episode_id"]
            if episode in result["credited_episodes"]:
                continue
            for record in payload["records"]:
                rid = record["record_id"]
                require(rid not in known or known[rid] == record, "Record identity reused with changed content")
                known[rid] = record
                result["observations"].append({"episode_id": episode,
                    "scene_family": admission["scene_family"], "attempt_id": admission["attempt_id"],
                    "attempt_sequence": event["sequence"], "record": record})
            result["credited_episodes"][episode] = {"attempt_id": admission["attempt_id"],
                                                   "attempt_event_sha256": event[SCHEMA_HASH_FIELD]}
            changed = True
        if not changed:
            return previous
        result.update(generation=previous["generation"] + 1,
                      parent_sha256=previous[SCHEMA_HASH_FIELD])
        return _seal(result)

    def _check_record_origin(self, admission, records):
        for record in records:
            require(record["source_scene_fingerprint_sha256"] == admission["scene_fingerprint"],
                    "Record source scene differs from admitted episode")
        if self.purpose != "production" or not _positive(admission, self.purpose) or not records:
            return
        allowed = admission.get("allowed_record_ids")
        require(isinstance(allowed, list) and set(r["record_id"] for r in records) <= set(allowed),
                "Positive records lack authoritative extraction binding")
        receipts = admission.get("record_extraction_receipts")
        require(isinstance(receipts, list) and receipts, "Missing verified record extraction receipts")
        extracted = set()
        for binding in receipts:
            receipt = read_sealed(verified(binding))
            require(receipt.get("schema") == "p555.extracted_local_experience.v1"
                    and receipt.get("episode_id") == admission["episode_id"],
                    "Record extraction episode/schema mismatch")
            _checksum(receipt.get("record_id"), "extracted record ID")
            for name in ("source_episode_artifact", "source_geometry_artifact", "extractor_source_artifact"):
                verified(receipt.get(name))
            require(receipt["source_episode_artifact"] in admission["source_bindings"],
                    "Extraction episode artifact not admitted")
            extracted.add(receipt["record_id"])
        require(set(r["record_id"] for r in records) <= extracted, "Missing corresponding extraction receipt")

    def _recover_locked(self):
        require(_read(self.root / "store.json") == self.descriptor, "Store descriptor changed")
        snapshot = _read(self._snapshot_path(0))
        require(snapshot == self._empty_snapshot(), "Genesis snapshot drift")
        events, closed, ids, known_records = [], {}, {}, {}
        files = sorted(p for p in (self.root / "journal").iterdir() if not p.name.startswith("."))
        for sequence, path in enumerate(files):
            require(path.name == "%08d.json" % sequence, "Journal is not contiguous")
            event = _read(path)
            require(set(_body(event)) == {"schema", "store_id", "purpose", "sequence", "previous_sha256", "kind", "payload"},
                    "Unexpected journal fields")
            require(event["schema"] == EVENT_SCHEMA and event["store_id"] == self.store_id
                    and event["purpose"] == self.purpose and type(event["sequence"]) is int
                    and event["sequence"] == sequence, "Journal identity drift")
            require(event["previous_sha256"] == (events[-1][SCHEMA_HASH_FIELD] if events else None),
                    "Journal hash chain broken")
            payload = event["payload"]
            cycle, _ = self._cycle(payload["cycle"])
            if event["kind"] == "attempt":
                require(set(payload) == {"cycle", "request", "admission", "records"}, "Unexpected attempt fields")
                require(cycle["cycle_id"] not in closed, "Attempt after cycle close")
                self._check_admission(cycle, payload["admission"])
                require(_records(payload["records"]) == payload["records"], "Record normalization drift")
                self._check_record_origin(payload["admission"], payload["records"])
                require(payload["admission"]["attempt_id"] not in ids, "Duplicate attempt in journal")
                ids[payload["admission"]["attempt_id"]] = event
                for record in payload["records"]:
                    rid = record["record_id"]
                    require(rid not in known_records or known_records[rid] == record,
                            "Record identity conflict")
                    known_records[rid] = record
                for rid in payload["admission"]["affected_record_ids"]:
                    require(any(o["record"]["record_id"] == rid for o in snapshot["observations"]),
                            "Critical failure references no previously committed record")
            elif event["kind"] == "commit":
                require(set(payload) == {"cycle", "additional_cycles", "attempt_event_sha256s", "snapshot"}, "Unexpected commit fields")
                require(isinstance(payload["additional_cycles"], list), "Invalid batch cycles")
                cycles = [cycle, *[self._cycle(c)[0] for c in payload["additional_cycles"]]]
                cycle_ids = {c["cycle_id"] for c in cycles}
                require(len(cycle_ids) == len(cycles), "Duplicate batch cycle")
                for member in cycles:
                    require(member["cycle_id"] not in closed, "Cycle committed twice")
                    require(member["generation"] == snapshot["generation"]
                            and member["snapshot_sha256"] == snapshot[SCHEMA_HASH_FIELD], "Commit violates snapshot CAS")
                attempts = [e for e in events if e["kind"] == "attempt"
                            and e["payload"]["cycle"]["cycle_id"] in cycle_ids]
                require(payload["attempt_event_sha256s"] == [e[SCHEMA_HASH_FIELD] for e in attempts],
                        "Commit omitted or added attempts")
                expected = self._transition(snapshot, attempts)
                require(payload["snapshot"] == expected, "Committed positive snapshot is not evidence-derived")
                snapshot = expected
                destination = self._snapshot_path(snapshot["generation"])
                if not destination.exists():
                    write_once(destination, snapshot)
                require(_read(destination) == snapshot, "Snapshot differs from committed transaction")
                closed.update({cid: event for cid in cycle_ids})
            else:
                raise ValueError("Unknown journal event kind")
            events.append(event)
        snapshots = sorted(p.name for p in (self.root / "snapshots").iterdir() if not p.name.startswith("."))
        require(snapshots == ["%08d.json" % n for n in range(snapshot["generation"] + 1)],
                "Orphan or missing positive snapshot")
        pointer = {"schema": "p555.memory_current.v1", "store_id": self.store_id,
                   "purpose": self.purpose, "generation": snapshot["generation"],
                   "snapshot_sha256": snapshot[SCHEMA_HASH_FIELD]}
        current = self.root / "CURRENT.json"
        if current.exists():
            old = _read(current)
            require(set(_body(old)) == set(pointer) and old.get("schema") == pointer["schema"],
                    "Unexpected CURRENT schema/fields")
            require(old.get("store_id") == self.store_id and old.get("purpose") == self.purpose,
                    "CURRENT identity drift")
            require(type(old.get("generation")) is int and 0 <= old["generation"] <= snapshot["generation"],
                    "CURRENT ahead of committed journal")
            require(old.get("snapshot_sha256") == _read(self._snapshot_path(old["generation"]))[SCHEMA_HASH_FIELD],
                    "CURRENT snapshot hash drift")
        if not current.exists() or old != _seal(pointer):
            _replace_pointer(current, pointer)
        return snapshot, events, closed, ids

    def recover(self):
        with self._locked():
            snapshot, events, _, _ = self._recover_locked()
            return _seal({"schema": "p555.memory_recovery.v1", "purpose": self.purpose,
                          "generation": snapshot["generation"], "snapshot_sha256": snapshot[SCHEMA_HASH_FIELD],
                          "journal_length": len(events), "positive_episode_count": len(snapshot["credited_episodes"]),
                          "real_positive_credit": len(snapshot["credited_episodes"]) if self.purpose == "production" else 0})

    def open_cycle(self, query_binding):
        require(isinstance(query_binding, dict), "query_binding must be an object")
        for key in ("query_id", "episode_identity", "scene_fingerprint_sha256"):
            _checksum(query_binding.get(key), key)
        for key in ("scene_family", "scene_id"):
            require(isinstance(query_binding.get(key), str) and query_binding[key], "Missing " + key)
        with self._locked():
            snapshot, _, closed, _ = self._recover_locked()
            # Reopening a no-positive-update query must allow a fresh attempt.
            # Repeated opens before commit remain idempotent.
            ordinal = sum(1 for cid, event in closed.items()
                for c in [event["payload"]["cycle"], *event["payload"]["additional_cycles"]]
                if c["cycle_id"] == cid and c["generation"] == snapshot["generation"]
                and c["query_binding"] == query_binding)
            body = {"schema": CYCLE_SCHEMA, "store_id": self.store_id, "purpose": self.purpose,
                    "generation": snapshot["generation"], "snapshot_sha256": snapshot[SCHEMA_HASH_FIELD],
                    "query_binding": _clone(query_binding), "cycle_ordinal": ordinal}
            body["cycle_id"] = digest(body)
            path = self.root / "cycles" / (body["cycle_id"] + ".json")
            if path.exists():
                require(_read(path) == _seal(body), "Existing cycle differs")
                return _read(path)
            return write_once(path, body)

    def _check_admission(self, cycle, admission):
        _positive(admission, self.purpose)
        binding = cycle["query_binding"]
        for name in ("attempt_id", "episode_id"):
            _checksum(admission.get(name), name)
        require(admission["episode_id"] == binding["episode_identity"], "Admission episode differs from cycle")
        require(admission.get("query_id") == binding["query_id"], "Admission query differs from cycle")
        require(admission.get("scene_fingerprint") == binding["scene_fingerprint_sha256"]
                and admission.get("scene_family") == binding["scene_family"]
                and admission.get("scene_id") == binding["scene_id"], "Admission scene identity drift")

    def record_attempt(self, cycle, admission_request_path, proposed_records=()):
        from episode_evidence import admit_episode
        checked_cycle, _ = self._cycle(cycle)
        request = artifact(admission_request_path)
        admission = admit_episode(Path(request["path"]), purpose=self.purpose)
        require(artifact(request["path"]) == request, "Admission request changed during verification")
        self._check_admission(checked_cycle, admission)
        records = _records(proposed_records)
        self._check_record_origin(admission, records)
        payload = {"cycle": checked_cycle, "request": request,
                   "admission": _clone(admission), "records": records}
        with self._locked():
            snapshot, events, closed, ids = self._recover_locked()
            prior = ids.get(admission["attempt_id"])
            if prior is not None:
                require(prior["payload"]["admission"] == payload["admission"]
                        and prior["payload"]["records"] == records, "Attempt replay conflicts")
                return _clone(prior)
            require(cycle["cycle_id"] not in closed, "Cycle already committed")
            known = {o["record"]["record_id"] for o in snapshot["observations"]}
            require(set(admission["affected_record_ids"]) <= known, "Unknown critical record")
            return self._append_event(events, "attempt", payload)

    def commit_cycle(self, cycle, *, additional_cycles=()):
        """CAS-publish one or more cycles pinned to the same N as one N+1.

        All attempts are re-admitted before publication. A stale batch fails as
        a whole; it is not silently rebased. Same-episode seeds/mirrors earn one
        global credit in the single derived snapshot.
        """
        from episode_evidence import admit_episode
        require(isinstance(additional_cycles, (list, tuple)), "Invalid additional cycles")
        with self._locked():
            snapshot, events, closed, _ = self._recover_locked()
            checked_cycle, _ = self._cycle(cycle)
            additional = [self._cycle(c)[0] for c in additional_cycles]
            cycles = [checked_cycle, *additional]
            cycle_ids = {c["cycle_id"] for c in cycles}
            require(len(cycle_ids) == len(cycles), "Duplicate batch cycle")
            if cycle["cycle_id"] in closed:
                prior = closed[cycle["cycle_id"]]
                require(all(closed.get(cid) == prior for cid in cycle_ids), "Batch replay conflicts")
                return _clone(prior)
            for member in cycles:
                require(member["cycle_id"] not in closed, "Batch includes closed cycle")
                require(member["generation"] == snapshot["generation"]
                        and member["snapshot_sha256"] == snapshot[SCHEMA_HASH_FIELD], "Snapshot CAS conflict")
            attempts = [e for e in events if e["kind"] == "attempt"
                        and e["payload"]["cycle"]["cycle_id"] in cycle_ids]
            for event in attempts:
                payload = event["payload"]
                require(artifact(payload["request"]["path"]) == payload["request"], "Admission request drift before commit")
                current = admit_episode(Path(payload["request"]["path"]), purpose=self.purpose)
                require(current == payload["admission"], "Admission evidence changed before commit")
                self._check_record_origin(current, payload["records"])
            updated = self._transition(snapshot, attempts)
            event = self._append_event(events, "commit", {"cycle": checked_cycle,
                "additional_cycles": additional,
                "attempt_event_sha256s": [e[SCHEMA_HASH_FIELD] for e in attempts], "snapshot": updated})
            self._recover_locked()
            return event

    def query_active(self, cycle, stage, invariant_key=None):
        require(stage in _STAGES, "Unknown memory stage")
        require(invariant_key is None or isinstance(invariant_key, (dict, str)), "Invalid invariant selector")
        if isinstance(invariant_key, str):
            _checksum(invariant_key, "invariant selector")
        with self._locked():
            current, events, _, _ = self._recover_locked()
            checked_cycle, frozen = self._cycle(cycle)
            groups = {}
            for observation in frozen["observations"]:
                record = observation["record"]
                key = group_key(record)
                groups.setdefault(key, []).append(observation)
            # Attribution may name a new record not present in frozen N. Map
            # against latest verified positives, but never read their positive
            # payload as a candidate for this cycle.
            record_parents = {o["record"]["record_id"]: group_key(o["record"], invariant_parent=True)
                              for o in current["observations"]}
            failures = {}
            for event in events:
                if event["kind"] != "attempt":
                    continue
                admission = event["payload"]["admission"]
                if admission["critical_memory_failure"]:
                    for key in {record_parents[rid] for rid in admission["affected_record_ids"]}:
                        failures.setdefault(key, []).append(event)
            summaries = {}
            for key, rows in groups.items():
                record = rows[0]["record"]
                negatives = failures.get(group_key(record, invariant_parent=True), [])
                last_critical = max((e["sequence"] for e in negatives), default=-1)
                eligible = [r for r in rows if r["attempt_sequence"] > last_critical]
                episodes = {r["episode_id"] for r in rows}
                families = {r["scene_family"] for r in rows}
                revalidated = last_critical < 0 or (bool(eligible) if record["scope"] == "scene_local" else (
                    len({r["episode_id"] for r in eligible}) >= POLICY["postcritical_successes"]
                    and len({r["scene_family"] for r in eligible}) >= POLICY["postcritical_scene_families"]))
                lower = _beta_lower(len(episodes), len(negatives))
                active = len(episodes) >= POLICY["minimum_successes"] and len(families) >= POLICY["minimum_scene_families"] and lower >= POLICY["minimum_lower_bound"]
                status = "quarantined" if not revalidated else "active" if active else "shadow" if len(episodes) > 1 else "candidate"
                summaries[key] = {"group_key": key, "status": status, "positive_episode_count": len(episodes),
                    "scene_family_count": len(families), "beta_lower_05": lower, "critical_failure_count": len(negatives),
                    "last_critical_sequence": last_critical, "eligible": eligible,
                    "active_basis": "independent_invariant_evidence"}
            selected, diagnostic = [], []
            for key, summary in summaries.items():
                record = groups[key][0]["record"]
                if record["stage"] != stage:
                    continue
                if invariant_key is not None and (record["invariant"] != invariant_key if isinstance(invariant_key, dict) else digest(record["invariant"]) != invariant_key):
                    continue
                info = {k: v for k, v in summary.items() if k != "eligible"}
                if record["scope"] == "scene_local":
                    parent = summaries.get(group_key(record, invariant_parent=True))
                    info["active_basis"] = POLICY["scene_local_active_basis"]
                    info["invariant_parent_group_key"] = group_key(record, invariant_parent=True)
                    if (parent is None or parent["status"] != "active" or info["status"] == "quarantined"
                            or record["source_scene_fingerprint_sha256"] != checked_cycle["query_binding"]["scene_fingerprint_sha256"]):
                        info["status"] = "quarantined" if (parent and parent["status"] == "quarantined") or info["status"] == "quarantined" else "candidate"
                    else:
                        info["status"] = "active"
                diagnostic.append(info)
                if info["status"] == "active" and summary["eligible"]:
                    # Observed geometry only. Never average poses or invent a template.
                    representative = min(summary["eligible"], key=lambda r: (r["record"]["record_id"], r["attempt_id"]))
                    selected.append({**info, "record": representative["record"],
                        "representative_attempt_id": representative["attempt_id"],
                        "representative_policy": "deterministic_observed_representative"})
            return _seal({"schema": "p555.active_memory_query.v1", "purpose": self.purpose,
                "cycle_id": cycle["cycle_id"], "read_generation": frozen["generation"],
                "read_snapshot_sha256": frozen[SCHEMA_HASH_FIELD],
                "live_journal_head_sha256": events[-1][SCHEMA_HASH_FIELD] if events else None,
                "stage": stage, "records": sorted(selected, key=lambda r: r["group_key"]),
                "groups": sorted(diagnostic, key=lambda r: r["group_key"]),
                "m0_bypass": not selected, "current_projection_and_physics_validation_still_required": True,
                "real_credit_counted": self.purpose == "production"})
