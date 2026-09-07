"""Fixed admission boundary for the consolidated package.

The complete real-motion evaluator and semantic authority have not yet been
ported. Production admission raises explicitly. Exact nine-gate fixtures are
available solely to test transactions/quarantine; they never grant real credit.
There is no caller-supplied authority, override, passed flag or callback registry.
"""
import math
from pathlib import Path
import re
from hsi.common.artifacts import artifact, verified, read_json, read_sealed, digest, require

REQUEST_SCHEMA = "p555.admission_request.v1"
ADMISSION_SCHEMA = "p555.episode_admission.v1"
FIXTURE_SCHEMA = "p555.fixture_episode_evidence.v1"
SCENE_SCHEMA = "p550.final_fixed_scene_understanding.v1"
GATES = ("collision_free", "keypose_reached", "interaction_surface_valid",
    "foot_slip_within_gate", "support_valid", "temporal_quality",
    "task_completed", "single_diffusion_chain", "trained_checkpoint")
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}")
PRODUCTION_ADMISSION_AVAILABLE = False


class ProductionAdmissionUnavailable(RuntimeError):
    """The current package cannot yet verify and admit a real complete motion."""


def _boolean(value, label):
    require(type(value) is bool, label + " must be a real boolean")
    return value



def _xy(value):
    require(isinstance(value, list) and len(value) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in value),
            "Missing finite query start XY")
    return [float(x) for x in value]



def _family(scene_id):
    # Conservative LINGO grouping: suffix variants cannot count as new families.
    match = re.match(r"^(\d+)(?:[-_]|$)", scene_id)
    return "lingo:" + str(int(match[1])).zfill(3) if match else "scene:" + scene_id



def _seal(value, field="receipt_payload_sha256"):
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}),
            "Evidence seal mismatch: " + field)
    return value



def _source_record(record):
    require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}
            and isinstance(record["path"], str) and Path(record["path"]).suffix == ".json"
            and type(record["bytes"]) is int and 0 <= record["bytes"] <= MAX_JSON_BYTES
            and isinstance(record["sha256"], str) and HEX64.fullmatch(record["sha256"]),
            "Source must be an exact bounded JSON artifact, never model weights")
    return record



class _Reader:
    def __init__(self):
        self.records = {}

    def check(self, record, *, within=None):
        require(isinstance(record, dict) and type(record.get("bytes")) is int
                and 0 <= record["bytes"] <= MAX_ARTIFACT_BYTES, "Oversized/missing evidence artifact")
        require(Path(record.get("path", "")).suffix.lower() not in
                {".safetensors", ".pt", ".pth", ".ckpt", ".bin"}, "Model weight hashing is not admission")
        key = (record.get("path"), record.get("bytes"), record.get("sha256"))
        if key not in self.records:
            path = verified(record)
            self.records[key] = dict(record)
        else:
            path = Path(record["path"])
        require(within is None or path.is_relative_to(within), "Evidence escapes its exact case: " + str(path))
        return path

    def read(self, record, *, schema=None, within=None, sealed=True):
        require(record.get("bytes", MAX_JSON_BYTES + 1) <= MAX_JSON_BYTES, "JSON evidence exceeds bound")
        path = self.check(record, within=within)
        value = read_sealed(path) if sealed else read_json(path)
        require(schema is None or value.get("schema") == schema, "Unexpected receipt schema")
        return value

    def geometry(self, record, scene_id):
        value = self.read(record)
        require(value.get("scene_id") == scene_id and isinstance(value.get("objects"), list),
                "Fixed geometry scene mismatch")
        for key in ("source_mesh", "source_occupancy"):
            self.check(value[key])
        fingerprint = digest({"mesh_sha256": value["source_mesh"]["sha256"],
                              "occupancy_sha256": value["source_occupancy"]["sha256"]})
        return value, fingerprint

    def scene(self, record, *, current=True):
        value = self.read(record, schema=SCENE_SCHEMA)
        require(value.get("human_instruction_read") is False
                and value.get("stage2_or_stage3_output_read") is False
                and (not current or value.get("semantic_backrest_review_complete") is True),
                "Not a final instruction-independent scene publication")
        require(isinstance(value.get("scene_id"), str) and value["scene_id"], "Missing scene identity")
        geometry, fingerprint = self.geometry(value["fixed_geometry"], value["scene_id"])
        self.read(value["fixed_semantics"])
        return value, geometry, fingerprint



def _verdict(reader, source, query, classification, reason, *, domain=None, checks=None):
    for record in reader.records.values():
        verified(record)  # Detect drift across the complete selected read transaction.
    identity = {k: v for k, v in query.items() if k != "scene_family"}
    episode = digest(identity)  # No output path, seed, retry count or timestamp.
    checks = dict(checks or {})
    checks["complete_query_identity_available"] = set(identity) == {
        "scene_id", "scene_fingerprint", "fixed_geometry_sha256", "instruction", "start_world_xy_m"}
    return {
        "schema": ADMISSION_SCHEMA,
        "attempt_id": digest({"source_sha256": source["sha256"], "query_sha256": episode}),
        "episode_id": episode, "query_id": episode, "classification": classification,
        "failure_domain": domain, "real_positive_credit_allowed": False,
        "critical_memory_failure": False, "affected_record_ids": [],
        "allowed_record_ids": [], "record_extraction_receipts": [],
        "scene_id": query.get("scene_id"), "scene_fingerprint": query.get("scene_fingerprint"),
        "scene_family": query.get("scene_family", _family(query.get("scene_id", "unknown"))),
        "source_bindings": list(reader.records.values()), "checks": checks, "reason": reason,
        "purpose": "production", "fixture_success_allowed": False,
    }



def _fixture(source):
    reader = _Reader()
    value = reader.read(source, schema=FIXTURE_SCHEMA)
    require(set(value) == {"schema", "namespace", "query_identity", "attempt_nonce", "checks",
                          "critical_memory_failure", "affected_record_ids", "receipt_payload_sha256"},
            "Unexpected fixture evidence fields")
    require(value["namespace"] == "test_fixture", "Missing fixture namespace")
    query = value["query_identity"]
    require(isinstance(query, dict) and set(query) ==
            {"scene_id", "scene_fingerprint", "scene_family", "instruction", "start_world_xy_m"},
            "Invalid fixture query")
    require(all(isinstance(query[k], str) and query[k] for k in
                ("scene_id", "scene_fingerprint", "scene_family", "instruction"))
            and HEX64.fullmatch(query["scene_fingerprint"]), "Invalid fixture identity")
    _xy(query["start_world_xy_m"])
    require(isinstance(value["attempt_nonce"], str) and value["attempt_nonce"], "Missing fixture attempt nonce")
    checks = value["checks"]
    require(isinstance(checks, dict) and set(checks) == set(GATES)
            and all(type(x) is bool for x in checks.values()), "Fixture needs exact nine boolean gates")
    passed = all(checks.values())
    critical = _boolean(value["critical_memory_failure"], "fixture critical flag")
    ids = value["affected_record_ids"]
    require(isinstance(ids, list) and len(set(ids)) == len(ids)
            and all(isinstance(x, str) and x for x in ids), "Invalid fixture affected IDs")
    require((not critical and not ids) or (critical and not passed and ids),
            "Critical attribution must be an explicit failed fixture")
    result = _verdict(reader, source, query, "success" if passed else "failure",
                      "Isolated mechanism fixture; never real positive credit",
                      domain=None if passed else "test_fixture", checks=checks)
    result.update(purpose="test_fixture", fixture_success_allowed=passed,
                  critical_memory_failure=critical, affected_record_ids=ids)
    return result



def admit_episode(request_path, *, purpose="production"):
    """Fixed fixture admission, with explicit fail-closed production behavior."""
    require(purpose in {"production", "test_fixture"}, "Unknown admission purpose")
    if purpose == "production":
        raise ProductionAdmissionUnavailable(
            "Complete-motion 34-gate, actual terminal-mesh, static 23-gate and "
            "actual semantic evidence authority is not integrated; no production credit is permitted")
    from .store import require_authority_sources
    require_authority_sources()
    request_binding = artifact(request_path)
    request = read_json(request_path)
    if "receipt_payload_sha256" in request:
        _seal(request)
    require(set(request) - {"receipt_payload_sha256"} == {"schema", "mode", "source_receipt"},
            "Unexpected admission request fields")
    require(request["schema"] == REQUEST_SCHEMA and request["mode"] == "test_fixture",
            "Only explicit isolated fixture admission is implemented")
    source = _source_record(request["source_receipt"])
    verified(source)
    result = _fixture(source)
    verified(request_binding)
    require_authority_sources()
    return result
