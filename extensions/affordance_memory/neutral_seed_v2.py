"""Strict v2 neutral identity, retaining frozen v1 numerical implementation.

Private function globals preserve all array/frame checks without modifying v1.
The five runtime-buffer fingerprints must reproduce from the exact bound asset;
an omitted/empty buffer dictionary can no longer disable runtime identity checks.
"""
from pathlib import Path
import types

import neutral_seed as retained
from memory_common import artifact, read_sealed, require, verified, write_once

RETAINED_SHA256 = "af4e5c8e47c7a9ca6d610aa1846e4c4c53017ea5614ea5a46f14c260dd379aa2"
SEED_SCHEMA = "p555.neutral_query_static_seed.v2"
RECEIPT_SCHEMA = "p555.neutral_query_static_seed_receipt.v2"
BUFFER_KEYS = frozenset(("v_template", "J_regressor", "lbs_weights", "shapedirs", "posedirs"))
NEUTRAL_SHA256 = retained.NEUTRAL_SHA256
SOURCE_KEYS = retained.SOURCE_KEYS
P523_FRAME = retained.P523_FRAME
module = retained.module
array_hash = retained.array_hash
verify_neutral_assets = retained.verify_neutral_assets


def _retained_source():
    record = artifact(retained.__file__)
    require(record["sha256"] == RETAINED_SHA256, "Frozen v1 neutral implementation drift")
    return record


def validate_buffer_identity(receipt):
    fingerprints = receipt.get("runtime_model_array_hashes")
    require(isinstance(fingerprints, dict) and set(fingerprints) == BUFFER_KEYS,
            "Neutral runtime buffer fingerprint set must contain exactly five keys")
    body = receipt["body_assets"]["candidate_body"]
    verified(body)
    require(body["sha256"] == NEUTRAL_SHA256, "Runtime fingerprints need the registered neutral asset")
    expected = retained.model_array_hashes(body["path"])
    require(fingerprints == expected, "Runtime fingerprints do not reproduce from the bound neutral asset")
    return expected


class P555NeutralSeed(retained.P555NeutralSeed):
    def build_p360_sample(self, dataset, device, plan):
        validate_buffer_identity(self.receipt)
        sample, audit = super().build_p360_sample(dataset, device, plan)
        audit.update(schema="p555.neutral_static_runtime_audit.v2", source=artifact(__file__),
                     retained_seed_implementation=_retained_source(),
                     five_runtime_buffers_recomputed_from_bound_asset=True)
        return sample, audit


def _write(path, value, **kwargs):
    if value.get("schema") == RECEIPT_SCHEMA:
        value = {**value, "retained_seed_implementation": _retained_source(),
                 "runtime_buffer_identity_policy": "exact_five_keys_recomputed_from_bound_neutral_asset"}
    return write_once(path, value, **kwargs)


def _private():
    _retained_source()
    scope = dict(vars(retained))
    scope.update(__file__=__file__, SEED_SCHEMA=SEED_SCHEMA, RECEIPT_SCHEMA=RECEIPT_SCHEMA,
                 P555NeutralSeed=P555NeutralSeed, write_once=_write)
    for name, value in vars(retained).items():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(retained):
            fn = types.FunctionType(value.__code__, scope, value.__name__, value.__defaults__, value.__closure__)
            fn.__kwdefaults__ = value.__kwdefaults__
            scope[name] = fn
    return scope


def build_seed(output, **kwargs):
    _private()["build_seed"](output, **kwargs)
    receipt = read_sealed(Path(output) / "receipt.json")
    validate_buffer_identity(receipt)
    return receipt


def load_seed(receipt_path):
    receipt = read_sealed(receipt_path)
    require(receipt.get("schema") == RECEIPT_SCHEMA, "Only v2 neutral receipt admitted by v2 loader")
    require(receipt.get("retained_seed_implementation") == _retained_source(), "Retained neutral source binding missing")
    require(receipt.get("runtime_buffer_identity_policy") == "exact_five_keys_recomputed_from_bound_neutral_asset",
            "Strict neutral identity policy missing")
    validate_buffer_identity(receipt)
    return _private()["load_seed"](receipt_path)
