"""Typed, bounded, surface-local experience prototypes (never complete poses).

Scene facts legitimately carry current world geometry in a separate store.
This schema is ONLY the transferable experience prototype consumed by the
success-credit store. Query/evidence receipts are not experience payloads.
"""
from __future__ import annotations

import math
import re
from typing import Any
from memory_common import canonical_bytes, digest, require

SCHEMA = 'p555.local_affordance.v1'
SHAPE_SCHEMA = 'p555.local_surface_shape.v1'
RESOLUTION = 32
STAGES = ('stage1_route', 'stage2_contact', 'stage3_guidance')
SCOPES = ('scene_local', 'invariant')
RECORD_KEYS = frozenset({'schema', 'record_id', 'stage', 'scope',
    'source_scene_fingerprint_sha256', 'invariant', 'shape', 'payload'})
INVARIANT_KEYS = frozenset({'action', 'target_semantic', 'surface_role', 'effector',
    'motion_phase', 'body_model_sha256', 'body_shape_bin', 'shape_family'})
SHAPE_KEYS = frozenset({'schema', 'resolution', 'dimensions_m', 'area_m2',
    'support_mask', 'height_valid_mask', 'heightmap_m'})
STAGE_ROLES = {
    'stage1_route': frozenset({'free_space_approach', 'egress_corridor'}),
    'stage2_contact': frozenset({'interaction_contact'}),
    'stage3_guidance': frozenset({'walkable_support', 'clearance', 'contact_release'}),
}
STAGE_EFFECTORS = {
    'stage1_route': frozenset({'root'}),
    'stage2_contact': frozenset({'pelvis_glute', 'hand', 'feet', 'body'}),
    'stage3_guidance': frozenset({'feet', 'body', 'root'}),
}
STAGE_PHASES = {
    'stage1_route': frozenset({'approach', 'egress'}),
    'stage2_contact': frozenset({'terminal_contact', 'contact_hold'}),
    'stage3_guidance': frozenset({'locomotion', 'transition', 'terminal_contact'}),
}
PAYLOAD_KEYS = {
    'stage1_route': frozenset({'approach_offset_local_xy_m', 'approach_direction_local_xy', 'clearance_target_m'}),
    'stage2_contact': frozenset({'contact_point_local_xyz_m', 'surface_normal_local_xyz',
        'root_offset_local_xyz_m', 'facing_local_xy', 'contact_phase'}),
    'stage3_guidance': frozenset({'sdf_weight_multiplier', 'contact_weight_multiplier', 'stance_weight_multiplier'}),
}
STAGE3_BOUNDS = {'sdf_weight_multiplier': (1.0, 1.25),
                 'contact_weight_multiplier': (0.8, 1.2),
                 'stance_weight_multiplier': (1.0, 1.25)}


def valid_sha(value: Any, label: str = 'SHA256') -> str:
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'Invalid ' + label)
    return value


def number(value: Any, low: float, high: float, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), label + ' must be finite numeric, not bool')
    require(low <= value <= high, label + ' outside registered bounds')
    return float(value)


def vector(value: Any, length: int, limit: float, label: str, unit: bool = False) -> list[float]:
    require(isinstance(value, list) and len(value) == length, label + ' has wrong shape')
    result = [number(v, -limit, limit, label) for v in value]
    if unit:
        require(abs(sum(v*v for v in result)-1) <= 1e-5, label + ' must have unit length')
    return result


def validate_shape(shape: dict) -> dict:
    require(isinstance(shape, dict) and set(shape) == SHAPE_KEYS, 'Unregistered local shape fields')
    require(shape['schema'] == SHAPE_SCHEMA, 'Unregistered local shape schema')
    require(type(shape['resolution']) is int and shape['resolution'] == RESOLUTION, 'Only registered 32x32 shapes are supported')
    dims = vector(shape['dimensions_m'], 3, 10.0, 'dimensions_m')
    require(dims[0] > 1e-5 and dims[1] > 1e-5 and dims[2] >= 0, 'Invalid metric shape dimensions')
    number(shape['area_m2'], 1e-8, 100.0, 'area_m2')
    for name in ('support_mask', 'height_valid_mask'):
        require(isinstance(shape[name], list) and len(shape[name]) == RESOLUTION**2, 'Invalid flattened ' + name)
        require(all(type(v) is bool for v in shape[name]), name + ' must contain actual booleans')
    require(sum(shape['support_mask']) >= 3, 'Insufficient actual triangle support')
    heights = shape['heightmap_m']
    require(isinstance(heights, list) and len(heights) == RESOLUTION**2, 'Invalid local heightmap shape')
    for support, valid, height in zip(shape['support_mask'], shape['height_valid_mask'], heights):
        number(height, -5.0, 5.0, 'local height')
        require(valid == support, 'Height validity must match actual triangle support, not filled AABB')
        require(valid or height == 0, 'Missing height cells must be explicit masked zeros')
    canonical_bytes(shape)
    return shape


def shape_family(shape: dict) -> str:
    """Conservative metric bins group candidate observations, not learned labels."""
    validate_shape(shape)
    d, w, z = shape['dimensions_m']
    coverage = sum(shape['support_mask']) / RESOLUTION**2
    # Quantize dimensions, never orientation/absolute position. Actual nearest
    # matching still checks masks, local heights and metric scale after grouping.
    return 'd%d_w%d_z%d_c%d' % (math.floor(d/.1 + 1e-8), math.floor(w/.1 + 1e-8),
                               math.floor(z/.05 + 1e-8), min(3, math.floor(coverage*4)))


def validate_invariant(stage: str, invariant: dict, shape: dict | None = None) -> dict:
    require(stage in STAGES, 'Unknown stage namespace')
    require(isinstance(invariant, dict) and set(invariant) == INVARIANT_KEYS, 'Unregistered invariant fields')
    for key, value in invariant.items():
        require(isinstance(value, str) and 0 < len(value) <= 128, 'Invariant must be short semantic identifiers: ' + key)
        require(re.fullmatch('[A-Za-z0-9_:-]+', value) is not None, 'Non-identifier invariant: ' + key)
    require(invariant['action'] in {'sit', 'stand_up', 'walk', 'reach', 'touch', 'lie', 'open', 'close'}, 'Unsupported action identifier')
    require(invariant['surface_role'] in STAGE_ROLES[stage], 'Surface role cannot be shared across stages')
    require(invariant['effector'] in STAGE_EFFECTORS[stage], 'Effector is incompatible with stage')
    require(invariant['motion_phase'] in STAGE_PHASES[stage], 'Phase is incompatible with stage')
    valid_sha(invariant['body_model_sha256'], 'body model identity')
    if shape is not None:
        require(invariant['shape_family'] == shape_family(shape), 'Shape family is not derived from metric geometry')
    return invariant


def validate_payload(stage: str, payload: dict) -> dict:
    require(stage in STAGES, 'Unknown stage namespace')
    require(isinstance(payload, dict) and set(payload) == PAYLOAD_KEYS[stage],
            'Payload allows only registered local low-dimensional fields; no world/pose/motion/GT data')
    if stage == 'stage1_route':
        vector(payload['approach_offset_local_xy_m'], 2, 2.0, 'approach offset')
        vector(payload['approach_direction_local_xy'], 2, 1.0, 'approach direction', unit=True)
        number(payload['clearance_target_m'], .28, 1.0, 'clearance target')
    elif stage == 'stage2_contact':
        contact = vector(payload['contact_point_local_xyz_m'], 3, 2.0, 'local contact')
        require(abs(contact[2]) <= .25, 'Contact cannot be far above/below its local surface')
        normal = vector(payload['surface_normal_local_xyz'], 3, 1.0, 'surface normal', unit=True)
        require(normal[2] >= .5, 'V1 interaction-contact memory requires an upward support normal')
        vector(payload['root_offset_local_xyz_m'], 3, 2.0, 'local root offset')
        vector(payload['facing_local_xy'], 2, 1.0, 'local facing', unit=True)
        number(payload['contact_phase'], 0.0, 1.0, 'contact phase')
    else:
        for key, (low, high) in STAGE3_BOUNDS.items():
            number(payload[key], low, high, key)
    return payload


def record_identity(record: dict) -> str:
    return digest({key: value for key, value in record.items() if key != 'record_id'})


def validate_record(record: dict) -> dict:
    require(isinstance(record, dict) and set(record) == RECORD_KEYS, 'Experience prototype fields are not registered')
    require(record['schema'] == SCHEMA and record['stage'] in STAGES and record['scope'] in SCOPES, 'Invalid experience schema/stage/scope')
    valid_sha(record['source_scene_fingerprint_sha256'], 'source scene fingerprint')
    validate_shape(record['shape'])
    validate_invariant(record['stage'], record['invariant'], record['shape'])
    validate_payload(record['stage'], record['payload'])
    valid_sha(record['record_id'], 'record id')
    require(record['record_id'] == record_identity(record), 'Experience record identity mismatch')
    return record


def make_record(*, stage: str, scope: str, source_scene_fingerprint_sha256: str,
                invariant: dict, shape: dict, payload: dict) -> dict:
    value = {'schema': SCHEMA, 'stage': stage, 'scope': scope,
             'source_scene_fingerprint_sha256': source_scene_fingerprint_sha256,
             'invariant': invariant, 'shape': shape, 'payload': payload}
    # Roundtrip detaches mutable caller dictionaries and disallows nonfinite values.
    import json
    value = json.loads(canonical_bytes(value))
    value['record_id'] = record_identity(value)
    return validate_record(value)


def group_key(record: dict) -> str:
    validate_record(record)
    fields = {key: record[key] for key in ('stage', 'scope', 'invariant')}
    if record['scope'] == 'scene_local':
        fields['source_scene_fingerprint_sha256'] = record['source_scene_fingerprint_sha256']
    return digest(fields)


def invariant_key(record: dict) -> str:
    validate_record(record)
    return digest(record['invariant'])
