"""Validate and print the P478 HSI/E2.2.6 runtime contract.

This CLI performs no generation.  It is intended as a fail-fast preflight for
tmux launchers before they construct the P360 condition function.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from hsi.stage3.causal_bridge import P360E226BridgeConfig, P360E226BridgeError
EXPECTED_ABLATIONS = ('M1_KEYPOSE', 'M2_KEYPOSE_SDF', 'M3_KEYPOSE_SDF_E226', 'M4_KEYPOSE_SDF_E226_ICGF', 'M5_M4_FEEDBACK_HOLD')

def load_contract(path: Path) -> tuple[dict[str, Any], P360E226BridgeConfig]:
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise P360E226BridgeError('contract root must be a mapping')
    if payload.get('schema') != 'p478.hsi_e226_bridge.v1':
        raise P360E226BridgeError('unexpected HSI bridge schema')
    bridge_value = payload.get('bridge')
    if not isinstance(bridge_value, Mapping):
        raise P360E226BridgeError('contract needs a bridge mapping')
    config = P360E226BridgeConfig.from_mapping(bridge_value)
    ablations = payload.get('ablations')
    if not isinstance(ablations, Mapping) or tuple(ablations) != EXPECTED_ABLATIONS:
        raise P360E226BridgeError('ablations must be ordered exactly as {}'.format(EXPECTED_ABLATIONS))
    for name in EXPECTED_ABLATIONS[2:]:
        value = ablations[name]
        if not isinstance(value, Mapping):
            raise P360E226BridgeError(f'{name} must be a mapping')
        if value.get('cond_fn') != 'P478.ReMoGenE226CondFn':
            raise P360E226BridgeError(f'{name} must use the P478 cond_fn')
        if value.get('base_goal_world_zup', 'missing') is not None:
            raise P360E226BridgeError(f'{name} must set base_goal_world_zup=null to avoid duplicate goal')
    if ablations['M4_KEYPOSE_SDF_E226_ICGF'].get('predicted_icgf') is not True:
        raise P360E226BridgeError('M4 must enable predicted query-time ICGF')
    if ablations['M5_M4_FEEDBACK_HOLD'].get('feedback_hold') is not True:
        raise P360E226BridgeError('M5 must enable feedback hold')
    forbidden = payload.get('causal_contract', {}).get('forbidden', ())
    required_forbidden = {'future_frames', 'ground_truth_root', 'ground_truth_pose', 'ground_truth_contact', 'ground_truth_icgf', 'posthoc_motion_edit'}
    if not required_forbidden.issubset(set(forbidden)):
        raise P360E226BridgeError('causal contract omits required forbidden inputs')
    return (dict(payload), config)

