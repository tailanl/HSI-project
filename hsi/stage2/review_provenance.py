"""Recompute decisions from actual recorded Qwen calls and exact image order."""
import hashlib
import json
from pathlib import Path
from ._common import artifact, read_sealed, require, verified
from . import compact_review, image_review, mesh_review


def validate(path, kind):
    value = read_sealed(path)
    module = image_review if kind == 'h3' else mesh_review if kind == 'refined' else None
    require(module is not None, 'Unknown semantic review kind')
    require(value['source'] == artifact(module.__file__), 'Semantic review source changed')
    generation = value['generation']
    call_path = Path(generation['call_receipt'])
    call = read_sealed(call_path)
    from hsi.stage1 import qwen
    require(call['source_code'] == artifact(qwen.__file__), 'Unrecognized Qwen transport')
    require(call['status'] == 'complete' and call['result'] == generation, 'Audit differs from recorded Qwen call')
    require(str(generation['model_id']).startswith('Qwen/'), 'Review requires configured Qwen model')
    require(generation['model_id'] == call['model']['model_id'] == generation['response_model'] == call['response']['model'],
            'Qwen model identity changed')
    choice = call['response']['choices'][0]
    require(choice['finish_reason'] == generation['finish_reason'] == 'stop', 'Truncated Qwen review')
    require(choice['message']['content'] == generation['raw_completion'], 'Raw response differs')
    for key in ('prompt', 'system_prompt'):
        require(hashlib.sha256(call[key].encode()).hexdigest() == generation[key + '_sha256'], 'Review prompt binding changed')
    checks = image_review.TRANSPORT_CHECKS if kind == 'h3' else mesh_review.CHECKS
    images = list(value['inputs'].values()) if kind == 'h3' else value['actual_input_images']
    require(len(images) == len(call['image_evidence']) == generation['image_count'], 'Qwen image count changed')
    for index, (record, expected) in enumerate(zip(call['image_evidence'], images)):
        require(record['label'] == f'IMAGE_{index}', 'Qwen image labels reordered')
        observed = {key: record[key] for key in ('path', 'bytes', 'sha256')}
        require(observed == expected, 'Qwen reviewed another image')
        verified(expected)
    raw = json.loads(generation['raw_completion'])
    require(value['compact_transport_only_no_checks_or_evidence_removed'] is True
            and value['ordered_boolean_checks'] == list(checks) and value['compact_raw_judgement'] == raw,
            'Compact response/check order changed')
    parsed = compact_review.normalize(raw, checks)
    decision = image_review.verdict(parsed) if kind == 'h3' else mesh_review.decide(parsed)
    require(value['judgement'] == parsed and value['decision'] == decision, 'Recorded Qwen decision is not reproducible')
    return {'audit': artifact(path), 'actual_call': artifact(call_path), 'kind': kind,
            'decision_recomputed': True, 'actual_image_order_verified': True}
