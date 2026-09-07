"""Missing real-motion authority must not turn caller flags into credit."""
import ast
from pathlib import Path

import pytest

from hsi.common.artifacts import digest, write_once
from hsi.memory import authority, store


@pytest.mark.parametrize('payload', [
    {'passed': True},
    {'schema': 'p555.admission_request.v2', 'mode': 'online_complete_motion', 'passed': True},
    {'schema': 'p555.fixture_episode_evidence.v1', 'namespace': 'test_fixture', 'passed': True},
])
def test_production_never_accepts_caller_positive(tmp_path, payload):
    request = tmp_path / 'request.json'
    write_once(request, payload)
    with pytest.raises(authority.ProductionAdmissionUnavailable):
        authority.admit_episode(request, purpose='production')
    bank = store.MemoryStore(tmp_path / 'bank')
    binding = dict(query_id=digest('q'), episode_identity=digest('q'),
        scene_fingerprint_sha256=digest('scene'), scene_family='fixture', scene_id='fixture')
    cycle = bank.open_cycle(binding)
    with pytest.raises(authority.ProductionAdmissionUnavailable):
        bank.record_attempt(cycle, request)
    assert bank.recover()['real_positive_credit'] == 0
    assert bank.query_active(cycle, 'stage2_contact')['m0_bypass'] is True


def test_no_legacy_runtime_loaders_or_project_root_constant():
    import hsi.memory
    root = Path(hsi.memory.__file__).parent
    for path in root.glob('*.py'):
        text = path.read_text()
        tree = ast.parse(text)
        assert 'agent9/methods/' not in text and 'agent9/runs/' not in text
        assert 'sys.path.insert' not in text and 'PROJECT' not in text
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {'exec', 'compile', '__import__'}
            if isinstance(node, ast.ImportFrom):
                assert node.level or node.module.startswith(('hsi.', 'pathlib', 'typing', 'contextlib', '__future__')) or node.module == 'PIL'


def test_new_store_does_not_claim_old_authority():
    assert store.POLICY['fixed_admission_authority'] == 'hsi.memory.authority.admit_episode'
    assert store.POLICY['production_admission_available'] is False
    assert authority.PRODUCTION_ADMISSION_AVAILABLE is False
    assert store.POLICY['minimum_successes'] == 8
    assert store.POLICY['minimum_scene_families'] == 4
    assert store.POLICY['minimum_lower_bound'] == .7
    assert store.POLICY['postcritical_successes'] == 2
    assert store.POLICY['postcritical_scene_families'] == 2
