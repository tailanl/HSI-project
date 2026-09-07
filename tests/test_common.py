import pytest
from hsi.common.artifacts import (write_once, read_sealed, read_json, artifact, verified,
                                  MemoryContractError, canonical_bytes)


def test_artifact_drift_and_write_once(tmp_path):
    path = tmp_path / "receipt.json"
    original = write_once(path, {"status": "test", "unicode": "场景"})
    assert read_sealed(path) == original
    record = artifact(path)
    assert verified(record) == path
    with pytest.raises(FileExistsError):
        write_once(path, {"changed": True})
    assert read_sealed(path) == original
    path.write_text('{"changed":true}')
    with pytest.raises(MemoryContractError):
        verified(record)


@pytest.mark.parametrize("content", ['{"x":1,"x":2}', '{"x":NaN}', '[]', '{'])
def test_bad_json_is_rejected(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(MemoryContractError):
        read_json(path)


def test_nonfinite_data_cannot_be_sealed():
    with pytest.raises(MemoryContractError):
        canonical_bytes({"x": float("nan")})
