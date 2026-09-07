"""Small filesystem/JSON contracts shared by the isolated P555 implementation.

SHA bindings detect accidental drift in a trusted local workspace. They do not
authenticate model execution against malicious processes with the same UID.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

_research_root = os.environ.get('HSI_RESEARCH_ROOT')
if not _research_root or not Path(_research_root).is_absolute():
    raise RuntimeError('Set HSI_RESEARCH_ROOT to the explicit original research-runtime root; no implicit workspace fallback')
PROJECT = Path(_research_root).resolve(strict=True)
if not PROJECT.is_dir():
    raise RuntimeError('HSI_RESEARCH_ROOT must name an existing research-runtime directory')
SCHEMA_HASH_FIELD = 'receipt_payload_sha256'
MAX_JSON_BYTES = 64 * 1024 * 1024


class MemoryContractError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise MemoryContractError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError) as error:
        raise MemoryContractError('Non-canonical or nonfinite JSON value') from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path: str | Path) -> dict:
    p = Path(path).resolve(strict=True)
    require(p.is_file(), 'Artifact must be a regular file: ' + str(p))
    before = p.stat()
    checksum = sha256(p)
    after = p.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(before) == identity(after), 'Artifact changed during read: ' + str(p))
    return {'path': str(p), 'bytes': after.st_size, 'sha256': checksum}


def verified(record: dict) -> Path:
    require(isinstance(record, dict) and set(record) == {'path', 'bytes', 'sha256'},
            'Artifact record must contain exact path/bytes/sha256 fields')
    require(isinstance(record['path'], str) and Path(record['path']).is_absolute(), 'Artifact path must be absolute')
    require(type(record['bytes']) is int and record['bytes'] >= 0, 'Invalid artifact size')
    require(isinstance(record['sha256'], str) and len(record['sha256']) == 64, 'Invalid artifact SHA256')
    actual = artifact(record['path'])
    require(actual == record, 'Artifact binding drift: ' + record['path'])
    return Path(actual['path'])


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path: str | Path) -> dict:
    p = Path(path)
    require(p.stat().st_size <= MAX_JSON_BYTES, 'JSON artifact exceeds size bound')
    try:
        data = json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda s: (_ for _ in ()).throw(MemoryContractError('Nonfinite JSON: ' + s)))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise MemoryContractError('Malformed JSON: ' + str(p)) from error
    require(isinstance(data, dict), 'Receipt JSON must be an object')
    canonical_bytes(data)
    return data


def read_sealed(path: str | Path) -> dict:
    value = read_json(path)
    expected = value.get(SCHEMA_HASH_FIELD)
    payload = {k: v for k, v in value.items() if k != SCHEMA_HASH_FIELD}
    require(isinstance(expected, str) and digest(payload) == expected, 'Receipt seal mismatch: ' + str(path))
    return value


def fsync_directory(path: str | Path) -> None:
    fd = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path: str | Path, value: dict, *, seal: bool = True) -> dict:
    """Durably install a complete JSON file without ever replacing a destination."""
    path = Path(path)
    require(isinstance(value, dict), 'Only object receipts can be published')
    output = dict(value)
    if seal:
        output.pop(SCHEMA_HASH_FIELD, None)
        output[SCHEMA_HASH_FIELD] = digest(output)
    canonical_bytes(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.' + path.name + '.', suffix='.tmp',
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # EEXIST is intentional: no race-prone replace.
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
