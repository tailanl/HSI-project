"""Artifact-only utilities; no historical runtime roots or model loaders."""
from pathlib import Path
import json
import math
import os
import time
from hsi.common.artifacts import artifact, digest, read_json as read, read_sealed, require, sha256, verified, write_once

HERE = Path(__file__).resolve().parent


def verify_artifact_tree(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            verified({key: value[key] for key in ('path', 'bytes', 'sha256')})
        for child in value.values():
            verify_artifact_tree(child)
    elif isinstance(value, list):
        for child in value:
            verify_artifact_tree(child)


def source_closure():
    return {path.name: artifact(path) for path in sorted(HERE.glob('*.py'))}
