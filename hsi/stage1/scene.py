"""Load an immutable, instruction-independent final Stage1A scene snapshot.

Source semantics and geometry are verified separately.  This loader does not
rerun perception, infer missing fields, reinterpret categories, or load a
saved navigation graph. Raw-scene construction is an explicit separate entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hsi.common.artifacts import read_sealed, require, verified


def verify_artifact_tree(value, seen=None):
    """Original scene-publication recursive artifact identity contract."""
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            record = {k: value[k] for k in ("path", "bytes", "sha256")}
            identity = (record["path"], record["bytes"], record["sha256"])
            if identity not in seen:
                verified(record)
                seen.add(identity)
        for item in value.values():
            verify_artifact_tree(item, seen)
    elif isinstance(value, list):
        for item in value:
            verify_artifact_tree(item, seen)
    return seen


@dataclass(frozen=True)
class FixedScene:
    publication_path: Path
    publication: dict[str, Any]
    geometry_path: Path
    geometry: dict[str, Any]
    semantics_path: Path
    semantics: dict[str, Any]


def load_final_scene(path: Path, *, verify_tree: bool = True) -> FixedScene:
    """Load the same final publication admitted by the retained Stage1B.

    ``verify_tree=False`` skips recursive archive reads only, not the final
    seal, direct geometry/semantic artifacts, or independence contracts.
    Production orchestration uses the strict default.
    """
    path = Path(path).resolve(strict=True)
    scene = read_sealed(path)
    require(scene.get("schema") == "p550.final_fixed_scene_understanding.v1"
            and scene.get("human_instruction_read") is False
            and scene.get("semantic_backrest_review_complete") is True,
            "Use the final instruction-independent Stage1A publication")
    if verify_tree:
        verify_artifact_tree(scene)
    geometry_path = verified(scene["fixed_geometry"])
    geometry = read_sealed(geometry_path)
    require(geometry["scene_id"] == scene["scene_id"]
            and geometry.get("task_instruction_read") is False
            and geometry.get("start_state_read") is False,
            "Fixed scene/geometry identity or independence drift")
    require(geometry["source_semantics"] == scene["fixed_semantics"],
            "Fixed scene semantics mismatch")
    semantics_path = verified(scene["fixed_semantics"])
    semantics = read_sealed(semantics_path)
    require(semantics.get("scene_id") == scene["scene_id"], "Fixed semantic scene mismatch")
    return FixedScene(path, scene, geometry_path, geometry, semantics_path, semantics)


def build_scene(*args, **kwargs):
    """Build a raw scene using the migrated scene-only perception/review chain."""
    from .scene_build import build
    return build(*args, **kwargs)
