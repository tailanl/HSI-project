import ast
import hashlib
import json
from pathlib import Path


def test_preserved_numeric_definitions_match_migration_baseline():
    root = Path(__file__).resolve().parents[1] / "hsi/stage1"
    manifest = json.loads((root / "ORIGINS.json").read_text())
    count = 0
    for module in manifest["modules"]:
        actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                  for n in ast.parse((root / module["module"]).read_text()).body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for record in module["unchanged_top_level_definitions"]:
            assert actual[record["name"]] == record["ast_sha256"]
            count += 1
    assert count == manifest["unchanged_definition_count"]
    assert count > 100


def test_runtime_has_no_old_workspace_source_loader():
    root = Path(__file__).resolve().parents[1] / "hsi/stage1"
    forbidden = ("spec_from_file_location", "sys.path.insert", "exec(", "inspect.getsource", "agent9/methods/", "agent6/runs/")
    for path in root.glob("*.py"):
        source = path.read_text()
        assert not any(marker in source for marker in forbidden), path


def test_scene_build_original_definition_pins():
    root = Path(__file__).resolve().parents[1] / "hsi/stage1/scene_build"
    manifest = json.loads((root / "ORIGINS.json").read_text())
    count = 0
    for module in manifest["modules"]:
        actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                  for n in ast.parse((root / module["module"]).read_text()).body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for record in module["unchanged_top_level_definitions"]:
            assert actual[record["name"]] == record["ast_sha256"]
            count += 1
    assert count == manifest["unchanged_definition_count"] == 37


def test_scene_build_has_no_old_source_loader_or_placeholder():
    root = Path(__file__).resolve().parents[1] / "hsi/stage1/scene_build"
    forbidden = ("spec_from_file_location", "sys.path.insert", "exec(", "inspect.getsource",
                 "agent9/methods/", "agent6/runs/", "NotImplementedError")
    for path in root.glob("*.py"):
        assert not any(marker in path.read_text() for marker in forbidden), path
