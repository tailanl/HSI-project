"""Frozen numerical golden vectors obtained from original and native kernels.

The same inputs were executed against both implementations during migration.
Only golden JSON comparison rounds floats to ten places; kernels do not round.
No historical source path is imported by these regression tests.
"""
from dataclasses import asdict
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from hsi.stage1.perception import renderer, atomic, multiscale


def test_sixty_nine_preserved_definitions_match_independent_source_fingerprints():
    root = Path(renderer.__file__).parent
    origins = json.loads((root / "ORIGINS.json").read_text(encoding="utf-8"))
    assert origins["runtime_imports_of_origin_paths"] is False
    assert origins["third_party_model_implementation_copied"] is False
    assert origins["unchanged_definition_count"] == 69
    count = 0
    for module in origins["modules"]:
        source = (root / module["module"]).read_text(encoding="utf-8")
        lines = source.splitlines()
        definitions = {node.name: node for node in ast.parse(source).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for row in module["unchanged_top_level_definitions"]:
            node = definitions[row["name"]]
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            preserved = "\n".join(lines[first - 1:node.end_lineno]) + "\n"
            assert hashlib.sha256(preserved.encode()).hexdigest() == row["source_definition_sha256"]
            count += 1
    assert count == 69


def normalized(value):
    if isinstance(value, np.ndarray):
        return normalized(value.tolist())
    if isinstance(value, np.generic):
        return normalized(value.item())
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def checksum(value):
    raw = json.dumps(normalized(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def test_all_sixteen_camera_calibrations_match_original():
    value = [asdict(camera) for camera in renderer.build_cameras(
        np.array([[100, 150], [200, 250]]), yaw_count=8, eye_height_m=1.35,
        look_height_m=.82, look_distance_m=1.8, width=640, height=480, vertical_fov_degrees=75.)]
    assert checksum(value) == "83af4eaaf2af53967b928a1182ec0a3fc5413dd03fd5cf7daef217e79c444109"


def test_every_multiscale_proposal_and_suppression_matches_original():
    depth = np.zeros((96, 96), dtype=np.float64)
    for y in range(25, 56):
        depth[y, 30:61] = .70 + .008 * (y - 25)
    k = np.array([[80., 0., 48.], [0., 80., 48.], [0., 0., 1.]])
    value = multiscale.propose_view_multiscale(depth, k, np.eye(4), width=96, height=96,
                iou_threshold=.90, containment_threshold=.98, containment_area_ratio_minimum=.82, maximum_rows=48)
    assert checksum(value) == "3cef5601386bfe05e1b37fa45a835bb05991f56e90f26e662eaad1d3a0ff6273"


def test_full_atomic_bridge_and_deduplication_audit_matches_original():
    t = np.linspace(0., 4., 100)
    left = np.stack((np.linspace(0., .1, 100), .01 * np.sin(t), .01 * np.cos(t)), axis=1)
    vertices = np.concatenate((left, left + np.array([1., 0., 0.])))
    observations = []
    for index, (view, support) in enumerate([(0, np.arange(0, 100)), (1, np.arange(20, 180)),
                                            (2, np.arange(100, 200)), (0, np.arange(0, 100))]):
        xyz = vertices[support]
        observations.append(atomic.Observation(str(index), view, f"view_{view}", f"anon_{index}",
            np.ones((20, 20), bool), support, np.stack((xyz.min(0), xyz.max(0))),
            np.median(xyz, 0), 1., {}, {}))
    value = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert checksum(value) == "0cee84e050aa60df009dfc13f1fbe677379e70939f1198d76b8ebfc971a40b85"


def test_anchor_locations_and_all_coverage_statistics_match_original():
    native = np.zeros((300, 100, 400), dtype=bool)
    native[150, :, :] = True
    native[[0, -1], :, :] = True
    native[:, :, [0, -1]] = True
    space = renderer.human_clear_space(native, body_z_min_m=.08, body_z_max_m=1.75,
                                       clearance_m=.28, minimum_component_cells=400)
    cells, rows = renderer.select_anchor_cells(space, anchor_count=6, minimum_anchor_clearance_m=.34)
    coverage = renderer.anchor_coverage_statistics(space["retained_free"], space["component_labels"],
                                                  cells, space["retained_component_ids"])
    assert checksum((cells, rows, coverage)) == "ce9052b518e30789be76fdee59bad3dca21e96026590c40a859ffa95d8127988"
