"""Synthetic contract tests, not receipts claiming real model/GPU execution."""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
from pathlib import Path
import sys
import subprocess

import numpy as np
from PIL import Image
import pytest

from hsi.common.artifacts import artifact, read_sealed, write_once, MemoryContractError
from hsi.stage1.perception import boxes, renderer, runtime, sam, sam_geometry, worker


def synthetic_render(root: Path, views: int = 2) -> tuple[Path, np.ndarray]:
    """One visible 400-vertex patch; every file lives in a temporary fixture."""
    root.mkdir(parents=True)
    rgb = root / "rgb.png"
    Image.new("RGB", (100, 100), (90, 110, 130)).save(rgb)
    depth_path = root / "depth.npy"
    np.save(depth_path, np.ones((100, 100), dtype=np.float32), allow_pickle=False)
    x, y = np.meshgrid(np.arange(20, 40), np.arange(20, 40))
    vertices = np.stack(((x.ravel() - 50) / 100, (y.ravel() - 50) / 100,
                         np.ones(x.size)), axis=-1)
    mesh = root / "mesh.obj"
    native = vertices @ sam_geometry.LINGO_YUP_TO_WORLD_ZUP
    mesh.write_text("".join("v %.12f %.12f %.12f\n" % tuple(row) for row in native)
                    + "f 1 2 21\nf 2 22 21\n", encoding="utf-8")
    occupancy = root / "occupancy.npy"
    np.save(occupancy, np.zeros((3, 3), dtype=bool), allow_pickle=False)
    rows = [{"view_index": index, "view_id": f"synthetic_{index}",
             "width": 100, "height": 100,
             "rgb": artifact(rgb), "depth": artifact(depth_path),
             "metric_depth_m": artifact(depth_path),
             "K": [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
             "world_to_camera": np.eye(4).tolist()}
            for index in range(views)]
    path = root / "receipt.json"
    write_once(path, {"schema": renderer.SCHEMA, "status": "fullscene_multiview_ready",
                     "scene_id": "synthetic", "view_count": views, "views": rows,
                     "mesh": {"vertex_count": len(vertices)},
                     "inputs": {"original_scene_mesh": artifact(mesh),
                                "query_occupancy": artifact(occupancy)},
                     "query_contract": {key: False for key in (
                         "instruction_read", "semantic_target_read", "support_candidate_read",
                         "motion_pose_contact_label_read", "planner_memory_or_hsi_read")}})
    return path, vertices


def synthetic_sam_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, views=2):
    render, vertices = synthetic_render(tmp_path / "render", views)
    rows = []
    for index in range(views):
        rows.append({"view_index": index, "view_id": f"synthetic_{index}", "proposals": [{
            "proposal_id": "ANON_TEST_000", "box_xywh": {"x": 18, "y": 18, "width": 25, "height": 25},
            "anonymous_proposal_score": 4.}]})
    boxes_path = tmp_path / "boxes.json"
    write_once(boxes_path, {"schema": sam_geometry.BOX_SCHEMA,
                           "status": "anonymous_sam31_box_prompts_ready", "scene_id": "synthetic",
                           "source_render_receipt": artifact(render), "views": rows,
                           "view_count": views, "total_anonymous_box_count": views,
                           "query_contract": {key: False for key in (
                               "instruction_read", "semantic_class_read_or_assigned",
                               "support_surface_candidates_read")}})
    weight = tmp_path / "synthetic.weight"
    weight.write_bytes(b"fixture only, no model")
    calls = []

    def fake_load(weight, *, sam_root):
        calls.append(("load", weight, sam_root))
        return object(), {"test_fixture_only": True, "checkpoint": artifact(weight)}

    def fake_infer(model, rgb_path, proposals, refine):
        calls.append(("infer", refine, len(proposals)))
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 20:40] = True
        cleaned, audit = sam_geometry.cleanup_mask(mask, proposals[0]["box_xywh"])
        return sam_geometry.mask_nms([{"proposal_id": proposals[0]["proposal_id"],
                                      "prompt": proposals[0]["box_xywh"], "mask": cleaned,
                                      "mask_audit": audit}]), .01

    monkeypatch.setattr(sam, "load_sam31", fake_load)
    monkeypatch.setattr(sam, "run_view_sam", fake_infer)
    args = argparse.Namespace(render_receipt=render, boxes_receipt=boxes_path,
                              output=tmp_path / "sam" / "receipt.json", weight=weight,
                              sam_root=tmp_path, instruction="", refine_iterations=3,
                              minimum_support_vertices=180)
    return args, calls, vertices


def test_static_sam_runner_publishes_only_strict_consensus_and_png(tmp_path, monkeypatch):
    args, calls, vertices = synthetic_sam_args(tmp_path, monkeypatch)
    result = sam.run(args)
    assert len(vertices) == 400
    assert [row[0] for row in calls] == ["load", "infer", "infer"]
    assert result["status"] == "atomic_instances_ready"
    assert result["instance_count"] == 1
    assert result["instruction"] == ""
    assert result["publish_gate_pass"] is False
    assert result["model"]["test_fixture_only"] is True
    assert result["query_contract"]["single_link_union_find_used"] is False
    instance = result["instances"][0]
    assert instance["support_vertex_count"] == 400
    assert len(instance["classifier_images"]) == 3
    with np.load(result["support_vertices"]["path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive[instance["instance_id"]], np.arange(400))
    assert not list(args.output.parent.rglob("*.html"))
    assert read_sealed(args.output) == result
    with pytest.raises(sam_geometry.SceneInstanceError, match="already exists"):
        sam.run(args)


def test_single_view_failure_keeps_audit_and_never_publishes_success(tmp_path, monkeypatch):
    args, calls, _ = synthetic_sam_args(tmp_path, monkeypatch, views=1)
    with pytest.raises(sam.AtomicFusionRejected) as error:
        sam.run(args)
    assert error.value.fusion_audit["published_component_count"] == 0
    assert error.value.fusion_audit["unpublished_observation_ids"] == ["OBSERVATION_0000"]
    assert not args.output.exists()


@pytest.mark.parametrize("field,value", [("instruction", "sit"), ("refine_iterations", 1),
                                         ("minimum_support_vertices", 10)])
def test_runner_does_not_accept_semantics_or_weakened_policy(tmp_path, monkeypatch, field, value):
    args, calls, _ = synthetic_sam_args(tmp_path, monkeypatch)
    setattr(args, field, value)
    with pytest.raises(sam_geometry.SceneInstanceError):
        sam.run(args)
    assert not calls
    assert not args.output.exists()


def test_boxes_process_every_view_and_preserve_masks_as_only_prompts(tmp_path):
    render_path, _ = synthetic_render(tmp_path / "render")
    result_path = boxes.run(render_path, tmp_path / "boxes" / "receipt.json")
    result = read_sealed(result_path)
    assert result["view_count"] == 2
    assert result["historical_feedback_read"] is False
    assert result["parameters"]["maximum_boxes_per_view"] == 48
    assert result["query_contract"]["boxes_are_only_visual_prompts_and_not_final_instances"] is True
    assert result["source_render_receipt"] == artifact(render_path)
    assert Path(result["visualization"]["path"]).suffix == ".png"
    with pytest.raises(MemoryContractError, match="already exists"):
        boxes.run(render_path, result_path)


def test_boxes_reject_drifted_depth_before_success(tmp_path):
    render_path, _ = synthetic_render(tmp_path / "render")
    depth = Path(read_sealed(render_path)["views"][0]["depth"]["path"])
    np.save(depth, np.full((100, 100), 2., dtype=np.float32))
    output = tmp_path / "boxes" / "receipt.json"
    with pytest.raises(Exception, match="drift"):
        boxes.run(render_path, output)
    assert not output.exists()


def test_renderer_default_is_current_6_by_8_and_calibration_is_rigid():
    args = renderer.parse_args(["--scene-id", "s", "--mesh", "m", "--occupancy", "o", "--output-dir", "d"])
    assert (args.anchor_count, args.yaw_count, args.width, args.height) == (6, 8, 640, 480)
    anchors = np.array([[100, 150], [200, 250]])
    cameras = renderer.build_cameras(anchors, yaw_count=8, eye_height_m=1.35,
                                    look_height_m=.82, look_distance_m=1.8, width=640,
                                    height=480, vertical_fov_degrees=75.)
    assert len(cameras) == 16
    for camera in cameras:
        matrix = camera.world_to_camera
        np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(matrix @ np.r_[camera.position_world_zup_m, 1.], [0., 0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-12)
    np.testing.assert_allclose(cameras[0].position_world_zup_m, [-.99, -.99, 1.35])


def test_occupancy_transform_and_anchor_component_coverage():
    native = np.zeros((300, 100, 400), dtype=bool)
    native[150, :, :] = True
    native[[0, -1], :, :] = True
    native[:, :, [0, -1]] = True
    space = renderer.human_clear_space(native, body_z_min_m=.08, body_z_max_m=1.75,
                                       clearance_m=.28, minimum_component_cells=400)
    cells, rows = renderer.select_anchor_cells(space, anchor_count=6, minimum_anchor_clearance_m=.34)
    assert len(rows) == 6
    assert len({row["component_id"] for row in rows}) == 2
    assert len({tuple(cell) for cell in cells}) == 6
    assert all(row["clearance_m"] >= .34 for row in rows)
    with pytest.raises(renderer.FullSceneRenderError, match="cover every"):
        renderer.select_anchor_cells(space, anchor_count=1, minimum_anchor_clearance_m=.34)
    with pytest.raises(renderer.FullSceneRenderError):
        renderer.native_to_world_occupancy(np.zeros((300, 100, 400), dtype=np.uint8))


def test_fixed_worker_failure_is_not_a_success_receipt(tmp_path, monkeypatch):
    output = tmp_path / "sam" / "receipt.json"
    monkeypatch.setattr(sam, "run", lambda args: (_ for _ in ()).throw(ModuleNotFoundError("SAM fixture unavailable")))
    with pytest.raises(ModuleNotFoundError):
        worker.main(["sam", "--render-receipt", "r", "--boxes-receipt", "b",
                     "--output", str(output), "--weight", "w", "--sam-root", "s"])
    failure = read_sealed(output.parent / "failure.json")
    assert failure["status"] == "blocked"
    assert failure["stage_gate_pass"] is False and failure["publish_gate_pass"] is False
    assert not output.exists()


def test_perception_sources_have_no_runtime_legacy_exec_or_hidden_asset_paths():
    root = Path(runtime.__file__).parent
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "agent9/" not in source and "agent6/" not in source
        assert "importlib.util" not in source and "_LAST_PUBLISHED_SUPPORTS" not in source
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "compile"}
    subprocess.run([sys.executable, "-B", "-c",
                    "from hsi.stage1.perception import renderer, sam, atomic, boxes; "
                    "import sys; assert 'torch' not in sys.modules and 'pyrender' not in sys.modules"],
                   check=True)
    assert all(Path(record["path"]).is_relative_to(root.parents[1])
               for record in runtime.source_closure())


def test_external_runtime_validation_requires_explicit_real_paths(tmp_path):
    config = runtime.PerceptionRuntime(Path(sys.executable), Path(sys.executable),
                                       tmp_path / "missing", tmp_path / "missing.weight")
    with pytest.raises(FileNotFoundError):
        config.validate()
    root = tmp_path / "sam_external"
    for name in ("comfy/sd.py", "comfy_extras/nodes_sam3.py", "comfy/ldm/sam3/detector.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture path; never imported\n")
    weight = tmp_path / "test.weight"
    weight.write_bytes(b"fixture")
    config = replace(config, sam_root=root, sam_weight=weight)
    config.validate()
    with pytest.raises(MemoryContractError, match="exactly one"):
        replace(config, gpu=-1).validate()
    with pytest.raises(MemoryContractError, match="timeout"):
        replace(config, stage_timeout_seconds=0).validate()


def test_native_runtime_orchestrates_fixed_workers_without_reuse(tmp_path, monkeypatch):
    """Only subprocess execution is stubbed; receipt pairing and final verification are real."""
    input_render, _ = synthetic_render(tmp_path / "input" / "synthetic")
    input_value = read_sealed(input_render)
    mesh = Path(input_value["inputs"]["original_scene_mesh"]["path"])
    occupancy = tmp_path / "input" / "synthetic.npy"
    np.save(occupancy, np.zeros((3, 3), dtype=bool))
    config = runtime.PerceptionRuntime(Path(sys.executable), Path(sys.executable), tmp_path, mesh)
    # Runtime dependency validation is separate from this process-routing test.
    monkeypatch.setattr(runtime.PerceptionRuntime, "validate", lambda self: None)
    calls = []

    def fake_worker(python, arguments, log, selected_runtime):
        calls.append((python, arguments, selected_runtime))
        options = dict(zip(arguments[1::2], arguments[2::2]))
        if arguments[0] == "render":
            assert options["--anchor-count"] == "6" and options["--yaw-count"] == "8"
            assert options["--width"] == "640" and options["--height"] == "480"
            target = Path(options["--output-dir"]) / "receipt.json"
            payload = dict(input_value)
            payload["inputs"] = {"original_scene_mesh": artifact(mesh), "query_occupancy": artifact(occupancy)}
            payload["views"] = [dict(input_value["views"][0], view_index=i, view_id=f"synthetic_{i}") for i in range(48)]
            payload["view_count"] = 48
            write_once(target, payload)
        else:
            assert options["--sam-root"] == str(tmp_path)
            target = Path(options["--output"])
            render_path = Path(options["--render-receipt"])
            write_once(target, {"schema": "p515.lingo_sam31_atomic_multiview_instances.v1",
                                "status": "atomic_instances_ready", "scene_id": "synthetic", "instruction": "",
                                "instance_count": 0, "test_fixture_only": True,
                                "source_render_receipt": artifact(render_path),
                                "source_original_scene_mesh": artifact(mesh),
                                "query_contract": {"instruction_available_to_segmentation_or_fusion": False,
                                                   "motion_pose_contact_keypose_planner_memory_or_hsi_read": False}})
        return .001

    monkeypatch.setattr(runtime, "_worker", fake_worker)
    output = tmp_path / "fresh_attempt"
    result = runtime.run("synthetic", mesh, occupancy, output, runtime=config)
    assert result == output / "sam" / "receipt.json"
    assert len(calls) == 2
    receipt = read_sealed(output / "receipt.json")
    assert receipt["sam"] == artifact(result)
    assert receipt["request"] == artifact(output / "request.json")
    assert receipt["instruction_read"] is False
    assert receipt["timing"]["anonymous_boxes_seconds"] >= 0
    with pytest.raises(MemoryContractError, match="empty attempt"):
        runtime.run("synthetic", mesh, occupancy, output, runtime=config)
