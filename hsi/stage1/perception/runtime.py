"""Explicit external environments; no historical directories or model code copy."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from hsi.common import artifacts
from hsi.common.artifacts import artifact, read_sealed, require, verified, write_once


@dataclass(frozen=True)
class PerceptionRuntime:
    render_python: Path
    sam_python: Path
    sam_root: Path
    sam_weight: Path
    gpu: int = 0
    stage_timeout_seconds: int = 7200

    def validate(self) -> None:
        for name in ("render_python", "sam_python"):
            executable = Path(getattr(self, name)).resolve(strict=True)
            require(executable.is_file() and os.access(executable, os.X_OK),
                    f"{name} must be an explicit executable")
        root = Path(self.sam_root).resolve(strict=True)
        require(root.is_dir(), "External SAM/ComfyUI source root is required")
        for relative in ("comfy/sd.py", "comfy_extras/nodes_sam3.py", "comfy/ldm/sam3/detector.py"):
            require((root / relative).resolve(strict=True).is_relative_to(root)
                    and (root / relative).is_file(), "Missing external SAM source: " + relative)
        require(Path(self.sam_weight).resolve(strict=True).is_file(), "SAM checkpoint is required")
        require(type(self.gpu) is int and self.gpu >= 0, "Select exactly one nonnegative GPU index")
        require(type(self.stage_timeout_seconds) is int and self.stage_timeout_seconds > 0,
                "Stage timeout must be positive")


def source_closure() -> list[dict[str, Any]]:
    root = Path(__file__).parent
    names = ("__init__.py", "runtime.py", "worker.py", "renderer.py", "depth_geometry.py",
             "multiscale.py", "boxes.py", "sam_geometry.py", "sam.py", "atomic.py")
    return [artifact(root / name) for name in names] + [artifact(Path(artifacts.__file__))]


def verify_sources(row: dict, sam_path: Path) -> dict:
    """Original exact scene-pair and recursive artifact verification contract."""
    from hsi.stage1.scene import verify_artifact_tree

    sam = read_sealed(sam_path)
    render = read_sealed(verified(sam["source_render_receipt"]))
    require(sam.get("schema") == "p515.lingo_sam31_atomic_multiview_instances.v1",
            "Require strict atomic SAM output")
    if sam["status"] != "atomic_instances_ready" or sam["scene_id"] != row["scene_id"] or render["scene_id"] != row["scene_id"]:
        raise ValueError("Scene perception identity/status drift")
    if sam["source_original_scene_mesh"] != row["mesh"] or render["inputs"]["original_scene_mesh"] != row["mesh"] or render["inputs"]["query_occupancy"] != row["occupancy"]:
        raise ValueError("Perception assets do not match the exact manifest pair")
    boundary = sam["query_contract"]
    if boundary["instruction_available_to_segmentation_or_fusion"] is not False or boundary["motion_pose_contact_keypose_planner_memory_or_hsi_read"] is not False:
        raise ValueError("Perception is not instruction/motion independent")
    require(sam.get("instruction") == "", "Scene-only publication must not retain a task")
    verify_artifact_tree(sam)
    verify_artifact_tree(render)
    return sam


def _environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    # Native package imports only; external SAM is added explicitly inside its worker.
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYOPENGL_PLATFORM"] = "egl"
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["EGL_DEVICE_ID"] = str(gpu)
    return environment


def _worker(python: Path, arguments: list[str], log: Path, runtime: PerceptionRuntime) -> float:
    started = time.monotonic()
    # O_EXCL avoids replacing an earlier attempt's log.
    with log.open("xb") as stream:
        result = subprocess.run(
            # Preserve a virtualenv executable's symlink spelling; resolving it
            # can silently select the base interpreter instead of that venv.
            [str(Path(python).expanduser().absolute()), "-B", "-m", "hsi.stage1.perception.worker", *arguments],
            env=_environment(runtime.gpu), cwd=Path(__file__).resolve().parents[3],
            stdout=stream, stderr=subprocess.STDOUT, timeout=runtime.stage_timeout_seconds,
            check=False,
        )
    require(result.returncode == 0, f"Perception worker failed ({result.returncode}); see {log}")
    return time.monotonic() - started


def run(scene_id: str, mesh_path: Path, occupancy_path: Path, output: Path,
        *, runtime: PerceptionRuntime) -> Path:
    """Run 48-view render → anonymous depth boxes → actual SAM/atomic instances.

    ``output`` is a new attempt directory. No previous receipt, NavMesh,
    semantic answer, task description, pose, or memory bank is an input.
    Existing completed output can be verified with :func:`verify_sources`;
    this function never relabels a reused receipt as a new execution.
    """
    require(isinstance(runtime, PerceptionRuntime), "An explicit PerceptionRuntime is required")
    runtime.validate()
    require(isinstance(scene_id, str) and scene_id and "/" not in scene_id
            and "\\" not in scene_id and scene_id not in (".", ".."), "Unsafe scene ID")
    mesh_path = Path(mesh_path).resolve(strict=True)
    occupancy_path = Path(occupancy_path).resolve(strict=True)
    require(mesh_path.parent.name == scene_id and occupancy_path.stem == scene_id,
            "LINGO scene ID must match mesh parent and occupancy filename")
    row = {"scene_id": scene_id, "mesh": artifact(mesh_path), "occupancy": artifact(occupancy_path)}
    output = Path(output).resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Perception output must be a new empty attempt directory")
    output.mkdir(parents=True, exist_ok=True)
    request = output / "request.json"
    write_once(request, {
        "schema": "hsi.stage1.scene_perception_request.v1", **row,
        "source_closure": source_closure(), "instruction_read": False,
        "runtime": {"render_python": artifact(runtime.render_python),
                    "sam_python": artifact(runtime.sam_python),
                    "sam_root": str(Path(runtime.sam_root).resolve(strict=True)),
                    "sam_weight": artifact(runtime.sam_weight), "gpu": runtime.gpu},
        "policy": {"anchor_count": 6, "yaw_count": 8, "width": 640, "height": 480,
                   "sam_refine_iterations": 3, "minimum_observation_support_vertices": 180},
    }, seal=True)
    render_path = output / "render" / "receipt.json"
    durations = {}
    durations["render_seconds"] = _worker(runtime.render_python, [
        "render", "--scene-id", scene_id, "--mesh", str(mesh_path),
        "--occupancy", str(occupancy_path), "--output-dir", str(render_path.parent),
        "--anchor-count", "6", "--yaw-count", "8", "--width", "640", "--height", "480",
    ], output / "render.log", runtime)
    render = read_sealed(render_path)
    require(render.get("view_count") == 48 and render.get("status") == "fullscene_multiview_ready",
            "The complete fresh 48-view render is required")
    from . import boxes
    started = time.monotonic()
    boxes_path = boxes.run(render_path, output / "boxes" / "receipt.json")
    durations["anonymous_boxes_seconds"] = time.monotonic() - started
    sam_path = output / "sam" / "receipt.json"
    durations["sam_and_atomic_seconds"] = _worker(runtime.sam_python, [
        "sam", "--render-receipt", str(render_path), "--boxes-receipt", str(boxes_path),
        "--output", str(sam_path), "--weight", str(Path(runtime.sam_weight).resolve(strict=True)),
        "--sam-root", str(Path(runtime.sam_root).resolve(strict=True)),
    ], output / "sam.log", runtime)
    sam = verify_sources(row, sam_path)
    for record in read_sealed(request)["source_closure"]:
        verified(record)
    verified(row["mesh"])
    verified(row["occupancy"])
    write_once(output / "receipt.json", {
        "schema": "hsi.stage1.scene_perception_execution.v1", "status": "atomic_instances_ready",
        "scene_id": scene_id, "request": artifact(request), "render": artifact(render_path),
        "boxes": artifact(boxes_path), "sam": artifact(sam_path), "timing": durations,
        "instance_count": sam["instance_count"], "new_execution": True,
        "instruction_read": False, "semantic_target_verified": False,
        "source_closure": source_closure(),
    }, seal=True)
    return sam_path
