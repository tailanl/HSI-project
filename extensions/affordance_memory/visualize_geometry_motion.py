"""Evidence-bound P555 v1 motion visualization, never a motion publisher.

Reuse evaluator-produced 10475-vertex frames verbatim. No SMPL-X loading,
skinning, IK, pose/root edit, scene crop, hidden occluder, H3, or diffusion call.
OpenGL is imported only by render(); validation and phase tests are CPU-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

import numpy as np

from memory_common import PROJECT, artifact, digest, read_json, read_sealed, require, verified, write_once

HERE = Path(__file__).resolve().parent
LOADER = PROJECT / "agent9/methods/p508_lingo_sam3_scene_first_stage1_20260831/code/render_lingo_fullscene_multiview_v1.py"
SCHEMA = "p555.actual_geometry_motion_visualization.v1"
FPS = 20.


def _checked(binding, *, sealed=True):
    path = verified(binding)
    return read_sealed(path) if sealed else read_json(path)


def _record(value):
    return {key: value[key] for key in ("path", "bytes", "sha256")}


def _array(value, shape, label):
    value = np.asarray(value)
    require(value.shape == shape and np.isfinite(value).all(), "Invalid " + label)
    return value


def validate_pairing(execution, evaluation, launch, adapter, packet, skin, proposal, review, rendered, view, bindings):
    """Small pure lineage check, separate from I/O for fail-closed CPU tests."""
    require(execution.get("schema") == "p555.geometry_stage3_execution.v1"
            and execution.get("dry_run") is False and execution.get("error") is None
            and execution.get("runner_returncode") == 0 and execution.get("full22_runtime_proof") is not None,
            "Not an actual completed typed execution")
    require(evaluation.get("schema") == "p555.geometry_stage3_motion_evaluation.v1"
            and evaluation.get("evaluation_complete") is True and evaluation.get("input_contract_valid") is True,
            "Not an actual complete v1 evaluation")
    require(evaluation["source_execution"] == bindings["execution"]
            and evaluation["source_adapter"] == launch["adapter"] == bindings["adapter"]
            and evaluation["source_packet"] == launch["packet"] == adapter["packet"] == bindings["packet"],
            "Cross execution/adapter/packet evaluation")
    for name, key in (("generated_motion.npz", "source_motion"), ("generation_metadata.json", "source_metadata"),
                      ("energy_log.jsonl", "source_energy_log")):
        require(execution["outputs"][name] == evaluation[key] == bindings[key], "Cross motion/metadata/energy binding")
    require(launch.get("schema") == "p555.geometry_stage3_launch.v1" and launch["dry_run"] is False
            and launch["affordance"] in {"on", "off"} and launch["affordance"] == evaluation["affordance"], "Ablation/dry-run drift")
    require(adapter["schema"] == "p555.geometry_stage3_adapter.v1" and packet["schema"] == "p555.current_geometry_stage3_packet.v1"
            and adapter["producer"] == packet["provenance"] == evaluation["actual_provider"] == "p555_current_geometry_ik",
            "Provider identity drift")
    require(packet["source_artifacts"] == adapter["inputs"]
            and packet["surface_contract"] == adapter["surface_contract"]
            and packet["scene_name"] == adapter["scene_id"] == evaluation["scene_id"]
            and packet["text"] == adapter["instruction"] == evaluation["instruction"], "Packet query/source drift")
    require(evaluation["generated_motion_modified"] is False and evaluation["h3_provider_claimed"] is False
            and evaluation["positive_credit"] == 0 and evaluation["thresholds_relaxed"] is False,
            "Evaluation edits motion or claims another provider/credit")
    require(skin["schema"] == "p555.actual_p519_motion_skinning.v1"
            and skin["source_motion"] == evaluation["source_motion"]
            and skin["source_metadata"] == evaluation["source_metadata"]
            and skin["outputs"]["vertices"] == evaluation["skinned_vertices"] == bindings["vertices"], "Cross motion/skinned vertices")
    require(skin["source_motion_edited"] is False and skin["all_source_frames_preserved"] is True
            and skin["renderer_original_numeric_function_used"] is True and skin["vertex_count"] == 10475,
            "Partial, edited or joint-proxy skinning")
    require(skin["body_sources"]["neutral_asset"]["sha256"] == adapter["inputs"]["body_model"]["sha256"], "Skin body model mismatch")
    require(proposal["schema"] == "p555.geometry_guided_keypose_proposal.v1"
            and proposal["outputs"]["candidate"] == adapter["inputs"]["candidate"] == bindings["candidate"]
            and proposal["inputs"]["stage1"] == adapter["inputs"]["stage1_execution"]
            and proposal["inputs"]["body_model"] == adapter["inputs"]["body_model"], "Candidate/actual Stage1/body drift")
    require(review["source_proposal"] == rendered["source_proposal"] == adapter["inputs"]["proposal"]
            and rendered["source_keypose"] == bindings["candidate"]
            and rendered["source_stage1"] == view["source_stage1_execution"] == adapter["inputs"]["stage1_execution"],
            "Cross candidate/render/view")
    require(view["target"] == adapter["inputs"]["stage1_target"]
            and rendered["original_camera"] == view["selection"]["selected_camera"] == bindings["camera"]
            and rendered["original_mesh"] == _record(adapter["inputs"]["original_scene_mesh"]) == bindings["mesh"],
            "Cross target/original camera/rawmesh")
    require(rendered["original_scene_rendered"] is True
            and rendered["original_scene_vertices_textures_or_camera_changed"] is False
            and rendered["actual_body_vertices_changed"] is False, "Source keypose scene/camera/body was altered")
    gates = evaluation["release_gates"]
    require(isinstance(gates, dict) and gates and all(type(v.get("passed")) is bool for v in gates.values()), "Invalid evaluation gates")
    failed = [key for key, value in gates.items() if value["passed"] is False]
    require(evaluation["failed_release_gates"] == failed and type(evaluation["motion_publishable"]) is bool
            and evaluation["motion_publishable"] == (not failed), "Evaluation publication label contradicts gates")
    return failed


def validate_camera(camera, image_size):
    require(camera["schema"] == "p508.lingo_fullscene_camera.v1"
            and camera["coordinate_system"] == "world_zup_metric"
            and camera["extrinsic_convention"] == "opencv_world_to_camera", "Unsupported original camera convention")
    k = _array(camera["K"], (3, 3), "camera intrinsics")
    pose = _array(camera["world_to_camera"], (4, 4), "camera extrinsics")
    require(np.array_equal(k, camera["intrinsics"]) and k[0, 0] > 0 and k[1, 1] > 0
            and np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0), "Intrinsics drift")
    require(np.allclose(pose[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            and abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-6, "Invalid camera rigid transform")
    require(tuple(image_size) == (camera["width"], camera["height"]), "Original source image/camera resolution differs")
    return k, pose


def select_frames(metadata, packet, evaluation, frame_count):
    """Eight real times; requested phases are never labelled completed contact."""
    require(type(frame_count) is int and frame_count >= 8, "Eight source frames are required")
    rows = metadata["primitive_records"]
    history = metadata["query_static_initial_history_frames"]
    require(type(history) is int and history == 2 and rows, "Unsupported primitive history")
    require([r["primitive_id"] for r in rows] == list(range(len(rows)))
            and frame_count == history + 8 * len(rows), "Motion/actual 8-frame primitive coverage drift")
    require(metadata["generated_frames_full"] == metadata["evaluation_generated_frames"] == frame_count,
            "Metadata frame coverage mismatch")
    phases = {n["ordinal"]: n["phase"] for n in packet["nodes"]}
    require(all(r["active_external_ordinal"] in phases for r in rows), "Unbound active primitive ordinal")
    starts = {phase: next((history + 8 * i for i, r in enumerate(rows)
                          if phases[r["active_external_ordinal"]] == phase), None)
              for phase in ("APPROACH", "CONTACT")}
    terminal = starts["CONTACT"]
    require(terminal is not None and terminal == evaluation["route"]["phase_split"]["terminal_start_frame_inclusive"],
            "Terminal phase/frame binding differs from actual evaluator")
    approach = starts["APPROACH"]
    # If a scheduler skips APPROACH, show the preterminal frame as a labelled
    # diagnostic, not as evidence that approach was reached.
    anchors = {0, frame_count - 1, terminal, approach if approach is not None else max(0, terminal - 1)}
    while len(anchors) < 8:
        candidates = [i for i in range(frame_count) if i not in anchors]
        anchors.add(max(candidates, key=lambda i: (min(abs(i - j) for j in anchors), -i)))
    result = []
    for index in sorted(anchors):
        phase = "INITIAL_HISTORY" if index < history else phases[rows[min((index - history)//8, len(rows)-1)]["active_external_ordinal"]]
        labels = []
        if index == 0: labels.append("INITIAL")
        if index == approach: labels.append("APPROACH PHASE START")
        if approach is None and index == max(0, terminal - 1): labels.append("APPROACH NOT ACTIVE: PRETERMINAL")
        if index == terminal: labels.append("TERMINAL PHASE START")
        if index == frame_count - 1: labels.append("LAST (NOT SUCCESS CLAIM)")
        result.append(dict(frame=index, seconds=index/FPS, phase=phase, label=" / ".join(labels) or phase))
    return result


def load_inputs(execution_path, evaluation_path):
    """Read/hash existing evidence only; deliberately do NOT call load_adapter,
    whose independent admission includes another costly body reconstruction.
    This report adds no scientific publication/positive-credit authority.
    """
    from PIL import Image
    bindings = dict(execution=artifact(execution_path), evaluation=artifact(evaluation_path), source=artifact(__file__))
    execution, evaluation = _checked(bindings["execution"]), _checked(bindings["evaluation"])
    launch = _checked(execution["launch"])
    bindings["launch"] = execution["launch"]
    bindings["adapter"] = launch["adapter"]
    adapter = _checked(bindings["adapter"])
    bindings["packet"] = adapter["packet"]
    packet = _checked(bindings["packet"])
    for key in ("source_motion", "source_metadata", "source_energy_log"):
        bindings[key] = evaluation[key]
        verified(bindings[key])
    bindings["skinning"] = evaluation["skinning"]
    skin = _checked(bindings["skinning"])
    bindings["vertices"] = evaluation["skinned_vertices"]
    bindings["proposal"], bindings["review"] = adapter["inputs"]["proposal"], adapter["inputs"]["qwen_review"]
    proposal, review = _checked(bindings["proposal"]), _checked(bindings["review"])
    bindings["keypose_render"] = review["source_render"]
    rendered = _checked(bindings["keypose_render"])
    bindings["view"] = rendered["source_view_selection"]
    view = _checked(bindings["view"])
    bindings["candidate"] = adapter["inputs"]["candidate"]
    bindings["camera"] = rendered["original_camera"]
    bindings["mesh"] = rendered["original_mesh"]
    bindings["reference"] = view["selection"]["selected_source_rgb"]
    bindings["navmesh_route"] = adapter["inputs"]["navmesh_route"]
    bindings["key_nodes"] = adapter["inputs"]["key_nodes"]
    bindings["target"] = adapter["inputs"]["stage1_target"]
    bindings["stage1"] = adapter["inputs"]["stage1_execution"]
    bindings["bundle"] = adapter["inputs"]["stage1_bundle"]
    bindings["body_model"] = adapter["inputs"]["body_model"]
    bindings["neutral_seed"] = launch["neutral_seed"]
    bindings["scene_loader"] = rendered["source_scene_loader"]
    require(verified(bindings["scene_loader"]) == LOADER, "Unexpected original scene loader")
    failed = validate_pairing(execution, evaluation, launch, adapter, packet, skin, proposal, review, rendered, view, bindings)
    for value in [*bindings.values(), *evaluation["sources"], *launch["sources"], skin["source"]]:
        verified(value)
    # Current LINGO mesh_low OBJ is self-contained geometry. Do not silently
    # allow an unbound mutable MTL/texture dependency into an exact-scene claim.
    mesh_path = Path(bindings["mesh"]["path"])
    require(mesh_path.suffix.lower() == ".obj", "Only current original OBJ scene assets are registered")
    with mesh_path.open(encoding="utf-8") as stream:
        require(not any(line.lstrip().startswith("mtllib ") for line in stream),
                "External OBJ material dependencies need explicit registration; refusing an unbound texture")
    camera = _checked(bindings["camera"], sealed=False)
    with Image.open(verified(bindings["reference"])) as im:
        validate_camera(camera, im.size)
        im.verify()
    target = _checked(bindings["target"])
    require(_record(target["artifacts"]["original_scene_mesh"]) == bindings["mesh"], "Actual target rawmesh mismatch")
    stage1, bundle = _checked(bindings["stage1"]), _checked(bindings["bundle"])
    require(stage1["target"] == bindings["target"] and stage1["bundle"] == bindings["bundle"]
            and bundle["artifacts"]["navmesh_route"] == bindings["navmesh_route"]
            and bundle["artifacts"]["key_nodes"] == bindings["key_nodes"], "Stage1 actual target/route bundle drift")
    route = _checked(bindings["navmesh_route"], sealed=False)
    keynodes = _checked(bindings["key_nodes"])
    require(keynodes["inputs"]["navmesh_route"] == bindings["navmesh_route"], "Keynodes belong to another route")
    with np.load(verified(bindings["source_motion"]), allow_pickle=False) as data:
        joints = data["joints"].copy()
        require(joints.ndim == 3 and joints.shape[1:] == (22, 3) and np.isfinite(joints).all(), "Invalid actual motion joints")
        require(np.array_equal(data["betas"], np.zeros_like(data["betas"])), "Generated body not neutral zero-shape")
    with np.load(verified(bindings["vertices"]), allow_pickle=False) as data:
        vertices, faces = data["vertices_world"].copy(), data["faces"].copy()
    frames = len(joints)
    _array(vertices, (frames, 10475, 3), "all original full-mesh frames")
    require(faces.shape == (20908, 3) and faces.dtype.kind in "iu" and faces.min() >= 0 and faces.max() < 10475, "Invalid full-body topology")
    with np.load(verified(bindings["candidate"]), allow_pickle=False) as data:
        require(str(data["schema"]) == "p555.current_geometry_ik_candidate.v1" and np.array_equal(faces, data["faces"]), "Candidate/skinned topology drift")
        candidate_root = _array(data["root_xyz_yaw"], (4,), "target keypose root").copy()
        contact = _array(data["contact_world_xyz_m"], (3,), "target contact").copy()
    require(np.array_equal(candidate_root, np.asarray(packet["nodes"][-1]["full_smplx_keypose"]["root_xyz_yaw"])),
            "Packet terminal keypose root differs from candidate")
    require(skin["frame_count"] == evaluation["execution"]["frames"] == evaluation["fullmesh_scene_collision"]["frame_count"] == frames
            and evaluation["fullmesh_scene_collision"]["available"] is True
            and evaluation["fullmesh_scene_collision"]["vertex_count"] == 10475, "Evaluation/full-mesh frame mismatch")
    require(evaluation["motion_quality"]["fps"] == FPS, "Unregistered motion timing convention")
    metadata = _checked(bindings["source_metadata"], sealed=False)
    selected = select_frames(metadata, packet, evaluation, frames)
    surface = next(r["surface"] for r in target["candidate_surfaces"] if r["candidate_id"] == target["selected_surface_id"])
    trajectory = np.asarray(route["route_world_xy_m"], dtype=float)
    require(trajectory.ndim == 2 and trajectory.shape[1] == 2 and len(trajectory) >= 2 and np.isfinite(trajectory).all(), "Invalid actual NavMesh route")
    return dict(bindings=bindings, execution=execution, evaluation=evaluation, launch=launch, packet=packet,
        vertices=vertices, faces=faces, joints=joints, camera=camera, selected=selected,
        route=trajectory, candidate_root=candidate_root, contact=contact, surface=surface, failed=failed)


def status_label(evaluation):
    return ("NUMERIC RELEASE GATES PASS / NO MEMORY CREDIT" if evaluation["motion_publishable"] else
            f"FAILED / NOT PUBLISHABLE | {len(evaluation['failed_release_gates'])} failed release gates")


def _font(size):
    from PIL import ImageFont
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def _annotated(pixels, title, subtitle):
    from PIL import Image, ImageDraw
    image = Image.fromarray(pixels)
    result = Image.new("RGB", (image.width, image.height + 66), "#f5f6f8")
    result.paste(image, (0, 66))  # Original camera frame is neither cropped nor stretched.
    draw = ImageDraw.Draw(result)
    draw.text((10, 7), title, font=_font(15), fill="#a52424")
    draw.text((10, 34), subtitle, font=_font(15), fill="#263145")
    return result


def contact_sheet(images, state, output):
    from PIL import Image, ImageDraw
    require(len(images) == 8 and len(state["selected"]) == 8, "Exactly eight source times required")
    width, height = images[0].size
    require(all(im.size == (width, height) for im in images), "Contact-sheet camera sizes differ")
    sheet = Image.new("RGB", (4 * width + 5 * 12, 2 * height + 3 * 12 + 92), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 9), f"Scene {state['evaluation']['scene_id']} | geometry affordance {state['launch']['affordance'].upper()} | ORIGINAL SCENE + ORIGINAL CAMERA",
              font=_font(24), fill="#16223a")
    draw.text((12, 42), status_label(state["evaluation"]), font=_font(21), fill="#a52424")
    draw.text((12, 69), "Actual evaluated mesh frames. Occluders retained. Phase labels are requested scheduler phases, not successful contact.", font=_font(17), fill="#374459")
    for i, im in enumerate(images):
        sheet.paste(im, (12 + (i % 4)*(width+12), 104 + (i//4)*(height+12)))
    sheet.save(output)


def trajectory_plot(state, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    route, roots = state["route"], state["joints"][:, 0, :2]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.plot(route[:, 0], route[:, 1], "--", color="#1b9e77", lw=2, label="Actual Stage1 NavMesh route")
    ax.plot(roots[:, 0], roots[:, 1], color="#c45125", lw=1.5, label="Generated pelvis root (joint 0)")
    points = ax.scatter(roots[:, 0], roots[:, 1], c=np.arange(len(roots))/FPS, s=8, cmap="magma", zorder=3)
    fig.colorbar(points, ax=ax, label="Actual motion time (seconds)", shrink=.75)
    for row in state["selected"]:
        xy = roots[row["frame"]]
        ax.annotate(str(row["frame"]), xy, xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.scatter(*route[-1], marker="s", s=100, color="#1b9e77", label="Standing route goal")
    ax.scatter(*state["candidate_root"][:2], marker="*", s=180, color="#246cc0", label="Target keypose root")
    ax.scatter(*state["contact"][:2], marker="x", s=110, color="#101010", label="Target gluteal contact point")
    ax.scatter(*roots[0], marker="o", s=80, facecolors="none", edgecolors="#c45125", label="Actual initial root")
    ax.scatter(*roots[-1], marker="D", s=65, color="#c45125", label="Actual last root")
    bounds = np.asarray(state["surface"]["bounds_world_zup_m"])
    ax.add_patch(Rectangle(bounds[0, :2], *(bounds[1, :2]-bounds[0, :2]), fill=False, hatch="//", edgecolor="#6d7890",
                           label="Target surface XY bounds (not its mask)"))
    ax.set(xlabel="World X (metres)", ylabel="World Y (metres)", aspect="equal")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"Scene {state['evaluation']['scene_id']} | affordance {state['launch']['affordance']}\n"+status_label(state["evaluation"]), fontsize=11)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render(execution_path, evaluation_path, output, *, animation="none", gpu=5, egl_device=None):
    """Explicit rendering entry; default PNG-only, optional every-frame MP4/GIF."""
    require(animation in {"none", "mp4", "gif"}, "Unknown animation format")
    require(type(gpu) is int and gpu >= 0, "Invalid explicit rendering GPU")
    egl_device = gpu if egl_device is None else egl_device
    require(type(egl_device) is int and egl_device >= 0, "Invalid explicit EGL device index")
    if animation == "mp4":
        require(importlib.util.find_spec("imageio_ffmpeg") is not None, "MP4 needs imageio-ffmpeg; use --animation gif in the current CPU environment")
    started = time.monotonic()
    state = load_inputs(execution_path, evaluation_path)
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite a visualization")
    output.mkdir(parents=True)
    # EGL enumerates its own devices; CUDA_VISIBLE_DEVICES does not remap its
    # index. Keep both explicit, never silently use physical EGL index zero.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYOPENGL_PLATFORM="egl", EGL_DEVICE_ID=str(egl_device))
    import pyrender
    import trimesh
    spec = importlib.util.spec_from_file_location("p555_motion_original_scene_loader", LOADER)
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    mesh = loader.load_scene_mesh(verified(state["bindings"]["mesh"]))
    scene = pyrender.Scene(bg_color=[.87, .88, .9, 1.], ambient_light=[.55, .55, .55])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    camera = state["camera"]
    k, pose = np.asarray(camera["K"]), loader.opencv_to_pyrender_pose(np.asarray(camera["world_to_camera"]))
    scene.add(pyrender.IntrinsicsCamera(k[0,0], k[1,1], k[0,2], k[1,2]), pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8), pose=pose)
    renderer = pyrender.OffscreenRenderer(camera["width"], camera["height"])
    from OpenGL.GL import glGetString, GL_RENDERER, GL_VERSION
    render_device = dict(cuda_visible_devices=str(gpu), requested_egl_device_index=egl_device,
        egl_device_name=renderer._platform._egl_device.name,
        gl_renderer=glGetString(GL_RENDERER).decode(), gl_version=glGetString(GL_VERSION).decode(),
        egl_index_is_not_inferred_from_cuda_visibility=True)
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.94, .43, .13, 1.], roughnessFactor=.85)
    selected = {r["frame"]: r for r in state["selected"]}
    images, frames, gif_images, writer = {}, [], [], None
    animation_path = output / ("motion." + animation) if animation != "none" else None
    try:
        if animation == "mp4":
            import imageio.v2 as imageio
            writer = imageio.get_writer(animation_path, fps=FPS, codec="libx264", macro_block_size=1, quality=8)
        indices = range(len(state["vertices"])) if animation != "none" else sorted(selected)
        for index in indices:
            # Use a fresh render mesh to avoid stale OpenGL VBOs. Never mutate
            # evaluator arrays and never recompute/regress/translate a body.
            body = trimesh.Trimesh(vertices=state["vertices"][index].copy(), faces=state["faces"].copy(), process=False)
            require(np.array_equal(body.vertices, state["vertices"][index]) and np.array_equal(body.faces, state["faces"]),
                    "Render library altered evaluated geometry")
            node = scene.add(pyrender.Mesh.from_trimesh(body, material=material, smooth=False))
            pixels, _ = renderer.render(scene)
            scene.remove_node(node)
            label = selected[index]["label"] if index in selected else "ACTUAL GENERATED FRAME"
            annotated = _annotated(pixels, status_label(state["evaluation"]), f"Frame {index} | {index/FPS:.2f}s | {label}")
            if index in selected:
                path = output / f"frame_{index:06d}.png"
                annotated.save(path)
                frames.append({**selected[index], "image": artifact(path)})
                images[index] = annotated
            if writer is not None: writer.append_data(np.asarray(annotated))
            if animation == "gif": gif_images.append(annotated)
    finally:
        renderer.delete()
        if writer is not None: writer.close()
    if animation == "gif":
        gif_images[0].save(animation_path, save_all=True, append_images=gif_images[1:], duration=50, loop=0, optimize=False, disposal=2)
    sheet_path, xy_path = output / "contact_sheet.png", output / "trajectory_xy.png"
    contact_sheet([images[r["frame"]] for r in state["selected"]], state, sheet_path)
    trajectory_plot(state, xy_path)
    for binding in state["bindings"].values(): verified(binding)
    result = dict(schema=SCHEMA, source=state["bindings"]["source"], inputs=state["bindings"],
        scene_id=state["evaluation"]["scene_id"], instruction=state["evaluation"]["instruction"],
        affordance=state["launch"]["affordance"], seed=state["launch"]["seed"],
        frame_count=len(state["vertices"]), vertex_count=10475, fps=FPS, frames=frames,
        contact_sheet=artifact(sheet_path), trajectory=artifact(xy_path), animation=artifact(animation_path) if animation_path else None,
        animation_all_source_frames_preserved=animation != "none", animation_generated=animation != "none",
        motion_publishable=state["evaluation"]["motion_publishable"], failed_release_gates=state["failed"],
        publication_status_source=state["bindings"]["evaluation"], renderer_grants_no_publication=True,
        actual_provider="p555_current_geometry_ik", h3_claimed=False, positive_credit=0,
        evaluated_vertices_reused_verbatim=True, body_reskinned=False, motion_or_root_modified=False,
        original_camera_preserved=True, original_raw_scene_full_mesh_retained=True, scene_occluders_hidden=False,
        frame_crop_or_camera_rescale=False, world_transform="retained_P508_LINGO_YUP_TO_WORLD_ZUP_once_scene_only",
        xy_plot_is_diagnostic_not_camera_view=True, render_device=render_device, elapsed_seconds=time.monotonic()-started)
    write_once(output / "receipt.json", result)
    from memory_common import fsync_directory
    readme = ("# Actual Stage3 motion visualization\n\n" + status_label(state["evaluation"]) + "\n\n"
        "- [Eight actual times](contact_sheet.png); [actual root vs route/target](trajectory_xy.png).\n"
        + (f"- [Full {len(state['vertices'])}-frame animation](motion.{animation}), 20 Hz.\n" if animation != "none" else "- Animation not requested.\n")
        + "\nThe original selected camera and complete raw scene are retained, including occluders. Initial body may be outside this fixed camera. "
        "No isolated-body image is substituted. Evaluation's 10475 vertices per frame are reused without skinning or motion/root edits. "
        "Phase labels describe the requested scheduler phase, not successful task completion. This is the geometric IK provider, not H3. Memory credit remains 0.\n\n"
        + "Failed evaluator gates:\n\n" + "\n".join("- `"+name+"`" for name in state["failed"]) + "\n")
    with (output / "README.md").open("x") as stream:
        stream.write(readme)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(output)
    return output / "receipt.json"


def pair_montage(first_path, second_path, output):
    """Same-input on/off comparison only; never implies that either passed."""
    from PIL import Image, ImageDraw
    records = [artifact(first_path), artifact(second_path)]
    rows = [_checked(r) for r in records]
    require(all(r["schema"] == SCHEMA for r in rows) and {r["affordance"] for r in rows} == {"on", "off"}, "Pair needs one actual on and off visualization")
    for row in rows:
        evaluation = _checked(row["inputs"]["evaluation"])
        require(row["motion_publishable"] == evaluation["motion_publishable"]
                and row["failed_release_gates"] == evaluation["failed_release_gates"]
                and row["affordance"] == evaluation["affordance"]
                and row["inputs"]["execution"] == evaluation["source_execution"], "Pair label differs from actual evaluation")
    for key in ("packet", "candidate", "camera", "mesh", "neutral_seed", "target", "navmesh_route"):
        require(rows[0]["inputs"][key] == rows[1]["inputs"][key], "Pair input differs: " + key)
        verified(rows[0]["inputs"][key])
    require(rows[0]["scene_id"] == rows[1]["scene_id"] and rows[0]["seed"] == rows[1]["seed"]
            and rows[0]["instruction"] == rows[1]["instruction"], "Pair query/seed mismatch")
    rows.sort(key=lambda r: r["affordance"])
    images = [Image.open(verified(r["contact_sheet"])).convert("RGB") for r in rows]
    require(images[0].size == images[1].size, "Pair original image dimensions differ")
    canvas = Image.new("RGB", (images[0].width, sum(im.height for im in images)+70), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "SAME INPUT + SEED | geometry affordance OFF / ON | each panel retains its own failed/pass label", font=_font(21), fill="#16223a")
    draw.text((12, 39), "Times independently selected from actual phase logs; not frame-aligned and not proof of learned-memory improvement.", font=_font(19), fill="#374459")
    offset = 70
    for im in images:
        canvas.paste(im, (0, offset)); offset += im.height
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite a pair montage")
    output.mkdir(parents=True)
    path = output / "affordance_off_on.png"
    canvas.save(path)
    with (output / "README.md").open("x") as stream:
        stream.write("# Same-input geometry affordance OFF / ON\n\n"
            "[Comparison PNG](affordance_off_on.png)\n\n"
            "Both panels retain their actual evaluator status. Packet, candidate, camera, raw scene, route, initial seed and random seed match. "
            "Frame times are independently selected from each actual phase log, not aligned samples. "
            "This is a geometric-guidance ablation, not proof of learned-memory improvement. Credit remains 0.\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_once(output / "receipt.json", dict(schema="p555.geometry_motion_pair_visualization.v1", source=artifact(__file__),
        inputs=records, image=artifact(path), same_packet_candidate_camera_scene_seed=True,
        frame_times_independently_phase_selected=True, proves_learning_improvement=False,
        case_statuses=[dict(affordance=r["affordance"], motion_publishable=r["motion_publishable"], failed_release_gates=r["failed_release_gates"]) for r in rows], positive_credit=0))
    return output / "receipt.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--pair", nargs=2, type=Path, help="Two completed same-input visualization receipts, off and on")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--animation", choices=("none", "mp4", "gif"), default="none")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--egl-device", type=int, help="Actual EGL enumeration index; defaults to --gpu, not CUDA-local index zero")
    args = parser.parse_args()
    require((bool(args.pair) and args.execution is None and args.evaluation is None)
            or (not args.pair and args.execution is not None and args.evaluation is not None),
            "Supply execution+evaluation OR a completed pair")
    path = pair_montage(*args.pair, args.output) if args.pair else render(args.execution, args.evaluation, args.output, animation=args.animation, gpu=args.gpu, egl_device=args.egl_device)
    print(path, flush=True)


if __name__ == "__main__":
    main()
