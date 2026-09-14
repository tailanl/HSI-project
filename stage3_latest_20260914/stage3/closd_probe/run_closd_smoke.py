"""Bounded original DiP -> PHC -> PhysX -> executed-state-feedback smoke.

Initialization is explicitly the canonical XML default pose, not AMASS. A
normalization-only dataset interface never yields a motion example. Original
learned networks, PD mapping, physical stepping and feedback code are retained.
All writes are inside a new attempt directory; vendor and body files are read-only.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
VENDOR = ROOT.parent / "vendor" / "CLoSD"
OLD = PROJECT / "agent9/methods/closd_unihsi_sdf_stage3_20260912"
V2 = PROJECT / "agent9/methods/closd_unihsi_sdf_stage3_20260912_v2"
TRACKER = ROOT / "assets/multitask_tracker/Humanoid.pth"
TRACKER_SHA = "7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac"
DIP_SHA = "3ca68ae54975946b3fe5c6cd64a6eafc439b5fc43c166bc9a98d1c43aeef6618"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def geometry_sources(task_kind):
    """Bind exact original physical/render assets, without inferred SMPL-X skin."""
    human = VENDOR / "closd/data/robot_cache/smpl_humanoid_0.xml"
    name = {"bench": "sofa.urdf", "reach": "location_marker.urdf", "strike": "strike_target.urdf"}[task_kind]
    target = VENDOR / "closd/data/assets/urdf" / name
    meshes = []
    for element in ET.parse(target).getroot().iter("mesh"):
        path = (target.parent / element.attrib["filename"]).resolve(strict=True)
        if path not in meshes:
            meshes.append(path)
    return {"humanoid_mjcf": binding(human), "target_urdf": binding(target),
            "target_meshes": [binding(path) for path in meshes],
            "humanoid_external_meshes": [], "humanoid_geometry": "original XML capsule and box primitives",
            "world_up_axis": "Z", "units": "meters", "quaternion_order": "xyzw",
            "floor": {"normal": [0, 0, 1], "distance": 0.0},
            "note": "URDF visual mesh retained; PhysX may convex-decompose its collision mesh"}


class NormalizationOnlyDataset:
    def __init__(self, mean, std):
        self.t2m_dataset = SimpleNamespace(mean=mean, std=std)
        self.num_actions = 1

    def __len__(self):
        return 0


class NormalizationOnlyLoader:
    def __init__(self, mean, std):
        self.dataset = NormalizationOnlyDataset(mean, std)

    def __iter__(self):
        return iter(())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=390918)
    parser.add_argument("--task", choices=("bench", "reach", "strike"), default="bench")
    parser.add_argument("--smpl-root", type=Path, default=Path("/home/lzsh2025/kimodo-viser/TSTMotion/datasets/smpl"))
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def run(args, report):
    if not 1 <= args.steps <= 300:
        raise ValueError("Bounded smoke supports 1..300 control steps")
    source_manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for item in source_manifest["files"]:
        if digest(VENDOR / item["path"]) != item["sha256"]:
            raise ValueError("Frozen official source changed: " + item["path"])
    dip = V2 / "weights/multi_target/model000300000.pt"
    if digest(TRACKER) != TRACKER_SHA or digest(dip) != DIP_SHA:
        raise ValueError("Official tracker/DiP checkpoint identity differs")
    smpl_files = [args.smpl_root / ("SMPL_%s.pkl" % name) for name in ("NEUTRAL", "MALE", "FEMALE")]
    mean_path, std_path = OLD / "weights/dip/Mean.npy", OLD / "weights/dip/Std.npy"
    snapshot = args.out / "source_snapshot.py"
    shutil.copyfile(__file__, snapshot)
    report["sources"] = [binding(path) for path in [snapshot, ROOT / "source_manifest.json", TRACKER, dip,
        V2 / "weights/multi_target/args.json", mean_path, std_path] + smpl_files +
        sorted(path for path in (OLD / "weights/distilbert").iterdir() if path.is_file())]
    report["source_commit"] = source_manifest["git_commit"]
    report["compatibility"] = [
        "Canonical XML static initialization; no AMASS or future motion used",
        "Normalization-only HumanML loader: empty iterator, official Mean/Std",
        "Local original BERT; no network access during run",
        "Headless recorder disabled; exact physical arrays saved for external rendering",
        "Training rewards disabled for inference; never used as success metric",
        "NumPy1.21 legacy aliases restored only in this subprocess",
    ]
    if args.validate_only:
        report["status"] = "inputs_verified_not_executed"
        return
    work = args.out / "working_copy"
    shutil.copytree(VENDOR, work)
    deps = work / "dependencies"
    (deps / "data/smpl").mkdir(parents=True)
    for source in smpl_files:
        (deps / "data/smpl" / source.name).symlink_to(source.resolve())
    os.chdir(work)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "runtime/torch_extensions")
    os.environ["MPLCONFIGDIR"] = str(args.out / "matplotlib_cache")
    os.environ["MAX_JOBS"] = "2"
    sys.path[:0] = [str(ROOT / "runtime/packages"), str(VENDOR), str(VENDOR / "closd")]
    import numpy as np
    for name, value in (("float", float), ("int", int), ("bool", bool), ("complex", complex), ("object", object), ("str", str), ("unicode", str)):
        if name not in np.__dict__:
            setattr(np, name, value)
    from isaacgym import gymapi, gymutil
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: this is not a standalone/CPU-fallback experiment")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    from easydict import EasyDict
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from closd.utils.flags import flags
    from closd.env.tasks import closd as closd_module
    from closd.env.tasks.closd_multitask import CLoSDMultiTask
    from closd.diffusion_planner.model import mdm as mdm_module
    from closd.diffusion_planner.model.BERT.BERT_encoder import load_bert
    from closd.learning.amp_network_builder import AMPBuilder
    from closd.utils.running_mean_std import RunningMeanStd

    mean, std = np.load(mean_path, allow_pickle=False), np.load(std_path, allow_pickle=False)
    if mean.shape != (263,) or std.shape != (263,) or not np.all(std > 0):
        raise ValueError("Invalid official native normalization")
    closd_module.get_dataset_loader = lambda **kwargs: NormalizationOnlyLoader(mean, std)
    mdm_module.load_bert = lambda ignored_path: load_bert(str(OLD / "weights/distilbert"))

    for key, value in dict(debug=False, follow=False, fixed=False, divide_group=False,
                          no_collision_check=False, fixed_path=False, real_path=False,
                          show_traj=False, server_mode=False, slow=False, real_traj=False,
                          im_eval=False, no_virtual_display=True, render_o3d=False,
                          test=True, add_proj=False, has_eval=False, trigger_input=False).items():
        setattr(flags, key, value)
    with initialize_config_dir(version_base=None, config_dir=str(VENDOR / "closd/data/cfg")):
        composed = compose(config_name="config", overrides=["env=closd_multitask", "robot=smpl_humanoid", "learning=im_big"])
    cfg = EasyDict(OmegaConf.to_container(composed, resolve=True))
    cfg.update(test=True, train=False, headless=True, no_virtual_display=True, no_log=True,
               device="cuda", device_type="cuda", device_id=0, rl_device="cuda:0",
               dependencies_path=str(deps), seed=args.seed, has_eval=False)
    cfg.env.update(num_envs=1, episode_length=args.steps + 2, stateInit="Default",
                   task_filter=args.task, show_markers=False, models=[], cycle_motion=False)
    cfg.env.dip.model_path = str(dip)
    cfg.env.dip.context_switch_prob = 0.0
    cfg.robot.has_shape_variation = False
    write_json(args.out / "resolved_config.json", cfg)

    class StaticInitialLibrary:
        """Only episode initialization metadata; never supplies future targets."""
        def __init__(self, task):
            self._motion_lengths = torch.tensor([float(args.steps + 2) * task.dt], device=task.device)
            self.mesh_parsers = None
        def get_motion_lengths(self):
            return self._motion_lengths

    class SmokeTask(CLoSDMultiTask):
        def _load_amass_gender_betas(self, config):
            self._amass_gender_betas = np.zeros((1, 17), dtype=np.float32)
        def _load_motion(self, ignored_path, *unused, **kwargs):
            self._motion_lib = StaticInitialLibrary(self)
            self._motion_train_lib = self._motion_lib
            self._motion_eval_lib = self._motion_lib
        def _compute_reward(self, actions):
            self.rew_buf.zero_()
        def create_viewer(self):
            self.viewer = None
            self.recording = False
            self.recorder_camera_handles = []
            self.viewing_env_idx = 0
            self.max_num_camera = 0
            self._video_queue = deque()
        def render(self, *unused, **kwargs):
            return None

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    gymutil.parse_sim_config(cfg.sim, sim_params)
    sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
    task = None
    arrays = defaultdict(list)
    feedback = []
    start = time.monotonic()
    try:
        task = SmokeTask(cfg, sim_params, gymapi.SIM_PHYSX, "cuda", 0, True)
        geometry = geometry_sources(args.task)
        geometry["actors"] = []
        for role, handle in (("humanoid", task.humanoid_handles[0]), ("target", task._target_handles[0])):
            names = list(task.gym.get_actor_rigid_body_names(task.envs[0], handle))
            indices = [int(task.gym.find_actor_rigid_body_index(task.envs[0], handle, name, gymapi.DOMAIN_SIM)) for name in names]
            geometry["actors"].append({"role": role, "rigid_body_names": names, "rigid_body_sim_indices": indices,
                                       "actor_scale": float(task.gym.get_actor_scale(task.envs[0], handle))})
        write_json(args.out / "geometry_manifest.json", geometry)
        report["geometry_manifest"] = binding(args.out / "geometry_manifest.json")
        checkpoint = torch.load(TRACKER, map_location="cpu", weights_only=False)
        if task.num_obs != 574 or task.num_actions != 69 or task.get_num_amp_obs() != 1960:
            raise ValueError("Physical task observation/action ABI does not match official checkpoint")
        builder = AMPBuilder()
        builder.load(cfg.learning.params.network)
        tracker = builder.build("amp", input_shape=(574,), actions_num=69, value_size=1,
                                num_seqs=1, amp_input_shape=(1960,)).to(task.device)
        state = checkpoint["model"]
        if not all(key.startswith("a2c_network.") for key in state):
            raise ValueError("Unknown actor checkpoint namespace")
        tracker.load_state_dict({key[len("a2c_network."):]: value for key, value in state.items()}, strict=True)
        normalizer = RunningMeanStd((574,)).to(task.device)
        normalizer.load_state_dict(checkpoint["running_mean_std"], strict=True)
        amp_normalizer = RunningMeanStd((1960,))
        amp_normalizer.load_state_dict(checkpoint["amp_input_mean_std"], strict=True)
        for model in (tracker, normalizer, amp_normalizer, task.mdm):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        report["tracker_load"] = {"strict": True, "obs": 574, "actions": 69, "amp_features": 1960,
                                  "epoch": int(checkpoint["epoch"])}
        original_build = task.build_completion_input
        def record_completion(*call_args, **call_kwargs):
            known = task.pose_buffer.detach().cpu().numpy().copy()
            result = original_build(*call_args, **call_kwargs)
            index = len(feedback)
            kwargs = result[0] if isinstance(result, tuple) else result
            entry = {"index": index, "control_frame": int(task.frame_idx), "actual_buffer": known}
            # The exact model-input prefix and actual buffer remain separate arrays.
            if isinstance(kwargs, dict) and "prefix" in kwargs:
                entry["native_prefix"] = kwargs["prefix"].detach().cpu().numpy().copy()
            if call_args and call_args[0] is not None:
                entry["context_switch_vec"] = call_args[0].detach().cpu().numpy().copy()
            feedback.append(entry)
            return result
        task.build_completion_input = record_completion
        with torch.no_grad():
            task.reset()
            arrays["initial_rigid_body_state"].append(task._rigid_body_state.detach().cpu().numpy().copy())
            for step in range(args.steps):
                obs = task.obs_buf.clone()
                normalized = normalizer(obs)
                mu, sigma = tracker.eval_actor({"obs": normalized})
                action = mu.clamp(-1, 1)
                if not torch.isfinite(action).all():
                    raise ValueError("Non-finite original actor action")
                reference_before = task.ref_body_pos.detach().cpu().numpy().copy()
                task.step(action)
                task.gym.fetch_results(task.sim, True)
                if not torch.isfinite(task._rigid_body_state).all():
                    raise ValueError("Non-finite physical state")
                for name, value in {"rigid_body_state": task._rigid_body_state,
                                    "dof_pos": task._dof_pos, "dof_vel": task._dof_vel,
                                    "root_states": task._root_states, "contact_forces": task._contact_forces,
                                    "target_contact_forces": task._tar_contact_forces,
                                    "action": action, "action_mu": mu, "observation": obs,
                                    "reference_after": task.ref_body_pos,
                                    "target_position": task._tar_pos, "state_machine": task.cur_state,
                                    "task_done": task.is_done, "terminated": task._terminate_buf,
                                    "reset_flag": task.reset_buf}.items():
                    arrays[name].append(value.detach().cpu().numpy().copy())
                arrays["reference_before"].append(reference_before)
                if step % 30 == 0:
                    print(json.dumps({"control_step": step, "replans": len(feedback)}), flush=True)
                # Preserve the first failed physical episode rather than silently resetting.
                if bool(task._terminate_buf.any()):
                    report["early_termination_frame"] = step
                    break
        report["status"] = "completed_physics_loop_not_quality_acceptance"
        report["steps_recorded"] = len(arrays["action"])
        report["requested_steps"] = args.steps
        report["control_fps"] = 1.0 / task.dt
        report["simulation_fps"] = 1.0 / sim_params.dt
        report["replanning_calls"] = len(feedback)
        report["physics_wall_seconds"] = time.monotonic() - start
        report["rigid_body_names"] = list(task._body_names)
        report["target_actor_kind"] = args.task
        report["body_model"] = "official canonical SMPL physical XML; not Stage2 actor/SMPL-X"
    finally:
        if arrays:
            np.savez_compressed(args.out / "physical_rollout.npz", **{k: np.stack(v) for k, v in arrays.items() if v})
            report["physical_rollout"] = binding(args.out / "physical_rollout.npz")
        if feedback:
            for entry in feedback:
                path = args.out / ("feedback_%03d.npz" % entry["index"])
                np.savez_compressed(path, **entry)
            report["feedback_files"] = [binding(args.out / ("feedback_%03d.npz" % item["index"])) for item in feedback]
        if task is not None:
            task.gym.destroy_sim(task.sim)


def main():
    args = parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "agent9.closd.original_loop_smoke.v1", "task": args.task,
              "seed": args.seed, "training": False, "quality_accepted": False,
              "stage12_connected": False, "sdf_connected": False}
    exit_code = 0
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed_attempt", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        exit_code = 1
    write_json(args.out / "execution.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "traceback")}, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
