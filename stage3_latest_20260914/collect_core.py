"""Explicit, source-only Stage3 extraction. Never copy weights or run outputs."""
from pathlib import Path
import hashlib
import json
import shutil

DEST = Path(__file__).resolve().parent
AGENT = DEST.parent
LATEST = AGENT / "methods/paper_structure_stage3_20260913"
OLD = AGENT / "methods/closd_unihsi_sdf_stage3_20260912"
V2 = AGENT / "methods/closd_unihsi_sdf_stage3_20260912_v2"
CLOSD = LATEST / "vendor/CLoSD"
UNIHSI = Path("/home/lzsh2025/workspace/UniHSI")
ALLOWED = {".py", ".yaml", ".yml", ".json", ".md", ".txt"}


def main():
    plan = []

    def add(root, subdir, names):
        for name in names:
            plan.append((root / name, Path(subdir) / name))

    add(LATEST, "stage3", [
        "export_stage12.py", "exact_heightmap.py", "reference_sdf.py",
        "pd_sdf_filter.py", "pd_sdf_filter_v2.py", "rerun_state_guarded_sdf.py",
        "unihsi_probe/run_release_rollout.py", "unihsi_probe/run_stage12_transfer_v3.py",
        "unihsi_probe/STAGE12_ADAPTER_BOUNDARY.md",
        "closd_probe/run_closd_smoke.py", "closd_probe/run_closd_convex_sdf.py",
        "closd_probe/convex_reference_sdf.py",
    ])
    add(CLOSD, "networks/closd", [
        "LICENSE", "closd/diffusion_planner/LICENSE",
        "closd/diffusion_planner/model/mdm.py",
        "closd/diffusion_planner/model/BERT/BERT_encoder.py",
        "closd/diffusion_planner/diffusion/gaussian_diffusion.py",
        "closd/diffusion_planner/diffusion/respace.py",
        "closd/diffusion_planner/diffusion/nn.py",
        "closd/diffusion_planner/diffusion/losses.py",
        "closd/diffusion_planner/utils/model_util.py",
        "closd/diffusion_planner/utils/sampler_util.py",
        "closd/diffusion_planner/utils/cond_util.py",
        "closd/diffusion_planner/utils/misc.py",
        "closd/learning/amp_network_builder.py", "closd/learning/network_builder.py",
        "closd/utils/running_mean_std.py",
        "closd/env/tasks/closd.py", "closd/env/tasks/closd_task.py",
        "closd/env/tasks/closd_multitask.py", "closd/env/tasks/humanoid_im.py",
        "closd/env/tasks/humanoid_amp.py", "closd/env/tasks/humanoid_amp_task.py",
        "closd/env/tasks/humanoid.py", "closd/env/tasks/base_task.py",
    ])
    add(UNIHSI, "networks/unihsi", [
        "README.md", "unihsi/learning/amp_network_builder.py",
        "unihsi/env/tasks/unihsi_scannet.py", "unihsi/env/tasks/humanoid_amp_task.py",
        "unihsi/env/tasks/humanoid_amp.py", "unihsi/env/tasks/humanoid.py",
        "unihsi/env/tasks/base_task.py",
    ])
    add(LATEST / "unihsi_probe/vendor_rl_games_1_1_4/rl-games-1.1.4", "networks/rl_games_core", [
        "README.md", "rl_games/algos_torch/network_builder.py",
        "rl_games/algos_torch/running_mean_std.py",
    ])
    # Existing kinematic SDF/keypose code used by the new branch, not a third
    # physical policy and not a newly trained network.
    add(OLD / "code", "sdf_motion/base", [
        "dip_backend.py", "hml_codec.py", "stage12_conditions.py",
        "sdf_guidance.py", "native_consistency.py", "mesh_realization.py",
    ])
    add(V2 / "code", "sdf_motion/target", [
        "target_backend.py", "target_coordinates.py", "generation_conditions.py",
        "run_target_pilot.py", "run_target_keypose_pilot.py", "guarded_refine.py",
    ])
    add(AGENT / "HSI-project", "shared", [
        "hsi/common/artifacts.py", "hsi/stage3_sequence/geometry.py",
        "hsi/stage3_sequence/constraints.py", "hsi/stage3_sequence/mesh_body.py",
        "experiments/stage3_upstream_adapter.py",
    ])
    add(CLOSD / "closd/data/cfg", "configs/closd", [
        "config.yaml", "env/closd_base.yaml", "env/closd_multitask.yaml",
        "env/dip/dip_defaults.yaml", "learning/im_big.yaml",
        "train/rlg/im_big.yaml", "sim/default_sim.yaml", "robot/smpl_humanoid.yaml",
    ])
    add(UNIHSI / "unihsi/data/cfg", "configs/unihsi", [
        "train/rlg/amp_humanoid_task_deep_layer.yaml",
        "humanoid_unified_interaction_scene_1.yaml",
    ])
    add(V2 / "weights/multi_target", "configs/dip_multi_target", ["args.json"])

    # Validate the entire bounded source list before copying anything.
    targets = set()
    for source, relative in plan:
        if not source.is_file() or source.is_symlink():
            raise ValueError("Missing/non-regular source: " + str(source))
        if source.suffix not in ALLOWED and source.name != "LICENSE":
            raise ValueError("Non-code asset forbidden: " + str(source))
        if source.stat().st_size > 2_000_000:
            raise ValueError("Unexpected large source: " + str(source))
        if relative in targets or (DEST / relative).exists():
            raise ValueError("Duplicate/existing target: " + str(relative))
        targets.add(relative)

    records = []
    for source, relative in plan:
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied = hashlib.sha256(target.read_bytes()).hexdigest()
        after = hashlib.sha256(source.read_bytes()).hexdigest()
        if before != copied or before != after:
            raise RuntimeError("Copy/source SHA mismatch: " + str(source))
        records.append(dict(source=str(source), destination=str(relative),
                            bytes=target.stat().st_size, sha256=copied))
    with (DEST / "MANIFEST.json").open("x") as out:
        json.dump(records, out, ensure_ascii=False, indent=2)
    summary = dict(status="source_only_copies_sha256_verified", file_count=len(records),
                   total_bytes=sum(r["bytes"] for r in records), weights_copied=0,
                   datasets_copied=0, media_copied=0, environments_copied=0,
                   source_files_modified=False, inference_performed=False,
                   standalone_runtime_claimed=False)
    with (DEST / "VERIFICATION.json").open("x") as out:
        json.dump(summary, out, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
