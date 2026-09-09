"""Corrected-contact R4-large training; frozen v1 sources remain unchanged.

Warm start imports weights and the explicit normalizer ONLY from the audited
50-step, interaction-only R4 checkpoint. V2 optimizer/RNG/step start afresh.
--resume is exact v2 continuation, not migration from the old broad-motion run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import stage3_joint_train as old_runner
from hsi.stage3_joint.corpus import JointCorpus
from hsi.stage3_joint.model import JointModelConfig, JointSequenceDenoiser
from hsi.stage3_sequence.diffusion import MotionDiffusion

digest, json_write, to_cpu = old_runner.digest, old_runner.json_write, old_runner.to_cpu
SCHEMA = "hsi.stage3_joint.training.v2"
TRAINING_POLICY_VERSION = "event_end_establish_geometric_stance_v2"
DEFAULT_WARM_START = REPO.parent / "runs/stage3_joint_20260908/large_joint_pilot01/joint_step000050.pt"
TRUSTED_WARM_START_SHA256 = "a3ede4954ff724a1d8e7a6ae1629db5768ad3e45715dec24e755aa2cf05fe1c0"
MIN_BROAD_TRAIN_WINDOWS = 13424
HISTORICAL_HELDOUT = frozenset(("028", "030", "031", "036", "043", "052", "055", "058", "067", "083"))


def training_api():
    # Deliberately lazy: CPU CLI/protocol checks do not need a geometry graph.
    from hsi.stage3_joint_v2.training import JointLossConfig, joint_training_loss, perturb_allowed_keyposes
    return JointLossConfig, joint_training_loss, perturb_allowed_keyposes


def source_binding(include_motion=True):
    if not include_motion:
        raise ValueError("V2 always binds the required broad motion loader")
    paths = []
    for directory in ("stage3_joint", "stage3_sequence", "stage3_joint_v2"):
        paths += sorted((REPO / "hsi" / directory).glob("*.py"))
    required = {"contacts.py", "objectives.py", "training.py", "sampling.py", "refine.py"}
    present = {p.name for p in paths if p.parent.name == "stage3_joint_v2"}
    if not required <= present:
        raise ValueError("V2 source package incomplete: " + str(sorted(required-present)))
    paths += [REPO / "experiments/stage3_joint_train.py", Path(__file__).resolve()]
    return {str(p.relative_to(REPO)): digest(p) for p in paths}


def verify_source_map(binding):
    if not isinstance(binding, dict) or not binding:
        raise ValueError("Checkpoint lacks explicit source bindings")
    for name, checksum in binding.items():
        path = (REPO / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(REPO) or not path.is_file() or digest(path) != checksum:
            raise ValueError("Checkpoint source changed or escaped repository: " + str(name))


def verify_manifest_root(root, checksum, name):
    if not root or not checksum or digest(Path(root) / "manifest.json") != checksum:
        raise ValueError(name + " manifest changed")


def verify_warm_start(payload, source_path, body_hash):
    if digest(source_path) != TRUSTED_WARM_START_SHA256:
        raise ValueError("Warm start must be the audited 50-step R4 checkpoint, not old broad-trained weights")
    if payload.get("schema") != old_runner.SCHEMA or payload.get("optimizer_steps") != 50:
        raise ValueError("Expected audited interaction-only R4 step50 checkpoint")
    if payload.get("motion_prepared_root") or payload.get("motion_manifest_sha256"):
        raise ValueError("V2 warm start cannot inherit broad-motion erroneous supervision")
    if payload.get("source_sha256") != old_runner.source_binding(include_motion=False):
        raise ValueError("Old checkpoint binding must exactly match frozen old runtime")
    verify_source_map(payload["source_sha256"])
    verify_manifest_root(payload.get("prepared_root"), payload.get("prepared_manifest_sha256"), "Warm-start interaction")
    if payload.get("body_asset_sha256") != body_hash:
        raise ValueError("Warm-start body asset differs")
    config = JointModelConfig.from_checkpoint(payload["model_config"])
    if config.layers != 16 or config.hidden_dim != 512 or not config.depth_expansion:
        raise ValueError("Expected the unchanged R4-large 16-layer D512 architecture")
    return config


def validate_corpus(dataset):
    if len(dataset.motion_train_indices) < MIN_BROAD_TRAIN_WINDOWS:
        raise ValueError("V2 requires all 13424+ sealed broad training windows, including for smoke tests")
    if len(dataset.interaction_train_indices) < 7:
        raise ValueError("Verified interaction training corpus unexpectedly shrank")
    motion = dataset.manifest.get("motion") or {}
    if motion.get("status") != "sealed_real_motion_windows" or not motion.get("completed_all_eligible_recordings"):
        raise ValueError("Only the complete sealed broad corpus may train V2")
    inherited = set(motion.get("heldout_base_scenes", []))
    actual = {str(dataset.records[i]["scene_id"]) for i in dataset.heldout_indices}
    train = {str(dataset.records[i]["scene_id"]) for i in dataset.train_indices}
    if not HISTORICAL_HELDOUT <= inherited or not HISTORICAL_HELDOUT <= actual or train & inherited:
        raise ValueError("Historical heldout scene groups were lost or leaked into training")
    return dict(broad_train_windows=len(dataset.motion_train_indices),
        interaction_train_records=len(dataset.interaction_train_indices),
        heldout_records=len(dataset.heldout_indices), preserved_heldout_scenes=sorted(inherited))


def fixed_evaluation_indices(dataset):
    """Exactly four heldout examples per source; prefer distinct base scenes."""
    result = []
    for source in (0, 1):
        candidates = sorted((i for i in dataset.heldout_indices if dataset.lookup[i][0] == source),
                            key=lambda i: dataset.records[i]["case_id"])
        chosen, seen = [], set()
        for index in candidates:
            scene = dataset.records[index]["scene_id"]
            if scene not in seen:
                chosen.append(index); seen.add(scene)
            if len(chosen) == 4:
                break
        chosen += [i for i in candidates if i not in chosen][:4-len(chosen)]
        if len(chosen) != 4:
            raise ValueError("Need four real heldout examples from each corpus")
        result += chosen
    return result


def policy_for(args):
    loss_config, _, _ = training_api()
    return dict(version=TRAINING_POLICY_VERSION, seed=args.seed, batch_size=1, lr=args.lr, weight_decay=.01,
        gradient_clip_norm=1., physics_warmup=args.physics_warmup,
        feedback_rounds="2 every fourth step, otherwise 1", augmentation_probability=.5,
        augmentation_max_degrees=3., gradient_accumulation=args.accumulate,
        interaction_probability=args.interaction_probability, loss_config=asdict(loss_config()))


def validate_args(args):
    if (not 1 <= args.steps <= 100000 or args.batch_size != 1 or not 1 <= args.accumulate <= 16
            or not np.isfinite(args.lr) or args.lr <= 0 or not 0 < args.interaction_probability < 1
            or min(args.checkpoint_every, args.evaluate_every, args.log_every, args.cpu_threads, args.physics_warmup) < 1):
        raise ValueError("Invalid bounded V2 training/mixture/reporting settings")
    if args.device.startswith("cuda"):
        visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
        if len(visible) != 1:
            raise ValueError("Launch V2 with exactly one explicitly visible GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU fallback")


def verify_resume(payload, state):
    if (payload.get("schema") != SCHEMA or payload.get("training_policy_version") != TRAINING_POLICY_VERSION
            or payload.get("policy_version") != TRAINING_POLICY_VERSION):
        raise ValueError("--resume accepts exact V2 checkpoints only, not old-loss continuation")
    for name in ("source_sha256", "prepared_manifest_sha256", "motion_manifest_sha256", "body_asset_sha256", "training_policy", "evaluation_cases"):
        if payload.get(name) != state[name]:
            raise ValueError("Exact V2 resume changed: " + name)
    verify_source_map(payload["source_sha256"])
    verify_manifest_root(payload.get("prepared_root"), payload.get("prepared_manifest_sha256"), "Previous interaction")
    verify_manifest_root(payload.get("motion_prepared_root"), payload.get("motion_manifest_sha256"), "Previous broad")
    required = {"optimizer_state", "rng_state", "generator_state", "torch_rng_state", "cuda_rng_state", "optimizer_steps"}
    if not required <= payload.keys() or not payload.get("optimizer_state_saved"):
        raise ValueError("Exact resume requires full optimizer and RNG state")
    if torch.device(payload["device"]).type != torch.device(state["device"]).type:
        raise ValueError("Exact resume cannot change CPU/GPU RNG backend")
    return JointModelConfig.from_checkpoint(payload["model_config"])


def verify_active_inputs(state):
    if source_binding() != state["source_sha256"]:
        raise RuntimeError("Training sources changed; refusing mixed-source checkpoint")
    for root, checksum, name in ((state["prepared_root"], state["prepared_manifest_sha256"], "Interaction"),
                                (state["motion_prepared_root"], state["motion_manifest_sha256"], "Broad")):
        verify_manifest_root(root, checksum, name)
    if digest(state["body_asset"]) != state["body_asset_sha256"]:
        raise RuntimeError("Body asset changed during training")
    if digest(state["source_checkpoint"]) != state["source_checkpoint_sha256"]:
        raise RuntimeError("Starting checkpoint changed during training")


def save_checkpoint(output, step, model, diffusion, optimizer, rng, generator, state):
    verify_active_inputs(state)
    device = next(model.parameters()).device
    payload = dict(state, optimizer_steps=step, model_config=asdict(model.config),
        model_state=to_cpu(model.state_dict()), diffusion_steps=diffusion.steps,
        diffusion_state=to_cpu(diffusion.state_dict()), optimizer_state=to_cpu(optimizer.state_dict()),
        rng_state=rng.bit_generator.state, generator_state=generator.get_state().cpu(),
        torch_rng_state=torch.random.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        optimizer_state_saved=True, motion_release_authorized=False)
    path = Path(output) / f"joint_step{step:06d}.pt"
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary); temporary.replace(path)
    receipt = dict(path=str(path), sha256=digest(path), optimizer_steps=step)
    json_write(Path(output) / "latest_checkpoint.json", receipt)
    return receipt


def evaluate(model, diffusion, dataset, indices, device, seed):
    """Fixed local noise, no GT-time conditioning; preserve training RNG."""
    _, loss_fn, _ = training_api()
    was_training = model.training
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    model.eval(); rows = []
    try:
        with torch.no_grad():
            for order, index in enumerate(indices):
                batch = dataset.batch([index], keypose_counts=[1+order%3], device=device)
                generator = torch.Generator(device=device).manual_seed(seed+index)
                result = loss_fn(model, diffusion, batch, generator=generator)
                rows.append(dict(index=index, case_id=dataset.records[index]["case_id"],
                    corpus="interaction" if dataset.lookup[index][0] == 0 else "broad_motion",
                    loss=float(result["loss"]), losses={k: float(v) for k, v in result["losses"].items()},
                    predicted_durations=result["schedule"].durations.cpu().tolist(),
                    supervised_durations=batch.timing.cpu().tolist(), positive_contacts=result["positive_contact_count"]))
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        model.train(was_training)
    return rows


def run(args):
    validate_args(args)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory, including exact resume")
    prepared, motion_root, body_path = (Path(p).resolve(strict=True) for p in (args.prepared, args.motion_prepared, args.body_asset))
    dataset = JointCorpus(prepared, body_path, device=device, motion_root=motion_root)
    corpus_audit = validate_corpus(dataset)
    eval_indices = fixed_evaluation_indices(dataset)
    source_path = Path(args.resume or args.warm_start or DEFAULT_WARM_START).resolve(strict=True)
    state = dict(schema=SCHEMA, training_policy_version=TRAINING_POLICY_VERSION, policy_version=TRAINING_POLICY_VERSION,
        source_sha256=source_binding(), source_checkpoint=str(source_path), source_checkpoint_sha256=digest(source_path),
        prepared_root=str(prepared), prepared_manifest_sha256=digest(prepared/"manifest.json"),
        motion_prepared_root=str(motion_root), motion_manifest_sha256=digest(motion_root/"manifest.json"),
        body_asset=str(body_path), body_asset_sha256=digest(body_path), training_policy=policy_for(args),
        train_indices=list(dataset.train_indices), heldout_indices=list(dataset.heldout_indices), corpus_audit=corpus_audit,
        evaluation_indices=eval_indices, evaluation_cases=[dataset.records[i]["case_id"] for i in eval_indices],
        optimizer_resumed=bool(args.resume), exact_data_resume=bool(args.resume), seed=args.seed,
        normalizer="exactly retained audited motion-only statistics; conditions metric, no second normalization",
        timestamp_conditioning=False, fps=15., variable_keypose_counts=[1, 2, 3],
        text_condition="null text embedding; observed coarse semantic IDs only",
        all_denoiser_parameters_trainable=True, total_requested_steps=args.steps,
        command=sys.argv, device=str(device), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        motion_release_authorized=False, status="preflight")
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    config = (verify_resume(payload, state) if args.resume else verify_warm_start(payload, source_path, state["body_asset_sha256"]))
    torch.random.default_generator.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    model = JointSequenceDenoiser(config).to(device)
    # No depth expansion, shape remapping, partial loading, or inherited optimizer.
    model.load_state_dict(payload["model_state"], strict=True)
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("All R4 parameters must remain trainable")
    diffusion = MotionDiffusion(len(payload["diffusion_state"]["alpha_bar"]),
        mean=payload["diffusion_state"]["mean"], scale=payload["diffusion_state"]["scale"]).to(device)
    diffusion.load_state_dict(payload["diffusion_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed+1)
    start_step = 0
    if args.resume:
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload["optimizer_steps"])
        rng.bit_generator.state = payload["rng_state"]
        generator.set_state(payload["generator_state"])
        torch.random.set_rng_state(payload["torch_rng_state"])
        if device.type == "cuda":
            if payload["cuda_rng_state"] is None:
                raise ValueError("Exact GPU resume cannot invent a missing CUDA RNG")
            torch.cuda.set_rng_state(payload["cuda_rng_state"], device)
        migration = payload["warm_start_migration"]
    else:
        # Reset AFTER construction/load: constructor random draws are not training RNG.
        torch.random.default_generator.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(args.seed)
        migration = dict(kind="weights_and_normalizer_only_from_audited_R4_step50",
            original_optimizer_steps=50, new_optimizer_steps=0, optimizer_state_loaded=False,
            old_rng_loaded=False, model_state_strict=True, architecture_changed=False,
            source_checkpoint=str(source_path), source_checkpoint_sha256=state["source_checkpoint_sha256"],
            prior_broad_erroneous_supervision_inherited=False, exact_old_training_resume=False)
    del payload
    state.update(architecture=config.architecture, model_config=asdict(config), loss_config=state["training_policy"]["loss_config"],
        trainable_parameters=sum(p.numel() for p in model.parameters()), starting_optimizer_steps=start_step,
        warm_start_migration=migration, status="training")
    stop_file = Path(args.request_stop_file).resolve() if args.request_stop_file else output/"REQUEST_STOP"
    state["request_stop_file"] = str(stop_file)
    output.mkdir(parents=True)
    json_write(output/"run.json", state); json_write(output/"warm_start_migration.json", migration)
    _, loss_fn, augment = training_api()
    model.train(); started = time.perf_counter(); last_saved = None; completed_step = start_step
    try:
        with (output/"train.jsonl").open("x", buffering=1) as log:
            for step in range(start_step+1, start_step+args.steps+1):
                tick = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                physics_scale = min(1., .1+.9*step/args.physics_warmup)
                row = dict(step=step, indices=[], keypose_counts=[], actual_keypose_counts=[], future_frames=[], case_ids=[],
                    losses={}, loss=0., positive_contacts=0, known_contacts=0, predicted_durations=[], supervised_durations=[])
                for _ in range(args.accumulate):
                    index = dataset.choose(rng, args.interaction_probability); count = int(rng.integers(1, 4))
                    batch = dataset.batch([index], keypose_counts=[count], device=device)
                    if rng.random() < .5:
                        batch = replace(batch, condition=augment(batch.condition, batch.body, generator=generator))
                    result = loss_fn(model, diffusion, batch, generator=generator, physics_scale=physics_scale,
                                     feedback_rounds=2 if step%4 == 0 else 1)
                    (result["loss"]/args.accumulate).backward()
                    row["loss"] += float(result["loss"].detach())/args.accumulate
                    for name, value in result["losses"].items():
                        row["losses"][name] = row["losses"].get(name, 0.)+float(value.detach())/args.accumulate
                    row["indices"].append(index); row["keypose_counts"].append(count)
                    row["case_ids"].append(batch.metadata[0]["case_id"])
                    row["actual_keypose_counts"] += batch.condition.keypose_mask.sum(1).cpu().tolist()
                    row["future_frames"] += batch.condition.total_frames.cpu().tolist()
                    row["positive_contacts"] += result["positive_contact_count"]; row["known_contacts"] += result["contact_known_count"]
                    row["predicted_durations"] += result["schedule"].durations.cpu().tolist()
                    row["supervised_durations"] += batch.timing.cpu().tolist()
                    del result, batch
                audit = old_runner.gradient_audit(model) if step <= start_step+2 or step%args.log_every == 0 else None
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); completed_step = step
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row.update(grad_norm=float(norm), physics_scale=physics_scale, gradient_accumulation=args.accumulate,
                    gradient_audit=audit, feedback_rounds=2 if step%4 == 0 else 1,
                    seconds=time.perf_counter()-tick, elapsed_seconds=time.perf_counter()-started,
                    cuda_peak_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                if step <= start_step+2 or step%args.log_every == 0:
                    print(json.dumps({k: row[k] for k in ("step", "loss", "seconds", "cuda_peak_mib")}), flush=True)
                    json_write(output/"status.json", dict(status="training", **row))
                stopping = stop_file.is_file()
                # A stop request saves BEFORE expensive validation and exits at a
                # complete optimizer-step boundary, never mid-accumulation.
                if not stopping and step%args.evaluate_every == 0:
                    rows = evaluate(model, diffusion, dataset, eval_indices, device, args.seed+7000)
                    json_write(output/f"heldout_loss_step{step:06d}.json", dict(step=step,
                        kind="fixed_noise_supervised_loss_not_generative_quality", examples=rows))
                if stopping or step%args.checkpoint_every == 0 or step == start_step+args.steps:
                    last_saved = save_checkpoint(output, step, model, diffusion, optimizer, rng, generator, state)
                    print(json.dumps({"checkpoint": last_saved}), flush=True)
                if stopping:
                    state["status"] = "stopped_clean_at_optimizer_boundary"
                    state["stop_request_observed"] = True
                    break
            else:
                state["status"] = "completed_requested_optimizer_steps"
        state.update(optimizer_steps=completed_step, latest_checkpoint=last_saved, elapsed_seconds=time.perf_counter()-started)
        verify_active_inputs(state)
        json_write(output/"run.json", state)
        json_write(output/"status.json", dict(status=state["status"], optimizer_steps=completed_step, latest_checkpoint=last_saved))
        return state
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", last_completed_optimizer_steps=completed_step,
                     latest_checkpoint=last_saved, elapsed_seconds=time.perf_counter()-started)
        json_write(output/"run.json", state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--motion-prepared", required=True)
    p.add_argument("--body-asset", required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--warm-start", help="Only the pinned audited R4 step50 checkpoint")
    source.add_argument("--resume", help="Exact V2 optimizer/RNG/source/data continuation")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=50000, help="Additional V2 optimizer steps (also for resume)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=3909)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--evaluate-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--physics-warmup", type=int, default=500)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--interaction-probability", type=float, default=.2)
    p.add_argument("--request-stop-file", help="Existence requests clean checkpoint+exit after next full optimizer step")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
