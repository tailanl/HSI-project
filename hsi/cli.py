"""Lazy CLI entrypoints: heavyweight stages own separate explicit workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsi import __version__
from hsi.common.artifacts import read_json


def parser():
    root = argparse.ArgumentParser(prog="hsi", description="Integrated human–scene interaction")
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("check-scene", help="Verify an immutable Stage1A scene snapshot")
    inspect.add_argument("--scene", type=Path, required=True)
    build = sub.add_parser('scene-build', help='Stage1A: raw scene to immutable FINAL scene understanding')
    build.add_argument('--scene-id', required=True)
    for flag in ('mesh','occupancy','output'):
        build.add_argument('--'+flag, type=Path, required=True)
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument('--perception-runtime', type=Path)
    source.add_argument('--atomic-receipt', type=Path, help='Verified current scene-only segmentation receipt')
    build.add_argument('--gpu', type=int, default=0)
    plan = sub.add_parser("plan", help="Fresh Stage1B text planning and cold NavMesh")
    plan.add_argument("--scene", type=Path, required=True)
    plan.add_argument("--instruction", required=True)
    plan.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), required=True)
    plan.add_argument("--smplx", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--qwen-endpoint", required=True, help="Explicit OpenAI-compatible Qwen URL ending in /v1")
    plan.add_argument("--qwen-model", required=True)
    plan.add_argument("--api-key-env", help="Environment variable name, never the key value")
    view = sub.add_parser("views", help="Current mask/depth checks with optional saved camera hints")
    view.add_argument("--stage1", type=Path, required=True)
    view.add_argument("--render", type=Path, required=True)
    view.add_argument("--output", type=Path, required=True)
    view.add_argument("--gpu", type=int, default=0)
    view.add_argument("--mode", choices=("full", "memory"), default="full")
    view.add_argument("--facts", type=Path)
    view.add_argument("--memory-store", type=Path)
    job = sub.add_parser("image-job", help="Bind the selected camera, crop and fresh target description")
    for flag in ("stage1", "view", "description", "output"):
        job.add_argument("--" + flag, type=Path, required=True)
    job.add_argument("--case-id", required=True)
    job.add_argument("--seed", type=int, required=True)
    image = sub.add_parser("image", help="Direct H3 native T=1 image inference, no server queue")
    for flag in ("job", "output", "comfy", "models"):
        image.add_argument("--" + flag, type=Path, required=True)
    image.add_argument("--gpu", type=int, default=0)
    recover = sub.add_parser("recover", help="HybrIK-X image articulation; not a physical keypose pass")
    for flag in ("image", "h3-receipt", "camera", "stage1-bundle", "config", "runtime", "output"):
        recover.add_argument("--" + flag, type=Path, required=True)
    recover.add_argument("--case-id", required=True)
    recover.add_argument("--gpu", type=int, default=0)
    stage2 = sub.add_parser('stage2', help='Complete current image keypose pipeline and strict final review')
    for flag in ('stage1','render','descriptions','output','comfy','models','recovery-runtime','segmentation'):
        stage2.add_argument('--'+flag, type=Path, required=True)
    stage2.add_argument('--gpu', type=int, default=0)
    stage2.add_argument('--seed', type=int, default=55000)
    stage2.add_argument('--mode', choices=('full','memory'), default='full')
    stage2.add_argument('--facts', type=Path)
    stage2.add_argument('--memory-store', type=Path)
    for command in (build, stage2):
        command.add_argument('--qwen-endpoint', required=True)
        command.add_argument('--qwen-model', required=True)
        command.add_argument('--api-key-env', help='Environment variable NAME; never a secret value')
    compile_motion = sub.add_parser('compile-motion', help='Verified Stage2 -> current route/contact motion packet')
    for flag in ('stage2','output','runtime-body','render-body'):
        compile_motion.add_argument('--'+flag, type=Path, required=True)
    motion = sub.add_parser('motion', help='One current ReMoGen run; output still requires full-mesh evaluation')
    for flag in ('adapter','output','runtime'):
        motion.add_argument('--'+flag, type=Path, required=True)
    motion.add_argument('--gpu', type=int, default=0)
    motion.add_argument('--seed', type=int, default=55000)
    motion.add_argument('--affordance', choices=('on','off'), default='on')
    motion.add_argument('--dry-run', action='store_true')
    evaluate = sub.add_parser('evaluate-motion', help='All-frame joint-driven SMPL-X and the fixed 34 physical release checks')
    for flag in ('execution','output','official-sources','mean-hands'):
        evaluate.add_argument('--'+flag, type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "check-scene":
        from hsi.stage1.scene import load_final_scene
        value = load_final_scene(args.scene)
        print(json.dumps({"scene_id": value.publication["scene_id"], "verified": True,
                          "publication": str(value.publication_path)}))
    elif args.command == 'scene-build':
        from hsi.stage1 import build_scene
        from hsi.stage1.qwen import QwenTextClient
        from hsi.stage1.perception.runtime import PerceptionRuntime
        runtime = PerceptionRuntime(**read_json(args.perception_runtime)) if args.perception_runtime else None
        result = build_scene(args.scene_id, args.mesh, args.occupancy, args.output,
            qwen_client=QwenTextClient(args.qwen_endpoint,args.qwen_model,api_key_env=args.api_key_env),
            perception_runtime=runtime, atomic_receipt=args.atomic_receipt, gpu=args.gpu)
        print(str(result))
    elif args.command == "plan":
        from hsi.stage1.pipeline import run
        from hsi.stage1.navigation import Stage1Runtime
        from hsi.stage1.qwen import QwenTextClient
        result = run(args.scene, args.instruction, args.start, args.output,
            qwen=QwenTextClient(args.qwen_endpoint, args.qwen_model, api_key_env=args.api_key_env),
            runtime=Stage1Runtime(args.smplx))
        return 0 if result.get("stage2_sequence_handoff_allowed") is True else 2
    elif args.command == "views":
        from hsi.stage2.views import select_views
        result = select_views(args.stage1, args.render, args.output, gpu=args.gpu, mode=args.mode,
                              facts_path=args.facts, store_root=args.memory_store)
        print(json.dumps({"output": str(args.output), "receipt": str(result)}))
    elif args.command == "image-job":
        from hsi.stage2.image_job import run
        run(stage1_path=args.stage1, view_path=args.view, description_path=args.description,
            output=args.output, case_id=args.case_id, seed=args.seed)
    elif args.command == "image":
        from hsi.stage2.h3 import generate
        generate(args.job, args.output, comfy_root=args.comfy, models_root=args.models, gpu=args.gpu)
    elif args.command == "recover":
        from hsi.stage2.recovery import recover
        recover(args.image, args.h3_receipt, args.camera, args.stage1_bundle, args.config, args.output,
                runtime=read_json(args.runtime), gpu=args.gpu, case_id=args.case_id)
    elif args.command == 'stage2':
        from hsi.stage2.pipeline import run
        from hsi.stage1.qwen import QwenTextClient
        result = run(args.stage1,args.render,args.descriptions,args.output,
            qwen_client=QwenTextClient(args.qwen_endpoint,args.qwen_model,api_key_env=args.api_key_env),
            comfy_root=args.comfy,models_root=args.models,recovery_runtime=read_json(args.recovery_runtime),
            segmentation=args.segmentation,gpu=args.gpu,seed=args.seed,mode=args.mode,
            facts_path=args.facts,store_root=args.memory_store)
        return 0 if result.get('verified_stage2_keypose') is True else 2
    elif args.command == 'compile-motion':
        from hsi.stage3.compiler import compile
        compile(args.stage2,args.output,runtime_body=args.runtime_body,render_body=args.render_body)
    elif args.command == 'motion':
        import os
        import sys
        if args.gpu < 0 or 'torch' in sys.modules:
            raise ValueError('Select one nonnegative GPU and launch motion in a fresh process')
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        from hsi.stage3.runtime import run
        from hsi.stage3.model_runtime import MotionRuntime
        run(args.adapter,args.output,runtime=MotionRuntime.from_mapping(read_json(args.runtime)),
            seed=args.seed,affordance=args.affordance=='on',dry_run=args.dry_run)
    elif args.command == 'evaluate-motion':
        from hsi.stage3.evaluation import evaluate
        result = evaluate(args.execution,args.output,
            official_sources={k:Path(v) for k,v in read_json(args.official_sources).items()},
            mean_hands_path=args.mean_hands,device='cpu')
        print(json.dumps(result,ensure_ascii=False))
    return 0
