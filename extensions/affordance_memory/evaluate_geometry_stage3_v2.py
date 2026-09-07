"""V2 admission wrapper around frozen full P519/P523 motion evaluation.

No metric, fullmesh reconstruction or publication threshold is changed.
V1 executions remain evaluable by their original, untouched v1 evaluator.
"""
import argparse
import inspect
from pathlib import Path
import types

import evaluate_geometry_stage3 as retained
import run_geometry_stage3_v2 as runner
from memory_common import artifact, require

SOURCE_PATH = Path(__file__).resolve(strict=True)
RETAINED_PATH = Path(retained.__file__).resolve(strict=True)
SCHEMA = "p555.geometry_stage3_motion_evaluation.v2"
RETAINED_SHA256 = "b3e49c5c0ae3751982b37f3d5b177fdbd104227de3eee9436ace69e9b37393c3"


def private_evaluator():
    require(artifact(RETAINED_PATH)["sha256"] == RETAINED_SHA256, "Frozen v1 fullmesh evaluator drift")
    runner.require_sources()
    scope = dict(vars(retained))
    scope.update(__file__=str(SOURCE_PATH), SCHEMA=SCHEMA, load_inputs=runner.load_inputs,
                 validate_affordance_runtime=runner.validate_affordance_runtime)
    for name, value in vars(retained).items():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(retained):
            fn = types.FunctionType(value.__code__, scope, value.__name__, value.__defaults__, value.__closure__)
            fn.__kwdefaults__ = value.__kwdefaults__
            scope[name] = fn
    source = inspect.getsource(retained.load_execution)
    source = runner._replace(source, '"p555.geometry_stage3_execution.v1"', repr(runner.SCHEMA))
    source = runner._replace(source, '"p555.geometry_stage3_launch.v1"', repr(runner.LAUNCH_SCHEMA))
    exec(compile(source, retained.__file__, "exec"), scope)
    original_load = scope["load_execution"]
    def strict_load(path):
        state = original_load(path)
        paths = [row["path"] for row in state["launch"]["sources"]]
        require(len(paths) == len(set(paths)), "Duplicate generation source binding")
        required = [runner.SOURCE_PATH, runner.compiler.__file__, runner.compiler.neutral.__file__,
                    runner.HERE / "stage3_affordance_guidance_v2.py", runner.RETAINED_PATH]
        for path in required:
            require(artifact(path) in state["launch"]["sources"], "V2 generation omitted an actual implementation source")
        runner.validate_network_proof(state["receipt"], state["launch"])
        return state
    scope["load_execution"] = strict_load
    source = inspect.getsource(retained.evaluate)
    source = runner._replace(source, "Path(__file__), EVALUATOR", 'Path(__file__), HERE / "evaluate_geometry_stage3.py", EVALUATOR')
    for name in ("run_geometry_stage3", "compile_geometry_stage3", "neutral_seed"):
        source = runner._replace(source, f'HERE / "{name}.py"', f'HERE / "{name}_v2.py"')
    source = runner._replace(source, '"actual_provider": "p555_current_geometry_ik"', f'"actual_provider": {runner.PROVENANCE!r}')
    exec(compile(source, retained.__file__, "exec"), scope)
    return scope


def evaluate(execution_path, output, *, device="cuda"):
    return private_evaluator()["evaluate"](execution_path, output, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = evaluate(args.execution, args.output, device=args.device)
    print({"motion_publishable": result["motion_publishable"], "failed_gates": result["failed_release_gates"]})
    raise SystemExit(0 if result["motion_publishable"] else 2)
