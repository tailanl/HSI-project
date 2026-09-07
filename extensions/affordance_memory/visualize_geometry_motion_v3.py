"""Actual Stage3 v2 lineage + unchanged numeric-overview/SEG rendering.

V1/V2 renderer files and input receipts are never rewritten. The v1 evidence
binder is cloned privately with six exact schema/provider string changes.
The frozen v2 numeric camera, scene masks and saved-vertex renderer are reused
without changing their algorithms. Only source closure and report metadata
are extended. No model loading, re-skinning or network-checkpoint checksum.

Import/load_inputs and tests are CPU-only. render() creates an explicit EGL
context; --gpu and --egl-device are separate physical/requested settings.
"""
from __future__ import annotations

import argparse
import ast
import copy
import inspect
from pathlib import Path
import types

import visualize_geometry_motion as retained_base
import visualize_geometry_motion_v2 as retained_overview
import visualize_stage3_guidance_comparison as comparison
from memory_common import artifact, require, verified

SOURCE = Path(__file__).resolve()
SCHEMA = "p555.actual_geometry_motion_overview_visualization.v3"
PINS = {
    Path(retained_base.__file__): "7f09899b933842ed705bec9c7e2fbbb6a37afb86020640a9613cc1854a95fa25",
    Path(retained_overview.__file__): "8c90cbedafd939b4db12a711dea212abd116e95407b00d886b784865fce40cbb",
    Path(comparison.__file__): "aa3417532d7bc104284e7c48660808574233584b475bfef13fe08ec4f2288d93",
}
SCHEMA_REPLACEMENTS = {
    "p555.geometry_stage3_execution.v1": "p555.geometry_stage3_execution.v2",
    "p555.geometry_stage3_motion_evaluation.v1": "p555.geometry_stage3_motion_evaluation.v2",
    "p555.geometry_stage3_launch.v1": "p555.geometry_stage3_launch.v2",
    "p555.geometry_stage3_adapter.v1": "p555.geometry_stage3_adapter.v2",
    "p555.current_geometry_stage3_packet.v1": "p555.current_geometry_stage3_packet.v2",
    "p555_current_geometry_ik": "p555_current_geometry_ik_v2",
}


def actual_sources():
    # Keep the v1 binder at index 1 for the retained v2 render's exact pin check.
    records = [artifact(SOURCE), *[artifact(path) for path in PINS]]
    for row, (path, checksum) in zip(records[1:], PINS.items()):
        require(row["sha256"] == checksum, "Frozen visualization dependency drift: " + str(path))
    return records


def clone_functions(module, *, overrides=None):
    """No global/sys.modules monkeypatch and no mutation of retained functions."""
    scope = dict(vars(module))
    if overrides:
        scope.update(overrides)
    for name, value in vars(module).items():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(module):
            fn = types.FunctionType(value.__code__, scope, value.__name__, value.__defaults__, value.__closure__)
            fn.__kwdefaults__ = value.__kwdefaults__
            scope[name] = fn
    return scope


def private_pairing(scope):
    """Replace exact AST string constants, never edit passed-in receipts."""
    tree = ast.parse(inspect.getsource(retained_base.validate_pairing))
    counts = {key:0 for key in SCHEMA_REPLACEMENTS}
    class Schemas(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in SCHEMA_REPLACEMENTS:
                counts[node.value] += 1
                return ast.copy_location(ast.Constant(SCHEMA_REPLACEMENTS[node.value]), node)
            return node
    tree = Schemas().visit(tree)
    require(all(count == 1 for count in counts.values()), "Frozen v1 schema adaptation anchor drift")
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), scope)
    return scope["validate_pairing"]


def checked_load(execution, evaluation, original_load):
    reader = comparison.Reader()
    source_execution = artifact(execution)
    executed = reader.json(source_execution)
    launch = reader.json(executed["launch"])
    mode = launch["affordance"]
    require(mode in {"off", "on"}, "Unknown actual affordance mode")
    # Full v2 source/net-proof/energy/34-gate checks, without importing a model
    # or rehashing checkpoint files. This reports evidence, never admits credit.
    case = comparison.load_case(execution,evaluation,mode,reader,comparison.retained_gate_function(reader))
    state = original_load(execution,evaluation)
    require(state["bindings"]["execution"] == case["source_execution"]
            and state["bindings"]["evaluation"] == case["source_evaluation"]
            and state["bindings"]["adapter"] == case["launch"]["adapter"]
            and state["bindings"]["packet"] == case["launch"]["packet"]
            and state["bindings"]["camera"] == case["original_camera"], "Overview/comparison evidence differed")
    for key in ("network_preflight", "network_postflight"):
        state["bindings"][key] = case["execution"][key]
    state["bindings"]["runtime_manifest"] = case["execution"]["outputs"]["run_manifest.json"]
    state["bindings"]["v3_visualizer_source"] = artifact(SOURCE)
    state["bindings"]["v2_network_gate_log_validator_source"] = artifact(comparison.__file__)
    state["v3_validation"] = {
        "input_execution_schema": "p555.geometry_stage3_execution.v2",
        "input_evaluation_schema": "p555.geometry_stage3_motion_evaluation.v2",
        "actual_provider": "p555_current_geometry_ik_v2",
        "original_34_release_gates_recomputed_from_evaluation": True,
        "actual_energy_log_full22_and_scheduler_phase_verified": True,
        "network_pre_post_receipt_lineage_and_manifest_verified": True,
        "network_checkpoint_files_rehashed_by_visualizer": False,
        "network_integrity_scope": "actual execution's sealed independently rehashed pre/post receipts; not a new model hash run or same-UID authentication",
        "body_identity_sha_rechecked_by_retained_binder": True,
        "geometry_overview_camera_and_segmentation_algorithms_unchanged_from_v2": True,
        "v1_v2_source_or_input_receipts_rewritten": False,
        "actual_guidance_activation": case["energy"],
        "positive_credit": 0,
    }
    reader.recheck()
    return state


def private_base():
    actual_sources()
    scope = clone_functions(retained_base)
    private_pairing(scope)
    original_load = scope["load_inputs"]
    def current_load(execution,evaluation):
        return checked_load(execution,evaluation,original_load)
    scope["load_inputs"] = current_load
    module = types.ModuleType("p555_private_v2_lineage_motion_binder")
    module.__dict__.update(scope)
    return module


def replace_once(source, old, new):
    require(source.count(old) == 1, "Frozen v2 render adaptation anchor drift")
    return source.replace(old,new)


def private_overview():
    base = private_base()
    scope = clone_functions(retained_overview,
        overrides={"base":base, "SCHEMA":SCHEMA, "__file__":str(SOURCE),
                   "actual_sources":actual_sources, "POLICY":copy.deepcopy(retained_overview.POLICY)})
    source = inspect.getsource(retained_overview.render)
    source = replace_once(source,"sources = [artifact(__file__), artifact(base.__file__)]", "sources = actual_sources()")
    source = replace_once(source,'write_once(output/"receipt.json",result)',
                          'result.update(state["v3_validation"])\n    write_once(output/"receipt.json",result)')
    exec(compile(source,str(SOURCE),"exec"),scope)
    return scope


def load_inputs(execution,evaluation):
    return private_base().load_inputs(execution,evaluation)


def render(execution,evaluation,output,*,animation="gif",gpu=5,egl_device=5):
    # Rendering stays opt-in; the host/controller owns GPU scheduling/PMON.
    require(not Path(output).exists(), "Refuse to overwrite visualization")
    return private_overview()["render"](execution,evaluation,output,animation=animation,gpu=gpu,egl_device=egl_device)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution",type=Path,required=True)
    parser.add_argument("--evaluation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--animation",choices=("none","gif"),default="gif")
    parser.add_argument("--gpu",type=int,default=5)
    parser.add_argument("--egl-device",type=int,default=5)
    args=parser.parse_args()
    print(render(args.execution,args.evaluation,args.output,animation=args.animation,gpu=args.gpu,egl_device=args.egl_device),flush=True)


if __name__ == "__main__":
    main()
