"""One fresh ID-only query sequence over an already completed Stage1A scene."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from hsi.common.artifacts import artifact, read_sealed, verified, write_once
from .scene import load_final_scene
from .binding import bind
from . import planning, descriptions
from .navigation import execute_route, check_stage1_handoff

def run(scene_path, instruction, start, output, *, qwen, runtime):
    started = time.monotonic()
    fixed = load_final_scene(Path(scene_path))
    geometry_path, geometry, semantics = fixed.geometry_path, fixed.geometry, fixed.semantics
    output = Path(output)
    if geometry["task_instruction_read"] is not False or geometry["start_state_read"] is not False:
        raise ValueError("Scene snapshot must precede the human query")
    if not instruction.strip() or len(start) != 2 or not np.isfinite(start).all():
        raise ValueError("Invalid text or start state")
    output.mkdir(parents=True, exist_ok=False)
    inventory = planning.inventory(semantics, geometry)
    prompt = """Plan EVERY action in the user's request, in order, using this fixed scene inventory as evidence.
Select supplied instance IDs, never invent furniture or choose unresolved masks. Keep multiple eligible alternatives for an unspecified choice; a separate geometric planner resolves reachability.
Do not substitute furniture categories, omit earlier or later interactions, or reduce a multi-object request to one target. WALK followed by SIT at the same object can be separate steps. Reference IDs are only distinct other objects explicitly required by a relation, never the target itself. If no verified target exists, retain the action with empty target_ids and explain.
Provide a concise intent and target-selection reason for each step. No coordinates, numbers describing geometry, angles, paths or bounding boxes.
SCENE_INVENTORY=""" + json.dumps(inventory, ensure_ascii=False) + "\nUSER_REQUEST=" + json.dumps(instruction, ensure_ascii=False)
    plan_path = output / "semantic_plan.json"
    queue_seconds = 0.0
    raw = qwen.call(prompt, planning.PLAN_SCHEMA, plan_path, max_tokens=1600)
    compiled, mapping, unsupported = planning.compile_supported_sequence(raw, inventory)
    compile_path = output / "semantic_compilation.json"
    write_once(compile_path, {"schema": "p550.ordered_interaction_semantic_compilation.v1",
        "source_actual_qwen_call": artifact(plan_path), "raw_plan": raw, "compiled_plan": compiled,
        "mapping": mapping, "unsupported_actions": unsupported, "source": artifact(__file__),
        "fixed_geometry": artifact(geometry_path), "qwen_computed_coordinates": False}, seal=True)
    result = {"schema": "p550.stage1_interaction_sequence_execution.v1", "scene_id": geometry["scene_id"],
        "instruction": instruction, "initial_start_world_xy_m": list(start), "source_fixed_geometry": artifact(geometry_path),
        "source_fixed_semantics": geometry["source_semantics"], "source_final_scene": artifact(fixed.publication_path),
        "semantic_plan": artifact(plan_path),
        "semantic_compilation": artifact(compile_path), "interaction_steps": [], "unsupported_actions": unsupported,
        "qwen_numeric_geometry": False, "qwen_plan_cooperative_queue_seconds": queue_seconds,
        "new_scene_understanding_per_interaction": False, "source": artifact(__file__),
        "approach_route_stitching_policy": "next NavMesh segment starts at prior free approach endpoint; physical sit/stand transitions remain Stage3",
        "stage3_motion_generated": False}
    if unsupported or raw["needs_clarification"]:
        result.update(status="semantic_sequence_retained_executor_unsupported" if unsupported else "semantic_clarification_required",
            stage2_sequence_handoff_allowed=False)
    else:
        current_start = list(start)
        for index, step in enumerate(compiled["steps"]):
            child_root = output / "interactions" / f"step_{index:02d}"
            child_root.mkdir(parents=True)
            derived_plan = child_root / "derived_semantic_step.json"
            write_once(derived_plan, {"schema": "p550.actual_qwen_sequence_step_binding.v1",
                "source_call": artifact(plan_path), "source_compilation": artifact(compile_path),
                "compiled_interaction_index": index, "step": step, "new_qwen_call_performed": False,
                "not_a_fresh_independent_plan": True}, seal=True)
            child = {"schema": "p550.stage1_query_execution.v1", "scene_id": geometry["scene_id"],
                "instruction": step["intent_description"], "source_fixed_geometry": artifact(geometry_path),
                "source_fixed_semantics": geometry["source_semantics"], "full_scene_snapshot": geometry["all_instances_attempted"],
                "new_segmentation_or_instance_classification": False, "qwen_computed_coordinates": False,
                "semantic_plan": artifact(derived_plan), "interaction_plan": {"steps": [step]},
                "source_sequence_semantic_compilation": artifact(compile_path), "interaction_index": index,
                "source_code": artifact(__file__)}
            child_started = time.monotonic()
            try:
                child.update(execute_route(semantics, geometry, step, child["instruction"],
                    current_start, child_root, derived_plan, runtime=runtime))
            except Exception as error:
                child.update(status="failed_not_publishable", reason=f"{type(error).__name__}: {error}")
            child["elapsed_seconds"] = time.monotonic() - child_started
            child_path = child_root / "stage1_execution.json"
            write_once(child_path, child, seal=True)
            row = {"interaction_index": index, "source_step_indices": mapping["execution_compilation"][index]["source_step_indices"],
                "action": step["action"], "status": child["status"], "stage1_execution": artifact(child_path),
                "route_start_world_xy_m": current_start, "requested_target_ids": step["target_ids"]}
            result["interaction_steps"].append(row)
            if child["status"] != "complete":
                result.update(status="failed_interaction_route", stage2_sequence_handoff_allowed=False,
                    unexecuted_interaction_indices=list(range(index + 1, len(compiled["steps"]))))
                break
            check_stage1_handoff(child_path)
            bound = bind(read_sealed(child_path))
            current_start = bound["snapped_goal_world_xy_m"]
            row.update(selected_target_id=bound["target_instance_id"], route_goal_world_xy_m=current_start)
            # Every numeric milestone receives actual task-bound Qwen prose.
            try:
                descriptions.run(child_path, child_root / "keynode_descriptions", qwen=qwen)
                row["keynode_descriptions"] = artifact(child_root / "keynode_descriptions/receipt.json")
            except Exception as error:
                row.update(status="keynode_description_failed", reason=f"{type(error).__name__}: {error}")
                result.update(status="failed_keynode_descriptions", stage2_sequence_handoff_allowed=False,
                    unexecuted_interaction_indices=list(range(index + 1, len(compiled["steps"]))))
                break
        else:
            result.update(status="complete_ordered_stage1_sequence", stage2_sequence_handoff_allowed=True,
                interaction_count=len(result["interaction_steps"]), all_source_actions_retained=True,
                all_navigation_segments_collision_checked=True, all_keynodes_described=True)
    result["elapsed_seconds"] = time.monotonic() - started
    write_once(output / "receipt.json", result, seal=True)
    print({"scene": result["scene_id"], "status": result["status"], "interactions": len(result["interaction_steps"]),
        "seconds": result["elapsed_seconds"]}, flush=True)
    return result

