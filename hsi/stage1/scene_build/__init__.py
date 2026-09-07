"""Complete instruction-independent scene construction and final publication.

Perception, geometry and model runtimes are explicit; no human query, generated
person, future motion or affordance bank is admitted by this interface.
"""
from ._common import *
from .checkpoints import phase, reuse, load_checkpoint
from .report import write_report
from . import semantics, geometry, shapes, recovery_candidates, parts, composition
from .publication import review_context, review_geometry, review_envelope, review_backrest
from ..scene import load_final_scene, verify_artifact_tree


def recover_parts(root, snapshot, *, qwen_client):
    ready = load_checkpoint(root, "recovered_semantics")
    if ready:
        return ready
    proposed = phase(root, "recovery_proposals", lambda out:
        recovery_candidates.run(snapshot, out / "receipt.json", limit=4))
    accepted, used, attempts = [], set(), []
    for index, proposal in enumerate(read_sealed(proposed)["proposals"]):
        if used.intersection(proposal["member_ids"]):
            continue
        try:
            path = phase(root, f"parts_{index:02d}", lambda out:
                parts.run(snapshot, proposal["member_ids"], out, qwen_client=qwen_client))
            review = read_sealed(path)
            attempts.append({"review": artifact(path), "status": "reviewed"})
            if composition.composition_eligible(review, snapshot):
                accepted.append(path)
                used.update(proposal["member_ids"])
        except Exception as error:
            attempts.append({"member_ids": proposal["member_ids"], "status": "unresolved",
                "reason": f"{type(error).__name__}: {error}"})
    if not (root / "recovery_outcome.json").exists():
        write_once(root / "recovery_outcome.json", {"schema": "p550.bounded_recovery_outcome.v1",
            "proposals": artifact(proposed), "attempts": attempts,
            "accepted_group_count": len(accepted), "original_atoms_never_removed": True})
    if not accepted:
        return reuse(root, "recovered_semantics", snapshot)
    return phase(root, "recovered_semantics", lambda out: composition.run(snapshot, accepted, out))


def understand_atoms(row, sam, root, *, qwen_client):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    final = root / "receipt.json"
    if final.exists():
        prior = read_sealed(final)
        require(prior["scene_assets"] == row and prior["source_atomic_instances"] == artifact(sam),
                "Existing scene output binds other assets or atomic perception")
        verify_artifact_tree(prior)
        return final
    snapshot = phase(root, "semantics", lambda out:
        semantics.run(sam, out, qwen_client=qwen_client), resume_partial=True)
    snapshot = recover_parts(root, snapshot, qwen_client=qwen_client)
    geo = phase(root, "geometry", lambda out: geometry.run(snapshot, out))
    generic = phase(root, "generic_shapes", lambda out: shapes.run(geo, out))
    report = write_report(root, snapshot, generic)
    semantic_data, geometry_data = read_sealed(snapshot), read_sealed(generic)
    unknowns = [r["instance_id"] for r in semantic_data["objects"]
        if r["decision"]["needs_visual_recovery"] and not r.get("represented_by_logical_object_id")]
    write_once(final, {"schema": "p550.complete_scene_only_processing.v1", "scene_id": row["scene_id"],
        "status": "processed_with_unresolved_instances" if unknowns or semantic_data["errors"] else "processed",
        "scene_assets": row, "fixed_semantics": artifact(snapshot), "fixed_geometry": artifact(generic),
        "source_atomic_instances": artifact(sam), "report": artifact(report),
        "all_instances_attempted": semantic_data["all_instances_attempted"],
        "source_atomic_count": semantic_data["source_instance_count"],
        "logical_object_count": len(semantic_data.get("logical_compositions", [])),
        "unresolved_instance_ids": unknowns, "per_instance_errors": semantic_data["errors"],
        "sit_usable_ids": [r["instance_id"] for r in geometry_data["objects"] if r["query_usable_for_sit"]],
        "generic_face_candidate_count": geometry_data["generic_face_candidate_count"],
        "human_instruction_read": False, "human_current_or_future_motion_read": False,
        "stage2_or_stage3_output_used": False, "affordance_memory_used": False,
        "elapsed_seconds_this_execution": time.monotonic()-started, "source_code": artifact(__file__)})
    return final


def build(scene_id, mesh_path, occupancy_path, output, *, qwen_client,
          perception_runtime=None, atomic_receipt=None, gpu=0):
    """Raw LINGO scene → immutable FINAL inventory with backrest review complete.

    ``atomic_receipt`` is explicit, hash-verified reuse of real SAM perception;
    it skips only rendering/SAM, never semantics or final publication gates.
    Omit it for a fresh raw-mesh build with an explicit PerceptionRuntime.
    """
    from .. import perception
    require(isinstance(scene_id, str) and scene_id and "/" not in scene_id
            and "\\" not in scene_id and scene_id not in (".", ".."), "Unsafe scene ID")
    require(type(gpu) is int and gpu >= 0, "Explicit nonnegative render GPU required")
    output = Path(output).resolve()
    row = {"scene_id": scene_id, "mesh": artifact(mesh_path), "occupancy": artifact(occupancy_path)}
    binding = {"schema": "hsi.stage1.scene_build_launch.v1", "scene_assets": row,
        "atomic_receipt": artifact(atomic_receipt) if atomic_receipt is not None else None,
        "gpu": gpu, "human_instruction_read": False,
        "qwen_model": qwen_client.model, "qwen_endpoint": qwen_client.endpoint,
        "qwen_revision": qwen_client.model_revision}
    launch = output / "launch.json"
    if launch.exists():
        prior = read_sealed(launch)
        require({k: v for k, v in prior.items() if k != "receipt_payload_sha256"} == binding,
                "Cannot resume scene build with changed assets, model or configuration")
    else:
        write_once(launch, binding)
    final = output / "receipt.json"
    if final.exists():
        return load_final_scene(final).publication_path
    if atomic_receipt is None:
        require(perception_runtime is not None, "Fresh perception requires explicit external runtime")
        require(perception_runtime.gpu == gpu, "Perception and supplemental-view GPU differ")
        sam = phase(output, "perception", lambda out:
            perception.run(scene_id, Path(mesh_path), Path(occupancy_path), out,
                           runtime=perception_runtime), filename="sam/receipt.json", gpu=gpu)
    else:
        sam = reuse(output, "perception", Path(atomic_receipt).resolve(strict=True))
    perception.verify_sources(row, sam)
    base = understand_atoms(row, sam, output / "scene_inventory", qwen_client=qwen_client)
    context = phase(output, "original_context", lambda out:
        review_context(base, out, gpu, qwen_client=qwen_client))
    spatial = phase(output, "spatial_and_direction", lambda out:
        review_geometry(context, out, gpu, qwen_client=qwen_client))
    envelope = phase(output, "envelope", lambda out:
        review_envelope(spatial, out, gpu, qwen_client=qwen_client))
    backrest = phase(output, "semantic_backrest", lambda out:
        review_backrest(envelope, out, gpu, qwen_client=qwen_client))
    result = read_sealed(backrest)
    result.update(integrated_scene_build_launch=artifact(launch),
        source_backrest_final_publication=artifact(backrest), integrated_source=artifact(__file__))
    write_once(final, result)
    return load_final_scene(final).publication_path


__all__ = ["build", "understand_atoms"]
