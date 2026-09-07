"""Immutable resumable checkpoints for ordinary in-process scene functions."""
from ._common import *
from ..scene import verify_artifact_tree


def checkpoint(root, label, receipt_path, reused=False):
    receipt = read_sealed(receipt_path)
    verify_artifact_tree(receipt)
    return write_once(root / "checkpoints" / (label+".json"), {
        "schema": "p550.scene_phase_checkpoint.v1", "phase": label,
        "receipt": artifact(receipt_path), "explicit_existing_artifact_reuse": reused,
        "new_execution_claim": not reused}, seal=True)


def load_checkpoint(root, label):
    path = root / "checkpoints" / (label+".json")
    if not path.exists():
        return None
    value = read_sealed(path)
    receipt_path = verified(value["receipt"])
    verify_artifact_tree(read_sealed(receipt_path))
    return receipt_path


def phase(root, label, operation, *, gpu=None, filename="receipt.json", resume_partial=False):
    """Call the migrated algorithm, never a historic script or command string.

    Only sealed artifacts are resumed. Failed attempts remain diagnostic evidence.
    ``gpu`` is retained as receipt metadata; model/render runtimes own allocation.
    """
    ready = load_checkpoint(root, label)
    if ready:
        return ready
    phase_root = root / "steps" / label
    attempts = sorted(phase_root.glob("attempt_*"))
    last = attempts[-1] if attempts else None
    if last is not None:
        target = last / "output" / filename
        if target.exists():
            checkpoint(root, label, target)
            return target
    if last is None or not resume_partial:
        last = phase_root / f"attempt_{len(attempts)+1:03d}"
        last.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run_index = len(list(last.glob("execution_*.json"))) + 1
    result = {"schema": "hsi.scene_phase_execution.v1", "phase": label,
        "gpu": gpu, "in_process_static_algorithm": True}
    try:
        operation(last / "output")
        target = last / "output" / filename
        checkpoint(root, label, target)
        result.update(status="complete", receipt=artifact(target))
        return target
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        write_once(last / f"execution_{run_index:03d}.json", result)


def reuse(root, label, path):
    ready = load_checkpoint(root, label)
    if ready:
        return ready
    checkpoint(root, label, path, reused=True)
    return path
