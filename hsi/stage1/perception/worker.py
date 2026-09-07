"""Fixed subprocess entry points for renderer and external SAM environments."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("render", "sam"):
        raise ValueError("Select the fixed render or sam worker")
    if argv[0] == "render":
        from . import renderer
        renderer.run(renderer.parse_args(argv[1:]))
        return 0
    from . import sam
    parser = argparse.ArgumentParser()
    for key in ("render-receipt", "boxes-receipt", "output", "weight", "sam-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv[1:])
    args.instruction = ""
    args.refine_iterations = 3
    args.minimum_support_vertices = 180
    try:
        sam.run(args)
    except Exception as error:
        from hsi.common.artifacts import artifact, write_once
        # Preserve diagnostics as a failure, never a successful stage publication.
        failed = args.output.parent / "failure.json"
        if not failed.exists():
            payload = {
                "schema": "hsi.stage1.scene_perception_failure.v1", "status": "blocked",
                "error_type": type(error).__name__, "reason": str(error),
                "stage_gate_pass": False, "publish_gate_pass": False,
                "source": artifact(Path(__file__)),
            }
            audit = getattr(error, "fusion_audit", None)
            if audit is not None:
                payload["atomic_instance_fusion"] = audit
            write_once(failed, payload, seal=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
