# Scene-only perception

`run(scene_id, mesh_path, occupancy_path, output, *, runtime)` runs the actual
scene pipeline and returns the strict SAM instance receipt. `output` must be
a new attempt directory; an existing attempt is never overwritten or counted
as a new execution. `verify_sources(row, sam_path)` verifies exact scene mesh
and occupancy identity and the original instruction-independent boundary.

```python
from pathlib import Path
from hsi.stage1.perception import PerceptionRuntime, run

runtime = PerceptionRuntime(
    render_python=Path("/external/render-env/bin/python"),
    sam_python=Path("/external/sam-env/bin/python"),
    sam_root=Path("/external/ComfyUI"),
    sam_weight=Path("/external/models/sam3.1_multiplex_fp16.safetensors"),
    gpu=0,
)
sam_receipt = run(scene_id, mesh_path, occupancy_path, output, runtime=runtime)
```

The renderer environment needs NumPy, SciPy, Pillow, Trimesh and Pyrender with
working EGL. The separate SAM environment needs those numerical/image
dependencies, PyTorch/CUDA, and the explicitly supplied ComfyUI installation
supporting `SAM3Model` and `SAM3_Detect.execute`. External checkpoint and SAM
node/detector sources are hashed, not copied into this package. No download,
service deployment, credentials, GPU scheduler or implicit environment is used.

The default production path preserves the current method:

- `renderer.py`: original occupancy transform, body-band clearance, connected
  components, deterministic farthest anchors and calibrated render. Six anchors
  × eight yaw angles at 640 × 480; RGB, metric depth and original camera matrices
  are retained. No navigation mesh is built or stored here.
- `depth_geometry.py` and `multiscale.py`: original discontinuity components,
  small/medium/large profiles, cross-scale NMS 0.90, containment 0.98 with area
  ratio 0.82, and a maximum of 48 anonymous visual prompts per view.
- `sam_geometry.py` and `sam.py`: real external SAM inference, refinement 3,
  original mask cleanup/NMS, depth-consistent backprojection to unchanged OBJ
  vertex IDs and a 180-vertex observation minimum. No category or task prompt.
- `atomic.py`: same-view 3-D NMS at 0.70; strong edges require 80 shared vertices
  and 16% of the smaller support. All original complete-link, average-score,
  strong-edge density, diameter and view-purity gates remain. Published support
  contains only vertices independently seen in at least two views. The final
  overlap veto remains 0.25 of the smaller consensus support.

Fusion uses explicit returned audits and consensus supports, without the old
module monkeypatch or global result side channel. The final compatibility
schema remains `p515.lingo_sam31_atomic_multiview_instances.v1` with status
`atomic_instances_ready`; producer/source bindings point to the new actual
modules. A successful segmentation is **not** final semantic target approval:
`publish_gate_pass` stays false until downstream semantic/geometry processing.

PNG outputs include the 48-view contact sheet, anchor map, anonymous-box
overlays, cyan-mask classifier crops and instance contact sheet. Calibrated
camera JSON, metric depth NPY, raw masks NPZ and consensus support NPZ are
retained. No HTML is emitted. Failed SAM attempts retain a sealed failure
receipt and logs; they do not publish a successful instance receipt.

Validation includes original atomic/mask/multiscale regressions and new CPU
contract tests. Four fixed golden vectors were computed by running old and
native kernels on identical inputs: full camera matrices, complete proposals
and suppressions, complete atomic fusion audit, and anchor/coverage statistics.
The orchestration tests stub external model/process boundaries in temporary
fixtures; they are **not** evidence of a real GPU/model scene run. Actual runtime
readiness and end-to-end image quality still require a configured model run.

The compatibility geometry is LINGO-specific: a boolean `(300, 100, 400)`
occupancy grid at 0.02 m, original mesh coordinates `[x, y_up, z]` transformed to
world `[x, -z, y_up]`, and scene ID matching the mesh parent and occupancy stem.
Source hashing protects local artifact consistency, not authenticity against
a malicious process with the same filesystem authority.
