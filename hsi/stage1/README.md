# Stage1: fixed scene, then instruction-dependent planning

This directory is one statically integrated implementation. `scene_build/`
contains the current scene-only reviews; `perception/` contains rendering,
anonymous depth boxes and strict atomic SAM fusion. The numeric route modules
are the retained current algorithm, not replacements or historic runtime loaders.

## Entry points

```python
from pathlib import Path
from hsi.stage1 import QwenTextClient, Stage1Runtime, build_scene, run
from hsi.stage1.perception import PerceptionRuntime

qwen = QwenTextClient(
    endpoint="http://127.0.0.1:8158/v1",  # Explicit user-configured service.
    model="qwen38-27b-fp8",              # Exact currently deployed alias.
    model_revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
)
fixed_scene = build_scene(
    scene_id, mesh_path, occupancy_path, scene_output,
    qwen_client=qwen, perception_runtime=PerceptionRuntime(
        render_python, sam_python, sam_source_root, sam_checkpoint, gpu=0), gpu=0,
)
run(fixed_scene, instruction, start_xy, query_output,
    qwen=qwen, runtime=Stage1Runtime(smplx_model_path=Path(smplx_file)))
```

The model alias and revision above document the tested deployment; no endpoint,
model weight or secret is downloaded or implicitly selected. Credentials, when
needed, are read from the explicitly named `api_key_env` variable. A declared
revision is not falsely reported as locally verified weight lineage.

For already completed Stage1A, call `run` with that explicit FINAL receipt.
`load_final_scene` verifies its seal, direct semantics/geometry identities and
recursive artifact hashes. To reuse perception but redo scene understanding,
pass `atomic_receipt=...` to `build_scene`; its exact mesh/occupancy binding and
instruction independence are checked. No implicit historical pilot search exists.

## Preserved algorithm and boundaries

Stage1A: 48 actual mesh views → anonymous depth proposals → SAM masks → strict
complete-link atomic 3-D fusion → category and completeness separately → bounded
two-order part-composition review → original untruncated context review (bounded
extra views only when missing) → spatial component/envelope proposals retaining
all residuals as unknown → three independently agreeing camera direction reviews
→ conflict-specific backrest/seat part review → immutable FINAL scene.

Qwen sees images, semantic text and candidate IDs, not coordinate requests.
Geometry computes masks, crops, candidate directions, contact surfaces and routes.
Category is not overwritten by dimensions, facing direction or geometry. Failed
or ambiguous objects remain unavailable for interaction without aborting unrelated
objects. This implementation does not add OfficialOrientation or change the
current direction algorithm.

Stage1B: fresh ID-only Qwen plan → ordered action validation → contact-front
region clipped by occupancy/neighbor clearance → fresh official `pynavmesh`
polygon graph and route → the original corridor interiorization → sparse
keypoints with required descriptions. Current execution supports SIT (including
same-object WALK then SIT); unsupported actions fail explicitly. It does not
pretend that all schema action names have a numerical executor.

Stage1A saves geometry/occupancy fields; these are not a NavMesh graph. Each
Stage1B query cold-builds its own graph. Routes and PNG diagnostics can be saved
as query results, but no graph cache is loaded or persisted. The private P555
third-party constructor source optimization is not vendored here: graph algorithm
and route math are retained, but previous optimized timing is not promised.

Scene contact sheets and route/keypoint images are PNG, with JSON receipts.
No HTML is generated. Supplemental object views require the caller environment
to contain the declared rendering dependencies; raw SAM/render workers use the
explicit external environments in `PerceptionRuntime`.

## Verification and origins

`ORIGINS.json` records 157 unchanged numeric top-level definitions;
`scene_build/ORIGINS.json` records 37 additional unchanged definitions plus every
adapted scene-review source. AST pins were produced/checked under Python 3.10.
The two PNG route modules retain P508's hash-bound renderer with P550's exact
two-node final-approach validation change folded in statically.

Real scene 014 CPU comparison reproduced all 34 object geometry records, every
surface array and all three navigation field arrays exactly (except new artifact
paths). The original 253-point route interiorization is bit-exact; its seven-node
route also rendered through the migrated PNG visualizer. A tiny nonseating CPU
fixture exercises all publication phases and FINAL reuse with a synthetic Qwen
transport. These checks are not a live SAM/Qwen full-scene accuracy result; no
new GPU/model execution was performed during this integration.

External code/weights are not copied here: SAM/ComfyUI, Qwen inference service,
SMPL-X, PyPI pynavmesh/pathfinder and rendering/scientific Python dependencies
must be installed/configured separately.
