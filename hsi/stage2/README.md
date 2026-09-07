# Stage2: current integrated single-image keypose method

The callable implementation is `pipeline.run`. It performs fresh geometric view
selection and SDF preparation in separate workers, obtains a Qwen target-appearance
description, invokes the external H3 single-image runtime, and runs HybrIK recovery
alongside the single-image Qwen check. Background consistency precedes placement
and refinement. Publication requires the original 18 physical gates, four neutral
sitting checks, and the actual nine-check Qwen review of the refined mesh.

Only unambiguous visibility-only H3 failures may enter the explicit recovery path.
The failed original judgement is retained. A physical candidate alone is not a
verified Stage2 output. `pipeline.validate_success` revalidates the source closure,
recorded model responses and image order, reruns background and posture checks,
rematerializes the actual SMPL-X body, and recomputes the 18 physical gates against
the current target-bound SDF before the Stage3 consumer can accept a receipt.

## Public entries

- `views.select_views(stage1_path, render_path, output, gpu=0, mode="full", ...)`
  evaluates the original camera inventory. Memory hits propose camera IDs only;
  current crop, owner/surface mask and depth gates are still executed. Rejected or
  missing hints fall back to the complete 48-view inventory. Original camera
  indices remain intact.
- `image_job.run(...)` prepares the exact crop/camera/target-bound image job.
- `h3.generate(...)` uses an explicit external ComfyUI installation and model
  manifests. It requires a fresh process and runs one 512-square temporal image
  latent, with the current eight-step sampler and both adapters.
- `recovery.recover(...)` invokes the external detector and HybrIK-X model. The
  real 55 rotations are projected to SO(3), kid shape is folded to ten neutral
  betas, and the complete native-Y-up/materializer-Z-up chain is audited. Camera
  translation never supplies world placement.
- `placement.run(...)` places the recovered articulation at the current Stage1
  contact and calls `refine.run`. World root translation is derived from the
  fixed gluteal contact subset; yaw and scale are not free optimizer variables.
- `pipeline.run(stage1_path, render_path, descriptions_path, output, *,
  qwen_client, comfy_root, models_root, recovery_runtime, segmentation, gpu=0,
  seed=55000, mode="full", facts_path=None, store_root=None)` produces the final
  sealed receipt. `recovery_runtime` is the explicit mapping accepted by
  `recovery.RuntimeConfig.from_mapping`; no model path is discovered implicitly.

The orchestration separates GPU/EGL workers from the CPU optimizer and does not
start a model server or a background queue. Qwen receives semantic images and
candidate identities, not a request to generate coordinates. NavMesh is not
persisted by Stage2; SDF tensors are separately prepared and hash-bound to the
current Stage1 plan.

## Numerical and execution boundaries

`refine.py` contains the materialized current objective, ranking and bounded search
as ordinary Python. It includes the strict penetration inner margins, bilateral
foot-area terms, neutral sitting loss, ankle/foot limits and the bounded 2-mm
microsearch. `body_model`, `contact_materialize`, `refine_physics`, `bound_sdf`,
`refine_target`, `refine_margins`, and `posture_quality` supply its actual numerical
closure. There is no runtime source rewriting, old-directory loader or placeholder
optimizer. Historical schema names inside numerical NPZ fields are retained wire
formats, not claims that an old source produced a new result.

The final public schema is `hsi.stage2.verified_keypose.v1`. It binds
`source_stage1`, `source_view`, `image_job`, `h3_receipt`, `h3_image`,
`recovery_receipt`, `physical_refine`, `background_consistency`,
`neutral_sitting_quality`, `post_refine_qwen`, and `keypose`. A failure leaves
`verified_stage2_keypose` and `stage3_handoff_allowed` false.

Current scope is neutral sitting on a supported sittable target. This is not a
universal action/keypose generator. Qwen, ComfyUI, HybrIK, SMPL-X assets and their
licenses are external dependencies; no third-party model implementation or weights
are packaged here. Texture completeness is limited by the supplied original mesh.

CPU tests execute full selector, optimizer/search and H3 caller control flow with
explicit device/model doubles, plus adversarial contracts and numerical checks.
Those tests do not constitute real GPU rendering, real pretrained-model inference,
or a completed model-to-model end-to-end validation of this new integration.
