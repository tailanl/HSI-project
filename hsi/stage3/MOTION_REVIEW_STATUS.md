# Actual-motion semantic source integration

The current source is consolidated in `motion_render`, `motion_camera`,
`motion_pictures`, `motion_contract`, and `motion_review`. These modules no longer
load historical workspace Python. The camera fit, actual scene/amodal masks,
terminal camera selection, deterministic real-frame policy, montage reconstruction,
and nine boolean checks with confidence at least 0.85 retain the current method.

`motion_render.render(execution_path, evaluation_path, sequence_path, output,
gpu=0)` uses the exact saved evaluated 10475-vertex frames and full original scene.
`motion_review.run(render_path, output, qwen_client=...)` calls the explicit Qwen
service with the original target images, actual trajectory montage, two actual
last-frame close views and actual four-frame terminal montage.

`motion_review.validate_review(...)` validates image/sequence/actual-Qwen binding.
This is not a replacement for independent full-motion physical validation. The
renderer checks the current evaluator and skin producer source bindings but does
not independently reskin saved vertices or recompute all physical metrics. It does
not grant positive Memory credit. Qwen weight files are not locally attested by
the generic external HTTP transport.

These final source integrations were added when the user requested prioritizing
code consolidation and stopping further comprehensive testing. They have not
been run end-to-end with real GPU rendering, motion models or Qwen. No new timing
or successful-motion claim is made for this integration.

Source origins: project-owned current motion semantic contract, rendering,
image-audit and helper implementations from the previous consolidated method.
Source paths are provenance only; they are not runtime dependencies. External
pyrender, trimesh, Qwen and licensed body/model assets remain separate dependencies.
