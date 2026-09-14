# Stage 1/2 → original UniHSI: partial-adapter boundary

Status: corrected private adapter `run_stage12_transfer_v2.py` is running in `unihsi_stage12_batch02`. **Version 1 / batch01 is invalid for effect evaluation:** it mistakenly treated the raw LINGO Y-up OBJ as world Z-up. Source hashes and action checks did not detect this semantic coordinate error. The corrected version directly uses the authoritative Stage 2 loader and binds a coordinate addendum. Ten V2 CPU contract tests passed; three actual contact surfaces match corrected world-mesh vertices within 8.42e-8 m. This is still **not a completed full Stage 1/2 → Stage 3 implementation**. The original ScanNet rollout is a separate physical-controller diagnostic; its source remains frozen.

## What can be preserved without changing the actor

The released actor consumes 223 body-state + 216 task features, and produces 28 normalized PD actions. Task features include a 9×9 local heightmap and per-body contact conditions. Preserve the real loop: observations → released actor → PD / PhysX → measured state / contact → next observations. The AMP discriminator is not an additional inference motion generator. This is the released MLP checkpoint, **not a reproduced paper CNN**.

Use a new private `UniHSI_ScanNet` subclass, without editing the original file `/home/lzsh2025/workspace/UniHSI/unihsi/env/tasks/unihsi_scannet.py`:

| Input / responsibility | Existing source hook | Minimal private adaptation |
| --- | --- | --- |
| Original LINGO scene mesh | `_load_mesh`, line 243 | Use `hsi.stage2.view_numeric.load_scene_mesh`, which maps raw Y-up `(x,y,z)` to world Z-up `(x,-z,y)`, then pass its vertices / face indices to PhysX. No additional ScanNet transformation, recentering, rescaling, or scene-name substitution. One environment has zero grid offset. |
| Contact surface and ordered route events | `_get_pcd_parts`, line 292; `process_contact`, line 195 | Construct the same 15-body / 200-point surface buffers, contact validity/type/direction, and standpoint buffers using explicit semantic correspondence. Route events are controller goals, never teleports or forced motion samples. |
| Geometry-conditioned observations | `_create_mesh_ground`, line 178; `_compute_task_obs`; `compute_strike_observations`, line 657 | Keep the trained 9×9 feature shape and original transformation conventions. Build a separate heightmap from a copy of scene vertices: the original heightmap helper mutates heights; never mutate the collision/rendering mesh. |
| Contact-driven event progression | `_reset_target`, line 435; reward/reset functions | Preserve measured physical feedback and record each transition. Do not replace contact completion with a frame number. Stop on physical failure; do not join reset episodes into one video. |

Sampling 200 surface points is an explicit representation reduction, not a claim that the entire upstream surface or full-body target is enforced. The implemented adapter explicitly maps the seat normal to the released world `up` direction, rejects errors over 15 degrees, and records the quantization error. It does not simply cast an arbitrary normal into an integer buffer. Initial native root XY and +X-facing yaw are obtained from the original SMPL-X root's +Z anatomical forward; native default height and DOF pose remain explicitly different. The shape-calibrated Stage 1 hip-axis yaw is 0, but saved root-rotation yaw is about -0.028 to -0.048 rad in these packets. V2 records this distinction and does not silently claim identical full-body orientation.

The optional `--sdf-filter` is a **new, experimental** bounded PD-action filter, not an original UniHSI component. It uses the current physical Jacobian and the original scene SDF to compare small articulation-change proxies, then sends the selected actions into actual PhysX. It records both nominal and actual actions, every gate outcome, and the count of genuinely changed frames. A rejected Jacobian check returns unchanged actions. This is not a learned dynamics predictor, full-surface guarantee, or successful full-pose conditioning. Default is OFF.

## Actual packet and non-negotiable limitations

Packet: `agent9/runs/paper_structure_stage3_20260913/stage12_packets/scene006/manifest.json`.

- Scene `006`, armchair `SCENE_INSTANCE_026`, surface `INTERACTION_SURFACE_CF4295DA2F1F74C7`; not ScanNet `0000_00`.
- Arrays SHA-256: `cdc1e65e4d6a0c3f6300eb49b5e6779c0e28eaa47785275333430ce4234a5345`.
- Original `mesh_low.obj` SHA-256: `719ef35fb410011dc6e6610bf0c675ca46f1495d5944e4a9cd0c17a259c6d430`.
- Initial root `[-0.09, -1.41, 0.9269709]`; route terminal XY `[0.16999993, 0.18999991]`; target root `[0.22442964, 0.80404234, 0.57784003]`. **Route terminal is the approach point, not the seated pelvis.**
- Original Stage 2 body has nonzero 10-D betas, 135-D pose, and 22 joints. The packet locks root XYZ, global orientation, shape and all 22 joints, with zero tolerance. Stage 2's earlier static-refinement allowances are explicitly not Stage 3 permissions.
- The released actor uses fixed 15-body / 28-DOF AMP geometry. It has no full-body SMPL-X keypose token or beta input. Directly mapping pelvis/feet/hands contacts cannot establish equivalence of the two articulated bodies or preserve the locked pose.
- A private physical scene/contact transfer may be tested only with these omissions recorded. It cannot be labelled exact Stage 2 keypose consumption, a shape-preserving rollout, or a completed Stage 3 implementation. No implicit neutral-body replacement, target snapping, output SMPL-X fitting, or invented retargeting success.

To meet the full target contract, first validate body/DOF retargeting and support geometry, then add a real pose/shape-conditioning mechanism and train or fine-tune the physical controller. Purely appending untrained CNN/keypose inputs to this frozen checkpoint is not a valid reproduction. If the task instead deliberately relaxes the upstream contract for a partial comparison, that change must be explicit.

## Reporting requirements for any future partial experiment

Keep original mesh/packet hashes, source closure, actor checkpoint, physics configuration, initial state, CoC mapping, and all post-physics frames. Separate successful event completion from physical failure and from any geometric/semantic target error. SDF may audit or guide permitted control choices, but cannot replace PhysX dynamics or authorize teleporting locked goals. Preserve target + other-object + floor coverage; SDF outside its finite bounds is unknown, not free.
