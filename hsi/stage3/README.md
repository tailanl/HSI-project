# Stage3: verified image keypose to current motion

The default is the current ordered ReMoGen rollout, clean-xstart physics, full-22
ICGF and raw-scene SDF. Triangle-contact/support guidance is added after one
complete parent energy evaluation. The raw parent field still includes the
target, and no whole-body or whole-target collision exemption is introduced.

## Entry points

- `compiler.compile(stage2_path, output, runtime_body=..., render_body=...)`
  revalidates actual Stage2 image/recovery/Qwen/full-mesh evidence and current
  Stage1 endpoint checks. It preserves the route, checks the bounded local sit
  transition, builds a fresh neutral zero-shape frame0 and writes the ordered
  root-only route → reorientation → full-22 contact packet. It currently accepts
  one feet-grounded sit interaction, not arbitrary sequences of actions.
- `runtime.run(adapter_path, output, runtime=MotionRuntime(...), seed=55000,
  affordance=True, dry_run=False)` invokes the current model once. Explicit
  checkpoint hashes, configurations, numerical code and CLIP checkpoint use are
  checked before/after generation. The source implementation is external; the
  project keypose adapters keep their current parameter names and invoke the
  official base forward through normal PyTorch hooks.
- `evaluation.evaluate(execution_path, output, official_sources=...,
  mean_hands_path=..., device="cpu")` independently validates the execution,
  builds the actual all-frame 10,475-vertex body with every predicted joint as a
  driver, and evaluates the fixed 34 release checks. Generation alone never
  authorizes release or awards Memory credit.

`MotionRuntime.from_mapping` requires the exact fields illustrated in
`configs/motion.example.json`. The external ReMoGen environment must also provide
its declared dependencies (including PyTorch3D, CLIP, SMPL-X, tyro and the current
model configuration classes). This package requires Python 3.10 or later; an old
Python 3.8 model environment is not silently treated as an installable runtime.

Pass the current mean-hand asset explicitly to preserve the current skinning
geometry. The API's explicit `None` diagnostic is the zero-hand fallback, not a
claim of parity with a run that used mean hands. Model assets, hands and external
metric implementations are not redistributed here.

## Preserved numerical policy

The rolling condition window has capacity 8. The current maximum is 40 motion
primitives plus the registered rolewise completion allowance. Semantic contact
IDs are 0/1/2; all 22 sparse joints remain in ICGF. Contact/SDF tail exemptions
apply only to the declared roles and final four frames. Current physics weights
remain slip/contact/floor/proxy = 4/1/8/2, with root-speed release at .85 m/s.
The evaluator retains 20 Hz, factor-two conservative occupancy pooling, fixed
world bounds and the original 34 physical gates. No output motion is edited.

Body-derived gluteal/foot proxy offsets are bounded, not clipped. Contact alias
handling subtracts the selected target before conservative pooling and retains
the complete parent obstacle checks. Static geometry is cached only within one
actual rollout and exact SDF/body identity; pose, motion and energy are not cached.

## Integration checks, not new model-performance claims

- CPU numerical tests cover the current SDF, causal bridge, physics, full-22
  repair, ordered packet, static neutral seed and release gates.
- With the installed official base class, 8 tiny-model forward combinations
  (scene control × adapter backend × active keypose) matched the old project
  adapter exactly; active adapter gradients matched too. No pretrained network
  checkpoint or GPU was used in that comparison.
- Independent joint-driven skinning matched the previous implementation on
  five sampled frames from a real 234-frame motion: maximum vertex difference
  `7.15e-7 m`, identical topology, and zero error in all 22 joint drivers.

These are integration/numerical comparisons, not a newly generated end-to-end
motion, an accuracy benchmark, or a speed improvement measurement.
