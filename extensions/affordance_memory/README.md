# Current Affordance Memory research extension

This directory preserves the current production-Memory and geometry-keypose
fast-path implementation and the local implementations its entry points depend
on. It is one dependency graph, not a menu of older methods. Files with version
suffixes are retained because the current entry explicitly imports or binds them.
The sources are not yet connected end-to-end to the consolidated `hsi` package.

## Primary entry

`run_geometry_stage2_memory_v2.py` is the geometry-keypose consumer entry.
It uses `memory_runtime_v2.prepare/query/validate_lookup` and
`fast_keypose_memory_v2.FastKeyposeEngine`; its actual geometry/render/Qwen
receipts are checked by `geometry_memory_contract_v2.py`.

`episode_evidence_v2.admit_episode` is the complete-motion Memory admission
entry. `memory_store_v2.MemoryStore` owns immutable cycles, journal/CAS commits,
activation and quarantine. These are the current entry points; do not switch to
the internally retained v1 dependencies as an alternative workflow.

The native consolidated facts, camera-hint bank and numerical local retrieval
remain in `hsi/memory/`. The native retrieval preparer is now also present in
`hsi/memory/runtime.py`. The native `authority.py` still rejects production
admission until this complete research authority has been connected to the new
execution/evaluation receipts. No permissive substitute is supplied.

## Preserved method

- Exact typed local records: contact point, support normal, root offset, facing
  and contact phase. No complete pose, motion, reusable Qwen conclusion or
  NavMesh payload is stored.
- Geometry fast path: the original analytic sitting articulation and optimizer,
  local-memory candidate veto and initialization ranking, original current SDF,
  full-body 23 numerical checks and independent actual-image Qwen review.
- Positive admission: actual complete motion, unchanged 34 physical release
  checks, original static 23 checks, measured final-four-frame meshes, original
  full task identity, independent nine-check motion semantics at confidence .85,
  and source-bound measured local extraction.
- Activation: at least eight successful episodes, four scene families and Beta
  5% lower bound .7; single-episode credit and live critical quarantine remain.

## Runtime boundary: code present, not yet integrated

These sources retain their original schema IDs, source pins, dynamic numerical
loaders and research receipt contracts. They do **not** become native producers
merely because they are placed here. `memory_common.py` now requires the explicit
`HSI_RESEARCH_ROOT` environment variable; there is no username-specific or inferred
workspace root. The current research layout below that configured root is still
required by the unadapted loaders. Original source pins have not been re-signed.
Thus this directory does not claim independent execution from a clean checkout.

The source map records original relative paths and SHA256 values, the packaged
SHA256 values and the sole root-configuration edit. Current numerical counterparts
already consolidated under `hsi/stage1`, `hsi/stage2` and `hsi/stage3` are listed
there; reconnecting those APIs and re-registering producer identity is still work
to do. The environment-specific reserved GPU lease hook, original external model
runtimes, model weights, datasets and execution receipts are deliberately not
included. No third-party model implementation is copied into this extension.

Per the user's instruction to prioritize having the current code present, no new
full integration tests or real model runs were performed for this extension.
