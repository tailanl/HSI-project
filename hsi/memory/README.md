# Memory

This is the consolidated implementation, not a loader for historical method
directories. Compatibility schema identifiers are retained so their strict field
contracts are not silently weakened. New receipts bind the actual package source
files; they do not claim the old selector or admission-authority source hashes.

## Implemented interfaces

```python
from hsi.memory.facts import build_scene_facts, load_scene_facts, load_descriptors
from hsi.memory.views import save_selection, query_views, validate_lookup
from hsi.memory.matching import load_current_surface, match_shapes, project_candidate, retrieve_current
from hsi.memory.extraction import derive_records
from hsi.memory.store import MemoryStore
from hsi.memory.runtime import prepare, query, validate_lookup as validate_experience_lookup
```

- `build_scene_facts(final_scene_path, output_dir)` publishes a new immutable
  descriptor directory from a complete, instruction-independent Stage1A receipt.
  `load_scene_facts(path)` verifies all retained geometry/semantic identities and
  direction evidence without model inference or raster recomputation. All unknown
  objects and all audited directions are retained. Each fact has zero credit.
- `save_selection(facts_path, view_receipt_path, store_root)` verifies a native
  `hsi.stage2.views` **full-mode, 48-camera** selection and saves an independent
  camera JSON, crop hint and complete historical audit. It calls the native
  selector validator and the current Stage1 binder. Subset-mode selections are
  deliberately not admitted for new bank entries in this version.
- `query_views(facts_path, stage1_path, store_root)` returns proposal-only camera
  hints. An absent store is a read-only miss. The key binds the fixed scene,
  object, true support faces, audited direction, owner mask and geometry hashes.
  The native Stage2 selector consumes those hints and recomputes its current
  crop/mask/depth checks, with a full-search fallback. A hint never grants a pose,
  semantic-review or physics pass. `validate_lookup(...)` repeats the current
  source/context checks immediately before separate consumers use a saved lookup.
- `match_shapes(source, current)` and `project_candidate(record, context,
  sdf_check)` are geometry diagnostics, not admission authorities. Metric scale,
  observed holes and the audited orientation are preserved. SDF evidence checks
  contact/root points only; full-body refinement and final checks remain required.
- `derive_records(context, motion, tail, triangles)` extracts low-dimensional
  contact/normal/root-offset/facing fields from the actually observed final four
  frames, using the original deterministic medoid and true triangle support.
  Extraction is not proof that the episode succeeded.
- `MemoryStore` contains the original immutable journal, snapshots, atomic
  pointer, CAS publication, single-episode credit deduplication, parent/local
  promotion and live critical-failure quarantine algorithms. The fixed authority
  is now `hsi.memory.authority.admit_episode`, with a new actual source manifest.
- `runtime.prepare(...)`, `query(...)` and `validate_lookup(...)` now retain the
  v2 frozen-cycle/live-quarantine retrieval path, with native Stage1 bindings,
  current facts, explicit neutral-body identity and current-mask SDF validation.
  `original_identity_from_stage1(...)` preserves the original complete same-target
  walk/sit task independently of the condensed contact step. This newest runtime
  migration has only been syntax-checked, not end-to-end exercised; it grants no
  credit and does not replace the still-unintegrated production authority.

## Production admission is intentionally unavailable

The current research implementation is preserved, not discarded: see the
repository's `extensions/affordance_memory/README.md`. Its current production entry and
required implementation chain are listed there. That extension has not been
fully wired into the native package; source availability is not a new runtime
verification or an authorization to grant credit.

The complete real-motion authority has **not** been integrated into this package.
Its current source and the geometry-memory fast-path dependency graph are present
in [the research extension](../../extensions/affordance_memory/README.md), with
one explicit current entry graph and original/new source hashes. They are code
preservation for the current method, not a claim of completed native integration.
Production `admit_episode` and `MemoryStore.record_attempt` therefore raise
`ProductionAdmissionUnavailable`; a caller's `passed=True`, a static pose pass,
an extracted record or a Qwen answer cannot substitute for the missing authority.
An empty production store can still be opened and queried as M0. Explicit
`purpose="test_fixture"` stores exercise transactions using synthetic nine-gate
fixtures; fixture successes never count as real credit and cannot cross-read
production stores.

Before enabling production learning, integrate the actual full-motion
34 gates, actual terminal meshes, the retained static 23 checks, original task
identity, actual semantic review and authoritative extraction binding. The old
online producer has not been replaced by a permissive shim. The native preparer
is present; no additional production-authority development or full testing was
performed after the user's instruction to prioritize preserving the current code.
Stage1-route and Stage3-guidance field namespaces are retained in the schema, but
their production experience producers remain unregistered. The real retrieval
and extraction path currently supports typed Stage2 sit contact only.

## Unchanged safety boundaries

View hard thresholds retain crop surface coverage `.95`, visible surface `.85`,
300 visible surface pixels, owner crop `.98`, owner visibility `.90`, depth
tolerance `.03 m` and 5,000 surface samples. Hypothetical body probes and camera
facing remain ranking terms, not action-type or human-pose acceptance gates.

Experience activation still requires at least 8 successful episodes from 4 scene
families and a Beta 5% lower confidence bound of at least `.7`. A critical failure
is visible even to older frozen read snapshots. Recovery requires postcritical
evidence in 2 families, and each local child needs its own fresh scene evidence.

The banks do not store NavMesh graphs, complete poses, motion arrays or reusable
Qwen conclusions. Stage1A still retains its scene semantics and references to
their original independent evidence; those references do not grant a new task a
pass. Current route receipts and original Qwen evidence belong to task/scene
evidence directories, not transferable experience payloads. Camera/facts receipts
still verify referenced source evidence: they are not self-contained archives
that allow deletion of their source files.

## Tests

`tests/test_memory_*.py` uses generated synthetic geometry/evidence only and needs
no old source tree, dataset, model, network or GPU. It covers the original
geometry/transaction negative cases, complete new camera save/query, explicit
production rejection, and frozen numerical observations measured from both old
and consolidated kernels. The numerical checksum tests round to 10 decimal
places only for JSON comparison; the implementation does not round its geometry.
