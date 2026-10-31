# Living roadmap for integration into ALF development

Last updated: October 31, 2026 (Increments 0–1 done locally).

This is the shared plan for gradually bringing the fork's intended capabilities
into `lanl/ALF:development`. It is separate from ongoing implementation on
`feature/unified-sampling-drivers`. Experimental implementations provide source
material and validation evidence; each development increment receives its own
review and acceptance checks.

The first release covers **batched reference labeling, ground-state MLMD,
uncertainty and summed ranking, and pooled top-K selection**. Later increments
extend sampling, acquisition, model interfaces, initialization, and workflows.

## Guiding constraints

**Change as little core ALF as possible, especially `alframework/__main__.py`.**

- Limit entrypoint changes to necessary dispatch, harvesting, timeout polling,
  and capacity accounting. Keep scientific policy out of that module.
- Keep batching/coordinator helpers in `tools`, sampling/acquisition/engine
  implementations in `samplers`, and model-specific adapters in the existing
  interface packages. Preserve upstream's callable/import-string extension style.
- Preserve existing task names, callable signatures, configuration files,
  `MoleculesObject` values, Parsl execution, storage conventions, and training APIs.
- Make new behavior opt-in. Existing main/development examples and normal
  workflows must continue working with their existing configurations.
- Preserve existing ASE samplers and trainers. Keep optional GPU and RDKit
  dependencies out of the base coordinator.
- Address ordinary batching correctness and demonstrated compatibility
  regressions. Broader edge-case hardening and unrelated cleanup are follow-up
  work, rather than prerequisites for every increment.

Each PR explains why its core changes are necessary and supplies compatibility
evidence for affected existing examples. A change in scientific defaults must
be a separate, explicit decision.

## Branch and maintenance workflow

Prepare small PRs from new branches in `pjlohr-LANL/ALF`, targeting official
`development`. Use isolated fork worktrees and preserve the experimental
branches. Preserve the official checkout's untracked examples and other local
work.

Before feature integration, synchronize development with current main. Recheck
their relationship before subsequent PRs, and keep fork main tracking official
main. At planning time, official main was `206ea44` and development was
`198f125`; implementation must verify live tips rather than assume these remain
current. The recent main PBC and HIPPYNN fixes belong in the integration baseline.

The roadmap is a living document. Each increment records its status,
dependencies, source implementation/commits, PR, acceptance evidence, and known
limitations. Update it when work merges or priorities change. Changes in order
should preserve the dependency and compatibility gates below.

## First-release increments

Status values: **planned**, **in progress**, **done locally** (committed on
local `development`, not pushed or merged upstream), and **merged** (in
official `lanl/ALF:development`). No increment is merged yet.

| Increment | Status | Deliverable | Acceptance gate |
| --- | --- | --- | --- |
| 0. Roadmap and baseline | done locally | Publish this document, synchronize development with main, and record existing-example/test behavior. | Baseline results and unavailable dependencies are documented. |
| 1. Shared batching helpers | done locally | Compatible grouping, waiting buffers, timeout submission, actual-width accounting, output normalization, and batch-builder ID preservation. | Focused tests pass; existing execution is unchanged until batching is selected. |
| 2. Labeling coordination | planned (next) | Batched labeling during bootstrap and active learning; explicit sequential compatibility for existing single-structure labelers. | Actual coordinator tests cover labeling, partial batches, storage, and retraining. |
| 3. Native PySEQM labeling | planned | Ground-state batch energies/forces, adapter-local padding and atom-order restoration, and per-member convergence handling. | Native batch results agree with individual evaluations; successful labels survive partial failures. |
| 4. Ground-state sampling and acquisition | planned | Minimal physical-inference/MLMD contracts, uncertainty and summed ranking, first-trigger returns, and an in-memory logical-cohort selector. | Shared observations produce equivalent screening, scores, stopping, and selection across execution strategies. |
| 5. TorchSim dynamics | planned | Supported NVIDIA/AMD ground-state batched MLMD. | CPU, NVIDIA, and Darwin MI250 mechanics and integration acceptance pass. |
| 6. ALCHEMI dynamics | planned | NVIDIA batched MLMD through the same contracts, in its separate environment. | Shared conformance tests and NVIDIA acceptance pass. |
| 7. Combined release validation | planned | Opt-in examples and integration/performance results. | All single/batched dynamics x single/batched labeling combinations complete representative ALF workflows. |

Port narrowly scoped implementations and tests. Each increment must be
independently reviewable and tested before dependent increments merge. Inspect
PBC propagation in the multi-structure condensed-phase builder as a focused
prerequisite; retain the recent main fixes. (Done in Increment 0; see below.)

### Progress log

Record each increment's commits, evidence, and limitations here as it lands.

#### Increment 0 — roadmap and baseline (done locally, 2026-10-31)

- Live tips checked on 2026-10-31: official main `206ea44`, development
  `198f125`. Development had no commits absent from main, so local
  `development` was fast-forwarded to `206ea44`; this brings in the main PBC
  and HIPPYNN fixes (`4f54fad`). Not pushed.
- Test baseline: `pytest` on development reports 47 passed both before
  (`198f125`) and after (`206ea44`) synchronization, in both the `atomistic`
  (Python 3.12, ASE 3.27, Parsl 2026.02) and `alf_env` (ASE 3.29, Parsl
  2026.07) conda environments.
- Run tests with `PYTHONPATH` or from the checkout root. Neither env imports
  this checkout by default: `atomistic` resolves `alframework` to
  `~/github/ALF` and `alf_env` to the fork checkout. `pytest.ini` sets
  `pythonpath = .`, so `pytest` from the repository root is correct. Ad hoc
  scripts must set `PYTHONPATH`.
- PBC prerequisite: `simple_multi_condensed_phase_builder_task` still built
  `Atoms(cell=...)` without `pbc=True`, so multi-builder condensed-phase
  systems (e.g. `examples/simple_water_multi_builder`) were nonperiodic even
  after main's fix to the single builder. Fixed the same way, with a
  regression test covering both builders and multi-builder ID order.
- Existing examples were not re-run end to end for this increment. Earlier
  Darwin runs (untracked `examples/simple_water_darwin_run_test` and
  `examples/fine_tuning_darwin_test`, July 2026, pre-sync) completed
  bootstrap+training and fine-tuning with ORCA 5.0.4 and HIPPYNN. Re-running
  one of them through the real coordinator is the compatibility check for
  Increment 2, the first increment that changes `__main__.py`.
- Unavailable dependencies on the test node (`cn211`, no GPU): PySEQM
  (`seqm` is an empty namespace in `atomistic`, absent in `alf_env`),
  TorchSim, ALCHEMI, GPU4PySCF, CUDA devices. ORCA 5.0.4 is available at
  `/usr/projects/ml4chem/Programs/orca_5.0.4/orca`; PySCF 2.14 is in
  `alf_env` only.

#### Increment 1 — shared batching helpers (done locally, 2026-10-31)

- New module `alframework/tools/batching.py`, with no core changes and no
  imports from existing modules other than `MoleculesObject`:
  - `batching_settings(config)` returns `None` unless the task config has a
    `batching` block (`batch_size`, `partial_policy` = `timeout` (default) |
    `full_only`, `max_wait_seconds` = 5.0 by default). Unknown keys are
    rejected.
  - `BatchBuffer` groups compatible molecules in memory. It releases full
    batches immediately and partial batches once their oldest member reaches
    the timeout. It supports `flush=True` for final drains and
    `seconds_until_ready()` for polling. Release order is deterministic
    (insertion-ordered buckets).
  - `default_batch_key` groups only by PBC; mixed sizes, compositions, and
    atom orders may share a batch. Adapters with stricter needs pass
    `key_function`. A selected excited state or a pinned model generation
    can later extend the key without changing the shared helper.
  - `flatten_molecule_output` normalizes single/nested/empty output while
    preserving order and object identity (batch-builder IDs survive
    regrouping).
  - `TaskWidths` records the actual input width per submitted task
    (unrecorded tasks count as 1), and `poll_delay` caps coordinator sleep at
    the nearest partial-batch deadline.
- Deliberately not ported from `sampler_batching.py`: excited-state selection
  inside the shared key, config-reload signatures, persisted buffers, and the
  unified/portable/legacy config dialects. Reload validation will be added in
  Increment 2 if the coordinator needs it.
- Evidence: `tests/test_batching.py` has 13 tests (settings validation,
  full/partial/timeout/full-only release, deadline retention, PBC separation,
  custom keys, ID preservation through regrouping, width accounting, poll
  delay). The full suite reports 61 passed in both envs. Nothing calls the
  module yet, so existing execution is unchanged.

### Batching and reference labeling

- Preserve molecule IDs, metadata, atom order, cells, and PBC. Builder width,
  dynamics width, labeling width, and acquisition cohort size are independent.
- Group by adapter-declared compatibility; splitting or combining builder
  outputs does not change molecular identity.
- Keep `QM_task` as the import-string entrypoint. Batch-capable implementations
  accept multiple molecules; compatibility adapters preserve single-object
  labeler interfaces. Keep existing training entrypoints and return conventions.
- Distinguish native multi-structure evaluation from sequential job packing in
  configuration and reporting. Keep units, force signs, convergence, and backend
  restrictions inside adapters.
- Retain successful labels and record unsuccessful members. Add no automatic
  retry policy beyond already-configured task handling.
- Use PySEQM as the first native batch labeler. PySCF/GPU4PySCF may use job
  packing when a suitable single-structure adapter is available. No supported
  native independent-geometry batch energy/force API was identified in its
  documented interface during planning; recheck capabilities when adding an
  adapter. See the [GPU4PySCF documentation](https://pyscf.org/user/gpu.html) and
  [PySEQM batch-input documentation](https://lanl.github.io/PYSEQM/Initialization.html).

Treat labeling as a reference-provider boundary, even while retaining the
historical `QM_task` name. A later foundation-MLIP teacher must be able to supply
labels through the same batching path without changes to the coordinator.

### Dynamics, screening, and identity

- Select grouping and engine explicitly through `driver.strategy` and
  `driver.engine`. Keep physical settings in the sampler's batching block.
- Support mixed-size ground-state systems, initial velocities, independent
  stopping, temperature/density schedules, and supported periodic checkpoints.
- Inference supplies physical predictions; sampling owns scientific checks;
  acquisition owns eligibility/ranking/selection; engines own integration,
  units, and synchronization.
- Preserve physical-force and geometry screening when configured. Screen
  unbiased physical predictions, excluding UDD/LCM/confinement forces.
- Keep sampling screens, optional post-label dataset screens, and trainer
  outlier filters distinct. Preserve current defaults and upstream training
  filter semantics; do not automatically enable one from another.
- Preserve genuine atom mappings through batching, labeler-specific sorting,
  and storage. Chemical symbols alone do not identify equal-element atoms.
  Existing untagged workflows remain usable without new mandatory identity tags.
- Submit full batches immediately and partial batches after five seconds,
  including polls with no new builder output. Account for actual input widths.

### Acquisition and trajectory outcomes

- Preserve ordinary energy/force uncertainty eligibility. Uncertainty acquisition
  uses the existing normalized ranking; summed acquisition uses the existing
  weighted sum for ranking only. A high summed score alone is not eligibility.
- First-trigger mode stops and returns one eligible frame per trajectory
  immediately. Pooled mode requires explicit cohort size and K.
- Select pooled top-K over a fixed logical cohort, independently of physical
  task boundaries, with either stop-at-first or continued collection.
- Pin one model generation per cohort. Admit and execute members incrementally
  so a cohort larger than available worker capacity can progress.
- Require full configured admission, with no admission timeout. Use stable
  trajectory/frame identities and deterministic ties. Add no mandatory
  per-trajectory quota or structural diversity filter.
- A later trajectory failure does not invalidate earlier eligible frames. Keep
  those frames in the pool and label them if they rank in the final top-K.
  Exclude the invalid frame and record the terminal outcome separately.
- Allow unaffected trajectories to continue when the engine can isolate a
  failure. Finalize after full admission and terminal outcomes for all members,
  including empty, rejected, or failed trajectories.
- For a whole-task failure, retain candidates already returned to the
  coordinator and use existing task-failure handling. Recovering frames held
  only inside a crashed worker is outside the initial recovery scope.
- Preserve old task-local selection settings; do not silently reinterpret them
  as new cohort-wide budgets.

### Recovery boundary

Keep waiting batches and logical cohorts in memory in this initial development
release. Add no persistent buffers, task journal, staged workflow, or new
restart/shutdown controls. Document that unfinished in-memory pools are not
restored after process restart. The experimental fork's durable implementation
is source material for the later recovery increment, not a prerequisite here.

## Subsequent capability increments

These intended development capabilities remain on the roadmap. The numbering
is the working order; independent initialization/replay contributions may proceed
once their helper and identity requirements are available.

| Increment | Capabilities to integrate | Important boundary |
| --- | --- | --- |
| 1. Composable sampling | Separate propagation from UDD, confinement, and other bias/force controls; support callable extensions and grouped non-MD evaluations. | Preserve physical predictions, formulas, switching order, and replica-local state. |
| 2. Additional acquisition strategies | Custom import-string strategies and geometry-based demonstrations, followed by production distance-based acquisition and diversity filtering. | Eligibility, ranking, trajectory policy, and selection remain separate; future selectors declare their retention needs. |
| 3. Model/training handoffs and distillation | Opaque model-artifact references, model-independent inference adapters, and foundation-MLIP reference labeling. | Record teacher identity, version, units, and label provenance separately from the student sampling model. |
| 4. Molecular excited states | PySEQM/GPU4PySCF labeling, multi-state HIPPYNN training, derived gap targets, state selection, gap diagnostics, and LCM. | Preserve per-state convergence, state/atom mappings, physical screening, and the distinction between predictions and propagation controls. |
| 5. RDKit initialization | Reference-centered KDG geometry generation, seed-specific screening, and FPS selection, with reproducible settings/provenance. | Optional preprocessing/builders; unlabeled seed geometries are not labeled training rows. |
| 6. Geometry-only HDF5 replay | Reuse stored geometries for sampling or relabeling, with original identities and source provenance. | Do not accidentally carry old labels into a newly requested labeling operation. |
| 7. Optional additional labels | Preserve optional nonadiabatic-coupling labeling and storage, including state-pair and atom-axis mappings. | Computing/storing NACs does not automatically make them energy/force training targets. |
| 8. Durable recovery and staged workflows | Persist batching/cohort handoffs; add retained-label budgets, ALF-to-GALF/configuration stages, exact dataset checkpoints, and endpoint training. | Preserve ordinary runs as the default; budgets count retained labeled frames, not candidates or submitted jobs. |
| 9. Further model and sampling extensions | Additional production model families, reactive batching, and separately planned methods. | Follow the same minimal-core-change and existing-workflow compatibility gates. |

The excited-state increment includes explicit topology validation and
topology-aware replay/storage where configured. Existing ordinary workflows
must not acquire a new fixed-topology requirement.

Budgeted workflows must distinguish selected candidates, labeling attempts,
successful labels, stored/retained frames, and rows surviving trainer filters.
Report these stages separately. Exact checkpoints and final endpoint training
refer to the chosen retained-data lineage; routine retraining and raw-store
overshoot are separate concepts.

### Known implementation follow-ups

The model/training-boundary increment must close two items recorded during the
recent fork review:

- The training handoff must tolerate valid trainer callables whose signature
  omits `ML_config`.
- Move model-family artifact validation behind adapters; common final-training
  validation must not assume HIPPYNN-style `model-NN` member directories.

## Compatibility and acceptance

Use tracked main/development examples as the practical compatibility baseline:
single- and multi-builder water, periodic condensed-phase systems, seeded
learning, fine-tuning, and reactive sampling. Preserve existing configurations
and task selections without rewriting them to adopt batching.

For each increment, run the current development suite and focused regressions.
Exercise representative workflows through the real coordinator with local test
backends. Run external backends where dependencies/resources are available,
distinguishing mocked integration checks, actual backend execution, unavailable
dependencies, and pre-existing failures. Avoid expanding an increment into
unrelated repair work.

First-release checks cover:

- All single/batched dynamics x single/batched labeling combinations through
  bootstrap, storage, and retraining.
- Uncertainty and summed ranking, first-trigger and pooled selection, and both
  stopping policies.
- Identical recorded observations grouped at widths 1, 2, 8, 32, and 50, with
  reordered task completion and identical selected frame IDs.
- Ties, empty trajectories, fewer than K candidates, cohorts larger than worker
  capacity, and eligible frames retained before a trajectory failure.
- Partial batches, actual capacity accounting, partial labeling failures,
  atom-order restoration, units, initial velocities, freezing, schedule timing,
  force refresh, and supported PBC.
- Optional-dependency isolation and unchanged screening/training defaults.

GPU-dependent increments record source revisions, dependencies, hardware,
numerical errors, throughput, memory, and selection latency. Retain the approved
float32 inference tolerances, `rtol=1e-5` and `atol=1e-5`; report QM precision
and convergence criteria separately. Report CPU correctness, hardware acceptance,
and production-potential accuracy separately. MI350 execution requires its own
hardware validation.

## Source material and contributor handoff

The portable implementation through `2eaa681` and unified strategy work through
`2e1cb64`/`b8248cc` are starting references, not wholesale integration units.
The checkpoint at `78ac39a` records remaining implementation work. Useful source
documents on the experimental branch include:

- `docs/unified_sampling.md` and `docs/portable_batched_sampling.md`.
- `docs/physical_force_screen.md` and `docs/storage_identity.md`.
- `examples/rdkit_initialization/README.md` and the HDF5 replay builder.
- `examples/excited_state_gpu4pyscf_nacr/README.md`.
- `docs/staged_workflows.md` and `docs/unified_sampling_progress_20261028.md`.

These references describe experimental-branch capabilities; some will not yet
exist on an early development integration branch. Keep branch/source identity
with validation records and update this document as capabilities land.
