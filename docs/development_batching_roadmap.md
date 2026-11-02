# Living roadmap for integration into ALF development

Last updated: October 31, 2026 (Increments 0–3 and 4a done locally; 4b next).

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
| 2. Labeling coordination | done locally | Batched labeling during bootstrap and active learning; explicit sequential compatibility for existing single-structure labelers. | Actual coordinator tests cover labeling, partial batches, storage, and retraining. |
| 3. Native PySEQM labeling | done locally | Ground-state batch energies/forces, adapter-local padding and atom-order restoration, and per-member convergence handling. | Native batch results agree with individual evaluations; successful labels survive partial failures. |
| 4. Ground-state sampling and acquisition | in progress (4a done; 4b next) | Minimal physical-inference/MLMD contracts, uncertainty and summed ranking, first-trigger returns, and an in-memory logical-cohort selector. | Shared observations produce equivalent screening, scores, stopping, and selection across execution strategies. |
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
- Rechecked 2026-11-02: official development advanced to `39dcd29` (based on
  `198f125`), which only restricts the GitHub docs, unit-test, and
  package-build workflows to `main`. It was merged into local `development`
  (`47d8835`, workflow files only, no conflicts); the suite still reports 102
  passed in both envs. Development PRs therefore receive no CI, so the local
  suite results recorded here are the test evidence. Official development
  still lacks main's `4f54fad` fixes, which these branches carry.
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

#### Increment 2 — labeling coordination (done locally, 2026-10-31)

- Opt in with a `batching` block in `QM_config`. `mode` is required and
  explicit:
  - `native`: `QM_task` accepts `molecule_objects` and labels them in one
    call.
  - `sequential`: job packing for an existing single-structure `QM_task`
    (`molecule_object`). One task on the labeler's own executors calls it per
    member.
  `batch_size`, `partial_policy`, and `max_wait_seconds` are shared with
  Increment 1. The signature is checked against the mode at startup.
  `full_only` requires `target_queued_QM >= batch_size`, because waiting
  structures count as queued.
- New `alframework/tools/labeling.py` (`LabelingCoordinator`,
  `label_sequentially`). `BatchedTaskQueue` was added to `batching.py`; its
  counts are in structures, so `bootstrap_set`, `target_queued_QM`, and
  `save_h5_threshold` keep their per-structure meaning. It flattens batch
  results into the ordinary `MoleculesObject` list, so `store_current_data`
  and training are unchanged.
- `__main__.py` changes are limited to dispatch, readiness polling, and the
  loop sleep: QM submissions (bootstrap, active learning, `--test_qm`) go
  through `labeling.submit`, then `labeling.submit_ready` runs each pass, and
  `time.sleep(60)` becomes `time.sleep(labeling.poll_delay(60))`. Without
  `batching`, `QM_task_queue` is the ordinary `parsl_task_queue`, tasks are
  built exactly as before, and the sleep stays 60 s. Net diff: +19/−28 lines.
- Partial failures: in `sequential` mode a member that raises is marked
  unconverged with `metadata['labeling_error']`; its batch-mates' labels are
  stored. `native` labelers mark failed members unconverged themselves.
  Unconverged members are skipped by the existing storage path. A whole-task
  failure counts once in `lifetime_failed_QM_tasks` (still in tasks);
  `QM_task_queue.failed_structures` holds the structure count but is not yet
  written to `status`. No retry policy was added.
- Evidence:
  - `tests/test_labeling_coordinator.py` (11 tests) runs the real
    `python -m alframework` entrypoint with local Parsl thread executors and
    mock builder/sampler/labeler/trainer tasks. It covers bootstrap → HDF5 →
    training → MLMD → labeling → second HDF5 → retraining for unbatched,
    sequential (width 2 and 5), and native (width 2 and 5) labeling. One
    bootstrap structure always fails. The width-5 cases can only finish
    through timed-out partial batches and check the shortened sleep.
  - Unit tests cover settings, signature checks, `full_only` validation,
    sequential partial-failure retention, and structure-based queue counts.
  - Full suite: 72 passed in both envs; the coordinator test was stable over
    15 repeated runs per env.
  - Real backend: `--test_qm` with ORCA 5.0.4 (HF/def2-SVP, 1 core, local
    thread executors on `cn211`) labeled a periodic 30-water box through
    `sequential` mode (SCF converged, PBC retained, stored to `qm_test.h5`),
    and again without batching. These were different random builder
    structures, so this checks execution, not numerical agreement. Scratch:
    `/tmp/alf_inc2_orca`.
- Not yet done: a Slurm/Darwin run of a full AL example through the
  coordinator with batching on and off. This is the outstanding
  real-coordinator compatibility check, and it needs a login-node `tmux`
  session.
- Known limitations: batching settings are read at startup and not
  re-read on the in-loop config reload; waiting structures are in memory
  only. The fork's `pyseqm_interface.py` labels one molecule per task and is
  excited-state only; Increment 3 adds the ground-state native batch path.

#### Increment 3 — native PySEQM labeling (done locally, 2026-10-31)

- New `alframework/qm_interfaces/pyseqm_interface.py`, which is ground-state
  only. It is written fresh rather than ported, because the fork module is
  excited-state, one molecule per task, and runs PySEQM in a subprocess.
  - `pyseqm_task(molecule_objects, QM_config)` runs on `alf_QM_executor`;
    select it with `"batching": {"mode": "native", ...}`.
  - `QM_config` keys: `method` (AM1 default), `scf_eps` (1e-8),
    `scf_converger` ([2], Pulay DIIS), `device` (`cpu` default; set `cuda`
    and place `alf_QM_executor` on GPU nodes for GPU labeling), and optional
    raw `pyseqm_parameters`.
  - Per-member `metadata['charge']` / `['mult']` are passed through
    (default 0/1).
- Adapter-local layout: each member is stably sorted heaviest-first (PySEQM
  rejects other orders) and zero-padded to the batch's largest molecule.
  Forces are restored to ALF's input order and trimmed of padding before
  storage. Mixed sizes and compositions share a batch.
- Units and signs: energy = PySEQM `Etot` in eV; forces = PySEQM `force` in
  eV/Å, confirmed to be −dE/dR by finite differences. Use
  `properties_list` factors for other units. The `heat of formation` (`Hf`)
  is not stored.
- Per-member failures: members flagged by PySEQM's `notconverged`, or with
  non-finite results, are marked unconverged with
  `metadata['labeling_error']`; their batch-mates are kept. If the batched
  solve raises, each member is retried alone once so one bad geometry
  cannot discard the batch. This isolates failures; it is not a retry
  policy. Periodic inputs and `excited_states` parameters are rejected.
- PySEQM source: `~/github/PYSEQM` at `31edd88` (`lanl/PYSEQM`). In
  `atomistic` it is pip-installed editable, but the editable finder is
  shadowed by a stale `site-packages/seqm/` directory that contains only
  `MolecularDynamics.py`, so `import seqm.Molecule` fails unless
  `~/github/PYSEQM` is on `PYTHONPATH`. In `alf_env` it is not installed.
  Run the PySEQM tests with
  `PYTHONPATH=/vast/home/pjlohr/github/PYSEQM`.
- Evidence (CPU, `cn211`, float64):
  - `tests/test_pyseqm_interface.py` has 8 tests. Six run without PySEQM:
    sorting/padding/restoration, PBC and excited-state rejection,
    unconverged-member retention, and failed-batch fallback with a fake
    runner. Two use real PySEQM:
    - A batch of [methanol, water, methanol] in non-sorted input order agrees
      with individual evaluations (|ΔE| < 1e-6 eV, |ΔF| < 1e-5 eV/Å;
      observed ≈ 1e-13 eV and 2e-7 eV/Å in the probe).
    - Forces match −dE/dx by central differences for every water atom.
  - Full suite: 80 passed in both envs with PySEQM on the path. Without it,
    `alf_env` reports 78 passed and 2 skipped (optional-dependency
    isolation).
  - Real coordinator: an ALF bootstrap (`python -m alframework`, local
    thread executors, stopped before HIPPYNN training) with a 5-wide mixed
    water/methanol builder and native PySEQM batches of 8 (5 s timeout)
    labeled 16/16 structures. HDF5 `data-0000.h5` had correct species
    ordering. Re-evaluating every stored frame individually reproduced the
    stored labels (max |ΔE| 6e-13 eV, max |ΔF| 5e-7 eV/Å). Scratch:
    `/tmp/alf_inc3_pyseqm`.
- Not yet done: GPU (`device: cuda`) execution and throughput/memory
  versus batch size; this node has no GPU. Retraining with PySEQM labels
  through HIPPYNN was not run here; storage/retraining through the
  coordinator is covered by Increment 2's tests.

#### Increment 4 — ground-state sampling and acquisition (in progress)

Decisions (2026-10-31, with the user):

- Keep development's flat sampler configuration (`dt`, `maxt`, `Escut`,
  `Fscut`, `Ncheck`, temperature/density ranges, `MLMD_calculator_options`,
  ...). Add only `driver`, `acquisition`, and `batching` blocks. The fork's
  `sampling`/`inference` schema is not adopted.
- Defer the physical-force screen (`max_force_cutoff`) until GALF is
  introduced. Development has no such screen, so adding one would change
  scientific behavior.
- Development has no ranking today; the uncertainty and summed scores come
  from the fork. The summed score is `w_energy·Es/Escut + w_force·Fsrms/Fscut`,
  with `w_energy` and `w_force` required and no defaults. The excited-state
  gap term is excluded.
- Factor Increment 2's buffer/submit/poll logic into a shared batched
  submitter used by both labeling and sampling, so `parallel_samplers`
  counts structures and one sleep honors both buffers' deadlines. Labeling
  behavior and tests are unchanged.

Sub-steps, each a separate commit:

| Step | Status | Deliverable | Acceptance |
| --- | --- | --- | --- |
| 4a. Inference contract and scoring | done locally | Per-member unbiased energies/forces from a list of structures (adapter for the existing ASE ensemble); `Es`/`Fs`/`Fsmax`/`Fsrms` with `MLMD_calculator`'s population-std formulas; eligibility (existing thresholds, `min_time`, `distcut` rejection); uncertainty and summed ranking; stable frame identity and deterministic ties. No `__main__.py` change. | Parity with `MLMD_calculator` on fakes and on the trained Darwin water ensemble (CPU). |
| 4b. Multi-trajectory MLMD, first-trigger | planned (next) | Opt-in task taking `molecule_objects`; `driver.strategy` `single`/`batched`, `driver.engine` `ase`; minimal engine contract (step, set temperature, scale cell, freeze, state); per-trajectory seeds from molecule IDs; configurable device; shared batched submitter in the coordinator. `simple_mlmd_sampling_task` untouched. | Identical frames/metadata for single vs batched from seeded inputs; real coordinator with mocks; real-coordinator CPU run with the trained water model. |
| 4c. In-memory cohort top-K | planned | `acquisition.selection = {mode: cohort_top_k, cohort_size, k}` and `trajectory_policy` `stop`/`continue`; incremental admission; model generation pinned per cohort; terminal outcomes per trajectory; exact global top-K with safe local pruning; handoff to labeling. No persistence. | Replay at widths 1, 2, 8, 32, 50 with reversed completion gives identical selections; ties, empty trajectories, < K candidates, cohort > capacity, failure after an eligible frame, model promotion mid-cohort. |

Out of scope for Increment 4: TorchSim/ALCHEMI (Increments 5/6), native
batched HIPPYNN inference (Increment 5), excited states/gap/LCM, plugin
bias/force controls, and cohort persistence.

Evidence available: the trained 8-member water ensemble at
`examples/simple_water_darwin_run_test/models/model-0000` (untracked,
July 2026) loads on CPU with `HIPNN_ASE_load_ensemble(..., device='cpu')`
in about 9 s. `MLMD_calculator` exposes per-member contributions, followed
by the well potential as a final extra entry.

4a record (done locally, 2026-10-31):

- New `alframework/ml_interfaces/ensemble_inference.py`:
  - `ASEEnsembleEvaluator.predict(atoms_list)` returns unbiased per-member
    energies `[models, systems]` and per-system forces
    `[models, atoms_i, 3]`. It evaluates one structure at a time and applies
    no UDD, well potential, or thresholds.
  - `ensemble_statistics` returns `Es`, `Fs`, `Fsmax`, `Fsrms`, and the mean
    energy.
- New `alframework/samplers/acquisition.py` (the name is free on
  development):
  - `acquisition_settings`: `uncertainty` (default) or `summed`, with
    required `w_energy`/`w_force`.
  - `assess_frame` returns `rejected` (`distcut`), `eligible`, or `continue`,
    only once `min_time` is reached. `ranking_score`, `frame_id`
    (`<trajectory_id>:step-NNNNNNNNN`), and `top_k` order ties by score,
    then trajectory ID, then step.
- Nonpositive cutoffs (examples use `Escut`/`Fscut = -1` to force every
  frame to trigger) still trigger under `uncertainty` and rank as `inf`
  ties, broken by trajectory/step. `summed` rejects nonpositive cutoffs at
  configuration time.
- Evidence: `tests/test_acquisition.py` has 22 tests.
  - Statistics match `MLMD_calculator` with and without the well potential,
    which is excluded from the physical predictions; mixed sizes work.
  - Eligibility matches a transcription of `mlmd_sampling`'s stopping logic
    in 10 boundary cases. A high summed score alone is not eligible.
    Scores match hand values. Top-K is invariant under 10 input shuffles.
  - Trained Darwin water ensemble (8 HIPPYNN members, CPU, float32): `Es`,
    `Fs`, `Fsmax` match `MLMD_calculator` within `rtol=atol=1e-5`. The test
    skips when that untracked model directory or `hippynn` is absent.
  - Full suite: 102 passed in both envs, with PySEQM on the path. The 8
    added warnings are HIPPYNN deprecation notices raised while loading the
    trained checkpoints.
- Nothing calls these modules yet; existing sampling is unchanged.

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
