# ALF development roadmap

Last updated: November 7, 2026.

This folder is the shared plan for bringing the fork's intended capabilities into
`lanl/ALF:development`.

| File | Covers | Record here |
| --- | --- | --- |
| This file | Status, standing rules, environments, acceptance rules | Status changes, test and docs records, environment policy |
| [`batching.md`](batching.md) | Increments 0-7 and their known follow-ups | Batching and ground-state results and follow-ups |
| [`excited_states.md`](excited_states.md) | The E1-E6 plan and Increments 8-11B | Excited-state results, including the E6 pilot |
| [`future.md`](future.md) | What's left: order of work, quick wins, and grouped plans | New plans and priority decisions |
| [`exascale.md`](exascale.md) | File I/O risks at scale, fix priorities, and node-local storage | I/O audit findings, Darwin profile results, and changes to the I/O priorities |

## Current status

- **Increments 0-10 and 11A are complete and merged.** Increment 7 was closed by user
  acceptance on November 5. The fixed-keto AcAc ALF loop passed Increment 11A.
- **Increment 11B / E6 is complete within the bounded CPP9 pilot scope.** The
  ordinary DCM active-learning loop passed with 32 fresh bootstrap frames and eight
  propagated, stored samples covering S0–S3 in a promoted four-member ensemble;
  see [Increment 11B](excited_states.md#increment-11b--completed-cpp9-active-learning-qualification-e6).
  Ten-node scaling, predictive accuracy and long-time stability remain unqualified.
- **Gap-seeking mechanics passed on all three engines**, and the ordinary
  base-to-gap-seeking sampling-window loop passed;
  see the [bounded hardware record](#gap-seeking-and-sampling-window-hardware-qualification--passed-2026-11-06).
- **Multi-node ensemble training is built;** its real run on Darwin is pending. So is confirmation of
  the fixes for allocations held after `--test_ml` and for dead executors. See the
  [record](#multi-node-training-2026-11-06).
- **Next:** safe restarts (tabled for now; see [`exascale.md`](exascale.md)), teacher labeling, then new
  uncertainty and acquisition methods. Long runs and staged campaigns come last, and may be
  removed; see [Order of work](future.md#order-of-work).
- **No merge to `main` is planned** (user decision, November 6).
- **Outside current acceptance:** legacy NeuroChem and MI350.

## Quick start for contributors and agents

Read these sections before starting work:
1. [Guiding constraints](#guiding-constraints)
2. [Branch and maintenance workflow](#branch-and-maintenance-workflow)
3. [Development and test environments](#development-and-test-environments)
4. [Increment 7 close-out](batching.md#increment-7-close-out) for accepted evidence and
   exclusions, then [the E1–E6 plan](excited_states.md#excited-state-alf-plan-e1e6), [Increment 8](excited_states.md#increment-8--completed-excited-state-contracts-and-labeling-2026-11-05), and
   [Order of work](future.md#order-of-work).

Standing rules:
- Commit locally on `development`. Never push; the user pushes.
- Verify live remote tips with `git ls-remote origin`.
- Keep each change small, opt-in, and separately reviewable. Leave
  `__main__.py`, task signatures, and existing configurations unchanged unless
  the section says otherwise.
- Do not modify untracked example directories, the user's personal conda
  environments, or the HIPPYNN/PySEQM checkouts.
- Run live campaigns from a frozen source snapshot and a frozen environment.
  Keep generated data outside Git, under `/vast/home/pjlohr/ALF_LANL/validation/`.
  Never infer a gate from a startup check.
- Record each result with its source revision, environment, and evidence path,
  and update the status table in this file and the relevant section. The table at the
  top of this file says which file each record belongs in.
- The fork is source material, not an integration target. Its own roadmap copy
  is stale; this folder is authoritative.

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

The current integration uses local `development`; agents commit locally and
the user handles pushes to official `development`. Keep changes small and
reviewable. When a separate contribution branch/PR is requested, prepare it in
`pjlohr-LANL/ALF` and target official `development`. Use isolated worktrees and
preserve experimental branches, untracked examples, and other local work.
Leave `main` unchanged except for separately authorized synchronization work.

Before feature integration, synchronize development with current main. Recheck
their relationship before subsequent PRs, and keep fork main tracking official
main. At planning time, official main was `206ea44` and development was
`198f125`; implementation must verify live tips rather than assume these remain
current. The recent main PBC and HIPPYNN fixes belong in the integration baseline.

The roadmap is a living document. Each increment records its status,
dependencies, source implementation/commits, PR, acceptance evidence, and known
limitations. Update it when work merges or priorities change. Changes in order
should preserve the dependency and compatibility gates below.

### Test suite scope and validation harnesses (2026-11-05)

`tests/` covers only `alframework`, with one file per module or area. The
Darwin validation harnesses and their tests were moved out of this repository
(from `508fc1e`):

- `examples/simple_water_campaign/`: the Increment 7 campaigns.
- `examples/excited_state_labeling/`: the Increment 8 labeling qualification.
- `examples/torchsim_benchmark/`: the engine benchmark.

They now live in the local repository `/vast/home/pjlohr/ALF_LANL/alf_validation/`.
Its `assemble.sh` combines an ALF revision with the harnesses into one
committed tree, which is the form the harnesses expect. Earlier entries in this
roadmap still cite the old `examples/...` paths; they refer to those historical
locations. Recorded runs keep their own frozen source copies under
`validation/`.

Before removal, the harness tests that exercised production code were moved
into ALF's own tests:
- `SamplerCoordinator` model pinning, to `test_sampler_coordinator.py`;
- `LabelingCoordinator` timeout tails, to `test_labeling_coordinator.py`;
- nonperiodic storage without `cell`, to `test_tools.py`.

Split suites were merged into these files:
- `test_qm_interfaces`;
- `test_pyseqm`;
- `test_samplers`;
- `test_hippynn_interface`;
- `test_excited_states`;
- `test_tools`.

Three files were renamed: `test_gpu`, `test_gpu4pyscf`, and `test_examples`.
Every in-scope test ID was preserved.

Results after the cleanup:

| Environment | Passed | Skipped |
| --- | ---: | ---: |
| `atomistic` | 338 | 40 |
| Fresh torch-free install | 321 | 24 |
| `alf_alchemi_02` | 356 | 22 |

The 65 harness tests pass in an assembled `alf_validation` tree.

The untracked July 2026 Darwin runs `examples/simple_water_darwin_run_test/`
and `examples/fine_tuning_darwin_test/` (464 MB) moved to
`/vast/home/pjlohr/ALF_LANL/validation/darwin_runs_202607/`. Older entries cite
their former paths; their own `parsl_configs.py` still names the old location.

Tests no longer read that run's trained 8-member checkpoint, because it cannot
exist on a fresh clone. The parity tests (ensemble statistics against
`MLMD_calculator`, batched and legacy HIPPYNN inference across engines, and
concurrent member loading) use a session fixture instead: a two-member HIPPYNN
water ensemble trained on CPU in about 10 s. They now run in any environment
with HIPPYNN, including the HIPPYNN CI job. They check code-path parity, not
potential accuracy.

The smaller ensemble exposed a real ordering bug.
- `HIPNN_ASE_load_ensemble` returned members in unsorted glob order, while the
  batched loader sorts.
- Ensemble means and spreads were unaffected, but per-member comparisons
  differed.
- Members are now loaded in sorted order (`eb20ae9`), with a regression test.

| Environment | Passed | Skipped |
| --- | ---: | ---: |
| `atomistic` | 339 | 40 |
| `alf_alchemi_02` | 357 | 22 |
| Fresh torch-free install | 322 | 24 |

### Test suite review and Actions coverage (2026-11-05)

A coverage and readability review found 59 cases that no Actions job could run.
A new `engines` job installs rdkit, TorchSim, ALCHEMI, and PySEQM `52850b4`, and
fails if any of them is unusable, so that a broken install cannot pass by
skipping. Actions now reports skip reasons.

The review removed 17 cases that added no coverage, and moved shared driver setup
and engine code into `tests/helpers/`. Unlike the earlier consolidation, it renamed
tests, so older entries may cite earlier test names.

**Naming tests.** State what should be true, one behavior per name, without `and`.
Keep the name of the tool, method, or component under test, because a test name is
the first thing seen when CI fails: `test_rdkit_seed_preparation_is_reproducible`,
`test_langevin_holds_the_target_temperature`. Tests shared by both packed engines
show the engine in their test ID, such as `[alchemi]`. Give parametrized cases
readable `ids=`.

**Test files.** File names stay generic, so one file can hold several tools:
`test_qm_interfaces.py` can hold ORCA and more, and `test_ml_interfaces.py` can hold
NeuroChem, HIPPYNN, and future models. Each test file starts with a docstring saying
what it covers, what is real or faked, and what makes its tests skip. `test_gpu.py`
became `test_gpu_selection.py`, because it needs no GPU.

Results on Actions (fork run `42173316850`, Ubuntu, Python 3.12, CPU):

| Job | Passed | Skipped | Job time |
| --- | ---: | ---: | ---: |
| `engines` | 533 | 0 | about 5 min |
| `hippynn` | 468 | 65 | about 3 min |
| `base` (torch-free) | 404 | 98 | about 1 min |

`engines` passed 3 runs in a row. Its first attempt failed 59 cases because the PyPI
`torchvision` pulled in by ALCHEMI doesn't work with CPU-only torch, so that job
installs both from the PyTorch CPU index.

### User documentation pass (2026-11-06)

`README.md` and `docs/source` now cover merged capabilities through Increment 10,
with new pages `user_guide/batching.rst`, `user_guide/acquisition.rst`, and
`user_guide/excited_states.rst`. User docs describe behavior and settings for people
running ALF, so they don't mention increments or campaigns.

**Held back until Increment 11A passes real qualification (user decision):**
`topology_check`, `max_force_cutoff`, `screen_audit_dir`, `friction_per_fs`,
`finite_bootstrap`, `rdkit_molecule_builder_task`, `group_by_selected_state`, and
`examples/keto_acac`. These still appear on the auto-generated API pages. The
`excited_states.rst` page has no worked example until `keto_acac` can be documented.

**Hold-back closed the same day.** After 11A passed, the guide gained these features,
DCM C-PCM labeling and `vertical_*` properties, the one-GPU-per-worker requirement for
GPU4PySCF, and pages for `examples/AcAc_excited_pyseqm` (renamed from `keto_acac`) and
`examples/CPP_excited_gpu4pyscf`. By user decision, the CPP page documents the whole
example and says nothing about E6, whose status stays in
[`excited_states.md`](excited_states.md).

`sphinx-docs.yml` now also builds for `development`, while deployment stays limited
to `main`. The public site therefore keeps describing `main`. No merge to `main` is
planned; see [Order of work](future.md#order-of-work).

The API pages were then repaired. `conf.py` enables `sphinx.ext.napoleon`, which
renders the code's Google-style `Args:` sections, and uses real Parsl during the
build with `python_app` replaced by a shim that keeps the plain function. Mocking
`parsl` had made autodoc skip every `*_task` entry point. The build has no warnings,
and the docs job now runs with `-W`, so a new warning fails it.

### Recent test review (2026-11-06)

A review of the tests added since `b995c15` found them passing but with gaps:

- Tests still asserted that GPU4PySCF labeling works on `cuda:1`, which the backend now
  rejects. They now use device 0, and a new test runs real device selection through
  labeling.
- `test_examples.py` claimed every tracked example but left out the AcAc and CPP
  examples. It now loads their configs. Portability checks stay in `alf_validation`.
- New tests cover the gas vertical spectrum, vertical-optics and PCM NAC failures, the
  vertical scale check, late-stage convergence metadata, the gas NAC solver, and custom
  dielectrics. Each was shown to fail when its library branch is broken.
- Eight tests were renamed to the naming rule above. `alf_validation` selectors that name
  the old tests need updating.

Local result, with the library fixes below: 580 passed, 2 skipped (`seqm` not installed).
Actions results at `5449542`: `engines` 582 passed; `hippynn` 517 passed, 65 skipped;
`base` 451 passed, 100 skipped.

The review also led to two GPU4PySCF library fixes; see
[Increment 11B](excited_states.md#increment-11b--completed-cpp9-active-learning-qualification-e6).

### Quick wins (2026-11-06)

Eleven small items from [`future.md`](future.md), each with its own commit and tests.
Each new test was shown to fail when its branch is broken, except one: the guard against
starting more training after the label budget only matters in a narrow timing window, and
no stable test reaches it.

| Item | Commit |
| --- | --- |
| ORCA regexes as raw strings | `98e5c7f` |
| Prune cohort candidates as frames arrive | `33f50d8` |
| `lifetime_failed_QM_structures` and `lifetime_failed_sampler_structures` in status | `964d0f5` |
| Bounded cohort history | `08131b4` |
| Reject sampler `full_only` batches that can never fill | `dfa6bb5` |
| Keep settings fixed at startup on reload | `2010875` |
| UDD and wells on the selected excited state | `d37e2dd` |
| Ground-state training keeps the scheduler's GPU mask | `1530fc5` |
| Reactive sampler device selection | `e02ac9d` |
| `selection.selector`, with `fps` diversity selection | `678cc43` |
| `maximum_stored_labels` stop rule | `c01076a` |

Local result: 635 passed, 2 skipped (`seqm` not installed). The real GPU runs for three of them
have since passed; see [the hardware record](#quick-win-hardware-qualification--passed-2026-11-06). Two behavior changes could affect
`alf_validation` on Darwin: editing a fixed setting mid-run now has no effect, and
`from_multiprocessing` training no longer sets `CUDA_VISIBLE_DEVICES`.

### Gap-seeking (2026-11-06)

Ported from the fork's ALCHEMI sampler to ASE, TorchSim, and ALCHEMI, in
`tools/gap_seeking.py`, `PackedMLMDEngine._apply_gap_seeking`, and `_StateMeanCalculator`. The
settings block, trigger, surface, and analytic force match the fork. Real mechanics
checks have passed; see the [hardware record](#gap-seeking-and-sampling-window-hardware-qualification--passed-2026-11-06).

Differences from the fork:

- It runs on all three engines, not only ALCHEMI.
- `gap_seeking` with `udd_bias_weight` is rejected at startup, because UDD's bias comes
  from the selected state's spread. Wells add on top of the surface, as in the fork.
- Pairs come from the ensemble's states, so no `dE#` properties are needed.
- The fork's `lcm_bias_Fmax` and `lcm_correction_Fmax` diagnostics and its
  `gap_seeking_switch_events` list aren't ported. The switch is recorded in
  `gap_seeking_*` metadata.

### Sampling windows (2026-11-06)

`sampling_windows` lets one run start on base sampling and move to UDD or gap-seeking at
stored-label counts, in `tools/sampling_windows.py` (user decision). The fork does this with
a staged runner of about 1,400 lines, where each stage is a separate run with a fresh
ensemble. That runner stays in [Group E](future.md#group-e-long-runs). Compared with it,
windows have no fresh ensembles, exact dataset checkpoints, or task journal, and a window can
only change `gap_seeking` and `MLMD_calculator_options`. A real base-to-gap-seeking
loop has passed; see the [hardware record](#gap-seeking-and-sampling-window-hardware-qualification--passed-2026-11-06).

### Multi-node training (2026-11-06)

Two opt-in tasks train one ensemble member per GPU on any number of nodes, for example 8 members on
2 nodes or 40 on 10: `train_HIPPYNN_ensemble_multinode_task` and
`train_excited_state_HIPPYNN_ensemble_multinode_task`, with helpers in `tools/ensemble_training.py`.
Single-node training stays the default, and the tasks need no change to `__main__.py`; the exit fixes
below are what changed it (user decision on retries: failed members run again, up to `member_retries`).

**How it works.** A `join_app` runs a control loop on the coordinator. It submits one app per member,
waits for them, and returns the usual `(completed, training_id)`, so the coordinator sees one task.
Members pick their GPU from their worker rank, because `model_id % width` would put members 0 and 4
on one GPU. Each trains in a fresh `spawn` child, using `ProcessPoolExecutor`, because
`multiprocessing.Pool` hangs if its child dies. Failed directories are renamed out of the `model-*`
pattern rather than deleted, and each attempt leaves a placement record beside them, since trainers
refuse a nonempty member directory.

**Local results.** 827 passed, 2 skipped (`seqm` not installed), including the exit fixes below, and the new tests also pass without Torch.
Actions results at `6e19e7a`: `engines` 829 passed; `hippynn` 728 passed, 101 skipped; `base` 625 passed,
169 skipped.
Each new branch was shown to fail its tests when broken.

Real Parsl workers on one machine (`HighThroughputExecutor`, two local blocks with two workers each,
four CPU excited-state members) also passed. A plain run trained all four across both pools. When one
pool was killed mid-training, Parsl reported `ManagerLost` for its two members. Only those were
retried, a replacement block started, the failed directories stayed, and all four members reloaded.

**Not shown yet.** Real nodes, real GPUs, the ground-state trainer through this path (it has no CPU
mode, so local tests fake it), and any scale. See the
[Darwin gate](future.md#group-c-placement-and-memory).

**Exit fixes found in review.** An older bug: when `--test_ml` finished, the node stayed held (user
report, also on `main`, which is unchanged). ALF called Parsl's `cleanup` only at a
`maximum_stored_labels` stop, and Parsl's own exit hook only logs a warning, so the interchange outlived
ALF and kept answering worker heartbeats. Locally, the interchange and worker pool were still alive 46 s
after `--test_ml`, past the 10 s heartbeat limit. `release_allocations_at_exit` now cleans up on
`exit()`, errors, Ctrl-C, and `SIGTERM`, but not `SIGKILL`. This changes `__main__.py`, because it owns
the Parsl lifetime. Nothing is left alive locally one second after `--test_ml`.

The review also fixed three problems in the first version of this work and one older bug:

- A crash during training kept the coordinator alive until training ended, because Python waits for
  Parsl's internal threads before exit hooks. The control loop now runs on a daemon thread. A crash test
  went from 123 s to 10 s.
- The excited-state loader read every member's whole `training_summary.json` on each sampler task,
  about 200 MB per file at 1M labels (older bug). It now reads a small `member_composition.json`, and
  falls back to the summary for existing models. The summary itself is unchanged (user decision).
- The guide wrongly said that Parsl releases idle blocks as members finish.
- The capacity check wrongly refused extra workers when `available_accelerators` gave each its own GPU.

**Dead executors.** Parsl permanently stops an executor once every block started so far has failed, and
ALF didn't notice. Locally, with a block whose `worker_init` failed, Parsl marked the ML executor dead
within seconds, and ALF was still looping at 100 s. `stop_if_executor_failed` now stops ALF with Parsl's
reason, after saving status, for every required executor (user decision: stop rather than keep sampling).
Labels finished since the last store are lost, as at a budget stop. The guide also recommends
`windowed_error_handler`. Locally, it survived 1 and 2 failed blocks and ran the task, and gave up after 3,
while the default gave up after 1.

### Gap-seeking and sampling-window hardware qualification — passed (2026-11-06)

The separate external harness is frozen at ALF `1055d9c` / `alf_validation`
`f64a70c` (assembled `2bcfafe`). Evidence is outside Git under
`validation/gap_windows_20261106/run-001/`, including `result.json`,
`evidence_summary.json`, per-engine results, loop events and training manifests.
It reused isolated copies of the accepted Increment 9 formaldehyde ensemble
and 32 labels, with the qualified
environments unchanged. Earlier E6 and quick-win passes retain their source boundaries.

By user choice this is a **mechanics qualification**, with a frozen **20 eV**
switch threshold, `sigma=3.5` and `alpha_eV=0.05`; it makes no near-crossing accuracy
claim. **All 15 GPU cases passed:** gap-seeking with/without wells at widths
one/four on ASE, TorchSim CUDA and ALCHEMI, plus a width-four no-switch control
on each engine. All 48 positive trajectories switched and propagated afterward;
all 12 control trajectories stayed on the selected state. Actual entered widths
matched submitted widths. Common-frame checks used an independent LCM derivative,
including both adjacent S1 pairs, and verified unchanged unbiased predictions,
uncertainty and physical-force screening. At the frozen `sigma=3.5`, maximum force
disagreement was **2.39e-7 eV/Å**. Large artificial forces at `sigma=1000` also
passed mixed `rtol=atol=1e-5` checks without contaminating the physical screen.
Trained-model surface-force finite-difference errors were at most
**1.67e-6 eV/Å** in preparation.

The ordinary ALCHEMI ALF loop **passed**, starting at 32 stored labels, switching
from base sampling to gap-seeking at exactly **40**, and exiting normally at **48**
through the production label-budget rule. Four width-four tasks executed in each
window. Independent FPS reconstruction matched all eight cohorts' selections.
All **16** selected, genuinely propagated IDs (eight per window) converged and
were stored in **four native width-four solves**. Every final member retained
all 16; the ensemble's actual training splits included all eight from each window.
All four members performed five real epochs and changed parameters in both
training rounds. Both ensembles promoted; model 2 reloaded with finite four-state
predictions and consistent member-derived gaps/population spreads. Maximum final
model-force finite-difference error was **2.11e-7 eV/Å**.

There were **zero failed tasks/structures, screening rejections, label overshoot,
or converged unstored labels**. The builder returned 36 geometries; 32 entered
dynamics, and one completed four-frame builder task remained unused at shutdown.
No sampler or QM work remained queued. All **ten owned Slurm jobs** were verified
terminated, with an independent empty-queue check; persistent supervision ended.
Source, inputs and passed-stage evidence fingerprints were reverified unchanged.

CPU checks passed **662 library tests, 78 unavailable-engine skips**, and
**96 affected external tests**. Persistent Darwin tmux supervision started the
four-hour queue-inclusive window at **19:25:44 Mountain** (deadline **23:25:44**);
the complete round, including queues and verified cleanup, took **20.05 minutes**.
Allocations were capped at 30 minutes. No library fix or dependency upgrade was
needed. Physical near-crossing accuracy, long-time behavior and full staged
campaigns remain unqualified.

### Quick-win hardware qualification — passed (2026-11-06)

All planned gates passed. Evidence is outside Git under
`validation/quickwins_20261106/run-006/`: `result.json`, `evidence_summary.json`,
frozen manifests, worker observations and closed datasets/checkpoints. The final
snapshot combines ALF `b7f38b5` and external `alf_validation` `e9d88f1` as
`3653590`. Its library is identical to `81a70cd`; the intervening ALF commits
only record qualification. Accepted Increment 9 formaldehyde models and 32
closed bootstrap labels were copied and verified. Accepted E6 inputs, results
and source boundaries were unchanged; no CPP9 campaign was repeated.

The QM and 24-case sampling passes were fingerprint-verified and reused from
`run-004/`, frozen at ALF `158bf71` / harness `569ab28` (`d8ee14c`). Ground-state
and ordinary-loop gates passed in `run-006/`. No library/API change or dependency
upgrade was needed. User guides were unchanged.

| Gate | Observed result |
| --- | --- |
| Selected-state propagation | All **24 cases** passed: ASE, TorchSim CUDA and ALCHEMI; widths one/four; baseline, UDD, well and combined. Each case propagated four molecules, preserved identities and recorded actual task widths. Common-frame physical/bias forces matched legacy formulas, including active 3 Å and 10 Å translated wells; maximum force discrepancy was **4.10e-8 eV/Å**. Physical screening and uncertainty remained unbiased. |
| Ground-state device selection | Both four-member, five-epoch cases passed: reordered four-GPU UUID visibility, then two visible GPUs with `gpus_per_node=4`. Inherited masks stayed unchanged; actual tensor devices and physical UUIDs covered four/two GPUs respectively. Every member changed parameters and reloaded. |
| Reactive execution | Real NEB and dimer tasks passed under reordered/restricted visibility, with **24/36 member-calculator evaluations** and correct physical GPU selection. No transition-state accuracy claim is made. |
| GPU4PySCF | Unbound four-GPU worker rejected before solving. Bound gas/DCM full and energy/force-only evaluations passed native cuTENSOR, convergence, finiteness, requested-response metadata and four production HDF5 stores. Full/reduced energy/force discrepancies were at most **2.73e-12 eV / 2.39e-7 eV/Å**. Formaldehyde uses zero offset. |
| Ordinary FPS/budget loop | Four actual width-four sampling cohorts supplied **eight new propagated labels**; all eight converged in **two native width-four solves**, were stored alongside the 32 bootstrap frames, and were retained by every promoted member. All eight occurred across actual training splits. Every member trained five epochs, changed parameters and reloaded with finite predictions and consistent gap diagnostics. Representative force finite-difference error: **2.79e-7 eV/Å**. |

All four FPS selections were independently reproduced from their recorded
16-frame candidate pools. The selected frames had propagated **5–16.25 fs**.
All 16 trajectories later reached a topology bond-window rejection; earlier
valid candidates survived under the configured continue policy. This validates
screening and workflow execution, not potential accuracy or long-time stability.

The production `maximum_stored_labels=40` rule stopped new admissions, allowed
training to finish and promote model 1, and exited normally without the old
harness admission-stop override. Counts: **8 attempts, 8 completed/converged,
40 total stored, zero overshoot, zero converged unstored labels, zero cancelled
QM calculations**. Four already-built geometries remained unused (one builder
result in the coordinator queue). Stored IDs, retention and split evidence are
in the final result and training-input manifests.

The available full CPU suite passed **591 tests, 46 skipped**: 22 TorchSim and
24 ALCHEMI cases unavailable in that CPU environment. The affected external
suite passed **88 tests** (delivery tests at harness `ad5addd`). Existing pruning/history/failure/exhaustion/reload/FPS
and shutdown checks passed; stale renamed test selectors were not found.
An additional event-controlled coordinator regression holds active training at
the budget for two polls, verifies that no apps are admitted, then releases the
result and requires promotion and normal cleanup. Task sleeps do not determine
completion order. Existing fast/slow and bootstrap-stop tests also passed, and
both real loops demonstrated normal budget shutdown. The controlled test covers
observable admission/exit behavior; no independent mutation-sensitive coverage
of the training-admission guard is claimed, because immediate exit after the
last promotion makes that branch redundant in this scenario.

Earlier attempts remain explicitly incomplete in their own snapshots. Harness
corrections addressed strict login-shell startup, single-driver batching blocks,
CuPy-free Torch UUID enumeration/normalization, observation imports before a
fork, and a builder signature incompatible with argument resolution. The first
ordinary loop trained/promoted successfully but executed only width-two QM
calls. Before the final loop, its partial-batch wait was frozen at **60 seconds**
instead of the inherited one second so two FPS cohorts fill width four; no
scientific setting or acquisition cutoff changed. Reusable PCM/calibration
launchers also bind one scheduler-visible UUID before CUDA imports.

Persistent frontend tmux supervision preserved the original queue-inclusive
window **18:12:19–22:12:19 Mountain on November 6**, including retries. The round
finished after **47.2 minutes**, with 30-minute allocation caps and one allocation
per ordinary-loop executor. All **nine jobs** referenced by accepted evidence
terminated; termination of **all 25 unique owned jobs** across attempts was also
verified. The final independent queue check was empty. LCM, large campaigns,
long-time stability, multi-node training, teacher labeling, NeuroChem and MI350
remain outside this round.

## Development and test environments

Policy, decided 2026-11-03: test development work against fresh installs with
up-to-date dependencies, not long-lived personal environments.
User clarification: this is the next acceptance priority after the running
Increment 7 retries, before new feature implementation. Compatibility with
current upstream releases is the target; `atomistic` and the other existing
environments supply historical comparison evidence, not the target dependency
versions. Keep the running retries frozen while completing them.

### Current state (audited 2026-11-03)

At that audit, no regression had run against a clean install of the development
checkout. The November 4 qualification below supersedes that environment status.

- **`atomistic`** (QM, training, CPU suites). Its editable `alf` points to
  `~/github/ALF`; tests import the development checkout only because they run
  from its root.
  - HIPPYNN is an editable `~/github/hippynn` at upstream `a66eba6` (February
    2026), with uncommitted local edits to `ase_interface/calculator.py`
    (energy `offset`) and `experiment/lightning_trainer.py`. ALF development
    does not use either edit, but all results so far ran against a modified
    HIPPYNN.
  - Parsl is 2025.10.27.
  - A stale `site-packages/seqm/` directory shadows the editable PySEQM, so
    commands need `PYTHONPATH=~/github/PYSEQM`.
- **TorchSim CUDA/ROCm.** These are venvs layered on `atomistic` (they include
  its system site-packages). They live in the fork's
  `debug/portable_gpu_20261027_8795a40/envs/` and inherit all of the above.
- **`alf_alchemi_02`** (Python 3.11): an editable development checkout, torch
  2.13, and `nvalchemi-toolkit==0.2.0`.
- **`alf_env_gpu4pyscf18`** (intended for E2): its editable `alf` is the fork
  checkout, and it carries ALCHEMI 0.1.0.
- **`setup.py`** has no lower bounds and `python_requires=">3.9"`, although the
  engines need Python 3.11/3.12. The versions used are recorded only in
  campaign manifests outside Git.
- **Location risk.** All of these environments live under `/vast/home`, where a
  recurring cleanup job has deleted every `core.py` file and broken them before.

As of the audit, PyPI offers HIPPYNN 0.1.3, Parsl 2026.11.2, and
`torch-sim-atomistic` 0.6.2.

### Fresh-environment recipe

1. **Build dated, self-contained environments from scratch** with current
   releases:
   - `alf-dev-<date>`: base, QM, and training;
   - TorchSim CUDA and TorchSim ROCm;
   - ALCHEMI;
   - GPU4PySCF, once E2 starts.

   No environment layers on a personal one.
   - Each installs the development checkout editable with its extras, for
     example `pip install -e ".[tests,full,torchsim]"`, after installing the
     torch build for its hardware (CPU, CUDA, or ROCm).
   - Resolve the newest stable releases available when the build starts,
     including HIPPYNN, Parsl, ASE, NumPy, h5py, PyTorch, TorchSim, and ALCHEMI.
     Select a supported Python and GPU runtime combination for each engine;
     record resolver downgrades rather than calling them the latest versions.
     Prereleases and arbitrary upstream development heads are separate tests.
   - ALCHEMI 0.2.0 remains the existing qualified baseline. Qualify the newest
     stable toolkit independently before changing the existing adapter pin;
     its old acceptance does not establish current-release compatibility.
     Any temporary pin needs a demonstrated failure, reason, and follow-up.
2. **Use upstream dependencies without local edits.** HIPPYNN is the PyPI
   release, or a recorded upstream commit when development needs unreleased
   work. PySEQM is a recorded upstream commit, with no shadowing directories.
3. **Freeze every build.** Write `pip freeze`, and `conda list --explicit` for
   conda environments, to an artifact outside Git. Record those paths with
   acceptance evidence, as for frozen source snapshots. Only the recipe belongs
   in the repository (for example, a short `docs/` page or missing extras).
4. **Refresh at fixed points.** Rebuild at the start of each increment and
   before an acceptance-gated push, never during a campaign. A running campaign
   keeps its frozen environment and frozen source.
5. **Gate each refresh.** The full CPU suites must pass in the fresh
   environments, with A100 and MI250 GPU smoke checks for the engine
   environments. If a newer dependency breaks ALF, fix it in a focused change
   or record an upper bound with its reason. Do not quietly fall back to an old
   environment.
   Package imports and mocked checks alone do not establish backend acceptance;
   the first refresh also runs the real workflow checks below. MI350 remains
   deferred.
6. **Add CI.** Running `unit-tests.yml` on `development` gives an automatic
   fresh-install CPU check on every push.
7. **Choose a location** outside the `core.py` cleanup's reach where possible
   (a `/projects` or scratch path); record it here once chosen.

Until the first fresh build passes, existing acceptance records keep their
recorded environments. Agents must state which environment produced each
result, and must not modify `atomistic`, `alf_env`, or the user's HIPPYNN and
PySEQM checkouts.

### Qualification progress (2026-11-04)

The user authorized the remaining Increment 7 work and selected
`/vast/home/pjlohr/ALF_LANL/environments/20261104/` for independent environments.
All five builds exclude personal/system site-packages: base CPU, NVIDIA
training, TorchSim CUDA, TorchSim ROCm, and ALCHEMI. Current-release resolution
and package checks are recorded outside Git under
`/vast/home/pjlohr/ALF_LANL/validation/inc7_closeout_20261104/`.
`pypi-audit.json`, `frozen-*.{txt,json}`, and `cpu-summary.json` retain versions,
dependency consistency checks, exact resolutions, skip reasons, and XML hashes.

The builds use Python 3.12.8, HIPPYNN 0.1.3, Parsl 2026.11.2, ASE 3.29.0,
NumPy 2.5.4, h5py 3.16.0, PyTorch 2.15.0 (CPU/CUDA 13.2/ROCm 7.14 variants),
TorchSim 0.6.2, and ALCHEMI 0.3.0 in their applicable environments. The clean
PySEQM checkout is upstream `52850b4`, installed without editing the user's
checkout; all eight real CPU interface checks passed. All environments passed
their dependency consistency checks. No older dependency was silently substituted.

| Fresh environment | Full CPU passed | Full CPU skipped |
| --- | ---: | ---: |
| Base CPU | 284 | 42 |
| TorchSim CUDA, CPU execution | 302 | 24 |
| TorchSim ROCm, CPU execution | 302 | 24 |
| ALCHEMI 0.3, CPU execution | 304 | 22 |

Those suites used new published-release CPU checkpoints. The old local HIPPYNN
checkpoints use newer development enum values (`SysAtom`) absent from PyPI
0.1.3 and do not load there; no cross-version conversion is claimed. A new
self-contained test trains two small members for two actual CPU epochs, checks
weight updates, reloads training databases, and evaluates through ALF. These
are test fixtures, separate from four-member campaign training acceptance.
Darwin's default GCC 8 cannot compile the C++20 code generated by current
PyTorch; selecting GCC 11.2 explicitly fixes that runtime prerequisite.

Local `9d88087` adds current config/task checks for all ten tracked master
configurations, the real CPU training/reload check, and base/HIPPYNN CI lanes
for `main` and `development`. NeuroChem imports remain explicitly unavailable
in the fresh stack. CI is prepared locally; the user still handles pushes.
`a01dab1` admits ALCHEMI 0.3 alongside 0.2 and updates the install extra; its
full CPU and focused conformance checks passed. NVIDIA toolkit acceptance is
still pending. `3a970c0` adds an independent installation recipe and explicit
compiler/CUDA site settings; final resource/version checks passed (31 tests,
two unavailable-engine skips).

The bounded runtime probe (`17376466`) passed tensor/autograd execution on an
A100 with all three fresh NVIDIA environments, and its termination was verified.
That probe is not engine or active-learning acceptance. The persistent supervisor
`alf-inc7-fresh-20261104` on `darwin-fe2` waits for the trajectory-failure case,
then runs NVIDIA engine checks, the twelve-case fresh water matrix, native
PySEQM, AMD engine checks, and the four MI250 cases sequentially. It uses frozen
`source-fresh/` at `3a970c0`, explicit queue-inclusive deadlines, owned-job
journals, and cleanup checks. `qualification-result.json` records actual gates;
preparation or a runtime probe cannot mark those campaigns passed.

Original-example compatibility is implemented locally at `c720d8c`, with 52
focused harness checks passing. Its adapters call the original example task
functions, retaining methods, units, architecture, and training/stopping settings.
Bounded smoke copies use four members, 16 bootstrap structures, eight newly
sampled training inputs, and 55-minute case budgets; the batching matrix keeps
its separate 32-structure bootstrap requirement. The examples' wB97X/def2-TZVP
labels need their own trained seed: reusing the PBE/def2-SVP campaign labels is
explicitly rejected. Seeded learning copies labels only and must train its own
initial ensemble; fine-tuning uses the example-local task and a compatible seed.
Missing backends are unverified, not passed. The separate persistent supervisor
`alf-inc7-examples-20261104` waits for verified current-release campaign completion
before allocating resources, using frozen `source-compatibility/` at `c720d8c`.
Results will be in `fresh-examples-run-001/` and `example-qualification-result.json`.
No real example-family gate is inferred from the passing config checks.

### First current-release compatibility gate

After recording the running retries and their cleanup, complete this gate
before starting E1. Preserve the existing feature priority order below it.

| Step | Required evidence |
| --- | --- |
| Resolve and install | Dated, independent base/training, TorchSim CUDA, TorchSim ROCm, and ALCHEMI builds. Install ALF from the recorded development revision. Record latest available versus installed versions, Python/runtime support, import paths, upstream revisions for source-only dependencies such as PySEQM, and every constraint. No personal site-packages, local dependency edits, or shadowing source directories. Run `pip check`. |
| CPU correctness | Full development tests in the fresh environments, including the real HIPPYNN small-molecule reader and larger-system regressions, training restarters in a fresh process, checkpoint inference, batching/cohorts, and failure handling. Report skips by reason and keep missing required backends incomplete. |
| NVIDIA workflow | Repeat the 12-case water matrix on the fresh compatible stacks, using the established limits, widths, numerical tolerances, and real acquisition/label/storage/retraining criteria. Demonstrate a four-member small-molecule bootstrap and both single/native PySEQM labeling cases. Reload training restarters and inference members at the current source revision, resolving the retry snapshot's old database-class path without restoring a separate database file. |
| AMD execution | Run the four TorchSim MI250 cases with the fresh ROCm sampler stack and newly qualified QM/training stack. Hardware-specific PyTorch wheels may differ; record their versions and runtime requirements explicitly. MI350 is outside this gate. |
| Reproducibility and maintenance | Store exact dependency resolutions, install commands, package-source identities, source fingerprints, and results outside Git. Track the install recipe and focused compatibility fixes in Git. Add fresh-install CPU CI for pushes and PRs targeting `development`, retaining `main` coverage and separating base-only optional-dependency isolation from HIPPYNN training coverage. |

Classify failures as dependency installation/resolution, ALF API compatibility,
numerical behavior, or site/runtime availability. An unavailable backend or
temporary upper bound remains an explicit gap in the latest-release claim;
success in an older environment cannot close it. Preserve the scientific
defaults and approved tolerances when repairing compatibility. Broad dependency
support is the goal; any minimum/maximum versions declared in packaging must
be supported by the observed tests rather than invented in advance.

GPU4PySCF receives the same clean current-release qualification when E2 begins;
it does not inherit acceptance from `alf_env_gpu4pyscf18`. New model/backend
dependencies receive this check when introduced. Later increments refresh
their affected stacks without upgrading an environment used by an active run.

## Increment status

Status values: **planned**, **in progress**, **done locally** (committed on
local `development`, not yet pushed), and **merged** (in official
`lanl/ALF:development`).

When an increment finishes, move its row to [`batching.md`](batching.md#delivered-increments) or
[`excited_states.md`](excited_states.md#delivered-increments), which hold the delivered ones.

The completed Increment 11B / E6 row is now in
[delivered excited-state increments](excited_states.md#delivered-increments). Remaining
work follows [Order of work](future.md#order-of-work).

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
with validation records and update this roadmap as capabilities land.
