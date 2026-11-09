# ALF development roadmap

Last updated: November 8, 2026.

This folder is the shared plan for bringing the fork's intended capabilities into
`lanl/ALF:development`.

| File | Covers | Record here |
| --- | --- | --- |
| This file | Status, standing rules, environments, acceptance rules | Status changes, test and docs records, environment policy |
| [`batching.md`](batching.md) | Increments 0-7 and their known follow-ups | Batching and ground-state results and follow-ups |
| [`excited_states.md`](excited_states.md) | The E1-E6 plan and Increments 8-11B | Excited-state results, including the E6 pilot |
| [`future.md`](future.md) | What's left: order of work, quick wins, and grouped plans | New plans and priority decisions |
| [`exascale.md`](exascale.md) | File I/O risks at scale, fix priorities, and node-local storage | I/O audit findings, Darwin profile results, and changes to the I/O priorities |
| [`qualification_round.md`](qualification_round.md) | The planned Darwin round for teacher labeling, fine-tuning, and restarts | Nothing; its results go in this file and `exascale.md` |

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
- **Multi-node ensemble training passed bounded Darwin qualification:** both trainers
  at eight members/two nodes, cancellation/retry recovery, and excited-state training
  at 20 members/five nodes. Exit cleanup, dead-executor handling and all three engines'
  sampling/plotting checks also passed; see the
  [November 7 hardware record](#current-development-hardware-qualification--passed-with-restart-limits-2026-11-07).
  The later bounded restart matrix also passed; see the November 8 record below.
- **Foundation-model teacher labeling is built,** with real MACE and real UMA (uma-s-1p2p1) checked
  locally. Teachers are off-the-shelf checkpoints only, and UMA's `omol` refuses periodic structures
  unless opted in. The bounded Darwin gate passed for both teachers and their students; see the
  [November 8 results](#darwin-qualification-results-2026-11-08) and the
  [periodic guard](#uma-teacher-off-the-shelf-only-and-a-periodic-guard-2026-11-07).
- **Restarts keep their labels:** atomic status and shard writes, the shard check in single-node
  training, and the replay manifest cache, then the coordinator lock, a holding area for finished
  labels, and restart reconciliation. See the [first](#restart-fixes-without-darwin-2026-11-07) and
  [second](#restarts-that-keep-labels-2026-11-07) records. All 18 bounded Darwin restart cases passed,
  including two interruptions during an open HDF5 store.
- **Fine-tuning with ALF is built for MACE, HIPPYNN, and UMA,** through one interface: ALF selects the
  samples, and each round fine-tunes copies of the pretrained model. GPU workflow checks passed for
  all three families; HIPPYNN and MACE also passed the DFT accuracy-improvement gate, while UMA did
  not. Two-node MACE/UMA fine-tuning and selective cancellation recovery passed. See the
  [November 8 results](#darwin-qualification-results-2026-11-08) and the earlier
  [MACE](#fine-tuning-with-alf-mace-2026-11-07), [HIPPYNN](#fine-tuning-with-alf-hippynn-2026-11-07), and
  [UMA](#fine-tuning-with-alf-uma-2026-11-07) records.
- **Remaining qualification:** investigate UMA's B4 accuracy regression. B7's tight periodic
  numerical checks and CPP accuracy training remain deferred; the historical HIP-HOP145 timing
  reproduction gate was not met. CPP HIP-NN64 capacity/timing and batch-64 checks passed.
  See the [results and limits](#darwin-qualification-results-2026-11-08). Then new
  uncertainty and acquisition methods (see
  [Order of work](future.md#order-of-work)). Long runs and staged campaigns come last, and may be
  removed; see [Order of work](future.md#order-of-work).
- **No merge to `main` is planned** (user decision, November 6).
- **Outside current acceptance:** legacy NeuroChem and MI350.

## Darwin qualification results (2026-11-08)

Results of the [November 7 plan](qualification_round.md), checked against the completed
supervisor and acceptance reports. This is a bounded workflow qualification. It does not establish
large-campaign accuracy, exascale recovery, or concurrent scaling of a single model.

| Section | Outcome | Evidence and practical limit |
| --- | --- | --- |
| B1: GPU4PySCF | Passed | Eight ground-state fixtures, single/sequential labeling, gas/DCM full and reduced requests, forced failure handling, and 64 separate DFT test frames. The opt-in S0 grid response was enabled; existing CPP defaults were preserved. |
| B2: restart acceptance | Passed, 18/18 | Every interruption row ran twice, including the real managed-child seed timeout and both open-store SIGKILL cases. Identical-command recovery preserved labels, trained/reloaded four members and reached promotion without manual status/shard repair. |
| B3: teacher labeling | Passed, MACE and UMA | Native GPU widths 1/8/32/128 and dtype/parity checks; two ordinary 48-label student loops, each with 16 propagated samples and two promotions. The held-out DFT set stayed outside training. |
| B4: DFT fine-tuning | Partial | HIPPYNN and MACE passed. UMA completed the workflow and both promotions, but failed the held-out accuracy-improvement gate. |
| B5: two-node fine-tuning | Passed, 4/4 | MACE and UMA, clean and cancelled-block cases. Four copies ran on four distinct host/GPU pairs across two nodes. Only the cancelled block's two copies retried; every lineage completed and each round promoted once. |
| B6: local-storage/settings | Measurements complete for three partitions | `general`, `ml4chem`, and `shared-redstone` were measured; post-allocation file survival and every other partition remain unqualified. |
| B7: periodic PySCF GPU | Incomplete/deferred | Molecular and k-point checks passed, and the temporary SCF checkpoint issue was fixed. Tight periodic energy/force/stress gates and the complete periodic student loop did not pass. |
| B8: CPP sizing | Capacity/timing follow-up passed | Four-member, one-epoch 50,000-row HIP-NN64 capacity runs passed at batches 64 and 128. The old HIP-HOP145 timing baseline did not reproduce within 15%; long accuracy training remains deferred. |

**B1 numerical checks.** CPU/GPU ground-state energy difference was `5.00e-7 eV` and the
representative force finite-difference error was `3.97e-6 eV/Å`. Gas/DCM full/reduced common
outputs agreed within the requested `1e-5 eV` / `1e-4 eV/Å` limits. Native GPU4PySCF 1.9.1,
PySCF 2.15.0 and cuTENSOR 2.8.1 were used, with explicit worker binding and preserved masks.

**B2 scope.** The final store cases each finished with 32 converged labels stored exactly once,
no held labels, model generation 2, and four members reloaded after five actual epochs.
Worker allocations could remain alive during the 120-second post-SIGKILL observation; owned
resources were explicitly cleaned before identical-command restart. This is not a claim that
SIGKILL automatically releases allocations. The harness's storage-return limit was made durable
across worker replacements; earlier incomplete attempts do not count as passes.

**B3 boundary.** Final CPU acceptance used the qualified GCC 11 compiler after an initial
frontend compiler mismatch. Frozen models, data and worker events were fingerprint-verified
unchanged. UMA's effective parameter dtype was recorded; requesting float64 does not imply
all upstream UMA parameters became float64. Neither teacher is claimed to improve on DFT here.

**B4 measured errors.** Each family started from its original foundation on every round and
performed initial training plus two promotions with four copies. All QM labels converged;
stored counts were 52/48/60 for HIPPYNN/MACE/UMA, including 20/16/28 propagated sampled IDs.
Mean per-copy errors on 64 held-out DFT frames, before and after the final promotion:

| Family | Force MAE before → after (eV/Å) | Offset-corrected energy MAE before → after (eV) |
| --- | ---: | ---: |
| HIPPYNN | 1.30034 → 1.15717 | 0.181912 → 0.167084 |
| MACE | 0.410482 → 0.100613 | 0.061944 → 0.013361 |
| UMA | 0.102667 → 1.274963 | 0.015293 → 0.178367 |

UMA's B5 scheduling/recovery pass does not close that B4 scientific gate. Its failed B4 recipe
needs investigation and a new qualification before claiming successful accuracy adaptation.

**B5 source and execution.** ALF `adf1aef` records each member's last actual placement outcome
in multi-node fine-tuning lineage. The preceding startup storage-schema fix was `7e4e3a4`.
The final harness was `alf_validation` `5b758bc`, with unchanged scientific settings and the
original two-hour queue-inclusive deadline. Retained MACE passes were fingerprint-verified;
only UMA was rerun after the disk-quota interruption. Final members performed 84 real optimizer
updates each for MACE and 151 for UMA, with changed finite parameters and unchanged child masks.
ALF's worker-rank CUDA assignment avoided Parsl's synthetic HIP/ROCr mask aliases. The observer
identified retries by their current Slurm job. All owned jobs and persistent supervisors ended.

**B7 limits.** Periodic energy discrepancies of approximately `2.6e-5`–`3.1e-5 eV` exceeded
the `1e-5 eV` gate. Representative force and stress finite-difference errors were
`1.464e-3 eV/Å` and `2.583e-4 eV/Å³`, above their gates. These were not waived as passed DFT
convergence checks. The user deferred further numerical investigation. Disabling the inherited
temporary GPU SCF checkpoint (`78af257`) passed a real GPU smoke check; no library API changed.

**B8 measured capacity.** On A100 40 GB, all four HIP-NN64 members used the actual
40,000/5,000/5,000 split of the repeated-row capacity fixture. Batch 64 took 166.9–168.4 s
per epoch, with maximum allocated/reserved CUDA peaks of 3.94/12.47 GiB. Batch 128 took
161.7–163.2 s, with peaks of 7.84/32.45 GiB. The CPP example retains batch 64: the small
timing gain at 128 cost substantially more reserved memory. A 300-epoch projection at batch
64 is about 14 hours per member, not a measured 300-epoch run or a prediction of convergence.
The repeated labels establish capacity, not predictive accuracy.

**Source boundaries and records.** B1 used ALF `f0d2560`; B2 `460a037`; B3 and the B7
checkpoint fix `78af257`; B4 `7e4e3a4`; B5 `adf1aef`. The final historical run identifiers were
`development_20261108/b1-run-004`, `b2-storage-followup-002`, `b3b7-run-005` plus
`b3-acceptance-gcc11-001`, `b4-run-001`, `b5-run-004` (retaining MACE from `b5-run-003`),
and `batchscale-run-001`. Compact reports, fingerprints, environment freezes and the source
revision map are retained under `<validation root>/records/20261108/`. Raw run directories
are retired under the retention policy below; a historical evidence path is provenance, not
a promise that generated datasets, checkpoints or snapshots remain on disk.

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
  Keep generated data outside Git, in the validation evidence directory on Darwin, written
  `<validation root>/` below.
  Never infer a gate from a startup check.
- Record each result with its source revision, environment, and evidence path,
  and update the status table in this file and the relevant section. The table at the
  top of this file says which file each record belongs in.
- After results are documented, user-authorized cleanup may retire bulk validation artifacts.
  Keep compact reports and fingerprints outside Git, and identify any deliberately retained
  reusable fixtures. Historical evidence paths may then refer to retired runs.
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

They now live in the separate `alf_validation` repository.
Its `assemble.sh` combines an ALF revision with the harnesses into one
committed tree, which is the form the harnesses expect. Earlier entries in this
roadmap still cite the old `examples/...` paths; they refer to those historical
locations. Recorded runs keep their own frozen source copies under
`validation/`; completed raw snapshots may subsequently be retired under the retention policy.

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
`<validation root>/darwin_runs_202607/`. Older entries cite
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

**Subsequent Darwin qualification passed November 7.** Both trainers ran eight members
on two real GPU nodes; excited-state training also ran 20 members on five nodes.
See the [hardware record](#current-development-hardware-qualification--passed-with-restart-limits-2026-11-07)
for retry, promotion, sampling and cleanup evidence and scope limits.

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

### Teacher labeling (2026-11-07)

A foundation model can label structures in place of QM, so that a HIPPYNN student learns from it:
`teacher_batch_task` for native batches and `teacher_task` for one structure at a time or
`"mode": "sequential"`, in `qm_interfaces/teacher_interface.py`. Ground-state energies and forces only.
See the [guide](../source/user_guide/qm_interfaces.rst).

- **Batches:** `TorchSimTeacher` runs a whole batch through any TorchSim model in one pass, using the
  first-party TorchSim models that MACE and fairchem ship. `AseTeacher` wraps any ASE calculator.
  A batch that fails is split in half until the failing structure is isolated.
- **Safeguards (from review, beyond the original plan):**
  - ALF stops at startup on unit factors other than 1.0, which catches Hartree factors copied from a
    QM configuration, and on properties a teacher can't give.
  - ALF refuses an `h5_dir` holding QM labels or another teacher's, unless `allow_mixed_labels` is set.
  - Structures with elements the teacher lacks, and charged or open-shell structures for MACE, are
    marked unconverged with a reason.
  - UMA requires an explicit `task`.
  - There's no force limit on labels (user question). Large forces are often real, and the training
    filters already handle outliers.
- **Provenance:** every label records the teacher, its package version, and its checkpoint's SHA-256.
  Storage writes these onto the shard as the `label_provenance` attribute, and both HIPPYNN trainers
  write `label_sources.json` after training. QM shards don't change.
- **A small generic hook:** `LabelingCoordinator` calls a labeler's `validate_config`, if it has one,
  at startup. `__main__.py` doesn't change.

**Real MACE, locally (user approved downloads).** MACE-MP-0 small, `mace-torch` 0.3.17, on CPU in a
throwaway environment:

- Batches of 1, 2, 4, and 8 matched one-at-a-time labels to 4e-15 eV and 3e-15 eV/Å, with molecules and
  a periodic copper cell in the same run.
- Labels matched MACE's own ASE calculator to 3e-14 eV.
- Forces matched a numerical derivative of the energy to 4e-8 eV/Å.
- A full ALF cycle stored real MACE labels with provenance, in both native and sequential modes.

That run found a real bug. Thread-based executors share a worker's teacher, and MACE keeps state between
calls, so concurrent sequential labels failed. A worker now uses its teacher from one thread at a time,
and a regression test covers it. Parsl's usual executor runs one task per process, so it wasn't affected.

**Not shown yet.**

- **GPU, and real student retraining:** both wait for the Darwin gate, since the ground-state trainer
  has no CPU mode.

**Local results.** 936 passed, 2 skipped (`seqm` not installed), and the teacher tests also pass without
torch, HIPPYNN, or TorchSim. All 40 deliberate breaks were caught, after tests were added for the five gaps
the first round found.

**Real UMA, locally (added later on November 7).** The user logged in with the `hf` CLI and accepted UMA's
license, so `facebook/UMA` became downloadable. uma-s-1p2p1 (2.3 GB, SHA-256 `b2673b85…d0d5ce`, FAIR
Chemistry License), fairchem-core 2.25.0, on CPU in a scratch environment, loaded from a local file as on
compute nodes:

- `omol`: batches of 1, 2, 4, and 8 gave identical energies to one-at-a-time labels, and forces within
  4e-6 eV/Å, for eight molecules, OH⁻ (charge −1), and the OH radical (multiplicity 2).
- Labels matched fairchem's own ASE calculator to 1.2e-4 eV and 5e-7 eV/Å, within float32. Forces matched a
  numerical derivative of the energy to 2e-4 eV/Å.
- Charge and spin reach the model: OH⁻, the OH radical, and neutral singlet OH all differ.
- Energies are totals: water −76.4335 Ha, an isolated O atom (triplet) −75.0780 Ha, so water's atomization
  energy is 9.99 eV. A local checkpoint loads without fairchem's separate reference energies, which only
  serve isolated atoms, and this checkpoint labels those itself.
- `omat`: a periodic copper cell, batched and alone, agreed exactly and matched fairchem's calculator to
  2e-6 eV, at −3.74 eV per atom.
- A full ALF cycle through the real coordinator stored real UMA labels with provenance (`task` `omol`, the
  checkpoint's SHA-256), in native and sequential modes, and `label_sources.json` named the teacher.

No code change was needed. The example and the [guide](../source/user_guide/teacher_student.rst) now name
uma-s-1p2p1 and show the `hf auth login` and `hf download` steps.

### Restart fixes without Darwin (2026-11-07)

Items 1-3 of the [`exascale.md`](exascale.md) priority list. None needed the Darwin profile.

- **Atomic writes (`cbadf9e`).**
  - Status is written through `write_status`, which writes a temporary file, fsyncs it and renames it,
    at all six coordinator sites.
  - A shard is written to `<shard>.partial` and renamed when complete. The name doesn't end in `.h5`, so
    every reader ignores it, and the next store overwrites a leftover one.
  - An existing shard is never replaced. A crash between the rename and the status write still stops a
    restart, now with a message saying what to check. Reconciling status with disk is item 6.
  - The shared helpers are in `tools/files.py`, which multi-node training's placement records now use too.
- **Shard check in single-node training (`8d151da`).** Both single-node HIPPYNN tasks take `h5_path`
  and `current_h5_id`, which `build_input_dict` already fills in, and wait up to 120 s for every stored
  shard to appear in the directory listing. `__main__.py` didn't change. Deleting a stored shard now
  stops training, instead of training without it.
- **Replay manifest cache (`6aedf2d`).**
  - Each worker indexes a source once per version: its SHA-256 and one frame count per group. A version is
    identified by path, inode, size and modified time. The manifest is a compact sequence, not one tuple
    per frame, and gives the same entries as before, so frame selection is unchanged.
  - A task opens each HDF5 source once. The RDKit and CPP builders hash their seed and reference once per
    worker per version, hash their manifest once per task, and open the seed twice per task instead of
    twice per molecule.
  - An edit in place that keeps both the size and the modified time wouldn't be noticed. Shards are
    renamed into place, so that can't happen to them.
- **Units page (`f1c9d3e`, user request).** [`units.rst`](../source/user_guide/units.rst) now says what
  each QM task reports, that HIPPYNN sampling expects eV and eV/Å, and what ALF doesn't check.

A test found that a Parsl app's `.func` returns errors as a `RemoteExceptionWrapper` instead of raising
them, so the new tests re-raise them.

**Local results.**
- Full suite: 966 passed.
- With HIPPYNN blocked: 856 passed, 110 skipped. With Torch blocked: 751 passed, 180 skipped.
- 29 deliberate breaks were caught, after one gap got a test: the hash cache had been tested only through
  the manifest cache, which hides it.
- Docs build with no warnings; 80 roadmap links checked, 0 broken.
- Not run on Darwin at this commit. The November 7 profile measured `ce711d8`, before these fixes: 50 of 50
  renamed files appeared on another node with matching contents, and the longest listing delay was
  0.240 s, far inside the 120 s wait. It didn't test a crash during a rename. See the
  [measurements](exascale.md#darwin-measurements-2026-11-07).

### Restarts that keep labels (2026-11-07)

Items 4-6 of the [`exascale.md`](exascale.md) priority list, after the Darwin profile showed `flock`
working across nodes, restarts during a finite bootstrap failing, and labels lost on any crash. To
continue a run that stopped, run the same command; no files need editing. User guide:
[Restarting after a crash](../source/user_guide/configuration.rst).

- **Coordinator lock.** `files.hold_lock` takes `flock` on `<status_path>.lock` before Parsl starts,
  in every mode, since even `--test_qm` rewrites status. A second coordinator stops and names the
  holder. The lock ends with its process, so a killed coordinator leaves nothing to remove.
- **Held labels.** Each loop moves finished QM work out of memory into `held_labels_dir` (default
  `<h5_dir>_held`, refused inside `h5_dir`), one atomically written pickle per loop, in
  `tools/restarts.py`. A store writes them as the next shard, recording the pieces in its
  `held_pieces` attribute, then removes them, then saves status. Labels stay out of `h5_dir` during
  training, so every member still trains on the same shards. Counts keep their old units: the
  bootstrap target counts held and queued successes, and `save_h5_threshold` still counts failed work
  since the last store, as the queue did.
- **Reconciliation on restart** (`reconcile_status`), printed as `Restart:` lines:
  - a shard stored before status was saved is counted, its pieces removed, and passed;
  - training IDs whose directories exist are skipped;
  - molecule IDs skip ahead by `parallel_samplers + maximum_builder_structures`, the most one unsaved
    loop can use;
  - a finite bootstrap rebuilds exactly the IDs it doesn't hold. Replay picks frames from the ID, so
    the bootstrap holds the same structures as a run that never stopped. `ase_calculator_task` moves an
    old scratch directory to `<dir>.previous-N` instead of raising.
- **Budget stop.** Labels finished but not stored are stored in one last shard, marked
  `stored_at_budget_stop`, which no model trained on.
- **Not covered.** A training running at the crash is lost; a later one replaces it. Unfinished
  builder, sampler, and QM work restarts. After `SIGKILL`, allocations stay until the scheduler ends
  them.

**Checked locally** with the real coordinator and stand-in tasks, crashing and restarting with the same
command: during a bootstrap, a finite bootstrap after every ID was handed out, during training, between
a store and saving status, between a store and removing its pieces, after an unsaved loop used new IDs,
and at a budget stop. Each restart stores every finished label exactly once, with `lifetime_stored_labels`
matching the shards.

- Full suite: 982 passed. With HIPPYNN blocked: 872 passed, 110 skipped. With Torch blocked: 767 passed,
  180 skipped. The restart tests passed eight runs in a row.
- All 26 deliberate breaks, plus one for the margin's size, were caught after three test fixes: the ID-reuse
  test crashed before its builders ran, and then within a loop too narrow to need the full margin, and no
  test had failing QM, so counting only converged labels toward `save_h5_threshold` went unnoticed.
- Docs build with no warnings; 96 roadmap links checked, 0 broken.
- Not run on Darwin. Rerunning section 3 of the [restart profile](exascale.md#darwin-profile-for-restart-work)
  at this commit would compare against the `ce711d8` baseline.

### Real QM in tests: dependency checks only, and two CI failures (2026-11-08)

User decision: tests catch changes in dependencies and check ALF's wrappers. Physics accuracy, such as SCF
convergence and agreement with finite differences, is for the user, `--test_qm`, and the qualification rounds.

- **`tests/test_pyscf.py`.**
  - **Removed** the physics comparisons: energies against PySCF directly, molecular, excited, periodic, and
    smeared forces against finite differences, and stress against ASE's numerical stress.
  - **Real PySCF now runs once per path ALF uses,** checking only that it runs and returns finite results of
    the right shape: molecule, open shell, TDA, periodic multigrid, fftdf stress, smearing, storage, a forced
    SCF failure, and the coordinator.
  - **New stand-in tests** cover what only the physics tests did: the conversion to eV and eV/Å, the force
    sign, stress symmetrization, units and Voigt order, and reporting the free energy when smearing. All 5
    deliberate breaks were caught.
- **`tests/test_pyseqm.py`** drops its finite-difference test, and keeps the batched-against-single run.
- **CI failures, both intermittent:**
  - **The open-shell PySCF tests used the OH radical,** whose degenerate π orbitals left its SCF oscillating
    near 1e-10 Eh; on CI it failed to converge in 100 cycles. They now use NH₂.
  - **`test_budget_stop_stores_the_labels_it_holds` relied on sleeps,** and on a slow runner no label was
    running when the budget was reached. Its stand-in QM now finishes a sampled label only once the next one
    is running, and the last training waits for the held labels. It's deterministic, and takes 0.5 s
    instead of several.
- Full suite: 1,160 passed. With HIPPYNN blocked: 1,050 passed, 110 skipped. With Torch blocked: 913
  passed, 212 skipped.

### CPP9 network sizing for Darwin, and restarting the CPP example (2026-11-08)

At 40,000 training structures, `hiphop145` is projected at about 66 minutes per epoch, about 14 epochs
per 16-hour Chicoma allocation. The user plans a smaller network (`hiphop64` or `hipnn64`), to be
measured on Darwin with the rest of the round.

- **`examples/CPP_excited_gpu4pyscf/benchmark_training.py`** times training and evaluation steps and
  records peak GPU memory, for every profile and batch. It builds the production S0–S3 energy and force
  graph through ALF's own builders, and runs each case in its own process, so out-of-memory results are
  recorded instead of stopping the run. It prints epoch projections for 40,000 training and 9,000
  evaluated structures. `prepare.py`'s network profiles moved into `apply_model_profile`, which both use.
- **[B8](qualification_round.md#b8-smaller-networks-for-cpp9-at-40000-training-structures):**
  - part 1, the script on one A100, checked against the November 6 `hiphop145` numbers;
  - part 2, one real epoch through `--test_ml` on the 50,000-row calibration fixture;
  - part 3, an accuracy comparison on the 1,000 historical labels.
- **CPU estimate** (one step, batch 8, against `hiphop145`): `hiphop100` 0.69× time and 0.87× memory;
  `hiphop64` 0.48× and 0.77×; `hipnn64` 0.05× and 0.23×. GPU ratios will differ, which is what B8 is for.
- **The CPP page now covers a coordinator that dies:**
  - cancel the run's `parsl.alf_*` Slurm jobs first, since a killed coordinator can't release them;
  - restart with the same command;
  - SIGTERM and Ctrl-C release the jobs themselves.
- One test runs the script's smallest case on CPU and checks its projection.

### Health check (2026-11-08)

A read-only review of the repo, tests, docs, and examples, after the spare-seed change (`38a3250`).

- **State.** 493 commits, 215 since November 1; 17k lines in `alframework/`, 6 TODOs. GPU evidence covers
  the CPP9 E6 pilot and multi-node training. Everything built since November 7 is checked on CPU only:
  restarts and spare seeds, teachers, fine-tuning, `pyscf_task` and periodic water, and the `E#`/`F#`
  names. The Darwin round and the Chicoma CPP9 campaign are what qualify it.
- **Tests.** 669 test functions in 37 files, each file with a docstring; three dependency tiers locally
  and in Actions. GPU4PySCF and HIPPYNN tests rely on stand-ins, as they must without a GPU. Twelve
  coordinator tests use `time.sleep`. The earlier full-suite stall was not reproduced.
- **Docs fixed in this pass:**
  - `exascale.md` no longer says ALF has no coordinator lock.
  - The README, "What is ALF", and the Parsl guide call `alframework/parsl_resource_configs` older
    profiles kept for reference (2023 accounts, an old executor layout), and point to the examples'
    `parsl_configs.py`; each profile says so in its docstring.
  - New example pages for `mace_fine_tuning` and `uma_fine_tuning`, from their READMEs.
- **Examples.** All 13 load in `tests/test_examples.py` and now all have pages. `IL`, `UO2`,
  `molten_salt`, and `reactive_sampling` keep site paths and are documented as templates; left as they
  are. Two untracked `__pycache__`-only folders left by the example moves were removed locally.
- **Not changed.** `hippynn_interface.py` (2,508 lines) and `pyscf/gpu4pyscf.py` (1,426) could be split,
  but not before the GPU rounds qualify them. This index (now about 1,500 lines) could move its dated
  records to an archive file.
- **Next, in order:** push, then bring up the CPP9 campaign on Chicoma (`--test_qm`, `--test_ml`, a small
  pilot with one restart, then 8 QM, 1 training, and 1 sampling node); the
  [Darwin round](qualification_round.md); training-time growth (see
  [Group C](future.md#group-c-placement-and-memory)); exascale items 7–8 after B6; then new uncertainty
  and acquisition methods.
- Docs build with no warnings (151 pages).

### Spare seeds for a finite bootstrap (2026-11-08)

Before a full CPP9 campaign on Chicoma (1,000 seeds, 10 GPU nodes), a finite bootstrap stopped with
"Finite bootstrap exhausted" if one seed's task was lost, for example at an allocation's time limit:
only returned tasks counted, and `prepare.py --count` made exactly `bootstrap_set` seeds. An
unconverged label counted as if it had succeeded, so the bootstrap ended a label short instead.
User decision: replace failed seeds from spares, and never stop while some labels converged.

- **`bootstrap_spares`** (master config, default `0`, needs `finite_bootstrap`; validated at startup in
  `batching.bootstrap_spare_count`). The seed list holds `bootstrap_set + bootstrap_spares` structures.
- **The finite bootstrap counts only converged labels** (`HeldLabels.converged()`), and hands out seed
  IDs up to `bootstrap_set + min(spares, failures)`. A failure is an unconverged held result or a task
  lost in this process. A spare is handed out only after a failure, so a clean bootstrap labels none.
- **Once the spares are used up,** ALF prints that it trains on fewer labels than `bootstrap_set`, and
  goes on. It stops with an error only if no seed converged. Each status report, and the end of the
  bootstrap, print converged, failed, and spare counts.
- **Restarts.** Held unconverged labels still count as failures. A task lost before the crash isn't
  held, so the restart rebuilds that seed and tries it again.
- **CPP example.** `prepare.py --spares`, default 5% of `--count` (50 for 1,000), writes
  `bootstrap_spares` and makes that many more seeds; the tracked `master_config.json` has 50. The AcAc
  and RDKit docs say how to keep spares from `seed_size`.
- **CPP Parsl template: 16-hour allocations for every role** (user decision, Chicoma's limit), in place
  of 30 minutes for QM and sampling and 1 hour for training. Labels and trainings still running at the
  limit are lost, and the bootstrap training alone is estimated at 1.3–7.5 hours on ~800 training frames
  (50–300 epochs at the Darwin batch-8 rate). The bootstrap waits on that training's result, so a
  training killed at walltime made the coordinator exit. The layout stays 8 QM nodes, 1 training node, and
  1 sampling node. The `ALF_*_WALLTIME` variables still override it.
- **Not covered.** A failed builder task's seeds aren't counted as failures; the bootstrap then trains on
  fewer. Ordinary (not finite) bootstraps are unchanged.

**Checked locally** with the real coordinator and stand-in tasks: a clean bootstrap labels no spares;
one unconverged and one lost seed are replaced by exactly two spares; with one spare, the run trains on
five of six and goes on; with nothing converged, it stops; a crash during a bootstrap with a failed seed
restarts and uses one spare. All 4 deliberate breaks were caught: counting returned instead of converged
labels, handing out spares before failures, no cap on spares, and stopping when they're used up.
Darwin B2 now includes a finite-bootstrap row with a failing seed.

- `test_finite_bootstrap_stops_when_structures_run_out`, where every builder or label fails, now expects
  the "no seed gave a converged label" stop instead of "Finite bootstrap exhausted".
- Full suite: 1,163 passed. With HIPPYNN blocked: 1,052 passed, 111 skipped. With Torch blocked: 910
  passed, 218 skipped. Docs build with no warnings; 133 roadmap links checked, 0 broken.

### PySCF and GPU4PySCF labeling, molecules and periodic cells (2026-11-08)

PySCF and GPU4PySCF are free and pip-installable, so they're the default QM for new examples (user
decision). PySCF was in the `full` extra; a `pyscf` extra now installs it alone.

**Decisions (user):**
- One labeler, `pyscf_task`, with `backend` `cpu` (PySCF) or `gpu` (GPU4PySCF), for molecules and periodic cells.
- An ALF-owned ASE calculator that the labeler uses.
- The qualified `gpu4pyscf_excited_state_task` keeps its code, but moves with the rest into one subpackage.
- Examples: a PySCF option in `simple_water` and a new periodic-water example.

**Built:**
- **`qm_interfaces/pyscf/calculator.py`: `PySCFCalculator`.**
  - It takes JSON-style settings, and picks a molecule or a periodic cell from `atoms.pbc`; mixed `pbc` is refused.
  - It reports `energy`, `free_energy`, `forces`, and periodic `stress`, plus `state_energies` and `state_forces` for TDA excited states of molecules.
  - Open shells use UKS, with charge and multiplicity taken from `atoms.info`.
  - The `periodic` block covers GTH basis and pseudopotential, a k-mesh, `df` (`multigrid`, `fftdf`, or `gdf`), `ke_cutoff`, and Fermi `smearing_eV`, for which the reported energy is the free energy its forces belong to.
  - The GPU path builds the same PySCF method and calls `to_gpu()`.
- **`qm_interfaces/pyscf/__init__.py`: `pyscf_task`.**
  - It labels state names (`E#`, `F#`, `dE##`) and `stress`, and is checked at startup.
  - A failed SCF or gradient gives an unconverged structure, with `qm_failed_stage`.
- **`run_gpu4pyscf_states(backend="cpu")`** runs the molecular TDA path with PySCF. Its GPU path is unchanged.
- **Examples and packaging:**
  - `examples/simple_water/master_config_pyscf.json` (ωB97X/def2-TZVP, as ORCA's);
  - `examples/periodic_water`: PBE/GTH-DZVP at Γ, a periodic HIPPYNN student, and GPU and CPU QM configs;
  - a `gpu4pyscf` extra in `setup.py`.
- **Docs:**
  - a PySCF/GPU4PySCF section in the QM guide, with settings, a CPU/GPU support table, and use outside ALF;
  - a periodic-water example page;
  - the PySCF option on the simple-water page;
  - cross-links from the excited-state guide and "Property names".

**Settled while prototyping on CPU, with PySCF 2.15:**
- **Gamma-point RKS gradients need the multigrid integrator.** They're fast: about 4 s for one water in a 5 Å box.
- **CPU stress needs k-point FFTDF.** At the Γ point it asserts FFTDF with plain `NumInt`, which can't give gradients there.
- **GDF has no CPU gradients**, and density-fitted TDA has no CPU gradients either, so CPU excited states need `density_fit: false`.
- **PySCF's stress is in Eh/Bohr³,** as `(1/V) dE/dε`, ASE's convention. PySCF's own ASE wrapper scales it by `HARTREE2EV/BOHR`, which is off by BOHR². Ours matches ASE's numerical stress to 1e-6 eV/Å³.
- **Randomly packed water can be too strained for periodic DFT.** A 5 Å box built with `min_dist` 1.25 and `shake` 0.05 had a 0.23 eV PBE gap and never converged, whatever the guess, level shift, damping, or DIIS settings. Boxes built with `min_dist` 1.6 and `shake` 0.02 converged at once, so the example uses those. Smearing of 0.27 eV converged the strained box, but 0.1 eV didn't.

**Checked locally, on CPU, with real PySCF:**
- **Against PySCF itself:** molecular energies, periodic energies, and TDA state energies matched it directly.
- **Finite differences:** ground-state, periodic, excited-state, and smeared forces matched finite differences; stress matched ASE's numerical stress.
- **As an ASE calculator:** an ASE VelocityVerlet run went through the calculator.
- **The task:** it stored `E0`/`F0` as `energy`/`forces`, plus `cell` and `stress` for periodic cells. HIPPYNN's reader and `shard_atoms` read the shards.
- **`--test_qm`:** a 9-water cluster at ωB97X/def2-SVP took 27 s, and a 15-water periodic box (7.7 Å) at PBE/GTH-DZVP took 57 s.
- **GPU routing** was checked with stand-ins only.

**Tests:** 40 in `tests/test_pyscf.py`, plus example-config tests. All 33 deliberate breaks were caught, after adding a test for the energy offset, which the first round missed. PySCF tests skip where PySCF isn't installed, as in CI's minimal job.

**Results.** Full suite, under coverage: 1147 passed in 2 min 40 s. With HIPPYNN blocked, as in CI: 1036 passed,
111 skipped. With Torch and PySCF blocked: 898 passed, 214 skipped. The docs build has no warnings, and all 132
roadmap links resolve.

One full-suite run under coverage stalled at the first CPU excited-state test for 4.5 hours at full CPU, and was
stopped. It didn't happen again: that test alone, with the 864 tests before it, and three further full runs all
finished in their usual time. The cause is unknown; if it recurs, start with that test.

**One subpackage (user decision).** The three PySCF modules are now `alframework/qm_interfaces/pyscf/`:
`__init__.py` (`pyscf_task`), `calculator.py` (`PySCFCalculator`), and `gpu4pyscf.py`, which was
`gpu4pyscf_interface.py`, moved with `git mv`. Its task's path is now
`alframework.qm_interfaces.pyscf.gpu4pyscf.gpu4pyscf_excited_state_task`, with no alias, so Darwin's B1
regenerates its configs with the new path. `ase_calculator_interface.py` is unchanged.

**A review pass (user question) found and fixed:**
- **`df: multigrid` on the GPU was wrong.** The calculator switched multigrid on in PySCF before `to_gpu()`. PySCF's
  CPU multigrid integrator has no `to_gpu`, so it would have been carried over unconverted, a CPU integrator in a GPU
  calculation. GPU4PySCF 1.9.1's RKS and UKS have their own `multigrid_numint()` (`multigrid_v3`), so it's now
  switched on after the move; a test checks the order. Its smearing defines `from_cpu`, and its gradient base class
  has `get_stress`, according to the GPU4PySCF source.
- **Hybrid functionals with periodic `multigrid`,** which handles pure functionals only, are refused per structure
  with the reason, instead of failing inside PySCF. This can't be a startup check, because one configuration also
  labels molecules, where `wb97x` is fine.
- **`--test_qm` crashed with `KeyError` when the label failed,** an older bug in `__main__.py`. It now prints the
  labeler's `qm_error` and stage, and exits with code 1. Parsl is cleaned up, as before.
- **PySCF isn't a core dependency.** It was in the `full` extra, so the docs saying it "comes with ALF" were wrong.
  A `pyscf` extra installs it alone, and the `gpu4pyscf` extra includes it.
- **Docs gaps, filled:**
  - the extras table in the installation guide;
  - the one-GPU-per-worker rule, which applies to `pyscf_task` on the GPU too;
  - the batching guide's task list;
  - the QM guide's related examples;
  - the repository README's QM list;
  - the teacher walkthrough's QM test set, which can come from the PySCF configuration.
- **New tests:**
  - `pyscf_task` through the real coordinator, with real PySCF labels, sequential and unbatched;
  - the multigrid order on the GPU, using stand-ins;
  - the hybrid refusal;
  - `--test_qm`'s failure report.
- **Checked and fine:** HIPPYNN's `PeriodicPairIndexer` counts the images it needs per direction, so the
  example's 4.5 Å cutoff in 7–8 Å boxes is fine.

After the pass: full suite under coverage, 1152 passed; with HIPPYNN blocked, 1041 passed and 111 skipped; with
Torch and PySCF blocked, 899 passed and 218 skipped. All 5 deliberate breaks of the fixes were caught, and the
docs build has no warnings.

**Not shown yet:** any GPU run, and a periodic HIPPYNN student trained on these labels, since the ground-state trainer has no CPU mode here. [B7 of the Darwin round](qualification_round.md#b7-pyscf_task-on-the-gpu-and-periodic-water) covers both.

### State names E#/F#, and GPU4PySCF ground-state DFT (2026-11-07)

Planning the Darwin round found that GPU4PySCF, ALF's only GPU DFT labeler, couldn't feed ground-state
training or fine-tuning. It needed at least one excited root and wrote the ground state as `sE0`/`F0`,
while those readers expect `energy`/`forces`.

**Decisions (user):**
- **State-aware code renames `sE#` to `E#`,** a clean break with no converter, since nothing had been run
  with this framework yet. Forces stay `F#` and gaps `dE01`.
- **Ground-state labelers and readers keep `energy`/`forces`,** for compatibility with existing
  ground-state codes. That covers ORCA, Q-Chem, VASP/ASE, the teacher, `pyseqm_task`, the ground-state
  trainer, and fine-tuning.
- **The two meet through `properties_list`:** the key names what a labeler computes, and the first element
  is the shard's dataset name. A ground-state run of a state-aware labeler writes
  `"E0": ["energy", ...], "F0": ["forces", ...]`. Before the decision, an ALF-wide rename of ground-state
  names was planned and dropped.

**Built:**
- **The rename:** `excited_state_tools` matches `^E(\d+)$` and builds gaps from `E#`; the GPU4PySCF and PySEQM
  interfaces, the excited trainer's loss names (`E{state}_RMSE`), sampler metadata rows
  (`E{state}_mean`/`_stdev`), the excited examples, docs, and tests use `E#`. A test checks that no `sE#`
  remains in code, examples, or user docs.
- **GPU4PySCF `nroots: 0`:** a ground-state calculation, the SCF energy and gradient with no TDA. It refuses
  gaps, couplings, and optical or vertical properties. TDA moved into `_run_tda`, so the excited path is
  otherwise unchanged. `qm_tda_converged` is `null` and `s1_gradient_source` is `none` when no TDA ran.
- **PySEQM's excited tasks with only `E0`/`F0`** run a ground-state AM1 solve instead of CIS, recording
  `qm_backend` `pyseqm`.
- **Docs:** a "Property names" section in the configuration guide, `nroots: 0` in the excited-state guide
  and the QM table, and a pointer from the fine-tuning guide.

**Checked locally,** with the existing fake PySCF/CuPy and PySEQM backends; neither backend is installed
here, so [B1 of the Darwin round](qualification_round.md#b1-gpu4pyscf-under-the-new-state-names) is the
real check. Under the fakes:
- an `nroots: 0` run equals an excited run's `E0`/`F0`, and ran no TDA;
- its labels stored as `energy`/`forces` are read by the HIPPYNN ground-state reader and by `shard_atoms`;
- the refusals hold;
- PySEQM's ground path uses no CIS and honours SCF convergence.

Writing the tests found one bug in the new code, which is fixed: a property check that failed before
`ground` was set raised `UnboundLocalError`. Of 18 deliberate breaks, 17 were caught after one test was
added. The one left, `td is not None` in place of `nroots > 0`, is equivalent.

**Results.** Full suite: 1100 passed. With HIPPYNN blocked, as in CI: 989 passed, 111 skipped,
and with Torch blocked: 868 passed, 197 skipped, both run before the last PySEQM test was added; that
test also passes with HIPPYNN blocked. The docs build has no warnings, and all 130 roadmap links resolve.

### UMA teacher: off-the-shelf only, and a periodic guard (2026-11-07)

**Scope (user decisions):**
- Only off-the-shelf UMA is supported as a teacher. A model fine-tuned by ALF is used directly, not as a teacher. The docs say so; the code doesn't enforce it.
- An `omol` teacher refuses periodic structures unless the user opts in.
- The fine-tuning examples are fixed in the same pass.

**What was wrong:**
- **fairchem's own docs** say `omol` training data are all aperiodic, so "any periodic systems should be treated with some caution". Sources: `docs/core/uma.md` and `docs/faq.md` in the fairchem repository; the FAQ points periodic systems to `omat`, `omc`, `odac`, or a catalysis task. fairchem itself gives no warning for this.
- **The examples gave different stages different geometries.**
  - `simple_condensed_phase_builder_task` always set `pbc=True`.
  - ORCA ignores the cell, and `simple_water`'s HIPPYNN student has `cell_key: null`, so both saw a cluster.
  - A teacher honoured `pbc=True`, so the non-periodic student learned periodic labels. The old MACE walkthrough output showed `PBCs: [True True True]`.
  - `examples/uma_fine_tuning` and `examples/mace_fine_tuning` likewise fine-tuned on periodic copies of ORCA's cluster labels.
  - With real UMA `omol`, one random 9-water geometry gave labels 1.24 eV apart, periodic versus cluster.

**Built:**
- **Teachers declare periodic support.** `TorchSimTeacher` and `AseTeacher` take `supports_periodic` (default true); a custom teacher without the attribute counts as supporting it. `load_uma_teacher` sets it from `uma_task_supports_periodic(task)`, which is false only for `omol`.
- **The refusal.** A periodic structure given to such a teacher is marked unconverged with a reason, unless `teacher.allow_periodic` is true, following the existing pattern for charged structures given to MACE. `allow_periodic` is validated at startup and recorded in `label_provenance`.
- **A builder setting.** `periodic` (default true) on both simple condensed-phase builder tasks. `false` builds a cluster inside the cell. Storage then writes no `cell`, and `shard_atoms` reads the shard back as non-periodic.
- **Examples.** `simple_water` (shared by the ORCA and teacher configurations), `uma_fine_tuning`, and `mace_fine_tuning` now set `"periodic": false`. `qm_config_teacher_uma.json` stays `omol`.
- **Docs.** In the teacher guide:
  - the off-the-shelf statement;
  - `allow_periodic` in the settings table;
  - a "Choosing a UMA task" note, and a note that student and teacher must see the same structures;
  - the walkthrough output refreshed from a real MACE run on a cluster;
  - Next Steps reworded.

  Also: the builder setting in `builders.rst` and `examples/simple_water.rst`, the `omol` row and the example note in the fine-tuning guide, and both fine-tuning READMEs.

**Checked locally, on CPU:**
- **MACE teacher.** MACE-MP-0 small, `--test_qm` on `simple_water`: a 12-water cluster, `PBCs: [False False False]`, −146.483 eV. The shard has no `cell`.
- **UMA teacher.** uma-s-1p2p1 `omol`, fairchem-core 2.25.0:
  - `--test_qm` labeled a 13-water cluster at −2079.38 eV per water, against −2079.89 eV for an isolated water, as expected for random packing. Provenance records `allow_periodic` false.
  - Directly: a periodic 9-water box was refused with the reason, labeled once `allow_periodic` was set, and the same geometry as a cluster was labeled.
- **Fine-tuning on cluster shards.** Two-copy rounds of MACE and of UMA ran through the real coordinator, with the real builder at `"periodic": false`, EMT labels, and TorchSim MD on the copies. In each, both rounds completed and were promoted, and 4 of the 12 stored clusters were sampled by the fine-tuned copies. No shard had a `cell`, and every frame read back non-periodic. The clusters were 5 or 6 waters for MACE and 3 for UMA.

**Tests:**
- a refusal with the reason, with the open structures in the same batch still labeled;
- labeling with `allow_periodic`;
- a custom teacher without the attribute;
- the library teachers' defaults;
- the UMA task helper;
- `allow_periodic` validation and provenance;
- a builder test from configuration to shard to `shard_atoms`;
- an examples test that the four cluster configurations are non-periodic.

All 13 deliberate breaks of the new lines were caught. Full suite: 1085 passed. With HIPPYNN
blocked, as in CI: 974 passed, 111 skipped. With Torch blocked: 854 passed, 196 skipped. The docs
build has no warnings, and all 114 roadmap links resolve.

### Fine-tuning docs and defaults pass (2026-11-07)

A pass over the fine-tuning guide, examples, and tests (user request), against each family's own advice:

- **The guide now defines fine-tuning in ALF:** each copy starts from a pretrained model's weights, with
  a fresh optimizer, on labels ALF selected, and every round starts again from the same pretrained model.
  A table contrasts it with training from scratch, resuming a training, and distillation. A second table
  says whose recipe ALF runs: MACE's and fairchem's own, and ALF's own for HIPPYNN, whose documentation
  covers restarting but not fine-tuning.
- **No model weights ship with ALF,** and the guide says so, with where MACE and UMA models come from and
  their terms: MIT or the Academic Software License per MACE model; the FAIR Chemistry License for UMA,
  with its acceptable-use policy, derivative copies, and acknowledgment in publications.
- **MACE defaults follow MACE's guidance** (user decision): `max_num_epochs` 6, `batch_size` 2, EMA 0.999,
  AMSGrad, gradient clipping 1.0, no weight decay, energy and force weights 10, and `rms_forces_scaling`;
  with replay, `lr` 1e-4, EMA 0.9999, and 20 epochs. `mace_options` overrides any of them. ALF warns at
  startup on `E0s: "average"`, which MACE advises against, and the example now uses `"estimated"`. Rerun
  on CPU with these defaults: two copies cut the held-out energy error against EMT from 0.94 eV to 0.011
  and 0.022 eV, better than the earlier 0.04 and 0.10 eV, in 23 s.
- **UMA defaults to 1 epoch,** fairchem's template value (user decision); the UMA example sets 10.
- **Internal details:** the roadmap's personal evidence and environment paths are now `<validation
  root>/` and `<environment root>/`, and node and front-end names are gone (user decision). Job and
  allocation IDs and partition names stay. The IL and UO2 examples write QM scratch to `qm_scratch/`
  instead of another user's home.
- **One package** (user question about the uneven names): the task and the shared shard reader are
  `ml_interfaces/fine_tuning/__init__.py`, and the adapters are `fine_tuning/hippynn.py`, `mace.py`, and
  `uma.py`, so adapter strings read `alframework.ml_interfaces.fine_tuning.mace.MaceFineTuner`. UMA no
  longer imports MACE's module, and the reader is now `shard_atoms`, apart from `restarts.shard_frames`.
- **Tests:** MACE's defaults, the replay defaults, a user override winning, the `average` warning, and
  UMA's default are covered; all 8 deliberate breaks of them were caught. No test references removed
  code or personal paths.

### Fine-tuning with ALF: UMA (2026-11-07)

The third adapter for the [fine-tuning interface](#fine-tuning-with-alf-mace-2026-11-07). User decisions: ALF
writes fairchem's training configuration itself, rather than needing fairchem's git repository; every UMA
task; charge and spin stored in shards; both reference-energy modes.

- **A prototype came first** (fairchem-core 2.25.0, `uma-s-1p2p1`, CPU, `HF_HUB_OFFLINE=1`) and settled seven
  questions: fine-tuning runs on CPU from a local checkpoint with no network
  (`initialize_finetuning_model(checkpoint_location=...)` in place of the template's download by name); the
  output is `<run_dir>/<timestamp_id>/checkpoints/final/inference_ckpt.pt`; it loads through `FairChemModel`;
  `omol` reads charge and spin through `r_data_keys`; the checkpoint's `tasks_config` holds each task's
  reference energies and normalizer, so keep mode is possible; `uma-s-1p2p1` fine-tunes as the template's
  `uma-s-1p1` does; and about 5 s per step on CPU for water. It also found that
  `python -m fairchem.core._cli` silently does nothing (the module has no `__main__` block, so ALF calls
  `main()`), that the job wants `CPU` or `CUDA` rather than lower case, and that an inference-only stress task
  breaks prediction from a model fine-tuned without stress.
- **`ml_interfaces/fine_tuning/uma.py`:** `UmaFineTuner` runs each copy in a child process
  (`python -m alframework.ml_interfaces.fine_tuning.uma`): it reads the shards, splits training and validation
  with the copy's seed, writes ASE-LMDB, refits references by least squares (`average`) or reads the
  checkpoint's (`foundation`), writes the configuration as JSON (valid YAML, so no PyYAML on the coordinator),
  runs fairchem's `main()`, and keeps `inference_ckpt.pt` as `uma.pt`. Resume checkpoints are removed unless
  `keep_training_checkpoints`.
- **`load_uma_batched_ensemble`** serves the batched engines. ALF's engines pass no charge or spin per
  structure, and fairchem would otherwise use spin 0, so `omol` sampling applies one `charge` and `spin`
  (default a neutral singlet) to every structure.
- **`ml_interfaces/torchsim_ensemble.py`:** one TorchSim evaluator for MACE and UMA, building a `SimState`
  with optional charge and spin.
- **Charge and spin in shards:** when any structure's metadata has `charge` or `mult`, `store_current_data`
  writes `total_charge` and `spin_multiplicity` for the whole shard, defaulting to 0 and 1. The HIPPYNN reader
  drops them unless a model names them, because mixing shards with and without them broke it ("Inconsistent
  property keys across ALF structure groups", reproduced before the change).
- **`examples/uma_fine_tuning`** (`omol`, ORCA labels, `E0s: "average"`), and a UMA section in the guide.

**Checked locally, on CPU, with real UMA** (`uma-s-1p2p1`, fairchem-core 2.25.0, in a scratch environment
layered on the ALF one):

- Two-copy rounds, 8 steps each, about 90 s: `omol` with refit, `omol` with keep (energies stay on UMA's
  scale, near −2080 eV for water), and `omat` with refit on periodic copper. All completed, recorded their
  lineage, and loaded through `load_uma_batched_ensemble`.
- OH⁻ (charge −1) and the OH radical (multiplicity 2) reached the training database with their charge and
  spin.
- Copies of one checkpoint disagreed clearly, unlike small HIPPYNN copies: 0.24 eV and 0.22 eV/Å for `omol`
  with refit, 0.035 eV and 0.042 eV/Å with keep.
- The real coordinator, with single water molecules, EMT labels through a teacher, UMA fine-tuning, and
  TorchSim MD on the copies: round 0 was promoted, its copies selected 4 structures, and round 1 fine-tuned
  from the original checkpoint on both shards and was promoted, in 14.5 minutes.

**Results.** Full suite: 1063 passed. With the packages CI's HIPPYNN job lacks blocked: 952 passed, 111
skipped. With Torch blocked: 832 passed, 196 skipped. The UMA tests run without fairchem. All 28 deliberate
breaks were caught. Docs build with no warnings; 112 roadmap links, 0 broken.

**Not shown yet.** GPU runs and GPU memory for several copies, fine-tuning on labels at UMA's own level of
theory, larger UMA checkpoints, the multi-node form with real UMA, and any fairchem version other than 2.25.

### Fine-tuning with ALF: HIPPYNN (2026-11-07)

A HIPPYNN adapter for the [fine-tuning interface](#fine-tuning-with-alf-mace-2026-11-07), based on the old
`examples/fine_tuning`, which now uses it. User decisions: convert that example and drop its local task;
accept a single model or an ensemble as the source; start every round from the pretrained model, as MACE
does (asked which is better: this keeps rounds reproducible and stops errors building up from round to
round; a "continue from the last round" setting can come later if round cost matters).

- **`ml_interfaces/fine_tuning/hippynn.py`:** `HippynnFineTuner` loads each source with
  `load_checkpoint(..., restart_db=False)`, keeps the graph and weights, and trains with a fresh Adam,
  `RaiseBatchSizeOnPlateau`, and `PatienceController`, as the old example did, in a child process per copy
  (`python -m alframework.ml_interfaces.fine_tuning.hippynn`) so each sees only its GPU.
  - Source: a model directory, or an ensemble of `model-*` members, where copy *i* starts from member *i*,
    cycling. A pretrained ensemble in `models/model-0000` is sampled with from the start, skipping the
    bootstrap, as in the old example.
  - Shards are read in place through `ALFPyAniDirectoryDB(files=...)`, not copied per member.
  - `E0s` is required: `foundation` keeps the energy head's atomic energies, and `average` refits them.
  - `freeze` takes graph node names such as `hipnn`, found through the graph's `names_dict`, or parameter
    prefixes. Integer parameters such as the species map are never trained.
  - `stopping_key` defaults to the one the pretrained model was trained with, from its saved controller;
    the old default, `Loss-Error`, doesn't exist in every model.
  - Small early rounds train: each split gets at least one structure, and when 10% of the training split
    is less than one structure, HIPPYNN scores the whole split each epoch. Fewer than 3 labels is an error.
  - The output is HIPPYNN's usual layout, so `HIPNN_ASE_load_ensemble` and `load_hippynn_batched_ensemble`
    load the copies unchanged. `h5_test_dir` uses `use_separate_test_set`.
- **The interface:** `train_member` now receives the copy's index, `foundation_model` may be a directory,
  and an adapter's `fingerprint()` puts each source file's SHA-256 in the lineage as
  `foundation_model_files`.
- **Docs:** the fine-tuning guide is rewritten around the one interface, with "two ways to start", common
  settings, HIPPYNN and MACE sections, freezing, and walkthroughs for both; the example page is rewritten.

**Checked locally, on CPU, with real HIPPYNN** (0.1.3), from small ensembles that `tests.helpers.hippynn`
trains:

- A round from a two-member ensemble fine-tuned three copies (member 0, 1, 0) in 18 s. Weights changed, the
  lineage listed all four source files' SHA-256, and `HIPNN_ASE_load_ensemble` and
  `load_hippynn_batched_ensemble` loaded the copies.
- The real coordinator, with single water molecules, EMT labels through a teacher, HIPPYNN fine-tuning, and
  TorchSim MD on the copies:
  - from a pretrained ensemble in `models/model-0000`: no bootstrap; all 4 stored labels were sampled with
    the pretrained ensemble, and round 1 was promoted;
  - from one model and a bootstrap: round 0 was promoted, sampling on its copies selected 3 structures, and
    round 1 fine-tuned from the original model on both shards and was promoted.
- **Copies of one model barely disagree.** On one water molecule, two copies fine-tuned for 3 epochs from
  one model differed by 0.014 eV in energy and 1.4e-6 eV/Å in forces; copies from the two members of an
  ensemble differed by 0.043 eV and 3.1e-3 eV/Å, and the pretrained ensemble itself by 0.83 eV and
  0.08 eV/Å. So the single-model loop above used cutoffs of zero. These are tiny models and few epochs,
  but it's the concern the roadmap raised for foundation models, and the guide now warns about it. Real
  runs should check `Es` and `Fsmax` against their cutoffs in early rounds.
- Two bugs found and fixed by these runs: a 10% split of a few labels was empty, and so was 10% of a small
  training split when HIPPYNN scored it; and HIPPYNN's integer species map can't take a gradient.

**Results.** Full suite: 1040 passed. With the packages CI's HIPPYNN job lacks blocked: 930 passed, 110
skipped. With Torch blocked: 811 passed, 194 skipped; the new HIPPYNN tests skip there. 27 deliberate breaks
were caught, after tests were added for three gaps: a copy whose log never finished, integer parameters, and
shards missing a dataset the model needs. Docs build with no warnings; 109 roadmap links, 0 broken.

**Not shown yet.** GPU runs, a real pretrained HIPPYNN model of useful size, and the multi-node form with
real HIPPYNN (it ran with a stand-in adapter).

### Fine-tuning with ALF: MACE (2026-11-07)

ALF now selects the samples that fine-tune an existing model (user decisions in
[Group D](future.md#group-d-model-families-and-labels)). Guides: [fine-tuning](../source/user_guide/fine_tuning.rst)
and [teacher–student labeling](../source/user_guide/teacher_student.rst).

- **The loop is unchanged.** The bootstrap is labeled, then each round's `ML_task` fine-tunes `n_models` copies
  of the original checkpoint on every stored label, each with seed `seed + i`. MD runs on the copies through
  the batched engines, and their disagreement selects frames as for any ensemble.
- **`ml_interfaces/fine_tuning/`:** `fine_tune_ensemble_task` (one GPU per copy at a time on one node, or
  one after another on CPU) and `fine_tune_ensemble_multinode_task` (one app per copy, with retries and
  placement records). Both wait for stored shards, refuse a nonempty ensemble directory, and write
  `label_sources.json` and `fine_tune_lineage.json`: checkpoint path and SHA-256, adapter, package version,
  effective settings, shards, seeds. A model family plugs in with an adapter of five methods.
- **`ml_interfaces/fine_tuning/mace.py`:** `MaceFineTuner` writes the shards as extended XYZ and runs
  `mace.cli.run_train` per copy, in its own directory, with `--multiheads_finetuning False` unless
  `replay_file` is set (MACE would otherwise download replay data). `E0s` is required. Settings ALF manages
  are refused in `mace_options`. `load_mace_batched_ensemble` evaluates every copy through MACE's TorchSim
  model, turning ALF's row-vector cells into TorchSim's column vectors.
- **`__main__.py`:** one startup hook, so an ML task's `validate_config` stops a bad configuration before the
  bootstrap, as labeling already does.
- **`h5_test_dir` fixed.** The ground-state trainer's separate test set changed the training arrays and
  called numpy methods on tensors, so it failed whenever it was set; the teacher guide recommends it. Now
  `use_separate_test_set` gives the test arrays the training arrays' form.
- **Docs:** a teacher–student page with a walkthrough, moved out of the QM page; a fine-tuning page; the
  old `examples/fine_tuning` page retitled as the older HIPPYNN-only example; `examples/mace_fine_tuning`.

**Checked locally, on CPU, with real MACE** (`mace-torch` 0.3.17, MACE-MP-0 small, in a scratch environment
layered on the ALF one):

- Two copies fine-tuned on 60 water frames labeled with EMT, so the labels differ from MACE, in 28 s. On
  held-out frames, energy error against EMT fell from 0.94 eV for the foundation model to 0.04 and 0.10 eV,
  after removing each model's constant offset. The copies disagreed by 0.39 eV and 0.30 eV/Å on average.
- The real coordinator, with the real condensed-phase builder, EMT labels through a teacher, MACE fine-tuning,
  and TorchSim MD on the copies: the bootstrap fine-tuned `model-0000`, sampling on it selected new water
  boxes, and the second round fine-tuned from the original checkpoint on both shards and was promoted, in 39 s.
- The teacher walkthrough's `--test_qm` ran with real MACE-MP-0 small; the page shows its output.

**Results.** Full suite: 1017 passed. With the packages CI's HIPPYNN job lacks blocked: 907 passed, 110
skipped. With Torch blocked: 800 passed, 182 skipped. 30 deliberate breaks were caught, after one test fix:
`E0s` given as JSON, with string keys, wasn't tested. Docs build with no warnings; 105 roadmap links, 0 broken.

**Not shown yet.** GPU runs, and the multi-node form with real MACE (it ran with a stand-in adapter).
Whether copies fine-tuned from one checkpoint disagree enough to track error is open; the roadmap's earlier
note expected them to agree too closely. Large label sets: each copy rewrites every label to extended XYZ
each round.

### Progress plots for batched and excited-state runs (2026-11-07)

`analysis_plot` was written for one ground-state trajectory per task. On excited-state runs its uncertainty
plots came out empty or skewed, because a screen rejection replaced `Es`, `Fs`, and `Fsmax` with `None`, and
every trajectory in a batch recorded the task's duration as its wall time. It also needed `meta_dir`, one of
the High rows in [`exascale.md`](exascale.md).

- **Metadata:** a rejected trajectory keeps its last valid uncertainty, and trajectories now record `peak_Es`,
  `peak_Fs`, `peak_Fsmax`, and their own wall time. `alf_validation` may need updating if it relies on the old
  values, which it couldn't be checked for here.
- **Summary:** with `trajectory_summary_path`, `SamplerCoordinator.harvest` appends one row per trajectory to
  one file, so plots need no `meta_dir` (user question: how does this work at exascale?).
- **Plots:** new outcome and per-state plots, and sampling-window lines. Plots with no data are skipped and
  reported, and `meta_dir` still works.

Checked with real excited-state trajectories through the real sampler using fake models, a force screen, and a
run through the real coordinator, and by looking at the images. Local result: 883 passed, 2 skipped (`seqm` not
installed), and the new tests also pass without Torch. At 1M rows, measured on a laptop: 350 MB, about 8 s
per update, and 1.3 GB peak memory. Selection and stopping don't use the changed fields.
Real GPU sampling and plotting on shared storage subsequently passed on all three
engines; see the [November 7 hardware record](#current-development-hardware-qualification--passed-with-restart-limits-2026-11-07).

### Current development hardware qualification — passed with restart limits (2026-11-07)

Qualified ALF **`ce711d8`**, preserving the earlier E6, quick-win and gap/window
source boundaries. Evidence is outside Git under
`validation/development_20261107/`: `qualification_summary.json` and
`qualification_summary.md`, frozen `run-002/` and `run-003/`, and the preserved-loop
recheck `loop-recheck-002/verification.json`. Harness revisions include `a50afa9`,
`60104bb`, `fa0ff20` and final verifier `b36e02d`, committed separately in
`alf_validation`. Qualified environments and accepted Increment 9 formaldehyde
inputs were reused without dependency upgrades. No ALF library fix was needed.

- **Both trainers:** eight members on two nodes/four GPUs each passed clean and
  cancelled-block runs. Placement showed eight distinct host/GPU pairs and actual
  optimizer overlap. Every member ran five epochs, changed parameters, preserved
  its GPU mask, loaded the same shards and reloaded. Only the four members on the
  cancelled block retried; failed directories were retained.
- **Scale:** 20 excited-state members trained concurrently on 20 distinct GPUs on
  five GPU nodes, two with A6000s and three with A5000s. All members reloaded;
  maximum representative force finite-difference error was **1.65e-6 eV/Å**.
  This is independent-member placement, not distributed training of one member;
  ten-node/40-member scaling and larger-dataset throughput remain unqualified.
- **Sampling and plots:** ASE, TorchSim CUDA and ALCHEMI passed actual widths
  one/four, S0/S1 propagation, initial-invalid-input isolation and identity checks
  using the new eight-member models. Exact harvested summary rows, peak/final
  diagnostics, state plots and legacy-pickle plots passed. Ground-state sampling
  also ran with the newly trained ensemble. New excited-state models loaded from
  `member_composition.json` without reading `training_summary.json`.
- **Exit paths:** real `--test_ml` allocations released automatically; coordinator
  `SIGTERM` and a permanently failed ML executor produced the expected exit and
  cleanup. The latter released the other executors too. This does not demonstrate
  `SIGKILL` cleanup or literal `scancel` of a batch-hosted coordinator.
- **Ordinary ALF loop:** cancellation with `member_retries=0` left failed generation
  1 unpromoted and allowed generation 2 to train/promote. All eight final members
  retained **20 new propagated labels**, with all 20 represented across actual
  training splits. The loop stored **52 frames** (32 bootstrap + 20 sampled),
  exceeding its 48-label budget by four. Independent FPS reconstruction, both
  base/gap-seeking windows, finite reloads and consistent gaps passed. Final
  representative force finite-difference error was **3.96e-6 eV/Å**.

The raw `run-003/result.json` remains **incomplete**, preserving its frozen
verifier's failure: it required another polling record after final promotion,
although ALF promoted and exited normally without another sleep. A separate
verification of unchanged evidence passed. It also accounts explicitly for four
sampler results completed after the budget stopped harvesting; they were reported
as dropped, not missing harvested summary rows. Shutdown reported two remaining QM
structures, zero finished-but-unstored QM structures and zero builder structures.
All original failed reports remain intact.

Filesystem/lock measurements passed, but restart work is **baseline evidence, not
crash-safe recovery acceptance**: 11 interruption observations completed and one
best-effort store-interruption repeat did not hit its trigger. Finite-bootstrap
restart failed; `SIGKILL` left allocations alive beyond the 120-second observation
window until owned-resource cleanup. See the
[measured I/O/restart record](exascale.md#darwin-measurements-2026-11-07).

The library suite passed **807 tests with 78 optional-backend skips**, counting the
single socket-restricted test rerun outside the sandbox. The external suite passed
196 tests; the final affected harness suite passed 25. Persistent frontend `tmux`
supervision used the original **08:41:58–12:41:58 Mountain** queue-inclusive budget;
cleanup finished at approximately **11:47:38**, before the deadline. Both attempts'
watchdogs and a final scheduler inspection verified all owned jobs terminated.
No user-guide changes, dependency upgrades, new campaign or push accompanied this
round. Safe restarts remain tabled; the priority order is unchanged.

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

### Validation artifacts and environment retention (2026-11-08)

The user authorized retiring completed bulk validation artifacts after their results were
recorded. `<validation root>/records/20261108/` keeps compact result/acceptance reports,
source and artifact fingerprints, installed-package manifests, cleanup inventories, and a
compressed archive of historical reports. Large checkpoints, repeated-row capacity fixtures,
Parsl worker trees, scratch directories, and redundant frozen ALF source trees are disposable.
Their historical paths in this roadmap are provenance references; they may no longer exist.
Reconstruct a run from the recorded ALF/harness revisions and settings rather than expecting
the original worker directory to remain available.

Small inputs retained for future focused checks are the accepted Increment 9 S0–S3 ensemble
and labels (`inc9_20261105/run-003`), B1's isolated 64-frame DFT test set, B3's final UMA-teacher
HIPPYNN student and its labeled shards, and the 1,000-row CPP source fixture. The 50,000-row
capacity copies can be regenerated. The original pretrained MACE and UMA checkpoints outside
`validation/` are untouched. The entire `github/molecule_smax` checkout, including its original
`uma-s-1p2p1.pt`, is explicitly outside this cleanup. Keeping these fixtures does not qualify
additional science.

The retained Increment 9 manifest uses the historical logical `sE#` keys. Current development
uses `E#`; the fixture reload check translates those keys in memory, retaining the same dataset
names, units, scales and values. It does not rewrite the accepted fixture or reapply an offset.

Keep the following nine frozen environments under `<environment root>/`:

| Relative path | Use |
| --- | --- |
| `20261105/inc11a/cpu` | CPU development tests and coordinator |
| `20261105/inc11a/ase` | ASE sampling |
| `20261105/inc11a/torchsim` | TorchSim CUDA and shared base packages for MACE/UMA |
| `20261105/inc11a/alchemi` | Qualified ALCHEMI sampling |
| `20261105/inc9-training-cuda` | Ground/excited HIPPYNN training |
| `20261105/inc8/pyseqm` | Qualified AM1/CIS labeling |
| `20261106/cpp9-gpu4pyscf-cutensor` | GPU4PySCF 1.9.1/PySCF 2.15.0 with native cuTENSOR |
| `20261108/mace` | MACE teacher/fine-tuning; overlays the retained TorchSim site-packages |
| `20261108/uma` | UMA teacher/fine-tuning with its separate Torch 2.13 build and retained base packages |

The five November 4 venvs and `20261105/inc8/gpu4pyscf` are superseded for current development.
Their retirement, together with `20261104/cache`, was explicitly approved. Package manifests
are preserved before removal. Hardlinked package files needed by retained environments remain
allocated; removing a cache directory does not imply reclaiming its entire displayed size.
No retained environment is moved, upgraded, or consolidated, and MACE/UMA's `.pth` dependencies
must remain valid. Personal conda environments and upstream HIPPYNN/PySEQM checkouts are outside
this cleanup. The original upstream source directories under the environment root are retained.

Cleanup completed with all 290 removal targets journaled: **76.85 GiB** of validation payloads
and **27.13 GiB** of superseded environments/cache, approximately **104 GiB** in total.
These are allocated file blocks whose final hardlink was removed; directory metadata is excluded.
The compressed historical report archive contains 66,101 verified members. All nine retained
environments passed the same role-specific import/module-discovery checks, with unchanged package
versions and module locations. Retained fixture hashes were unchanged; both four-member ensembles
reloaded and produced finite CPU predictions with public PyTorch kernels and GCC 11, and all six
retained HDF5 files were read successfully. The execution journal and checks are in
`<validation root>/records/20261108/`.

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
- **Location risk.** All of these environments live on the shared home filesystem, where a
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
`<environment root>/20261104/` for independent environments.
All five builds exclude personal/system site-packages: base CPU, NVIDIA
training, TorchSim CUDA, TorchSim ROCm, and ALCHEMI. Current-release resolution
and package checks are recorded outside Git under
`<validation root>/inc7_closeout_20261104/`.
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
`alf-inc7-fresh-20261104` on a Darwin front end waits for the trajectory-failure case,
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
