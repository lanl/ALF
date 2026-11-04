# Living roadmap for integration into ALF development

Last updated: November 4, 2026 (Increments 0–6 and the Increment 7 prerequisites merged at `24028da`;
core water, cohort/summed, MI250, and PySEQM matrices passed; two controlled-failure
cases passed, while the trajectory-failure case remains incomplete. MI350 is deferred outside
current acceptance. Post-release order decided: excited-state ALF, then
composable sampling with batched UDD/gap ALF, then staged campaigns, with
multi-node training and teacher labeling as parallel tracks; see **Decided
priority order**. Fresh-install environment policy added; see **Development
and test environments**).

## Quick start for contributors and agents

Read these sections before starting work:
1. **Guiding constraints**
2. **Branch and maintenance workflow**
3. **Development and test environments**
4. The current increment's section: **Increment 7 close-out**, then
   **Decided priority order** and its E1–E6 table.

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
  and update this roadmap's status table and the relevant section.
- The fork is source material, not an integration target. Its own roadmap copy
  is stale; this document is authoritative.

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

## Development and test environments

Policy, decided 2026-11-03: test development work against fresh installs with
up-to-date dependencies, not long-lived personal environments.
User clarification: this is the next acceptance priority after the running
Increment 7 retries, before new feature implementation. Compatibility with
current upstream releases is the target; `atomistic` and the other existing
environments supply historical comparison evidence, not the target dependency
versions. Keep the running retries frozen while completing them.

### Current state (audited 2026-11-03)

No regression so far has run against a clean install of the development checkout.

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

## First-release increments

Status values: **planned**, **in progress**, **done locally** (committed on
local `development`, not yet pushed), and **merged** (in official
`lanl/ALF:development`).

On 2026-11-02 the user pushed local `development` directly to official
`development` (tip `d5a0395`), rather than through separate fork PRs.
Increments 0–3 and 4a–4b are therefore merged. The user pushed again the same day
(tip `86f62ca`), merging 4c and Increment 5. A live remote check on November 2
confirmed official development at `be7fb64`, including Increment 6 and the
shared-test rename. A later push (tip `24028da`, verified November 2) merged the
Increment 7 prerequisite fixes and their acceptance record. Main remains
`206ea44`, already included in development.
On November 3, a live remote check confirmed development at `d4bf99d`, including
the core campaign harness and MPI/deadline repair; main remains `206ea44`.
Older progress entries retain their dated status at the time of writing; the
table below is the current status. Agents still commit locally only; the user
pushes.

| Increment | Status | Deliverable | Acceptance gate |
| --- | --- | --- | --- |
| 0. Roadmap and baseline | merged (`d5a0395`) | Publish this document, synchronize development with main, and record existing-example/test behavior. | Baseline results and unavailable dependencies are documented. |
| 1. Shared batching helpers | merged (`d5a0395`) | Compatible grouping, waiting buffers, timeout submission, actual-width accounting, output normalization, and batch-builder ID preservation. | Focused tests pass; existing execution is unchanged until batching is selected. |
| 2. Labeling coordination | merged (`d5a0395`) | Batched labeling during bootstrap and active learning; explicit sequential compatibility for existing single-structure labelers. | Actual coordinator tests cover labeling, partial batches, storage, and retraining. |
| 3. Native PySEQM labeling | merged (`d5a0395`) | Ground-state batch energies/forces, adapter-local padding and atom-order restoration, and per-member convergence handling. | Native batch results agree with individual evaluations; successful labels survive partial failures. |
| 4. Ground-state sampling and acquisition | merged (`86f62ca`) | Minimal physical-inference/MLMD contracts, uncertainty and summed ranking, first-trigger returns, and an in-memory logical-cohort selector. | Shared observations produce equivalent screening, scores, stopping, and selection across execution strategies. |
| 5. TorchSim dynamics | merged (`86f62ca`) | Supported NVIDIA/AMD ground-state batched MLMD. | CPU, NVIDIA, and Darwin MI250 mechanics and integration acceptance pass. |
| 6. ALCHEMI dynamics | merged (`be7fb64`) | ALCHEMI 0.2.0 NVIDIA batched MLMD through the same contracts, in its separate environment. | Shared conformance tests and NVIDIA acceptance pass; TorchSim CUDA/ROCm regressions pass. |
| 7. Combined release validation | in progress | Opt-in examples and integration/performance results. | All single/batched dynamics x single/batched labeling combinations complete representative ALF workflows. |

Port narrowly scoped implementations and tests. Each increment must be
independently reviewable and tested before dependent increments merge. Inspect
PBC propagation in the multi-structure condensed-phase builder as a focused
prerequisite; retain the recent main fixes. (Done in Increment 0; see below.)

### Increment 7 — core water matrix accepted; remaining release validation open

The engine work is merged, and the twelve-case Darwin water campaign now
validates first-trigger active learning across all three engines. Cohort/summed
acquisition and the other backend gates below remain open. Keep this increment focused on existing workflows and the
concrete correctness gaps listed under **Known implementation follow-ups**.
Longer-term plugin interfaces and durable recovery remain later increments.

1. Address relevant focused fixes before relying on the affected workflow:
   periodic `distcut` screening, scheduler GPU visibility, and the legacy ASE
   loader's concurrent `chdir` behavior. Make each a separate, reviewable change;
   do not combine them with a sampler redesign or fork-roadmap synchronization.
2. Run tracked examples through the real Darwin/Parsl coordinator, keeping an
   unchanged legacy ASE example as the compatibility baseline. Cover single
   and batched dynamics with ASE, TorchSim, and ALCHEMI, crossed with single
   labeling and sequential/native batching where supported. Native PySEQM
   needs molecular inputs: periodic dynamics support does not make PySEQM a
   periodic labeler. Do not silently clear PBC to make a combination run.
3. Exercise bootstrap → sampling → acquisition → labeling → HDF5 → actual
   retraining/model promotion, including first-trigger, uncertainty/summed
   ranking, cohort stop/continue, partial batches, and partial labeling failures.
   Check that older cohorts finish on their pinned model. Local coordinator
   tests with mock labelers/trainers are already covered; they are not this
   production-backend campaign.
4. Record candidate-to-labeling latency and cohort completion/selection latency
   as well as throughput, queue occupancy, and labeled/stored frame counts.
   Increment 6 measured inference/dynamics throughput, not selection latency.
   Name every memory metric explicitly; peak Torch allocation excludes possible
   Warp and other process allocations.
5. Finish outstanding backend acceptance separately: native PySEQM GPU labeling
   and retraining on its labels, a trained HIPPYNN cell-input checkpoint, longer
   potential-stability checks. MI350 testing is deferred by the user and is outside
   current Increment 7 acceptance; future execution needs separate qualification. An unavailable
   item must stay explicitly open rather than being inferred from MI250/A100
   results or a molecular checkpoint's periodic pair adaptation.

Keep the flat sampler configuration and existing task entrypoints throughout.
Record source/dependency/checkpoint fingerprints with results, but keep large
environment dumps and dated validation reports in local artifacts, outside Git.

#### Core water campaign matrix (2026-11-02)

The opt-in `examples/simple_water_campaign/` suite prepares twelve immutable
profiles and runs the real coordinator through observed production task
functions. Every builder returns four periodic water boxes. Dynamics cross
ASE/TorchSim/ALCHEMI at widths one/four with single ORCA labeling or sequential
packs of four. ASE width one retains the legacy sampler. ORCA uses
PBE/def2-SVP/RI-J; the HIPPYNN ensemble has four members. Other scientific
settings stay with the water example, except for a 0.1 ps trajectory cap.

One real bootstrap targets 32 converged stored frames and trains the initial
ensemble. Each isolated case copies those pinned training inputs/checkpoints
and must promote a genuinely trained ensemble with at least eight sampled,
converged, stored IDs in its training input. New labels/models are not shared.
Existing uncertainty cutoffs are retained; acquisition shortfalls remain
incomplete. Full-width sampler and QM tasks must actually complete before a
batched case can pass. The runner checks HDF5 units, forces, coordinates, cells,
atom sorting, member training logs, and checkpoint loading; it records
retention/split counts separately from training-input IDs.

The elapsed ceiling is four hours for shared preflights/bootstrap and four
hours per engine group (sixteen hours total), including queue waits. Cases
are capped at 55 minutes including cleanup. The supervisor journals owned
Slurm jobs, verifies cancellation, and refuses further allocations after an
unverified cleanup. Generated data/logs/checkpoints stay outside Git. Native
PySEQM GPU labeling, full MI250 campaigns, cohort/summed acquisition, deliberate
failure campaigns, and long-time stability remain separate acceptance work.

QM/training use the existing `atomistic` environment; TorchSim uses its
qualified CUDA environment. ALCHEMI's Python 3.11 computation is isolated in a
data-only subprocess adapter behind Python 3.12 Parsl transport, avoiding
cross-version function serialization or modifications to existing environments.
This process startup is part of the campaign harness rather than a standalone
engine benchmark. Production task signatures and `__main__.py` are unchanged.

Focused harness tests currently pass (15 checks), including real local Parsl
submission auditing, the real multi-builder adapter, independent seed copies,
late-bootstrap accounting, HDF5 validation, owned-job cleanup, deadlines, and
Python 3.12-to-3.11 data transfer. All twelve live cases passed, including actual
full-width execution. See the example README for commands and artifact layout.

The first live preflight at `cf29d50` reached both single and packed ORCA
calls, but MPI refused eight ranks because the initial CPU profile exposed
only one Slurm task/slot. Jobs `17373266`/`17373267` were cancelled and cleanup
verified. The validation profile now reserves sixteen MPI slots and divides
worker affinity blocks; ORCA ranks use one thread each. The repaired attempt
retains the first attempt's budget origin rather than resetting its deadlines.

The repaired `cd46b3f` attempt passed the real preflight: all four recorded
boxes converged in both single and sequential-batched ORCA execution, with
zero energy/force differences. The persistent supervisor on `darwin-fe2`
completed all twelve cases at 22:06 MST on November 2, within approximately
2 hours 53 minutes of the original budget origin, including the MPI repair.
The final reports were checked on November 3. Artifacts are outside Git under
`/vast/home/pjlohr/ALF_LANL/validation/inc7_water_matrix_20261102/`:
`run-001/` retains the failed MPI attempt, `run-002/` is the completed attempt,
`suite-002.log` records stage progress, and `validation-summary.json` records
CPU test evidence and the actual imported environment versions. Final results
are in `run-002/result.json` (`status: passed`, `complete_matrix: true`) and
each case's `result.json`.

| Engine | Passed cases | Case elapsed range, including cleanup |
| --- | ---: | ---: |
| ASE | 4/4 | 11.3–13.4 min |
| TorchSim CUDA | 4/4 | 9.3–14.4 min |
| ALCHEMI 0.2 | 4/4 | 9.6–15.5 min |

The shared bootstrap stored 32 converged structures and trained four loadable
initial members. Every case promoted model 1 with eight new sampled IDs in
its training-input manifest and four valid output members with actual epochs.
All members retained 40 frames (33 train, 4 validation, 3 test). Across cases,
128 labeling attempts converged; 96 new frames entered the accepted training
inputs, with the remaining 32 labels recorded as overshoot. Each case stored
40 frames including its seed. Actual sampler widths were one/four as required,
and actual QM task widths were one/four as required. Individual/packed
prediction checks passed for every case. Bootstrap and all twelve case cleanup
reports verified termination of owned jobs. These short workflows establish
execution correctness, not engine speedup or bulk-water potential accuracy.

Complete CPU regressions with the final MPI/deadline checks:

| Environment | Passed | Skipped | Elapsed |
| --- | ---: | ---: | ---: |
| atomistic | 230 | 38 | 133.19 s |
| TorchSim CUDA environment, CPU execution | 248 | 20 | 180.52 s |
| alf_alchemi_02 | 248 | 20 | 178.29 s |

The skips are unavailable optional backends. No production coordinator,
training implementation, or existing example was modified for this harness.

#### Remaining campaigns — implemented; external acceptance pending (2026-11-03)

Keep Increment 7 open and extend the opt-in campaign harness before starting
new model, training, or acquisition capabilities. The next implemented matrix is
twelve cases: ASE/TorchSim/ALCHEMI crossed with uncertainty/summed ranking and
cohort `stop`/`continue`. Use dynamics width four, sequential ORCA packs of
four, logical cohort size eight, and K = four. Four active trajectory slots
must admit a cohort incrementally; cohort size is independent of task width.
Keep the existing water science settings and 0.1 ps bound. Summed weights are
fixed at energy = one and force RMS = one; ranking does not change eligibility cutoffs.

Reuse the accepted shared bootstrap only through a new, verified seed-import
option: check training-input/checkpoint hashes and configuration compatibility,
then copy into fresh isolated case directories with fresh status. Do not edit
the accepted run or inherit pending work, new labels, or promoted models.
The runner now implements this suite-level seed import, including real CPU
loads/predictions for all four initial members. The accepted seed has passed
this verification against the new profiles without changing its files.

Require at least two completed cohorts, eight selected sampled IDs traced
through converged labels and storage into real retraining, and four loadable
promoted members per case. Report actual admitted/terminal counts, candidate
counts, selected IDs/scores, pinned model generations, actual task widths,
selection latency, labeling latency, and candidate-memory measurements.
Insufficient natural acquisition remains incomplete. For identical recorded
observations, replay grouping widths 1, 2, 4, 8, 32, and 50 and reordered
completion must preserve selected IDs and deterministic ties. Independent
stochastic engine trajectories need not select identical structures.

Add targeted model-promotion coverage proving an already open cohort finishes
on its pinned generation. Keep sampling/acquisition configuration fixed during
each run; hot reload and durable recovery remain separate work. Address any
demonstrated cohort correctness bug in a focused fix. Online candidate
retention becomes a prerequisite only if measured memory prevents the bounded
`continue` campaign. Retain persistent supervision, owned-job cleanup, the
existing resource ceilings, and bounded case/group deadlines.

After that stage, proceed with:

1. A focused partial-batch and deliberate partial-label-failure campaign,
   proving valid siblings survive and counts/storage/retraining remain correct.
2. A separate nonperiodic molecular campaign for native PySEQM GPU batching
   and retraining; retain periodic water geometry in the water campaign.
3. A full TorchSim MI250 active-learning campaign, trained cell-input checkpoint
   acceptance, and longer stability work using a suitable qualified potential.
   MI350 is deferred by the user, outside the current acceptance schedule.

These are separate acceptance gates. Passing the core water matrix closes
only that gate, not all of Increment 7. Training GPU-assignment changes,
multi-node training, non-ensemble uncertainty, diversity selection, and durable
recovery keep their existing later scope.

The implemented CLI adds `--campaign core|cohort|failures|pyseqm|mi250` and
`--import-seed`; the default core command remains supported. Preparation and
single-case execution remain available. `remaining.py` supervises four campaigns
sequentially from a persistent login session, stopping if owned-job cleanup
cannot be verified. No production task signatures or coordinator code changed.

| Campaign | Cases | Maximum elapsed, including waits/cleanup |
| --- | ---: | ---: |
| Cohort/summed acquisition | 12 | 16 h; four-hour setup and engine groups |
| Controlled failures | 3, plus real three-member timeout preflight | 4 h |
| Native PySEQM GPU | 2 active-learning cases; native widths 1/2/4/8 in preflight | 4 h; bootstrap at most 1 h |
| TorchSim MI250 | 4 first-trigger dynamics/labeling combinations | 4 h |

Case limits remain 55 minutes including cleanup. Controlled failures use only
validation wrappers: one label exception, one unconverged flag after a real
converged ORCA evaluation, and one trajectory failing after an eligible frame.
Successful siblings must reach real training; invalid labels must be absent
from storage/training. A pre-failure candidate must actually survive selection
and reach labeling/training, or that case remains incomplete.

The native PySEQM campaign has its own nonperiodic water/methanol AM1 bootstrap,
32 stored structures, and four initial members, with carbon added to the model's
species. It requests only energy/forces with unit factors one. Its heterogeneous
builder uses deterministic 0.03 Å perturbations and the 0.7 Å geometry screen.
Single/native GPU comparisons require absolute differences no larger than
1e-6 eV and 1e-5 eV/Å and include a force-sign finite difference. Native fallback
calls cannot satisfy full-batch coverage. Sampling and labeling share one
serialized GPU allocation; training remains separate on four NVIDIA GPUs.

The MI250 campaign changes only sampler hardware/environment; ORCA and training
retain their established CPU/NVIDIA environments. All campaigns retain real
eight-new-label lineage and four-member training requirements. Configuration
alone, synthetic labels, and model counters do not establish acceptance.

Audits now distinguish generated candidates, returned trajectories, coordinator
selections, QM attempts/convergence, stored IDs, and training inputs. Selection
replay checks unpruned top-K and configured eligibility/ranking. Candidate-array
bytes, process lifetime peak RSS, and peak Torch allocation are separate metrics.
Trained cell-input and qualified potential-stability gates stay explicitly
open when their required inputs are absent; MI350 is deferred and does not block
the current campaign acceptance. The combined runner does
not mark all of Increment 7 complete merely because its runnable campaigns pass.

Implementation is committed locally at `1b0c813`, with observed-hardware guards
at `5dadbd8` and the sampling-observer/retry repair at `0e1dbe5`. Full CPU
regressions passed before these final guards/observer changes; subsequent
focused checks cover them and all campaign harness checks:

| Environment | Full passed | Full skipped | Final focused passed |
| --- | ---: | ---: | ---: |
| atomistic | 245 | 38 | 34 |
| TorchSim CUDA, CPU execution | 263 | 20 | 34 |
| alf_alchemi_02 | 263 | 20 | 34 |
| TorchSim ROCm, CPU execution | 263 | 20 | 34 |

The separate real PySEQM CPU suite passed all eight checks with pinned source
`31edd887`; GPU native batching and real retraining remain separate live gates.
Prepared profiles submit no jobs, and the final frozen source successfully
verified the accepted water seed with four loaded initial members.

The first executed attempt (`run-002`, `5dadbd8`) passed prediction preflight
but sampling failed because the new allocator-stat observer ran before CUDA
initialization in a fresh worker. It was stopped and all three owned Slurm
jobs were verified terminated. The fix initializes the selected GPU before
resetting its statistics; production sampling/training behavior is unchanged.

The replacement supervisor ran on `darwin-fe2` in tmux session
`alf-inc7-remaining-20261103-03`, using frozen revision `0e1dbe5` from the detached
`source-memory-fix/` worktree. All artifacts are outside Git under
`/vast/home/pjlohr/ALF_LANL/validation/inc7_remaining_20261103/`.
`run-001/` contains preparation only; `run-002/` retains the failed observer
attempt; `run-003/` has finished. The original cohort budget origin
`1793730532.3461077` is retained. First-case, ASE-group, and overall deadlines
count the earlier attempt and repair time; per-case `deadlines.json` records them.
`suite-003.log`, campaign logs/results, and `validation-summary.json` retain live
progress, CPU test evidence, and fingerprints. At startup the first ASE cohort
case passed its CUDA prediction preflight and returned four real builder boxes;
no remaining campaign gate is inferred from that startup check. Final accepted
case counts must come from each campaign's result and cleanup evidence.

Final `run-003` results: all twelve cohort/summed cases and all four TorchSim
MI250 cases passed, each with eight new sampled labels in a real promoted
four-member ensemble. All 18 cleanup reports (12 cohort cases, four MI250
cases, and the two failed-stage bootstraps) verified termination of owned jobs.
The controlled-failure stage stopped during its partial-batch preflight because
the standalone preflight did not create the sampling-output directory. Native
PySEQM GPU preflight passed widths 1/2/4/8 (maximum absolute errors about
2.3e-13 eV and 1.6e-7 eV/Å), but bootstrap training failed in HIPPYNN's automatic
atom-axis inference for three- and six-atom molecules.

The blocker repair creates the preflight output directory and adds an ALF-side
HIPPYNN database adapter. Small-molecule atom axes come from the configured
coordinate/force/charge keys; Cartesian components, cells, dipoles, and
quadrupoles are not inferred as atom axes. Larger-only datasets retain the
upstream reader path, and checkpoint restarters retain the explicit schema.
The adapter lives in the existing `hippynn_interface.py`, with deferred HIPPYNN
imports and an importable class path for checkpoint reload; it requires no
separate database module.
The external HIPPYNN checkout is not modified. Nonperiodic storage acceptance
now follows ALF's existing convention of omitting cell datasets.

Only the two affected campaigns will be retried from a new frozen snapshot;
the accepted core/cohort/MI250 evidence remains available. The user's new
blocker-fix request authorizes a new bounded attempt, with at most four hours
per failed stage including queue waits and cleanup. Earlier attempts and their
deadlines remain recorded. `remaining.py --campaigns` selects those stages;
selected-stage success is distinct from a complete four-campaign result.

The retry started at 18:25 MST on November 3 from frozen revision `e665690`
(`validation/inc7_blocker_fixes_20261103/run-001/`, `--campaigns pyseqm failures`,
accepted water seed imported). The native PySEQM preflight passed again at widths
1/2/4/8, and the small-molecule bootstrap passed the HIPPYNN atom-axis stage that
failed before: four members are training. Final results are not yet recorded;
no campaign gate is inferred from these startup checks.

Local `088aa94` (after the frozen snapshot) moves the adapter class from
`hippynn_database.py` into `hippynn_interface.py` without changing its behavior;
57 focused tests pass at the tip. Inference loads checkpoints with
`restart_db=False` and is unaffected. HIPPYNN database restarters in the retry's
training checkpoints record the old module path, so restarting training from
those checkpoints under the tip would fail to resolve the class. Before pushing,
either retrain and reload a member at the tip, or keep a re-export at the old
path.

#### Controlled-failure rerun and investigation (2026-11-04)

Harness-only repair `c7485d9` accounts for the deliberately lost label before
retraining in the label-failure cases, and retries slow Slurm cancellation/query
commands within its cleanup window. The frozen rerun is outside Git at
`/vast/home/pjlohr/ALF_LANL/validation/inc7_failures_rerun_20261104/run-001/`.
Both `label_exception` and `label_unconverged` passed, each with 11 new sampled
IDs in a trained four-member ensemble. Bootstrap and all three case cleanup
reports verified termination. Earlier Darwin/cleanup failures remain recorded;
the prior PySEQM single/native cases both passed with eight new sampled labels
and real four-member retraining.

`trajectory_failure` reached its deadline and remains incomplete. Its fixed
target failed immediately after the first eligible frame. That frame was
correctly retained in the failed trajectory's terminal report, but ranked
254th of 257 unpruned candidates in its cohort (score 3.8868; fourth-place
score 14.3623). It correctly lost global top-K selection. Model 1 trained on
eight new sampled inputs and was promoted at 29.23 minutes; all four members
completed actual epochs. Model 2 trained on 16 new sampled inputs but was not
promoted before interruption. Neither training input contains a frame from
the failed trajectory, so the required pre-failure-frame lineage is missing.
Increasing the time budget cannot change this closed cohort's selection.

Offline replay verified all four completed cohorts against unpruned candidates,
configured eligibility/ranking, widths 1/2/4/8/32/50, and reordered completion.
The observed behavior does not demonstrate a production sampler/selector bug.
The harness timing/target needs a separate focused revision before another
attempt: allow naturally ranked pre-failure candidates to accumulate, test
both winning and losing retained candidates, and retain the actual
selection/label/storage/training requirement. Do not alter scientific cutoffs,
boost scores, force selection, or count this attempt as passed. The diagnostic
and fingerprints are in the rerun artifact directory's
`investigation/trajectory_failure.{json,md}`. No new jobs were launched by the
investigation.

#### Increment 7 close-out

1. Record the retry's PySEQM and controlled-failure results and cleanup
   evidence; repair only demonstrated failures, as focused changes.
2. Document the opt-in settings in the user guide (`docs/source/user_guide`
   does not yet describe `driver`, `batching`, `acquisition`, or labeling batch
   settings), with pointers to the tracked TorchSim/ALCHEMI water configs and
   the campaign README.
3. Complete the **First current-release compatibility gate** before new
   feature implementation and the next acceptance-gated push. Build fresh
   environments, run the CPU and real workflow checks, and enable fresh-install
   CPU CI on `development`. Resolve versions at build time; the dated package
   audit above is historical evidence, not a dependency lock.
4. Record supported versions and any explicit incompatibilities from that
   gate. Existing `atomistic` and other qualified environments remain useful
   comparisons but are not the compatibility target.
   Include the example config-load test and the first existing-example
   compatibility smoke matrix (see **Existing-example compatibility**).
5. Keep the deferred gates explicitly open: trained cell-input checkpoint,
   long-time potential stability, and MI350. They do not block the next increment.

#### Focused prerequisites (2026-11-02)

- **PBC screening — merged; CPU/GPU accepted.** Legacy MLMD,
  batched ASE, TorchSim, and ALCHEMI share a minimum-image distance helper.
  Only diagonal self-pairs are excluded; distinct overlapping atoms retain
  distance zero. Fewer than two atoms yield infinity. Cutoffs, strict `<`,
  timing, and `min_time` behavior are unchanged; self-image screening is not
  added. The compatibility import `batched_mlmd.minimum_distance` remains.
  Focused geometry/legacy/batched tests: 44 passed in `atomistic`.
  ALCHEMI CPU packed-screen regressions: 2 passed, 2 TorchSim skips.
- **GPU allocation — merged; CPU/GPU accepted.** Legacy/batched
  MLMD and UDD preserve inherited CUDA/HIP/ROCr visibility and select runtime
  local ordinals by worker rank. Unmasked assignment retains the bounded
  `gpus_per_node` cap; explicit CPU skips GPU discovery and `cuda:N` pins a
  local ordinal. Legacy GPU and batched automatic CPU/GPU defaults remain.
  NeuroChem inference passes its local ordinal without changing CUDA visibility;
  its Torch-free path uses CUDA mask width or the unmasked configured count.
  Focused device/interface/legacy checks: 65 passed, 1 trained-model deselected
  in `atomistic`. Training and reactive samplers remain separate.
- **Legacy checkpoint loading — merged; CPU/GPU accepted.** Inference calls HIPPYNN's
  explicit-filename loader with absolute `experiment_structure.pt` and
  `best_checkpoint.pt` paths. Checkpoint restoration, CPU mapping, units,
  float32/device conversion, and the calculator API are preserved. The batched
  loader still uses `best_model.pt`. Focused loading/acquisition/MLMD checks:
  49 passed in `atomistic`, including simultaneous distinguishable checkpoint
  loads with `chdir` forbidden and real trained-member prediction parity.

Keep these as separate local commits. CPU suites and NVIDIA/MI250 smoke jobs
gate this prerequisite stage; the full Increment 7 campaign remains planned.
Large logs and environment/checkpoint fingerprints belong in local artifacts.

Implementation commits: `adfb238` (PBC), `d61e533` (GPU assignment), and
`2a1708a` (legacy checkpoint loading); acceptance record `24028da`. All are in
official development at `24028da`. Full CPU regressions at `2a1708a`:

| Environment | Passed | Skipped | Elapsed |
| --- | ---: | ---: | ---: |
| atomistic | 215 | 38 | 134.64 s |
| TorchSim CUDA environment, CPU execution | 233 | 20 | 184.39 s |
| alf_alchemi_02 | 233 | 20 | 177.03 s |

All skips are unavailable optional backends. Artifacts are outside Git at
`/vast/home/pjlohr/ALF_LANL/validation/inc7_prerequisites_20261102/`:
`cpu-summary.json`, `results/cpu-*.{log,xml}`, the detached `source/` worktree,
and hardware manifests containing source/package/checkpoint fingerprints.
GPU acceptance passed against that frozen `2a1708a` source. The first
hardware attempts (`17373104`/`17373105`) stopped before ALF execution because
the harness relied on unavailable Parsl launcher scripts. The harness now uses
its selected interpreter to launch Parsl modules; ALF code was unchanged.

| Device / engine | Passed | Skipped | Elapsed |
| --- | ---: | ---: | ---: |
| A100 / TorchSim (job `17373146`) | 76 | 20 | 90.26 s |
| A100 / ALCHEMI (job `17373146`) | 78 | 18 | 101.11 s |
| MI250 / TorchSim (job `17373147`) | 76 | 20 | 71.83 s |

The GPU suites include dynamics/engine-boundary conformance, raw-member
prediction parity with the approved float32/roundoff criteria, coordinator
routing, device-selection unit checks, and concurrent checkpoint-loading regressions.
Engine tests explicitly use `cuda:1`; coordinator fixtures retain their own CPU
settings. Unavailable optional engines account for the skips.

Separate real Parsl high-throughput smoke runs cover inherited, reordered
two-device, and singleton visibility for all three engine/device cases. Two
workers each execute two successive tasks per case; selected local ordinals
remain stable and inherited CUDA/HIP/ROCr variables remain unchanged. NVIDIA
subsets use allocated GPU UUIDs; MI250 subsets use ROCr device lists. Each task
runs actual legacy MLMD and two packed trajectories with the trained ensemble;
all complete. Singleton allocations correctly map both workers to `cuda:0`.

`acceptance-summary.json` combines the CPU/GPU counts, source/checkpoint
fingerprints, HIPPYNN source/local-change record, and per-worker evidence.
The runtime/benchmark tree SHA256 is
`066a5abe7f578a5ad2ff42d3a23c1099f2c6de5abfaba4bd4fefef0b1d264f1f`.
All 44 runtime/benchmark file hashes match the current development checkout.
No cutoffs, inference tolerances, or training behavior changed. These checks
close the focused prerequisites; the production active-learning campaign,
training GPU assignment, reactive-sampler allocation, and remaining backend
acceptance stay open as described above.

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

#### Increment 4 — ground-state sampling and acquisition (merged)

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
| 4a. Inference contract and scoring | merged (`d5a0395`) | Per-member unbiased energies/forces from a list of structures (adapter for the existing ASE ensemble); `Es`/`Fs`/`Fsmax`/`Fsrms` with `MLMD_calculator`'s population-std formulas; eligibility (existing thresholds, `min_time`, `distcut` rejection); uncertainty and summed ranking; stable frame identity and deterministic ties. No `__main__.py` change. | Parity with `MLMD_calculator` on fakes and on the trained Darwin water ensemble (CPU). |
| 4b. Multi-trajectory MLMD, first-trigger | merged (`d5a0395`) | Opt-in task taking `molecule_objects`; `driver.strategy` `single`/`batched`, `driver.engine` `ase`; minimal engine contract (step, set temperature, scale cell, freeze, state); per-trajectory seeds from molecule IDs; configurable device; shared batched submitter in the coordinator. `simple_mlmd_sampling_task` untouched. | Identical frames/metadata for single vs batched from seeded inputs; real coordinator with mocks; real-coordinator CPU run with the trained water model. |
| 4c. In-memory cohort top-K | merged (`86f62ca`) | `acquisition.selection = {mode: cohort_top_k, cohort_size, k}` and `trajectory_policy` `stop`/`continue`; incremental admission; model generation pinned per cohort; terminal outcomes per trajectory; exact global top-K with safe local pruning; handoff to labeling. No persistence. | Replay at widths 1, 2, 8, 32, 50 with reversed completion gives identical selections; ties, empty trajectories, < K candidates, cohort > capacity, failure after an eligible frame, model promotion mid-cohort. |

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

4b record (done locally, 2026-11-02):

- New `alframework/samplers/batched_mlmd.py`. `mlmd_trajectories_task(molecule_objects, sampler_config,
  model_path, current_model_id, gpus_per_node)` runs several independent ASE Langevin trajectories in one
  task and returns one `MoleculesObject` per input, in order.
  - `atoms` is the selected frame when the trajectory is eligible, `None` otherwise.
  - It reuses the flat `simple_mlmd_sampling_task` keys and adds the optional `driver`, `batching`,
    `acquisition`, `random_seed`, and `device`.
  - Neurochem (`use_potential_specific_code`) is rejected.
- Engine contract (`ASEEngine`): `step(indices, n)`, `observe(i)`, `atoms(i)`, `set_temperature`,
  `scale_cell`, `finish(i)`.
  - `run_mlmd_trajectories` is the lockstep loop. It reproduces `mlmd_sampling`: initial step, a check
    every `Ncheck`, schedules from `annealing_schedule`, and density scaling.
  - Each trajectory stops independently at its first eligible frame, or on a `distcut` rejection.
    Stopped members are never stepped again while batch-mates continue.
  - A trajectory whose evaluation raises is marked `failed` (`metadata['sampling_error']`) without
    affecting the others.
  - Screening uses unbiased per-member predictions read from `MLMD_calculator`'s contributions, so it
    excludes UDD and the well potential. UDD remains `MLMD_calculator_options.udd_bias_weight`.
- Metadata keeps every legacy `meta_dict` key and the `meta_dir` pickle/`.traj` files. It adds `Fsrms`,
  `trajectory_outcome` (`eligible`/`rejected`/`completed`/`failed`), `acquisition_strategy`,
  `ranking_score` (eligible only), `step`, `frame_id`, and `random_seed`.
- Seeds: with `random_seed`, each trajectory's `RandomState` is derived from SHA-256 of the seed and the
  molecule ID. It drives schedule draws, the `trajectory_frequency` coin flip, and the Langevin noise.
  Without it, OS entropy is used and recorded. `device` defaults to `cuda:0` when available, else `cpu`.
- Coordinator: Increment 2's logic was factored into `batching.BatchedSubmitter`. `LabelingCoordinator`
  now subclasses it, and its tests pass unmodified.
  - New `tools/sampler_coordinator.py` (`SamplerCoordinator`, `driver_settings`). Legacy
    `molecule_object` samplers keep an ordinary queue and one task per structure; a `driver` or
    `batching` block with a legacy task is an error.
  - `molecule_objects` samplers use a structure-counting `BatchedTaskQueue`: `single` gives width-1
    tasks; `batched` uses the sampler's `batching` block. `torchsim`/`alchemi` raise
    `NotImplementedError` (Increments 5/6).
  - `__main__.py` changes are limited to the three sampler submission sites (active learning and
    `--test_sampler`), one `sampling.submit_ready` per pass, and the sleep
    `labeling.poll_delay(sampling.poll_delay(60))`. Net +11/−17 lines.
- Evidence:
  - `tests/test_batched_mlmd.py` (14 tests) uses a new bounded fake ensemble
    (`tests/helpers/fakes.PairSpringCalculator`; spring constants differ per member, so uncertainty grows
    with thermal motion).
    - Legacy parity: six cases (trigger, trigger at start, never, `distcut`, `min_time`, periodic
      density schedule) reproduce `mlmd_sampling` exactly with the same seeded stream: frame,
      `Es`/`Fs`/`Fsmax`, crit flags, `temps`/`denss`, positions, cell.
    - Width/order invariance: eight trajectories stopping at steps 11–76, one `distcut` rejection, and
      one that never triggers give identical results as 8×1, 1×8, 3+5, and reversed.
    - Also covered: stopped members are not stepped while batch-mates run; failure isolation; UDD moves
      atoms while the recorded `Es` stays unbiased; seeds; metadata/`.traj` files; device selection.
    - On the trained Darwin ensemble (CPU), `single` and `batched` agree exactly.
  - `tests/test_sampler_coordinator.py` (5 tests) runs the real entrypoint with the real
    `mlmd_trajectories_task`, a fake ensemble loader, and mock builder/QM/ML. `single` and `batched`
    (width 3) both complete bootstrap → MLMD → labeling → HDF5 → retraining, with sampler structures
    within `parallel_samplers`. Stable over 10 repeats per env. Full suite: 121 passed in both envs.
  - Trained model, real coordinator (CPU, `cn211`, scratch `/tmp/alf_inc4b`): active learning resumed
    from a copy of the Darwin `model-0000`/`data-0000.h5`.
    - Setup: real condensed-phase builder (30–45 atom water boxes), batched `mlmd_trajectories_task`
      (width 3, `device: cpu`), and mock QM.
    - Result: 9/9 harvested trajectories were eligible and all were labeled. Every one triggered at
      the first check (Es 3.7–7.8 eV vs `Escut` 0.5). That is expected for the 100-structure smoke-test
      model, so this run checks execution, not sampling quality.
- Findings from that run (not 4b defects, recorded for follow-up):
  - `distcut` defaults to 1.2 Å in both `mlmd_sampling` and the new task. The tracked water examples
    omit `distcut`, so dense water (O–H ≈ 0.96 Å) is rejected at the first check under either sampler.
    `simple_water_darwin_run_test` avoided this only because its `maxt` ends sampling before any check
    triggers. The scratch run set `distcut: 0.7`.
    - Resolved 2026-11-02 at the user's direction. The four tracked water examples (`simple_water`,
      `simple_water_multi_builder`, `fine_tuning`, `seeded_active_learning`) now set `"distcut":0.7`,
      matching the existing Darwin fine-tuning example.
    - The code default (1.2 Å) is unchanged, so other examples and user configs behave as before.
    - The stored Darwin water bootstrap frames have minimum interatomic distances of 0.84–0.90 Å:
      every frame is below 1.2 and none is below 0.7.
    - Re-running the trained-model scratch loop with `simple_water`'s unmodified sampler settings (plus
      only batching/`maxt`/`device`) gave 6/6 trajectories eligible (force triggers) and labeled, with
      none rejected and no failed tasks. Before the change, every trajectory was rejected at the first
      check.
  - `HIPNN_ASE_calculator` loads checkpoints inside `hippynn.tools.active_directory` (process-wide
    `os.chdir`) with a relative model path. Two sampler tasks loading concurrently in one process
    (`ThreadPoolExecutor`) therefore race. One task failed this way in the scratch run. This affects the
    legacy sampler equally; HTEX workers are separate processes.
    - Pre-existing; not introduced by batching (checked 2026-11-02).
      - The loader is unchanged on `origin/main` (`206ea44`) and dates to the 2023 initial commits; no
        batching commit touches `hippynn_interface.py`.
      - `simple_mlmd_sampling_task` on main also loads the ensemble inside every task.
        `mlmd_trajectories_task` loads once per batch, so it loads less often.
      - Every shipped Parsl config (`alframework/parsl_resource_configs/*`, `examples/*/parsl_configs.py`)
        uses `HighThroughputExecutor`, so shipped workflows do not trigger it. Only thread-based or other
        single-process setups with overlapping loads do.
    - Open follow-up, deferred by the user. Make it a separate small commit, outside the batching
      increments: load via absolute paths with `load_checkpoint` instead of changing directory, or
      serialize the load with a lock.
    - Increment 5's `load_hippynn_batched_ensemble` already loads from absolute paths, so it is not
      affected; `HIPNN_ASE_calculator` (the `ase_calculator` path) still is.
- Limitations: a stopped trajectory's frame reaches labeling when its task finishes, i.e. when its
  slowest batch-mate stops. Results are identical; only latency differs. Batches use the model current
  at submission (cohort pinning is 4c). Settings are read once at startup. Only the ASE engine exists;
  inference is still one structure at a time.

4c record (done locally, 2026-11-02, commit `a74d97b`):

- Decisions (user, 2026-11-02):
  - `trajectory_policy: continue` keeps every flagged frame, with an optional
    `max_candidates_per_trajectory` cap that is unset by default.
  - A cohort is pinned to the model generation current when it opened and finishes on that model.
  - Scoring was made extensible: a named term table instead of hardcoded formulas.
- Configuration (sampler `acquisition` block; flat physics keys unchanged):
  - `selection: {mode: first_trigger | cohort_top_k, cohort_size, k, max_candidates_per_trajectory}`.
    `first_trigger` (the default) is exactly 4b. `cohort_size` and `k` are required for `cohort_top_k`.
  - `trajectory_policy: stop | continue`. `continue` requires `cohort_top_k`.
  - Unknown keys, including the fork's task-local `return_top_k`, are errors rather than reinterpreted.
  - Legacy `molecule_object` samplers with a `selection` block are rejected.
- Scoring (`acquisition.SCORE_TERMS`): `energy` = Es/Escut, `force_mean` = Fs/Fscut,
  `force_max` = Fsmax/(3·Fscut), `force_rms` = Fsrms/Fscut.
  - `uncertainty` takes the max of the three trigger terms; `summed` is a weighted sum of configured terms
    (`weights: {term: w}`). `w_energy`/`w_force` remain shorthand for `energy`/`force_rms`.
  - Eligibility for every strategy is still main's threshold test.
  - In first-trigger mode the strategy only sets the `ranking_score` metadata; returned frames are
    unchanged.
  - Adding a term (e.g. a future excited-state gap uncertainty) is one table entry naming its signal
    and normalizing cutoff. Scoring raises if inference does not provide that signal. No change is needed
    in pools or the coordinator.
- Sampler (`batched_mlmd.run_mlmd_trajectories`):
  - In cohort mode each eligible frame becomes a candidate holding positions, cell, PBC, numbers,
    signals, flags, score and `frame_id`. `stop` ends the trajectory there; `continue` keeps
    sampling.
  - Candidates collected before a later `distcut` rejection or failure are kept.
  - Before returning, each trajectory keeps at most its best `k` candidates.
    An explicit cap below `k` truncates that trajectory's contribution and can
    change the unrestricted global top-K. Without such a cap, local top-K
    pruning preserves the final scalar ranking. This is currently return-time
    pruning, not a bound on candidate memory during the trajectory.
  - Returned molecules have `atoms=None` and carry `acquisition_candidates`, so nothing is labeled
    before the cohort selects.
- Pool (new `tools/cohorts.py`, `CohortPool`):
  - Admission is incremental (`cohort-<model>-<n>` IDs, per-member `trajectory_ordinal`), with no
    admission timeout and no partial-cohort selection.
  - A cohort finalizes when it is full and every member is terminal (`eligible`, `completed`,
    `rejected`, `failed`). Selection is `top_k` (score, then trajectory ID, then step). Selected frames
    become `MoleculesObject`s named `<trajectory_id>-s<step>`, with cohort, rank, score and signal
    metadata.
  - Finished-cohort summaries record outcomes, candidate counts and selected frame IDs.
- Model promotion: the open cohort keeps admitting until full and its members run on its pinned model;
  the next cohort opens on the then-current model. A cohort never mixes generations, and sampler
  batches never mix models (the buffer key includes the pinned model, and each task receives the
  group's `current_model_id`). Without cohorts, multi-trajectory tasks use the model current at
  submission, as before.
- Coordinator:
  - `SamplerCoordinator.submit` admits each structure and pins its model. `harvest` records results
    and returns each finished cohort's selection; without cohorts it returns results with `atoms` set,
    which is the prior logic.
  - `BatchedTaskQueue` now records each task's inputs, so members of a whole-task failure are
    finalized as `failed` and their cohort can still complete.
  - `__main__.py`: the harvest loop calls `sampling.harvest(...)`, and cohort status is printed each pass
    (+5/−3 lines).
- Evidence:
  - `tests/test_cohorts.py` (9 tests), with 50 real spring-ensemble trajectories per case under `stop`,
    `continue`, and `continue` with cap 2, including `distcut` rejections and low-spread members.
    - Selected frame IDs and ranks are identical at task widths 1, 2, 8, 32, and 50 in forward,
      reversed, and shuffled completion order.
    - The selection equals the global top-K over all candidates; worker pruning gives the same
      selection as no pruning.
    - Also covered: ties, fewer than K candidates, empty cohorts, and duplicate/unknown reports. A
      failure after eligible frames keeps them; whole-task failure members are finalized. A cohort of 10
      completes at capacity 3 with exactly one selection, and promotion is pinned as above.
  - `tests/test_acquisition.py` adds explicit-term parity with the `w_*` shorthand, a registered dummy
    `gap` term (including the missing-signal error), and 11 new validation cases. The 4a/4b expectations
    are unchanged.
  - The real-entrypoint coordinator test gains `cohort-stop` and `cohort-continue` cases (`cohort_size`
    6, `k` 2, `parallel_samplers` 4). Both complete bootstrap → MLMD → selection → labeling → retraining.
    Only selected `-s<step>` frames are labeled, at most 2 per `cohort_id`. Stable over 6 repeats per
    env.
  - Full suite: 143 passed in both envs (with PySEQM on the path).
- Limitations:
  - Cohorts and candidates are in memory only; unfinished cohorts are lost on restart.
  - Candidates of a trajectory whose task fails as a whole are lost, since they never reached the
    coordinator; frames held only inside a crashed worker are out of scope.
  - Candidate payloads include full positions, so very large cohorts with `continue` and no cap hold
    `cohort_size × min(k, cap)` frames in coordinator memory.
  - Selection is scalar top-K only; no diversity filter.

#### Increment 5 — TorchSim dynamics (done locally, 2026-11-02, commit `a80c23e`)

Decisions (user, 2026-11-02):

- Per-trajectory Langevin noise. BAOAB is built from TorchSim's `momentum_step`/`position_step`, with an
  O-step drawing each trajectory's noise from its own stream (the 4b seed derived from its molecule ID).
  TorchSim's stock `nvt_langevin_step` uses one generator per batch, so a trajectory's noise would depend
  on its batch-mates. Finished trajectories are removed from the packed state, not frozen.
- Batched HIPPYNN inference lives in the existing `ml_interfaces/hippynn_interface.py`, beside
  `HIPNN_ASE_load_ensemble`.

Implementation:

- `driver.engine: torchsim` is accepted (ALCHEMI is added in Increment 6 below). New
  `samplers/torchsim_engine.py` (`TorchSimEngine`) implements the `ASEEngine` contract, so
  `run_mlmd_trajectories`, acquisition, cohorts, and the coordinator are unchanged. TorchSim is imported
  only when this engine is selected.
  - One packed `MDState` per task; one batched ensemble evaluation per MD step; float64 integration state.
  - ASE units (Å, eV, amu, ASE time; `dt` in fs). Friction 0.02 inverse ASE time, as `mlmd_sampling`.
    As in ASE `Langevin(fixcm=True)`, the noise carries no net momentum and is rescaled by `sqrt(N/(N-1))`.
  - Propagation force = member mean + optional well potential + UDD, with `MLMD_calculator`'s formulas;
    screening uses the unbiased per-member predictions.
  - `scale_cell` scales one system's cell and positions and refreshes forces before the next step.
  - A system with non-finite predictions fails alone; an exception in the batched call fails the stepped
    members. Trajectory files (`trajectory_frequency`) are written from the engine.
- Inference: `batched_calculator` (with `batched_calculator_options`) selects a batched evaluator,
  `evaluate(numbers, positions, cells, pbc, counts) -> (energies [models, systems], forces [models, atoms, 3])`.
  Without it, the TorchSim engine evaluates the `ase_calculator` ensemble one structure at a time.
  `batched_calculator` with the ASE engine is a configuration error.
- `hippynn_interface.load_hippynn_batched_ensemble`:
  - loads members from absolute paths (no `active_directory` chdir, so it avoids the 4b loader race);
  - evaluates each member on the padded batch in `model-*` order. `hippynn.graphs.make_ensemble` was not
    used because it does not preserve member order (found when per-member energies disagreed);
  - periodic systems with cell-less checkpoints (all tracked water examples) get HIPPYNN periodic pair
    finders in place of the open ones, as HIPPYNN's ASE calculator does; cell-trained checkpoints use their
    own cell input. Mixed periodicity is rejected;
  - selects portable HIPPYNN kernels under ROCm.
- `setup.py` adds a `torchsim` extra (`torch-sim-atomistic>=0.6.2`). Opt-in example
  `examples/simple_water/mlmd_config_torchsim.json`; acceptance/throughput harness
  `examples/torchsim_benchmark/benchmark.py`.
- Test environment: the fork's TorchSim 0.6.2 venvs (`ALF_fork/ALF/debug/portable_gpu_20261027_8795a40/envs/
  torchsim-{cuda,rocm}`, layered on `atomistic`; Torch 2.9.0+cu128 / +rocm6.4).

CPU evidence (`cn213`):

- `tests/test_dynamics_engines.py` (10 tests; skipped where `torch_sim` is absent):
  - zero friction and temperature reproduce ASE velocity Verlet (6e-16 Å after 40 steps);
  - results identical alone, together, split 3+4, and reversed, in first-trigger and cohort `continue`
    mode (spring ensemble, mixed sizes, a `distcut` rejection);
  - well + UDD forces and energy equal `MLMD_calculator`'s; Langevin holds 300 K within 3%;
    density schedules scale cells, also identically when batched; a NaN member fails alone;
  - trained Darwin water ensemble (8 members): batched HIPPYNN equals the per-structure ASE calculators
    within `rtol=atol=1e-5` for molecules and 24–36-atom periodic boxes; every collected frame's
    `Es`/`Fs`/`Fsmax` from a TorchSim run equals the ASE ensemble's at that geometry.
- Real-entrypoint coordinator test gains `torchsim` and `torchsim-cohort-continue` cases; both complete
  bootstrap → MLMD → (selection) → labeling → retraining. Stable over 5 repeats.
- Full suite: 155 passed in the TorchSim venv; 143 passed, 3 skipped (the TorchSim test module and the two
  TorchSim coordinator cases) in `atomistic` and `alf_env` (PySEQM on the path).

GPU acceptance (Darwin, 2026-11-02; source frozen at `ALF_fork/ALF/debug/torchsim_inc5_20261102/source`,
base `a74d97b` plus the Increment 5 working tree later committed as `a80c23e`; artifacts in its `results/`):

- Environments: the fork's TorchSim 0.6.2 venvs; Torch `2.9.0+cu128` (CUDA 12.8) on NVIDIA and
  `2.9.0+rocm6.4` (HIP 6.4) on AMD; HIPPYNN portable kernels (`HIPPYNN_USE_CUSTOM_KERNELS=False`).
- A100-SXM4-40GB, `shared-redstone` cn1, job 17372775: harness passed at all widths;
  `tests/test_dynamics_engines.py` + `tests/test_sampler_coordinator.py` 19 passed (including the trained
  ensemble and real-coordinator TorchSim cases). (Job 17372745 on `shared-gpu-ampere` was cancelled
  unrun: per-user QOS limit.)
- MI250X/MI250, `shared-gpu-amd-mi250` cn782: harness passed at all widths (job 17372746). Its test run
  skipped the 3 trained-model tests because the untracked model was not in the frozen copy; rerun with it
  linked, job 17372771: 19 passed.
- Harness (`examples/torchsim_benchmark/benchmark.py`): the trained 8-member Darwin water ensemble on
  periodic water boxes from the condensed-phase builder (27–45 atoms); batched vs per-structure predictions
  on the same device, `rtol=atol=1e-5`; then timed inference (20 calls) and TorchSim dynamics (50 steps,
  0.25 fs, 300 K) after 5 warm-up steps. Energy differences up to 0.01 eV are float32 roundoff on
  ~−16,600 eV totals (relative < 1e-6). Single short runs, no repeat confidence intervals.

| GPU | Systems | Inference systems/s | Dynamics system-steps/s | Peak MiB | Max ΔE (eV) | Max ΔF (eV/Å) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100-SXM4-40GB | 1 | 11.6 | 11.4 | 25 | 0.0059 | 2.9e-06 |
| A100-SXM4-40GB | 2 | 22.8 | 22.4 | 33 | 0.0059 | 3.8e-06 |
| A100-SXM4-40GB | 8 | 89.8 | 88.0 | 71 | 0.0059 | 5.2e-06 |
| A100-SXM4-40GB | 32 | 308.9 | 290.7 | 234 | 0.0059 | 5.7e-06 |
| A100-SXM4-40GB | 50 | 406.7 | 384.0 | 368 | 0.0059 | 7.6e-06 |
| MI250X/MI250 | 1 | 15.6 | 15.7 | 160 | 0 | 1.9e-06 |
| MI250X/MI250 | 2 | 31.5 | 31.1 | 167 | 0.0039 | 3.8e-06 |
| MI250X/MI250 | 8 | 123.9 | 122.1 | 207 | 0.0078 | 6.2e-06 |
| MI250X/MI250 | 32 | 329.2 | 310.8 | 368 | 0.0078 | 7.6e-06 |
| MI250X/MI250 | 50 | 397.5 | 381.5 | 503 | 0.0098 | 8.6e-06 |

Batching raises throughput ~34x (A100) and ~24x (MI250X) from width 1 to 50 for this small model.
Remaining: MI350-family hardware; a trained periodic (cell-input) checkpoint; longer stability runs; an
end-to-end Slurm active-learning run with `engine: torchsim` (combined validation, Increment 7).

Limitations: one shared PBC per batch; NVT Langevin only (no NPT); trajectories in one task advance in
lockstep, so a stopped frame reaches labeling when its slowest batch-mate stops (as in 4b); GPU results are
not bitwise reproducible across devices; inference throughput depends on HIPPYNN's padded evaluation.

#### Increment 6 — ALCHEMI 0.2 (merged at `be7fb64`, 2026-11-02)

Decisions (user, 2026-11-02):

- Target toolkit 0.2.0. Clone `tmn_alchemi` into `alf_alchemi_02`, preserving
  its Python 3.11, Torch 2.13.0+cu130, ops 0.5.0, and Warp 1.18.0 stack.
  Add HIPPYNN, Parsl, ALF, and test dependencies only in the clone. Keep the
  original environment and the validated TorchSim environments unchanged.
- Reuse the current molecule-ID-derived NumPy streams and center-of-mass
  thermostat correction. ALCHEMI's stock global-seed Langevin kernel is not
  used: its random draws depend on packed atom indices.
- Use ALCHEMI deterministic velocity updates with adapter-owned BAOAB drifts
  and the shared stochastic step. Preserve `dt` in fs and friction 0.02 in
  inverse ASE time, initial velocities, schedules, and current UDD/well options.
- Extract shared evaluation/force/noise helpers without importing either
  engine's optional dependencies. Reuse the existing batched HIPPYNN loader,
  acquisition, cohorts, and coordinator; no `__main__.py` changes are expected.
- Extend the existing benchmark with `--engine torchsim|alchemi` (default
  `torchsim`). Record the executing development revision and source hashes,
  not an enclosing repository's revision when source is copied elsewhere.
- Validate CPU checkpoint inference before engine development, then CPU
  conformance and full suites. NVIDIA acceptance first launches Torch and Warp
  kernels to check compatibility with the cloned CUDA stack, then tests widths
  1, 2, 8, 32, and 50 with the trained water ensemble. Re-run TorchSim CUDA/ROCm
  regressions after extracting shared helpers. Keep `rtol=atol=1e-5`.
- Full Slurm active learning, a trained cell-input checkpoint, and MI350 are
  later acceptance work. The old ASE-loader `chdir` race and fork-roadmap sync
  remain separate changes. Agents commit locally; the user pushes.
- MI250 float32 uncertainty acceptance (user, 2026-11-02): accept the observed
  small spread difference when physical member predictions pass `rtol=atol=1e-5`.
  Verify candidate signals against their actual member predictions and bound
  differences from individual evaluation by propagated, measured member roundoff.
  Do not alter inference precision or acquisition cutoffs. The initial direct
  spread comparison failed; retain that result and its diagnostic in the report.

Implementation: `5001880` (shared packed calculations and ALCHEMI adapter) and
`fb88905` (benchmark, source identity, and opt-in example). `__main__.py`,
acquisition, cohorts, labeling, and the HIPPYNN loader are unchanged.

- The clone is complete; toolkit 0.2.0, ops 0.5.0, Warp 1.18.0, and Torch
  2.13.0+cu130 retain their original versions. Four uncached Conda runtime
  packages were fetched using the source environment's exact package manifest.
  Added Parsl 2025.10.27, pytest 9.0.1, and HIPPYNN from revision `a66eba61`
  (package metadata 0.1.3.post0.dev15).
- CPU checkpoint compatibility: all eight trained water members load; mixed
  3/6-atom predictions agree with individual ASE evaluations within
  `rtol=atol=1e-5` (max energy error 0, max force error 1.43e-6 eV/Angstrom).
- Original-stack GPU compatibility probe: A100-SXM4-40GB, cn1, driver
  595.71.05, job 17372861. Both Torch and native ALCHEMI/Warp kernels passed.
  This was a compatibility probe, not final implementation acceptance.
- Full CPU suites: 166 passed / 16 skipped in `alf_alchemi_02`, 164 / 18 in
  the TorchSim environment, and 148 / 34 in `atomistic`, with PySEQM on the
  path. Optional engines account for differing skips.
- Shared conformance now covers both engines, both ranking strategies,
  initial velocities/kinetic energy/triclinic cells, temperature changes,
  earlier candidates surviving a later failure, and dependency isolation.
  `ALF_TEST_ENGINE_DEVICE=cuda:0` explicitly selects GPU test execution.
- Shared tests were renamed to `tests/test_dynamics_engines.py` in `be7fb64`;
  `tests/test_engine_boundaries.py` imports its shared fixtures. Test contents
  were unchanged and all 182 tests collected in both qualified environments.
  Frozen earlier acceptance logs retain the old `test_torchsim_engine.py` name.
- The cloned stack contains another package named `tests`. ALF's tests now
  form an explicit local package to avoid that collision and remain excluded
  from the distribution wheel.

- NVIDIA A100 acceptance: job 17372891 passed benchmarks at widths 1, 2,
  8, 32, and 50 for both engines. Final conformance job 17372945 passed
  27 ALCHEMI / 16 skipped and 25 TorchSim / 18 skipped, including the
  trained-model check and real coordinator cases.
- Darwin MI250 TorchSim regression: job 17372892 passed all benchmark widths;
  its direct spread comparison failed. Diagnostic 17372937 established
  float32 member roundoff; after the user-approved test refinement in
  `d6743a7`, job 17372946 passed 25 / 18 skipped. The original failure remains
  in the artifacts. Inference tolerances and acquisition cutoffs are unchanged.
- Benchmarks ran frozen commit `b63e3df`; final GPU tests ran `d6743a7`.
  Imported runtime/benchmark source hashes match. Width-50 dynamics throughput:
  376.05 system-steps/s (A100 ALCHEMI), 382.48 (A100 TorchSim), and 381.47
  (MI250 TorchSim). Memory reports are peak PyTorch allocations, excluding
  possible Warp allocations. Thermostat means passed the 3% criterion.
- Local validation artifacts (`alchemi_increment6_validation_20261102.md` and
  `alchemi_02_environment_20261102.json`, kept outside Git in the artifact
  directory below) record exact versions, HIPPYNN revision and pre-existing
  source modifications, checkpoint fingerprints, executed revisions, test results,
  throughput/memory tables,
  the uncertainty decision, and known limitations.

Artifacts: `ALF_fork/ALF/debug/alchemi_inc6_20261102/`. Implementation and
CPU/GPU engine acceptance are merged in official development at `be7fb64`.
The dated environment/report files remain local artifacts, excluded from Git.
This is not production-potential accuracy validation. Full Darwin active
learning, a trained cell-input checkpoint, and MI350 remain Increment 7 or later.

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
  `driver.engine`. Keep the flat physics/calculator settings and physical
  grouping settings in `batching`; do not adopt the fork's `sampling`/`inference`
  configuration dialect (Increment 4 decision).
- Support mixed-size ground-state systems, initial velocities, independent
  stopping, temperature/density schedules, and supported periodic checkpoints.
- Inference supplies physical predictions; sampling owns scientific checks;
  acquisition owns eligibility/ranking/selection; engines own integration,
  units, and synchronization.
- Preserve main's `distcut` geometry screen and unbiased ensemble uncertainty.
  Defer the optional physical-force screen (`max_force_cutoff`) until GALF
  (Increment 4 decision). When introduced, it must exclude artificial bias and
  confinement forces rather than change propagation.
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
- First-trigger mode stops at one eligible frame per trajectory. Its result
  reaches the coordinator when the whole sampler task returns, so a slower
  batch-mate can delay labeling. Pooled mode requires explicit cohort size and K.
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

These intended development capabilities remain on the roadmap. Row numbers
are stable identifiers; the working order is set in **Decided priority order**
below. Independent initialization/replay contributions may proceed
once their helper and identity requirements are available. These are future
capabilities, not features already supplied by Increment 6 or promises that every
existing sampler can already run through the new engines.

| Increment | Capabilities to integrate | Important boundary |
| --- | --- | --- |
| 1. Composable sampling | Separate propagation from UDD, confinement, and other bias/force controls; support callable extensions and grouped non-MD evaluations. | Preserve physical predictions, formulas, switching order, and replica-local state. Folded into the batched UDD/gap ALF step (see **Decided priority order**). |
| 2. Additional acquisition strategies | Custom import-string strategies and geometry-based demonstrations, followed by production distance-based acquisition and diversity-aware batch selection within cohorts (and later against the existing training set). | Eligibility, ranking, trajectory policy, and selection remain separate; future selectors declare their retention needs. The selector/retention interface follows E5 (see **Decided priority order**). |
| 3. Model/training handoffs and distillation | Opaque model-artifact references, model-independent inference adapters, and foundation-MLIP reference labeling. | Record teacher identity, version, units, and label provenance separately from the student sampling model. Teacher labeling (3a) is a parallel track after Increment 7 (see **Decided priority order**); handoffs and distillation (3b) remain later. Fine-tuning tiers F1–F3 are planned in **Later steps**. |
| 4. Molecular excited states | PySEQM/GPU4PySCF labeling, multi-state HIPPYNN training, derived gap targets, state selection, gap diagnostics, and LCM. | Preserve per-state convergence, state/atom mappings, physical screening, and the distinction between predictions and propagation controls. |
| 5. RDKit initialization | Reference-centered KDG geometry generation, seed-specific screening, and FPS selection, with reproducible settings/provenance. | Optional preprocessing/builders; unlabeled seed geometries are not labeled training rows. |
| 6. Geometry-only HDF5 replay | Reuse stored geometries for sampling or relabeling, with original identities and source provenance. | Do not accidentally carry old labels into a newly requested labeling operation. |
| 7. Optional additional labels | Preserve optional nonadiabatic-coupling labeling and storage, including state-pair and atom-axis mappings. | Computing/storing NACs does not automatically make them energy/force training targets. |
| 8. Durable recovery and staged workflows | Persist batching/cohort handoffs; add retained-label budgets, ALF-to-GALF/configuration stages, exact dataset checkpoints, and endpoint training. | Preserve ordinary runs as the default; budgets count retained labeled frames, not candidates or submitted jobs. |
| 9. Further model and sampling extensions | Additional production model families, reactive batching, and separately planned methods. | Follow the same minimal-core-change and existing-workflow compatibility gates. |
| 10. Multi-node ML training | Train ensemble members across nodes, then optionally data-parallel training of one member across GPUs/nodes. | Keep the `ML_task` entrypoint, `(completed, training_id)` return, member directory layout, and model-promotion semantics; single-node training remains the default. Parallel track, independent of the excited-state sequence (see **Decided priority order**). |
| 11. Non-ensemble uncertainty | Pluggable uncertainty estimators, including single-network methods, alongside the existing ensemble disagreement. | New signals carry their own names, units, and cutoffs; `Es`/`Fs`/`Fsmax` and `Escut`/`Fscut` keep their ensemble-std meaning. First planned use: foundation-model fine-tuning (F3). |

Items 10 and 11 and the diversity part of item 2 were added at the user's
request on November 2, 2026; their order relative to the other rows is not yet
decided. Readiness notes from a review of development at `24028da`:

- **Multi-node training (10).** `train_HIPPYNN_ensemble_task` is one Parsl
  app on `alf_ML_executor` that runs `multiprocessing.Pool(gpus_per_node)` on
  one node, one member per GPU. Each member sets `CUDA_VISIBLE_DEVICES` itself,
  which is the "training pools" GPU-visibility follow-up above. The tracked
  Darwin/Chicoma configs give the ML executor one node per block. `__main__.py`
  waits for a single ML task (`ML_task_queue.get_number() < 1`) and promotes the
  model after it returns.
  - The first step is to run members across nodes: they train independently,
    so this is task placement, not distributed numerics. A Parsl `join_app`
    that fans out one app per member (or per node) and returns the existing
    `(completed, training_id)` keeps `__main__.py` unchanged. A multi-node
    block with `srun` placement is the alternative. Both need the
    local-ordinal GPU assignment already used by the samplers, and the
    per-member `completed` handling for partial failures must be kept.
  - A later step is data-parallel training of one member, for datasets or
    models too large for one GPU. HIPPYNN's Lightning trainer
    (`hippynn.experiment.lightning_trainer`) is marked experimental and lacks
    some features. Adopting it changes checkpoint layout and the batched
    loader's `best_model.pt` assumption, so it is a separate decision.
- **Non-ensemble uncertainty (11).** This is partly ready.
  - Ready: acquisition reads a signal dictionary, and ranking terms are a
    table (`SCORE_TERMS`). Packed engines keep physical predictions separate
    from propagation.
  - Ensemble assumptions to factor out:
    - both evaluators require at least two members, and the evaluator contract
      returns per-member `[models, ...]` arrays;
    - `run_mlmd_trajectories` calls `ensemble_statistics` directly;
    - eligibility (`uncertainty_flags`) hard-codes `Es`/`Fs`/`Fsmax` against
      `Escut`/`Fscut`/`3*Fscut`;
    - propagation uses the member mean;
    - UDD needs member deviations;
    - legacy samplers read `MLMD_calculator`'s `energy_stdev` results;
    - training assumes `n_models` member directories.
  - Proposed contract: the evaluator returns a mean prediction plus named
    uncertainty signals, with ensemble statistics as the default estimator.
    Eligibility then reads configurable signal/cutoff pairs. Methods that need
    repeated stochastic passes (for example MC dropout) can still use the
    member layout.
  - A single-network estimator may provide only energy uncertainty or no
    per-atom force uncertainty, so eligibility must allow a subset of signals.
  - Calibration and cutoffs need per-method validation; ensemble cutoffs must
    not be reused.
  - UDD-style biasing needs a differentiable uncertainty, which ties this to
    composable sampling (item 1).
- **Diversity-aware acquisition (2).** The cohort pool is the natural place
  for it.
  - Ready: `CohortPool.ready()` selects with `top_k(cohort.candidates, k)`, and
    candidates already carry numbers, positions, cell, PBC, and signals. A
    selector interface, such as score-weighted farthest-point or k-center
    selection on geometric descriptors, can replace that call.
  - Constraints:
    - `_prune` and `max_candidates_per_trajectory` drop candidates by score
      inside the worker before the pool sees them, so a diversity selector must
      declare its own retention (as noted under Known implementation
      follow-ups).
    - First-trigger mode has no pool.
    - Selection must stay invariant to grouping and completion order:
      canonicalize candidates by `(trajectory_id, step)` first and break ties
      deterministically.
    - Model-latent descriptors would require evaluators to return features
      and candidates to carry them, which costs memory.
    - Diversity against the existing training set needs read access to the
      HDF5 store; it is a separate step from within-cohort diversity.
  - The `acquisition.selection` keys are validated against a fixed set, so new
    selectors need a `mode` or import-string extension.

### Decided priority order (2026-11-03)

After Increment 7 closes, the user set this order. Each step will take many
small, separately reviewable changes over time.

1. **Excited-state ALF** (row 4, pulling forward the parts of rows 6 and 7 that
   it needs). The target is the excited-state work on the fork's
   `production_excited_states`. The headline example is the
   cycloparaphenylene ring (cpp) run as a full ALF active-learning example.
   Steps are E1–E6 below.
2. **Composable sampling with batched UDD and gap ALF** (row 1 and the
   sampling half of row 4). The bias/control interface comes first. UDD,
   wells, LCM gap seeking, and the deferred physical-force screen are built on
   it, with gap-aware acquisition.
3. **Staged campaigns across engines** (row 8): durable handoffs, staged
   configurations, and budgets.

Two further items sit outside that sequence:

- **Multi-node training (row 10), a parallel track.** It is independent of
  steps 1–3 and can start whenever there is capacity. The first step places
  members across nodes. Data-parallel training of one member is a later,
  separate decision.
- **Acquisition selector/retention interface (first part of row 2), after
  E5.** A pluggable cohort selector that declares its candidate-retention
  needs replaces the hard-wired `top_k` in `CohortPool.ready()` and the
  score-only `_prune` in `batched_mlmd.py`. Top-K remains the default with
  identical selections.
- **Foundation-model teacher labeling (row 3a), a parallel track.** A
  foundation MLIP supplies reference labels instead of QM. It is independent of
  steps 1–3 and can start after the Increment 7 close-out. Ground-state
  teachers come first; excited-state teachers wait until a suitable model
  exists.
- **Fine-tuning with ALF (rows 3b and 11), a later goal in three tiers.** ALF
  already fine-tunes ALF-trained HIPPYNN ensembles through the example-local
  task in `examples/fine_tuning/`.
  - **F1, library HIPPYNN fine-tuning.** It can start any time after the
    current-release compatibility gate.
  - **F2, multi-state fine-tuning.** It follows E3.
  - **F3, foundation-model fine-tuning under active learning.** It follows E5,
    teacher labeling (3a), and multi-node training.

  See **Later steps**.

Rationale (2026-11-03): the user set composable sampling and multi-node
training as the priorities after excited states, to make other uncertainty and
acquisition methods easier.
- **Composable sampling is folded into step 2** because that step already
  rebuilds UDD and LCM in the packed engines. Porting them ad hoc first would
  mean writing them twice.
- **Multi-node training runs in parallel** because it touches only training
  placement. It buys throughput: larger ensembles and multi-state ring
  training. It does not add method extensibility.
- **Method extensibility comes from three seams:**
  1. named uncertainty signals from inference (E5);
  2. eligibility from configured signal/cutoff pairs (E5);
  3. the selector/retention interface above.
- **How later methods attach:** single-network estimators (row 11) plug into
  seams 1–2. Differentiable estimators used for biasing plug into the
  composable-sampling interface. Diversity selection (row 2) plugs into seam 3.

Rows 2 (remaining strategies), 3, 5, 9, and 11 follow later. While
the work is ported, the fork remains the production environment for excited
states; `production_excited_states` (`c364fa8`) is the source material. Its
excited-state sources match `feature/unified-sampling-drivers` byte for byte.

#### What the fork provides today

- **The cpp9 example (`examples/cpp9_gpu4pyscf`) is a standalone trial, not
  an active-learning loop.**
  - `run_trial.py` and `cpp_trial/` drive two Parsl sessions:
    1. label a fixed pool of 1,000 perturbed geometries with GPU4PySCF
       (CAM-B3LYP/6-31G, RKS/TDA, `nroots=3` giving S0–S3, analytic
       gradients, and NACs for pairs (1,2), (1,3), (2,3));
    2. train four multi-state HIPPYNN members, then exit.
  - There is no sampler, no acquisition, and no `alframework.__main__`.
  - `cpp_trial/qm.py` is a local copy of the core QM interface that adds
    transition dipoles and oscillator strengths.
  - Its audit (`docs/cpp9_example_audit_2026-10-21.md` on the fork) records
    CPU tests with mocks and one CPU PySCF comparison only. Real GPU QM and
    training were not validated there, and rejected frames are not refilled.
- **Self-contained excited-state core.** These depend on ordinary task
  entrypoints, `molecules_class`, `pyanitools`, and each other, not on the
  fork's staged workflow, run control, or sampling dialect:
  - `tools/excited_state_tools.py`: the `sE#`/`F#`/`dE#` key contract;
    the state count is inferred from contiguous keys in `properties_list`.
  - `tools/atom_order.py`.
  - The `pair_atomic` storage scope and `nac_pairs` index in
    `tools/tools.py::store_current_data`.
  - `qm_interfaces/gpu4pyscf_interface.py`: `run_gpu4pyscf_states`,
    `gpu4pyscf_excited_state_task`, singlet RKS/TDA, and optional NACs.
  - PySEQM AM1 CIS through `pyseqm_excited_state_task`, run as one molecule
    per child process.
  - `ml_interfaces/excited_state_hippynn_interface.py`:
    - one shared trunk with an energy head and force gradient per state;
    - optional derived gap targets;
    - its own explicit HDF5 loader;
    - `train_excited_state_HIPPYNN_ensemble_task`;
    - nonperiodic only.
  - `builders/h5_replay_builder.py`.
- **Excited-state sampling exists only in the fork's ALCHEMI sampler**
  (`samplers/alchemi_sampling.py`, `model_mode='excited_state'`).
  - The propagated state comes from `selected_state` metadata or from
    `state_selection` `fixed`/`batch_cycle` (`tools/sampler_batching.py`).
    Every member of a batch shares one state.
  - Uncertainty uses `Es`/`Fs`/`Fsmax` on the propagated state, with
    per-state and `dE##_stdev` diagnostics.
  - Summed scoring has an optional gap-uncertainty term.
  - LCM `gap_seeking` permanently switches a triggered replica to the
    smoothed surface.
  - The fork's ASE sampler has no excited-state path. Its full-loop examples
    (`excited_state_gpu4pyscf`, `_nacr`, `excited_state_pyseqm`) also rely on
    the fork's `run_control` and staged workflow.
- **What dev lacks.**
  - There is no GPU4PySCF interface.
  - `pyseqm_parameters` rejects `excited_states`.
  - `train_HIPNN_model` fits one energy.
  - The evaluator contract assumes one energy per system: `energies [M, S]`,
    `HippynnBatchedEnsemble`, `PackedMLMDEngine._evaluate`/`observe`,
    `ensemble_statistics`, and `uncertainty_flags`.
  - Existing hooks:
    - the `SCORE_TERMS` docstring anticipates a gap term;
    - `default_batch_key` anticipates a selected-state key;
    - storage writes only keys listed in `properties_list`, so new keys are
      opt-in.
  - Per-state keys are atom-first (`F#` is `[atoms, 3]` per frame), so the
    existing `"atomic"` sorting handles them. Only NACs need the new scope.

#### Excited-state ALF plan (E1–E6)

Each step is a separate local change with its own CPU tests. Existing
configurations, task signatures, and ground-state results stay unchanged.
Steps E1–E4 can proceed in parallel once E1 lands.

| Step | Deliverable | Acceptance gate |
| --- | --- | --- |
| E1. Contract and storage | Port the state/gap key contract, `atom_order`, and the `pair_atomic`/`nac_pairs` storage scope. | Multi-state and NAC HDF5 round-trip tests; atom order restored; existing `properties_list` behavior unchanged. |
| E2. Excited-state labelers | GPU4PySCF TDA as an ordinary `QM_task`, with single and sequential packing (no native batch). Promote cpp9's transition dipoles and oscillator strengths into the interface. PySEQM CIS excited states. | Mocked CPU tests and a CPU PySCF comparison; then real A100 GPU4PySCF labels with the cpp9 label checks (shapes, finiteness, offset applied once, oscillator strengths ≥ 0, NAC pair order). |
| E3. Multi-state training | The excited-state HIPPYNN trainer as a separate `ML_task`. Reconcile its loader with dev's small-molecule `ALFPyAniDirectoryDB` rather than keeping two readers. | Real four-member GPU training; members reload; the ground-state trainer is unchanged. |
| E4. External initial geometries | A builder that admits externally generated, unlabeled geometries (xyz or geometry-only HDF5) as bootstrap/sampling seeds. Uses the scoped parts of `h5_replay_builder` (row 6). The random ring generator stays outside ALF. | Identities, atom order, and provenance survive into labeling inputs; no stored labels are carried into new labeling. |
| E5. Sampling on a chosen state | Multi-state evaluator output (per-state member arrays; existing ground-state shapes remain the default). A `selected_state` from metadata or `fixed`/`batch_cycle`, with a selected-state batch key. Per-state diagnostics and a named gap signal with its own cutoff as a `SCORE_TERMS` entry. Eligibility reads configured signal/cutoff pairs; `Es`/`Fs`/`Fsmax` keep their meaning. | CPU parity with fakes across ASE, TorchSim, and ALCHEMI; a bounded molecular excited-state loop through the real coordinator. |
| E6. cpp ring full ALF example | An ordinary master configuration run through `alframework.__main__`: (1) bootstrap from a random initial dataset of ring structures made by an external generator; (2) cpp9-science GPU4PySCF labels, with NACs stored but not trained; (3) multi-state training; (4) excited-state MLMD and acquisition; (5) iterate. The fork's standalone driver is not ported as-is; its settings and checks are reused. | A real, bounded Darwin campaign like the water matrix: at least one round selects new sampled ring structures, labels them with converged GPU4PySCF, stores them, and promotes a retrained, loadable four-member ensemble. |

E6 is the first point where dev can do the cpp ring work, so it follows
sampling (E5) rather than preceding it. Open decisions:
- whether PySEQM CIS uses dev's native batch path or the fork's one-molecule
  child process;
- which state(s) the ring campaign propagates;
- whether its first campaign enables gap-uncertainty ranking.

#### Later steps (outline; detail after E5)

- **Composable sampling with batched UDD and gap ALF.**
  - Start with the bias/control interface in the packed engines (row 1). Each
    control:
    - receives physical predictions and returns propagation forces;
    - keeps replica-local state (for example, the permanent LCM switch);
    - never changes the unbiased predictions used for screening and
      uncertainty;
    - applies in a fixed, documented switching order.
  - First plugins:
    - UDD on the selected state;
    - wells;
    - LCM gap seeking as a per-replica permanent mode switch.
  - The deferred physical-force screen (`max_force_cutoff`) excludes bias and
    LCM forces.
  - Gap-aware summed acquisition.
  - Gate: parity with the current `MLMD_calculator` UDD/well formulas and with
    the fork's ALCHEMI LCM behavior, across ASE, TorchSim, and ALCHEMI.
- **Multi-node training (parallel track).**
  - Today `train_HIPPYNN_ensemble_task` runs `multiprocessing.Pool` on one
    node. Fan members out across `alf_ML_executor` blocks that span nodes,
    via a `join_app` or per-member apps, returning the existing
    `(completed, training_id)`. `__main__.py` is unchanged.
  - Use local-ordinal GPU assignment, as for samplers (`tools/gpu.py`); keep
    per-member `completed` handling and the existing member directory layout.
  - Gate: a real two-node Darwin training of four members, then promotion and
    reload.
  - Data-parallel training of one member (HIPPYNN's experimental Lightning
    trainer, with a checkpoint-layout change) is a separate decision.
- **Acquisition selector/retention interface (after E5).**
  - The selector receives canonicalized candidates and declares what workers
    must retain.
  - Top-K is the default.
  - Gate: identical selections in the existing `tests/test_cohorts.py`
    width/completion-order replay; online scalar top-K retention closes the
    candidate-memory follow-up.
- **Foundation-model teacher labeling (row 3a, parallel track).** The
  labeling boundary already supports this: `LabelingCoordinator`
  (`tools/labeling.py`) packs `native` batches for any `QM_task` that accepts
  `molecule_objects`, as PySEQM does. Storage, retraining, and promotion see
  only labels and a converged flag.
  - **Adapter:** an ordinary `QM_task`, native over `molecule_objects`, with a
    single-structure form for sequential packing. It:
    - loads the teacher once per worker;
    - evaluates with packed batched inference, reusing the evaluator's padding
      and atom-order conventions, or with a per-structure ASE calculator;
    - converts outputs to ALF's eV and eV/Å;
    - selects its GPU through `tools/gpu.py` local ordinals.
  - **Why `ase_calculator_task` is not enough:** it passes `QM_run_command`
    and the whole `QM_config` to the calculator constructor, and it requires
    `calc.converged`.
  - **Converged rule:** a teacher needs its own rule, such as finite outputs
    plus optional domain checks.
  - **Provenance:** record the teacher's name, version, checkpoint hash, units,
    and adapter settings with its labels, separately from the student model.
    `store_current_data` writes only `properties_list` keys, so this needs an
    explicit metadata path.
  - **Site:** a GPU-configured `alf_QM_executor`. This is configuration only,
    as in the PySEQM GPU campaign.
  - **Gate:**
    - single and native-batch parity at widths 1/2/4/8;
    - a bounded water-style active-learning campaign in which teacher labels
      are stored, used for real retraining, and promoted;
    - provenance present in the stored data.
  - **Out of scope for 3a; remains in 3b or row 9:** mixed teacher/QM labels
    (multi-fidelity or delta learning), teacher-driven dynamics,
    fine-tuning/distillation handoffs, and opaque model-artifact references.
- **Fine-tuning with ALF (later goal; added 2026-11-03).**
  - **What exists.** `examples/fine_tuning/` (and its Darwin variant) uses a
    local `ML_task`, `fine_tune_ml_task.fine_tune_HIPPYNN_ensemble_task`. It:
    - loads a trained HIPPYNN seed ensemble from `models/model-0000/`;
    - keeps the checkpoint graph and weights, with a fresh optimizer and
      controller at a lower learning rate;
    - rebuilds the training database from `h5store/`, filtered to the
      checkpoint's inputs and targets;
    - writes each adapted ensemble to the next model slot.

    `__main__.py` is unchanged and `--test_ml` runs fine-tuning only. The
    example is part of the compatibility baseline. Fine-tuning is example
    code, not library code, and it covers HIPPYNN ensembles in ALF's own
    layout only.
  - **F1, library HIPPYNN fine-tuning (row 3b).**
    - Move the task beside `train_HIPPYNN_ensemble_task` as an ordinary
      `ML_task`.
    - Make its settings explicit: source checkpoint, optimizer reset, learning
      rate, frozen layers, and training-data filter.
    - Record lineage: the parent checkpoint hash and generation.
    - Keep the example working by pointing it at the library task, with its
      results unchanged.
    - Inherit multi-node placement, and do not conflict with it.
    - Gate: the existing fine-tuning example gives equivalent results through
      the library task, including a real Darwin fine-tuning round.
  - **F2, multi-state fine-tuning (after E3).** Start a multi-state model from
    a pretrained ground-state or earlier multi-state HIPPYNN; for example,
    seed the cpp ring model from a ground-state model. Use the F1 lineage and
    settings, with an explicit mapping from source to new heads and targets.
  - **F3, foundation-model fine-tuning under active learning (rows 3b and
    11).** ALF samples with the model being fine-tuned and labels with QM or a
    teacher (3a). It needs:
    - **Uncertainty.** Foundation models are usually single networks.
      Options: fine-tuned ensembles (expensive), heads on a shared backbone,
      or single-network estimators (row 11) through the E5 named-signal
      contract. F3 is the natural first use of row 11.
    - **Model-family adapters** for training (behind `ML_task`) and inference
      (behind the evaluator contract).
    - **Opaque model-artifact references,** replacing the batched loader's
      `best_model.pt` and the trainers' `model-NN` assumptions. This must
      preserve the existing HIPPYNN layout.
    - **Provenance:** the base model, version, license, and which weights
      changed.
    - **Engines:** TorchSim and ALCHEMI already host foundation-model
      calculators, so the engine side may need little new work. Each model
      still needs its own qualification.

    Gate: a bounded active-learning campaign in which a fine-tuned foundation
    model's uncertainty selects frames that are labeled, stored, used for
    fine-tuning, and promoted, with lineage and provenance recorded.
  - **First F3 targets: MACE and UMA (notes from 2026-11-03).** These
    describe the packages found installed, not an acceptance claim. Each
    package must still pass the clean current-release qualification.
    - **Availability.** `alf_alchemi_02` contains `mace-torch` 0.3.16 and
      `fairchem-core` 2.23.0. The batched engines already provide calculators:
      - TorchSim: `torch_sim/models/mace.py` and `fairchem.py`;
      - ALCHEMI 0.2: `nvalchemi/models/mace.py` and `uma.py`.
    - **Inference.** ALF needs an evaluator adapter, the counterpart of
      `load_hippynn_batched_ensemble`.
      - It is selected through `batched_calculator` and returns per-member
        energies `[M, S]` and forces `[M, atoms, 3]` in eV and eV/Å. Engine
        changes are not expected.
      - The legacy sampler could use an `ase_calculator` loader returning one
        ASE calculator per member. This is untested.
    - **MACE training adapter.** `mace_run_train` provides:
      - `--foundation_model`, and `--freeze` for frozen layers;
      - multihead fine-tuning (`--multihead_finetuning`, `--pt_train_file`),
        which replays pretraining data to limit forgetting.

      The adapter exports `h5store/` to extxyz in eV and eV/Å and trains each
      member. It keeps outputs in ALF's `model-NN` layout and returns
      `(completed, training_id)`.
    - **UMA training adapter.** `create_uma_finetune_dataset.py` builds the
      dataset, fits normalizers and linear references, and writes a config
      from fairchem's fine-tuning template. The fairchem CLI then trains from
      the UMA checkpoint. The adapter must select and record the task head
      (`omol` for molecules, `omat` for materials).
    - **Reference level and energy offsets.** Fine-tuning on ALF's labels
      (for example ORCA PBE/def2-SVP) moves the model away from its pretraining
      level of theory. Make these explicit and record them:
      - MACE E0s;
      - UMA linear references;
      - `omol` charge and spin inputs.
    - **Training placement.** Both trainers support multi-GPU training
      natively (MACE `--distributed`, fairchem DDP). For these families,
      multi-node training means placing members and then delegating to the
      native distributed training, not building a new data-parallel trainer.
    - **Uncertainty.** Members fine-tuned from one foundation checkpoint tend
      to agree too closely, so their disagreement underestimates error. MACE
      copies are affordable but share this problem; several UMA copies are
      also expensive. Options, roughly cheapest first:
      1. a heterogeneous committee (for example MACE, UMA, and HIPPYNN), which
         the per-member evaluator contract already allows;
      2. multiple heads on a shared backbone;
      3. single-network estimators (row 11) through E5's named signals.

      Calibrate cutoffs for each method; do not reuse ensemble `Escut`/`Fscut`.
    - **Access, license, and environments.**
      - UMA weights are gated on Hugging Face: workers need an accepted
        license and a token.
      - Record each base checkpoint, version, and license: UMA's FAIR
        Chemistry license and MACE's model licenses.
      - Each family pins its own torch/e3nn range, so each gets its own
        qualified environment, separate from the HIPPYNN environment.
    - **Suggested order.** Each step has a bounded campaign gate like teacher
      labeling. The first two need no fine-tuning.
      1. UMA or MACE as a teacher (3a), labeling for HIPPYNN students.
      2. A frozen committee as the sampling model: the evaluator adapter
         only, to prove the engine, evaluator, and uncertainty plumbing.
      3. MACE fine-tuning (more mature entrypoints and smaller models).
      4. UMA fine-tuning.
- **Staged campaigns across engines** (row 8): durable cohort/handoff
  persistence, a run-control equivalent, staged configurations, and
  retained-label budgets. The fork's `run_control`, `workflow_io`,
  `staged_workflow`, and `dataset_screening` are source material.

The excited-state increment includes explicit topology validation and
topology-aware replay/storage where configured. Existing ordinary workflows
must not acquire a new fixed-topology requirement.

Budgeted workflows must distinguish selected candidates, labeling attempts,
successful labels, stored/retained frames, and rows surviving trainer filters.
Report these stages separately. Exact checkpoints and final endpoint training
refer to the chosen retained-data lineage; routine retraining and raw-store
overshoot are separate concepts.

### Known implementation follow-ups

The following watch list was checked against development at `be7fb64` on
November 2, with the first three rows updated for the Increment 7
prerequisite fixes (`2a1708a`, merged at `24028da`). It records concrete limitations and
future edits, not a requirement to handle every edge case before continuing. Fix affected scientific behavior
explicitly, preserve the ordinary examples, and keep implementation in helpers
or adapters wherever possible.

| Area | Current behavior or gap | Follow-up and boundary |
| --- | --- | --- |
| Periodic geometry screening | Fixed in the Increment 7 prerequisites: a shared minimum-image screen excludes diagonal self-pairs and retains physical overlaps. CPU, A100, and MI250 regressions pass. | This screen compares distinct atoms, not an atom's own periodic images. Cutoffs and screening timing are unchanged. |
| Scheduler GPU visibility | Fixed for legacy/batched MLMD, UDD, and NeuroChem inference: inherited masks remain unchanged and automatic assignment uses local ordinals. Simulated allocation checks and real A100/MI250 reused-worker runs pass. | Training pools and reactive samplers are separate follow-ups; external Parsl/site configuration must provide valid worker visibility. NeuroChem has adapter/unit coverage, not hardware qualification in this stage. |
| Legacy ASE checkpoint loading | Fixed: `HIPNN_ASE_calculator()` loads explicit absolute checkpoint paths without `chdir`. Concurrent member loading, prediction parity, and the twelve-case water active-learning matrix pass. | Checkpoint format and restoration are unchanged; the batched loader keeps its separate weights format. Cohort and other backend acceptance remain open. |
| Configuration reload and pinning | The main loop reloads task configurations, but driver/buffer settings and cohort selection are constructed at startup. Cohorts pin a model generation, not an immutable sampling/acquisition configuration. | Add explicit reload safeguards or configuration snapshots at admission/submission. Changing engine, size/K, thresholds, weights, or stopping policy mid-cohort must not mix meanings between startup coordination and later task configs. Until then, keep these settings fixed for a run. |
| Cohort admission | A cohort requires full membership and all terminal reports, with no admission timeout or partial selection. A permanently undersupplied builder leaves it unfinished. | Document and observe admitted/finished counts during Increment 7. Keep incremental admission for cohorts larger than worker capacity; do not introduce an automatic partial-cohort policy without a separate decision. |
| Queue thresholds and strict batches | `full_only` can leave a PBC/model group waiting indefinitely when it cannot fill within available capacity. The inherited `minimum_QM` harvest gate can also hold a small number of finished sampler results. | Exercise small capacities and fragmented groups with the existing timeout policy. Check that configured completion thresholds are reachable. Add focused diagnostics/validation or drain behavior only as a separate compatibility change. The labeler already checks `target_queued_QM >= batch_size` for strict batches. |
| Candidate memory and caps | `continue` appends candidate geometries throughout the trajectory; `_prune()` runs only at the end. `max_candidates_per_trajectory` bounds the returned payload, not peak worker memory. A cap below K can change the unrestricted global top-K. | Add online scalar top-K retention with a proof/test that selection is unchanged when no smaller explicit cap is requested. Future diversity/distance selectors must declare different retention needs. Bound completed-cohort summary history for long runs separately. |
| Packed inference memory | HIPPYNN pads to the largest system in a task. Capacity is specified in systems, without an atom/pair-memory budget or automatic OOM splitting. A failed batched evaluator call can fail the whole active group. | Profile heterogeneous sizes before increasing widths. Consider adapter-declared memory limits and explicit retry/splitting later, preserving atom order, model pinning, identities, and per-trajectory random streams. This is not a requirement to add every OOM recovery path now. |
| Capacity and result latency | Finished members leave GPU packed state, but their queue width is released only when the whole task completes and is harvested. Earlier first-trigger frames wait for slower batch-mates. There is no inflight replacement or result streaming. | Measure this delay in Increment 7 and keep physical capacity, candidate storage, and pending QM work distinct. Streaming or replacement would need an explicit task/result protocol change; fixed task membership remains current behavior. |
| Restart and failed work | Buffers, cohorts, model admission sequence, and handoffs are in memory. No new retry journal or exactly-once QM handoff exists. Candidates survive a later isolated failure only if the task returns them; crashed-worker-only frames are lost. | Keep this limitation visible for ordinary screen/tmux runs as well as staged workflows. Add durable snapshots, logical IDs, replay, and handoff recovery in the later recovery increment, without requiring new pause controls now. |
| Status and CI | Task-failure counters and failed-structure counts are different; `failed_structures` is not yet part of persisted status. CI workflows target `main`, not `development`. | Improve count/latency reporting without changing existing budget units. Consider enabling development CI separately; current acceptance relies on explicitly recorded local/backend checks and optional-dependency skips. |

Current strategy separation is deliberately incomplete. The multi-trajectory
task supports Langevin MLMD and existing UDD/well options; it is not yet a
general propagator/bias plugin API or a grouped non-MD optimization interface.
The engine contract and packed base remain MD-oriented. Acquisition has the
two built-in strategies and an extensible score-term table, but no public
import-string eligibility/observation/retention interface, geometry-only
strategy, or diversity filter. Legacy reactive tasks remain available through
their existing entrypoints; they have not been ported to the new engines.

Keep capability boundaries explicit in future changes: native HIPPYNN supports
nonperiodic or fully periodic systems, not partial PBC; packed engines reject
ASE constraints; ALCHEMI targets exactly toolkit 0.2.0 and CPU/NVIDIA, with AMD
served by TorchSim. NPT, excited-state/LCM sampling, and periodic excited states
have not been added here. A toolkit upgrade or new model adapter needs its own
qualification rather than inheriting the existing acceptance results.

The approved float32 comparison uses raw member `rtol=atol=1e-5`, verifies
reported uncertainty reductions, and propagates measured member roundoff into
spread comparisons. It does not promise identical flags/rankings for a frame
arbitrarily close to a cutoff, or identical trained-model trajectories across
devices. Preserve that distinction when changing screening or ranking. The
water smoke model also produces sensitivity-range warnings and short distances
in the brief benchmark: engine correctness is not potential stability or
accuracy validation.

The model/training-boundary increment must also address two porting requirements
recorded during the recent experimental-fork review. They are not claims that
the corresponding new handoff/final-training machinery already exists here:

- The training handoff must tolerate valid trainer callables whose signature
  omits `ML_config`.
- Move model-family artifact validation behind adapters; common final-training
  validation must not assume HIPPYNN-style `model-NN` member directories.

Deferred documentation housekeeping: the fork's copy of this roadmap still
has the October 31 status. Synchronize it in a separate documentation change;
this development copy is authoritative for merged integration status. Keep
experimental-fork capabilities distinct from those available in development.

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

### Existing-example compatibility (planned, 2026-11-03)

No planned increment is meant to make the original examples unusable; every
new capability stays opt-in. The evidence is narrower than that intent, so
these checks are part of the future plan.

**Current state.**
- All ten tracked examples still select the original tasks:
  - `simple_mlmd_sampling_task`, or `reactive_sampling`;
  - the ORCA, VASP, and ASE labelers;
  - `train_HIPPYNN_ensemble_task`, `fine_tune_HIPPYNN_ensemble_task`, or
    `train_ANI_model_task`;
  - the original builders.
- The only deliberate config change is `"distcut": 0.7` in four water
  examples, made at the user's direction; the code default stays 1.2 Å.
- Since these changes, only the water family has run through the real
  coordinator: the legacy ASE width-one cases of the campaign matrix.
- IL, molten salt (VASP), UO2 (ANI/NeuroChem), reactive sampling,
  fine-tuning, and seeded active learning have CPU tests and unchanged code
  paths. They have not been re-run end to end.

**Risk by planned step.**
- **Low:** E1–E4, teacher labeling, and the selector interface. They add new
  tasks or opt-in branches. Watch the shared helpers:
  - `store_current_data`, where E1 adds the NAC scope;
  - `hippynn_interface.py`, where E3 must leave the ground-state trainer
    unchanged.
- **Medium:**
  - E5's evaluator and eligibility generalization: ground-state shapes and
    `Es`/`Fs`/`Fsmax` must remain the default.
  - Multi-node training: every HIPPYNN example uses
    `train_HIPPYNN_ensemble_task`, so single-node placement stays the default
    with identical behavior.
- **Highest: composable sampling with UDD and wells.** Five examples set
  `well_params`: IL, molten salt, simple water, fine-tuning, and seeded active
  learning. Build the plugins for the packed engines. Leave `MLMD_calculator`
  and `mlmd_sampling` unchanged, or prove formula parity if code must be
  shared.
- **Staged campaigns and recovery:** keep ordinary runs as the default; do not
  port the fork's `run_control` changes to `__main__.py` wholesale.
- **Fresh-install policy:** newer dependencies can break examples without any
  ALF change. ANI/NeuroChem and VASP paths are the least exercised.

**Planned checks.**
1. **Compatibility smoke matrix.** One bounded real-coordinator run per
   tracked example family. Run it:
   - at the first fresh-environment build;
   - after E5, after composable sampling, and after multi-node training;
   - at each release point.

   Water, fine-tuning, and seeded active learning run with ORCA on Darwin.
   IL, molten salt (VASP), UO2 (ANI), and reactive sampling need their
   backends. Record a missing backend as "unverified, backend unavailable",
   never as passed.
2. **Config-load test.** A CPU test that loads every tracked example's JSON
   configurations and validates them against current code, including
   `acquisition_settings`, `driver_settings`, labeling batch settings, and
   task import strings. It is cheap and suitable for CI.
3. **UDD/well parity.** In the composable-sampling step, test the new plugins
   against the legacy `MLMD_calculator` well and UDD forces, using the
   examples' actual `well_params` values (`r_start` 3.0 and 10).

**Breaking changes need their own increment.** Examples would stop working only
through a deliberate decision, which no current increment makes. Examples of
such decisions:
- removing the legacy sampler or `MLMD_calculator`;
- changing code defaults such as `distcut` or the uncertainty cutoffs;
- adopting the fork's `sampling`/`inference` configuration dialect, already
  rejected in Increment 4;
- dropping a dependency such as NeuroChem.

Any such retirement is its own increment: an explicit decision, a deprecation
period, and migrated example configurations. It is never a side effect of
another change.

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
