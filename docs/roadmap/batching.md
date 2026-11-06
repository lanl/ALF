# Batching and ground-state roadmap

Increments 0-7: batched labeling, ground-state MLMD, acquisition, and the combined release validation, with their follow-ups. See the [roadmap index](README.md) for current status and standing rules.

Port narrowly scoped implementations and tests. Each increment must be
independently reviewable and tested before dependent increments merge. Inspect
PBC propagation in the multi-structure condensed-phase builder as a focused
prerequisite; retain the recent main fixes. (Done in Increment 0; see below.)

## Increment 7 — closed by user acceptance (2026-11-05)

The November 5 close-out decision below supersedes earlier open-gate and retry
instructions in this section. Preserve historical outcomes as recorded; no
additional Increment 7 campaigns or harness repairs are required to move on.

The engine work is merged, and the twelve-case Darwin water campaign now
validates first-trigger active learning across all three engines. Cohort/summed
acquisition and the other backend gates below remain open. Keep this increment focused on existing workflows and the
concrete correctness gaps listed under [Known implementation follow-ups](#known-implementation-follow-ups).
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

### Core water campaign matrix (2026-11-02)

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

### Remaining campaigns — implemented; external acceptance pending (2026-11-03)

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

### Controlled-failure rerun and investigation (2026-11-04)

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

The follow-up harness repair at `791dd05` moves the controlled step failure
after the final observation of a fully observed cohort, targeting an active
trajectory whose retained frame naturally makes global top-K. It changes only
fault placement; ranking, cutoffs, and selection remain unchanged. If there is
no active winning trajectory, the harness does not force a failure or selection.
The focused sampler/cohort suite passed all 35 checks. Its bounded real retry is
`inc7_closeout_20261104/failure-run-001/`, supervised by the persistent
`alf-inc7-closeout-failure-20261104` session on `darwin-fe2`, with frozen
`source-failure/` at `791dd05`. That attempt finished incomplete: eight new
labels reached a promoted ensemble, but no failure was injected. Its winners
were rejected before the final step, so the late-active-winner condition never
held. Cleanup verified termination. The overly strict outer dependency stopped
the fresh campaigns and example checks before they started; those stopped
supervisors and their evidence remain preserved.

The user authorized repair and relaunch, emphasizing reuse of passed evidence.
Local `acb4bb9` replaces the stochastic trigger with a recorded-input regression
fixture. Its first eight inputs are immutable, genuinely generated builder
outputs from that attempt, with the same four-member initial model, random
streams, cells, PBC, and scientific settings. The planned failure follows the
previously observed natural top-ranked frame, `mol-0000-0000000007` at step 26,
before its later geometry rejection. The first two builder tasks explicitly
record fixture replay; all subsequent builders and the fresh coverage matrices
retain their ordinary generation path. No stored QM labels or old candidates
are replayed into acquisition: trajectories, ranking, labels, storage, and
retraining run again. Acceptance now requires that exact planned frame in the
promoted ensemble's training-input lineage. Changed fixtures, configurations,
models, or a missed injection produce an explicit error instead of another
55-minute wait for an impossible condition.

Focused checks passed: 54 in the established stack and 66 in the fresh stack,
including exact-frame lineage, fixture integrity, early failure retention,
independent-stage continuation, and owned-job cleanup. Independent qualification
stages now continue after a failed scientific/harness check when its resources
are verified terminated. Unverified cleanup still stops further allocations;
a missing fresh seed blocks only its dependent MI250 campaign. The separate
example checks have their own wB97X seed and do not require every PBE case to pass.

The new persistent supervisor is `alf-inc7-relaunch-20261104` on `darwin-fe2`,
using frozen `source/` at `acb4bb9`. Its root is
`/vast/home/pjlohr/ALF_LANL/validation/inc7_relaunch_20261104/`;
`relaunch-manifest.json`, `result.json`, and `supervisor.log` identify inputs,
source, stage outcomes, and progress. It first retries only the remaining
controlled-failure case, then runs the unexecuted current-release hardware,
water, native PySEQM, MI250, and original-example checks sequentially. Passed
historical matrices, the other two failure cases, and fresh CPU checks remain
accepted in their recorded environments. MI350 and user-guide updates remain
deferred. No newly launched gate is claimed passed before its actual result.

### Increment 7 close-out

**Closed by user acceptance, November 5.** Following the successful batching,
failure-handling, and fresh-dependency campaigns, the user directed development
to move on without further example-harness fixes or campaign reruns. The five
incomplete original-example checks below are accepted validation gaps, not
passing results. Their logs and outcomes remain unchanged. This decision
supersedes the previous requirement to complete every existing-example smoke
check before new feature implementation. The current-release gate is accepted
with these explicit gaps; fresh-install CPU CI is prepared locally and remote
CI execution still awaits the user's push.

Next work is excited-state ALF, starting at E1 in the E1–E6 plan. Trained
cell-input checkpoints, long-time potential stability, MI350, and user-guide
updates remain deferred. NeuroChem is excluded as recorded below.

November 5 scope decision: the user excluded legacy ANI/NeuroChem (the UO2
example) from current Increment 7 acceptance. Retain its unavailable-backend
record as unverified; it is not a passing qualification and no longer blocks
close-out. This does not remove the backend or change its production interface.

The November 4 relaunch finished with verified resource cleanup for every stage:
the controlled trajectory failure passed, the fresh water matrix passed 12/12,
native PySEQM passed 2/2, MI250 passed 4/4, and both hardware smoke stages passed.
Original-example compatibility remains incomplete: simple water and fine-tuning
passed; multi-builder water and seeded active learning stopped at the seed
architecture guard; IL, reactive sampling, and molten salt reached their case
deadlines. The shared water seed has a vector model, whereas multi-builder
water requests a larger scalar model and seeded active learning omits that
architecture setting. These are harness seed-reuse failures before execution,
not evidence that those example workflows failed. IL completed one bootstrap
QM label; molten salt had two VASP calculations started and no returned labels;
reactive sampling completed 16 bootstrap labels and four-member initial
training, then had no returned sampling tasks by the deadline. These timeouts
do not establish either successful end-to-end compatibility or an ALF defect.
The retained evidence is under `inc7_relaunch_20261104/fresh-examples/`.

The original close-out checklist is retained for context and is superseded by
the acceptance decision above:

1. Record the retry's PySEQM and controlled-failure results and cleanup
   evidence; repair only demonstrated failures, as focused changes.
2. Complete the **First current-release compatibility gate** before new
   feature implementation and the next acceptance-gated push. Build fresh
   environments, run the CPU and real workflow checks, and enable fresh-install
   CPU CI on `development`. Resolve versions at build time; the dated package
   audit above is historical evidence, not a dependency lock.
3. Record supported versions and any explicit incompatibilities from that
   gate. Existing `atomistic` and other qualified environments remain useful
   comparisons but are not the compatibility target.
   Include the example config-load test and the first existing-example
   compatibility smoke matrix (see [Existing-example compatibility](future.md#existing-example-compatibility-planned-2026-11-03)).
4. Keep the deferred gates explicitly open: trained cell-input checkpoint,
   long-time potential stability, and MI350. They do not block the next increment.

User-guide updates were deferred at the user's direction (2026-11-04), outside
the scope of Increment 7 acceptance. They now describe `driver`, `batching`,
`acquisition`, and labeling batch settings, with pointers to the tracked
TorchSim/ALCHEMI water configs; see [User documentation pass](README.md#user-documentation-pass-2026-11-06).
Roadmap status and validation evidence records remain part of close-out.

### Focused prerequisites (2026-11-02)

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

## Progress log

Record each increment's commits, evidence, and limitations here as it lands.

### Increment 0 — roadmap and baseline (done locally, 2026-10-31)

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

### Increment 1 — shared batching helpers (done locally, 2026-10-31)

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

### Increment 2 — labeling coordination (done locally, 2026-10-31)

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

### Increment 3 — native PySEQM labeling (done locally, 2026-10-31)

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

### Increment 4 — ground-state sampling and acquisition (merged)

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

### Increment 5 — TorchSim dynamics (done locally, 2026-11-02, commit `a80c23e`)

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

### Increment 6 — ALCHEMI 0.2 (merged at `be7fb64`, 2026-11-02)

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

## Batching and reference labeling

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

## Dynamics, screening, and identity

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

## Acquisition and trajectory outcomes

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

## Recovery boundary

Keep waiting batches and logical cohorts in memory in this initial development
release. Add no persistent buffers, task journal, staged workflow, or new
restart/shutdown controls. Document that unfinished in-memory pools are not
restored after process restart. The experimental fork's durable implementation
is source material for the later recovery increment, not a prerequisite here.

## Known implementation follow-ups

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
| Status and CI | Task-failure counters and failed-structure counts are different; `failed_structures` is not yet part of persisted status. | Improve count/latency reporting without changing existing budget units. Unit tests and the docs build run on `development`; real backend acceptance still relies on explicitly recorded runs outside CI. |

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
ASE constraints; ALCHEMI supports the qualified toolkit 0.2.0/0.3.0 versions on
CPU/NVIDIA, with AMD served by TorchSim. NPT, LCM sampling, and periodic
excited states remain later work. A toolkit upgrade or new model adapter needs its own
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
