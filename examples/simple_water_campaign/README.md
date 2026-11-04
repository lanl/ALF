# Darwin water batching campaign

This opt-in validation suite runs real ORCA labeling, HIPPYNN training, and ALF
coordinator workflows. It does not change the ordinary water examples or ALF's
production task signatures.

Every builder task returns four periodic water boxes. The twelve cases cross
ASE, TorchSim, and ALCHEMI dynamics at widths one/four with single ORCA tasks or
sequential packs of four ORCA calculations. ASE width one uses the legacy
sampler; ASE width four uses the multi-trajectory ASE engine. Native PySEQM
labeling, cohort acquisition, and a full MI250 campaign are separate work.

The immutable profile uses PBE/def2-SVP/RI-J, a four-member ensemble, a shared
32-structure bootstrap target, and at least eight new sampled labels per case.
It preserves the example's uncertainty cutoffs, training settings, and force
controls. Trajectories are capped at 0.1 ps. Periodic geometry is retained;
ORCA still evaluates finite clusters, following the existing example's
convention. Passing this suite does not establish bulk-water potential accuracy.

## Prepare and run

Use a fresh artifact directory **outside the checkout** and a fixed source
checkout. With the established Darwin environment:

```bash
python examples/simple_water_campaign/run.py \
  --run-root /path/to/artifacts --prepare-only
python examples/simple_water_campaign/run.py \
  --run-root /path/to/artifacts
```

To execute one case, supply e.g. `--case torchsim-w4-q4`. Its shared bootstrap
is still built and trained, but only that matrix case runs. A one-case pass
never marks the complete matrix accepted. `--prepare-only` submits no jobs.
Do not edit generated JSON: the runner verifies its configuration and source
manifest before execution. A root executes only once; use another directory
for a new attempt so earlier failures remain reviewable. For a harness repair,
pass `--budget-start` with the first attempt's `execution.json` start timestamp
to keep its original bootstrap and sixteen-hour overall deadlines.

The default site configuration uses Darwin's `general` CPU partition,
`shared-redstone` NVIDIA partition, account `y2020-bf`, ORCA 5.0.4/OpenMPI 4.1.1,
and the existing `atomistic`, TorchSim CUDA, and `alf_alchemi_02` environments.
`--site overrides.json` overrides fields from `darwin_site()`; the resolved
configuration is saved in the artifact root. Environments are not installed
or modified by this runner. Training and QM use `atomistic`; only sampler
workers use the selected engine's environment. The unchanged training pool
requires a compatible complete four-GPU allocation and is checked in a
separate child process before training, without initializing CUDA in the
training worker before its pool forks.

ALCHEMI uses Python 3.11 while the coordinator uses Python 3.12. Its Parsl
transport process therefore uses the coordinator-compatible interpreter and
delegates sampler/prediction execution to the qualified ALCHEMI interpreter.
Only data objects cross this boundary in protocol-4 pickle files; Python
functions/code objects and Torch models are not transferred between versions.
Backend logs and actual computation-worker identities are retained. The extra
process startup is part of this validation harness, so its throughput is not
a standalone engine benchmark.

Launch from a Darwin login node in a persistent `tmux` session. The suite
allows four hours for preflights/bootstrap and four hours per engine group,
including queue waits. Each case has a 55-minute cap, including cleanup;
unused deadlines are not transferred to another stage. Resource ceilings are
one CPU allocation with two eight-core ORCA workers, one training allocation,
and one sampling allocation. Each GPU allocation has at most one block.
The CPU allocation reserves sixteen MPI slots. Each ORCA worker inherits an
exclusive eight-core affinity block; Open MPI leaves that affinity unchanged.

## Evidence and success

`events/` contains atomic sidecar records of actual submissions, completion,
worker imports/dependencies, builder geometry, selected structures, labels,
and queue occupancy. No synthetic labels or trainer results are used. The
preflight evaluates the same four structures individually and in a sequential
QM pack, retaining the configured QM repeat-evaluation tolerances. Before
each case, individual and packed predictions are compared on those recorded
structures at the approved float32 tolerances, including measured roundoff
bounds on uncertainty differences.

`training/model-NNNN.json` fingerprints the exact closed HDF5 files/IDs supplied
to each real training task, the resulting four checkpoint members, training
epochs, and retention/split counts parsed from their logs. Input-frame counts
and post-filter retention counts are reported separately. Model promotion
alone is insufficient: at least eight sampled IDs must have converged labels,
stored rows, and membership in the promoted model's training input. Stored
coordinates, cells, forces, units, and atom sorting are checked against QM
observations. Every new member is loaded and evaluated before acceptance.

Configured batching is insufficient too: a width-four sampler must actually
complete a four-trajectory task, and a packed-labeling case must complete a
four-structure QM task. Timeout partial batches are permitted, but a case
without a full-width execution remains incomplete.

The seed is copied into isolated case directories, including the original
training-input files and initial ensemble. Status files and pending work are
never copied. New labels/models are never shared between cases, and seed
fingerprints are checked again at acceptance. This tests twelve active-learning
branches from one real bootstrap, rather than twelve independent bootstraps.

The supervisor interrupts the infinite coordinator loop after verified success
or its deadline. It runs Parsl cleanup, cancels only Slurm job IDs journalled by
that run's provider, and verifies that they disappear. Unfinished tasks and
overshoot remain in the audit. Cleanup failure stops further allocations.

Each case writes `result.json`, `cleanup.json`, `prediction.json`, and logs.
The root `result.json` passes only when all twelve cases pass; natural
acquisition shortfalls, missing full-width executions, and timeouts remain
incomplete. Sampler-return-to-QM-start latency includes queue/packing delay;
it does not measure a frame's earlier wait inside an unfinished dynamics task.

## Remaining Increment 7 campaigns

The accepted core matrix is retained unchanged. Additional opt-in profiles use
`--campaign cohort|failures|pyseqm|mi250`; omit this argument for the original
core matrix. Case names are printed by preparation and appear under `cases/`.
Each profile supports `--prepare-only` and `--case NAME`.

```bash
python examples/simple_water_campaign/run.py \
  --campaign cohort --run-root /path/to/new/cohort \
  --import-seed /path/to/accepted/core/seed --prepare-only
python examples/simple_water_campaign/remaining.py \
  --run-root /path/to/new/remaining \
  --import-seed /path/to/accepted/core/seed
```

The second command runs all four campaigns sequentially. Launch it from a
persistent login-node tmux session with the established base interpreter and a
fixed source checkout. Each campaign has its own log and result. Every root
executes once. Repairs use a fresh root and the affected campaign's original
`execution.json` start via `run.py --budget-start`; never reset a failed
attempt's deadline or edit generated configurations. Other stages may proceed
after a failed case, but unverified owned-job cleanup stops all later work.

Seed import verifies the accepted bootstrap report, exact closed HDF5 inputs,
member fingerprints and epochs, scientific configuration, property units, and
all four checkpoint loads/predictions. The seed and its sibling `bootstrap/`
directory must remain available. Fragment libraries may move between source
snapshots only when their contents and builder settings match. Fresh case
copies never inherit status or pending work. PySEQM rejects seed import and
builds its own molecular bootstrap.

| Campaign | Cases and required execution | Elapsed ceiling |
| --- | --- | --- |
| cohort | Three engines × uncertainty/summed × stop/continue; dynamics/QM widths four; cohort eight, K four | 16 h, including four-hour setup and four hours per engine |
| failures | Label exception, unconverged label, and trajectory failure after an eligible candidate; real surviving labels/training | 4 h |
| pyseqm | ASE dynamics width four × native GPU labeling widths one/four; separate 32-frame molecular bootstrap | 4 h; bootstrap at most 1 h |
| mi250 | TorchSim ROCm dynamics widths one/four × ORCA widths one/four | 4 h |

Every active-learning case retains the 55-minute ceiling including cleanup,
eight-new-label lineage target, and four trained/loadable promoted members.
Water scientific settings remain unchanged. Summed ranking uses explicit
energy and force-RMS weights of one; it does not change eligibility. Cohort
cases require at least two completed cohorts; `continue` requires an observed
trajectory with multiple candidates. Audits capture candidates before pruning,
admission/model pinning, terminal reports, and the coordinator's selected child
IDs. Replay checks unpruned global top-K and grouping/completion-order parity.
The 50-trajectory replay fixture and controlled promotion tests are CPU contract
checks, not claims of width-50 hardware execution in these campaigns.

Failure injection exists only in validation wrappers, once for predetermined
`mol-0000-0000000000`. It is recorded separately from backend failures. The
unconverged-flag case requires that member's real ORCA evaluation to have
converged before the injected flag. Failed labels must never enter storage or
training, and a successful sibling from the same packed task must reach the
accepted training input. The trajectory failure preserves prior candidates;
one of those must actually be selected, labeled, and trained on. Missing that
natural selection remains incomplete. A separate real-Parsl preflight submits
three-member sampler and QM tails after 60 seconds without forced flushing.

PySEQM uses nonperiodic water/methanol, deterministic 0.03 Å perturbations,
AM1, float64, and SCF epsilon 1e-8. Its model includes carbon, requests only
energy/forces with unit factors one, and disables density schedules. Existing
water cutoffs and the 0.1 ps sampling cap remain in force. Preflight compares
identical inputs at actual native widths 1/2/4/8 against individual GPU calls,
with absolute limits 1e-6 eV and 1e-5 eV/Å, plus a force-sign finite difference.
Fallback-to-single calls cannot establish native batch coverage. Sampling,
building, and labeling share one serialized GPU worker allocation; training
uses its separate four-GPU allocation. A pinned PySEQM source path precedes the
stale installed package without modifying the environment.

MI250 changes only the sampler interpreter, ROCm initialization, and sampler
partition. Training remains in the NVIDIA base environment and ORCA remains
on the established CPU allocation. Scheduler visibility masks are preserved.

Candidate creation, task return, selection, QM start, and promotion timestamps
are separate events. Memory names distinguish generated candidate-array bytes,
process lifetime peak RSS (KiB), and task peak Torch allocated bytes; the latter
excludes Warp and other non-Torch allocations. Queue snapshots retain unfinished
work, and results report converged-label overshoot separately from accepted IDs.

The combined runner always keeps Increment 7 open while trained cell-input,
qualified potential-stability, or other required gates remain unresolved.
`runnable_campaigns_passed` reports the four executed campaign gates separately.
No replacement scientific fixtures are generated for missing inputs. MI350
testing is deferred by the user and is outside current Increment 7 acceptance.

For an early repair during the first cohort case, `remaining.py --budget-start`
accepts that original campaign's `execution.json` start. It retains the overall,
first ASE group, and first-case deadlines conservatively; repair time is not
refunded. Later stages that have not started receive their normal budgets.
This is not a general resume or durable-recovery interface.

To retry only failed stages after a completed run, supply e.g.
`remaining.py --campaigns pyseqm failures --run-root NEW_ROOT --import-seed SEED`.
The new authorized attempt has fresh bounded stage budgets and preserves the
earlier artifacts. `selected_campaigns_passed` reports just those requested
stages; it cannot establish `runnable_campaigns_passed` for all four campaigns.

The standalone partial-batch preflight creates its metadata directory before
submission. HIPPYNN training uses ALF's database adapter for groups below seven
atoms, padding only declared atom axes; larger-only datasets use the upstream
layout inference. This supports molecular water/methanol without modifying the
qualified HIPPYNN dependency or padding Cartesian/system properties. Native
nonperiodic frames retain ALF's storage convention with no cell dataset.
