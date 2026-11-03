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
