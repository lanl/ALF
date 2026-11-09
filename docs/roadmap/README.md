# ALF development roadmap

Last updated: November 8, 2026.

This is the plan for ALF's `development` branch: what has been built, what has been shown to work on
real hardware, the decisions that shape it, and what comes next. Detailed work logs and validation
evidence are kept outside the repository.

## Status

"Shown on GPUs" means a bounded run on real hardware that exercised the workflow end to end at small
scale. These runs show the mechanics work. They don't establish model accuracy, long-run stability,
or performance at scale, except where a row says so.

| Area | Built | Shown on GPUs | Open |
| --- | --- | --- | --- |
| Batched labeling and sampling | Native batched labeling (PySEQM) and sequential packing for single-structure labelers. Batched MD on ASE, TorchSim (NVIDIA and AMD), and ALCHEMI. Uncertainty and summed ranking, first-trigger and pooled selection, top-K and farthest-point selection. | Every combination of single or batched dynamics with single or batched labeling, through bootstrap, storage, and retraining. | Result streaming, and replacing finished trajectories in flight. |
| Excited states | State, gap, and coupling storage. GPU4PySCF TDA labels (energies, forces, couplings, optical properties, C-PCM solvent) and PySEQM CIS labels. Multi-state HIPPYNN training with a shared trunk. Selected-state sampling and acquisition, chemical topology screening, UDD and wells, gap-seeking, and sampling windows. | Active-learning loops for acetylacetone (PySEQM) and a CPP9 ring (GPU4PySCF, in DCM). UDD, wells, and gap-seeking mechanics on all three engines. | Accuracy and long-time stability. Gap-seeking near real crossings. |
| Restarts | Atomic status and shard writes, a coordinator lock, finished labels held on disk until stored, reconciliation on restart, and spare seeds for a finite bootstrap. | 18 interruption and restart cases, including kills during a store and a finite bootstrap that used a spare. | A killed coordinator leaves its allocations for the scheduler. A training running at a crash starts over. |
| Multi-node training | One ensemble member per GPU across nodes, with retries of failed members only. | HIPPYNN with 8 members on 2 nodes and 20 on 5 nodes. MACE and UMA fine-tuning on 2 nodes, with a cancelled node retried. | Larger deployments. Data-parallel training of a single member. |
| Teacher labeling | Off-the-shelf MACE and UMA checkpoints as labelers. UMA's `omol` task refuses periodic structures unless opted in. | Both teachers on GPU, each with a real HIPPYNN student trained on its labels. | Teacher accuracy against DFT isn't claimed. |
| Fine-tuning | Adapters for HIPPYNN, MACE, and UMA. ALF selects the samples, and each round fine-tunes copies of the pretrained model on all labels so far. | The workflow for all three on DFT labels. Held-out accuracy improved for HIPPYNN and MACE. | UMA's held-out accuracy got worse; its recipe needs investigation. |
| PySCF labeling (`pyscf_task`) | One labeler for molecules and periodic cells, on CPU (PySCF) or GPU (GPU4PySCF), through ALF's `PySCFCalculator` ASE calculator. | GPU4PySCF ground-state DFT and the excited path. Molecular and k-point checks on GPU. | On GPU, periodic energies, forces, and stress missed their tight agreement checks, and the periodic student loop didn't finish. |
| CPP9 training size | The CPP example defaults to `hipnn64` at batch 64. | About 167 s per epoch at 40,000 training structures on a 40 GB A100, about 4 GiB allocated per member. | Accuracy at this size and batch, including the learning rate. |

## Design decisions

- **Change core ALF as little as possible,** especially `alframework/__main__.py`. New behavior is
  opt-in, existing task names, signatures, configurations, and storage conventions are preserved, and
  optional GPU and RDKit dependencies stay out of the base coordinator.
- **State names.** State-aware code names energies `E0, E1, …`, forces `F0, F1, …`, and gaps `dE01`,
  …. Ground-state labelers keep `energy` and `forces`. In `properties_list`, the key is what a labeler
  computes and the first element is the dataset name stored in the shard, so
  `"E0": ["energy", "system", 1.0]` lets a state-aware labeler feed ground-state trainers.
- **Labels are held outside `h5_dir` until a store,** so every ensemble member trains on the same shards.
- **A finite bootstrap counts only converged labels.** Spare seeds (`bootstrap_spares`) replace
  failed ones. If the spares run out, ALF trains on what converged, and stops only if nothing did.
- **Teachers are off-the-shelf checkpoints only.**
- **Fine-tuning restarts from the pretrained model every round,** on all labels so far, so rounds don't
  compound earlier fine-tuning.
- **PySCF and GPU4PySCF are the default QM for new examples,** since they're free and install with pip.
- **Single-node training stays the default;** multi-node training is a separate task.
- **Tests check ALF's wrappers and catch dependency changes,** not physics. Unit, sign, and convention
  checks use stand-ins with known numbers. Real QM runs once per code path ALF uses, asserting only
  that it runs and returns finite results of the right shape. Accuracy is checked with `--test_qm` and
  in qualification runs.
- **`development` is the integration branch.** No merge to `main` is planned.
- **Outside current acceptance:** legacy NeuroChem, and AMD MI350 hardware.

## Known limits

- **Allocations after a hard kill.** `SIGTERM` and Ctrl-C release ALF's allocations; a `SIGKILL` or a
  lost login node doesn't, and the jobs run until their walltime unless cancelled.
- **Training time grows with the data.** Each round trains from scratch on all labels, so trainings
  eventually outgrow a job's time limit, and a training killed at walltime is lost.
- **Packed inference memory.** HIPPYNN pads each task to its largest system, with no automatic split
  when a batch runs out of memory.
- **In-memory state.** Waiting batches and sampling cohorts aren't restored after a restart, and
  frames held only inside a crashed worker are lost.
- **Cohorts and strict batches can wait.** A cohort needs full admission, with no timeout, and a
  `full_only` batch can wait indefinitely if its group can't fill.
- **Stored QM labels aren't filtered.** Runtime force and topology screens apply to sampled frames only.
- **Node-local storage.** QM scratch still runs on shared storage. Where `/tmp` is a RAM-backed file
  system, as on the partitions measured, scratch there competes with host memory.
- **File I/O at scale.** Builders cache the replay manifest per worker, but trainers read every shard,
  each sampler task loads every model member, and metadata is written per structure.
- **UMA fine-tuning disk use.** fairchem's intermediate checkpoints take far more space than the final
  model while training runs.

## Next

1. **CPP9 at scale.** A full campaign on 10 GPU nodes (8 for labeling, 1 for training, 1 for
   sampling). Check `--test_qm` and `--test_ml` on the new cluster, then run a small pilot with one
   kill and restart, before the full run.
2. **UMA fine-tuning accuracy.** Find why held-out errors grew. The refit of per-element reference
   energies on few labels is one suspect.
3. **Running at scale.** A guide section on settings (`HDF5_USE_FILE_LOCKING`, `TMPDIR`,
   `PYSCF_TMPDIR`, `MPLCONFIGDIR`, Parsl run and log directories), then QM in node-local scratch, with
   failed calculations copied back to shared storage.
4. **Training time.** Cap epochs, or start each retrain from the previous ensemble; the HIPPYNN
   fine-tuning adapter may serve for the second.
5. **New uncertainty and acquisition methods.** Single-network uncertainty estimators alongside
   ensemble disagreement, custom selectors through import strings, score-aware diversity selection,
   and diversity against the existing training set. New signals carry their own names and cutoffs; the
   ensemble `Es`/`Fs` signals keep their meaning. Any selector must stay independent of task grouping
   and completion order.
6. **Long runs and staged campaigns,** last, and possibly dropped: durable cohorts and handoffs,
   staged configurations with fresh ensembles per stage, retained-label budgets, and filtering stored
   labels after QM.

## Not yet planned

Each needs its own decision: NPT sampling, periodic excited states, surface hopping, native forces for
mixed-composition CIS batches, grouped non-MD evaluations and reactive batching, a general
bias/control interface for future biased dynamics, data-parallel training of one member, and AMD MI350
hardware.

## Working on `development`

- Keep each change small, opt-in, and reviewable on its own, with tests. Leave `__main__.py`, task
  signatures, and existing example configurations unchanged unless a change needs them.
- The original examples must keep working with their own configurations and tasks.
  `tests/test_examples.py` loads every tracked example.
- Run the test suite three ways: with every optional dependency; with HIPPYNN but without TorchSim,
  ALCHEMI, RDKit, or PySEQM; and without Torch. CI runs the same three. The docs build must have no
  warnings.
- Run qualification work from a fixed source revision and environment, and keep generated data and
  evidence outside the repository.
- Record outcomes here as short summaries: what was shown, its limits, and what changed. Leave out
  site details such as job IDs, node and partition names, accounts, and file-system paths.
