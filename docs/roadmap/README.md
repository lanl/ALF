# ALF development roadmap

Last updated November 27, 2026.

This is the plan for ALF's `development` branch: what has been built, what has been shown to work on
real hardware, the decisions that shape it, and what comes next. Detailed work logs and validation
evidence are kept outside the repository.

## Current focus on `ALF_NTK`

The [NTK roadmap](uncertainty-integration.md#current-priority-ntk) covers uncertainty development on this branch.
The [ground-state HIPPYNN path](uncertainty-integration.md#ground-state-hippynn-basis-and-sampling)
shares a prepared NTK basis across replay and sampling. The [benchmark handoff](uncertainty-integration.md#benchmark-handoff)
covers the fixed-pool QM-force study and packed GPU measurements on the target machine.

## ALF 2.0 case-study handoff

Both example workflows are ready for configuration and builder checks. The next milestone is
a short run on the target hardware that produces real reference labels, trains a model, and
completes a sampling and retraining round. Use the example READMEs for commands and setup.

| Case study | Available in the working tree | First compute-machine run |
| --- | --- | --- |
| [Chalcogenide adsorption](../../examples/surface_training/README.md#chalcogenide-adsorption-with-uma-oc25) | The built-in surface builder cuts zinc-blende (110) slabs from CdSe, CdTe, and HgTe bulk files and decorates them with methylamine or methylphosphine from SMILES. Each pose has 151 atoms and uses fixed-cell TorchSim dynamics with all atoms mobile. The pilot samples the six slab–ligand pairs with equal probability, targets 48 accepted labels with up to 12 spares, and keeps HgSe in a separate holdout config. Full UMA-S 1.2.1 OC25 fine-tuning starts from the same foundation checkpoint each round. | Set up VASP, the licensed PAWs, the UMA checkpoint, Fairchem, and TorchSim on one CUDA worker. Qualify the bulk references and RPBE-D3 settings, including a matched SOC check on HgTe and HgSe. Run one VASP label, then verify bootstrap training, batched MD, and a second fit that uses newly acquired labels. |
| [Mixed-molecule excited states](../../examples/molecular_excited_states/README.md#broader-chromophore-pilot) | The shared RDKit module prepares a 35-molecule chromophore panel with 27 training molecules and eight whole-molecule holdouts, plus a separate optional validation panel. The builder cycles through training molecules for 216 initial candidates, with up to 108 spares. The preset starts with 64-feature HIP-HOP-NN, supports up to five excited roots, and enables optical labels. The carbonyl \(S_0/S_1\) preset is a smaller workflow check. | Set up GPU4PySCF, HIPPYNN, RDKit, and ALCHEMI. Qualify 6-31G* and 6-31+G* on representative relaxed and distorted chromophores, including larger training molecules. Use the root, gradient, convergence, time, and memory checks to choose the production state range. Verify one real label and a complete active-learning round before scaling. |

### Comparisons after the short runs

1. **Surfaces.** Compare pretrained, seed-only, iterative, and fixed-pool UMA against the same
   independent VASP poses. Report accepted counts by slab–ligand pair and match the label
   budgets where applicable. Evaluate relative pose energies, ligand and near-site forces,
   and HgSe transfer. Compare fixed-cell trajectories
   from matched starts using donor distance and tilt. Run ordinary batched MD before the separate
   UDD comparison and check the direction and strength of the bias.
2. **Molecules.** Compare 64 and 145 features on identical labels and pick the width on validation
   molecules. Use the selected width for ensemble, NTK, and random acquisition with matched
   accepted label counts and propagation settings. Compare ordinary adiabatic MD and later
   gap-seeking in a separate experiment. Report state-resolved force and gap errors, small-gap
   yield, and labeling and sampling cost. The expansion target is 120 training, 18 validation,
   and 24 test molecules. Supply that larger manifest through the existing panel input.
3. **Molecular properties.** After the dynamics checkpoint is qualified, fit the optional
   oscillator-strength readout on saved optical labels. For neutral partial charges, qualify
   the smaller \(S_0/S_1\) permanent-dipole dataset, then fit the charge readout through the
   dipole loss. Evaluate both readouts on separate molecules. Keep their results with the
   parent dynamics checkpoint and label provenance.

### Validation and transfer

- Local checks cover both examples' configuration and builder commands; the surface builder's
  bulk cutting, element anchors, fixed cell and frozen atoms; a CPU HipNN train–sample–retrain
  cycle with stand-in labels; the readouts against stand-in solvers; and single-model NTK
  gap-seeking in ASE, TorchSim and ALCHEMI. The non-slow suite passes, and the docs build with
  warnings as errors.
- Before transfer, make the commit containing the examples, tests, and roadmap available on the
  destination machine, or transfer the complete working tree. A branch checkout only contains
  committed files. Record the exact revision and any patches used for each run.
- Archive resolved configurations, panel and pose manifests, PAW and checkpoint identifiers,
  software versions, split and label inventories, attempted and accepted label counts, timing and
  memory records, and the evaluation outputs. Use qualified runs and independent results to update
  the manuscript figures and conclusions.

## Status

"Shown on GPUs" means a bounded run on real hardware that exercised the workflow end to end at small
scale. These runs show the mechanics work. They don't establish model accuracy, long-run stability,
or performance at scale, except where a row says so.

| Area | Built | Shown on GPUs | Open |
| --- | --- | --- | --- |
| Batched labeling and sampling | Native batched labeling (PySEQM) and sequential packing for single-structure labelers. Batched MD on ASE, TorchSim (NVIDIA and AMD), and ALCHEMI. Uncertainty and summed ranking, first-trigger and pooled selection, top-K and farthest-point selection. | Every combination of single or batched dynamics with single or batched labeling, through bootstrap, storage, and retraining. | Result streaming, and replacing finished trajectories in flight. |
| Excited states | State, gap, and coupling storage. GPU4PySCF TDA labels (energies, forces, couplings, optical properties, C-PCM solvent) and PySEQM CIS labels. Multi-state HIPPYNN training with a shared trunk. Selected-state sampling and acquisition, chemical topology screening, UDD and wells, gap-seeking, and sampling windows. | Active-learning loops for acetylacetone (PySEQM) and a CPP9 ring (GPU4PySCF, in DCM). UDD, wells, and gap-seeking mechanics on all three engines. | Accuracy and long-time stability. Gap-seeking near real crossings. |
| Restarts | Atomic status and shard writes, a coordinator lock, finished labels held on disk until stored, reconciliation on restart, and spare seeds for a finite bootstrap. | 18 interruption and restart cases, including kills during a store and a finite bootstrap that used a spare. | A killed coordinator leaves its allocations for the scheduler. A training running at a crash starts over. |
| Multi-node training | Independent ensemble members across nodes, with retries of failed members only. HIPPYNN also supports data-parallel training of one member across GPUs and nodes. | HIPPYNN with 8 members on 2 nodes and 20 on 5 nodes. MACE and UMA fine-tuning on 2 nodes, with a cancelled node retried. [Single-member scaling measurements](../source/user_guide/running_at_scale.rst). | Scaling and end-to-end qualification on the target campaign. |
| Teacher labeling | Off-the-shelf MACE and UMA checkpoints as labelers. UMA's `omol` task refuses periodic structures unless opted in. | Both teachers on GPU, each with a real HIPPYNN student trained on its labels. | Teacher accuracy against DFT isn't claimed. |
| Fine-tuning | Adapters for HIPPYNN, MACE, and UMA. ALF selects the samples, and each round fine-tunes copies of the pretrained model on all labels so far. | The workflow for all three on DFT labels. Held-out accuracy improved for HIPPYNN and MACE. | UMA's held-out accuracy got worse; its recipe needs investigation. |
| PySCF labeling (`pyscf_task`) | One labeler for molecules and periodic cells, on CPU (PySCF) or GPU (GPU4PySCF), through ALF's `PySCFCalculator` ASE calculator. | GPU4PySCF ground-state DFT and the excited path. Molecular and k-point checks on GPU. | On GPU, periodic energies, forces, and stress missed their tight agreement checks, and the periodic student loop didn't finish. |
| CPP9 training size | Explicit HIPPYNN kernel selection and configurable model, batch, evaluation batch, and epoch limit. | All 1,000 DCM bootstrap labels converged and were stored. Four HIP-HOP64 members each completed three epochs, evaluation, and checkpoint reloads on those labels, using Triton, training/evaluation batch 200, and one 40 GB A100 per member. Peak sampled device use was 31.8 GiB. Dense PyTorch at batch 200 failed while requesting a single 51.3 GiB intermediate. | Full 50,000-label training, the projected 200-epoch walltime, predictive accuracy, and long-run stability remain unqualified. |

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
- **The HIPPYNN kernel is part of resource qualification.** `pytorch` and `triton` select
  implementations of HIPPYNN's message-passing operations, not different networks or loss functions;
  PyTorch still handles training and differentiation. Triton avoids large dense intermediates and
  can improve GPU throughput, with compilation/autotuning overhead and small floating-point
  differences from the order of reductions. A memory or speed improvement doesn't establish better
  predictive accuracy. Benchmarks must explicitly select and record the kernel actually used by
  each worker role, along with the precision and graph settings.
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
- **Benchmark and production kernels can differ.** The older CPP example benchmark didn't apply the
  configured `hippynn_kernel`: HIPPYNN chose its default accelerated implementation, while production
  explicitly used dense PyTorch. The resulting batch-200 memory estimate didn't qualify that
  production configuration. Selecting Triton explicitly passed real four-member training on the
  saved bootstrap data. The CPP example benchmark now selects and records its kernel explicitly,
  as does the newer `python -m alframework.benchmark`;
  still verify a short real training, evaluation, and checkpoint reload before scaling. Geometry
  and neighbor counts, batch size, force derivatives, precision, and kernel all affect memory;
  dataset size alone doesn't determine GPU use.
- **Allocated GPUs aren't necessarily usable.** A training allocation advertised four GPUs through
  NVML, but CUDA initialization found only three and every member failed before training. The node
  was independently marked for draining because of PyTorch crashes; this doesn't prove a physical
  card failure. Site recovery excluded the node and added fresh-process CUDA initialization and a
  small computation on every training GPU before worker registration. The 1,000 stored labels were
  retained for the retry.
- **Packed inference memory.** HIPPYNN pads each task to its largest system, with no automatic split
  when a batch runs out of memory.
- **In-memory state.** Waiting batches and sampling cohorts aren't restored after a restart, and
  frames held only inside a crashed worker are lost.
- **Cohorts and strict batches can wait.** A cohort needs full admission, with no timeout, and a
  `full_only` batch can wait indefinitely if its group can't fill.
- **Stored QM labels aren't filtered.** Runtime force and topology screens apply to sampled frames only.
- **Node-local storage.** QM scratch still runs on shared storage. Where `/tmp` is a RAM-backed file
  system, as on the partitions measured, scratch there competes with host memory.
- **File I/O at scale.** Builders cache the replay manifest and sampler workers reuse their last
  ensemble, but trainers still read every shard and metadata is written per structure.
- **UMA fine-tuning disk use.** fairchem's intermediate checkpoints take far more space than the final
  model while training runs.

## Next

1. **CPP9 at scale.** Complete the campaign with a ceiling of 10 GPU nodes (8 for labeling, 1 for
   training, 1 for sampling). The earlier pilot exercised a coordinator kill/restart and resumed
   labeling. The 1,000-label DCM bootstrap and three-epoch HIP-HOP64/Triton training qualification
   are complete; the production configuration
   uses training/evaluation batch 200, at most 200 epochs, and a 50,000-label stopping threshold.
   Confirm production training, sampling, and subsequent retraining, then measure training walltime
   and peak memory as the dataset and sampled geometries grow. Verify actual executor ports from
   compute nodes, bound repeated worker-registration failures, and check allocated GPUs before
   accepting a worker; successful node allocation alone doesn't establish worker health.
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
bias/control interface for future biased dynamics, and AMD MI350
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
