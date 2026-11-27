# Uncertainty integration in ALF

Use this guide to qualify uncertainty providers on `ALF_NTK`.
The [acquisition guide](../source/user_guide/acquisition.rst)
defines the score fields and selection settings. ALF thresholds named scalar
scores and keeps the corresponding atom arrays with selected structures.

## Current priority: NTK

The first goal is to establish whether one potential's NTK force scores improve
frame selection at a fixed QM labeling budget and what the scoring costs.

1. **Use the exact training split and make saved bases reusable.** The HIPPYNN
   excited-state loader now fits the train indices recorded in each member's
   `training_input_manifest.json` and checks the checkpoint, split fingerprint,
   and settings on cache reuse. The core also reads the previous version 2 basis
   format. ALF now writes separate X-MACE training and validation files and
   ties the basis to their split manifest. Full and LoRA tests train and fit
   the basis on the same 15 of 16 frames. Retrain older ALF X-MACE members
   without a verified split before benchmarking. The [ground-state
   HIPPYNN basis lifecycle](#ground-state-hippynn-basis-and-sampling) uses explicit
   preparation and a shared replay and sampler loader.
2. **Benchmark force-based acquisition on a fixed pool.** Start with one trained
   ground-state HIPPYNN potential and a pool with QM forces. Keep the evaluation
   frames out of model and basis fitting, including the models used for the
   ensemble baseline. Use a separate calibration split if tuning score settings.
   Compare the per-atom NTK force score with ensemble disagreement and local
   environment novelty on identical frames. Report large force errors missed at
   equal frame budgets and score time. Keep the pool, error threshold, and frame
   budget fixed across methods.
3. **Measure GPU cost before changing the batching strategy.** Profile the same
   workload on one supported GPU. Record basis fitting time, scoring time per
   check, end-to-end sampler throughput, and peak GPU memory across trajectory
   batch sizes. Compare exact and projected force scores where useful. Use those
   results to decide which operations to batch across structures, then compare
   fixed and adaptive selection on the same pool and label budget.

## Benchmark handoff

Run the larger measurements on the target machine. Use a benchmark directory
outside the repository for inputs, scores, logs, and profiles.

1. **Pin the source and environment.** Record the ALF revision, any applied
   patches, Python and package versions, GPU type, driver, and device assignment.
   Start with HIPPYNN. Use an editable install of that ALF checkout and the target
   machine's PyTorch GPU build. Use the same HIPPYNN revision for training,
   preparation, and sampling so the serialized checkpoint graph loads. TorchSim
   is the packed route on CUDA and ROCm.
   ALCHEMI is an additional packed route on NVIDIA CUDA. Both use `cuda:0` as a
   PyTorch device string. On one allocated GPU, run these checks from the repo root.

   ```sh
   python -m pytest tests/test_ntk_core.py tests/test_adaptive_ntk.py
   ALF_TEST_NTK_DEVICE=cuda:0 python -m pytest tests/test_hippynn_ntk.py
   python examples/atom_resolved_uncertainty/ntk_replay.py --device cuda:0 --engine torchsim
   ```

   For an ALCHEMI run, repeat the replay with `--engine alchemi` in that worker
   environment. These checks use synthetic examples. Save their logs as
   environment qualification evidence.
   To qualify a single fine-tuned model, run the corresponding test on one
   allocated GPU. Use a node-local fixture directory when shared storage
   disallows HDF5 file locks, as on the Frontier home filesystem.

   ```sh
   export ALF_TEST_NTK_DEVICE=cuda:0
   python -m pytest tests/test_fine_tuning.py \
     -k 'prepares_single_member_ntk and cuda' --basetemp="$(mktemp -d)"
   # In the X-MACE environment:
   python -m pytest tests/test_fine_tuning_xmace.py \
     -k 'fine_tuning_then_ntk_end_to_end and cuda' --basetemp="$(mktemp -d)"
   ```

   The HIPPYNN test fine-tunes one LoRA model, fits on the filtered training
   rows, and checks selected-atom NTK acquisition through TorchSim. The X-MACE
   test checks fine-tuning, basis preparation on the assigned GPU, basis reload,
   and multi-state energy, gap, force, and projected Hessian scoring.
2. **Prepare one real fixed pool.** Start with the [public QM benchmark](#public-qm-benchmark).
   Pin one deployed ground-state checkpoint,
   its exact member training rows, and the ensemble checkpoints for the baseline.
   Prepare a QM-labeled pool with stable frame IDs, atomic numbers, coordinates,
   cells and periodicity where applicable, and forces in eV/Å in the same atom
   order as each structure. Keep the pool disjoint from every model's training
   rows and from the NTK and novelty reference sets. Keep a separate labeled
   calibration set if choosing score settings or thresholds from data. Save the
   checkpoint and data hashes, ordered frame IDs, and split definitions.
3. **Score the pool before starting MD.** Train with `record_ntk_training_split`
   enabled and use the [preparation command](#ground-state-hippynn-basis-and-sampling).
   For other real inputs, use a study driver around the prepared estimator. Use
   `predict(atoms, forces=True)` to get the deployed forces and exact per-atom
   scores, and `score_forces(structure, force_probes=8)` for projected RMS on the
   same basis. Rank exact NTK by the maximum atom score and projected NTK by the
   frame RMS. Verify prediction agreement with the deployed calculator on a few
   frames before ranking the full pool. Rank frames by the maximum atom score
   for the two other baselines. For ensemble disagreement, use the norm of the three
   component standard deviations across members for each atom. Use
   `LocalEnvironmentNovelty.score_atoms` with reference environments from
   training. Define the reference frame error
   as the maximum atom norm of the deployed model's force error against QM, and
   use that same error for every ranking. Fix the error threshold and frame
   budgets before evaluating the pool. Break score ties by frame ID and report
   the number and fraction of large-error frames missed at each budget. Save
   the per-frame results so the budget curves can be reproduced without rescoring.
4. **Measure packed MD on the target hardware.** Start with one GPU. Keep the
   checkpoint, frames, step count, check interval, precision, and reporting fixed.
   Record MD-only throughput, throughput with acquisition checks, synchronized
   scoring time per check, and peak GPU memory as trajectories per GPU increase.
   Separate checkpoint load, basis fitting, and warmup from steady-state timings.
   Compare batched sampler decisions and atom arrays with single-frame replay at
   the same geometries before scaling. Record the actual HIPPYNN kernels used in
   MD and in NTK, along with failures and memory limits. Extend to the target
   allocation using the verified single-GPU configuration.

Keep the command, resolved settings, basis ID and metadata, force-error threshold,
frame scores and selections, timing and memory measurements, and raw logs with
the study. Add the result and its limits to the compatibility notes below.

### Public QM benchmark

Run [revised MD17 aspirin](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)
first and [MD22 Ac-Ala3-NHMe](https://www.sgdml.org/docs/datasets.html) second.
Download each official NPZ into a benchmark directory. Use the revised aspirin
file with `old_indices`. Start with a fixed group of atoms around the aspirin
ester or one peptide bond, chosen by inspecting the input atom order before
scoring. The commands below use the same HIPPYNN environment for all steps.
For Figshare v4 aspirin (`rmd17_aspirin.npz`, file ID `62265757`), the zero-based
ester indices are `11 8 12` for the carbonyl carbon, carbonyl oxygen, and
bridging oxygen. The CPU smoke run verified this assignment by connectivity
across its 48 training candidates.

```sh
work=/path/to/benchmark/aspirin
driver=examples/atom_resolved_uncertainty/benchmark_ntk.py
python "$driver" prepare --dataset rmd17-aspirin \
    --source "$work/rmd17_aspirin.npz" --out "$work/data"
```

For the peptide, use `--dataset md22-ala` and a separate work directory.
Preparation keeps the archive atom order, converts the labels to eV and eV/Å,
and records the input checksum, source row IDs, and trajectory ordering.
Training, calibration, and test draws come from the first 60%, next 20%, and
last 20% of the ordered frames. The defaults draw 1,000 training candidates,
256 calibration frames, and 1,024 test frames after excluding 1,000 trajectory
indices after each boundary. rMD17 uses its original MD indices. MD22 uses
archive row order, which is the assumption for the blocked split. These are
replay splits for acquisition evaluation.

Train at least two members from `data/training` with a different seed per member
and with exact training-row recording enabled. This four-member example gives
one reproducible starting configuration.

```sh
python - "$work" cuda:0 <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import torch
from alframework.ml_interfaces.hippynn.interface import train_HIPNN_model

work = Path(sys.argv[1]).resolve()
device = sys.argv[2]
data = json.loads((work / "data/dataset.json").read_text())
network = dict(possible_species=[0, *sorted(set(data["atomic_numbers"]))],
               n_features=16, n_sensitivities=10, n_interaction_layers=2,
               n_atom_layers=2, dist_soft_min=0.5, dist_soft_max=4.5, dist_hard_max=5.5)
for index in range(4):
    torch.manual_seed(17 + index)
    np.random.seed(17 + index)
    member = work / "models" / f"model-{index:02d}"
    member.mkdir(parents=True)
    train_HIPNN_model(
        str(member), str(work / "data/training"), energy_key="energy",
        coordinates_key="coordinates", species_key="species", force_key="forces",
        network_params=dict(network), device_string=device, record_ntk_training_split=True,
    )
PY
```

Use member 0's `T-RMSE` and `ForceRMSE` from
`final_metric.valid` in `models/model-00/training_summary.json` for the basis
scales. Set `atom_indices` to the chosen zero-based indices as a space-separated
list before running replay.

```sh
read -r energy_scale force_scale < <(
    python - "$work/models/model-00/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

valid = json.loads(Path(sys.argv[1]).read_text())["final_metric"]["valid"]
print(valid["T-RMSE"], valid["ForceRMSE"])
PY
)
python -m alframework.ml_interfaces.hippynn.ntk_basis prepare "$work/models" --member 0 \
    --device cuda:0 --energy-scale "$energy_scale" --force-scale "$force_scale"
python "$driver" score --data-dir "$work/data" --model-dir "$work/models" --member 0 \
    --device cuda:0 --indices $atom_indices --out "$work/replay"
```

Replay verifies that each ensemble checkpoint used the prepared training
shard and loads the selected member's saved basis. It compares selected-atom
ensemble disagreement, exact selected-atom NTK, projected whole-frame NTK,
and selected-force NTK with total-energy uncertainty. The common force error
is the largest norm of the deployed member's force error on a selected atom.
Energy errors use a constant offset fitted on calibration frames. The 90th
percentiles on calibration set the large-error thresholds and the joint
ranking scales. The combined score takes the larger normalized score, as in
ALF's `largest` strategy.

Keep `dataset.json`, `calibration.csv`, `test.csv`, and `summary.json`.
The summary records the frames picked and large errors missed at 1%, 5%, and
10% labeling budgets. It also gives warmed per-frame timings. The ensemble
timing covers one frame's member predictions, and the NTK timings cover each
sensitivity request. Use the packed MD handoff above for end-to-end
throughput, peak GPU memory, and the cost of scoring at the MD check interval.

## Ground-state HIPPYNN basis and sampling

Set `record_ntk_training_split` in the ML configuration. The ordinary and
data-parallel trainers record the exact training rows and source fingerprints
after screening and splitting. Prepare one basis after training. The scales
below are illustrative and the fit should use scales chosen from validation data.

```sh
python -m alframework.ml_interfaces.hippynn.ntk_basis prepare models/model-0000 \
    --member 0 --device cuda:0 --energy-scale 0.05 --force-scale 0.10
```

Use `load_hippynn_ground_state_ntk_estimator` for replay and
`load_hippynn_ground_state_ntk` as the packed calculator loader from
`alframework.ml_interfaces.hippynn.ntk_basis`. Compare `force_score="projected"`
with eight probes and `force_score="exact"` against the same artifact. For a
projected run use `ntk_force_rms`. For an exact run use `ntk_force_max` and retain
the atom arrays for analysis. See the [acquisition configuration](../source/user_guide/acquisition.rst#prepared-ground-state-hippynn-force-ntk).
For older checkpoints with independently recorded training geometries, the
explicit-geometry estimator API remains available.

To compare force uncertainty at a reaction site, use the same
`acquisition.force_atoms` selection for ensemble and NTK. Fixed indices and
per-structure ASE boolean masks are supported. Use `ensemble_force_selected_max`
for ensemble disagreement and `ntk_force_selected_max` with
`force_score="selected"` for prepared HIPPYNN. The latter requests three
sensitivity rows per selected atom. Optional `energy_score=true` adds one
total-energy row. When comparing selected-force scores against QM, use the same
selected atom set to define the reference force error for each frame. Measure
acquisition check time against the full exact and projected force modes.

## Local CPU timing and integration checks

On 2026-11-26, short TorchSim runs used one AMD EPYC 7763 CPU thread, four MD
steps per check, and medians from five warmed cycles. With eight 21-atom
rMD17 aspirin trajectories, the whole-batch check took 2.5 ms for a two-member
ensemble, 45 ms for eight projected force probes, 234 ms for exact scores on
three selected atoms, and 241 ms for exact scores on every atom. One-member
and two-member MD alone took 3.4 and 6.0 ms per packed step, respectively.

In a separate single-frame replay with the same 16-dimensional aspirin basis,
`auto` picked forward mode for the nine selected force components. Reverse
mode cut the median from 28 to 5.4 ms with matching scores on four held-out
frames. Compare both differentiation backends at realistic sketch widths,
atom counts, and trajectory widths on the target GPU before changing `auto`.

With four synthetic three-atom, four-state X-MACE trajectories, the packed
check took 0.32 s for energy and gap scores, 0.87 s for exact per-state force
scores, and 7.44 s for two projected Hessian probes per state. The timing basis
used the 12 explicit training frames and was tested on separate validation
frames. A separate sampler check accepted `ntk_dE12` on the fixed S1 loader
and identified a missing multi-state evaluator hook. NTK now exposes physical
state predictions and forwards scoring through the state-selection interface.
Use `scores.gap_pairs` such as `[[1, 2], [1, 3]]` for coupling-aware acquisition.
The HIPPYNN CPU regression checks selected-state dynamics and nonadjacent
gap scores in ASE, TorchSim, and ALCHEMI. A fine-tuned X-MACE CPU smoke run
selected a held-out frame through the same state-aware TorchSim sampler.

## Fine-tuning and GPU qualification

On 2026-11-26, the following tests passed on one AMD Instinct MI250X with
PyTorch 2.10.0 and ROCm 7.1.

- Ground-state HIPPYNN LoRA fine-tuned one member with `fine_tune.ntk` enabled.
  It prepared a float32 basis on the 13 filtered training rows, restored it
  on GPU, and selected a frame with exact two-atom force NTK in TorchSim.
- Excited-state X-MACE LoRA trained and prepared a float32 basis on its 15
  training frames using the assigned GPU. It reloaded the basis and scored
  state energies, forces, gaps, and projected Hessians on GPU.
- Separate fixed-pool runs scored 64 held-out rMD17 aspirin frames on the same
  saved float64 basis with sketch width 16. `--ntk-engine reverse` reduced
  median exact three-atom score time from 34.1 to 9.0 ms per frame and joint
  force/energy scoring from 34.2 to 10.6 ms. The two-member ensemble evaluation
  took 8.6 ms per frame. Both NTK backends selected the same frames at every
  tested budget, with score differences below 1.2e-14 across calibration and
  test. These are scoring timings on a small pool. Measure packed throughput
  at the desired check interval and compare missed force errors on a larger pool.

Use the opt-in `fine_tune.ntk` block for automatic HIPPYNN basis preparation
and `ntk_engine` in that sampler's `batched_calculator_options` to reproduce
the backend comparison. Ground-state MACE has a stand-alone NTK adapter.
Integrating its exact training split and automatic fit into the fine-tuning
task is still open.

### Single-Model UMA

For conservative UMA-S `omol`, ALF supports frozen-backbone and radial LoRA
fine-tuning. `fine_tune.ntk` fits a single-member basis on the exported
training database. TorchSim runs one EMA model and evaluates NTK at checks.
LoRA export writes a native `uma.pt` alongside EMA factors for NTK. CPU and
Frontier MI250X smokes with Fairchem 2.21 and UMA-S 1.2 passed two training
steps in each mode, exact frozen-weight checks, stock Fairchem prediction
parity, rotation and atom permutation checks, finite differences for energy
and force NTK rows, and frame selection through TorchSim. The Frontier test
also rebuilt the unmerged LoRA model from the retained resume configuration
and loaded its final training weights. Prepare larger labeled pools and pin
the Fairchem version for production qualification.

To reproduce the GPU comparison, take an aspirin data directory from the
[public QM benchmark](#public-qm-benchmark). Run from the ALF checkout on one
GPU. Use a node-local directory for model artifacts and retain the summary and
per-frame scores.

```sh
python -m examples.uma_fine_tuning.benchmark_ntk train \
  --checkpoint "$uma_checkpoint" --dataset-dir "$aspirin_data" \
  --out "$node_work/uma-aspirin" --device cuda:0
python -m examples.uma_fine_tuning.benchmark_ntk measure \
  --model-dir "$node_work/uma-aspirin/models" --dataset-dir "$aspirin_data" \
  --out "$report_dir" --device cuda:0 --indices 11 8 12 --width 4
```

The benchmark fine-tunes two LoRA members on the same input pool with separate
validation splits. It reports basis fit time, matched-budget coverage against
QM, and synchronized timing and peak memory for two-member disagreement and
one-member NTK. It uses the same selected atoms and four MD steps per check.

The pair was trained and the same checkpoints were replayed on one MI250X
with Fairchem 2.21, PyTorch 2.8, and ROCm 6.4. The
44-frame basis took 54.4 s to prepare, including checkpoint load and basis
save, with 5.08 GiB peak GPU allocation. On 64 test frames, exact three-atom
NTK and a two-member ensemble each missed 4, 1, and 0 of 5 large force errors
at budgets of 1, 4, and 7 frames. Eight projected whole-frame probes missed
all five at those budgets. At batch width 4, exact NTK achieved 18.0
trajectory-steps/s with 1.91 GiB peak allocation. The ensemble achieved 16.6
trajectory-steps/s with 2.59 GiB. These results use one small aspirin split
and four fine-tuning steps per member. Repeat on larger pools and check
intervals before drawing conclusions about ranking quality or throughput.

## Single-model uncertainty across architectures

The goal is to decide which frames need QM labels while running one potential.
Prioritize providers that use atomic geometry and standard calculator outputs
so the same implementation works across model architectures. Score individual
atoms and use a named reduction, such as the largest atom score, for frame
acquisition. NTK and `uq-mlip` are single-model methods that need
architecture-specific access to parameter derivatives or internal model
features. NTK keeps that access in a small adapter: the core
(`alframework.ml_interfaces.ntk`) works for any model that returns energies,
and adapters exist for HIPPYNN and MACE. Both methods use the same acquisition
interface.

Two portable candidates to evaluate are local environment novelty, measured
by distance to training environments of the same element, and a per-atom force
error estimator fit to QM residuals using local geometry and the potential's
predicted forces. Fit the error estimator on structures excluded from potential
training and evaluate it on a separate test set.

Test each portable provider with two model architectures through the same
acquisition settings. Use per-atom ensemble force disagreement as a reference
and compare missed large force errors and scoring time at equal QM labeling
budgets on the same held-out structures.

### GPU-batched scoring

New portable providers should score a batch of active trajectories at each
acquisition check in TorchSim and ALCHEMI. Use packed device tensors for atomic
numbers, positions, cells, periodicity, and structure boundaries. Reuse the
unbiased forces already computed for those positions when the provider needs
them. Compute per-atom scores and frame reductions on the GPU, return the
frame scores to acquisition, and keep the atom arrays in input order for frames
retained by sampling. ASE can reuse the scoring code for one frame.

The packed engine's `observe_uncertainty_batch` hook receives the current geometry
and unbiased predictions for all active structures. Local environment novelty
scores the packed structures together. NTK accepts the same hook and computes
derivative rows one structure at a time. The GPU profile above should measure
this scoring cost separately from ordinary batched energy and force inference.

Qualify GPU scoring with mixed atom counts and supported periodic inputs.
Compare scores and selection decisions against single-frame replay. Measure
scoring time, end-to-end sampler throughput, and peak GPU memory as the number
of trajectories per GPU grows. Target TorchSim on NVIDIA CUDA and AMD ROCm,
and ALCHEMI on NVIDIA CUDA.

## What the providers measure

| Provider | Inputs and fitted artifacts | Output and ALF use |
| --- | --- | --- |
| Local environment novelty | Reference geometries and an element-resolved radial descriptor | Distance to the nearest same-element reference environment per atom, with a frame maximum for acquisition. The score needs calibration against the deployed model's force errors. |
| Energy, force, and Hessian NTK | One HIPPYNN graph (molecular or periodic, one or more states) or MACE model, its original training geometries, and energy, force, and optional Hessian scales | Posterior standard deviations of the linearized model: energies, energy gaps, every force component (per-atom scores and a frame maximum), or k random projections for the RMS force and Hessian-element uncertainty (`ntk_force_rms`, `ntk_hessian`, Projected Hessian Learning probes), and optional per-atom Hessian scores. Geometries trained on fixed-probe or full Hessians add Hessian-vector-product rows to the fit. Compare with quantum forces and Hessians to establish what a warning means. |
| Energy AdaptNTK | One molecular HIPPYNN energy graph and a saved basis fit on its original training geometries | One energy-gradient row and initial energy score per frame. The coordinator uses the rows to update scores after each selection. Per-atom force NTK can be reported by the same evaluator. |
| PNNL `uq-mlip` | A MACE checkpoint and a quantile regressor fit on its frozen per-atom embeddings and internal atomic energies | `uq_mlip` is the interval half-width for each atom's internal energy. `uq_mlip_max` is the scalar acquisition score. The interface paper studies a relative width normalized by the sum of the bounds. These quantities have different scales and require separate cutoffs. Compare the package score with held-out reference forces before using it as a force-error warning. |

Read the [interface study](https://doi.org/10.1038/s41524-026-02051-8),
[PNNL package](https://github.com/pnnl/UQ-MLIP), and
[AdaptNTK preprint](https://arxiv.org/html/2609.00488v1#S4.SS2) for the methods.
The ALF integration targets MACE for `uq-mlip`; PNNL also publishes UMA and
CHGNet extractors. Those backends require separate qualification.

## Compatibility and evidence

**V** verified for the numbered smoke test, **I** implemented with hardware
verification pending, **P** planned, **—** unsupported in the current path.
An entry such as CPU/CPU/CPU gives the MLIP, embedding or gradient extraction,
and score devices. A separate CPU selection stage is indicated where relevant.

| Provider and ALF path | Geometry | CPU | NVIDIA CUDA | AMD ROCm | Apple MPS |
| --- | --- | --- | --- | --- | --- |
| Local novelty replay, ASE MD, packed TorchSim and ALCHEMI MD | Molecules and full-rank periodic cells | V [6] | I | I via TorchSim | — |
| Force NTK replay and ASE MD | Molecule | V [1] [7] | I | I | I (V [1] before the [7] rewrite) |
| Force NTK packed TorchSim | Molecule | V [2] [7] | I | I | — |
| Force NTK packed ALCHEMI | Molecule | V [2] | I | — | — |
| HIPPYNN NTK replay: energy, force, projected Hessian, and gap scores | Molecule, periodic solid, multi-state molecule | V [7] | I | I | I |
| MACE NTK replay: energy, force, and projected Hessian scores | Molecule, periodic solid | V [7] | I | I | P |
| X-MACE NTK replay: state energies, gaps, forces, projected Hessians, and smooth outputs, with crossing flags | Molecule | V [8] | I | I | P |
| X-MACE NTK replay | Periodic solid | P | P | P | P |
| X-MACE fine-tuning from a ground-state MACE, with the NTK basis fit after training (full and LoRA) | Molecule | V [9] | I | I | — |
| HIPPYNN excited-state ensemble member NTK loader, with a cached basis | Molecule | V [9] | I | I | — |
| NTK for HIPPYNN coupling models on a frozen energy model (outputs adapter, physical-coupling uncertainty) | Molecule | V [10] | I | I | — |
| NTK on LoRA fine-tuned HIPPYNN copies (linearized in the LoRA factors and heads) | Molecule | V [10] | I | I | — |
| NTK fit with Hessian-vector-product rows (fixed probes or full Hessians) | Molecule | V [7] | I | I | I |
| Energy AdaptNTK fixed pool | Molecule | V [3] | I | I | V [3] |
| Energy AdaptNTK TorchSim cohort | Molecule | V [3] | I | I | — |
| Energy AdaptNTK ALCHEMI cohort | Molecule | V [3] | I | — | — |
| Energy AdaptNTK ASE cohort | All | — | — | — | — |
| Force NTK ASE MD and packed MD | Periodic solid or interface | I | I | I via ASE or TorchSim | I via ASE |
| Energy AdaptNTK replay and packed MD | Periodic solid or interface | P | P | P via replay or TorchSim | P for replay |
| `uq-mlip` replay and ASE MD | Molecule, periodic solid | V [4] CPU/CPU/CPU | I CUDA/CUDA/CPU | I ROCm/ROCm/CPU | V [5] MPS/MPS/CPU, float32 checkpoint |
| `uq-mlip` replay and ASE MD | Periodic interface qualification | P | P | P | P |
| `uq-mlip` packed TorchSim | Molecule, periodic solid or interface | P | P | P | — |
| `uq-mlip` packed ALCHEMI | Molecule, periodic solid or interface | P | P | — | — |
| Per-provider NEB and dimer acquisition | Molecule or periodic geometry where supported | P | P | P | P |
| Per-provider UDD bias and excited-state gap seeking | All | — | — | — | — |

The ground-state providers do not implement a state gap or a differentiable
bias potential. ALF's existing UDD and gap-seeking paths use ensemble spread.
NEB and dimer currently read ensemble spreads directly. Packed ALF uses a
float64 MD state, which excludes MPS, and ALCHEMI supports CPU and NVIDIA CUDA.
Adiabatic state energies are not smooth where states cross, so per-state
Hessian scores grow like 1/gap there. Each state therefore reports
`ntk_near_crossing_<state>`. Near a crossing, acquisition should read the gap
scores and the scores of smooth model outputs, such as X-MACE's state matrix or
couplings scaled by the gap. These matter for the planned photochemical
reactive sampling, which spends time near crossings.
The posterior covers only the trainable parameters, so a LoRA or frozen-layer
fine-tune of a foundation model is linearized in what was trained.
HIPPYNN NTK takes periodic graphs (with a cell input) for fully periodic cells.
Its rows use HIPPYNN's plain-PyTorch kernels in `torch.func` forward mode, at a
cost of about one evaluation per sketch dimension. MACE rows use batched reverse
mode, at a cost of about one backward pass per row, because MACE's forward pass
can't run inside `torch.func`.
The ASE `uq-mlip` provider uses a separate embedding forward when ASE
evaluates the calculator. Scoring only at acquisition checks is a target for
the packed provider.

## Set up and replay the MACE provider

Use a sampler environment with the appropriate PyTorch build and install the
pinned PNNL revision with MACE. The example fits a temporary regressor, runs
the replay and MD checks, and prints a JSON report. Supply a local copy of the
[public MACE-MP-0b small checkpoint](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model)
as `mace.model`.

```sh
python -m pip install -e .
python -m pip install 'uq-mlip[mace] @ git+https://github.com/pnnl/UQ-MLIP.git@61ae6e171db4f64382b81df34673b93edc4cee42'
OMP_NUM_THREADS=2 python examples/atom_resolved_uncertainty/uq_mlip_replay.py mace.model --device cpu
OMP_NUM_THREADS=2 python examples/atom_resolved_uncertainty/uq_mlip_replay.py mace.model --device cuda:0
```

Run the last command on Linux with a CUDA build of PyTorch for NVIDIA.
The same PyTorch device spelling applies to ROCm in an AMD environment.
Native hardware runs are pending for both. XGBoost stays on CPU in either
case. On macOS, XGBoost needs a discoverable OpenMP runtime.

On Apple MPS, use a float32 checkpoint. MACE loads checkpoint tensors onto
the requested device before its dtype conversion, so the original float64
checkpoint fails at load time on MPS. Create a separate float32 checkpoint
and fit its own regressor:

```sh
python -c 'import torch; from mace.calculators import mace_mp; model = mace_mp(model="./mace.model", device="cpu", return_raw_model=True); torch.save(model.float(), "mace-fp32.model")'
OMP_NUM_THREADS=2 PYTORCH_ENABLE_MPS_FALLBACK=0 python examples/atom_resolved_uncertainty/uq_mlip_replay.py mace-fp32.model --device mps
```

For use in an ALF run, follow the [sampler configuration and fitting example](../source/user_guide/acquisition.rst#pnnl-per-atom-uncertainty-with-mace).

## Reproduce the checks

Run commands from the repository root. The checks below establish score
agreement and sampling mechanics on synthetic data. Predictive accuracy and
production throughput require separate measurements.

1. On Apple M3 Max at ALF `b5563e7`, run
   `python examples/atom_resolved_uncertainty/ntk_replay.py` on CPU and add
   `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=0` for the Apple GPU.
   The synthetic HIP-NN spring replay matches standalone scores and warnings
   and checks selected-frame atom scores on short trajectories.
2. On Apple M3 Max at ALF `c4f4e81`, run
   `python examples/atom_resolved_uncertainty/ntk_replay.py --device cpu --engine torchsim`
   and repeat with `--engine alchemi`. Both CPU engines matched six replay
   decisions and retained per-atom scores on two synthetic HIP-NN trajectories.
3. On Apple M3 Max at ALF `46e91b7`, run
   `python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py --device cpu --engine torchsim`
   and repeat with `--engine alchemi`. Each synthetic MD cohort retained four
   candidates after pruning eight and released three for labeling. Run
   `PYTORCH_ENABLE_MPS_FALLBACK=0 python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py --device mps`
   for the fixed-pool Apple GPU check. The small matrix updates use CPU float64
   on every device. Both replays used the same 11-frame synthetic geometry pool.
4. On Apple M3 Max at ALF `c4f4e81`, run the CPU command above with
   `uq-mlip` 0.1.0 at the pinned revision, MACE 0.3.17, PyTorch 2.8.0,
   XGBoost 3.3.0, and ASE 3.29.0. The source checkpoint SHA-256 was
   `7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0`.
   The example fit 12 displaced water and silicon structures, replayed six
   held-out frames with 12 matching decisions, and preserved atom arrays in
   both first-trigger selections and the one-frame cohort.
5. On the same machine and revisions as the CPU check, run the MPS command above with
   CPU fallback disabled. The converted checkpoint SHA-256 in this run was
   `e6514a6f7972ad2c15269e9951b7377ac6005039d2afcfe6ca7c91283a2fe1d8`.
   The replay and MD checks passed with MACE and embedding extraction on MPS
   and the quantile regressor on CPU.
6. Run `python -m pytest tests/test_local_novelty.py` with PyTorch and TorchSim
   installed. CPU tests compared packed and ASE scores for mixed atom counts
   with molecular, slab, and fully periodic cells, and checked acquisition on
   short molecular trajectories in ASE, TorchSim, and ALCHEMI. For device
   qualification, set `ALF_TEST_ENGINE_DEVICE=cuda:0` on the target hardware.
7. On a Linux CPU at ALF `4731ad9`, with PyTorch 2.10 and HIPPYNN, run
   `python -m pytest tests/test_ntk_core.py tests/test_hippynn_ntk.py tests/test_adaptive_ntk.py`.
   With mace-torch 0.3.17 also installed, run `tests/test_ntk_mace.py`. These tests check:
   - rows against parameter finite differences, for both engines, molecular and periodic, HIPPYNN and
     MACE;
   - a pair potential that is linear in its weights against closed-form Bayesian linear regression;
   - the projected force and Hessian scores against exact sums;
   - periodic translation and permutation invariance, and supercell scaling;
   - Hessian-row fitting, and multi-state gap scores.

   The replay examples in checks 1 and 3 give the same decisions with ASE and TorchSim.
8. In an environment with X-MACE installed from github.com/rhyan10/X-MACE at `03d71eb`, run
   `python -m pytest tests/test_ntk_xmace.py` on a Linux CPU. Small random `ExcitedMACE` and
   `AutoencoderExcitedMACE` models with three states were checked for:
   - per-state energy, force and HVP rows, and the rows of the smooth outputs, against parameter finite
     differences;
   - batched and looped reverse mode agreeing;
   - eigenvalues of the autoencoder's `state_matrix` reproducing the state energies;
   - a frozen backbone staying out of the posterior;
   - energies and forces matching X-MACE's `ExcitedMACECalculator`.

   `tests/test_ntk_core.py` checks the crossing behaviour on a two-state model as the coupling goes from
   1e-2 to 1e-4: adiabatic Hessian scores grow about tenfold per decade, while the scores of the
   diabatic elements and of the gap stay finite and unchanged.

9. With the same X-MACE environment, run `python -m pytest tests/test_fine_tuning_xmace.py`, and in an
   environment with HIPPYNN, `python -m pytest tests/test_hippynn_interface.py -k excited_state_ntk`.
   The tests check:
   - multi-state shards becoming X-MACE training frames, with smooth couplings NAC × |ΔE|;
   - X-MACE's training command;
   - a two-epoch fine-tune of a three-state `ExcitedMACE` from a saved ground-state MACE, with and
     without LoRA, whose NTK basis loads for packed MD, with the backbone left out under LoRA;
   - a trained two-state HIPPYNN member's NTK predictions matching ALF's excited-state ensemble, in
     both property scales, with a cached basis.

10. Run `python -m pytest tests/test_ntk_outputs.py tests/test_hippynn_lora.py` with HIPPYNN. The tests check:
    - output rows against parameter finite differences;
    - batched against looped reverse mode;
    - the physical-coupling uncertainty formula;
    - LoRA starting at the pretrained model, pickling, merging, and agreeing in both NTK engines, for
      Hipnn and HipHopnn.

Add the ALF revision, upstream revision, checkpoint identity, package versions,
device placement, dataset and split, command, and observed result when you
verify a new combination. For acquisition quality, report large force errors
missed at equal frame flagging rates, separately for bulk and interfacial atoms,
and record scoring time. The MACE NTK adapter allows a same-potential
comparison between NTK and `uq-mlip`. Compare fixed and adaptive batching at the same pool
and label budget.
