# Uncertainty integration in ALF

Use this guide to qualify uncertainty providers on `ALF_NTK`.
The [acquisition guide](../source/user_guide/acquisition.rst)
defines the score fields and selection settings. ALF thresholds named scalar
scores and keeps the corresponding atom arrays with selected structures.

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

The packed engine now supports `observe_uncertainty_batch`, which receives the
current geometry and unbiased predictions for all active structures. Local
environment novelty is the first provider using it. NTK uses the single-frame
`observe_uncertainty` hook during packed runs.

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

9. With the same X-MACE environment, run `python -m pytest tests/test_fine_tuning_xmace.py`, and in an
   environment with HIPPYNN, `python -m pytest tests/test_hippynn_interface.py -k excited_state_ntk`.
   The tests check:
   - multi-state shards becoming X-MACE training frames, with smooth couplings NAC × |ΔE|;
   - X-MACE's training command;
   - a two-epoch fine-tune of a three-state `ExcitedMACE` from a saved ground-state MACE, with and
     without LoRA, whose NTK basis loads for packed MD, with the backbone left out under LoRA;
   - a trained two-state HIPPYNN member's NTK predictions matching ALF's excited-state ensemble, in
     both property scales, with a cached basis.

   `tests/test_ntk_core.py` checks the crossing behaviour on a two-state model as the coupling goes from
   1e-2 to 1e-4: adiabatic Hessian scores grow about tenfold per decade, while the scores of the
   diabatic elements and of the gap stay finite and unchanged.

Add the ALF revision, upstream revision, checkpoint identity, package versions,
device placement, dataset and split, command, and observed result when you
verify a new combination. For acquisition quality, report large force errors
missed at equal frame flagging rates, separately for bulk and interfacial atoms,
and record scoring time. The MACE NTK adapter allows a same-potential
comparison between NTK and `uq-mlip`. Compare fixed and adaptive batching at the same pool
and label budget.
