# Uncertainty integration in ALF

Use this guide to qualify uncertainty providers on
`feature/atom-resolved-uncertainty`. The [acquisition guide](../source/user_guide/acquisition.rst)
defines the score fields and selection settings. ALF thresholds named scalar
scores and keeps the corresponding atom arrays with selected structures.

## What the providers measure

| Provider | Inputs and fitted artifacts | Output and ALF use |
| --- | --- | --- |
| HIPPYNN force NTK | One molecular HIP-NN or HIP-HOP energy graph, its original training geometries, and energy and force scales | A parameter-sensitivity score for each force component, reduced to a score per atom and a maximum for frame acquisition. Compare with quantum forces to establish what a warning means. |
| Energy AdaptNTK | One molecular HIPPYNN energy graph and a saved basis fit on its original training geometries | One energy-gradient row and initial energy score per frame. The coordinator uses the rows to update scores after each selection. Per-atom force NTK can be reported by the same evaluator. |
| PNNL `uq-mlip` | A MACE checkpoint and a quantile regressor fit on its frozen per-atom embeddings and internal atomic energies | The package returns an interval half-width for each atom's internal energy. The interface paper studies a relative width normalized by the sum of the bounds. These quantities have different scales and require separate cutoffs. Compare the package score with held-out reference forces before using it as a force-error warning. |

Read the [interface study](https://doi.org/10.1038/s41524-026-02051-8),
[PNNL package](https://github.com/pnnl/UQ-MLIP), and
[AdaptNTK preprint](https://arxiv.org/html/2609.00488v1#S4.SS2) for the methods.
The ALF integration targets MACE for `uq-mlip`; PNNL also publishes UMA and
CHGNet extractors. Those backends require separate qualification.

## Compatibility and evidence

**V** verified for the linked smoke test, **I** implemented with hardware
verification pending, **P** planned, **—** unsupported in the current path.
An entry such as CPU/CPU/CPU gives the MLIP, embedding or gradient extraction,
and score devices. A separate CPU selection stage is indicated where relevant.

| Provider and ALF path | Geometry | CPU | NVIDIA CUDA | AMD ROCm | Apple MPS |
| --- | --- | --- | --- | --- | --- |
| Force NTK replay and ASE MD | Molecule | V [1] | I | I | V [1] |
| Force NTK packed TorchSim | Molecule | V [2] | I | I | — |
| Force NTK packed ALCHEMI | Molecule | V [2] | I | — | — |
| Energy AdaptNTK fixed pool | Molecule | V [3] | I | I | V [3] |
| Energy AdaptNTK TorchSim cohort | Molecule | V [3] | I | I | — |
| Energy AdaptNTK ALCHEMI cohort | Molecule | V [3] | I | — | — |
| Energy AdaptNTK ASE cohort | All | — | — | — | — |
| Force NTK or energy AdaptNTK replay/packed MD | Periodic solid or interface | P | P | P via TorchSim | P for replay |
| `uq-mlip` replay and ASE MD | Molecule, periodic solid | P | P | P | P |
| `uq-mlip` replay and ASE MD | Periodic interface qualification | P | P | P | P |
| `uq-mlip` packed TorchSim | Molecule, periodic solid or interface | P | P | P | — |
| `uq-mlip` packed ALCHEMI | Molecule, periodic solid or interface | P | P | — | — |
| Per-provider NEB and dimer acquisition | Molecule or periodic geometry where supported | P | P | P | P |
| Per-provider UDD bias and excited-state gap seeking | All | — | — | — | — |

The ground-state providers do not implement a state gap or a differentiable
bias potential. ALF's existing UDD and gap-seeking paths use ensemble spread.
NEB and dimer currently read ensemble spreads directly. Packed ALF uses a
float64 MD state, which excludes MPS, and ALCHEMI supports CPU and NVIDIA CUDA.
Periodic HIPPYNN energy and force inference exists independently of periodic
NTK. The NTK graph and estimator currently require molecular inputs.

## Reproduce the checks

Run commands from the repository root. The checks below establish score
agreement and sampling mechanics on synthetic data. Predictive accuracy and
production throughput require separate measurements.

1. On Apple M3 Max at ALF `b5563e7`, run
   `python examples/atom_resolved_uncertainty/ntk_replay.py` on CPU and add
   `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=0` for the Apple GPU.
   The synthetic HIP-NN spring replay matches standalone scores and warnings
   and checks selected-frame atom scores on short trajectories.
2. At ALF `46e91b7`, run
   `python -m pytest tests/test_hippynn_ntk.py tests/test_dynamics_engines.py`
   with TorchSim and ALCHEMI installed. The Apple M3 Max CPU regression
   checks packed atom ordering, observation timing, and the installed engines.
   For a combined force-NTK and native-engine smoke, use `ntk_replay.py` with
   `--device cpu --engine torchsim` or `--engine alchemi`.
3. On Apple M3 Max at ALF `46e91b7`, run
   `python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py --device cpu --engine torchsim`
   and repeat with `--engine alchemi`. Each synthetic MD cohort retained four
   candidates after pruning eight and released three for labeling. Run
   `PYTORCH_ENABLE_MPS_FALLBACK=0 python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py --device mps`
   for the fixed-pool Apple GPU check. The small matrix updates use CPU float64
   on every device. Both replays used the same 11-frame synthetic geometry pool.

Add the ALF revision, upstream revision, checkpoint identity, package versions,
device placement, dataset and split, command, and observed result when you
verify a new combination. For acquisition quality, report large force errors
missed at equal frame flagging rates, separately for bulk and interfacial atoms,
and record scoring time. Compare fixed and adaptive batching at the same pool
and label budget.
