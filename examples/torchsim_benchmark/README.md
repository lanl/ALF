# Batched MLMD engine acceptance

The existing benchmark accepts `--engine torchsim` (the default) or
`--engine alchemi`. Each engine uses the same batched HIPPYNN evaluator,
physical predictions, and ALF force/noise calculations. The benchmark runs
inference and dynamics, not QM, training, or a full active-learning campaign.

Use separate sampler-worker environments. TorchSim requires Python 3.12+;
the ALCHEMI adapter targets toolkit 0.2.0 and Python 3.11+. Install the matching
CUDA/ROCm PyTorch build first, then the `torchsim` or `alchemi` ALF extra.
ALCHEMI supports CPU correctness checks and NVIDIA CUDA execution; use
TorchSim on AMD. A working import alone does not validate CUDA: execute both
Torch and Warp operations on the allocated device.

To select ALCHEMI in an ordinary ALF run, set the master sampler task to
`alframework.samplers.batched_mlmd.mlmd_trajectories_task` and use
`examples/simple_water/mlmd_config_alchemi.json` as an opt-in sampler example.
It retains the flat MLMD settings and adds
`"driver": {"strategy": "batched", "engine": "alchemi"}`. Singleton execution
uses `strategy: single` with no `batching` block. Acquisition and labeling
configuration remain independent of the engine.

From the checkout root, with the selected worker environment:

```bash
PYTHONPATH="$PWD" python examples/torchsim_benchmark/benchmark.py \
  --engine alchemi --device cuda:0 --model /path/to/model-0000 \
  --batch-sizes 1 2 8 32 50 --output /path/to/alchemi-results.json
```

Use `--structures /path/to/structures.extxyz` to evaluate identical fixed
geometries across environments. Without it, the script builds periodic water
boxes. Outputs include checkpoint hashes, source-file hashes, actual imported
checkout identity, package versions, hardware, timings, and parity errors.
A source copy inside another repository does not claim the enclosing repo's
revision. Memory is **peak PyTorch allocated memory**, not total GPU process
memory; Warp and other allocations may be outside this metric.

Float32 predictions use `rtol=atol=1e-5`, independently of acquisition
thresholds. Conformance records the actual member predictions and checks that
candidate uncertainties are their correct reductions. Comparisons with
individual evaluations account for measured input roundoff: the change in an
ensemble standard deviation is bounded by `std(batched - individual)`. Force
mean/max/RMS uncertainty bounds follow from those component bounds. This avoids
applying total-energy relative tolerances directly to a much smaller energy
spread; acquisition cutoffs are unchanged.

Integration state is float64. Friction remains 0.02 inverse ASE
time; `dt` remains fs. ALCHEMI uses native deterministic half kicks and the
same per-trajectory NumPy thermostat as TorchSim, rather than the toolkit's
batch-dependent stochastic kernel. Reports measure this adapter, not the
stock high-level ALCHEMI Langevin runner.

Shared conformance tests retain their historical filename:

```bash
PYTHONPATH="$PWD" python -m pytest tests/test_torchsim_engine.py \
  tests/test_engine_boundaries.py tests/test_sampler_coordinator.py
```

They test whichever qualified engines are installed. Inside an allocated GPU
job, set `ALF_TEST_ENGINE_DEVICE=cuda:0` to exercise actual GPU mechanics and
checkpoint predictions. Default tests use CPU. Seeded grouping comparisons
require exact frame identities and outcomes; GPU float64 roundoff is compared
at `1e-12`, while CPU grouping retains exact equality checks.

Thermostat tests measure 300 K, then independent 200/400 K targets, with 3%
mean-temperature acceptance. The trained water smoke model is an integration
fixture; these checks establish neither potential accuracy nor long-term
production stability.
