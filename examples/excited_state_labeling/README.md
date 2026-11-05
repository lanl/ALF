# Excited-state labeling qualification

This opt-in check exercises fixed small molecules through the production
labelers, Parsl `LabelingCoordinator`, and HDF5 storage. It runs no training or
dynamics. GPU4PySCF supports single and sequential tasks; PySEQM AM1/CIS also
supports native multi-molecule solves. Both label S0–S3. Current upstream PySEQM requires a common sorted species
sequence for all-state analytic CIS forces. Mixed input lists are partitioned
by composition into native solves, then restored to original submission order.
The report distinguishes submitted width, compatibility groups, actual solve
widths and exception retries. True mixed-composition native CIS forces are
deferred by user decision; ground-state mixed batching remains unchanged.

The PySEQM entrypoints are `pyseqm_excited_state_task` (one molecule) and
`pyseqm_excited_state_batch_task` (`molecule_objects`, for native batching) in
`alframework.qm_interfaces.pyseqm_interface`. The GPU4PySCF entrypoint is
`gpu4pyscf_excited_state_task` in its own interface module. They accept
`QM_config`, `properties_list`, optional `sampler_config`, and `gpus_per_node`.
Ground-state entrypoints are unchanged.

State energy properties use `sE0`…`sEn` with scope `system`; force properties
use `F0`…`Fn` with scope `atomic`. Optional `dE01` denotes E1−E0. NACs use
`nacr` with scope `pair_atomic` and are stored as `[frames, pairs, atoms, 3]`,
with a constant `nac_pairs` dataset. GPU4PySCF NAC units are inverse Angstrom;
transition dipoles are atomic units and oscillator strengths dimensionless.
Energies and forces returned by both adapters are eV and eV/Angstrom before
`properties_list` conversion. `energy_offset_eV` is subtracted once from each
state energy; it does not affect gaps, forces or optical properties.

Create isolated environments using `install_environment.sh`. No existing
personal or qualification environment should be modified. The site JSON needs
absolute `gpu4pyscf_python` and `pyseqm_python` interpreter paths; optional site
fields are `partition`, `account`, and `cuda_module`. Darwin defaults are
`shared-redstone`, `y2020-bf`, and `cuda/13.2.1`.

After committing the source, prepare without submitting:

```bash
python examples/excited_state_labeling/qualify.py --prepare-only \
  --root /absolute/validation/run-001 --site /absolute/site.json
```

Select either backend with `--backend gpu4pyscf` or `--backend pyseqm`.
Preparation exports committed source and records fixture/source hashes.
From a persistent Darwin front-end session, run the generated frozen runner:

```bash
python /absolute/validation/run-001/source/examples/excited_state_labeling/qualify.py \
  --execute --root /absolute/validation/run-001
```

Allocations run sequentially for up to 30 minutes each. The supervisor bounds
the whole run to two hours including queue waits. It records exact owned job
IDs, cancels only those jobs, and verifies termination before continuing.
Results include environments, numerical errors, task/backend widths, and HDF5
fingerprints. Missing or timed-out checks remain incomplete.
