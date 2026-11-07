# Fine-tuning MACE with ALF

ALF fine-tunes copies of a MACE foundation model on structures it selects. Each round starts every
copy from the original checkpoint and trains on all labels so far. The copies' disagreement during MD
picks the next structures to label with ORCA.

1. Download a MACE checkpoint to `foundation/mace-mp-0-small.model`; compute nodes often have no network.
2. Install `mace-torch` and ALF's `torchsim` extra in the ML and sampler worker environments.
3. Set `QM_run_command` in `orca_config.json`, and the machine settings in `parsl_configs.py`.
4. Check the stages, then run:

```bash
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json --test_qm
python -m alframework --master master_config.json
```

ORCA gives Hartree, so `properties_list` converts to eV and eV/Å, as MACE expects. `E0s` is
`average`, which fits new atomic energies to the ORCA labels; use `foundation` when the labels share
the foundation model's level of theory. See the fine-tuning guide in the ALF documentation.
