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

ORCA gives Hartree, so `properties_list` converts to eV and eV/Å, as MACE expects. The builder
makes water clusters (`"periodic": false`), because ORCA labels each structure as a cluster and
the fine-tuned model must learn the same structure. `E0s` is
`estimated`, as MACE's fine-tuning guidance recommends when isolated-atom energies aren't available;
use `foundation` only when the labels share the foundation model's level of theory. The other
defaults follow MACE's naive fine-tuning recipe:
https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html

ALF ships no model weights. You supply the checkpoint, and its license applies: MACE-MP-0 is MIT
licensed, while several other MACE foundation models use the Academic Software License. See the
fine-tuning guide in the ALF documentation.
