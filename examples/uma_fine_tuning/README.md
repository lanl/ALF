# Fine-tuning UMA with ALF

ALF fine-tunes copies of a UMA checkpoint's `omol` task on water structures it selects and labels
with ORCA. Each round starts every copy from the original checkpoint and trains on all labels so far;
the copies' disagreement during MD picks the next structures.

1. On a machine with network access, accept UMA's license at https://huggingface.co/facebook/UMA,
   then run `hf auth login` and
   `hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir foundation`.
   Compute nodes only read the file.
2. Install `fairchem-core` and ALF's `torchsim` extra in the ML and sampler worker environments.
3. Set `QM_run_command` in `orca_config.json`, and the machine in `parsl_configs.py`.
4. Check the stages, then run:

```bash
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json --test_qm
python -m alframework --master master_config.json
```

ALF ships no model weights. UMA's weights are covered by the FAIR Chemistry License, which you
accept on Hugging Face; fine-tuned copies are derivative works, so sharing them is subject to it,
and publications using UMA must acknowledge it.

ORCA's level of theory differs from UMA's `omol` (ωB97M-V), so `E0s` is `average`, which refits
UMA's per-element reference energies to the labels. Sampling treats every structure as a neutral
singlet (`charge` 0 and `spin` 1 in `mlmd_config.json`). The builder makes water clusters
(`"periodic": false`): ORCA labels each structure as a cluster, and `omol` was trained only on
non-periodic structures. `epochs` is 10, above fairchem's default
of 1, because ALF's early rounds have few labels. See the fine-tuning guide in the ALF documentation,
and fairchem's own: https://facebookresearch.github.io/fairchem/fine-tuning
