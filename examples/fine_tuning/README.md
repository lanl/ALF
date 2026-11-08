# Fine-tuning a pretrained HIPPYNN ensemble with ALF

ALF samples with a pretrained HIPPYNN ensemble, labels the structures where its members disagree,
and fine-tunes copies of that ensemble on the labels. Each round starts every copy from the
pretrained weights and trains on all labels so far, with a fresh optimizer.

1. Put the pretrained ensemble in `models/model-0000/model-00`, `model-01`, and so on. Each member
   needs `experiment_structure.pt`, `best_checkpoint.pt`, and `best_model.pt`. ALF samples with it
   from the start, so no bootstrap is needed, and `hippynn_config.json` fine-tunes from it.
2. Optionally, put earlier labels in `h5store/` as ALF-format shards. Every round trains on them too.
3. Put starting structures in `fragment_library/`, set `QM_run_command` in `orca_config.json`, and
   set the machine in `parsl_configs.py`.
4. Check the stages, then run:

```bash
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json --test_sampler
python -m alframework --master master_config.json --test_qm
python -m alframework --master master_config.json
```

To fine-tune once on existing labels, without sampling, run `--test_ml` instead.

You supply the pretrained ensemble; ALF ships no model weights. HIPPYNN has no official
fine-tuning recipe, so this one is ALF's: the pretrained network and weights, a fresh optimizer,
and the steps of ALF's HIPPYNN trainer.

Labels must be in the units the pretrained model was trained on; ALF's samplers expect eV and
eV/Å. HIPPYNN checkpoints are loaded with PyTorch, which can run code in the file, so only use
checkpoints you trust. See the fine-tuning guide in the ALF documentation.
