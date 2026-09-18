# Fine-Tuning Existing Model Example

This example shows how to start ALF from an existing HIPPYNN ensemble and
fine-tune that ensemble on new ALF data. The example contains a local custom
`ML_task` in `fine_tune_ml_task.py`, so ALF's main program is unchanged.

Here, fine-tuning means the common MLIP pattern: load pretrained model weights,
continue supervised training on new target data with a smaller learning rate,
and write the adapted model as the next ALF ensemble. It does not reuse the old
optimizer state. This is not a full HIPPYNN training restart: ALF reuses the
checkpoint graph and weights, creates a fresh optimizer/controller, and rebuilds
the training database from the current `h5store/`.

The local task intentionally loads checkpoints with `restart_db=False`. The
checkpoint evaluator's `db_info` defines the expected HDF5 inputs and targets,
and the staged HDF5 copies are filtered to match that loaded graph. HIPPYNN
checkpoint loading uses PyTorch serialization, so only fine-tune from checkpoint
files that you trust. For the upstream restart behavior this example builds on,
see the HIPPYNN restart documentation:
https://lanl.github.io/hippynn/examples/restarting.html.

If you only want to bring your own structures or HDF5 data without starting
from model weights, use `examples/seeded_active_learning`.

Before running, add a trained seed ensemble under:

```text
models/model-0000/model-00/
models/model-0000/model-01/
...
```

Optional prior ALF-format training data can be placed under:

```text
h5store/data-0000.h5
h5store/data-0001.h5
...
```

To fine-tune existing checkpoints on existing labeled data without running
MLMD, start without `status.txt` and run only the ML stage:

```bash
python -m alframework --master master_config.json --test_ml
```

This reads the HDF5 store and writes the adapted ensemble to the next model
slot, usually `models/model-0001`, without launching builders, samplers, or QM
labeling.

The fine-tuning task stages filtered HDF5 copies inside each output
`model-XX/staged_h5/` directory before HIPPYNN loads the data. When `cell_key`
is `null`, unused `cell` datasets are dropped from those staged copies so new
periodic sampler output can be combined with older non-periodic HDF5 batches.
The original files in `h5store/` are not modified. If `cell_key` is set, every
HDF5 group must contain that dataset.

If `status.txt` is absent and `models/model-0000` exists, ALF will detect the
seed model and use it for sampling. New labels are written to `h5store/`, and
the fine-tuned ensemble is written to `models/model-0001`.

The local ML task loads each source member from:

```text
experiment_structure.pt
best_checkpoint.pt
```

inside each `models/model-0000/model-XX/` directory. On later ALF training
rounds, it fine-tunes from the current model id and writes the next model id.

Confirm that fine-tuning actually ran by checking each
`models/model-0001/model-XX/training_log.txt` for training epochs and
`Training complete`; the new `best_model.pt` and `best_checkpoint.pt` should be
written during that run.

Run staged checks from this directory:

```bash
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json --test_sampler
python -m alframework --master master_config.json --test_qm
python -m alframework --master master_config.json --test_ml
```

Then start the fine-tuning active-learning run:

```bash
python -m alframework --master master_config.json
```
