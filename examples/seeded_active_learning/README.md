# Seeded Active Learning Example

This example starts ALF from user-provided structures and optional existing
ALF/HIPPYNN HDF5 data. It is useful when you already have representative
starting structures or labels and want to avoid a random bootstrap stage.

Put starting structures under:

```text
fragment_library/*.cfg
```

If you already have labeled data, put it under:

```text
h5store/data-0000.h5
h5store/data-0001.h5
...
```

If `status.txt` is absent and `h5store/data-0000.h5` exists, ALF detects the
existing HDF5 store, skips random bootstrap labeling, and trains
`models/model-0000` from those data. The configured builder then draws future
starting structures from `fragment_library/`.

Run staged checks from this directory:

```bash
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json --test_ml
python -m alframework --master master_config.json --test_sampler
python -m alframework --master master_config.json --test_qm
```

Then start active learning:

```bash
python -m alframework --master master_config.json
```
