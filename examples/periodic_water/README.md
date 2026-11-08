# Periodic water with PySCF

ALF learns liquid water in periodic boxes, labeled with periodic DFT (PBE, GTH-DZVP, Γ point) from
PySCF or GPU4PySCF, and trains a periodic HIPPYNN ensemble (`cell_key: "cell"`).

- `qm_config.json` runs on a GPU (`"backend": "gpu"`), and needs GPU4PySCF: `pip install "alf[gpu4pyscf]"`
  for CUDA 12, or the builds matching your CUDA.
- `qm_config_cpu.json` runs on the CPU with PySCF, which comes with ALF. Use it for bring-up by setting it
  as `QM_config_path` in `master_config.json`. A 15-water box took about a minute on a laptop.

Set the machine in `parsl_configs.py`, then:

```bash
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json --test_qm
python -m alframework --master master_config.json
```

The builder makes periodic boxes (`"periodic": true`) with `min_dist` 1.6 and `shake` 0.02: randomly
packed boxes that are too strained can fail to converge. See the PySCF section of the QM interfaces guide.
