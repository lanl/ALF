# Surface builder example

This example places CO near lead sites on a pseudocubic MAPbI₃(100) slab built
from the Materials Project cell `mp-977013`. The C–N, C–H, and N–H distance
cutoffs keep methylammonium ions intact during cleavage. The adslab has 96
substrate atoms plus two CO atoms and 20 Å of vacuum. The substrate is
10.52–12.82 Å thick, depending on the termination, and its in-plane vectors
are approximately 12.88 × 12.98 Å.

To generate and inspect structures without running dynamics or a labeler, run
this from the example directory with ALF installed:

```bash
python - <<'PY'
import json
from ase.io import write
from alframework.builders.surface import build_surface

with open("builder_config.json") as file:
    config = json.load(file)

for mode in ("random", "binding_sites"):
    config["adsorbates"][0]["mode"] = mode
    candidates = [build_surface(f"preview-{i}", config).get_atoms() for i in range(5)]
    write(f"{mode}.extxyz", candidates)

config.pop("adsorbates")
write("bare.extxyz", build_surface("preview-bare", config).get_atoms())
PY
```

To generate candidates in an ALF run, use
`alframework.builders.surface.surface_builder_task` as `builder_task` and point
`builder_config_path` at the supplied JSON. Put the input geometries under the
run's `master_directory`, or use absolute filenames. Choose a labeler and
sampler that support periodic cells and the system's elements. The candidates
use a fully periodic cell with a vacuum gap and are starting points for
relaxation.

See the [Cu(111) training example](../surface_training/README.md) to connect
the surface builder to VASP labels, UMA fine-tuning, and HIPPYNN training.

Use [`builder_mp_materials.json`](builder_mp_materials.json) to sample clean
(100) slabs of all four Materials Project compounds under a 200-atom cap.
Each candidate has a 10 Å minimum substrate thickness, 8 Å hard lateral
floors, 9 Å lateral targets, and 20 Å of vacuum.

For a sweep of clean orthorhombic MAPbI₃ facets, use
[`builder_sweep.json`](builder_sweep.json) as `builder_config_path`. This
config examines primitive indices through `max_miller: 1`, samples surviving
facet families equally, and caps each structure at 300 atoms. Facet families
that fail the lattice bounds, atom cap, or bond protection are reported and
skipped. The example's (010) family is skipped by the atom cap.

To mix symmetric copper slabs carrying CO and clean MAPbI₃ slabs in a ratio
of two to one on average, use [`builder_materials.json`](builder_materials.json).
Try it from this directory:

```bash
python - <<'PY'
import json
from ase.io import write
from alframework.builders.surface import build_surface

with open("builder_materials.json") as file:
    config = json.load(file)
samples = [build_surface(f"mix-{i}", config) for i in range(12)]
for sample in samples:
    print(sample.get_moleculeid(), sample.get_metadata()["surface_builder"]["material_id"])
write("materials.extxyz", [sample.get_atoms() for sample in samples])
PY
```

No extra installation is needed for facet symmetry, face symmetry, or XYZ
adsorbates. To try water from SMILES, install ALF with `python -m pip install
'.[topology]'` from the checkout, then replace the top-level adsorbate in
`builder_materials.json` with `{"smiles": "O", "anchor_indices": [0]}`. The
anchor is oxygen. ALF adds the hydrogens and counts them against the atom cap.

ALF centers the slab or the entire adslab along `z`, leaving half of
`vacuum_A` on either side.

## Bulk structure sources

The saved Materials Project snapshots were exported from the output of
`death_star/add_materials_to_mongodb/2_add_bulk_materials_to_mongodb/get_bulk_materials_from_mp.ipynb`
in `high_throughput_DFT`. That notebook fetched conventional cells. These
are the structures saved before the separate Death Star PBE bulk relaxation.

| Bulk structure | Materials Project | Bulk atoms | Clean (100) slab atoms |
| --- | --- | ---: | ---: |
| Pseudocubic MAPbI₃ | [mp-977013](https://materialsproject.org/materials/mp-977013) | 12 | 96 |
| Pseudocubic FAPbI₃ | [mp-977014](https://materialsproject.org/materials/mp-977014) | 12 | 96 |
| Cubic CsPbBr₃ | [mp-600089](https://materialsproject.org/materials/mp-600089) | 5 | 60 |
| Cubic CsPbI₃ | [mp-1069538](https://materialsproject.org/materials/mp-1069538) | 5 | 60 |

`mapi_100K.extxyz` is an ASE conversion of `geometry_7.in` from the
[Hybrid Perovskite Database, dataset 2900](https://materials.hybrid3.duke.edu/materials/dataset/2900),
available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The 100 K atomic structure is from Weller et al., *Chem. Commun.* **51**,
4180–4183 (2015), [doi:10.1039/C4CC09944C](https://doi.org/10.1039/C4CC09944C).

To try a compact slab from the orthorhombic cell, override these fields in
`builder_config.json`:

```json
{
  "bulk_file": "mapi_100K.extxyz",
  "allow_independent_terminations": true,
  "slab_symmetry": "require"
}
```

With the other settings in that file, this gives 12.21 Å of substrate
thickness, 12.63 × 8.58 Å in plane, and 74 atoms including CO. Removing the
adsorbate gives a 72-atom slab with equivalent faces.
