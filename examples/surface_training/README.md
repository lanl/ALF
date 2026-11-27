# Train surface models on CO/Cu(111)

The two masters use the same periodic VASP RPBE labeler and the same builder.
The builder generates clean Cu(111) slabs and CO on atop, bridge, and hollow
sites. It gives each candidate a small, reproducible coordinate perturbation,
so the first set includes different forces even on clean slabs. Sampling uses
fixed-cell MD to preserve the vacuum gap of each candidate. Each trajectory
is limited to 0.1 ps in the example.

| Master | Training | Sampling |
| --- | --- | --- |
| `master_uma.json` | Fine-tune two copies of the UMA-S `oc20` task | UMA ensemble through TorchSim, on CUDA |
| `master_hippynn.json` | Train two periodic HIPPYNN models from scratch | HIPPYNN ensemble through ASE, on CPU |

Run the commands from this directory. The builder uses the neighboring
`surface_adsorption` example's Cu and CO files. If you move the example, update
the relative paths in `builder_config.json`.

1. Install ALF and provide VASP and the Cu, C, and O PAW files. Set
   `VASP_PP_PATH` and edit `QM_run_command` in `vasp_config.json` if you use
   a launcher or wrapper. The example uses 450 eV, Gaussian smearing of 0.1 eV,
   and a 20 Å total vacuum gap before the initial perturbation. The
   `kpoint_spacing_A_inv` setting of 0.15 Å⁻¹ gives a Γ-centered 5 × 5 × 1 mesh
   for this Cu(111) cell and adjusts the in-plane counts for other cells.
   The surface builder marks `c` as a vacuum axis with one k-point.
   Static labels use `EDIFF=1e-7` and allow up to 200 electronic steps.
   Converge the spacing, cutoff, smearing, slab thickness, and vacuum for
   your application.
2. For UMA, install `fairchem-core` and ALF's `torchsim` extra in the GPU
   worker environment. Accept the [UMA checkpoint terms](https://huggingface.co/facebook/UMA)
   on a machine with network access, then run
   `hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir foundation`.
   The trainer and sampler both select `oc20`. The example uses
   `E0s: "foundation"` to keep that task's energy references for this narrow
   RPBE example. The workers read the checkpoint from disk.
3. For HIPPYNN, install HIPPYNN in the worker environment. A single-host Parsl
   configuration and a debug configuration are supplied for both masters.
   Adjust execution resources and the labeling command for your site.

Choose either master for the following checks:

```bash
master=master_uma.json  # or master_hippynn.json
python -m alframework --master "$master" --check_config
python -m alframework --master "$master" --test_builder
python -m alframework --master "$master" --test_qm
python -m alframework --master "$master"
```

`--check_config` validates the inputs. `--test_builder` constructs a candidate,
and `--test_qm` runs one real VASP single point and writes `qm_test.h5` for
inspection. Move that file after each check if you test both masters in this
directory. A full run builds its bootstrap set, trains the ensemble, then
labels structures selected during MD. The two masters have separate status
files and keep shards, models, and scratch files under their own `runs/` paths. The small
label budget is meant to demonstrate the workflow. If the ensemble agrees on
every trajectory, ALF may collect no further labels. Set acquisition thresholds
from your validation errors for production work.

## Prepare seeds and set constraints

The example uses the supplied CO coordinates. For larger SMILES adsorbates,
`relax_smiles: true` minimizes the embedded geometry with RDKit before
placement. You can also supply an externally relaxed structure file. ALF
labels each selected MD frame at its sampled geometry. A gas-phase
adsorption-energy reference is a separate DFT calculation.

Use relaxed bulk input cells for the desired lattice constants. To sample
strain, supply strained bulk cells as additional materials and run
fixed-cell MD for each candidate. For a frozen substrate in the HIPPYNN
example, use a separate builder config with `fixed_bottom_layers: 1`
and point that master to it. The scalar ASE sampler and the batched driver's
ASE engine support this setting. Keep `end_dens` null and `translate_to_center`
false. The shared example leaves both slabs fully mobile, which also works
with the UMA TorchSim sampler.

To test electrostatic finite-size effects, set `slab_dipole_correction: true`
at the top level of the VASP config. Test identical clean and decorated slab
geometries in separate scratch directories. Compare dipole off and on and
compare 0.15 Å⁻¹ with 0.12 Å⁻¹ (6 × 6 × 1 on the example cell) using forces
and relative site energies. Start fresh label sets when changing the reference
settings.

## Compare models on the same DFT data

Once one run has written finalized `data-*.h5` shards, copy those files to the
other master's `h5/` directory in a **fresh** run. Copy the bootstrap shard
alone to compare the first training round, or copy a fixed snapshot of all the
finalized shards for a larger comparison. For example:

```bash
mkdir -p runs/hippynn/h5
cp runs/uma/h5/data-0000.h5 runs/hippynn/h5/
python -m alframework --master master_hippynn.json --test_ml
python -m alframework --master master_hippynn.json --test_sampler
```

`--test_ml` trains on the shards already in the selected run directory.
`--test_sampler` loads its trained model and attempts one trajectory. It may
finish without selecting a frame if the acquisition thresholds are not reached.
The same commands work with the UMA master and its own training artifacts.
VASP's force-consistent `free_energy` and forces are stored as `energy` and
`forces` in eV and eV/Å. Both trainers use the same labels and periodic cells.
They choose their own internal validation splits. For a direct comparison,
evaluate both on the same independent VASP frames excluded from both training
runs. Compare force errors, including the adsorbate atoms, and relative
atop, bridge, and hollow energies at the same composition. For adsorption
energies, use clean-slab and isolated-CO reference energies from a consistent
DFT protocol.

## Choose a starting model for another surface

UMA's pretrained `oc20` task targets RPBE dry adsorption. Its off-the-shelf
predictions do not use r²SCAN. You can initialize a model from UMA and fine-tune
it against r²SCAN energies and forces. Use a separate run and fit energy
references for the new labels rather than reusing the RPBE references. A fixed
composition can only identify reference energy combinations, so add independent
compositions if you need transferable element references.

[MACE-MH-1](https://github.com/ACEsuit/mace/releases) is another starting point.
It has an `oc20_usemppbe` PBE surface head and a separate `matpes_r2scan` head.
Compare the chosen heads on held-out structures at your target level of theory.
For MAPbI₃, start from the [ordered bulk and bond-protection example](../surface_adsorption/README.md),
update the HIPPYNN species if training from scratch, and label that material
with your chosen DFT setup before deciding which model transfers better.
