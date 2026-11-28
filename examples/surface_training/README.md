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

## Perovskite adsorption pilot

`master_perovskite_uma.json` fine-tunes the UMA-S 1.2.1 `oc25` task for H₂O and NH₃ on
CsPbBr₃, CsPbI₃, CsSnBr₃, and CsSnI₃ pseudocubic (100). This crossed
design tests whether the Pb-to-Sn change depends on halide. Prepare the four
cubic bulk references with VASP and the Cs, Pb, Sn, Br, I, H, O, and N PAWs.
The bulk script uses the checked-in Materials Project Pb cells as prototypes,
substitutes Sn to generate the tin starting cells, and minimizes each
end member's RPBE-D3 free energy with respect to its cubic lattice parameter.
It records the starting structures, fixed reciprocal meshes, and energy
samples in `perovskite_bulk_oc25/manifest.json`.

Use one licensed PBE PAW release and record the selected setups for Cs, Pb,
Sn, Br, I, H, O, and N. Set `VASP_PP_PATH` to the parent of the
`potpaw_PBE` directory and make it visible to the bulk job and every QM
worker. ASE reads the element files and assembles each job's `POTCAR`.
For a versioned directory such as `potpaw_PBE.64`, set
`VASP_PP_VERSION=64`. Check the selected PAWs' `ENMAX` values when
qualifying the 450 eV cutoff. [ASE VASP setup](https://ase.gitlab.io/ase/ase/calculators/vasp.html#pseudopotentials)

```bash
python perovskite_bulk.py
python -m alframework --master master_perovskite_uma.json --check_config
python -m alframework --master master_perovskite_uma.json
```

Supply the local UMA checkpoint as described above. The builder samples
CsX and MX₂ faces with 20 Å vacuum, at least 10 Å of substrate, and a
200-atom cap. The 48 bootstrap structures cycle evenly through the 16
composition, adsorbate, and termination combinations. The cell stays fixed
and all atoms can move. The bulk and slab settings use Gaussian smearing
with `SIGMA = 0.05 eV`, `ENCUT = 450 eV`, `ISPIN = 1`, and
`LSORBIT = False`. Bulk preparation and slab labels use RPBE-D3
with zero damping (`IVDW = 11`), matching OC25's dispersion treatment. Slab labels
use an in-plane reciprocal mesh, one point along the vacuum axis, and a
dipole correction centered on the slab's mass-weighted center. The matched
cubic cells define a comparison of compositions at fixed crystal topology.
The stable iodide polymorphs depend on temperature and environment.
[CsSnI₃ phase thermodynamics](https://doi.org/10.1021/acs.chemmater.2c03475)

The fine-tuned checkpoint predicts the full RPBE-D3 energy and forces
directly. The run uses fresh status, data, model, and scratch paths under
the OC25 setup. [OC25 DFT methods](https://arxiv.org/html/2509.17862v2#S5.SS1.SSS3) · [VASP DFT-D3](https://vasp.at/wiki/DFT-D3)

For a geometry-only check, use
`python perovskite_bulk.py --preview --output /tmp/alf-perovskite-preview`
and pass `--bulk-dir /tmp/alf-perovskite-preview` to the benchmark's
`prepare` subcommand.

For an independent comparison, hold out bridge and hollow initial poses,
label them with the same VASP settings, and score pretrained UMA OC20,
pretrained UMA OC25, and fine-tuned UMA OC25 on those exact geometries.
The MACE-MH-1 surface head provides an additional PBE baseline.

```bash
for partition in end-members alloy; do
    python perovskite_benchmark.py prepare "${partition}.extxyz" --partition "$partition"
    python perovskite_benchmark.py label "${partition}.extxyz" "${partition}_dft.extxyz" --scratch "runs/heldout_${partition}"
    python perovskite_benchmark.py score "${partition}_dft.extxyz" uma /path/to/pretrained_uma.pt --uma-task oc20
    python perovskite_benchmark.py score "${partition}_dft.extxyz" uma /path/to/pretrained_uma.pt --uma-task oc25
    python perovskite_benchmark.py score "${partition}_dft.extxyz" mace /path/to/mace-mh-1.model
    python perovskite_benchmark.py score "${partition}_dft.extxyz" uma /path/to/fine_tuned_uma.pt --uma-task oc25
done
```

The alloy partition reserves CsPb₀.₅Sn₀.₅I₃ for composition transfer. It uses
ordered checkerboard and stripe B-site arrangements on the same interpolated
pseudocubic lattice and includes both Pb-atop and Sn-atop poses with both
ligands. Keep the two alloy arrangements and all benchmark labels outside
the training run. The default is 64 held-out end-member poses and 16 alloy poses.
[All-inorganic Pb–Sn alloy experiments](https://doi.org/10.1039/D0RA10110A)

The scorer reports adsorbate and nearby substrate force errors and relative
pose energies at fixed composition, including separate errors by alloy
binding metal. Run short fixed-cell trajectories from
the same held-out geometry, velocity seed and thermostat settings:

```bash
python perovskite_benchmark.py md end-members.extxyz vasp dft_md.extxyz --scratch runs/heldout_md --frame 0 --seed 42
python perovskite_benchmark.py md end-members.extxyz uma uma_md.extxyz --checkpoint /path/to/fine_tuned_uma.pt --frame 0 --seed 42
python perovskite_benchmark.py trajectory dft_md.extxyz
python perovskite_benchmark.py trajectory uma_md.extxyz
```

Repeat across independent starting poses and compare donor distances to Pb
and Sn and ligand-axis tilts. Keep those trajectories outside the fitting data.

## Chalcogenide adsorption with UMA OC25

The dedicated case study uses methylamine and methylphosphine on zinc-blende
(110) CdSe, CdTe, HgTe, and HgSe. The first three compositions supply 48
bootstrap labels, eight for each material–ligand pair. The builder places the
donor near exposed Cd or Hg and varies distance, tilt, azimuth, and lateral
offset. A separate fixed pool provides a non-adaptive label baseline. The
holdout pool includes independent poses on seen compositions and every HgSe
pose. The preparation manifest records material, ligand, cell partition,
metal–donor distance, and donor-to-carbon tilt.

From the repository root:

```bash
python -m examples.surface_training.chalcogenide prepare /scratch/chalc-preview --preview
python -m examples.surface_training.chalcogenide relax /scratch/chalc-preview /scratch/relaxed-slabs
python -m examples.surface_training.chalcogenide prepare /scratch/chalc --slabs /scratch/relaxed-slabs \
  --checkpoint /path/to/uma-s-1p2p1.pt --vasp-command vasp_std
```

The preview uses approximate prototype bulk cells. Qualify the lattice constants
and clean slab dimensions for production. Supply relaxed, stoichiometric,
periodic slabs named after their compositions with the normal along +z.
The preparer uses a common cell for each composition and leaves at least 20 Å
of vacuum above the occupied slab and ligand extent. The slab plus ligand has
at most 200 atoms. The VASP configuration uses RPBE-D3 with zero damping,
450 eV, Gaussian smearing of 0.05 eV, a Γ-centered reciprocal mesh with
0.15 Å⁻¹ in-plane spacing, and a slab dipole correction. Use the same licensed
PBE PAW release on all workers. Set `VASP_PP_PATH` and optionally
`VASP_PP_VERSION` as in the perovskite example. Check reference convergence and
compare SOC on matched HgTe and HgSe poses before generating training labels.

In the prepared run directory:

```bash
python -m alframework --master master_config.json --check_config
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json --test_qm
python -m alframework --master master_config.json
```

Both UMA members start each fit from the original OC25 checkpoint and train
on cumulative accepted labels with the existing 20:2 energy and force loss.
The pilot retains the foundation elemental references. Check PAW and energy
conventions before comparing absolute energies. The first fit uses bootstrap
labels. Later generations drive fixed-cell batched TorchSim trajectories.
Adjust the acquisition thresholds if ordinary MD finds too few eligible poses.

From the repository root, label and score the same independent structures at
each checkpoint:

```bash
python -m examples.surface_training.chalcogenide label /scratch/chalc holdout holdout.extxyz
python -m examples.surface_training.chalcogenide label /scratch/chalc holdout holdout-soc.extxyz --soc --count 8
python -m examples.surface_training.chalcogenide score holdout.extxyz /path/to/uma-s-1p2p1.pt
python -m examples.surface_training.chalcogenide score holdout.extxyz /scratch/chalc/runs/models/model-0000
```

For a fixed-pool comparison, copy the prepared run to a fresh directory
before training. Put the finalized shared bootstrap shard in its `runs/h5/`,
label the same number of additional poses as used by the AL comparison with
`label /scratch/chalc fixed /scratch/fixed-run/runs/h5/data-0001.h5 --count N`,
and run `--test_ml` in the fixed run. Keep the holdout pool outside training.

Use `chalcogenide md RUN OUTPUT --frame 0 --seed 42` to generate the VASP
reference trajectory. Pass a UMA file or ensemble with `--checkpoint` to run
the corresponding model trajectory from the same start. The `trajectory`
subcommand reports donor distance and tilt percentiles from the saved frames.
Use `sample RUN MODEL OUTPUT --udd-weight -0.1` for a separate ordinary versus
UDD batch on the same held-out starts. The negative coefficient lowers the
biased energy in high-disagreement regions. Check stability and bias strength
on a short pilot before extending the sampling budget.

## Extend the low-index facet coverage

| Material family | Starting facet | Follow-up facets |
| --- | --- | --- |
| Cubic halide perovskites | Pseudocubic (100), with CsX and MX₂ terminations | (110) and (111) |
| Zinc-blende chalcogenides | Stoichiometric (110) | (100) and (111) |

Select follow-up orientations explicitly with `miller_indices` in the
builder config. Screen clean slabs for stoichiometry, termination, polarity,
and reconstruction before decorating them. Qualify a charge-compensation
or reconstruction model for polar terminations before adding them to a
labeling run. Verify thickness and lateral size under the atom budget for
each facet. Use facet-specific scores, and hold out a facet on a composition
present in training to measure facet transfer independently of composition
transfer.
[Cubic CsPbBr₃ low-index surfaces](https://doi.org/10.1039/D0CP04893C) · [CdTe(100) and (111) polarity and reconstructions](https://doi.org/10.1016/j.apsusc.2020.146832)
