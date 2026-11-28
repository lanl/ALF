# Learn S₀ and S₁ surfaces from a SMILES panel

This example trains HIPPYNN from scratch on gas-phase CAM-B3LYP/6-31G* TDA
energies and forces. The supplied panel has 24 training molecules and six
whole-molecule holdouts across five carbonyl families. It spans 4–28 atoms
including hydrogens. Replace `panel.json` with a JSON list of SMILES or add
named records with `family` and `split` fields for a larger panel. The default
limit is 30 atoms including hydrogen. Each input must describe a distinct
neutral, connected, closed-shell molecule.

The panel pairs carbonyls with reference vertical excitations (formaldehyde,
acetaldehyde, acetone, acrolein, glyoxal, and propynal) and experimentally
studied methylglyoxal and cyclobutanone with related molecules that vary
branching, ring size, carbonyl position, conjugation, or flexibility.
3-Methylbutan-2-one pairs branching with the two pentanone isomers.
4-Methylcyclohexanone varies substitution at fixed ring size. Propynal
adds a triple bond conjugated with the carbonyl.
The cyclobutanone 200 nm experiment starts in S₂. The S₀/S₁ model here
tests the lower surfaces and off-equilibrium gradients.
[QUESTDB](https://arxiv.org/abs/2011.14675) · [Methylglyoxal spectroscopy](https://doi.org/10.1029/95JD00541) · [Cyclobutanone experiment](https://doi.org/10.1063/5.0266559)

The six whole-molecule tests are 2-methylpropanal (branching), pentan-3-one
(carbonyl position), cycloheptanone (ring size), methacrolein (substitution
next to the conjugated carbonyl), diacetyl (α-dicarbonyl substitution), and
nonan-2-one (larger, more flexible ketone). Their purposes are also recorded
in the panel manifest.

Install ALF and its topology and ALCHEMI extras, HIPPYNN, and GPU4PySCF in
the worker environments. From the repository root, prepare study runs with the
same panel and seed:

```bash
python examples/molecular_excited_states/prepare.py /path/to/ensemble --mode ensemble
python examples/molecular_excited_states/prepare.py /path/to/ntk --mode ntk
python examples/molecular_excited_states/prepare.py /path/to/random --mode random
```

Before production labeling, run the CPU PySCF qualification on one
prespecified molecule from each family:

```bash
python examples/molecular_excited_states/qualify.py /path/to/ensemble basis_check.json
```

The script relaxes each ground state with CAM-B3LYP/6-31G* and evaluates the
lowest three singlet TDA roots with 6-31G* and 6-31+G* at the same relaxed
and C=O-stretched geometries. It matches roots by the absolute overlap
of their normalized transition densities and reports oscillator strengths.
The report separates cases that change root order, have matched overlap
below 0.8, or shift a matched excitation by more than 0.2 eV. Inspect
flagged cases before interpreting the production results. The
`s1_basis_sensitive` field identifies flags affecting the production root.
Use `--all`
to extend the check to all training molecules. These diagnostic calculations use
conventional CPU integrals. Production forces use the GPU4PySCF settings in
the generated QM configuration.

Add `--local` to use the single-host Parsl configuration on a workstation with
one GPU. With the default Slurm configuration, set `ALF_PARTITION` and adapt the
worker resources. Use the same settings for all comparisons. From each run
directory:

```bash
python -m alframework --master master_config.json --check_config
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json --test_qm
python -m alframework --master master_config.json
```

The example uses ALF's `rdkit_molecules` module for geometry preparation.
It embeds reference conformers and generates screened distortion pools with
farthest-point seed selection. The same functionality is available outside
this example via `python -m alframework.builders.rdkit_molecules`.
`prepare.py` supplies the study's QM, HIPPYNN, and acquisition settings.
The built-in collection builder cycles through the training seed files in
panel order and draws one geometry per molecule on each pass. Failed labels
can be replaced from later passes, so accepted counts can vary by molecule.
Later iterations start from seed geometries or completed label shards and
propagate short S₀ or S₁ trajectories. The molecule's topology follows it
through MD, labeling, and replay. The same HIPPYNN fits both states with a
shared train-split elemental energy reference. Force losses and NTK use real
atoms only.

The initial settings rank by ensemble gap spread, single-model NTK gap
uncertainty, or seeded random selection over frames passing the physical
screens. All use the same bootstrap, label caps, and cohort sizes. Compare
results at equal **accepted** QM label counts. Calibrate the initial ensemble
and NTK acquisition cutoffs on the pilot so each policy yields adequate
candidates, and record how many trajectories and QM calculations each policy
needed. Set `ALF_TASK_TIMELINE_DIR` to a run-specific shared directory in
each worker environment. The timeline includes ML training task seconds and
records cold NTK fitting as the `ntk_fit` phase of its sampler task.

## Independent evaluation

Create and score an external test shard using the same QM settings:

```bash
python examples/molecular_excited_states/evaluate.py label-holdout /path/to/ensemble holdout.h5
python examples/molecular_excited_states/evaluate.py score /path/to/ensemble/models/model-0000 holdout.h5
```

Keep this shard outside the run's HDF5 directories and use it for every policy.
The scorer checks for molecule overlap with all training and internal evaluation
data. It reports per-molecule S₀/S₁ energy and force MAEs and the gap MAE.
Use `compare` to verify a matched budget and evaluate the same held-out shard:

```bash
python examples/molecular_excited_states/evaluate.py compare holdout.h5 \
  --ensemble /path/to/ensemble/models/model-0001 \
  --ntk /path/to/ntk/models/model-0001 \
  --random /path/to/random/models/model-0001
```

The command reads the actual label counts from the training manifests and
gives each held-out molecule equal weight in the summary force and gap MAEs.
Use `--timelines ENSEMBLE NTK RANDOM` with the worker timelines through
the corresponding study endpoints to include stage timing and NTK fit
seconds, which are already part of sampler time. Report time separately
from the S₁ force and S₁–S₀ gap errors.

For transfer to different conformations of known molecules, separately generate
and label complete independent trajectories with unique IDs, store each frame
with its `molecule_id` and an ALF-style `<trajectory>-s<step>` frame ID, then
score with `--trajectory-holdout`. Use more than one split of the panel or
hold out an entire family to study how transfer changes with chemical distance.

S₁ is the lowest singlet TDA root at each geometry. Inspect state character
and basis sensitivity on a subset of conformers before production labeling,
particularly for diffuse excited states and near-degenerate roots.

## Broader chromophore pilot

Use the 35-molecule chromophore preset for the paper pilot. It has 27 training
molecules and eight whole-molecule holdouts across six families. Para-terphenyl
and triphenylamine put molecules above 30 atoms in training. Meta-terphenyl and
trans,trans-1,4-distyrylbenzene test transfer at 32 and 40 atoms. The optional
six-molecule validation panel reserves a separate set for model-width and
property-readout selection.

```bash
python -m examples.molecular_excited_states.prepare /scratch/chrom --preset chromophore --local \
  --validation-panel examples/molecular_excited_states/property_validation.json
python -m examples.molecular_excited_states.qualify /scratch/chrom basis-check.json --nroots 5 --gradients
```

The preset defaults to `hiphop64`. The 27 training molecules contribute eight
initial candidates each for a bootstrap target of 216 accepted labels, with
up to 108 spares from the remaining four seeds per molecule. The total label
budget is 32 per training molecule on average. The preset records oscillator
strengths and transition dipoles. RDKit conformers are deduplicated
before ALF creates the distortion pools. Distinct conformer pools are interleaved
within a molecule while bootstrap files are interleaved across molecules. The
parent manifest records the source pools and hashes. For a shorter preparation
use `--pool-size 12 --seed-size 8`, which uses all seeds for the initial
bootstrap and leaves zero spares.

Qualify the chosen basis and state count before collecting production labels.
`--gpu-labels` times complete GPU4PySCF labels for both bases and reports sampled
process GPU memory when `nvidia-smi` is available. Prepare a fresh run with
`--basis '6-31+g*'` and `--nroots 5` if that basis and state count pass. The run
configuration supplies the state schema to both the labeler and the sampler.
For a later gap-seeking comparison, prepare a separate run with
`--gap-seeking-after LABELS` after qualifying the ordinary-dynamics checkpoint.
Use the same dynamics settings for the ensemble, NTK, and random acquisition
comparison. Choose the width using the same training and validation data with
`--model hiphop64` and `--model hiphop145`.

For production, pass `--panel production.json` with the chromophore preset and
120 training, 18 validation, and 24 test molecules. Use `train`, `validation`,
and `holdout` in the `split` field. Include the source identifier, family, and
SMILES in each record. Supply the exact same manifest, basis, state count and
seed to all policy comparisons. `evaluate.py score` and `compare` take
`--run /scratch/chrom` to load the state schema. To label the validation set,
use `evaluate.py label-holdout /scratch/chrom validation.h5 --split validation`.

### Optional property readouts

Run these commands from the repository root with a finished dynamics checkpoint.
The readout is a separate fitting stage on the frozen atomic representation.
Use molecule-disjoint optical or dipole data for readout fitting, validation,
and evaluation. The scorer checks these splits against the dynamics training
manifest as well.

```bash
run=/scratch/chrom
member="$run/models/model-0000/model-00"
python -m examples.molecular_excited_states.properties fit oscillator "$run" "$member" strengths.pt \
  --training "$run"/h5store/data-*.h5 --validation validation.h5
python -m examples.molecular_excited_states.properties score strengths.pt holdout.h5
python -m examples.molecular_excited_states.properties predict strengths.pt new-geometries.extxyz
```

The oscillator head predicts nonnegative dimensionless strengths for each
\(S_0\)-to-\(S_n\) transition, in ascending root order. It minimizes MSE on the
strengths. The scorer combines them with model excitation energies to calculate
broadened absorption overlap and reports per-transition MAE.

For a small permanent-dipole pilot, select shards that contain only the
molecules you want to label. The CPU calculation obtains \(S_0\) from the SCF
density and \(S_1\) from finite differences of the state energy in an electric
field. Use `--check-step` on a small reference set to validate sensitivity to
the field step. Store the sidecars outside the dynamics label directory.

```bash
python -m examples.molecular_excited_states.properties label-dipoles "$run" train-dipoles.json \
  "$run/h5store/data-0000.h5" --frames-per-molecule 2
python -m examples.molecular_excited_states.properties label-dipoles "$run" valid-dipoles.json \
  validation.h5 --frames-per-molecule 2 --check-step
python -m examples.molecular_excited_states.properties label-dipoles "$run" test-dipoles.json \
  holdout.h5 --frames-per-molecule 2
python -m examples.molecular_excited_states.properties fit charge "$run" "$member" charges.pt \
  --training train-dipoles.json --validation valid-dipoles.json
python -m examples.molecular_excited_states.properties score charges.pt test-dipoles.json
```

The charge readout projects the atomic charges to zero total per molecule and
state. It minimizes componentwise dipole MSE in e Å. The scorer reports the
latent charges in atom order, permanent dipoles, and the error in the change in
dipole from \(S_0\) to \(S_1\). Use the same `predict` subcommand with
`charges.pt` for new neutral geometries.
