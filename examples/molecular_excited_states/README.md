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
to extend the check to the full panel. These diagnostic calculations use
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

Preparation creates one geometry pool and fixed-topology reference per molecule.
RDKit embeds the reference and generates diverse conformations and local
distortions, with geometric screening and farthest-point seed selection.
Bootstrap interleaves equal numbers of seeds from each training molecule.
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
