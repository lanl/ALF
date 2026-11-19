# Staged QM labeling through ASE

Run two fixed-geometry calculations on one QM worker when the second needs a file produced by the
first. The first stage requests energy and prepares an initial electronic state. The target stage
supplies the energy and forces stored by ALF.

| Workflow | Prepared input | Files for an ALF run |
| --- | --- | --- |
| ORCA orbital warm start | Small-basis orbitals in `orca.gbw` | `staged_master_config_fragment.json` and `staged_orca_qm_config.json` |
| VASP spin-polarized label | Nonmagnetic charge density in `CHGCAR` | `staged_vasp_master_config_fragment.json` and `staged_vasp_qm_config.json` |

## Run one structure

With ALF installed, run from the repository root in the environment that can launch your licensed QM
program. ORCA uses H₂ for the smoke run. VASP uses a two-atom periodic bcc iron cell and needs the
Fe PAW potential available to ASE.

```bash
python examples/staged_ase_qm/run_workflow.py orca --command /absolute/path/to/orca
python examples/staged_ase_qm/run_workflow.py vasp --command /absolute/path/to/vasp_std
```

Use `--command` to supply the same wrapper or launch command the QM worker uses. The script prints
the stage directories and final energy and forces. Each backend has a separate default scratch
directory. Use `--scratch-dir` to choose another location.

## ORCA orbital warm start

The `guess` stage converges a smaller-basis calculation. ALF moves its `orca.gbw` to `guess.gbw` in
the `target` directory. The target reads the guess with `MORead` and `%moinp`, then computes the
larger-basis energy and gradient on the sampled geometry. Check `target/orca.out` for the SCF
startup report that it read the guess file and for normal termination.

A better initial guess can help a difficult SCF, including open-shell systems. For production,
choose the charge, multiplicity, methods, and basis sets for your chemistry. Compare both stages'
combined cost and the final state with a target-only calculation before adopting the warm start.
[ORCA initial guess and restart guide](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/initialguess.html)

## VASP spin-polarized label

The `charge_density` stage runs with `ISPIN=1` and writes `CHGCAR`. The `magnetic` stage reads that
charge density with `ICHARG=1` and initializes its wavefunctions with `ISTART=0`. Its
`initial_magnetic_moments` setting gives every Fe atom 2.2 μB, so ASE writes a `MAGMOM` that fits a
cell of any size; VASP uses it because the `CHGCAR` has no magnetization. Check `magnetic/vasp.out`
for a report that VASP read `CHGCAR`, and check the electronic convergence and final magnetization
in `magnetic/OUTCAR`. For a compatible wavefunction restart, the same handoff can pass on `WAVECAR`
with `LWAVE` enabled at the source and `ISTART=1` at the target. Confirm that VASP read the supplied
file. [VASP magnetic initialization](https://vasp.at/wiki/MAGMOM),
[VASP restart output](https://vasp.at/wiki/Restart_and_output_files_cheat_sheet)

## Use in an ALF run

Merge the selected master fragment into your run's master configuration and copy the QM
configuration it names. Set `QM_run_command` for both stages and match resources to the QM executor.
Keep ASE energy and force multipliers at 1.0 for eV and eV/Angstrom. Run
`python -m alframework --master master_config.json --test_qm` on your installation before starting
an active-learning run.

Stage files stay in the QM scratch directory after a label. For production, add
`"remove_scratch_on_success": true` beside `stages` to delete each molecule's directory once its
label passes ALF's checks; a failed label's directory is kept for inspection.

ALF's tests run both recipes through ASE's own input writers with stand-ins for the licensed
programs: a stand-in ORCA executable, and a stand-in for the VASP run on iron cells of 2 and 16
atoms. The commands above run the real programs for local qualification.

For a single ORCA calculation, see `examples/simple_water`; the QM interface guide describes single
ORCA and Q-Chem configurations.