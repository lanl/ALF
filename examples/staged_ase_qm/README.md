# Staged QM labeling through ASE

Run two fixed-geometry calculations on one QM worker when the second needs a file
produced by the first. The first stage requests energy and prepares an initial
electronic state. The target stage supplies the energy and forces stored by ALF.

| Workflow | Prepared input | Files for an ALF run |
| --- | --- | --- |
| ORCA orbital warm start | Small-basis orbitals in `orca.gbw` | `staged_master_config_fragment.json` and `staged_orca_qm_config.json` |
| VASP spin-polarized label | Nonmagnetic charge density in `CHGCAR` | `staged_vasp_master_config_fragment.json` and `staged_vasp_qm_config.json` |

## Run one structure

With ALF installed, run from the repository root in the environment that can
launch your licensed QM program. ORCA uses H₂ for the smoke run. VASP uses a
two-atom periodic bcc iron cell and needs the Fe PAW potential available to ASE.

```bash
python examples/staged_ase_qm/run_workflow.py orca --command /absolute/path/to/orca
python examples/staged_ase_qm/run_workflow.py vasp --command /absolute/path/to/vasp_std
```

Use `--command` to supply the same wrapper or launch command the QM worker
uses. The script prints the stage directories and final energy and forces.
Each backend has a separate default scratch directory. Use `--scratch-dir`
to choose another location.

## ORCA orbital warm start

The `guess` stage converges a smaller-basis calculation. ALF copies its
`orca.gbw` to `guess.gbw` in the `target` directory. The target reads the
guess with `MORead` and `%moinp`, then computes the larger-basis energy and
gradient on the sampled geometry. Check `target/orca.out` for the SCF startup
report that it read the guess file and for normal termination.

A better initial guess can help a difficult SCF, including open-shell systems.
For production, choose the charge, multiplicity, methods, and basis sets for
your chemistry. Compare both stages' combined cost and the final state with a
target-only calculation before adopting the warm start. [ORCA initial guess and restart guide](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/initialguess.html)

## VASP spin-polarized label

The `charge_density` stage runs with `ISPIN=1` and writes `CHGCAR`. The
`magnetic` stage reads the copied charge density with `ICHARG=1` and
initializes its wavefunctions with `ISTART=0`. Explicit `MAGMOM` values seed
the `ISPIN=2` calculation. The two entries match the example cell's two Fe
atoms. Check `magnetic/vasp.out` for a report that VASP read `CHGCAR`, and
check the electronic convergence and final magnetization in `magnetic/OUTCAR`.
For a compatible wavefunction restart, the same handoff can copy `WAVECAR`
with `LWAVE` enabled at the source and `ISTART=1` at the target. Confirm
that VASP read the supplied file. [VASP magnetic initialization](https://vasp.at/wiki/MAGMOM), [VASP restart output](https://vasp.at/wiki/Restart_and_output_files_cheat_sheet)

## Use in an ALF run

Merge the selected master fragment into your run's master configuration and
copy the QM configuration it names. Set `QM_run_command` for both stages
and match resources to the QM executor. Keep ASE energy and force multipliers
at 1.0 for eV and eV/Angstrom. Run
`python -m alframework --master master_config.json --test_qm` on your
installation before starting an active-learning run.

The ORCA integration test runs ASE's input writer and output parser against
a stand-in executable. The VASP test uses ASE's Vasp calculator settings
with a stand-in for the calculation. The commands above run the licensed
backends for local qualification.

For a single ORCA or Q-Chem calculation, use `master_config_fragment.json`
with `orca_qm_config.json` or set `QM_config_path` to `qchem_qm_config.json`.
ASE appends ORCA's input filename to the command. Set its core count with
`%pal nprocs N end`. For Q-Chem, `QM_run_command` is the complete command
including `PREFIX.inp` and `PREFIX.out`. Supply your Q-Chem wrapper's thread
flags in that command, or omit it and let ASE construct the command from
`np` and `nt`.
