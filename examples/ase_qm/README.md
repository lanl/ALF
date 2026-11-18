# ORCA and Q-Chem through ASE

QM configuration fragments for labeling with ASE's ORCA and Q-Chem calculators. Add them to a run
that has its own builder, sampler, trainer and Parsl configuration; they aren't a complete run.
The QM interface guide explains each setting.

| File | Use |
| --- | --- |
| `master_config_fragment.json` | `QM_task`, scratch directory and `properties_list` for `ase_calculator_task` |
| `orca_qm_config.json` | One ORCA single point |
| `qchem_qm_config.json` | One Q-Chem single point; point `QM_config_path` at it instead |
| `staged_master_config_fragment.json` | The same for `ase_workflow_task` |
| `staged_orca_qm_config.json` | Two ORCA stages; the second, with the larger basis, gives the label |

1. Merge a master fragment into the run's master configuration, and copy the QM configuration it
   names.
2. Replace the command placeholder, and choose the method, basis, SCF settings and cores for your
   chemistry; the ones here only show the format.
3. Keep the energy and force multipliers at 1.0: ASE gives eV and eV/Angstrom.
4. Label one structure with `python -m alframework --master master_config.json --test_qm` before
   a full run.

## ORCA

`QM_run_command` is the ORCA executable, or a wrapper that loads its environment; ASE adds the input
file name and captures the output:

```bash
#!/bin/bash
set -e
source /path/to/orca-environment.sh
exec /path/to/orca "$@"
```

Set cores with `%pal nprocs N end` in `orcablocks`, matching the QM executor's CPUs. Forces need
`EnGrad`. ASE doesn't read quadrupoles or Hirshfeld charges; the legacy ORCA tasks do.

## Q-Chem

`QM_run_command` is the whole command, with ASE's `PREFIX.inp` and `PREFIX.out`, and is used as
written, so a wrapper must add its own thread flags (for example `exec qchem -nt 4 "$@"`). Leave it
out to have ASE build its command from `np` and `nt`. The legacy Q-Chem task calls its command
differently, so its wrappers can't be reused unchanged.

## Workflows

In `staged_orca_qm_config.json`, each stage is an independent calculation from the same geometry,
and nothing passes between stages; the last stage gives the label. To repeat one calculation and
average it only when the repeats agree, use `"result_mode": "mean_if_agree"` with identical stages
and `agreement` tolerances, as `examples/IL` does.
