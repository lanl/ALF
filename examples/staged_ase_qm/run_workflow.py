"""Run one staged label on H2 with ORCA or on bcc iron with VASP."""

import argparse
import json
from pathlib import Path

from ase import Atoms
from ase.build import bulk

from alframework.qm_interfaces.ase.workflow import label_workflow
from alframework.tools.molecules import MoleculesObject


EXAMPLE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["orca", "vasp"])
    parser.add_argument("--command", required=True, help="Backend executable or worker command")
    parser.add_argument("--scratch-dir", type=Path, help="Override the example's scratch directory")
    args = parser.parse_args()

    if args.backend == "orca":
        master_file = "staged_master_config_fragment.json"
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        molecule_id = "h2"
    else:
        master_file = "staged_vasp_master_config_fragment.json"
        atoms = bulk("Fe", "bcc", a=2.86, cubic=True)
        molecule_id = "bcc-fe"

    master = json.loads((EXAMPLE_DIR / master_file).read_text())
    config = json.loads((EXAMPLE_DIR / master["QM_config_path"]).read_text())
    for stage in config["stages"]:
        stage["QM_run_command"] = args.command
    molecule = MoleculesObject(atoms, molecule_id)
    scratch = args.scratch_dir or master["QM_scratch_dir"]
    result = label_workflow(molecule, config, scratch, master["properties_list"])
    metadata = result.get_metadata()
    for stage in metadata["qm_workflow"]["stages"]:
        print(f"{stage['name']}: {stage['directory']}")
    if not result.check_convergence():
        raise SystemExit(f"Label failed in {metadata['qm_failed_stage']}: {metadata['qm_error']}")

    properties = result.get_results()
    print(f"Final energy (eV): {properties['energy']}")
    print("Final forces (eV/Angstrom):")
    print(properties["forces"])


if __name__ == "__main__":
    main()
