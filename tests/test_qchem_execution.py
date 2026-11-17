"""The legacy Q-Chem generator runs its configured command and labels only output it can fully read."""

import shlex
import sys

import numpy as np
import pytest
from ase import Atoms

from alframework.qm_interfaces.legacy.qchem_DFT_interface import qchemGenerator

HARTREE_IN_EV = 27.2113834
BOHR_IN_ANGSTROM = 0.5291772083


def qchem_stdout(natoms=2):
    """Output with an energy of -2.5 Hartree and every gradient component 0.1 Hartree/Bohr."""
    lines = ["Total energy in the final basis set = -2.5", "Gradient of SCF Energy"]
    for offset in range(0, natoms, 6):
        count = min(6, natoms - offset)
        lines.append(" atom headers")
        lines += [f"{axis + 1} " + " ".join(["0.1"] * count) for axis in range(3)]
    return "\n".join(lines)


def printing(text):
    return shlex.quote(sys.executable) + " -c " + shlex.quote(f"print({text!r})")


@pytest.mark.parametrize("natoms", [2, 7])
def test_the_wrapper_runs_beside_its_input_with_the_environment_and_its_output_is_converted(tmp_path, natoms):
    script = tmp_path / "fake qchem.py"
    script.write_text("import os, sys\nfrom pathlib import Path\n"
                      'assert Path("qchem.in").exists()\n'
                      "assert Path(sys.argv[1]).is_dir()\n"
                      'assert os.environ["LOADED_FOR_QCHEM"] == "yes"\n'
                      f"print({qchem_stdout(natoms)!r})\n")
    environment = tmp_path / "load env.sh"
    environment.write_text("export LOADED_FOR_QCHEM=yes\n")
    calc = qchemGenerator(scratch_path=str(tmp_path / "scratch space"), store_path=str(tmp_path / "stored"),
                          qchem_command=shlex.quote(sys.executable) + " " + shlex.quote(str(script)),
                          qchem_env_file=str(environment), unit={"energy": "ev", "length": "angstrom"})
    result = calc.single_point(Atoms("H" * natoms))
    assert result["converged"]
    assert result["energy"] == pytest.approx(-2.5 * HARTREE_IN_EV)
    np.testing.assert_allclose(result["forces"], np.full((natoms, 3), -0.1 * HARTREE_IN_EV / BOHR_IN_ANGSTROM))
    assert (tmp_path / "stored" / "qchem" / "qchem.out").exists()


@pytest.mark.parametrize("command,environment", [("exit 7", None), ("true", "missing.sh")])
def test_a_failed_command_or_environment_is_unconverged(tmp_path, command, environment):
    calc = qchemGenerator(scratch_path=str(tmp_path), qchem_command=command, qchem_env_file=environment)
    result = calc.single_point(Atoms("H"))
    assert not result["converged"] and "exited" in result["error"]


@pytest.mark.parametrize("text", ["", "Total energy in the final basis set = -1",
                                  "Total energy in the final basis set = nan",
                                  "Gradient of SCF Energy\nheader\n1 nope"])
def test_missing_or_unreadable_output_is_unconverged(tmp_path, text):
    result = qchemGenerator(scratch_path=str(tmp_path), qchem_command=printing(text)).single_point(Atoms("H"))
    assert not result["converged"] and result["error"]
