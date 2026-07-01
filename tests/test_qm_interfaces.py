import numpy as np
import pytest
from ase import Atoms

from alframework.qm_interfaces.orca5_interface import orcaGenerator
from alframework.qm_interfaces.qchem_DFT_interface import qchemGenerator
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.interface_checks import check_molecule_result, check_task_convergence


def write_orca_outputs(path):
    (path / "orca.engrad").write_text(
        """The current total energy in Eh
#
-2.500000
#
The current gradient in Eh/bohr
#
0.100000
0.200000
0.300000
0.400000
0.500000
0.600000
#
"""
    )
    (path / "orca.log").write_text(
        """SCF CONVERGED AFTER 6 CYCLES
TOTAL RUN TIME: 0 days
HIRSHFELD ANALYSIS
header
SPIN  
  0 H 0.10 0.01
  1 H 0.20 0.02
TOTAL
"""
    )
    (path / "orca_property.txt").write_text(
        """Electric_Properties
header
Total Dipole moment:

    X 1.0
    Y 2.0
    Z 3.0
---------------------
Electric_Properties
header
Total quadrupole moment

    XX 1.0 0.1 0.2
    YY 0.1 2.0 0.3
    ZZ 0.2 0.3 3.0
# --------------
Total Something Energy: -2.4
"""
    )


def test_orca_input_writer_and_parser(tmp_path):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    generator = orcaGenerator(
        scratch_path=str(tmp_path) + "/",
        nproc=2,
        orcainput="engrad HF",
        orcablocks="%maxcore 100",
    )

    generator.write_orca_input(atoms, charge=0, multiplicity=1, job_path=str(tmp_path) + "/", filename="orca.inp")
    input_text = (tmp_path / "orca.inp").read_text()
    assert "! engrad HF" in input_text
    assert "%pal nproc 2 end" in input_text
    assert "* xyzfile 0 1 input.xyz" in input_text

    write_orca_outputs(tmp_path)
    assert generator.check_normal_termination(str(tmp_path / "orca.log"))
    parsed = generator.parse_output(
        str(tmp_path) + "/",
        "orca",
        natom=2,
        properties=["energy", "forces", "dipole", "quadrupole", "hirshfeld", "hirshfeld_spin"],
    )

    assert parsed["energy"] == -2.5
    np.testing.assert_allclose(parsed["forces"], -np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    np.testing.assert_allclose(parsed["dipole"], [1.0, 2.0, 3.0])
    assert parsed["quadrupole"].shape == (3, 3)
    np.testing.assert_allclose(parsed["hirshfeld"], [0.10, 0.20])
    np.testing.assert_allclose(parsed["hirshfeld_spin"], [0.01, 0.02])


@pytest.mark.parametrize(
    "unit,energy_scale,force_scale",
    [
        ({"energy": "hartree", "length": "bohr"}, 1.0, 1.0),
        ({"energy": "ev", "length": "angstrom"}, 27.2113834, 27.2113834 / 0.5291772083),
    ],
)
def test_orca_parser_applies_unit_conversion(tmp_path, unit, energy_scale, force_scale):
    write_orca_outputs(tmp_path)
    generator = orcaGenerator(unit=unit)

    parsed = generator.parse_output(str(tmp_path) + "/", "orca", natom=2, properties=["energy", "forces"])

    assert parsed["energy"] == -2.5 * energy_scale
    np.testing.assert_allclose(parsed["forces"], -np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) * force_scale)


@pytest.mark.parametrize(
    "unit",
    [
        {"energy": "kcal", "length": "bohr"},
        {"energy": "hartree", "length": "nanometer"},
    ],
)
def test_orca_generator_rejects_unknown_units(unit):
    with pytest.raises(KeyError):
        orcaGenerator(unit=unit)


def test_qchem_input_writer(tmp_path):
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]])
    generator = qchemGenerator(scratch_path=str(tmp_path), qcheminput="JOBTYPE FORCE", qchemblocks="$pcm\n$end")
    input_path = tmp_path / "qchem.in"

    generator.write_qchem_input(atoms, charge=-1, mult=2, filename=str(input_path))

    text = input_path.read_text()
    assert "$molecule\n-1 2\n" in text
    assert "8 0.0 0.0 0.0" in text
    assert "1 0.0 0.7 0.7" in text
    assert "$rem\nJOBTYPE FORCE\n$end" in text


def test_orca_task_can_be_tested_with_mocked_single_point(tmp_path, monkeypatch):
    from alframework.qm_interfaces import orca5_interface

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    molecule = MoleculesObject(atoms, "h")
    expected = {"energy": -1.0, "forces": np.zeros((1, 3)), "converged": True}

    def fake_single_point(self, molecule, prefix="orca", properties=None):
        return expected

    monkeypatch.setattr(orca5_interface.orcaGenerator, "single_point", fake_single_point)
    task_func = getattr(orca5_interface.orca_calculator_task, "func", orca5_interface.orca_calculator_task)
    result = task_func(
        molecule,
        {
            "ncpu": 1,
            "orca_env_file": None,
            "QM_run_command": "orca",
            "orcasimpleinput": "HF",
            "orcablocks": "",
        },
        str(tmp_path),
        {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]},
    )

    check_molecule_result(result, natoms=1, required_properties=["energy", "forces", "converged"])
    check_task_convergence(result, True)
    assert result.get_results() == expected
