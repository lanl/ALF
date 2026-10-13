from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator

from alframework.tools.molecules_class import MoleculesObject


def check_energy_force_results(results, natoms):
    assert "energy" in results
    assert "forces" in results
    assert np.asarray(results["energy"]).shape in [(), (1,)]
    np.testing.assert_equal(np.asarray(results["forces"]).shape, (natoms, 3))


def check_molecule_result(result, natoms, required_properties):
    assert isinstance(result, MoleculesObject)
    if result.get_atoms() is not None:
        assert len(result.get_atoms()) == natoms
    for prop in required_properties:
        assert prop in result.get_results()
    if {"energy", "forces"}.issubset(required_properties):
        check_energy_force_results(result.get_results(), natoms)


def check_task_convergence(result, expected):
    assert isinstance(expected, bool)
    assert result.check_convergence() is expected


def check_ase_calculator_interface(calc, atoms, properties):
    assert isinstance(calc, Calculator)
    assert set(properties).issubset(calc.implemented_properties)
    calc.calculate(atoms=atoms, properties=properties)
    for prop in properties:
        assert prop in calc.results
    if {"energy", "forces"}.issubset(properties):
        check_energy_force_results(calc.results, len(atoms))


def check_scratch_directory(path, moleculeid):
    assert (Path(path) / moleculeid).is_dir()
