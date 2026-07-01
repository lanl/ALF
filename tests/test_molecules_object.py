import numpy as np
import pytest
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject, compare_chemical_composition


def water():
    return Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]])


def test_molecules_object_state_and_legacy_indexing():
    atoms = water()
    molecule = MoleculesObject(atoms, "water-0")

    assert len(molecule) == 3
    assert repr(molecule) == "water-0"
    assert molecule.get_moleculeid() == "water-0"
    assert molecule.get_atoms() is atoms
    assert not molecule.check_stored_results()
    assert molecule.check_convergence() is None

    molecule.store_results({"energy": -76.0, "forces": np.ones((3, 3))})
    molecule.store_results({"energy": -75.5, "dipole": np.zeros(3)}, replace=False)
    molecule.update_metadata({"source": "unit-test"})
    molecule.update_metadata({"source": "ignored", "sample": 1}, replace=False)
    molecule.set_converged_flag(True)

    assert molecule.get_results()["energy"] == -76.0
    assert "dipole" in molecule.get_results()
    assert molecule.get_metadata() == {"source": "unit-test", "sample": 1}
    assert molecule.check_stored_results()
    assert molecule.check_convergence() is True

    with pytest.warns(DeprecationWarning):
        assert molecule[0] == "water-0"
    with pytest.raises(IndexError):
        molecule[3]


def test_update_and_append_atoms():
    molecule = MoleculesObject(Atoms("H", positions=[[0.0, 0.0, 0.0]]), "h")

    molecule.append_atoms(Atoms("O", positions=[[0.0, 0.0, 1.0]]))
    assert molecule.get_atoms().get_chemical_formula() == "HO"

    molecule.update_atoms(None)
    assert molecule.get_atoms() is None


def test_equality_uses_other_coordinates_regression():
    molecule_a = MoleculesObject(water(), "a")
    translated = water()
    translated.positions[1, 1] += 0.2
    molecule_b = MoleculesObject(translated, "b")

    assert molecule_a != molecule_b
    assert molecule_a == MoleculesObject(water(), "c")


def test_composition_and_signature_are_order_stable():
    molecule_a = MoleculesObject(water(), "a")
    molecule_b = MoleculesObject(Atoms("H2O", positions=[[3, 0, 0], [4, 0, 0], [5, 0, 0]]), "b")

    assert compare_chemical_composition(molecule_a, molecule_b)
    assert "O0.0000000.0000000.000000" in molecule_a.get_system_signature()

    with pytest.raises(AssertionError):
        compare_chemical_composition(molecule_a, object())
