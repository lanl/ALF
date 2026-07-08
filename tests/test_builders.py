import itertools
import random

import numpy as np
import pytest

from alframework.builders.builders import (
    atomic_system_builder,
    construct_simulation_box,
    create_atomic_system,
    to_mic,
)
from alframework.tools.molecules_class import MoleculesObject


@pytest.mark.parametrize(
    "atom_charges,target_num_atoms",
    [
        ({"Na": 1, "Cl": -1}, 8),
        ({"Mg": 2, "Cl": -1}, 9),
        ({"Ca": 2, "F": -1, "O": -2}, 10),
    ],
)
def test_create_atomic_system_is_charge_neutral(atom_charges, target_num_atoms):
    random.seed(1)

    system = create_atomic_system(atom_charges, target_num_atoms)

    total_charge = sum(atom_charges[atom] * count for atom, count in system.items())
    assert total_charge == 0
    assert sum(system.values()) >= target_num_atoms - 1
    assert set(system).issubset(atom_charges)


def test_to_mic_wraps_distances_into_minimum_image():
    distances = np.array([[5.5, -5.5, 0.0], [1.0, 2.0, -6.0]])

    np.testing.assert_allclose(to_mic(distances, box_length=10.0), [[-4.5, 4.5, 0.0], [1.0, 2.0, 4.0]])


def test_construct_simulation_box_respects_minimum_distance():
    np.random.seed(5)
    coords = construct_simulation_box(
        {"Na": 2, "Cl": 2},
        min_distance=1.0,
        box_length=8.0,
        scale_coords=False,
        max_iter=20,
        max_tries=5,
    )

    assert coords.shape == (4, 3)
    assert np.all(coords >= 0.0)
    assert np.all(coords < 8.0)
    for first, second in itertools.combinations(coords, 2):
        assert np.linalg.norm(to_mic(first - second, 8.0)) > 1.0


def test_construct_simulation_box_can_return_scaled_coordinates():
    np.random.seed(5)
    coords = construct_simulation_box(
        {"Na": 1, "Cl": 1},
        min_distance=1.0,
        box_length=8.0,
        scale_coords=True,
    )

    assert coords.shape == (2, 3)
    assert np.all(coords >= 0.0)
    assert np.all(coords <= 1.0)


def test_construct_simulation_box_rejects_too_dense_system():
    with pytest.raises(ValueError):
        construct_simulation_box({"Na": 10}, min_distance=2.0, box_length=3.0)


def test_atomic_system_builder_returns_molecule_ready_atoms():
    random.seed(2)
    np.random.seed(2)

    atom_charges = {"Na": 1, "Cl": -1}
    target_num_atoms = 4
    atoms = atomic_system_builder(atom_charges, target_num_atoms=target_num_atoms, min_distance=0.8, box_length=8.0)
    molecule = MoleculesObject(atoms, "salt")

    assert atoms.pbc.all()
    np.testing.assert_allclose(atoms.cell.lengths(), [8.0, 8.0, 8.0])
    assert len(atoms) >= target_num_atoms - 1
    assert len(molecule) == len(atoms)
    assert set(atoms.get_chemical_symbols()).issubset(atom_charges)
    assert sum(atom_charges[symbol] for symbol in atoms.get_chemical_symbols()) == 0
