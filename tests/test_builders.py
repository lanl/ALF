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


def test_create_atomic_system_is_charge_neutral():
    random.seed(1)

    system = create_atomic_system({"Na": 1, "Cl": -1}, target_num_atoms=8)

    assert sum({"Na": 1, "Cl": -1}[atom] * count for atom, count in system.items()) == 0
    assert sum(system.values()) >= 7


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
    for first, second in itertools.combinations(coords, 2):
        assert np.linalg.norm(to_mic(first - second, 8.0)) > 1.0


def test_construct_simulation_box_rejects_too_dense_system():
    with pytest.raises(ValueError):
        construct_simulation_box({"Na": 10}, min_distance=2.0, box_length=3.0)


def test_atomic_system_builder_returns_molecules_ready_atoms():
    random.seed(2)
    np.random.seed(2)

    atoms = atomic_system_builder({"Na": 1, "Cl": -1}, target_num_atoms=4, min_distance=0.8, box_length=8.0)
    molecule = MoleculesObject(atoms, "salt")

    assert atoms.pbc.all()
    assert atoms.cell.lengths().tolist() == [8.0, 8.0, 8.0]
    assert len(molecule) >= 3
