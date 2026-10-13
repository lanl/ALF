import numpy as np
import pytest
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.fakes import FixedCalculator


@pytest.fixture
def h2_atoms():
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


@pytest.fixture
def water_atoms():
    return Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]])


@pytest.fixture
def periodic_water_atoms(water_atoms):
    atoms = water_atoms.copy()
    atoms.set_cell(np.eye(3) * 8.0)
    atoms.set_pbc(True)
    return atoms


@pytest.fixture
def water_molecule(periodic_water_atoms):
    return MoleculesObject(periodic_water_atoms, "water-0")


@pytest.fixture
def fixed_calculator_factory():
    return FixedCalculator
