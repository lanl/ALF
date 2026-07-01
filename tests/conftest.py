import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from alframework.tools.molecules_class import MoleculesObject


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


class FixedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy, forces):
        super().__init__()
        self.energy = energy
        self.forces = np.array(forces, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self.energy
        self.results["forces"] = self.forces.copy()


@pytest.fixture
def fixed_calculator_factory():
    return FixedCalculator
