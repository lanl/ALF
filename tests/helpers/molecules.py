"""Shared test geometries. Always ``.copy()`` these before changing or attaching anything."""

from ase import Atoms

# Close to a real water molecule, for tests that care about ordinary bond lengths.
WATER = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])

# Equilateral, with every pair at the rest length of ``fakes.PairSpringCalculator``: ensemble
# members start in perfect agreement, and their uncertainty grows as the molecule heats up.
SPRING_WATER = Atoms("OH2", positions=[[0, 0, 0], [1.2, 0, 0], [0.6, 1.0392, 0]])
