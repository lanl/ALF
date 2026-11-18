"""Small labeled structures for label validation, storage and ASE labeling tests."""

import numpy as np
from ase import Atoms

from alframework.tools.molecules import MoleculesObject

# ASE's units, stored as they are.
PROPERTIES = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}


def labeled_h2(name, pbc=False, converged=True):
    """An H2 label in an 8 Angstrom cell; ``pbc`` says which directions are periodic."""
    molecule = MoleculesObject(Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[8] * 3, pbc=pbc), name)
    molecule.store_results({"energy": -1.0, "forces": np.zeros((2, 3))})
    molecule.set_converged_flag(converged)
    return molecule


def unlabeled_h2(name="h2"):
    """A nonperiodic H2 near its bond length, waiting to be labeled."""
    return MoleculesObject(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]), name)
