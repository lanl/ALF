"""Structures to benchmark on: a water cluster of any size, a bundled CPP reference, or any file."""
from pathlib import Path

import numpy as np

CPP_REFERENCES = Path(__file__).resolve().parents[2] / 'examples' / 'CPP_excited_gpu4pyscf' / 'references'


def water_cluster(n_atoms):
    """About ``n_atoms`` atoms of water molecules on a 3.1 Angstrom grid, so neighbors don't clash."""
    from ase import Atoms
    from ase.build import molecule
    count = max(1, round(n_atoms / 3))
    side = int(np.ceil(count ** (1 / 3)))
    water = molecule('H2O')
    atoms = Atoms()
    for index in range(count):
        copy = water.copy()
        copy.translate(3.1 * np.array(np.unravel_index(index, (side, side, side)), dtype=float))
        atoms += copy
    return atoms


def load_system(name, n_atoms=96):
    """``water``, ``cppN`` (from a clone), or a structure file ASE can read."""
    from ase.io import read
    if name == 'water':
        return water_cluster(n_atoms)
    if name.startswith('cpp') and name[3:].isdigit():
        path = CPP_REFERENCES / f'{name}_s0.xyz'
        if not path.exists():
            raise FileNotFoundError(f'{name} needs an ALF clone with {path}; use --system water or a file.')
        return read(path)
    path = Path(name).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'No structure {name!r}: use water, cppN (6-12), or a structure file.')
    return read(path)
