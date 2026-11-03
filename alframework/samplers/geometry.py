"""Geometry checks shared by the legacy and batched MLMD samplers."""

import numpy as np


def minimum_distance(atoms):
    """Minimum image distance between distinct atoms, or infinity without pairs.

    Only diagonal self-pairs are excluded: coincident distinct atoms, including
    periodic equivalents, have distance zero. An atom's own periodic images
    are outside this interatomic screen.
    """
    if len(atoms) < 2:
        return float('inf')
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())
