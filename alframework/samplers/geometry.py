"""Geometry checks shared by the legacy and batched MLMD samplers."""

import numpy as np

# The closest pair is found first with a neighbor search within this radius (Angstrom), which costs
# O(N) instead of the O(N^2) all-pairs minimum-image matrix. The search returns every periodic image of
# every pair closer than the radius, so if it finds any pair, its closest is the exact minimum-image
# minimum. Only structures with no pair this close (dilute gases) fall back to the full matrix.
SEARCH_RADIUS = 3.0


def minimum_distance(atoms):
    """Minimum image distance between distinct atoms, or infinity without pairs.

    Only diagonal self-pairs are excluded: coincident distinct atoms, including
    periodic equivalents, have distance zero. An atom's own periodic images
    are outside this interatomic screen.
    """
    if len(atoms) < 2:
        return float('inf')
    from ase.neighborlist import neighbor_list
    first, second, distances = neighbor_list('ijd', atoms, SEARCH_RADIUS, self_interaction=False)
    distinct = first != second
    if distinct.any():
        return float(distances[distinct].min())
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())
