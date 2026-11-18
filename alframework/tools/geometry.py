"""Geometry checks shared by the legacy and batched MLMD samplers."""

import numpy as np

# The closest pair is found first with a neighbor search within this radius (Angstrom), which costs
# O(N) instead of the O(N^2) all-pairs minimum-image matrix. The search returns every periodic image of
# every pair closer than the radius, so if it finds any pair, its closest is the exact minimum-image
# minimum. Only structures with no pair this close (dilute gases) fall back to the full matrix.
SEARCH_RADIUS = 3.0
# Molecules (no periodic direction) up to this many atoms use the all-pairs matrix directly: without a
# cell, a neighbor search costs ~10x more than the matrix at molecular sizes (0.8 vs 7.9 ms at 120 atoms).
ALL_PAIRS_LIMIT = 1000


def minimum_distance(atoms):
    """Minimum image distance between distinct atoms, or infinity without pairs.

    Only diagonal self-pairs are excluded: coincident distinct atoms, including
    periodic equivalents, have distance zero. An atom's own periodic images
    are outside this interatomic screen.
    """
    if len(atoms) < 2:
        return float('inf')
    if not np.any(atoms.pbc) and len(atoms) <= ALL_PAIRS_LIMIT:
        return _all_pairs_minimum(atoms)
    from ase.neighborlist import neighbor_list
    first, second, distances = neighbor_list('ijd', atoms, SEARCH_RADIUS, self_interaction=False)
    distinct = first != second
    if distinct.any():
        return float(distances[distinct].min())
    return _all_pairs_minimum(atoms)


def _all_pairs_minimum(atoms):
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def _batched_neighbor_search():
    """A batched neighbor search over a packed batch, on the batch's device: ALCHEMI's when installed,
    else TorchSim's pure-PyTorch cell list (which runs on CUDA, ROCm and CPU). ``None`` without TorchSim."""
    try:
        from torch_sim.neighbors.alchemiops import ALCHEMIOPS_AVAILABLE
        if ALCHEMIOPS_AVAILABLE:
            from torch_sim.neighbors.alchemiops import alchemiops_nl_cell_list
            return alchemiops_nl_cell_list
        from torch_sim.neighbors.torch_nl import torch_nl_linked_cell
        return torch_nl_linked_cell
    except ImportError:
        return None


def batched_minimum_distances(positions, cells, system_idx, n_systems):
    """Each fully periodic system's minimum interatomic distance, for a whole packed batch at once.

    ``positions`` [atoms, 3] and ``cells`` [systems, 3, 3] (ASE's row-vector convention) are torch tensors on
    the batch's device; ``system_idx`` [atoms] gives each atom's system. Same quantity as
    ``minimum_distance``, found with one batched neighbor search within ``SEARCH_RADIUS``: a system with
    no pair that close gets infinity, and callers fall back to ``minimum_distance`` for it. Returns
    ``None`` when no batched search is available. Values agree with ``minimum_distance`` to float64
    rounding (the order of operations differs).
    """
    import torch
    search = _batched_neighbor_search()
    if search is None:
        return None
    positions, cells = positions.to(torch.float64), cells.to(torch.float64)
    pbc = torch.ones((n_systems, 3), dtype=torch.bool, device=positions.device)
    mapping, system_mapping, shifts = search(positions, cells, pbc, SEARCH_RADIUS, system_idx)
    first, second = mapping
    vectors = positions[second] - positions[first] + torch.einsum(
        'pi,pij->pj', shifts.to(positions.dtype), cells[system_mapping])
    distinct = first != second  # an atom's own periodic images are outside this screen
    distances = torch.full((n_systems,), float('inf'), dtype=positions.dtype, device=positions.device)
    return distances.scatter_reduce_(0, system_mapping[distinct], vectors[distinct].norm(dim=1), reduce='amin')
