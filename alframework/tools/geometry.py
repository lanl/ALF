"""Geometry operations shared by builders, samplers, and training."""

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


# The neighbor searches ALF's batched code can use, by the name a configuration gives (such as a batched
# evaluator's neighbor_backend). Every one finds the same pairs: both directions, and an atom's own periodic images.
NEIGHBOR_BACKENDS = ('auto', 'alchemiops', 'torch_linked_cell', 'torch_n2', 'vesin', 'ase')


def _system_neighbors(points, box, cutoff, backend):
    """(i, j, S) for all pairs within ``cutoff`` of one fully periodic system, with ASE or vesin on the CPU."""
    if backend == 'vesin':
        import vesin
        i, j, shifts = vesin.NeighborList(cutoff=cutoff, full_list=True).compute(
            points=points, box=box, periodic=True, quantities='ijS')
    else:
        from ase.neighborlist import primitive_neighbor_list
        i, j, shifts = primitive_neighbor_list('ijS', [True] * 3, box, points, cutoff, self_interaction=False)
    return np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64), np.asarray(shifts, dtype=np.float64)


def _per_system_search(backend):
    """A search with TorchSim's batched interface that runs ASE or vesin one system at a time on the CPU, for
    fully periodic systems. It returns each system's pairs in turn, in that library's order."""
    def search(positions, cells, pbc, cutoff, system_idx, self_interaction=False):
        import torch
        points, boxes = positions.detach().cpu().double().numpy(), cells.detach().cpu().double().numpy()
        counts = torch.bincount(system_idx, minlength=len(boxes)).tolist()
        firsts, seconds, systems, shifts, offset = [], [], [], [], 0
        for system, count in enumerate(counts):
            i, j, cell_shifts = _system_neighbors(points[offset:offset + count], boxes[system], cutoff, backend)
            firsts.append(i + offset)
            seconds.append(j + offset)
            systems.append(np.full(len(i), system, dtype=np.int64))
            shifts.append(cell_shifts)
            offset += count
        device = positions.device
        mapping = torch.as_tensor(np.stack([np.concatenate(firsts), np.concatenate(seconds)]), device=device)
        return (mapping, torch.as_tensor(np.concatenate(systems), device=device),
                torch.as_tensor(np.concatenate(shifts), device=device))
    return search


def neighbor_search(backend='auto'):
    """A batched neighbor search, by name (:data:`NEIGHBOR_BACKENDS`), with TorchSim's interface.

    ``search(positions, cells, pbc, cutoff, system_idx)`` takes a packed batch, ``positions`` [atoms, 3], ``cells``
    [systems, 3, 3] in ASE's row-vector convention, ``pbc`` [systems, 3] and each atom's system, and returns
    ``(mapping, system_mapping, shifts)``: the pairs [2, pairs], each pair's system, and its periodic shift in
    cell vectors, so a pair's vector is ``positions[second] - positions[first] + shift @ cell``.

    - ``auto``: on the batch's device, ALCHEMI's cell list when installed (NVIDIA), else TorchSim's PyTorch cell
      list, which runs on CUDA, ROCm and CPU. ``None`` without TorchSim, so a caller can fall back.
    - ``alchemiops``, ``torch_linked_cell``, ``torch_n2`` (all pairs within each system): TorchSim's, on the device.
    - ``vesin``, ``ase``: one system at a time on the CPU, for fully periodic systems, without TorchSim.
    """
    if backend not in NEIGHBOR_BACKENDS:
        raise ValueError(f'neighbor_backend must be one of {list(NEIGHBOR_BACKENDS)}; received {backend!r}.')
    if backend == 'vesin':
        try:
            import vesin  # noqa: F401
        except ImportError:
            raise ValueError("neighbor_backend 'vesin' needs vesin; install ALF's vesin extra.") from None
    if backend in ('vesin', 'ase'):
        return _per_system_search(backend)
    try:
        from torch_sim.neighbors.alchemiops import ALCHEMIOPS_AVAILABLE
        from torch_sim.neighbors import alchemiops_nl_cell_list, torch_nl_linked_cell, torch_nl_n2
    except ImportError:
        if backend == 'auto':
            return None
        raise ValueError(f'neighbor_backend {backend!r} needs TorchSim (torch-sim-atomistic).') from None
    if backend == 'alchemiops' or (backend == 'auto' and ALCHEMIOPS_AVAILABLE):
        if not ALCHEMIOPS_AVAILABLE:
            raise ValueError("neighbor_backend 'alchemiops' needs nvalchemi-toolkit-ops, which runs on NVIDIA GPUs.")
        return alchemiops_nl_cell_list
    return torch_nl_n2 if backend == 'torch_n2' else torch_nl_linked_cell


def batched_minimum_distances(positions, cells, system_idx, n_systems, backend='auto'):
    """Each fully periodic system's minimum interatomic distance, for a whole packed batch at once.

    ``positions`` [atoms, 3] and ``cells`` [systems, 3, 3] (ASE's row-vector convention) are torch tensors on
    the batch's device; ``system_idx`` [atoms] gives each atom's system. Same quantity as
    ``minimum_distance``, found with one batched neighbor search within ``SEARCH_RADIUS``: a system with
    no pair that close gets infinity, and callers fall back to ``minimum_distance`` for it. Returns
    ``None`` when no batched search is available. Values agree with ``minimum_distance`` to float64
    rounding (the order of operations differs). ``backend`` names the search (:func:`neighbor_search`).
    """
    import torch
    search = neighbor_search(backend)
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


# Pair distances one chunk of padded frames holds at once in padded_minimum_distances (frames x atoms x atoms).
PAIR_BUDGET = 1 << 24


def padded_minimum_distances(positions, mask, device=None, pair_budget=PAIR_BUDGET):
    """Each frame's minimum interatomic distance, without periodic images, for a padded batch.

    ``positions`` [frames, atoms, 3] and ``mask`` [frames, atoms] (true for real atoms; padding has species 0)
    are NumPy arrays or torch tensors. In chunks of about ``pair_budget`` pairs on ``device`` (the positions' own
    by default), in float64, cdist's fast matrix-product form finds each frame's closest pair of real atoms, and
    that pair's distance is then computed directly, so the value is exact even far from the origin (the fast form
    alone loses precision there). Only two pairs within about 1e-8 Å of each other can be told apart wrongly,
    which no screen notices. A frame with fewer than two real atoms gets infinity. Frames of more than
    ``ALL_PAIRS_LIMIT`` atoms use ``minimum_distance``. Returns a float64 CPU tensor.
    """
    import torch
    positions = torch.as_tensor(positions)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    target = torch.device(device) if device is not None else positions.device
    frames, atoms = mask.shape
    found = torch.full((frames,), float('inf'), dtype=torch.float64)
    if atoms < 2 or frames == 0:
        return found
    if atoms > ALL_PAIRS_LIMIT:
        from ase import Atoms
        for frame in range(frames):
            real = mask[frame]
            found[frame] = minimum_distance(Atoms(positions=positions[frame][real].double().cpu().numpy()))
        return found
    step = max(1, pair_budget // (atoms * atoms))
    distinct = ~torch.eye(atoms, dtype=torch.bool, device=target)
    for start in range(0, frames, step):
        chunk = slice(start, start + step)
        points = positions[chunk].to(target, torch.float64)
        real = mask[chunk].to(target)
        pairs = real[:, :, None] & real[:, None, :] & distinct
        rough = torch.cdist(points, points).masked_fill(~pairs, float('inf')).flatten(1)
        closest = rough.argmin(dim=1)
        rows = torch.arange(len(points), device=target)
        exact = (points[rows, closest // atoms] - points[rows, closest % atoms]).norm(dim=1)
        found[chunk] = torch.where(torch.isfinite(rough[rows, closest]), exact, float('inf')).cpu()
    return found


def random_rotation_matrix(deflection=1.0, randnums=None):
    """Returns a random rotation matrix

    Args:
        deflection (float): Pole deflection.
        randnums (ndarray): A (3,) numpy array containing the parameters theta, phi, and z.

    Returns:
        (ndarray): Random 3x3 rotation matrix.

    """
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    r = np.sqrt(z)

    V = [np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)]

    st = np.sin(theta)
    ct = np.cos(theta)

    R_z = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

    M = (np.outer(V, V) - np.eye(3)) @ R_z

    return M
