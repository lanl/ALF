"""ALF's geometry operations (alframework.tools.geometry)."""
import numpy as np
import pytest


def _padded_molecules(sizes, atoms, seed=0):
    rng = np.random.default_rng(seed)
    positions = np.zeros((len(sizes), atoms, 3))
    mask = np.zeros((len(sizes), atoms), dtype=bool)
    for frame, size in enumerate(sizes):
        positions[frame, :size] = rng.uniform(0, 4.0, size=(size, 3))
        positions[frame, size:] = 0.01 * frame  # padding sits close to real atoms, and must be ignored
        mask[frame, :size] = True
    return positions, mask


def test_padded_minimum_distances_match_each_molecule_alone():
    from ase import Atoms
    from alframework.tools.geometry import minimum_distance, padded_minimum_distances
    sizes = [2, 5, 9, 1, 12, 7]
    positions, mask = _padded_molecules(sizes, 12)
    found = padded_minimum_distances(positions, mask, pair_budget=200).numpy()  # several chunks
    for frame, size in enumerate(sizes):
        expected = minimum_distance(Atoms(positions=positions[frame, :size])) if size > 1 else np.inf
        assert found[frame] == pytest.approx(expected, rel=0, abs=1e-12)


def test_padded_minimum_distances_are_exact_for_close_pairs_far_from_the_origin():
    from alframework.tools.geometry import padded_minimum_distances
    positions = np.full((1, 40, 3), 1.0e4) + np.arange(40)[None, :, None] * np.array([3.0, 0, 0])
    positions[0, 7] = positions[0, 6] + np.array([1e-4, 0, 0])
    found = padded_minimum_distances(positions, np.ones((1, 40), dtype=bool))
    assert float(found[0]) == pytest.approx(1e-4, rel=1e-9)
