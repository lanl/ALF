"""Gap-stratified splits keep the near-degenerate share of every set equal."""

import numpy as np
import pytest

from alframework.tools.splits import frame_gaps, gap_stratified_split, split_settings


def near_degenerate_fraction(gaps, positions, cutoff):
    return float(np.mean(np.abs(gaps[positions]) < cutoff))


def test_every_set_holds_the_same_share_of_small_gaps():
    rng = np.random.default_rng(1)
    gaps = np.concatenate([rng.uniform(0, 0.01, 1480), rng.uniform(0.05, 3.0, 8520)])  # 14.8% near-degenerate
    train, valid, test = gap_stratified_split(gaps, test_size=0.1, valid_size=0.1, bins=20, seed=42)
    assert len(np.unique(np.concatenate([train, valid, test]))) == len(gaps)
    assert (len(train), len(valid), len(test)) == (8000, 1000, 1000)
    fractions = [near_degenerate_fraction(gaps, part, 0.01) for part in (train, valid, test)]
    assert max(fractions) - min(fractions) < 0.005
    # The seed changes membership but not the balance.
    other = gap_stratified_split(gaps, test_size=0.1, valid_size=0.1, bins=20, seed=7)
    assert not np.array_equal(other[2], test)
    assert gap_stratified_split(gaps, test_size=0.1, valid_size=0.1, bins=20, seed=42)[2].tolist() == test.tolist()


def test_frame_gap_is_the_smallest_listed_pair_gap():
    energies = {0: np.array([0.0, 0.0]), 1: np.array([3.0, 2.0]), 2: np.array([3.1, 4.0])}
    np.testing.assert_allclose(frame_gaps(energies, [(0, 1), (1, 2)]), [0.1, 2.0])
    np.testing.assert_allclose(frame_gaps(energies, [(1, 2)]), [0.1, 2.0])


@pytest.mark.parametrize("gaps, sizes", [
    (np.arange(64), (52, 6, 6)),
    (np.ones(64), (52, 6, 6)),
    (np.arange(3), (1, 1, 1)),
])
def test_small_pools_keep_disjoint_splits_with_held_out_frames(gaps, sizes):
    parts = gap_stratified_split(gaps, test_size=0.1, valid_size=0.1)
    assert tuple(map(len, parts)) == sizes
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(gaps)))
    if len(gaps) == 64 and gaps[0] != gaps[-1]:
        for part in parts[1:]:
            assert gaps[part].min() < 16 and gaps[part].max() >= 48


def test_split_settings_default_and_refusals():
    assert split_settings(None, 4) == {"method": "random", "bins": 20, "pairs": [(0, 1), (1, 2), (2, 3)]}
    assert split_settings({"method": "gap_stratified", "pairs": [[1, 2]], "bins": 5}, 4)["pairs"] == [(1, 2)]
    for options, match in (({"method": "kmeans"}, "method"), ({"bins": 0}, "bins"), ({"seed": 1}, "not"),
                           ({"method": "gap_stratified", "pairs": [[1, 7]]}, "pairs")):
        with pytest.raises(ValueError, match=match):
            split_settings(options, 4)
    with pytest.raises(ValueError, match="sum"):
        gap_stratified_split(np.ones(10), test_size=0.5, valid_size=0.5)
    with pytest.raises(ValueError, match="three frames"):
        gap_stratified_split(np.ones(2), test_size=0.1, valid_size=0.1)
