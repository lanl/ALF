"""Train/validation/test splits that keep rare near-degenerate frames evenly represented.

A random split can leave the validation or test set short of the few frames where two states nearly
cross, which are the frames excited-state dynamics depends on most. A gap-stratified split ranks the frames
by energy gap and distributes each held-out set across bins of similar gaps. The seed breaks ties and
chooses which frames in each bin go into each split.
"""

import numpy as np

SPLIT_METHODS = ("random", "gap_stratified")


def gap_stratified_split(gaps, *, test_size, valid_size, bins=20, seed=0):
    """Return position arrays ``(train, valid, test)`` balanced on ``|gaps|``.

    Round the requested fractions on the full dataset and reserve at least one frame for each set.
    Allocate test frames across the gap bins first, then allocate validation frames across the remaining
    frames. Cumulative rounding keeps the totals exact even when a bin is too small to hold all three sets.
    """
    gaps = np.abs(np.asarray(gaps, dtype=float)).reshape(-1)
    if not np.isfinite(gaps).all():
        raise ValueError("Gap-stratified splitting needs finite gaps.")
    if not (0 < test_size < 1 and 0 < valid_size < 1 and test_size + valid_size < 1):
        raise ValueError("test_size and valid_size must be positive fractions that sum to less than 1.")
    count = len(gaps)
    if count < 3:
        raise ValueError("Gap-stratified splitting needs at least three frames.")
    n_test = min(count - 2, max(1, round(test_size * count)))
    n_valid = min(count - n_test - 1, max(1, round(valid_size * count)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    order = order[np.argsort(gaps[order], kind="stable")]
    members_by_bin = np.array_split(order, max(1, min(int(bins), count)))
    sizes = np.array([len(members) for members in members_by_bin])
    test_counts = _distribute(n_test, sizes)
    valid_counts = _distribute(n_valid, sizes - test_counts)
    train, valid, test = [], [], []
    for members, test_count, valid_count in zip(members_by_bin, test_counts, valid_counts):
        rng.shuffle(members)
        valid_end = test_count + valid_count
        test.append(members[:test_count])
        valid.append(members[test_count:valid_end])
        train.append(members[valid_end:])
    return tuple(np.sort(np.concatenate(part)) for part in (train, valid, test))


def _distribute(total, sizes):
    cumulative = np.rint(total * sizes.cumsum() / sizes.sum()).astype(int)
    return np.diff(cumulative, prepend=0)


def frame_gaps(energies, pairs):
    """The smallest ``|E_j − E_i|`` over ``pairs`` for every frame. ``energies`` maps a state to ``[frames]``."""
    if not pairs:
        raise ValueError("Give at least one state pair to stratify on.")
    return np.min(np.stack([np.abs(np.asarray(energies[j], dtype=float) - np.asarray(energies[i], dtype=float))
                            for i, j in pairs]), axis=0)


def split_settings(options, n_states):
    """Validate a ``split`` block: ``{"method": "random" | "gap_stratified", "bins": 20, "pairs": [[1, 2]]}``.

    ``pairs`` defaults to every adjacent pair of states, and the frame's gap is the smallest of them.
    """
    options = dict(options or {})
    unknown = set(options) - {"method", "bins", "pairs"}
    if unknown:
        raise ValueError(f"split takes method, bins, and pairs, not {sorted(unknown)}.")
    method = options.get("method", "random")
    if method not in SPLIT_METHODS:
        raise ValueError(f"split.method must be one of {SPLIT_METHODS}, not {method!r}.")
    bins = options.get("bins", 20)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("split.bins must be a positive integer.")
    pairs = options.get("pairs")
    if pairs is None:
        pairs = [[s, s + 1] for s in range(n_states - 1)]
    pairs = [tuple(int(state) for state in pair) for pair in pairs]
    if method == "gap_stratified":
        if not pairs:
            raise ValueError("A gap-stratified split needs at least two states.")
        for i, j in pairs:
            if not (0 <= i < n_states and 0 <= j < n_states and i != j):
                raise ValueError(f"split.pairs entry {[i, j]} must name two different states below {n_states}.")
    return {"method": method, "bins": bins, "pairs": pairs}
