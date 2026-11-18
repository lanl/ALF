"""Allocate seed molecule IDs and validate spare counts during bootstrap."""


def bootstrap_spare_count(master_config):
    """Return the validated ``bootstrap_spares``: extra seeds a finite bootstrap labels in place of failed ones."""
    spares = master_config.get('bootstrap_spares', 0)
    if isinstance(spares, bool) or not isinstance(spares, int) or spares < 0:
        raise ValueError(f'bootstrap_spares must be a non-negative integer; received {spares!r}.')
    if spares and not master_config.get('finite_bootstrap', False):
        raise ValueError('bootstrap_spares needs finite_bootstrap: an ordinary bootstrap already keeps building '
                         'until it has enough labels.')
    return spares


def bootstrap_molecule_ids(start, total, width, finite=False):
    """Bound an opt-in finite source without changing ordinary bootstrap admission."""
    return next_bootstrap_ids(start, total, width, finite)[0]


def next_bootstrap_ids(start, total, width, finite=False, skip=frozenset()):
    """Return up to ``width`` bootstrap IDs from ``start``, and the counter to continue from.

    IDs in ``skip`` are passed over, so a restarted finite bootstrap rebuilds only what it doesn't hold.
    """
    if finite and (isinstance(total, bool) or not isinstance(total, int) or total < 1):
        raise ValueError('finite_bootstrap requires a positive integer bootstrap_set')
    ids, index = [], start
    while len(ids) < width and not (finite and index >= total):
        moleculeid = f'mol-boot-{index:010d}'
        if moleculeid not in skip:
            ids.append(moleculeid)
        index += 1
    return ids, index
