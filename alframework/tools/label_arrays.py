"""Assemble the groups of ALF label shards into whole, padded arrays: one row per frame, atoms padded to the most
atoms of any frame. Trainers hand these arrays to their own dataset classes; nothing here imports one.

Which axis of a column holds atoms is known, not guessed from lengths: ``species``, the coordinates, topology atom
IDs and the caller's atom keys hold atoms on axis 1, and so do the shard's ``"atomic"`` properties; its
``"pair_atomic"`` properties, ``[frames, pairs, atoms, 3]``, hold them on axis 2. Every other column keeps its
shape, which must agree across groups. A column stored once per group, such as ``species`` grouped by formula, is
repeated for each of the group's frames.
"""
import json

import numpy as np

from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS
from alframework.tools.label_hdf5 import PROPERTY_SCHEMA

# Atom axes of property kinds in a shard's property schema (properties_list), as stored.
PROPERTY_ATOM_AXES = {'atomic': 1, 'pair_atomic': 2}
# Columns a group may lack, and the value a frame without them takes (see data_shards.STRUCTURE_STATE_DATASETS).
STATE_DEFAULTS = {'total_charge': 0, 'spin_multiplicity': 1}
# Names a species column goes by, matched without regard to case, when a trainer doesn't name it.
SPECIES_KEYS = ('species', 'atomic_numbers', 'z', 'atom_types', 'atomic_number')


def detect_species_key(keys):
    """The one key among ``keys`` that names species (``SPECIES_KEYS``, any case)."""
    keys = list(keys)
    matches = [key for key in keys if key.casefold() in SPECIES_KEYS]
    if len(matches) != 1:
        found = 'several species datasets, ' + ', '.join(matches) if matches else 'no species dataset'
        raise ValueError(f'Found {found} among {keys}; set species_key to name it.')
    return matches[0]


def print_arrays(arrays):
    """A table of each array's name, dtype and shape, for training logs. Takes NumPy arrays or tensors."""
    row = '| {:<30} | {:<18} | {:<28} |'
    line = '-' * len(row.format('', '', ''))
    print(line, row.format('Name', 'dtype', 'shape'), line, sep='\n')
    for key, value in arrays.items():
        value = value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        print(row.format(key, repr(value.dtype), repr(value.shape)))
    print(line)


def shard_atom_axes(files):
    """The atom axis of each property dataset named in the shards' property schemas (``alf_label_properties``)."""
    import h5py
    axes = {}
    for file in files:
        with h5py.File(file, 'r') as handle:
            recorded = handle.attrs.get(PROPERTY_SCHEMA)
        for spec in (json.loads(recorded) if recorded else {}).values():
            if len(spec) >= 2 and str(spec[1]).lower() in PROPERTY_ATOM_AXES:
                axis = PROPERTY_ATOM_AXES[str(spec[1]).lower()]
                if axes.setdefault(spec[0], axis) != axis:
                    raise ValueError(f'{file}: {spec[0]!r} is stored as {spec[1]!r} here but with atoms on axis '
                                     f'{axes[spec[0]]} in another shard.')
    return axes


def atom_axes(files, species_key, coordinates_key, atom_keys=()):
    """Every column's atom axis: the shards' schema (:func:`shard_atom_axes`), then the structural columns and
    ``atom_keys``, which hold atoms on axis 1."""
    axes = shard_atom_axes(files)
    for key in (species_key, coordinates_key, TOPOLOGY_ATOM_IDS, *atom_keys):
        if key is not None:
            axes[key] = 1
    return axes


def _frames(batch, count_key):
    return len(batch[count_key])


def _column_dtype(key, values):
    """One dtype for a column's groups: the widest text, or the numeric type all of them cast to safely."""
    kinds = {value.dtype.kind for value in values}
    if len(kinds) == 1 and kinds <= {'S', 'U'}:
        kind = kinds.pop()
        width = max(value.dtype.itemsize // (4 if kind == 'U' else 1) for value in values)
        return np.dtype(f'{kind}{max(width, 1)}')
    if kinds & {'S', 'U', 'O'}:
        raise ValueError(f'{key!r} mixes text and numbers, or text kinds, across label groups.')
    return np.result_type(*values)


def assemble_label_arrays(batches, axes, species_key, count_key, defaults=STATE_DEFAULTS):
    """Whole arrays from ``batches`` (one dict of a group's columns each, in order).

    ``axes`` maps a column to its atom axis (:func:`atom_axes`); ``count_key`` names a column with one row per
    frame. A column some groups lack is filled with its ``defaults`` value for their frames, and is an error
    without one. Adds ``sys_number``, each frame's group index, as HIPPYNN's readers do.
    """
    if not batches:
        raise ValueError('No structure groups in the label shards.')
    counts = [_frames(batch, count_key) for batch in batches]
    total = sum(counts)
    n_atoms_max = max(np.shape(batch[species_key])[-1] for batch in batches)
    keys = list(dict.fromkeys(key for batch in batches for key in batch))
    arrays = {}
    for key in keys:
        present = [(batch[key], count) for batch, count in zip(batches, counts) if key in batch]
        if len(present) != len(batches) and key not in defaults:
            raise ValueError(f'{key!r} is stored in some label groups but not in others.')
        values = [np.asarray(value) for value, _ in present]
        axis = axes.get(key)
        trailing = set()
        for value, count in zip(values, (count for _, count in present)):
            if value.ndim == 0:
                raise ValueError(f'{key!r} is a single value; label columns have a row per frame.')
            shape = list(value.shape[1:])
            if value.shape[0] not in (count, 1) or (value.shape[0] == 1 and count != 1 and key != species_key):
                raise ValueError(f'{key!r} has {value.shape[0]} rows in a group of {count} frames.')
            if axis is not None:
                if value.ndim <= axis:
                    raise ValueError(f'{key!r} has no atom axis {axis}; its shape is {value.shape}.')
                shape[axis - 1] = None
            trailing.add(tuple(shape))
        if len(trailing) > 1:
            raise ValueError(f'{key!r} has different shapes in different label groups, {sorted(map(str, trailing))}; '
                             'if it is an atom property, name it in the trainer\'s atom keys.')
        shape = [n_atoms_max if size is None else size for size in next(iter(trailing))]
        dtype = _column_dtype(key, values)
        out = np.zeros((total, *shape), dtype=dtype) if key not in defaults else np.full(
            (total, *shape), defaults[key], dtype=dtype)
        start = 0
        for batch, count in zip(batches, counts):
            if key in batch:
                value = np.asarray(batch[key])
                if value.shape[0] == 1 and count != 1:
                    value = np.broadcast_to(value, (count, *value.shape[1:]))
                out[(slice(start, start + count), *(slice(0, size) for size in value.shape[1:]))] = value
            start += count
        arrays[key] = out
    arrays['sys_number'] = np.repeat(np.arange(len(batches), dtype=np.int64), counts)
    return arrays
