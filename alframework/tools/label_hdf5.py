"""Write HDF5 label groups and check the backing files of virtual label shards."""

import json
import warnings
from pathlib import Path

import h5py
import numpy as np


LABEL_PARTS = 'alf_label_parts'
PROPERTY_SCHEMA = 'alf_label_properties'
LABEL_UNITS = 'alf_label_units'
SCHEMA_VERSION = 'alf_label_schema_version'
# How a shard groups its frames (label_h5_layout.group_by). "formula": one group per sorted species list, the
# species stored once per group. "atom_count": one group per atom count (and periodicity and topology tagging), the
# species stored for each frame. Shards without the attribute group by formula.
GROUP_BY = 'alf_label_group_by'
GROUP_BY_CHOICES = ('formula', 'atom_count')
# A formula-grouped shard with more groups than this, and more than MANY_GROUPS_PER_LABEL groups per label, spends
# most of its storage and reading time on per-group work.
MANY_GROUPS = 10_000
MANY_GROUPS_PER_LABEL = 0.1


def split_group_by(h5_options):
    """``(group_by, writer options)`` from a ``label_h5_layout``'s writer options: the grouping is the caller's,
    the rest configure :class:`LabelH5Writer`."""
    options = dict(h5_options or {})
    group_by = options.pop('group_by', None) or 'formula'
    if group_by not in GROUP_BY_CHOICES:
        raise ValueError(f'label_h5_layout group_by must be one of {list(GROUP_BY_CHOICES)}, not {group_by!r}.')
    return group_by, options


def many_groups_warning(path, groups, frames, group_by):
    """Warn when grouping by formula leaves nearly one group per label, which is slow to store and to read."""
    if group_by == 'formula' and groups > MANY_GROUPS and groups > MANY_GROUPS_PER_LABEL * frames:
        warnings.warn(f'{path} holds {groups:,} formula groups for {frames:,} labels, so storing and reading it is '
                      'dominated by per-group work. With many compositions, label_h5_layout group_by "atom_count" '
                      'groups frames by atom count instead.', stacklevel=3)


def discover_label_shards(path):
    """Select published HDF5 label shards in lexical order, or one explicit file."""
    root = Path(path).expanduser().absolute()
    if root.is_file():
        return [str(root)]
    return sorted(str(path) for path in root.glob('*')
                  if path.suffix in ('.h5', '.hdf5') and path.is_file())


def label_units_setting(properties, units=None):
    """Units declared for the stored HDF5 columns."""
    if units is None:
        return {}
    if not isinstance(units, dict):
        raise ValueError('label_units must map HDF5 dataset names to unit strings.')
    datasets = {spec[0] for spec in properties.values()}
    for key, unit in units.items():
        if key not in datasets or not isinstance(unit, str) or not unit.strip():
            raise ValueError(f'label_units[{key!r}] must name a stored dataset and give a nonempty unit string.')
    return {key: value.strip() for key, value in units.items()}


def label_schema_attributes(properties, units=None):
    """Metadata for the conversion already applied to a shard's rows."""
    units = label_units_setting(properties, units)
    attributes = {PROPERTY_SCHEMA: json.dumps(properties, sort_keys=True), SCHEMA_VERSION: 1}
    if units:
        attributes[LABEL_UNITS] = json.dumps(units, sort_keys=True)
    return attributes


def validate_training_units(shards, targets=None):
    """Check explicit units across one training inventory, including legacy shards."""
    targets = None if targets is None else set(targets)
    declared = {}
    units_by_shard = {}
    for path in shards:
        with h5py.File(path, 'r') as handle:
            units = json.loads(handle.attrs.get(LABEL_UNITS, '{}'))
        units_by_shard[path] = units
        for key, unit in units.items():
            if targets is not None and key not in targets:
                continue
            previous_unit, previous_path = declared.setdefault(key, (unit, path))
            if unit != previous_unit:
                raise ValueError(f'{path}: {key!r} declares units {unit!r}, but {previous_path} declares '
                                 f'{previous_unit!r}.')
    unknown = sorted({key for units in units_by_shard.values() for key in declared if key not in units})
    if unknown:
        warnings.warn(f'Training shards mix declared and unknown stored units for {", ".join(unknown)}.',
                      stacklevel=2)


def validate_label_parts(handle):
    """Check a virtual label shard before HDF5 can treat a missing source as fill values."""
    parts = json.loads(handle.attrs.get(LABEL_PARTS, '[]'))
    directory = Path(handle.filename).parent
    for relative_path in parts:
        source = directory / relative_path
        if not source.is_file():
            raise FileNotFoundError(f'{handle.filename}: required label part {source} is missing')


class LabelH5Writer:
    """Write groups of label arrays; the caller chooses the groups (see ``GROUP_BY``).

    ``complib`` and ``complevel`` select lossless compression. ``shuffle`` adds the byte-shuffle filter
    to numeric arrays, and ``frame_chunk`` groups that many whole frames into each chunk.
    """

    def __init__(self, store_file, mode='w-', complib='gzip', complevel=6, shuffle=False, frame_chunk=None,
                 **file_options):
        self.store = h5py.File(store_file, mode=mode, **file_options)
        self.clib = complib
        self.clev = None if complib in (None, 'lzf') else complevel
        self.shuffle = shuffle
        self.frame_chunk = frame_chunk

    def store_data(self, store_loc, **kwargs):
        """Create a group and write its datasets."""
        group = self.store.create_group(store_loc)
        for name, values in kwargs.items():
            if type(values) is list and values and type(values[0]) in (str, np.str_):
                values = [value.encode('utf8') for value in values]

            options = dict(compression=self.clib, compression_opts=self.clev)
            array = np.asarray(values)
            if self.shuffle and array.dtype.kind in 'biuf':
                options['shuffle'] = True
            if self.frame_chunk and array.ndim >= 1 and array.size and (self.clib or self.shuffle):
                options['chunks'] = (min(int(self.frame_chunk), array.shape[0]),) + array.shape[1:]
            group.create_dataset(name, data=values, **options)

    def cleanup(self):
        """Close the HDF5 file."""
        self.store.close()
