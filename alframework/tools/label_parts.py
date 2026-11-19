"""Durable label batches and an atomic HDF5 view of each training generation."""

import json
import os
from pathlib import Path

import h5py
import numpy as np

from alframework.tools.files import fsync_file
from alframework.tools.pyanitools import LABEL_PARTS, validate_label_parts
from alframework.tools.restarts import HeldLabels, shard_frames
from alframework.tools.data_shards import PAIR_INDEX_DATASET, STRUCTURE_STATE_DATASETS, store_current_data
from alframework.tools.label_sources import LEGACY_LABEL_PROVENANCE, SHARD_LABEL_SOURCES, write_shard_label_sources


STATIC = frozenset(('species', PAIR_INDEX_DATASET))
STATE_DEFAULTS = dict(zip(STRUCTURE_STATE_DATASETS, (0, 1)))
REJECTED_IDS = 'alf_rejected_label_ids'
PROPERTY_SCHEMA = 'alf_label_properties'


def part_directory(snapshot):
    snapshot = Path(snapshot)
    return snapshot.parent / '.alf-label-parts' / snapshot.name


class HDF5HeldLabels(HeldLabels):
    """Each held batch is an immutable file in its generation's private directory."""

    def __init__(self, snapshot, properties, batch_size, h5_options=None):
        self.h5_options = dict(h5_options or {})
        self.properties = properties
        self.schema = json.dumps(properties, sort_keys=True)
        self.batch_size = batch_size
        self.select_snapshot(snapshot)

    def select_snapshot(self, snapshot):
        self.directory = part_directory(snapshot)
        self.failed_since_store = 0
        self._successful = self._converged = self._ids = self._periodic = None
        self._next_piece = 0

    def pieces(self):
        return sorted(path.name for path in self.directory.glob('part-*.h5'))

    def _scan(self):
        converged, rejected, ids, periodicity = 0, 0, set(), {}
        pieces = self.pieces()
        for name in pieces:
            with h5py.File(self.directory / name, 'r') as handle:
                if handle.attrs[PROPERTY_SCHEMA] != self.schema:
                    raise ValueError(f'{handle.filename}: resume with the same label properties and units.')
                failed_ids = json.loads(handle.attrs[REJECTED_IDS])
                rejected += len(failed_ids)
                ids.update(failed_ids)
                for formula, group in handle.items():
                    stored_ids = group['_id'].asstr()[:]
                    converged += len(stored_ids)
                    ids.update(stored_ids)
                    periodicity[formula] = 'cell' in group
        self._successful = converged + rejected
        self._converged, self._ids, self._periodic = converged, ids, periodicity
        self._next_piece = int(Path(pieces[-1]).stem.split('-')[-1]) + 1 if pieces else 0

    def hold(self, results, failed=0):
        if self._ids is None:
            self._scan()
        self.failed_since_store += failed
        for start in range(0, len(results), self.batch_size):
            chunk = results[start:start + self.batch_size]
            self._write_piece(chunk)

    def _write_piece(self, results):
        if not self.directory.is_dir():
            self.directory.mkdir(parents=True)
            fsync_file(self.directory.parent)
            fsync_file(self.directory.parent.parent)
        accepted, rejected = [], []
        for result in results:
            self._keep_one_periodicity(result)
            if result.check_convergence():
                accepted.append(result)
            else:
                rejected.append(result.get_moleculeid())
        path = self.directory / f'part-{self._next_piece:08d}.h5'
        try:
            stored = store_current_data(str(path), accepted, self.properties,
                                        attributes={REJECTED_IDS: json.dumps(rejected), PROPERTY_SCHEMA: self.schema},
                                        h5_options={'libver': 'latest', 'track_order': True, **self.h5_options})
            fsync_file(self.directory)
        except BaseException:
            self._ids = None  # Scan durable pieces and reconstruct the cursor if the caller retries.
            raise
        self._next_piece += 1
        self._successful += len(results)
        self._converged += stored
        self._ids.update(result.get_moleculeid() for result in results)

    def commit(self, snapshot, properties, attributes):
        if json.dumps(properties, sort_keys=True) != self.schema:
            raise ValueError('Label properties and units must stay fixed while a generation is being written.')
        parts = [self.directory / name for name in self.pieces()]
        return publish_snapshot(Path(snapshot), parts, attributes)

    def recover_snapshot(self, snapshot):
        # Published views still reference these files. Restart must keep them.
        return shard_frames(snapshot)


def publish_snapshot(path, parts, attributes):
    """Keep the per-formula schema, row order, and the parts' label sources."""
    if path.exists():
        raise FileExistsError(path)
    relative_paths = [os.path.relpath(part, path.parent) for part in parts]
    groups, sources, frames = {}, {}, 0
    for part, relative in zip(parts, relative_paths):
        with h5py.File(part, 'r') as handle:
            recorded = handle.attrs.get(SHARD_LABEL_SOURCES, handle.attrs.get(LEGACY_LABEL_PROVENANCE, '[]'))
            for source in json.loads(recorded):
                sources.setdefault(json.dumps(source, sort_keys=True, default=str), source)
            for formula, group in handle.items():
                size = len(group['_id'])
                frames += size
                static = {key: group[key][:] for key in STATIC.intersection(group.keys())}
                columns = {key: (value.dtype, value.shape) for key, value in group.items() if key not in STATIC}
                if formula not in groups:
                    groups[formula] = (static, [])
                expected, segments = groups[formula]
                if static.keys() != expected.keys() or any(
                    not np.array_equal(value, expected[key]) for key, value in static.items()
                ):
                    raise ValueError(f'{part}:{formula} has inconsistent species or pair ordering.')
                segments.append((relative, size, columns))

    write_state = any(STATE_DEFAULTS.keys() & columns.keys()
                      for _, segments in groups.values() for _, _, columns in segments)
    partial = path.with_suffix(path.suffix + '.partial')
    try:
        with h5py.File(partial, 'w', libver='latest', track_order=True) as handle:
            for formula, (static, segments) in sorted(groups.items()):
                group = handle.create_group(formula)
                for name, data in static.items():
                    group.create_dataset(name, data=data)
                _write_virtual_columns(group, segments, write_state)
            handle.attrs.update(attributes)
            handle.attrs[LABEL_PARTS] = json.dumps(relative_paths)
            write_shard_label_sources(handle, list(sources.values()))
            validate_label_parts(handle)
        fsync_file(partial)
        os.replace(partial, path)
        fsync_file(path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return frames


def _write_virtual_columns(group, segments, write_state):
    keys = set(segments[0][2]) - STATE_DEFAULTS.keys()
    frames = sum(size for _, size, _ in segments)
    if any(columns.keys() - STATE_DEFAULTS.keys() != keys for _, _, columns in segments):
        raise ValueError(f'{group.name} has inconsistent data columns across label parts.')
    if write_state:
        keys.update(STATE_DEFAULTS)
    for name in sorted(keys):
        shapes = {columns[name][1][1:] for _, _, columns in segments if name in columns}
        if len(shapes) > 1:
            raise ValueError(f'{group.name}/{name} has inconsistent shapes across label parts.')
        shape = next(iter(shapes), ())
        dtypes = [columns[name][0] for _, _, columns in segments if name in columns]
        dtype = np.result_type(*dtypes) if dtypes else np.int64
        if not dtypes:
            group.create_dataset(name, shape=(frames,), dtype=dtype, fillvalue=STATE_DEFAULTS[name])
            continue
        layout = h5py.VirtualLayout(shape=(frames, *shape), dtype=dtype)
        offset = 0
        for source, size, columns in segments:
            if name in columns:
                _, source_shape = columns[name]
                if source_shape[0] != size:
                    raise ValueError(f'{source}:{group.name}/{name} is misaligned with its label IDs.')
                layout[offset:offset + size] = h5py.VirtualSource(source, f'{group.name}/{name}', shape=source_shape)
            offset += size
        group.create_virtual_dataset(name, layout, fillvalue=STATE_DEFAULTS.get(name))
