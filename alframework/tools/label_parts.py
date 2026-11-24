"""Durable label batches and an atomic HDF5 view of each training generation."""

import json
import os
from pathlib import Path

import h5py
import numpy as np

from alframework.tools.files import fsync_file
from alframework.tools.label_hdf5 import (
    LABEL_PARTS, LABEL_UNITS, PROPERTY_SCHEMA, SCHEMA_VERSION, LabelH5Writer, label_schema_attributes,
    validate_label_parts,
)
from alframework.tools.restarts import HeldLabels, shard_frames
from alframework.tools.data_shards import PAIR_INDEX_DATASET, STRUCTURE_STATE_DATASETS, store_current_data
from alframework.tools.label_sources import LEGACY_LABEL_PROVENANCE, SHARD_LABEL_SOURCES, write_shard_label_sources


STATIC = frozenset(('species', PAIR_INDEX_DATASET))
STATE_DEFAULTS = dict(zip(STRUCTURE_STATE_DATASETS, (0, 1)))
REJECTED_IDS = 'alf_rejected_label_ids'


# With bounded storage, the longest a coordinator loop pass spends writing label parts beyond its first one.
HOLD_SECONDS = 30.0

# data_by_reference "labels": QM workers write their labels as parts here, and return receipts.
INCOMING = 'incoming'
# The receipts a generation has adopted, one JSON line each, in its part directory.
MANIFEST = 'parts.jsonl'
# On a merged part: the manifest lines (receipts) it holds, so a restart doesn't count them twice.
MERGED_FROM = 'alf_merged_from'
# Processes that merge worker parts into parts of label_storage_batch_size before a snapshot is published.
MERGE_WORKERS = 16
PERIODICITY_CONFLICT = ('the next shard already holds {kind} {formula} frames, and a formula group cannot mix '
                        'periodic and nonperiodic frames.')


def incoming_directory(snapshot):
    """Where QM workers write label parts for the snapshots beside ``snapshot``, whatever their generation."""
    return part_directory(snapshot).parent / INCOMING


class LabelReceipt:
    """What a QM worker returns for the labels it wrote as one part, instead of the labeled structures.

    ``path`` is the part (``None`` when no label converged), ``formulas`` maps each formula group to its IDs and
    periodicity, and ``rejected`` maps each unconverged or invalid label's ID to the reason.
    """

    def __init__(self, path, formulas, rejected):
        self.path = path
        self.formulas = formulas
        self.rejected = rejected

    def ids(self):
        return [i for group in self.formulas.values() for i in group['ids']] + list(self.rejected)

    def __len__(self):
        return sum(len(group['ids']) for group in self.formulas.values()) + len(self.rejected)

    def as_record(self):
        return {'path': self.path, 'formulas': self.formulas, 'rejected': self.rejected}

    @classmethod
    def from_record(cls, record):
        return cls(record['path'], record['formulas'], record['rejected'])

    def __reduce__(self):
        return (LabelReceipt, (self.path, self.formulas, self.rejected))


def _failure_reason(molecule):
    metadata = molecule.get_metadata() or {}
    return str(metadata.get('labeling_error') or metadata.get('qm_error') or 'the label did not converge')


def write_label_part(results, directory, properties, schema, h5_options=None, label_units=None):
    """On a QM worker: screen ``results`` as the coordinator would, store the converged ones as one part in
    ``directory``, and return a :class:`LabelReceipt`.

    Labels that can't be stored are marked unconverged first (``screen_labels``), and a formula group keeps one
    periodicity within the part, as ``HDF5HeldLabels`` does across parts.
    """
    import uuid
    from alframework.tools.batching import flatten_molecule_output
    from alframework.tools.data_shards import compute_empirical_formula
    from alframework.tools.label_validation import reject_label, screen_labels

    results = flatten_molecule_output(results)
    screen_labels(results, properties)
    formulas, rejected, accepted = {}, {}, []
    for result in results:
        if result.check_convergence():
            atoms = result.get_atoms()
            formula = compute_empirical_formula(atoms.get_chemical_symbols())
            periodic = bool(atoms.pbc.all())
            group = formulas.setdefault(formula, {'ids': [], 'periodic': periodic})
            if group['periodic'] != periodic:
                reject_label(result, PERIODICITY_CONFLICT.format(
                    kind='periodic' if group['periodic'] else 'nonperiodic', formula=formula))
            else:
                group['ids'].append(result.get_moleculeid())
                accepted.append(result)
                continue
        rejected[result.get_moleculeid()] = _failure_reason(result)
    formulas = {formula: group for formula, group in formulas.items() if group['ids']}
    path = None
    if accepted:
        if json.loads(schema) != json.loads(json.dumps(properties)):
            raise ValueError('Worker label properties differ from the configured generation schema.')
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = str(Path(directory) / f'task-{uuid.uuid4().hex}.h5')
        store_current_data(path, accepted, properties, label_units=label_units,
                           attributes={REJECTED_IDS: json.dumps(sorted(rejected))},
                           h5_options={'libver': 'latest', 'track_order': True, **(h5_options or {})})
        fsync_file(directory)
    return LabelReceipt(path, formulas, rejected)


def _validate_part_schema(handle, metadata):
    if handle.attrs.get(SCHEMA_VERSION, metadata[SCHEMA_VERSION]) != metadata[SCHEMA_VERSION]:
        raise ValueError(f'{handle.filename}: unsupported label schema version for this generation.')
    for attribute, default in ((PROPERTY_SCHEMA, None), (LABEL_UNITS, '{}')):
        recorded = handle.attrs.get(attribute, default)
        if recorded is None or json.loads(recorded) != json.loads(metadata.get(attribute, default)):
            raise ValueError(f'{handle.filename}: label properties or units differ from this generation.')


def merge_label_parts(job):
    """Merge worker parts into one part: each source read once, its formula groups appended in order.

    ``job`` is ``(target, sources, rejected, metadata, h5_options, lines)``, where ``sources`` lists
    ``(path, excluded)`` with the formula groups to leave out of that part and ``lines`` the manifest lines it
    holds. The part is written to ``target`` all at once, and records ``lines`` (``MERGED_FROM``). Returns
    ``target`` and the converged labels it holds.
    """
    target, sources, rejected, metadata, h5_options, lines = job
    groups, label_sources = {}, {}
    for path, excluded in sources:
        with h5py.File(path, 'r') as handle:
            _validate_part_schema(handle, metadata)
            recorded = handle.attrs.get(SHARD_LABEL_SOURCES, handle.attrs.get(LEGACY_LABEL_PROVENANCE, '[]'))
            for source in json.loads(recorded):
                label_sources.setdefault(json.dumps(source, sort_keys=True, default=str), source)
            for formula, group in handle.items():
                if formula in excluded:
                    continue
                columns = {key: dataset[()] for key, dataset in group.items()}
                merged = groups.setdefault(formula, {'static': {}, 'pieces': []})
                for key in STATIC.intersection(columns):
                    value = columns.pop(key)
                    if key in merged['static'] and not np.array_equal(merged['static'][key], value):
                        raise ValueError(f'{path}:{formula} has species or pair ordering that differs from the '
                                         'other parts of its formula.')
                    merged['static'][key] = value
                merged['pieces'].append(columns)
    stored = 0
    partial = Path(str(target) + '.partial')
    try:
        packer = LabelH5Writer(str(partial), mode='w', **{'libver': 'latest', 'track_order': True,
                                                       **(h5_options or {})})
        try:
            for formula, merged in groups.items():
                pieces = merged['pieces']
                keys = set().union(*pieces)
                if any(set(piece) - STATE_DEFAULTS.keys() != keys - STATE_DEFAULTS.keys() for piece in pieces):
                    raise ValueError(f'{formula} has different datasets in different worker parts.')
                data = dict(merged['static'])
                for key in sorted(keys):
                    data[key] = np.concatenate([
                        piece[key] if key in piece else np.full(len(piece['_id']), STATE_DEFAULTS[key])
                        for piece in pieces])
                stored += len(data['_id'])
                packer.store_data(formula, **data)
            write_shard_label_sources(packer.store, list(label_sources.values()))
            packer.store.attrs[REJECTED_IDS] = json.dumps(sorted(rejected))
            packer.store.attrs.update(metadata)
            packer.store.attrs[MERGED_FROM] = json.dumps(list(lines))
        finally:
            packer.cleanup()
        fsync_file(partial)
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return str(target), stored


def hold_ready(take_one_part, waiting, batch_size, *, drain=False, seconds=HOLD_SECONDS, clock=None):
    """Write label parts until enough are written for this loop pass; returns how many were written.

    ``take_one_part()`` holds up to ``batch_size`` finished labels; ``waiting()`` counts the finished labels still
    in the queue. A pass writes one part, then more while a full part is waiting, for up to ``seconds``, so finished
    labels don't wait in memory while the loop writes one part per pass (at large scale a pass takes seconds).
    ``drain`` writes everything. Without a ``batch_size`` (unbounded storage) one call holds everything at once.
    """
    import time
    clock = clock or time.monotonic
    started, written = clock(), 0
    while True:
        take_one_part()
        written += 1
        left = waiting() if batch_size is not None else 0
        if not left or (not drain and (left < batch_size or clock() - started >= seconds)):
            return written


def part_directory(snapshot):
    snapshot = Path(snapshot)
    return snapshot.parent / '.alf-label-parts' / snapshot.name


class HDF5HeldLabels(HeldLabels):
    """Each held batch is an immutable file in its generation's private directory.

    With ``data_by_reference`` labels, QM workers write their own parts and the coordinator adopts their receipts
    (:meth:`adopt`) into the generation's manifest; :meth:`commit` merges those parts into parts of
    ``batch_size`` labels, with ``merge_workers`` processes, before publishing, so readers see the same layout.
    """

    def __init__(self, snapshot, properties, batch_size, h5_options=None, *, merge_workers=MERGE_WORKERS,
                 keep_worker_parts=False, label_units=None):
        self.h5_options = dict(h5_options or {})
        self.properties = properties
        self.label_units = label_units
        self.schema_attributes = label_schema_attributes(properties, label_units)
        self.schema = self.schema_attributes[PROPERTY_SCHEMA]
        self.batch_size = batch_size
        self.merge_workers = merge_workers
        self.keep_worker_parts = keep_worker_parts
        self.select_snapshot(snapshot)

    def select_snapshot(self, snapshot):
        self.directory = part_directory(snapshot)
        self.failed_since_store = 0
        self._successful = self._converged = self._ids = self._periodic = None
        self._receipts = None
        self._next_piece = 0

    def pieces(self):
        return sorted(path.name for path in self.directory.glob('part-*.h5'))

    def manifest(self):
        return self.directory / MANIFEST

    def _scan(self):
        converged, rejected, ids, periodicity, merged = 0, 0, set(), {}, set()
        pieces = self.pieces()
        for name in pieces:
            with h5py.File(self.directory / name, 'r') as handle:
                _validate_part_schema(handle, self.schema_attributes)
                merged.update(json.loads(handle.attrs.get(MERGED_FROM, '[]')))
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
        # Adopted receipts not yet merged; a merged part already counts the ones it holds.
        self._receipts = []
        records = self._manifest_records()
        self._manifest_lines = len(records)
        for line, record in enumerate(records):
            if line not in merged:
                record['_line'] = line
                self._count_receipt(record)

    def _manifest_records(self):
        """The manifest's receipts. A last line a crash cut short was never adopted: it is cut off, so the next
        append starts a line of its own."""
        path = self.manifest()
        if not path.exists():
            return []
        data = path.read_bytes()
        if data and not data.endswith(b'\n'):
            data = data[:data.rfind(b'\n') + 1]
            with open(path, 'r+b') as stream:
                stream.truncate(len(data))
                os.fsync(stream.fileno())
        return [json.loads(line) for line in data.decode().splitlines()]

    def _count_receipt(self, record):
        excluded = record.get('excluded', {})
        for formula, group in record['formulas'].items():
            if formula not in excluded:
                self._periodic.setdefault(formula, group['periodic'])
                self._converged += len(group['ids'])
        self._successful += sum(len(group['ids']) for group in record['formulas'].values()) + len(record['rejected'])
        self._ids.update(i for group in record['formulas'].values() for i in group['ids'])
        self._ids.update(record['rejected'])
        self._receipts.append(record)

    def hold(self, results, failed=0):
        if self._ids is None:
            self._scan()
        self.failed_since_store += failed
        for start in range(0, len(results), self.batch_size):
            chunk = results[start:start + self.batch_size]
            self._write_piece(chunk)

    def adopt(self, receipts, failed=0):
        """Record QM workers' parts by their receipts, with one sync of the manifest for the call.

        A formula group whose periodicity conflicts with what this generation already holds is left out of the
        part when it is merged, and its labels count as rejected, as :meth:`hold` would hold them.
        """
        if self._ids is None:
            self._scan()
        if not receipts:
            self.failed_since_store += failed
            return
        self._ensure_directory()
        records = []
        periodicity = self._periodic.copy()
        for receipt in receipts:
            record = receipt.as_record()
            excluded = {}
            for formula, group in record['formulas'].items():
                held = periodicity.setdefault(formula, group['periodic'])
                if held != group['periodic']:
                    excluded[formula] = PERIODICITY_CONFLICT.format(kind='periodic' if held else 'nonperiodic',
                                                                    formula=formula)
            if excluded:
                record['excluded'] = excluded
            records.append(record)
        first = self._manifest_lines
        manifest = self.manifest()
        created = not manifest.exists()
        with open(manifest, 'a') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if created:
            fsync_file(self.directory)
        self.failed_since_store += failed
        self._manifest_lines += len(records)
        for line, record in enumerate(records, first):
            record['_line'] = line
            self._count_receipt(record)

    def _ensure_directory(self):
        if not self.directory.is_dir():
            self.directory.mkdir(parents=True)
            fsync_file(self.directory.parent)
            fsync_file(self.directory.parent.parent)

    def _write_piece(self, results):
        self._ensure_directory()
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
                                        attributes={REJECTED_IDS: json.dumps(rejected)}, label_units=self.label_units,
                                        h5_options={'libver': 'latest', 'track_order': True, **self.h5_options})
            fsync_file(self.directory)
        except BaseException:
            self._ids = None  # Scan durable pieces and reconstruct the cursor if the caller retries.
            raise
        self._next_piece += 1
        self._successful += len(results)
        self._converged += stored
        self._ids.update(result.get_moleculeid() for result in results)

    def merge_jobs(self):
        """The adopted receipts, in manifest order, as merges of about ``batch_size`` converged labels each."""
        if self._ids is None:
            self._scan()
        jobs, sources, rejected, lines, labels = [], [], {}, [], 0

        def close():
            if lines:
                target = self.directory / f'part-{self._next_piece + len(jobs):08d}.h5'
                jobs.append((str(target), list(sources), dict(rejected), self.schema_attributes,
                             self.h5_options, list(lines)))
                sources.clear()
                rejected.clear()
                lines.clear()

        for record in self._receipts:
            lines.append(record['_line'])
            excluded = record.get('excluded', {})
            rejected.update(record['rejected'])
            for formula, reason in excluded.items():
                rejected.update(dict.fromkeys(record['formulas'][formula]['ids'], reason))
            kept = sum(len(group['ids']) for formula, group in record['formulas'].items() if formula not in excluded)
            if record['path'] is not None:
                sources.append((record['path'], sorted(excluded)))
                labels += kept
            if labels >= self.batch_size:
                close()
                labels = 0
        close()
        return jobs

    def merge(self):
        """Merge the adopted worker parts into this generation's parts, in parallel; returns how many."""
        jobs = self.merge_jobs()
        if not jobs:
            return 0
        if len(jobs) == 1 or self.merge_workers <= 1:
            for job in jobs:
                merge_label_parts(job)
        else:
            import concurrent.futures
            import multiprocessing
            # Spawned, not forked: the coordinator holds Parsl's threads and sockets.
            context = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.merge_workers, len(jobs)),
                                                        mp_context=context) as pool:
                list(pool.map(merge_label_parts, jobs))
        fsync_file(self.directory)
        self._ids = None  # The merged parts now hold what the receipts described.
        return len(jobs)

    def commit(self, snapshot, properties, attributes):
        if json.dumps(properties, sort_keys=True) != self.schema:
            raise ValueError('Label properties and units must stay fixed while a generation is being written.')
        if self._ids is None:
            self._scan()
        worker_parts = [record['path'] for record in self._receipts if record['path'] is not None]
        self.merge()
        parts = [self.directory / name for name in self.pieces()]
        stored = publish_snapshot(Path(snapshot), parts, {**attributes, **self.schema_attributes})
        if not self.keep_worker_parts:
            for path in worker_parts:
                Path(path).unlink(missing_ok=True)
        return stored

    def recover_snapshot(self, snapshot):
        # Published views still reference these files. Restart must keep them.
        return shard_frames(snapshot)


def remove_orphan_worker_parts(held):
    """Remove worker parts that ``held``'s generation hasn't adopted: a worker wrote them, but the coordinator
    stopped before recording their receipts, or after publishing the snapshot that merged them. Returns how many."""
    incoming = held.directory.parent / INCOMING
    if not incoming.is_dir():
        return 0
    adopted = {record['path'] for record in held._manifest_records()}
    removed = 0
    for path in incoming.glob('task-*.h5'):
        if str(path) not in adopted:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


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
