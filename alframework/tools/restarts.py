"""Keep finished labels on disk until they're stored, and make status agree with disk on a restart.

The coordinator stores labels into ``h5_dir`` only between trainings, so that every ensemble member
trains on the same shards. Until then, finished QM results were held only in memory, and a crash
lost them. :class:`HeldLabels` writes them to a holding area outside ``h5_dir`` as they finish, and
:func:`reconcile_status` picks up what a crash left behind.
"""
import json
import os
import pickle
from pathlib import Path

import h5py

from alframework.tools.files import write_bytes_atomic, write_json_atomic
from alframework.tools.label_hdf5 import validate_label_parts
from alframework.tools.label_validation import reject_label

HELD_PIECES = 'held_pieces'
# Marks the shard stored when a label budget stops a run. No model trained on it.
STORED_AT_BUDGET_STOP = 'stored_at_budget_stop'
FORMAT = 1


def held_labels_directory(master_config):
    """``held_labels_dir`` if set, otherwise ``<h5_dir>_held`` beside ``h5_dir``. Never inside ``h5_dir``."""
    h5_dir = Path(master_config['h5_dir']).resolve()
    configured = master_config.get('held_labels_dir')
    directory = Path(configured).resolve() if configured else h5_dir.parent / (h5_dir.name + '_held')
    if directory == h5_dir or h5_dir in directory.parents:
        raise ValueError(f'held_labels_dir ({directory}) must be outside h5_dir ({h5_dir}), because trainers '
                         'read everything in h5_dir.')
    return directory


LABEL_LAYOUT_KEYS = {'compression': 'complib', 'level': 'complevel', 'shuffle': 'shuffle', 'frame_chunk': 'frame_chunk'}


def label_h5_options(master_config):
    """HDF5 writer options from the optional ``label_h5_layout``: how stored label datasets are compressed
    and chunked. All choices are lossless; without it labels are written with gzip level 6, as always."""
    layout = master_config.get('label_h5_layout')
    if layout is None:
        return {}
    if not isinstance(layout, dict) or set(layout) - set(LABEL_LAYOUT_KEYS):
        raise ValueError(f'label_h5_layout must be a dictionary with keys from {sorted(LABEL_LAYOUT_KEYS)}.')
    compression = layout.get('compression', 'gzip')
    if compression not in ('gzip', 'lzf', None):
        raise ValueError("label_h5_layout compression must be 'gzip', 'lzf' or null.")
    level = layout.get('level', 6)
    if compression == 'gzip' and (isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 9):
        raise ValueError('label_h5_layout level must be an integer from 0 to 9 for gzip.')
    chunk = layout.get('frame_chunk')
    if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1):
        raise ValueError('label_h5_layout frame_chunk must be a positive integer.')
    if not isinstance(layout.get('shuffle', False), bool):
        raise ValueError('label_h5_layout shuffle must be true or false.')
    return {LABEL_LAYOUT_KEYS[key]: value for key, value in dict(layout, compression=compression, level=level).items()}


def label_storage_batch_size(master_config):
    size = master_config.get('label_storage_batch_size')
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
        raise ValueError('label_storage_batch_size must be a positive integer.')
    return size


def open_held_labels(master_config, current_h5_id):
    from alframework.tools.label_parts import HDF5HeldLabels, part_directory

    size = label_storage_batch_size(master_config)
    legacy = held_labels_directory(master_config)
    pattern = master_config['h5_path']
    snapshot = Path(pattern.format(current_h5_id))
    while snapshot.exists():
        current_h5_id += 1
        snapshot = Path(pattern.format(current_h5_id))
    if size is None:
        if any(part_directory(snapshot).glob('part-*.h5')):
            raise ValueError('An unfinished HDF5 generation exists. Resume with label_storage_batch_size set.')
        return HeldLabels(legacy, h5_options=label_h5_options(master_config),
                          label_units=master_config.get('label_units'))
    if any(legacy.glob('held-*.pkl')):
        raise ValueError('Held pickle files exist. Resume with the original storage mode to commit them first.')
    from alframework.tools.sampled_frames import data_by_reference_settings
    reference = data_by_reference_settings(master_config) or {}
    return HDF5HeldLabels(snapshot, master_config['properties_list'], size, h5_options=label_h5_options(master_config),
                          merge_workers=reference.get('merge_workers', 1),
                          keep_worker_parts=bool(reference.get('keep_files')),
                          label_units=master_config.get('label_units'))


class HeldLabels:
    """Finished QM results that aren't stored yet, kept as atomically written pieces.

    Counts use the QM queue's units (tasks, or structures for batched labeling).
    Retraining and a finite bootstrap count converged labels; a bootstrap that isn't finite, and
    ``maximum_completed_QM``, count every result that returned, converged or not. ``finished()`` also
    includes failed tasks, which hold no results and are counted only for this process.
    """

    PIECE_SIZE = 1000

    def __init__(self, directory, h5_options=None, *, label_units=None):
        self.directory = Path(directory)
        self.h5_options = dict(h5_options or {})
        self.label_units = label_units
        self.directory.mkdir(parents=True, exist_ok=True)
        self.failed_since_store = 0
        self._successful = None
        self._converged = None
        self._ids = None
        self._periodic = None

    def pieces(self):
        return sorted(path.name for path in self.directory.glob('held-*.pkl'))

    def _read(self, name):
        with open(self.directory / name, 'rb') as handle:
            piece = pickle.load(handle)
        if piece.get('format') != FORMAT:
            raise ValueError(f'{self.directory / name} has an unknown format; it was written by another ALF version.')
        return piece

    def _scan(self):
        self._successful, self._converged, self._ids, self._periodic = 0, 0, set(), {}
        for name in self.pieces():
            piece = self._read(name)
            self._successful += piece['successful']
            self._converged += sum(1 for result in piece['results'] if result.check_convergence())
            self._ids.update(result.get_moleculeid() for result in piece['results'])
            for result in piece['results']:
                self._keep_one_periodicity(result)

    def _keep_one_periodicity(self, result):
        """A shard's formula group holds only periodic or only nonperiodic frames, so a converged label that
        would mix them in the next shard is held unconverged instead: storing it would fail on every restart."""
        if not result.check_convergence():
            return
        from alframework.tools.data_shards import compute_empirical_formula
        atoms = result.get_atoms()
        formula = compute_empirical_formula(atoms.get_chemical_symbols())
        periodic = bool(atoms.pbc.all())
        held = self._periodic.setdefault(formula, periodic)
        if held != periodic:
            kind = 'periodic' if held else 'nonperiodic'
            reject_label(result, f'the next shard already holds {kind} {formula} frames, and a formula group '
                                 'cannot mix periodic and nonperiodic frames.')

    def hold(self, results, failed=0):
        """Write ``results`` as new pieces of at most ``PIECE_SIZE`` each, and track failures until the next store.

        Bounded pieces keep a store's memory bounded too: it reads one piece at a time, and when most QM tasks
        finish together a single piece would otherwise hold nearly every label of the cycle. A converged label
        whose periodicity conflicts with held labels of its formula is held unconverged.
        """
        self.failed_since_store += failed
        if not results:
            return
        results = list(results)
        if self._periodic is None:
            self._scan()
        for result in results:
            self._keep_one_periodicity(result)
        existing = self.pieces()
        number = int(existing[-1][len('held-'):-len('.pkl')]) + 1 if existing else 0
        for start in range(0, len(results), self.PIECE_SIZE):
            chunk = results[start:start + self.PIECE_SIZE]
            piece = dict(format=FORMAT, successful=len(chunk), results=chunk)
            write_bytes_atomic(self.directory / f'held-{number:08d}.pkl', pickle.dumps(piece, pickle.HIGHEST_PROTOCOL))
            number += 1
        self._successful += len(results)
        self._converged += sum(1 for result in results if result.check_convergence())
        self._ids.update(result.get_moleculeid() for result in results)

    def successful(self):
        """Held results, converged or not: the tasks that returned."""
        if self._successful is None:
            self._scan()
        return self._successful

    def converged(self):
        """Held results with a converged label, the ones a store keeps."""
        if self._converged is None:
            self._scan()
        return self._converged

    def finished(self):
        return self.successful() + self.failed_since_store

    def molecule_ids(self):
        if self._ids is None:
            self._scan()
        return set(self._ids)

    def take(self):
        """Return every held result and the names of the pieces holding them. Nothing is removed."""
        names = self.pieces()
        return list(self.iter_results(names)), names

    def iter_results(self, names):
        """Yield the results held in ``names``, reading one piece at a time."""
        for name in names:
            yield from self._read(name)['results']

    def discard(self, names):
        """Remove pieces once a shard holds them, and start counting toward the next store."""
        for name in names:
            path = self.directory / name
            if path.exists():
                path.unlink()
        self.failed_since_store = 0
        self._successful = self._converged = self._ids = self._periodic = None

    def commit(self, snapshot, properties, attributes):
        from alframework.tools.data_shards import store_current_data

        pieces = self.pieces()
        stored = store_current_data(snapshot, self.iter_results(pieces), properties,
                                    attributes={HELD_PIECES: json.dumps(pieces), **attributes},
                                    h5_options=getattr(self, 'h5_options', None), label_units=self.label_units)
        self.discard(pieces)
        return stored

    def recover_snapshot(self, snapshot):
        frames = shard_frames(snapshot)
        self.discard(shard_held_pieces(snapshot))
        return frames

    def select_snapshot(self, snapshot):
        """Legacy held pieces use the same directory for each generation."""


def shard_frames(h5path):
    """The number of stored structures in a shard."""
    with h5py.File(h5path, 'r') as handle:
        validate_label_parts(handle)
        return sum(int(group['_id'].shape[0]) for group in handle.values()
                   if isinstance(group, h5py.Group) and '_id' in group)


def shard_held_pieces(h5path):
    with h5py.File(h5path, 'r') as handle:
        value = handle.attrs.get(HELD_PIECES)
    if value is None:
        return []
    return json.loads(value.decode() if isinstance(value, bytes) else value)


def in_bootstrap(status):
    return status['current_h5_id'] == 0 and status['current_model_id'] < 0


def reconcile_status(status, master_config, held):
    """Bring a loaded status up to date with what a crash left on disk. Returns the changes, as text.

    - A shard at ``current_h5_id`` was stored but status wasn't saved: count it, discard the
      pieces it holds, and move past it.
    - A model directory at ``current_training_id`` belongs to a training that started: never
      reuse its ID.
    - Molecule IDs may have been used after status was last saved. A finite bootstrap rebuilds
      exactly the IDs it doesn't hold, which give the same structures; otherwise IDs skip ahead
      by the most one loop iteration can use.
    """
    changes = []
    while os.path.exists(master_config['h5_path'].format(status['current_h5_id'])):
        shard = master_config['h5_path'].format(status['current_h5_id'])
        frames = held.recover_snapshot(shard)
        status['lifetime_stored_labels'] = status.get('lifetime_stored_labels', 0) + frames
        status['current_h5_id'] += 1
        changes.append(f'{os.path.basename(shard)} was stored before the crash ({frames} labels); continuing after it')
    first_training = status['current_training_id']
    while os.path.exists(master_config['model_path'].format(status['current_training_id'])):
        status['current_training_id'] += 1
    if status['current_training_id'] != first_training:
        changes.append(f'training IDs {first_training}-{status["current_training_id"] - 1} were already started; '
                       f'the next training is {status["current_training_id"]}')
    if in_bootstrap(status) and master_config.get('finite_bootstrap', False):
        status['current_molecule_id'] = 0
        changes.append(f'finite bootstrap: {held.successful()} labels are held ({held.converged()} converged); '
                       'rebuilding the rest')
    else:
        margin = master_config.get('parallel_samplers', 0) + master_config.get('maximum_builder_structures', 1)
        status['current_molecule_id'] += margin
        changes.append(f'molecule IDs skip ahead by {margin}, past any used after status was saved')
    held_count = held.successful()
    if held_count:
        changes.append(f'{held_count} finished labels were held from before the crash and will be stored')
    return changes


def find_empty_directory(pattern):
    """Return the first unused index for a run's indexed path pattern.

    Args:
        pattern (str): A format string for files or directories.

    Returns:
        int: The first index whose formatted path does not exist.

    """
    curI = 0
    while os.path.exists(pattern.format(curI)):
        curI += 1

    return curI


def write_status(status_path, status):
    """Save the coordinator's status all at once, so a crash mid-write can't stop a restart."""
    write_json_atomic(status_path, status)


# The master configuration's opt-in: skip rewriting a status file that hasn't changed.
STATUS_ON_CHANGE = 'write_status_on_change'


def status_on_change_setting(master_config):
    """The checked ``write_status_on_change`` setting, ``False`` unless set."""
    value = (master_config or {}).get(STATUS_ON_CHANGE, False)
    if not isinstance(value, bool):
        raise ValueError(f'{STATUS_ON_CHANGE} must be true or false; received {value!r}.')
    return value


class StatusFile:
    """Writes the coordinator's status file, by default on every call.

    With ``only_on_change`` (the master configuration's ``write_status_on_change``), a call whose status and path
    match the last one written skips the write. Every change is still written when it's passed in, so a restart
    finds the same status; what's skipped is rewriting the same bytes, each a sync and a rename on the shared
    filesystem, on the passes where nothing happened, such as while a stage waits for its nodes.
    """

    def __init__(self, only_on_change=False):
        self.only_on_change = only_on_change
        self._written = None
        self.skipped = 0

    def write(self, status_path, status):
        """Write ``status`` to ``status_path``, unless it's skipped as unchanged; returns whether it was written."""
        written = (status_path, json.dumps(status, sort_keys=True, default=str))
        if self.only_on_change and written == self._written:
            self.skipped += 1
            return False
        write_status(status_path, status)
        self._written = written
        return True
