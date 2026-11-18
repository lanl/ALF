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


class HeldLabels:
    """Finished QM results that aren't stored yet, kept as atomically written pieces.

    Counts use the QM queue's units (tasks, or structures for batched labeling).
    Retraining and a finite bootstrap count converged labels; a bootstrap that isn't finite, and
    ``maximum_completed_QM``, count every result that returned, converged or not. ``finished()`` also
    includes failed tasks, which hold no results and are counted only for this process.
    """

    PIECE_SIZE = 1000

    def __init__(self, directory):
        self.directory = Path(directory)
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


def shard_frames(h5path):
    """The number of stored structures in a shard."""
    with h5py.File(h5path, 'r') as handle:
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
        frames = shard_frames(shard)
        held.discard(shard_held_pieces(shard))
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
