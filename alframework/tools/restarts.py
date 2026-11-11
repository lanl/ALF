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

from alframework.tools.files import write_bytes_atomic

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

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.failed_since_store = 0
        self._successful = None
        self._converged = None
        self._ids = None

    def pieces(self):
        return sorted(path.name for path in self.directory.glob('held-*.pkl'))

    def _read(self, name):
        with open(self.directory / name, 'rb') as handle:
            piece = pickle.load(handle)
        if piece.get('format') != FORMAT:
            raise ValueError(f'{self.directory / name} has an unknown format; it was written by another ALF version.')
        return piece

    def _scan(self):
        self._successful, self._converged, self._ids = 0, 0, set()
        for name in self.pieces():
            piece = self._read(name)
            self._successful += piece['successful']
            self._converged += sum(1 for result in piece['results'] if result.check_convergence())
            self._ids.update(result.get_moleculeid() for result in piece['results'])

    def hold(self, results, failed=0):
        """Write ``results`` as a new piece, and track failures until the next store."""
        self.failed_since_store += failed
        if not results:
            return
        existing = self.pieces()
        number = int(existing[-1][len('held-'):-len('.pkl')]) + 1 if existing else 0
        piece = dict(format=FORMAT, successful=len(results), results=list(results))
        write_bytes_atomic(self.directory / f'held-{number:08d}.pkl', pickle.dumps(piece, pickle.HIGHEST_PROTOCOL))
        if self._successful is not None:
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
        results = []
        for name in names:
            results.extend(self._read(name)['results'])
        return results, names

    def discard(self, names):
        """Remove pieces once a shard holds them, and start counting toward the next store."""
        for name in names:
            path = self.directory / name
            if path.exists():
                path.unlink()
        self.failed_since_store = 0
        self._successful = self._converged = self._ids = None


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
