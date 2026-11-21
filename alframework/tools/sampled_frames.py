"""Spool finished sampler batches and transfer their frame references to QM."""

import os
import pickle
import time
from collections import Counter, deque
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Optional

from alframework.tools.config import load_module_from_string
from alframework.tools.task_timeline import phase, timed
from alframework.tools.trajectory_summary import trajectory_row

# The master configuration's opt-in handoff of data by reference through shared files, and what it can hand off.
DATA_BY_REFERENCE = 'data_by_reference'
HANDOFFS = ('sampler_frames', 'labels')
REFERENCE_SETTINGS = ('directory', *HANDOFFS, 'keep_files', 'merge_workers')
RENAMED = {'sampler_frame_spool': 'data_by_reference with "directory" and "sampler_frames": true',
           'keep_sampler_frames': 'data_by_reference "keep_files"'}


def data_by_reference_settings(master_config):
    """The checked ``data_by_reference`` block, or ``None`` when the run doesn't hand data off by reference.

    ``{"directory": <shared path>, "sampler_frames": true, "labels": true, "keep_files": false}``: with
    ``sampler_frames``, each sampler task writes its eligible frames to a file in ``directory`` and returns
    references, which QM workers resolve. ``directory`` is relative to ``master_directory`` unless absolute, and
    every worker that writes or reads the files must see it at the same path; only ``sampler_frames`` needs it.
    With ``labels``, each QM task writes
    its labels as an HDF5 part beside the label store and returns a receipt, and ``merge_workers`` processes
    (default 16) merge the parts into parts of ``label_storage_batch_size`` before each snapshot is published.
    ``keep_files`` keeps every file after the run, for debugging. Each handoff is off unless set to ``true``, and
    a block that turns none on is refused.
    """
    master_config = master_config or {}
    for old, new in RENAMED.items():
        if old in master_config:
            raise ValueError(f'{old} was renamed: use {new}.')
    block = master_config.get(DATA_BY_REFERENCE)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f'{DATA_BY_REFERENCE} must be a dictionary with keys from {list(REFERENCE_SETTINGS)}.')
    unknown = sorted(set(block) - set(REFERENCE_SETTINGS))
    if unknown:
        raise ValueError(f'{DATA_BY_REFERENCE} has unknown settings {unknown}; it takes {list(REFERENCE_SETTINGS)}.')
    directory = block.get('directory')
    if (block.get('sampler_frames') or directory is not None) and (not isinstance(directory, str)
                                                                    or not directory.strip()):
        raise ValueError(f'{DATA_BY_REFERENCE}.directory must name a shared directory.')
    for key in (*HANDOFFS, 'keep_files'):
        if not isinstance(block.get(key, False), bool):
            raise ValueError(f'{DATA_BY_REFERENCE}.{key} must be true or false.')
    if not any(block.get(key, False) for key in HANDOFFS):
        raise ValueError(f'{DATA_BY_REFERENCE} hands nothing off: set one of {list(HANDOFFS)} to true.')
    from alframework.tools.label_parts import MERGE_WORKERS
    workers = block.get('merge_workers', MERGE_WORKERS)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f'{DATA_BY_REFERENCE}.merge_workers must be a positive integer.')
    if block.get('labels', False) and master_config.get('label_storage_batch_size') is None and \
            'label_storage_batch_size' not in (master_config.get('scaling') or {}):
        raise ValueError(f'{DATA_BY_REFERENCE} labels needs bounded label storage: set label_storage_batch_size.')
    return dict(directory=str(Path(master_config.get('master_directory') or '.') / directory) if directory else None,
                **{key: block.get(key, False) for key in (*HANDOFFS, 'keep_files')}, merge_workers=workers)


class SampledFrame(NamedTuple):
    """A scalar trajectory summary and, for an eligible frame, its location in a worker file."""

    batch: int
    index: int
    path: Optional[str]
    pbc: Optional[tuple]
    summary: dict

    def get_moleculeid(self):
        return self.summary['id']


class FrameRetirement:
    """Track references until the coordinator has recorded their terminal QM outcomes."""

    def __init__(self):
        self._pending = Counter()
        self._held = Counter()
        self._checkpoint_ready = Counter()
        self._unlink_ready = deque()
        self._warned = False

    def register(self, frames):
        self._pending.update(frame.path for frame in frames if frame.path is not None)

    def labels_held(self, succeeded, failed, *, durable):
        self._checkpoint_ready.update(failed)
        (self._checkpoint_ready if durable else self._held).update(succeeded)

    def snapshot_published(self):
        self._checkpoint_ready.update(self._held)
        self._held.clear()

    def checkpointed(self, *, seconds=1.0):
        for path, count in self._checkpoint_ready.items():
            left = self._pending[path] - count
            if left < 0:
                raise RuntimeError(f'More QM inputs retired than registered for {path}.')
            if left:
                self._pending[path] = left
            else:
                del self._pending[path]
                self._unlink_ready.append(path)
        self._checkpoint_ready.clear()

        started = time.monotonic()
        removed = 0
        for _ in range(len(self._unlink_ready)):
            if time.monotonic() - started >= seconds:
                break
            path = self._unlink_ready.popleft()
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                self._unlink_ready.append(path)
                if not self._warned:
                    print(f'Could not remove sampler frame file {path}: {exc}', flush=True)
                    self._warned = True
            else:
                removed += 1
        if removed:
            print(f'Sampler spool: retired {removed} batch files after QM checkpoint.', flush=True)


def stage_sampled_frames(molecules, directory, batch):
    """Return references only after the worker has finished writing its unique batch file."""
    eligible = [molecule for molecule in molecules if molecule.get_atoms() is not None]
    path = None
    if eligible:
        try:
            with NamedTemporaryFile(dir=directory, prefix='batch-', suffix='.pkl', delete=False) as stream:
                path = stream.name
                pickle.dump(eligible, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            if path is not None:
                Path(path).unlink(missing_ok=True)
            raise

    frames = []
    index = 0
    for molecule in molecules:
        summary = trajectory_row(molecule.get_moleculeid(), molecule.get_metadata())
        atoms = molecule.get_atoms()
        if atoms is None:
            frames.append(SampledFrame(batch, index, None, None, summary))
        else:
            frames.append(SampledFrame(batch, index, path, tuple(map(bool, atoms.get_pbc())), summary))
            index += 1
    return frames


@timed('QM')
def label_sampled_frames(molecule_objects, task_path, task_inputs, label_parts=None):
    """Resolve frames on the native QM worker, then call the configured labeler.

    With ``label_parts`` (``data_by_reference`` labels), the labels are written as one part and a
    :class:`~alframework.tools.label_parts.LabelReceipt` is returned instead
    (:func:`~alframework.tools.label_parts.write_label_part`, given ``label_parts`` as keywords).
    """
    from parsl.app.errors import RemoteExceptionWrapper

    batches = {}
    inputs = []
    with phase('read_frames'):
        for frame in molecule_objects:
            if not isinstance(frame, SampledFrame):
                inputs.append(frame)
                continue
            if frame.path not in batches:
                with open(frame.path, 'rb') as stream:
                    batches[frame.path] = pickle.load(stream)
            molecule = batches[frame.path][frame.index]
            if molecule.get_moleculeid() != frame.get_moleculeid():
                raise ValueError(f'Sampled frame ID does not match the receipt for {frame.get_moleculeid()}.')
            inputs.append(molecule)
    with phase('load_labeler'):
        labeler = load_module_from_string(task_path).func
    with phase('label'):
        result = labeler(molecule_objects=inputs, **task_inputs)
    if isinstance(result, RemoteExceptionWrapper):
        result.reraise()
    if label_parts is not None:
        from alframework.tools.label_parts import write_label_part
        with phase('write_labels'):
            return write_label_part(result, **label_parts)
    return result
