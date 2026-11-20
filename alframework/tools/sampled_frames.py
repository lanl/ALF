"""Spool finished sampler batches and transfer their frame references to QM."""

import os
import pickle
import time
from collections import Counter, deque
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Optional

from alframework.tools.config import load_module_from_string
from alframework.tools.task_timeline import timed
from alframework.tools.trajectory_summary import trajectory_row


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
def label_sampled_frames(molecule_objects, task_path, task_inputs):
    """Resolve frames on the native QM worker, then call the configured labeler."""
    from parsl.app.errors import RemoteExceptionWrapper

    batches = {}
    inputs = []
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
    result = load_module_from_string(task_path).func(molecule_objects=inputs, **task_inputs)
    if isinstance(result, RemoteExceptionWrapper):
        result.reraise()
    return result
