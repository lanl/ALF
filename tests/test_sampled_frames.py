"""Keep the sampler wave, ordering and QM inputs intact when the handoff uses files."""

import pickle
from pathlib import Path

import pytest
from ase import Atoms

from alframework.tools import sampled_frames
from alframework.tools.batching import BatchBuffer, SampledFrameQueue
from alframework.tools.labeling_coordinator import LabelingCoordinator
from alframework.tools.molecules import MoleculesObject
from alframework.tools.sampled_frames import label_sampled_frames, stage_sampled_frames
from tests.helpers.fakes import FakeTask


def test_frame_receipts_keep_ids_metadata_and_periodicity(tmp_path, monkeypatch):
    periodic = MoleculesObject(Atoms('H', cell=[5, 5, 5], pbc=True), 'periodic')
    gas = MoleculesObject(Atoms('He', positions=[[0, 1, 2]]), 'gas')
    gas.update_metadata({'private': 'metadata seen by the native QM task'})
    rejected = MoleculesObject(Atoms('H'), 'rejected')
    rejected.update_atoms(None)
    rejected.update_metadata({'trajectory_outcome': 'rejected'})
    a, b, c = pickle.loads(pickle.dumps(stage_sampled_frames([periodic, rejected, gas], tmp_path, 1)))
    assert a.path == c.path and (a.index, c.index) == (0, 1)
    assert b.path is None and b.summary['outcome'] == 'rejected'
    buffer = BatchBuffer(2)
    buffer.extend([a, c])
    assert buffer.pop_ready(flush=True) == [[a], [c]]

    class Labeler:
        @staticmethod
        def func(molecule_objects, QM_config):
            assert QM_config == {'test': True}
            return molecule_objects

    monkeypatch.setattr(sampled_frames, 'load_module_from_string', lambda path: Labeler)
    for _ in range(2):  # A retry can still read the complete staged batch.
        result = label_sampled_frames([c, periodic], 'fixture.labeler', {'QM_config': {'test': True}})
        assert result[0].get_moleculeid() == 'gas'
        assert result[0].get_atoms() == gas.get_atoms()
        assert result[0].get_metadata() == gas.get_metadata()
        assert result[1] is periodic
    assert Path(c.path).is_file()


def test_harvesting_ready_receipts_preserves_wave_count_and_order(tmp_path):
    first = stage_sampled_frames([MoleculesObject(Atoms('H'), 'first')], tmp_path, 0)
    second = stage_sampled_frames([MoleculesObject(Atoms('H'), 'second')], tmp_path, 1)
    # A later batch finishes while the earlier one's full input is still in flight.
    ready = FakeTask('exec_done', second)
    queued = FakeTask('pending', first, done=False)
    queue = SampledFrameQueue(BatchBuffer(2))
    queue.add_task(queued, width=1)
    queue.add_task(ready, width=1)
    queue.collect()
    assert queue.task_list == [queued]
    assert queue.get_number() == queue.get_task_number() == 2
    assert queue.get_completed_number() == 1
    queued._status = 'exec_done'
    queued._done = True
    frames, failures = queue.get_task_results()
    assert failures == 0
    assert [frame.get_moleculeid() for frame in frames] == ['first', 'second']
    assert queue.get_number() == queue.get_completed_number() == 0


def test_early_harvest_keeps_failed_inputs_and_width():
    molecule = MoleculesObject(Atoms('H'), 'failed')
    queue = SampledFrameQueue(BatchBuffer(2))
    queue.add_task(FakeTask('failed', exception=RuntimeError('worker failed')), width=1, inputs=[molecule])
    queue.collect()
    assert queue.get_number() == queue.get_completed_number() == queue.get_failed_number() == 1
    assert queue.get_exec_done_number() == 0
    assert queue.get_task_results() == ([], 1)
    assert queue.get_failed_number() == 0
    assert queue.failed_structures == 1
    assert queue.failed_inputs == [molecule]


def test_unpublished_batch_is_removed_on_write_failure(tmp_path, monkeypatch):
    def fail(descriptor):
        raise OSError('test fsync failure')

    monkeypatch.setattr(sampled_frames.os, 'fsync', fail)
    with pytest.raises(OSError, match='fsync failure'):
        stage_sampled_frames([MoleculesObject(Atoms('H'), 'first')], tmp_path, 0)
    assert list(tmp_path.iterdir()) == []


def test_reference_handoff_requires_native_qm():
    with pytest.raises(ValueError, match='requires native QM batching'):
        LabelingCoordinator(None, 'fixture.labeler', {}, sampled_frame_handoff=True)
