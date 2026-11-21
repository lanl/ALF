"""Keep the sampler wave, ordering and QM inputs intact when the handoff uses files."""

import pickle
from pathlib import Path

import pytest
from ase import Atoms

from alframework.tools import sampled_frames
from alframework.tools.batching import BatchBuffer, BatchedTaskQueue, SampledFrameQueue
from alframework.tools.labeling_coordinator import LabelingCoordinator
from alframework.tools.molecules import MoleculesObject
from alframework.tools.sampled_frames import FrameRetirement, label_sampled_frames, stage_sampled_frames
from alframework.tools.sampler_coordinator import SamplerCoordinator
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


def test_keep_files_preserves_the_spool_and_logs_its_size(tmp_path, capsys):
    from types import SimpleNamespace
    from alframework.tools.capabilities import declares

    def sampler(molecule_objects, frame_spool=None):
        pass

    undeclared = SimpleNamespace(func=sampler)
    with pytest.raises(ValueError, match="declare 'frame_spool'"):
        SamplerCoordinator(undeclared, 'fixture.sampler', {'Escut': 0.0, 'Fscut': 0.0},
                           {'master_directory': str(tmp_path), 'maximum_stored_labels': 1,
                            'data_by_reference': {'directory': 'frames', 'sampler_frames': True}})
    sampler = declares('frame_spool')(sampler)

    config = {'Escut': 0.0, 'Fscut': 0.0}
    master = {'master_directory': str(tmp_path), 'maximum_stored_labels': 1,
              'data_by_reference': {'directory': 'frames', 'sampler_frames': True, 'keep_files': True}}
    task = SimpleNamespace(func=sampler)
    sampling = SamplerCoordinator(task, 'fixture.sampler', config, master)
    assert Path(sampling.frame_directory).parent == tmp_path / 'frames'
    frame, = stage_sampled_frames([MoleculesObject(Atoms('H'), 'a')], sampling.frame_directory, 0)
    assert sampling.frame_retirement is None and sampling.harvest([frame]) == [frame]
    sampling.cleanup_frames()
    path = Path(frame.path)
    assert path.is_file()
    assert f'retained 1 batch files ({path.stat().st_size:,} bytes) at {path.parent}' in capsys.readouterr().out


@pytest.mark.parametrize('block, message', [
    ('frames', 'must be a dictionary'),
    ({'directory': 'frames'}, 'hands nothing off'),
    ({'directory': 'frames', 'sampler_frames': False, 'keep_files': True}, 'hands nothing off'),
    ({'sampler_frames': True}, 'directory must name a shared directory'),
    ({'directory': 'frames', 'sampler_frames': 'yes'}, 'sampler_frames must be true or false'),
    ({'directory': 'frames', 'sampler_frames': True, 'qm': True}, r"unknown settings \['qm'\]"),
], ids=['not_a_block', 'nothing_on', 'only_keep_files', 'no_directory', 'not_boolean', 'unknown'])
def test_data_by_reference_settings_refuse_mistakes(block, message):
    with pytest.raises(ValueError, match=message):
        sampled_frames.data_by_reference_settings({'data_by_reference': block})


@pytest.mark.parametrize('old', ['sampler_frame_spool', 'keep_sampler_frames'])
def test_the_earlier_handoff_names_point_to_data_by_reference(old):
    with pytest.raises(ValueError, match=f'{old} was renamed: use data_by_reference'):
        sampled_frames.data_by_reference_settings({old: 'frames'})


def test_data_by_reference_is_off_without_its_block_and_resolves_its_directory(tmp_path):
    assert sampled_frames.data_by_reference_settings({}) is None
    settings = sampled_frames.data_by_reference_settings(
        {'master_directory': str(tmp_path), 'data_by_reference': {'directory': 'frames', 'sampler_frames': True}})
    assert settings == {'directory': str(tmp_path / 'frames'), 'sampler_frames': True, 'keep_files': False}
    absolute = sampled_frames.data_by_reference_settings(
        {'master_directory': str(tmp_path), 'data_by_reference': {'directory': '/shared/frames', 'sampler_frames': True}})
    assert absolute['directory'] == '/shared/frames'


@pytest.mark.parametrize('durable', [True, False])
def test_retire_only_after_the_last_input_and_its_outcome_are_recorded(tmp_path, durable):
    a, b, c = [MoleculesObject(Atoms('H'), name) for name in ('a', 'b', 'c')]
    first, second = stage_sampled_frames([a, b], tmp_path, 0), stage_sampled_frames([c], tmp_path, 1)
    lifecycle = FrameRetirement()
    lifecycle.register(first + second)
    labeler = object.__new__(LabelingCoordinator)
    labeler.queue = queue = BatchedTaskQueue()
    retry = FakeTask('pending', [a], done=False)
    queue.add_task(retry, inputs=first[:1])
    queue.add_task(FakeTask('exec_done', []), inputs=first[1:])  # terminal partial output
    queue.add_task(FakeTask('exec_done', [c]), inputs=second)
    results, failed, passed, errors = labeler.take_finished_with_references(None)
    assert results == [c] and not failed
    lifecycle.labels_held(passed, errors, durable=durable)
    assert Path(second[0].path).exists()  # write_status has not completed
    lifecycle.checkpointed()
    assert Path(second[0].path).exists() != durable
    assert Path(first[0].path).exists()

    retry._status = 'failed'
    retry._done = True
    retry._exception = RuntimeError('QM exhausted its retries')
    results, failed, passed, errors = labeler.take_finished_with_references(None)
    assert not results and failed == 1
    lifecycle.labels_held(passed, errors, durable=durable)
    lifecycle.checkpointed(seconds=0)  # a saturated poll can defer unlinks
    assert Path(first[0].path).exists()
    lifecycle.checkpointed()
    assert Path(first[0].path).exists() != durable

    lifecycle.snapshot_published()
    lifecycle.checkpointed()
    assert not list(tmp_path.iterdir())


def test_a_reference_qm_task_records_reading_loading_and_labeling(tmp_path, monkeypatch):
    from alframework.tools.task_timeline import ENV, read_timeline
    gas = MoleculesObject(Atoms('He', positions=[[0, 1, 2]]), 'gas')
    [frame] = stage_sampled_frames([gas], tmp_path, 1)

    class Labeler:
        @staticmethod
        def func(molecule_objects, QM_config):
            return molecule_objects

    monkeypatch.setattr(sampled_frames, 'load_module_from_string', lambda path: Labeler)
    monkeypatch.setenv(ENV, str(tmp_path / 'timeline'))
    label_sampled_frames([frame], 'fixture.labeler', {'QM_config': {}})
    [record] = read_timeline(str(tmp_path / 'timeline'))
    assert record['stage'] == 'QM' and set(record['phases']) == {'read_frames', 'load_labeler', 'label'}
    assert set(record['phase_imports']) == set(record['phases'])
