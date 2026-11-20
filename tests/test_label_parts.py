import json
import shutil

import h5py
import numpy as np
import pytest

from alframework.tools import label_parts, pyanitools
from alframework.tools.batching import BatchedTaskQueue
from alframework.tools.label_parts import HDF5HeldLabels, part_directory
from alframework.tools.restarts import open_held_labels, shard_frames
from alframework.tools.data_shards import store_current_data
from alframework.tools.label_sources import LABEL_SOURCE, read_label_sources
from tests.helpers.driver import run_alf_driver, stored_ids
from tests.test_restarts import LOOP, PROPERTIES, labeled, restart_tasks


def test_snapshot_matches_legacy_layout_and_moves_with_its_parts(tmp_path):
    samples = [labeled('a'), labeled('longer-id'), labeled('last')]
    samples[0].update_metadata({LABEL_SOURCE: {'teacher': 'first'}, 'mult': 2})
    samples[1].update_metadata({LABEL_SOURCE: {'teacher': 'second'}, 'charge': -1})
    samples[1].get_atoms().set_chemical_symbols('H3')
    samples[2].update_metadata({LABEL_SOURCE: {'teacher': 'first'}})
    expected = tmp_path / 'control.h5'
    assert store_current_data(str(expected), samples, PROPERTIES) == 3

    directory = tmp_path / 'source'
    snapshot = directory / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=1)
    held.hold(samples)
    assert len(held.pieces()) == 3
    assert list(directory.glob('*.h5')) == []
    assert held.commit(snapshot, PROPERTIES, {}) == 3
    relocated = tmp_path / 'relocated'
    shutil.move(directory, relocated)
    snapshot = relocated / snapshot.name
    assert shard_frames(snapshot) == 3
    assert read_label_sources(snapshot) == read_label_sources(expected)
    with h5py.File(snapshot) as after, h5py.File(expected) as before:
        assert set(after) == set(before)
        for formula, group in before.items():
            assert set(after[formula]) == set(group)
            for name, dataset in group.items():
                assert after[formula][name].dtype == dataset.dtype
                np.testing.assert_array_equal(after[formula][name][:], dataset[:])

    next(part_directory(snapshot).glob('part-*.h5')).unlink()
    with pytest.raises(FileNotFoundError, match='required label part'):
        pyanitools.anidataloader(str(snapshot))
    with pytest.raises(FileNotFoundError, match='required label part'):
        shard_frames(snapshot)


@pytest.mark.parametrize('published', [False, True])
def test_recover_unacknowledged_part(tmp_path, monkeypatch, published):
    snapshot = tmp_path / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=2)
    held.hold([labeled('first')])
    rejected = labeled('rejected')
    rejected.set_converged_flag(False)
    second = [labeled('second'), rejected]
    if published:
        def fail_after_rename(path):
            raise OSError('interrupted')

        monkeypatch.setattr(label_parts, 'fsync_file', fail_after_rename)
    else:
        def fail_before_rename(*args, **kwargs):
            raise OSError('interrupted')

        monkeypatch.setattr(label_parts, 'store_current_data', fail_before_rename)

    with pytest.raises(OSError, match='interrupted'):
        held.hold(second)
    monkeypatch.undo()
    (part_directory(snapshot) / 'part-99999999.h5.partial').write_bytes(b'interrupted')
    recovered = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=1)
    if not published:
        assert recovered.molecule_ids() == {'first'}
        recovered.hold(second)
    assert (recovered.successful(), recovered.converged()) == (3, 2)
    assert recovered.molecule_ids() == {'first', 'second', 'rejected'}
    assert recovered.commit(snapshot, PROPERTIES, {}) == 2
    assert shard_frames(snapshot) == 2


def test_restart_after_snapshot_publication_with_real_coordinator(tmp_path, monkeypatch):
    original = HDF5HeldLabels.commit

    def publish_then_crash(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError('interrupted after snapshot publication')

    settings = dict(LOOP, label_storage_batch_size=2, finite_bootstrap=True)
    built = []
    with monkeypatch.context() as patch:
        patch.setattr(HDF5HeldLabels, 'commit', publish_then_crash)
        with pytest.raises(OSError, match='after snapshot publication'):
            run_alf_driver(tmp_path, patch, **restart_tasks(tmp_path, built), **settings)
    directory = tmp_path / 'h5'
    bootstrap = stored_ids(directory)
    assert len(bootstrap) == settings['bootstrap_set']
    status_file = tmp_path / 'status.json'
    assert json.loads(status_file.read_text())['current_h5_id'] == 0
    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **settings)
    all_ids = stored_ids(directory)
    status = json.loads(status_file.read_text())
    assert status['current_h5_id'] == 2
    assert status['lifetime_stored_labels'] == len(set(all_ids)) == len(all_ids)
    assert all_ids[:len(bootstrap)] == bootstrap


def test_queue_harvest_bounds_structures_and_accounts_for_failures():
    class Task:
        def __init__(self, name, width, failed=False):
            self.name = name
            self.width = width
            self.failed = failed

        def task_status(self):
            return 'failed' if self.failed else 'exec_done'

        def result(self):
            if self.failed:
                raise RuntimeError('simulated task failure')
            return [labeled(self.name)] * self.width

    queue = BatchedTaskQueue()
    for task in (Task('oversized', 4), Task('failure', 2, failed=True), Task('later', 2)):
        queue.add_task(task, width=task.width)
    labels, failures = queue.get_task_results(max_items=3)
    assert len(labels) == 4 and failures == 0
    assert queue.get_completed_number() == 4
    labels, failures = queue.get_task_results(max_items=3)
    assert labels == [] and failures == 1 and queue.failed_structures == 2
    labels, failures = queue.get_task_results(max_items=3)
    assert len(labels) == 2 and failures == 0 and queue.get_number() == 0


def test_resume_mode_cannot_hide_pending_labels(tmp_path):
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=1)
    held.hold([labeled()])
    config = {'h5_dir': str(snapshot.parent), 'h5_path': str(snapshot.parent / 'data-{:04d}.h5')}
    with pytest.raises(ValueError, match='unfinished HDF5 generation'):
        open_held_labels(config, 0)


@pytest.mark.parametrize('workers', [1, 2])
def test_hippynn_loader_reads_the_same_arrays_from_a_virtual_shard(tmp_path, monkeypatch, workers):
    pytest.importorskip('hippynn')
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB

    labels = [labeled(f'mol-{i}', energy=-i) for i in range(3)]
    if workers == 2:
        labels[-1].get_atoms().set_chemical_symbols('H3')
    monkeypatch.setenv('ALF_H5_READ_WORKERS', str(workers))
    snapshot = tmp_path / 'parts' / 'data.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=2)
    held.hold(labels)
    held.commit(snapshot, PROPERTIES, {})
    control = tmp_path / 'control'
    control.mkdir()
    store_current_data(str(control / 'data.h5'), labels, PROPERTIES)
    options = dict(inputs=['species', 'coordinates'], targets=['energy', 'forces'], seed=0, quiet=True)
    before = ALFPyAniDirectoryDB(str(control), **options)
    after = ALFPyAniDirectoryDB(str(snapshot.parent), **options)
    assert set(before.arr_dict) == set(after.arr_dict)
    for key, value in before.arr_dict.items():
        np.testing.assert_array_equal(value.numpy(), after.arr_dict[key].numpy())

    next(part_directory(snapshot).glob('part-*.h5')).unlink()
    with pytest.raises(FileNotFoundError, match='required label part'):
        ALFPyAniDirectoryDB(str(snapshot.parent), **options)


LAYOUTS = [{'compression': 'lzf', 'shuffle': True, 'frame_chunk': 2},
           {'compression': 'gzip', 'level': 1, 'shuffle': True, 'frame_chunk': 1024},
           {'compression': None, 'frame_chunk': 2}]


def _datasets(path):
    with h5py.File(path) as handle:
        return {(formula, name): dataset[()] for formula, group in handle.items() for name, dataset in group.items()}


@pytest.mark.parametrize('layout', LAYOUTS)
def test_a_label_layout_stores_exactly_the_default_values(tmp_path, layout):
    """label_h5_layout changes only compression and chunking: every dataset, dtype and value is the same, from
    both the legacy store and HDF5 parts, and the readers ALF and HIPPYNN use read them unchanged."""
    from alframework.tools.restarts import HeldLabels, label_h5_options
    samples = [labeled('a'), labeled('longer-id'), labeled('last')]
    samples[1].get_atoms().set_chemical_symbols('H3')
    control = tmp_path / 'control.h5'
    store_current_data(str(control), samples, PROPERTIES)
    options = label_h5_options({'label_h5_layout': layout})
    legacy = HeldLabels(tmp_path / 'held', h5_options=options)
    legacy.hold(samples)
    legacy.commit(str(tmp_path / 'legacy.h5'), PROPERTIES, {})
    parts = HDF5HeldLabels(tmp_path / 'parts' / 'data-0000.h5', PROPERTIES, batch_size=2, h5_options=options)
    parts.hold(samples)
    parts.commit(tmp_path / 'parts' / 'data-0000.h5', PROPERTIES, {})
    expected = _datasets(control)
    for path in (tmp_path / 'legacy.h5', tmp_path / 'parts' / 'data-0000.h5'):
        found = _datasets(path)
        assert found.keys() == expected.keys()
        for key, value in expected.items():
            assert found[key].dtype == value.dtype and np.array_equal(found[key], value), (path.name, key)
        assert shard_frames(path) == 3
        assert len(list(pyanitools.anidataloader(str(path)))) == len(list(pyanitools.anidataloader(str(control))))
    with h5py.File(tmp_path / 'legacy.h5') as handle:
        coordinates = next(iter(handle.values()))['coordinates']
        assert coordinates.compression == layout['compression']
        assert coordinates.shuffle == layout.get('shuffle', False)


@pytest.mark.parametrize('layout, message', [({'compression': 'zstd'}, 'compression'),
                                             ({'level': 12}, 'level'), ({'frame_chunk': 0}, 'frame_chunk'),
                                             ({'shuffle': 'yes'}, 'shuffle'), ({'chunks': 4}, 'keys')])
def test_a_label_layout_rejects_unknown_settings(layout, message):
    from alframework.tools.restarts import label_h5_options
    with pytest.raises(ValueError, match=message):
        label_h5_options({'label_h5_layout': layout})


def test_without_a_layout_labels_are_written_as_before():
    from alframework.tools.restarts import label_h5_options
    assert label_h5_options({}) == {}


class _Queue:
    """Finished labels waiting in a QM queue, taken a part at a time."""

    def __init__(self, waiting, batch):
        self.waiting, self.batch, self.parts = waiting, batch, 0

    def take_one_part(self):
        self.waiting -= min(self.batch, self.waiting) if self.batch else self.waiting
        self.parts += 1


def test_a_pass_writes_every_full_part_that_is_waiting():
    from alframework.tools.label_parts import hold_ready
    queue = _Queue(waiting=8 * 4 + 3, batch=8)
    assert hold_ready(queue.take_one_part, lambda: queue.waiting, 8) == 4
    assert queue.waiting == 3, "the pass stops at a part that isn't full; the next pass or a drain takes it"
    assert hold_ready(queue.take_one_part, lambda: queue.waiting, 8, drain=True) == 1 and queue.waiting == 0


def test_a_pass_stops_writing_parts_after_its_time_limit():
    from alframework.tools.label_parts import hold_ready
    queue = _Queue(waiting=8 * 100, batch=8)
    ticks = iter(range(1000))
    written = hold_ready(queue.take_one_part, lambda: queue.waiting, 8, seconds=3, clock=lambda: next(ticks))
    assert written == 3 and queue.waiting == 8 * 97, "one part per clock tick, three ticks allowed"
    written = hold_ready(queue.take_one_part, lambda: queue.waiting, 8, drain=True, seconds=3,
                         clock=lambda: next(ticks))
    assert written == 97 and queue.waiting == 0, "a drain ignores the limit"


def test_without_bounded_storage_one_call_holds_everything():
    from alframework.tools.label_parts import hold_ready
    queue = _Queue(waiting=1000, batch=None)
    assert hold_ready(queue.take_one_part, lambda: queue.waiting, None) == 1 and queue.waiting == 0


def _snapshot(tmp_path, count=7, batch_size=2):
    labels = [labeled(f'mol-{i}', energy=-i) for i in range(count)]
    snapshot = tmp_path / 'parts' / 'data.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=batch_size)
    held.hold(labels)
    held.commit(snapshot, PROPERTIES, {})
    return snapshot


def test_a_snapshot_reads_the_same_from_its_parts_as_through_hdf5(tmp_path):
    from alframework.tools import h5_parallel
    snapshot = _snapshot(tmp_path)
    for path in h5_parallel.group_paths(str(snapshot)):
        direct = h5_parallel.read_group(str(snapshot), path)
        with h5py.File(snapshot) as handle:
            assert any(dataset.is_virtual for dataset in handle[path].values()), "the test needs virtual datasets"
            for key, dataset in handle[path].items():
                reference = dataset[()]
                if reference.dtype.kind == 'S' and reference.ndim == 1:
                    assert direct[key] == [item.decode('ascii') for item in reference]
                else:
                    assert direct[key].dtype == reference.dtype
                    np.testing.assert_array_equal(direct[key], reference)


def test_selected_snapshot_columns_read_in_part_order_and_preserve_fill_values(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from alframework.tools import h5_parallel

    snapshot = tmp_path / 'data.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=1)
    labels = [labeled(f'mol-{i}', energy=-i) for i in range(5)]
    labels[2].update_metadata({'charge': -1})
    held.hold(labels)
    held.commit(snapshot, PROPERTIES, {})
    path, = h5_parallel.group_paths(str(snapshot))
    keys = {'energy', '_id', 'total_charge'}
    with h5py.File(snapshot) as handle:
        expected = {key: handle[path][key][()] for key in keys}
    np.testing.assert_array_equal(expected['total_charge'], [0, 0, -1, 0, 0])

    open_parts = []
    opener = h5py.File

    @contextmanager
    def track(name, *args, **kwargs):
        with opener(name, *args, **kwargs) as handle:
            if name != str(snapshot):
                assert not any(part.id.valid for part in open_parts)
                open_parts.append(handle)
            yield handle

    monkeypatch.setattr(h5py, 'File', track)
    group, = h5_parallel.read_groups(str(snapshot), [path], 1, keys)
    assert len(open_parts) == 5
    assert set(group) == keys | {'path'}
    assert group['_id'] == [item.decode() for item in expected['_id']]
    for key in ('energy', 'total_charge'):
        assert group[key].dtype == expected[key].dtype
        np.testing.assert_array_equal(group[key], expected[key])


def test_a_snapshot_is_sized_by_its_parts_when_choosing_readers(tmp_path, monkeypatch):
    from alframework.tools import h5_parallel
    snapshot = _snapshot(tmp_path)
    parts = sum(path.stat().st_size for path in part_directory(snapshot).glob('part-*.h5'))
    assert h5_parallel.stored_bytes(str(snapshot)) == snapshot.stat().st_size + parts
    monkeypatch.delenv(h5_parallel.READ_WORKERS, raising=False)
    monkeypatch.setattr(h5_parallel, 'PARALLEL_MIN_BYTES', snapshot.stat().st_size + 1)
    assert h5_parallel.workers_for(str(snapshot), 4) > 1, "the parts, not the small snapshot, make it large"


def test_a_mapping_alf_does_not_write_falls_back_to_hdf5(tmp_path):
    from alframework.tools import h5_parallel
    source = tmp_path / 'source.h5'
    with h5py.File(source, 'w') as handle:
        handle['x'] = np.arange(12.0).reshape(4, 3)
    layout = h5py.VirtualLayout(shape=(2, 3), dtype='f8')
    layout[:, :] = h5py.VirtualSource(str(source), 'x', shape=(4, 3))[1:3, :]  # part of the source, not all of it
    with h5py.File(tmp_path / 'view.h5', 'w') as handle:
        handle.create_group('g').create_virtual_dataset('x', layout)
        handle['g/species'] = np.array([b'H'])
    data = h5_parallel.read_group(str(tmp_path / 'view.h5'), '/g')
    np.testing.assert_array_equal(data['x'], np.arange(12.0).reshape(4, 3)[1:3])
