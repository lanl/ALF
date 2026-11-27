import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from alframework.tools import label_parts
from alframework.tools.batching import BatchedTaskQueue
from alframework.tools.h5_parallel import read_shard
from alframework.tools.label_parts import HDF5HeldLabels, part_directory
from alframework.tools.restarts import open_held_labels, shard_frames
from alframework.tools.data_shards import store_current_data
from alframework.tools.label_sources import LABEL_SOURCE, read_label_sources
from tests.helpers.driver import run_alf_driver, stored_ids
from tests.helpers.driver import LOOP, restart_tasks
from tests.test_restarts import PROPERTIES, labeled


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
        list(read_shard(snapshot, workers=1))
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


def test_coordinator_trim_runs_after_each_snapshot(tmp_path, monkeypatch):
    from alframework.tools import memory

    snapshots = []

    def configure(master):
        assert master['coordinator_trim_after_snapshot'] is True
        return lambda: snapshots.append(stored_ids(tmp_path / 'h5'))

    monkeypatch.setattr(memory, 'snapshot_memory_trim', configure)
    settings = dict(LOOP, label_storage_batch_size=2, finite_bootstrap=True, coordinator_trim_after_snapshot=True)
    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, []), **settings)
    assert len(snapshots) == 2
    assert len(snapshots[0]) == settings['bootstrap_set']
    assert snapshots[1][:len(snapshots[0])] == snapshots[0]
    assert len(snapshots[1]) > len(snapshots[0])


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
    from alframework.ml_interfaces.hippynn.interface import ALFPyAniDirectoryDB

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
        assert len(list(read_shard(path, workers=1))) == len(list(read_shard(control, workers=1)))
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
    assert label_h5_options({}) == {}, 'without a layout, labels are written as before'


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


def test_the_configured_reader_count_comes_before_the_environment(tmp_path, monkeypatch):
    from alframework.tools import h5_parallel
    snapshot = str(_snapshot(tmp_path))
    monkeypatch.setenv(h5_parallel.READ_WORKERS, '1')
    assert h5_parallel.workers_for(snapshot, 4) == 1
    assert h5_parallel.workers_for(snapshot, 4, requested=3) == 3
    assert h5_parallel.workers_for(snapshot, 2, requested=3) == 2, "never more readers than groups"
    for wrong in (0, True, 2.5, '4'):
        with pytest.raises(ValueError, match='h5_read_workers must be a positive integer'):
            h5_parallel.workers_for(snapshot, 4, requested=wrong)


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


def test_label_units_are_recorded_on_shards_and_checked_as_one_inventory(tmp_path):
    from alframework.tools.label_hdf5 import LABEL_UNITS, PROPERTY_SCHEMA, SCHEMA_VERSION, validate_training_units

    ordinary = tmp_path / 'ordinary.h5'
    store_current_data(str(ordinary), [labeled('a')], PROPERTIES, label_units={'energy': 'eV'})
    snapshot = tmp_path / 'snapshot.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=1, label_units={'energy': 'Hartree'})
    held.hold([labeled('b')])
    assert held.commit(snapshot, PROPERTIES, {}) == 1
    for path, unit in ((ordinary, 'eV'), (snapshot, 'Hartree')):
        with h5py.File(path, 'r') as handle:
            assert handle.attrs[SCHEMA_VERSION] == 1
            assert json.loads(handle.attrs[PROPERTY_SCHEMA]) == PROPERTIES
            assert json.loads(handle.attrs[LABEL_UNITS]) == {'energy': unit}
    with pytest.raises(ValueError, match='declares units'):
        validate_training_units([ordinary, snapshot], ['energy'])
    legacy = tmp_path / 'legacy.h5'
    with h5py.File(legacy, 'w'):
        pass
    with pytest.warns(UserWarning, match='mix declared and unknown'):
        validate_training_units([legacy, ordinary], ['energy'])


def test_a_reader_never_sees_a_snapshot_that_is_still_being_written(tmp_path, monkeypatch):
    """The snapshot appears under its name only once complete; a publish that fails leaves no snapshot at all."""
    import contextlib

    labels = [labeled(f'mol-{i}', energy=-i) for i in range(3)]
    real, seen = label_parts.replaced_atomically, []

    @contextlib.contextmanager
    def watched(path, **options):
        with real(path, **options) as partial:
            yield partial
            seen.append((Path(path).exists(), Path(partial).exists()))  # written, not yet renamed

    monkeypatch.setattr(label_parts, 'replaced_atomically', watched)
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=2)
    held.hold(labels)
    assert held.commit(snapshot, PROPERTIES, {}) == 3
    assert (False, True) in seen, 'while the snapshot was written, only its partial file existed'
    assert sorted(read_shard(str(snapshot))) and snapshot.exists()

    failing = tmp_path / 'failing' / 'data-0000.h5'
    held = HDF5HeldLabels(failing, PROPERTIES, batch_size=2)
    held.hold(labels)
    monkeypatch.setattr(label_parts, '_write_virtual_columns', lambda *_: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        held.commit(failing, PROPERTIES, {})
    assert not failing.exists() and not failing.with_name(failing.name + '.partial').exists()
