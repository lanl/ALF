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


def test_hippynn_loader_reads_the_same_arrays_from_a_virtual_shard(tmp_path):
    pytest.importorskip('hippynn')
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB

    labels = [labeled(f'mol-{i}', energy=-i) for i in range(3)]
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
