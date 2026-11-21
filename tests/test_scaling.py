"""The master configuration's scaling block: one place for the settings that change how ALF runs at scale."""
import copy
import json
from types import SimpleNamespace

import pytest

from alframework.tools import config as workflow_config
from alframework.tools import scaling
from alframework.tools.capabilities import declares


def _task(*features):
    @declares(*features)
    def func():
        pass
    return SimpleNamespace(func=func)


SAMPLER_FEATURES = ('returned_metadata', 'returned_momenta', 'metadata_files', 'frame_spool')
TASKS = {'my.sampler': _task(*SAMPLER_FEATURES),
         'my.trainer': _task('shared_training_data', 'stream_final_evaluation', 'h5_read_workers'),
         'my.fine_tuner': _task('h5_read_workers'),
         'my.other_sampler': _task(),
         'my.other_trainer': _task()}

BLOCK = {'parsl_data_serializer': 'pickle', 'poll_interval': 1, 'coordinator_trim_after_snapshot': True,
         'disable_transparent_huge_pages': True, 'label_storage_batch_size': 8192,
         'label_h5_layout': {'compression': 'gzip', 'level': 1, 'shuffle': True, 'frame_chunk': 1024},
         'data_by_reference': {'directory': 'frames', 'sampler_frames': True},
         'trajectory_summary_path': 'records/trajectories.jsonl', 'coordinator_memory_sample': 30,
         'coordinator_memory_path': '/shared/records/memory.csv', 'coordinator_loop_path': 'records/loop.csv',
         'write_status_on_change': True, 'returned_metadata': 'compact', 'returned_momenta': False, 'metadata_files': 'per_task',
         'shared_training_data': True, 'stream_final_evaluation': True, 'h5_read_workers': 8}


@pytest.fixture(autouse=True)
def tasks(monkeypatch):
    monkeypatch.setattr(workflow_config, 'load_module_from_string', TASKS.__getitem__)


def _master(**settings):
    return {'master_directory': '/run/', 'sampler_task': 'my.sampler', 'ML_task': 'my.trainer', **settings}


def test_the_block_gives_each_configuration_what_setting_it_there_would(tmp_path):
    # The master settings in their usual place, read by load_config_file as the coordinator reads them.
    in_place = {name: value for name, value in BLOCK.items() if name in scaling.MASTER_SETTINGS}
    (tmp_path / 'master.json').write_text(json.dumps(dict(_master(), **in_place, master_directory=str(tmp_path))))
    expected = workflow_config.load_config_file(str(tmp_path / 'master.json'))

    master, sampler, ml = _master(scaling=copy.deepcopy(BLOCK), master_directory=expected['master_directory']), {}, {}
    assert scaling.apply_scaling(master, sampler, ml) == BLOCK
    for name in (*scaling.MASTER_SETTINGS, 'trajectory_summary_dir', 'coordinator_memory_dir', 'coordinator_loop_dir'):
        assert master[name] == expected[name], name
    assert sampler == {name: BLOCK[name] for name in scaling.SAMPLER_SETTINGS}
    assert ml == {name: BLOCK[name] for name in scaling.ML_SETTINGS}


def test_paths_resolve_against_the_master_directory_unless_absolute():
    master = _master(scaling={'coordinator_loop_path': 'records/loop.csv',
                              'coordinator_memory_path': '/shared/memory.csv'})
    scaling.apply_scaling(master, {}, {})
    assert master['coordinator_loop_path'] == '/run/records/loop.csv'
    assert master['coordinator_loop_dir'] == '/run/records/'
    assert master['coordinator_memory_path'] == '/shared/memory.csv'


def test_without_a_block_nothing_changes():
    master, sampler, ml = _master(), {'a': 1}, {'b': 2}
    before = copy.deepcopy((master, sampler, ml))
    assert scaling.apply_scaling(master, sampler, ml) is None
    assert (master, sampler, ml) == before


def test_a_setting_in_both_places_must_agree():
    master = _master(scaling={'returned_momenta': False, 'poll_interval': 1}, poll_interval=1)
    scaling.apply_scaling(master, {'returned_momenta': False}, {})
    with pytest.raises(ValueError, match="returned_momenta is True in the sampler configuration but False in scaling"):
        scaling.apply_scaling(_master(scaling={'returned_momenta': False}), {'returned_momenta': True}, {})
    with pytest.raises(ValueError, match="poll_interval is 60 in the master configuration but 1 in scaling"):
        scaling.apply_scaling(_master(scaling={'poll_interval': 1}, poll_interval=60), {}, {})


@pytest.mark.parametrize('name, place', [('batching', 'sampler or QM'), ('parallel_samplers', 'master'),
                                         ('gpus_per_member', 'the ML configuration'),
                                         ('member_nodes', 'the ML configuration')])
def test_settings_that_size_the_work_stay_in_their_own_configurations(name, place):
    with pytest.raises(ValueError, match=f'{name} sizes the work, so it belongs in .*{place}'):
        scaling.apply_scaling(_master(scaling={name: 1}), {}, {})


def test_unknown_settings_and_a_block_that_is_not_one_are_refused():
    with pytest.raises(ValueError, match="scaling has no setting 'poll_intervall'"):
        scaling.apply_scaling(_master(scaling={'poll_intervall': 1}), {}, {})
    with pytest.raises(ValueError, match='scaling must be a dictionary'):
        scaling.apply_scaling(_master(scaling=['pickle']), {}, {})


@pytest.mark.parametrize('name, task', [
    ('returned_metadata', dict(sampler_task='my.other_sampler')),
    ('metadata_files', dict(sampler_task='my.other_sampler')),
    ('shared_training_data', dict(ML_task='my.fine_tuner')),
    ('h5_read_workers', dict(ML_task='my.other_trainer')),
])
def test_a_setting_the_configured_task_does_not_read_is_refused(name, task):
    with pytest.raises(ValueError, match=f"scaling {name} is read only by tasks that declare '{name}'"):
        scaling.apply_scaling(dict(_master(scaling={name: BLOCK[name]}), **task), {}, {})


def test_fine_tuning_takes_the_reader_count():
    ml = {}
    scaling.apply_scaling(dict(_master(scaling={'h5_read_workers': 4}), ML_task='my.fine_tuner'), {}, ml)
    assert ml == {'h5_read_workers': 4}


def test_a_reload_gets_the_startup_block_without_checking_tasks_again(monkeypatch):
    startup = scaling.apply_scaling(_master(scaling={'returned_momenta': False, 'poll_interval': 1}), {}, {})
    monkeypatch.setattr(workflow_config, 'load_module_from_string', lambda name: pytest.fail('tasks loaded again'))
    reloaded, sampler = _master(scaling={'poll_interval': 5}), {}
    scaling.apply_scaling(reloaded, sampler, {}, startup, check_tasks=False)
    assert reloaded['poll_interval'] == 1 and sampler == {'returned_momenta': False}


def test_a_large_run_without_the_measured_settings_is_warned_about():
    master, sampler = _master(parallel_samplers=8192), TASKS['my.sampler']
    warning = scaling.large_run_warning(master, {}, sampler)
    for name in ('parsl_data_serializer "pickle"', 'label_storage_batch_size', 'metadata_files "per_task"',
                 'returned_metadata "compact"'):
        assert name in warning
    assert 'parallel_samplers 8,192' in warning
    done = dict(master, parsl_data_serializer='pickle', label_storage_batch_size=8192)
    assert scaling.large_run_warning(done, {'metadata_files': 'per_task', 'returned_metadata': 'compact'},
                                     sampler) is None
    assert scaling.large_run_warning(dict(master, parallel_samplers=4095), {}, sampler) is None
    assert scaling.large_run_warning(done, {}, TASKS['my.other_sampler']) is None, \
        'sampler settings only for a sampler that reads them'


def test_reports_find_monitoring_paths_in_the_block():
    assert scaling.setting({'coordinator_loop_path': 'a.csv'}, 'coordinator_loop_path') == 'a.csv'
    assert scaling.setting({'scaling': {'coordinator_loop_path': 'b.csv'}}, 'coordinator_loop_path') == 'b.csv'
    assert scaling.setting({}, 'coordinator_memory_path', 'coordinator_memory.csv') == 'coordinator_memory.csv'


def test_the_coordinator_runs_with_the_block_as_with_each_setting_in_place(tmp_path, monkeypatch):
    """The same run, with its master settings in the scaling block or each in its usual place, stores the same
    labels the same way, and the block's loop timing, trim and bounded storage all take effect."""
    from tests.helpers.driver import run_alf_driver
    from tests.test_restarts import LOOP, restart_tasks
    from alframework.tools.label_hdf5 import LABEL_PARTS

    import h5py
    settings = dict(label_storage_batch_size=2, coordinator_trim_after_snapshot=True, poll_interval=1,
                    parsl_data_serializer='pickle', coordinator_loop_path='records/loop.csv')
    results = {}
    for arm in ('in_place', 'block'):
        run_dir = tmp_path / arm
        run_dir.mkdir()
        extra = settings if arm == 'in_place' else {'scaling': settings}
        with monkeypatch.context() as patch:
            run = run_alf_driver(run_dir, patch, **restart_tasks(run_dir, []), **LOOP, finite_bootstrap=True,
                                 **extra)
        bootstrap = run_dir / 'h5' / 'data-0000.h5'
        with h5py.File(bootstrap) as handle:
            assert LABEL_PARTS in handle.attrs, 'bounded storage published the shard over parts'
            first_ids = sorted(value.decode() for group in handle.values() for value in group['_id'][()])
        assert (run_dir / 'records' / 'loop.csv').stat().st_size > 0
        assert run.status['lifetime_stored_labels'] > len(first_ids), 'a retraining stored more labels'
        # The finite bootstrap labels the same seeds either way; later shards depend on thread timing.
        results[arm] = (first_ids, max(run.sleeps))
    assert results['block'] == results['in_place'] and results['block'][1] == 1
