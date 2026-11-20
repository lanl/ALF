"""Tasks declare the settings they read, and every role's task can check its configuration at startup."""
import functools
from types import SimpleNamespace

import pytest

from alframework.tools.capabilities import declared, declares, require, supports, validate_task_config


def test_a_declaration_is_kept_by_wrappers_and_parsl_apps():
    from parsl import python_app

    @declares('returned_metadata')
    @declares('metadata_files')
    def sampler(molecule_objects):
        return molecule_objects

    @functools.wraps(sampler)
    def wrapper(*args, **kwargs):
        return sampler(*args, **kwargs)

    app = python_app(executors=['alf_sampler_executor'])(wrapper)
    for task in (sampler, wrapper, app):
        assert declared(task) == {'returned_metadata', 'metadata_files'}
        assert supports(task, 'metadata_files') and not supports(task, 'frame_spool')


def test_an_undeclared_feature_is_refused_with_how_to_declare_it():
    with pytest.raises(ValueError, match=r"returned_momenta is read only by tasks that declare 'returned_momenta', "
                                         r"and my\.sampler does not.*declares\('returned_momenta'\)"):
        require(SimpleNamespace(func=lambda: None), 'my.sampler', 'returned_momenta')
    with pytest.raises(ValueError, match='setting name'):
        declares('')


def test_the_built_in_tasks_declare_what_they_read():
    from alframework.ml_interfaces import fine_tuning, hippynn_interface
    from alframework.samplers import batched_mlmd, mlmd_sampling

    assert declared(batched_mlmd.mlmd_trajectories_task) == {'returned_metadata', 'returned_momenta', 'metadata_files',
                                                            'frame_spool'}
    assert declared(mlmd_sampling.simple_mlmd_sampling_task) == set()
    assert declared(hippynn_interface.train_HIPPYNN_ensemble_multinode_task) == {
        'shared_training_data', 'stream_final_evaluation', 'h5_read_workers'}
    assert declared(hippynn_interface.train_HIPPYNN_ensemble_task) == {'h5_read_workers'}
    assert declared(fine_tuning.fine_tune_ensemble_task) == {'h5_read_workers'}


def test_validate_task_config_calls_the_hook_when_there_is_one():
    seen = []

    def task():
        pass
    validate_task_config(task, {'a': 1}, {})  # no hook, nothing to check
    task.validate_config = lambda config, master: seen.append((config, master))
    validate_task_config(SimpleNamespace(func=task), {'a': 1}, {'b': 2})
    assert seen == [({'a': 1}, {'b': 2})]


@pytest.mark.parametrize('role', ['builder', 'sampler'])
def test_the_coordinator_checks_builder_and_sampler_configs_before_the_bootstrap(tmp_path, monkeypatch, role):
    from tests.helpers.driver import run_alf_driver
    from tests.test_restarts import LOOP, restart_tasks

    tasks = restart_tasks(tmp_path, [])

    def refuse(config, master_config):
        raise ValueError(f'{role} config refused')
    tasks[role].func.validate_config = refuse
    try:
        with pytest.raises(ValueError, match=f'{role} config refused'):
            run_alf_driver(tmp_path, monkeypatch, **tasks, **LOOP)
    finally:
        del tasks[role].func.validate_config
    assert not list((tmp_path / 'h5').glob('*.h5')), 'nothing was labeled'
