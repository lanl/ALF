"""alf_task: one declaration gives a task its role's executor, timeline record and failure cleanup."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from alframework.tools import tasks
from alframework.tools.capabilities import declared
from alframework.tools.preflight import bind_task
from alframework.tools.tasks import (ROLES, Role, alf_join_task, alf_task, executor_label, executor_stages,
                                     relabel_executors, role)
from tests.helpers.driver import loaded_parsl

ROOT = Path(__file__).resolve().parents[1]
PARSL_DECORATORS = {'python_app', 'join_app', 'bash_app'}


@alf_task('QM')
def labeler_task(molecule_object):
    """A labeler that preflight binds by its import path."""
    return molecule_object * 2


def _layout(config):
    return 'cuda', 8, 1


@alf_join_task('ML', declares=('shared_training_data',), member_layout=_layout)
def trainer_task(ML_config):
    return None


def _decorator_name(node):
    node = node.func if isinstance(node, ast.Call) else node
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, 'id', None)


def test_every_parsl_app_in_alf_and_its_examples_is_made_in_alframework_tools_tasks():
    """An app written with @python_app by hand could name the wrong executor or miss the timeline."""
    by_hand = []
    for path in sorted([*(ROOT / 'alframework').rglob('*.py'), *(ROOT / 'examples').rglob('*.py')]):
        if path == ROOT / 'alframework' / 'tools' / 'tasks.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                made = {_decorator_name(decorator) for decorator in node.decorator_list} & PARSL_DECORATORS
            else:
                made = isinstance(node, ast.Call) and _decorator_name(node) in PARSL_DECORATORS
            if made:
                by_hand.append(f'{path.relative_to(ROOT)}:{node.lineno}')
    assert by_hand == []


def test_alf_task_applies_the_standard_stack(tmp_path, monkeypatch):
    def validate(config, master_config):
        pass

    @alf_task('sampler', declares=('returned_metadata',), validate_config=validate)
    def my_sampler_task(molecule_objects, sampler_config):
        """Sample."""
        return len(molecule_objects)

    assert my_sampler_task.executors == ['alf_sampler_executor']
    assert my_sampler_task.func.__name__ == 'my_sampler_task' and my_sampler_task.func.__doc__ == 'Sample.'
    assert declared(my_sampler_task) == {'returned_metadata'}
    assert my_sampler_task.func.validate_config is validate
    monkeypatch.setenv('ALF_TASK_TIMELINE_DIR', str(tmp_path))
    assert my_sampler_task.func([1, 2], {}) == 2
    [record] = [json.loads(line) for path in tmp_path.glob('*.jsonl') for line in path.read_text().splitlines()]
    assert (record['stage'], record['task'], record['outcome']) == ('sampler', 'my_sampler_task', 'ok')


def test_an_alf_task_runs_on_its_roles_executor_and_records_its_call(tmp_path, monkeypatch):
    monkeypatch.setenv('ALF_TASK_TIMELINE_DIR', str(tmp_path / 'timeline'))
    with loaded_parsl(tmp_path, 'alf_QM_executor'):
        assert labeler_task(21).result(timeout=30) == 42
    [record] = [json.loads(line) for path in (tmp_path / 'timeline').glob('*.jsonl')
                for line in path.read_text().splitlines()]
    assert (record['stage'], record['task'], record['outcome']) == ('QM', 'labeler_task', 'ok')


def test_a_configured_standby_executor_joins_the_tasks_executor():
    executors = {'alf_QM_executor': object(), 'alf_QM_standby_executor': object()}
    assert bind_task('tests.test_tasks.labeler_task', executors).executors == [
        'alf_QM_executor', 'alf_QM_standby_executor']
    assert bind_task('tests.test_tasks.labeler_task', {'alf_QM_executor': object()}).executors == ['alf_QM_executor']
    with pytest.raises(ValueError, match="unconfigured Parsl executors \\['alf_QM_executor'\\]"):
        bind_task('tests.test_tasks.labeler_task', {'alf_QM_standby_executor': object()})


def test_a_failed_alf_task_frees_its_frames_before_the_error_leaves_it(monkeypatch):
    from alframework.tools import task_failure_cleanup
    released = []
    monkeypatch.setattr(task_failure_cleanup, 'release_failed_task', released.append)

    @alf_task('QM')
    def my_labeler_task(molecule_object):
        raise RuntimeError('out of memory')

    try:
        my_labeler_task.func(None)  # Parsl's func may return the error wrapped instead of raising it.
    except RuntimeError:
        pass
    assert [str(error) for error in released] == ['out of memory']


def test_the_startup_check_needs_the_executor_a_join_task_submits_its_members_to():
    assert declared(trainer_task) == {'shared_training_data'} and trainer_task.join
    with pytest.raises(ValueError, match="unconfigured Parsl executors \\['alf_ML_executor'\\]"):
        bind_task('tests.test_tasks.trainer_task', {})
    assert bind_task('tests.test_tasks.trainer_task', {'alf_ML_executor': object()}).func.member_layout is _layout


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="not 'labeler'"):
        alf_task('labeler')


def test_builders_and_samplers_share_an_executor_and_its_run_records():
    assert executor_label('builder') == executor_label('sampler') == 'alf_sampler_executor'
    assert executor_stages() == {'alf_sampler_executor': ('builder', 'sampler'), 'alf_QM_executor': ('QM',),
                                 'alf_ML_executor': ('ML',)}


@pytest.fixture
def renamed_labeler(monkeypatch):
    """The labeler role renamed, as it may be one day: role, executor and stage names apart, older names accepted."""
    monkeypatch.setattr(tasks, 'ROLES', {**ROLES, 'labeler': Role('QM', 'alf_labeler_executor',
                                                                  accepts=('alf_QM_executor',))})
    monkeypatch.delitem(tasks.ROLES, 'QM')
    monkeypatch.setattr(tasks, 'ROLE_ALIASES', {'QM': 'labeler'})


def test_a_former_role_name_declares_a_task_of_the_renamed_role(renamed_labeler):
    @alf_task('QM')
    def my_labeler_task(molecule_object):
        return molecule_object

    assert role('QM') is role('labeler')
    assert my_labeler_task.executors == ['alf_labeler_executor']
    assert executor_stages()['alf_QM_executor'] == executor_stages()['alf_labeler_executor'] == ('QM',)


def test_a_parsl_configuration_with_a_former_executor_label_runs_under_the_current_one(renamed_labeler):
    executors = [SimpleNamespace(label=label) for label in
                 ('alf_sampler_executor', 'alf_QM_executor', 'alf_QM_standby_executor', 'alf_ML_executor')]
    assert relabel_executors(executors) == [('alf_QM_executor', 'alf_labeler_executor'),
                                            ('alf_QM_standby_executor', 'alf_labeler_standby_executor')]
    assert [executor.label for executor in executors] == [
        'alf_sampler_executor', 'alf_labeler_executor', 'alf_labeler_standby_executor', 'alf_ML_executor']
    assert relabel_executors(executors) == []
    with pytest.raises(ValueError, match="both 'alf_QM_executor' and 'alf_labeler_executor'"):
        relabel_executors([SimpleNamespace(label='alf_QM_executor'), SimpleNamespace(label='alf_labeler_executor')])


def test_a_run_whose_parsl_configuration_uses_a_former_label_binds_its_tasks(tmp_path, monkeypatch):
    """As a run's tasks and Parsl configuration load at startup, after the sampler executor is renamed."""
    from alframework.benchmark.__main__ import _run_tasks
    monkeypatch.setattr(tasks, 'ROLES', {**ROLES, 'sampler': Role('sampler', 'alf_md_executor',
                                                                  accepts=('alf_sampler_executor',))})
    (tmp_path / 'renamed_run.py').write_text(
        'from parsl.config import Config\n'
        'from parsl.executors import ThreadPoolExecutor\n'
        'from alframework.tools.tasks import alf_task\n'
        "config = Config(executors=[ThreadPoolExecutor(label='alf_sampler_executor')])\n"
        "@alf_task('sampler')\n"
        'def sampler_task(molecule_objects):\n'
        '    return molecule_objects\n')
    monkeypatch.delitem(sys.modules, 'renamed_run', raising=False)
    master = {'parsl_configuration': 'renamed_run.config', 'builder_task': 'renamed_run.missing',
              'sampler_task': 'renamed_run.sampler_task'}
    executors, _, sampler = _run_tasks(master, tmp_path)
    assert list(executors) == ['alf_md_executor'] and sampler.executors == ['alf_md_executor']


@pytest.mark.parametrize('name', ['chicoma', 'darwin'])
def test_alfs_parsl_configurations_label_their_executors_as_the_roles_expect(name):
    tree = ast.parse((ROOT / 'alframework' / 'parsl_resource_configs' / f'{name}.py').read_text())
    labels = {keyword.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
              for keyword in node.keywords if keyword.arg == 'label' and isinstance(keyword.value, ast.Constant)}
    assert {current.executor for current in ROLES.values()} <= labels


def test_the_docs_build_keeps_alf_tasks_as_plain_functions(monkeypatch):
    """The docs build (docs/source/conf.py) replaces parsl.python_app with one that keeps the function, so that
    autodoc shows tasks; alf_task looks python_app up when decorating, so it gets the replacement."""
    import parsl
    monkeypatch.setattr(parsl, 'python_app', lambda function=None, **_: function or (lambda func: func))

    @alf_task('builder')
    def my_builder_task(moleculeid):
        return moleculeid

    assert callable(my_builder_task) and not hasattr(my_builder_task, 'executors')
    assert my_builder_task('m0') == 'm0'
