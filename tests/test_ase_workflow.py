"""The ASE workflow task: several ASE calculations on one fixed geometry, in order or repeated to agree."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator

from alframework.qm_interfaces.ase import generic
from alframework.qm_interfaces.ase.workflow import ase_workflow_task, label_workflow, validate_ase_workflow_config
from alframework.tools.labeling import LabelingCoordinator, label_sequentially
from alframework.tools.data_shards import store_current_data
from tests.helpers.labels import PROPERTIES, unlabeled_h2


def settings(mode='last', count=2):
    config = {'stages': [{'name': f'stage_{i}', 'ASE_calculator': 'ase.calculators.emt.EMT', 'input': {}}
                          for i in range(count)], 'result_mode': mode}
    if mode == 'mean_if_agree':
        config['agreement'] = {'energy': 1., 'forces': 1.}
    return config


@pytest.fixture
def controlled(monkeypatch):
    calls = []
    responses = []

    class Controlled(Calculator):
        implemented_properties = ['energy', 'forces', 'dipole', 'extra']

        def __init__(self, directory, **options):
            self.options = deepcopy(options)
            super().__init__(directory=directory)
            self.metadata = {'custom_state': 'kept'}

        def calculate(self, atoms, properties, system_changes):
            calls.append((self.directory, atoms.copy(), self.options))
            assert atoms.calc is None
            assert not list(Path(self.directory).iterdir())
            (Path(self.directory) / 'ran').write_text('stage executed')
            response = responses.pop(0) if responses else {}
            if isinstance(response, Exception):
                raise response
            self.results = {'energy': 0., 'forces': np.zeros((len(atoms), 3)), 'dipole': np.ones(3)}
            self.results.update(response)
            # State must not be passed to the next stage.
            atoms.info['charge'] = 999
    monkeypatch.setattr(generic, 'load_module_from_config', lambda *_: Controlled)
    return calls, responses


def test_stages_run_in_order_from_the_same_geometry_and_the_last_one_gives_the_label(tmp_path, controlled):
    calls, responses = controlled
    responses.extend([{'energy': 1.}, {'energy': 2.}])
    config = settings()
    config['stages'][1]['input'] = {'method': 'target'}
    before = deepcopy(config)
    obj = unlabeled_h2()
    obj.update_metadata({'charge': -1, 'mult': 2})
    label_workflow(obj, config, str(tmp_path), PROPERTIES)
    assert obj.check_convergence(), obj.metadata
    assert obj.get_results()['energy'] == 2.
    assert [Path(call[0]).name for call in calls] == ['stage_0', 'stage_1']
    assert [call[1].info['charge'] for call in calls] == [-1, -1]
    assert calls[1][2] == {'method': 'target'}
    assert obj.atoms.info == {} and config == before
    assert obj.metadata['custom_state'] == 'kept'
    assert [item['converged'] for item in obj.metadata['qm_workflow']['stages']] == [True, True]


def test_a_retried_workflow_moves_the_earlier_attempt_aside(tmp_path, controlled):
    obj = label_workflow(unlabeled_h2(), settings(), str(tmp_path), PROPERTIES)
    label_workflow(obj, settings(), str(tmp_path), PROPERTIES)
    assert obj.check_convergence(), obj.metadata
    assert (tmp_path / 'h2.previous-0' / 'stage_0' / 'ran').exists()


@pytest.mark.parametrize('property,difference,accepted', [
    ('energy', .5, True), ('energy', 1., False), ('energy', 1.1, False),
    ('forces', .5, True), ('forces', 1., False), ('forces', 1.1, False)])
def test_repeats_are_averaged_only_when_they_agree_within_the_tolerances(tmp_path, controlled, property,
                                                                         difference, accepted):
    calls, responses = controlled
    value = difference if property == 'energy' else np.full((2, 3), difference)
    responses.extend([{}, {property: value}])
    obj = label_workflow(unlabeled_h2(), settings('mean_if_agree'), str(tmp_path),
                         dict(PROPERTIES, dipole=['dipole', 'system', 1.]))
    assert obj.check_convergence() is accepted, obj.metadata
    if accepted:
        np.testing.assert_allclose(obj.get_results()[property], value / 2)
        np.testing.assert_allclose(obj.get_results()['dipole'], np.ones(3))
    else:
        assert obj.metadata['qm_failed_stage'] == 'agreement'


def test_every_pair_of_repeats_must_agree(tmp_path, controlled):
    _, responses = controlled
    responses.extend([{'energy': 0.}, {'energy': .75}, {'energy': -.75}])
    obj = label_workflow(unlabeled_h2(), settings('mean_if_agree', 3), str(tmp_path), PROPERTIES)
    assert not obj.check_convergence()
    assert obj.metadata['qm_workflow']['max_deviation']['energy'] == 1.5


@pytest.mark.parametrize('index', [0, 1])
def test_a_failed_stage_stops_the_workflow_and_names_the_stage(tmp_path, controlled, index):
    calls, responses = controlled
    responses.extend([{}] * index + [RuntimeError('controlled failure')])
    obj = unlabeled_h2()
    obj.store_results({'energy': -999., 'forces': np.ones((2, 3))})
    label_workflow(obj, settings(count=3), str(tmp_path / 'failed'), PROPERTIES)
    assert not obj.check_convergence() and len(calls) == index + 1
    assert obj.metadata['qm_failed_stage'] == f'stage_{index}:calculation'
    assert len(obj.metadata['qm_workflow']['stages']) == index + 1
    good = label_workflow(unlabeled_h2('good'), settings(), str(tmp_path / 'good'), PROPERTIES)
    assert store_current_data(str(tmp_path / 'labels.h5'), [obj, good], PROPERTIES) == 1


@pytest.mark.parametrize('change', [
    {'stages': []}, {'stages': 'bad'}, {'result_mode': 'average'}, {'agreement': {}}, {'unknown': 1},
    {'stages': [{'name': '../escape', 'ASE_calculator': 'ase.calculators.emt.EMT'}]},
])
def test_an_invalid_workflow_stops_at_startup_and_writes_no_scratch(tmp_path, change):
    config = settings()
    config.update(change)
    with pytest.raises((ValueError, TypeError)):
        LabelingCoordinator(ase_workflow_task, 'workflow', config, {'properties_list': PROPERTIES})
    obj = label_workflow(unlabeled_h2(), config, str(tmp_path / 'scratch'), PROPERTIES)
    assert not obj.check_convergence() and not (tmp_path / 'scratch').exists()


@pytest.mark.parametrize('damage', ['names', 'class', 'options', 'command', 'count', 'energy', 'forces', 'missing',
                                    'properties'])
def test_repeats_must_be_identical_and_have_both_tolerances(damage):
    config = settings('mean_if_agree')
    properties = PROPERTIES
    second = config['stages'][1]
    if damage == 'names':
        second['name'] = config['stages'][0]['name']
    elif damage == 'class':
        second['ASE_calculator'] = 'ase.calculators.lj.LennardJones'
    elif damage == 'options':
        second['input'] = {'asap_cutoff': True}
    elif damage == 'command':
        second['QM_run_command'] = '/other'
    elif damage == 'count':
        config['stages'].pop()
    elif damage == 'missing':
        config['agreement'].pop('forces')
    elif damage == 'properties':
        properties = {'energy': PROPERTIES['energy']}
    else:
        config['agreement'][damage] = float('nan') if damage == 'energy' else False
    with pytest.raises((ValueError, TypeError)):
        validate_ase_workflow_config(config, {'properties_list': properties})


def test_a_sequential_batch_of_workflows_keeps_the_labels_that_succeed(tmp_path):
    good, bad = unlabeled_h2('good'), unlabeled_h2('../bad')
    results = label_sequentially([good, bad], ase_workflow_task.func,
        {'QM_config': settings('mean_if_agree'), 'QM_scratch_dir': str(tmp_path), 'properties_list': PROPERTIES})
    assert [item.check_convergence() for item in results] == [True, False]
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('tolerance', [0., -1., float('inf'), '0.1', None])
def test_tolerances_must_be_positive_and_finite(tolerance):
    config = settings('mean_if_agree')
    config['agreement']['energy'] = tolerance
    with pytest.raises(ValueError, match='positive and finite'):
        validate_ase_workflow_config(config, {'properties_list': PROPERTIES})


def test_an_invalid_geometry_fails_before_any_scratch_is_written(tmp_path):
    obj = unlabeled_h2()
    obj.atoms.positions[0, 0] = np.nan
    label_workflow(obj, settings(), str(tmp_path / 'scratch'), PROPERTIES)
    assert not obj.check_convergence() and not (tmp_path / 'scratch').exists()


def test_large_finite_labels_that_agree_are_averaged_without_overflow(tmp_path, controlled):
    _, responses = controlled
    responses.extend([{'energy': 1e308}, {'energy': 1e308}])
    obj = label_workflow(unlabeled_h2(), settings('mean_if_agree'), str(tmp_path), PROPERTIES)
    assert obj.check_convergence(), obj.metadata
    assert obj.get_results()['energy'] == 1e308


def test_a_workflow_task_still_runs_after_parsl_serializes_it(tmp_path):
    from parsl.serialize import pack_apply_message, unpack_apply_message
    function, args, kwargs = unpack_apply_message(pack_apply_message(ase_workflow_task.func,
        (unlabeled_h2(), settings('mean_if_agree'), str(tmp_path), PROPERTIES), {}))
    result = function(*args, **kwargs)
    assert result.check_convergence(), result.metadata
    assert callable(ase_workflow_task.func.validate_config)


def test_repeats_with_different_label_shapes_do_not_agree(tmp_path, controlled):
    _, responses = controlled
    responses.extend([{'extra': np.ones(2)}, {'extra': np.ones((2, 1))}])
    obj = label_workflow(unlabeled_h2(), settings('mean_if_agree'), str(tmp_path),
                         dict(PROPERTIES, extra=['extra', 'atomic', 1.]))
    assert not obj.check_convergence() and obj.metadata['qm_failed_stage'] == 'agreement'
    assert 'different shapes' in obj.metadata['qm_error']
