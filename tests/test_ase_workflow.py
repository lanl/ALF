"""The ASE workflow task: several ASE calculations on one fixed geometry, in order or repeated to agree."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator

from alframework.qm_interfaces.ase import generic
from alframework.qm_interfaces.ase.workflow import ase_workflow_task, label_workflow, validate_ase_workflow_config
from alframework.tools.labeling_coordinator import LabelingCoordinator, label_sequentially
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
    assert [stage['name'] for stage in obj.metadata['label_source']['stages']] == ['stage_0', 'stage_1']


def test_a_good_label_carries_no_per_stage_record_and_a_failed_one_does(tmp_path, controlled):
    _, responses = controlled
    good = label_workflow(unlabeled_h2('good'), settings(), str(tmp_path), PROPERTIES)
    assert good.check_convergence()
    assert 'qm_workflow' not in good.metadata and 'qm_agreement_max_deviation' not in good.metadata
    responses.extend([{}, RuntimeError('controlled failure')])
    failed = label_workflow(unlabeled_h2('failed'), settings(), str(tmp_path), PROPERTIES)
    assert not failed.check_convergence()
    stages = failed.metadata['qm_workflow']['stages']
    assert [(stage['name'], stage['converged']) for stage in stages] == [('stage_0', True), ('stage_1', False)]
    assert stages[0]['directory'] == str(tmp_path / 'failed' / 'stage_0')


def test_retrying_a_failed_label_drops_its_old_per_stage_record(tmp_path, controlled):
    _, responses = controlled
    responses.append(RuntimeError('controlled failure'))
    molecule = label_workflow(unlabeled_h2(), settings(), str(tmp_path), PROPERTIES)
    assert 'qm_workflow' in molecule.metadata
    assert label_workflow(molecule, settings(), str(tmp_path), PROPERTIES).check_convergence()
    assert 'qm_workflow' not in molecule.metadata


def dependent_settings():
    config = settings()
    prep, target = config['stages']
    prep.update(ASE_calculator='example.Producer', properties=['energy'])
    target.update(ASE_calculator='example.Consumer', inputs_from_stages=[
        {'stage': 'stage_0', 'source': 'orbitals.bin', 'destination': 'guess.bin'},
    ])
    return config


@pytest.fixture
def dependent_calculators(monkeypatch):
    writers, received = [], []

    class Producer(Calculator):
        implemented_properties = ['energy']

        def calculate(self, atoms, properties, system_changes):
            assert properties == ['energy']
            writers.pop(0)(Path(self.directory) / 'orbitals.bin')
            self.results = {'energy': 1.0}

    class Consumer(Calculator):
        implemented_properties = ['energy', 'forces']

        def __init__(self, directory):
            super().__init__(directory=directory)
            guess = Path(directory) / 'guess.bin'
            received.append(guess.read_bytes())
            guess.write_bytes(b'target-updated orbitals')

        def calculate(self, atoms, properties, system_changes):
            assert properties == ['energy', 'forces']
            self.results = {'energy': 2.0, 'forces': np.ones((len(atoms), 3))}

    classes = {'example.Producer': Producer, 'example.Consumer': Consumer}
    monkeypatch.setattr(generic, 'load_module_from_config', lambda config, key: classes[config[key]])
    return writers, received


def test_a_stage_reads_its_input_before_calculator_construction(tmp_path, dependent_calculators):
    writers, received = dependent_calculators
    writers.append(lambda path: path.write_bytes(b'prep orbitals'))
    config = dependent_settings()
    validate_ase_workflow_config(config, {'properties_list': PROPERTIES})
    molecule = unlabeled_h2()
    positions = molecule.atoms.positions.copy()
    result = label_workflow(molecule, config, str(tmp_path), PROPERTIES)
    assert result.check_convergence(), result.metadata
    assert received == [b'prep orbitals']
    assert (tmp_path / 'h2/stage_1/guess.bin').read_bytes() == b'target-updated orbitals'
    assert result.get_results()['energy'] == 2.0
    np.testing.assert_array_equal(result.get_results()['forces'], np.ones((2, 3)))
    np.testing.assert_array_equal(result.atoms.positions, positions)
    assert result.metadata['label_source']['stages'][1]['inputs_from_stages'] == config['stages'][1]['inputs_from_stages']


@pytest.mark.parametrize('write_artifact', [
    lambda path: None,
    lambda path: path.touch(),
    lambda path: path.mkdir(),
    lambda path: path.symlink_to(__file__),
], ids=['missing', 'empty', 'directory', 'symlink'])
def test_bad_stage_input_stops_before_calculator_construction(tmp_path, dependent_calculators, write_artifact):
    writers, received = dependent_calculators
    writers.append(write_artifact)
    result = label_workflow(unlabeled_h2(), dependent_settings(), str(tmp_path), PROPERTIES)
    assert not result.check_convergence()
    assert result.get_metadata()['qm_failed_stage'] == 'stage_1:input_files'
    assert 'orbitals.bin' in result.get_metadata()['qm_error']
    assert not received


def test_a_retry_does_not_read_an_artifact_from_the_previous_attempt(tmp_path, dependent_calculators):
    writers, received = dependent_calculators
    writers.extend([lambda path: path.write_bytes(b'first attempt'), lambda path: None])
    config = dependent_settings()
    assert label_workflow(unlabeled_h2(), config, str(tmp_path), PROPERTIES).check_convergence()
    second = label_workflow(unlabeled_h2(), config, str(tmp_path), PROPERTIES)
    assert not second.check_convergence()
    assert second.get_metadata()['qm_failed_stage'] == 'stage_1:input_files'
    assert (tmp_path / 'h2.previous-0/stage_1/guess.bin').read_bytes() == b'target-updated orbitals'
    assert received == [b'first attempt']


def test_a_file_s_last_reader_moves_it_and_earlier_readers_copy_it(tmp_path, dependent_calculators):
    writers, received = dependent_calculators
    writers.append(lambda path: path.write_bytes(b'prep orbitals'))
    config = dependent_settings()
    config['stages'].append(dict(deepcopy(config['stages'][1]), name='stage_2'))
    assert label_workflow(unlabeled_h2(), config, str(tmp_path), PROPERTIES).check_convergence()
    assert received == [b'prep orbitals', b'prep orbitals']
    assert not (tmp_path / 'h2/stage_0/orbitals.bin').exists()


def test_a_workflow_removes_its_scratch_after_a_good_label_when_asked(tmp_path, dependent_calculators):
    writers, _ = dependent_calculators
    writers.append(lambda path: path.write_bytes(b'prep orbitals'))
    config = dict(dependent_settings(), remove_scratch_on_success=True)
    assert label_workflow(unlabeled_h2(), config, str(tmp_path), PROPERTIES).check_convergence()
    assert not (tmp_path / 'h2').exists()


def test_a_failed_workflow_keeps_its_scratch(tmp_path, dependent_calculators):
    writers, _ = dependent_calculators
    writers.append(lambda path: None)
    config = dict(dependent_settings(), remove_scratch_on_success=True)
    assert not label_workflow(unlabeled_h2(), config, str(tmp_path), PROPERTIES).check_convergence()
    assert (tmp_path / 'h2/stage_0').is_dir()


def test_scratch_removal_is_a_workflow_setting_not_a_stage_setting(dependent_calculators):
    config = dependent_settings()
    config['stages'][0]['remove_scratch_on_success'] = True
    with pytest.raises(ValueError, match='Unknown ASE stage fields'):
        validate_ase_workflow_config(config, {'properties_list': PROPERTIES})


def test_each_stage_calculates_with_its_own_initial_magnetic_moments(tmp_path, controlled):
    calls, _ = controlled
    config = settings()
    config['stages'][1]['initial_magnetic_moments'] = {'H': 1.0}
    validate_ase_workflow_config(config, {'properties_list': PROPERTIES})
    obj = label_workflow(unlabeled_h2(), config, str(tmp_path), PROPERTIES)
    assert obj.check_convergence(), obj.metadata
    first, second = (atoms.get_initial_magnetic_moments() for _, atoms, _ in calls)
    np.testing.assert_array_equal(first, [0.0, 0.0])
    np.testing.assert_array_equal(second, [1.0, 1.0])


def test_a_stage_s_initial_magnetic_moments_must_be_numbers():
    config = settings()
    config['stages'][1]['initial_magnetic_moments'] = {'H': 'large'}
    with pytest.raises(ValueError, match='finite number'):
        validate_ase_workflow_config(config, {'properties_list': PROPERTIES})


@pytest.mark.parametrize('change,error', [
    ({'stage': 'stage_1'}, 'earlier stage'),
    ({'stage': 'missing'}, 'earlier stage'),
    ({'source': '../orbitals.bin'}, 'filenames without directories'),
    ({'destination': '/tmp/guess.bin'}, 'filenames without directories'),
    ({'destination': r'..\guess.bin'}, 'filenames without directories'),
])
def test_invalid_stage_file_paths_are_rejected_at_startup(dependent_calculators, change, error):
    config = dependent_settings()
    config['stages'][1]['inputs_from_stages'][0].update(change)
    with pytest.raises(ValueError, match=error):
        validate_ase_workflow_config(config, {'properties_list': PROPERTIES})


def test_duplicate_stage_file_destinations_are_rejected_at_startup(dependent_calculators):
    config = dependent_settings()
    inputs = config['stages'][1]['inputs_from_stages']
    inputs.append(dict(inputs[0], source='other.bin'))
    with pytest.raises(ValueError, match='Duplicate stage input destination'):
        validate_ase_workflow_config(config, {'properties_list': PROPERTIES})


@pytest.mark.parametrize('stage,properties,error', [
    (0, ['dipole'], 'must be in properties_list'),
    (0, ['energy', 'energy'], 'distinct property names'),
    (1, ['energy'], 'Only preparatory stages'),
])
def test_invalid_stage_properties_are_rejected_at_startup(dependent_calculators, stage, properties, error):
    config = dependent_settings()
    config['stages'][stage]['properties'] = properties
    with pytest.raises(ValueError, match=error):
        validate_ase_workflow_config(config, {'properties_list': PROPERTIES})


def test_mean_if_agree_rejects_stage_handoffs():
    config = settings('mean_if_agree')
    config['stages'][1]['inputs_from_stages'] = [
        {'stage': 'stage_0', 'source': 'orbitals.bin', 'destination': 'guess.bin'},
    ]
    with pytest.raises(ValueError, match='inputs_from_stages requires result_mode last'):
        validate_ase_workflow_config(config, {'properties_list': PROPERTIES})


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
        assert obj.metadata['qm_agreement_max_deviation'][property] == pytest.approx(difference)
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


def test_a_repeat_that_fails_leaves_no_mean_and_names_its_stage(tmp_path, controlled):
    calls, responses = controlled
    responses.extend([{}, RuntimeError('controlled failure')])
    obj = label_workflow(unlabeled_h2(), settings('mean_if_agree'), str(tmp_path), PROPERTIES)
    assert not obj.check_convergence() and len(calls) == 2
    assert obj.metadata['qm_failed_stage'] == 'stage_1:calculation'
    assert store_current_data(str(tmp_path / 'labels.h5'), [obj], PROPERTIES) == 0


@pytest.mark.parametrize('change,error', [
    ({'stages': []}, 'nonempty stages list'), ({'stages': 'bad'}, 'nonempty stages list'),
    ({'result_mode': 'average'}, 'result_mode must be'), ({'agreement': {}}, 'only valid with mean_if_agree'),
    ({'unknown': 1}, r"Unknown ASE workflow fields: \['unknown'\]"),
    ({'stages': [{'name': '../escape', 'ASE_calculator': 'ase.calculators.emt.EMT'}]}, 'safe filename'),
])
def test_an_invalid_workflow_stops_at_startup_and_writes_no_scratch(tmp_path, change, error):
    config = settings()
    config.update(change)
    with pytest.raises((ValueError, TypeError), match=error):
        LabelingCoordinator(ase_workflow_task, 'workflow', config, {'properties_list': PROPERTIES})
    obj = label_workflow(unlabeled_h2(), config, str(tmp_path / 'scratch'), PROPERTIES)
    assert not obj.check_convergence() and not (tmp_path / 'scratch').exists()


@pytest.mark.parametrize('damage,error', [
    ('names', 'Duplicate ASE stage name'), ('class', 'identical calculator classes'),
    ('options', 'identical calculator classes'), ('command', 'identical calculator classes'),
    ('count', 'at least two independent stages'), ('energy', 'energy tolerance must be positive'),
    ('forces', 'forces tolerance must be positive'), ('missing', 'must provide energy'),
    ('properties', 'requires requested energy and forces')])
def test_repeats_must_be_identical_and_have_both_tolerances(damage, error):
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
    with pytest.raises(ValueError, match=error):
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


def test_a_workflow_label_records_every_stage_as_its_source(tmp_path, dependent_calculators):
    writers, _ = dependent_calculators
    writers.append(lambda path: path.write_bytes(b'prep orbitals'))
    config = dependent_settings()
    config['stages'][1]['initial_magnetic_moments'] = {'H': 1.0}
    molecule = label_workflow(unlabeled_h2(), config, str(tmp_path), PROPERTIES)
    source = molecule.get_metadata()['label_source']
    assert source['labeler'] == 'ase_workflow_task' and source['result_mode'] == 'last'
    first, second = source['stages']
    assert (first['name'], first['calculator'], first['properties']) == ('stage_0', 'example.Producer', ['energy'])
    assert second['properties'] == ['energy', 'forces']
    assert second['inputs_from_stages'] == config['stages'][1]['inputs_from_stages']
    assert second['initial_magnetic_moments'] == {'H': 1.0}


def test_repeated_stages_record_their_agreement_tolerances_in_the_source(tmp_path, controlled):
    molecule = label_workflow(unlabeled_h2(), settings('mean_if_agree'), str(tmp_path), PROPERTIES)
    assert molecule.check_convergence(), molecule.metadata
    assert molecule.get_metadata()['label_source']['agreement'] == settings('mean_if_agree')['agreement']
