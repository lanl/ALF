"""The generic ASE labeling task, with real ASE calculators and calculators that fail on purpose."""
from copy import deepcopy
import subprocess
import sys

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import BaseCalculator, all_changes
from ase.calculators.emt import EMT

from alframework.qm_interfaces.ase import generic as interface
from alframework.tools.label_validation import screen_labels
from alframework.tools.labeling import LabelingCoordinator, label_sequentially
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import store_current_data
from tests.helpers.ase_qm import EMT_CONFIG, label_with_ase
from tests.helpers.labels import PROPERTIES, unlabeled_h2


@pytest.mark.parametrize('nested', [False, True])
def test_an_in_process_calculator_labels_without_a_command(tmp_path, nested):
    config = dict(EMT_CONFIG, input={'asap_cutoff': True}) if nested else dict(EMT_CONFIG, asap_cutoff=True)
    config['batching'] = {'mode': 'sequential', 'batch_size': 2}
    before = deepcopy(config)
    obj = unlabeled_h2()
    original = obj.atoms.copy()
    reference = original.copy()
    reference.calc = EMT(asap_cutoff=True)
    result = label_with_ase(obj, config, str(tmp_path), PROPERTIES)
    assert result is obj and obj.check_convergence(), obj.metadata
    assert obj.get_results()['energy'] == pytest.approx(reference.get_potential_energy())
    np.testing.assert_allclose(obj.get_results()['forces'], reference.get_forces())
    assert config == before and obj.atoms.calc is None
    np.testing.assert_array_equal(obj.atoms.positions, original.positions)


@pytest.mark.parametrize('flag,good', [(True, True), (np.bool_(True), True), (False, False),
                                     (None, False), (1, False), (np.array([True]), False)])
def test_a_calculator_s_own_convergence_flag_decides_the_label(tmp_path, monkeypatch, flag, good):
    class FlaggedEMT(EMT):
        converged = flag
    monkeypatch.setattr(interface, 'load_module_from_config', lambda *_: FlaggedEMT)
    obj = label_with_ase(unlabeled_h2(), EMT_CONFIG, str(tmp_path), PROPERTIES)
    assert obj.check_convergence() is good
    if not good:
        assert obj.metadata['qm_failed_stage'] == 'validation'


@pytest.mark.parametrize('damage', ['missing', 'nan', 'shape', 'energy_shape', 'overflow'])
def test_missing_or_invalid_fresh_results_fail_the_label(tmp_path, monkeypatch, damage):
    class DamagedEMT(EMT):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            if damage == 'missing':
                self.results.pop('forces')
            elif damage == 'nan':
                self.results['energy'] = np.nan
            elif damage == 'shape':
                self.results['forces'] = np.zeros((1, 3))
            elif damage == 'energy_shape':
                self.results['energy'] = np.ones(2)
            else:
                self.results['energy'] = 1e300
    monkeypatch.setattr(interface, 'load_module_from_config', lambda *_: DamagedEMT)
    obj = unlabeled_h2()
    obj.store_results({'energy': -1., 'forces': np.zeros((2, 3))})
    obj.set_converged_flag(True)
    properties = deepcopy(PROPERTIES)
    if damage == 'overflow':
        properties['energy'][2] = 1e300
    label_with_ase(obj, EMT_CONFIG, str(tmp_path), properties)
    assert not obj.check_convergence()
    assert obj.metadata['qm_error'] and obj.metadata['qm_failed_stage'] == 'validation'


def test_calculators_without_atoms_in_their_constructor_are_given_all_changes(tmp_path, monkeypatch):
    class ModernCalculator(BaseCalculator):
        implemented_properties = ['energy', 'forces']

        def __init__(self, *, directory):
            super().__init__()
            self.directory = directory

        def calculate(self, atoms, properties, system_changes):
            assert system_changes == all_changes
            self.results = {'energy': -1., 'forces': np.zeros((len(atoms), 3))}
    monkeypatch.setattr(interface, 'load_module_from_config', lambda *_: ModernCalculator)
    obj = label_with_ase(unlabeled_h2(), EMT_CONFIG, str(tmp_path), PROPERTIES)
    assert obj.check_convergence(), obj.metadata


def test_a_calculator_cannot_change_the_geometry_or_the_configuration(tmp_path, monkeypatch):
    class Mutator(EMT):
        def __init__(self, *, nested, **kwargs):
            nested['items'].append('changed')
            super().__init__(**kwargs)

        def calculate(self, atoms, properties, system_changes):
            atoms.positions[0, 0] += 1
            super().calculate(atoms, properties, system_changes)
    monkeypatch.setattr(interface, 'load_module_from_config', lambda *_: Mutator)
    config = dict(EMT_CONFIG, input={'nested': {'items': []}})
    obj = unlabeled_h2()
    original = obj.atoms.copy()
    label_with_ase(obj, config, str(tmp_path), PROPERTIES)
    assert not obj.check_convergence() and 'geometry' in obj.metadata['qm_error']
    assert config['input']['nested']['items'] == []
    np.testing.assert_array_equal(obj.atoms.positions, original.positions)


@pytest.mark.parametrize('extra,properties', [
    ({'input': {}, 'asap_cutoff': True}, PROPERTIES),
    ({'input': []}, PROPERTIES),
    ({'input': {'directory': '/elsewhere'}}, PROPERTIES),
    ({'input': {'ASE_calculator': 'bad'}}, PROPERTIES),
    ({'input': {'label': '/elsewhere/h2'}}, PROPERTIES),
    ({'QM_run_command': ''}, PROPERTIES),
    ({}, {'missing': ['missing', 'system', 1.0]}),
    ({}, {'energy': ['energy', 'typo', 1.0]}),
    ({}, {}),
])
def test_invalid_settings_stop_the_coordinator_at_startup(extra, properties):
    with pytest.raises((ValueError, TypeError)):
        LabelingCoordinator(interface.ase_calculator_task, 'ase', dict(EMT_CONFIG, **extra),
                            {'properties_list': properties})


def test_a_retried_label_moves_the_earlier_attempt_aside(tmp_path):
    directory = tmp_path / 'h2'
    directory.mkdir()
    (directory / 'old.out').write_text('previous calculation')
    obj = unlabeled_h2()
    obj.update_metadata({'qm_error': 'previous error', 'qm_failed_stage': 'calculation'})
    label_with_ase(obj, EMT_CONFIG, str(tmp_path), PROPERTIES)
    assert obj.check_convergence() and 'qm_error' not in obj.metadata
    assert (tmp_path / 'h2.previous-0/old.out').read_text() == 'previous calculation'
    assert not (directory / 'old.out').exists()


@pytest.mark.parametrize('name', ['../outside', '/absolute', '..', ''])
def test_a_molecule_id_cannot_point_outside_the_scratch_directory(tmp_path, name):
    obj = label_with_ase(unlabeled_h2(name), EMT_CONFIG, str(tmp_path / 'scratch'), PROPERTIES)
    assert not obj.check_convergence() and 'Molecule ID' in obj.metadata['qm_error']
    assert not (tmp_path / 'scratch').exists()


def test_a_failed_label_in_a_sequential_batch_keeps_its_siblings(tmp_path):
    good, bad = unlabeled_h2('good'), MoleculesObject(Atoms('He'), 'unsupported')
    results = label_sequentially([good, bad], interface.ase_calculator_task.func,
        {'QM_config': EMT_CONFIG, 'QM_scratch_dir': str(tmp_path / 'scratch'), 'properties_list': PROPERTIES})
    screen_labels(results, PROPERTIES)
    assert [obj.check_convergence() for obj in results] == [True, False]
    assert bad.metadata['qm_failed_stage'] == 'calculation'
    shard = tmp_path / 'data.h5'
    assert store_current_data(str(shard), results, PROPERTIES) == 1
    with h5py.File(shard) as handle:
        assert handle['H02/energy'].shape[0] == 1


def test_pyscf_through_the_generic_task_matches_pyscf_task(tmp_path):
    pytest.importorskip('pyscf')
    from alframework.qm_interfaces.pyscf import label_pyscf_molecule
    options = {'xc': 'pbe', 'basis': 'sto-3g', 'grids_level': 1}
    config = {'ASE_calculator': 'alframework.qm_interfaces.pyscf.calculator.PySCFCalculator', 'input': options}
    generic = label_with_ase(unlabeled_h2(), config, str(tmp_path), PROPERTIES)
    specialized = label_pyscf_molecule(unlabeled_h2(), options,
        {'E0': ['energy', 'system', 1.0], 'F0': ['forces', 'atomic', 1.0]})
    assert generic.check_convergence(), generic.metadata
    assert specialized.check_convergence(), specialized.metadata
    assert generic.get_results()['energy'] == pytest.approx(specialized.get_results()['E0'], abs=1e-9)
    np.testing.assert_allclose(generic.get_results()['forces'], specialized.get_results()['F0'], atol=1e-9)


@pytest.mark.parametrize('batched', [False, True])
def test_the_coordinator_replaces_a_failed_ase_label_and_a_restart_keeps_its_shards(tmp_path, monkeypatch, batched):
    import json
    from parsl import python_app
    from tests.helpers.driver import MODEL_PATTERN, run_alf_driver
    trained = []
    monkeypatch.setattr(interface, 'load_module_from_config', lambda *_: EMT)

    @python_app(executors=['alf_builder_executor'])
    def builder(moleculeid):
        if moleculeid.endswith('0000000000'):
            return MoleculesObject(Atoms('He'), moleculeid)
        return unlabeled_h2(moleculeid)

    @python_app(executors=['alf_sampler_executor'])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=['alf_ML_executor'])
    def ml(current_training_id):
        trained.append(current_training_id)
        (tmp_path / MODEL_PATTERN.format(current_training_id)).mkdir(parents=True)
        return [True], current_training_id

    config = dict(EMT_CONFIG)
    if batched:
        config['batching'] = {'mode': 'sequential', 'batch_size': 2, 'max_wait_seconds': .01}
    kwargs = dict(builder=builder, sampler=sampler, qm=interface.ase_calculator_task, ml=ml,
        qm_config=config, QM_scratch_dir=str(tmp_path / 'scratch'), bootstrap_set=2,
        finite_bootstrap=True, bootstrap_spares=1, parallel_samplers=2, target_queued_QM=2,
        save_h5_threshold=0, maximum_stored_labels=2, poll_interval=.01)
    run_alf_driver(tmp_path, monkeypatch, **kwargs)
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['lifetime_stored_labels'] == 2 and trained == [0]
    shards = {path.name: path.read_bytes() for path in (tmp_path / 'h5').glob('*.h5')}
    run_alf_driver(tmp_path, monkeypatch, **kwargs)
    assert trained == [0]
    assert shards == {path.name: path.read_bytes() for path in (tmp_path / 'h5').glob('*.h5')}


def test_startup_checks_survive_the_plain_decorator_the_docs_build_uses():
    # docs/source/conf.py replaces Parsl's python_app with a decorator that returns the plain function.
    subprocess.run([sys.executable, '-B', '-c', '''
import parsl
parsl.python_app = lambda *args, **kwargs: lambda function: function
from alframework.qm_interfaces.ase.generic import ase_calculator_task
assert callable(ase_calculator_task.validate_config)
'''], check=True)
