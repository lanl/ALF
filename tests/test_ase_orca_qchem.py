"""ORCA and Q-Chem through ASE's own calculators, run against stand-in programs."""
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.orca import ORCA
from ase.calculators.qchem import QChem

from alframework.qm_interfaces.ase import generic as interface
from alframework.qm_interfaces.ase.orca import create_calculator as create_orca
from alframework.qm_interfaces.ase.qchem import create_calculator as create_qchem
from alframework.tools.molecules import MoleculesObject
from alframework.tools.data_shards import store_current_data
from tests.helpers.ase_qm import executable, label_with_ase, orca_command, qchem_command, qchem_output
from tests.helpers.labels import PROPERTIES, unlabeled_h2


@pytest.mark.parametrize('natoms', [2, 7])
def test_qchem_runs_the_configured_command_with_the_molecule_s_charge_and_spin(tmp_path, natoms):
    command = executable(tmp_path, 'import sys\nfrom pathlib import Path\n'
        'assert len(sys.argv) == 3\n'
        'assert "ASE_CALCULATOR" not in Path(sys.argv[1]).read_text()\n'
        f'Path(sys.argv[2]).write_text({qchem_output(natoms)!r})\n') + ' PREFIX.inp PREFIX.out'
    config = {'ASE_calculator': 'ase.calculators.qchem.QChem', 'QM_run_command': command,
        'input': {'method': 'hf', 'basis': 'sto-3g', 'nt': 2},
        'batching': {'mode': 'sequential', 'batch_size': 2}}
    before = deepcopy(config)
    obj = MoleculesObject(Atoms('H' * natoms), 'charged')
    obj.update_metadata({'charge': -1, 'mult': 2})
    label_with_ase(obj, config, str(tmp_path / 'scratch space'), PROPERTIES)
    assert obj.check_convergence(), obj.metadata
    text = (tmp_path / 'scratch space/charged/qchem.inp').read_text()
    assert '$molecule\n   -1 2\n' in text
    for field in ('ASE_CALCULATOR', 'QM_RUN_COMMAND', 'BATCHING'):
        assert field not in text
    assert config == before and obj.atoms.info == {}
    assert obj.get_results()['energy'] == pytest.approx(-2.5 * units.Hartree)
    np.testing.assert_allclose(obj.get_results()['forces'], np.full((natoms, 3), -.1 * units.Hartree / units.Bohr))


@pytest.mark.parametrize('output,exit_code', [('', 0), ('SCF failed to converge', 0),
                                         (qchem_output().replace('-2.5000000000', 'nan'), 0),
                                         (qchem_output(), 7)])
def test_a_failed_or_unreadable_qchem_run_is_unconverged_with_its_error(tmp_path, output, exit_code):
    command = qchem_command(tmp_path, output, exit_code)
    obj = label_with_ase(unlabeled_h2(), {'ASE_calculator': 'ase.calculators.qchem.QChem', 'QM_run_command': command},
               str(tmp_path / 'scratch'), PROPERTIES)
    assert not obj.check_convergence() and obj.metadata['qm_error_type']


def test_orca_runs_the_configured_command_with_the_molecule_s_charge_and_spin(tmp_path):
    config = {'ASE_calculator': 'ase.calculators.orca.ORCA', 'QM_run_command': orca_command(tmp_path),
              'input': {'orcasimpleinput': 'EnGrad HF STO-3G', 'orcablocks': '%pal nprocs 2 end',
                        'charge': 0, 'mult': 1}}
    before = deepcopy(config)
    properties = dict(PROPERTIES, dipole=['dipole', 'system', 1.0])
    obj = unlabeled_h2()
    obj.atoms.info.update(charge=0, mult=1)
    obj.update_metadata({'charge': -1, 'mult': 2})
    label_with_ase(obj, config, str(tmp_path / 'scratch'), properties)
    assert obj.check_convergence(), obj.metadata
    text = (tmp_path / 'scratch/h2/orca.inp').read_text()
    assert '*xyz -1 2' in text and '%pal nprocs 2 end' in text
    assert 'QM_run_command' not in text and 'ASE_calculator' not in text
    assert obj.atoms.info == {'charge': 0, 'mult': 1} and config == before
    assert obj.get_results()['energy'] == pytest.approx(-2.5 * units.Hartree)
    np.testing.assert_allclose(obj.get_results()['forces'], np.full((2, 3), -.1 * units.Hartree / units.Bohr))
    np.testing.assert_allclose(obj.get_results()['dipole'], np.array([1, 2, 3]) * units.Bohr)


@pytest.mark.parametrize('changes', [{'normal': False}, {'energy': 'nan'}, {'dipole': False},
                                     {'gradient': False}, {'exit_code': 7}])
def test_a_failed_or_incomplete_orca_run_does_not_become_a_label(tmp_path, changes):
    config = {'ASE_calculator': 'ase.calculators.orca.ORCA', 'QM_run_command': orca_command(tmp_path, **changes),
              'input': {'orcasimpleinput': 'EnGrad HF STO-3G'}}
    properties = dict(PROPERTIES, dipole=['dipole', 'system', 1.])
    obj = label_with_ase(unlabeled_h2(), config, str(tmp_path / 'scratch'), properties)
    assert not obj.check_convergence(), obj.get_results()
    expected = 'calculation' if 'normal' in changes or 'exit_code' in changes else 'validation'
    assert obj.metadata['qm_failed_stage'] == expected


@pytest.mark.parametrize('backend', ['orca', 'qchem'])
def test_orca_and_qchem_refuse_periodic_structures_before_running(tmp_path, backend):
    config = {'ASE_calculator': f'ase.calculators.{backend}.' + ('ORCA' if backend == 'orca' else 'QChem')}
    if backend == 'orca':
        config.update(QM_run_command='/never-executed/orca', input={'orcasimpleinput': 'EnGrad HF STO-3G'})
    obj = unlabeled_h2()
    obj.atoms.set_cell([4, 4, 4])
    obj.atoms.pbc = True
    label_with_ase(obj, config, str(tmp_path), PROPERTIES)
    assert not obj.check_convergence() and 'nonperiodic' in obj.metadata['qm_error']


@pytest.mark.parametrize('options,command,properties', [
    ({'orcasimpleinput': 'HF STO-3G'}, '/orca', PROPERTIES),
    ({'orcasimpleinput': 'Opt HF STO-3G'}, '/orca', PROPERTIES),
    ({'orcasimpleinput': 'EnGrad HF STO-3G', 'orca_env_file': 'env.sh'}, '/orca', PROPERTIES),
    ({'orcasimpleinput': 'EnGrad HF STO-3G', 'orcablocks': '%coords end'}, '/orca', PROPERTIES),
    ({'orcasimpleinput': 'EnGrad HF STO-3G', 'mult': True}, '/orca', PROPERTIES),
    ({'orcasimpleinput': 'EnGrad HF STO-3G'}, None, PROPERTIES),
    ({'orcasimpleinput': 'EnGrad HF STO-3G'}, '/orca', {'quadrupole': ['q', 'system', 1.]}),
])
def test_unsupported_orca_settings_stop_the_coordinator_at_startup(options, command, properties):
    config = {'ASE_calculator': 'ase.calculators.orca.ORCA',
              'QM_run_command': command, 'input': options}
    with pytest.raises((TypeError, ValueError)):
        interface.validate_ase_config(config, {'properties_list': properties})


def test_alf_settings_apply_only_to_ase_s_own_orca_and_qchem_classes():
    class CustomQChem(QChem):
        pass
    assert interface._backend(QChem).__name__.endswith('.qchem')
    assert interface._backend(ORCA).__name__.endswith('.orca')
    assert interface._backend(CustomQChem) is None
    assert interface._configuration({'ASE_calculator': 'ase.calculators.orca.ORCA',
        'QM_run_command': '/orca', 'input': {'orcasimpleinput': 'EnGrad HF STO-3G'}})[0] is ORCA


def test_qchem_without_a_command_uses_ase_s_command_and_cpu_flags(tmp_path):
    calc = create_qchem(unlabeled_h2().atoms, str(tmp_path), {'np': 2, 'nt': 3}, None)
    assert type(calc) is QChem
    assert calc.command == 'qchem -np 2 -nt 3 PREFIX.inp PREFIX.out'


@pytest.mark.parametrize('backend', ['orca', 'qchem'])
def test_a_new_calculator_does_not_keep_the_previous_molecule_s_charge_and_spin(tmp_path, backend):
    if backend == 'orca':
        create = create_orca
        command = orca_command(tmp_path)
        options = {'charge': 1, 'mult': 2, 'orcasimpleinput': 'EnGrad HF STO-3G'}
        path = tmp_path / 'run/orca.inp'
        expected = ('*xyz -1 2', '*xyz 1 2')
    else:
        create = create_qchem
        command = qchem_command(tmp_path)
        options = {'charge': 1, 'multiplicity': 2}
        path = tmp_path / 'run/qchem.inp'
        expected = ('$molecule\n   -1 2', '$molecule\n   1 2')
    atoms = unlabeled_h2().atoms
    atoms.info.update(charge=-1, mult=2)
    for text in expected:
        calc = create(atoms, str(tmp_path / 'run'), options, command)
        assert type(calc) is (ORCA if backend == 'orca' else QChem)
        calc.calculate(atoms, list(PROPERTIES), system_changes=['positions'])
        assert text in path.read_text()
        atoms.info.clear()


@pytest.mark.parametrize('options', [{'jobtype': 'OPT'}, {'jobtype': 'SP'}, {'nt': 0}, {'np': True}])
def test_unsupported_qchem_jobs_or_resources_stop_the_coordinator_at_startup(options):
    with pytest.raises(ValueError):
        interface.validate_ase_config({'ASE_calculator': 'ase.calculators.qchem.QChem', 'input': options},
                                      {'properties_list': PROPERTIES})


@pytest.mark.parametrize('backend', ['orca', 'qchem'])
def test_the_ase_qm_example_configurations_are_valid(backend):
    directory = Path(__file__).resolve().parents[1] / 'examples' / 'staged_ase_qm'
    config = json.loads((directory / f'{backend}_qm_config.json').read_text())
    master = json.loads((directory / 'master_config_fragment.json').read_text())
    interface.validate_ase_config(config, master)
    assert all(schema[2] == 1.0 for schema in master['properties_list'].values())


@pytest.mark.parametrize('backend', ['orca', 'qchem'])
def test_configured_charge_and_spin_are_stored_with_the_label(tmp_path, backend):
    if backend == 'orca':
        command = orca_command(tmp_path)
        options = {'orcasimpleinput': 'EnGrad HF STO-3G', 'charge': 1, 'mult': 2}
        path = 'ase.calculators.orca.ORCA'
        input_file, state_text = 'orca.inp', '*xyz 1 2'
    else:
        command = qchem_command(tmp_path)
        options = {'charge': 1, 'multiplicity': 2}
        path = 'ase.calculators.qchem.QChem'
        input_file, state_text = 'qchem.inp', '$molecule\n   1 2'
    obj = label_with_ase(unlabeled_h2(), {'ASE_calculator': path, 'QM_run_command': command, 'input': options},
               str(tmp_path / 'scratch'), PROPERTIES)
    assert obj.check_convergence(), obj.metadata
    assert state_text in (tmp_path / 'scratch/h2' / input_file).read_text()
    assert obj.metadata['charge'] == 1 and obj.metadata['mult'] == 2
    assert obj.metadata['qm_backend'] == backend
    shard = tmp_path / 'state.h5'
    store_current_data(str(shard), [obj], PROPERTIES)
    with h5py.File(shard) as handle:
        np.testing.assert_array_equal(handle['H02/total_charge'][()], [1])
        np.testing.assert_array_equal(handle['H02/spin_multiplicity'][()], [2])
