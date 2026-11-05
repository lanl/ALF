"""HIPPYNN interface: deferred imports, checkpoint loading, and the small-molecule database adapter."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import subprocess
import threading
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
import pickle
import h5py

from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_calculator


def test_interface_import_defers_hippynn_and_torch():
    subprocess.run([sys.executable, '-c', """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    assert name.split('.')[0] not in ('torch', 'hippynn'), name
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import alframework.ml_interfaces.hippynn_interface
"""], capture_output=True, check=True)


def test_cpu_training_checkpoints_reload_through_alf(tmp_path):
    pytest.importorskip('hippynn')
    from tests.helpers.hippynn import train_cpu_ensemble
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble
    from hippynn.experiment.serialization import load_checkpoint
    from tests.test_acquisition import WATER
    reports = train_cpu_ensemble(tmp_path/'ensemble')
    assert len(reports) == 2 and all(r['epochs'] >= 2 for r in reports)
    for directory in sorted((tmp_path/'ensemble').glob('model-*')):
        bundle = load_checkpoint(str(directory/'experiment_structure.pt'),
                                 str(directory/'best_checkpoint.pt'), restart_db=True, map_location='cpu')
        assert bundle['database'].coordinates_key == 'coordinates'
    for calculator in HIPNN_ASE_load_ensemble(str(tmp_path/'ensemble'), device='cpu'):
        atoms = WATER.copy()
        atoms.calc = calculator
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()


def test_absolute_checkpoint_paths_allow_concurrent_loading(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    directory = Path.cwd()
    calls = []
    for index in range(2):
        member = tmp_path / str(index)
        member.mkdir()
        (member / 'experiment_structure.pt').write_text(f'member-{index}')
        (member / 'best_checkpoint.pt').write_text(f'checkpoint-{index}')
        (member / 'best_model.pt').write_text('wrong-format')

    def load(structure, state, map_location):
        assert Path(structure).is_absolute() and Path(state).is_absolute()
        assert Path(state).name == 'best_checkpoint.pt'
        assert map_location == 'cpu'
        barrier.wait(timeout=10)
        identity = (Path(structure).read_text(), Path(state).read_text())
        calls.append(identity)
        model = SimpleNamespace(node_from_name=lambda key: (identity, key))
        return {'training_modules': SimpleNamespace(model=model)}

    class Calculator:
        def __init__(self, energy_node, en_unit):
            self.node = energy_node
            self.unit = en_unit
            self.conversions = []

        def to(self, target):
            self.conversions.append(target)

    serialization = ModuleType('hippynn.experiment.serialization')
    serialization.load_checkpoint = load
    interface = ModuleType('hippynn.interfaces.ase_interface')
    interface.HippynnCalculator = Calculator
    monkeypatch.setitem(sys.modules, serialization.__name__, serialization)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32', device=lambda x: x))

    def forbidden(*args):
        pytest.fail('Inference loading must not change process cwd')

    monkeypatch.setattr(os, 'chdir', forbidden)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: HIPNN_ASE_calculator(
            os.path.relpath(tmp_path / str(i), directory), energy_key='custom-energy', device='cpu'), range(2)))
    for index, calc in enumerate(results):
        assert calc.node == ((f'member-{index}', f'checkpoint-{index}'), 'custom-energy')
        assert calc.unit == 1.
        assert calc.conversions == ['float32', 'cpu']
    assert len(calls) == 2
    assert Path.cwd() == directory

    def missing(structure, state, **kwargs):
        Path(structure).read_bytes()

    serialization.load_checkpoint = missing
    with pytest.raises(FileNotFoundError):
        HIPNN_ASE_calculator(tmp_path / 'missing', device='cpu')
    assert Path.cwd() == directory


def test_concurrent_trained_members_match_legacy_predictions(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('hippynn')
    from ase import units
    from hippynn.experiment import serialization
    from hippynn.interfaces.ase_interface import HippynnCalculator
    from hippynn.tools import active_directory
    from tests.test_acquisition import TRAINED_ENSEMBLE, WATER

    members = [TRAINED_ENSEMBLE / f'model-{i:02d}' for i in range(2)]
    if not all((member / 'best_checkpoint.pt').is_file() for member in members):
        pytest.skip('trained Darwin checkpoint members not available')

    expected = []
    # Capture the original loader's predictions before forbidding cwd changes.
    for member in members:
        with active_directory(str(member)):
            bundle = serialization.load_checkpoint_from_cwd(map_location='cpu')
        calc = HippynnCalculator(bundle['training_modules'].model.node_from_name('energy'), en_unit=units.eV)
        calc.to(torch.float32)
        calc.to(torch.device('cpu'))
        atoms = WATER.copy()
        atoms.calc = calc
        expected.append((atoms.get_potential_energy(), atoms.get_forces()))

    original_load = serialization.load_checkpoint
    barrier = threading.Barrier(2)

    def concurrent_load(*args, **kwargs):
        barrier.wait(timeout=30)
        return original_load(*args, **kwargs)

    def forbidden(*args):
        pytest.fail('Inference loading must not change process cwd')

    monkeypatch.setattr(serialization, 'load_checkpoint', concurrent_load)
    monkeypatch.setattr(os, 'chdir', forbidden)
    before = Path.cwd()

    def predict(index):
        atoms = WATER.copy()
        atoms.calc = HIPNN_ASE_calculator(os.path.relpath(members[index]), device='cpu')
        return atoms.get_potential_energy(), atoms.get_forces()

    with ThreadPoolExecutor(max_workers=2) as pool:
        predictions = list(pool.map(predict, [0, 1, 1, 0]))
    for index, (energy, forces) in zip([0, 1, 1, 0], predictions):
        np.testing.assert_allclose(energy, expected[index][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(forces, expected[index][1], rtol=1e-5, atol=1e-5)
    assert Path.cwd() == before


# The database adapter tests need HIPPYNN itself; the loading tests above do not.
@pytest.fixture
def readers():
    pytest.importorskip('hippynn')
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB
    from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
    return ALFPyAniDirectoryDB, PyAniDirectoryDB


def write_dataset(path, sizes=(3, 6), custom=False):
    coordinates, forces, species = ('xyz', 'grad', 'Z') if custom else ('coordinates', 'forces', 'species')
    with h5py.File(path, 'w') as handle:
        for index, size in enumerate(sizes):
            group = handle.create_group(f'group-{index}')
            group[species] = np.array(['H']*size, dtype='S')
            group[coordinates] = np.arange(2*size*3, dtype=float).reshape(2, size, 3) + 100*index
            group[forces] = np.full((2, size, 3), index+1.)
            group['energy'] = [index+1., index+2.]
            group['cell'] = np.repeat((np.eye(3)*10)[None], 2, axis=0)
            group['dipole'] = np.full((2, 3), 7.)
            group['quadrupole'] = np.full((2, 3, 3), 9.)
    return dict(coordinates_key=coordinates, atom_keys=(coordinates, forces), species_key=species)


@pytest.mark.parametrize('sizes', [(3,), (6,), (3, 6), (6, 3), (3, 9)])
def test_small_groups_pad_only_atom_axis_and_preserve_components(tmp_path, sizes, readers):
    ALFPyAniDirectoryDB, PyAniDirectoryDB = readers
    schema = write_dataset(tmp_path/'data.h5', sizes)
    db = ALFPyAniDirectoryDB(str(tmp_path), inputs=None, targets=None, allow_unfound=True,
                            quiet=True, seed=1, **schema)
    arrays = {k: v.numpy() for k,v in db.arr_dict.items()}
    assert arrays['coordinates'].shape == (2*len(sizes), max(sizes), 3)
    assert arrays['cell'].shape == (2*len(sizes), 3, 3)
    assert arrays['dipole'].shape == (2*len(sizes), 3)
    assert arrays['quadrupole'].shape == (2*len(sizes), 3, 3)
    for index, size in enumerate(sizes):
        frames = slice(2*index, 2*index+2)
        np.testing.assert_array_equal(arrays['forces'][frames,:size], index+1.)
        np.testing.assert_array_equal(arrays['forces'][frames,size:], 0.)
        np.testing.assert_array_equal(arrays['species'][frames,:size], 1)
        np.testing.assert_array_equal(arrays['species'][frames,size:], 0)
        expected = np.arange(2*size*3).reshape(2,size,3) + 100*index
        np.testing.assert_array_equal(arrays['coordinates'][frames,:size], expected)
    np.testing.assert_array_equal(arrays['dipole'], 7.)
    np.testing.assert_array_equal(arrays['quadrupole'], 9.)


def test_custom_schema_survives_checkpoint_restarter(tmp_path, readers):
    ALFPyAniDirectoryDB, PyAniDirectoryDB = readers
    schema = write_dataset(tmp_path/'data.h5', custom=True)
    db = ALFPyAniDirectoryDB(str(tmp_path), inputs=None, targets=None, allow_unfound=True,
                            quiet=True, seed=1, **schema)
    assert type(db).__module__ == 'alframework.ml_interfaces.hippynn_interface'
    assert type(db).__qualname__ == 'ALFPyAniDirectoryDB'
    restarted = pickle.loads(pickle.dumps(db.restarter)).attempt_restart()
    assert restarted.coordinates_key == 'xyz'
    assert 'grad' in restarted.atom_keys
    for key, value in db.arr_dict.items():
        np.testing.assert_array_equal(value.numpy(), restarted.arr_dict[key].numpy())
    # A fresh worker must resolve the class without a previous explicit import.
    subprocess.run([sys.executable, '-c', """
import pickle
import sys
restarter = pickle.loads(sys.stdin.buffer.read())
database = restarter.attempt_restart()
assert type(database).__module__ == 'alframework.ml_interfaces.hippynn_interface'
assert database.coordinates_key == 'xyz'
assert database.arr_dict['xyz'].shape == (4, 6, 3)
"""], input=pickle.dumps(db.restarter), capture_output=True, check=True)


def test_existing_larger_groups_match_upstream_arrays(tmp_path, readers):
    ALFPyAniDirectoryDB, PyAniDirectoryDB = readers
    write_dataset(tmp_path/'data.h5', sizes=(8, 10))
    kwargs = dict(directory=str(tmp_path), inputs=None, targets=None, allow_unfound=True,
                  quiet=True, seed=1)
    actual, expected = ALFPyAniDirectoryDB(**kwargs), PyAniDirectoryDB(**kwargs)
    assert set(actual.arr_dict) == set(expected.arr_dict)
    for key, value in expected.arr_dict.items():
        np.testing.assert_array_equal(actual.arr_dict[key].numpy(), value.numpy())


def test_unknown_varying_property_is_rejected_instead_of_guessing_axis(tmp_path, readers):
    ALFPyAniDirectoryDB, PyAniDirectoryDB = readers
    write_dataset(tmp_path/'data.h5')
    with h5py.File(tmp_path/'data.h5', 'a') as handle:
        handle['group-0']['custom_atomic'] = np.zeros((2, 3))
        handle['group-1']['custom_atomic'] = np.zeros((2, 6))
    with pytest.raises(ValueError, match='custom_atomic'):
        ALFPyAniDirectoryDB(str(tmp_path), inputs=None, targets=None, allow_unfound=True,
                            quiet=True, seed=1)
