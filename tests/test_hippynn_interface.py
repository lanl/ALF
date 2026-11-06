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
import json

from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_calculator
from tests.helpers.fakes import fake_training_pool


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


def test_concurrent_trained_members_match_legacy_predictions(monkeypatch, trained_ensemble):
    import torch
    from ase import units
    from hippynn.experiment import serialization
    from hippynn.interfaces.ase_interface import HippynnCalculator
    from hippynn.tools import active_directory
    from tests.test_acquisition import WATER

    members = [trained_ensemble / f'model-{i:02d}' for i in range(2)]

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


# Multi-state training contracts and graphs.
from alframework.ml_interfaces import hippynn_interface as ml


@pytest.mark.parametrize('network_choice', [0, 1])
def test_multistate_cpu_training_reload_and_force_gradients(tmp_path, network_choice):
    torch = pytest.importorskip('torch')
    pytest.importorskip('hippynn')
    from hippynn.experiment.serialization import load_checkpoint
    from hippynn.graphs import GraphModule
    data = tmp_path / 'data'
    data.mkdir()
    coordinates = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    coordinates = np.repeat(coordinates, 12, axis=0)
    coordinates[:, 1, 0] += np.linspace(0., .1, 12)
    coordinates[-1, 1] = [.1, 0., 0.]  # Explicitly filtered frame.
    energies = .01 * (coordinates**2).sum(axis=(1, 2))
    with h5py.File(data / 'data.h5', 'w') as h:
        g = h.create_group('water')
        g['species'] = np.array(['O', 'H', 'H'], dtype='S')
        g['coordinates'] = coordinates
        g['_id'] = np.array([f'frame-{i}' for i in range(12)], dtype='S')
        g['topology_atom_ids'] = np.tile(np.arange(3), (12, 1))
        for state in range(2):
            g[f'sE{state}'] = energies + .1 * state
            g[f'F{state}'] = -.02 * coordinates
        g['dE01'] = np.full(12, .1)
        g['nacr'] = np.zeros((12, 1, 3, 3))
        g['nac_pairs'] = [[0, 1]]
    config = _config(network_choice=network_choice, n_models=1, max_epochs=2,
                     batch_size=4, eval_batch_size=4, termination_patience=10,
                     gap_targets={'enabled': True}, torch_threads=1, fraction_train_eval=1.,
                     network_params=dict(possible_species=[0, 1, 8], n_features=4,
                                         n_sensitivities=4, n_interaction_layers=1, n_atom_layers=1,
                                         dist_soft_min=.4, dist_soft_max=2., dist_hard_max=3.,
                                         **({'l_max': 1, 'n_max': 2} if network_choice else {})))
    member = tmp_path / 'model-00'
    ml.train_single_excited_state_model(model_id=0, model_dir=str(member), h5_train_dir=str(data),
                                        properties_list=_properties(with_gap=True), ML_config=config)
    summary = json.loads((member / 'training_summary.json').read_text())
    manifest = json.loads((member / 'training_input_manifest.json').read_text())
    assert summary['epochs'] == 2 and summary['parameters_changed']
    assert summary['hippynn_kernel'] == 'pytorch'
    assert manifest['filtered_indices'] == [11]
    assert sorted(i for split in manifest['splits'].values() for i in split) == list(range(11))
    bundle = load_checkpoint(str(member / 'experiment_structure.pt'),
                             str(member / 'best_checkpoint.pt'), map_location='cpu', restart_db=True)
    database = bundle['database']
    assert database.excited_state_properties == _properties(with_gap=True)
    assert {key: values['indices'].tolist() for key, values in database.splits.items()} == manifest['splits']
    model = bundle['training_modules'].model.to(torch.float64)
    e0, e1, f0, gap = [ml._target(model, k) for k in ('sE0', 'sE1', 'F0', 'dE01')]
    graph = GraphModule(model.input_nodes, [e0, e1, f0, gap])
    species = torch.tensor([[8, 1, 1]])
    xyz = torch.tensor(coordinates[:1], dtype=torch.float64, requires_grad=True)
    args = [species if node.db_name == 'species' else xyz for node in model.input_nodes]
    predictions = graph(*args)
    assert all(torch.isfinite(value).all() for value in predictions)
    torch.testing.assert_close(predictions[3], predictions[1] - predictions[0])
    step = 1e-4
    shifted = []
    for sign in (1, -1):
        x = xyz.detach().clone().requires_grad_(True)
        with torch.no_grad():
            x[0, 1, 0] += sign * step
        shifted.append(graph(*[species if node.db_name == 'species' else x for node in model.input_nodes])[0])
    numerical = -(shifted[0] - shifted[1]).item() / (2 * step)
    assert abs(numerical - predictions[2][0, 1, 0].item()) < 1e-3
    with pytest.raises(FileExistsError):
        ml.train_single_excited_state_model(model_id=0, model_dir=str(member), h5_train_dir=str(data),
                                            properties_list=_properties(with_gap=True), ML_config=config)
    ml.train_single_excited_state_model(model_id=1, model_dir=str(tmp_path/'model-01'),
                                        h5_train_dir=str(data), properties_list=_properties(with_gap=True),
                                        ML_config=config)
    from ase import Atoms
    evaluator = ml.load_hippynn_excited_state_ensemble(tmp_path, 'cpu', _properties(with_gap=True))
    atoms = Atoms('OHH', positions=coordinates[0])
    atoms.set_array('topology_atom_ids', np.arange(3))
    other = atoms.copy(); other.positions[1, 0] += .02
    packed = evaluator.predict_states([atoms, other])
    assert packed['state_energies'].shape == (2, 2, 2)
    assert packed['state_forces'][0].shape == (2, 2, 3, 3)
    for index, a in enumerate([atoms, other]):
        single = evaluator.predict_states([a])
        np.testing.assert_allclose(single['state_energies'][:, :, 0],
                                   packed['state_energies'][:, :, index], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(single['state_forces'][0], packed['state_forces'][index], rtol=1e-5, atol=1e-5)
    order = [2, 0, 1]
    permuted = evaluator.predict_states([atoms[order]])
    np.testing.assert_allclose(permuted['state_energies'], packed['state_energies'][:, :, :1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(permuted['state_forces'][0], packed['state_forces'][0][:, :, order], rtol=1e-5, atol=1e-5)
    scaled_schema = {k: [v[0], v[1], 2.] for k, v in _properties(with_gap=True).items()}
    scaled = ml.load_hippynn_excited_state_ensemble(tmp_path, 'cpu', scaled_schema).predict_states([atoms])
    np.testing.assert_allclose(scaled['state_energies']*2, packed['state_energies'][:, :, :1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scaled['state_forces'][0]*2, packed['state_forces'][0], rtol=1e-5, atol=1e-5)
    for graph, _ in evaluator._graphs:
        graph.to(torch.float64)
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[1, 0] += step; minus.positions[1, 0] -= step
    fd = -(evaluator.predict_states([plus])['state_energies'] -
           evaluator.predict_states([minus])['state_energies']) / (2*step)
    np.testing.assert_allclose(fd[:, :, 0], evaluator.predict_states([atoms])['state_forces'][0][:, :, 1, 0], atol=1e-3)
    periodic = atoms.copy(); periodic.pbc = True
    with pytest.raises(ValueError, match='nonperiodic'):
        evaluator.predict_states([periodic])
    bad = atoms.copy(); bad.numbers[0] = 6
    with pytest.raises(ValueError, match='composition'):
        evaluator.predict_states([bad])
    bad = atoms.copy(); bad.arrays['topology_atom_ids'][:] = 0
    with pytest.raises(ValueError, match='permutation'):
        evaluator.predict_states([bad])


def test_multistate_devices_preserve_visibility(monkeypatch):
    fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-one,GPU-two')
    assert ml.excited_training_device('from_multiprocessing', 3, 4) == 'cuda:1'
    assert ml.excited_training_device('cuda:0', 3, 4) == 'cuda:0'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-one,GPU-two'
    with pytest.raises(ValueError, match='outside'):
        ml.excited_training_device('cuda:2', 0, 4)
    fake.cuda.device_count = lambda: 0
    with pytest.raises(RuntimeError, match='No GPU'):
        ml.excited_training_device('cuda', 0, 4)
    assert ml.excited_training_device('cpu', 0, 0) == 'cpu'


def test_multistate_partial_failure_and_pool_cleanup(tmp_path, monkeypatch):
    def worker_result(params):
        return dict(ok=params['model_id'] == 0, payload=dict(model_dir=params['model_dir'],
                                                             model_id=params['model_id'], error='bad member'))
    FakePool, pool = fake_training_pool(member_log=lambda index: 'Training complete' if index == 0 else None,
                                        result=worker_result)
    monkeypatch.setattr(ml.multiprocessing, 'get_context', lambda kind: SimpleNamespace(Pool=FakePool))
    completed, identity = ml.train_excited_state_ensemble(ML_config=_config(), h5_dir='unused',
                            model_path=str(tmp_path / 'ensemble-{:d}'), current_training_id=3,
                            gpus_per_node=0, properties_list=_properties())
    assert completed == [True, False] and identity == 3
    assert pool.processes == 1
    assert pool.calls == ['close', 'join']
    assert json.loads((tmp_path / 'ensemble-3/model-01/training_error.json').read_text())['error'] == 'bad member'


def test_multistate_reader_restores_identical_element_identity(tmp_path):
    pytest.importorskip('hippynn')
    _write_group(tmp_path / 'data.h5', 'HH', species=('H', 'H'), count=3)
    with h5py.File(tmp_path / 'data.h5', 'a') as h:
        g = h['HH']
        g['topology_atom_ids'] = [[0, 1], [1, 0], [0, 1]]
        positions = g['coordinates'][:]
        forces = np.arange(18.).reshape(3, 2, 3)
        g['F1'][...] = forces
    arrays, _ = ml.load_excited_state_h5_arrays(str(tmp_path), _properties())
    np.testing.assert_array_equal(arrays['coordinates'][1], positions[1, ::-1])
    np.testing.assert_array_equal(arrays['F1'][1], forces[1, ::-1])

def _properties(with_gap=False):
    properties = {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
    }
    if with_gap:
        properties["dE01"] = ["dE01", "system", 1.0]
    return properties


def _config(**overrides):
    config = {
        "n_models": 2,
        "cell_key": None,
        "train_forces": True,
        "network_choice": 0,
        "valid_size": 0.2,
        "test_size": 0.2,
        "device_string": "cpu",
        "exports": {"pdf": False, "png": False, "csv": False},
    }
    config.update(overrides)
    return config


def _write_group(
    path: Path,
    name: str,
    *,
    species=("H", "O"),
    count=3,
    missing=None,
    nonfinite=None,
):
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as store:
        group = store.create_group(name)
        values = {
            "species": np.asarray(species, dtype="S2"),
            "coordinates": np.arange(
                count * len(species) * 3,
                dtype=float,
            ).reshape(count, len(species), 3),
            "sE0": np.linspace(-2.0, -1.0, count),
            "F0": np.zeros((count, len(species), 3)),
            "sE1": np.linspace(-1.5, -0.5, count),
            "F1": np.ones((count, len(species), 3)),
            "dE01": np.full(count, 0.5),
        }
        if nonfinite is not None:
            values[nonfinite] = np.asarray(values[nonfinite]).copy()
            values[nonfinite].reshape(-1)[0] = np.nan
        for key, value in values.items():
            if key != missing:
                group.create_dataset(key, data=value)


def test_explicit_h5_loader_reads_two_states_and_small_molecules(tmp_path):
    pytest.importorskip('hippynn')
    shard = tmp_path / "data-0000.h5"
    _write_group(shard, "HO", count=4)

    arrays, summary = ml.load_excited_state_h5_arrays(
        str(tmp_path),
        _properties(),
        configured_n_atoms=2,
        configured_possible_species=[0, 1, 8],
    )

    assert set(arrays) == {"coordinates", "species", "sE0", "F0", "sE1", "F1"}
    assert arrays["coordinates"].shape == (4, 2, 3)
    assert arrays["species"].shape == (4, 2)
    assert arrays["species"].dtype == np.int64
    assert arrays["sE0"].dtype == np.float32
    assert arrays["F1"].shape == (4, 2, 3)
    assert summary["n_structures"] == 4
    assert summary["n_atoms"] == 2
    assert summary["atomic_numbers"] == [1, 8]
    assert [row["energy_db_name"] for row in summary["state_table"]] == [
        "sE0",
        "sE1",
    ]


def test_explicit_h5_loader_uses_configured_database_names(tmp_path):
    pytest.importorskip('hippynn')
    properties = {
        "sE0": ["state_zero_energy", "system", 1.0],
        "F0": ["state_zero_force", "atomic", 1.0],
    }
    shard = tmp_path / "data.h5"
    with h5py.File(shard, "w") as store:
        group = store.create_group("H2")
        group["species"] = np.asarray(["H", "H"], dtype="S2")
        group["coordinates"] = np.zeros((3, 2, 3))
        group["state_zero_energy"] = np.arange(3, dtype=float)
        group["state_zero_force"] = np.zeros((3, 2, 3))

    arrays, summary = ml.load_excited_state_h5_arrays(
        str(shard),
        properties,
    )

    assert set(arrays) == {
        "coordinates",
        "species",
        "state_zero_energy",
        "state_zero_force",
    }
    assert summary["state_table"][0]["energy_key"] == "sE0"
    assert summary["state_table"][0]["energy_db_name"] == "state_zero_energy"


def test_explicit_h5_loader_reads_and_validates_gap_targets(tmp_path):
    pytest.importorskip('hippynn')
    _write_group(tmp_path / "data.h5", "HO")

    arrays, summary = ml.load_excited_state_h5_arrays(
        str(tmp_path),
        _properties(with_gap=True),
    )

    np.testing.assert_allclose(arrays["dE01"], arrays["sE1"] - arrays["sE0"])
    assert summary["gap_table"][0]["gap_key"] == "dE01"


def test_explicit_h5_loader_rejects_inconsistent_gap_targets(tmp_path):
    pytest.importorskip('hippynn')
    path = tmp_path / "data.h5"
    _write_group(path, "HO")
    with h5py.File(path, "a") as store:
        store["HO"]["dE01"][:] = 0.25

    with pytest.raises(ValueError, match="inconsistent"):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(with_gap=True),
        )


@pytest.mark.parametrize(
    ("problem", "with_gap", "message"),
    [
        ({"missing": "dE01"}, True, "missing required datasets"),
        ({"nonfinite": "dE01"}, True, "nonfinite"),
        ({"missing": "F1"}, False, "F1"),
        ({"nonfinite": "coordinates"}, False, "nonfinite"),
        ({"nonfinite": "sE0"}, False, "nonfinite"),
        ({"nonfinite": "F1"}, False, "nonfinite"),
    ],
    ids=["gap_missing", "gap_nonfinite", "force_missing", "coordinates_nonfinite", "energy_nonfinite",
         "force_nonfinite"],
)
def test_explicit_h5_loader_rejects_bad_datasets(tmp_path, problem, with_gap, message):
    pytest.importorskip('hippynn')
    _write_group(tmp_path / "data.h5", "HO", **problem)

    with pytest.raises((KeyError, ValueError), match=message):
        ml.load_excited_state_h5_arrays(str(tmp_path), _properties(with_gap=with_gap))


def test_explicit_h5_loader_rejects_mixed_atomic_ordering(tmp_path):
    pytest.importorskip('hippynn')
    shard = tmp_path / "data.h5"
    _write_group(shard, "first", species=("H", "O"))
    _write_group(shard, "second", species=("O", "H"))

    with pytest.raises(ValueError, match="exact atomic-number sequence"):
        ml.load_excited_state_h5_arrays(str(tmp_path), _properties())


def test_explicit_h5_loader_validates_configured_shape_and_species(tmp_path):
    pytest.importorskip('hippynn')
    _write_group(tmp_path / "data.h5", "HO")

    with pytest.raises(ValueError, match="n_atoms=3"):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(),
            configured_n_atoms=3,
        )
    with pytest.raises(ValueError, match="missing atomic numbers"):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(),
            configured_possible_species=[0, 1],
        )
    with pytest.raises(ValueError, match="begin with padding"):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(),
            configured_possible_species=[1, 8],
        )


@pytest.mark.parametrize("count", [0, 2, 4])
def test_filtered_split_capacity_rejects_insufficient_survivors(count):
    with pytest.raises(ValueError, match="too few structures"):
        ml.validate_filtered_split_capacity(
            count,
            test_size=0.2,
            valid_size=0.2,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"cell_key": "cell"}, "nonperiodic"),
        ({"train_forces": False}, "train_forces=true"),
        ({"export_force_gradients": False}, "export_force_gradients=true"),
        ({"network_choice": 2}, "network_choice"),
        ({"optimizer": "SGD"}, "AdamW"),
        ({"gap_targets": {"enabled": True}}, "gap targets"),
        ({"exports": {"csv": True}}, "CSV"),
        ({"remove_high_energy_cut": -1.0}, "remove_high_energy_cut"),
        ({"remove_high_forces_std": np.inf}, "remove_high_forces_std"),
    ],
)
def test_training_config_rejects_unsupported_features(override, message):
    with pytest.raises((ValueError, TypeError), match=message):
        ml.validate_excited_state_training_config(
            _config(**override),
            _properties(),
        )


def test_training_properties_require_energy_force_kinds_and_unique_names():
    wrong_kind = _properties()
    wrong_kind["F0"] = ["F0", "system", 1.0]
    with pytest.raises(ValueError, match="atomic property"):
        ml.validate_excited_state_training_properties(wrong_kind)

    duplicate_name = _properties()
    duplicate_name["F0"] = ["sE0", "atomic", 1.0]
    with pytest.raises(ValueError, match="unique HDF5"):
        ml.validate_excited_state_training_properties(duplicate_name)


def test_loss_matches_fork_formula_exactly():
    actual = ml.compose_excited_state_loss(
        [(2.0, 1.0), (4.0, 3.0)],
        [(6.0, 5.0), (8.0, 7.0)],
        9.0,
        n_atoms=2,
        energy_weight=2.0,
        force_weight=3.0,
        l2_weight=0.5,
    )
    expected = (
        2.0 * ((2.0 + 1.0) + (4.0 + 3.0))
        + 3.0 * ((6.0 + 5.0) + (8.0 + 7.0)) / np.sqrt(6.0)
        + 0.5 * 9.0
    )
    assert actual == pytest.approx(expected)


def test_loss_adds_weighted_derived_gap_terms():
    actual = ml.compose_excited_state_loss(
        [(2.0, 1.0)],
        [(6.0, 5.0)],
        9.0,
        gap_error_terms=[(4.0, 3.0)],
        n_atoms=2,
        energy_weight=2.0,
        force_weight=3.0,
        gap_weight=0.25,
        l2_weight=0.5,
    )
    expected = (
        2.0 * (2.0 + 1.0)
        + 3.0 * (6.0 + 5.0) / np.sqrt(6.0)
        + 0.25 * (4.0 + 3.0)
        + 0.5 * 9.0
    )
    assert actual == pytest.approx(expected)


def _small_shared_network():
    from hippynn.graphs import inputs

    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    positions.requires_grad = True
    network = ml._build_shared_network(
        species,
        positions,
        network_choice=0,
        network_params={
            "possible_species": [0, 1],
            "n_features": 4,
            "n_sensitivities": 4,
            "dist_soft_min": 0.5,
            "dist_soft_max": 2.0,
            "dist_hard_max": 2.5,
            "n_interaction_layers": 1,
            "n_atom_layers": 1,
            "sensitivity_type": "inverse",
            "resnet": True,
        },
    )
    return network, positions


def test_graph_uses_one_shared_trunk_and_all_state_database_names():
    hippynn = pytest.importorskip("hippynn")
    network, positions = _small_shared_network()
    table = ml.validate_excited_state_training_config(
        _config(),
        _properties(),
    )
    energy_outputs, force_outputs, total_loss, validation = (
        ml.build_excited_state_training_graph(
            network,
            positions,
            table,
            n_atoms=2,
            energy_weight=1.0,
            force_weight=1.0,
            l2_weight=2.0e-5,
        )
    )
    _, db_info = hippynn.experiment.assemble_for_training(
        total_loss,
        validation,
    )

    assert [output.db_name for _, output in energy_outputs] == ["sE0", "sE1"]
    assert [output.db_name for _, output in force_outputs] == ["F0", "F1"]
    assert set(db_info["inputs"]) == {"species", "coordinates"}
    assert {"sE0", "F0", "sE1", "F1"}.issubset(db_info["targets"])
    assert all(
        network in output.parents[0].parents
        for _, output in energy_outputs
    )


def test_graph_exposes_derived_gap_target_without_independent_head():
    hippynn = pytest.importorskip("hippynn")
    from alframework.tools.excited_state_tools import derive_gap_property_table

    network, positions = _small_shared_network()
    properties = _properties(with_gap=True)
    state_table = ml.validate_excited_state_training_config(
        _config(
            gap_targets={
                "enabled": True,
                "pairs": [[0, 1]],
                "weight": 0.5,
            }
        ),
        properties,
    )
    gap_table = derive_gap_property_table(
        properties,
        gap_config={"pairs": [[0, 1]]},
        require_properties=True,
    )
    _, _, total_loss, validation = ml.build_excited_state_training_graph(
        network,
        positions,
        state_table,
        gap_table=gap_table,
        n_atoms=2,
        energy_weight=1.0,
        force_weight=1.0,
        gap_weight=0.5,
        l2_weight=2.0e-5,
    )
    _, db_info = hippynn.experiment.assemble_for_training(
        total_loss,
        validation,
    )

    assert "dE01" in db_info["targets"]
    assert {"dE01_RMSE", "dE01_MAE"}.issubset(validation)


@pytest.mark.parametrize(
    ("gap_targets", "message"),
    [
        ({"enabled": True, "mode": "head", "pairs": [[0, 1]]}, "derived"),
        ({"enabled": True, "pairs": [[0, 1]], "weight": -1.0}, "weight"),
    ],
    ids=["independent_head", "negative_weight"],
)
def test_gap_training_config_rejects_independent_heads_and_bad_weight(gap_targets, message):
    with pytest.raises(ValueError, match=message):
        ml.validate_excited_state_training_config(_config(gap_targets=gap_targets), _properties(with_gap=True))


def test_model_member_seeds_are_deterministic():
    config = {"random_seed": 42}
    assert [ml.model_member_seed(config, index) for index in range(3)] == [
        42,
        43,
        44,
    ]


def test_task_signature_and_destructive_options(tmp_path, monkeypatch):
    assert ml.train_excited_state_HIPPYNN_ensemble_task.executors == [
        "alf_ML_executor"
    ]
    with pytest.raises(ValueError, match="remove_existing=True"):
        ml.validate_excited_state_task_options(
            remove_existing=True,
            h5_test_dir=None,
        )
    with pytest.raises(ValueError, match="h5_test_dir"):
        ml.validate_excited_state_task_options(
            remove_existing=False,
            h5_test_dir="separate-test-data",
        )


def test_nacr_datasets_in_a_shard_are_never_loaded_for_training(tmp_path, monkeypatch):
    """A shard labeled with coupling trains exactly as an E/F-only shard does.

    NACR is stored so a dataset need not be relabeled later, but the MLIP must
    still learn energies and forces only. That holds because the loader reads
    only the names derived from the sE#/F#/dE# patterns, so the extra datasets
    are invisible rather than merely unused.
    """
    pytest.importorskip('hippynn')

    plain = tmp_path / "plain"
    with_nacr = tmp_path / "with_nacr"
    plain.mkdir()
    with_nacr.mkdir()
    _write_group(plain / "data-0000.h5", "HO", count=4)
    _write_group(with_nacr / "data-0000.h5", "HO", count=4)

    # Add coupling datasets exactly as store_current_data would.
    with h5py.File(with_nacr / "data-0000.h5", "a") as store:
        store["HO"].create_dataset("nacr", data=np.ones((4, 1, 2, 3)))
        store["HO"].create_dataset(
            "nac_pairs", data=np.asarray([[1, 2]], dtype=np.int64)
        )

    loader_kwargs = {
        "configured_n_atoms": 2,
        "configured_possible_species": [0, 1, 8],
    }
    original_read = h5py.Dataset.__getitem__
    def guarded_read(dataset, key):
        assert not dataset.name.endswith(('/nacr', '/nac_pairs'))
        return original_read(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded_read)
    plain_arrays, plain_summary = ml.load_excited_state_h5_arrays(
        str(plain), _properties(), **loader_kwargs
    )
    nacr_arrays, nacr_summary = ml.load_excited_state_h5_arrays(
        str(with_nacr), _properties(), **loader_kwargs
    )

    assert set(nacr_arrays) == set(plain_arrays)
    assert "nacr" not in nacr_arrays
    assert "nac_pairs" not in nacr_arrays
    for key, values in plain_arrays.items():
        np.testing.assert_allclose(nacr_arrays[key], values)
    assert nacr_summary["n_structures"] == plain_summary["n_structures"]
    assert nacr_summary["state_table"] == plain_summary["state_table"]



"""Compare training selection to the upstream HIPPYNN implementation."""
import hashlib

import h5py
import numpy as np
import pytest

from alframework.ml_interfaces import hippynn_interface as ml


PROPERTIES = {'sE0': ['sE0', 'system', 1.], 'F0': ['F0', 'atomic', 1.]}
TABLE = ml.validate_excited_state_training_properties(PROPERTIES)


def arrays(count=12):
    coordinates = np.zeros((count,2,3), dtype=np.float64)
    coordinates[:,1,0] = 1.
    return {'coordinates': coordinates,
            'species': np.tile([1,1], (count,1)).astype(np.int64),
            'sE0': np.full(count, -2.), 'F0': np.full((count,2,3),10.),
            'indices': np.arange(count), 'frame_ids': 500 + np.arange(count)}


def upstream_selection(data, options):
    torch = pytest.importorskip('torch')
    Database = pytest.importorskip('hippynn.databases.database').Database
    values = {key: torch.as_tensor(value.copy()) for key,value in data.items()}
    values['sE0'] = values['sE0'].reshape(-1,1)
    database = Database(values, inputs=['species','coordinates'], targets=['sE0','F0'],
                        seed=42, quiet=True)
    database.remove_high_property('F0', True, species_key='species',
                                  cut=options.get('remove_high_forces_cut'),
                                  std_factor=options.get('remove_high_forces_std'))
    database.remove_high_property('sE0', False, species_key='species',
                                  cut=options.get('remove_high_energy_cut'),
                                  std_factor=options.get('remove_high_energy_std'))
    return database.arr_dict['indices'].cpu().numpy()


def compare(data, options):
    before = {key: value.copy() for key,value in data.items()}
    expected = upstream_selection(data, options)
    filtered, summary = ml.filter_excited_state_training_arrays(
        data, TABLE, coordinates_key='coordinates', dist_soft_min=.5, **options)
    np.testing.assert_array_equal(filtered['indices'], expected)
    for key,value in before.items():
        np.testing.assert_array_equal(data[key], value)  # Source arrays are unchanged.
        np.testing.assert_array_equal(filtered[key], value[expected])
    return filtered, summary


@pytest.mark.parametrize('options', [
    {'remove_high_forces_cut':40., 'remove_high_energy_cut':4.},
    {'remove_high_forces_std':2., 'remove_high_energy_std':2.},
    {'remove_high_forces_cut':40., 'remove_high_forces_std':2.,
     'remove_high_energy_cut':4., 'remove_high_energy_std':2.},
    {'remove_high_forces_cut':None, 'remove_high_forces_std':None},
])
def test_fixed_standardized_combined_and_disabled_filters_match_upstream(options):
    data = arrays()
    data['F0'][-1] = 110.
    data['F0'][-2] = 20.
    data['sE0'][-3] = 8.
    compare(data, options)


@pytest.mark.parametrize('cut, expected', [(2., list(range(6))),
                                         (np.nextafter(2.,0.), list(range(2,6)))])
def test_nonzero_mean_and_exact_component_cutoff(cut, expected):
    data = arrays(6)
    data['F0'][0], data['F0'][1] = 8., 12.
    filtered, _ = compare(data, {'remove_high_forces_cut':cut})
    assert filtered['indices'].tolist() == expected


@pytest.mark.parametrize('options', [{'remove_high_forces_cut':0.}, {'remove_high_forces_std':0.},
                                    {'remove_high_forces_cut':0.,'remove_high_forces_std':2.}])
def test_zero_variance_large_forces_are_not_norm_filtered(options):
    data = arrays()
    # Every vector has a large norm, but every component equals the mean.
    data['F0'][:] = 1000.
    filtered, _ = compare(data, options)
    assert len(filtered['indices']) == 12


def test_std_statistics_are_recomputed_after_fixed_cut():
    data = arrays()
    data['F0'][-1], data['F0'][-2] = 110.,20.
    filtered, summary = compare(data, {'remove_high_forces_cut':40., 'remove_high_forces_std':2.})
    assert filtered['indices'].tolist() == list(range(10))
    assert summary['static_cut_failures']['F0'] == 1
    assert summary['standard_deviation_failures']['F0'] == 1


def test_h5_training_force_filter_keeps_original_store(tmp_path):
    pytest.importorskip('hippynn')
    path = tmp_path/'data-0000.h5'
    data = arrays()
    data['F0'][-1] = 110.
    data['F0'][-2] = 20.
    with h5py.File(path,'w') as handle:
        group = handle.create_group('H02')
        for key in ('coordinates','sE0','F0'):
            group[key] = data[key]
        group['species'] = np.array([b'H',b'H'])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, _ = ml.load_excited_state_h5_arrays(str(tmp_path), PROPERTIES)
    filtered, _ = ml.filter_excited_state_training_arrays(
        loaded, TABLE, coordinates_key='coordinates', dist_soft_min=.5,
        remove_high_forces_cut=40., remove_high_forces_std=2.)
    assert len(filtered['coordinates']) == 10
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def multistate_arrays():
    data = arrays(20)
    data['sE1'] = data['sE0'] + .5
    data['F1'] = data['F0'].copy()
    data['dE01'] = data['sE1'] - data['sE0']
    properties = {**PROPERTIES, 'sE1': ['sE1', 'system', 1.],
                  'F1': ['F1', 'atomic', 1.], 'dE01': ['dE01', 'system', 1.]}
    return data, ml.validate_excited_state_training_properties(properties)


@pytest.mark.parametrize('options', [
    {'remove_high_forces_cut': 5., 'remove_high_energy_cut': 5.},
    {'remove_high_forces_std': 2., 'remove_high_energy_std': 2.},
    {'remove_high_forces_cut': 5., 'remove_high_forces_std': 2.,
     'remove_high_energy_cut': 5., 'remove_high_energy_std': 2.},
    {'remove_high_forces_cut': None, 'remove_high_forces_std': None},
], ids=['fixed', 'standardized', 'combined', 'disabled'])
def test_multistate_union_is_order_independent_and_preserves_all_rows(options):
    data, table = multistate_arrays()
    data['sE1'][16] = 20.
    data['F0'][17] = 30.
    data['F1'][18] = -10.
    data['dE01'] = data['sE1'] - data['sE0']
    before = {key: value.copy() for key, value in data.items()}
    forward, summary = ml.filter_excited_state_training_arrays(
        data, table, coordinates_key='coordinates', dist_soft_min=.5, **options)
    reverse, _ = ml.filter_excited_state_training_arrays(
        data, list(reversed(table)), coordinates_key='coordinates', dist_soft_min=.5, **options)
    expected = [i for i in range(20) if i not in (16, 17, 18)] if any(
        value is not None for value in options.values()) else list(range(20))
    assert forward['indices'].tolist() == expected
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)
        np.testing.assert_array_equal(forward[key], reverse[key])
        np.testing.assert_array_equal(forward[key], value[expected])
    np.testing.assert_array_equal(forward['dE01'], forward['sE1'] - forward['sE0'])
    if len(expected) < len(data['indices']):
        stage = 'static_cut' if options.get('remove_high_forces_cut') is not None else 'standard_deviation'
        assert summary['per_state_failures']['0']['forces'][stage] == 1
        assert summary['per_state_failures']['1']['forces'][stage] == 1
        assert summary['per_state_failures']['1']['energy'][stage] == 1
        assert summary['removed_static_cut_union'] + summary['removed_standard_deviation_union'] == 3
    assert summary['final_structures'] == len(expected)


def test_minimum_distance_is_applied_before_multistate_statistics():
    data, table = multistate_arrays()
    data['coordinates'][0, 1, 0] = .4
    data['sE1'][0] = 1000.
    data['dE01'] = data['sE1'] - data['sE0']
    filtered, summary = ml.filter_excited_state_training_arrays(
        data, table, coordinates_key='coordinates', dist_soft_min=.5,
        remove_high_energy_std=2.)
    expected = list(range(1, 20))
    for key, value in data.items():
        np.testing.assert_array_equal(filtered[key], value[expected])
    assert summary['removed_min_distance'] == 1
    assert summary['removed_standard_deviation_union'] == 0


def test_multistate_zero_variance_retains_all_rows():
    data, table = multistate_arrays()
    filtered, summary = ml.filter_excited_state_training_arrays(
        data, table, coordinates_key='coordinates', dist_soft_min=.5,
        remove_high_energy_std=2., remove_high_forces_std=2.)
    for key, value in data.items():
        np.testing.assert_array_equal(filtered[key], value)
    assert summary['final_structures'] == len(data['indices'])
