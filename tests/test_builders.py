"""Geometry builders: condensed-phase and salt builders, external geometry import, and RDKit seed
preparation.

Tests that call RDKit skip without it.
"""

import hashlib
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers
from ase.io import write

from alframework.builders import rdkit_molecule as seed
from alframework.builders.builders import (
    atomic_system_builder,
    construct_simulation_box,
    create_atomic_system,
    simple_condensed_phase_builder_task,
    simple_multi_condensed_phase_builder_task,
    to_mic,
)
from alframework.builders.external_geometry import (
    build_external_geometries,
    external_geometry_builder_task,
    geometry_manifest,
)
from alframework.tools.molecules_class import MoleculesObject


@pytest.mark.parametrize(
    "atom_charges,target_num_atoms",
    [
        ({"Na": 1, "Cl": -1}, 8),
        ({"Mg": 2, "Cl": -1}, 9),
        ({"Ca": 2, "F": -1, "O": -2}, 10),
    ],
    ids=["NaCl", "MgCl2", "CaFO"],
)
def test_atomic_system_is_charge_neutral(atom_charges, target_num_atoms):
    random.seed(1)

    system = create_atomic_system(atom_charges, target_num_atoms)

    total_charge = sum(atom_charges[atom] * count for atom, count in system.items())
    assert total_charge == 0
    assert sum(system.values()) >= target_num_atoms - 1
    assert set(system).issubset(atom_charges)


def test_distances_wrap_to_minimum_image():
    distances = np.array([[5.5, -5.5, 0.0], [1.0, 2.0, -6.0]])

    np.testing.assert_allclose(to_mic(distances, box_length=10.0), [[-4.5, 4.5, 0.0], [1.0, 2.0, 4.0]])


def test_box_respects_minimum_distance():
    np.random.seed(5)
    coords = construct_simulation_box(
        {"Na": 2, "Cl": 2},
        min_distance=1.0,
        box_length=8.0,
        scale_coords=False,
        max_iter=20,
        max_tries=5,
    )

    assert coords.shape == (4, 3)
    assert np.all(coords >= 0.0)
    assert np.all(coords < 8.0)
    for first, second in itertools.combinations(coords, 2):
        assert np.linalg.norm(to_mic(first - second, 8.0)) > 1.0


def test_box_can_return_scaled_coordinates():
    np.random.seed(5)
    coords = construct_simulation_box(
        {"Na": 1, "Cl": 1},
        min_distance=1.0,
        box_length=8.0,
        scale_coords=True,
    )

    assert coords.shape == (2, 3)
    assert np.all(coords >= 0.0)
    assert np.all(coords <= 1.0)


def test_overfull_box_is_rejected():
    with pytest.raises(ValueError, match="Density is too high"):
        construct_simulation_box({"Na": 10}, min_distance=2.0, box_length=3.0)


def test_salt_builder_returns_periodic_neutral_atoms():
    random.seed(2)
    np.random.seed(2)

    atom_charges = {"Na": 1, "Cl": -1}
    target_num_atoms = 4
    atoms = atomic_system_builder(atom_charges, target_num_atoms=target_num_atoms, min_distance=0.8, box_length=8.0)
    molecule = MoleculesObject(atoms, "salt")

    assert atoms.pbc.all()
    np.testing.assert_allclose(atoms.cell.lengths(), [8.0, 8.0, 8.0])
    assert len(atoms) >= target_num_atoms - 1
    assert len(molecule) == len(atoms)
    assert set(atoms.get_chemical_symbols()).issubset(atom_charges)
    assert sum(atom_charges[symbol] for symbol in atoms.get_chemical_symbols()) == 0


def test_condensed_phase_builders_return_periodic_systems():

    library = (
        str(Path(__file__).resolve().parents[1] / "examples" / "simple_water_multi_builder" / "fragment_library") + "/"
    )
    settings = {
        "molecule_library_dir": library,
        "cell_range": [[6.0, 6.0]] * 3,
        "solute_molecule_options": [["water.xyz"]],
        "Rrange": [0.5, 0.5],
    }
    builder_config = {"solvent_molecules": {"water.xyz": 1.0}, "min_dist": 1.25, "max_patience": 50}
    random.seed(3)
    np.random.seed(3)

    single = simple_condensed_phase_builder_task.func("single-0", builder_config, **settings)
    multi = simple_multi_condensed_phase_builder_task.func(["multi-0", "multi-1"], builder_config, **settings)

    assert single.get_atoms().pbc.all()
    assert [system.get_moleculeid() for system in multi] == ["multi-0", "multi-1"]
    for system in multi:
        assert system.get_atoms().pbc.all()


@pytest.fixture
def geometry_sources(tmp_path):
    atoms = Atoms("OHH", positions=[[0, 0, 0], [0.95, 0, 0], [0, 0.95, 0]], cell=np.eye(3) * 8, pbc=[True, False, True])
    atoms.set_array("topology_atom_ids", np.arange(3))
    atoms.calc = SinglePointCalculator(atoms, energy=-42.0, forces=np.ones((3, 3)))
    write(tmp_path / "source.xyz", atoms, format="extxyz")
    permutation = [0, 2, 1]
    with h5py.File(tmp_path / "source.h5", "w") as h:
        g = h.create_group("water")
        g["coordinates"] = [atoms.positions[permutation]]
        g["species"] = np.array(["O", "H", "H"], dtype="S")
        g["topology_atom_ids"] = [permutation]
        g["_id"] = np.array(["source-water"], dtype="S")
        g["energy"] = [-123.0]
        g["forces"] = np.ones((1, 3, 3))
    return tmp_path


def test_external_geometry_drops_old_labels(geometry_sources):
    config = dict(sources=["source.xyz", "source.h5"], selection_mode="sequential")
    items = external_geometry_builder_task.func(
        moleculeid="ignored",
        moleculeids=["new-0", "new-1"],
        builder_config=config,
        master_directory=str(geometry_sources),
    )
    assert [m.get_moleculeid() for m in items] == ["new-0", "new-1"]
    for item in items:
        assert item.get_results() == {} and item.converged is None
        assert item.atoms.calc is None and item.atoms.info == {}
        assert len(item.get_metadata()["geometry_source_sha256"]) == 64
    np.testing.assert_array_equal(items[0].atoms.pbc, [True, False, True])
    np.testing.assert_allclose(items[0].atoms.cell, np.eye(3) * 8)
    np.testing.assert_array_equal(items[1].atoms.arrays["topology_atom_ids"], [0, 2, 1])
    assert items[1].get_metadata()["geometry_source_id"] == "source-water"
    single = external_geometry_builder_task.func("new-1", builder_config=config, master_directory=str(geometry_sources))
    np.testing.assert_allclose(single.atoms.positions, items[1].atoms.positions)


def test_external_geometry_hash_selection_depends_only_on_id(geometry_sources):
    config = dict(sources=["source.xyz", "source.h5"], selection_seed=19)
    a = build_external_geometries(["a", "b", "c"], config, str(geometry_sources))
    b = build_external_geometries(["c", "a"], config, str(geometry_sources))
    for x, y in [(a[2], b[0]), (a[0], b[1])]:
        assert x.get_metadata() == y.get_metadata()
        np.testing.assert_array_equal(x.atoms.positions, y.atoms.positions)


@pytest.mark.parametrize(
    "ids,mode,error,reason",
    [
        (["new-2"], "sequential", IndexError, "exhausted"),
        (["new"], "sequential", ValueError, "trailing integer"),
        (["a", "a"], "deterministic_hash", ValueError, "Duplicate"),
        (["a"], "bad", ValueError, "selection_mode"),
    ],
    ids=["past_the_end", "id_without_number", "duplicate_ids", "unknown_mode"],
)
def test_external_geometry_rejects_invalid_selection(geometry_sources, ids, mode, error, reason):
    with pytest.raises(error, match=reason):
        build_external_geometries(
            ids, dict(sources=["source.xyz", "source.h5"], selection_mode=mode), str(geometry_sources)
        )


@pytest.mark.parametrize("bad_ids", [[0, 0, 2], [0.0, 1.0, 2.0]], ids=["duplicate_id", "float_ids"])
def test_external_geometry_rejects_invalid_topology(geometry_sources, bad_ids):
    with h5py.File(geometry_sources / "source.h5", "a") as h:
        del h["water/topology_atom_ids"]
        h["water/topology_atom_ids"] = [bad_ids]
    with pytest.raises(ValueError, match="integer permutation"):
        build_external_geometries(["a"], dict(sources=["source.h5"]), str(geometry_sources))


def test_plain_xyz_gets_no_invented_topology(tmp_path):
    (tmp_path / "plain.xyz").write_text("2\nplain molecule\nH 0 0 0\nH 0 0 .74\n")
    m = build_external_geometries(["a"], dict(sources=["plain.xyz"]), str(tmp_path))[0]
    assert "topology_atom_ids" not in m.atoms.arrays
    assert not m.atoms.pbc.any()


def test_external_geometry_rejects_empty_or_malformed_sources(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        geometry_manifest({})
    with h5py.File(tmp_path / "bad.h5", "w") as h:
        h["species"] = [1, 1]
        h["coordinates"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="frames, atoms, 3"):
        geometry_manifest(dict(sources=["bad.h5"]), str(tmp_path))


# ---- Preparing molecular seeds with RDKit (an optional builder capability) ----

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/AcAc_excited_pyseqm"


# ---- Portable excited-state examples and reusable CPPn preparation ----

CPP_EXAMPLE = ROOT / 'examples/CPP_excited_gpu4pyscf'


def load_example_prepare(folder):
    import importlib.util
    spec = importlib.util.spec_from_file_location('prepare_'+folder.name, folder/'prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_cpp(size):
    """A regular para-linked macrocycle, independent of shipped CPP6/9 files."""
    radius = (1.47+2*1.4*np.cos(np.pi/size))/(2*np.sin(np.pi/size))
    positions, symbols = [], []
    for index in range(size):
        angle = 2*np.pi*index/size
        radial = np.array([np.cos(angle), np.sin(angle), 0.])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.])
        for vertex in range(6):
            theta = np.pi*vertex/3
            unit = tangent*np.cos(theta)+np.array([0., 0., np.sin(theta)])
            positions.append(radius*radial+1.4*unit)
            symbols.append('C')
            if vertex not in (0, 3):
                positions.append(radius*radial+2.48*unit)
                symbols.append('H')
    return Atoms(symbols=symbols, positions=positions)


@pytest.mark.parametrize('size', [5, 6, 8, 9, 12])
def test_ring_sampler_infers_any_cpp_size(size):
    from alframework.builders import ring_molecule as ring
    atoms = synthetic_cpp(size)
    config = {**ring.DEFAULT_SETTINGS, 'count': 8}
    topology = ring.infer_topology(atoms, config['screening'])
    assert topology['nring'] == size
    first = ring.generate_samples(atoms, config, 8, 42)
    second = ring.generate_samples(atoms, config, 8, 42)
    np.testing.assert_array_equal(first['coordinates'], second['coordinates'])
    np.testing.assert_array_equal(np.bincount(first['tier_index']), [4, 4])
    assert first['coordinates'].shape == (8, size*10, 3)
    for frame in first['coordinates']:
        assert ring.screen_geometry(frame, topology, config['screening']) is None
    rotated = ring.rotate_phenyls(atoms.positions, topology, np.linspace(-30, 30, size))
    pairs = topology['bonds']
    np.testing.assert_allclose(np.linalg.norm(rotated[pairs[:, 0]]-rotated[pairs[:, 1]], axis=1), topology['bond_lengths'], atol=1e-13)


@pytest.mark.parametrize('size', [6, 9])
def test_cpp_example_can_move_and_replay_without_labels(tmp_path, size):
    from alframework.builders.ring_molecule import build_ring_molecules
    module = load_example_prepare(CPP_EXAMPLE)
    output = module.prepare(tmp_path/'original', ring_size=size, count=8, generate_seeds=True)
    moved = tmp_path/'moved'
    output.rename(moved)
    config = json.loads((moved/'builder_config.json').read_text())
    molecules = build_ring_molecules([f'mol-boot-{i:010d}' for i in range(8)], config, master_directory=moved)
    assert [m.get_moleculeid() for m in molecules] == [f'mol-boot-{i:010d}' for i in range(8)]
    for molecule in molecules:
        assert molecule.get_results() == {}
        assert not molecule.atoms.has('momenta')
        np.testing.assert_array_equal(molecule.atoms.arrays['topology_atom_ids'], np.arange(size*10))
        assert molecule.get_metadata()['ring_tier'] in (0, 1)
    with pytest.raises(IndexError, match='exhausted'):
        build_ring_molecules(['mol-boot-0000000008'], config, master_directory=moved)
    (moved/'h5store').mkdir()
    shutil.copyfile(moved/'seeds/seed.h5', moved/'h5store/data-0000.h5')
    with h5py.File(moved/'h5store/data-0000.h5', 'a') as handle:
        handle[f'cpp{size}']['sE0'] = np.ones(8)
    runtime = build_ring_molecules(['runtime-123'], config, 'h5store/data-{:04d}.h5', 1, moved)
    assert runtime[0].get_results() == {}
    assert not runtime[0].get_metadata()['replay_external_bootstrap']
    assert runtime[0].get_metadata()['geometry_source_id'].startswith(f'cpp{size}-seed-')
    (moved/'seeds/reference.xyz').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        build_ring_molecules(['mol-boot-0000000000'], config, master_directory=moved)


@pytest.mark.parametrize('model', ['hiphop145', 'hiphop64', 'hipnn64'])
@pytest.mark.parametrize('folder', [EXAMPLES, CPP_EXAMPLE], ids=['acac', 'cpp'])
def test_excited_example_preparation_only_and_model_profiles(tmp_path, model, folder):
    module = load_example_prepare(folder)
    output = module.prepare(tmp_path/'run', model=model)
    assert not (output/'seeds').exists()
    assert not (output/'status.json').exists()
    ml = json.loads((output/'hippynn_config.json').read_text())
    assert ml['n_models'] == 4
    assert ml['network_params']['n_features'] == (145 if model=='hiphop145' else 64)
    assert ml['network_choice'] == (0 if model=='hipnn64' else 1)
    assert ('l_max' in ml['network_params']) == (model != 'hipnn64')
    master = json.loads((output/'master_config.json').read_text())
    assert master['master_directory'] == 'pwd' and master['finite_bootstrap']
    for key in ('builder_config_path', 'sampler_config_path', 'QM_config_path', 'ML_config_path'):
        assert (output/master[key]).is_file() and not Path(master[key]).is_absolute()
    assert not json.loads((output/'preparation.json').read_text())['campaign_submitted']
    with pytest.raises(FileExistsError):
        module.prepare(output)


def test_acac_seed_preparation_remains_portable(tmp_path):
    pytest.importorskip('rdkit')
    module = load_example_prepare(EXAMPLES)
    output = module.prepare(tmp_path/'original', pool_size=24, seed_size=12, generate_seeds=True)
    moved = tmp_path/'moved'
    output.rename(moved)
    config = json.loads((moved/'builder_config.json').read_text())
    molecules = seed.build_rdkit_molecules(['mol-boot-0000000000'], config, master_directory=moved)
    assert len(molecules) == 1 and molecules[0].get_results() == {}
    assert json.loads((moved/'master_config.json').read_text())['bootstrap_set'] == 12
    for path in moved.rglob('*.json'):
        assert str(tmp_path) not in path.read_text()


def test_custom_cpp_reference_and_offset_are_explicit(tmp_path):
    module = load_example_prepare(CPP_EXAMPLE)
    reference = tmp_path/'cpp8.xyz'
    write(reference, synthetic_cpp(8))
    with pytest.raises(ValueError, match='explicit energy offset'):
        module.prepare(tmp_path/'bad', ring_size=8, reference=reference)
    with pytest.raises(ValueError, match='disagrees'):
        module.prepare(tmp_path/'bad', ring_size=9, reference=reference, energy_offset_ev=-123.)
    output = module.prepare(tmp_path/'cpp8', ring_size=8, reference=reference, energy_offset_ev=-123., count=4, generate_seeds=True)
    assert json.loads((output/'QM_config.json').read_text())['energy_offset_eV'] == -123.
    assert json.loads((output/'hippynn_config.json').read_text())['n_atoms'] == 80


def test_failed_ring_preparation_is_atomic(tmp_path, monkeypatch):
    from alframework.builders import ring_molecule as ring
    config = json.loads((CPP_EXAMPLE/'seed_cpp9.json').read_text())
    monkeypatch.setattr(ring, 'screen_geometry', lambda *args: 'test_rejection')
    with pytest.raises(ValueError, match='anchor fails screening'):
        ring.prepare_ring_pool(config, tmp_path/'failed', config_directory=CPP_EXAMPLE)
    assert not (tmp_path/'failed').exists()


def test_ring_attempt_exhaustion_publishes_nothing(tmp_path, monkeypatch):
    from alframework.builders import ring_molecule as ring
    config = json.loads((CPP_EXAMPLE/'seed_cpp9.json').read_text())
    config['count'] = 2
    config['screening']['max_attempts_per_structure'] = 2
    original_screen = ring.screen_geometry
    def reject_distortion(positions, topology, settings):
        if np.array_equal(positions, atoms.positions):
            return original_screen(positions, topology, settings)
        return 'deliberate_rejection'
    from ase.io import read
    atoms = read(CPP_EXAMPLE/'references/cpp9_s0.xyz')
    monkeypatch.setattr(ring, 'screen_geometry', reject_distortion)
    with pytest.raises(RuntimeError, match='Cannot fill.*deliberate_rejection'):
        ring.prepare_ring_pool(config, tmp_path/'failed', config_directory=CPP_EXAMPLE)
    assert not (tmp_path/'failed').exists()


def test_ring_builder_submits_plural_and_scalar_tasks(tmp_path):
    from alframework.builders.ring_molecule import ring_molecule_builder_task
    from tests.helpers.driver import loaded_parsl
    output = load_example_prepare(CPP_EXAMPLE).prepare(tmp_path/'run', ring_size=6, count=4, generate_seeds=True)
    config = json.loads((output/'builder_config.json').read_text())
    with loaded_parsl(tmp_path/'runinfo', 'alf_sampler_executor'):
        batch = ring_molecule_builder_task(moleculeid='ignored', moleculeids=[f'mol-boot-{i:010d}' for i in range(4)],
                                            builder_config=config, master_directory=str(output)).result()
        single = ring_molecule_builder_task(moleculeid='mol-boot-0000000000', builder_config=config,
                                             master_directory=str(output)).result()
    assert len(batch) == 4
    np.testing.assert_array_equal(single.atoms.positions, batch[0].atoms.positions)


@pytest.mark.parametrize('folder', [EXAMPLES, CPP_EXAMPLE], ids=['acac', 'cpp'])
def test_example_parsl_template_uses_site_settings_and_submits_nothing(tmp_path, monkeypatch, folder):
    import importlib.util
    from parsl.providers import SlurmProvider
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ALF_PARTITION', 'my_partition')
    monkeypatch.setenv('ALF_QM_PYTHON', '/my qm/bin/python')
    monkeypatch.setenv('ALF_QM_WORKER_INIT', 'export LD_LIBRARY_PATH=/my/cutensor/lib:$LD_LIBRARY_PATH')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    def forbidden(*args, **kwargs):
        raise AssertionError('Template import must not submit')
    monkeypatch.setattr(SlurmProvider, 'submit', forbidden)
    spec = importlib.util.spec_from_file_location('portable_parsl', folder/'parsl_configs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for executor in module.config.executors:
        assert executor.provider.init_blocks == 0
        assert executor.provider.partition == 'my_partition'
        assert executor.provider.worker_init.endswith('\n')
        assert 'CUDA_VISIBLE_DEVICES' not in executor.provider.worker_init
    qm = next(e for e in module.config.executors if e.label=='alf_QM_executor')
    assert qm.interchange_launch_cmd[0] == '/my qm/bin/python'
    assert 'LD_LIBRARY_PATH=/my/cutensor/lib' in qm.provider.worker_init
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'


@pytest.mark.parametrize('size', [6, 9])
def test_cpp_example_contracts_and_topology_preflight(tmp_path, monkeypatch, size):
    from alframework.ml_interfaces.hippynn_interface import validate_excited_state_training_config
    from alframework.qm_interfaces.gpu4pyscf_interface import validate_gpu4pyscf_config, validate_gpu4pyscf_properties
    from alframework.tools.molecular_topology import load_reference_topology, validate_fixed_topology
    from alframework.tools.tools import load_config_file
    from alframework.builders.ring_molecule import build_ring_molecules
    pytest.importorskip('rdkit')
    output = load_example_prepare(CPP_EXAMPLE).prepare(tmp_path/'run', ring_size=size, count=4, generate_seeds=True)
    monkeypatch.chdir(output)
    master = load_config_file('master_config.json')
    configs = {role:load_config_file(master[f'{role}_config_path'], master['master_directory']) for role in ('builder','sampler','QM','ML')}
    assert len(validate_excited_state_training_config(configs['ML'], master['properties_list'])) == 4
    qm = validate_gpu4pyscf_config(configs['QM'])
    validate_gpu4pyscf_properties(master['properties_list'], nroots=qm['nroots'], compute_nacr=True)
    reference = load_reference_topology(configs['sampler'], master_directory=master['master_directory'])
    molecules = build_ring_molecules(['mol-boot-0000000000'], configs['builder'], master_directory=output)
    assert validate_fixed_topology(molecules[0].atoms, reference).valid
    metadata = json.loads((output/'references'/f'cpp{size}_s0.json').read_text())
    assert qm['energy_offset_eV'] == metadata['ground_energy_hartree']*metadata['hartree_to_ev']
    assert configs['ML']['batch_size'] == configs['ML']['eval_batch_size'] == configs['ML']['scheduler_options']['max_batch_size'] == 8
    assert configs['ML']['fraction_train_eval'] == .1


def example_config(name):
    return json.loads((EXAMPLES / f"seed_{name}.json").read_text())


def test_rdkit_fps_selection_is_deterministic():
    assert seed.ladder_quotas(50000, 6) == [8334, 8334, 8333, 8333, 8333, 8333]
    assert seed.ladder_quotas(5000, 6) == [834, 834, 833, 833, 833, 833]
    assert [seed.stratum_seed(42, level, 0) for level in range(6)] == [
        438055385,
        1262664333,
        1985621771,
        1596702514,
        702239142,
        649246379,
    ]
    features = np.arange(4, dtype=float)[:, None]
    np.testing.assert_array_equal(seed.farthest_point_order(features, 4), [1, 3, 0, 2])
    np.testing.assert_array_equal(seed.farthest_point_order(np.zeros((4, 2)), 4), np.arange(4))
    coordinates = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    numbers = np.array([8, 1, 1])
    descriptor = seed.pair_descriptor(coordinates, numbers)
    np.testing.assert_allclose(descriptor, [[np.sqrt(5), 1.0, 2.0]])
    order = [2, 0, 1]
    np.testing.assert_array_equal(seed.pair_descriptor(coordinates[:, order] + 10.0, numbers[order]), descriptor)
    with pytest.raises(ValueError, match="FPS"):
        seed.farthest_point_order(features, 5)


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"seed_size": 50001}, "seed_size"),
        ({"threads": True}, "threads"),
        ({"bond_scales": [0.0, float("nan")]}, "bond_scales"),
        ({"angle_scales": [0.0]}, "same length"),
        ({"geometry_screen": {"typo": 1.0}}, "geometry_screen"),
    ],
    ids=["seed_larger_than_pool", "boolean_threads", "nan_scale", "mismatched_ladders", "unknown_screen_setting"],
)
def test_rdkit_rejects_invalid_seed_settings(override, reason):
    with pytest.raises(ValueError, match=reason):
        seed.resolve_config({**example_config("keto"), **override}, base_directory=EXAMPLES)


def test_rdkit_geometry_screen_reports_the_right_failure():
    context = {
        "numbers": np.array([1, 1]),
        "bonds": np.array([[0, 1]]),
        "bond_centers": np.array([1.0]),
        "pairs": np.array([[0, 1]]),
        "nonbonded": np.empty((0, 2), dtype=int),
    }
    screen = seed.DEFAULT_SETTINGS["geometry_screen"]
    for distance in (0.85, 1.0, 1.50):
        assert seed.geometry_failure([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], context, screen) is None
    assert seed.geometry_failure([[0.0, 0.0, 0.0], [0.849, 0.0, 0.0]], context, screen) == "bond_window"
    assert seed.geometry_failure([[0.0, 0.0, 0.0], [1.501, 0.0, 0.0]], context, screen) == "bond_window"
    assert seed.geometry_failure(np.full((2, 3), np.nan), context, screen) == "non_finite"
    loose = {**screen, "bond_min_scale": 0.1}
    assert seed.geometry_failure([[0.0, 0.0, 0.0], [0.69, 0.0, 0.0]], context, loose) == "min_distance"
    context.update(
        numbers=np.array([1, 1, 1]), pairs=np.array([[0, 1], [0, 2], [1, 2]]), nonbonded=np.array([[0, 2], [1, 2]])
    )
    assert (
        seed.geometry_failure([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.1, 0.0]], context, screen)
        == "inferred_extra_bond"
    )
    assert (
        seed.geometry_failure([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.82, 0.0]], context, screen)
        == "nonbonded_close"
    )


def test_rdkit_polar_hydrogen_bounds_are_wider(tmp_path):
    pytest.importorskip("rdkit")
    reference = tmp_path / "water.xyz"
    reference.write_text("3\nwater\nO 0 0 0\nH .96 0 0\nH -.24 .93 0\n")
    config = seed.resolve_config({"smiles": "O", "reference_geometry": str(reference)})
    context = seed._reference_context(config)
    polar = seed.widened_bounds_matrix(context["molecule"], context["centers"], config, 5)
    ordinary = seed.widened_bounds_matrix(context["molecule"], context["centers"], {**config, "polar_h": False}, 5)
    center = context["centers"][0, 1]
    assert polar[0, 1] == pytest.approx(center * 1.1)
    assert polar[1, 0] == pytest.approx(center * 0.9)
    assert polar[0, 1] > ordinary[0, 1]
    assert polar[1, 0] < ordinary[1, 0]


def test_rdkit_rejects_same_elements_with_different_topology():
    pytest.importorskip("rdkit")
    config = {**example_config("keto"), "reference_geometry": "enol.xyz"}
    with pytest.raises(ValueError, match="connectivity|bond set"):
        seed._reference_context(seed.resolve_config(config, base_directory=EXAMPLES))


@pytest.fixture(scope="module", params=["enol", "keto"])
def prepared_seed(request, tmp_path_factory):
    """A small prepared seed, generated twice as two separate runs would."""
    pytest.importorskip("rdkit")
    config = {**example_config(request.param), "pool_size": 24, "seed_size": 12}
    directory = tmp_path_factory.mktemp(request.param)
    output, repeated = directory / "first", directory / "second"
    report = seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    seed.prepare_rdkit_seed(config, repeated, config_directory=EXAMPLES)
    return SimpleNamespace(report=report, output=output, repeated=repeated)


def test_rdkit_seed_preparation_records_provenance(prepared_seed):
    report = prepared_seed.report
    assert report["pool_count"] == 24 and report["selected_count"] == 12
    assert report["embedding_settings"]["useBasicKnowledge"] is True
    assert report["embedding_settings"]["useExpTorsionAnglePrefs"] is False
    assert report["embedding_settings"]["force_field_optimization"] is False
    for file, digest in report["artifacts_sha256"].items():
        assert hashlib.sha256((prepared_seed.output / file).read_bytes()).hexdigest() == digest


def test_rdkit_seed_preparation_is_reproducible(prepared_seed):
    with h5py.File(prepared_seed.output / "seed.h5") as first, h5py.File(prepared_seed.repeated / "seed.h5") as second:
        for key in first["molecule"]:
            np.testing.assert_array_equal(first["molecule"][key][()], second["molecule"][key][()])


def test_rdkit_seed_is_a_stratified_sample_of_the_pool(prepared_seed):
    with (
        h5py.File(prepared_seed.output / "pool.h5") as pool_file,
        h5py.File(prepared_seed.output / "seed.h5") as seed_file,
    ):
        pool, selected = pool_file["molecule"], seed_file["molecule"]
        indices = selected["pool_index"][()]
        assert len(np.unique(indices)) == 12
        species = selected["species"][()].astype("U2")
        assert species.ndim == 1
        assert np.all(np.diff([atomic_numbers[value] for value in species]) >= 0)
        np.testing.assert_array_equal(np.bincount(selected["strata"][()][:, 0]), [2] * 6)
        np.testing.assert_array_equal(selected["fps_rank"][()], [0, 1] * 6)
        np.testing.assert_array_equal(selected["coordinates"][()], pool["coordinates"][()][indices])
        np.testing.assert_array_equal(selected["strata"][()], pool["strata"][()][indices])
        np.testing.assert_array_equal(selected["fps_rank"][()], pool["fps_rank"][()][indices])
        assert not any(key.startswith(("sE", "F", "dE")) for key in selected)


def test_rdkit_builder_replays_prepared_geometries(prepared_seed):
    with h5py.File(prepared_seed.output / "seed.h5") as handle:
        selected = handle["molecule"]
        canonical = selected["coordinates"][()][:, np.argsort(selected["topology_atom_ids"][()])]
    builder = json.loads((prepared_seed.output / "builder_config.json").read_text())
    molecules = seed.build_rdkit_molecules([f"mol-boot-{i:010d}" for i in range(12)], builder)
    assert len(molecules) == 12
    for index, molecule in enumerate(molecules):
        assert molecule.get_results() == {}
        order = np.argsort(molecule.atoms.arrays["topology_atom_ids"])
        np.testing.assert_array_equal(molecule.atoms.positions[order], canonical[index])
        assert molecule.get_metadata()["rdkit_fps_rank"] == index % 2
    with pytest.raises(IndexError, match="exhausted"):
        seed.build_rdkit_molecules(["mol-boot-0000000012"], builder)


def test_rdkit_runtime_replay_strips_labels(prepared_seed, tmp_path):
    builder = json.loads((prepared_seed.output / "builder_config.json").read_text())
    labels = tmp_path / "labels"
    labels.mkdir()
    shutil.copyfile(prepared_seed.output / "seed.h5", labels / "data-0000.h5")
    with h5py.File(labels / "data-0000.h5", "a") as handle:
        handle["molecule"]["energy"] = np.ones(12)
    replay = seed.build_rdkit_molecules(["runtime-1"], builder, str(labels / "data-{:04d}.h5"), 1)
    assert replay[0].get_results() == {} and not replay[0].get_metadata()["replay_external_bootstrap"]


def test_failed_rdkit_preparation_writes_nothing(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    from rdkit.Chem import AllChem

    config = {**example_config("keto"), "pool_size": 6, "seed_size": 6}
    output = tmp_path / "failed"
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="quota"):
        seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    assert not output.exists()
    output.mkdir()
    marker = output / "keep"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    assert marker.read_text() == "unchanged"


def test_rdkit_command_line_finds_reference_next_to_config(tmp_path):
    pytest.importorskip("rdkit")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {**example_config("enol"), "pool_size": 12, "seed_size": 6}
    (config_dir / "enol.xyz").write_bytes((EXAMPLES / "enol.xyz").read_bytes())
    path = config_dir / "seed.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "prepared"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alframework.builders.rdkit_molecule",
            "--config",
            str(path),
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "provenance.json").read_text())
    assert report["selected_count"] == 6
    assert report["settings"]["reference_geometry"] == str(config_dir / "enol.xyz")
