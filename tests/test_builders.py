import itertools
import random

import numpy as np
import pytest

from alframework.builders.builders import (
    atomic_system_builder,
    construct_simulation_box,
    create_atomic_system,
    simple_condensed_phase_builder_task,
    simple_multi_condensed_phase_builder_task,
    to_mic,
)
from alframework.tools.molecules_class import MoleculesObject


@pytest.mark.parametrize(
    "atom_charges,target_num_atoms",
    [
        ({"Na": 1, "Cl": -1}, 8),
        ({"Mg": 2, "Cl": -1}, 9),
        ({"Ca": 2, "F": -1, "O": -2}, 10),
    ],
)
def test_create_atomic_system_is_charge_neutral(atom_charges, target_num_atoms):
    random.seed(1)

    system = create_atomic_system(atom_charges, target_num_atoms)

    total_charge = sum(atom_charges[atom] * count for atom, count in system.items())
    assert total_charge == 0
    assert sum(system.values()) >= target_num_atoms - 1
    assert set(system).issubset(atom_charges)


def test_to_mic_wraps_distances_into_minimum_image():
    distances = np.array([[5.5, -5.5, 0.0], [1.0, 2.0, -6.0]])

    np.testing.assert_allclose(to_mic(distances, box_length=10.0), [[-4.5, 4.5, 0.0], [1.0, 2.0, 4.0]])


def test_construct_simulation_box_respects_minimum_distance():
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


def test_construct_simulation_box_can_return_scaled_coordinates():
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


def test_construct_simulation_box_rejects_too_dense_system():
    with pytest.raises(ValueError):
        construct_simulation_box({"Na": 10}, min_distance=2.0, box_length=3.0)


def test_atomic_system_builder_returns_molecule_ready_atoms():
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


def test_condensed_phase_builders_return_periodic_systems_with_requested_ids():
    from pathlib import Path

    library = str(Path(__file__).resolve().parents[1] / "examples" / "simple_water_multi_builder" / "fragment_library") + "/"
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
    import h5py
    from ase import Atoms
    from ase.io import write
    from ase.calculators.singlepoint import SinglePointCalculator
    atoms = Atoms('OHH', positions=[[0, 0, 0], [.95, 0, 0], [0, .95, 0]],
                  cell=np.eye(3) * 8, pbc=[True, False, True])
    atoms.set_array('topology_atom_ids', np.arange(3))
    atoms.calc = SinglePointCalculator(atoms, energy=-42., forces=np.ones((3, 3)))
    write(tmp_path / 'source.xyz', atoms, format='extxyz')
    permutation = [0, 2, 1]
    with h5py.File(tmp_path / 'source.h5', 'w') as h:
        g = h.create_group('water')
        g['coordinates'] = [atoms.positions[permutation]]
        g['species'] = np.array(['O', 'H', 'H'], dtype='S')
        g['topology_atom_ids'] = [permutation]
        g['_id'] = np.array(['source-water'], dtype='S')
        g['energy'] = [-123.]
        g['forces'] = np.ones((1, 3, 3))
    return tmp_path


def test_external_geometry_import_strips_labels_preserves_identity(geometry_sources):
    from alframework.builders.external_geometry import external_geometry_builder_task
    config = dict(sources=['source.xyz', 'source.h5'], selection_mode='sequential')
    items = external_geometry_builder_task.func(moleculeid='ignored', moleculeids=['new-0', 'new-1'],
                                               builder_config=config, master_directory=str(geometry_sources))
    assert [m.get_moleculeid() for m in items] == ['new-0', 'new-1']
    for item in items:
        assert item.get_results() == {} and item.converged is None
        assert item.atoms.calc is None and item.atoms.info == {}
        assert len(item.get_metadata()['geometry_source_sha256']) == 64
    np.testing.assert_array_equal(items[0].atoms.pbc, [True, False, True])
    np.testing.assert_allclose(items[0].atoms.cell, np.eye(3) * 8)
    np.testing.assert_array_equal(items[1].atoms.arrays['topology_atom_ids'], [0, 2, 1])
    assert items[1].get_metadata()['geometry_source_id'] == 'source-water'
    single = external_geometry_builder_task.func('new-1', builder_config=config,
                                                master_directory=str(geometry_sources))
    np.testing.assert_allclose(single.atoms.positions, items[1].atoms.positions)


def test_external_geometry_hash_selection_is_stateless(geometry_sources):
    from alframework.builders.external_geometry import build_external_geometries
    config = dict(sources=['source.xyz', 'source.h5'], selection_seed=19)
    a = build_external_geometries(['a', 'b', 'c'], config, str(geometry_sources))
    b = build_external_geometries(['c', 'a'], config, str(geometry_sources))
    for x, y in [(a[2], b[0]), (a[0], b[1])]:
        assert x.get_metadata() == y.get_metadata()
        np.testing.assert_array_equal(x.atoms.positions, y.atoms.positions)


@pytest.mark.parametrize('ids,mode,error', [
    (['new-2'], 'sequential', IndexError), (['new'], 'sequential', ValueError),
    (['a', 'a'], 'deterministic_hash', ValueError), (['a'], 'bad', ValueError),
])
def test_external_geometry_rejects_invalid_selection(geometry_sources, ids, mode, error):
    from alframework.builders.external_geometry import build_external_geometries
    with pytest.raises(error):
        build_external_geometries(ids, dict(sources=['source.xyz', 'source.h5'], selection_mode=mode),
                                  str(geometry_sources))


@pytest.mark.parametrize('bad_ids', [[0, 0, 2], [0., 1., 2.]])
def test_external_geometry_rejects_invalid_topology(geometry_sources, bad_ids):
    import h5py
    from alframework.builders.external_geometry import build_external_geometries
    with h5py.File(geometry_sources / 'source.h5', 'a') as h:
        del h['water/topology_atom_ids']
        h['water/topology_atom_ids'] = [bad_ids]
    with pytest.raises(ValueError, match='integer permutation'):
        build_external_geometries(['a'], dict(sources=['source.h5']), str(geometry_sources))


def test_external_geometry_plain_xyz_does_not_invent_topology(tmp_path):
    from alframework.builders.external_geometry import build_external_geometries
    (tmp_path / 'plain.xyz').write_text('2\nplain molecule\nH 0 0 0\nH 0 0 .74\n')
    m = build_external_geometries(['a'], dict(sources=['plain.xyz']), str(tmp_path))[0]
    assert 'topology_atom_ids' not in m.atoms.arrays
    assert not m.atoms.pbc.any()


def test_external_geometry_rejects_empty_and_malformed_h5(tmp_path):
    import h5py
    from alframework.builders.external_geometry import geometry_manifest
    with pytest.raises(ValueError, match='nonempty'):
        geometry_manifest({})
    with h5py.File(tmp_path / 'bad.h5', 'w') as h:
        h['species'] = [1, 1]
        h['coordinates'] = np.zeros((2, 3))
    with pytest.raises(ValueError, match='frames, atoms, 3'):
        geometry_manifest(dict(sources=['bad.h5']), str(tmp_path))


# Molecular RDKit/FPS preparation is an optional builder capability.
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import h5py
from ase.data import atomic_numbers
from alframework.builders import rdkit_molecule as seed
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT/'examples/keto_acac'

def example_config(name):
    return json.loads((EXAMPLES/f'seed_{name}.json').read_text())

def test_rdkit_quotas_descriptor_and_fps_are_deterministic():
    assert seed.ladder_quotas(50000, 6) == [8334, 8334, 8333, 8333, 8333, 8333]
    assert seed.ladder_quotas(5000, 6) == [834, 834, 833, 833, 833, 833]
    assert [seed.stratum_seed(42, level, 0) for level in range(6)] == [
        438055385, 1262664333, 1985621771, 1596702514, 702239142, 649246379]
    features = np.arange(4, dtype=float)[:, None]
    np.testing.assert_array_equal(seed.farthest_point_order(features, 4), [1, 3, 0, 2])
    np.testing.assert_array_equal(seed.farthest_point_order(np.zeros((4, 2)), 4), np.arange(4))
    coordinates = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]]])
    numbers = np.array([8, 1, 1])
    descriptor = seed.pair_descriptor(coordinates, numbers)
    np.testing.assert_allclose(descriptor, [[np.sqrt(5), 1., 2.]])
    order = [2, 0, 1]
    np.testing.assert_array_equal(seed.pair_descriptor(coordinates[:, order] + 10., numbers[order]), descriptor)
    with pytest.raises(ValueError, match='FPS'):
        seed.farthest_point_order(features, 5)


@pytest.mark.parametrize('override', [
    {'seed_size': 50001}, {'threads': True}, {'bond_scales': [0., float('nan')]},
    {'angle_scales': [0.]}, {'geometry_screen': {'typo': 1.}},
])
def test_rdkit_invalid_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        seed.resolve_config({**example_config('keto'), **override}, base_directory=EXAMPLES)


def test_rdkit_seed_screen_boundaries_and_rejection_priority():
    context = {'numbers': np.array([1, 1]), 'bonds': np.array([[0, 1]]),
               'bond_centers': np.array([1.]), 'pairs': np.array([[0, 1]]),
               'nonbonded': np.empty((0, 2), dtype=int)}
    screen = seed.DEFAULT_SETTINGS['geometry_screen']
    for distance in (.85, 1., 1.50):
        assert seed.geometry_failure([[0., 0., 0.], [distance, 0., 0.]], context, screen) is None
    assert seed.geometry_failure([[0., 0., 0.], [.849, 0., 0.]], context, screen) == 'bond_window'
    assert seed.geometry_failure([[0., 0., 0.], [1.501, 0., 0.]], context, screen) == 'bond_window'
    assert seed.geometry_failure(np.full((2, 3), np.nan), context, screen) == 'non_finite'
    loose = {**screen, 'bond_min_scale': .1}
    assert seed.geometry_failure([[0., 0., 0.], [.69, 0., 0.]], context, loose) == 'min_distance'
    context.update(numbers=np.array([1, 1, 1]), pairs=np.array([[0, 1], [0, 2], [1, 2]]),
                   nonbonded=np.array([[0, 2], [1, 2]]))
    assert seed.geometry_failure([[0., 0., 0.], [1., 0., 0.], [0., .1, 0.]], context, screen) == 'inferred_extra_bond'
    assert seed.geometry_failure([[0., 0., 0.], [1., 0., 0.], [0., .82, 0.]], context, screen) == 'nonbonded_close'


def test_rdkit_reference_centers_and_polar_hydrogen_widening(tmp_path):
    pytest.importorskip('rdkit')
    reference = tmp_path / 'water.xyz'
    reference.write_text('3\nwater\nO 0 0 0\nH .96 0 0\nH -.24 .93 0\n')
    config = seed.resolve_config({'smiles': 'O', 'reference_geometry': str(reference)})
    context = seed._reference_context(config)
    polar = seed.widened_bounds_matrix(context['molecule'], context['centers'], config, 5)
    ordinary = seed.widened_bounds_matrix(context['molecule'], context['centers'], {**config, 'polar_h': False}, 5)
    center = context['centers'][0, 1]
    assert polar[0, 1] == pytest.approx(center * 1.1)
    assert polar[1, 0] == pytest.approx(center * .9)
    assert polar[0, 1] > ordinary[0, 1]
    assert polar[1, 0] < ordinary[1, 0]


def test_rdkit_matching_species_are_not_a_topology_mapping():
    pytest.importorskip('rdkit')
    config = {**example_config('keto'), 'reference_geometry': 'enol.xyz'}
    with pytest.raises(ValueError, match='connectivity|bond set'):
        seed._reference_context(seed.resolve_config(config, base_directory=EXAMPLES))


@pytest.mark.parametrize('name', ['enol', 'keto'])
def test_rdkit_preparation_is_reproducible_and_replays_tagged_geometry_only(tmp_path, name):
    pytest.importorskip('rdkit')
    config = {**example_config(name), 'pool_size': 24, 'seed_size': 12}
    output = tmp_path / name
    report = seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    repeated = tmp_path / (name + '-repeat')
    seed.prepare_rdkit_seed(config, repeated, config_directory=EXAMPLES)
    assert report['pool_count'] == 24 and report['selected_count'] == 12
    assert report['embedding_settings']['useBasicKnowledge'] is True
    assert report['embedding_settings']['useExpTorsionAnglePrefs'] is False
    assert report['embedding_settings']['force_field_optimization'] is False
    for file, digest in report['artifacts_sha256'].items():
        assert hashlib.sha256((output / file).read_bytes()).hexdigest() == digest
    with h5py.File(output / 'pool.h5') as pool, h5py.File(output / 'seed.h5') as selected, h5py.File(repeated / 'seed.h5') as again:
        p, s, r = pool['molecule'], selected['molecule'], again['molecule']
        indices = s['pool_index'][()]
        assert len(np.unique(indices)) == 12
        species = s['species'][()].astype('U2')
        assert species.ndim == 1
        assert np.all(np.diff([atomic_numbers[value] for value in species]) >= 0)
        np.testing.assert_array_equal(np.bincount(s['strata'][()][:, 0]), [2] * 6)
        np.testing.assert_array_equal(s['fps_rank'][()], [0, 1] * 6)
        np.testing.assert_array_equal(s['coordinates'][()], p['coordinates'][()][indices])
        np.testing.assert_array_equal(s['strata'][()], p['strata'][()][indices])
        np.testing.assert_array_equal(s['fps_rank'][()], p['fps_rank'][()][indices])
        for key in s:
            np.testing.assert_array_equal(s[key][()], r[key][()])
        assert not any(key.startswith(('sE', 'F', 'dE')) for key in s)
        reorder = np.argsort(s['topology_atom_ids'][()])
        canonical = s['coordinates'][()][:, reorder]
    builder=json.loads((output/'builder_config.json').read_text())
    molecules = seed.build_rdkit_molecules([f'mol-boot-{i:010d}' for i in range(12)],builder)
    assert len(molecules)==12
    for index,molecule in enumerate(molecules):
        assert molecule.get_results()=={}
        order=np.argsort(molecule.atoms.arrays['topology_atom_ids'])
        np.testing.assert_array_equal(molecule.atoms.positions[order],canonical[index])
        assert molecule.get_metadata()['rdkit_fps_rank']==index%2
    with pytest.raises(IndexError,match='exhausted'):
        seed.build_rdkit_molecules(['mol-boot-0000000012'],builder)
    # Runtime closed label replay strips deliberately injected source labels.
    labels=tmp_path/'labels';labels.mkdir()
    import shutil
    shutil.copyfile(output/'seed.h5',labels/'data-0000.h5')
    with h5py.File(labels/'data-0000.h5','a') as handle:
        handle['molecule']['energy']=np.ones(12)
    replay=seed.build_rdkit_molecules(['runtime-1'],builder,str(labels/'data-{:04d}.h5'),1)
    assert replay[0].get_results()=={} and not replay[0].get_metadata()['replay_external_bootstrap']


def test_rdkit_exhausted_embedding_and_existing_outputs_fail_without_overwrite(tmp_path, monkeypatch):
    pytest.importorskip('rdkit')
    from rdkit.Chem import AllChem
    config = {**example_config('keto'), 'pool_size': 6, 'seed_size': 6}
    output = tmp_path / 'failed'
    monkeypatch.setattr(AllChem, 'EmbedMultipleConfs', lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match='quota'):
        seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    assert not output.exists()
    output.mkdir()
    marker = output / 'keep'
    marker.write_text('unchanged')
    with pytest.raises(FileExistsError):
        seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    assert marker.read_text() == 'unchanged'


def test_rdkit_cli_resolves_reference_relative_to_config(tmp_path):
    pytest.importorskip('rdkit')
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    config = {**example_config('enol'), 'pool_size': 12, 'seed_size': 6}
    (config_dir / 'enol.xyz').write_bytes((EXAMPLES / 'enol.xyz').read_bytes())
    path = config_dir / 'seed.json'
    path.write_text(json.dumps(config))
    output = tmp_path / 'prepared'
    result = subprocess.run([sys.executable, '-m', 'alframework.builders.rdkit_molecule',
                             '--config', str(path), '--output-dir', str(output)], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'provenance.json').read_text())
    assert report['selected_count'] == 6
    assert report['settings']['reference_geometry'] == str(config_dir / 'enol.xyz')
