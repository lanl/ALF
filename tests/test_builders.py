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
