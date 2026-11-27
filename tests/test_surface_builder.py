import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.io import read, write
from ase.neighborlist import neighbor_list
from parsl import python_app
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from alframework.builders.external_geometry import build_external_geometries
from alframework.builders.surface import build_surface, surface_builder_task
from alframework.builders.surface_symmetry import bulk_operations, slab_has_equivalent_faces
from alframework.testing import check_builder_task
from alframework.tools.data_shards import store_current_data
from tests.helpers.driver import PROPERTIES, run_alf_driver


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'surface_adsorption'


@pytest.fixture
def copper(tmp_path):
    write(tmp_path / 'copper.extxyz', bulk('Cu', 'fcc', cubic=True))
    return dict(bulk_file='copper.extxyz', miller_indices=[[1, 1, 1]])


def _co_placement(**settings):
    return dict(file=str(EXAMPLE / 'co.xyz'), anchor_indices=[0], **settings)


def _miller_index_of(candidate):
    return tuple(candidate.get_metadata()['surface_builder']['miller_index'])


def _check_cell(atoms, vacuum):
    np.testing.assert_allclose(atoms.cell[2, :2], 0, atol=1e-10)
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    np.testing.assert_allclose(normal / np.linalg.norm(normal), [0, 0, 1], atol=1e-10)
    assert atoms.cell[2, 2] > 0 and atoms.pbc.all()
    z = atoms.positions[:, 2]
    np.testing.assert_allclose((z.min(), atoms.cell[2, 2] - z.max()), [vacuum / 2] * 2)


@pytest.mark.parametrize('kind', ['random', 'ontop', 'bridge', 'hollow'])
def test_co_placement_preserves_geometry_and_uses_the_periodic_surface(copper, tmp_path, kind):
    mode = 'random' if kind == 'random' else 'binding_sites'
    config = dict(copper, adsorbates=[_co_placement(mode=mode, site_types=[kind])], vacuum_A=18, face='bottom')
    molecule = build_surface('seed', config, tmp_path)
    atoms = molecule.get_atoms()
    metadata = molecule.get_metadata()['surface_builder']
    site = np.array(metadata['site'])
    anchor = metadata['anchor_atom_index']
    assert metadata['site_type'] == kind
    assert metadata['anchor_index'] == 0 and atoms[anchor].symbol == 'C'
    np.testing.assert_allclose(atoms.positions[anchor, :2], site[:2])
    assert atoms.positions[anchor, 2] > site[2]
    assert atoms.get_distance(anchor, anchor + 1) == pytest.approx(1.13)
    _check_cell(atoms, 18)
    assert not atoms.constraints
    assert set(atoms.get_tags()) == {0, 1, 2}
    distances = atoms.get_all_distances(mic=True)[-2:, :-2]
    radii = covalent_radii[atoms.numbers]
    assert np.all(distances + 1e-8 >= radii[-2:, None] + radii[:-2] + 0.1)
    if kind == 'ontop':
        vectors = atoms.positions[:-2] - site
        assert np.min(find_mic(vectors, atoms.cell, pbc=(True, True, False))[1]) < 1e-6


def test_alf_builder_contract_and_periodic_hdf5_round_trip(copper, tmp_path):
    clean, = check_builder_task(surface_builder_task, builder_config=copper, master_directory=tmp_path, count=1)
    assert clean.get_atoms().get_chemical_symbols() == ['Cu'] * len(clean)
    assert np.ptp(clean.get_atoms().positions[:, 2]) >= 10 - 1e-8
    assert np.all(clean.get_atoms().cell.lengths()[:2] >= 8)
    config = dict(copper, adsorbates=[_co_placement()], seed=12)
    candidates = check_builder_task(surface_builder_task, builder_config=config, master_directory=tmp_path)
    sample = candidates[0]
    replay = build_surface(sample.get_moleculeid(), config, tmp_path)
    np.testing.assert_array_equal(sample.get_atoms().positions, replay.get_atoms().positions)
    assert sample.get_metadata() == replay.get_metadata()
    assert not np.allclose(sample.get_atoms().positions[-2:], candidates[1].get_atoms().positions[-2:])

    for item in candidates:
        item.store_results(dict(energy=-1.0, forces=np.zeros((len(item), 3))))
        item.set_converged_flag(True)
    shard = tmp_path / 'slab.h5'
    store_current_data(str(shard), candidates, PROPERTIES)
    with h5py.File(shard) as handle:
        group, = handle.values()
        assert group['coordinates'].shape == (2, len(sample), 3)
        assert group['cell'].shape == (2, 3, 3)
    restored, = build_external_geometries(['mol-0'], dict(sources=[str(shard)], selection_mode='sequential'))
    assert restored.get_atoms().pbc.all()
    np.testing.assert_allclose(restored.get_atoms().cell, sample.get_atoms().cell)


def _organic_components(slab, cutoffs):
    slab = slab.copy()
    slab.pbc = (True, True, False)
    left, right = neighbor_list('ij', slab, cutoffs)
    graph = csr_matrix((np.ones(len(left)), (left, right)), shape=(len(slab), len(slab)))
    _, components = connected_components(graph, directed=False)
    symbols = np.asarray(slab.get_chemical_symbols())
    return [Counter(symbols[components == component])
            for component in np.unique(components[np.isin(symbols, ['C', 'N', 'H'])])]


def test_mapi_cleavage_keeps_wrapped_methylammonium_intact():
    config = json.loads((EXAMPLE / 'builder_config.json').read_text())
    config.update(bulk_file='mapi_100K.extxyz', face='both', min_slab_thickness_A=15,
                  min_in_plane_A=10, vacuum_A=18, surface_depth_A=4)
    for key in ('target_a_A', 'target_b_A', 'max_atoms'):
        config.pop(key)
    cutoffs = {(x, y): r for x, y, r in config['protected_bonds']}
    crystal = read(EXAMPLE / config['bulk_file'])
    offsets = neighbor_list('S', crystal, cutoffs)
    assert np.any(offsets[:, 0] != 0)

    cuts = set()
    faces = set()
    ion = {'C': 1, 'N': 1, 'H': 6}
    for index in range(16):
        molecule = build_surface(f'mapi-{index}', config, EXAMPLE)
        metadata = molecule.get_metadata()['surface_builder']
        cuts.add(metadata['termination_shift'])
        faces.add(metadata['face'])
        components = _organic_components(molecule.get_atoms()[:-2], cutoffs)
        assert components and all(component == ion for component in components)
    assert len(cuts) > 1
    assert faces == {'top', 'bottom'}

    # Removing protection permits a real (100) termination with a split ion.
    bare = dict(config, adsorbates=[], protected_bonds=[])
    unprotected = (build_surface(f'mapi-{index}', bare, EXAMPLE) for index in range(16))
    assert any(any(component != ion for component in _organic_components(m.get_atoms(), cutoffs))
               for m in unprotected)


@pytest.mark.parametrize('mode', ['random', 'binding_sites'])
def test_mp_mapi_builds_co_on_the_smaller_unit_cell(mode):
    config = json.loads((EXAMPLE / 'builder_config.json').read_text())
    config['adsorbates'][0]['mode'] = mode
    cutoffs = {(x, y): r for x, y, r in config['protected_bonds']}
    for index in range(16):
        sample = build_surface(f'preview-{index}', config, EXAMPLE)
        substrate = sample.atoms[sample.atoms.get_tags() != 2]
        assert len(sample) == 98 and sample.atoms[-2:].get_chemical_symbols() == ['C', 'O']
        assert 10 <= np.ptp(substrate.positions[:, 2]) <= 13
        np.testing.assert_allclose(sample.atoms.cell.lengths()[:2], [12.87672, 12.98422], atol=0.001)
        assert _organic_components(substrate, cutoffs) == [{'C': 1, 'N': 1, 'H': 6}] * 8
        _check_cell(sample.atoms, 20)


def test_mp_pool_builds_all_four_saved_materials():
    config = json.loads((EXAMPLE / 'builder_mp_materials.json').read_text())
    expected = {
        'mp-977013': ({'C': 1, 'H': 6, 'N': 1, 'Pb': 1, 'I': 3}, 8),
        'mp-977014': ({'C': 1, 'H': 5, 'N': 2, 'Pb': 1, 'I': 3}, 8),
        'mp-600089': ({'Cs': 1, 'Pb': 1, 'Br': 3}, 12),
        'mp-1069538': ({'Cs': 1, 'Pb': 1, 'I': 3}, 12),
    }
    for material in config['materials']:
        source = read(EXAMPLE / material['bulk_file'])
        formula, _ = expected[material['id']]
        assert source.info['materials_project_id'] == material['id']
        assert Counter(source.symbols) == formula and source.pbc.all()
    observed = set()
    for index in range(24):
        sample = build_surface(f'pool-{index}', config, EXAMPLE)
        name = sample.metadata['surface_builder']['material_id']
        observed.add(name)
        formula, repeats = expected[name]
        assert Counter(sample.atoms.symbols) == {symbol: count * repeats for symbol, count in formula.items()}
        assert _miller_index_of(sample) == (1, 0, 0) and len(sample) <= 200
        _check_cell(sample.atoms, 20)
    assert observed == set(expected)


@pytest.mark.parametrize('decorate', [False, True])
def test_independent_cuts_make_a_thinner_stoichiometric_symmetric_mapi_slab(decorate):
    config = json.loads((EXAMPLE / 'builder_config.json').read_text())
    config.update(bulk_file='mapi_100K.extxyz', slab_symmetry='require', allow_independent_terminations=True)
    if not decorate:
        config.pop('adsorbates')
    cutoffs = {(x, y): r for x, y, r in config['protected_bonds']}
    for min_b, expected_b, formulas in [(8, 8.57689, 6), (9, 17.15378, 12)]:
        sample = build_surface('compact', dict(config, min_b_A=min_b), EXAMPLE)
        info = sample.metadata['surface_builder']
        substrate = sample.atoms[sample.atoms.get_tags() != 2]
        assert info['slab_symmetric'] and slab_has_equivalent_faces(substrate, 0.001)
        assert info['upper_termination_shift'] - info['termination_shift'] == pytest.approx(1.5)
        assert np.ptp(substrate.positions[:, 2]) == pytest.approx(12.209898)
        assert Counter(substrate.symbols) == {'C': formulas, 'H': 6 * formulas, 'N': formulas,
                                              'I': 3 * formulas, 'Pb': formulas}
        assert _organic_components(substrate, cutoffs) == [{'C': 1, 'N': 1, 'H': 6}] * formulas
        np.testing.assert_allclose(sample.atoms.cell.lengths()[:2], [12.6293, expected_b])
        assert len(sample) == 12 * formulas + (2 if decorate else 0)
        _check_cell(sample.atoms, 20)
    whole = build_surface('compact', dict(config, allow_independent_terminations=False), EXAMPLE)
    substrate = whole.atoms[whole.atoms.get_tags() != 2]
    assert np.ptp(substrate.positions[:, 2]) == pytest.approx(16.642768)
    assert 'upper_termination_shift' not in whole.metadata['surface_builder']


def test_rejects_a_facet_that_must_cross_a_protected_bond(tmp_path):
    # Every (100) cut through a simple cubic one-atom lattice breaks a protected C–C bond.
    write(tmp_path / 'network.extxyz', Atoms('C', cell=[2, 2, 2], pbc=True))
    config = dict(bulk_file='network.extxyz', miller_indices=[[1, 0, 0]], protected_bonds=[['C', 'C', 2.1]])
    with pytest.raises(ValueError, match=r'no intact slab for Miller \(1, 0, 0\)'):
        build_surface('seed', config, tmp_path)


def test_automatic_and_explicit_facets_and_xyz_need_no_extras(copper, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'spglib', None)
    monkeypatch.setitem(sys.modules, 'rdkit', None)
    config = dict(copper, adsorbates=[_co_placement()])
    explicit = build_surface('seed', config, tmp_path)
    assert _miller_index_of(explicit) == (1, 1, 1)
    metadata = explicit.metadata['surface_builder']
    assert metadata['adsorbate_charge'] == dict(net=0, assumed=True, sources=[])
    assert metadata['bulk_charge'] == dict(net=0, assumed=True, sources=[])
    config.pop('miller_indices')
    automatic = build_surface('seed', dict(config, max_miller=1, slab_symmetry='require'), tmp_path)
    assert automatic.metadata['surface_builder']['slab_symmetric'] is True
    with pytest.raises(ValueError, match='exactly one'):
        build_surface('seed', dict(copper, max_miller=1), tmp_path)


def test_enumerates_crystal_specific_facet_families(copper, tmp_path):
    config = dict(bulk_file=copper['bulk_file'], max_miller=1)
    samples = [build_surface(f'facet-{i}', config, tmp_path) for i in range(300)]
    counts = Counter(map(_miller_index_of, samples))
    assert set(counts) == {(1, 0, 0), (1, 1, 0), (1, 1, 1)}
    assert all(75 <= count <= 125 for count in counts.values())
    replay = build_surface('facet-0', config, tmp_path)
    assert replay.metadata == samples[0].metadata
    np.testing.assert_array_equal(replay.atoms.positions, samples[0].atoms.positions)

    expanded = dict(config, max_miller=2)
    observed = {_miller_index_of(build_surface(f'facet-{i}', expanded, tmp_path)) for i in range(100)}
    assert observed == set(counts) | {(2, 1, 0), (2, 1, 1), (2, 2, 1)}

    # Different axis lengths prevent the symmetry from identifying these Miller directions.
    write(tmp_path / 'orthorhombic.extxyz', Atoms('Cu', cell=[3, 4, 5], pbc=True))
    orthorhombic = dict(config, bulk_file='orthorhombic.extxyz')
    observed = {_miller_index_of(build_surface(f'facet-{i}', orthorhombic, tmp_path)) for i in range(100)}
    assert observed == {(1, 0, 0), (0, 1, 0), (0, 0, 1),
                        (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)}


def test_bulk_symmetry_checks_the_motif_and_handles_skew_cell_bases():
    crystal = bulk('Cu', cubic=True)
    transform = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
    original = bulk_operations(crystal, 0.001)
    crystal.set_cell(transform @ crystal.cell)
    skew = bulk_operations(crystal, 0.001)
    inverse = np.rint(np.linalg.inv(transform)).astype(int)
    expected = transform @ original @ inverse
    assert {tuple(op.flat) for op in skew} == {tuple(op.flat) for op in expected}

    # A cubic lattice alone does not imply cubic symmetry for the decorated crystal.
    motif = Atoms('CuNO', scaled_positions=[[0, 0, 0], [0.17, 0.23, 0.39], [0.41, 0.11, 0.28]],
                  cell=[5, 5, 5], pbc=True)
    np.testing.assert_array_equal(bulk_operations(motif, 0.001), [np.eye(3, dtype=int)])
    hexagonal = bulk('Mg', 'hcp', a=3.2, c=5.2)
    assert len(bulk_operations(hexagonal, 0.001)) == 24


def test_slab_symmetry_matches_face_swaps_and_filters_before_placement(tmp_path):
    face_swap = Atoms('CuNCuN', positions=[[0, 0, 0], [1, 1, 0.6], [2, 0, 3], [3, 1, 2.4]],
                      cell=[4, 4, 8], pbc=(True, True, False))
    assert slab_has_equivalent_faces(face_swap, 0.001)
    face_swap.positions[3, 2] += 0.01
    assert not slab_has_equivalent_faces(face_swap, 0.001)
    assert slab_has_equivalent_faces(face_swap, 0.02)

    asymmetric = Atoms('CuNCu', positions=[[0, 0, 0], [0.4, 0.2, 1], [0.2, 0.3, 3]],
                       cell=[4, 4, 6], pbc=True)
    write(tmp_path / 'asymmetric.extxyz', asymmetric)
    config = dict(bulk_file='asymmetric.extxyz', miller_indices=[[0, 0, 1]])
    candidate = build_surface('seed', dict(config, slab_symmetry='report'), tmp_path)
    assert candidate.metadata['surface_builder']['slab_symmetric'] is False
    with pytest.raises(ValueError, match='top and bottom faces are inequivalent'):
        build_surface('seed', dict(config, slab_symmetry='require'), tmp_path)


def test_axis_bounds_and_atom_cap(copper, tmp_path):
    config = dict(copper, miller_indices=[[1, 0, 0]], min_in_plane_A=6,
                  min_a_A=9, max_a_A=11, max_b_A=8)
    slab = build_surface('seed', config, tmp_path).get_atoms()
    lattice_a = read(tmp_path / copper['bulk_file']).cell.lengths()[0]
    np.testing.assert_allclose(slab.cell.lengths()[:2], [3 * lattice_a, 2 * lattice_a])
    with pytest.raises(ValueError, match=r'no intact slab for Miller .*\|a\| cannot fit'):
        build_surface('seed', dict(config, max_a_A=10), tmp_path)
    with pytest.raises(ValueError, match=r'min_b_A must not exceed max_b_A'):
        build_surface('seed', dict(config, min_b_A=9), tmp_path)

    write(tmp_path / 'water.xyz', molecule('H2O'))
    adsorbates = [_co_placement(), dict(file='water.xyz', anchor_indices=[0])]
    at_limit = dict(config, adsorbates=adsorbates, max_atoms=len(slab) + 3)
    observed = {len(build_surface(f'cap-{i}', at_limit, tmp_path)) for i in range(15)}
    assert observed == {len(slab) + 2, len(slab) + 3}
    with pytest.raises(ValueError, match=r'3 reserved for the adsorbate .*max_atoms='):
        build_surface('seed', dict(at_limit, max_atoms=len(slab) + 2), tmp_path)


def test_lateral_targets_choose_the_closest_feasible_integer_repeats(tmp_path):
    write(tmp_path / 'bulk.extxyz', Atoms('Cu', cell=[4, 5, 3], pbc=True))
    config = dict(bulk_file='bulk.extxyz', miller_indices=[[0, 0, 1]], min_slab_thickness_A=2,
                  min_a_A=4, min_b_A=5, target_a_A=8, target_b_A=10, adsorbates=[_co_placement()])
    assert len(build_surface('seed', config, tmp_path)) == 10
    capped = build_surface('seed', dict(config, max_atoms=6), tmp_path)
    assert len(capped) == 6
    np.testing.assert_allclose(capped.atoms.cell.lengths()[:2], [4, 10])
    tied = build_surface('seed', dict(config, target_a_A=6), tmp_path)
    np.testing.assert_allclose(tied.atoms.cell.lengths()[:2], [4, 10])
    bounded = build_surface('seed', dict(config, max_b_A=8), tmp_path)
    np.testing.assert_allclose(bounded.atoms.cell.lengths()[:2], [8, 5])


def test_automatic_sweep_tries_an_equivalent_orientation_and_reports_rejections(copper, tmp_path):
    # (110) has its long vector along a. An equivalent (011) orientation fits the requested bounds.
    config = dict(bulk_file=copper['bulk_file'], max_miller=1, min_a_A=3.4, max_a_A=4,
                  min_b_A=4.9, max_b_A=5.4)
    with pytest.warns(UserWarning, match=r'skipped facet families: \(1, 0, 0\).*\(1, 1, 1\)'):
        assert _miller_index_of(build_surface('seed', config, tmp_path)) == (0, 1, 1)
    with pytest.raises(ValueError, match=r'no facet families survived max_miller.*max_atoms=1'):
        build_surface('seed', dict(config, max_atoms=1), tmp_path)


def test_mapi_sweep_preserves_ions_and_respects_the_atom_cap():
    config = json.loads((EXAMPLE / 'builder_sweep.json').read_text())
    cutoffs = {(x, y): r for x, y, r in config['protected_bonds']}
    with pytest.warns(UserWarning, match=r'skipped facet families: \(0, 1, 0\).*max_atoms=300'):
        structures = [build_surface(f'sweep-{i}', config, EXAMPLE) for i in range(30)]
    facets = set()
    for item in structures:
        facets.add(_miller_index_of(item))
        atoms = item.get_atoms()
        assert len(atoms) <= config['max_atoms']
        _check_cell(atoms, 18)
        components = _organic_components(atoms, cutoffs)
        assert components and all(component == {'C': 1, 'N': 1, 'H': 6} for component in components)
    assert len(facets) > 1 and (0, 1, 0) not in facets


def test_smiles_embeds_explicit_hydrogens_and_counts_them(copper, tmp_path):
    pytest.importorskip('rdkit')
    size = len(build_surface('seed', copper, tmp_path))
    config = dict(copper, adsorbates=[dict(smiles='O', anchor_indices=[0])], max_atoms=size + 3)
    structure = build_surface('seed', config, tmp_path)
    assert structure.atoms[-3:].get_chemical_symbols() == ['O', 'H', 'H']
    np.testing.assert_array_equal(build_surface('seed', config, tmp_path).atoms.positions, structure.atoms.positions)
    metadata = structure.metadata['surface_builder']
    assert metadata['adsorbate_smiles'] == 'O'
    assert metadata['adsorbate_charge'] == dict(net=0, assumed=False, sources=['rdkit:formal_charge'])
    with pytest.raises(ValueError, match='3 reserved for the adsorbate'):
        build_surface('seed', dict(config, max_atoms=size + 2), tmp_path)
    for bad, reason in [('[Na+].[Cl-]', 'disconnected fragments'), ('[NH4+]', 'nonzero net charge'),
                        ('not_a_smiles', 'could not parse')]:
        with pytest.raises(ValueError, match=reason):
            build_surface('seed', dict(copper, adsorbates=[dict(smiles=bad, anchor_indices=[0])]), tmp_path)
    with pytest.raises(ValueError, match='input atom indices'):
        build_surface('seed', dict(config, adsorbates=[dict(smiles='O', anchor_indices=[1])]), tmp_path)


def test_bond_file_preserves_geometry_and_validates_formal_charge(copper, tmp_path):
    chem = pytest.importorskip('rdkit.Chem')
    from rdkit.Chem import AllChem

    water = chem.AddHs(chem.MolFromSmiles('O'))
    AllChem.EmbedMolecule(water, randomSeed=4)
    path = tmp_path / 'water.sdf'
    with chem.SDWriter(str(path)) as writer:
        writer.write(water)
    config = dict(copper, adsorbates=[dict(file=path.name, anchor_indices=[0])])
    structure = build_surface('seed', config, tmp_path)
    positions = water.GetConformer().GetPositions()
    expected = Atoms('OH2', positions=positions)
    np.testing.assert_allclose(structure.atoms[-3:].get_all_distances(), expected.get_all_distances(), atol=2e-4)
    assert structure.metadata['surface_builder']['adsorbate_charge']['sources'] == ['rdkit:formal_charge']
    # Labeling a neutral bond graph with a nonzero input charge is a conflicting assertion.
    config['adsorbates'][0]['charge'] = 1
    with pytest.raises(ValueError, match='water.sdf: conflicting charge declarations'):
        build_surface('seed', config, tmp_path)


def test_smiles_cleanup_is_reproducible_and_preserves_the_anchor(copper, tmp_path):
    pytest.importorskip('rdkit')
    from alframework.builders.surface_inputs import load_adsorbate

    entry = dict(smiles='CCO', anchor_indices=[2])
    embedded = load_adsorbate(entry, tmp_path, 0)
    entry['relax_smiles'] = True
    relaxed = load_adsorbate(entry, tmp_path, 0)
    assert relaxed.anchor_count == embedded.anchor_count == 3
    np.testing.assert_array_equal(relaxed.atoms.numbers, embedded.atoms.numbers)
    assert not np.allclose(relaxed.atoms.get_all_distances(), embedded.atoms.get_all_distances())
    config = dict(copper, adsorbates=[entry])
    candidate = build_surface('seed', config, tmp_path)
    metadata = candidate.get_metadata()['surface_builder']
    assert metadata['adsorbate_force_field'] == 'MMFF94'
    assert metadata['anchor_index'] == 2 and candidate.atoms[metadata['anchor_atom_index']].symbol == 'O'
    np.testing.assert_array_equal(candidate.atoms.positions, build_surface('seed', config, tmp_path).atoms.positions)


@pytest.mark.parametrize('face', ['top', 'bottom'])
def test_fixed_layers_match_the_oriented_substrate_without_selecting_adsorbates(copper, tmp_path, face):
    from ase.geometry import get_layers

    config = dict(copper, fixed_bottom_layers=2, face=face, adsorbates=[_co_placement()])
    molecule = build_surface('seed', config, tmp_path)
    atoms = molecule.get_atoms()
    substrate = atoms.get_tags() != 2
    layers, _ = get_layers(atoms[substrate], (0, 0, 1))
    expected = np.flatnonzero(substrate)[layers < 2]
    np.testing.assert_array_equal(molecule.get_metadata()['fixed_substrate_indices'], expected)
    np.testing.assert_array_equal(atoms.constraints[0].get_indices(), expected)
    with pytest.raises(ValueError, match='must leave a mobile substrate layer'):
        build_surface('seed', dict(config, fixed_bottom_layers=1000), tmp_path)


@pytest.mark.parametrize(('positions', 'reason'), [
    ([[0, 0, 0], [0, 0, 4]], 'disconnected fragments'),
    ([[0, 0, 0], [0, 0, 0.01]], 'overlapping atoms'),
])
def test_rejects_invalid_coordinate_adsorbates_even_when_other_entries_are_valid(copper, tmp_path, positions, reason):
    write(tmp_path / 'bad.xyz', Atoms('OH', positions=positions))
    config = dict(copper, adsorbates=[_co_placement(), dict(file='bad.xyz', anchor_indices=[0])])
    with pytest.raises(ValueError, match=f'bad.xyz:.*{reason}'):
        build_surface('seed', config, tmp_path)


def test_rejects_charged_bulk_and_adsorbate_files(copper, tmp_path):
    ion = Atoms('OH', positions=[[0, 0, 0], [0, 0, 1]], info={'charge': -1})
    write(tmp_path / 'ion.extxyz', ion)
    config = dict(copper, adsorbates=[dict(file='ion.extxyz', anchor_indices=[0])])
    with pytest.raises(ValueError, match='ion.extxyz: nonzero net charge'):
        build_surface('seed', config, tmp_path)
    config['adsorbates'][0]['charge'] = 0
    with pytest.raises(ValueError, match='ion.extxyz: conflicting charge declarations'):
        build_surface('seed', config, tmp_path)
    with pytest.raises(ValueError, match='nonzero net charge'):
        build_surface('seed', dict(copper, bulk_charge=1), tmp_path)

    crystal = read(tmp_path / copper['bulk_file'])
    crystal.set_initial_charges([0, 0.3, 0, 0])
    write(tmp_path / 'charged_bulk.extxyz', crystal)
    with pytest.raises(ValueError, match='charged_bulk.extxyz: nonzero net charge.*file:initial_charges'):
        build_surface('seed', dict(copper, bulk_file='charged_bulk.extxyz'), tmp_path)


def test_material_pool_samples_by_weight_and_applies_per_material_overrides():
    config = json.loads((EXAMPLE / 'builder_materials.json').read_text())
    candidates = [build_surface(f'pool-{i}', config, EXAMPLE) for i in range(60)]
    counts = Counter(item.metadata['surface_builder']['material_id'] for item in candidates)
    assert set(counts) == {'copper', 'mapi'}
    assert counts['copper'] > counts['mapi']
    first = build_surface('pool-0', config, EXAMPLE)
    assert first.metadata == candidates[0].metadata
    np.testing.assert_array_equal(first.atoms.positions, candidates[0].atoms.positions)
    for item in candidates:
        metadata = item.metadata['surface_builder']
        if metadata['material_id'] == 'copper':
            assert item.atoms[-2:].get_chemical_symbols() == ['C', 'O']
            assert metadata['slab_symmetric'] is True
        else:
            assert _miller_index_of(item) == (1, 0, 0)
            assert not (item.atoms.get_tags() == 2).any()
    config['materials'][1]['bulk_charge'] = 1
    with pytest.raises(ValueError, match="Material 'mapi': .*nonzero net charge"):
        build_surface('pool-0', config, EXAMPLE)


@pytest.mark.parametrize('automatic', [False, True])
def test_coordinator_test_builder_mode_uses_surface_app(copper, tmp_path, monkeypatch, capsys, automatic):
    @python_app(executors=['alf_sampler_executor'])
    def sampler(molecule_object):
        raise AssertionError('builder test should stop before sampling')

    @python_app(executors=['alf_QM_executor'])
    def qm(molecule_object):
        raise AssertionError('builder test should stop before labeling')

    @python_app(executors=['alf_ML_executor'])
    def ml():
        raise AssertionError('builder test should stop before training')

    config = dict(copper, adsorbates=[_co_placement()])
    if automatic:
        material = dict(config, id='copper')
        material.pop('miller_indices')
        config = dict(materials=[dict(material, max_miller=1, max_atoms=250, slab_symmetry='require')],
                      min_a_A=8, max_a_A=13, min_b_A=7, max_b_A=13)
    with pytest.raises(SystemExit) as stopped:
        run_alf_driver(tmp_path, monkeypatch, builder=surface_builder_task, sampler=sampler, qm=qm, ml=ml,
                       builder_config=config,
                       arguments=['--test_builder'], parallel_samplers=1, target_queued_QM=1)
    assert stopped.value.code in (0, None)
    assert 'Builder testing returned:\nSystem: C' in capsys.readouterr().out
