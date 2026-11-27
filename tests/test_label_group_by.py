"""label_h5_layout group_by "atom_count": frames are grouped by atom count, with species for each frame, instead of
one group per sorted species list. Every frame keeps the same values; only the grouping changes."""
import json
import warnings

import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools import label_hdf5
from alframework.tools.data_shards import group_key, store_current_data
from alframework.tools.h5_parallel import read_shard
from alframework.tools.label_hdf5 import GROUP_BY
from alframework.tools.label_parts import HDF5HeldLabels, incoming_directory, part_directory, write_label_part
from alframework.tools.molecules import MoleculesObject
from alframework.tools.restarts import label_h5_options, shard_frames

PROPERTIES = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}
SCHEMA = json.dumps(PROPERTIES, sort_keys=True)
BY_ATOM_COUNT = {'group_by': 'atom_count'}
# Molecules of six atoms in four compositions, one of four atoms and one of seven.
FORMULAS = ('C2H4', 'CH3OH', 'N2H4', 'H2O2', 'CH3CN', 'C2H5')


def _labels(count, formulas=FORMULAS, periodic=False):
    """``count`` labels of ``formulas`` in turn; with ``periodic``, one more, a periodic formic acid (five atoms)."""
    labels = []
    rng = np.random.default_rng(0)
    for index in range(count + bool(periodic)):
        atoms = Atoms(formulas[index % len(formulas)] if index < count else 'HCOOH')
        atoms.set_positions(rng.normal(size=(len(atoms), 3)) * 2)
        if index == count:
            atoms.set_cell([12.0, 12.0, 12.0])
            atoms.set_pbc(True)
        molecule = MoleculesObject(atoms, f'mol-{index:04d}')
        molecule.store_results({'energy': -1.0 - index, 'forces': rng.normal(size=(len(atoms), 3))})
        molecule.set_converged_flag(True)
        if index % 4 == 1:
            molecule.update_metadata({'charge': -1, 'mult': 2})
        labels.append(molecule)
    return labels


def _frames(path):
    """Every stored frame by ID: its species and the values of each column, whatever the grouping."""
    frames = {}
    for group in read_shard(str(path), workers=1):
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in group['_id']]
        species = np.asarray(group['species'])
        for index, name in enumerate(ids):
            row = {key: np.asarray(value)[index] for key, value in group.items()
                   if key not in ('path', '_id', 'species')}
            row['species'] = [value.decode() if isinstance(value, bytes) else str(value)
                              for value in (species[index] if species.ndim > 1 else species)]
            frames[name] = row
    return frames


def _assert_same_frames(first, second):
    assert first.keys() == second.keys()
    for name, row in first.items():
        assert row.keys() == second[name].keys(), name
        for key, value in row.items():
            np.testing.assert_array_equal(value, second[name][key], err_msg=f'{name}:{key}')


def test_group_names_carry_atom_count_periodicity_and_topology_tagging():
    atoms = Atoms('CH4')
    assert group_key(atoms) == 'C01_H04'
    assert group_key(atoms, 'atom_count') == 'atoms_0005'
    atoms.set_cell([5, 5, 5])
    atoms.set_pbc(True)
    assert group_key(atoms, 'atom_count') == 'atoms_0005_periodic'
    atoms.set_pbc([True, True, False])
    assert group_key(atoms, 'atom_count') == 'atoms_0005_pbc110'
    atoms.set_pbc(False)
    atoms.set_array('topology_atom_ids', np.arange(5))
    assert group_key(atoms, 'atom_count') == 'atoms_0005_tagged'


def test_a_shard_grouped_by_atom_count_holds_the_same_frames(tmp_path):
    labels = _labels(12, periodic=True)
    store_current_data(str(tmp_path / 'formula.h5'), labels, PROPERTIES)
    store_current_data(str(tmp_path / 'atoms.h5'), labels, PROPERTIES, h5_options=BY_ATOM_COUNT)
    with h5py.File(tmp_path / 'atoms.h5') as handle:
        assert handle.attrs[GROUP_BY] == 'atom_count'
        assert sorted(handle) == ['atoms_0004', 'atoms_0005_periodic', 'atoms_0006', 'atoms_0007']
        species = handle['atoms_0006']['species']
        assert species.shape == (len(handle['atoms_0006']['_id']), 6), 'species for each frame'
    with h5py.File(tmp_path / 'formula.h5') as handle:
        assert GROUP_BY not in handle.attrs, 'the default layout is written as before'
        assert len(handle) == 7, 'six formulas and the periodic one'
    _assert_same_frames(_frames(tmp_path / 'atoms.h5'), _frames(tmp_path / 'formula.h5'))
    assert shard_frames(tmp_path / 'atoms.h5') == 13


def test_species_within_each_frame_are_sorted_as_in_formula_groups(tmp_path):
    labels = _labels(6)
    store_current_data(str(tmp_path / 'atoms.h5'), labels, PROPERTIES, h5_options=BY_ATOM_COUNT)
    frames = _frames(tmp_path / 'atoms.h5')
    for molecule in labels:
        numbers = molecule.get_atoms().get_atomic_numbers()
        expected = [molecule.get_atoms().get_chemical_symbols()[i] for i in np.argsort(numbers)]
        assert frames[molecule.get_moleculeid()]['species'] == expected


@pytest.mark.parametrize('merge_workers', [1, 2])
def test_worker_parts_grouped_by_atom_count_publish_the_same_frames(tmp_path, merge_workers):
    labels = _labels(23)
    control = tmp_path / 'control' / 'data-0000.h5'
    held = HDF5HeldLabels(control, PROPERTIES, batch_size=5)
    held.hold(labels)
    assert held.commit(control, PROPERTIES, {}) == 23

    snapshot = tmp_path / 'atoms' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=5, h5_options=BY_ATOM_COUNT, merge_workers=merge_workers)
    receipts = [write_label_part(labels[start:start + 4], incoming_directory(snapshot), PROPERTIES, SCHEMA,
                                 h5_options=BY_ATOM_COUNT) for start in range(0, 23, 4)]
    assert set(receipts[0].formulas) == {'atoms_0004', 'atoms_0006'}
    held.adopt(receipts)
    assert held.commit(snapshot, PROPERTIES, {}) == 23
    with h5py.File(snapshot) as handle:
        assert handle.attrs[GROUP_BY] == 'atom_count'
        assert sorted(handle) == ['atoms_0004', 'atoms_0006', 'atoms_0007']
        assert handle['atoms_0006']['species'].is_virtual
    for part in part_directory(snapshot).glob('part-*.h5'):
        with h5py.File(part) as handle:
            assert handle.attrs[GROUP_BY] == 'atom_count'
    _assert_same_frames(_frames(snapshot), _frames(control))


def test_parts_held_by_the_coordinator_group_by_atom_count_too(tmp_path):
    labels = _labels(11)
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4, h5_options=BY_ATOM_COUNT)
    held.hold(labels[:6])
    held.hold(labels[6:])
    assert (held.successful(), held.converged()) == (11, 11)
    assert held.commit(snapshot, PROPERTIES, {}) == 11
    control = tmp_path / 'control.h5'
    store_current_data(str(control), labels, PROPERTIES)
    _assert_same_frames(_frames(snapshot), _frames(control))


def test_a_restart_counts_an_atom_count_generation_from_its_parts(tmp_path):
    labels = _labels(9)
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4, h5_options=BY_ATOM_COUNT)
    held.adopt([write_label_part(labels[:5], incoming_directory(snapshot), PROPERTIES, SCHEMA,
                                 h5_options=BY_ATOM_COUNT)])
    held.hold(labels[5:])
    restarted = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4, h5_options=BY_ATOM_COUNT)
    assert (restarted.successful(), restarted.converged()) == (9, 9)
    assert restarted.molecule_ids() == {m.get_moleculeid() for m in labels}
    assert restarted.commit(snapshot, PROPERTIES, {}) == 9


def test_a_generation_refuses_parts_grouped_another_way(tmp_path):
    labels = _labels(4)
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4, h5_options=BY_ATOM_COUNT)
    held.adopt([write_label_part(labels, incoming_directory(snapshot), PROPERTIES, SCHEMA)])
    with pytest.raises(ValueError, match='groups its labels by formula'):
        held.commit(snapshot, PROPERTIES, {})


def test_the_layout_setting_accepts_only_known_groupings():
    assert label_h5_options({'label_h5_layout': {'group_by': 'atom_count'}})['group_by'] == 'atom_count'
    assert 'group_by' not in label_h5_options({'label_h5_layout': {'compression': 'lzf'}})
    with pytest.raises(ValueError, match='group_by must be one of'):
        label_h5_options({'label_h5_layout': {'group_by': 'species'}})


def test_many_formula_groups_per_label_warn_and_name_the_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(label_hdf5, 'MANY_GROUPS', 3)
    labels = _labels(12)
    with pytest.warns(UserWarning, match='group_by "atom_count"'):
        store_current_data(str(tmp_path / 'formula.h5'), labels, PROPERTIES)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        store_current_data(str(tmp_path / 'atoms.h5'), labels, PROPERTIES, h5_options=BY_ATOM_COUNT)


def test_seed_geometries_read_from_an_atom_count_shard(tmp_path):
    from alframework.builders.external_geometry import _h5_frame
    labels = _labels(6)
    store_current_data(str(tmp_path / 'atoms.h5'), labels, PROPERTIES, h5_options=BY_ATOM_COUNT)
    with h5py.File(tmp_path / 'atoms.h5') as handle:
        group = handle['atoms_0006']
        for index, name in enumerate(group['_id'].asstr()[:]):
            atoms, source = _h5_frame(group, index)
            original = next(m.get_atoms() for m in labels if m.get_moleculeid() == name)
            assert source == name
            assert sorted(atoms.get_chemical_symbols()) == sorted(original.get_chemical_symbols())


def test_fine_tuning_reads_atom_count_shards_frame_by_frame(tmp_path):
    from alframework.ml_interfaces.fine_tuning import shard_atoms
    labels = _labels(7)
    store_current_data(str(tmp_path / 'formula.h5'), labels, PROPERTIES)
    store_current_data(str(tmp_path / 'atoms.h5'), labels, PROPERTIES, h5_options=BY_ATOM_COUNT)

    def by_energy(path):
        return {round(atoms.info['REF_energy'], 9): atoms for atoms in shard_atoms([str(path)], read_workers=1)}

    formula, atoms = by_energy(tmp_path / 'formula.h5'), by_energy(tmp_path / 'atoms.h5')
    assert formula.keys() == atoms.keys()
    for energy, frame in formula.items():
        assert frame.get_chemical_symbols() == atoms[energy].get_chemical_symbols()
        np.testing.assert_array_equal(frame.positions, atoms[energy].positions)
        np.testing.assert_array_equal(frame.arrays['REF_forces'], atoms[energy].arrays['REF_forces'])


@pytest.mark.parametrize('layouts', [('formula', 'atom_count'), ('atom_count',)])
def test_hippynn_reads_atom_count_shards_and_mixed_directories_as_formula_shards(tmp_path, layouts):
    pytest.importorskip('hippynn')
    from alframework.ml_interfaces.hippynn.interface import ALFPyAniDirectoryDB
    labels = _labels(14)

    def load(directory):
        database = ALFPyAniDirectoryDB(directory=str(directory), inputs=None, targets=None, allow_unfound=True,
                                       quiet=True, seed=1, read_workers=1, species_key='species',
                                       coordinates_key='coordinates', atom_keys=('coordinates', 'forces'))
        # HIPPYNN keeps no string columns, so frames are matched by their energies, unique in these labels.
        energies = np.asarray(database.arr_dict['energy']).reshape(-1)
        return database, {round(float(value), 9): index for index, value in enumerate(energies)}

    (tmp_path / 'control').mkdir()
    store_current_data(str(tmp_path / 'control' / 'data.h5'), labels, PROPERTIES)
    (tmp_path / 'mixed').mkdir()
    halves = (labels, ) if len(layouts) == 1 else (labels[:7], labels[7:])
    for index, (layout, chunk) in enumerate(zip(layouts, halves)):
        store_current_data(str(tmp_path / 'mixed' / f'data-{index}.h5'), chunk, PROPERTIES,
                           h5_options={'group_by': layout})
    control, control_rows = load(tmp_path / 'control')
    mixed, mixed_rows = load(tmp_path / 'mixed')
    assert control_rows.keys() == mixed_rows.keys()
    order = [mixed_rows[name] for name in control_rows]
    for key in ('species', 'coordinates', 'energy', 'forces'):
        np.testing.assert_array_equal(np.asarray(control.arr_dict[key])[list(control_rows.values())],
                                      np.asarray(mixed.arr_dict[key])[order], err_msg=key)



def test_a_stored_shard_is_described_by_its_groups_and_grouping(tmp_path):
    from alframework.tools.restarts import describe_shard
    labels = _labels(12)
    store_current_data(str(tmp_path / 'formula.h5'), labels, PROPERTIES)
    store_current_data(str(tmp_path / 'atoms.h5'), labels, PROPERTIES, h5_options=BY_ATOM_COUNT)
    assert describe_shard(tmp_path / 'formula.h5', 12) == '12 labels in 6 groups (by formula)'
    assert describe_shard(tmp_path / 'atoms.h5', 12) == '12 labels in 3 groups (by atom_count)'


def test_species_bytes_convert_to_atomic_numbers_as_symbols_do():
    pytest.importorskip('hippynn')
    from alframework.ml_interfaces.hippynn.interface import _species_numbers
    symbols = np.array([['H', 'C', 'O', 'Cl'], ['Zn', 'F', 'H', 'H']])
    expected = [[1, 6, 8, 17], [30, 9, 1, 1]]
    np.testing.assert_array_equal(_species_numbers(np.char.encode(symbols, 'utf-8')), expected)
    np.testing.assert_array_equal(_species_numbers(symbols), expected)
    np.testing.assert_array_equal(_species_numbers(np.char.encode(symbols[0], 'utf-8')), [expected[0]])
    with pytest.raises(KeyError):
        _species_numbers(np.array([[b'Xx']]))


def test_the_neurochem_interface_refuses_atom_count_groups():
    from alframework.tools.restarts import check_label_grouping_for_ml
    neurochem = 'alframework.ml_interfaces.neurochem_interface.train_ANI_model_task'
    with pytest.raises(ValueError, match='NeuroChem'):
        check_label_grouping_for_ml(BY_ATOM_COUNT, neurochem)
    check_label_grouping_for_ml({}, neurochem)
    check_label_grouping_for_ml(BY_ATOM_COUNT, 'alframework.ml_interfaces.hippynn.interface.train_HIPNN_model_task')


EXCITED = {'E0': ['E0', 'system', 1.0], 'F0': ['F0', 'atomic', 1.0], 'E1': ['E1', 'system', 1.0],
           'F1': ['F1', 'atomic', 1.0], 'nacs': ['nacs', 'pair_atomic', 1.0]}


def _excited_labels(count):
    rng = np.random.default_rng(1)
    labels = []
    for index in range(count):
        atoms = Atoms('OH2', positions=rng.normal(size=(3, 3)))
        molecule = MoleculesObject(atoms, f'exc-{index:03d}')
        molecule.store_results({'E0': -2.0 - index, 'F0': rng.normal(size=(3, 3)), 'E1': -1.0 - index,
                                'F1': rng.normal(size=(3, 3)), 'nacs': rng.normal(size=(1, 3, 3))})
        molecule.update_metadata({'nac_pairs': np.array([[0, 1]])})
        molecule.set_converged_flag(True)
        labels.append(molecule)
    return labels


def test_excited_state_labels_with_pair_order_read_the_same_in_both_groupings(tmp_path):
    pytest.importorskip('hippynn')
    from alframework.ml_interfaces.hippynn import interface as ml
    labels = _excited_labels(9)
    trainer = {key: spec for key, spec in EXCITED.items() if key != 'nacs'}
    loaded = {}
    for group_by in ('formula', 'atom_count'):
        snapshot = tmp_path / group_by / 'data-0000.h5'
        held = HDF5HeldLabels(snapshot, EXCITED, batch_size=4, h5_options={'group_by': group_by})
        held.hold(labels)
        held.commit(snapshot, EXCITED, {})
        with h5py.File(snapshot) as handle:
            group, = handle.values()
            np.testing.assert_array_equal(group['nac_pairs'][()], [[0, 1]])
            assert group['nacs'].shape == (9, 1, 3, 3)
        arrays, summary = ml.load_excited_state_h5_arrays(str(snapshot), trainer)
        order = np.argsort([frame['id'] for frame in summary['frames']])
        loaded[group_by] = {key: np.asarray(value)[order] if np.ndim(value) and len(value) == 9 else value
                            for key, value in arrays.items() if key != 'indices'}
    assert loaded['formula'].keys() == loaded['atom_count'].keys()
    for key, value in loaded['formula'].items():
        np.testing.assert_array_equal(value, loaded['atom_count'][key], err_msg=key)


def test_pair_orders_that_differ_within_an_atom_count_group_are_refused(tmp_path):
    labels = _excited_labels(2)
    labels[1].update_metadata({'nac_pairs': np.array([[1, 2]])})
    with pytest.raises(RuntimeError, match='different'):
        store_current_data(str(tmp_path / 'atoms.h5'), labels, EXCITED, h5_options=BY_ATOM_COUNT)
