"""ALF's assembly of label groups into whole padded arrays (alframework.tools.label_arrays), without any trainer."""
import json

import h5py
import numpy as np
import pytest

from alframework.tools.label_arrays import assemble_label_arrays, atom_axes, shard_atom_axes
from alframework.tools.label_hdf5 import PROPERTY_SCHEMA

AXES = {'species': 1, 'coordinates': 1, 'forces': 1}


def _group(frames, atoms, offset=0.0, **extra):
    group = dict(species=np.arange(1, atoms + 1)[None], energy=np.arange(frames) + offset,
                 coordinates=np.full((frames, atoms, 3), offset), forces=np.full((frames, atoms, 3), -offset),
                 cell=np.repeat(np.eye(3)[None], frames, axis=0))
    group.update(extra)
    return group


def test_groups_are_padded_to_the_most_atoms_and_species_repeat_for_each_frame():
    arrays = assemble_label_arrays([_group(2, 3, 1.0), _group(3, 5, 2.0)], AXES, 'species', 'coordinates')
    assert arrays['coordinates'].shape == (5, 5, 3) and arrays['cell'].shape == (5, 3, 3)
    np.testing.assert_array_equal(arrays['species'], [[1, 2, 3, 0, 0]] * 2 + [[1, 2, 3, 4, 5]] * 3)
    np.testing.assert_array_equal(arrays['coordinates'][:2, 3:], 0.0)
    np.testing.assert_array_equal(arrays['coordinates'][2:], 2.0)
    np.testing.assert_array_equal(arrays['energy'], [1, 2, 2, 3, 4])
    np.testing.assert_array_equal(arrays['sys_number'], [0, 0, 1, 1, 1])


def test_species_stored_for_each_frame_is_kept_per_frame():
    group = _group(2, 3, species=np.array([[1, 1, 8], [1, 6, 8]]))
    arrays = assemble_label_arrays([group, _group(1, 4)], AXES, 'species', 'coordinates')
    np.testing.assert_array_equal(arrays['species'], [[1, 1, 8, 0], [1, 6, 8, 0], [1, 2, 3, 4]])


def test_pair_atomic_properties_pad_their_atom_axis():
    axes = dict(AXES, nacs=2)
    groups = [_group(2, 3, nacs=np.ones((2, 1, 3, 3))), _group(1, 4, nacs=np.ones((1, 1, 4, 3)))]
    arrays = assemble_label_arrays(groups, axes, 'species', 'coordinates')
    assert arrays['nacs'].shape == (3, 1, 4, 3)
    np.testing.assert_array_equal(arrays['nacs'][:2, :, 3], 0.0)


def test_a_column_of_unknown_shape_is_refused_not_guessed():
    groups = [_group(2, 3, custom=np.zeros((2, 3))), _group(2, 4, custom=np.zeros((2, 4)))]
    with pytest.raises(ValueError, match="'custom' has different shapes"):
        assemble_label_arrays(groups, AXES, 'species', 'coordinates')
    arrays = assemble_label_arrays(groups, dict(AXES, custom=1), 'species', 'coordinates')
    assert arrays['custom'].shape == (4, 4)


def test_state_columns_some_groups_lack_take_their_defaults():
    groups = [_group(2, 3, total_charge=np.array([-1, -1]), spin_multiplicity=np.array([2, 2])), _group(1, 3)]
    arrays = assemble_label_arrays(groups, AXES, 'species', 'coordinates')
    np.testing.assert_array_equal(arrays['total_charge'], [-1, -1, 0])
    np.testing.assert_array_equal(arrays['spin_multiplicity'], [2, 2, 1])
    with pytest.raises(ValueError, match="'dipole' is stored in some"):
        assemble_label_arrays([_group(1, 3, dipole=np.zeros((1, 3))), _group(1, 3)], AXES, 'species', 'coordinates')


def test_rows_must_match_the_group_frame_count():
    with pytest.raises(ValueError, match="'energy' has 3 rows in a group of 2 frames"):
        assemble_label_arrays([_group(2, 3, energy=np.zeros(3))], AXES, 'species', 'coordinates')


def test_text_columns_keep_their_widest_value():
    groups = [_group(1, 3, _id=np.array(['a'])), _group(1, 3, _id=np.array(['longer-id']))]
    np.testing.assert_array_equal(assemble_label_arrays(groups, AXES, 'species', 'coordinates')['_id'],
                                  ['a', 'longer-id'])


def test_atom_axes_come_from_each_shards_property_schema(tmp_path):
    schema = {'energy': ['energy', 'system', 1.0], 'forces': ['grad', 'atomic', 1.0],
              'nacs': ['nac', 'pair_atomic', 1.0]}
    with h5py.File(tmp_path / 'a.h5', 'w') as handle:
        handle.attrs[PROPERTY_SCHEMA] = json.dumps(schema)
    with h5py.File(tmp_path / 'b.h5', 'w'):
        pass  # a shard written before ALF recorded its schema
    files = [str(tmp_path / 'a.h5'), str(tmp_path / 'b.h5')]
    assert shard_atom_axes(files) == {'grad': 1, 'nac': 2}
    assert atom_axes(files, 'species', 'xyz', ('charges',)) == {'grad': 1, 'nac': 2, 'species': 1, 'xyz': 1,
                                                                'topology_atom_ids': 1, 'charges': 1}
    with h5py.File(tmp_path / 'b.h5', 'a') as handle:
        handle.attrs[PROPERTY_SCHEMA] = json.dumps({'forces': ['grad', 'pair_atomic', 1.0]})
    with pytest.raises(ValueError, match="'grad' is stored as 'pair_atomic'"):
        shard_atom_axes(files)
