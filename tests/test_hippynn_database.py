"""Real-reader regressions for small ALF molecules and existing larger inputs."""
import pickle
import subprocess
import sys

import h5py
import numpy as np
import pytest

pytest.importorskip('hippynn')
from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB
from hippynn.databases.h5_pyanitools import PyAniDirectoryDB


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
def test_small_groups_pad_only_atom_axis_and_preserve_components(tmp_path, sizes):
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


def test_custom_schema_survives_checkpoint_restarter(tmp_path):
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


def test_existing_larger_groups_match_upstream_arrays(tmp_path):
    write_dataset(tmp_path/'data.h5', sizes=(8, 10))
    kwargs = dict(directory=str(tmp_path), inputs=None, targets=None, allow_unfound=True,
                  quiet=True, seed=1)
    actual, expected = ALFPyAniDirectoryDB(**kwargs), PyAniDirectoryDB(**kwargs)
    assert set(actual.arr_dict) == set(expected.arr_dict)
    for key, value in expected.arr_dict.items():
        np.testing.assert_array_equal(actual.arr_dict[key].numpy(), value.numpy())


def test_unknown_varying_property_is_rejected_instead_of_guessing_axis(tmp_path):
    write_dataset(tmp_path/'data.h5')
    with h5py.File(tmp_path/'data.h5', 'a') as handle:
        handle['group-0']['custom_atomic'] = np.zeros((2, 3))
        handle['group-1']['custom_atomic'] = np.zeros((2, 6))
    with pytest.raises(ValueError, match='custom_atomic'):
        ALFPyAniDirectoryDB(str(tmp_path), inputs=None, targets=None, allow_unfound=True,
                            quiet=True, seed=1)
