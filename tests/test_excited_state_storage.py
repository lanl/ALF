"""Excited-state storage preserves explicit pair and topology identities."""
import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools.atom_order import canonical_orders, validated_atom_ids
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import store_current_data

PROPERTIES = {'sE0': ['e0', 'system', 2.0], 'sE1': ['e1', 'system', 2.0],
              'F1': ['f1', 'atomic', 3.0], 'nacr': ['nac', 'pair_atomic', 4.0]}


def molecule(order=(0, 1, 2, 3), tagged=True):
    atoms = Atoms('CHHO', positions=np.arange(12).reshape(4, 3) / 10)
    if tagged:
        atoms.set_array('topology_atom_ids', np.arange(4))
    mol = MoleculesObject(atoms[list(order)], 'frame-' + ''.join(map(str, order)))
    forces = np.arange(12).reshape(4, 3).astype(float)
    mol.store_results({'sE0': -2., 'sE1': 1., 'F1': forces[list(order)],
                       'nacr': np.stack([forces, forces + 100])[:, list(order)]})
    mol.update_metadata({'nac_pairs': [[1, 2], [1, 3]]})
    mol.set_converged_flag(True)
    return mol


def test_multistate_pair_atomic_and_same_element_identity_roundtrip(tmp_path):
    path = tmp_path/'labels.h5'
    store_current_data(str(path), [molecule(), molecule((2, 3, 0, 1))], PROPERTIES)
    with h5py.File(path) as h5:
        g = next(iter(h5.values()))
        orders = canonical_orders(g['topology_atom_ids'][:], 2, 4)
        np.testing.assert_array_equal(g['nac_pairs'][:], [[1, 2], [1, 3]])
        np.testing.assert_allclose(g['e0'][:], [-4, -4])
        np.testing.assert_allclose(g['e1'][:] - g['e0'][:], [6, 6])
        for i, order in enumerate(orders):
            np.testing.assert_allclose(g['coordinates'][i][order], np.arange(12).reshape(4, 3)/10)
            np.testing.assert_allclose(g['f1'][i][order], np.arange(12).reshape(4, 3)*3)
            np.testing.assert_allclose(g['nac'][i][:, order],
                                      np.stack([np.arange(12).reshape(4, 3),
                                                np.arange(12).reshape(4, 3)+100])*4)


@pytest.mark.parametrize('pairs', [None, [[1, 2]], [[1, 2], [1, 2]],
                                   [[2, 1], [1, 3]], [[1.5, 2], [1, 3]]])
def test_bad_pair_index_rejected_before_write(tmp_path, pairs):
    mol = molecule()
    mol.update_metadata({'nac_pairs': pairs})
    with pytest.raises((ValueError, RuntimeError)):
        store_current_data(str(tmp_path/'bad.h5'), [mol], PROPERTIES)
    assert not (tmp_path/'bad.h5').exists()


def test_changed_pair_mapping_and_mixed_topology_are_rejected(tmp_path):
    a, b = molecule(), molecule()
    b.update_metadata({'nac_pairs': [[1, 2], [2, 3]]})
    with pytest.raises(RuntimeError, match='different'):
        store_current_data(str(tmp_path/'bad.h5'), [a, b], PROPERTIES)
    with pytest.raises(ValueError, match='tagged'):
        store_current_data(str(tmp_path/'bad.h5'), [a, molecule(tagged=False)], PROPERTIES)


def test_topology_ids_require_explicit_permutation():
    for ids in ([0, 0, 2], [0., 1., 2.], [0, 1, 3]):
        with pytest.raises(ValueError):
            validated_atom_ids(ids, 3)
