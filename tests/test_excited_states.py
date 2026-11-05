"""Excited-state key contracts and topology-aware multi-state/NAC storage."""
import pytest
import h5py
import numpy as np
from ase import Atoms

from alframework.tools.excited_state_tools import (
    derive_gap_property_table,
    derive_state_property_table,
    gap_key_for_pair,
    parse_gap_key,
)
from alframework.tools.atom_order import canonical_orders, validated_atom_ids
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import store_current_data


def test_flattened_excited_state_contract():
    table = derive_state_property_table(
        {
            "sE0": ["singlet_0_energy", "system", 1.0],
            "F0": ["singlet_0_forces", "atomic", 1.0],
            "sE1": ["singlet_1_energy", "system", 1.0],
            "F1": ["singlet_1_forces", "atomic", 1.0],
        },
        require_forces=True,
    )

    assert [row["state"] for row in table] == [0, 1]
    assert table[1]["energy_key"] == "sE1"
    assert table[1]["force_db_name"] == "singlet_1_forces"


def test_excited_state_contract_requires_contiguous_energies():
    with pytest.raises(ValueError, match="contiguous"):
        derive_state_property_table(
            {"sE0": ["e0"], "F0": ["f0"], "sE2": ["e2"], "F2": ["f2"]},
            require_forces=True,
        )


def test_excited_state_contract_requires_each_force_for_sampling():
    with pytest.raises(ValueError, match="F1"):
        derive_state_property_table(
            {"sE0": ["e0"], "F0": ["f0"], "sE1": ["e1"]},
            require_forces=True,
        )


def _gap_properties():
    return {
        "sE0": ["sE0", "system", 2.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 2.0],
        "F1": ["F1", "atomic", 1.0],
        "sE2": ["sE2", "system", 2.0],
        "F2": ["F2", "atomic", 1.0],
        "dE01": ["gap_01", "system", 2.0],
        "dE12": ["gap_12", "system", 2.0],
    }


def test_gap_keys_support_compact_and_multi_digit_forms():
    assert parse_gap_key("dE01") == (0, 1)
    assert parse_gap_key("dE0_10") == (0, 10)
    assert parse_gap_key("sE1") is None
    assert gap_key_for_pair(1, 2) == "dE12"
    assert gap_key_for_pair(0, 10) == "dE0_10"


def test_gap_table_defaults_to_properties_and_supports_pair_selection():
    properties = _gap_properties()
    all_rows = derive_gap_property_table(properties)
    selected = derive_gap_property_table(
        properties,
        gap_config={"pairs": [[1, 2]]},
        require_properties=True,
    )

    assert [row["gap_key"] for row in all_rows] == ["dE01", "dE12"]
    assert selected == [all_rows[1]]
    assert selected[0]["gap_db_name"] == "gap_12"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda properties: properties.__setitem__(
                "dE01", ["gap_01", "atomic", 2.0]
            ),
            "system property",
        ),
        (
            lambda properties: properties.__setitem__(
                "dE01", ["gap_01", "system", 1.0]
            ),
            "identical property scales",
        ),
    ],
)
def test_gap_table_rejects_bad_schema_and_scale(mutator, message):
    properties = _gap_properties()
    mutator(properties)
    with pytest.raises(ValueError, match=message):
        derive_gap_property_table(properties)


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([[1, 0]], "lower to higher"),
        ([[0, 1], [0, 1]], "Duplicate"),
        ([[0, 3]], "outside"),
        ([[0, 2]], "requires"),
        ([[0, 1.5]], "nonnegative integer"),
    ],
)
def test_gap_table_rejects_invalid_configured_pairs(pairs, message):
    with pytest.raises(ValueError, match=message):
        derive_gap_property_table(
            _gap_properties(),
            gap_config={"pairs": pairs},
            require_properties=True,
        )


def test_state_scales_and_orphan_forces_cannot_silently_disagree():
    with pytest.raises(ValueError, match='identical'):
        derive_state_property_table({'sE0': ['e0', 'system', 1.], 'sE1': ['e1', 'system', 2.]})
    with pytest.raises(ValueError, match='without energies'):
        derive_state_property_table({'sE0': ['e0', 'system', 1.], 'F1': ['f1', 'atomic', 1.]})


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
