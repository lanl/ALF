"""Excited-state labels: property names (sE#, F#, dE##), gap tables, and storing multi-state and NACR
labels in HDF5 with atom identity kept.
"""

import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools.atom_order import canonical_orders, validated_atom_ids
from alframework.tools.excited_state_tools import (
    derive_gap_property_table,
    derive_state_property_table,
    gap_key_for_pair,
    parse_gap_key,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import store_current_data


def test_state_table_lists_each_state():
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


@pytest.mark.parametrize(
    ("properties", "require_forces", "message"),
    [
        ({"sE0": ["e0"], "F0": ["f0"], "sE2": ["e2"], "F2": ["f2"]}, True, "contiguous"),
        ({"sE0": ["e0"], "F0": ["f0"], "sE1": ["e1"]}, True, "F1"),
        ({"sE0": ["e0", "system", 1.0], "sE1": ["e1", "system", 2.0]}, False, "identical"),
        ({"sE0": ["e0", "system", 1.0], "F1": ["f1", "atomic", 1.0]}, False, "without energies"),
    ],
    ids=["skipped_state", "missing_force", "different_energy_scales", "force_without_energy"],
)
def test_state_table_rejects_inconsistent_properties(properties, require_forces, message):
    with pytest.raises(ValueError, match=message):
        derive_state_property_table(properties, require_forces=require_forces)


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


def test_gap_keys_handle_multi_digit_states():
    assert parse_gap_key("dE01") == (0, 1)
    assert parse_gap_key("dE0_10") == (0, 10)
    assert parse_gap_key("sE1") is None
    assert gap_key_for_pair(1, 2) == "dE12"
    assert gap_key_for_pair(0, 10) == "dE0_10"


def test_gap_table_can_select_pairs():
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
            lambda properties: properties.__setitem__("dE01", ["gap_01", "atomic", 2.0]),
            "system property",
        ),
        (
            lambda properties: properties.__setitem__("dE01", ["gap_01", "system", 1.0]),
            "identical property scales",
        ),
    ],
)
def test_gap_table_rejects_wrong_kind_or_scale(mutator, message):
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
    ids=["reversed", "duplicate", "out_of_range", "missing_property", "non_integer"],
)
def test_gap_table_rejects_invalid_pairs(pairs, message):
    with pytest.raises(ValueError, match=message):
        derive_gap_property_table(
            _gap_properties(),
            gap_config={"pairs": pairs},
            require_properties=True,
        )


PROPERTIES = {
    "sE0": ["e0", "system", 2.0],
    "sE1": ["e1", "system", 2.0],
    "F1": ["f1", "atomic", 3.0],
    "nacr": ["nac", "pair_atomic", 4.0],
}


def molecule(order=(0, 1, 2, 3), tagged=True):
    atoms = Atoms("CHHO", positions=np.arange(12).reshape(4, 3) / 10)
    if tagged:
        atoms.set_array("topology_atom_ids", np.arange(4))
    mol = MoleculesObject(atoms[list(order)], "frame-" + "".join(map(str, order)))
    forces = np.arange(12).reshape(4, 3).astype(float)
    mol.store_results(
        {"sE0": -2.0, "sE1": 1.0, "F1": forces[list(order)], "nacr": np.stack([forces, forces + 100])[:, list(order)]}
    )
    mol.update_metadata({"nac_pairs": [[1, 2], [1, 3]]})
    mol.set_converged_flag(True)
    return mol


def test_stored_nacr_labels_follow_topology_ids(tmp_path):
    path = tmp_path / "labels.h5"
    store_current_data(str(path), [molecule(), molecule((2, 3, 0, 1))], PROPERTIES)
    with h5py.File(path) as h5:
        g = next(iter(h5.values()))
        orders = canonical_orders(g["topology_atom_ids"][:], 2, 4)
        np.testing.assert_array_equal(g["nac_pairs"][:], [[1, 2], [1, 3]])
        np.testing.assert_allclose(g["e0"][:], [-4, -4])
        np.testing.assert_allclose(g["e1"][:] - g["e0"][:], [6, 6])
        for i, order in enumerate(orders):
            np.testing.assert_allclose(g["coordinates"][i][order], np.arange(12).reshape(4, 3) / 10)
            np.testing.assert_allclose(g["f1"][i][order], np.arange(12).reshape(4, 3) * 3)
            np.testing.assert_allclose(
                g["nac"][i][:, order], np.stack([np.arange(12).reshape(4, 3), np.arange(12).reshape(4, 3) + 100]) * 4
            )


@pytest.mark.parametrize(
    "pairs",
    [None, [[1, 2]], [[1, 2], [1, 2]], [[2, 1], [1, 3]], [[1.5, 2], [1, 3]]],
    ids=["missing", "too_few", "duplicate", "out_of_order", "non_integer"],
)
def test_bad_nac_pairs_block_the_write(tmp_path, pairs):
    mol = molecule()
    mol.update_metadata({"nac_pairs": pairs})
    with pytest.raises((ValueError, RuntimeError), match="nac_pairs"):
        store_current_data(str(tmp_path / "bad.h5"), [mol], PROPERTIES)
    assert not (tmp_path / "bad.h5").exists()


def test_mixed_nac_pairs_or_topology_cannot_share_a_file(tmp_path):
    a, b = molecule(), molecule()
    b.update_metadata({"nac_pairs": [[1, 2], [2, 3]]})
    with pytest.raises(RuntimeError, match="different"):
        store_current_data(str(tmp_path / "bad.h5"), [a, b], PROPERTIES)
    with pytest.raises(ValueError, match="tagged"):
        store_current_data(str(tmp_path / "bad.h5"), [a, molecule(tagged=False)], PROPERTIES)


def test_topology_ids_must_be_a_permutation():
    for ids in ([0, 0, 2], [0.0, 1.0, 2.0], [0, 1, 3]):
        with pytest.raises(ValueError, match="permutation"):
            validated_atom_ids(ids, 3)
