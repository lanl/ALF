import pytest

from alframework.tools.excited_state_tools import (
    derive_gap_property_table,
    derive_state_property_table,
    gap_key_for_pair,
    parse_gap_key,
)


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
