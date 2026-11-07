"""LCM gap seeking: settings, pair choice, and the switch of a replica's forces.

Uses analytic multi-state fakes, so nothing here needs QM or a trained model. Tests that step the
packed engines run once per installed engine, and ASE always runs. Tests that need torch skip without it.
"""

import numpy as np
import pytest

from alframework.tools.gap_seeking import adjacent_pairs, gap_seeking_settings, lcm_energy_forces, select_trigger
from alframework.tools.sampler_coordinator import state_settings

BLOCK = {"enabled": True, "trigger_gap_threshold_eV": 0.05, "sigma": 3.5, "alpha_eV": 0.05}


def excited(**settings):
    return {"model_mode": "excited_state", "gap_seeking": {**BLOCK, **settings}}


def test_lcm_settings_are_off_unless_enabled():
    assert gap_seeking_settings({}) is None
    assert gap_seeking_settings({"gap_seeking": {"enabled": False, "anything": 1}}) is None


def test_lcm_settings_accept_the_fork_block():
    block = {
        **BLOCK,
        "mode": "levine_coe_martinez_switch",
        "switch_policy": "stay_fixed",
        "candidate_pairs": "adjacent",
    }
    assert gap_seeking_settings({"model_mode": "excited_state", "gap_seeking": block}) == {
        "trigger_gap_threshold_eV": 0.05,
        "sigma": 3.5,
        "alpha_eV": 0.05,
    }


@pytest.mark.parametrize(
    "config,message",
    [
        ({"gap_seeking": "yes"}, "dictionary"),
        ({"gap_seeking": {"enabled": 1}}, "Boolean"),
        (excited(extra=1), "Unknown"),
        (excited(mode="other"), "mode"),
        (excited(switch_policy="hysteresis"), "switch_policy"),
        (excited(candidate_pairs="all"), "candidate_pairs"),
        ({"gap_seeking": BLOCK}, "excited_state"),
        ({**excited(), "MLMD_calculator_options": {"udd_bias_weight": 0.1}}, "udd_bias_weight"),
        (excited(sigma=0.0), "sigma"),
        (excited(alpha_eV=float("nan")), "alpha_eV"),
        (excited(trigger_gap_threshold_eV=True), "trigger_gap_threshold_eV"),
        ({"model_mode": "excited_state", "gap_seeking": {"enabled": True, "sigma": 1.0, "alpha_eV": 1.0}}, "requires"),
    ],
    ids=[
        "not_a_dictionary",
        "enabled_not_boolean",
        "unknown_option",
        "unsupported_mode",
        "unsupported_switch_policy",
        "unsupported_pairs",
        "ground_state",
        "with_udd",
        "zero_sigma",
        "nan_alpha",
        "boolean_threshold",
        "missing_threshold",
    ],
)
def test_lcm_settings_reject_invalid_values(config, message):
    with pytest.raises(ValueError, match=message):
        gap_seeking_settings(config)


def test_lcm_settings_are_checked_with_the_other_state_settings():
    assert state_settings(excited())["gap_seeking"]["sigma"] == 3.5
    with pytest.raises(ValueError, match="excited_state"):
        state_settings({"gap_seeking": BLOCK})


@pytest.mark.parametrize(
    "selected,n_states,expected",
    [(0, 4, [(0, 1)]), (1, 4, [(0, 1), (1, 2)]), (3, 4, [(2, 3)])],
    ids=["lowest", "middle", "highest"],
)
def test_lcm_candidates_are_adjacent_pairs_with_the_selected_state(selected, n_states, expected):
    assert adjacent_pairs(selected, n_states) == expected


def test_lcm_needs_an_adjacent_state():
    with pytest.raises(ValueError, match="adjacent"):
        adjacent_pairs(0, 1)


def test_lcm_trigger_takes_the_smallest_absolute_gap():
    assert select_trigger([0.0, 0.03, 0.04], [(0, 1), (1, 2)], 0.05) == {"pair": (1, 2), "gap_eV": pytest.approx(0.01)}


def test_lcm_trigger_measures_an_inverted_gap_by_its_size():
    assert select_trigger([0.04, 0.0, 1.0], [(0, 1), (1, 2)], 0.05) == {"pair": (0, 1), "gap_eV": pytest.approx(-0.04)}


def test_lcm_trigger_breaks_ties_by_lowest_pair():
    assert select_trigger([0.0, 0.03, 0.06], [(1, 2), (0, 1)], 0.05)["pair"] == (0, 1)


def test_lcm_trigger_stays_off_above_the_threshold():
    assert select_trigger([0.0, 0.06], [(0, 1)], 0.05) is None


def test_lcm_trigger_rejects_a_nonfinite_gap():
    with pytest.raises(ValueError, match="nonfinite"):
        select_trigger([0.0, float("inf")], [(0, 1)], 0.05)


def test_lcm_formula_matches_the_fork_on_hand_computed_values():
    # Same inputs as the fork's formula test: E = 2.0 and 2.02, forces 2.0 and 5.0, sigma 3.5, alpha 0.05.
    gap = 0.02
    root = np.sqrt(gap * gap + 1e-12)
    denominator = root + 0.05
    expected_energy = 2.01 + 3.5 * gap * gap / denominator
    derivative = 3.5 * (2.0 * gap * denominator - gap**3 / root) / denominator**2
    energy, forces = lcm_energy_forces(2.0, 2.02, np.full((2, 3), 2.0), np.full((2, 3), 5.0), 3.5, 0.05)
    assert energy == pytest.approx(expected_energy)
    np.testing.assert_allclose(forces, 3.5 + derivative * 3.0)


def test_lcm_force_is_the_gradient_of_its_energy():
    # Two states with analytic energies E = k x^2 + c, so F = -2 k x.
    x0 = np.array([[0.3, -0.2, 0.1], [0.5, 0.4, -0.6]])
    k_lower, k_upper, c_lower, c_upper = 0.7, 1.1, 0.0, 0.02

    def energy_at(x):
        return lcm_energy_forces(
            k_lower * (x * x).sum() + c_lower, k_upper * (x * x).sum() + c_upper, -2 * k_lower * x, -2 * k_upper * x, 3.5, 0.05
        )[0]

    _, forces = lcm_energy_forces(
        k_lower * (x0 * x0).sum() + c_lower, k_upper * (x0 * x0).sum() + c_upper, -2 * k_lower * x0, -2 * k_upper * x0, 3.5, 0.05
    )
    step = 1e-5
    for index in np.ndindex(x0.shape):
        plus, minus = x0.copy(), x0.copy()
        plus[index] += step
        minus[index] -= step
        assert forces[index] == pytest.approx(-(energy_at(plus) - energy_at(minus)) / (2 * step), rel=1e-6, abs=1e-7)


def test_lcm_formula_works_on_tensors():
    torch = pytest.importorskip("torch")
    energy, forces = lcm_energy_forces(
        torch.tensor(2.0), torch.tensor(2.02), torch.full((2, 3), 2.0), torch.full((2, 3), 5.0), 3.5, 0.05
    )
    expected_energy, expected_forces = lcm_energy_forces(2.0, 2.02, np.full((2, 3), 2.0), np.full((2, 3), 5.0), 3.5, 0.05)
    assert float(energy) == pytest.approx(expected_energy, rel=1e-5)
    np.testing.assert_allclose(forces.numpy(), expected_forces, rtol=1e-4)
