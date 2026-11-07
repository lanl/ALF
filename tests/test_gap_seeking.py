"""LCM gap seeking: settings, pair choice, and the switch of a replica's forces.

Uses analytic multi-state fakes, so nothing here needs QM or a trained model. Tests that step the
packed engines run once per installed engine, and ASE always runs. Tests that need torch skip without it.
"""

import copy

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from alframework.ml_interfaces.ensemble_inference import SelectedStateEvaluator, ensemble_statistics
from alframework.samplers import batched_mlmd
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.samplers.batched_mlmd import ASEEngine, sample_trajectories
from alframework.tools.gap_seeking import adjacent_pairs, gap_seeking_settings, lcm_energy_forces, select_trigger
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_coordinator import SamplerCoordinator, state_settings
from tests.helpers.engines import DEVICE, engine_class, require_engine

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


def test_lcm_setting_is_fixed_on_reload():
    config = dict(excited(), Escut=1.0, Fscut=1.0)
    coordinator = SamplerCoordinator(batched_mlmd.mlmd_trajectories_task, "batched", config)
    restored = coordinator.keep_startup_settings(dict(config, gap_seeking={**BLOCK, "sigma": 9.0}))
    assert restored["gap_seeking"]["sigma"] == 3.5


# ---- Engines and the trajectory loop ----


class CrossingStates:
    """Three states where each gap depends on its own coordinates: gap(0,1) = sum x^2, gap(1,2) = sum y^2.

    The members differ by a constant, so the spread stays fixed and the mean gaps are exact.
    """

    n_models = 2
    n_states = 3

    def evaluate_states(self, numbers, positions, cells, pbc, counts):
        import torch

        x = torch.as_tensor(positions, dtype=torch.float64)
        system = torch.repeat_interleave(
            torch.arange(len(counts), device=x.device), torch.as_tensor(counts, device=x.device)
        )
        u = x.new_zeros(len(counts)).index_add_(0, system, x[:, 0] ** 2)
        v = x.new_zeros(len(counts)).index_add_(0, system, x[:, 1] ** 2)
        state_energies = torch.stack([torch.zeros_like(u), u, u + v])
        force_u, force_v = torch.zeros_like(x), torch.zeros_like(x)
        force_u[:, 0], force_v[:, 1] = -2 * x[:, 0], -2 * x[:, 1]
        state_forces = torch.stack([torch.zeros_like(x), force_u, force_u + force_v])
        offsets = x.new_tensor([-0.1, 0.1])[:, None, None]
        return state_energies[None] + offsets, state_forces[None].repeat(2, 1, 1, 1)

    def predict_states(self, atoms_list):
        import torch

        counts = [len(atoms) for atoms in atoms_list]
        positions = torch.tensor(np.concatenate([atoms.positions for atoms in atoms_list]))
        energies, forces = self.evaluate_states(None, positions, None, None, counts)
        starts = np.cumsum([0, *counts])
        return dict(
            state_energies=energies.numpy(),
            state_forces=[forces[:, :, lo:hi].numpy() for lo, hi in zip(starts[:-1], starts[1:])],
        )


class SelectedStateMember(Calculator):
    """One member of CrossingStates on one state, for MLMD_calculator references."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, member, state):
        super().__init__()
        self.member, self.state = member, state

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        states = CrossingStates().predict_states([atoms])
        self.results = {
            "energy": float(states["state_energies"][self.member, self.state, 0]),
            "forces": states["state_forces"][0][self.member, self.state],
        }


NEAR_FIRST_PAIR = [[0.01, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]]
NEAR_SECOND_PAIR = [[1.5, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.5]]
FAR_FROM_BOTH = [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]]


def water_like(positions):
    return Atoms("OH2", positions=positions)


def run_config(engine, **settings):
    config = dict(
        dt=0.1,
        maxt=0.001,
        Ncheck=2,
        distcut=0.1,
        min_time=0.0,
        Escut=1000.0,
        Fscut=1000.0,
        random_seed=4,
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
        model_mode="excited_state",
        state_selection={"mode": "fixed", "state": 1},
        gap_seeking=dict(BLOCK),
        driver={"strategy": "batched", "engine": engine},
        batching={"batch_size": 8},
    )
    config.update(settings)
    if config["gap_seeking"] is None:
        del config["gap_seeking"]
    return config


def sample(engine, positions, **settings):
    molecules = [MoleculesObject(water_like(p), f"mol-{index}") for index, p in enumerate(positions)]
    return sample_trajectories(molecules, CrossingStates(), run_config(engine, **settings), device=DEVICE)


@pytest.fixture(params=["ase", "torchsim", "alchemi"])
def engine_kind(request):
    pytest.importorskip("torch")
    if request.param != "ase":
        require_engine(request.param)
    return request.param


def build_engine(kind, positions, options=None):
    models = SelectedStateEvaluator(CrossingStates(), 1)
    atoms = [water_like(p) for p in positions]
    rngs = [np.random.RandomState(index) for index in range(len(atoms))]
    temperatures = [0.0] * len(atoms)
    if kind == "ase":
        return ASEEngine(atoms, models, 0.1, temperatures, rngs, calculator_options=options), models
    return engine_class(kind)(atoms, models, 0.1, temperatures, rngs, calculator_options=options), models


def propagation_forces(engine, index):
    """The forces the integrator will use next for this trajectory."""
    if isinstance(engine, ASEEngine):
        return engine.atoms(index).get_forces()
    engine.observe(index)
    return engine.state.forces[engine._slices[index]].cpu().numpy()


def expected_lcm(positions, pair):
    states = CrossingStates().predict_states([water_like(positions)])
    energies = states["state_energies"][:, :, 0].mean(0)
    forces = states["state_forces"][0].mean(0)
    lower, upper = pair
    return lcm_energy_forces(float(energies[lower]), float(energies[upper]), forces[lower], forces[upper], 3.5, 0.05)


def test_lcm_forces_match_the_formula_after_a_switch(engine_kind):
    engine, _ = build_engine(engine_kind, [NEAR_FIRST_PAIR])
    propagation_forces(engine, 0)  # Forces already computed on the selected state must not be reused.
    engine.set_lcm(0, (0, 1), 3.5, 0.05)
    np.testing.assert_allclose(propagation_forces(engine, 0), expected_lcm(NEAR_FIRST_PAIR, (0, 1))[1], atol=1e-10)


def test_lcm_well_adds_on_top_of_the_surface(engine_kind):
    options = {"well_params": {"r_start": 0.5, "force": 0.1}}
    positions = FAR_FROM_BOTH
    engine, _ = build_engine(engine_kind, [positions], options)
    engine.set_lcm(0, (0, 1), 3.5, 0.05)
    reference = MLMD_calculator([SelectedStateMember(member, 1) for member in range(2)], **copy.deepcopy(options))
    atoms = water_like(positions)
    reference.calculate(atoms, properties=["energy", "forces"])
    selected_mean = CrossingStates().predict_states([atoms])["state_forces"][0].mean(0)[1]
    expected = reference.results["forces"] - selected_mean + expected_lcm(positions, (0, 1))[1]
    np.testing.assert_allclose(propagation_forces(engine, 0), expected, atol=1e-10)


def test_lcm_leaves_observations_unbiased(engine_kind):
    engine, models = build_engine(engine_kind, [NEAR_FIRST_PAIR])
    engine.set_lcm(0, (0, 1), 3.5, 0.05)
    unbiased = models.predict([water_like(NEAR_FIRST_PAIR)])
    assert not np.allclose(propagation_forces(engine, 0), unbiased["forces"][0].mean(0))
    observed = engine.observe(0)["prediction"]
    np.testing.assert_allclose(observed["forces"][0], unbiased["forces"][0], atol=1e-10)
    assert ensemble_statistics(observed)[0] == pytest.approx(ensemble_statistics(unbiased)[0])


def test_lcm_applies_only_to_the_switched_replica(engine_kind):
    engine, models = build_engine(engine_kind, [FAR_FROM_BOTH, NEAR_FIRST_PAIR])
    engine.set_lcm(1, (0, 1), 3.5, 0.05)
    unswitched = models.predict([water_like(FAR_FROM_BOTH)])["forces"][0].mean(0)
    np.testing.assert_allclose(propagation_forces(engine, 0), unswitched, atol=1e-10)
    np.testing.assert_allclose(propagation_forces(engine, 1), expected_lcm(NEAR_FIRST_PAIR, (0, 1))[1], atol=1e-10)


def test_lcm_survives_another_replica_leaving_the_packed_state(engine_name):
    engine, _ = build_engine(engine_name, [FAR_FROM_BOTH, NEAR_FIRST_PAIR])
    engine.set_lcm(1, (0, 1), 3.5, 0.05)
    engine.finish(0)
    assert not engine.step([1], 2)
    positions = engine.atoms(1).get_positions()
    forces = engine.state.forces.cpu().numpy()
    np.testing.assert_allclose(forces, expected_lcm(positions, (0, 1))[1], atol=1e-10)


def test_lcm_replica_can_leave_the_packed_state(engine_name):
    engine, models = build_engine(engine_name, [NEAR_FIRST_PAIR, FAR_FROM_BOTH])
    engine.set_lcm(0, (0, 1), 3.5, 0.05)
    engine.finish(0)
    assert not engine.step([1], 2)
    unswitched = models.predict([engine.atoms(1)])["forces"][0].mean(0)
    np.testing.assert_allclose(engine.state.forces.cpu().numpy(), unswitched, atol=1e-10)


def test_lcm_switch_records_the_pair_and_trigger(engine_kind):
    near, far = sample(engine_kind, [NEAR_FIRST_PAIR, FAR_FROM_BOTH])
    switched = near.get_metadata()
    assert switched["gap_seeking_current_mode"] == "lcm" and switched["gap_seeking_switched"] is True
    assert switched["gap_seeking_pair"] == [0, 1]
    assert abs(switched["gap_seeking_trigger_gap_eV"]) <= BLOCK["trigger_gap_threshold_eV"]
    assert switched["gap_seeking_trigger_step"] >= 1
    assert "gap_seeking_trigger_time_ps" in switched
    untouched = far.get_metadata()
    assert untouched["gap_seeking_current_mode"] == "selected_state" and untouched["gap_seeking_switched"] is False
    assert "gap_seeking_pair" not in untouched


def test_lcm_each_replica_picks_its_own_pair(engine_kind):
    first, second = sample(engine_kind, [NEAR_FIRST_PAIR, NEAR_SECOND_PAIR])
    assert first.get_metadata()["gap_seeking_pair"] == [0, 1]
    assert second.get_metadata()["gap_seeking_pair"] == [1, 2]


def test_lcm_switch_changes_the_trajectory(engine_kind):
    plain = sample(engine_kind, [NEAR_FIRST_PAIR], gap_seeking=None)[0]
    switched = sample(engine_kind, [NEAR_FIRST_PAIR])[0]
    assert not np.allclose(plain.get_metadata()["positions"], switched.get_metadata()["positions"])


def test_lcm_disabled_block_leaves_the_trajectory_unchanged(engine_kind):
    absent = sample(engine_kind, [NEAR_FIRST_PAIR], gap_seeking=None)[0]
    disabled = sample(engine_kind, [NEAR_FIRST_PAIR], gap_seeking={"enabled": False})[0]
    np.testing.assert_array_equal(absent.get_metadata()["positions"], disabled.get_metadata()["positions"])
    assert "gap_seeking_switched" not in disabled.get_metadata()


def test_lcm_gives_the_same_trajectories_alone_or_batched(engine_kind):
    positions = [NEAR_FIRST_PAIR, NEAR_SECOND_PAIR, FAR_FROM_BOTH]
    batched = {molecule.get_moleculeid(): molecule for molecule in sample(engine_kind, positions)}
    for index, p in enumerate(positions):
        molecule = MoleculesObject(water_like(p), f"mol-{index}")
        alone = sample_trajectories([molecule], CrossingStates(), run_config(engine_kind), device=DEVICE)[0]
        expected = batched[f"mol-{index}"].get_metadata()
        assert alone.get_metadata()["gap_seeking_switched"] == expected["gap_seeking_switched"]
        np.testing.assert_allclose(alone.get_metadata()["positions"], expected["positions"], atol=1e-8)


def test_lcm_switches_a_replica_only_once():
    calls = []

    class CountingEngine(ASEEngine):
        def set_lcm(self, index, pair, sigma, alpha_eV):
            calls.append(index)
            super().set_lcm(index, pair, sigma, alpha_eV)

    pytest.importorskip("torch")
    molecule = MoleculesObject(water_like(NEAR_FIRST_PAIR), "mol-0")
    sample_trajectories([molecule], CrossingStates(), run_config("ase"), engine_class=CountingEngine)
    assert calls == [0]


def test_lcm_never_switches_a_replica_that_was_rejected(engine_kind):
    result = sample(engine_kind, [NEAR_FIRST_PAIR], max_force_cutoff=1e-6)[0]
    assert result.get_metadata()["trajectory_outcome"] == "rejected"
    assert result.get_metadata()["gap_seeking_switched"] is False


def test_lcm_never_switches_a_replica_that_stopped_on_uncertainty(engine_kind):
    result = sample(engine_kind, [NEAR_FIRST_PAIR], Escut=1e-6)[0]
    assert result.get_metadata()["trajectory_outcome"] == "eligible"
    assert result.get_metadata()["gap_seeking_switched"] is False


def test_lcm_forces_do_not_reach_the_force_screen(engine_kind):
    # A huge sigma makes the surface forces far larger than the cutoff, which only sees physical forces.
    block = {**BLOCK, "sigma": 1.0e6}
    result = sample(engine_kind, [NEAR_FIRST_PAIR], gap_seeking=block, max_force_cutoff=100.0, maxt=0.0004)[0]
    metadata = result.get_metadata()
    assert metadata["gap_seeking_switched"] is True
    positions = np.asarray(metadata["screen_audit"]["initial"]["positions"])
    states = CrossingStates().predict_states([water_like(positions)])
    unbiased = float(np.linalg.norm(states["state_forces"][0].mean(0), axis=-1).max())
    assert metadata["screen_audit"]["initial"]["diagnostics"]["Fmeanmax"] == pytest.approx(unbiased)


def test_lcm_candidates_taken_after_a_switch_carry_its_mode(engine_kind):
    acquisition = {"trajectory_policy": "continue", "selection": {"mode": "cohort_top_k", "cohort_size": 1, "k": 50}}
    result = sample(engine_kind, [NEAR_FIRST_PAIR], acquisition=acquisition, Escut=1e-6, maxt=0.002)[0]
    modes = [candidate["metadata"]["gap_seeking_current_mode"] for candidate in result.get_metadata()["acquisition_candidates"]]
    assert "lcm" in modes


def test_lcm_needs_an_adjacent_state_in_the_ensemble():
    pytest.importorskip("torch")

    class OneState(CrossingStates):
        n_states = 1

    config = run_config("ase", state_selection={"mode": "fixed", "state": 0})
    with pytest.raises(ValueError, match="adjacent"):
        sample_trajectories([MoleculesObject(water_like(FAR_FROM_BOTH), "mol-0")], OneState(), config)

