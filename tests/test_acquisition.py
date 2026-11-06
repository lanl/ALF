"""Acquisition: deciding which sampled frames are worth labeling.

Covers ensemble uncertainty, the rules that make a frame eligible, ranking scores, top-k selection,
and gap uncertainty. One test uses a trained HIPPYNN model and skips without HIPPYNN.
"""

import numpy as np
import pytest
from ase import Atoms

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble
from alframework.samplers import acquisition
from alframework.samplers.acquisition import (
    acquisition_settings,
    assess_frame,
    frame_id,
    ranking_score,
    top_k,
)
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from tests.helpers.fakes import FixedCalculator
from tests.helpers.molecules import WATER

METHANE = Atoms(
    "CH4", positions=[[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]
)
CONFIG = {"Escut": 0.5, "Fscut": 0.2, "distcut": 0.7, "min_time": 0.01}


class PositionDependentCalculator(FixedCalculator):
    """Distinct, geometry-dependent members so statistics are nontrivial."""

    def __init__(self, seed):
        super().__init__(0.0, np.zeros((1, 3)))
        self.seed = seed

    def calculate(self, atoms=None, properties=("energy",), system_changes=()):
        super(FixedCalculator, self).calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results["energy"] = float(self.seed * np.sum(positions**2) - len(atoms))
        self.results["forces"] = -2 * self.seed * positions + 0.1 * self.seed


def fake_ensemble(n=4):
    return [PositionDependentCalculator(0.5 + 0.25 * index) for index in range(n)]


def mlmd_reference(calculators, atoms, well_params=None):
    calculator = MLMD_calculator(list(calculators), well_params=well_params)
    calculator.calculate(
        atoms, properties=["energy", "forces", "energy_stdev", "forces_stdev_mean", "forces_stdev_max"]
    )
    return calculator.results


@pytest.mark.parametrize("well_params", [None, {"r_start": 0.5, "force": 0.1}], ids=["no_well", "with_well"])
def test_uncertainty_matches_mlmd_calculator(well_params):
    calculators = fake_ensemble()
    rows = ensemble_statistics(ASEEnsembleEvaluator(calculators).predict([WATER, METHANE]))
    for atoms, row in zip((WATER, METHANE), rows):
        reference = mlmd_reference(fake_ensemble(), atoms, dict(well_params) if well_params else None)
        assert row["Es"] == pytest.approx(reference["energy_stdev"])
        assert row["Fs"] == pytest.approx(reference["forces_stdev_mean"])
        assert row["Fsmax"] == pytest.approx(reference["forces_stdev_max"])
        # The physical mean energy excludes the well potential's contribution.
        members = np.array(reference["energy_contributions"][:4])
        assert row["energy"] == pytest.approx(members.mean())
        assert row["Fsrms"] >= row["Fs"]


def test_mixed_molecule_sizes_evaluate_together():
    prediction = ASEEnsembleEvaluator(fake_ensemble(3)).predict([WATER, METHANE])
    assert prediction["energies"].shape == (3, 2)
    assert [forces.shape for forces in prediction["forces"]] == [(3, 3, 3), (3, 5, 3)]
    with pytest.raises(ValueError, match="at least two"):
        ASEEnsembleEvaluator(fake_ensemble(1))


@pytest.mark.parametrize(
    "Es,Fs,Fsmax,min_distance,time_ps,expected",
    [
        (0.1, 0.1, 0.1, 1.0, 0.02, "continue"),
        (0.6, 0.1, 0.1, 1.0, 0.02, "eligible"),
        (0.1, 0.3, 0.1, 1.0, 0.02, "eligible"),
        (0.1, 0.1, 0.61, 1.0, 0.02, "eligible"),  # The max force cutoff is 3 * Fscut = 0.6.
        (0.1, 0.1, 0.59, 1.0, 0.02, "continue"),
        (0.6, 0.3, 0.7, 1.0, 0.0, "continue"),
        (0.1, 0.1, 0.1, 0.5, 0.02, "rejected"),
        (0.6, 0.1, 0.1, 0.5, 0.02, "rejected"),
        (0.1, 0.1, 0.1, 0.5, 0.0, "continue"),
        (0.5, 0.2, 0.6, 1.0, 0.01, "continue"),
    ],
    ids=[
        "below_every_cutoff",
        "energy_too_uncertain",
        "mean_force_too_uncertain",
        "max_force_too_uncertain",
        "max_force_just_under_cutoff",
        "uncertain_before_min_time",
        "close_contact",
        "close_contact_beats_uncertainty",
        "close_contact_before_min_time",
        "exactly_at_every_cutoff",
    ],
)
def test_frame_eligibility_follows_the_stopping_rules(Es, Fs, Fsmax, min_distance, time_ps, expected):
    signals = {"Es": Es, "Fs": Fs, "Fsmax": Fsmax, "Fsrms": Fs}
    decision = assess_frame(signals, min_distance, time_ps, CONFIG, acquisition_settings(CONFIG))
    assert decision["outcome"] == expected
    assert ("ranking_score" in decision) is (expected == "eligible")


def test_negative_cutoffs_trigger_every_frame():
    config = {"Escut": -1.0, "Fscut": -1.0}
    signals = {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0, "Fsrms": 0.0}
    decision = assess_frame(signals, 1.5, 0.0, config, acquisition_settings(config))
    assert decision["outcome"] == "eligible" and decision["ranking_score"] == np.inf
    with pytest.raises(ValueError, match="requires a positive Escut"):
        acquisition_settings({**config, "acquisition": {"strategy": "summed", "w_energy": 1, "w_force": 1}})


def test_ranking_score_formulas():
    signals = {"Es": 1.0, "Fs": 0.3, "Fsmax": 1.2, "Fsrms": 0.4}
    uncertainty = acquisition_settings(CONFIG)
    assert ranking_score(signals, uncertainty, CONFIG) == pytest.approx(2.0)
    summed = acquisition_settings({**CONFIG, "acquisition": {"strategy": "summed", "w_energy": 0.5, "w_force": 2.0}})
    assert ranking_score(signals, summed, CONFIG) == pytest.approx(0.5 * 2.0 + 2.0 * 2.0)
    # The same score through the explicit term table.
    explicit = acquisition_settings(
        {**CONFIG, "acquisition": {"strategy": "summed", "weights": {"energy": 0.5, "force_rms": 2.0}}}
    )
    assert ranking_score(signals, explicit, CONFIG) == ranking_score(signals, summed, CONFIG)


def test_new_score_terms_join_summed_ranking(monkeypatch):
    monkeypatch.setitem(acquisition.SCORE_TERMS, "custom_gap", ("gap_std", "gap_std_cut", 1.0))
    config = {
        **CONFIG,
        "gap_std_cut": 0.1,
        "acquisition": {"strategy": "summed", "weights": {"energy": 1.0, "custom_gap": 2.0}},
    }
    settings = acquisition_settings(config)
    signals = {"Es": 1.0, "Fs": 0.3, "Fsmax": 1.2, "Fsrms": 0.4, "gap_std": 0.05}
    assert ranking_score(signals, settings, config) == pytest.approx(1.0 / 0.5 + 2.0 * 0.05 / 0.1)
    with pytest.raises(ValueError, match="'gap_std' signal"):
        ranking_score({key: value for key, value in signals.items() if key != "gap_std"}, settings, config)


def test_high_score_alone_is_not_eligible():
    config = {**CONFIG, "acquisition": {"strategy": "summed", "w_energy": 100.0, "w_force": 100.0}}
    signals = {"Es": 0.4, "Fs": 0.1, "Fsmax": 0.1, "Fsrms": 0.19}
    decision = assess_frame(signals, 1.0, 1.0, config, acquisition_settings(config))
    assert decision["outcome"] == "continue"


@pytest.mark.parametrize(
    "block,error",
    [
        ({"strategy": "ranked"}, "strategy"),
        ({"strategy": "summed", "w_energy": 1.0}, "w_force"),
        ({"strategy": "summed", "w_energy": -1, "w_force": 1}, "nonnegative"),
        ({"strategy": "summed", "w_energy": 0, "w_force": 0}, "positive"),
        ({"strategy": "summed", "weights": {"unknown": 1.0}}, "Unknown acquisition score terms"),
        ({"strategy": "summed", "weights": {"energy": 1.0}, "w_energy": 1, "w_force": 1}, "not both"),
        ({"strategy": "uncertainty", "w_energy": 1.0}, "only to strategy 'summed'"),
        ({"trajectory_policy": "continue"}, "requires selection.mode 'cohort_top_k'"),
        ({"selection": {"mode": "cohort_top_k", "k": 2}}, "selection.cohort_size"),
        ({"selection": {"mode": "cohort_top_k", "cohort_size": 4}}, "selection.k"),
        ({"selection": {"mode": "cohort_top_k", "cohort_size": 4, "k": 0}}, "positive integer"),
        ({"selection": {"mode": "first_trigger", "k": 2}}, "require acquisition.selection.mode"),
        (
            {"selection": {"mode": "cohort_top_k", "cohort_size": 4, "k": 2, "top_k": 1}},
            "Unknown acquisition.selection",
        ),
        ({"return_top_k": 4}, "Unknown acquisition settings"),
    ],
    ids=[
        "unknown_strategy",
        "missing_force_weight",
        "negative_weight",
        "all_zero_weights",
        "unknown_score_term",
        "weights_plus_w_energy",
        "weights_on_uncertainty_strategy",
        "continue_without_cohorts",
        "cohort_without_size",
        "cohort_without_k",
        "zero_k",
        "k_without_cohort_mode",
        "unknown_selection_key",
        "unknown_setting",
    ],
)
def test_invalid_settings_are_rejected(block, error):
    with pytest.raises(ValueError, match=error):
        acquisition_settings({**CONFIG, "acquisition": block})


def test_top_k_ignores_input_order():
    candidates = [
        {"trajectory_id": f"mol-{t}", "step": s, "ranking_score": score}
        for t, s, score in [(2, 10, 3.0), (1, 20, 3.0), (1, 10, 3.0), (3, 5, 9.0), (4, 1, 1.0)]
    ]
    expected = [("mol-3", 5), ("mol-1", 10), ("mol-1", 20)]
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = list(rng.permutation(candidates))
        assert [(c["trajectory_id"], c["step"]) for c in top_k(shuffled, 3)] == expected
    assert len(top_k(candidates[:2], 5)) == 2
    assert frame_id("mol-0000-0000000007", 40) == "mol-0000-0000000007:step-000000040"


def test_trained_hippynn_uncertainty_matches_mlmd_calculator(trained_ensemble):

    calculators = HIPNN_ASE_load_ensemble(str(trained_ensemble), device="cpu")
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(3):
        atoms = WATER.copy()
        atoms.rattle(0.05, seed=int(rng.integers(1 << 30)))
        frames.append(atoms)
    rows = ensemble_statistics(ASEEnsembleEvaluator(calculators).predict(frames))
    for atoms, row in zip(frames, rows):
        reference = mlmd_reference(calculators, atoms, {"r_start": 3.0, "force": 0.1})
        # float32 inference: the approved tolerances.
        assert row["Es"] == pytest.approx(float(reference["energy_stdev"]), rel=1e-5, abs=1e-5)
        assert row["Fs"] == pytest.approx(float(reference["forces_stdev_mean"]), rel=1e-5, abs=1e-5)
        assert row["Fsmax"] == pytest.approx(float(reference["forces_stdev_max"]), rel=1e-5, abs=1e-5)


def _gap_statistics():
    """Two members whose energies disagree by 10 in both states, so their gaps agree exactly."""
    energies = np.array([[[10.0], [12.0]], [[20.0], [22.0]]])
    prediction = dict(
        energies=energies[:, 1],
        forces=[np.zeros((2, 2, 3))],
        state_energies=energies,
        state_forces=[np.zeros((2, 2, 2, 3))],
        gap_pair=[0, 1],
    )
    return ensemble_statistics(prediction)[0]


def _gap_config(eligibility):
    return dict(
        Escut=100.0,
        Fscut=100.0,
        distcut=0.7,
        gap_std_cut=0.01,
        model_mode="excited_state",
        gap_uncertainty={"pair": [0, 1]},
        acquisition={"strategy": "summed", "weights": {"energy": 1.0, "gap": 1.0}, "eligibility": eligibility},
    )


def test_gap_spread_ignores_shared_errors():
    stats = _gap_statistics()
    assert stats["Es"] == 5.0 and stats["gap_std"] == 0.0


def test_gap_uncertainty_alone_can_trigger_labeling():
    config = _gap_config(["gap"])
    settings = acquisition_settings(config)
    stats = _gap_statistics()
    assert assess_frame(stats, 1.0, 0.0, config, settings)["outcome"] == "continue"
    stats["gap_std"] = 0.02
    decision = assess_frame(stats, 1.0, 0.0, config, settings)
    assert decision["outcome"] == "eligible" and not decision["Ecrit"]
    assert decision["acquisition_trigger_flags"] == {"gap": True}
    assert decision["acquisition_score_terms"]["gap"] == 2.0


def test_gap_uncertainty_must_be_enabled_to_trigger():
    config = _gap_config(["energy"])
    stats = {**_gap_statistics(), "gap_std": 0.02}
    assert assess_frame(stats, 1.0, 0.0, config, acquisition_settings(config))["outcome"] == "continue"


def test_gap_ranking_requires_the_gap_signal():
    config = _gap_config(["gap"])
    stats = _gap_statistics()
    del stats["gap_std"]
    with pytest.raises(ValueError, match="gap_std"):
        ranking_score(stats, acquisition_settings(config), config)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_gap_cutoffs_are_rejected(cutoff):
    config = dict(
        model_mode="excited_state",
        gap_uncertainty={"pair": [0, 1]},
        gap_std_cut=cutoff,
        Escut=1.0,
        Fscut=1.0,
        acquisition={"eligibility": ["gap"]},
    )
    with pytest.raises(ValueError, match="positive finite"):
        acquisition_settings(config)
