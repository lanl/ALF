from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
from alframework.samplers.acquisition import (
    acquisition_settings,
    assess_frame,
    frame_id,
    ranking_score,
    top_k,
)
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from tests.helpers.fakes import FixedCalculator

WATER = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
METHANE = Atoms("CH4", positions=[[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                                  [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]])
TRAINED_ENSEMBLE = (Path(__file__).resolve().parents[1]
                    / "examples/simple_water_darwin_run_test/models/model-0000")
CONFIG = {"Escut": 0.5, "Fscut": 0.2, "distcut": 0.7, "min_time": 0.01}


class PositionDependentCalculator(FixedCalculator):
    """Distinct, geometry-dependent members so statistics are nontrivial."""

    def __init__(self, seed):
        super().__init__(0.0, np.zeros((1, 3)))
        self.seed = seed

    def calculate(self, atoms=None, properties=("energy",), system_changes=()):
        super(FixedCalculator, self).calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results["energy"] = float(self.seed * np.sum(positions ** 2) - len(atoms))
        self.results["forces"] = -2 * self.seed * positions + 0.1 * self.seed


def fake_ensemble(n=4):
    return [PositionDependentCalculator(0.5 + 0.25 * index) for index in range(n)]


def mlmd_reference(calculators, atoms, well_params=None):
    calculator = MLMD_calculator(list(calculators), well_params=well_params)
    calculator.calculate(atoms, properties=["energy", "forces", "energy_stdev",
                                            "forces_stdev_mean", "forces_stdev_max"])
    return calculator.results


@pytest.mark.parametrize("well_params", [None, {"r_start": 0.5, "force": 0.1}])
def test_statistics_match_mlmd_calculator_and_exclude_well_potential(well_params):
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


def test_evaluator_shapes_support_mixed_sizes():
    prediction = ASEEnsembleEvaluator(fake_ensemble(3)).predict([WATER, METHANE])
    assert prediction["energies"].shape == (3, 2)
    assert [forces.shape for forces in prediction["forces"]] == [(3, 3, 3), (3, 5, 3)]
    with pytest.raises(ValueError):
        ASEEnsembleEvaluator(fake_ensemble(1))


def reference_mlmd_decision(signals, min_distance, time_ps, config):
    """The stopping logic of mlmd_sampling, transcribed for comparison."""
    Ecrit = signals["Es"] > config["Escut"]
    Fcrit = signals["Fs"] > config["Fscut"]
    Fmcrit = signals["Fsmax"] > 3 * config["Fscut"]
    distcrit = min_distance < config.get("distcut", 1.2)
    if time_ps >= config.get("min_time", 0) and (Ecrit or Fcrit or Fmcrit or distcrit):
        return "rejected" if distcrit else "eligible"
    return "continue"


@pytest.mark.parametrize("Es,Fs,Fsmax,min_distance,time_ps", [
    (0.1, 0.1, 0.1, 1.0, 0.02),   # below every cutoff
    (0.6, 0.1, 0.1, 1.0, 0.02),   # energy
    (0.1, 0.3, 0.1, 1.0, 0.02),   # mean force
    (0.1, 0.1, 0.61, 1.0, 0.02),  # max force uses 3 * Fscut
    (0.1, 0.1, 0.59, 1.0, 0.02),
    (0.6, 0.3, 0.7, 1.0, 0.0),    # before min_time
    (0.1, 0.1, 0.1, 0.5, 0.02),   # close contact only
    (0.6, 0.1, 0.1, 0.5, 0.02),   # close contact wins over uncertainty
    (0.1, 0.1, 0.1, 0.5, 0.0),    # close contact before min_time is not checked
    (0.5, 0.2, 0.6, 1.0, 0.01),   # exactly at cutoffs and at min_time
])
def test_eligibility_matches_existing_mlmd_stopping(Es, Fs, Fsmax, min_distance, time_ps):
    signals = {"Es": Es, "Fs": Fs, "Fsmax": Fsmax, "Fsrms": Fs}
    decision = assess_frame(signals, min_distance, time_ps, CONFIG, acquisition_settings(CONFIG))
    assert decision["outcome"] == reference_mlmd_decision(signals, min_distance, time_ps, CONFIG)
    assert ("ranking_score" in decision) is (decision["outcome"] == "eligible")


def test_negative_cutoffs_trigger_every_checked_frame_as_before():
    config = {"Escut": -1.0, "Fscut": -1.0}
    signals = {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0, "Fsrms": 0.0}
    decision = assess_frame(signals, 1.5, 0.0, config, acquisition_settings(config))
    assert decision["outcome"] == "eligible" and decision["ranking_score"] == np.inf
    with pytest.raises(ValueError, match="positive Escut"):
        acquisition_settings({**config, "acquisition": {"strategy": "summed", "w_energy": 1, "w_force": 1}})


def test_uncertainty_and_summed_scores():
    signals = {"Es": 1.0, "Fs": 0.3, "Fsmax": 1.2, "Fsrms": 0.4}
    uncertainty = acquisition_settings(CONFIG)
    assert ranking_score(signals, uncertainty, 0.5, 0.2) == pytest.approx(2.0)
    summed = acquisition_settings({**CONFIG, "acquisition": {"strategy": "summed", "w_energy": 0.5, "w_force": 2.0}})
    assert ranking_score(signals, summed, 0.5, 0.2) == pytest.approx(0.5 * 2.0 + 2.0 * 2.0)


def test_high_summed_score_alone_is_not_eligible():
    config = {**CONFIG, "acquisition": {"strategy": "summed", "w_energy": 100.0, "w_force": 100.0}}
    signals = {"Es": 0.4, "Fs": 0.1, "Fsmax": 0.1, "Fsrms": 0.19}
    decision = assess_frame(signals, 1.0, 1.0, config, acquisition_settings(config))
    assert decision["outcome"] == "continue"


@pytest.mark.parametrize("block,error", [
    ({"strategy": "ranked"}, "strategy"),
    ({"strategy": "summed", "w_energy": 1.0}, "w_force"),
    ({"strategy": "summed", "w_energy": -1, "w_force": 1}, "nonnegative"),
    ({"strategy": "summed", "w_energy": 0, "w_force": 0}, "positive"),
])
def test_invalid_acquisition_settings(block, error):
    with pytest.raises(ValueError, match=error):
        acquisition_settings({**CONFIG, "acquisition": block})


def test_top_k_is_deterministic_and_independent_of_input_order():
    candidates = [{"trajectory_id": f"mol-{t}", "step": s, "ranking_score": score}
                  for t, s, score in [(2, 10, 3.0), (1, 20, 3.0), (1, 10, 3.0), (3, 5, 9.0), (4, 1, 1.0)]]
    expected = [("mol-3", 5), ("mol-1", 10), ("mol-1", 20)]
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = list(rng.permutation(candidates))
        assert [(c["trajectory_id"], c["step"]) for c in top_k(shuffled, 3)] == expected
    assert len(top_k(candidates[:2], 5)) == 2
    assert frame_id("mol-0000-0000000007", 40) == "mol-0000-0000000007:step-000000040"


@pytest.mark.skipif(not TRAINED_ENSEMBLE.is_dir(), reason="trained Darwin water ensemble not available")
def test_trained_hippynn_ensemble_statistics_match_mlmd_calculator():
    pytest.importorskip("hippynn")
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble

    calculators = HIPNN_ASE_load_ensemble(str(TRAINED_ENSEMBLE), device="cpu")
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
