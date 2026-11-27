"""Uncertainty a model reports of its own, besides an ensemble's spread, chooses frames in both sampler families."""
import numpy as np
import pytest

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics, member_uncertainty
from alframework.samplers import acquisition
from alframework.samplers.acquisition import acquisition_settings, assess_frame, require_members, uses_spread
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.tools.molecules import MoleculesObject
from alframework.tools.trajectory_summary import OWN_UNCERTAINTY_PREFIX, trajectory_row
from tests.helpers.fakes import PairSpringCalculator
from tests.helpers.molecules import SPRING_WATER

OWN = {"terms": {"stretch": {"uncertainty": "stretch_energy", "cutoff": 0.05}}, "eligibility": ["stretch"],
       "strategy": "largest"}


class SelfUncertainSpring(PairSpringCalculator):
    """One model that reports its own uncertainty: here its spring energy, which grows away from equilibrium."""

    def calculate(self, atoms=None, properties=("energy",), system_changes=()):
        super().calculate(atoms, properties, system_changes or ["positions"])
        self.results["alf_uncertainty"] = {"stretch_energy": float(self.results["energy"])}


@pytest.mark.parametrize("term, message", [
    ({"uncertainty": "x"}, 'must be {"uncertainty"'),
    ({"uncertainty": "", "cutoff": 1.0}, "must name what the model reports"),
    ({"uncertainty": "x", "cutoff": 0}, "positive number"),
    ({"uncertainty": "x", "cutoff": True}, "positive number"),
])
def test_a_term_of_your_own_must_name_an_uncertainty_and_a_cutoff(term, message):
    with pytest.raises(ValueError, match=message):
        acquisition.custom_terms({"acquisition": {"terms": {"mine": term}}})
    with pytest.raises(ValueError, match="built-in term"):
        acquisition.custom_terms({"acquisition": {"terms": {"energy": {"uncertainty": "x", "cutoff": 1.0}}}})


def test_a_run_of_only_its_own_terms_needs_no_spread_cutoffs():
    settings = acquisition_settings({"acquisition": OWN})
    assert settings["eligibility"] == ["stretch"] and settings["strategy"] == "largest"
    with pytest.raises(ValueError, match="requires Escut"):
        acquisition_settings({})  # the default energy and force terms still need them
    with pytest.raises(ValueError, match="Unknown acquisition eligibility terms \\['mine'\\]"):
        acquisition_settings({"acquisition": dict(OWN, eligibility=["mine"])})


@pytest.mark.parametrize("config, spread", [
    ({"Escut": 1.0, "Fscut": 1.0}, True),
    ({"acquisition": OWN}, False),
    ({"acquisition": dict(OWN, strategy="uncertainty"), "Escut": 1.0, "Fscut": 1.0}, True),
    ({"acquisition": dict(OWN, strategy="summed", weights={"stretch": 1.0})}, False),
    ({"acquisition": dict(OWN, strategy="summed", weights={"stretch": 1.0, "energy": 0.5}), "Escut": 1.0}, True),
], ids=["default", "own_terms", "own_eligibility_spread_ranking", "summed_own", "summed_with_energy"])
def test_uses_spread_says_whether_a_run_needs_two_members(config, spread):
    assert uses_spread(config) is spread
    one = [PairSpringCalculator(3.5)]
    if spread:
        with pytest.raises(ValueError, match="has one member, so no spread"):
            require_members(one, config)
    else:
        require_members(one, config)
    require_members([PairSpringCalculator(3.5)] * 2, config)


def test_a_frame_is_chosen_and_ranked_by_the_models_own_uncertainty():
    config = {"acquisition": OWN, "distcut": 0.5}
    settings = acquisition_settings(config)
    low = dict(Es=0.0, Fs=0.0, Fsmax=0.0, stretch_energy=0.01)
    high = dict(low, stretch_energy=0.2)
    assert assess_frame(low, 1.0, 1.0, config, settings)["outcome"] == "continue"
    decision = assess_frame(high, 1.0, 1.0, config, settings)
    assert decision["outcome"] == "eligible" and decision["ranking_score"] == pytest.approx(4.0)
    assert decision["acquisition_trigger_flags"] == {"stretch": True}
    assert decision["acquisition_cutoffs"] == {"stretch": 0.05}
    assert (decision["Ecrit"], decision["Fcrit"], decision["Fmcrit"]) == (False, False, False)
    assert assess_frame(high, 0.1, 1.0, config, settings)["outcome"] == "rejected", "distcut still applies"


def test_an_evaluators_uncertainty_joins_its_spread():
    prediction = {"energies": np.zeros((1, 2)), "forces": [np.zeros((1, 3, 3)), np.zeros((1, 2, 3))],
                  "uncertainty": [{"dropout": 0.1}, {"dropout": 0.3}]}
    rows = ensemble_statistics(prediction)
    assert [row["dropout"] for row in rows] == [0.1, 0.3] and rows[0]["Es"] == 0.0
    for bad, message in (([{"dropout": 0.1}], "for 1 structures, not 2"),
                         ([{"Es": 1.0}, {}], "name of a built-in signal"),
                         ([{"dropout": float("nan")}, {}], "not a finite number")):
        with pytest.raises(ValueError, match=message):
            ensemble_statistics(dict(prediction, uncertainty=bad))


def test_ase_ensembles_average_what_their_members_report():
    assert member_uncertainty([{"alf_uncertainty": {"a": 1.0}}, {"alf_uncertainty": {"a": 3.0, "b": 2.0}}, {}]) == \
        {"a": 2.0, "b": 2.0}
    evaluator = ASEEnsembleEvaluator([SelfUncertainSpring(3.0), SelfUncertainSpring(5.0)])
    prediction = evaluator.predict([SPRING_WATER.copy()])
    expected = np.mean(prediction["energies"][:, 0])
    assert prediction["uncertainty"][0]["stretch_energy"] == pytest.approx(expected)
    calculator = MLMD_calculator([SelfUncertainSpring(3.0)])
    atoms = SPRING_WATER.copy()
    calculator.calculate(atoms, properties=["energy", "forces", "energy_stdev", "forces_stdev_mean",
                                            "forces_stdev_max"])
    assert calculator.results["energy_stdev"] == 0.0
    assert calculator.results["alf_uncertainty"]["stretch_energy"] > 0


SAMPLER = dict(dt=0.5, maxt=0.5, Ncheck=5, distcut=0.3, min_time=0.0, random_seed=5,
               srt_temp=[3000.0, 3000.0], end_temp=[3000.0, 3000.0], amp_temp=[0.0, 0.0], per_temp=[1.0, 1.0],
               amp_dens=None, per_dens=None, end_dens=None)


def _molecules(count):
    return [MoleculesObject(SPRING_WATER.copy(), f"mol-0000-{index:010d}") for index in range(count)]


@pytest.mark.parametrize("engine", ["ase", "torchsim"])
def test_one_model_chooses_frames_by_its_own_uncertainty_in_the_multi_trajectory_sampler(engine):
    from alframework.samplers.batched_mlmd import sample_trajectories
    from tests.helpers.engines import require_engine

    if engine != "ase":
        require_engine(engine)
    config = dict(SAMPLER, acquisition=OWN, driver={"engine": engine})
    results = sample_trajectories(_molecules(3), [SelfUncertainSpring(3.5)], config)
    outcomes = [molecule.get_metadata()["trajectory_outcome"] for molecule in results]
    assert "eligible" in outcomes
    for molecule in results:
        metadata = molecule.get_metadata()
        assert metadata["Es"] == 0.0, "one member has no spread"
        assert metadata["max_uncertainty"]["stretch_energy"] > 0
        row = trajectory_row(molecule.get_moleculeid(), metadata)
        assert row[OWN_UNCERTAINTY_PREFIX + "stretch_energy"] == pytest.approx(metadata["max_uncertainty"]["stretch_energy"])
        if metadata["trajectory_outcome"] == "eligible":
            assert metadata["acquisition_trigger_flags"] == {"stretch": True}
    with pytest.raises(ValueError, match="has one member, so no spread"):
        sample_trajectories(_molecules(1), [SelfUncertainSpring(3.5)], dict(SAMPLER, Escut=1.0, Fscut=1.0))


def test_one_model_chooses_frames_by_its_own_uncertainty_in_the_one_trajectory_sampler():
    from alframework.samplers.mlmd_sampling import mlmd_sampling

    schedule = dict(Tamp=0.0, Tper=1.0, Tsrt=3000.0, Tend=3000.0, Ramp=None, Rper=None, Rend=None)
    common = {key: SAMPLER[key] for key in ("dt", "maxt", "Ncheck", "distcut", "min_time")}
    [molecule] = _molecules(1)
    result = mlmd_sampling(molecule, MLMD_calculator([SelfUncertainSpring(3.5)]), acquisition=OWN, **common,
                           **schedule)
    metadata = result.get_metadata()
    assert result.get_atoms() is not None, "the frame whose stretch passed its cutoff goes to labeling"
    assert metadata["max_uncertainty"]["stretch_energy"] > 0.05 and metadata["Es"] == 0.0
    # Without the model's own terms the same single model would never trigger: its spread is 0.
    [molecule] = _molecules(1)
    quiet = mlmd_sampling(molecule, MLMD_calculator([SelfUncertainSpring(3.5)]), Escut=1.0, Fscut=1.0, **common,
                          **schedule)
    assert quiet.get_atoms() is None and quiet.get_metadata()["Ecrit"] is False


def test_the_summary_reads_the_largest_own_uncertainty_back(tmp_path):
    from alframework.tools.trajectory_summary import TrajectorySummary, read_trajectory_summary

    summary = TrajectorySummary(tmp_path / "trajectories.jsonl")
    plain, own = _molecules(2)
    plain.update_metadata({"trajectory_outcome": "completed", "Es": 0.1})
    own.update_metadata({"trajectory_outcome": "eligible", "max_uncertainty": {"stretch_energy": 0.4}})
    summary.append([plain, own])
    columns = read_trajectory_summary(tmp_path / "trajectories.jsonl")
    values = list(columns[OWN_UNCERTAINTY_PREFIX + "stretch_energy"])
    assert np.isnan(values[0]) and values[1] == pytest.approx(0.4)


def test_the_coordinator_refuses_one_member_when_the_run_reads_the_spread(tmp_path, monkeypatch):
    from tests.helpers.driver import run_alf_driver
    from tests.test_restarts import LOOP, restart_tasks

    with pytest.raises(ValueError, match=r"The ML configuration \(n_models 1\) has one member"):
        run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, []), **LOOP,
                       sampler_config={"Escut": 1.0, "Fscut": 1.0}, ml_config={"n_models": 1})


def test_a_one_member_run_that_reads_its_own_uncertainty_trains_and_retrains(tmp_path, monkeypatch):
    from tests.helpers.driver import run_alf_driver
    from tests.test_restarts import LOOP, restart_tasks

    run = run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, []), **LOOP,
                         sampler_config={"acquisition": OWN}, ml_config={"n_models": 1})
    assert run.status["current_model_id"] >= 1
