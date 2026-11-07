"""MLMD sampling of several trajectories at once on the ASE engine.

Checks that results match legacy mlmd_sampling, that batching doesn't change them, and that one
failed trajectory leaves the others alone. Some tests skip without torch, HIPPYNN, or RDKit.
"""

import copy
import time

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from alframework.ml_interfaces.ensemble_inference import SelectedStateEvaluator, ensemble_statistics
from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble
from alframework.samplers import batched_mlmd
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.samplers.batched_mlmd import (
    ASEEngine,
    run_mlmd_trajectories,
    sample_trajectories,
    trajectory_rng,
    trajectory_seed,
)
from alframework.samplers.mlmd_sampling import mlmd_sampling
from alframework.tools.cohorts import CohortPool
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import annealing_schedule
from tests.helpers.fakes import PairSpringCalculator, QuadraticStateEnsemble, QuadraticStateMember
from tests.helpers.fakes import spring_ensemble as fake_ensemble
from tests.helpers.molecules import SPRING_WATER

BOX = SPRING_WATER.copy()
BOX.set_cell(np.eye(3) * 12.0)
BOX.set_pbc(True)
SCHEDULE = dict(Tamp=200.0, Tper=0.02, Tsrt=3000.0, Tend=2500.0, Ramp=None, Rper=None, Rend=None)
BASE = dict(dt=0.5, maxt=0.05, Ncheck=5, distcut=0.5, min_time=0.0, Fscut=100.0)
CASES = {
    "trigger": (dict(BASE, Escut=0.012), SCHEDULE),
    "trigger_at_start": (dict(BASE, Escut=-1.0), SCHEDULE),
    "never": (dict(BASE, Escut=100.0), SCHEDULE),
    "distcut": (dict(BASE, Escut=100.0, distcut=1.15), SCHEDULE),
    "min_time": (dict(BASE, Escut=0.005, min_time=0.03), SCHEDULE),
    "density": (dict(BASE, Escut=100.0), dict(SCHEDULE, Ramp=0.002, Rper=0.02, Rend=0.004)),
}


def run_legacy(atoms, config, schedule, seed):
    np.random.seed(seed)
    molecule = MoleculesObject(atoms.copy(), "legacy")
    return mlmd_sampling(molecule, MLMD_calculator(fake_ensemble()), **config, **schedule)


def run_new(atoms_list, config, schedules, rngs, models=None):
    temperatures = [
        annealing_schedule(0.0, config["maxt"], s["Tamp"], s["Tper"], s["Tsrt"], s["Tend"]) for s in schedules
    ]
    engine = ASEEngine([a.copy() for a in atoms_list], models or fake_ensemble(), config["dt"], temperatures, rngs)
    molecules = [MoleculesObject(engine.atoms(i), f"mol-{i}") for i in range(len(atoms_list))]
    return run_mlmd_trajectories(molecules, engine, schedules, config), engine


@pytest.mark.parametrize("case", sorted(CASES))
def test_single_trajectory_matches_legacy_mlmd_sampling(case):
    config, schedule = CASES[case]
    atoms = BOX if case == "density" else SPRING_WATER
    legacy = run_legacy(atoms, config, schedule, seed=11)
    # mlmd_sampling uses the global numpy stream, which RandomState(11) reproduces.
    (new,), _ = run_new([atoms], config, [schedule], [np.random.RandomState(11)])
    old_md, new_md = legacy.get_metadata(), new.get_metadata()

    assert (legacy.get_atoms() is None) == (new.get_atoms() is None)
    for key in (
        "Es",
        "Fs",
        "Fsmax",
        "distmin",
        "simulationtime",
        "Ecrit",
        "Fcrit",
        "Fmcrit",
        "distcrit",
        "Rsrt",
        "Tamp",
        "Tend",
    ):
        assert old_md[key] == new_md[key], key
    np.testing.assert_array_equal(old_md["temps"], new_md["temps"])
    np.testing.assert_array_equal(old_md["denss"], new_md["denss"])
    np.testing.assert_array_equal(old_md["positions"], new_md["positions"])
    np.testing.assert_array_equal(np.asarray(old_md["cell"]), np.asarray(new_md["cell"]))
    expected = {
        "trigger": "eligible",
        "trigger_at_start": "eligible",
        "never": "completed",
        "distcut": "rejected",
        "min_time": "eligible",
        "density": "completed",
    }[case]
    assert new_md["trajectory_outcome"] == expected
    if expected == "eligible":
        np.testing.assert_array_equal(legacy.get_atoms().get_positions(), new.get_atoms().get_positions())
        assert new_md["ranking_score"] > 1.0 or case == "trigger_at_start"
    if case == "density":
        assert len(new_md["denss"]) > 0 and new_md["denss"][0] != new_md["denss"][-1]


def mixed_inputs():
    """Eight trajectories that stop at different points (or never)."""
    molecules = []
    for index in range(8):
        atoms = SPRING_WATER.copy()
        if index == 6:
            atoms.positions[2] = atoms.positions[1] + [0.0, 0.3, 0.0]  # H-H close contact
        if index == 7:
            atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])  # one spring: low spread
        molecules.append(MoleculesObject(atoms, f"mol-0000-{index:010d}"))
    return molecules


GROUP_CONFIG = dict(
    BASE,
    Escut=0.012,
    maxt=0.05,
    random_seed=5,
    srt_temp=[2500.0, 3500.0],
    end_temp=[2500.0, 3500.0],
    amp_temp=[0.0, 200.0],
    per_temp=[0.01, 0.05],
    amp_dens=None,
    per_dens=None,
    end_dens=None,
)


def sample_in_groups(groups):
    results = {}
    for group in groups:
        molecules = [MoleculesObject(m.get_atoms().copy(), m.get_moleculeid()) for m in group]
        for molecule in sample_trajectories(molecules, fake_ensemble(), dict(GROUP_CONFIG)):
            results[molecule.get_moleculeid()] = molecule
    return results


def test_batching_does_not_change_results():
    inputs = mixed_inputs()
    reference = sample_in_groups([[m] for m in inputs])
    outcomes = sorted({r.get_metadata()["trajectory_outcome"] for r in reference.values()})
    assert outcomes == ["completed", "eligible", "rejected"]
    steps = {r.get_metadata()["step"] for r in reference.values()}
    assert len(steps) >= 3, "trajectories should stop at different steps"
    for groups in ([inputs], [inputs[:3], inputs[3:]], [list(reversed(inputs))]):
        grouped = sample_in_groups(groups)
        for name, expected in reference.items():
            got = grouped[name]
            assert (got.get_atoms() is None) == (expected.get_atoms() is None)
            if expected.get_atoms() is not None:
                np.testing.assert_array_equal(got.get_atoms().get_positions(), expected.get_atoms().get_positions())
            for key in ("trajectory_outcome", "step", "Es", "Fs", "temps", "random_seed", "frame_id"):
                assert got.get_metadata()[key] == expected.get_metadata()[key], (name, key)


def test_stopped_trajectory_stays_frozen():
    config = dict(BASE, Escut=0.012)
    atoms_list = [SPRING_WATER.copy() for _ in range(3)]
    atoms_list[0].positions *= 1.1  # starts strained: triggers at the first check
    rngs = [np.random.RandomState(seed) for seed in (1, 2, 3)]
    results, engine = run_new(atoms_list, config, [SCHEDULE] * 3, rngs)
    steps = [r.get_metadata()["step"] for r in results]
    assert results[0].get_metadata()["trajectory_outcome"] == "eligible"
    assert steps[0] == 1 and engine.steps[0] == 1
    assert min(steps[1:]) > steps[0]
    # The stopped member's frame equals its solo run.
    (solo,), _ = run_new([atoms_list[0]], config, [SCHEDULE], [np.random.RandomState(1)])
    np.testing.assert_array_equal(solo.get_atoms().get_positions(), results[0].get_atoms().get_positions())


class FailingEngine(ASEEngine):
    def observe(self, index):
        if index == 1 and self.steps[index] > 1:
            raise RuntimeError("model evaluation failed")
        return super().observe(index)


def test_one_failed_trajectory_spares_the_others():
    inputs = mixed_inputs()[:3]
    clean = sample_in_groups([inputs])
    molecules = [MoleculesObject(m.get_atoms().copy(), m.get_moleculeid()) for m in inputs]
    failed = sample_trajectories(molecules, fake_ensemble(), dict(GROUP_CONFIG), engine_class=FailingEngine)
    assert failed[1].get_metadata()["trajectory_outcome"] == "failed"
    assert failed[1].get_atoms() is None
    assert "model evaluation failed" in failed[1].get_metadata()["sampling_error"]
    for index in (0, 2):
        name = inputs[index].get_moleculeid()
        assert failed[index].get_metadata()["step"] == clean[name].get_metadata()["step"]
        assert failed[index].get_metadata()["Es"] == clean[name].get_metadata()["Es"]


def test_udd_bias_does_not_affect_screening():

    def run(options):
        molecule = MoleculesObject(SPRING_WATER.copy(), "mol-0000-0000000000")
        return sample_trajectories([molecule], fake_ensemble(), dict(GROUP_CONFIG, MLMD_calculator_options=options))[0]

    plain, udd = run({}), run({"udd_bias_weight": 0.05})
    assert udd.get_metadata()["trajectory_outcome"] == "eligible"
    assert udd.get_metadata()["step"] != plain.get_metadata()["step"] or not np.array_equal(
        udd.get_atoms().get_positions(), plain.get_atoms().get_positions()
    )
    # The recorded Es is the unbiased ensemble spread at the returned (biased-trajectory) frame.
    engine = ASEEngine([udd.get_atoms().copy()], fake_ensemble(), 0.5, [300.0], [np.random.RandomState(0)])
    assert udd.get_metadata()["Es"] == pytest.approx(ensemble_statistics(engine.observe(0)["prediction"])[0]["Es"])


def test_each_trajectory_gets_its_own_reproducible_seed():
    assert trajectory_seed("a", 1) == trajectory_seed("a", 1) != trajectory_seed("b", 1)
    assert trajectory_seed("a", None) != trajectory_seed("a", None)
    assert trajectory_rng(trajectory_seed("a", 1)).rand() == trajectory_rng(trajectory_seed("a", 1)).rand()


def test_metadata_files_are_written_per_trajectory(tmp_path):
    config = dict(GROUP_CONFIG, meta_dir=str(tmp_path), trajectory_frequency=1.0, trajectory_interval=1)
    molecules = [MoleculesObject(m.get_atoms().copy(), m.get_moleculeid()) for m in mixed_inputs()[:2]]
    sample_trajectories(molecules, fake_ensemble(), config)
    for molecule in molecules:
        assert (tmp_path / f"metadata-{molecule.get_moleculeid()}.p").exists()
        assert (tmp_path / f"metadata-{molecule.get_moleculeid()}.traj").stat().st_size > 0


def test_trained_hippynn_gives_same_results_batched(trained_ensemble):

    models = HIPNN_ASE_load_ensemble(str(trained_ensemble), device="cpu")
    config = dict(
        GROUP_CONFIG,
        Escut=0.5,
        Fscut=1.0,
        maxt=0.01,
        distcut=0.7,
        srt_temp=[250.0, 350.0],
        end_temp=[250.0, 350.0],
        amp_temp=[0.0, 20.0],
        per_temp=[5.0, 10.0],
        MLMD_calculator_options={"well_params": {"r_start": 3.0, "force": 0.1}},
    )
    inputs = mixed_inputs()[:4]

    def run(groups):
        out = {}
        for group in groups:
            for molecule in sample_trajectories(
                [MoleculesObject(m.get_atoms().copy(), m.get_moleculeid()) for m in group], models, dict(config)
            ):
                out[molecule.get_moleculeid()] = molecule.get_metadata()
        return out

    single, batched = run([[m] for m in inputs]), run([inputs])
    for name in single:
        for key in ("trajectory_outcome", "step", "Es", "Fs", "Fsmax"):
            assert single[name][key] == batched[name][key], (name, key)
        np.testing.assert_array_equal(single[name]["positions"], batched[name]["positions"])


def test_excited_ase_sampling_follows_the_selected_state():
    pytest.importorskip("torch")
    a = SPRING_WATER.copy()
    a.set_array("topology_atom_ids", np.array([0, 2, 1]))
    models = SelectedStateEvaluator(QuadraticStateEnsemble(), 1, [1, 2])
    engine = ASEEngine([a], models, 0.25, [0.0], [np.random.RandomState(1)])
    expected = models.predict([a])
    np.testing.assert_allclose(a.get_forces(), expected["forces"][0].mean(0))
    assert ensemble_statistics(engine.observe(0)["prediction"]) == ensemble_statistics(expected)
    molecule = MoleculesObject(a.copy(), "excited")
    molecule.update_metadata(dict(selected_state=1, source="fixture"))
    config = dict(
        dt=0.25,
        maxt=0.001,
        Ncheck=1,
        Escut=0.001,
        Fscut=0.01,
        distcut=0.5,
        min_time=0.0,
        model_mode="excited_state",
        gap_uncertainty={"pair": [1, 2]},
        acquisition={"selection": {"mode": "cohort_top_k", "cohort_size": 1, "k": 1}},
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
    )
    pool = CohortPool({"cohort_size": 1, "k": 1})
    pool.admit(molecule, 0)
    result = sample_trajectories([molecule], QuadraticStateEnsemble(), config)[0]
    pool.record(result)
    (frame,) = pool.ready()
    assert frame.get_metadata()["selected_state"] == 1 and frame.get_metadata()["source"] == "fixture"
    np.testing.assert_array_equal(frame.atoms.arrays["topology_atom_ids"], [0, 2, 1])
    assert "dE12_stdev" in frame.get_metadata()
    for bad, reason in (
        ({"mode": "batch_cycle", "states": [0, 1]}, "state cycling"),
        ({"mode": "fixed", "state": 9}, "absent"),
    ):
        config["state_selection"] = bad
        with pytest.raises(ValueError, match=reason):
            sample_trajectories([MoleculesObject(SPRING_WATER.copy(), "bad")], QuadraticStateEnsemble(), config)


SELECTED_STATE_BIAS = {
    "well": {"well_params": {"r_start": 0.5, "force": 0.1}},
    "udd": {"udd_bias_weight": 0.3},
    "well_and_udd": {"well_params": {"r_start": 0.5, "force": 0.1}, "udd_bias_weight": 0.3},
}


@pytest.mark.parametrize("options", SELECTED_STATE_BIAS.values(), ids=SELECTED_STATE_BIAS.keys())
def test_ase_selected_state_bias_matches_mlmd_calculator(options):
    pytest.importorskip("torch")
    atoms = SPRING_WATER.copy()
    models = SelectedStateEvaluator(QuadraticStateEnsemble(), 1)
    ASEEngine([atoms], models, 0.25, [0.0], [np.random.RandomState(1)], calculator_options=options)
    reference = MLMD_calculator([QuadraticStateMember(member, 1) for member in range(2)], **copy.deepcopy(options))
    reference.calculate(atoms, properties=["energy", "forces"])
    np.testing.assert_allclose(atoms.get_forces(), reference.results["forces"], atol=1e-10)
    assert atoms.get_potential_energy() == pytest.approx(reference.results["energy"])


def test_ase_selected_state_bias_leaves_observations_unbiased():
    pytest.importorskip("torch")
    atoms = SPRING_WATER.copy()
    models = SelectedStateEvaluator(QuadraticStateEnsemble(), 1)
    options = SELECTED_STATE_BIAS["well_and_udd"]
    engine = ASEEngine([atoms], models, 0.25, [0.0], [np.random.RandomState(1)], calculator_options=options)
    unbiased = models.predict([atoms])
    assert not np.allclose(atoms.get_forces(), unbiased["forces"][0].mean(0))
    observed = engine.observe(0)["prediction"]
    np.testing.assert_allclose(observed["forces"][0], unbiased["forces"][0])
    assert ensemble_statistics(observed) == ensemble_statistics(unbiased)


def test_ase_selected_state_bias_changes_the_trajectory():
    pytest.importorskip("torch")
    config = dict(
        dt=0.25,
        maxt=0.01,
        Ncheck=2,
        Escut=1000.0,
        Fscut=1000.0,
        distcut=0.1,
        min_time=0.0,
        model_mode="excited_state",
        state_selection={"mode": "fixed", "state": 1},
        random_seed=1,
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
    )

    def final_positions(**options):
        molecule = MoleculesObject(SPRING_WATER.copy(), "excited")
        result = sample_trajectories([molecule], QuadraticStateEnsemble(), dict(config, **options))[0]
        return result.get_metadata()["positions"]

    plain = final_positions()
    assert not np.allclose(plain, final_positions(MLMD_calculator_options={"udd_bias_weight": 0.3}))


def test_force_screen_on_selected_state_ignores_well_forces():
    pytest.importorskip("torch")
    config = dict(
        dt=0.1,
        maxt=0.001,
        Ncheck=1,
        Escut=1000.0,
        Fscut=1000.0,
        distcut=0.1,
        min_time=0.0,
        model_mode="excited_state",
        state_selection={"mode": "fixed", "state": 1},
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
        max_force_cutoff=1000.0,
        MLMD_calculator_options={"well_params": {"r_start": 0.1, "force": 1000.0, "mass_weighted": False}},
    )
    atoms = SPRING_WATER.copy()
    states = QuadraticStateEnsemble().predict_states([atoms])
    unbiased = float(np.linalg.norm(states["state_forces"][0].mean(0), axis=-1).max())
    result = sample_trajectories([MoleculesObject(atoms, "excited")], QuadraticStateEnsemble(), config)[0]
    assert result.get_metadata()["screen_audit"]["initial"]["diagnostics"]["Fmeanmax"] == pytest.approx(unbiased)
    assert result.get_metadata()["step"] > 0, "large well forces alone must not reject the first frame"


def test_force_screen_ignores_well_forces():
    a = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]])
    config = dict(
        dt=0.1,
        maxt=0.001,
        Ncheck=1,
        Escut=100.0,
        Fscut=100.0,
        min_time=0.0,
        distcut=0.1,
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
        max_force_cutoff=16.0,
        MLMD_calculator_options={"well_params": {"r_start": 0.1, "force": 1000.0, "mass_weighted": False}},
    )
    out = sample_trajectories(
        [MoleculesObject(a, "well")], [PairSpringCalculator(1.0), PairSpringCalculator(1.0)], config
    )[0]
    assert out.get_metadata()["screen_audit"]["initial"]["diagnostics"]["Fmeanmax"] == pytest.approx(0.0)
    assert out.get_metadata()["step"] > 0  # Large well forces alone did not reject at step zero.


def test_earlier_candidate_survives_later_topology_rejection(tmp_path):
    pytest.importorskip("rdkit")
    atoms = Atoms("OHH", positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239, 0.927, 0]])
    reference = tmp_path / "water.xyz"
    write(reference, atoms)
    atoms.set_array("topology_atom_ids", np.arange(3))

    class BreakBondEngine(ASEEngine):
        def step(self, indices, n):
            failures = super().step(indices, n)
            if self.steps[0] >= 2:
                self._atoms[0].positions[1] *= 2.0
            return failures

    config = dict(
        dt=0.1,
        maxt=0.001,
        Ncheck=1,
        Escut=0.001,
        Fscut=0.01,
        min_time=0.0,
        distcut=0.7,
        max_force_cutoff=16.0,
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
        model_mode="excited_state",
        state_selection={"mode": "fixed", "state": 1},
        topology_check=dict(
            enabled=True,
            reference_conformer_path=str(reference),
            reference_charge=0,
            bond_min_scale=0.85,
            bond_max_scale=1.5,
            connectivity_scale=1.25,
        ),
        acquisition=dict(trajectory_policy="continue", selection=dict(mode="cohort_top_k", cohort_size=1, k=1)),
    )
    molecule = MoleculesObject(atoms, "late-rejection")
    pool = CohortPool(dict(cohort_size=1, k=1))
    pool.admit(molecule, 0)
    result = sample_trajectories([molecule], QuadraticStateEnsemble(), config, engine_class=BreakBondEngine)[0]
    assert result.get_metadata()["trajectory_outcome"] == "rejected"
    assert result.get_metadata()["topology_valid"] is False
    pool.record(result)
    (selected,) = pool.ready()
    assert selected.get_metadata()["topology_valid"] is True
    assert selected.get_metadata()["screen_reject_reason"] is None
    assert selected.get_metadata()["step"] == 1
    audit = result.get_metadata()["screen_audit"]
    assert all(audit[k] is not None for k in ("initial", "last_valid", "rejected"))


# Metadata for progress plots.


class StretchingEngine(ASEEngine):
    """Pulls a chosen trajectory's atoms apart from a chosen step, so the force screen rejects it."""

    stretch_from = {}

    def step(self, indices, n):
        failures = super().step(indices, n)
        for index in indices:
            if self.steps[index] >= self.stretch_from.get(index, np.inf):
                self._atoms[index].positions[1] *= 20.0
        return failures


def screened_config(**settings):
    return dict(BASE, dt=0.5, maxt=0.05, Ncheck=5, Escut=100.0, Fscut=100.0, distcut=0.01, max_force_cutoff=100.0, **settings)


def run_screened(stretch_from, count=1, config=None, engine_class=StretchingEngine, observed=None):
    config = config or screened_config()
    schedule = dict(SCHEDULE, Tamp=0.0, Tsrt=300.0, Tend=300.0)

    class Engine(engine_class):
        pass

    Engine.stretch_from = stretch_from
    engine = Engine([SPRING_WATER.copy() for _ in range(count)], fake_ensemble(), config["dt"], [300.0] * count,
                    [np.random.RandomState(index) for index in range(count)])
    if observed is not None:
        real_observe = engine.observe

        def observe(index):
            observation = real_observe(index)
            observed.append(ensemble_statistics(observation["prediction"])[0])
            return observation

        engine.observe = observe
    molecules = [MoleculesObject(engine.atoms(index), f"mol-0000-{index:010d}") for index in range(count)]
    return run_mlmd_trajectories(molecules, engine, [schedule] * count, config)


def test_rejected_trajectory_keeps_its_last_valid_uncertainty():
    observed = []
    metadata = run_screened({0: 11}, observed=observed)[0].get_metadata()
    assert metadata["trajectory_outcome"] == "rejected"
    # The first observation is the screen before dynamics, and the last is the rejected frame.
    valid = observed[1:-1]
    assert len(valid) >= 2
    assert metadata["Es"] == pytest.approx(valid[-1]["Es"])
    assert metadata["Fs"] == pytest.approx(valid[-1]["Fs"])
    assert metadata["Fsmax"] == pytest.approx(valid[-1]["Fsmax"])
    assert metadata["Es"] != pytest.approx(observed[-1]["Es"]), "the rejected frame is physically suspect"
    assert metadata["Fmeanmax"] > 100.0, "the screen's own values must stay"


def test_trajectory_rejected_on_its_first_check_has_no_uncertainty():
    metadata = run_screened({0: 1})[0].get_metadata()
    assert metadata["trajectory_outcome"] == "rejected"
    assert metadata["Es"] is None and metadata["peak_Es"] is None
    assert metadata["Fmeanmax"] > 100.0


def test_trajectory_rejected_before_dynamics_has_no_uncertainty():
    metadata = run_screened({}, config=screened_config() | {"max_force_cutoff": 1e-9})[0].get_metadata()
    assert metadata["trajectory_outcome"] == "rejected"
    assert metadata["Es"] is None and metadata["peak_Fsmax"] is None


def test_peak_uncertainty_is_the_highest_value_and_not_the_last(monkeypatch):
    values = iter([0.1, 0.9, 0.3, 0.2] + [0.2] * 100)

    def statistics(prediction):
        value = next(values)
        return [dict(Es=value, Fs=value / 2, Fsmax=value * 2, Fsrms=value, energy=0.0)]

    monkeypatch.setattr(batched_mlmd, "ensemble_statistics", statistics)
    metadata = run_screened({}, config=dict(BASE, Escut=100.0, Fscut=100.0))[0].get_metadata()
    assert metadata["Es"] == pytest.approx(0.2)
    assert metadata["peak_Es"] == pytest.approx(0.9)
    assert metadata["peak_Fs"] == pytest.approx(0.45)
    assert metadata["peak_Fsmax"] == pytest.approx(1.8)


def test_peak_uncertainty_ignores_the_rejected_frame():
    observed = []
    metadata = run_screened({0: 11}, observed=observed)[0].get_metadata()
    assert metadata["peak_Es"] == pytest.approx(max(row["Es"] for row in observed[1:-1]))
    assert metadata["peak_Es"] < max(row["Es"] for row in observed)


def test_trajectories_in_one_task_record_their_own_wall_time():
    class SlowEngine(StretchingEngine):
        def step(self, indices, n):
            time.sleep(0.01)
            return super().step(indices, n)

    early, late = run_screened({0: 6}, count=2, engine_class=SlowEngine)
    assert early.get_metadata()["trajectory_outcome"] == "rejected"
    assert late.get_metadata()["trajectory_outcome"] == "completed"
    early_time, late_time = early.get_metadata()["realtime_simulation"], late.get_metadata()["realtime_simulation"]
    # Records are written one after another at the end, so a small gap wouldn't show per-trajectory timing.
    assert early_time > 0 and late_time - early_time > 0.05
