from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alframework.samplers import batched_mlmd
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.samplers.batched_mlmd import (
    ASEEngine,
    mlmd_trajectories_task,
    run_mlmd_trajectories,
    sample_trajectories,
    trajectory_rng,
    trajectory_seed,
)
from alframework.samplers.mlmd_sampling import mlmd_sampling
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import annealing_schedule
from tests.helpers.fakes import spring_ensemble as fake_ensemble
from tests.test_acquisition import TRAINED_ENSEMBLE

# Equilateral water at the springs' rest length: zero uncertainty that grows as it heats.
WATER = Atoms("OH2", positions=[[0, 0, 0], [1.2, 0, 0], [0.6, 1.0392, 0]])
BOX = WATER.copy()
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
    temperatures = [annealing_schedule(0.0, config["maxt"], s["Tamp"], s["Tper"], s["Tsrt"], s["Tend"])
                    for s in schedules]
    engine = ASEEngine([a.copy() for a in atoms_list], models or fake_ensemble(), config["dt"], temperatures, rngs)
    molecules = [MoleculesObject(engine.atoms(i), f"mol-{i}") for i in range(len(atoms_list))]
    return run_mlmd_trajectories(molecules, engine, schedules, config), engine


@pytest.mark.parametrize("case", sorted(CASES))
def test_single_trajectory_reproduces_legacy_mlmd_sampling(case):
    config, schedule = CASES[case]
    atoms = BOX if case == "density" else WATER
    legacy = run_legacy(atoms, config, schedule, seed=11)
    # mlmd_sampling uses the global numpy stream, which RandomState(11) reproduces.
    (new,), _ = run_new([atoms], config, [schedule], [np.random.RandomState(11)])
    old_md, new_md = legacy.get_metadata(), new.get_metadata()

    assert (legacy.get_atoms() is None) == (new.get_atoms() is None)
    for key in ("Es", "Fs", "Fsmax", "distmin", "simulationtime", "Ecrit", "Fcrit", "Fmcrit", "distcrit",
                "Rsrt", "Tamp", "Tend"):
        assert old_md[key] == new_md[key], key
    np.testing.assert_array_equal(old_md["temps"], new_md["temps"])
    np.testing.assert_array_equal(old_md["denss"], new_md["denss"])
    np.testing.assert_array_equal(old_md["positions"], new_md["positions"])
    np.testing.assert_array_equal(np.asarray(old_md["cell"]), np.asarray(new_md["cell"]))
    expected = {"trigger": "eligible", "trigger_at_start": "eligible", "never": "completed",
                "distcut": "rejected", "min_time": "eligible", "density": "completed"}[case]
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
        atoms = WATER.copy()
        if index == 6:
            atoms.positions[2] = atoms.positions[1] + [0.0, 0.3, 0.0]   # H-H close contact
        if index == 7:
            atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])   # one spring: low spread
        molecules.append(MoleculesObject(atoms, f"mol-0000-{index:010d}"))
    return molecules


GROUP_CONFIG = dict(BASE, Escut=0.012, maxt=0.05, random_seed=5,
                    srt_temp=[2500.0, 3500.0], end_temp=[2500.0, 3500.0], amp_temp=[0.0, 200.0],
                    per_temp=[0.01, 0.05], amp_dens=None, per_dens=None, end_dens=None)


def sample_in_groups(groups):
    results = {}
    for group in groups:
        molecules = [MoleculesObject(m.get_atoms().copy(), m.get_moleculeid()) for m in group]
        for molecule in sample_trajectories(molecules, fake_ensemble(), dict(GROUP_CONFIG)):
            results[molecule.get_moleculeid()] = molecule
    return results


def test_grouping_and_order_do_not_change_results():
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


def test_stopped_trajectories_are_not_stepped_while_batch_mates_continue():
    config = dict(BASE, Escut=0.012)
    atoms_list = [WATER.copy() for _ in range(3)]
    atoms_list[0].positions *= 1.1       # starts strained: triggers at the first check
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


def test_failed_trajectory_does_not_affect_batch_mates():
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


def test_udd_bias_moves_atoms_but_screening_uses_unbiased_predictions():
    from alframework.ml_interfaces.ensemble_inference import ensemble_statistics

    def run(options):
        molecule = MoleculesObject(WATER.copy(), "mol-0000-0000000000")
        return sample_trajectories([molecule], fake_ensemble(), dict(GROUP_CONFIG, MLMD_calculator_options=options))[0]

    plain, udd = run({}), run({"udd_bias_weight": 0.05})
    assert udd.get_metadata()["trajectory_outcome"] == "eligible"
    assert udd.get_metadata()["step"] != plain.get_metadata()["step"] or \
        not np.array_equal(udd.get_atoms().get_positions(), plain.get_atoms().get_positions())
    # The recorded Es is the unbiased ensemble spread at the returned (biased-trajectory) frame.
    engine = ASEEngine([udd.get_atoms().copy()], fake_ensemble(), 0.5, [300.0], [np.random.RandomState(0)])
    assert udd.get_metadata()["Es"] == pytest.approx(ensemble_statistics(engine.observe(0)["prediction"])[0]["Es"])


def test_seeds_are_reproducible_and_id_specific():
    assert trajectory_seed("a", 1) == trajectory_seed("a", 1) != trajectory_seed("b", 1)
    assert trajectory_seed("a", None) != trajectory_seed("a", None)
    assert trajectory_rng(trajectory_seed("a", 1)).rand() == trajectory_rng(trajectory_seed("a", 1)).rand()


def test_metadata_files_and_task_signature(tmp_path):
    config = dict(GROUP_CONFIG, meta_dir=str(tmp_path), trajectory_frequency=1.0, trajectory_interval=1)
    molecules = [MoleculesObject(m.get_atoms().copy(), m.get_moleculeid()) for m in mixed_inputs()[:2]]
    sample_trajectories(molecules, fake_ensemble(), config)
    for molecule in molecules:
        assert (tmp_path / f"metadata-{molecule.get_moleculeid()}.p").exists()
        assert (tmp_path / f"metadata-{molecule.get_moleculeid()}.traj").stat().st_size > 0
    import inspect
    assert "molecule_objects" in inspect.signature(mlmd_trajectories_task.func).parameters


@pytest.mark.skipif(not TRAINED_ENSEMBLE.is_dir(), reason="trained Darwin water ensemble not available")
def test_trained_ensemble_single_and_batched_agree():
    pytest.importorskip("hippynn")
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble

    models = HIPNN_ASE_load_ensemble(str(TRAINED_ENSEMBLE), device="cpu")
    config = dict(GROUP_CONFIG, Escut=0.5, Fscut=1.0, maxt=0.01, distcut=0.7, srt_temp=[250.0, 350.0],
                  end_temp=[250.0, 350.0], amp_temp=[0.0, 20.0], per_temp=[5.0, 10.0],
                  MLMD_calculator_options={"well_params": {"r_start": 3.0, "force": 0.1}})
    inputs = mixed_inputs()[:4]

    def run(groups):
        out = {}
        for group in groups:
            for molecule in sample_trajectories([MoleculesObject(m.get_atoms().copy(), m.get_moleculeid())
                                                 for m in group], models, dict(config)):
                out[molecule.get_moleculeid()] = molecule.get_metadata()
        return out

    single, batched = run([[m] for m in inputs]), run([inputs])
    for name in single:
        for key in ("trajectory_outcome", "step", "Es", "Fs", "Fsmax"):
            assert single[name][key] == batched[name][key], (name, key)
        np.testing.assert_array_equal(single[name]["positions"], batched[name]["positions"])


def test_device_defaults_and_overrides(monkeypatch):
    import os
    monkeypatch.delenv("PARSL_WORKER_RANK", raising=False)
    assert batched_mlmd.sampling_device({"device": "cpu"}) == "cpu"
    assert batched_mlmd.sampling_device({}) in ("cpu", "cuda:0")
    monkeypatch.setenv("PARSL_WORKER_RANK", "5")
    before = dict(os.environ)
    assert batched_mlmd.sampling_device({"device": "cpu"}, gpus_per_node=4) == 'cpu'
    assert dict(os.environ) == before
