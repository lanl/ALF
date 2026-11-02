"""TorchSim engine (driver.engine 'torchsim'); skipped where torch_sim is not installed."""
import copy

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.verlet import VelocityVerlet

pytest.importorskip("torch_sim")

from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator  # noqa: E402
from alframework.samplers.batched_mlmd import sample_trajectories  # noqa: E402
from alframework.samplers.torchsim_engine import TorchSimEngine  # noqa: E402
from alframework.tools.molecules_class import MoleculesObject  # noqa: E402
from tests.helpers.fakes import spring_ensemble  # noqa: E402
from tests.test_acquisition import TRAINED_ENSEMBLE  # noqa: E402

WATER = Atoms("OH2", positions=[[0, 0, 0], [1.25, 0, 0], [0.6, 1.0392, 0.05]])
CONFIG = dict(dt=0.5, maxt=0.1, Ncheck=5, distcut=0.5, min_time=0.0, Escut=0.012, Fscut=100.0, random_seed=4,
              srt_temp=[2500.0, 3500.0], end_temp=[2500.0, 3500.0], amp_temp=[0.0, 200.0], per_temp=[0.01, 0.05],
              amp_dens=None, per_dens=None, end_dens=None,
              driver={"strategy": "batched", "engine": "torchsim"}, batching={"batch_size": 8})


class Tether(Calculator):
    """Harmonic tether to the origin, for thermostat checks."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, k):
        super().__init__()
        self.k = k

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.get_positions()
        self.results = {"energy": 0.5 * self.k * (x ** 2).sum(), "forces": -self.k * x}


def inputs():
    molecules = []
    for index in range(7):
        atoms = WATER.copy() if index % 2 == 0 else Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]])
        if index == 5:
            atoms.positions[1] = [0.4, 0, 0]          # distcut rejection at the first check
        molecules.append(MoleculesObject(atoms, f"mol-0000-{index:010d}"))
    return molecules


def run(groups, config=CONFIG, models=spring_ensemble):
    results = {}
    for group in groups:
        copies = [MoleculesObject(m.get_atoms().copy(), m.get_moleculeid()) for m in group]
        for molecule in sample_trajectories(copies, models(), copy.deepcopy(config)):
            results[molecule.get_moleculeid()] = molecule
    return results


def test_zero_friction_zero_temperature_is_velocity_verlet():
    reference = WATER.copy()
    reference.calc = MLMD_calculator(spring_ensemble())
    VelocityVerlet(reference, 0.5 * units.fs).run(40)
    engine = TorchSimEngine([WATER.copy()], spring_ensemble(), 0.5, [0.0], [np.random.RandomState(0)], friction=0.0)
    engine.step([0], 40)
    np.testing.assert_allclose(engine.atoms(0).get_positions(), reference.get_positions(), atol=1e-12)
    assert engine.steps == [40]


def test_grouping_and_order_do_not_change_results():
    reference = run([[m] for m in inputs()])
    assert {"eligible", "rejected"} <= {r.get_metadata()["trajectory_outcome"] for r in reference.values()}
    assert len({r.get_metadata()["step"] for r in reference.values()}) >= 3
    for groups in ([inputs()], [inputs()[:3], inputs()[3:]], [list(reversed(inputs()))]):
        grouped = run(groups)
        for name, expected in reference.items():
            got = grouped[name]
            for key in ("trajectory_outcome", "step", "Es", "Fs", "temps", "frame_id"):
                assert got.get_metadata()[key] == expected.get_metadata()[key], (name, key)
            if expected.get_atoms() is not None:
                np.testing.assert_array_equal(got.get_atoms().get_positions(), expected.get_atoms().get_positions())


def test_cohort_candidates_do_not_depend_on_grouping():
    config = dict(CONFIG, acquisition={"trajectory_policy": "continue",
                                       "selection": {"mode": "cohort_top_k", "cohort_size": 7, "k": 3}})
    single, batched = run([[m] for m in inputs()], config), run([inputs()], config)
    for name in single:
        assert [c["frame_id"] for c in single[name].get_metadata()["acquisition_candidates"]] == \
            [c["frame_id"] for c in batched[name].get_metadata()["acquisition_candidates"]]
    assert max(len(r.get_metadata()["acquisition_candidates"]) for r in single.values()) > 1


def test_well_and_udd_forces_match_mlmd_calculator():
    options = {"well_params": {"r_start": 0.5, "force": 0.1}, "udd_bias_weight": 0.3}
    atoms = Atoms("OH2CH", positions=np.random.default_rng(0).normal(size=(5, 3)) * 1.6)
    engine = TorchSimEngine([atoms.copy()], spring_ensemble(), 0.5, [0.0], [np.random.RandomState(0)],
                            calculator_options=options)
    calculator = MLMD_calculator(spring_ensemble(), **copy.deepcopy(options))
    calculator.calculate(atoms, properties=["energy", "forces"])
    np.testing.assert_allclose(engine.state.forces.numpy(), calculator.results["forces"], atol=1e-12)
    assert float(engine.state.energy[0]) == pytest.approx(calculator.results["energy"])


def test_langevin_temperature():
    rng = np.random.default_rng(0)
    systems = [Atoms("H" * 8, positions=rng.normal(size=(8, 3)) * 0.05) for _ in range(32)]
    # Strong friction (inverse ASE time) so the short run holds many independent samples.
    engine = TorchSimEngine(systems, [Tether(4.0), Tether(5.0)], 1.0, [300.0] * 32,
                            [np.random.RandomState(i) for i in range(32)], friction=0.5)
    temperatures = []
    for block in range(120):
        engine.step(list(range(32)), 10)
        if block >= 20:
            temperatures.append(np.mean([engine.atoms(i).get_temperature() for i in range(32)]))
    # As in ASE's Langevin (fixcm), the zero-net-momentum noise is rescaled by sqrt(N/(N-1)),
    # so the kinetic temperature over all 3N degrees of freedom is the target.
    assert np.mean(temperatures) == pytest.approx(300.0, rel=0.03)


def test_density_schedule_scales_cells():
    box = WATER.copy()
    box.set_cell(np.eye(3) * 12.0)
    box.set_pbc(True)
    config = dict(CONFIG, Escut=100.0, amp_dens=[0.002, 0.002], per_dens=[0.02, 0.02], end_dens=[0.004, 0.004])
    molecules = [MoleculesObject(box.copy(), "box-a"), MoleculesObject(box.copy(), "box-b")]
    results = run([molecules], config)
    for name, result in results.items():
        metadata = result.get_metadata()
        assert metadata["trajectory_outcome"] == "completed"
        assert len(metadata["denss"]) > 0 and metadata["denss"][0] != metadata["denss"][-1]
        assert not np.allclose(np.asarray(metadata["cell"]), box.cell.array)
    solo = run([[molecules[0]]], config)["box-a"].get_metadata()
    np.testing.assert_array_equal(np.asarray(solo["cell"]), np.asarray(results["box-a"].get_metadata()["cell"]))


def test_failed_member_does_not_affect_batch_mates():
    class Exploding(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            self.inner.calculate(atoms, properties, system_changes)
            self.results = dict(self.inner.results)
            if len(atoms) == 2:              # only the H2 member, after a few steps
                self.calls += 1
                if self.calls > 3:
                    self.results["forces"] = self.results["forces"] * np.nan

    clean = run([inputs()[:3]])
    failed = run([inputs()[:3]], models=lambda: [Exploding(c) for c in spring_ensemble()])
    assert failed["mol-0000-0000000001"].get_metadata()["trajectory_outcome"] == "failed"
    assert "non-finite" in failed["mol-0000-0000000001"].get_metadata()["sampling_error"]
    for name in ("mol-0000-0000000000", "mol-0000-0000000002"):
        assert failed[name].get_metadata()["step"] == clean[name].get_metadata()["step"]
        assert failed[name].get_metadata()["Es"] == clean[name].get_metadata()["Es"]


@pytest.mark.skipif(not TRAINED_ENSEMBLE.is_dir(), reason="trained Darwin water ensemble not available")
@pytest.mark.parametrize("periodic", [False, True], ids=["molecular", "periodic"])
def test_batched_hippynn_matches_ase_calculators(periodic):
    pytest.importorskip("hippynn")
    import torch
    from alframework.ml_interfaces.hippynn_interface import load_hippynn_batched_ensemble
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble
    from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator

    rng = np.random.default_rng(1)
    systems = []
    for size, waters in ((6.5, 8), (7.5, 12), (5.0, 1)):
        positions = []
        for _ in range(waters):
            center = rng.uniform(0, size, 3)
            positions += [center, center + [0.96, 0, 0], center + [-0.24, 0.93, 0]]
        systems.append(Atoms("OH2" * waters, positions=positions, cell=np.eye(3) * size, pbc=periodic))
    batched = load_hippynn_batched_ensemble(str(TRAINED_ENSEMBLE), device="cpu")
    reference = ASEEnsembleEvaluator(HIPNN_ASE_load_ensemble(str(TRAINED_ENSEMBLE), device="cpu")).predict(systems)
    energies, forces = batched.evaluate(
        torch.as_tensor(np.concatenate([a.numbers for a in systems])),
        torch.as_tensor(np.concatenate([a.positions for a in systems])),
        torch.as_tensor(np.stack([a.cell.array for a in systems])), systems[0].pbc, [len(a) for a in systems])
    np.testing.assert_allclose(energies.numpy(), reference["energies"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forces.numpy(), np.concatenate(reference["forces"], axis=1), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not TRAINED_ENSEMBLE.is_dir(), reason="trained Darwin water ensemble not available")
def test_trained_ensemble_sampling_with_batched_hippynn():
    pytest.importorskip("hippynn")
    from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
    from alframework.ml_interfaces.hippynn_interface import load_hippynn_batched_ensemble
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble

    batched = load_hippynn_batched_ensemble(str(TRAINED_ENSEMBLE), device="cpu")
    config = dict(CONFIG, Escut=0.5, Fscut=1.0, maxt=0.02, distcut=0.7, srt_temp=[250.0, 350.0],
                  end_temp=[250.0, 350.0], amp_temp=[0.0, 20.0], per_temp=[5.0, 10.0],
                  MLMD_calculator_options={"well_params": {"r_start": 3.0, "force": 0.1}},
                  acquisition={"trajectory_policy": "continue",
                               "selection": {"mode": "cohort_top_k", "cohort_size": 3, "k": 50}})
    molecules = [MoleculesObject(Atoms("OH2" * n, positions=np.concatenate(
        [[[3.0 * i, 0, 0], [3.0 * i + 0.96, 0, 0], [3.0 * i - 0.24, 0.93, 0]] for i in range(n)])), f"w-{n}")
        for n in (1, 2, 3)]
    results = sample_trajectories(molecules, batched, config)
    reference = ASEEnsembleEvaluator(HIPNN_ASE_load_ensemble(str(TRAINED_ENSEMBLE), device="cpu"))
    # Every collected frame's screening signals equal the per-structure ASE ensemble's at that geometry.
    candidates = [c for r in results for c in r.get_metadata()["acquisition_candidates"]]
    assert len(candidates) > 3
    for candidate in candidates:
        frame = Atoms(numbers=candidate["numbers"], positions=candidate["positions"])
        expected = ensemble_statistics(reference.predict([frame]))[0]
        for key in ("Es", "Fs", "Fsmax"):
            assert candidate["signals"][key] == pytest.approx(expected[key], rel=1e-4, abs=1e-5), key
