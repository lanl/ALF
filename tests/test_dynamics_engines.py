"""Packed dynamics engines: TorchSim and ALCHEMI.

Each test runs once per engine, shown in its test ID, and skips engines that aren't installed.
Physics is checked against ASE and MLMD_calculator. The whole file needs torch. Set
ALF_TEST_ENGINE_DEVICE to run on a GPU.
"""

import copy

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.verlet import VelocityVerlet

pytest.importorskip("torch")

from ase.io import write

from alframework.ml_interfaces.ensemble_inference import (
    ASEEnsembleEvaluator,
    SelectedStateEvaluator,
    ensemble_statistics,
)
from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble, load_hippynn_batched_ensemble
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator  # noqa: E402
from alframework.samplers.batched_mlmd import sample_trajectories  # noqa: E402
from alframework.tools.molecules_class import MoleculesObject  # noqa: E402
from tests.helpers.engines import DEVICE, Tether, engine_class  # noqa: E402
from tests.helpers.fakes import (
    QuadraticStateEnsemble,
    QuadraticStateMember,
    spring_ensemble,  # noqa: E402
)

WATER = Atoms("OH2", positions=[[0, 0, 0], [1.25, 0, 0], [0.6, 1.0392, 0.05]])
CONFIG = dict(
    dt=0.5,
    maxt=0.1,
    Ncheck=5,
    distcut=0.5,
    min_time=0.0,
    Escut=0.012,
    Fscut=100.0,
    random_seed=4,
    srt_temp=[2500.0, 3500.0],
    end_temp=[2500.0, 3500.0],
    amp_temp=[0.0, 200.0],
    per_temp=[0.01, 0.05],
    amp_dens=None,
    per_dens=None,
    end_dens=None,
    driver={"strategy": "batched", "engine": "torchsim"},
    batching={"batch_size": 8},
)


def inputs():
    molecules = []
    for index in range(7):
        atoms = WATER.copy() if index % 2 == 0 else Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]])
        if index == 5:
            atoms.positions[1] = [0.4, 0, 0]  # distcut rejection at the first check
        molecules.append(MoleculesObject(atoms, f"mol-0000-{index:010d}"))
    return molecules


@pytest.mark.parametrize("overlap", [False, True], ids=["across_periodic_boundary", "overlapping_atoms"])
def test_close_contacts_are_rejected(engine_name, overlap):
    bad = Atoms("H2", positions=[[0.1, 0, 0], [0.1 if overlap else 9.9, 0, 0]], cell=[10] * 3, pbc=True)
    good = Atoms("H2", positions=[[1, 0, 0], [2, 0, 0]], cell=[10] * 3, pbc=True)
    config = dict(
        CONFIG,
        dt=0.5,
        maxt=0.001,
        Ncheck=1,
        Escut=1.0,
        Fscut=1.0,
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
        MLMD_calculator_options={"well_params": None},
    )
    molecules = [MoleculesObject(a, name) for a, name in [(bad, "bad"), (good, "good")]]
    results = sample_trajectories(molecules, [Tether(0.0), Tether(0.0)], config, engine_class=engine_class(engine_name))
    assert [m.get_metadata()["trajectory_outcome"] for m in results] == ["rejected", "completed"]
    assert results[0].get_metadata()["distmin"] == pytest.approx(0.0 if overlap else 0.2)
    assert results[1].get_metadata()["distmin"] == pytest.approx(1.0)


def run(groups, config=CONFIG, models=spring_ensemble, engine_name="torchsim"):
    config = copy.deepcopy(config)
    config["driver"]["engine"] = engine_name
    results = {}
    for group in groups:
        copies = [MoleculesObject(m.get_atoms().copy(), m.get_moleculeid()) for m in group]
        for molecule in sample_trajectories(copies, models(), copy.deepcopy(config), device=DEVICE):
            results[molecule.get_moleculeid()] = molecule
    return results


def test_zero_friction_langevin_matches_velocity_verlet(engine_name):
    reference = WATER.copy()
    reference.calc = MLMD_calculator(spring_ensemble())
    VelocityVerlet(reference, 0.5 * units.fs).run(40)
    engine = engine_class(engine_name)(
        [WATER.copy()], spring_ensemble(), 0.5, [0.0], [np.random.RandomState(0)], friction=0.0
    )
    engine.step([0], 40)
    np.testing.assert_allclose(engine.atoms(0).get_positions(), reference.get_positions(), atol=1e-12)
    assert engine.steps == [40]


def test_batching_does_not_change_results(engine_name):
    reference = run([[m] for m in inputs()], engine_name=engine_name)
    assert {"eligible", "rejected"} <= {r.get_metadata()["trajectory_outcome"] for r in reference.values()}
    assert len({r.get_metadata()["step"] for r in reference.values()}) >= 3
    for groups in ([inputs()], [inputs()[:3], inputs()[3:]], [list(reversed(inputs()))]):
        grouped = run(groups, engine_name=engine_name)
        for name, expected in reference.items():
            got = grouped[name]
            for key in ("trajectory_outcome", "step", "Es", "Fs", "temps", "frame_id"):
                if DEVICE != "cpu" and key in {"Es", "Fs"}:
                    assert got.get_metadata()[key] == pytest.approx(expected.get_metadata()[key], rel=1e-12, abs=1e-12)
                else:
                    assert got.get_metadata()[key] == expected.get_metadata()[key], (name, key)
            if expected.get_atoms() is not None:
                if DEVICE == "cpu":
                    np.testing.assert_array_equal(got.get_atoms().get_positions(), expected.get_atoms().get_positions())
                else:
                    np.testing.assert_allclose(
                        got.get_atoms().get_positions(), expected.get_atoms().get_positions(), rtol=1e-12, atol=1e-12
                    )


@pytest.mark.parametrize("strategy", ["uncertainty", "summed"])
def test_batching_does_not_change_candidates(engine_name, strategy):
    config = dict(
        CONFIG,
        acquisition={
            "trajectory_policy": "continue",
            "strategy": strategy,
            **({"weights": {"energy": 1.0, "force_rms": 1.0}} if strategy == "summed" else {}),
            "selection": {"mode": "cohort_top_k", "cohort_size": 7, "k": 3},
        },
    )
    single, batched = (
        run([[m] for m in inputs()], config, engine_name=engine_name),
        run([inputs()], config, engine_name=engine_name),
    )
    for name in single:
        assert [c["frame_id"] for c in single[name].get_metadata()["acquisition_candidates"]] == [
            c["frame_id"] for c in batched[name].get_metadata()["acquisition_candidates"]
        ]
    assert max(len(r.get_metadata()["acquisition_candidates"]) for r in single.values()) > 1


def test_well_udd_forces_match_mlmd_calculator(engine_name):
    options = {"well_params": {"r_start": 0.5, "force": 0.1}, "udd_bias_weight": 0.3}
    atoms = Atoms("OH2CH", positions=np.random.default_rng(0).normal(size=(5, 3)) * 1.6)
    engine = engine_class(engine_name)(
        [atoms.copy()], spring_ensemble(), 0.5, [0.0], [np.random.RandomState(0)], calculator_options=options
    )
    calculator = MLMD_calculator(spring_ensemble(), **copy.deepcopy(options))
    calculator.calculate(atoms, properties=["energy", "forces"])
    np.testing.assert_allclose(engine.state.forces.cpu().numpy(), calculator.results["forces"], atol=1e-12)
    assert float(engine.state.energy[0]) == pytest.approx(calculator.results["energy"])


def test_selected_state_well_udd_forces_match_mlmd_calculator(engine_name):
    options = {"well_params": {"r_start": 0.5, "force": 0.1}, "udd_bias_weight": 0.3}
    atoms = Atoms("OH2CH", positions=np.random.default_rng(0).normal(size=(5, 3)) * 1.6)
    models = SelectedStateEvaluator(QuadraticStateEnsemble(), 1)
    engine = engine_class(engine_name)(
        [atoms.copy()], models, 0.5, [0.0], [np.random.RandomState(0)], calculator_options=options
    )
    reference = MLMD_calculator([QuadraticStateMember(member, 1) for member in range(2)], **copy.deepcopy(options))
    reference.calculate(atoms, properties=["energy", "forces"])
    np.testing.assert_allclose(engine.state.forces.cpu().numpy(), reference.results["forces"], atol=1e-10)
    assert float(engine.state.energy[0]) == pytest.approx(reference.results["energy"])


def test_selected_state_bias_leaves_observations_unbiased(engine_name):
    options = {"well_params": {"r_start": 0.5, "force": 0.1}, "udd_bias_weight": 0.3}
    atoms = Atoms("OH2CH", positions=np.random.default_rng(0).normal(size=(5, 3)) * 1.6)
    models = SelectedStateEvaluator(QuadraticStateEnsemble(), 1)
    engine = engine_class(engine_name)(
        [atoms.copy()], models, 0.5, [0.0], [np.random.RandomState(0)], calculator_options=options
    )
    expected = models.predict([atoms])
    assert not np.allclose(engine.state.forces.cpu().numpy(), expected["forces"][0].mean(0))
    assert ensemble_statistics(engine.observe(0)["prediction"])[0] == pytest.approx(ensemble_statistics(expected)[0])


def test_langevin_holds_the_target_temperature(engine_name):
    rng = np.random.default_rng(0)
    systems = [Atoms("H" * 8, positions=rng.normal(size=(8, 3)) * 0.05) for _ in range(32)]
    # Strong friction (inverse ASE time) so the short run holds many independent samples.
    engine = engine_class(engine_name)(
        systems,
        [Tether(4.0), Tether(5.0)],
        1.0,
        [300.0] * 32,
        [np.random.RandomState(i) for i in range(32)],
        friction=0.5,
    )
    temperatures = []
    for block in range(120):
        engine.step(list(range(32)), 10)
        if block >= 20:
            temperatures.append(np.mean([engine.atoms(i).get_temperature() for i in range(32)]))
    # As in ASE's Langevin (fixcm), the zero-net-momentum noise is rescaled by sqrt(N/(N-1)),
    # so the kinetic temperature over all 3N degrees of freedom is the target.
    initial = float(np.mean(temperatures))
    assert initial == pytest.approx(300.0, rel=0.03)
    targets = np.array([200.0, 400.0] * 16)
    for index, target in enumerate(targets):
        engine.set_temperature(index, target)
    changed = []
    for block in range(120):
        assert not engine.step(list(range(32)), 10)
        if block >= 20:
            changed.append([engine.atoms(i).get_temperature() for i in range(32)])
    changed = np.asarray(changed)
    measured = {float(target): float(changed[:, targets == target].mean()) for target in (200.0, 400.0)}
    for target, mean in measured.items():
        assert mean == pytest.approx(target, rel=0.03)


def test_density_schedule_changes_the_cell(engine_name):
    box = WATER.copy()
    box.set_cell(np.eye(3) * 12.0)
    box.set_pbc(True)
    config = dict(CONFIG, Escut=100.0, amp_dens=[0.002, 0.002], per_dens=[0.02, 0.02], end_dens=[0.004, 0.004])
    molecules = [MoleculesObject(box.copy(), "box-a"), MoleculesObject(box.copy(), "box-b")]
    results = run([molecules], config, engine_name=engine_name)
    for name, result in results.items():
        metadata = result.get_metadata()
        assert metadata["trajectory_outcome"] == "completed"
        assert len(metadata["denss"]) > 0 and metadata["denss"][0] != metadata["denss"][-1]
        assert not np.allclose(np.asarray(metadata["cell"]), box.cell.array)
    solo = run([[molecules[0]]], config, engine_name=engine_name)["box-a"].get_metadata()
    np.testing.assert_array_equal(np.asarray(solo["cell"]), np.asarray(results["box-a"].get_metadata()["cell"]))


@pytest.mark.parametrize("pooled", [False, True], ids=["stop_policy", "continue_policy"])
def test_one_failed_trajectory_spares_the_others(engine_name, pooled):
    class Exploding(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            self.inner.calculate(atoms, properties, system_changes)
            self.results = dict(self.inner.results)
            if len(atoms) == 2:  # only the H2 member, after a few steps
                self.calls += 1
                if self.calls > 3:
                    self.results["forces"] = self.results["forces"] * np.nan

    config = copy.deepcopy(CONFIG)
    if pooled:
        # Establish an eligible frame before injecting a later failure.
        config["Escut"] = 1e-12
        config["acquisition"] = {
            "trajectory_policy": "continue",
            "selection": {"mode": "cohort_top_k", "cohort_size": 3, "k": 3},
        }
    clean = run([inputs()[:3]], config=config, engine_name=engine_name)
    failed = run(
        [inputs()[:3]], config=config, models=lambda: [Exploding(c) for c in spring_ensemble()], engine_name=engine_name
    )
    assert failed["mol-0000-0000000001"].get_metadata()["trajectory_outcome"] == "failed"
    assert "non-finite" in failed["mol-0000-0000000001"].get_metadata()["sampling_error"]
    if pooled:
        assert failed["mol-0000-0000000001"].get_metadata()["acquisition_candidates"]
    for name in ("mol-0000-0000000000", "mol-0000-0000000002"):
        assert failed[name].get_metadata()["step"] == clean[name].get_metadata()["step"]
        if DEVICE == "cpu":
            assert failed[name].get_metadata()["Es"] == clean[name].get_metadata()["Es"]
        else:
            assert failed[name].get_metadata()["Es"] == pytest.approx(
                clean[name].get_metadata()["Es"], rel=1e-12, abs=1e-12
            )


@pytest.mark.parametrize("periodic", [False, True], ids=["molecular", "periodic"])
def test_batched_hippynn_matches_ase_calculators(periodic, engine_name, trained_ensemble):
    import torch

    rng = np.random.default_rng(1)
    systems = []
    for size, waters in ((6.5, 8), (7.5, 12), (5.0, 1)):
        positions = []
        for _ in range(waters):
            center = rng.uniform(0, size, 3)
            positions += [center, center + [0.96, 0, 0], center + [-0.24, 0.93, 0]]
        systems.append(Atoms("OH2" * waters, positions=positions, cell=np.eye(3) * size, pbc=periodic))
    batched = load_hippynn_batched_ensemble(str(trained_ensemble), device=DEVICE)
    reference = ASEEnsembleEvaluator(HIPNN_ASE_load_ensemble(str(trained_ensemble), device=DEVICE)).predict(systems)
    energies, forces = batched.evaluate(
        torch.as_tensor(np.concatenate([a.numbers for a in systems])),
        torch.as_tensor(np.concatenate([a.positions for a in systems])),
        torch.as_tensor(np.stack([a.cell.array for a in systems])),
        systems[0].pbc,
        [len(a) for a in systems],
    )
    np.testing.assert_allclose(energies.cpu().numpy(), reference["energies"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forces.cpu().numpy(), np.concatenate(reference["forces"], axis=1), rtol=1e-5, atol=1e-5)


def _periodic_waters(seed, boxes):
    rng = np.random.default_rng(seed)
    systems = []
    for cell, waters in boxes:
        positions = []
        for _ in range(waters):
            center = rng.uniform(0, 1, 3) @ np.asarray(cell)
            positions += [center, center + [0.96, 0, 0], center + [-0.24, 0.93, 0]]
        systems.append(Atoms("OH2" * waters, positions=positions, cell=cell, pbc=True))
    return systems


def _batched_and_reference(batched, trained_ensemble, systems):
    import torch

    energies, forces = batched.evaluate(
        torch.as_tensor(np.concatenate([a.numbers for a in systems])),
        torch.as_tensor(np.concatenate([a.positions for a in systems])),
        torch.as_tensor(np.stack([a.cell.array for a in systems])),
        systems[0].pbc,
        [len(a) for a in systems],
    )
    reference = ASEEnsembleEvaluator(HIPNN_ASE_load_ensemble(str(trained_ensemble), device=DEVICE)).predict(systems)
    np.testing.assert_allclose(energies.cpu().numpy(), reference["energies"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forces.cpu().numpy(), np.concatenate(reference["forces"], axis=1), rtol=1e-5, atol=1e-5)


# A skewed cell and a cell smaller than the cutoff, where atoms pair with their own images.
SKEWED_AND_SMALL = ([[7.0, 0, 0], [2.0, 6.5, 0], [1.0, 1.5, 6.0]], 10), (np.eye(3) * 5.0, 1)


def test_batched_hippynn_reuses_pairs_until_an_atom_moves_half_the_skin(neighbor_ensemble):
    """Pairs are found once for all members and kept, Verlet-style, while no atom has moved more than
    half the skin; the results match HIPPYNN's ASE calculators before and after a rebuild."""
    trained_ensemble = neighbor_ensemble
    batched = load_hippynn_batched_ensemble(str(trained_ensemble), device=DEVICE, pair_skin=1.0)
    systems = _periodic_waters(2, SKEWED_AND_SMALL)
    _batched_and_reference(batched, trained_ensemble, systems)
    assert batched.pairs.builds == 1

    rng = np.random.default_rng(3)
    for atoms in systems:  # every atom moves 0.4 A in a random direction, under half the skin
        directions = rng.normal(size=atoms.positions.shape)
        atoms.positions += 0.4 * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    _batched_and_reference(batched, trained_ensemble, systems)
    assert batched.pairs.builds == 1

    systems[0].positions[4] += [0.6, 0.0, 0.0]  # one atom past half the skin
    _batched_and_reference(batched, trained_ensemble, systems)
    assert batched.pairs.builds == 2


def test_batched_hippynn_rebuilds_pairs_for_a_different_batch(neighbor_ensemble):
    trained_ensemble = neighbor_ensemble
    batched = load_hippynn_batched_ensemble(str(trained_ensemble), device=DEVICE)
    systems = _periodic_waters(4, SKEWED_AND_SMALL)
    _batched_and_reference(batched, trained_ensemble, systems)
    _batched_and_reference(batched, trained_ensemble, systems[:1])  # a trajectory left the batch
    assert batched.pairs.builds == 2
    stretched = systems[0].copy()
    stretched.set_cell(stretched.cell.array * 1.01, scale_atoms=False)
    _batched_and_reference(batched, trained_ensemble, [stretched])
    assert batched.pairs.builds == 3


def test_periodic_trained_hippynn_uses_shared_pairs(trained_periodic_ensemble):
    from hippynn.graphs.nodes.inputs import CellNode

    batched = load_hippynn_batched_ensemble(str(trained_periodic_ensemble), device=DEVICE)
    assert all(any(isinstance(n, CellNode) for n in model.input_nodes) for model in batched._members)
    systems = _periodic_waters(7, SKEWED_AND_SMALL)
    _batched_and_reference(batched, trained_periodic_ensemble, systems)
    assert batched.pairs is not None, "periodic-trained checkpoints must activate the external neighbor list"
    assert batched.pairs.builds == 1, "one neighbor build serves all ensemble members"
    _batched_and_reference(batched, trained_periodic_ensemble, systems)
    assert batched.pairs.builds == 1
    for atoms in systems:
        atoms.pbc = False
    with pytest.raises(ValueError, match="requires fully periodic"):
        _batched_and_reference(batched, trained_periodic_ensemble, systems)


@pytest.mark.parametrize("backend", ["ase", "vesin"])
def test_zero_skin_rebuilds_neighbors_even_for_identical_coordinates(trained_periodic_ensemble, backend):
    if backend == "vesin":
        pytest.importorskip("vesin")
    batched = load_hippynn_batched_ensemble(str(trained_periodic_ensemble), device=DEVICE, pair_skin=0,
                                          neighbor_backend=backend)
    systems = _periodic_waters(8, SKEWED_AND_SMALL)
    _batched_and_reference(batched, trained_periodic_ensemble, systems)
    _batched_and_reference(batched, trained_periodic_ensemble, systems)
    assert batched.pairs.builds == 2


def test_batched_hippynn_selects_the_requested_kernels(trained_ensemble):
    import hippynn

    previous = hippynn.custom_kernels.CUSTOM_KERNELS_ACTIVE
    try:
        load_hippynn_batched_ensemble(str(trained_ensemble), device=DEVICE, hippynn_kernel="pytorch")
        assert hippynn.custom_kernels.CUSTOM_KERNELS_ACTIVE == "pytorch"
    finally:
        hippynn.custom_kernels.set_custom_kernels(previous or "pytorch")


def test_batched_hippynn_rejects_a_negative_skin(trained_ensemble):
    batched = load_hippynn_batched_ensemble(str(trained_ensemble), device=DEVICE, pair_skin=-0.1)
    with pytest.raises(ValueError, match="pair_skin"):
        _batched_and_reference(batched, trained_ensemble, _periodic_waters(5, SKEWED_AND_SMALL[:1]))


def test_batched_hippynn_sampling_uncertainty_matches_ase(engine_name, trained_ensemble):

    batched = load_hippynn_batched_ensemble(str(trained_ensemble), device=DEVICE)
    # Keep the actual physical member outputs at each observed geometry. Float32
    # total-energy rounding can be small relative to E but noticeable relative to
    # std(E); distinguish prediction parity from correctness of its reduction.
    recorded = {}

    class RecordingEvaluator:
        n_models = batched.n_models

        def evaluate(self, numbers, positions, cells, pbc, counts):
            energies, forces = batched.evaluate(numbers, positions, cells, pbc, counts)
            numbers = numbers.cpu().numpy()
            positions = positions.detach().cpu().numpy()
            offset = 0
            for system, count in enumerate(counts):
                section = slice(offset, offset + count)
                key = (numbers[section].tobytes(), positions[section].tobytes())
                recorded[key] = {
                    "energies": energies[:, system : system + 1].detach().cpu().numpy().copy(),
                    "forces": [forces[:, section].detach().cpu().numpy().copy()],
                }
                offset += count
            return energies, forces

    config = dict(
        CONFIG,
        Escut=0.5,
        Fscut=1.0,
        maxt=0.02,
        distcut=0.7,
        srt_temp=[250.0, 350.0],
        end_temp=[250.0, 350.0],
        amp_temp=[0.0, 20.0],
        per_temp=[5.0, 10.0],
        MLMD_calculator_options={"well_params": {"r_start": 3.0, "force": 0.1}},
        acquisition={"trajectory_policy": "continue", "selection": {"mode": "cohort_top_k", "cohort_size": 3, "k": 50}},
    )
    molecules = [
        MoleculesObject(
            Atoms(
                "OH2" * n,
                positions=np.concatenate(
                    [[[3.0 * i, 0, 0], [3.0 * i + 0.96, 0, 0], [3.0 * i - 0.24, 0.93, 0]] for i in range(n)]
                ),
            ),
            f"w-{n}",
        )
        for n in (1, 2, 3)
    ]
    config["driver"] = dict(CONFIG["driver"], engine=engine_name)
    results = sample_trajectories(molecules, RecordingEvaluator(), config, device=DEVICE)
    reference = ASEEnsembleEvaluator(HIPNN_ASE_load_ensemble(str(trained_ensemble), device=DEVICE))
    # Compare raw predictions at the approved tolerance, exact signal reductions,
    # and the propagation of measured member roundoff into each uncertainty.
    candidates = [c for r in results for c in r.get_metadata()["acquisition_candidates"]]
    assert len(candidates) > 3
    for candidate in candidates:
        frame = Atoms(numbers=candidate["numbers"], positions=candidate["positions"])
        prediction = recorded[(frame.numbers.tobytes(), frame.positions.tobytes())]
        individual = reference.predict([frame])
        np.testing.assert_allclose(prediction["energies"], individual["energies"], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(prediction["forces"][0], individual["forces"][0], rtol=1e-5, atol=1e-5)
        actual, expected = ensemble_statistics(prediction)[0], ensemble_statistics(individual)[0]
        delta_e = prediction["energies"].astype(float) - individual["energies"]
        delta_f = prediction["forces"][0].astype(float) - individual["forces"][0]
        component_bound = np.std(delta_f, axis=0)
        # std is the norm of centered members / sqrt(n). The reverse triangle
        # inequality bounds its change by std(delta); mean/max/RMS inherit it.
        bounds = {
            "Es": float(np.std(delta_e)),
            "Fs": float(component_bound.mean()),
            "Fsmax": float(component_bound.max()),
            "Fsrms": float(np.sqrt(np.mean(component_bound**2))),
        }
        for key, bound in bounds.items():
            assert candidate["signals"][key] == pytest.approx(actual[key], rel=1e-12, abs=1e-12), key
            assert abs(actual[key] - expected[key]) <= bound + 1e-12, (key, bound)


def test_excited_states_stay_correct_after_a_trajectory_finishes(engine_name):
    model = SelectedStateEvaluator(QuadraticStateEnsemble(), 1, [1, 2])
    atoms = [WATER.copy() for _ in range(3)]
    for index, a in enumerate(atoms):
        a.positions *= 1 + 0.1 * index
    engine = engine_class(engine_name)(
        atoms, model, 0.1, [0.0] * 3, [np.random.RandomState(i) for i in range(3)], friction=0.0
    )
    for index in range(3):
        reference = ensemble_statistics(model.predict([atoms[index]]))[0]
        assert ensemble_statistics(engine.observe(index)["prediction"])[0] == pytest.approx(reference)
    # Remove an interior system, then actually step and evaluate the surviving layout.
    engine.finish(1)
    assert engine.step([0, 2], 1) == {}
    assert engine._rows == [0, 2]
    for index in (0, 2):
        observed = ensemble_statistics(engine.observe(index)["prediction"])[0]
        reference = ensemble_statistics(model.predict([engine.atoms(index)]))[0]
        assert observed == pytest.approx(reference)
    assert engine.steps == [1, 0, 1]


@pytest.mark.parametrize("ncheck", [7, 125])
def test_timing_boundaries_preserve_packed_trajectories(engine_name, monkeypatch, ncheck):
    from alframework.samplers import batched_mlmd

    config = copy.deepcopy(CONFIG)
    config.update(Ncheck=ncheck, maxt=0.125, Escut=-1.0, min_time=0.10, distcut=0.01,
                  srt_temp=[0.0, 0.0], end_temp=[0.0, 0.0], amp_temp=[0.0, 0.0])
    molecules = inputs()[:2]
    timed = run([molecules], config=config, engine_name=engine_name)
    monkeypatch.setattr(batched_mlmd, "STEP_TIMING_INTERVAL", 10**9)
    unsplit = run([molecules], config=config, engine_name=engine_name)
    for name, result in timed.items():
        actual, expected = result.get_metadata(), unsplit[name].get_metadata()
        assert actual["delta_realtime_step_100"] > 0
        assert actual["delta_realtime_step_200"] > 0
        for key in ("step", "frame_id", "trajectory_outcome", "Es", "Fs"):
            assert actual[key] == expected[key], key
        np.testing.assert_array_equal(actual["positions"], expected["positions"])
        np.testing.assert_array_equal(actual["temps"], expected["temps"])


def test_topology_or_force_rejection_stops_before_first_step(engine_name, tmp_path):
    reference = Atoms("OHH", positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239, 0.927, 0]])
    path = tmp_path / "water.xyz"
    write(path, reference)
    reference.set_array("topology_atom_ids", np.arange(3))
    config = dict(
        CONFIG,
        driver=dict(engine=engine_name, strategy="single"),
        model_mode="excited_state",
        state_selection=dict(mode="fixed", state=1),
        min_time=0.0,
        maxt=0.001,
        Ncheck=1,
        max_force_cutoff=16.0,
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        friction_per_fs=0.1,
        topology_check=dict(
            enabled=True,
            reference_conformer_path=str(path),
            reference_charge=0,
            bond_min_scale=0.85,
            bond_max_scale=1.5,
            connectivity_scale=1.25,
        ),
    )
    config.pop("batching")

    def molecules():
        items = [MoleculesObject(reference.copy(), str(i)) for i in range(4)]
        items[1].atoms.positions[1] *= 2.0
        return items

    out = sample_trajectories(molecules(), QuadraticStateEnsemble(), config, engine_class=engine_class(engine_name))
    assert [m.get_moleculeid() for m in out] == ["0", "1", "2", "3"]
    assert out[1].get_metadata()["trajectory_outcome"] == "rejected" and out[1].get_metadata()["step"] == 0
    assert out[1].get_metadata()["screen_reject_reason"] == "topology_bond_window"
    assert all(out[i].get_metadata()["step"] > 0 for i in (0, 2, 3))
    assert all(m.get_metadata()["sampling_initial_active_width"] == 3 for m in out)
    config["max_force_cutoff"] = 1.0
    out = sample_trajectories(molecules(), QuadraticStateEnsemble(), config, engine_class=engine_class(engine_name))
    assert all(m.atoms is None and m.get_metadata()["step"] == 0 for m in out)
    assert out[0].get_metadata()["screen_reject_reason"] == "physical_force"


def _periodic_boxes():
    rng = np.random.default_rng(11)
    skewed = Atoms("H40", scaled_positions=rng.uniform(0, 1, (40, 3)),
                   cell=[[7.0, 0, 0], [3.5, 6.0, 0], [2.0, 2.5, 5.5]], pbc=True)
    tiny = Atoms("H2", scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.7, 0.4]], cell=[2.0, 2.2, 1.9], pbc=True)
    dilute = Atoms("H3", positions=rng.uniform(0, 30, (3, 3)), cell=[30] * 3, pbc=True)  # no pair within 3 A
    return [skewed, tiny, dilute]


def _screened_engine(engine_name):
    systems = _periodic_boxes()
    engine = engine_class(engine_name)([a.copy() for a in systems], [Tether(0.1), Tether(0.1)], 0.5,
                                       [300.0] * len(systems), [np.random.RandomState(i) for i in range(len(systems))])
    engine.batched_distance_screen = True  # the default off the CPU
    return engine


def test_batched_distance_screen_matches_minimum_distance_once_per_step(engine_name, monkeypatch):
    """Every system's distance comes from one batched search per step; values match the per-system screen
    (a skewed cell, a cell smaller than the search radius, and a dilute box that falls back to it)."""
    from alframework.samplers import packed_engine
    from alframework.samplers.geometry import minimum_distance

    searches = []
    batched = packed_engine.batched_minimum_distances
    monkeypatch.setattr(packed_engine, "batched_minimum_distances",
                        lambda *args: searches.append(1) or batched(*args))
    engine = _screened_engine(engine_name)
    for step in range(3):
        for index in range(3):
            expected = minimum_distance(engine.atoms(index))
            assert engine.observe(index)["min_distance"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert len(searches) == step + 1
        engine.step([0, 1, 2], 1)


def test_batched_distance_screen_falls_back_without_a_batched_search(engine_name, monkeypatch):
    from alframework.samplers import geometry
    from alframework.samplers.geometry import minimum_distance

    monkeypatch.setattr(geometry, "_batched_neighbor_search", lambda: None)
    engine = _screened_engine(engine_name)
    for index in range(3):
        assert engine.observe(index)["min_distance"] == pytest.approx(minimum_distance(engine.atoms(index)),
                                                                      rel=1e-12, abs=1e-12)
