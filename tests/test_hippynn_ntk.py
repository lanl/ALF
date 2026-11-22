"""Numerical checks for the optional energy and force NTK provider."""

import copy
import os

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")
pytest.importorskip("hippynn")

from hippynn import custom_kernels
from hippynn.graphs import GraphModule, inputs, networks, targets

from alframework.ml_interfaces.hippynn.compat import system_energy
from alframework.ml_interfaces.hippynn.ntk import EnergyForceNTK, NTKASECalculator, NTKPackedEvaluator
from alframework.samplers.engines.packed import PackedMLMDEngine
from alframework.testing import check_evaluator


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    previous = custom_kernels.CUSTOM_KERNELS_ACTIVE
    custom_kernels.set_custom_kernels(False)
    try:
        yield
    finally:
        custom_kernels.set_custom_kernels(previous)


@pytest.fixture(params=["Hipnn", "HipHopnn"])
def graph(request):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        species = inputs.SpeciesNode("species")
        positions = inputs.PositionsNode("positions")
        network = getattr(networks, request.param)("hipnn", (species, positions), module_kwargs={
            "possible_species": [0, 1, 8], "n_features": 8, "n_sensitivities": 4,
            "dist_soft_min": 0.5, "dist_soft_max": 2.5, "dist_hard_max": 3.0,
            "n_interaction_layers": 1, "n_atom_layers": 1,
        })
        energy = system_energy(targets.HEnergyNode("energy", network))
        return GraphModule((species, positions), (energy,)).double().eval()


@pytest.fixture
def atoms():
    return Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.03, 0.07], [-0.24, 0.93, -0.06]])


def test_energy_and_force_features_match_parameter_finite_differences(graph, atoms):
    estimator = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=16, seed=13)
    with torch.enable_grad():
        _, _, energy_row, force_rows = estimator._tangents(atoms)
    trainable = [p for p in graph.parameters() if p.requires_grad]
    original = [p.detach().clone() for p in trainable]

    def predict_perturbed(step):
        perturbed = copy.deepcopy(graph)
        params = (p for p in perturbed.parameters() if p.requires_grad)
        with torch.no_grad():
            for p, reference, projection in zip(params, original, estimator._projection):
                p.copy_(reference + step * projection[:, 0].reshape_as(p))
        positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)
        energy = perturbed(torch.tensor(atoms.numbers)[None], positions[None])[0].sum()
        forces = -torch.autograd.grad(energy, positions)[0]
        return energy.detach(), forces

    plus_e, plus_f = predict_perturbed(1e-5)
    minus_e, minus_f = predict_perturbed(-1e-5)
    torch.testing.assert_close(energy_row[0], (plus_e - minus_e) / 2e-5, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(force_rows[:, :, 0], (plus_f - minus_f) / 2e-5, atol=1e-6, rtol=1e-5)


def test_streamed_ntk_scores_match_direct_solve_and_preserve_atom_order(graph, atoms):
    stretched = atoms.copy()
    stretched.positions[1] *= 1.15
    training = [atoms, stretched]
    energy_scale, force_scale = 0.2, 0.8
    estimator = EnergyForceNTK(graph, iter(training), energy_scale=energy_scale, force_scale=force_scale,
                               sketch_dim=16, seed=13)
    rows = []
    for frame in training:
        _, _, e_row, f_rows = estimator._tangents(frame)
        rows.extend([e_row[None] / energy_scale, f_rows.flatten(0, 1) / force_scale])
    observations = torch.cat(rows)
    precision = observations.T @ observations + estimator.ridge * torch.eye(16, dtype=torch.float64)
    _, _, e_row, f_rows = estimator._tangents(stretched)
    targets = torch.cat([e_row[None], f_rows.flatten(0, 1)])
    variance = estimator.ridge * (targets * torch.linalg.solve(precision, targets.T).T).sum(1)
    result = estimator.predict_energy_forces(stretched)
    torch.testing.assert_close(result["energy_uncertainty_score"], variance[0].sqrt())
    torch.testing.assert_close(result["force_component_uncertainty_score"], variance[1:].sqrt().reshape(3, 3))
    torch.testing.assert_close(result["force_uncertainty_score"], variance[1:].reshape(3, 3).sum(1).sqrt())
    torch.testing.assert_close(result["max_force_uncertainty_score"], result["force_uncertainty_score"].max())
    permutation = [2, 0, 1]
    reordered = estimator.predict_energy_forces(stretched[permutation])
    for key in ("forces", "force_component_uncertainty_score", "force_uncertainty_score"):
        torch.testing.assert_close(reordered[key], result[key][permutation])
    assert estimator.training_structure_count == len(training)


def test_inference_contexts_and_ase_periodicity_guard(graph, atoms):
    estimator = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8)
    result = estimator.predict_energy_forces(atoms)
    assert result["max_force_uncertainty_score"] > 0
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            fitted = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8)
            observed = fitted.predict_energy_forces(atoms)
        for key, expected in result.items():
            torch.testing.assert_close(observed[key], expected)
    calculator = NTKASECalculator(estimator)
    calculator.calculate(atoms)
    np.testing.assert_allclose(calculator.results["alf_atomic_uncertainty"]["ntk_force"],
                               result["force_uncertainty_score"])
    periodic = atoms.copy()
    periodic.set_cell([6.0] * 3)
    periodic.set_pbc(True)
    with pytest.raises(ValueError, match="nonperiodic"):
        calculator.calculate(periodic)
    assert not calculator.results


@pytest.mark.parametrize("device", list(dict.fromkeys(("cpu", os.environ.get("ALF_TEST_NTK_DEVICE", "cpu")))))
def test_float32_matches_float64_reference(graph, atoms, device):
    training = [atoms, atoms.copy()]
    training[1].positions[1] *= 1.15
    query = atoms.copy()
    query.positions[2] *= 1.1
    scales = dict(energy_scale=0.2, force_scale=0.8, sketch_dim=16, seed=13)
    ref = EnergyForceNTK(graph, training, **scales).predict_energy_forces(query)
    # The GPU case exercises the default precision. An explicitly requested GPU
    # must be usable so the qualification command cannot pass by skipping it.
    dtype = torch.float32 if device == "cpu" else None
    estimator = EnergyForceNTK(graph, training, **scales, device=device, dtype=dtype)
    assert estimator.dtype == torch.float32
    for value in (*estimator._parameters, *estimator._projection, estimator._factor):
        assert value.device.type == torch.device(device).type
    with torch.inference_mode():
        observed = estimator.predict_energy_forces(query)
    for key, expected in ref.items():
        assert observed[key].device.type == torch.device(device).type
        torch.testing.assert_close(observed[key].cpu().double(), expected, rtol=3e-3, atol=5e-5)


def test_packed_ntk_scores_are_lazy_and_follow_active_atom_order(graph, atoms, monkeypatch):
    second = Atoms("H2", positions=[[0., 0., 0.], [1.3, 0., 0.]])
    estimator = EnergyForceNTK(graph, [atoms, second], energy_scale=0.2, force_scale=0.8, dtype=torch.float32)
    evaluator = NTKPackedEvaluator(estimator)
    score_calls = []
    observe = evaluator.observe_uncertainty

    def record_observe(numbers, positions, cell, pbc):
        score_calls.append(numbers.clone())
        return observe(numbers, positions, cell, pbc)

    class TestPackedEngine(PackedMLMDEngine):
        def _baoab(self):
            # Change the current geometry without a dependency on an integrator.
            self.state.positions *= 1.001
            return self._evaluate()

    monkeypatch.setattr(evaluator, "observe_uncertainty", record_observe)
    engine = TestPackedEngine([atoms.copy(), second.copy()], evaluator, 0.5, [300., 300.],
                              [np.random.RandomState(2), np.random.RandomState(3)])
    # Refresh ordinary forces and screen a frame without invoking NTK scoring.
    engine.scale_cell(0, 1.02)
    assert not engine.step([0, 1], 2)
    assert "uncertainty" not in engine.observe_physical(0)["prediction"]
    assert not score_calls

    for index in (0, 1):
        snapshot = engine.atoms(index).copy()
        reference = estimator.predict_energy_forces(snapshot)
        prediction = engine.observe(index)["prediction"]
        np.testing.assert_allclose(prediction["energies"][0, 0], reference["energy"], rtol=3e-5, atol=1e-6)
        np.testing.assert_allclose(prediction["forces"][0][0], reference["forces"], rtol=3e-5, atol=1e-6)
        np.testing.assert_allclose(prediction["atomic_uncertainty"][0]["ntk_force"],
                                   reference["force_uncertainty_score"], rtol=3e-5, atol=1e-6)
    assert [len(numbers) for numbers in score_calls] == [3, 2]

    engine.finish(0)
    engine.scale_cell(1, 1.05)
    assert not engine.step([1], 1)
    assert len(score_calls) == 2
    observed = engine.observe(1)["prediction"]["atomic_uncertainty"][0]["ntk_force"]
    remaining = engine.atoms(1)
    np.testing.assert_array_equal(score_calls[-1], remaining.numbers)
    expected = estimator.predict_energy_forces(remaining)["force_uncertainty_score"]
    np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=1e-6)
    assert check_evaluator(evaluator, [remaining])[0]["ntk_force_max"] == pytest.approx(float(expected.max()))
    engine.finish(1)
