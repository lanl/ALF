"""Numerical checks for the optional energy and force NTK provider."""

import copy

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")
pytest.importorskip("hippynn")

from hippynn import custom_kernels
from hippynn.graphs import GraphModule, inputs, networks, targets

from alframework.ml_interfaces.hippynn.compat import system_energy
from alframework.ml_interfaces.hippynn.ntk import EnergyForceNTK, NTKASECalculator


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
