"""Train a molecular HIP-NN from scratch and replay its energy and force NTK scores."""

import json

import numpy as np
import torch
from ase import Atoms
from hippynn import custom_kernels
from hippynn.graphs import GraphModule, inputs, networks, targets

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
from alframework.ml_interfaces.hippynn.compat import system_energy
from alframework.ml_interfaces.hippynn.ntk import EnergyForceNTK, NTKASECalculator
from alframework.samplers.acquisition import acquisition_settings, assess_frame
from alframework.tools.geometry import minimum_distance


def molecule(distance):
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])


def train_graph(energy_scale, force_scale):
    """Fit the energy and its negative gradient to an analytic spring."""
    torch.manual_seed(7)
    species = inputs.SpeciesNode("species")
    positions = inputs.PositionsNode("positions")
    network = networks.Hipnn("hipnn", (species, positions), module_kwargs={
        "possible_species": [0, 1], "n_features": 8, "n_sensitivities": 4,
        "dist_soft_min": 0.35, "dist_soft_max": 2.5, "dist_hard_max": 3.0,
        "n_interaction_layers": 1, "n_atom_layers": 1,
    })
    graph = GraphModule((species, positions), (system_energy(targets.HEnergyNode("energy", network)),)).double()
    distances = np.linspace(0.85, 1.45, 24)
    structures = [molecule(distance) for distance in distances]
    numbers = torch.from_numpy(np.stack([atoms.numbers for atoms in structures]))
    coordinates = torch.from_numpy(np.stack([atoms.positions for atoms in structures])).requires_grad_(True)
    stretch = torch.from_numpy(distances - 1.1)
    # E = k (R - R0)^2 / 2, k = 2 eV / Å².
    reference_energy = stretch.square()
    reference_forces = torch.zeros_like(coordinates)
    reference_forces[:, 0, 0] = 2.0 * stretch
    reference_forces[:, 1, 0] = -2.0 * stretch
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.01)
    parameters = tuple(p for p in graph.parameters() if p.requires_grad)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        energy = graph(numbers, coordinates)[0].squeeze(-1)
        forces = -torch.autograd.grad(energy.sum(), coordinates, create_graph=True)[0]
        energy_loss = ((energy - reference_energy) / energy_scale).square()
        force_loss = ((forces - reference_forces) / force_scale).square().sum((1, 2))
        (energy_loss + force_loss).mean().backward(inputs=parameters)
        optimizer.step()
    return graph.eval(), structures


def direct_scores(result):
    return {
        "ntk_energy": float(result["energy_uncertainty_score"]),
        "ntk_force_max": float(result["max_force_uncertainty_score"]),
    }


def main():
    custom_kernels.set_custom_kernels(False)
    torch.set_num_threads(2)
    energy_scale, force_scale = 0.1, 1.0
    graph, training = train_graph(energy_scale, force_scale)
    estimator = EnergyForceNTK(graph, training, energy_scale=energy_scale, force_scale=force_scale, seed=7)
    calibration = [estimator.predict_energy_forces(molecule(r)) for r in (0.89, 1.01, 1.19, 1.39)]
    scores = ("ntk_energy", "ntk_force_max")
    cutoffs = {name: float(np.median([direct_scores(row)[name] for row in calibration])) for name in scores}
    config = {
        "distcut": 0.3,
        "acquisition": {
            "terms": {name: {"uncertainty": name, "cutoff": cutoffs[name]} for name in scores},
            "eligibility": list(scores),
            "strategy": "largest",
        },
    }
    settings = acquisition_settings(config)
    distances = (0.72, 0.92, 1.04, 1.26, 1.51, 1.66)
    structures = [molecule(r) for r in distances]
    prediction = ASEEnsembleEvaluator([NTKASECalculator(estimator)]).predict(structures)
    report = []
    for i, (distance, atoms, signals, arrays) in enumerate(zip(
            distances, structures, ensemble_statistics(prediction), prediction["atomic_uncertainty"])):
        standalone = estimator.predict_energy_forces(atoms)
        expected_scores = direct_scores(standalone)
        np.testing.assert_allclose(prediction["energies"][0, i], standalone["energy"])
        np.testing.assert_allclose(prediction["forces"][i][0], standalone["forces"])
        np.testing.assert_allclose(arrays["ntk_force"], standalone["force_uncertainty_score"])
        np.testing.assert_allclose([signals[name] for name in scores], [expected_scores[name] for name in scores])
        min_distance = minimum_distance(atoms)
        decision = assess_frame(signals, min_distance, 0.0, config, settings)
        triggers = {name: expected_scores[name] > cutoffs[name] for name in scores}
        expected_outcome = "rejected" if min_distance < config["distcut"] else (
            "eligible" if any(triggers.values()) else "continue"
        )
        if decision["outcome"] != expected_outcome or decision["acquisition_trigger_flags"] != triggers:
            raise AssertionError(f"ALF and standalone decisions differ at distance {distance} Å.")
        report.append({
            "distance_A": distance, "scores": {name: signals[name] for name in scores},
            "ntk_force_by_atom": arrays["ntk_force"].tolist(), "trigger_flags": triggers,
            "decision": decision["outcome"],
        })
    print(json.dumps({"training_structures": len(training), "cutoffs": cutoffs,
                      "matching_decisions": len(report), "replay": report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
