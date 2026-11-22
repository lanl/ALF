"""Train a molecular HIP-NN from scratch and replay its energy and force NTK scores."""

import argparse
import json

import numpy as np
import torch
from ase import Atoms
from hippynn import custom_kernels
from hippynn.graphs import GraphModule, inputs, networks, targets

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
from alframework.ml_interfaces.hippynn.compat import system_energy
from alframework.ml_interfaces.hippynn.ntk import EnergyForceNTK, NTKASECalculator, NTKPackedEvaluator
from alframework.samplers.acquisition import acquisition_settings, assess_frame
from alframework.samplers.batched_mlmd import sample_trajectories
from alframework.tools.geometry import minimum_distance
from alframework.tools.molecules import MoleculesObject


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


def trajectory_smoke(estimator, structures, engine):
    """Run a short trajectory and verify each selected frame's atom scores."""
    score = "ntk_force_max"
    cutoff = 0.1 * min(direct_scores(estimator.predict_energy_forces(atoms))[score] for atoms in structures)
    config = {
        "driver": {"engine": engine}, "dt": 0.25, "maxt": 0.002, "Ncheck": 2,
        "min_time": 0.0005, "distcut": 0.3, "random_seed": 7, "metadata_files": "none",
        "srt_temp": [300.0, 300.0], "end_temp": [300.0, 300.0], "amp_temp": [0.0, 0.0],
        "per_temp": [1.0, 1.0], "amp_dens": None, "per_dens": None, "end_dens": None,
        "acquisition": {
            "terms": {score: {"uncertainty": score, "cutoff": cutoff}},
            "eligibility": [score], "strategy": "largest",
        },
    }
    molecules = [MoleculesObject(atoms.copy(), f"mol-0000-{i:010d}") for i, atoms in enumerate(structures)]
    models = [NTKASECalculator(estimator)] if engine == "ase" else NTKPackedEvaluator(estimator)
    selected = sample_trajectories(molecules, models, config, device=str(estimator.device))
    for molecule in selected:
        metadata = molecule.get_metadata()
        if metadata["trajectory_outcome"] != "eligible":
            raise AssertionError(f"NTK trajectory smoke test failed: {metadata.get('sampling_error', metadata)}")
        result = estimator.predict_energy_forces(molecule.get_atoms())
        np.testing.assert_allclose(metadata["atomic_uncertainty"]["ntk_force"],
                                   result["force_uncertainty_score"].cpu(), rtol=3e-3, atol=5e-5)
    return {"engine": engine, "selected": len(selected), "steps": [m.get_metadata()["step"] for m in selected]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, mps, or a PyTorch cuda device (CUDA or ROCm).")
    parser.add_argument("--dtype", choices=("float32", "float64"), help="Override the scoring precision.")
    parser.add_argument("--engine", default="ase", choices=("ase", "torchsim", "alchemi"),
                        help="Engine to use for the short trajectory check.")
    args = parser.parse_args()
    if torch.device(args.device).type == "mps" and args.engine != "ase":
        parser.error("Use ASE for MPS. ALF's packed integration state is float64.")
    if args.engine == "alchemi" and torch.device(args.device).type == "cuda" and torch.version.hip is not None:
        parser.error("Use TorchSim for ROCm. ALCHEMI's GPU integrator uses NVIDIA CUDA.")
    custom_kernels.set_custom_kernels(False)
    torch.set_num_threads(2)
    energy_scale, force_scale = 0.1, 1.0
    graph, training = train_graph(energy_scale, force_scale)
    options = dict(energy_scale=energy_scale, force_scale=force_scale, seed=7)
    estimator = EnergyForceNTK(graph, training, **options, device=args.device,
                               dtype=getattr(torch, args.dtype) if args.dtype else None)
    reference = (estimator if estimator.device.type == "cpu" and estimator.dtype == torch.float64
                 else EnergyForceNTK(graph, training, **options))
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
        np.testing.assert_allclose(prediction["energies"][0, i], standalone["energy"].cpu())
        np.testing.assert_allclose(prediction["forces"][i][0], standalone["forces"].cpu())
        np.testing.assert_allclose(arrays["ntk_force"], standalone["force_uncertainty_score"].cpu())
        np.testing.assert_allclose([signals[name] for name in scores], [expected_scores[name] for name in scores])
        min_distance = minimum_distance(atoms)
        decision = assess_frame(signals, min_distance, 0.0, config, settings)
        triggers = {name: expected_scores[name] > cutoffs[name] for name in scores}
        expected_outcome = "rejected" if min_distance < config["distcut"] else (
            "eligible" if any(triggers.values()) else "continue"
        )
        if decision["outcome"] != expected_outcome or decision["acquisition_trigger_flags"] != triggers:
            raise AssertionError(f"ALF and standalone decisions differ at distance {distance} Å.")
        cpu = reference.predict_energy_forces(atoms)
        for key, value in standalone.items():
            torch.testing.assert_close(value.cpu().double(), cpu[key], rtol=3e-3, atol=5e-5)
        cpu_scores = direct_scores(cpu)
        for name in scores:
            # Treat a decision within the score comparison tolerance as a numerical tie.
            if abs(cpu_scores[name] - cutoffs[name]) > 5e-5 + 3e-3 * abs(cutoffs[name]):
                if triggers[name] != (cpu_scores[name] > cutoffs[name]):
                    raise AssertionError(f"{name} decision differs from the float64 reference at {distance} Å.")
        report.append({
            "distance_A": distance, "scores": {name: signals[name] for name in scores},
            "ntk_force_by_atom": arrays["ntk_force"].tolist(), "trigger_flags": triggers,
            "decision": decision["outcome"],
        })
    print(json.dumps({"training_structures": len(training), "cutoffs": cutoffs,
                      "device": str(estimator.device), "dtype": str(estimator.dtype),
                      "trajectory": trajectory_smoke(estimator, structures[-2:], args.engine),
                      "matching_decisions": len(report), "replay": report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
