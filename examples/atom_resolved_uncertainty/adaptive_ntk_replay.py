"""Train a toy HIP-NN and compare fixed and adaptive energy-NTK selection."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from ase.io import read
from hippynn import custom_kernels

from alframework.ml_interfaces.hippynn.ntk import EnergyNTK, NTKPackedEvaluator
from alframework.samplers.adaptive_ntk import initial_energy_score, select_ntk_candidates
from alframework.samplers.batched_mlmd import mlmd_trajectories_task, sample_trajectories
from alframework.tools.molecules import MoleculesObject
from alframework.tools.sampler_coordinator import SamplerCoordinator
from ntk_replay import molecule, train_graph


def fixed_pool(estimator, structures, budget):
    candidates = []
    for index, atoms in enumerate(structures):
        candidates.append({
            "trajectory_id": "replay", "step": index,
            "energy_ntk": estimator.candidate_features(atoms.numbers, atoms.positions, atoms.pbc),
        })
    return {
        "candidate_count": len(candidates),
        "budget": budget,
        "fixed": [{"frame": row["step"], **row["adaptive_ntk"]}
                  for row in select_ntk_candidates(candidates, budget, adaptive=False)],
        "adaptive": [{"frame": row["step"], **row["adaptive_ntk"]}
                     for row in select_ntk_candidates(candidates, budget)],
    }


def cohort_smoke(graph, basis, device, engine, cap, budget):
    # Simulate two separate workers opening the same fitted basis.
    models = [NTKPackedEvaluator(EnergyNTK.from_basis(graph, basis, device=device)) for _ in range(2)]
    molecules = [MoleculesObject(molecule(distance), f"mol-0000-{i:010d}")
                 for i, distance in enumerate((0.78, 1.64))]
    initial = []
    for model, item in zip(models, molecules):
        atoms = item.get_atoms()
        initial.append(initial_energy_score({
            "energy_ntk": model.estimator.candidate_features(atoms.numbers, atoms.positions, atoms.pbc)
        }))
    config = {
        "driver": {"engine": engine}, "dt": 0.25, "maxt": 0.003, "Ncheck": 2, "distcut": 0.3,
        "random_seed": 7, "metadata_files": "none",
        "srt_temp": [300.0, 300.0], "end_temp": [300.0, 300.0], "amp_temp": [0.0, 0.0],
        "per_temp": [1.0, 1.0], "amp_dens": None, "per_dens": None, "end_dens": None,
        "acquisition": {
            "terms": {"energy_ntk": {"uncertainty": "ntk_energy", "cutoff": 0.1 * np.sqrt(min(initial))}},
            "eligibility": ["energy_ntk"], "strategy": "largest", "trajectory_policy": "continue",
            "selection": {"mode": "cohort_top_k", "selector": "adaptive_ntk",
                          "cohort_size": 2, "k": budget, "max_candidates_per_trajectory": cap},
        },
    }
    coordinator = SamplerCoordinator(mlmd_trajectories_task, "batched", config)
    for item in molecules:
        coordinator.pool.admit(item, 0)
    results = [sample_trajectories([item], model, config, device=device)[0]
               for item, model in zip(molecules, models)]
    pool = []
    for result in results:
        metadata = result.get_metadata()
        if metadata["trajectory_outcome"] != "eligible":
            raise AssertionError(f"Adaptive NTK sampler failed: {metadata.get('sampling_error', metadata)}")
        retained = metadata["acquisition_candidates"]
        if len(retained) != cap or metadata["acquisition_candidates_pruned"] <= 0:
            raise AssertionError("Expected each worker to prune candidates to the configured cap.")
        if result.get_atoms() is not None:
            raise AssertionError("A sampler returned a frame before cohort selection.")
        pool.extend(retained)
    if coordinator.harvest(results[1:]):
        raise AssertionError("An incomplete cohort must wait for its other worker.")
    chosen = coordinator.harvest(results[:1])
    expected = select_ntk_candidates(pool, budget)
    if [frame.get_metadata()["frame_id"] for frame in chosen] != [row["frame_id"] for row in expected]:
        raise AssertionError("Coordinator and replay selections disagree on the same candidates.")
    if len(chosen) != min(budget, len(pool)) or coordinator.harvest([]):
        raise AssertionError("The coordinator did not release exactly one batch within budget.")
    return {
        "engine": engine, "retained": len(pool), "pruned": coordinator.pool.finished[-1]["candidates_pruned"],
        "sent_to_labeling": [{"frame_id": frame.get_metadata()["frame_id"], **frame.get_metadata()["adaptive_ntk"]}
                             for frame in chosen],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--engine", choices=("torchsim", "alchemi"), help="Also run an MD cohort on this engine.")
    parser.add_argument("--pool", type=Path, help="Replay all frames in this nonperiodic hydrogen geometry file.")
    parser.add_argument("--budget", type=int, default=3, help="Label budget for both fixed-pool selections.")
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive.")
    if args.engine and torch.device(args.device).type == "mps":
        parser.error("Use MPS for fixed-pool replay. The packed MD state uses float64.")
    if args.engine == "alchemi" and torch.device(args.device).type == "cuda" and torch.version.hip is not None:
        parser.error("Use TorchSim for ROCm. ALCHEMI's GPU integrator uses NVIDIA CUDA.")
    distances = (0.72, 0.721, 0.723, 0.93, 1.03, 1.24, 1.51, 1.65, 1.66, 1.67, 1.9)
    structures = read(args.pool, index=":") if args.pool else [molecule(distance) for distance in distances]
    custom_kernels.set_custom_kernels(False)
    torch.set_num_threads(2)
    graph, training = train_graph(energy_scale=0.1, force_scale=1.0)
    fitted = EnergyNTK(graph, training, device=args.device, seed=7)
    with TemporaryDirectory(prefix="alf-adaptive-ntk-") as directory:
        path = Path(directory) / "energy_ntk.npz"
        fitted.save_basis(path)
        estimator = EnergyNTK.from_basis(graph, path, device=args.device)
        replay = fixed_pool(estimator, structures, args.budget)
        if args.pool is None:
            replay["distances_A"] = distances
        report = {"device": str(estimator.device), "basis_id": estimator.basis_id, "fixed_pool": replay}
        if args.engine:
            report["md_cohort"] = cohort_smoke(graph, path, args.device, args.engine, cap=2, budget=3)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
