"""TorchSim/ALCHEMI batched-MLMD acceptance and throughput on one device.

Builds periodic water boxes with ALF's condensed-phase builder, then for each
batch size:

1. compares batched HIPPYNN predictions (``load_hippynn_batched_ensemble``) with
   per-structure evaluation, within float32 tolerances ``rtol=atol=1e-5``;
2. times batched inference and the selected engine's dynamics steps;
3. records peak PyTorch-allocated device memory (not total process memory).

Writes one JSON artifact. It runs no scheduler jobs and no QM.

    python examples/torchsim_benchmark/benchmark.py --model models/model-0000 \
        --device cuda:0 --batch-sizes 1 2 8 32 50 --output results.json
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import alframework

from alframework.builders.builders import condensed_phase_builder, readMolFiles
from alframework.ml_interfaces.hippynn_interface import load_hippynn_batched_ensemble
from alframework.samplers.batched_mlmd import make_engine
from alframework.tools.molecules_class import MoleculesObject

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parents[0] / "simple_water" / "fragment_library/"


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    """Identify imported code, including copies inside an unrelated checkout."""
    root = Path(alframework.__file__).resolve().parent.parent
    if root != HERE.parent.parent:
        raise RuntimeError(f"Benchmark imports ALF from {root}; set PYTHONPATH to {HERE.parent.parent}.")
    git_root = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                              capture_output=True, text=True)
    revision, dirty = None, None
    if git_root.returncode == 0 and Path(git_root.stdout.strip()).resolve() == root:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                             '--untracked-files=no'], text=True).strip())
    files = sorted((root / 'alframework').rglob('*.py')) + [Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)): fingerprint(path) for path in files}
    tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'revision': revision, 'working_tree_dirty': dirty, 'root': str(root),
            'files_sha256': hashes, 'tree_sha256': tree}


def water_boxes(count, seed):
    import random
    from ase import Atoms
    random.seed(seed)
    np.random.seed(seed)
    library, _ = readMolFiles(str(LIBRARY) + "/")
    boxes = []
    while len(boxes) < count:
        cell = np.random.uniform(6.0, 8.0, 3)
        start = MoleculesObject(Atoms(cell=cell, pbc=True), f"box-{len(boxes)}")
        system = condensed_phase_builder(start, library, solute_molecules=["water.xyz"],
                                         solvent_molecules={"water.xyz": 1.0},
                                         density=np.random.uniform(0.9, 1.1), min_dist=1.25, max_patience=500,
                                         center_first_molecule=True, shake=0.05)
        boxes.append(system.get_atoms())
    return boxes


def pack(atoms_list, device):
    return (torch.as_tensor(np.concatenate([a.numbers for a in atoms_list]), device=device),
            torch.as_tensor(np.concatenate([a.positions for a in atoms_list]), device=device),
            torch.as_tensor(np.stack([a.cell.array for a in atoms_list]), device=device),
            atoms_list[0].pbc, [len(a) for a in atoms_list])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    device = torch.device(args.device)
    model = load_hippynn_batched_ensemble(args.model, device=str(device))
    if args.structures:
        from ase.io import read
        boxes = read(args.structures, index=':')
        if len(boxes) < max(args.batch_sizes):
            raise ValueError('The structure file has fewer frames than the largest batch size.')
    else:
        boxes = water_boxes(max(args.batch_sizes), args.seed)
    source = source_identity()
    packages = {}
    for name in ('torch', 'hippynn', 'ase', 'numpy', 'parsl', 'torch-sim-atomistic',
                 'nvalchemi-toolkit', 'nvalchemi-toolkit-ops', 'warp-lang'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    checkpoints = sorted(Path(args.model).glob('model-*/experiment_structure.pt')) + \
        sorted(Path(args.model).glob('model-*/best_model.pt'))
    report = {"device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
              "hip": torch.version.hip, "node": platform.node(),
              "engine": args.engine, "packages": packages,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "model": str(Path(args.model).resolve()), "n_models": model.n_models,
              "atoms_per_box": [len(b) for b in boxes], "rtol": args.rtol, "atol": args.atol,
              "source": source['revision'], "source_identity": source,
              "checkpoint_sha256": {str(p.resolve()): fingerprint(p) for p in checkpoints},
              "structures_sha256": fingerprint(Path(args.structures)) if args.structures else None,
              "memory_metric": "peak PyTorch allocated MiB; excludes non-PyTorch allocations",
              "parameters": {'dt_fs': .25, 'temperature_K': 300., 'warmup': args.warmup,
                             'inference_calls': args.inference_calls, 'md_steps': args.md_steps},
              "cases": []}
    individual = [model.evaluate(*pack([box], device)) for box in boxes]
    for width in args.batch_sizes:
        systems = boxes[:width]
        energies, forces = model.evaluate(*pack(systems, device))
        ref_e = torch.cat([e for e, _ in individual[:width]], dim=1)
        ref_f = torch.cat([f for _, f in individual[:width]], dim=1)
        energy_error = float((energies - ref_e).abs().max())
        force_error = float((forces - ref_f).abs().max())
        passed = bool(torch.all((energies - ref_e).abs() <= args.atol + args.rtol * ref_e.abs())
                      and torch.all((forces - ref_f).abs() <= args.atol + args.rtol * ref_f.abs()))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.warmup):
            model.evaluate(*pack(systems, device))
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.inference_calls):
            model.evaluate(*pack(systems, device))
        synchronize(device)
        inference = width * args.inference_calls / (time.perf_counter() - start)
        engine = make_engine(args.engine, [b.copy() for b in systems], model, 0.25, [300.0] * width,
                             [np.random.RandomState(i) for i in range(width)], device=str(device))
        warmup_failures = engine.step(list(range(width)), args.warmup)
        synchronize(device)
        start = time.perf_counter()
        failures = engine.step(list(range(width)), args.md_steps)
        synchronize(device)
        dynamics = width * args.md_steps / (time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
        case = {"batch_size": width, "max_energy_error_eV": energy_error, "max_force_error_eV_A": force_error,
                "parity_passed": passed, "inference_systems_per_s": inference,
                "dynamics_system_steps_per_s": dynamics,
                "md_failures": len({**warmup_failures, **failures}), "peak_MiB": peak}
        report["cases"].append(case)
        for index in range(width):
            engine.finish(index)
        print(json.dumps(case), flush=True)
    report["passed"] = all(c["parity_passed"] and c["md_failures"] == 0 for c in report["cases"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Ensemble directory containing model-* members.")
    parser.add_argument("--engine", choices=['torchsim', 'alchemi'], default='torchsim')
    parser.add_argument("--structures", help="Optional fixed extxyz structures, shared across worker environments.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 8, 32, 50])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--inference-calls", type=int, default=20)
    parser.add_argument("--md-steps", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
