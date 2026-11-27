"""Prepare public molecular QM data and compare ensemble and NTK acquisition on fixed pools."""

import argparse
import csv
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from ase import Atoms, units

from alframework.tools.files import cached_sha256, write_json_atomic


DATASETS = {
    "rmd17-aspirin": ("nuclear_charges", "coords", "energies", "forces", "old_indices"),
    "md22-ala": ("z", "R", "E", "F", None),
}
DATA_MANIFEST = "dataset.json"
TRAIN_SHARD = "training/data.h5"
TRAIN_GROUP = "molecule"
TRAIN_KEYS = {"species": "species", "coordinates": "coordinates", "cell": None,
              "energy": "energy", "forces": "forces"}
METHODS = ("ensemble_selected", "ntk_selected", "ntk_projected", "ntk_selected_energy")
BUDGETS = (0.01, 0.05, 0.10)
CALIBRATION_QUANTILE = 0.90


def _public_arrays(dataset, source):
    species, xyz, energy, force, chronology = DATASETS[dataset]
    with np.load(source, allow_pickle=False) as archive:
        numbers = archive[species]
        positions = archive[xyz]
        energies = archive[energy].reshape(-1) * (units.kcal / units.mol)
        forces = archive[force] * (units.kcal / units.mol)
        time_index = archive[chronology] if chronology else np.arange(len(positions))
    if (numbers.ndim != 1 or numbers.dtype.kind not in "iu" or not len(numbers)
            or np.any((numbers < 1) | (numbers > 118))):
        raise ValueError("Expected a fixed vector of atomic numbers in source atom order.")
    size = len(positions)
    if (not size or positions.shape != (size, len(numbers), 3) or forces.shape != positions.shape
            or energies.shape != (size,) or any(not np.isfinite(x).all() for x in (positions, energies, forces))):
        raise ValueError("Expected matching finite coordinates, total energies, and forces.")
    if time_index.shape != (size,) or time_index.dtype.kind not in "iu" or len(np.unique(time_index)) != size:
        raise ValueError("Expected distinct integer original trajectory indices.")
    return numbers, positions, energies, forces, time_index


def _split_rows(time_index, counts, seed, gap):
    order = np.argsort(time_index)
    first, second = (int(fraction * len(order)) for fraction in (0.6, 0.8))
    windows = (order[:first], order[first:second], order[second:])
    rng = np.random.default_rng(seed)
    result = {}
    for name, count, window in zip(("training", "calibration", "test"), counts, windows):
        if result:
            previous = order[first - 1 if name == "calibration" else second - 1]
            window = window[time_index[window] > int(time_index[previous]) + gap]
        if not 0 < count <= len(window):
            raise ValueError(f"{name} needs {count} frames but its trajectory block has {len(window)} available.")
        rows = rng.choice(window, size=count, replace=False)
        result[name] = rows[np.argsort(time_index[rows])]
    return result


def prepare(dataset, source, out, *, train_count=1000, calibration_count=256, test_count=1024, seed=0, gap=1000):
    """Write the ALF training shard and two independent replay pools."""
    import h5py

    if gap < 0 or seed < 0:
        raise ValueError("The trajectory gap and seed must be nonnegative.")
    if dataset == "rmd17-aspirin" and train_count > 1000:
        raise ValueError("The rMD17 release recommends at most 1000 structures for fitting.")
    numbers, positions, energies, forces, time_index = _public_arrays(dataset, source)
    splits = _split_rows(time_index, (train_count, calibration_count, test_count), seed, gap)
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {out}.")
    shard = out / TRAIN_SHARD
    shard.parent.mkdir(parents=True)
    rows = splits["training"]
    with h5py.File(shard, "w") as handle:
        group = handle.create_group(TRAIN_GROUP)
        group["species"] = np.asarray(Atoms(numbers=numbers).get_chemical_symbols(), dtype="S")
        group["coordinates"] = positions[rows]
        group["energy"] = energies[rows]
        group["forces"] = forces[rows]

    files = {"training": TRAIN_SHARD, "calibration": "calibration.npz", "test": "test.npz"}
    for name in ("calibration", "test"):
        rows = splits[name]
        np.savez_compressed(out / files[name], numbers=numbers, positions=positions[rows],
                            energies=energies[rows], forces=forces[rows], source_rows=rows)
    manifest = {
        "dataset": dataset, "input_sha256": cached_sha256(source), "atomic_numbers": numbers.tolist(),
        "trajectory_order": DATASETS[dataset][-1] or "archive row order", "seed": seed, "guard_frames": gap,
        "splits": {name: {"file": files[name], "sha256": cached_sha256(out / files[name]),
                          "source_rows": rows.tolist(), "trajectory_indices": time_index[rows].tolist()}
                   for name, rows in splits.items()},
    }
    write_json_atomic(out / DATA_MANIFEST, manifest)
    return manifest


def _verified_models(model_dir, training):
    from alframework.ml_interfaces.hippynn.ntk_basis import CHECKPOINT_FILES, MANIFEST_FILE

    models = sorted(Path(model_dir).glob("model-*/"))
    if len(models) < 2:
        raise ValueError("The ensemble baseline requires at least two HIPPYNN members.")
    results = []
    for model in models:
        path = model / MANIFEST_FILE
        if not path.is_file():
            raise ValueError(f"{path} is missing. Train each member with record_ntk_training_split=True.")
        record = json.loads(path.read_text())
        sources = record["files"]
        if (record["keys"] != TRAIN_KEYS or len(sources) != 1
                or sources[0]["sha256"] != training["sha256"] or sources[0]["parts"]):
            raise ValueError(f"{model} must be trained from the benchmark's prepared training shard.")
        groups = sources[0]["groups"]
        if (len(groups) != 1 or groups[0]["path"] != f"/{TRAIN_GROUP}" or not groups[0]["rows"]
                or any(not 0 <= i < len(training["source_rows"]) for i in groups[0]["rows"])):
            raise ValueError(f"{model} has invalid training row identities.")
        checkpoints = {name: cached_sha256(model / name) for name in CHECKPOINT_FILES}
        if checkpoints != record["checkpoint"]:
            raise ValueError(f"{model} has changed since training recorded its split.")
        results.append({"member": model.name, "checkpoints": checkpoints, "split_sha256": cached_sha256(path)})
    return results


def _read_pool(data_dir, manifest, name):
    spec = manifest["splits"][name]
    path = data_dir / spec["file"]
    if cached_sha256(path) != spec["sha256"]:
        raise ValueError(f"{path} does not match the benchmark data manifest.")
    with np.load(path, allow_pickle=False) as saved:
        pool = dict(saved)
    if (pool["source_rows"].tolist() != spec["source_rows"]
            or pool["numbers"].tolist() != manifest["atomic_numbers"]):
        raise ValueError(f"{path} has a different frame or atom order from the manifest.")
    return pool


def _time_call(call, synchronize, durations):
    synchronize()
    start = perf_counter()
    value = call()
    synchronize()
    durations.append(perf_counter() - start)
    return value


def _score_pool(pool, indices, member, ensemble, estimator, probes, *, verify_predictions=False):
    import torch

    device = estimator.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    numbers = torch.as_tensor(pool["numbers"], device=device)
    cells = torch.zeros((1, 3, 3), device=device)
    pbc = (False, False, False)
    durations = {name: [] for name in METHODS}
    records = []
    for frame, (xyz, energy, forces) in enumerate(zip(pool["positions"], pool["energies"], pool["forces"])):
        xyz = torch.as_tensor(xyz, device=device)
        structure = estimator.adapter.structure(numbers, xyz)
        # Use the deployed member's physical forces as the common reference, even for ensemble rankings.
        prediction = _time_call(lambda: ensemble.evaluate(numbers, xyz, cells, pbc, [len(numbers)]),
                                synchronize, durations["ensemble_selected"])
        member_energies, member_forces = (value.detach().cpu().numpy() for value in prediction)
        selected = _time_call(lambda: estimator.score_forces(structure, atom_indices=indices),
                              synchronize, durations["ntk_selected"])
        projected = _time_call(lambda: estimator.score_forces(structure, force_probes=probes),
                               synchronize, durations["ntk_projected"])
        joint = _time_call(lambda: estimator.score_forces(structure, atom_indices=indices, include_energy=True),
                           synchronize, durations["ntk_selected_energy"])
        if verify_predictions and frame < 3:
            offline = estimator.score(structure, forces=False)
            np.testing.assert_allclose(offline["forces"][0].cpu(), member_forces[member], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(offline["energy"][0].cpu(), member_energies[member, 0], rtol=1e-5, atol=1e-4)
        target_forces = member_forces[member, indices] - forces[indices]
        spread = np.std(member_forces[:, indices, :].astype(float), axis=0)
        records.append({
            "source_row": int(pool["source_rows"][frame]),
            "selected_force_error_eV_per_A": float(np.linalg.norm(target_forces, axis=-1).max()),
            "energy_residual_eV": float(member_energies[member, 0] - energy),
            "ensemble_selected": float(np.linalg.norm(spread, axis=-1).max()),
            "ntk_selected": float(selected["max_force_uncertainty_score"][0]),
            "ntk_projected": float(projected["force_rms_uncertainty"][0]),
            "ntk_joint_force": float(joint["max_force_uncertainty_score"][0]),
            "ntk_total_energy": float(joint["energy_uncertainty_score"][0]),
        })
    return records, durations


def _calibrate(records, force_score_cutoff, energy_score_cutoff):
    def quantile(key):
        return float(np.quantile([row[key] for row in records], CALIBRATION_QUANTILE))

    joint_force = quantile("ntk_joint_force") if force_score_cutoff is None else force_score_cutoff
    total_energy = quantile("ntk_total_energy") if energy_score_cutoff is None else energy_score_cutoff
    if any(not math.isfinite(cutoff) or cutoff <= 0 for cutoff in (joint_force, total_energy)):
        raise ValueError("Joint score cutoffs must be positive and finite. Supply explicit cutoffs for zero spread.")
    shift = float(np.median([row["energy_residual_eV"] for row in records]))
    return {"quantile": CALIBRATION_QUANTILE, "joint_force_cutoff_eV_per_A": joint_force,
            "total_energy_cutoff_eV": total_energy, "energy_residual_offset_eV": shift,
            "large_force_error_eV_per_A": quantile("selected_force_error_eV_per_A"),
            "large_energy_error_eV": float(np.quantile(
                [abs(row["energy_residual_eV"] - shift) for row in records], CALIBRATION_QUANTILE))}


def _budget_metrics(records, calibration):
    for row in records:
        row["energy_error_eV"] = abs(row["energy_residual_eV"] - calibration["energy_residual_offset_eV"])
        row["ntk_selected_energy"] = max(row["ntk_joint_force"] / calibration["joint_force_cutoff_eV_per_A"],
                                         row["ntk_total_energy"] / calibration["total_energy_cutoff_eV"])
        row["large_force_error"] = row["selected_force_error_eV_per_A"] > calibration["large_force_error_eV_per_A"]
        row["large_energy_error"] = row["energy_error_eV"] > calibration["large_energy_error_eV"]
    result = {}
    totals = {key: sum(row[key] for row in records) for key in ("large_force_error", "large_energy_error")}
    for method in METHODS:
        ranked = sorted(records, key=lambda row: (-row[method], row["source_row"]))
        result[method] = []
        for budget in BUDGETS:
            k = math.ceil(budget * len(ranked))
            entry = {"budget_fraction": budget, "frames": k, "selected_source_rows": [
                row["source_row"] for row in ranked[:k]]}
            for name, total in totals.items():
                missed = total - sum(row[name] for row in ranked[:k])
                entry[f"{name}_missed"] = missed
                entry[f"{name}_missed_fraction"] = missed / total if total else None
            result[method].append(entry)
    return {"test_error_counts": totals, "methods": result}


def score(data_dir, model_dir, out, indices, *, member=0, basis_file="ntk_basis.npz", device="cpu", probes=8,
          force_score_cutoff=None, energy_score_cutoff=None, ntk_engine="auto"):
    """Replay a verified basis and matching ensemble, then report fixed-budget error coverage."""
    from alframework.ml_interfaces.hippynn.interface import HippynnBatchedEnsemble
    from alframework.ml_interfaces.hippynn.ntk_basis import load_hippynn_ground_state_ntk_estimator
    from alframework.samplers.acquisition import resolve_force_atom_indices

    data_dir, out = Path(data_dir), Path(out)
    manifest_path = data_dir / DATA_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    rows = [row for part in manifest["splits"].values() for row in part["source_rows"]]
    if len(rows) != len(set(rows)):
        raise ValueError("Training, calibration, and test pools must be disjoint.")
    if not indices or any(type(i) is not int or i < 0 for i in indices) or len(indices) != len(set(indices)):
        raise ValueError("Provide distinct nonnegative atom indices in the source atom order.")
    indices = list(resolve_force_atom_indices({"indices": tuple(indices)},
                   Atoms(numbers=manifest["atomic_numbers"]), manifest["dataset"]))
    if probes < 1:
        raise ValueError("Projected force scoring needs at least one probe.")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {out}.")

    model_records = _verified_models(model_dir, manifest["splits"]["training"])
    cal_pool = _read_pool(data_dir, manifest, "calibration")
    test_pool = _read_pool(data_dir, manifest, "test")
    ntk = load_hippynn_ground_state_ntk_estimator(model_dir, device=device, member=member, basis_file=basis_file,
                                                ntk_engine=ntk_engine)
    ensemble = HippynnBatchedEnsemble(model_dir, device=device)
    cal_records, _ = _score_pool(cal_pool, indices, member, ensemble, ntk, probes, verify_predictions=True)
    calibration = _calibrate(cal_records, force_score_cutoff, energy_score_cutoff)
    test_records, times = _score_pool(test_pool, indices, member, ensemble, ntk, probes)
    report = {
        "dataset": manifest["dataset"], "dataset_manifest_sha256": cached_sha256(manifest_path),
        "basis_id": ntk.basis_id, "basis_metadata": ntk.metadata, "reference_member": member,
        "models": model_records, "atom_indices": indices, "probes": probes, "device": device,
        "ntk_engine": ntk_engine,
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in ("torch", "hippynn", "ase", "numpy")}},
        "calibration": calibration, **_budget_metrics(test_records, calibration),
        "test_time_per_frame_seconds": {
            method: {"median": float(np.median(t)), "total": sum(t)} for method, t in times.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("calibration", cal_records), ("test", test_records)):
        with (out / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json_atomic(out / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Make an ALF shard and two held-out replay pools.")
    prep.add_argument("--dataset", choices=DATASETS, required=True)
    prep.add_argument("--source", required=True, help="Downloaded public NPZ file.")
    prep.add_argument("--out", required=True)
    prep.add_argument("--train-count", type=int, default=1000)
    prep.add_argument("--calibration-count", type=int, default=256)
    prep.add_argument("--test-count", type=int, default=1024)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--gap", type=int, default=1000, help="Excluded trajectory indices after each split boundary.")
    replay = commands.add_parser("score", help="Compare scores on a fixed QM-labeled test pool.")
    replay.add_argument("--data-dir", required=True)
    replay.add_argument("--model-dir", required=True, help="Generation directory with model-* members.")
    replay.add_argument("--out", required=True)
    replay.add_argument("--indices", nargs="+", type=int, required=True, help="Selected zero-based atom indices.")
    replay.add_argument("--member", type=int, default=0)
    replay.add_argument("--basis-file", default="ntk_basis.npz")
    replay.add_argument("--device", default="cpu", help="cpu or a PyTorch cuda device (CUDA or ROCm).")
    replay.add_argument("--probes", type=int, default=8)
    replay.add_argument("--ntk-engine", choices=("auto", "reverse", "forward"), default="auto",
                        help="Differentiation backend used to score frames with the same fitted basis.")
    replay.add_argument("--force-score-cutoff", type=float, help="Override calibration cutoff in eV/Å.")
    replay.add_argument("--energy-score-cutoff", type=float, help="Override calibration cutoff in eV.")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            result = prepare(**args)
            print(json.dumps({name: len(spec["source_rows"]) for name, spec in result["splits"].items()}))
        else:
            result = score(**args)
            print(json.dumps(result["test_error_counts"]))
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
