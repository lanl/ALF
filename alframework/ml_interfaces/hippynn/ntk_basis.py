"""Prepare and share a ground-state HIPPYNN NTK basis after training."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from alframework.tools.files import cached_sha256, write_json_atomic

MANIFEST_FILE = "ntk_training_split.json"
BASIS_FILE = "ntk_basis.npz"
PREPARE_COMMAND = "python -m alframework.ml_interfaces.hippynn.ntk_basis prepare"
CHECKPOINT_FILES = ("experiment_structure.pt", "best_model.pt")


def _shard_fingerprint(path):
    import h5py

    from alframework.tools.label_hdf5 import LABEL_PARTS, validate_label_parts

    with h5py.File(path, "r") as shard:
        validate_label_parts(shard)
        parts = json.loads(shard.attrs.get(LABEL_PARTS, "[]"))
    return {"sha256": cached_sha256(path),
            "parts": {name: cached_sha256(Path(path).parent / name) for name in parts}}


def capture_training_split(database, original_indices, *, species_key, coordinates_key, cell_key,
                           energy_key, force_key):
    """Capture the train rows before a distributed trainer can release the split arrays."""
    train = database.splits["train"]
    selected = np.isin(original_indices, train["indices"].cpu().numpy())
    if selected.sum() != len(train["indices"]) or len(np.unique(original_indices)) != len(original_indices):
        raise ValueError("Cannot identify the HIPPYNN training rows by their database indices.")

    files = {}
    start = 0
    for path, group, count in database.alf_frame_groups:
        stop = start + count
        rows = np.flatnonzero(selected[start:stop]).tolist()
        if rows:
            files.setdefault(path, []).append({"path": group, "rows": rows})
        start = stop
    if start != len(original_indices) or not files:
        raise ValueError("The HIPPYNN source groups do not match the original training database.")

    keys = {"species": species_key, "coordinates": coordinates_key, "cell": cell_key,
            "energy": energy_key, "forces": force_key}
    dtypes = {role: str(train[key].dtype).split(".")[-1] for role, key in keys.items()
              if role in {"coordinates", "cell"} and key is not None}
    sources = [{"path": path, **_shard_fingerprint(path), "groups": groups} for path, groups in files.items()]
    return {"keys": keys, "dtypes": dtypes, "files": sources, "training_structure_count": int(selected.sum())}


def _checkpoint_fingerprint(directory):
    return {name: cached_sha256(directory / name) for name in CHECKPOINT_FILES}


def write_training_split(split):
    """Publish the captured split with the finished checkpoint in the trainer's working directory."""
    write_json_atomic(MANIFEST_FILE, {**split, "checkpoint": _checkpoint_fingerprint(Path.cwd())})


def _member_manifest(model_directory, member):
    root = Path(model_directory)
    members = [root] if (root / CHECKPOINT_FILES[0]).is_file() else sorted(root.glob("model-*/"))
    if type(member) is not int or not 0 <= member < len(members):
        raise ValueError(f"{model_directory} has {len(members)} members, not member {member}.")
    directory = members[member]
    path = directory / MANIFEST_FILE
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{path} is missing. Train the checkpoint with record_ntk_training_split enabled.") from None
    if not manifest["keys"]["forces"]:
        raise ValueError("Force NTK preparation requires a HIPPYNN checkpoint trained on forces.")
    return directory, manifest


def _split_fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _checked_shards(manifest, shards):
    files = manifest["files"]
    paths = [entry["path"] for entry in files] if shards is None else list(shards)
    if len(paths) != len(files):
        raise ValueError("Provide one path per NTK training shard in manifest order.")
    for path, entry in zip(paths, files):
        if _shard_fingerprint(path) != {key: entry[key] for key in ("sha256", "parts")}:
            raise ValueError(f"{path} does not match the NTK training manifest in that position.")
    return paths


def _training_geometries(manifest, paths):
    from ase import Atoms

    from alframework.ml_interfaces.hippynn.interface import _species_numbers
    from alframework.tools.h5_parallel import read_group

    keys, dtypes = manifest["keys"], manifest["dtypes"]
    species_key, coordinates_key, cell_key = (keys[key] for key in ("species", "coordinates", "cell"))
    wanted = {species_key, coordinates_key, cell_key} - {None}
    for path, source in zip(paths, manifest["files"]):
        for selection in source["groups"]:
            group = read_group(path, selection["path"], keys=wanted)
            numbers = _species_numbers(group[species_key])[0]
            real = numbers > 0
            positions = np.asarray(group[coordinates_key], dtype=dtypes["coordinates"])
            cells = None if cell_key is None else np.asarray(group[cell_key], dtype=dtypes["cell"])
            for row in selection["rows"]:
                yield Atoms(numbers=numbers[real], positions=positions[row, real],
                            cell=None if cells is None else cells[row], pbc=cells is not None)


def _adapter(model, manifest, device, dtype=None):
    from hippynn.graphs import GraphModule

    from alframework.ml_interfaces.hippynn.interface import _target
    from alframework.ml_interfaces.hippynn.ntk import HippynnNTKAdapter

    graph = GraphModule(model.input_nodes, [_target(model, manifest["keys"]["energy"])])
    return HippynnNTKAdapter(graph, device=device, dtype=dtype)


def prepare_hippynn_ground_state_ntk(model_directory, *, energy_scale, force_scale, device="cpu", member=0,
                                     basis_file=BASIS_FILE, shards=None, sketch_dim=64, seed=0, ridge_factor=1e-3):
    """Fit one checkpoint's energy and force basis or reuse a matching prepared artifact."""
    from alframework.ml_interfaces.hippynn.interface import _load_member
    from alframework.ml_interfaces.hippynn.ntk import _matching_basis
    from alframework.ml_interfaces.ntk import NTKEstimator

    energy_scale, force_scale, ridge_factor = map(float, (energy_scale, force_scale, ridge_factor))
    if any(not math.isfinite(value) or value <= 0 for value in (energy_scale, force_scale, ridge_factor)):
        raise ValueError("Energy scale, force scale, and ridge factor must be positive and finite.")
    if type(sketch_dim) is not int or sketch_dim < 1:
        raise ValueError("sketch_dim must be a positive integer.")
    directory, manifest = _member_manifest(model_directory, member)
    if manifest.get("checkpoint") != _checkpoint_fingerprint(directory):
        raise ValueError(f"{directory} does not match the checkpoint recorded in its NTK training manifest.")
    adapter = _adapter(_load_member(str(directory), "cpu"), manifest, device)
    provenance = {"training_split_sha256": _split_fingerprint(manifest), "ridge_factor": ridge_factor}
    expected = {"fit_provenance": provenance, "model": adapter.digest(), "sketch_dim": sketch_dim,
                "seed": int(seed), "dtype": str(adapter.dtype).split(".")[-1],
                "training_structure_count": manifest["training_structure_count"],
                "scales": {"energy": energy_scale, "force": force_scale, "hessian": None}}
    path = directory / basis_file
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = _checked_shards(manifest, shards)
    with path.with_name(f".{path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            restored = _matching_basis(adapter, path, expected)
            if restored is None:
                raise ValueError(f"{path} holds a different NTK fit. Choose another --basis-file.")
            return restored
        estimator = NTKEstimator(adapter, _training_geometries(manifest, paths), energy_scale=energy_scale,
                                 force_scale=force_scale, sketch_dim=sketch_dim, seed=seed,
                                 ridge_factor=ridge_factor, fit_provenance=provenance)
        if estimator.training_structure_count != manifest["training_structure_count"]:
            raise ValueError("NTK basis frame count does not match the recorded train split.")
        estimator.save_basis(path)
        return estimator


def _load_prepared_basis(model_directory, device, member, basis_file, ntk_engine):
    import torch

    from alframework.ml_interfaces.hippynn.interface import _load_member
    from alframework.ml_interfaces.ntk.estimator import NTKEstimator, read_basis_metadata

    directory, manifest = _member_manifest(model_directory, member)
    path = directory / basis_file
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        raise ValueError(f"{path} is missing. Run {PREPARE_COMMAND} first.") from None
    try:
        with stream:
            metadata = read_basis_metadata(stream)
            fingerprint = (metadata.get("fit_provenance") or {}).get("training_split_sha256")
            if (fingerprint != _split_fingerprint(manifest)
                    or metadata.get("training_structure_count") != manifest["training_structure_count"]
                    or metadata["scales"]["force"] is None):
                raise ValueError("the basis does not match the recorded training split and force fit")
            model = _load_member(str(directory), "cpu")
            adapter = _adapter(model, manifest, device, dtype=getattr(torch, metadata["dtype"]))
            stream.seek(0)
            estimator = NTKEstimator.from_basis(adapter, stream, engine=ntk_engine)
    except ValueError as exc:
        raise ValueError(f"Cannot load {path}: {exc}. Run {PREPARE_COMMAND} with a matching basis filename.") from exc
    return estimator, model, manifest["keys"]


def load_hippynn_ground_state_ntk_estimator(model_directory, device="cpu", *, member=0, basis_file=BASIS_FILE,
                                            ntk_engine="auto"):
    """Load a prepared ground-state basis for offline replay."""
    estimator, _, _ = _load_prepared_basis(model_directory, device, member, basis_file, ntk_engine)
    return estimator


def load_hippynn_ground_state_ntk(model_directory, device="cuda:0", *, member=0, basis_file=BASIS_FILE,
                                 force_score="projected", force_probes=8, hippynn_kernel="auto",
                                 pair_skin=1.0, neighbor_backend="ase", energy_score=False, ntk_engine="auto"):
    """Load one HIPPYNN member for packed dynamics and NTK acquisition."""
    from alframework.ml_interfaces.hippynn.interface import HippynnBatchedEnsemble
    from alframework.ml_interfaces.hippynn.ntk import GroundStateNTKPackedEvaluator

    estimator, model, keys = _load_prepared_basis(model_directory, device, member, basis_file, ntk_engine)
    inference = HippynnBatchedEnsemble(model_directory, device=device, energy_key=keys["energy"],
                                      force_key=keys["forces"], hippynn_kernel=hippynn_kernel,
                                      pair_skin=pair_skin, neighbor_backend=neighbor_backend, _member=model)
    return GroundStateNTKPackedEvaluator(estimator, inference, force_score=force_score, force_probes=force_probes,
                                        energy_score=energy_score)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    prepare = parser.add_subparsers(dest="command", required=True).add_parser("prepare")
    prepare.add_argument("model_directory", help="A checkpoint directory or a generation with model-* members.")
    prepare.add_argument("--member", type=int, default=0)
    prepare.add_argument("--energy-scale", type=float, required=True, help="Energy scale in eV.")
    prepare.add_argument("--force-scale", type=float, required=True, help="Force scale in eV/Å.")
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument("--sketch-dim", type=int, default=64)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--ridge-factor", type=float, default=1e-3)
    prepare.add_argument("--basis-file", default=BASIS_FILE, help="Filename relative to the member directory.")
    prepare.add_argument("--shards", nargs="+", help="Relocated shard paths in manifest order.")
    options = vars(parser.parse_args(argv))
    options.pop("command")
    estimator = prepare_hippynn_ground_state_ntk(**options)
    print(json.dumps({"basis_id": estimator.basis_id, "device": str(estimator.device), **estimator.metadata}, indent=2))


if __name__ == "__main__":
    main()
