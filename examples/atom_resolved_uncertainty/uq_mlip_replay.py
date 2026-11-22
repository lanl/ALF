"""Verify the PNNL provider on an isolated molecule and a periodic solid."""

import argparse
from contextlib import redirect_stdout
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform
import sys
from tempfile import TemporaryDirectory

from ase.build import bulk, molecule
import numpy as np
import torch

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
from alframework.ml_interfaces.uq_mlip import fit_mace_uq
from alframework.samplers.acquisition import acquisition_settings, assess_frame
from alframework.samplers.batched_mlmd import mlmd_trajectories_task, sample_trajectories, worker_ensemble
from alframework.tools.geometry import minimum_distance
from alframework.tools.molecules import MoleculesObject
from alframework.tools.sampler_coordinator import SamplerCoordinator


LOADER = "alframework.ml_interfaces.uq_mlip.load_mace_uq"
SCORE = "uq_mlip"


def fixtures(jitters, seed):
    rng = np.random.default_rng(seed)
    frames = []
    for magnitude in jitters:
        for atoms in (molecule("H2O"), bulk("Si", "diamond", a=5.43, cubic=True)):
            atoms.positions += rng.normal(scale=magnitude, size=atoms.positions.shape)
            frames.append(atoms)
    return frames


def settings(cutoff):
    return {
        "driver": {"engine": "ase"}, "ase_calculator": LOADER,
        "dt": 0.25, "maxt": 0.001, "Ncheck": 2, "distcut": 0.4, "random_seed": 12,
        "srt_temp": [100., 100.], "end_temp": [100., 100.], "amp_temp": [0., 0.],
        "per_temp": [1., 1.], "amp_dens": None, "per_dens": None, "end_dens": None,
        "acquisition": {
            "terms": {SCORE: {"uncertainty": "uq_mlip_max", "cutoff": cutoff}},
            "eligibility": [SCORE], "strategy": "largest",
        },
    }


def expected(reference, atoms):
    scored = atoms.copy()
    reference.calculate(scored)
    return reference.results


def replay(models, reference, frames):
    prediction = ASEEnsembleEvaluator(models).predict(frames)
    report = []
    for i, (atoms, signals, atomic) in enumerate(zip(
            frames, ensemble_statistics(prediction), prediction["atomic_uncertainty"])):
        truth = expected(reference, atoms)
        # Independent float32 forwards on an accelerator can reduce forces
        # in different orders. Keep score and decision checks separate.
        np.testing.assert_allclose(prediction["energies"][0, i], truth["energy"], atol=3e-5, rtol=1e-6)
        np.testing.assert_allclose(prediction["forces"][i][0], truth["forces"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(atomic[SCORE], truth["uq"])
        peak = float(truth["uq"].max())
        np.testing.assert_allclose(signals["uq_mlip_max"], peak)
        for cutoff in (peak * 0.5, peak * 1.5):
            config = settings(cutoff)
            decision = assess_frame(signals, minimum_distance(atoms), 0., config, acquisition_settings(config))
            want = "eligible" if peak > cutoff else "continue"
            assert decision["outcome"] == want
            assert decision["acquisition_trigger_flags"][SCORE] == (peak > cutoff)
            report.append({"periodic": bool(atoms.pbc.all()), "atoms": len(atoms),
                           "cutoff": cutoff, "score": peak, "decision": want})
    return report


def check_md(models, reference, frames, meta_root, device, *, cohort):
    cutoff = 0.01 * min(float(expected(reference, atoms)["uq"].max()) for atoms in frames)
    config = settings(cutoff)
    config["meta_dir"] = str(meta_root)
    molecules = [MoleculesObject(atoms.copy(), f"mol-0000-{i:010d}") for i, atoms in enumerate(frames)]
    if cohort:
        config["acquisition"]["selection"] = {"mode": "cohort_top_k", "cohort_size": 2, "k": 1}
        coordinator = SamplerCoordinator(mlmd_trajectories_task, "batched", config)
        for item in molecules:
            coordinator.pool.admit(item, 0)
    sampled = sample_trajectories(molecules, models, config, device=device)
    for result in sampled:
        assert result.get_metadata()["trajectory_outcome"] == "eligible", result.get_metadata()
    paths = sorted(meta_root.rglob("metadata-*.p"))
    assert len(paths) == len(frames)
    records = [pickle.loads(path.read_bytes()) for path in paths]
    if cohort:
        assert all(item.get_atoms() is None for item in sampled)
        chosen = coordinator.harvest(sampled)
        candidates = [candidate for record in records for candidate in record["acquisition_candidates"]]
        winner = min(candidates, key=lambda value: (-value["ranking_score"], value["trajectory_id"], value["step"]))
        assert len(chosen) == 1
        assert chosen[0].get_metadata()["frame_id"] == winner["frame_id"]
        durable = [winner["metadata"]]
        original = [frames[int(winner["trajectory_id"].split("-")[-1])]]
    else:
        chosen, durable, original = sampled, records, frames
    for item, saved, initial in zip(chosen, durable, original):
        atoms = item.get_atoms()
        np.testing.assert_array_equal(atoms.numbers, initial.numbers)
        np.testing.assert_array_equal(atoms.pbc, initial.pbc)
        np.testing.assert_allclose(atoms.cell, initial.cell)
        scores = expected(reference, atoms)["uq"]
        np.testing.assert_allclose(item.get_metadata()["atomic_uncertainty"][SCORE], scores)
        np.testing.assert_allclose(saved["atomic_uncertainty"][SCORE], scores)
    return {"selected": len(chosen), "periodic": [bool(item.get_atoms().pbc.all()) for item in chosen]}


def run(checkpoint, device):
    from mace.calculators import mace_mp
    from uq_mlip import UQCalculator, UQModel, get_extractor

    checkpoint = checkpoint.resolve(strict=True)
    torch.set_num_threads(2)
    calibration = fixtures((0., 0.01, 0.02, 0.04, 0.06, 0.09), 7)
    frames = fixtures((0.035, 0.075), 17)
    # Reverse atom order in the held-out frames as an additional alignment check.
    frames += [atoms[::-1] for atoms in frames[:2]]
    with TemporaryDirectory(prefix="alf-uq-mlip-") as directory:
        root = Path(directory)
        (root / "mace.model").symlink_to(checkpoint)
        manifest = fit_mace_uq(root, calibration, device=device, n_estimators=24)
        models = worker_ensemble({"ase_calculator": LOADER}, str(root), device)
        assert models is worker_ensemble({"ase_calculator": LOADER}, str(root), device)
        regressor = UQModel.from_dir(manifest.parent)
        base = mace_mp(model=str(checkpoint), device=device, default_dtype="float32")
        extractor = get_extractor("mace", model=str(checkpoint), device=device, default_type="float32")
        reference = UQCalculator(base, regressor, extractor=extractor)
        return {
            "versions": {name: version(name) for name in ("uq-mlip", "mace-torch", "ase", "torch", "xgboost")},
            "platform": platform.platform(),
            "devices": {"mlip": device, "embedding": device, "regressor": "cpu"},
            "fit_structures": len(calibration), "fit": json.loads(manifest.read_text()),
            "replay": replay(models, reference, frames),
            "first_trigger": check_md(models, reference, frames[:2], root / "first", device, cohort=False),
            "top_k": check_md(models, reference, frames[:2], root / "cohort", device, cohort=True),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a local MACE MP checkpoint.")
    parser.add_argument("--device", default="cpu", help="MACE inference and embeddings device. XGBoost uses CPU.")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        report = run(args.checkpoint, args.device)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
