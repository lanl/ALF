"""Optional MACE atom-resolved uncertainty using PNNL's quantile regressor."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from ase.calculators.calculator import Calculator, all_changes

from alframework.ml_interfaces.ensemble_inference import atomic_uncertainty_arrays
from alframework.tools.files import write_json_atomic


TESTED_UQ_MLIP_REVISION = "61ae6e171db4f64382b81df34673b93edc4cee42"


def _checkpoint(model_directory):
    directory = Path(model_directory).resolve()
    paths = sorted(directory.glob("model-*/mace.model"))
    if (directory / "mace.model").is_file():
        paths.insert(0, directory / "mace.model")
    if len(paths) != 1 or not paths[0].is_file():
        raise ValueError(f"UQ-MLIP requires exactly one mace.model in {directory} or one model-* member.")
    return paths[0]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uq_dependencies():
    try:
        from mace.calculators import mace_mp
        from uq_mlip import UQCalculator, UQModel, get_extractor
    except ImportError as exc:
        raise ImportError("MACE UQ requires PNNL uq-mlip with its mace extra.") from exc
    return mace_mp, UQCalculator, UQModel, get_extractor


def fit_mace_uq(model_directory, structures, *, device="cpu", lower_alpha=0.05,
                upper_alpha=0.95, n_estimators=100):
    """Fit and save a CPU quantile regressor for one MACE generation.

    Pass representative ASE Atoms and finish fitting before workers load the
    model directory. The device selects MACE embedding extraction. The fitted
    values are MACE's internal atomic energies.
    """
    if not 0 < lower_alpha < upper_alpha < 1:
        raise ValueError("UQ-MLIP quantiles must satisfy 0 < lower_alpha < upper_alpha < 1.")
    checkpoint = _checkpoint(model_directory)
    _, _, uq_model, get_extractor = _uq_dependencies()
    regressor_directory = Path(model_directory) / "uq_mlip"
    checkpoint_hash = _sha256(checkpoint)
    extractor = get_extractor("mace", model=str(checkpoint), device=device, default_type="float32")
    regressor = uq_model(regressor_directory, lower_alpha=lower_alpha, upper_alpha=upper_alpha,
                         n_estimators=n_estimators, device="cpu")
    regressor.fit(extractor.extract(structures))
    manifest_path = regressor_directory / "manifest.json"
    write_json_atomic(manifest_path, {
        "checkpoint_sha256": checkpoint_hash,
        "regressor_sha256": _sha256(regressor.model_path),
        "lower_alpha": lower_alpha,
        "upper_alpha": upper_alpha,
        "uq_mlip_version": version("uq-mlip"),
        "tested_uq_mlip_revision": TESTED_UQ_MLIP_REVISION,
    })
    return manifest_path


class _ALFMaceUQCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, calculator):
        super().__init__()
        self.calculator = calculator

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        if not len(self.atoms):
            raise ValueError("UQ-MLIP requires a nonempty structure.")
        # The upstream calculator writes arrays on its input. Give it the
        # calculator's copy so an MD state retains only the requested metadata.
        self.calculator.calculate(self.atoms, properties, system_changes)
        raw = self.calculator.results
        scores = atomic_uncertainty_arrays({"uq_mlip": raw["uq"]}, len(self.atoms))
        self.results = {
            "energy": raw["energy"],
            "forces": raw["forces"].copy(),
            "alf_uncertainty": {"uq_mlip_max": float(scores["uq_mlip"].max())},
            "alf_atomic_uncertainty": {"uq_mlip": scores["uq_mlip"].copy()},
        }


def load_mace_uq(model_directory, device="cpu"):
    """ALF ASE loader for one immutable MACE checkpoint and fitted regressor."""
    checkpoint = _checkpoint(model_directory)
    regressor_directory = Path(model_directory) / "uq_mlip"
    manifest_path = regressor_directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing UQ-MLIP manifest {manifest_path}. Fit the regressor with fit_mace_uq.")
    manifest = json.loads(manifest_path.read_text())
    if _sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("UQ-MLIP regressor was fitted for a different MACE checkpoint.")

    mace_mp, calculator, uq_model, get_extractor = _uq_dependencies()
    regressor = uq_model(regressor_directory, lower_alpha=manifest["lower_alpha"],
                         upper_alpha=manifest["upper_alpha"], device="cpu")
    if not regressor.model_path.is_file() or _sha256(regressor.model_path) != manifest["regressor_sha256"]:
        raise ValueError("UQ-MLIP regressor file is missing or differs from the fitted manifest.")
    regressor.load()
    base = mace_mp(model=str(checkpoint), device=device, default_dtype="float32")
    extractor = get_extractor("mace", model=str(checkpoint), device=device, default_type="float32")
    return [_ALFMaceUQCalculator(calculator(base, regressor, extractor=extractor))]
