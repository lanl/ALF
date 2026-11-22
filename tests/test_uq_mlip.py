"""Adapter and bundle checks without optional MACE or PNNL dependencies."""

import hashlib
import json

from ase import Atoms
import numpy as np
import pytest

from alframework.ml_interfaces import uq_mlip
from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics


class StandInUQCalculator:
    def __init__(self, values):
        self.values = values

    def calculate(self, atoms, properties, changes):
        self.results = {"energy": 1.5, "forces": np.zeros((len(atoms), 3)), "uq": self.values}
        atoms.arrays["uq"] = self.values


def test_ase_adapter_reports_atom_scores_in_input_order_without_mutating_the_input():
    atoms = Atoms("HOSi", positions=[[0, 0, 0], [1, 0, 0], [0, 2, 0]], cell=[5, 5, 5], pbc=True)
    source = StandInUQCalculator(np.array([0.12, 0.32, 0.24]))
    adapter = uq_mlip._ALFMaceUQCalculator(source)
    result = ASEEnsembleEvaluator([adapter]).predict([atoms])
    assert result["energies"][0, 0] == 1.5
    np.testing.assert_array_equal(result["forces"][0], np.zeros((1, 3, 3)))
    assert ensemble_statistics(result)[0]["uq_mlip_max"] == pytest.approx(0.32)
    assert "uq" not in atoms.arrays
    source.values[:] = 0.
    np.testing.assert_array_equal(result["atomic_uncertainty"][0]["uq_mlip"], [0.12, 0.32, 0.24])


@pytest.mark.parametrize("values", [np.array([1.]), np.array([0., np.nan]), np.array([0., -1.])])
def test_ase_adapter_rejects_invalid_atom_scores(values):
    adapter = uq_mlip._ALFMaceUQCalculator(StandInUQCalculator(values))
    with pytest.raises(ValueError, match="uq_mlip"):
        adapter.calculate(Atoms("H2"))
    assert not adapter.results


def test_bundle_rejects_missing_manifest_and_wrong_checkpoint(tmp_path):
    checkpoint = tmp_path / "mace.model"
    checkpoint.write_bytes(b"another MACE checkpoint")
    with pytest.raises(ValueError, match="Missing UQ-MLIP manifest"):
        uq_mlip.load_mace_uq(tmp_path)
    bundle = tmp_path / "uq_mlip"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256(b"original MACE checkpoint").hexdigest(),
    }))
    with pytest.raises(ValueError, match="different MACE checkpoint"):
        uq_mlip.load_mace_uq(tmp_path)


def test_bundle_rejects_a_replaced_regressor_before_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "mace.model"
    checkpoint.write_bytes(b"original MACE checkpoint")
    bundle = tmp_path / "uq_mlip"
    bundle.mkdir()
    (bundle / "regressor.pkl").write_bytes(b"another regressor")
    (bundle / "manifest.json").write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "regressor_sha256": hashlib.sha256(b"original regressor").hexdigest(),
        "lower_alpha": 0.05, "upper_alpha": 0.95,
    }))

    class Regressor:
        def __init__(self, directory, **kwargs):
            self.model_path = directory / "regressor.pkl"

    monkeypatch.setattr(uq_mlip, "_uq_dependencies", lambda: (None, None, Regressor, None))
    with pytest.raises(ValueError, match="differs from the fitted manifest"):
        uq_mlip.load_mace_uq(tmp_path)
