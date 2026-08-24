import json

import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import (
    annealing_schedule,
    build_input_dict,
    compute_empirical_formula,
    load_config_file,
    random_rotation_matrix,
    store_current_data,
    system_checker,
)


def test_annealing_schedule_linear():
    assert annealing_schedule(t=0.0, tmax=10.0, amp=0.0, per=2.0, srt=100.0, end=300.0) == 100.0
    assert annealing_schedule(t=5.0, tmax=10.0, amp=0.0, per=2.0, srt=100.0, end=300.0) == 200.0
    assert annealing_schedule(t=10.0, tmax=10.0, amp=0.0, per=2.0, srt=100.0, end=300.0) == 300.0


def test_annealing_schedule_with_sinusoidal_oscillation():
    result = annealing_schedule(t=1.0, tmax=10.0, amp=50.0, per=2.0, srt=100.0, end=300.0)

    assert np.isclose(result, 170.0)


def test_compute_empirical_formula():
    assert compute_empirical_formula(["H", "O", "H", "C"]) == "C01_H02_O01"


def test_random_rotation_matrix():
    rotation = random_rotation_matrix(randnums=np.array([0.25, 0.5, 0.75]))

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(abs(np.linalg.det(rotation)), 1.0)


def test_system_checker_accepts_valid_system_and_rejects_bad_data():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    valid = [{"moleculeid": "h2"}, atoms, {"energy": np.array([-1.0])}]

    assert system_checker(valid)
    assert not system_checker([{"moleculeid": "bad"}, atoms, {"forces": np.array([np.nan])}], kill_on_fail=False)
    with pytest.raises(RuntimeError):
        system_checker([{}, atoms, {}])


def test_load_config_file_paths(tmp_path):
    config_path = tmp_path / "master.json"
    config_path.write_text(
        json.dumps(
            {
                "master_directory": "pwd",
                "scratch_dir": "scratch",
                "model_path": "models/model-{:04d}",
                "absolute_dir": "/already/absolute",
            }
        )
    )

    config = load_config_file(str(config_path))

    assert config["master_directory"].endswith("/")
    assert config["scratch_dir"].endswith("/scratch")
    assert config["model_path"].endswith("/models/model-{:04d}")
    assert config["model_dir"].endswith("/models/")
    assert config["absolute_dir"] == "/already/absolute"


def test_store_current_data_writes_converged_sorted_h5(tmp_path):
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    molecule = MoleculesObject(atoms, "water-0")
    molecule.store_results({"energy": -1.5, "forces": np.arange(9).reshape(3, 3)})
    molecule.set_converged_flag(True)

    unconverged = MoleculesObject(atoms.copy(), "skip-me")
    unconverged.store_results({"energy": 0.0, "forces": np.zeros((3, 3))})
    unconverged.set_converged_flag(False)

    h5_path = tmp_path / "data.h5"
    store_current_data(
        str(h5_path),
        [molecule, unconverged],
        {"energy": ["energy", "system", 2.0], "forces": ["forces", "atomic", 0.5]},
    )

    with h5py.File(h5_path, "r") as h5:
        assert list(h5.keys()) == ["H02_O01"]
        group = h5["H02_O01"]
        assert [item.decode("utf-8") for item in group["species"][()]] == ["H", "H", "O"]
        assert [item.decode("utf-8") for item in group["_id"][()]] == ["water-0"]
        np.testing.assert_allclose(group["energy"][()], [-3.0])
        assert group["forces"].shape == (1, 3, 3)
        assert group["cell"].shape == (1, 3, 3)
