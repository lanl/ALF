"""ML model interfaces: training tasks and ensemble loading for HIPPYNN and NeuroChem.

Trainers are faked, so these check how ALF calls them. Tests for further ML models belong here.
"""

import os

import pytest
from parsl.app.errors import RemoteExceptionWrapper

from alframework.ml_interfaces import hippynn_interface
from alframework.tools import ensemble_training
from alframework.tools.tools import build_input_dict
from tests.helpers.fakes import fake_training_pool, import_neurochem_with_fake_backend


def test_hippynn_training_configures_each_member(tmp_path, monkeypatch):

    FakePool, pool = fake_training_pool()
    monkeypatch.setattr(hippynn_interface.multiprocessing, "Pool", FakePool)

    completed, training_id = hippynn_interface.train_HIPPYNN_ensemble_task.func(
        ML_config={"n_models": 2, "energy_key": "energy", "species_key": "species"},
        h5_dir="h5store",
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_training_id=3,
        gpus_per_node=2,
    )

    assert completed == [True, True]
    assert training_id == 3
    assert (tmp_path / "models" / "model-0003" / "label_sources.json").is_file()
    assert pool.processes == 2
    assert "close" in pool.calls
    assert pool.func is hippynn_interface.train_HIPNN_model_wrapper
    assert pool.params_list == [
        {
            "energy_key": "energy",
            "species_key": "species",
            "model_dir": str(tmp_path / "models" / "model-0003" / "model-00"),
            "h5_train_dir": "h5store",
            "from_multiprocessing_nGPU": 2,
        },
        {
            "energy_key": "energy",
            "species_key": "species",
            "model_dir": str(tmp_path / "models" / "model-0003" / "model-01"),
            "h5_train_dir": "h5store",
            "from_multiprocessing_nGPU": 2,
        },
    ]


def test_hippynn_training_reports_failed_members(tmp_path, monkeypatch):

    logs = ["Training complete\n", "Stopped before convergence\n"]
    FakePool, _ = fake_training_pool(member_log=logs.__getitem__)
    monkeypatch.setattr(hippynn_interface.multiprocessing, "Pool", FakePool)

    completed, training_id = hippynn_interface.train_HIPPYNN_ensemble_task.func(
        ML_config={"n_models": 2},
        h5_dir="h5store",
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_training_id=4,
        gpus_per_node=1,
    )

    assert completed == [True, False]
    assert training_id == 4


def single_node_trainer(kind, tmp_path, monkeypatch):
    """Return a call to one single-node training task, with training replaced by a recorder."""
    trained = []
    if kind == "ground_state":
        FakePool, _ = fake_training_pool()

        class RecordingPool(FakePool):
            def __init__(self, processes):
                trained.append(processes)
                super().__init__(processes)

        monkeypatch.setattr(hippynn_interface.multiprocessing, "Pool", RecordingPool)
        task, extra = hippynn_interface.train_HIPPYNN_ensemble_task.func, {}
    else:
        monkeypatch.setattr(hippynn_interface, "train_excited_state_ensemble",
                            lambda **kwargs: trained.append(kwargs) or ([True], kwargs["current_training_id"]))
        task, extra = hippynn_interface.train_excited_state_HIPPYNN_ensemble_task.func, {"properties_list": {}}

    def train(**kwargs):
        result = task(ML_config={"n_models": 1}, h5_dir=str(tmp_path / "h5"),
                      model_path=str(tmp_path / "models" / "model-{:04d}"), current_training_id=2,
                      gpus_per_node=1, **extra, **kwargs)
        if isinstance(result, RemoteExceptionWrapper):  # Parsl's .func returns errors instead of raising.
            result.reraise()
        return result

    return train, trained


TRAINERS = pytest.mark.parametrize("kind", ["ground_state", "excited_state"])


@TRAINERS
def test_single_node_training_waits_for_every_stored_shard(kind, tmp_path, monkeypatch):
    h5 = tmp_path / "h5"
    h5.mkdir()
    for name in ("data-0000.h5", "data-0001.h5", "seed.h5"):  # Extra files are allowed.
        (h5 / name).touch()
    real_listdir, listings = os.listdir, []

    def stale_listdir(path):
        listings.append(path)
        names = real_listdir(path)
        return [name for name in names if name != "data-0001.h5"] if len(listings) < 3 else names

    monkeypatch.setattr(ensemble_training.os, "listdir", stale_listdir)
    monkeypatch.setattr(ensemble_training.wait_for_shards, "__defaults__", (30, 0.001))
    train, trained = single_node_trainer(kind, tmp_path, monkeypatch)
    train(h5_path=str(h5 / "data-{:04d}.h5"), current_h5_id=2)
    assert len(listings) >= 3 and len(trained) == 1


@TRAINERS
def test_single_node_training_stops_when_a_stored_shard_never_appears(kind, tmp_path, monkeypatch):
    (tmp_path / "h5").mkdir()
    (tmp_path / "h5" / "data-0000.h5").touch()
    monkeypatch.setattr(ensemble_training.wait_for_shards, "__defaults__", (0, 0.001))
    train, trained = single_node_trainer(kind, tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="data-0001.h5"):
        train(h5_path=str(tmp_path / "h5" / "data-{:04d}.h5"), current_h5_id=2)
    assert trained == []


@TRAINERS
def test_single_node_training_without_h5_path_skips_the_check(kind, tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble_training.wait_for_shards, "__defaults__", (0, 0.001))
    train, trained = single_node_trainer(kind, tmp_path, monkeypatch)
    train(current_h5_id=2)
    assert len(trained) == 1


@pytest.mark.parametrize("task", [hippynn_interface.train_HIPPYNN_ensemble_task,
                                  hippynn_interface.train_excited_state_HIPPYNN_ensemble_task],
                         ids=["ground_state", "excited_state"])
def test_coordinator_passes_single_node_training_its_shards(task):
    master = {"h5_dir": "h5", "h5_path": "h5/data-{:04d}.h5", "model_path": "models/model-{:04d}",
              "gpus_per_node": 1, "properties_list": {}}
    status = {"current_training_id": 4, "current_h5_id": 3}
    inputs = build_input_dict(task.func, [{"ML_config": {}}, master, status], raise_on_fail=True)
    assert inputs["h5_path"] == "h5/data-{:04d}.h5" and inputs["current_h5_id"] == 3


def test_hippynn_ensemble_members_load_in_name_order(monkeypatch):

    found_by_glob = ["models/model-0003/model-02/", "models/model-0003/model-00/", "models/model-0003/model-01/"]
    in_member_order = sorted(found_by_glob)
    calls = []

    def fake_calculator(model_dir, device="cuda:0"):
        calls.append((model_dir, device))
        return f"calculator:{model_dir}"

    monkeypatch.setattr(hippynn_interface.glob, "glob", lambda pattern: found_by_glob)
    monkeypatch.setattr(hippynn_interface, "HIPNN_ASE_calculator", fake_calculator)

    calculators = hippynn_interface.HIPNN_ASE_load_ensemble("models/model-0003", device="cpu")

    assert calculators == [f"calculator:{model_dir}" for model_dir in in_member_order]
    assert calls == [(model_dir, "cpu") for model_dir in in_member_order]


def test_neurochem_training_passes_settings_to_trainer(monkeypatch):
    neurochem_interface = import_neurochem_with_fake_backend(monkeypatch)
    captured = {}

    class FakeNeuroChemTrainer:
        def __init__(
            self,
            ensemble_size,
            gpuids,
            force_training=True,
            periodic=False,
            rmhighe=False,
            rmhighf=False,
            build_test=True,
            remove_existing=False,
        ):
            captured["trainer_kwargs"] = {
                "ensemble_size": ensemble_size,
                "gpuids": gpuids,
                "force_training": force_training,
                "periodic": periodic,
                "rmhighe": rmhighe,
                "rmhighf": rmhighf,
                "build_test": build_test,
                "remove_existing": remove_existing,
            }

        def train_models(self, configuration):
            captured["training_config"] = configuration.copy()
            return ["network"], [True, False]

    monkeypatch.setattr(neurochem_interface, "NeuroChemTrainer", FakeNeuroChemTrainer)
    monkeypatch.setattr(neurochem_interface.np.random, "randint", lambda high: 12345)

    completed, training_id = neurochem_interface.train_ANI_model_task.func(
        ML_config={
            "ensemble_size": 2,
            "force_training": False,
            "periodic": True,
            "rmhighe": 50.0,
            "build_test": False,
            "aev_params": {"elements": ["H", "O"]},
        },
        h5_dir="h5store",
        model_path="models/model-{:04d}",
        current_training_id=6,
        gpus_per_node=3,
    )

    assert completed == [True, False]
    assert training_id == 6
    assert captured["trainer_kwargs"] == {
        "ensemble_size": 2,
        "gpuids": [0, 1, 2],
        "force_training": False,
        "periodic": True,
        "rmhighe": 50.0,
        "rmhighf": False,
        "build_test": False,
        "remove_existing": False,
    }
    assert captured["training_config"]["ensemble_path"] == "models/model-0006"
    assert captured["training_config"]["data_store"] == "h5store"
    assert captured["training_config"]["seed"] == 12345
    assert captured["training_config"]["aev_params"] == {"elements": ["H", "O"]}


def test_neurochem_calculator_builds_ani_ensemble(monkeypatch):
    neurochem_interface = import_neurochem_with_fake_backend(monkeypatch)
    captured = {}

    def fake_batchedensemblemolecule(*args):
        captured["batched_args"] = args
        return "batched-model"

    def fake_aniens(model):
        captured["aniens_model"] = model
        return "aniens-calculator"

    monkeypatch.setattr(neurochem_interface.os, "listdir", lambda path: ["network.params", "notes.txt"])
    monkeypatch.setattr(neurochem_interface, "batchedensemblemolecule", fake_batchedensemblemolecule)
    monkeypatch.setattr(neurochem_interface, "ANIENS", fake_aniens)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")

    calculator = neurochem_interface.NeuroChemCalculator({"model_path": "/models/model-0006", "Nn": 8, "gpu": "2"})

    assert calculator == "aniens-calculator"
    assert captured["aniens_model"] == "batched-model"
    assert captured["batched_args"] == (
        "/models/model-0006/network.params",
        "/models/model-0006/sae_linfit.dat",
        "/models/model-0006",
        8,
        2,
    )
    assert neurochem_interface.os.environ["CUDA_VISIBLE_DEVICES"] == "3,5,7"
