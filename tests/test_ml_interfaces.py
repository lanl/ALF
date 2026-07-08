import importlib
import sys
import types


def test_hippynn_train_ensemble_task_builds_model_configs(tmp_path, monkeypatch):
    from alframework.ml_interfaces import hippynn_interface

    captured = {}

    class FakePool:
        def __init__(self, processes):
            captured["processes"] = processes
            captured["closed"] = False

        def map(self, func, params_list):
            captured["func"] = func
            captured["params_list"] = params_list
            for params in params_list:
                model_dir = tmp_path / params["model_dir"]
                model_dir.mkdir(parents=True)
                (model_dir / "training_log.txt").write_text("Training complete\n")
            return [None for _ in params_list]

        def close(self):
            captured["closed"] = True

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
    assert captured["processes"] == 2
    assert captured["closed"]
    assert captured["func"] is hippynn_interface.train_HIPNN_model_wrapper
    assert captured["params_list"] == [
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


def test_hippynn_train_ensemble_task_reports_partial_failure(tmp_path, monkeypatch):
    from alframework.ml_interfaces import hippynn_interface

    class FakePool:
        def __init__(self, processes):
            pass

        def map(self, func, params_list):
            log_texts = ["Training complete\n", "Stopped before convergence\n"]
            for params, log_text in zip(params_list, log_texts):
                model_dir = tmp_path / params["model_dir"]
                model_dir.mkdir(parents=True)
                (model_dir / "training_log.txt").write_text(log_text)
            return [None for _ in params_list]

        def close(self):
            pass

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


def test_hippynn_ase_load_ensemble_returns_calculators_for_model_dirs(monkeypatch):
    from alframework.ml_interfaces import hippynn_interface

    model_dirs = ["models/model-0003/model-00/", "models/model-0003/model-01/"]
    calls = []

    def fake_calculator(model_dir, device="cuda:0"):
        calls.append((model_dir, device))
        return f"calculator:{model_dir}"

    monkeypatch.setattr(hippynn_interface.glob, "glob", lambda pattern: model_dirs)
    monkeypatch.setattr(hippynn_interface, "HIPNN_ASE_calculator", fake_calculator)

    calculators = hippynn_interface.HIPNN_ASE_load_ensemble("models/model-0003", device="cpu")

    assert calculators == [f"calculator:{model_dir}" for model_dir in model_dirs]
    assert calls == [(model_dir, "cpu") for model_dir in model_dirs]


def import_neurochem_interface(monkeypatch):
    fake_anitraintools = types.ModuleType("anitraintools")
    fake_ase_interface = types.ModuleType("ase_interface")
    fake_ase_interface.aniensloader = object()
    fake_ase_interface.ANIENS = lambda model: ("ANIENS", model)
    fake_ase_interface.batchedensemblemolecule = lambda *args: ("batchedensemblemolecule", args)

    monkeypatch.setitem(sys.modules, "anitraintools", fake_anitraintools)
    monkeypatch.setitem(sys.modules, "ase_interface", fake_ase_interface)
    monkeypatch.delitem(sys.modules, "alframework.ml_interfaces.neurochem_interface", raising=False)

    return importlib.import_module("alframework.ml_interfaces.neurochem_interface")


def test_neurochem_train_task_wires_trainer_and_training_config(monkeypatch):
    neurochem_interface = import_neurochem_interface(monkeypatch)
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
    neurochem_interface = import_neurochem_interface(monkeypatch)
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
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    calculator = neurochem_interface.NeuroChemCalculator(
        {"model_path": "/models/model-0006", "Nn": 8, "gpu": "2"}
    )

    assert calculator == "aniens-calculator"
    assert captured["aniens_model"] == "batched-model"
    assert captured["batched_args"] == (
        "/models/model-0006/network.params",
        "/models/model-0006/sae_linfit.dat",
        "/models/model-0006",
        8,
        0,
    )
    assert neurochem_interface.os.environ["CUDA_VISIBLE_DEVICES"] == "2"
