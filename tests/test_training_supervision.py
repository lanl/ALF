"""Single-node ground-state training runs each member in a supervised child, one per device at a time."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from parsl.app.errors import RemoteExceptionWrapper

from alframework.ml_interfaces import hippynn_interface as ml
from alframework.tools import ensemble_training as training
from tests.helpers.fakes import training_member


def test_a_member_that_dies_does_not_hang_training_or_stop_later_members(tmp_path):
    code = """
import json, sys
from alframework.tools.ensemble_training import run_members_in_slots
from tests.helpers.fakes import training_member
members = [dict(action='exit'), dict(action='raise'), dict(output=sys.argv[1])]
print(json.dumps(run_members_in_slots(training_member, members, ['cpu'])), flush=True)
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    process = subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path / "done")], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = process.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("Training hung after a member died")
    assert process.returncode == 0, err
    assert json.loads(out.splitlines()[-1]) == [False, False, True]
    assert (tmp_path / "done").read_text() == "cpu"


def test_members_never_share_a_device_and_each_gets_the_held_out_directory(tmp_path, monkeypatch):
    active, seen, lock = set(), [], threading.Lock()

    def run(function, params):
        device = params["device_string"]
        with lock:
            assert device not in active, "two members on one device"
            active.add(device)
            seen.append(params)
        time.sleep(0.01)
        with lock:
            active.remove(device)
        return True

    monkeypatch.setattr(training, "run_member_in_child", run)
    members = ml._ground_state_member_params({"n_models": 5}, "train", str(tmp_path / "model-{}"), 0, 2, "heldout")
    assert training.run_members_in_slots(training_member, members, ["cuda:0", "cuda:1"]) == [True] * 5
    assert all(params["h5_test_dir"] == "heldout" for params in seen)
    assert {params["device_string"] for params in seen} == {"cuda:0", "cuda:1"}


def test_a_held_out_directory_in_ml_config_that_disagrees_with_the_task_s_is_refused():
    with pytest.raises(ValueError, match="Conflicting"):
        ml._ground_state_member_params({"n_models": 1, "h5_test_dir": "other"}, "train", "m{}", 0, 1, "heldout")


@pytest.mark.parametrize("n_models", [0, -1, 1.5, True, "2"])
def test_n_models_must_be_a_positive_integer(n_models):
    with pytest.raises(ValueError, match="n_models"):
        ml._ground_state_member_params({"n_models": n_models}, "train", "m{}", 0, 1)


def test_ensemble_training_refuses_to_remove_an_existing_model_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(ml.ensemble_training, "wait_for_shards", lambda shards: None)
    result = ml.train_HIPPYNN_ensemble_task.func({"n_models": 1}, str(tmp_path), str(tmp_path / "model-{}"), 0, 1,
                                                 remove_existing=True)
    with pytest.raises(ValueError, match="remove_existing"):
        result.reraise()


def test_training_devices_are_numbered_within_the_scheduler_s_mask(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,6")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    assert ml._single_node_training_devices({}, 8) == ["cuda:0", "cuda:1"]
    assert ml._single_node_training_devices({"device_string": "1,0"}, 8) == ["cuda:1", "cuda:0"]
    assert ml._single_node_training_devices({"device_string": "cpu"}, 0) == ["cpu"]
    assert os.environ["ROCR_VISIBLE_DEVICES"] == "4,6", "the mask is not rewritten"
    with pytest.raises(ValueError):
        ml._single_node_training_devices({"device_string": "0,0"}, 2)


def test_a_real_cpu_ensemble_evaluates_on_the_held_out_directory(tmp_path):
    pytest.importorskip("hippynn")
    from tests.helpers.hippynn import TINY_NETWORK, write_water_shard

    write_water_shard(tmp_path / "train")
    write_water_shard(tmp_path / "heldout", frames=8)
    config = dict(n_models=2, energy_key="energy", force_key="forces", species_key="species",
                  coordinates_key="coordinates", network_params=dict(TINY_NETWORK), device_string="cpu",
                  valid_size=0.25, test_size=0.25, plot_every=1000,
                  controller_options=dict(batch_size=4, eval_batch_size=4, max_epochs=1, termination_patience=5),
                  scheduler_options=dict(max_batch_size=8, patience=5, factor=0.5))
    result = ml.train_HIPPYNN_ensemble_task.func(config, str(tmp_path / "train"), str(tmp_path / "model-{}"), 0, 0,
                                                 h5_test_dir=str(tmp_path / "heldout"))
    if isinstance(result, RemoteExceptionWrapper):
        result.reraise()
    assert result == ([True, True], 0)
    for member in (tmp_path / "model-0").glob("model-*"):
        log = (member / "training_log.txt").read_text()
        assert str(tmp_path / "heldout") in log and "Training complete" in log
