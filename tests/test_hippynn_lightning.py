"""Training one ensemble member on several devices with PyTorch Lightning, and the checks on gpus_per_member.

The training tests run two CPU processes (gloo) on tiny water models for a few epochs. They check
that a member trained this way is written and reloaded like a single-device member, not its accuracy.
"""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from alframework.ml_interfaces import hippynn_interface as ml
from alframework.tools import ensemble_training

STATES = {"E0": ["E0", "system", 1.0], "F0": ["F0", "atomic", 1.0],
          "E1": ["E1", "system", 1.0], "F1": ["F1", "atomic", 1.0]}
NETWORK = dict(possible_species=[0, 1, 8], n_features=4, n_sensitivities=4, n_interaction_layers=1,
               n_atom_layers=1, dist_soft_min=0.4, dist_soft_max=2.0, dist_hard_max=3.0)


def _water(directory, frames=16, excited=False):
    rng = np.random.default_rng(3)
    coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])[None]
    coordinates = coordinates + rng.normal(0.0, 0.02, (frames, 3, 3))
    energies = (coordinates**2).sum(axis=(1, 2))
    directory.mkdir()
    with h5py.File(directory / "data.h5", "w") as handle:
        group = handle.create_group("water")
        group["species"] = np.array(["O", "H", "H"], dtype="S")
        group["coordinates"] = coordinates
        if excited:
            group["_id"] = np.array([f"frame-{i}" for i in range(frames)], dtype="S")
            group["topology_atom_ids"] = np.tile(np.arange(3), (frames, 1))
            for state in range(2):
                group[f"E{state}"] = energies + 0.1 * state
                group[f"F{state}"] = -2 * coordinates
        else:
            group["energy"] = energies
            group["forces"] = -2 * coordinates


def _check_member(member, devices=2, batch_size=4):
    log = (member / "training_log.txt").read_text()
    assert log.count("Training complete") == 1
    assert "Total epoch time" in log, "rank 0's epoch output belongs in the member's log"
    assert not (member / "lightning_rank0_log.txt").exists()
    record = json.loads((member / "lightning_training.json").read_text())
    assert record["devices"] == devices and record["epochs"] >= 1
    assert record["batch_size"] == batch_size and record["batch_size_per_device"] == batch_size // devices
    from hippynn.experiment.serialization import load_checkpoint

    bundle = load_checkpoint(str(member / "experiment_structure.pt"), str(member / "best_checkpoint.pt"),
                             map_location="cpu")
    assert bundle["controller"].batch_size == batch_size, "checkpoints store the configured total"


def test_excited_state_member_trains_on_two_devices(tmp_path):
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    _water(tmp_path / "data", excited=True)
    member = tmp_path / "ensemble" / "model-00"
    config = dict(n_models=1, cell_key=None, train_forces=True, network_choice=0, valid_size=0.25, test_size=0.25,
                  device_string="cpu", exports={"pdf": False, "png": False, "csv": False}, max_epochs=3,
                  batch_size=4, eval_batch_size=4, termination_patience=10, torch_threads=1,
                  fraction_train_eval=1.0, network_params=NETWORK, gpus_per_member=2)
    ml.train_single_excited_state_model(model_id=0, model_dir=str(member), h5_train_dir=str(tmp_path / "data"),
                                        properties_list=STATES, ML_config=config)
    _check_member(member)
    summary = json.loads((member / "training_summary.json").read_text())
    assert summary["gpus_per_member"] == 2 and summary["parameters_changed"]
    model = ml._load_member(member, "cpu")
    for key in ("E0", "F0", "E1", "F1"):
        ml._target(model, key)


def test_ground_state_member_trains_on_two_devices(tmp_path):
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    _water(tmp_path / "data")
    member = tmp_path / "ensemble" / "model-00"
    member.mkdir(parents=True)
    ml.train_HIPNN_model(str(member), str(tmp_path / "data"), energy_key="energy", coordinates_key="coordinates",
                         species_key="species", force_key="forces", network_params=dict(NETWORK),
                         valid_size=0.25, test_size=0.25, plot_every=1000,
                         controller_options=dict(batch_size=4, eval_batch_size=4, max_epochs=3,
                                                 termination_patience=10),
                         scheduler_options=dict(max_batch_size=8, patience=5, factor=0.5),
                         device_string="cpu", gpus_per_member=2)
    _check_member(member)
    assert ml._ground_state_member_complete(str(member))
    ml.HIPNN_ASE_calculator(str(member), energy_key="energy", device="cpu")


def test_batch_sizes_must_divide_across_devices():
    pytest.importorskip("pytorch_lightning")
    from alframework.ml_interfaces.hippynn_lightning import _split_batches

    controller = SimpleNamespace(batch_size=64, eval_batch_size=30, scheduler_list=[])
    with pytest.raises(ValueError, match="eval_batch_size 30"):
        _split_batches(controller, 4)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_gpus_per_member_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="gpus_per_member"):
        ensemble_training.validated_gpus_per_member(value)


def test_member_device_gives_a_multi_gpu_member_the_first_visible_gpu():
    assert ensemble_training.member_device("from_multiprocessing", 4, gpus_per_member=4) == "cuda:0"
    assert ensemble_training.member_device("cpu", 4, gpus_per_member=4) == "cpu"


def executor(workers, accelerators=None):
    pool = SimpleNamespace(workers_per_node=workers, provider=SimpleNamespace(nodes_per_block=1, max_blocks=4))
    if accelerators is not None:
        pool.available_accelerators = accelerators
    return pool


def test_check_capacity_counts_every_gpu_of_a_member():
    ensemble_training.check_capacity(4, 4, "from_multiprocessing", executor(1), gpus_per_member=4)
    with pytest.raises(ValueError, match="share a GPU"):
        ensemble_training.check_capacity(4, 4, "from_multiprocessing", executor(4), gpus_per_member=2)


def test_check_capacity_reads_each_workers_gpus_from_available_accelerators():
    ensemble_training.check_capacity(4, 4, "from_multiprocessing", executor(2, ["0,1", "2,3"]), gpus_per_member=2)
    with pytest.raises(ValueError, match="available_accelerators"):
        ensemble_training.check_capacity(4, 4, "from_multiprocessing", executor(4, ["0", "1", "2", "3"]),
                                         gpus_per_member=2)


def test_single_node_tasks_refuse_several_gpus_per_member():
    ensemble_training.single_gpu_members({"gpus_per_member": 1})
    with pytest.raises(ValueError, match="multi-node"):
        ensemble_training.single_gpu_members({"gpus_per_member": 4})
