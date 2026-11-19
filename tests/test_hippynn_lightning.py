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
    assert f"Testing model on {devices} devices, eval batch size {batch_size // devices} each" in log, \
        "the final test is split over the member's devices, each at its share of the evaluation batch"
    assert not list(member.glob("final_test_rank*.pt")), "the ranks' shares are removed after the final test"
    from alframework.ml_interfaces._hippynn_compat import checkpoints
    load_checkpoint = checkpoints().load_checkpoint

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


def test_ground_state_member_trains_on_two_devices_inside_a_slurm_step(tmp_path, monkeypatch):
    """Under Parsl's SrunLauncher each worker runs inside a Slurm step. Lightning must still see ALF's ranks,
    not SLURM_PROCID 0 in every rank, which made all of them bind the same port (seen on Frontier). ALF names
    Lightning's environment whether or not Slurm is there, so this one training covers both cases."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    for key, value in dict(SLURM_NTASKS="1", SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NODEID="0",
                           SLURM_JOB_NAME="alf_ml_executor", SLURM_JOB_ID="1").items():
        monkeypatch.setenv(key, value)
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


def test_the_split_final_test_reports_what_one_device_would(tmp_path, monkeypatch):
    """The ranks predict every n-th evaluation batch and the parent computes hippynn's metrics over all of them:
    the same numbers as hippynn's test_model on one device, for every split, after a real two-device training."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    from hippynn.experiment import test_model
    from alframework.ml_interfaces import hippynn_lightning as hl

    seen = {}
    final_test = hl._final_test

    def recording(evaluator, database, n_devices, metric_tracker, when="FinalTraining"):
        seen.update(evaluator=evaluator, database=database, n_devices=n_devices)
        seen["split"] = final_test(evaluator, database, n_devices, metric_tracker, when)
        return seen["split"]

    monkeypatch.setattr(hl, "_final_test", recording)
    _water(tmp_path / "data", frames=24)
    member = tmp_path / "model-00"
    member.mkdir()
    ml.train_HIPNN_model(str(member), str(tmp_path / "data"), energy_key="energy", coordinates_key="coordinates",
                         species_key="species", force_key="forces", network_params=dict(NETWORK),
                         valid_size=0.25, test_size=0.25, plot_every=1000,
                         controller_options=dict(batch_size=4, eval_batch_size=4, max_epochs=2,
                                                 termination_patience=10),
                         scheduler_options=dict(max_batch_size=8, patience=5, factor=0.5),
                         device_string="cpu", gpus_per_member=2)
    assert seen["n_devices"] == 2
    evaluator = seen["evaluator"]
    evaluator.plot_maker = None
    reference = {}
    tracker = SimpleNamespace(register_metrics=lambda metrics, when: reference.update(metrics),
                              evaluation_print=lambda *args, **kwargs: None)
    test_model(seen["database"], evaluator, batch_size=2, when="check", metric_tracker=tracker)
    assert set(seen["split"]) == set(reference) == {"train", "valid", "test"}
    for name in reference:
        for key, value in reference[name].items():
            assert seen["split"][name][key] == pytest.approx(value, rel=1e-5, abs=1e-8), (name, key)


NODE_LAUNCHER = str(__import__("pathlib").Path(__file__).parent / "helpers" / "node_launcher.py")


def _train_on_two_nodes(tmp_path, launcher_options=()):
    """A ground-state member on two stand-in nodes of two CPU devices each, started as srun would start them."""
    import sys
    _water(tmp_path / "data", frames=32)
    member = tmp_path / "ensemble" / "model-00"
    member.mkdir(parents=True)
    ml.train_HIPNN_model(str(member), str(tmp_path / "data"), energy_key="energy", coordinates_key="coordinates",
                         species_key="species", force_key="forces", network_params=dict(NETWORK),
                         valid_size=0.25, test_size=0.25, plot_every=1000,
                         controller_options=dict(batch_size=8, eval_batch_size=8, max_epochs=2,
                                                 termination_patience=10),
                         scheduler_options=dict(max_batch_size=16, patience=5, factor=0.5),
                         device_string="cpu", gpus_per_member=4, member_nodes=2,
                         member_launcher=[sys.executable, NODE_LAUNCHER, "2", *launcher_options, "--"])
    return member


def test_a_member_trains_across_two_nodes(tmp_path, monkeypatch):
    """Two node processes, each starting its own two ranks, train one model as a world of four, and the member
    is written like any other. With ALF_LIGHTNING_STATS every rank records its node, steps and peak memory."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    monkeypatch.setenv("ALF_LIGHTNING_STATS", "1")
    member = _train_on_two_nodes(tmp_path)
    _check_member(member, devices=4, batch_size=8)
    record = json.loads((member / "lightning_training.json").read_text())
    assert record["nodes"] == 2
    assert "on 2 nodes, 2 each" in (member / "training_log.txt").read_text()
    assert not (member / "lightning_rendezvous.json").exists()
    stats = [json.loads(path.read_text()) for path in sorted((member / "lightning_stats").glob("rank-*.json"))]
    assert [(entry["rank"], entry["node"], entry["local_rank"]) for entry in stats] == \
        [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]
    for entry in stats:
        assert entry["world"] == 4 and entry["step_count"] >= 2 and entry["peak_rss_gib"] > 0
        assert len(entry["epoch_seconds"]) == record["epochs"]
    ml.HIPNN_ASE_calculator(str(member), energy_key="energy", device="cpu")


def test_a_failed_node_fails_the_member(tmp_path):
    """When one node's process fails, the launcher stops the others, and the member reports the failure
    rather than waiting for the missing node's ranks."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    with pytest.raises(RuntimeError, match="on 2 nodes failed"):
        _train_on_two_nodes(tmp_path, ["--fail-node", "1"])
    assert not (tmp_path / "ensemble" / "model-00" / "lightning_rendezvous.json").exists()


def test_node_rank_comes_from_the_first_variable_set():
    from alframework.ml_interfaces.hippynn_lightning import node_rank

    assert node_rank({"SLURM_PROCID": "3"}) == 3
    assert node_rank({"ALF_NODE_RANK": "1", "SLURM_PROCID": "3"}) == 1
    assert node_rank({"ALF_NODE_RANK": "", "PMI_RANK": "2"}) == 2
    with pytest.raises(RuntimeError, match="ALF_NODE_RANK"):
        node_rank({})


def test_member_nodes_must_share_the_gpus_and_have_a_launcher():
    pytest.importorskip("pytorch_lightning")
    from alframework.ml_interfaces.hippynn_lightning import _local_devices

    assert _local_devices(16, 2, ["srun"]) == 8
    assert _local_devices(8, 1, None) == 8
    with pytest.raises(ValueError, match="divide evenly"):
        _local_devices(12, 8, ["srun"])
    with pytest.raises(ValueError, match="member_launcher"):
        _local_devices(16, 2, None)
    with pytest.raises(ValueError, match="list of words"):
        _local_devices(16, 2, "srun -N2")
    with pytest.raises(ValueError, match="member_nodes"):
        _local_devices(16, 0, ["srun"])


def test_member_nodes_need_several_gpus_per_member(tmp_path):
    with pytest.raises(ValueError, match="member_nodes is 2, but gpus_per_member is 1"):
        ml.train_HIPNN_model(str(tmp_path), str(tmp_path), energy_key="energy", coordinates_key="coordinates",
                             species_key="species", device_string="cpu", member_nodes=2)


def test_shared_split_arrays_come_back_unchanged(tmp_path, monkeypatch):
    """The ranks map exactly the arrays the parent wrote: same values, dtypes and shapes, for every split."""
    pytest.importorskip("pytorch_lightning")
    import torch
    from alframework.ml_interfaces import hippynn_lightning as hl

    monkeypatch.chdir(tmp_path)
    splits = {"train": {"coordinates": torch.randn(5, 3, 3), "species": torch.tensor([[1, 8, 1]] * 5),
                        "mask": torch.tensor([True, False, True, True, False])},
              "valid": {"coordinates": torch.randn(2, 3, 3), "species": torch.tensor([[1, 8, 1]] * 2),
                        "mask": torch.tensor([True, True])}}
    database = SimpleNamespace(splits={name: dict(split) for name, split in splits.items()})
    returned = hl._share_splits(database)
    assert database.splits == {} and returned.keys() == splits.keys()
    reader = SimpleNamespace()
    hl._attach_splits(reader, tmp_path / hl.SHARED)
    for name, split in splits.items():
        for key, tensor in split.items():
            mapped = reader.splits[name][key]
            assert mapped.dtype == tensor.dtype and torch.equal(mapped, tensor), (name, key)


def test_a_two_node_member_trains_on_shared_node_local_arrays(tmp_path, monkeypatch):
    """With ALF_LIGHTNING_SHARED_DATA the split arrays go to files that every node copies to its local
    directory and every rank maps; the member is written as usual and both copies are removed afterwards."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    local = tmp_path / "node-local"
    local.mkdir()
    monkeypatch.setenv("ALF_LIGHTNING_SHARED_DATA", "1")
    monkeypatch.setenv("ALF_NODE_LOCAL_DIR", str(local))
    member = _train_on_two_nodes(tmp_path)
    _check_member(member, devices=4, batch_size=8)
    assert not (member / "lightning_data").exists()
    assert list(local.iterdir()) == []
    ml.HIPNN_ASE_calculator(str(member), energy_key="energy", device="cpu")


@pytest.mark.parametrize("total, batch_size, n_devices", [(10, 3, 2), (10, 3, 3), (9, 3, 4), (1, 4, 2)])
def test_each_rank_gathers_only_its_own_final_test_batches(total, batch_size, n_devices):
    """Together the ranks' batches are exactly the unshuffled loader's, with the same numbers."""
    pytest.importorskip("pytorch_lightning")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from alframework.ml_interfaces.hippynn_lightning import _owned_batches

    loader = DataLoader(TensorDataset(torch.arange(total), torch.arange(total) * 10), batch_size=batch_size)
    expected = list(enumerate(loader))
    found = sorted((pair for rank in range(n_devices) for pair in _owned_batches(loader, batch_size, rank, n_devices)),
                   key=lambda pair: pair[0])
    assert [number for number, _ in found] == [number for number, _ in expected]
    for (_, got), (_, want) in zip(found, expected):
        assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_check_capacity_counts_a_multi_node_member_by_its_gpus_on_each_node(capsys):
    """16 GPUs on 2 nodes is 8 per node: one worker per node fits 8-GPU nodes, two would share GPUs."""
    ensemble_training.check_capacity(2, 8, "from_multiprocessing", executor(1), gpus_per_member=16, member_nodes=2)
    assert "Warning" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="share a GPU"):
        ensemble_training.check_capacity(2, 8, "from_multiprocessing", executor(2), gpus_per_member=16,
                                         member_nodes=2)


def test_check_capacity_warns_when_member_node_groups_are_fewer_than_members(capsys):
    # executor(1): 4 blocks of 1 node, so 2 groups of 2 nodes for 4 members.
    ensemble_training.check_capacity(4, 8, "from_multiprocessing", executor(1), gpus_per_member=16, member_nodes=2)
    assert "2 slots for 4 ensemble members" in capsys.readouterr().out


@pytest.mark.parametrize("member_nodes, gpus_per_member", [(0, 8), (True, 8), (2, 1), (3, 16), (1.5, 8)])
def test_member_nodes_must_share_gpus_per_member(member_nodes, gpus_per_member):
    with pytest.raises(ValueError, match="member_nodes"):
        ensemble_training.validated_member_nodes(member_nodes, gpus_per_member)
