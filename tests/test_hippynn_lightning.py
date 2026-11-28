"""Training one ensemble member on several devices with PyTorch Lightning, and the checks on gpus_per_member.

The training tests run two CPU processes (gloo) on tiny water models for a few epochs. They check
that a member trained this way is written and reloaded like a single-device member, not its accuracy.
"""

import json
from types import SimpleNamespace

import pytest

from alframework.ml_interfaces.hippynn import interface as ml
from alframework.tools import ensemble_training
from tests.helpers.hippynn import TINY_NETWORK, write_water_shard

STATES = {"E0": ["E0", "system", 1.0], "F0": ["F0", "atomic", 1.0],
          "E1": ["E1", "system", 1.0], "F1": ["F1", "atomic", 1.0]}


def _check_member(member, devices=2, batch_size=4):
    log = (member / "training_log.txt").read_text()
    assert log.count("Training complete") == 1
    assert "Total epoch time" in log, "rank 0's epoch output belongs in the member's log"
    assert not (member / "lightning_rank0_log.txt").exists()
    record = json.loads((member / "lightning_training.json").read_text())
    assert record["devices"] == devices and record["epochs"] >= 1
    assert record["batch_size"] == batch_size and record["batch_size_per_device"] == batch_size // devices
    assert isinstance(record["seed"], int) and f"Member seed: {record['seed']}" in log, \
        "a member's drawn seed is recorded, so a failed member can be reproduced"
    assert f"Testing model on {devices} devices, eval batch size {batch_size // devices} each" in log, \
        "the final test is split over the member's devices, each at its share of the evaluation batch"
    assert not list(member.glob("final_test_rank*.pt")), "the ranks' shares are removed after the final test"
    from alframework.ml_interfaces.hippynn.compat import checkpoints
    load_checkpoint = checkpoints().load_checkpoint

    bundle = load_checkpoint(str(member / "experiment_structure.pt"), str(member / "best_checkpoint.pt"),
                             map_location="cpu")
    assert bundle["controller"].batch_size == batch_size, "checkpoints store the configured total"


@pytest.mark.slow
def test_excited_state_member_trains_on_two_devices(tmp_path):
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    write_water_shard(tmp_path / "data", excited=True)
    member = tmp_path / "ensemble" / "model-00"
    config = dict(n_models=1, cell_key=None, train_forces=True, network_choice=0, valid_size=0.25, test_size=0.25,
                  device_string="cpu", exports={"pdf": False, "png": False, "csv": False}, max_epochs=3,
                  batch_size=4, eval_batch_size=4, termination_patience=10, torch_threads=1,
                  fraction_train_eval=1.0, network_params=TINY_NETWORK, gpus_per_member=2)
    ml.train_single_excited_state_model(model_id=0, model_dir=str(member), h5_train_dir=str(tmp_path / "data"),
                                        properties_list=STATES, ML_config=config)
    _check_member(member)
    summary = json.loads((member / "training_summary.json").read_text())
    assert summary["gpus_per_member"] == 2 and summary["parameters_changed"]
    model = ml._load_member(member, "cpu")
    for key in ("E0", "F0", "E1", "F1"):
        ml._target(model, key)


@pytest.mark.slow
def test_ground_state_member_trains_on_two_devices_inside_a_slurm_step(tmp_path, monkeypatch):
    """Under Parsl's SrunLauncher each worker runs inside a Slurm step. Lightning must still see ALF's ranks,
    not SLURM_PROCID 0 in every rank, which made all of them bind the same port (seen on an HPC system). ALF names
    Lightning's environment whether or not Slurm is there, so this one training covers both cases. It shares the
    split arrays as files the ranks map (shared_training_data), as a one-node member does without node-local
    copies."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    for key, value in dict(SLURM_NTASKS="1", SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NODEID="0",
                           SLURM_JOB_NAME="alf_ml_executor", SLURM_JOB_ID="1").items():
        monkeypatch.setenv(key, value)
    write_water_shard(tmp_path / "data")
    member = tmp_path / "ensemble" / "model-00"
    member.mkdir(parents=True)
    ml.train_HIPNN_model(str(member), str(tmp_path / "data"), energy_key="energy", coordinates_key="coordinates",
                         species_key="species", force_key="forces", network_params=dict(TINY_NETWORK),
                         valid_size=0.25, test_size=0.25, plot_every=1000,
                         controller_options=dict(batch_size=4, eval_batch_size=4, max_epochs=3,
                                                 termination_patience=10),
                         scheduler_options=dict(max_batch_size=8, patience=5, factor=0.5),
                         device_string="cpu", gpus_per_member=2, shared_training_data=True,
                         record_ntk_training_split=True)
    _check_member(member)
    assert "and shared arrays" in (member / "training_log.txt").read_text()
    assert not (member / "lightning_data").exists(), "the shared arrays are removed after training"
    assert ml._ground_state_member_complete(str(member))
    ml.HIPNN_ASE_calculator(str(member), energy_key="energy", device="cpu")
    _check_training_summary(member, epochs=3, gpus_per_member=2)
    manifest = json.loads((member / "ntk_training_split.json").read_text())
    summary = json.loads((member / "training_summary.json").read_text())
    assert manifest["training_structure_count"] == summary["split_counts"]["train"]


def _check_training_summary(member, epochs, gpus_per_member):
    """A ground-state member's training_summary.json: epoch times, metrics per epoch, and the final test."""
    summary = json.loads((member / "training_summary.json").read_text())
    assert summary["epochs"] == epochs and len(summary["epoch_seconds"]) == epochs
    assert len(summary["epoch_metric"]) == epochs and summary["average_epoch_time"] > 0
    assert summary["gpus_per_member"] == gpus_per_member
    assert {"train", "valid", "test"} <= set(summary["final_metric"])
    test = summary["final_metric"]["test"]
    assert "ForceRMSE" in test and test["ForceRMSE"] >= 0
    assert summary["split_counts"]["test"] > 0 and summary["metric"]["valid"]


@pytest.mark.slow
def test_a_single_device_ground_state_member_records_its_training(tmp_path):
    pytest.importorskip("hippynn")
    write_water_shard(tmp_path / "data")
    member = tmp_path / "model-00"
    member.mkdir()
    ml.train_HIPNN_model(str(member), str(tmp_path / "data"), energy_key="energy", coordinates_key="coordinates",
                         species_key="species", force_key="forces", network_params=dict(TINY_NETWORK),
                         valid_size=0.25, test_size=0.25, plot_every=1000,
                         controller_options=dict(batch_size=4, eval_batch_size=4, max_epochs=2,
                                                 termination_patience=10, fraction_train_eval=1.0),
                         scheduler_options=dict(max_batch_size=8, patience=5, factor=0.5),
                         device_string="cpu", gpus_per_member=1)
    _check_training_summary(member, epochs=2, gpus_per_member=1)


def test_batch_sizes_must_divide_across_devices():
    pytest.importorskip("pytorch_lightning")
    from alframework.ml_interfaces.hippynn.lightning import _split_batches

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


# The environment's own route is covered by test_training_options_come_from_ml_config_before_the_environment.
@pytest.mark.slow
@pytest.mark.parametrize('stream, setting', [(False, 'environment'), (True, 'ML_config')])
def test_the_split_final_test_reports_what_one_device_would(tmp_path, monkeypatch, stream, setting):
    """The ranks predict every n-th evaluation batch and the parent computes hippynn's metrics over all of them:
    the same numbers as hippynn's test_model on one device, for every split, after a real two-device training.
    Streaming is chosen by ``ALF_LIGHTNING_STREAM_FINAL``, or by ``stream_final_evaluation``, which wins."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    from hippynn.experiment import test_model
    from alframework.ml_interfaces.hippynn import lightning as hl

    seen = {}
    train = hl._train_ranks_and_finish

    def recording(modules, database, controller, tracker, settings, local_devices, launcher, plan, **options):
        seen.update(evaluator=modules.evaluator, database=database, n_devices=settings['n_devices'])
        assert (plan is not None) == stream
        result = train(modules, database, controller, tracker, settings, local_devices, launcher, plan, **options)
        seen['split'] = result.other_metric_values['FinalTraining']
        return result

    by_config = {} if setting == 'environment' else {'stream_final_evaluation': stream}
    monkeypatch.setenv('ALF_LIGHTNING_STREAM_FINAL', '1' if stream and not by_config else '0')
    monkeypatch.setattr(hl, '_train_ranks_and_finish', recording)
    write_water_shard(tmp_path / "data", frames=25)
    member = tmp_path / "model-00"
    member.mkdir()
    ml.train_HIPNN_model(str(member), str(tmp_path / "data"), energy_key="energy", coordinates_key="coordinates",
                         species_key="species", force_key="forces", network_params=dict(TINY_NETWORK),
                         valid_size=0.25, test_size=0.25, plot_every=1000,
                         controller_options=dict(batch_size=4, eval_batch_size=4, max_epochs=2,
                                                 termination_patience=10),
                         scheduler_options=dict(max_batch_size=8, patience=5, factor=0.5),
                         device_string="cpu", gpus_per_member=2, **by_config)
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
    assert not list(member.glob('final-evaluation-*'))


def test_streamed_histogram_has_the_whole_split_edges_and_counts(tmp_path):
    pytest.importorskip('hippynn')
    import pickle

    import numpy as np
    from alframework.ml_interfaces.hippynn.final_evaluation import _edges, _extend_ranges, _histogram_counts

    x = np.array([-20, 4, 2, 7, 20, 4, 2, 4, -12], dtype=np.float32)
    y = np.array([3, 1, 2, 7, 18, -3, 2, 0, -17], dtype=np.float32)
    for predicted in (y, np.full_like(y, 1.5)):
        expected, x_edges, y_edges = np.histogram2d(x, predicted, bins=10)
        with (tmp_path / 'plot.pkl').open('w+b') as spool:
            ranks = []
            for rows in (slice(0, 6, 2), slice(1, 6, 2), slice(6, None)):
                parts = [(x[rows], predicted[rows])]
                ranges = [None]
                _extend_ranges(ranges, parts)
                ranks.append(ranges[0])
                pickle.dump(('test', parts), spool)
            edges = _edges(SimpleNamespace(bins=10), ranks)
            np.testing.assert_array_equal(edges[0], x_edges)
            np.testing.assert_array_equal(edges[1], y_edges)
            counts, = _histogram_counts(spool, {'test': [edges]})['test']
            np.testing.assert_array_equal(counts, expected)


NODE_LAUNCHER = str(__import__("pathlib").Path(__file__).parent / "helpers" / "node_launcher.py")


def _train_on_two_nodes(tmp_path, launcher_options=()):
    """A ground-state member on two stand-in nodes of two CPU devices each, started as srun would start them."""
    import sys
    write_water_shard(tmp_path / "data", frames=32)
    member = tmp_path / "ensemble" / "model-00"
    member.mkdir(parents=True)
    ml.train_HIPNN_model(str(member), str(tmp_path / "data"), energy_key="energy", coordinates_key="coordinates",
                         species_key="species", force_key="forces", network_params=dict(TINY_NETWORK),
                         valid_size=0.25, test_size=0.25, plot_every=1000,
                         controller_options=dict(batch_size=8, eval_batch_size=8, max_epochs=2,
                                                 termination_patience=10),
                         scheduler_options=dict(max_batch_size=16, patience=5, factor=0.5),
                         device_string="cpu", gpus_per_member=4, member_nodes=2,
                         member_launcher=[sys.executable, NODE_LAUNCHER, "2", *launcher_options, "--"])
    return member


@pytest.mark.slow
def test_a_member_trains_across_two_nodes_on_shared_node_local_arrays(tmp_path, monkeypatch):
    """Two node processes, each starting its own two ranks, train one model as a world of four, and the member
    is written like any other. With ALF_LIGHTNING_STATS every rank records its node, steps and peak memory. With
    ALF_LIGHTNING_SHARED_DATA the split arrays go to files that every node copies to its local directory and every
    rank maps; streaming evaluation removes its temporary reports as well. One training checks all three, since
    each two-node training starts six processes that import PyTorch and HIPPYNN."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    local = tmp_path / "node-local"
    local.mkdir()
    monkeypatch.setenv("ALF_LIGHTNING_STATS", "1")
    monkeypatch.setenv("ALF_LIGHTNING_SHARED_DATA", "1")
    monkeypatch.setenv("ALF_LIGHTNING_STREAM_FINAL", "1")
    monkeypatch.setenv("ALF_NODE_LOCAL_DIR", str(local))
    member = _train_on_two_nodes(tmp_path)
    _check_member(member, devices=4, batch_size=8)
    record = json.loads((member / "lightning_training.json").read_text())
    assert record["nodes"] == 2 and record["final_evaluation"] == "streaming"
    assert not (member / "lightning_data").exists()
    assert not list(member.glob("final-evaluation-*"))
    assert list(local.iterdir()) == []
    assert "on 2 nodes, 2 each" in (member / "training_log.txt").read_text()
    assert not (member / "lightning_rendezvous.json").exists()
    stats = [json.loads(path.read_text()) for path in sorted((member / "lightning_stats").glob("rank-*.json"))]
    assert [(entry["rank"], entry["node"], entry["local_rank"]) for entry in stats] == \
        [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]
    for entry in stats:
        assert entry["world"] == 4 and entry["step_count"] >= 2 and entry["peak_rss_gib"] > 0
        assert len(entry["epoch_seconds"]) == record["epochs"]
    ml.HIPNN_ASE_calculator(str(member), energy_key="energy", device="cpu")


@pytest.mark.slow
def test_a_failed_node_fails_the_member(tmp_path):
    """When one node's process fails, the launcher stops the others, and the member reports the failure
    rather than waiting for the missing node's ranks."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    with pytest.raises(RuntimeError, match="on 2 nodes failed"):
        _train_on_two_nodes(tmp_path, ["--fail-node", "1"])
    assert not (tmp_path / "ensemble" / "model-00" / "lightning_rendezvous.json").exists()


@pytest.mark.slow
def test_a_non_finite_training_loss_stops_the_member_at_the_end_of_the_epoch(tmp_path, capfd):
    """A NaN force label makes a batch's loss NaN, and the gradient all-reduce then spreads it to every rank's
    weights. All ranks stop together after that epoch, rather than training to NaN for every remaining epoch and
    failing in the final evaluation, and each rank that saw it names its host and batch. Every frame carries one
    NaN component, so the first training batch is non-finite whichever frames the random split puts in training."""
    pytest.importorskip("hippynn")
    pytest.importorskip("pytorch_lightning")
    import h5py
    import numpy as np
    write_water_shard(tmp_path / "data")
    with h5py.File(tmp_path / "data" / "data.h5", "r+") as handle:
        forces = handle["water/forces"][()]
        forces[:, 0, 0] = np.nan
        handle["water/forces"][...] = forces
    member = tmp_path / "ensemble" / "model-00"
    member.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        ml.train_HIPNN_model(str(member), str(tmp_path / "data"), energy_key="energy", coordinates_key="coordinates",
                             species_key="species", force_key="forces", network_params=dict(TINY_NETWORK),
                             valid_size=0.25, test_size=0.25, plot_every=1000,
                             controller_options=dict(batch_size=4, eval_batch_size=4, max_epochs=20,
                                                     termination_patience=30),
                             scheduler_options=dict(max_batch_size=8, patience=5, factor=0.5),
                             device_string="cpu", gpus_per_member=2)
    output = capfd.readouterr()
    text = output.out + output.err + (member / "training_log.txt").read_text()
    assert "not finite in epoch 0" in text
    assert "non-finite training loss first at batch" in text
    assert text.count("Total epoch time") == 1, "the member stops after the epoch that went non-finite"


def test_non_finite_final_evaluation_predictions_are_named():
    pytest.importorskip("hippynn")
    from types import SimpleNamespace as Plot
    import numpy as np
    from alframework.ml_interfaces.hippynn.final_evaluation import _edges
    with pytest.raises(ValueError, match="predictions are not finite"):
        _edges(Plot(bins=10), [(0.0, 1.0, np.nan, np.nan), (0.0, 2.0, 0.5, 1.5)])


def test_node_rank_comes_from_the_first_variable_set():
    from alframework.ml_interfaces.hippynn.lightning import node_rank

    assert node_rank({"SLURM_PROCID": "3"}) == 3
    assert node_rank({"ALF_NODE_RANK": "1", "SLURM_PROCID": "3"}) == 1
    assert node_rank({"ALF_NODE_RANK": "", "PMI_RANK": "2"}) == 2
    with pytest.raises(RuntimeError, match="ALF_NODE_RANK"):
        node_rank({})


def test_a_member_on_several_nodes_needs_a_launcher_and_splits_its_gpus():
    pytest.importorskip("pytorch_lightning")
    from alframework.ml_interfaces.hippynn.lightning import _local_devices

    assert _local_devices(16, 2, ["srun"]) == 8
    assert _local_devices(8, 1, None) == 8
    with pytest.raises(ValueError, match="member_launcher"):
        _local_devices(16, 2, None)
    with pytest.raises(ValueError, match="list of words"):
        _local_devices(16, 2, "srun -N2")


def test_training_and_the_member_launcher_check_member_nodes_as_the_startup_check_does(tmp_path):
    """One rule (ensemble_training.validated_member_nodes) for the startup check, the trainer and the launcher."""
    with pytest.raises(ValueError, match="gpus_per_member 1 is a total over member_nodes 2"):
        ml.train_HIPNN_model(str(tmp_path), str(tmp_path), energy_key="energy", coordinates_key="coordinates",
                             species_key="species", device_string="cpu", member_nodes=2)
    pytest.importorskip("pytorch_lightning")
    from alframework.ml_interfaces.hippynn.lightning import _local_devices
    with pytest.raises(ValueError, match="divide evenly"):
        _local_devices(12, 8, ["srun"])


def test_shared_split_arrays_come_back_unchanged(tmp_path, monkeypatch):
    """The ranks map exactly the arrays the parent wrote: same values, dtypes and shapes, for every split."""
    pytest.importorskip("pytorch_lightning")
    import torch
    from alframework.ml_interfaces.hippynn import lightning as hl

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


@pytest.mark.parametrize("value, variable, expected", [
    (None, None, False), (None, "1", True), (None, "0", False), (True, None, True), (True, "0", True),
    (False, "1", False)])
def test_training_options_come_from_ml_config_before_the_environment(value, variable, expected):
    from alframework.ml_interfaces.hippynn import lightning as hl
    shared = {} if variable is None else {hl.SHARED_DATA_VARIABLE: variable}
    stream = {} if variable is None else {hl.STREAM_FINAL_VARIABLE: variable}
    assert hl.shared_data_enabled(value, shared) is expected
    assert hl.stream_final_enabled(value, stream) is expected


@pytest.mark.parametrize("name", ["shared_training_data", "stream_final_evaluation"])
def test_a_training_option_must_be_true_false_or_null(name):
    with pytest.raises(ValueError, match=f"{name} must be true, false or null"):
        ml.train_HIPNN_model("model", "data", energy_key="energy", coordinates_key="coordinates",
                             species_key="species", **{name: "yes"})


@pytest.mark.parametrize("total, batch_size, n_devices", [(10, 3, 2), (10, 3, 3), (9, 3, 4), (1, 4, 2)])
def test_each_rank_gathers_only_its_own_final_test_batches(total, batch_size, n_devices):
    """Together the ranks' batches are exactly the unshuffled loader's, with the same numbers."""
    pytest.importorskip("pytorch_lightning")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from alframework.ml_interfaces.hippynn.lightning import _owned_batches

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
