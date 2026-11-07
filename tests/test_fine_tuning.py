"""Fine-tuning existing models in ALF's loop: the general task, the MACE adapter, and MACE sampling.

MACE isn't installed in the test environment, so MACE's own training is checked by the command
ALF builds, and the general task runs with ``tests.helpers.fake_fine_tuner``.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alframework.ml_interfaces import fine_tuning
from alframework.ml_interfaces.fine_tuning import (child_environment, fine_tune_ensemble, fine_tune_settings,
                                                   validate_fine_tune_config)
from alframework.ml_interfaces.mace_interface import MaceFineTuner, shard_frames
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import store_current_data
from tests.helpers.fake_fine_tuner import trained
from tests.helpers.molecules import WATER

PROPERTIES = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}
FAKE = "tests.helpers.fake_fine_tuner.FakeFineTuner"
MACE = "alframework.ml_interfaces.mace_interface.MaceFineTuner"


def ml_config(foundation, **settings):
    return {"fine_tune": {**dict(adapter=FAKE, foundation_model=str(foundation), n_models=3, seed=10, device="cpu"),
                          **settings}}


def labeled(name, atoms=None, energy=-1.0):
    atoms = (atoms if atoms is not None else WATER).copy()
    molecule = MoleculesObject(atoms, name)
    forces = np.arange(3 * len(atoms), dtype=float).reshape(-1, 3) / 10
    molecule.store_results({"energy": energy, "forces": forces})
    molecule.set_converged_flag(True)
    return molecule


@pytest.fixture
def run(tmp_path):
    (tmp_path / "foundation.model").write_bytes(b"pretrained weights")
    (tmp_path / "h5").mkdir()
    store_current_data(str(tmp_path / "h5" / "data-0000.h5"), [labeled("a"), labeled("b", energy=-2.0)], PROPERTIES)
    return tmp_path


def fine_tune(run, config, training_id=0, **kwargs):
    return fine_tune_ensemble(config, str(run / "h5"), str(run / "models" / "model-{:04d}"), training_id,
                              master_directory=str(run), **kwargs)


# --- Settings -------------------------------------------------------------------------------------

def test_settings_resolve_local_files_against_the_master_directory(tmp_path):
    settings = fine_tune_settings({"fine_tune": dict(adapter=FAKE, foundation_model="models/base.model",
                                                     replay_file="replay/mp.xyz")}, str(tmp_path))
    assert settings["foundation_model"] == str(tmp_path / "models/base.model")
    assert settings["replay_file"] == str(tmp_path / "replay/mp.xyz")
    assert (settings["n_models"], settings["seed"], settings["device"]) == (4, 0, "cuda")


@pytest.mark.parametrize("block, message", [
    (None, "needs a 'fine_tune' block"),
    (dict(foundation_model="x"), "fine_tune.adapter"),
    (dict(adapter=FAKE), "fine_tune.foundation_model"),
    (dict(adapter=FAKE, foundation_model="x", n_models=1), "at least 2"),
    (dict(adapter=FAKE, foundation_model="x", n_models=True), "at least 2"),
    (dict(adapter=FAKE, foundation_model="x", seed=-1), "seed"),
    (dict(adapter=FAKE, foundation_model="x", device="cuda:1"), "device"),
], ids=["missing", "adapter", "foundation", "one_copy", "boolean", "seed", "device"])
def test_settings_refuse_mistakes(block, message):
    with pytest.raises(ValueError, match=message):
        fine_tune_settings({} if block is None else {"fine_tune": block})


def test_startup_check_needs_the_foundation_model_and_asks_the_adapter(run):
    validate_fine_tune_config(ml_config(run / "foundation.model"), {"master_directory": str(run)})
    with pytest.raises(ValueError, match="Foundation model not found"):
        validate_fine_tune_config(ml_config(run / "missing.model"), {})
    with pytest.raises(ValueError, match="fake adapter rejects"):
        validate_fine_tune_config(ml_config(run / "foundation.model"), {"reject_fine_tuning": True})


# --- The general task -------------------------------------------------------------------------------

def test_each_round_fine_tunes_every_copy_from_the_original_checkpoint(run):
    config = ml_config(run / "foundation.model")
    assert fine_tune(run, config, 0) == ([True, True, True], 0)
    store_current_data(str(run / "h5" / "data-0001.h5"), [labeled("c")], PROPERTIES)
    assert fine_tune(run, config, 1) == ([True, True, True], 1)

    for round_, shards in ((0, ["data-0000.h5"]), (1, ["data-0000.h5", "data-0001.h5"])):
        members = [trained(run / "models" / f"model-{round_:04d}" / f"model-{member:02d}") for member in range(3)]
        assert {member["source"] for member in members} == {str(run / "foundation.model")}, \
            "every round starts from the foundation model, never from the previous round"
        assert [member["seed"] for member in members] == [10, 11, 12]
        assert [member["member"] for member in members] == [0, 1, 2]
        assert all(member["data"] == shards and member["device"] == "cpu" for member in members)


def test_a_round_records_its_lineage_and_label_sources(run):
    config = ml_config(run / "foundation.model", failing_seeds=[11])
    assert fine_tune(run, config)[0] == [True, False, True], "one failed copy doesn't stop the others"
    lineage = json.loads((run / "models" / "model-0000" / fine_tuning.LINEAGE).read_text())
    import hashlib
    assert lineage["foundation_model"] == str(run / "foundation.model")
    assert lineage["foundation_model_sha256"] == hashlib.sha256(b"pretrained weights").hexdigest()
    assert (lineage["adapter"], lineage["package"], lineage["package_version"]) == (FAKE, "fake-tuner", "1.0")
    assert lineage["shards"] == ["data-0000.h5"] and lineage["seeds"] == [10, 11, 12]
    assert lineage["completed"] == [True, False, True]
    assert lineage["settings"]["failing_seeds"] == [11]
    assert json.loads((run / "models" / "model-0000" / "label_sources.json").read_text())["shards"] == 1


def test_a_round_waits_for_every_stored_shard(run, monkeypatch):
    monkeypatch.setattr(fine_tuning.ensemble_training.wait_for_shards, "__defaults__", (0.05, 0.01))
    with pytest.raises(FileNotFoundError, match="data-0001.h5"):
        fine_tune(run, ml_config(run / "foundation.model"), h5_path=str(run / "h5" / "data-{:04d}.h5"),
                  current_h5_id=2)
    assert not (run / "models").exists()


def test_a_round_refuses_a_nonempty_directory(run):
    (run / "models" / "model-0000").mkdir(parents=True)
    (run / "models" / "model-0000" / "left.txt").write_text("from a crashed round")
    with pytest.raises(FileExistsError, match="nonempty ensemble directory"):
        fine_tune(run, ml_config(run / "foundation.model"))


def test_copies_share_the_gpus_one_at_a_time(run, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    fine_tune(run, ml_config(run / "foundation.model", device="cuda", train_seconds=0.3), gpus_per_node=4)
    members = [trained(run / "models" / "model-0000" / f"model-{member:02d}") for member in range(3)]
    assert {member["device"] for member in members} == {"cuda"}
    assert {member["visible"] for member in members} == {"4", "6"}, "both masked GPUs are used, and no others"
    for gpu in ("4", "6"):
        runs = sorted((member["started"], member["finished"]) for member in members if member["visible"] == gpu)
        assert all(first[1] <= second[0] for first, second in zip(runs, runs[1:])), "two copies shared a GPU"
    assert members[0]["started"] < members[1]["finished"] and members[1]["started"] < members[0]["finished"], \
        "copies on different GPUs run at the same time"


def test_child_environment_picks_one_gpu_within_the_worker_mask(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    assert child_environment("cpu")[0] == "cpu"
    assert child_environment("cuda:2")[1]["CUDA_VISIBLE_DEVICES"] == "2"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3, 7")
    assert child_environment("cuda:1") == ("cuda", {**os.environ, "CUDA_VISIBLE_DEVICES": "7"})
    with pytest.raises(ValueError, match="outside"):
        child_environment("cuda:2")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    with pytest.raises(NotImplementedError, match="HIP"):
        child_environment("cuda:0")


def test_multinode_round_trains_each_copy_as_its_own_app(run):
    from tests.helpers.driver import loaded_parsl

    with loaded_parsl(run / "runinfo", "alf_ML_executor"):
        future = fine_tuning.fine_tune_ensemble_multinode_task(
            ml_config(run / "foundation.model", failing_seeds=[12]), str(run / "h5"),
            str(run / "models" / "model-{:04d}"), 0, master_directory=str(run), member_retries=0)
        assert future.result(timeout=120) == ([True, True, False], 0)
    root = run / "models" / "model-0000"
    assert [trained(root / f"model-{member:02d}")["seed"] for member in (0, 1)] == [10, 11]
    assert json.loads((root / fine_tuning.LINEAGE).read_text())["shards"] == ["data-0000.h5"]
    assert len(list(run.glob("models/model-0000/placement-model-*-attempt-1.json"))) == 3


def test_coordinator_checks_the_fine_tuning_config_before_the_bootstrap(tmp_path, monkeypatch):
    from parsl import python_app

    from tests.helpers.driver import run_alf_driver

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        raise AssertionError("no structure should be built")

    with pytest.raises(ValueError, match="Foundation model not found"):
        run_alf_driver(tmp_path, monkeypatch, builder=builder, sampler=builder, qm=builder,
                       ml=fine_tuning.fine_tune_ensemble_task, ml_config=ml_config(tmp_path / "missing.model"),
                       bootstrap_set=2, parallel_samplers=1, target_queued_QM=1, save_h5_threshold=1)
    assert not (tmp_path / "h5").exists() or not list((tmp_path / "h5").iterdir())


# --- MACE -------------------------------------------------------------------------------------------

def mace(tmp_path, **settings):
    base = dict(adapter=MACE, foundation_model=str(tmp_path / "foundation.model"), E0s="average")
    base.update(settings)
    return MaceFineTuner(fine_tune_settings({"fine_tune": base}, str(tmp_path)))


def arguments(command):
    return {part.split("=", 1)[0][2:]: part.split("=", 1)[1] if "=" in part else True
            for part in command[3:]}


def test_mace_command_fine_tunes_from_the_foundation_model_without_replay(tmp_path):
    adapter = mace(tmp_path)
    command = adapter.command("train.xyz", "member", 7, "cuda")
    assert command[1:3] == ["-m", "mace.cli.run_train"]
    args = arguments(command)
    assert args["foundation_model"] == str(tmp_path / "foundation.model")
    assert args["multiheads_finetuning"] == "False", "MACE would otherwise download replay data"
    assert "pt_train_file" not in args and "freeze" not in args
    assert (args["seed"], args["device"], args["E0s"], args["train_file"]) == ("7", "cuda", "average", "train.xyz")
    assert (args["energy_key"], args["forces_key"]) == ("REF_energy", "REF_forces")
    assert args["work_dir"] == args["model_dir"] == "member"


def test_mace_command_turns_on_replay_freezing_and_extra_options(tmp_path):
    # JSON gives string keys and may give whole numbers; MACE needs integer atomic numbers.
    adapter = mace(tmp_path, replay_file="replay.xyz", freeze=3, E0s={"8": -2042, "1": -13.6},
                   mace_options={"swa": True, "ema": False, "patience": 5})
    args = arguments(adapter.command("train.xyz", "member", 1, "cpu"))
    assert args["multiheads_finetuning"] == "True" and args["pt_train_file"] == str(tmp_path / "replay.xyz")
    assert args["freeze"] == "3" and args["E0s"] == "{8: -2042.0, 1: -13.6}"
    assert args["swa"] is True and "ema" not in args and args["patience"] == "5"


@pytest.mark.parametrize("settings, message", [
    (dict(E0s=None), "E0s"),
    (dict(E0s="guess"), "E0s"),
    (dict(dtype="float16"), "dtype"),
    (dict(learning_rate=0), "learning_rate"),
    (dict(freeze=True), "freeze"),
    (dict(valid_fraction=1.0), "valid_fraction"),
    (dict(mace_options={"lr": 0.1, "seed": 3}), r"may not set \['lr', 'seed'\]"),
], ids=["no_E0s", "bad_E0s", "dtype", "learning_rate", "freeze", "valid_fraction", "managed_options"])
def test_mace_refuses_mistakes(tmp_path, settings, message):
    if settings.get("E0s", "") is None:
        block = dict(adapter=MACE, foundation_model="x")
        with pytest.raises(ValueError, match=message):
            MaceFineTuner(fine_tune_settings({"fine_tune": block}))
        return
    with pytest.raises(ValueError, match=message):
        mace(tmp_path, **settings)


def test_mace_checks_where_the_run_stores_energies_and_forces(tmp_path):
    mace(tmp_path).validate({"properties_list": PROPERTIES})
    with pytest.raises(ValueError, match="fine_tune.forces_dataset"):
        mace(tmp_path).validate({"properties_list": {"energy": ["energy", "system", 27.2],
                                                     "forces": ["gradients", "atomic", 51.4]}})
    mace(tmp_path, forces_dataset="gradients").validate(
        {"properties_list": {"energy": ["energy", "system", 1.0], "forces": ["gradients", "atomic", 1.0]}})
    with pytest.raises(ValueError, match="replay_file not found"):
        mace(tmp_path, replay_file="missing.xyz").validate({"properties_list": PROPERTIES})


def test_mace_training_data_matches_the_shards(tmp_path):
    box = Atoms("OH2OH2", positions=WATER.positions.tolist() + (WATER.positions + 2.5).tolist(),
                cell=[6.0, 6.5, 7.0], pbc=True)
    store_current_data(str(tmp_path / "data-0000.h5"), [labeled("w1"), labeled("w2", energy=-3.0)], PROPERTIES)
    store_current_data(str(tmp_path / "data-0001.h5"), [labeled("box", box, energy=-5.0)], PROPERTIES)
    shards = [str(tmp_path / "data-0000.h5"), str(tmp_path / "data-0001.h5")]

    frames = shard_frames(shards)
    assert [frame.info["REF_energy"] for frame in frames] == [-1.0, -3.0, -5.0]
    order = np.argsort(WATER.numbers, kind="stable")
    np.testing.assert_allclose(frames[0].positions, WATER.positions[order])
    np.testing.assert_allclose(frames[0].arrays["REF_forces"], labeled("w1").get_results()["forces"][order])
    assert list(frames[0].numbers) == sorted(WATER.numbers) and not frames[0].pbc.any()
    assert frames[2].pbc.all()
    np.testing.assert_allclose(frames[2].cell.lengths(), [6.0, 6.5, 7.0])

    from ase.io import read

    data = mace(tmp_path).prepare_data(shards, tmp_path)
    written = read(data, index=":")
    assert [frame.info["REF_energy"] for frame in written] == [-1.0, -3.0, -5.0]
    np.testing.assert_allclose(written[2].arrays["REF_forces"], frames[2].arrays["REF_forces"])
    assert sorted(os.listdir(tmp_path)) == ["data-0000.h5", "data-0001.h5", "train.xyz"]
    with pytest.raises(ValueError, match="no labeled frames"):
        mace(tmp_path).prepare_data([], tmp_path)


def test_mace_failure_reports_the_end_of_its_log(tmp_path, monkeypatch):
    import subprocess

    def fails(command, cwd, env, stdout, stderr):
        stdout.write("epoch 1\nRuntimeError: CUDA out of memory\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fails)
    with pytest.raises(RuntimeError, match="exited with 1: epoch 1 | RuntimeError: CUDA out of memory"):
        mace(tmp_path).train_member("train.xyz", tmp_path, 0, 0, "cpu", dict(os.environ))
    assert not mace(tmp_path).member_complete(tmp_path)


def test_mace_batched_ensemble_evaluates_every_copy_on_the_batch():
    torch = pytest.importorskip("torch")
    from alframework.ml_interfaces.mace_interface import MaceBatchedEnsemble

    seen = []

    class FakeMace:
        def __init__(self, scale):
            self.scale = scale

        def __call__(self, state):
            seen.append(state)
            n_systems = int(state["system_idx"].max()) + 1
            return {"energy": torch.full((n_systems,), self.scale, dtype=torch.float64),
                    "forces": state["positions"] * self.scale}

    ensemble = MaceBatchedEnsemble("unused", device="cpu", dtype="float64", models=[FakeMace(1.0), FakeMace(2.0)])
    rows = torch.tensor([[[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.0, 7.0]]] * 2, dtype=torch.float64)
    positions = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    energies, forces = ensemble.evaluate(torch.tensor([8, 1, 1, 8, 1]), positions, rows, [True, True, True], [3, 2])
    assert energies.shape == (2, 2) and forces.shape == (2, 5, 3)
    torch.testing.assert_close(forces[1], positions * 2)
    state = seen[0]
    assert state["system_idx"].tolist() == [0, 0, 0, 1, 1] and state["pbc"].tolist() == [True, True, True]
    torch.testing.assert_close(state["cell"][0], rows[0].T, msg="TorchSim cells are column vectors")
    with pytest.raises(ValueError, match="at least two"):
        MaceBatchedEnsemble("unused", device="cpu", models=[FakeMace(1.0)])


def test_mace_example_configuration_is_valid():
    example = Path(__file__).resolve().parents[1] / "examples" / "mace_fine_tuning"
    master = json.loads((example / "master_config.json").read_text())
    settings = fine_tune_settings(json.loads((example / "fine_tune_config.json").read_text()), str(example))
    assert settings["adapter"] == MACE and settings["n_models"] >= 2
    MaceFineTuner(settings).validate(master)
    sampler = json.loads((example / "mlmd_config.json").read_text())
    assert sampler["batched_calculator"] == "alframework.ml_interfaces.mace_interface.load_mace_batched_ensemble"


def test_lineage_records_the_adapters_defaults(run):
    from alframework.ml_interfaces.fine_tuning import write_lineage

    (run / "models").mkdir()
    settings = fine_tune_settings({"fine_tune": dict(adapter=MACE, foundation_model=str(run / "foundation.model"),
                                                     E0s="foundation")})
    write_lineage(run / "models", settings, [str(run / "h5" / "data-0000.h5")])
    recorded = json.loads((run / "models" / fine_tuning.LINEAGE).read_text())["settings"]
    assert (recorded["E0s"], recorded["dtype"], recorded["max_num_epochs"]) == ("foundation", "float64", 50)
    assert recorded["replay_file"] is None and recorded["mace_options"] == {}


# --- HIPPYNN ----------------------------------------------------------------------------------------
# Real fine-tuning on CPU, from tiny ensembles that tests.helpers.hippynn trains.

HIPPYNN = "alframework.ml_interfaces.hippynn_fine_tuning.HippynnFineTuner"
QUICK = {"max_epochs": 2, "batch_size": 4, "eval_batch_size": 4, "termination_patience": 10}


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    pytest.importorskip("hippynn")
    from tests.helpers.hippynn import train_cpu_ensemble

    root = tmp_path_factory.mktemp("pretrained")
    train_cpu_ensemble(root / "ensemble")
    return root / "ensemble"


def water_labels(directory, count=16, scale=1.5, seed=3):
    """Water whose energy surface differs from the pretrained one, so fine-tuning changes the model."""
    import h5py

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])[None] \
        + rng.normal(0.0, 0.03, (count, 3, 3))
    with h5py.File(directory / "data-0000.h5", "w") as handle:
        group = handle.create_group("water")
        group["species"] = np.array(["O", "H", "H"], dtype="S")
        group["coordinates"] = coordinates
        group["energy"] = scale * (coordinates**2).sum(axis=(1, 2))
        group["forces"] = -2 * scale * coordinates
        group["_id"] = np.array([f"w{index}".encode() for index in range(count)])
    return directory


def hippynn_tuner(source, **settings):
    from alframework.ml_interfaces.hippynn_fine_tuning import HippynnFineTuner

    block = {**dict(adapter=HIPPYNN, foundation_model=str(source), E0s="foundation", device="cpu",
                    learning_rate=1e-3, controller_options=QUICK), **settings}
    return HippynnFineTuner(fine_tune_settings({"fine_tune": block}))


def fine_tune_in_process(tmp_path, tuner, member=0, seed=0, count=16):
    """Run one copy's training in this process, as the child process would."""
    from alframework.ml_interfaces.hippynn_fine_tuning import fine_tune_hippynn_member

    labels = water_labels(tmp_path / "h5", count=count)
    member_dir = tmp_path / "copy"
    fine_tune_hippynn_member(dict(source=str(tuner.source(member)), data=tuner.prepare_data(
        [str(labels / "data-0000.h5")], member_dir), member_dir=str(member_dir), seed=seed, device="cpu",
        settings=tuner.settings))
    assert tuner.member_complete(member_dir)
    return member_dir


def state(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def test_hippynn_round_fine_tunes_copies_that_alf_samplers_load(tmp_path, pretrained):
    torch = pytest.importorskip("torch")
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble, load_hippynn_batched_ensemble

    water_labels(tmp_path / "h5")
    config = {"fine_tune": dict(adapter=HIPPYNN, foundation_model=str(pretrained), n_models=2, device="cpu",
                                E0s="average", learning_rate=1e-3, controller_options=QUICK)}
    validate_fine_tune_config(config, {"properties_list": PROPERTIES})
    assert fine_tune(tmp_path, config, 1) == ([True, True], 1)

    root = tmp_path / "models" / "model-0001"
    lineage = json.loads((root / fine_tuning.LINEAGE).read_text())
    assert sorted(lineage["foundation_model_files"]) == [
        "model-00/best_checkpoint.pt", "model-00/experiment_structure.pt",
        "model-01/best_checkpoint.pt", "model-01/experiment_structure.pt"]
    assert lineage["package"] == "hippynn" and lineage["settings"]["E0s"] == "average"
    assert [json.loads((root / f"model-{m:02d}" / "fine_tune_args.json").read_text())["source"]
            for m in (0, 1)] == [str(pretrained / "model-00"), str(pretrained / "model-01")]
    for member in (0, 1):
        before, after = state(pretrained / f"model-{member:02d}" / "best_model.pt"), state(root / f"model-{member:02d}" / "best_model.pt")
        assert any(not torch.equal(before[key], after[key]) for key in before if before[key].is_floating_point())

    for calculator in HIPNN_ASE_load_ensemble(str(root), device="cpu"):
        atoms = WATER.copy()
        atoms.calc = calculator
        assert np.isfinite(atoms.get_potential_energy())
    energies, forces = load_hippynn_batched_ensemble(str(root), device="cpu").evaluate(
        torch.tensor(WATER.numbers), torch.tensor(WATER.positions), torch.zeros(1, 3, 3), [False] * 3, [3])
    assert energies.shape == (2, 1) and torch.isfinite(forces).all()


def test_hippynn_source_is_one_model_or_each_member_in_turn(pretrained):
    single = hippynn_tuner(pretrained / "model-01")
    assert [single.source(member) for member in range(3)] == [pretrained / "model-01"] * 3
    ensemble = hippynn_tuner(pretrained)
    assert [ensemble.source(member).name for member in range(3)] == ["model-00", "model-01", "model-00"]
    assert sorted(single.fingerprint()) == ["best_checkpoint.pt", "experiment_structure.pt"]


def test_hippynn_frozen_nodes_keep_their_weights(tmp_path, pretrained):
    torch = pytest.importorskip("torch")
    from hippynn.experiment.serialization import load_checkpoint

    from alframework.ml_interfaces.hippynn_fine_tuning import frozen_parameters

    member_dir = fine_tune_in_process(tmp_path, hippynn_tuner(pretrained, freeze=["hipnn"]))
    source = load_checkpoint(str(pretrained / "model-00" / "experiment_structure.pt"),
                             str(pretrained / "model-00" / "best_checkpoint.pt"), restart_db=False, map_location="cpu")
    frozen = set(frozen_parameters(source["training_modules"].model, ["hipnn"]))
    before, after = source["training_modules"].model.state_dict(), state(member_dir / "best_model.pt")
    assert frozen and all(torch.equal(before[name], after[name]) for name in frozen)
    assert any(not torch.equal(before[name], after[name]) for name in before
               if name not in frozen and before[name].is_floating_point()), "the energy head should still train"
    with pytest.raises(ValueError, match=r"'nonesuch' matches no parameters.*'henergy'"):
        frozen_parameters(source["training_modules"].model, ["nonesuch"])


@pytest.mark.parametrize("E0s", ["foundation", "average"])
def test_hippynn_E0s_are_kept_or_refit(tmp_path, pretrained, E0s):
    torch = pytest.importorskip("torch")
    # Freeze the energy head, so only a refit can change its atomic energies.
    member_dir = fine_tune_in_process(tmp_path, hippynn_tuner(pretrained, E0s=E0s, freeze=["henergy"]))
    from hippynn.experiment.serialization import load_checkpoint
    from hippynn.graphs import find_unique_relative
    from hippynn.graphs.nodes.targets import HEnergyNode

    model = load_checkpoint(str(pretrained / "model-00" / "experiment_structure.pt"),
                            str(pretrained / "model-00" / "best_checkpoint.pt"), restart_db=False,
                            map_location="cpu")["training_modules"].model
    henergy = find_unique_relative(model.nodes_to_compute, HEnergyNode)
    e0 = f"moddict.{model.names_dict[henergy]}.layers.0.weight"  # The per-element energies.
    before, after = model.state_dict(), state(member_dir / "best_model.pt")
    assert torch.equal(before[e0], after[e0]) == (E0s == "foundation")


def test_hippynn_uses_a_separate_test_set(tmp_path, pretrained, monkeypatch):
    from alframework.ml_interfaces import hippynn_interface

    used = []
    real = hippynn_interface.use_separate_test_set
    monkeypatch.setattr(hippynn_interface, "use_separate_test_set",
                        lambda database, directory, *args: used.append(directory) or real(database, directory, *args))
    test_dir = water_labels(tmp_path / "qm_test", count=6, seed=9)
    fine_tune_in_process(tmp_path, hippynn_tuner(pretrained, h5_test_dir=str(test_dir)))
    assert used == [str(test_dir)]


def test_hippynn_refuses_an_unknown_stopping_key(tmp_path, pretrained):
    with pytest.raises(ValueError, match=r"stopping_key 'Loss-Error' is not one of the model's validation losses, \['Loss'\]"):
        fine_tune_in_process(tmp_path, hippynn_tuner(pretrained, stopping_key="Loss-Error"))


@pytest.mark.parametrize("settings, message", [
    (dict(E0s=None), "E0s is required"),
    (dict(E0s="estimated"), "E0s is required"),
    (dict(learning_rate=-1), "learning_rate"),
    (dict(valid_size=0), "valid_size"),
    (dict(freeze="hipnn"), "freeze must be a list"),
], ids=["no_E0s", "mace_E0s", "learning_rate", "valid_size", "freeze_string"])
def test_hippynn_refuses_mistakes(tmp_path, settings, message):
    from alframework.ml_interfaces.hippynn_fine_tuning import HippynnFineTuner

    block = {**dict(adapter=HIPPYNN, foundation_model=str(tmp_path), E0s="foundation"), **settings}
    if block["E0s"] is None:
        del block["E0s"]
    with pytest.raises(ValueError, match=message):
        HippynnFineTuner(fine_tune_settings({"fine_tune": block}))


def test_hippynn_startup_check_finds_the_model_files(tmp_path):
    from alframework.ml_interfaces.hippynn_fine_tuning import HippynnFineTuner

    def check(**settings):
        block = {**dict(adapter=HIPPYNN, foundation_model=str(tmp_path / "source"), E0s="foundation"), **settings}
        HippynnFineTuner(fine_tune_settings({"fine_tune": block})).validate({"properties_list": PROPERTIES})

    (tmp_path / "source" / "model-00").mkdir(parents=True)
    with pytest.raises(ValueError, match="neither a HIPPYNN model"):
        check()
    (tmp_path / "source" / "model-00" / "experiment_structure.pt").write_bytes(b"graph")
    with pytest.raises(ValueError, match="best_checkpoint.pt not found"):
        check()
    (tmp_path / "source" / "model-00" / "best_checkpoint.pt").write_bytes(b"weights")
    check()
    with pytest.raises(ValueError, match="'gradients', which properties_list doesn't store"):
        check(force_key="gradients")
    with pytest.raises(ValueError, match="h5_test_dir not found"):
        check(h5_test_dir=str(tmp_path / "missing"))


def test_hippynn_reads_shards_in_place(tmp_path):
    from alframework.ml_interfaces.hippynn_fine_tuning import HippynnFineTuner

    tuner = HippynnFineTuner(fine_tune_settings({"fine_tune": dict(adapter=HIPPYNN, foundation_model="x",
                                                                    E0s="foundation")}))
    assert tuner.prepare_data([str(tmp_path / "h5" / "data-0000.h5"), str(tmp_path / "h5" / "seed.h5")], tmp_path) \
        == {"directory": str(tmp_path / "h5"), "files": ["data-0000.h5", "seed.h5"]}
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="share one directory"):
        tuner.prepare_data([str(tmp_path / "a" / "x.h5"), str(tmp_path / "b" / "y.h5")], tmp_path)
    with pytest.raises(ValueError, match="no labeled shards"):
        tuner.prepare_data([], tmp_path)


def test_hippynn_failure_reports_the_end_of_its_training_log(tmp_path, monkeypatch):
    import subprocess

    from alframework.ml_interfaces.hippynn_fine_tuning import HippynnFineTuner

    (tmp_path / "source").mkdir()
    for name in ("experiment_structure.pt", "best_checkpoint.pt"):
        (tmp_path / "source" / name).write_bytes(b"x")
    tuner = HippynnFineTuner(fine_tune_settings({"fine_tune": dict(adapter=HIPPYNN, foundation_model=str(tmp_path / "source"),
                                                                    E0s="foundation")}))

    def fails(command, cwd, env, stdout, stderr):
        (Path(cwd) / "training_log.txt").write_text("epoch 1\nTraceback\nRuntimeError: CUDA out of memory\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fails)
    member_dir = tmp_path / "copy"
    member_dir.mkdir()
    with pytest.raises(RuntimeError, match="exited with 1: epoch 1 | Traceback | RuntimeError: CUDA out of memory"):
        tuner.train_member({"directory": "h5", "files": []}, member_dir, 0, 0, "cpu", dict(os.environ))
    assert json.loads((member_dir / "fine_tune_args.json").read_text())["source"] == str(tmp_path / "source")
    assert not tuner.member_complete(member_dir), "a log without 'Training complete' and no best_model.pt"


def test_hippynn_fine_tuning_imports_without_torch():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    assert name.split('.')[0] not in ('torch', 'hippynn'), name
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import alframework.ml_interfaces.hippynn_fine_tuning
"""], check=True, capture_output=True)


def test_hippynn_example_configuration_is_valid(tmp_path):
    from alframework.ml_interfaces.hippynn_fine_tuning import HippynnFineTuner

    example = Path(__file__).resolve().parents[1] / "examples" / "fine_tuning"
    master = json.loads((example / "master_config.json").read_text())
    assert master["ML_task"] == "alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_task"
    config = json.loads((example / "hippynn_config.json").read_text())
    # The example fine-tunes the ensemble it first samples with, which you provide.
    assert config["fine_tune"]["foundation_model"] == master["model_path"].format(0)
    for member in ("model-00", "model-01"):
        (tmp_path / "models" / "model-0000" / member).mkdir(parents=True)
        for name in ("experiment_structure.pt", "best_checkpoint.pt"):
            (tmp_path / "models" / "model-0000" / member / name).write_bytes(b"x")
    settings = fine_tune_settings(config, str(tmp_path))
    assert settings["adapter"] == HIPPYNN
    tuner = HippynnFineTuner(settings)
    tuner.validate(master)
    assert len(tuner.sources()) == 2


@pytest.mark.parametrize("count", [3, 5], ids=["three", "five"])
def test_hippynn_fine_tunes_a_small_first_round(tmp_path, pretrained, count):
    """A 10% split of a few labels is empty, and so is 10% of their training split; both get at least one."""
    member_dir = fine_tune_in_process(tmp_path, hippynn_tuner(pretrained), count=count)
    log = (member_dir / "training_log.txt").read_text()
    assert "Training complete" in log


def test_hippynn_refuses_too_few_labels(tmp_path, pretrained):
    with pytest.raises(ValueError, match="at least 3 labeled structures"):
        fine_tune_in_process(tmp_path, hippynn_tuner(pretrained), count=2)


def test_hippynn_copy_is_complete_only_with_its_model_and_a_finished_log(tmp_path):
    from alframework.ml_interfaces.hippynn_fine_tuning import HippynnFineTuner

    tuner = HippynnFineTuner(fine_tune_settings({"fine_tune": dict(adapter=HIPPYNN, foundation_model="x",
                                                                    E0s="foundation")}))
    (tmp_path / "best_model.pt").write_bytes(b"weights")
    (tmp_path / "training_log.txt").write_text("Epoch 0\nEpoch 1\n")
    assert not tuner.member_complete(tmp_path), "stopped before HIPPYNN finished"
    (tmp_path / "training_log.txt").write_text("Epoch 0\nTraining complete.\n")
    assert tuner.member_complete(tmp_path)


def test_hippynn_never_trains_or_freezes_integer_parameters(pretrained):
    pytest.importorskip("torch")
    from hippynn.experiment.serialization import load_checkpoint

    from alframework.ml_interfaces.hippynn_fine_tuning import frozen_parameters, trainable_names

    model = load_checkpoint(str(pretrained / "model-00" / "experiment_structure.pt"),
                            str(pretrained / "model-00" / "best_checkpoint.pt"), restart_db=False,
                            map_location="cpu")["training_modules"].model
    integer = [name for name, parameter in model.named_parameters() if not parameter.is_floating_point()]
    assert integer and not set(integer) & set(trainable_names(model)), "the species map is a parameter, not a weight"
    with pytest.raises(ValueError, match="matches no parameters"):
        frozen_parameters(model, [integer[0]])


def test_hippynn_names_what_the_shards_lack(tmp_path, pretrained):
    import h5py

    labels = water_labels(tmp_path / "h5")
    with h5py.File(labels / "data-0000.h5", "a") as handle:
        del handle["water"]["forces"]
    from alframework.ml_interfaces.hippynn_fine_tuning import fine_tune_hippynn_member

    tuner = hippynn_tuner(pretrained)
    with pytest.raises(ValueError, match=r"needs \['forces'\], which the shards lack"):
        fine_tune_hippynn_member(dict(source=str(tuner.source(0)), data=tuner.prepare_data(
            [str(labels / "data-0000.h5")], tmp_path / "copy"), member_dir=str(tmp_path / "copy"), seed=0,
            device="cpu", settings=tuner.settings))
