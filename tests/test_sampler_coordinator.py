"""Sampling through the real ALF coordinator, with local stand-ins for its tasks.

Covers the full active learning loop, batching, cohorts, and finite bootstrap. TorchSim and ALCHEMI
cases skip if not installed.
"""

import os

import numpy as np
import pytest
from ase import Atoms
from parsl import python_app

from alframework.samplers import batched_mlmd
from alframework.samplers.mlmd_sampling import simple_mlmd_sampling_task
from alframework.tools import tools as tool_module
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_coordinator import SamplerCoordinator, driver_settings
from tests.helpers.driver import MODEL_PATTERN, loaded_parsl, run_alf_driver, stored_ids
from tests.helpers.engines import require_engine
from tests.helpers.molecules import SPRING_WATER

SAMPLER = dict(
    dt=0.5,
    maxt=0.05,
    Escut=0.012,
    Fscut=100.0,
    Ncheck=5,
    distcut=0.5,
    random_seed=3,
    srt_temp=[2500.0, 3500.0],
    end_temp=[2500.0, 3500.0],
    amp_temp=[0.0, 200.0],
    per_temp=[0.01, 0.05],
    amp_dens=None,
    per_dens=None,
    end_dens=None,
    device="cpu",
    ase_calculator="tests.helpers.fakes.load_spring_ensemble",
)
TORCHSIM = {"strategy": "batched", "engine": "torchsim"}
ALCHEMI = {"strategy": "batched", "engine": "alchemi"}
PARALLEL_SAMPLERS = 4


@pytest.mark.parametrize(
    "driver,policy",
    [
        ({"strategy": "single"}, None),
        ({"strategy": "batched"}, None),
        ({"strategy": "batched"}, "stop"),
        ({"strategy": "batched"}, "continue"),
        (TORCHSIM, None),
        (TORCHSIM, "continue"),
        (ALCHEMI, None),
        (ALCHEMI, "stop"),
        (ALCHEMI, "continue"),
        ({"strategy": "single", "engine": "alchemi"}, None),
    ],
    ids=[
        "single",
        "batched",
        "cohort-stop",
        "cohort-continue",
        "torchsim",
        "torchsim-cohort-continue",
        "alchemi",
        "alchemi-cohort-stop",
        "alchemi-cohort-continue",
        "alchemi-single",
    ],
)
def test_active_learning_loop_completes_a_full_cycle(tmp_path, monkeypatch, driver, policy):
    require_engine(driver.get("engine"))
    training_calls, sampler_widths, labeled, engines = [], [], [], []

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(SPRING_WATER.copy(), moleculeid)

    original_sample = batched_mlmd.sample_trajectories

    def recording_sample(molecule_objects, models, sampler_config, engine_class=None, device=None):
        sampler_widths.append(len(molecule_objects))
        engines.append(batched_mlmd.driver_settings(sampler_config)["engine"])
        return original_sample(molecule_objects, models, sampler_config, engine_class, device)

    monkeypatch.setattr(batched_mlmd, "sample_trajectories", recording_sample)

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        labeled.append((molecule_object.get_moleculeid(), molecule_object.get_metadata().get("cohort_id")))
        molecule_object.store_results({"energy": -1.0, "forces": np.zeros((len(molecule_object), 3))})
        molecule_object.set_converged_flag(True)
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        training_calls.append((current_training_id, stored_ids(tmp_path / "h5")))
        (tmp_path / MODEL_PATTERN.format(current_training_id)).mkdir(parents=True)
        return [True] * 4, current_training_id

    sampler_config = dict(SAMPLER, driver=driver)
    if driver.get("engine") in {"torchsim", "alchemi"}:
        sampler_config["device"] = os.environ.get("ALF_TEST_ENGINE_DEVICE", "cpu")
    if policy is not None:
        # Cohorts of 6 with capacity 4: admission must continue while members run.
        sampler_config["acquisition"] = {
            "trajectory_policy": policy,
            "selection": {"mode": "cohort_top_k", "cohort_size": 6, "k": 2},
        }
    if policy == "continue" and driver.get("engine") == "alchemi":
        sampler_config["acquisition"].update(strategy="summed", weights={"energy": 1.0, "force_rms": 1.0})
    if driver["strategy"] == "batched":
        sampler_config["batching"] = {"batch_size": 3, "max_wait_seconds": 0.05}

    run = run_alf_driver(
        tmp_path,
        monkeypatch,
        builder=builder,
        sampler=batched_mlmd.mlmd_trajectories_task,
        qm=qm,
        ml=ml,
        sampler_config=sampler_config,
        bootstrap_set=2,
        parallel_samplers=PARALLEL_SAMPLERS,
        target_queued_QM=4,
        save_h5_threshold=3,
        maximum_completed_QM=8,
        gpus_per_node=0,
    )

    assert run.status["lifetime_failed_sampler_tasks"] == 0 and run.status["lifetime_failed_QM_tasks"] == 0
    assert training_calls[0][0] == 0 and training_calls[1][0] == 1
    sampled_ids = [value for value in training_calls[1][1] if value.startswith("mol-0000-")]
    assert sampled_ids, "sampled frames must reach the second HDF5 shard"
    if policy is not None:
        # Only cohort selections are labeled: k per cohort, named <trajectory>-s<step>.
        active = [(name, cohort) for name, cohort in labeled if name.startswith("mol-0000-")]
        assert active and all("-s" in name and cohort for name, cohort in active)
        assert len(active) == len(set(active))
        per_cohort = {}
        for name, cohort in active:
            per_cohort.setdefault(cohort, []).append(name)
        assert all(len(frames) <= 2 for frames in per_cohort.values()), per_cohort
        assert any(len(frames) == 2 for frames in per_cohort.values()), per_cohort
    assert set(engines) == {driver.get("engine", "ase")}
    assert max(run.sampler_queue_sizes) <= PARALLEL_SAMPLERS
    if driver["strategy"] == "single":
        assert set(sampler_widths) == {1}
    else:
        assert max(sampler_widths) > 1 and max(sampler_widths) <= 3


def test_driver_settings_are_validated():
    assert driver_settings({})["strategy"] == "single"
    with pytest.raises(ValueError, match="requires a sampler 'batching'"):
        driver_settings({"driver": {"strategy": "batched"}})
    with pytest.raises(ValueError, match="requires driver.strategy 'batched'"):
        driver_settings({"batching": {"batch_size": 2}})
    assert driver_settings({"driver": {"engine": "torchsim"}})["engine"] == "torchsim"
    assert driver_settings({"driver": {"engine": "alchemi"}})["engine"] == "alchemi"
    assert driver_settings({"driver": {"engine": "alchemi"}, "batched_calculator": "x.y"})["engine"] == "alchemi"
    with pytest.raises(ValueError, match="batched_calculator requires"):
        driver_settings({"batched_calculator": "x.y"})
    with pytest.raises(ValueError, match="driver.engine"):
        driver_settings({"driver": {"engine": "lammps"}})
    with pytest.raises(ValueError, match="Unknown driver"):
        driver_settings({"driver": {"stratgy": "single"}})


def test_legacy_sampler_rejects_batching_settings():
    coordinator = SamplerCoordinator(simple_mlmd_sampling_task, "mlmd", {"dt": 0.5})
    assert type(coordinator.queue) is tool_module.parsl_task_queue and not coordinator.batched
    for key in ("driver", "batching"):
        with pytest.raises(ValueError, match="molecule_objects"):
            SamplerCoordinator(simple_mlmd_sampling_task, "mlmd", {key: {}})


@pytest.mark.parametrize(
    "policy,capacity,rejected",
    [("full_only", 3, True), ("full_only", 4, False), ("timeout", 3, False)],
    ids=["cannot_fill", "can_fill", "timeout_releases_partial_batches"],
)
def test_sampler_full_only_batch_must_be_able_to_fill(policy, capacity, rejected):
    config = dict(
        SAMPLER,
        driver={"strategy": "batched"},
        batching={"batch_size": 4, "partial_policy": policy, "max_wait_seconds": 1.0},
    )
    master = {"parallel_samplers": capacity}
    if rejected:
        with pytest.raises(ValueError, match="parallel_samplers"):
            SamplerCoordinator(batched_mlmd.mlmd_trajectories_task, "batched", config, master)
    else:
        SamplerCoordinator(batched_mlmd.mlmd_trajectories_task, "batched", config, master)


def test_batched_sampler_counts_molecules():
    cutoffs = {"Escut": 0.5, "Fscut": 0.3}
    single = SamplerCoordinator(batched_mlmd.mlmd_trajectories_task, "batched", cutoffs)
    assert single.batched and single.settings["batch_size"] == 1 and single.poll_delay(60) == 60
    batched = SamplerCoordinator(
        batched_mlmd.mlmd_trajectories_task,
        "batched",
        {**cutoffs, "driver": {"strategy": "batched"}, "batching": {"batch_size": 4}},
    )
    batched.submit(MoleculesObject(SPRING_WATER.copy(), "a"), [{"current_model_id": 0}])
    assert batched.queue.get_number() == 1 and batched.poll_delay(60) == pytest.approx(5.0, abs=0.1)


def test_cohort_stays_on_the_model_it_started_with(tmp_path):
    @python_app(executors=["pinning"])
    def sampler(molecule_objects, sampler_config, current_model_id):
        for m in molecule_objects:
            m.update_metadata(
                dict(executed_model_id=current_model_id, trajectory_outcome="completed", acquisition_candidates=[])
            )
        return molecule_objects

    config = dict(
        driver=dict(engine="ase", strategy="batched"),
        batching=dict(batch_size=2, partial_policy="timeout", max_wait_seconds=0),
        acquisition=dict(selection=dict(mode="cohort_top_k", cohort_size=3, k=1)),
        Escut=20,
        Fscut=0.3,
    )
    with loaded_parsl(tmp_path / "parsl", "pinning") as dfk:
        coordinator = SamplerCoordinator(sampler, "test.sampler", config)
        for name, model in [("a", 0), ("b", 0), ("c", 1), ("d", 1)]:
            coordinator.submit(MoleculesObject(Atoms("H"), name), [dict(sampler_config=config, current_model_id=model)])
        coordinator.submit_ready([dict(sampler_config=config, current_model_id=1)])
        # Wait through actual Parsl results, then use the normal harvest path.
        for task in list(dfk.tasks.values()):
            task["app_fu"].result()
        results, failed = coordinator.queue.get_task_results()
        assert failed == 0
        assert {m.get_moleculeid(): m.get_metadata()["executed_model_id"] for m in results} == dict(a=0, b=0, c=0, d=1)
        assert coordinator.harvest(results) == []
        assert coordinator.pool.finished[0]["model_id"] == 0
        assert coordinator.status()["open"][0]["model_id"] == 1


def test_state_assignment_ignores_batch_size(tmp_path):
    @python_app(executors=["state_admission"])
    def capture(molecule_objects, current_model_id):
        assert len({m.get_metadata()["selected_state"] for m in molecule_objects}) == 1
        assert all(m.get_metadata()["sampling_model_id"] == current_model_id for m in molecule_objects)
        return molecule_objects

    with loaded_parsl(tmp_path / "parsl", "state_admission"):
        snapshots = []
        for width in (1, 2, 4):
            config = dict(
                Escut=1.0,
                Fscut=1.0,
                model_mode="excited_state",
                state_selection={"mode": "batch_cycle", "states": [0, 1]},
                driver={"strategy": "single" if width == 1 else "batched"},
            )
            if width > 1:
                config["batching"] = {"batch_size": width, "partial_policy": "timeout", "max_wait_seconds": 60}
            coordinator = SamplerCoordinator(capture, "test.capture", config)
            for index in range(8):
                molecule = MoleculesObject(SPRING_WATER.copy(), f"state-{index}")
                if index == 2:
                    molecule.update_metadata({"selected_state": 1})
                coordinator.submit(molecule, [{"current_model_id": 7}])
            coordinator.submit_ready([{"current_model_id": 7}], flush=True)
            results = [m for future in reversed(coordinator.queue.task_list) for m in future.result()]
            snapshots.append(
                sorted(
                    (
                        m.get_moleculeid(),
                        m.get_metadata()["selected_state"],
                        m.get_metadata()["sampling_admission_ordinal"],
                    )
                    for m in results
                )
            )
        assert snapshots[0] == snapshots[1] == snapshots[2]
        assert [state for _, state, _ in snapshots[0]] == [0, 1, 1, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("failure", [None, "builder", "qm"], ids=["all_succeed", "builder_fails", "qm_fails"])
def test_finite_bootstrap_stops_when_structures_run_out(tmp_path, monkeypatch, failure):
    requested, training, sampled = [], [], []

    @python_app(executors=["finite_executor"])
    def builder(moleculeid=None, moleculeids=None):
        ids = moleculeids if moleculeids is not None else [moleculeid]
        requested.extend(ids)
        if failure == "builder":
            raise RuntimeError("deliberate builder failure")
        return [MoleculesObject(SPRING_WATER.copy(), name) for name in ids]

    @python_app(executors=["finite_executor"])
    def label(molecule_objects):
        if failure == "qm":
            raise RuntimeError("deliberate label failure")
        for m in molecule_objects:
            m.store_results(dict(energy=-1.0, forces=np.zeros((3, 3))))
            m.set_converged_flag(True)
        return molecule_objects

    @python_app(executors=["finite_executor"])
    def train(current_training_id):
        training.extend(stored_ids(tmp_path / "h5"))
        return [True] * 4, current_training_id

    @python_app(executors=["finite_executor"])
    def sampler(molecule_object):
        sampled.append(molecule_object.get_moleculeid())
        return molecule_object

    settings = dict(
        builder=builder,
        sampler=sampler,
        qm=label,
        ml=train,
        executors=("finite_executor",),
        qm_config={
            "batching": {"mode": "native", "batch_size": 4, "partial_policy": "timeout", "max_wait_seconds": 0.001}
        },
        # Run until a list returned by the builder has also reached the sampler.
        stop_when=lambda status: status["current_model_id"] >= 0 and bool(sampled),
        bootstrap_set=5,
        finite_bootstrap=True,
        maximum_builder_structures=4,
        parallel_samplers=8,
        target_queued_QM=8,
        save_h5_threshold=7,
        gpus_per_node=0,
    )
    if failure:
        with pytest.raises(RuntimeError, match="Finite bootstrap exhausted"):
            run_alf_driver(tmp_path, monkeypatch, **settings)
    else:
        run_alf_driver(tmp_path, monkeypatch, **settings)

    bootstrap = [name for name in requested if name.startswith("mol-boot-")]
    assert sorted(bootstrap) == [f"mol-boot-{i:010d}" for i in range(5)]
    assert len(training) == (0 if failure else 5)
