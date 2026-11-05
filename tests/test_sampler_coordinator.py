"""Multi-trajectory sampling through the real ALF coordinator with local tasks."""
import json
import os
import runpy
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import parsl
import pytest
from ase import Atoms
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

from alframework.samplers import batched_mlmd
from alframework.samplers.mlmd_sampling import simple_mlmd_sampling_task
from alframework.tools import tools as tool_module
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_coordinator import SamplerCoordinator, driver_settings

WATER = Atoms("OH2", positions=[[0, 0, 0], [1.2, 0, 0], [0.6, 1.0392, 0]])
SAMPLER = dict(dt=0.5, maxt=0.05, Escut=0.012, Fscut=100.0, Ncheck=5, distcut=0.5, random_seed=3,
               srt_temp=[2500.0, 3500.0], end_temp=[2500.0, 3500.0], amp_temp=[0.0, 200.0],
               per_temp=[0.01, 0.05], amp_dens=None, per_dens=None, end_dens=None, device="cpu",
               ase_calculator="tests.helpers.fakes.load_spring_ensemble")


def stored_ids(h5_dir):
    ids = []
    for path in sorted(Path(h5_dir).glob("data-*.h5")):
        with h5py.File(path) as handle:
            for group in handle.values():
                ids.extend(value.decode() for value in group["_id"][()])
    return ids


TORCHSIM = {"strategy": "batched", "engine": "torchsim"}
ALCHEMI = {"strategy": "batched", "engine": "alchemi"}


@pytest.mark.parametrize("driver,policy", [({"strategy": "single"}, None), ({"strategy": "batched"}, None),
                                           ({"strategy": "batched"}, "stop"), ({"strategy": "batched"}, "continue"),
                                           (TORCHSIM, None), (TORCHSIM, "continue"),
                                           (ALCHEMI, None), (ALCHEMI, "stop"), (ALCHEMI, "continue"),
                                           ({"strategy": "single", "engine": "alchemi"}, None)],
                         ids=["single", "batched", "cohort-stop", "cohort-continue", "torchsim",
                              "torchsim-cohort-continue", "alchemi", "alchemi-cohort-stop",
                              "alchemi-cohort-continue", "alchemi-single"])
def test_coordinator_bootstraps_samples_batches_labels_and_retrains(tmp_path, monkeypatch, driver, policy):
    if driver.get("engine") == "torchsim":
        pytest.importorskip("torch_sim")
    if driver.get("engine") == "alchemi":
        from importlib.metadata import PackageNotFoundError, version
        try:
            from alframework.samplers.alchemi_engine import SUPPORTED_VERSIONS
            qualified = version('nvalchemi-toolkit') in SUPPORTED_VERSIONS
        except PackageNotFoundError:
            qualified = False
        if not qualified:
            pytest.skip('ALCHEMI 0.2 has its own worker environment.')
    monkeypatch.chdir(tmp_path)
    model_pattern = str(tmp_path / "models/model-{:04d}")
    training_calls, sampler_widths, labeled, engines = [], [], [], []

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(WATER.copy(), moleculeid)

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
        Path(model_pattern.format(current_training_id)).mkdir(parents=True)
        return [True] * 4, current_training_id

    config = Config(executors=[ThreadPoolExecutor(label=f"alf_{name}_executor", max_threads=2)
                               for name in ("builder", "sampler", "QM", "ML")],
                    strategy="none", run_dir=str(tmp_path / "runinfo"))
    modules = {"mock.config": config, "mock.builder": builder, "mock.qm": qm, "mock.ml": ml,
               "mock.sampler": batched_mlmd.mlmd_trajectories_task}
    monkeypatch.setattr(tool_module, "load_module_from_string", modules.__getitem__)

    sampler_config = dict(SAMPLER, driver=driver)
    if driver.get('engine') in {'torchsim', 'alchemi'}:
        sampler_config['device'] = os.environ.get('ALF_TEST_ENGINE_DEVICE', 'cpu')
    if policy is not None:
        # Cohorts of 6 with capacity 4: admission must continue while members run.
        sampler_config["acquisition"] = {"trajectory_policy": policy,
                                         "selection": {"mode": "cohort_top_k", "cohort_size": 6, "k": 2}}
    if policy == "continue" and driver.get("engine") == "alchemi":
        sampler_config['acquisition'].update(strategy='summed', weights={'energy': 1., 'force_rms': 1.})
    if driver["strategy"] == "batched":
        sampler_config["batching"] = {"batch_size": 3, "max_wait_seconds": 0.05}
    master = dict(master_directory="pwd", builder_task="mock.builder", sampler_task="mock.sampler",
                  QM_task="mock.qm", ML_task="mock.ml", parsl_configuration="mock.config",
                  builder_config_path="builder.json", sampler_config_path="sampler.json",
                  QM_config_path="qm.json", ML_config_path="ml.json", status_path="status.json",
                  model_path=model_pattern, h5_path=str(tmp_path / "h5/data-{:04d}.h5"), bootstrap_set=2,
                  properties_list={"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]},
                  parallel_samplers=4, target_queued_QM=4, minimum_QM=0, save_h5_threshold=3,
                  maximum_completed_QM=8, gpus_per_node=0)
    for name, values in [("master", master), ("builder", {}), ("sampler", sampler_config), ("qm", {}), ("ml", {})]:
        (tmp_path / f"{name}.json").write_text(json.dumps(values))
    monkeypatch.setattr(sys, "argv", ["alframework", "--master", str(tmp_path / "master.json")])

    original_sleep = time.sleep
    passes, sampler_counts = [], []

    class Finished(Exception):
        pass

    def driver_sleep(seconds):
        frame = sys._getframe(1)
        if frame.f_globals.get("__name__") != "__main__":
            return original_sleep(seconds)
        passes.append(seconds)
        if "sampler_task_queue" in frame.f_globals:
            sampler_counts.append(frame.f_globals["sampler_task_queue"].get_number())
        if json.loads((tmp_path / "status.json").read_text())["current_model_id"] >= 1:
            raise Finished
        assert len(passes) < 3000, "coordinator did not complete active learning"
        original_sleep(min(seconds, 0.01))

    monkeypatch.setattr(time, "sleep", driver_sleep)
    try:
        with pytest.raises(Finished):
            runpy.run_module("alframework", run_name="__main__")
    finally:
        try:
            parsl.dfk().cleanup()
        finally:
            parsl.clear()

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["lifetime_failed_sampler_tasks"] == 0 and status["lifetime_failed_QM_tasks"] == 0
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
    assert max(sampler_counts) <= master["parallel_samplers"]
    if driver["strategy"] == "single":
        assert set(sampler_widths) == {1}
    else:
        assert max(sampler_widths) > 1 and max(sampler_widths) <= 3


def test_driver_settings_validation():
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


def test_legacy_sampler_is_unchanged_and_rejects_new_blocks():
    coordinator = SamplerCoordinator(simple_mlmd_sampling_task, "mlmd", {"dt": 0.5})
    assert type(coordinator.queue) is tool_module.parsl_task_queue and not coordinator.batched
    for key in ("driver", "batching"):
        with pytest.raises(ValueError, match="molecule_objects"):
            SamplerCoordinator(simple_mlmd_sampling_task, "mlmd", {key: {}})


def test_multi_trajectory_sampler_counts_structures():
    cutoffs = {"Escut": 0.5, "Fscut": 0.3}
    single = SamplerCoordinator(batched_mlmd.mlmd_trajectories_task, "batched", cutoffs)
    assert single.batched and single.settings["batch_size"] == 1 and single.poll_delay(60) == 60
    batched = SamplerCoordinator(batched_mlmd.mlmd_trajectories_task, "batched",
                                 {**cutoffs, "driver": {"strategy": "batched"}, "batching": {"batch_size": 4}})
    batched.submit(MoleculesObject(WATER.copy(), "a"), [{"current_model_id": 0}])
    assert batched.queue.get_number() == 1 and batched.poll_delay(60) == pytest.approx(5.0, abs=0.1)


def test_real_sampler_coordinator_pins_admission_and_task_generation(tmp_path):
    import parsl
    from parsl import python_app
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor
    from ase import Atoms
    from alframework.tools.molecules_class import MoleculesObject
    from alframework.tools.sampler_coordinator import SamplerCoordinator
    @python_app(executors=['pinning'])
    def sampler(molecule_objects, sampler_config, current_model_id):
        for m in molecule_objects:
            m.update_metadata(dict(executed_model_id=current_model_id, trajectory_outcome='completed', acquisition_candidates=[]))
        return molecule_objects
    config = dict(driver=dict(engine='ase', strategy='batched'),
                  batching=dict(batch_size=2, partial_policy='timeout', max_wait_seconds=0),
                  acquisition=dict(selection=dict(mode='cohort_top_k', cohort_size=3, k=1)), Escut=20, Fscut=.3)
    dfk = parsl.load(Config(executors=[ThreadPoolExecutor(label='pinning', max_threads=1)], run_dir=str(tmp_path / 'parsl')))
    try:
        coordinator = SamplerCoordinator(sampler, 'test.sampler', config)
        for name, model in [('a', 0), ('b', 0), ('c', 1), ('d', 1)]:
            coordinator.submit(MoleculesObject(Atoms('H'), name), [dict(sampler_config=config, current_model_id=model)])
        coordinator.submit_ready([dict(sampler_config=config, current_model_id=1)])
        # Wait through actual Parsl results, then use the normal harvest path.
        for task in list(dfk.tasks.values()):
            task['app_fu'].result()
        results, failed = coordinator.queue.get_task_results()
        assert failed == 0
        assert {m.get_moleculeid(): m.get_metadata()['executed_model_id'] for m in results} == dict(a=0, b=0, c=0, d=1)
        assert coordinator.harvest(results) == []
        assert coordinator.pool.finished[0]['model_id'] == 0
        assert coordinator.status()['open'][0]['model_id'] == 1
    finally:
        dfk.cleanup()
        parsl.clear()


def test_excited_state_admission_is_independent_of_width_and_completion_order(tmp_path):
    @python_app(executors=['state_admission'])
    def capture(molecule_objects, current_model_id):
        assert len({m.get_metadata()['selected_state'] for m in molecule_objects}) == 1
        assert all(m.get_metadata()['sampling_model_id'] == current_model_id for m in molecule_objects)
        return molecule_objects
    dfk = parsl.load(Config(executors=[ThreadPoolExecutor(label='state_admission',max_threads=2)],
                            run_dir=str(tmp_path/'parsl')))
    try:
        snapshots = []
        for width in (1,2,4):
            config = dict(Escut=1.,Fscut=1.,model_mode='excited_state',
                          state_selection={'mode':'batch_cycle','states':[0,1]},
                          driver={'strategy':'single' if width==1 else 'batched'})
            if width > 1:
                config['batching'] = {'batch_size':width,'partial_policy':'timeout','max_wait_seconds':60}
            coordinator = SamplerCoordinator(capture, 'test.capture', config)
            for index in range(8):
                molecule = MoleculesObject(WATER.copy(), f'state-{index}')
                if index == 2:
                    molecule.update_metadata({'selected_state':1})
                coordinator.submit(molecule, [{'current_model_id':7}])
            coordinator.submit_ready([{'current_model_id':7}], flush=True)
            results = [m for future in reversed(coordinator.queue.task_list) for m in future.result()]
            snapshots.append(sorted((m.get_moleculeid(),m.get_metadata()['selected_state'],
                                     m.get_metadata()['sampling_admission_ordinal']) for m in results))
        assert snapshots[0] == snapshots[1] == snapshots[2]
        assert [state for _,state,_ in snapshots[0]] == [0,1,1,1,0,1,0,1]
    finally:
        dfk.cleanup(); parsl.clear()
