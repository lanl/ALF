"""The trajectory summary: one row per trajectory, written by the coordinator for progress plots.

Rows come from real sampler metadata, and the coordinator test runs the real loop and real ASE-engine
trajectories against fake spring models. The packed-engine and excited-state metadata is covered by
the sampler tests, because the summary only reads what they record.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from parsl import python_app

from alframework.samplers import batched_mlmd
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.plotting import analysis_plot
from alframework.tools.sampler_coordinator import SamplerCoordinator
from alframework.tools.tools import build_input_dict, load_config_file
from alframework.tools.trajectory_summary import COLUMNS, TrajectorySummary, read_trajectory_summary, trajectory_row
from tests.helpers.driver import MODEL_PATTERN, run_alf_driver
from tests.helpers.molecules import WATER

BATCHED = dict(trajectory_outcome="rejected", screen_reject_reason="physical_force", selected_state=np.int64(1),
               sampling_window=2, simulationtime=0.5, realtime_simulation=np.float64(3.0), Es=0.1, Fs=0.2, Fsmax=0.3,
               peak_Es=0.4, peak_Fs=0.5, peak_Fsmax=0.6, gap_std=0.7, peak_gap_std=0.8, positions=np.zeros((3, 3)), acquisition_candidates=[{"big": 1}])


def molecule(moleculeid="mol-0003-0000000012", **metadata):
    result = MoleculesObject(Atoms("H"), moleculeid)
    result.update_metadata(metadata)
    return result


def test_trajectory_row_keeps_only_what_plots_need():
    row = trajectory_row("mol-0003-0000000012", BATCHED)
    assert row == {"id": "mol-0003-0000000012", "model": 3, "outcome": "rejected", "reason": "physical_force", "state": 1,
                   "window": 2, "simtime": 0.5, "realtime": 3.0, "Es": 0.1, "Fs": 0.2, "Fsmax": 0.3, "peak_Es": 0.4,
                   "peak_Fs": 0.5, "peak_Fsmax": 0.6, "gap_std": 0.7, "peak_gap_std": 0.8}
    json.dumps(row)


def test_trajectory_row_for_legacy_metadata_has_no_outcome():
    row = trajectory_row("mol-0000-0000000001", dict(Es=0.1, Fs=0.2, Fsmax=0.3, simulationtime=1.0, realtime_simulation=2.0))
    assert row["outcome"] is None and row["state"] is None and row["peak_Es"] is None
    assert row["peak_gap_std"] is None
    assert row["Es"] == 0.1


def test_trajectory_row_for_a_rejection_before_dynamics_has_only_what_is_known():
    row = trajectory_row("mol-0001-0000000002", dict(trajectory_outcome="rejected", simulationtime=0.0,
                                                     screen_reject_reason="topology_identity"))
    assert (row["model"], row["outcome"], row["reason"], row["simtime"]) == (1, "rejected", "topology_identity", 0.0)
    assert row["Es"] is None and row["realtime"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad", None], ids=["nan", "inf", "text", "none"])
def test_trajectory_row_replaces_values_json_cannot_hold(value):
    row = trajectory_row("mol-0000-0000000001", dict(Es=value))
    assert row["Es"] is None
    assert "NaN" not in json.dumps(row)


def test_trajectory_row_without_a_model_in_its_id():
    assert trajectory_row("test_builder", {})["model"] is None


def test_summary_records_results_and_members_of_failed_tasks(tmp_path):
    path = tmp_path / "new" / "summary.jsonl"
    summary = TrajectorySummary(path)
    summary.append([molecule(**BATCHED)], [molecule("mol-0003-0000000013", selected_state=1)])
    summary.append([molecule("mol-0004-0000000001", trajectory_outcome="completed")])
    columns = read_trajectory_summary(path)
    assert columns["model"] == [3, 3, 4]
    assert columns["outcome"] == ["rejected", "task_failed", "completed"]
    assert columns["state"] == [1, 1, None]
    assert set(columns) == set(COLUMNS)
    assert columns["Es"][0] == 0.1 and np.isnan(columns["Es"][1])


def test_summary_writes_nothing_for_no_results(tmp_path):
    TrajectorySummary(tmp_path / "summary.jsonl").append([])
    assert not list(tmp_path.iterdir())


def test_summary_write_failure_warns_once_and_does_not_raise(tmp_path, capsys):
    (tmp_path / "file").write_text("")
    summary = TrajectorySummary(tmp_path / "file" / "summary.jsonl")
    summary.append([molecule()])
    summary.append([molecule()])
    assert capsys.readouterr().out.count("Could not write the trajectory summary") == 1


def test_summary_survives_a_partial_last_line_from_a_crash(tmp_path):
    path = tmp_path / "summary.jsonl"
    path.write_text(json.dumps(trajectory_row("mol-0000-0000000001", {})) + '\n{"id": "mol-0000-00000')
    assert read_trajectory_summary(path)["model"] == [0]
    TrajectorySummary(path).append([molecule("mol-0001-0000000001")])
    assert read_trajectory_summary(path)["model"] == [0, 1], "the next row must not be lost with the partial one"


def test_summary_keeps_one_row_for_a_repeated_trajectory(tmp_path):
    path = tmp_path / "summary.jsonl"
    summary = TrajectorySummary(path)
    summary.append([molecule(trajectory_outcome="failed"), molecule("mol-0003-0000000099")])
    summary.append([molecule(trajectory_outcome="completed")])
    columns = read_trajectory_summary(path)
    assert columns["outcome"] == ["completed", None]


def test_summary_skips_lines_that_are_not_rows(tmp_path):
    path = tmp_path / "summary.jsonl"
    path.write_text('[1, 2]\n{"no id": 1}\n\n' + json.dumps(trajectory_row("mol-0000-0000000001", {})) + "\n")
    assert read_trajectory_summary(path)["model"] == [0]


# The coordinator.


@python_app(executors=["alf_sampler_executor"])
def batched_sampler(molecule_objects):
    return molecule_objects


@python_app(executors=["alf_sampler_executor"])
def single_sampler(molecule_object):
    return molecule_object


def coordinator(sampler_task, tmp_path=None, **master):
    settings = {"Escut": 1.0, "Fscut": 1.0}
    return SamplerCoordinator(sampler_task, "sampler", settings, master or None)


def test_coordinator_writes_nothing_without_the_setting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = [molecule()]
    assert coordinator(single_sampler).harvest(results) == results
    assert not list(tmp_path.iterdir())


def test_coordinator_records_every_result_including_those_not_selected(tmp_path):
    path = tmp_path / "summary.jsonl"
    selected, dropped = molecule("mol-0000-0000000001"), molecule("mol-0000-0000000002")
    dropped.update_atoms(None)
    assert coordinator(batched_sampler, trajectory_summary_path=str(path)).harvest([selected, dropped]) == [selected]
    assert read_trajectory_summary(path)["model"] == [0, 0]


def test_coordinator_records_members_of_failed_batched_tasks(tmp_path):
    path = tmp_path / "summary.jsonl"
    sampling = coordinator(batched_sampler, trajectory_summary_path=str(path))
    sampling.queue.failed_inputs = [molecule("mol-0000-0000000003")]
    sampling.harvest([])
    assert read_trajectory_summary(path)["outcome"] == ["task_failed"]
    assert sampling.queue.failed_inputs == []


def test_coordinator_failed_summary_does_not_change_the_results(tmp_path):
    (tmp_path / "file").write_text("")
    results = [molecule()]
    assert coordinator(single_sampler, trajectory_summary_path=str(tmp_path / "file" / "s.jsonl")).harvest(results) == results


def test_summary_and_plots_through_the_real_coordinator_without_meta_dir(tmp_path, monkeypatch):
    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(WATER.copy(), moleculeid)

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        molecule_object.store_results({"energy": -1.0, "forces": np.zeros((len(molecule_object), 3))})
        molecule_object.set_converged_flag(True)
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        (tmp_path / MODEL_PATTERN.format(current_training_id)).mkdir(parents=True)
        return [True] * 4, current_training_id

    sampler_config = dict(dt=0.5, maxt=0.05, Escut=0.012, Fscut=100.0, Ncheck=5, distcut=0.5, random_seed=3,
                          srt_temp=[2500.0, 3500.0], end_temp=[2500.0, 3500.0], amp_temp=[0.0, 200.0], per_temp=[0.01, 0.05],
                          amp_dens=None, per_dens=None, end_dens=None, device="cpu",
                          ase_calculator="tests.helpers.fakes.load_spring_ensemble",
                          driver={"strategy": "batched"}, batching={"batch_size": 2, "max_wait_seconds": 0.05})
    run_alf_driver(
        tmp_path, monkeypatch, builder=builder, sampler=batched_mlmd.mlmd_trajectories_task, qm=qm, ml=ml,
        sampler_config=sampler_config, bootstrap_set=2, parallel_samplers=4, target_queued_QM=4, save_h5_threshold=3,
        maximum_completed_QM=8, gpus_per_node=0, trajectory_summary_path="progress/summary.jsonl", plot_dir="status_plots/",
    )
    path = tmp_path / "progress/summary.jsonl"
    columns = read_trajectory_summary(path)
    assert columns["model"] and set(columns["outcome"]) <= {"eligible", "completed", "rejected", "failed"}
    assert not np.isnan(columns["peak_Es"]).any()
    assert not list(Path(tmp_path).glob("**/metadata-*.p"))

    master = load_config_file(str(tmp_path / "master.json"))
    arguments = build_input_dict(analysis_plot, [master, load_config_file(str(tmp_path / "sampler.json"), master["master_directory"])])
    assert arguments["trajectory_summary_path"] == str(path)
    written = analysis_plot(**arguments)
    assert {"simtime", "Es", "Fs", "Fsmax", "outcomes"} <= set(written)
    assert (tmp_path / "status_plots/analysis-plot-outcomes.png").stat().st_size > 1000
