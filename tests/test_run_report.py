"""Run reports: reading a run's records into tables, and the figures and summary written from them.

The run here is synthetic: three training rounds, with a task timeline, Parsl block submissions, a trajectory
summary, label shards, member training summaries and coordinator memory samples, at known times.
"""

import csv
import json
import os

import h5py
import numpy as np
import pytest

from alframework.tools import run_records, run_report

T0 = 1_700_000_000.0
ROUND_FINISH = {0: T0 + 1000, 1: T0 + 2000, 2: T0 + 3000}  # when each training round's members finished


def _task(stage, start, seconds, outcome="ok", host="node-a", gpu=None):
    return dict(stage=stage, task=f"{stage}_task", items=4, start=start, end=start + seconds, seconds=seconds,
                outcome=outcome, host=host, pid=1, slurm_job="7", parsl_block="0", gpu_peak_gb=gpu)


def make_run(run_dir, rounds=3, with_memory=True, excited=False):
    """A run directory with every record a report reads; ``excited``: two selected states, gap spreads, and test
    errors per state, as an excited-state run records them."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "master_config.json").write_text(json.dumps(dict(
        h5_path="h5store/data-{:04d}.h5", model_path="models/ensemble-{:04d}",
        trajectory_summary_path="plots/trajectories.jsonl", gpus_per_node=4)))
    # Tasks: each cycle samples, labels and trains after the previous round finished.
    timeline = run_dir / "task_timeline"
    timeline.mkdir()
    records = [_task("builder", T0 + 10, 2)]
    for cycle in range(-1, rounds - 1):
        base = T0 + 100 if cycle < 0 else ROUND_FINISH[cycle] + 10
        records += [_task("sampler", base + i, 100 + 10 * i, gpu=3.0 + i) for i in range(4)]
        records += [_task("QM", base + 300 + i, 50, outcome="ok" if i else "TimeoutError", host=f"node-{i}")
                    for i in range(3)]
        records += [_task("ML", base + 500, 400)]
    with open(timeline / "node-a-1.jsonl", "w") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in records)
    # One block per executor, each submitted 60 s before its stage's first task.
    scripts = run_dir / "runinfo" / "000" / "submit_scripts"
    scripts.mkdir(parents=True)
    for executor, first in (("alf_sampler_executor", T0 + 10), ("alf_QM_executor", T0 + 400),
                            ("alf_ML_executor", T0 + 600)):
        (scripts / f"parsl.{executor}.block-0.{first - 60:.4f}").write_text("#!/bin/bash\n")
    # Trajectories: models 0..rounds-1, with the spread falling and fewer candidates as models improve.
    (run_dir / "plots").mkdir()
    with open(run_dir / "plots" / "trajectories.jsonl", "w") as handle:
        for model in range(rounds):
            for index in range(10):
                row = dict(id=f"mol-{model}-{index}", model=model,
                           outcome="eligible" if index < 6 - 2 * model else "completed",
                           simtime=1.0, realtime=10.0, Es=1.0 / (model + 1) + 0.01 * index,
                           Fs=0.5 / (model + 1), Fsmax=2.0)
                if excited:
                    # The gap spread belongs to a pair of states: adjacent pairs, recorded by their lower state.
                    row.update(state=index % 4, gap_std=0.2 / (model + 1) * (1 + index % 3),
                               gap_lower_state=index % 3)
                handle.write(json.dumps(row) + "\n")
    # Label shards: 100, 200, 300 labels.
    (run_dir / "h5store").mkdir()
    for number in range(rounds):
        with h5py.File(run_dir / "h5store" / f"data-{number:04d}.h5", "w") as handle:
            handle.create_group("water")["_id"] = np.arange(100 * (number + 1))
    # Members: two per round, with summaries written when the round finished.
    for number in range(rounds):
        for member in range(2):
            directory = run_dir / "models" / f"ensemble-{number:04d}" / f"model-{member:02d}"
            directory.mkdir(parents=True)
            path = directory / "training_summary.json"
            path.write_text(json.dumps(dict(
                epochs=3, epoch_seconds=[5.0, 4.0, 4.0 + member], average_epoch_time=4.5, gpus_per_member=2,
                metric={"valid": {"ForceRMSE": 1.0}},
                final_metric={"test": {f"{kind}{state}_{error}": (1 + state) / (number + 1) * scale
                                       for kind in "EF" for state in range(4)
                                       for error, scale in (("RMSE", 1.0), ("MAE", 0.8))} if excited else
                              {"ForceRMSE": 1.0 / (number + 1) + 0.1 * member, "ForceMAE": 0.8 / (number + 1),
                               "T-RMSE": 10.0 / (number + 1), "T-MAE": 8.0 / (number + 1),
                               "TperAtom-RMSE": 1.0 / (number + 1), "TperAtom-MAE": 0.7 / (number + 1)},
                              "valid": {"ForceRMSE": 0.9}})))
            os.utime(path, (ROUND_FINISH[number], ROUND_FINISH[number]))
    if with_memory:
        with open(run_dir / "coordinator_memory.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "pid", "ppid", "command", "rss_kib", "pss_kib"])
            for step in range(5):
                writer.writerow([T0 + 100 * step, 10, 1, "python -m alframework", 2**20 * (step + 1), 2**20 * step])
                writer.writerow([T0 + 100 * step, 11, 10, "interchange", 2**19, 2**19])
    return run_dir


def test_records_assign_each_task_the_model_in_use_when_it_started(tmp_path):
    records = run_records.run_records(make_run(tmp_path / "run"))
    tasks = records["tasks"]
    assert sorted(tasks["cycle"].unique()) == [-1, 0, 1]
    first_samplers = tasks[(tasks["stage"] == "sampler")].sort_values("start")
    assert first_samplers["cycle"].iloc[0] == -1 and first_samplers["cycle"].iloc[-1] == 1
    assert list(records["round_finishes"].index) == [0, 1, 2]


def test_a_run_started_from_a_copied_ensemble_counts_its_first_tasks_as_cycle_zero():
    import pandas as pd
    finishes = pd.Series({1: 100.0, 2: 200.0})  # round 0 was copied in, not trained
    assert list(run_records.assign_cycles([50.0, 150.0, 250.0], finishes)) == [0, 1, 2]
    assert list(run_records.assign_cycles([50.0], pd.Series(dtype=float))) == [-1]


def test_records_read_block_waits_labels_training_and_memory(tmp_path):
    records = run_records.run_records(make_run(tmp_path / "run"))
    waits = records["block_waits"].set_index("executor")["wait_seconds"]
    assert waits.round(3).to_dict() == {"alf_sampler_executor": 60.0, "alf_QM_executor": 60.0,
                                        "alf_ML_executor": 60.0}
    assert records["labels"]["labels"].tolist() == [100, 200, 300]
    assert records["labels"]["cumulative_labels"].tolist() == [100, 300, 600]
    training = records["training"]
    assert len(training) == 6 and set(training["round"]) == {0, 1, 2}
    assert training.loc[training["round"] == 2, "final.test.ForceRMSE"].min() == pytest.approx(1 / 3)
    memory = records["memory"]
    assert memory["coordinator_gib"].max() == pytest.approx(5.0)  # the root process's RSS
    assert memory["tree_gib"].max() == pytest.approx(4.5)  # PSS summed over the tree
    assert records["run_start"] == T0 - 50  # the first block's submission, before any task


def test_numbered_paths_find_rounds_after_a_removed_one(tmp_path):
    for number in (1, 2, 10):
        (tmp_path / f"ensemble-{number:04d}").mkdir()
    (tmp_path / "ensemble-other").mkdir()
    found = run_records.numbered_paths(str(tmp_path / "ensemble-{:04d}"))
    assert [number for number, _ in found] == [1, 2, 10]


def test_the_report_writes_every_figure_and_a_summary(tmp_path):
    out = tmp_path / "report"
    written = run_report.write_report(make_run(tmp_path / "run"), out)
    assert set(written) == {"run-report-stages.png", "run-report-task-times.png", "run-report-spread.png",
                            "run-report-labels.png", "run-report-training.png", "run-report-health.png",
                            "run_report.json"}
    summary = json.loads((out / "run_report.json").read_text())
    assert summary["skipped"] == []
    assert summary["stages"]["QM"]["failed"] == 3 and summary["stages"]["sampler"]["tasks"] == 12
    assert summary["stages"]["sampler"]["task_seconds"]["max"] == 130
    gaps = {(gap["cycle"], gap["between"]): gap["seconds"] for gap in summary["stage_gaps"]}
    # The last sampler starts 3 s after the cycle begins and takes 130 s; QM starts 300 s after it begins.
    assert gaps[(0, "sampler to QM")] == pytest.approx(300 - 133)
    spread = {row["model"]: row for row in summary["spread_per_cycle"]}
    assert spread[0]["candidate_share"] == pytest.approx(0.6) and spread[2]["candidate_share"] == pytest.approx(0.2)
    assert spread[0]["Es"]["median"] > spread[2]["Es"]["median"]
    assert [row["labels"] for row in summary["labels_per_round"]] == [100, 200, 300]
    rounds = {row["round"]: row for row in summary["training_rounds"]}
    assert rounds[1]["median_epoch_seconds"] == pytest.approx(4.5)
    assert summary["coordinator_memory"]["peak_coordinator_gib"] == pytest.approx(5.0)
    for name in written:
        assert (out / name).stat().st_size > 0


def test_the_summary_names_slow_hosts():
    import pandas as pd
    tasks = pd.DataFrame([dict(stage="QM", host="slow" if i < 3 else f"fast-{i % 4}", seconds=300 if i < 3 else 100)
                          for i in range(15)])
    slow = run_report.slow_hosts(tasks)
    assert [row["host"] for row in slow] == ["slow"] and slow[0]["median_seconds"] == 300


def test_a_run_without_records_gets_a_summary_saying_what_was_missing(tmp_path):
    (tmp_path / "run").mkdir()
    written = run_report.write_report(tmp_path / "run", tmp_path / "report")
    assert written == ["run_report.json"]
    summary = json.loads((tmp_path / "report" / "run_report.json").read_text())
    assert len(summary["skipped"]) == len(run_report.FIGURES)


def test_the_command_writes_the_report_into_the_run(tmp_path, capsys):
    run_dir = make_run(tmp_path / "run", with_memory=False)
    run_report.main([str(run_dir)])
    assert (run_dir / "run_report" / "run-report-health.png").exists()
    assert "run_report.json" in capsys.readouterr().out


def test_the_coordinator_refreshes_the_report_in_its_own_process(tmp_path):
    from alframework.tools.progress_plots import ProgressPlots
    run_dir = make_run(tmp_path / "run")
    updates = ProgressPlots("alframework.tools.run_report.live_report")
    assert updates.update({"run_dir": str(run_dir), "out_dir": str(tmp_path / "live"),
                           "master_config_path": str(run_dir / "master_config.json")})
    updates.wait(timeout=120)
    assert updates.process.returncode == 0
    assert json.loads((tmp_path / "live" / "run_report.json").read_text())["skipped"] == []


def test_an_excited_state_run_is_reported_per_state(tmp_path):
    out = tmp_path / "report"
    written = run_report.write_report(make_run(tmp_path / "run", excited=True), out)
    assert "run-report-spread.png" in written and "run-report-training.png" in written
    summary = json.loads((out / "run_report.json").read_text())
    assert summary["skipped"] == []
    cycle = {row["model"]: row for row in summary["spread_per_cycle"]}[0]
    assert set(cycle["states"]) == {"0", "1", "2", "3"} and cycle["gap_std"]["median"] > 0
    assert set(cycle["gap_pairs"]) == {"0", "1", "2"} and cycle["gap_pairs"]["1"]["median"] == pytest.approx(0.4)
    assert "gap_std" not in cycle["states"]["1"], "a gap is a pair's, not the selected state's"
    # Every state's errors are kept, not only the first few metrics.
    last = {row["round"]: row for row in summary["training_rounds"]}[2]
    assert last["final.test.F2_RMSE"] == pytest.approx(1.0) and last["final.test.E0_RMSE"] == pytest.approx(1 / 3)
    assert run_report._state_metrics(run_records.run_records(tmp_path / "run")["training"])["F"][2] == \
        {"RMSE": "final.test.F2_RMSE", "MAE": "final.test.F2_MAE"}
    assert last["final.test.F2_MAE"] == pytest.approx(0.8), "MAE is kept with RMSE"


def test_sampling_gpu_time_per_candidate_comes_from_the_cycle_s_sampler_tasks(tmp_path):
    records = run_records.run_records(make_run(tmp_path / "run"))
    rows = {row["model"]: row for row in run_report.spread_per_cycle(records["trajectories"], records["tasks"])}
    # Cycle 0's four sampler tasks take 100 to 130 s; model 0 gave six candidates.
    assert rows[0]["sampling_gpu_hours_per_candidate"] == pytest.approx(460 / 3600 / 6)
    assert "sampling_gpu_hours_per_candidate" not in rows[2]  # no sampler task ran with model 2


def test_every_state_keeps_a_line_past_the_palette():
    from alframework.tools.plot_style import CATEGORICAL, OVERFLOW, series_colors
    colors = [color for color, _ in series_colors(range(11))]
    assert colors[:8] == list(CATEGORICAL) and colors[8:] == [OVERFLOW] * 3


def test_ground_state_errors_keep_rmse_and_mae_for_each_quantity(tmp_path):
    training = run_records.run_records(make_run(tmp_path / "run"))["training"]
    ground = run_report._ground_metrics(training)
    assert list(ground) == ["T", "TperAtom", "Force"], "energy first"
    assert ground["Force"] == {"RMSE": "final.test.ForceRMSE", "MAE": "final.test.ForceMAE"}
