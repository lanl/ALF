"""Progress plots: real PNG files from a trajectory summary or from metadata pickles.

Uses made-up metadata in the form the samplers record, so it needs no models. See
``test_batched_mlmd.py`` for how samplers fill it in, and ``test_trajectory_summary.py`` for a run
through the real coordinator.
"""

import json
import pickle

import numpy as np
import pytest
from matplotlib.axes import Axes

from alframework.tools.plotting import OUTCOME_COLORS, _chosen_models, _table, analysis_plot
from alframework.tools.config import build_input_dict
from alframework.tools.trajectory_summary import trajectory_row

NUMERIC = ("realtime", "Es", "Fs", "Fsmax", "peak_Es", "peak_Fs", "peak_Fsmax")
LEGACY_PLOTS = {"realtime", "log_realtime", "simtime", "log_simtime", "Es", "Fs", "Fsmax"}


def finished(**extra):
    return {"trajectory_outcome": "completed", "simulationtime": 0.5, "realtime_simulation": 3.0, "Es": 0.01, "Fs": 0.02,
            "Fsmax": 0.03, "peak_Es": 0.04, "peak_Fs": 0.05, "peak_Fsmax": 0.06, **extra}


def rejected_keeping_uncertainty(**extra):
    return finished(**{"trajectory_outcome": "rejected", **extra})


def rejected_at_once(**extra):
    return dict(trajectory_outcome="rejected", simulationtime=0.0, realtime_simulation=0.0, Es=None, Fs=None, Fsmax=None,
                peak_Es=None, peak_Fs=None, peak_Fsmax=None, **extra)


LEGACY = dict(simulationtime=0.5, realtime_simulation=3.0, Es=0.01, Fs=0.02, Fsmax=0.03)


def write_summary(path, trajectories):
    with open(path, "w") as handle:
        for number, (model, metadata) in enumerate(trajectories):
            handle.write(json.dumps(trajectory_row(f"mol-{model:04d}-{number:010d}", metadata)) + "\n")
    return str(path)


def write_pickles(directory, trajectories):
    directory.mkdir(exist_ok=True)
    for number, (model, metadata) in enumerate(trajectories):
        with open(directory / f"metadata-mol-{model:04d}-{number:010d}.p", "wb") as handle:
            pickle.dump(metadata, handle)
    return str(directory)


def plot(tmp_path, trajectories, source):
    """Plot from either source, returning the plot names and the directory."""
    plots = tmp_path / "plots"
    if source == "summary":
        written = analysis_plot(plot_dir=str(plots), trajectory_summary_path=write_summary(tmp_path / "s.jsonl", trajectories))
    else:
        written = analysis_plot(plot_dir=str(plots), meta_dir=write_pickles(tmp_path / "meta", trajectories))
    return set(written), plots


@pytest.fixture(params=["summary", "pickles"])
def source(request):
    return request.param


def test_analysis_plot_gives_uncertainty_plots_for_rejected_trajectories(tmp_path, source):
    trajectories = [(model, rejected_keeping_uncertainty()) for model in range(3) for _ in range(3)]
    written, plots = plot(tmp_path, trajectories, source)
    assert LEGACY_PLOTS | {"outcomes"} == written
    for name in written:
        assert (plots / f"analysis-plot-{name}.png").stat().st_size > 1000


def test_analysis_plot_skips_and_reports_plots_with_no_data(tmp_path, source, capsys):
    written, plots = plot(tmp_path, [(0, rejected_at_once()), (1, dict(rejected_at_once(), simulationtime=0.5))], source)
    assert "Es" not in written and not (plots / "analysis-plot-Es.png").exists()
    assert "simtime" in written
    assert "skipped for lack of data" in capsys.readouterr().out


def test_analysis_plot_treats_zero_times_as_missing_on_log_plots(tmp_path, source):
    written, _ = plot(tmp_path, [(0, rejected_at_once()), (0, finished())], source)
    assert {"simtime", "log_simtime"} <= written
    table = _table({name: [row[name] for row in (trajectory_row("mol-0000-0000000001", rejected_at_once()),)]
                    for name in trajectory_row("x", {}) if name != "id"})
    assert np.isnan(table["log_simtime"].iloc[0])


def test_analysis_plot_gives_the_original_plots_for_legacy_metadata(tmp_path, source):
    written, _ = plot(tmp_path, [(model, dict(LEGACY)) for model in range(2) for _ in range(2)], source)
    assert written == LEGACY_PLOTS, "legacy samplers record no outcomes, so there is no outcome plot"


def test_analysis_plot_uses_peak_uncertainty_when_recorded():
    table = _table({name: [row[name] for row in (trajectory_row("mol-0000-0000000001", finished()),
                                                 trajectory_row("mol-0000-0000000002", LEGACY))]
                    for name in trajectory_row("x", {}) if name != "id"})
    assert table["Es"].tolist() == pytest.approx([0.04, 0.01])


def test_analysis_plot_titles_say_when_uncertainty_is_a_peak(tmp_path, monkeypatch):
    titles = []
    real = Axes.set_title
    monkeypatch.setattr(Axes, "set_title", lambda self, title, **kw: (titles.append(title), real(self, title, **kw))[1])
    plot(tmp_path, [(0, finished())], "summary")
    assert "Peak Energy Uncertainty" in titles
    titles.clear()
    plot(tmp_path, [(0, dict(LEGACY))], "summary")
    assert "Energy Uncertainty" in titles


def test_analysis_plot_reads_per_task_subdirectories(tmp_path):
    from alframework.tools.plotting import _read_pickles
    meta = tmp_path / "meta"
    write_pickles(meta, [(0, finished())])
    write_pickles(meta / "task-mol-0000-0000000005", [(0, finished()), (0, finished())])
    (meta / "other").mkdir()
    write_pickles(meta / "other", [(0, finished())])  # only task-* subdirectories are per-task layout
    assert len(_read_pickles(str(meta))["model"]) == 3


def test_analysis_plot_reads_one_file_per_task(tmp_path):
    import pickle
    from alframework.tools.plotting import _read_pickles
    meta = tmp_path / "meta"
    meta.mkdir()
    records = [dict(finished(), moleculeid=f"mol-0001-{n:010d}") for n in range(3)]
    with open(meta / "metadata-task-mol-0001-0000000000.p", "wb") as handle:
        pickle.dump(records, handle)
    columns = _read_pickles(str(meta))
    assert columns["model"] == [1, 1, 1]


def test_analysis_plot_ignores_a_truncated_pickle_and_other_files(tmp_path):
    meta = tmp_path / "meta"
    write_pickles(meta, [(0, finished())])
    (meta / "metadata-mol-0000-0000000099.p").write_bytes(b"\x80\x04trunc")
    (meta / "metadata-mol-boot-0000000099.p").write_bytes(b"not read")
    (meta / "notes.txt").write_text("not read")
    assert "simtime" in analysis_plot(meta_dir=str(meta), plot_dir=str(tmp_path / "plots"))


def test_analysis_plot_tolerates_metadata_with_missing_keys(tmp_path, source):
    written, _ = plot(tmp_path, [(0, {}), (1, {"simulationtime": 0.5}), (1, finished())], source)
    assert "simtime" in written


def test_analysis_plot_draws_a_single_model(tmp_path, source):
    written, _ = plot(tmp_path, [(0, finished())], source)
    assert "simtime" in written


def test_analysis_plot_accepts_model_numbers_above_9999(tmp_path, source):
    written, _ = plot(tmp_path, [(12345, finished())], source)
    assert "simtime" in written


def test_chosen_models_are_real_models_spread_across_the_run():
    table = _table({"model": [0, 3, 7, 20, 21, 90], "simtime": [1.0] * 6, **{name: [None] * 6 for name in NUMERIC}})
    chosen = _chosen_models(table, 3)
    assert chosen.tolist() == [0, 7, 90]
    assert _chosen_models(table, 50).tolist() == [0, 3, 7, 20, 21, 90]


def test_analysis_plot_limits_how_many_models_it_shows(tmp_path):
    trajectories = [(model, finished()) for model in range(30)]
    ticks = []
    real = Axes.set_xticks

    def spy(self, positions, labels=None, **kw):
        ticks.append(list(labels))
        return real(self, positions, labels, **kw)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Axes, "set_xticks", spy)
        analysis_plot(plot_dir=str(tmp_path / "p"), meta_n_cols=5, trajectory_summary_path=write_summary(tmp_path / "s.jsonl", trajectories))
    assert all(len(labels) == 5 and labels[0] == "0" and labels[-1] == "29" for labels in ticks)


def test_analysis_plot_splits_by_state_only_with_several_states(tmp_path, source):
    one, _ = plot(tmp_path, [(0, finished(selected_state=1)), (1, finished(selected_state=1))], source)
    assert not any("state" in name for name in one)
    several, plots = plot(tmp_path, [(0, finished(selected_state=1)), (0, finished(selected_state=2)),
                                     (1, rejected_at_once(selected_state=2))], source)
    assert {"Es-state1", "Es-state2", "simtime-state1", "simtime-state2"} <= several
    assert (plots / "analysis-plot-Fsmax-state2.png").exists()


def test_outcome_plot_appears_only_when_outcomes_are_recorded(tmp_path):
    written, _ = plot(tmp_path, [(0, dict(LEGACY))], "summary")
    assert "outcomes" not in written
    written, _ = plot(tmp_path, [(0, finished()), (0, rejected_at_once()), (1, dict(trajectory_outcome="task_failed"))], "summary")
    assert "outcomes" in written


def test_outcome_plot_keeps_each_outcome_on_its_own_color(tmp_path, monkeypatch):
    bars = []
    real = Axes.bar

    def spy(self, x, height, width=0.8, bottom=None, **kw):
        bars.append((kw["label"], kw["color"], list(height), list(bottom)))
        return real(self, x, height, width, bottom, **kw)

    monkeypatch.setattr(Axes, "bar", spy)
    plot(tmp_path, [(0, finished()), (0, rejected_at_once()), (0, rejected_at_once()), (1, rejected_at_once())], "summary")
    # Shares of each model's trajectories: model 0 has one completed and two rejected, model 1 one rejected.
    assert bars[0][:2] == ("completed", OUTCOME_COLORS["completed"])
    assert bars[0][2] == pytest.approx([100 / 3, 0])
    assert bars[1][0] == "rejected" and bars[1][1] == OUTCOME_COLORS["rejected"]
    assert bars[1][2] == pytest.approx([200 / 3, 100]) and bars[1][3] == pytest.approx([100 / 3, 0]), \
        "bars must stack"


def test_outcome_plot_gives_unknown_outcomes_a_color(tmp_path):
    written, _ = plot(tmp_path, [(0, dict(trajectory_outcome="surprising"))], "summary")
    assert "outcomes" in written


def test_analysis_plot_marks_where_sampling_windows_start(tmp_path, monkeypatch):
    lines = []
    real = Axes.axvline
    monkeypatch.setattr(Axes, "axvline", lambda self, x, **kw: (lines.append(x), real(self, x, **kw))[1])
    trajectories = [(0, finished(sampling_window=0)), (1, finished(sampling_window=0)), (2, finished(sampling_window=1))]
    written, _ = plot(tmp_path, trajectories, "summary")
    assert len(lines) == len(written), "one line on every plot"
    assert set(lines) == {1.5}


def test_analysis_plot_draws_no_window_lines_without_windows(tmp_path, monkeypatch):
    lines = []
    monkeypatch.setattr(Axes, "axvline", lambda self, x, **kw: lines.append(x))
    plot(tmp_path, [(0, finished()), (1, finished())], "summary")
    assert not lines


def test_analysis_plot_closes_its_figures(tmp_path):
    import matplotlib.pyplot as plt

    plot(tmp_path, [(0, finished(selected_state=1)), (1, finished(selected_state=2))], "summary")
    assert not plt.get_fignums()


def test_analysis_plot_explains_when_it_has_no_source(tmp_path, capsys):
    assert analysis_plot(plot_dir=str(tmp_path)) == []
    assert "need plot_dir, and either" in capsys.readouterr().out
    assert analysis_plot(meta_dir=str(tmp_path)) == []


def test_analysis_plot_waits_quietly_for_a_summary_that_does_not_exist_yet(tmp_path, capsys):
    assert analysis_plot(plot_dir=str(tmp_path / "p"), trajectory_summary_path=str(tmp_path / "later.jsonl")) == []
    assert capsys.readouterr().out == ""


def test_analysis_plot_prefers_the_summary_over_pickles(tmp_path):
    meta = write_pickles(tmp_path / "meta", [(0, dict(LEGACY))])
    summary = write_summary(tmp_path / "s.jsonl", [(0, finished())])
    assert "outcomes" in analysis_plot(meta_dir=meta, plot_dir=str(tmp_path / "p"), trajectory_summary_path=summary)


def test_analysis_plot_finds_its_arguments_in_configs():
    arguments = build_input_dict(analysis_plot, [{"plot_dir": "plots/", "builder_task": "x"}, {"meta_dir": "sampling/"}])
    assert arguments == {"plot_dir": "plots/", "meta_dir": "sampling/"}


def candidate(**extra):
    return finished(**{"trajectory_outcome": "eligible", "peak_Es": 0.3, "peak_Fs": 0.05, "peak_Fsmax": 0.06, **extra})


CUTOFFS = dict(Escut=0.2, Fscut=0.1)


def test_uncertainty_plots_draw_the_sampler_threshold(tmp_path, monkeypatch):
    lines = []
    real = Axes.axhline
    monkeypatch.setattr(Axes, "axhline", lambda self, y=0, **kw: (lines.append(y), real(self, y, **kw))[1])
    analysis_plot(plot_dir=str(tmp_path / "p"), trajectory_summary_path=write_summary(
        tmp_path / "s.jsonl", [(0, finished()), (0, candidate())]), **CUTOFFS)
    # Es at Escut, Fs at Fscut, Fsmax at 3 * Fscut (the samplers' rule), and 1 on the trigger-time plot.
    for threshold in (0.2, 0.1, 0.3, 1.0):
        assert any(line == pytest.approx(threshold) for line in lines), threshold


def test_tuning_and_trigger_plots_need_the_cutoffs(tmp_path):
    trajectories = [(0, finished()), (0, candidate()), (1, candidate()), (1, finished())]
    summary = write_summary(tmp_path / "s.jsonl", trajectories)
    written = analysis_plot(plot_dir=str(tmp_path / "a"), trajectory_summary_path=summary)
    assert "cutoff-tuning" not in written and "trigger-time" not in written
    written = analysis_plot(plot_dir=str(tmp_path / "b"), trajectory_summary_path=summary, **CUTOFFS)
    assert {"cutoff-tuning", "trigger-time", "sampling-cost"} <= set(written)


def test_a_trajectory_triggers_on_its_worst_eligible_signal():
    from alframework.tools.plotting import _cutoffs, _eligible_signals, _trigger_ratio
    import pandas as pd
    thresholds = _cutoffs(Escut=0.2, Fscut=0.1, gap_std_cut=0.05)
    assert thresholds == {"Es": 0.2, "Fs": 0.1, "Fsmax": pytest.approx(0.3), "gap_std": 0.05}
    table = pd.DataFrame({"Es": [0.1, 0.1], "Fs": [0.05, 0.25], "Fsmax": [0.06, 0.06], "gap_std": [0.2, 0.0]})
    default = _eligible_signals(None, thresholds)
    assert default == ["Es", "Fs", "Fsmax"], "the gap counts only when acquisition.eligibility names it"
    assert _trigger_ratio(table, default, thresholds).tolist() == pytest.approx([0.5, 2.5])
    with_gap = _eligible_signals({"eligibility": ["energy", "gap"]}, thresholds)
    assert with_gap == ["Es", "gap_std"]
    assert _trigger_ratio(table, with_gap, thresholds).tolist() == pytest.approx([4.0, 0.5])


def test_excited_state_gaps_get_one_plot_per_pair_of_states(tmp_path):
    trajectories = [(0, finished(selected_state=state, gap_std=0.1, peak_gap_std=0.2, gap_std_lower_state=lower))
                    for state in (1, 2) for lower in (0, 1, 2)]
    written, _ = plot(tmp_path, trajectories, "summary")
    assert {"gap_std", "gap_std-states0-1", "gap_std-states1-2", "gap_std-states2-3"} <= set(written)
    assert "gap_std-state1" not in written, "a gap is a pair's, not the selected state's"


def test_ground_state_runs_do_not_report_a_missing_gap_plot(tmp_path, capsys):
    written, _ = plot(tmp_path, [(0, finished())], "summary")
    assert "gap_std" not in written and "gap_std" not in capsys.readouterr().out


def test_rejections_are_plotted_by_reason(tmp_path):
    trajectories = [(0, rejected_at_once(screen_reject_reason=reason)) for reason in ("min_distance", "max_force")]
    written, _ = plot(tmp_path, trajectories, "summary")
    assert "rejections" in written


def test_sampling_cost_needs_candidates(tmp_path):
    written, _ = plot(tmp_path, [(0, finished())], "summary")
    assert "sampling-cost" not in written
    written, _ = plot(tmp_path, [(0, finished()), (0, candidate())], "summary")
    assert "sampling-cost" in written


def test_the_cutoffs_and_acquisition_come_from_the_sampler_config():
    arguments = build_input_dict(analysis_plot, [{"plot_dir": "plots/"}, {"Escut": 0.2, "Fscut": 0.1,
                                                                          "acquisition": {"eligibility": ["gap"]}}])
    assert arguments["Escut"] == 0.2 and arguments["acquisition"] == {"eligibility": ["gap"]}
