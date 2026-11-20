"""A report on an ALF run, finished or running: figures and a JSON summary from what the run records.

  python -m alframework.tools.run_report RUN_DIR [--out DIR] [--master master_config.json] [--timeline DIR]

It reads the records described in :mod:`alframework.tools.run_records` and writes, to ``--out`` (default
``RUN_DIR/run_report``):

- ``run-report-stages.png``: each stage's GPUs busy and nodes in use over the run, as percentages of what it ran
  on (``gpus_per_node`` in the master config), with its failed tasks shaded cumulatively, the waits for Parsl
  blocks shaded, and the time each training round finished marked;
- ``run-report-task-times.png``: each stage's task times per cycle (median, 90th percentile and longest);
- ``run-report-spread.png``: the ensemble's energy and force spread per cycle, and the share of trajectories
  that gave QM a candidate;
- ``run-report-labels.png``: labels stored per training round, and in total;
- ``run-report-training.png``: epoch time and the final test errors per round, one point per member;
- ``run-report-health.png``: GPU memory peaks per stage, the coordinator's memory over the run, and failed tasks;
- ``run_report.json``: the numbers behind them, with the gaps between stages and the slowest hosts.

A figure whose records are missing is skipped, and the summary lists what was missing. With ``run_report: true``
in the master config, the coordinator refreshes the report with its progress plots, in a separate process.
"""
import argparse
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # This runs on a head node, usually without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alframework.tools.plot_style import GRID, MUTED, NOTE, RC, SHADE as QUEUE_SHADE, TEXT, series_colors, title_case
from alframework.tools.plot_style import figure as _figure, key, label_end as _label_end, panel as _panel
from alframework.tools.plot_style import save
from alframework.tools.run_records import load_master_config, run_records

# Stages in pipeline order, each with a fixed color from the reference palette.
STAGES = (("builder", "Builders", "#898781"), ("sampler", "Sampling", "#2a78d6"), ("QM", "QM labeling", "#eb6834"),
          ("ML", "ML training", "#24916b"))
STAGE_NAME = {stage: name for stage, name, _ in STAGES}
STAGE_COLOR = {stage: color for stage, _, color in STAGES}
MEMBER_COLOR = "#2a78d6"
SECONDARY = "#eb6834"


def _time_axis(span_seconds):
    """Minutes for a run of up to three hours, hours beyond."""
    return (60.0, "Minutes since the run started") if span_seconds <= 3 * 3600 else \
        (3600.0, "Hours since the run started")


def _save(figure, out_dir, name, written):
    save(figure, os.path.join(out_dir, name))
    written.append(name)


def _percentiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return dict(count=0)
    return dict(count=int(values.size), median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                max=float(values.max()), min=float(values.min()))


def _clean(value):
    """``value`` as strict JSON: non-finite numbers become ``null``, NumPy scalars plain numbers."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


# ---- Summary numbers ----

def stage_summary(tasks):
    """Per stage: tasks, failures, items, wall span, and task-time percentiles."""
    summary = {}
    for stage, group in tasks.groupby("stage"):
        summary[stage] = dict(tasks=int(len(group)), failed=int((group["outcome"] != "ok").sum()),
                              items=int(pd.to_numeric(group["items"], errors="coerce").fillna(0).sum()),
                              first_start=float(group["start"].min()), last_end=float(group["end"].max()),
                              wall_seconds=float(group["end"].max() - group["start"].min()),
                              task_seconds=_percentiles(group["seconds"]))
    return summary


def stage_gaps(tasks):
    """Per cycle, the time from the last task of one stage to the first of the next, in pipeline order:
    sampling to QM, and QM to ML. Negative when the next stage started before the previous one ended."""
    gaps = []
    for cycle, group in tasks.groupby("cycle"):
        ends = group.groupby("stage")["end"].max()
        starts = group.groupby("stage")["start"].min()
        for before, after in (("sampler", "QM"), ("QM", "ML")):
            if before in ends and after in starts:
                gaps.append(dict(cycle=int(cycle), between=f"{before} to {after}",
                                 seconds=float(starts[after] - ends[before])))
    return gaps


def slow_hosts(tasks, factor=1.5, least_tasks=3):
    """Hosts whose median task time in a stage is at least ``factor`` times that stage's median."""
    slow = []
    for stage, group in tasks.groupby("stage"):
        overall = group["seconds"].median()
        if not overall or not np.isfinite(overall):
            continue
        per_host = group.groupby("host")["seconds"].agg(["median", "count"])
        for host, row in per_host[(per_host["count"] >= least_tasks)
                                  & (per_host["median"] >= factor * overall)].iterrows():
            slow.append(dict(stage=stage, host=host, median_seconds=float(row["median"]), tasks=int(row["count"]),
                             stage_median_seconds=float(overall)))
    return sorted(slow, key=lambda row: -row["median_seconds"] / row["stage_median_seconds"])


SPREADS = (("Es", "Energy spread", "eV"), ("Fs", "Mean force spread", "eV/Å"), ("Fsmax", "Largest force spread",
                                                                                   "eV/Å"),
           ("gap_std", "Gap spread", "eV"))


def _whole(value):
    """A state number as an integer when it is one (a column with gaps reads as floats), else as it is."""
    try:
        return int(value) if float(value).is_integer() else value
    except (TypeError, ValueError):
        return value


def _peak(group, name):
    return group[f"peak_{name}"].where(group[f"peak_{name}"].notna(), group[name])


def spread_per_cycle(trajectories, tasks=None):
    """Per model: trajectories, QM candidates and their share, the spread percentiles (using each trajectory's peak),
    per selected state when a run has several, the gap spread per pair of states when the pair is recorded, and the
    sampling GPU-hours per candidate when the task timeline has the cycle's sampler tasks."""
    rows = []
    states = sorted({_whole(state) for state in trajectories["state"].dropna().unique()}, key=str) \
        if "state" in trajectories else []
    for model, group in trajectories.groupby("model"):
        outcomes = group["outcome"].dropna()
        candidates = int((outcomes == "eligible").sum())
        row = dict(model=int(model), trajectories=int(len(group)), candidates=candidates,
                   candidate_share=float((outcomes == "eligible").mean()) if len(outcomes) else None)
        for name, _, _ in SPREADS:
            if name in group and _peak(group, name).notna().any():
                row[name] = _percentiles(_peak(group, name))
        if len(states) > 1:
            row["states"] = {}
            for state in states:
                part = group[group["state"].map(_whole) == state]
                if len(part):
                    known = part["outcome"].dropna()
                    row["states"][str(state)] = dict(
                        trajectories=int(len(part)),
                        candidate_share=float((known == "eligible").mean()) if len(known) else None,
                        **{name: _percentiles(_peak(part, name)) for name, _, _ in SPREADS
                           if name != "gap_std" and name in part and _peak(part, name).notna().any()})
        # A gap belongs to a pair of states, not to the state a trajectory ran on: with adjacent pairs, the sampler
        # records the lower state of the pair whose gap spread was largest.
        if "gap_lower_state" in group and group["gap_lower_state"].notna().any():
            row["gap_pairs"] = {}
            for lower, part in group.groupby("gap_lower_state"):
                values = _peak(part, "gap_std")
                if values.notna().any():
                    row["gap_pairs"][str(int(float(lower)))] = _percentiles(values)
        if tasks is not None and not tasks.empty and candidates:
            sampler = tasks[(tasks["stage"] == "sampler") & (tasks["cycle"] == model)]
            if len(sampler):
                row["sampling_gpu_hours_per_candidate"] = float(sampler["seconds"].sum() / 3600 / candidates)
        rows.append(row)
    return rows


# ---- Figures ----

def _bins(t0, end):
    width = max((end - t0) / 400, 5.0)
    edges = np.arange(t0, end + width, width)
    return edges, edges[:-1] + width / 2


def _occupancy(stage_tasks, edges, weight=1.0):
    """Per bin: the weighted number of tasks running (averaged over the bin) and the hosts running any."""
    width = edges[1] - edges[0]
    busy = np.zeros(len(edges) - 1)
    hosts = [set() for _ in busy]
    for start, end, host in stage_tasks[["start", "end", "host"]].itertuples(index=False):
        first, last = int((start - edges[0]) // width), int((end - edges[0]) // width)
        for index in range(max(first, 0), min(last, len(busy) - 1) + 1):
            overlap = min(end, edges[index + 1]) - max(start, edges[index])
            if overlap > 0:
                busy[index] += weight * overlap / width
                hosts[index].add(host)
    return busy, np.array([len(names) for names in hosts], dtype=float)


def _gpus_per_task(records, stage):
    """GPUs one task of ``stage`` holds: a training member's ``gpus_per_member``, else one."""
    training = records["training"]
    if stage == "ML" and not training.empty and "gpus_per_member" in training:
        value = pd.to_numeric(training["gpus_per_member"], errors="coerce").median()
        return float(value) if np.isfinite(value) and value > 0 else 1.0
    return 1.0


def _minutes(seconds):
    return f"{seconds / 60:,.1f} min" if seconds < 3 * 3600 else f"{seconds / 3600:,.1f} h"


def plot_stages(records, out_dir, written):
    """Each stage's GPUs busy and nodes in use, as percentages of what it ran on, with cumulative failures shaded;
    the waits for Parsl blocks shaded gray; dashed lines where models finished."""
    tasks, waits = records["tasks"], records["block_waits"]
    if tasks.empty:
        return False
    gpus_per_node = records.get("gpus_per_node")
    stages = [stage for stage, _, _ in STAGES if (tasks["stage"] == stage).any()]
    t0 = records["run_start"]
    end = max(tasks["end"].max(), *records["round_finishes"].tolist()) if len(records["round_finishes"]) \
        else tasks["end"].max()
    scale, xlabel = _time_axis(end - t0)
    edges, centers = _bins(t0, end)
    x = (centers - t0) / scale
    figure, axes = _figure(len(stages), "Stages over the run",
                           "Percentages are of the GPUs and nodes each stage ran on. Waiting for a block covers the "
                           "scheduler's queue and the block's startup.")
    for axis, stage in zip(axes, stages):
        color, group = STAGE_COLOR[stage], tasks[tasks["stage"] == stage]
        for wait in waits[waits["stage"].str.split("/").apply(lambda names: stage in names)].itertuples():
            if np.isfinite(wait.first_task):
                axis.axvspan((wait.submitted - t0) / scale, (wait.first_task - t0) / scale, color=QUEUE_SHADE, lw=0,
                             zorder=0)
        per_task = _gpus_per_task(records, stage)
        busy, hosts = _occupancy(group, edges, per_task)
        cpu_stage = stage == "builder" or not gpus_per_node
        # A task on several nodes (a multi-node training member) records only its first node's host.
        nodes_per_task = 1 if cpu_stage else max(1, math.ceil(per_task / float(gpus_per_node)))
        nodes = group["host"].nunique() * nodes_per_task
        capacity = max(busy.max(), 1e-9) if cpu_stage else nodes * float(gpus_per_node)
        busy_pct, nodes_pct = 100 * busy / capacity, 100 * hosts * nodes_per_task / max(nodes, 1)
        failed = group.loc[group["outcome"] != "ok", "end"]
        if len(failed):
            per_bin = np.histogram(failed, bins=edges)[0]
            failed_pct = 100 * np.cumsum(per_bin) * per_task / capacity
            axis.fill_between(x, failed_pct, step="mid", color=color, alpha=0.55, lw=0)
        axis.fill_between(x, busy_pct, color=color, alpha=0.15, lw=0)
        axis.plot(x, busy_pct, color=color, lw=1.6)
        axis.plot(x, nodes_pct, color=color, lw=1.0, ls=(0, (4, 2)))
        for number, finished in records["round_finishes"].items():
            axis.axvline((finished - t0) / scale, color=MUTED, ls=(0, (2, 2)), lw=0.8)
        axis.set_ylim(0, max(105, axis.get_ylim()[1]))
        entries = [("GPUs busy" if not cpu_stage else "Tasks running", dict(color=color, lw=1.6)),
                   ("Nodes in use", dict(color=color, lw=1.0, ls=(0, (4, 2))))]
        if len(failed):
            entries.append(("Failed so far", {"area": color, "alpha": 0.55}))
        if any(stage in names for names in waits["stage"].str.split("/")):
            entries.append(("Waiting for a block", {"area": QUEUE_SHADE}))
        if len(records["round_finishes"]):
            entries.append(("New model (ALF iteration)", dict(color=MUTED, lw=0.8, ls=(0, (2, 2)))))
        key(axis, entries)
        axis.set_yticks([0, 50, 100], ["0", "50%", "100%"])
        axis.set_ylabel(title_case("busy"))
        active = group["end"].max() - group["start"].min()
        unit = "of peak (CPU tasks)" if stage == "builder" else \
            ("of peak; set gpus_per_node for % of GPUs" if cpu_stage else f"{nodes:,} node{'s' * (nodes != 1)}")
        _panel(axis, STAGE_NAME[stage], f"{len(group):,} task{'s' * (len(group) != 1)}  ·  {unit}  ·  "
               f"active {_minutes(active)}  ·  {len(failed):,} failed")
    for number, finished in records["round_finishes"].items():
        # Left of its line near the right edge, so it stays clear of the key.
        right = (finished - t0) / (end - t0) > 0.75
        axes[0].annotate(f"Iteration {number}", ((finished - t0) / scale, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(-3 if right else 3, -2), textcoords="offset points", va="top",
                         ha="right" if right else "left", fontsize=NOTE, color=MUTED)
    axes[-1].set_xlabel(title_case(xlabel))
    for axis in axes:
        axis.set_xlim(0, (end - t0) / scale * 1.06)
    _save(figure, out_dir, "run-report-stages.png", written)
    return True


def plot_task_times(records, out_dir, written):
    tasks = records["tasks"]
    if tasks.empty:
        return False
    stages = [stage for stage, _, _ in STAGES if (tasks["stage"] == stage).any()]
    figure, axes = _figure(len(stages), "Task times per ALF iteration",
                           "A long top whisker is a straggler the whole stage waited for.", height=1.5)
    for axis, stage in zip(axes, stages):
        group, color = tasks[tasks["stage"] == stage], STAGE_COLOR[stage]
        cycles = sorted(group["cycle"].unique())
        axis.boxplot([group.loc[group["cycle"] == cycle, "seconds"].to_numpy() for cycle in cycles],
                     positions=range(len(cycles)), widths=0.5, whis=(0, 100), patch_artist=True,
                     boxprops=dict(facecolor=color, edgecolor=color, alpha=0.3), medianprops=dict(color=color, lw=2),
                     whiskerprops=dict(color=MUTED, lw=1), capprops=dict(color=MUTED, lw=1))
        axis.set_xticks(range(len(cycles)), [str(cycle) for cycle in cycles])
        pad = max(0, 4 - len(cycles)) / 2  # A few cycles keep a box's width, rather than filling the panel.
        axis.set_xlim(-0.5 - pad, len(cycles) - 0.5 + pad)
        seconds = group["seconds"]
        axis.set_ylim(0, seconds.max() * 1.15 if seconds.max() > 0 else 1)
        _panel(axis, STAGE_NAME[stage], f"median {seconds.median():,.1f} s  ·  90% within "
               f"{seconds.quantile(0.9):,.1f} s  ·  longest {seconds.max():,.1f} s", "Seconds")
    key(axes[0], [("Middle half", {"area": MUTED, "alpha": 0.3}), ("Median", dict(color=MUTED, lw=2)),
                  ("Shortest to longest", dict(color=MUTED, lw=1))])
    axes[-1].set_xlabel(title_case("ALF iteration (the model in use when the task started)"))
    _save(figure, out_dir, "run-report-task-times.png", written)
    return True


def _cycle_axis(axis, models):
    axis.set_xticks(models, [str(model) for model in models])
    axis.set_xlim(models.min() - 0.6, models.max() + 0.6 + 0.12 * max(1, np.ptp(models)))


def plot_spread(records, out_dir, written):
    rows = spread_per_cycle(records["trajectories"], records["tasks"])
    if not rows:
        return False
    models = np.array([row["model"] for row in rows])
    states = sorted({state for row in rows for state in row.get("states", {})}, key=lambda state: (len(state), state))
    spreads = [(name, title, unit) for name, title, unit in SPREADS if name != "Fsmax" and any(name in row for row in rows)]
    gpu = np.array([row.get("sampling_gpu_hours_per_candidate", np.nan) for row in rows])
    panels = len(spreads) + 1 + int(np.isfinite(gpu).any())
    by_state = " Per-state lines use the state a trajectory ran on; gap lines, the pair of states." if states else ""
    figure, axes = _figure(panels, "Ensemble spread per ALF iteration",
                           "Spread is each trajectory's peak disagreement between the ensemble's members. All of these "
                           "should fall as the models improve." + by_state, height=1.5)
    ends = {}  # Each panel's key entries, for the lines drawn per state or pair.
    pairs = sorted({lower for row in rows for lower in row.get("gap_pairs", {})}, key=int)
    for axis, (name, title, unit) in zip(axes, spreads):
        if name == "gap_std" and pairs:  # One line per pair of states, by its lower state.
            for color, lower in series_colors(pairs):
                medians = np.array([row.get("gap_pairs", {}).get(lower, {}).get("median", np.nan) for row in rows])
                axis.plot(models, medians, color=color, lw=1.5, marker="o", ms=3.5)
                ends.setdefault(axis, []).append((f"States {lower}–{int(lower) + 1}",
                                                  dict(color=color, lw=1.5, marker="o", ms=3.5)))
            last = rows[-1].get(name, {})
            _panel(axis, "Gap spread per pair of states", f"last iteration, all pairs: median "
                   f"{last.get('median', np.nan):.3g} {unit}", unit)
        elif states and name != "gap_std":  # Excited states: each selected state's median.
            for color, state in series_colors(states):
                medians = np.array([row.get("states", {}).get(state, {}).get(name, {}).get("median", np.nan)
                                    for row in rows])
                axis.plot(models, medians, color=color, lw=1.5, marker="o", ms=3.5)
                ends.setdefault(axis, []).append((f"State {state}", dict(color=color, lw=1.5, marker="o", ms=3.5)))
            last = rows[-1].get(name, {})
            _panel(axis, title, f"last iteration, all states: median {last.get('median', np.nan):.3g} {unit}", unit)
        else:
            medians = np.array([row.get(name, {}).get("median", np.nan) for row in rows])
            p90 = np.array([row.get(name, {}).get("p90", np.nan) for row in rows])
            axis.fill_between(models, medians, p90, color=MEMBER_COLOR, alpha=0.12, lw=0)
            axis.plot(models, p90, color=MEMBER_COLOR, lw=1, ls=(0, (4, 2)))
            axis.plot(models, medians, color=MEMBER_COLOR, lw=1.6, marker="o", ms=4)
            key(axis, [("Median", dict(color=MEMBER_COLOR, lw=1.6, marker="o", ms=4)),
                       ("90th percentile", dict(color=MEMBER_COLOR, lw=1, ls=(0, (4, 2))))])
            _panel(axis, title, f"last iteration: median {medians[-1]:.3g} {unit}, 90% below {p90[-1]:.3g} {unit}", unit)
        axis.set_ylim(bottom=0)
    shares = np.array([100 * row["candidate_share"] if row["candidate_share"] is not None else np.nan
                       for row in rows])
    share_axis = axes[len(spreads)]
    bars = share_axis.bar(models, shares, color=MEMBER_COLOR, width=0.6, alpha=0.85)
    if len(models) <= 16:
        share_axis.bar_label(bars, labels=[f"{share:.0f}%" if np.isfinite(share) else "" for share in shares],
                             fontsize=NOTE, color=MUTED, padding=2)
    _panel(share_axis, "Trajectories that gave QM a candidate",
           f"{sum(row['candidates'] for row in rows):,} candidates from "
           f"{sum(row['trajectories'] for row in rows):,} trajectories", "Percent")
    share_axis.set_ylim(0, max(100, np.nanmax(shares) if np.isfinite(shares).any() else 0) * 1.12)
    if np.isfinite(gpu).any():
        axis = axes[-1]
        axis.plot(models, gpu * 60, color=STAGE_COLOR["sampler"], lw=1.6, marker="o", ms=4)
        last = gpu[np.isfinite(gpu)][-1] * 60
        _panel(axis, "Sampling GPU time per QM candidate", f"last iteration: {last:.3g} GPU-minutes", "GPU-minutes")
        axis.set_ylim(bottom=0)
    axes[-1].set_xlabel(title_case("ALF iteration"))
    for axis in axes:
        _cycle_axis(axis, models)
    for axis, entries in ends.items():
        key(axis, entries)
    _save(figure, out_dir, "run-report-spread.png", written)
    return True


def plot_labels(records, out_dir, written):
    labels = records["labels"].dropna(subset=["labels"])
    if labels.empty:
        return False
    rounds = labels["round"].to_numpy()
    color = STAGE_COLOR["QM"]
    figure, axes = _figure(2, "Labels per ALF iteration", "Counted from each iteration's HDF5 shard.", height=1.5)
    bars = axes[0].bar(rounds, labels["labels"], color=color, width=0.6, alpha=0.85)
    if len(rounds) <= 16:
        axes[0].bar_label(bars, labels=[f"{value:,.0f}" for value in labels["labels"]], fontsize=NOTE, color=MUTED,
                          padding=2)
    _panel(axes[0], "Added", f"median {labels['labels'].median():,.0f} per round", "Labels")
    axes[1].plot(rounds, labels["cumulative_labels"], color=color, lw=1.6, marker="o", ms=4)
    _label_end(axes[1], rounds, labels["cumulative_labels"], f"{labels['cumulative_labels'].iloc[-1]:,.0f}", color)
    _panel(axes[1], "In total", f"{len(rounds)} iteration{'s' * (len(rounds) != 1)}", "Labels", "ALF iteration")
    for axis in axes:
        axis.set_xticks(rounds, [str(number) for number in rounds])
        axis.set_xlim(rounds.min() - 0.6, rounds.max() + 0.6 + 0.12 * max(1, np.ptp(rounds)))
        axis.set_ylim(0, axis.get_ylim()[1] * 1.1)
    _save(figure, out_dir, "run-report-labels.png", written)
    return True


STATE_METRIC = re.compile(r"^final\.test\.(?P<kind>[EF])(?P<state>\d+)_(?P<error>RMSE|MAE)$")
ERROR_METRIC = re.compile(r"^final\.test\.(?P<quantity>.+?)[-_]?(?P<error>RMSE|MAE)$")
# HIPPYNN's names for what a ground-state member's errors measure.
QUANTITY = {"T": "Energy", "TperAtom": "Energy per Atom", "Force": "Force"}


def _error_columns(training):
    """Every final test error, as ``{quantity: {"RMSE": column, "MAE": column}}``: per state for excited states
    (quantities ``("E", state)`` and ``("F", state)``), else per HIPPYNN quantity (energy, energy per atom, force)."""
    found = {}
    for column in training.columns:
        state = STATE_METRIC.match(column)
        plain = ERROR_METRIC.match(column)
        if state:
            found.setdefault((state["kind"], int(state["state"])), {})[state["error"]] = column
        elif plain:
            found.setdefault(plain["quantity"], {})[plain["error"]] = column
    return found


def _state_metrics(training):
    """Excited-state members' final test errors by state: ``{"E" or "F": {state: {"RMSE": column, "MAE": ...}}}``."""
    by_state = {}
    for quantity, errors in _error_columns(training).items():
        if isinstance(quantity, tuple):
            by_state.setdefault(quantity[0], {})[quantity[1]] = errors
    return {kind: dict(sorted(states.items())) for kind, states in by_state.items()}


def _ground_metrics(training):
    """Ground-state final test errors, energy first, at most four quantities."""
    errors = {quantity: value for quantity, value in _error_columns(training).items() if not isinstance(quantity, tuple)}
    order = list(QUANTITY) + sorted(set(errors) - set(QUANTITY))
    return {key: errors[key] for key in order if key in errors}


def _medians(training, column, rounds):
    return pd.to_numeric(training[column], errors="coerce").groupby(training["round"]).median().reindex(rounds)


def plot_training(records, out_dir, written):
    training = records["training"]
    if training.empty:
        return False
    rounds = np.array(sorted(training["round"].unique()))
    by_state, ground = _state_metrics(training), _ground_metrics(training)
    panels = [("epoch", "Epoch Time")]
    if by_state:
        panels += [(kind, f"Final Test {'Energy' if kind == 'E' else 'Force'} Error per State") for kind in ("E", "F")
                   if kind in by_state]
    else:
        panels += [(quantity, f"Final Test {QUANTITY.get(quantity, quantity)} Error") for quantity in list(ground)[:4]]
    figure, axes = _figure(len(panels), "Training per ALF iteration",
                           "Epoch time is each member's last epoch. Errors are the final test of each member's best "
                           "model, in the units of the training data" + (", as the median over members." if by_state
                                                                          else "."), height=1.6)
    members = training["member"][training["round"] == rounds[-1]].nunique()
    for axis, (kind, title) in zip(axes, panels):
        ends = []
        if kind == "epoch":
            values = pd.to_numeric(training["last_epoch_time"], errors="coerce")
            axis.scatter(training["round"], values, s=14, color=MEMBER_COLOR, alpha=0.55, lw=0)
            medians = values.groupby(training["round"]).median().reindex(rounds)
            axis.plot(rounds, medians, color=TEXT, lw=1.3)
            key(axis, [("Members", dict(color=MEMBER_COLOR, lw=0, marker="o", ms=5, alpha=0.6)),
                       ("Median", dict(color=TEXT, lw=1.3))])
            _panel(axis, title, f"iteration {rounds[-1]}: median {medians.iloc[-1]:.3g} s, {members} members", "Seconds")
        elif kind in ("E", "F"):
            for color, (state, errors) in series_colors(by_state[kind].items()):
                for error, style in (("RMSE", "-"), ("MAE", (0, (4, 2)))):
                    if error in errors:
                        medians = _medians(training, errors[error], rounds)
                        axis.plot(rounds, medians, color=color, lw=1.5, ls=style, marker="o" if error == "RMSE"
                                  else None, ms=4)
                ends.append((f"State {state}", dict(color=color, lw=1.5, marker="o", ms=4)))
            _panel(axis, title, f"iteration {rounds[-1]}, {members} members", "Error")
            key(axis, ends + [("RMSE: solid", dict(lw=1.5)), ("MAE: dashed", dict(lw=1.5, ls=(0, (4, 2))))])
        else:
            errors = ground[kind]
            for error, style in (("RMSE", "-"), ("MAE", (0, (4, 2)))):
                if error in errors:
                    values = pd.to_numeric(training[errors[error]], errors="coerce")
                    axis.scatter(training["round"], values, s=14, color=MEMBER_COLOR, alpha=0.4, lw=0)
                    medians = values.groupby(training["round"]).median().reindex(rounds)
                    axis.plot(rounds, medians, color=MEMBER_COLOR, lw=1.5, ls=style)
            last = ", ".join(f"{error} {_medians(training, column, rounds).iloc[-1]:.3g}"
                             for error, column in errors.items())
            _panel(axis, title, f"iteration {rounds[-1]} medians: {last}", "Error")
            key(axis, [("Members", dict(color=MEMBER_COLOR, lw=0, marker="o", ms=5, alpha=0.5)),
                       ("RMSE (median)", dict(color=MEMBER_COLOR, lw=1.5)),
                       ("MAE (median)", dict(color=MEMBER_COLOR, lw=1.5, ls=(0, (4, 2))))])
        axis.set_xticks(rounds, [str(number) for number in rounds])
        axis.set_xlim(rounds.min() - 0.6, rounds.max() + 0.6 + 0.12 * max(1, np.ptp(rounds)))
    axes[-1].set_xlabel(title_case("ALF iteration"))
    _save(figure, out_dir, "run-report-training.png", written)
    return True


def plot_health(records, out_dir, written):
    tasks, memory = records["tasks"], records["memory"]
    gpu = tasks.dropna(subset=["gpu_peak_gb"]) if not tasks.empty else tasks
    failures = int((tasks["outcome"] != "ok").sum()) if not tasks.empty else None
    panels = [name for name, present in (("gpu", not gpu.empty), ("memory", not memory.empty),
                                         ("failed", bool(failures))) if present]
    if not panels:
        return False
    note = "" if failures is None else (" No task failed." if not failures else f" {failures:,} tasks failed.")
    figure, axes = _figure(len(panels), "Run health", note.strip() or None, height=1.5)
    t0 = records["run_start"]
    end = max(tasks["end"].max() if not tasks.empty else t0, memory["time"].max() if not memory.empty else t0)
    scale, xlabel = _time_axis(end - t0)
    for axis, panel in zip(axes, panels):
        if panel == "gpu":
            stages = [stage for stage, _, _ in STAGES if (gpu["stage"] == stage).any()]
            boxes = axis.boxplot([gpu.loc[gpu["stage"] == stage, "gpu_peak_gb"] for stage in stages], vert=False,
                                 widths=0.5, patch_artist=True, tick_labels=[STAGE_NAME[stage] for stage in stages],
                                 whis=(0, 100), medianprops=dict(lw=2), whiskerprops=dict(color=MUTED),
                                 capprops=dict(color=MUTED))
            for box, median, stage in zip(boxes["boxes"], boxes["medians"], stages):
                box.set(facecolor=STAGE_COLOR[stage], edgecolor=STAGE_COLOR[stage], alpha=0.3)
                median.set(color=STAGE_COLOR[stage])
            axis.grid(axis="x", color=GRID, lw=0.8)
            axis.grid(axis="y", visible=False)
            highest = gpu.loc[gpu["gpu_peak_gb"].idxmax()]
            key(axis, [("Middle half", {"area": MUTED, "alpha": 0.3}), ("Median", dict(color=MUTED, lw=2)),
                       ("Lowest to highest", dict(color=MUTED, lw=1))])
            _panel(axis, "Peak GPU memory per task", f"highest {highest['gpu_peak_gb']:.1f} GB "
                   f"({STAGE_NAME.get(highest['stage'], highest['stage'])}, {highest['host']})", xlabel="GB")
        elif panel == "memory":
            x = (memory["time"] - t0) / scale
            axis.plot(x, memory["tree_gib"], color=MEMBER_COLOR, lw=1.6)
            axis.plot(x, memory["coordinator_gib"], color=SECONDARY, lw=1.2)
            key(axis, [("All processes (PSS)", dict(color=MEMBER_COLOR, lw=1.6)),
                       ("Coordinator (RSS)", dict(color=SECONDARY, lw=1.2))])
            _panel(axis, "Coordinator memory", f"peak {memory['tree_gib'].max():.1f} GiB with child processes, "
                   f"{memory['coordinator_gib'].max():.1f} GiB coordinator", "GiB", xlabel)
            axis.set_ylim(bottom=0)
            axis.set_xlim(0, (end - t0) / scale * 1.02)
        else:
            failed = tasks[tasks["outcome"] != "ok"].sort_values("end")
            entries = []
            for stage, _, color in STAGES:
                times = failed.loc[failed["stage"] == stage, "end"].to_numpy()
                if len(times):
                    x = (np.concatenate([[t0], times]) - t0) / scale
                    axis.step(x, np.arange(len(times) + 1), where="post", color=color, lw=1.6)
                    entries.append((f"{STAGE_NAME[stage]} ({len(times):,})", dict(color=color, lw=1.6)))
            key(axis, entries)
            reasons = failed["outcome"].value_counts()
            _panel(axis, "Failed tasks so far", ", ".join(f"{count:,} {reason}" for reason, count
                                                          in reasons.head(3).items()), "Tasks", xlabel)
            axis.set_ylim(bottom=0)
            axis.set_xlim(0, (end - t0) / scale * 1.02)
    _save(figure, out_dir, "run-report-health.png", written)
    return True


FIGURES = (("stages", plot_stages, "task timeline"), ("task_times", plot_task_times, "task timeline"),
           ("spread", plot_spread, "trajectory summary"), ("labels", plot_labels, "stored labels"),
           ("training", plot_training, "training summaries"), ("health", plot_health, "task timeline and memory"))


def summary(records):
    tasks, training = records["tasks"], records["training"]
    rounds = []
    for number, group in training.groupby("round") if not training.empty else ():
        row = dict(round=int(number), members=int(len(group)),
                   median_epoch_seconds=float(pd.to_numeric(group["last_epoch_time"], errors="coerce").median()))
        for errors in _error_columns(training).values():
            for column in errors.values():
                row[column] = float(pd.to_numeric(group[column], errors="coerce").median())
        rounds.append(row)
    memory = records["memory"]
    return _clean(dict(
        paths=records["paths"],
        stages=stage_summary(tasks) if not tasks.empty else {},
        stage_gaps=stage_gaps(tasks) if not tasks.empty else [],
        block_waits=records["block_waits"].to_dict(orient="records"),
        slow_hosts=slow_hosts(tasks) if not tasks.empty else [],
        spread_per_cycle=spread_per_cycle(records["trajectories"], records["tasks"]),
        labels_per_round=records["labels"].to_dict(orient="records"),
        training_rounds=rounds,
        coordinator_memory=dict(peak_tree_gib=float(memory["tree_gib"].max()),
                                peak_coordinator_gib=float(memory["coordinator_gib"].max())) if not memory.empty
        else {},
    ))


def write_report(run_dir, out_dir=None, master_config_path=None, timeline_dir=None):
    """Write the report's figures and ``run_report.json`` for the run in ``run_dir``. Returns the files written."""
    run_dir = os.path.abspath(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "run_report")
    master = (json.loads(open(master_config_path).read()) if master_config_path
              else load_master_config(run_dir))
    records = run_records(run_dir, master, timeline_dir)
    records["gpus_per_node"] = master.get("gpus_per_node")
    os.makedirs(out_dir, exist_ok=True)
    written, missing = [], []
    with plt.rc_context(RC):
        for name, plot, needs in FIGURES:
            try:
                if not plot(records, out_dir, written):
                    missing.append(f"{name} (no {needs})")
            except Exception as error:  # One figure's problem must not stop the others.
                missing.append(f"{name} ({type(error).__name__}: {error})")
                plt.close("all")
    report = summary(records)
    report["figures"], report["skipped"] = written, missing
    with open(os.path.join(out_dir, "run_report.json"), "w") as handle:
        json.dump(report, handle, indent=1, allow_nan=False)
    written.append("run_report.json")
    if missing:
        print("Run report skipped: " + "; ".join(missing), flush=True)
    return written


def live_report(run_dir, out_dir=None, master_config_path=None):
    """The coordinator's entry point (``run_report: true``), run in its own process with the progress plots."""
    return write_report(run_dir, out_dir, master_config_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--out", help="where to write (default RUN_DIR/run_report)")
    parser.add_argument("--master", help="the master config (default RUN_DIR/master_config.json)")
    parser.add_argument("--timeline", help="the task timeline directory (default RUN_DIR/task_timeline)")
    args = parser.parse_args(argv)
    for name in write_report(args.run_dir, args.out, args.master, args.timeline):
        print(name)


if __name__ == "__main__":
    main(sys.argv[1:])
