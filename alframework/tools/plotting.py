"""Progress plots: how sampling changes as the model improves, with one box per model.

Plots come from a trajectory summary (see :mod:`alframework.tools.trajectory_summary`) when
``trajectory_summary_path`` is set, or otherwise from the pickles in ``meta_dir``. The summary is the
better choice for a large run, because it's one file written by the coordinator.
"""
import os
import pickle
import re
from array import array

import matplotlib
matplotlib.use("Agg")  # This runs on a head node, usually without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alframework.tools.trajectory_summary import COLUMNS, read_trajectory_summary, trajectory_row

METADATA_FILE = re.compile(r'metadata-(mol-\d+-\d+)\.p')
# The reference categorical palette, in its fixed order. An outcome keeps its color in every run.
OUTCOME_COLORS = {"completed": "#2a78d6", "eligible": "#1baf7a", "rejected": "#eb6834", "failed": "#eda100",
                  "task_failed": "#e87ba4"}
OTHER_OUTCOME_COLOR = "#6250d6"
BOX_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _read_pickles(meta_dir):
    """Read every metadata pickle into columns, skipping any that are partial or unreadable."""
    rows = []
    for filename in os.listdir(meta_dir):
        match = METADATA_FILE.fullmatch(filename)
        if match is None:
            continue
        try:
            with open(os.path.join(meta_dir, filename), 'rb') as handle:
                rows.append(trajectory_row(match.group(1), pickle.load(handle)))
        except Exception:  # A worker may still be writing it.
            continue
    return {name: [row[name] for row in rows] for name in COLUMNS}


def _table(columns):
    """Build the table to plot, with numeric columns, and peak uncertainty preferred when recorded."""
    table = pd.DataFrame({name: np.asarray(columns[name]) if isinstance(columns[name], array) else columns[name]
                          for name in COLUMNS if name in columns}, columns=list(COLUMNS))
    table = table[table["model"].notna()].copy()
    table["model"] = table["model"].astype(int)
    for name in ("simtime", "realtime", "Es", "Fs", "Fsmax", "peak_Es", "peak_Fs", "peak_Fsmax"):
        table[name] = pd.to_numeric(table[name], errors="coerce").astype(float)
    for name in ("Es", "Fs", "Fsmax"):
        table[name] = table[f"peak_{name}"].where(table[f"peak_{name}"].notna(), table[name])
    for name in ("simtime", "realtime"):
        table[f"log_{name}"] = np.log10(table[name].where(table[name] > 0))
    return table


def _chosen_models(table, most):
    """Return the models to plot, evenly spread across those that exist, including both ends."""
    models = np.sort(table["model"].unique())
    if len(models) <= most:
        return models
    return models[np.unique(np.linspace(0, len(models) - 1, most).round().astype(int))]


def _style(figure, axis, title, ylabel):
    figure.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=TEXT, loc="left")
    axis.set_xlabel("Model", color=MUTED)
    axis.set_ylabel(ylabel, color=MUTED)
    axis.tick_params(colors=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def _mark_windows(axis, table, models):
    """Draw a line at the first plotted model of each sampling window after the first."""
    windows = table[table["window"].notna() & table["model"].isin(models)]
    if windows.empty:
        return
    first_models = windows.groupby("window")["model"].min().sort_values()
    for window, model in first_models.iloc[1:].items():
        position = int(np.searchsorted(models, model)) - 0.5
        axis.axvline(position, color=MUTED, linestyle="--", linewidth=1)
        axis.annotate(f"window {window}", (position, 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                      textcoords="offset points", va="top", color=MUTED, fontsize=8)


def _new_figure(count):
    return plt.subplots(figsize=(max(6.4, 0.28 * count + 1.5), 4.8), layout="constrained")


def _box_plot(table, name, title, ylabel, path, models, windows):
    """Write one box plot, with a box for each model that has data. Returns whether it had any."""
    values = table[table[name].notna() & table["model"].isin(models)]
    if values.empty:
        return False
    positions, groups = [], []
    for position, model in enumerate(models):
        group = values.loc[values["model"] == model, name].to_numpy()
        if len(group):
            positions.append(position)
            groups.append(group)
    figure, axis = _new_figure(len(models))
    axis.boxplot(groups, positions=positions, widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor=BOX_COLOR, edgecolor=BOX_COLOR, alpha=0.35),
                 medianprops=dict(color=BOX_COLOR, linewidth=2), whiskerprops=dict(color=MUTED, linewidth=1),
                 capprops=dict(color=MUTED, linewidth=1), flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                                                          markeredgecolor=MUTED, alpha=0.6))
    axis.set_xticks(range(len(models)), [str(model) for model in models], rotation=45 if len(models) > 12 else 0)
    axis.set_xlim(-0.75, len(models) - 0.25)
    _style(figure, axis, title, ylabel)
    _mark_windows(axis, windows, models)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return True


def _outcome_plot(table, path, models, windows):
    """Write a stacked bar of trajectory outcomes per model. Returns whether any were recorded."""
    known = table[table["outcome"].notna() & table["model"].isin(models)]
    if known.empty:
        return False
    counts = known.groupby(["model", "outcome"]).size().unstack(fill_value=0).reindex(models, fill_value=0)
    ordered = [name for name in OUTCOME_COLORS if name in counts] + sorted(set(counts) - set(OUTCOME_COLORS))
    figure, axis = _new_figure(len(models))
    bottom = np.zeros(len(models))
    for name in ordered:
        axis.bar(range(len(models)), counts[name].to_numpy(), 0.7, bottom=bottom, label=name, edgecolor=SURFACE,
                 linewidth=1.5, color=OUTCOME_COLORS.get(name, OTHER_OUTCOME_COLOR))
        bottom += counts[name].to_numpy()
    axis.set_xticks(range(len(models)), [str(model) for model in models], rotation=45 if len(models) > 12 else 0)
    axis.set_xlim(-0.75, len(models) - 0.25)
    _style(figure, axis, "Trajectory outcomes", "Trajectories")
    _mark_windows(axis, windows, models)
    legend = axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), reverse=True)
    for text in legend.get_texts():
        text.set_color(MUTED)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return True


def analysis_plot(meta_dir=None, plot_dir=None, meta_n_cols=50, trajectory_summary_path=None):
    """Write progress plots, reading a trajectory summary if one is set, or else ``meta_dir``.

    Args:
        meta_dir (str): Directory of per-trajectory metadata pickles.
        plot_dir (str): Directory to write the plots to.
        meta_n_cols (int): Maximum number of models to show on a plot.
        trajectory_summary_path (str): File the coordinator writes one row per trajectory to.

    Returns:
        (list): The names of the plots written.
    """
    if plot_dir is None or (meta_dir is None and trajectory_summary_path is None):
        print("Progress plots need plot_dir, and either trajectory_summary_path or meta_dir, so none were made.", flush=True)
        return []
    if trajectory_summary_path is not None:
        if not os.path.exists(trajectory_summary_path):
            return []  # Nothing has been sampled yet.
        table = _table(read_trajectory_summary(trajectory_summary_path))
    else:
        table = _table(_read_pickles(meta_dir))
    if table.empty:
        return []
    os.makedirs(plot_dir, exist_ok=True)
    models = _chosen_models(table, meta_n_cols)
    peak = "Peak " if table["peak_Es"].notna().any() else ""
    plots = (("realtime", "Wall time per trajectory", "Seconds"),
             ("log_realtime", "Wall time per trajectory, log scale", "log10(seconds)"),
             ("simtime", "Simulated time per trajectory", "Picoseconds"),
             ("log_simtime", "Simulated time per trajectory, log scale", "log10(picoseconds)"),
             ("Es", f"{peak}energy uncertainty".capitalize(), "eV"),
             ("Fs", f"{peak}mean force uncertainty".capitalize(), "eV/Å"),
             ("Fsmax", f"{peak}largest force uncertainty".capitalize(), "eV/Å"))
    states = sorted(table["state"].dropna().unique(), key=str)
    written, empty = [], []
    for name, title, ylabel in plots:
        if _box_plot(table, name, title, ylabel, os.path.join(plot_dir, f"analysis-plot-{name}.png"), models, table):
            written.append(name)
        else:
            empty.append(name)
        if len(states) > 1:
            for state in states:
                subset = table[table["state"] == state]
                label = f"{name}-state{state}"
                if _box_plot(subset, name, f"{title}, state {state}", ylabel,
                             os.path.join(plot_dir, f"analysis-plot-{label}.png"), models, table):
                    written.append(label)
    if _outcome_plot(table, os.path.join(plot_dir, "analysis-plot-outcomes.png"), models, table):
        written.append("outcomes")
    if empty:
        print("Progress plots skipped for lack of data: " + ", ".join(empty), flush=True)
    return written
