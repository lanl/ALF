"""Plot sampling progress across successive model ensembles.

Plots come from a trajectory summary (see :mod:`alframework.tools.trajectory_summary`) when
``trajectory_summary_path`` is set, or otherwise from the pickles in ``meta_dir``. The summary is the
better choice for a large run, because it's one file written by the coordinator.

With the sampler's cutoffs (``Escut``, ``Fscut``, ``gap_std_cut``, found in the sampler config like the other
arguments), the uncertainty plots draw each cutoff and say how many trajectories passed it, and the cutoff-tuning
and trigger plots show how QM demand would move with the cutoffs. Which signals trigger QM follows the sampler's
``acquisition.eligibility`` (energy and both force signals by default; add ``"gap"`` for excited-state gaps).
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
from matplotlib.ticker import FuncFormatter

from alframework.tools.plot_style import (CATEGORICAL, GRID, LABEL, MUTED, NOTE, RC, SURFACE, WIDTH, figure,
                                          label_end, panel, save, thousands, title_case)
from alframework.tools.trajectory_summary import COLUMNS, read_trajectory_summary, trajectory_row

METADATA_FILE = re.compile(r'metadata-(mol-\d+-\d+)\.p')
TASK_METADATA_FILE = re.compile(r'metadata-task-(mol-\d+-\d+)\.p')
MOLECULE_ID = re.compile(r'mol-\d+-\d+')
# The reference categorical palette, in its fixed order. An outcome keeps its color in every run.
OUTCOME_COLORS = {"completed": "#2a78d6", "eligible": "#1baf7a", "rejected": "#eb6834", "failed": "#eda100",
                  "task_failed": "#e87ba4"}
OTHER_OUTCOME_COLOR = "#6250d6"
BOX_COLOR = "#2a78d6"
CUTOFF_COLOR = "#eb6834"
# Each uncertainty signal: its column, the acquisition term that uses it, the cutoff key, the cutoff multiplier,
# its title, and unit. They match alframework.samplers.acquisition.SCORE_TERMS.
SIGNALS = (("Es", "energy", "Escut", 1.0, "energy uncertainty", "eV"),
           ("Fs", "force_mean", "Fscut", 1.0, "mean force uncertainty", "eV/Å"),
           ("Fsmax", "force_max", "Fscut", 3.0, "largest force uncertainty", "eV/Å"),
           ("gap_std", "gap", "gap_std_cut", 1.0, "gap uncertainty", "eV"))
DEFAULT_ELIGIBILITY = ("energy", "force_mean", "force_max")
CANDIDATE = "eligible"


def _metadata_files(meta_dir):
    """Files in ``meta_dir`` and, for ``meta_dir_layout`` "per_task", in its ``task-*`` subdirectories."""
    for entry in os.scandir(meta_dir):
        if entry.is_dir() and entry.name.startswith('task-'):
            yield from ((os.path.join(entry.path, name), name) for name in os.listdir(entry.path))
        else:
            yield entry.path, entry.name


def _read_pickles(meta_dir):
    """Read every metadata pickle into columns, skipping any that are partial or unreadable."""
    rows = []
    for path, filename in _metadata_files(meta_dir):
        match = METADATA_FILE.fullmatch(filename)
        bundle = TASK_METADATA_FILE.fullmatch(filename)
        if match is None and bundle is None:
            continue
        try:
            with open(path, 'rb') as handle:
                loaded = pickle.load(handle)
            if match is not None:
                rows.append(trajectory_row(match.group(1), loaded))
            else:  # metadata_files "per_task": a list of dicts, each with its molecule id
                rows.extend(trajectory_row(record['moleculeid'], record) for record in loaded
                            if MOLECULE_ID.fullmatch(str(record.get('moleculeid'))))
        except Exception:  # A worker may still be writing it.
            continue
    return {name: [row[name] for row in rows] for name in COLUMNS}


def _table(columns):
    """Build the table to plot, with numeric columns, and peak uncertainty preferred when recorded."""
    table = pd.DataFrame({name: np.asarray(columns[name]) if isinstance(columns[name], array) else columns[name]
                          for name in COLUMNS if name in columns}, columns=list(COLUMNS))
    table = table[table["model"].notna()].copy()
    table["model"] = table["model"].astype(int)
    for name in ("simtime", "realtime", "Es", "Fs", "Fsmax", "peak_Es", "peak_Fs", "peak_Fsmax", "gap_std",
                 "peak_gap_std"):
        table[name] = pd.to_numeric(table[name], errors="coerce").astype(float)
    for name in ("Es", "Fs", "Fsmax", "gap_std"):
        table[name] = table[f"peak_{name}"].where(table[f"peak_{name}"].notna(), table[name])
    for name in ("simtime", "realtime"):
        table[f"log_{name}"] = np.log10(table[name].where(table[name] > 0))
    return table


def _whole(value):
    """A state number as an integer when it is one (a column with gaps reads as floats), else as it is."""
    try:
        return int(value) if float(value).is_integer() else value
    except (TypeError, ValueError):
        return value


def _chosen_models(table, most):
    """Return the models to plot, evenly spread across those that exist, including both ends."""
    models = np.sort(table["model"].unique())
    if len(models) <= most:
        return models
    return models[np.unique(np.linspace(0, len(models) - 1, most).round().astype(int))]


def _model_axis(axis, models):
    axis.set_xticks(range(len(models)), [str(model) for model in models], rotation=45 if len(models) > 12 else 0)
    axis.set_xlim(-0.75, len(models) - 0.25 + 0.04 * max(len(models), 4))
    axis.set_xlabel(title_case("Model"))


def _mark_windows(axis, table, models):
    """Draw a line at the first plotted model of each sampling window after the first."""
    windows = table[table["window"].notna() & table["model"].isin(models)]
    if windows.empty:
        return
    first_models = windows.groupby("window")["model"].min().sort_values()
    for window, model in first_models.iloc[1:].items():
        position = int(np.searchsorted(models, model)) - 0.5
        axis.axvline(position, color=MUTED, linestyle="--", linewidth=1)
        axis.annotate(f"Window {window}", (position, 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                      textcoords="offset points", va="top", color=MUTED, fontsize=NOTE)


def _new_figure(count, caption):
    return figure(1, caption=caption, height=3.6, width=max(WIDTH, 0.24 * count + 3.0))


def _cutoffs(Escut=None, Fscut=None, gap_std_cut=None):
    """Each signal's trigger threshold (cutoff times multiplier) where its cutoff is known."""
    given = {"Escut": Escut, "Fscut": Fscut, "gap_std_cut": gap_std_cut}
    found = {}
    for name, _, key, multiplier, _, _ in SIGNALS:
        value = given.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
            found[name] = float(value) * multiplier
    return found


def _eligible_signals(acquisition, thresholds):
    """The signals that trigger QM, per ``acquisition.eligibility``, among those with a known cutoff."""
    terms = (acquisition or {}).get("eligibility", DEFAULT_ELIGIBILITY) if isinstance(acquisition, dict) else \
        DEFAULT_ELIGIBILITY
    return [name for name, term, *_ in SIGNALS if term in terms and name in thresholds]


def _box_plot(table, name, title, ylabel, path, models, windows, threshold=None):
    """Write one box plot, with a box for each model that has data. Returns whether it had any.

    Boxes span the middle half, whiskers the 5th to 95th percentiles, and a tick marks each model's largest value,
    so a long tail doesn't hide the rest. With ``threshold``, a line marks it and the numbers say how many passed it."""
    values = table[table[name].notna() & table["model"].isin(models)]
    if values.empty:
        return False
    positions, groups = [], []
    for position, model in enumerate(models):
        group = values.loc[values["model"] == model, name].to_numpy()
        if len(group):
            positions.append(position)
            groups.append(group)
    caption = ("Box: the middle half of each model's trajectories, with the median. Whiskers: 5th to 95th "
               "percentile. Tick: the largest.")
    if threshold is not None:
        caption += " Orange line: the sampler's trigger threshold."
    fig, (axis,) = _new_figure(len(models), caption)
    axis.boxplot(groups, positions=positions, widths=0.6, patch_artist=True, whis=(5, 95), showfliers=False,
                 boxprops=dict(facecolor=BOX_COLOR, edgecolor=BOX_COLOR, alpha=0.3),
                 medianprops=dict(color=BOX_COLOR, linewidth=2), whiskerprops=dict(color=MUTED, linewidth=1),
                 capprops=dict(color=MUTED, linewidth=1))
    axis.scatter(positions, [group.max() for group in groups], marker="_", s=60, color=MUTED, linewidths=1.2,
                 zorder=3)
    last = groups[-1]
    numbers = f"model {models[positions[-1]]}: median {np.median(last):.3g}, {len(last):,} trajectories"
    if threshold is not None:
        axis.axhline(threshold, color=CUTOFF_COLOR, linestyle=(0, (4, 2)), linewidth=1.2)
        label_end(axis, [len(models) - 0.5], [threshold], f"Threshold {threshold:.3g}", CUTOFF_COLOR, dy=9)
        numbers += f", {100 * np.mean(last > threshold):.0f}% above the threshold"
    _model_axis(axis, models)
    panel(axis, title, numbers, ylabel)
    _mark_windows(axis, windows, models)
    save(fig, path)
    return True


def _outcome_plot(table, path, models, windows):
    """Write each model's trajectory outcomes as shares of its trajectories. Returns whether any were recorded."""
    known = table[table["outcome"].notna() & table["model"].isin(models)]
    if known.empty:
        return False
    counts = known.groupby(["model", "outcome"]).size().unstack(fill_value=0).reindex(models, fill_value=0)
    ordered = [name for name in OUTCOME_COLORS if name in counts] + sorted(set(counts) - set(OUTCOME_COLORS))
    totals = counts.sum(axis=1).replace(0, np.nan).to_numpy()
    fig, (axis,) = _new_figure(len(models), f"Each bar is one model's trajectories, {int(np.nansum(totals)):,} in "
                               "all. 'eligible' trajectories gave QM a candidate; 'completed' ran to the end without one.")
    bottom = np.zeros(len(models))
    for name in ordered:
        share = np.nan_to_num(100 * counts[name].to_numpy() / totals)
        axis.bar(range(len(models)), share, 0.7, bottom=bottom, label=name, edgecolor=SURFACE, linewidth=1.5,
                 color=OUTCOME_COLORS.get(name, OTHER_OUTCOME_COLOR))
        if share[-1] >= 6:  # Name each outcome on the last model's bar, where it fits.
            axis.annotate(name, (len(models) - 1 + 0.4, bottom[-1] + share[-1] / 2), xytext=(4, 0),
                          textcoords="offset points", va="center", fontsize=NOTE, annotation_clip=False,
                          color=OUTCOME_COLORS.get(name, OTHER_OUTCOME_COLOR))
        bottom += share
    last = counts.iloc[-1]
    _model_axis(axis, models)
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
    panel(axis, "Trajectory outcomes", f"model {models[-1]}: " + ", ".join(
        f"{100 * last[name] / max(last.sum(), 1):.0f}% {name}" for name in ordered if last[name]), "Trajectories")
    _mark_windows(axis, windows, models)
    legend = axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), reverse=True, fontsize=NOTE)
    for text in legend.get_texts():
        text.set_color(MUTED)
    save(fig, path)
    return True


def _trigger_ratio(table, signals, thresholds):
    """Each trajectory's largest signal over its threshold: above 1, it triggered (or would have)."""
    ratios = [table[name] / thresholds[name] for name in signals]
    return pd.concat(ratios, axis=1).max(axis=1, skipna=True) if ratios else pd.Series(dtype=float)


def _cutoff_tuning_plot(table, path, models, signals, thresholds):
    """For the latest few models: the share of trajectories that would trigger QM if every cutoff were scaled."""
    latest = list(models[-4:])
    scales = np.geomspace(0.25, 4, 121)
    panels = [("all", "Any eligible signal")] + [(name, title.capitalize()) for name, _, _, _, title, _ in SIGNALS
                                                 if name in signals]
    fig, axes = figure(len(panels), "Cutoff tuning",
                       "The share of each model's trajectories whose peak passes the threshold, if every cutoff were "
                       "multiplied by the factor on the x axis (1 = the cutoffs now). With trajectories stopped at "
                       "their first trigger, peaks above a threshold were cut short, so to the right of 1 the shares "
                       "are lower bounds.", height=1.6)
    for number, (axis, (signal, title)) in enumerate(zip(axes, panels)):
        for color, model in zip(CATEGORICAL, latest):
            group = table[table["model"] == model]
            ratio = _trigger_ratio(group, signals, thresholds) if signal == "all" else \
                group[signal] / thresholds[signal]
            ratio = ratio.dropna().to_numpy()
            if not len(ratio):
                continue
            share = 100 * (ratio[None, :] > scales[:, None]).mean(axis=1)
            axis.plot(scales, share, color=color, lw=1.6, label=f"Model {model}")
        axis.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        axis.set_xscale("log")
        axis.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25×", "0.5×", "1×", "2×", "4×"])
        axis.set_xticks([], minor=True)
        axis.set_ylim(0, 105)
        axis.set_yticks([0, 50, 100], ["0", "50%", "100%"])
        if number == 0:  # One key for every panel: the lines meet at 0 on the right, so they can't be labeled there.
            legend = axis.legend(loc="upper right", reverse=True)
            for text, line in zip(legend.get_texts(), legend.get_lines()):
                text.set_color(line.get_color())
        group = table[table["model"] == latest[-1]]
        now = _trigger_ratio(group, signals, thresholds) if signal == "all" else group[signal] / thresholds[signal]
        now = now.dropna()
        panel(axis, title, f"model {latest[-1]} now: {100 * (now > 1).mean():.0f}% above" if len(now) else None,
              "Share of Trajectories")
    axes[-1].set_xlabel(title_case("Cutoffs multiplied by"))
    save(fig, path)
    return True


def _trigger_time_plot(table, path, models, signals, thresholds):
    """Latest model: when each trajectory stopped against how close it came to triggering."""
    model = models[-1]
    group = table[table["model"] == model]
    ratio = _trigger_ratio(group, signals, thresholds)
    data = pd.DataFrame({"time": group["simtime"], "ratio": ratio}).dropna()
    data = data[data["ratio"] > 0]
    if data.empty:
        return False
    fig, (axis,) = figure(1, caption="Each trajectory of the latest model: simulated time when it stopped, and its "
                          "peak uncertainty over the trigger threshold (above 1: it gave QM a candidate). Darker: more "
                          "trajectories. Early triggers mean the model fails fast; a crowd just under 1 at the end of "
                          "the run means the cutoffs barely held.", height=3.6)
    if len(data) > 500:
        hexes = axis.hexbin(data["time"], data["ratio"], gridsize=60, yscale="log", bins="log", cmap="Blues",
                            mincnt=1, linewidths=0)
        bar = fig.colorbar(hexes, ax=axis, pad=0.01)
        bar.set_label("Trajectories", color=MUTED, fontsize=LABEL)
        bar.outline.set_visible(False)
    else:
        axis.scatter(data["time"], data["ratio"], s=10, color=BOX_COLOR, alpha=0.6, lw=0)
        axis.set_yscale("log")
    axis.axhline(1.0, color=CUTOFF_COLOR, lw=1.2, ls=(0, (4, 2)))
    axis.grid(axis="x", color=GRID, lw=0.8)
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    axis.yaxis.set_major_formatter(plain)
    axis.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: f"{value:g}" if f"{value:g}"[0] in "25" else ""))
    triggered = data["ratio"] > 1
    panel(axis, f"When model {model}'s trajectories stopped",
          f"{100 * triggered.mean():.0f}% triggered, median at {data.loc[triggered, 'time'].median():.3g} ps"
          if triggered.any() else "none triggered", "Peak uncertainty / threshold", "Simulated time (ps)")
    save(fig, path)
    return True


def _rejection_plot(table, path, models, windows):
    """Each model's rejected trajectories by the reason recorded for them."""
    rejected = table[(table["outcome"] == "rejected") & table["reason"].notna() & table["model"].isin(models)]
    if rejected.empty:
        return False
    counts = rejected.groupby(["model", "reason"]).size().unstack(fill_value=0).reindex(models, fill_value=0)
    reasons = list(counts.sum().sort_values(ascending=False).index)
    if len(reasons) > len(CATEGORICAL):  # A ninth reason folds into "other", rather than a generated color.
        counts["other"] = counts[reasons[len(CATEGORICAL) - 1:]].sum(axis=1)
        reasons = reasons[:len(CATEGORICAL) - 1] + ["other"]
    fig, (axis,) = _new_figure(len(models), f"Rejected trajectories by the reason recorded for them, "
                               f"{int(counts.to_numpy().sum()):,} in all.")
    bottom = np.zeros(len(models))
    for color, reason in zip(CATEGORICAL, reasons):
        axis.bar(range(len(models)), counts[reason].to_numpy(), 0.7, bottom=bottom, label=reason, color=color,
                 edgecolor=SURFACE, linewidth=1.5)
        bottom += counts[reason].to_numpy()
    _model_axis(axis, models)
    thousands(axis)
    top = counts.iloc[-1]
    panel(axis, "Why trajectories were rejected", f"model {models[-1]}: " + ", ".join(
        f"{top[reason]:,} {reason}" for reason in reasons[:3] if top[reason]), "Trajectories")
    _mark_windows(axis, windows, models)
    legend = axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), reverse=True, fontsize=NOTE)
    for text in legend.get_texts():
        text.set_color(MUTED)
    save(fig, path)
    return True


def _sampling_cost_plot(table, path, models, windows):
    """How much simulation each QM candidate took, and how much found nothing."""
    known = table[table["outcome"].notna() & table["simtime"].notna() & table["model"].isin(models)]
    if known.empty or not (known["outcome"] == CANDIDATE).any():
        return False
    rows = []
    for model in models:
        group = known[known["model"] == model]
        candidates = int((group["outcome"] == CANDIDATE).sum())
        total = group["simtime"].sum()
        rows.append((candidates, total / candidates if candidates else np.nan,
                     100 * group.loc[group["outcome"] != CANDIDATE, "simtime"].sum() / total if total > 0 else np.nan))
    per_candidate = np.array([row[1] for row in rows])
    wasted = np.array([row[2] for row in rows])
    fig, axes = figure(2, "Sampling cost",
                       "Simulated time spent per QM candidate, and the share of it in trajectories that gave none "
                       "(they ran to the end, or a screen rejected them). Tighter cutoffs find candidates sooner; "
                       "looser ones spend more simulation per candidate.", height=1.8,
                       width=max(WIDTH, 0.24 * len(models) + 3.0))
    positions = np.arange(len(models))
    axes[0].plot(positions, per_candidate, color=BOX_COLOR, lw=1.6, marker="o", ms=4)
    panel(axes[0], "Simulated time per QM candidate", f"model {models[-1]}: {per_candidate[-1]:.3g} ps over "
          f"{rows[-1][0]:,} candidates", "Picoseconds")
    axes[1].bar(positions, wasted, 0.6, color=BOX_COLOR, alpha=0.85)
    panel(axes[1], "Simulated time that found no candidate", f"model {models[-1]}: {wasted[-1]:.0f}%", "Percent")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        _model_axis(axis, models)
        axis.set_ylim(bottom=0)
        _mark_windows(axis, windows, models)
    save(fig, path)
    return True


def analysis_plot(meta_dir=None, plot_dir=None, meta_n_cols=50, trajectory_summary_path=None, Escut=None,
                  Fscut=None, gap_std_cut=None, acquisition=None):
    """Write progress plots, reading a trajectory summary if one is set, or else ``meta_dir``.

    Args:
        meta_dir (str): Directory of per-trajectory metadata pickles.
        plot_dir (str): Directory to write the plots to.
        meta_n_cols (int): Maximum number of models to show on a plot.
        trajectory_summary_path (str): File the coordinator writes one row per trajectory to.
        Escut, Fscut, gap_std_cut (float): The sampler's cutoffs, to draw them and tune them.
        acquisition (dict): The sampler's acquisition settings, for which signals trigger QM.

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
    thresholds = _cutoffs(Escut, Fscut, gap_std_cut)
    signals = _eligible_signals(acquisition, thresholds)
    peak = "Peak " if table["peak_Es"].notna().any() else ""
    plots = [("realtime", "Wall time per trajectory", "Seconds"),
             ("log_realtime", "Wall time per trajectory, log scale", "log10(seconds)"),
             ("simtime", "Simulated time per trajectory", "Picoseconds"),
             ("log_simtime", "Simulated time per trajectory, log scale", "log10(picoseconds)")]
    plots += [(name, f"{peak}{title}".capitalize(), unit) for name, _, _, _, title, unit in SIGNALS]
    states = sorted({_whole(state) for state in table["state"].dropna().unique()}, key=lambda state: (str(type(state)), state))
    written, empty = [], []

    def path(name):
        return os.path.join(plot_dir, f"analysis-plot-{name}.png")

    with plt.rc_context(RC):
        for name, title, ylabel in plots:
            threshold = thresholds.get(name)
            if _box_plot(table, name, title, ylabel, path(name), models, table, threshold):
                written.append(name)
            elif name != "gap_std":  # Only excited-state runs with gap_uncertainty record it.
                empty.append(name)
            if name == "gap_std":  # A gap belongs to a pair of states: one plot per recorded pair, by its lower state.
                pairs = sorted(table["gap_lower_state"].dropna().astype(float).astype(int).unique())
                for lower in pairs if len(pairs) > 1 else ():
                    subset = table[pd.to_numeric(table["gap_lower_state"], errors="coerce") == lower]
                    label = f"gap_std-states{lower}-{lower + 1}"
                    if _box_plot(subset, name, f"{title}, States {lower}–{lower + 1}", ylabel, path(label), models,
                                 table, threshold):
                        written.append(label)
            elif len(states) > 1:
                for state in states:
                    subset = table[table["state"].map(_whole) == state]
                    label = f"{name}-state{state}"
                    if _box_plot(subset, name, f"{title}, state {state}", ylabel, path(label), models, table,
                                 threshold):
                        written.append(label)
        extras = [("outcomes", lambda: _outcome_plot(table, path("outcomes"), models, table)),
                  ("rejections", lambda: _rejection_plot(table, path("rejections"), models, table)),
                  ("sampling-cost", lambda: _sampling_cost_plot(table, path("sampling-cost"), models, table))]
        if signals:
            extras += [("cutoff-tuning", lambda: _cutoff_tuning_plot(table, path("cutoff-tuning"), models, signals,
                                                                      thresholds)),
                       ("trigger-time", lambda: _trigger_time_plot(table, path("trigger-time"), models, signals,
                                                                    thresholds))]
        for name, draw in extras:
            if draw():
                written.append(name)
    if empty:
        print("Progress plots skipped for lack of data: " + ", ".join(empty), flush=True)
    return written
