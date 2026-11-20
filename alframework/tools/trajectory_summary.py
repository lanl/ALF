"""One compact row per trajectory, written by the coordinator to a single file.

The progress plots read this instead of ``meta_dir``, which writes one pickle per trajectory into
one directory, from every sampler worker. Here the coordinator is the only writer, using metadata it
already receives through Parsl, so a large run adds one file and no load from the workers.
"""
import json
import math
from array import array
import os
import re

MODEL_IN_ID = re.compile(r'mol-(\d+)-')
NUMBERS = {"simtime": "simulationtime", "realtime": "realtime_simulation", "Es": "Es", "Fs": "Fs", "Fsmax": "Fsmax",
           "peak_Es": "peak_Es", "peak_Fs": "peak_Fs", "peak_Fsmax": "peak_Fsmax",
           "gap_std": "gap_std", "peak_gap_std": "peak_gap_std",
           "initial_kinetic_temperature_K": "initial_kinetic_temperature_K",
           "kinetic_temperature_K": "kinetic_temperature_K", "thermostat_temperature_K": "thermostat_temperature_K"}
LABELS = {"outcome": "trajectory_outcome", "reason": "screen_reject_reason", "state": "selected_state",
          "window": "sampling_window", "gap_lower_state": "gap_std_lower_state"}
COLUMNS = ("model", *LABELS, *NUMBERS)
# A row also holds ``max_uncertainty_<name>``, the largest value during the trajectory, for each uncertainty of the
# model's own that the run's acquisition terms read.
OWN_UNCERTAINTY_PREFIX = "max_uncertainty_"
TASK_FAILED = "task_failed"


def _number(value):
    """Return a finite float, or ``None`` for anything else, because JSON can't hold NaN."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _label(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    try:
        return int(value)  # NumPy integers.
    except (TypeError, ValueError):
        return str(value)


def trajectory_row(moleculeid, metadata, outcome=None):
    """Return one trajectory's row. ``outcome`` overrides what its metadata says.

    Args:
        moleculeid (str): The trajectory's ID, such as ``mol-0003-0000000012``.
        metadata (dict): Its sampler metadata.
        outcome (str): Optional outcome, for a trajectory whose task failed and returned nothing.

    Returns:
        (dict): The trajectory's ID and every column in ``COLUMNS``.
    """
    match = MODEL_IN_ID.match(str(moleculeid))
    row = {"id": str(moleculeid), "model": int(match.group(1)) if match else None}
    row.update({column: _label(metadata.get(key)) for column, key in LABELS.items()})
    row.update({column: _number(metadata.get(key)) for column, key in NUMBERS.items()})
    # The largest values of the model's own uncertainty the run's acquisition terms read (acquisition.terms).
    for name, largest in (metadata.get("max_uncertainty") or {}).items():
        row[OWN_UNCERTAINTY_PREFIX + str(name)] = _number(largest)
    if outcome is not None:
        row["outcome"] = outcome
    return row


class TrajectorySummary:
    """Append trajectory rows to one file. A failure warns once, and never stops the run."""

    def __init__(self, path):
        self.path = str(path)
        self._warned = False

    def append(self, structures, failed=()):
        """Record harvested results, and the members of tasks that failed as a whole."""
        self.append_rows((trajectory_row(molecule.get_moleculeid(), molecule.get_metadata())
                          for molecule in structures), failed)

    def append_rows(self, rows, failed=()):
        """Write precomputed or streamed rows without assembling an entire wave in memory."""
        from itertools import chain

        try:
            rows = chain(rows, (trajectory_row(molecule.get_moleculeid(), molecule.get_metadata(), TASK_FAILED)
                                for molecule in failed))
            first = next(rows, None)
            if first is None:
                return
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "ab+") as handle:
                # A crash can leave a partial line. Finish it, so that only that row is lost.
                if handle.seek(0, os.SEEK_END) > 0:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
                for row in chain((first,), rows):
                    handle.write((json.dumps(row) + "\n").encode("utf-8"))
        except Exception as exc:
            if not self._warned:
                print(f"Could not write the trajectory summary {self.path}: {type(exc).__name__}: {exc}", flush=True)
                self._warned = True


def read_trajectory_summary(path):
    """Read a summary into columns, skipping damaged lines.

    A restart can repeat a trajectory, in which case its last row wins. Numbers are stored in compact
    arrays and repeated labels are shared, because a file can hold millions of rows.

    Args:
        path (str): The summary file.

    Returns:
        (dict): For each name in ``COLUMNS``, and each ``max_uncertainty_<name>`` any row has, a list of labels or an
        array of floats with NaN for missing values, all the same length.
    """
    columns = {name: array("d") if name in NUMBERS else [] for name in COLUMNS}
    own = {}  # max_uncertainty_<name> columns, created when a row first has one
    positions = {}
    shared = {}

    def value(name, row):
        item = row.get(name)
        if name in NUMBERS:
            return math.nan if item is None else item
        return shared.setdefault(item, item)

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                position = positions.get(row["id"])
                values = [value(name, row) for name in COLUMNS]
            except (ValueError, TypeError, KeyError):
                continue
            if position is None:
                position = positions[row["id"]] = len(columns["model"])
                for name, item in zip(COLUMNS, values):
                    columns[name].append(item)
                for name in own:
                    own[name].append(math.nan)
            else:
                for name, item in zip(COLUMNS, values):
                    columns[name][position] = item
            for name, item in row.items():
                if name.startswith(OWN_UNCERTAINTY_PREFIX) and isinstance(item, (int, float)):
                    if name not in own:
                        own[name] = array("d", [math.nan] * len(columns["model"]))
                    own[name][position] = item
    columns.update(own)
    return columns
