"""Sampling windows: change how one run samples as its stored labels grow.

A run can start on ordinary sampling and move to UDD or gap-seeking later. It stays one
run, with one dataset and an ensemble that keeps retraining. Each window lists the
settings it replaces in the base sampler configuration; windows don't build on each other.
"""

UNTIL = "until_stored_labels"
OVERRIDES = ("gap_seeking", "MLMD_calculator_options")


def validated_windows(config):
    """Return the validated ``sampling_windows`` list, or ``None`` when it isn't set."""
    windows = config.get("sampling_windows")
    if windows is None:
        return None
    if not isinstance(windows, list) or not windows:
        raise ValueError("sampling_windows must be a nonempty list.")
    previous = 0
    for number, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"sampling_windows[{number}] must be a dictionary.")
        unknown = set(window) - {UNTIL, *OVERRIDES}
        if unknown:
            raise ValueError(f"sampling_windows[{number}] can't set {sorted(unknown)}; "
                             f"it can set {list(OVERRIDES)}.")
        if number == len(windows) - 1:
            if UNTIL in window:
                raise ValueError(f"The last sampling window must not set {UNTIL}.")
            continue
        limit = window.get(UNTIL)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= previous:
            raise ValueError(f"sampling_windows[{number}] requires an integer {UNTIL} above {previous}.")
        previous = limit
    return windows


def window_number(windows, stored_labels):
    """Return the number of the window that applies at this count of stored labels."""
    for number, window in enumerate(windows):
        if UNTIL not in window or stored_labels < window[UNTIL]:
            return number
    raise ValueError("The last sampling window must not set a limit.")


def windowed_config(config, windows, number):
    """Return a copy of the base configuration with one window's settings applied."""
    overrides = {key: value for key, value in windows[number].items() if key != UNTIL}
    return {**config, **overrides, "sampling_window": number}
