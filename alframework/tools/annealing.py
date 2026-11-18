"""Annealing schedules used by sampling tasks."""

import numpy as np


def annealing_schedule(t, tmax, amp, per, srt, end):
    """Defines the overall temperature profile in the molecular dynamics simulation.

    Args:
        t (float): Current simulation time.
        tmax (float): Total simulation time.
        amp (float): Amplitude of the temperature oscillation.
        per (float): Period of the temperature oscillation.
        srt (float): Initial temperature.
        end (float): Final temperature.

    Returns:
        (float): Temperature at step 't' given the specified annealing schedule.

    """
    linear = t / tmax
    linear_T = (1 - linear) * srt + linear * end

    return amp * np.sin(np.pi * t / per)**2 + linear_T
