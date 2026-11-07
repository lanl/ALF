"""Gap-seeking for excited-state sampling, using the Levine-Coe-Martinez penalty surface.

A replica running on its selected state switches permanently to that surface
for one adjacent state pair, once that pair's ensemble-mean gap falls below a
threshold. The surface drives the replica toward the crossing. It only changes
propagation forces: screening and uncertainty still read unbiased predictions.

The formulas work on tensors and on NumPy arrays alike, so this module needs no torch.
"""

import math

FIXED_OPTIONS = {
    "mode": "levine_coe_martinez_switch",
    "switch_policy": "stay_fixed",
    "candidate_pairs": "adjacent",
}
PARAMETERS = ("trigger_gap_threshold_eV", "sigma", "alpha_eV")
_SMOOTHING = 1.0e-12


def gap_seeking_settings(config):
    """Return validated ``gap_seeking`` parameters, or ``None`` when it is off."""
    raw = config.get("gap_seeking")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("gap_seeking must be a dictionary.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("gap_seeking.enabled must be Boolean.")
    if not enabled:
        return None
    unknown = set(raw) - {"enabled", *FIXED_OPTIONS, *PARAMETERS}
    if unknown:
        raise ValueError(f"Unknown gap_seeking options: {sorted(unknown)}.")
    for key, supported in FIXED_OPTIONS.items():
        if raw.get(key, supported) != supported:
            raise ValueError(f"gap_seeking.{key} must be '{supported}'.")
    if config.get("model_mode", "ground_state") != "excited_state":
        raise ValueError("gap_seeking requires model_mode excited_state.")
    if (config.get("MLMD_calculator_options") or {}).get("udd_bias_weight") is not None:
        # UDD's bias comes from the selected state's spread, which doesn't describe a gap-seeking replica.
        raise ValueError("gap_seeking can't be combined with udd_bias_weight.")
    settings = {}
    for name in PARAMETERS:
        if name not in raw:
            raise ValueError(f"Enabled gap_seeking requires {name}.")
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"gap_seeking.{name} must be finite and positive.")
        settings[name] = float(value)
    return settings


def adjacent_pairs(selected_state, n_states):
    """Return the ordered adjacent state pairs that contain ``selected_state``."""
    pairs = [(state, state + 1) for state in (selected_state - 1, selected_state) if 0 <= state < n_states - 1]
    if not pairs:
        raise ValueError(f"gap_seeking needs a state adjacent to state {selected_state}; "
                         f"the ensemble has {n_states} state(s).")
    return pairs


def select_trigger(mean_state_energies, pairs, threshold):
    """Return the pair and gap to switch to, or ``None``.

    Picks the smallest absolute ensemble-mean gap, and the lowest pair on a tie.

    Args:
        mean_state_energies: Ensemble-mean energy of each state, for one system.
        pairs (list): Candidate ``(lower, upper)`` state pairs.
        threshold (float): Largest absolute gap, in eV, that triggers a switch.
    """
    candidates = []
    for lower, upper in pairs:
        gap = float(mean_state_energies[upper]) - float(mean_state_energies[lower])
        if not math.isfinite(gap):
            raise ValueError(f"gap_seeking received a nonfinite gap for states {lower} and {upper}.")
        candidates.append((abs(gap), lower, upper, gap))
    best = min(candidates)
    return None if best[0] > threshold else {"pair": (best[1], best[2]), "gap_eV": best[3]}


def gap_seeking_energy_forces(lower_energy, upper_energy, lower_forces, upper_forces, sigma, alpha_eV):
    """Return the gap-seeking energy and its analytic force, from two states' ensemble means."""
    gap = upper_energy - lower_energy
    smooth_abs_gap = (gap * gap + _SMOOTHING) ** 0.5
    denominator = smooth_abs_gap + alpha_eV
    energy = 0.5 * (lower_energy + upper_energy) + sigma * gap * gap / denominator
    gap_derivative = sigma * (2.0 * gap * denominator - gap * gap * gap / smooth_abs_gap) / (denominator * denominator)
    return energy, 0.5 * (lower_forces + upper_forces) + gap_derivative * (upper_forces - lower_forces)
