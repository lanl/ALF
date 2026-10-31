"""Acquisition: frame eligibility, ranking, and identity for MLMD sampling.

Eligibility is ALF's existing uncertainty trigger, unchanged from
``mlmd_sampling``:

- ``Es > Escut``, ``Fs > Fscut``, or ``Fsmax > 3 * Fscut``;
- only once simulated time reaches ``min_time``;
- a frame whose minimum interatomic distance is below ``distcut`` is rejected
  (it ends the trajectory without a candidate), regardless of uncertainty.

Ranking orders eligible frames and never makes an ineligible frame eligible.
Configure it with the sampler's ``acquisition`` block::

    "acquisition": {"strategy": "uncertainty"}
    "acquisition": {"strategy": "summed", "w_energy": 1.0, "w_force": 1.0}

``uncertainty`` ranks by the worst normalized violation,
``max(Es/Escut, Fs/Fscut, Fsmax/(3*Fscut))``. ``summed`` ranks by
``w_energy*Es/Escut + w_force*Fsrms/Fscut``; both weights are required.

Frames are identified by ``(trajectory_id, step)``. Ties in score are broken
by trajectory ID and then step, so selections do not depend on how
trajectories were grouped into tasks or the order in which tasks finished.
"""

import numpy as np

STRATEGIES = ("uncertainty", "summed")


def _weight(block, name):
    if name not in block:
        raise ValueError(f"acquisition.strategy 'summed' requires acquisition.{name}.")
    value = block[name]
    if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"acquisition.{name} must be a nonnegative number; received {value!r}.")
    return float(value)


def acquisition_settings(sampler_config):
    """Return validated acquisition settings; ``uncertainty`` when unconfigured."""
    block = dict(sampler_config.get("acquisition") or {})
    strategy = block.get("strategy", "uncertainty")
    if strategy not in STRATEGIES:
        raise ValueError(f"acquisition.strategy must be one of {STRATEGIES}; received {strategy!r}.")
    settings = {"strategy": strategy}
    if strategy == "summed":
        settings["w_energy"] = _weight(block, "w_energy")
        settings["w_force"] = _weight(block, "w_force")
        if settings["w_energy"] == 0 and settings["w_force"] == 0:
            raise ValueError("acquisition.strategy 'summed' requires a positive w_energy or w_force.")
    for cutoff in ("Escut", "Fscut"):
        if cutoff not in sampler_config:
            raise ValueError(f"Sampler configuration requires {cutoff}.")
    if strategy == "summed" and (sampler_config["Escut"] <= 0 or sampler_config["Fscut"] <= 0):
        # Normalizing by a nonpositive cutoff would invert or break the ranking.
        raise ValueError("acquisition.strategy 'summed' requires positive Escut and Fscut.")
    return settings


def uncertainty_flags(signals, Escut, Fscut):
    """Apply the existing energy, mean-force, and max-force thresholds."""
    Ecrit = signals["Es"] > Escut
    Fcrit = signals["Fs"] > Fscut
    Fmcrit = signals["Fsmax"] > 3 * Fscut
    return {"Ecrit": bool(Ecrit), "Fcrit": bool(Fcrit), "Fmcrit": bool(Fmcrit),
            "uncertain": bool(Ecrit or Fcrit or Fmcrit)}


def ranking_score(signals, settings, Escut, Fscut):
    if settings["strategy"] == "summed":
        return float(settings["w_energy"] * signals["Es"] / Escut + settings["w_force"] * signals["Fsrms"] / Fscut)
    ratios = []
    for value, cutoff in ((signals["Es"], Escut), (signals["Fs"], Fscut), (signals["Fsmax"], 3 * Fscut)):
        # A nonpositive cutoff always triggers; rank such channels above every finite ratio.
        ratios.append(value / cutoff if cutoff > 0 else np.inf)
    return float(max(ratios))


def assess_frame(signals, min_distance, time_ps, sampler_config, settings):
    """Classify one observed frame.

    Returns:
        (dict): ``outcome`` is ``"rejected"`` (``distcut`` violation, ending
        the trajectory), ``"eligible"`` (a candidate), or ``"continue"``; plus
        the uncertainty flags, ``distcrit``, and, for eligible frames,
        ``ranking_score``.
    """
    Escut, Fscut = sampler_config["Escut"], sampler_config["Fscut"]
    flags = uncertainty_flags(signals, Escut, Fscut)
    flags["distcrit"] = bool(min_distance < sampler_config.get("distcut", 1.2))
    checked = time_ps >= sampler_config.get("min_time", 0)
    if checked and flags["distcrit"]:
        flags["outcome"] = "rejected"
    elif checked and flags["uncertain"]:
        flags["outcome"] = "eligible"
        flags["ranking_score"] = ranking_score(signals, settings, Escut, Fscut)
    else:
        flags["outcome"] = "continue"
    return flags


def frame_id(trajectory_id, step):
    return "{}:step-{:09d}".format(trajectory_id, int(step))


def ranking_key(candidate):
    """Sort key: highest score first, then trajectory ID, then step."""
    return (-float(candidate["ranking_score"]), str(candidate["trajectory_id"]), int(candidate["step"]))


def top_k(candidates, k):
    """Return the ``k`` best candidates in deterministic rank order."""
    return sorted(candidates, key=ranking_key)[:int(k)]
