"""Acquisition: frame eligibility, ranking, selection, and identity for MLMD sampling.

Eligibility is ALF's existing uncertainty trigger, unchanged from
``mlmd_sampling``:

- ``Es > Escut``, ``Fs > Fscut``, or ``Fsmax > 3 * Fscut``;
- only once simulated time reaches ``min_time``;
- a frame whose minimum interatomic distance is below ``distcut`` is rejected
  (it ends the trajectory), regardless of uncertainty.

Ranking orders eligible frames and never makes an ineligible frame eligible.
Scores are built from normalized terms in :data:`SCORE_TERMS`:

==============  ====================
``energy``      ``Es / Escut``
``force_mean``  ``Fs / Fscut``
``force_max``   ``Fsmax / (3*Fscut)``
``force_rms``   ``Fsrms / Fscut``
==============  ====================

``uncertainty`` (default) ranks by the largest of ``energy``, ``force_mean``,
and ``force_max``: the worst violation of the three trigger thresholds.
``summed`` ranks by a weighted sum of the configured terms::

    "acquisition": {"strategy": "summed", "weights": {"energy": 1.0, "force_rms": 1.0}}

``w_energy``/``w_force`` are accepted as shorthand for the ``energy`` and
``force_rms`` weights. A new term (for example an excited-state gap
uncertainty) is one :data:`SCORE_TERMS` entry naming the signal it reads and
the configuration cutoff that normalizes it; the inference adapter must supply
that signal.

Selection is configured with ``trajectory_policy`` and ``selection``::

    "acquisition": {"trajectory_policy": "continue",
                    "selection": {"mode": "cohort_top_k", "cohort_size": 50, "k": 16,
                                  "max_candidates_per_trajectory": 4}}

``first_trigger`` (default) returns each trajectory's first eligible frame.
``cohort_top_k`` pools eligible frames from ``cohort_size`` trajectories and
labels the ``k`` best (see :mod:`alframework.tools.cohorts`). Under
``trajectory_policy`` ``continue`` a trajectory keeps sampling after an
eligible frame, optionally keeping only its best
``max_candidates_per_trajectory`` frames.

Frames are identified by ``(trajectory_id, step)``. Ties in score are broken
by trajectory ID and then step, so selections do not depend on how
trajectories were grouped into tasks or the order in which tasks finished.
"""

import numpy as np

STRATEGIES = ("uncertainty", "summed")
TRAJECTORY_POLICIES = ("stop", "continue")
SELECTION_MODES = ("first_trigger", "cohort_top_k")
COHORT_SELECTORS = ("top_k", "fps")

# name: (signal key from ensemble statistics, cutoff key in sampler_config, cutoff multiplier)
SCORE_TERMS = {
    "energy": ("Es", "Escut", 1.0),
    "force_mean": ("Fs", "Fscut", 1.0),
    "force_max": ("Fsmax", "Fscut", 3.0),
    "force_rms": ("Fsrms", "Fscut", 1.0),
    "gap": ("gap_std", "gap_std_cut", 1.0),
}
UNCERTAINTY_TERMS = ("energy", "force_mean", "force_max")
WEIGHT_SHORTHAND = {"w_energy": "energy", "w_force": "force_rms"}


def _nonnegative(value, name):
    if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a nonnegative number; received {value!r}.")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")
    return int(value)


def _summed_weights(block):
    weights = dict(block.get("weights") or {})
    shorthand = {WEIGHT_SHORTHAND[key]: block[key] for key in WEIGHT_SHORTHAND if key in block}
    if weights and shorthand:
        raise ValueError("Give acquisition.weights or w_energy/w_force, not both.")
    weights = weights or shorthand
    if not weights:
        raise ValueError("acquisition.strategy 'summed' requires acquisition.weights "
                         "(or w_energy and w_force).")
    if shorthand and set(shorthand) != set(WEIGHT_SHORTHAND.values()):
        missing = [key for key, term in WEIGHT_SHORTHAND.items() if term not in shorthand]
        raise ValueError(f"acquisition.strategy 'summed' requires acquisition.{missing[0]}.")
    unknown = set(weights) - set(SCORE_TERMS)
    if unknown:
        raise ValueError(f"Unknown acquisition score terms: {sorted(unknown)}; available: {sorted(SCORE_TERMS)}.")
    weights = {term: _nonnegative(value, f"acquisition weight {term!r}") for term, value in weights.items()}
    if not any(weights.values()):
        raise ValueError("acquisition.strategy 'summed' requires at least one positive weight.")
    return weights


def _selection(block):
    raw = dict(block.get("selection") or {})
    unknown = set(raw) - {"mode", "cohort_size", "k", "max_candidates_per_trajectory", "group_by_selected_state",
                          "selector"}
    if unknown:
        raise ValueError(f"Unknown acquisition.selection settings: {sorted(unknown)}.")
    mode = raw.get("mode", "first_trigger")
    if mode not in SELECTION_MODES:
        raise ValueError(f"acquisition.selection.mode must be one of {SELECTION_MODES}; received {mode!r}.")
    selection = {"mode": mode}
    if mode == "first_trigger":
        if set(raw) - {"mode"}:
            raise ValueError("cohort_size, k, and max_candidates_per_trajectory require "
                             "acquisition.selection.mode 'cohort_top_k'.")
        return selection
    for key in ("cohort_size", "k"):
        if key not in raw:
            raise ValueError(f"acquisition.selection.mode 'cohort_top_k' requires selection.{key}.")
        selection[key] = _positive_integer(raw[key], f"acquisition.selection.{key}")
    group = raw.get('group_by_selected_state', False)
    if not isinstance(group, bool):
        raise ValueError('selection.group_by_selected_state must be Boolean')
    if group:
        selection['group_by_selected_state'] = True
    cap = raw.get("max_candidates_per_trajectory")
    selection["max_candidates_per_trajectory"] = None if cap is None else \
        _positive_integer(cap, "acquisition.selection.max_candidates_per_trajectory")
    selector = raw.get("selector", "top_k")
    if selector not in COHORT_SELECTORS:
        raise ValueError(f"acquisition.selection.selector must be one of {COHORT_SELECTORS}; received {selector!r}.")
    if selector != "top_k":
        if cap is None:
            raise ValueError(f"acquisition.selection.selector '{selector}' requires "
                             "max_candidates_per_trajectory, which sets how many frames workers keep for it.")
        selection["selector"] = selector
    return selection


def acquisition_settings(sampler_config):
    """Return validated acquisition settings; ``uncertainty``/``first_trigger`` when unconfigured."""
    block = dict(sampler_config.get("acquisition") or {})
    unknown = set(block) - {"strategy", "weights", "trajectory_policy", "selection", "eligibility", *WEIGHT_SHORTHAND}
    if unknown:
        raise ValueError(f"Unknown acquisition settings: {sorted(unknown)}.")
    strategy = block.get("strategy", "uncertainty")
    if strategy not in STRATEGIES:
        raise ValueError(f"acquisition.strategy must be one of {STRATEGIES}; received {strategy!r}.")
    for cutoff in ("Escut", "Fscut"):
        if cutoff not in sampler_config:
            raise ValueError(f"Sampler configuration requires {cutoff}.")
    settings = {"strategy": strategy}
    eligibility = block.get('eligibility', list(UNCERTAINTY_TERMS))
    if not isinstance(eligibility, (list, tuple)) or not eligibility or any(not isinstance(t, str) for t in eligibility) or len(set(eligibility)) != len(eligibility):
        raise ValueError('acquisition.eligibility requires distinct named score terms')
    if set(eligibility) - set(SCORE_TERMS):
        raise ValueError('Unknown acquisition eligibility terms')
    settings['eligibility'] = list(eligibility)
    if 'eligibility' in block:
        for term in eligibility:
            cutoff = SCORE_TERMS[term][1]
            if not np.isfinite(float(sampler_config.get(cutoff, 0))) or sampler_config.get(cutoff, 0) <= 0:
                raise ValueError(f'Explicit eligibility requires a positive finite {cutoff}')
    if strategy == "summed":
        settings["weights"] = _summed_weights(block)
        for term, weight in settings["weights"].items():
            cutoff_key = SCORE_TERMS[term][1]
            if weight > 0 and (not np.isfinite(float(sampler_config.get(cutoff_key, 0))) or sampler_config.get(cutoff_key, 0) <= 0):
                # Normalizing by a nonpositive cutoff would invert or break the ranking.
                raise ValueError(f"acquisition.strategy 'summed' term {term!r} requires a positive {cutoff_key}.")
    elif "weights" in block or set(WEIGHT_SHORTHAND) & set(block):
        raise ValueError("acquisition weights apply only to strategy 'summed'.")
    policy = block.get("trajectory_policy", "stop")
    if policy not in TRAJECTORY_POLICIES:
        raise ValueError(f"acquisition.trajectory_policy must be one of {TRAJECTORY_POLICIES}; received {policy!r}.")
    settings["trajectory_policy"] = policy
    used = set(eligibility) | {key for key, weight in settings.get('weights', {}).items() if weight > 0}
    if 'gap' in used:
        from alframework.tools.sampler_coordinator import state_settings
        states = state_settings(sampler_config)
        if states is None or states['gap_pair'] is None:
            raise ValueError('Gap acquisition requires excited_state mode and gap_uncertainty.pair or .pairs')
    settings["selection"] = _selection(block)
    if settings['selection'].get('group_by_selected_state') and sampler_config.get('model_mode') != 'excited_state':
        raise ValueError('State-grouped cohorts require model_mode excited_state')
    if policy == "continue" and settings["selection"]["mode"] != "cohort_top_k":
        raise ValueError("acquisition.trajectory_policy 'continue' requires selection.mode 'cohort_top_k'.")
    return settings


def cohort_selection(sampler_config):
    """Return the ``cohort_top_k`` selection settings, or ``None`` for first-trigger sampling."""
    selection = acquisition_settings(sampler_config)["selection"]
    return selection if selection["mode"] == "cohort_top_k" else None


def uncertainty_flags(signals, Escut, Fscut):
    """Apply the existing energy, mean-force, and max-force thresholds."""
    Ecrit = signals["Es"] > Escut
    Fcrit = signals["Fs"] > Fscut
    Fmcrit = signals["Fsmax"] > 3 * Fscut
    return {"Ecrit": bool(Ecrit), "Fcrit": bool(Fcrit), "Fmcrit": bool(Fmcrit),
            "uncertain": bool(Ecrit or Fcrit or Fmcrit)}


def score_term(term, signals, sampler_config):
    """Return one normalized score term."""
    signal, cutoff_key, multiplier = SCORE_TERMS[term]
    if signal not in signals:
        raise ValueError(f"Acquisition score term {term!r} requires the {signal!r} signal from inference.")
    if not np.isfinite(signals[signal]):
        raise ValueError(f'Nonfinite acquisition signal {signal!r}')
    cutoff = multiplier * sampler_config[cutoff_key]
    # A nonpositive cutoff always triggers; rank such channels above every finite ratio.
    return signals[signal] / cutoff if cutoff > 0 else np.inf


def ranking_score(signals, settings, sampler_config):
    if settings["strategy"] == "summed":
        return float(sum(weight * score_term(term, signals, sampler_config)
                         for term, weight in settings["weights"].items() if weight > 0))
    return float(max(score_term(term, signals, sampler_config) for term in UNCERTAINTY_TERMS))


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
    terms = settings.get('eligibility', UNCERTAINTY_TERMS)
    trigger_flags = {}
    for term in terms:
        score_term(term, signals, sampler_config)  # Validate the requested signal.
        signal, cutoff, multiplier = SCORE_TERMS[term]
        trigger_flags[term] = bool(signals[signal] > sampler_config[cutoff] * multiplier)
    flags['uncertain'] = any(trigger_flags.values())
    flags['acquisition_trigger_flags'] = trigger_flags
    flags['acquisition_cutoffs'] = {term: sampler_config[SCORE_TERMS[term][1]] * SCORE_TERMS[term][2]
                                     for term in terms}
    flags["distcrit"] = bool(min_distance < sampler_config.get("distcut", 1.2))
    checked = time_ps >= sampler_config.get("min_time", 0)
    if checked and flags["distcrit"]:
        flags["outcome"] = "rejected"
    elif checked and flags["uncertain"]:
        flags["outcome"] = "eligible"
        flags["ranking_score"] = ranking_score(signals, settings, sampler_config)
        weighted = settings.get('weights', {term: 1. for term in UNCERTAINTY_TERMS})
        flags['acquisition_score_terms'] = {term: weight * score_term(term, signals, sampler_config)
                                            for term, weight in weighted.items() if weight > 0}
    else:
        flags["outcome"] = "continue"
    return flags


def frame_id(trajectory_id, step):
    return "{}:step-{:09d}".format(trajectory_id, int(step))


def candidate_key(candidate):
    return (str(candidate["trajectory_id"]), int(candidate["step"]))


def ranking_key(candidate):
    """Sort key: highest score first, then trajectory ID, then step."""
    return (-float(candidate["ranking_score"]), str(candidate["trajectory_id"]), int(candidate["step"]))


def top_k(candidates, k):
    """Return the ``k`` best candidates in deterministic rank order."""
    return sorted(candidates, key=ranking_key)[:int(k)]


def fps_select(candidates, k):
    """Return up to ``k`` geometrically diverse candidates, by farthest-point sampling.

    Candidates are put in a canonical order first, so that the result doesn't depend on how
    trajectories were grouped or the order in which they finished. Pair-distance descriptors
    need one composition and no periodic images.
    """
    if not candidates:
        return []
    from alframework.builders.rdkit_molecule import farthest_point_order, pair_descriptor
    ordered = sorted(candidates, key=lambda candidate: (str(candidate["trajectory_id"]), int(candidate["step"])))
    if any(np.any(candidate["pbc"]) for candidate in ordered):
        raise ValueError("FPS cohort selection supports nonperiodic molecules only.")
    if len({tuple(sorted(np.asarray(candidate["numbers"]).tolist())) for candidate in ordered}) > 1:
        raise ValueError("FPS cohort selection needs every candidate in a cohort to have the same composition.")
    features = np.concatenate([pair_descriptor(np.asarray(candidate["positions"])[None], candidate["numbers"])
                               for candidate in ordered])
    return [ordered[row] for row in farthest_point_order(features, min(int(k), len(ordered)))]


def select_candidates(candidates, selection):
    """Return the frames a finished cohort sends to QM, in rank order."""
    if selection.get("selector", "top_k") == "fps":
        return fps_select(candidates, selection["k"])
    return top_k(candidates, selection["k"])


def worker_retention_limit(selection):
    """Return how many frames one trajectory must keep for this selection.

    Top-K can never select a frame outside a trajectory's own best ``k``. FPS can, so it keeps
    what ``max_candidates_per_trajectory`` allows, choosing the highest-scoring frames.
    """
    cap = selection.get("max_candidates_per_trajectory")
    if selection.get("selector", "top_k") == "fps":
        return cap
    return selection["k"] if cap is None else min(selection["k"], cap)
