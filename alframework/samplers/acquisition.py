"""Acquisition: frame eligibility, ranking, selection, and identity for MLMD sampling.

Eligibility is ALF's existing uncertainty trigger, unchanged from
``mlmd_sampling``:

- ``Es > Escut``, ``Fs > Fscut``, or ``Fsmax > 3 * Fscut``;
- only once simulated time reaches ``min_time``;
- a frame whose minimum interatomic distance is below ``distcut`` is rejected
  (it ends the trajectory), regardless of uncertainty. With
  ``acquisition.distcut_frames`` "label" it is a frame to label instead.

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

``random`` admits frames that pass the time and physical screens and ranks
them by a seeded score derived from their trajectory ID and step.

Selection is configured with ``trajectory_policy`` and ``selection``::

    "acquisition": {"trajectory_policy": "continue",
                    "selection": {"mode": "cohort_top_k", "cohort_size": 50, "k": 16,
                                  "max_candidates_per_trajectory": 4}}

``first_trigger`` (default) returns each trajectory's first eligible frame.
``final_frame`` runs each trajectory to its configured duration and returns the
final frame. Uncertainty and distance thresholds produce diagnostics in this
mode. Explicit topology and physical-force screens still reject frames.
``distcut_frames`` is "reject" (default) or "label". With "label", a ``distcut``
violation is a frame to label rather than a rejection: ``first_trigger`` stops
there and labels it, and ``final_frame`` still runs to its duration but labels
the first such frame instead of the final one. It doesn't apply to
``cohort_top_k``.
``cohort_top_k`` pools eligible frames from ``cohort_size`` trajectories and
labels the ``k`` best (see :mod:`alframework.tools.cohorts`). Under
``trajectory_policy`` ``continue`` a trajectory keeps sampling after an
eligible frame, optionally keeping only its best
``max_candidates_per_trajectory`` frames.

Frames are identified by ``(trajectory_id, step)``. Ties in score are broken
by trajectory ID and then step, so selections do not depend on how
trajectories were grouped into tasks or the order in which tasks finished.
"""

import hashlib

import numpy as np

STRATEGIES = ("uncertainty", "summed", "largest", "random")
TRAJECTORY_POLICIES = ("stop", "continue")
SELECTION_MODES = ("first_trigger", "final_frame", "cohort_top_k")
DISTCUT_FRAMES = ("reject", "label")
COHORT_SELECTORS = ("top_k", "fps", "adaptive_ntk")

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
# The settings each built-in term needs, for runs that use only terms of their own.
SPREAD_CUTOFFS = ("Escut", "Fscut")
ENSEMBLE_FORCE_SELECTED = "ensemble_force_selected_max"
NTK_FORCE_SELECTED = "ntk_force_selected_max"
SELECTED_FORCE_SIGNALS = (ENSEMBLE_FORCE_SELECTED, NTK_FORCE_SELECTED)


def custom_terms(sampler_config):
    """The run's own acquisition terms, ``acquisition.terms``: ``{name: {"uncertainty": key, "cutoff": value}}``.

    A term reads the named per-structure ``uncertainty`` an evaluator or ASE calculator reports of its own (see
    :mod:`alframework.ml_interfaces.ensemble_inference`) and is normalized by ``cutoff``, like the built-in terms.
    Names can't reuse a built-in term's.
    """
    block = (sampler_config.get("acquisition") or {}).get("terms")
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError('acquisition.terms must map names to {"uncertainty": ..., "cutoff": ...}.')
    terms = {}
    for name, term in block.items():
        if name in SCORE_TERMS:
            raise ValueError(f"acquisition.terms {name!r} is a built-in term; choose another name.")
        if not isinstance(term, dict) or set(term) != {"uncertainty", "cutoff"}:
            raise ValueError(f'acquisition.terms {name!r} must be {{"uncertainty": ..., "cutoff": ...}}.')
        if not isinstance(term["uncertainty"], str) or not term["uncertainty"]:
            raise ValueError(f"acquisition.terms {name!r} uncertainty must name what the model reports.")
        cutoff = term["cutoff"]
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError(f"acquisition.terms {name!r} cutoff must be a positive number; received {cutoff!r}.")
        terms[name] = {"uncertainty": term["uncertainty"], "cutoff": float(cutoff)}
    return terms


def term_signal_and_cutoff(term, sampler_config):
    """The signal a term reads and the cutoff that normalizes it."""
    if term in SCORE_TERMS:
        signal, cutoff_key, multiplier = SCORE_TERMS[term]
        return signal, multiplier * sampler_config[cutoff_key]
    custom = (sampler_config.get("acquisition") or {}).get("terms") or {}
    if term not in custom:
        raise ValueError(f"Unknown acquisition term {term!r}.")
    return custom[term]["uncertainty"], float(custom[term]["cutoff"])


def used_terms(settings):
    """Every term a run's acquisition reads: its eligibility, its ranking, and its positive weights."""
    ranking = {"uncertainty": UNCERTAINTY_TERMS, "largest": settings["eligibility"],
               "random": (),
               "summed": [term for term, weight in settings.get("weights", {}).items() if weight > 0]}
    return set(settings["eligibility"]) | set(ranking[settings["strategy"]])


def uses_spread(sampler_config):
    """Whether acquisition reads an ensemble spread and therefore needs two or more members."""
    settings = acquisition_settings(sampler_config)
    return (any(term in SCORE_TERMS for term in used_terms(settings))
            or ENSEMBLE_FORCE_SELECTED in settings.get("selected_force_signals", ()))


def resolve_force_atom_indices(selection, atoms, trajectory_id):
    """Resolve a validated selection in this trajectory's original atom order."""
    if selection is None:
        return None
    context = f"acquisition.force_atoms for trajectory {trajectory_id!r}"
    if "indices" in selection:
        indices = selection["indices"]
        if max(indices) >= len(atoms):
            raise ValueError(f"{context}: an index is out of range for {len(atoms)} atoms.")
        return indices
    name = selection["array"]
    mask = atoms.arrays.get(name)
    if not isinstance(mask, np.ndarray) or mask.shape != (len(atoms),) or mask.dtype.kind != "b":
        raise ValueError(f"{context}: ASE atom array {name!r} must be a boolean vector of length {len(atoms)}.")
    indices = tuple(np.flatnonzero(mask).tolist())
    if not indices:
        raise ValueError(f"{context}: ASE atom array {name!r} selects no atoms.")
    return indices


def selected_force_diagnostics(signals, sources, indices, *, member_forces=None, atomic=None, selected=None):
    """Add selected-atom signals before eligibility and return the corresponding diagnostics."""
    scores = {}
    for source in sources:
        if source == ENSEMBLE_FORCE_SELECTED:
            if member_forces is None or len(member_forces) < 2:
                raise ValueError(f"{source} requires forces from at least two ensemble members.")
            spread = np.std(np.asarray(member_forces, dtype=float)[:, indices, :], axis=0)
            values = np.linalg.norm(spread, axis=-1)
        elif selected and source in selected:
            values = np.asarray(selected[source], dtype=float)
        elif atomic and "ntk_force" in atomic:
            values = np.asarray(atomic["ntk_force"], dtype=float)[list(indices)]
        else:
            raise ValueError(f"{source} requires exact NTK atom scores. Use force_score='selected' or 'exact'.")
        if values.shape != (len(indices),) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid per-atom uncertainty values for {source}.")
        signals[source] = float(values.max())
        scores[source] = values.copy()
    return {"force_atom_indices": list(indices), "selected_force_uncertainty": scores}


SINGLE_MEMBER = ("{what} has one member, so no spread, but this run's acquisition reads the ensemble's spread "
                 "(the default energy and force terms, Escut and Fscut, or another built-in term). Train two or more "
                 "members, or choose frames by the model's own uncertainty: name it in acquisition.terms "
                 "and use only those terms in acquisition.eligibility.")


def member_count(models):
    """How many members a loaded ensemble has: its ``n_models``, the length of a list of calculators, or ``None``."""
    count = getattr(models, 'n_models', None)
    if count is None and isinstance(models, (list, tuple)):
        count = len(models)
    return count


def require_members(models, sampler_config, what='The sampler\'s model'):
    """Refuse a one-member model for a run whose acquisition reads the ensemble's spread."""
    if member_count(models) == 1 and uses_spread(sampler_config):
        raise ValueError(SINGLE_MEMBER.format(what=what))


def _nonnegative(value, name):
    if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a nonnegative number; received {value!r}.")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")
    return int(value)


def _summed_weights(block, known=tuple(SCORE_TERMS)):
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
    unknown = set(weights) - set(known)
    if unknown:
        raise ValueError(f"Unknown acquisition score terms: {sorted(unknown)}; available: {sorted(known)}.")
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
    if mode != "cohort_top_k":
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
    unknown = set(block) - {"strategy", "weights", "trajectory_policy", "selection", "eligibility", "terms",
                            "distcut_frames", "force_atoms", *WEIGHT_SHORTHAND}
    if unknown:
        raise ValueError(f"Unknown acquisition settings: {sorted(unknown)}.")
    strategy = block.get("strategy", "uncertainty")
    if strategy not in STRATEGIES:
        raise ValueError(f"acquisition.strategy must be one of {STRATEGIES}; received {strategy!r}.")
    custom = custom_terms(sampler_config)
    known = {*SCORE_TERMS, *custom}
    settings = {"strategy": strategy}
    eligibility = block.get('eligibility', [] if strategy == 'random' else list(UNCERTAINTY_TERMS))
    if strategy == 'random' and eligibility:
        raise ValueError('Random acquisition accepts all frames that pass the time and geometry screens.')
    if not isinstance(eligibility, (list, tuple)) or (not eligibility and strategy != 'random') or any(not isinstance(t, str) for t in eligibility) or len(set(eligibility)) != len(eligibility):
        raise ValueError('acquisition.eligibility requires distinct named score terms')
    if set(eligibility) - known:
        raise ValueError(f'Unknown acquisition eligibility terms {sorted(set(eligibility) - known)}; '
                         f'available: {sorted(known)}.')
    settings['eligibility'] = list(eligibility)
    if 'eligibility' in block:
        for term in (term for term in eligibility if term in SCORE_TERMS):
            cutoff = SCORE_TERMS[term][1]
            if not np.isfinite(float(sampler_config.get(cutoff, 0))) or sampler_config.get(cutoff, 0) <= 0:
                raise ValueError(f'Explicit eligibility requires a positive finite {cutoff}')
    if strategy == "summed":
        settings["weights"] = _summed_weights(block, known)
        for term, weight in settings["weights"].items():
            if term not in SCORE_TERMS:
                continue
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
    if "force_atoms" in block:
        selection = block["force_atoms"]
        if isinstance(selection, dict) and selection.keys() == {"indices"}:
            indices = selection["indices"]
            if (not isinstance(indices, (list, tuple)) or not indices
                    or any(type(index) is not int or index < 0 for index in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError("acquisition.force_atoms.indices requires distinct nonnegative integer indices.")
            settings["force_atoms"] = {"indices": tuple(indices)}
        elif (isinstance(selection, dict) and selection.keys() == {"array"}
              and isinstance(selection["array"], str) and selection["array"]):
            settings["force_atoms"] = {"array": selection["array"]}
        else:
            raise ValueError('acquisition.force_atoms requires either {"indices": [...]} or {"array": "..."}')
    used_signals = {custom[term]["uncertainty"] for term in used_terms(settings) & custom.keys()}
    selected_sources = tuple(source for source in SELECTED_FORCE_SIGNALS if source in used_signals)
    if selected_sources:
        if "force_atoms" not in settings:
            raise ValueError("Selected force uncertainty requires acquisition.force_atoms.")
        settings["selected_force_signals"] = selected_sources
    used = set(eligibility) | {key for key, weight in settings.get('weights', {}).items() if weight > 0}
    if 'gap' in used:
        from alframework.tools.sampler_coordinator import state_settings
        states = state_settings(sampler_config)
        if states is None or states['gap_pair'] is None:
            raise ValueError('Gap acquisition requires excited_state mode and gap_uncertainty.pair or .pairs')
    # Escut and Fscut are needed by the built-in terms; a run that uses only terms of its own can leave them out.
    for cutoff in sorted({SCORE_TERMS[term][1] for term in used_terms(settings) if term in SCORE_TERMS}):
        if cutoff not in sampler_config:
            raise ValueError(f"Sampler configuration requires {cutoff}.")
    settings["selection"] = _selection(block)
    if settings['selection'].get('group_by_selected_state') and sampler_config.get('model_mode') != 'excited_state':
        raise ValueError('State-grouped cohorts require model_mode excited_state')
    if policy == "continue" and settings["selection"]["mode"] != "cohort_top_k":
        raise ValueError("acquisition.trajectory_policy 'continue' requires selection.mode 'cohort_top_k'.")
    distcut_frames = block.get("distcut_frames", "reject")
    if distcut_frames not in DISTCUT_FRAMES:
        raise ValueError(f"acquisition.distcut_frames must be one of {DISTCUT_FRAMES}; received {distcut_frames!r}.")
    if distcut_frames == "label" and settings["selection"]["mode"] == "cohort_top_k":
        raise ValueError("acquisition.distcut_frames 'label' requires selection.mode 'first_trigger' or 'final_frame'.")
    settings["distcut_frames"] = distcut_frames
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
    signal, cutoff = term_signal_and_cutoff(term, sampler_config)
    if signal not in signals:
        raise ValueError(f"Acquisition score term {term!r} requires the {signal!r} signal from inference.")
    if not np.isfinite(signals[signal]):
        raise ValueError(f'Nonfinite acquisition signal {signal!r}')
    # A nonpositive cutoff always triggers; rank such channels above every finite ratio.
    return signals[signal] / cutoff if cutoff > 0 else np.inf


def ranking_score(signals, settings, sampler_config):
    if settings["strategy"] == "summed":
        return float(sum(weight * score_term(term, signals, sampler_config)
                         for term, weight in settings["weights"].items() if weight > 0))
    terms = settings["eligibility"] if settings["strategy"] == "largest" else UNCERTAINTY_TERMS
    return float(max(score_term(term, signals, sampler_config) for term in terms))


def assess_frame(signals, min_distance, time_ps, sampler_config, settings, *, trajectory_id=None, step=None):
    """Classify one observed frame.

    Returns:
        (dict): ``outcome`` is ``"rejected"`` (``distcut`` violation, ending
        the trajectory), ``"eligible"`` (a candidate, which with ``distcut_frames``
        "label" includes a ``distcut`` violation), or ``"continue"``; plus
        the uncertainty flags, ``distcrit``, and, for eligible frames,
        ``ranking_score``.
    """
    if all(key in sampler_config for key in SPREAD_CUTOFFS) and all(key in signals for key in ("Es", "Fs", "Fsmax")):
        flags = uncertainty_flags(signals, sampler_config["Escut"], sampler_config["Fscut"])
    else:  # a run that uses only terms of its own
        flags = {"Ecrit": False, "Fcrit": False, "Fmcrit": False}
    terms = settings.get('eligibility', UNCERTAINTY_TERMS)
    trigger_flags, cutoffs = {}, {}
    for term in terms:
        score_term(term, signals, sampler_config)  # Validate the requested signal.
        signal, cutoffs[term] = term_signal_and_cutoff(term, sampler_config)
        trigger_flags[term] = bool(signals[signal] > cutoffs[term])
    flags['uncertain'] = any(trigger_flags.values())
    flags['acquisition_trigger_flags'] = trigger_flags
    flags['acquisition_cutoffs'] = cutoffs
    flags["distcrit"] = bool(min_distance < sampler_config.get("distcut", 1.2))
    checked = time_ps >= sampler_config.get("min_time", 0)
    label_contact = flags["distcrit"] and settings.get("distcut_frames", "reject") == "label"
    if checked and flags["distcrit"] and not label_contact:
        flags["outcome"] = "rejected"
    elif checked and settings.get('strategy') == 'random':
        key = f'{sampler_config.get("random_seed", 0)}|{frame_id(trajectory_id, step)}'.encode()
        score = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
        flags.update(outcome='eligible', ranking_score=score, acquisition_score_terms={'random': score})
    elif checked and (flags["uncertain"] or label_contact):
        flags["outcome"] = "eligible"
        flags["ranking_score"] = ranking_score(signals, settings, sampler_config)
        ranked = terms if settings.get('strategy') == 'largest' else UNCERTAINTY_TERMS
        weighted = settings.get('weights', {term: 1. for term in ranked})
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
    selector = selection.get("selector", "top_k")
    if selector == "fps":
        return fps_select(candidates, selection["k"])
    if selector == "adaptive_ntk":
        from alframework.samplers.adaptive_ntk import select_ntk_candidates
        return select_ntk_candidates(candidates, selection["k"])
    return top_k(candidates, selection["k"])


def worker_retention_limit(selection):
    """Return how many frames one trajectory must keep for this selection.

    Top-K can never select a frame outside a trajectory's own best ``k``. FPS and adaptive
    NTK can, so they keep what the explicit per-trajectory cap allows.
    """
    cap = selection.get("max_candidates_per_trajectory")
    if selection.get("selector", "top_k") in ("fps", "adaptive_ntk"):
        return cap
    return selection["k"] if cap is None else min(selection["k"], cap)
