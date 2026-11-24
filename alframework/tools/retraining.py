"""When a run retrains: the master configuration's ``retraining`` block.

::

    "retraining": {"after": 2048, "counts": "sampled"}

``after`` (N) starts a retraining once N have come in since the last one, counted as ``counts`` says:

- ``"labels"`` (the default): new converged labels.
- ``"sampled"``: sampled structures that are resolved, as long as at least one new label is held. A structure is
  resolved once its label has finished, converged or not, or once sampling ended without one to label: a
  trajectory rejected for ``distcut``, a topology change or a physical-force screen, one that never triggered
  acquisition or wasn't selected from its cohort, and the members of a sampler task that failed. A round of N
  trajectories then retrains once each is accounted for, however many of them yield a label, instead of waiting for
  more sampling to make up the ones that didn't.

The older ``save_h5_threshold`` still works as before, as ``"counts": "labels"`` with ``after`` one more than it: it
retrains once more than ``save_h5_threshold`` labels are held. A configuration gives one or the other.

The block is read where it's used, so an edit applies at the next reload; an edit that isn't valid is refused, and
the run keeps its previous settings.
"""
import math
import numbers

RETRAINING = 'retraining'
OLDER_THRESHOLD = 'save_h5_threshold'
COUNTS = ('labels', 'sampled')
RETRAINING_KEYS = ('after', 'counts')
# In the status file: sampled structures resolved without a label since labels were last stored.
SAMPLED_WITHOUT_LABEL = 'sampled_without_label_since_store'


MISSING = (f'Set {RETRAINING}.after in the master configuration: how many new labels start a retraining (or the '
           f'older {OLDER_THRESHOLD}).')


def retraining_settings(master_config, *, required=True):
    """``{'after': N, 'counts': 'labels' | 'sampled'}`` from the master configuration's ``retraining`` block or its
    older ``save_h5_threshold``. Without either, ``None`` if not ``required``, as for the test modes, which never retrain."""
    block, threshold = master_config.get(RETRAINING), master_config.get(OLDER_THRESHOLD)
    if block is None:
        if threshold is None:
            if not required:
                return None
            raise ValueError(MISSING)
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            raise ValueError(f'{OLDER_THRESHOLD} must be a number; received {threshold!r}.')
        return {'after': math.floor(threshold) + 1, 'counts': 'labels'}
    if threshold is not None:
        raise ValueError(f'Give {RETRAINING} or {OLDER_THRESHOLD}, not both. {RETRAINING}.after N retrains once N new '
                         f'labels are held, as {OLDER_THRESHOLD} N - 1 did.')
    if not isinstance(block, dict):
        raise ValueError(f'{RETRAINING} must be a dictionary, such as {{"after": 2048}}.')
    unknown = sorted(set(block) - set(RETRAINING_KEYS))
    if unknown:
        raise ValueError(f'Unknown {RETRAINING} keys {unknown}; it takes {list(RETRAINING_KEYS)}.')
    after = block.get('after')
    if isinstance(after, bool) or not isinstance(after, int) or after < 1:
        raise ValueError(f'{RETRAINING}.after must be a positive integer; received {after!r}.')
    counts = block.get('counts', 'labels')
    if counts not in COUNTS:
        raise ValueError(f'{RETRAINING}.counts must be one of {list(COUNTS)}; received {counts!r}.')
    return {'after': after, 'counts': counts}


def resolved(settings, *, converged, finished, sampled_without_label):
    """How far the run is toward ``settings['after']``, in the unit ``settings['counts']`` names.

    ``converged`` counts held labels that converged and ``finished`` every held result plus QM failures, both since
    labels were last stored; ``sampled_without_label`` counts structures whose sampling ended without a frame to
    label in that time.
    """
    return converged if settings['counts'] == 'labels' else finished + sampled_without_label


def ready_to_retrain(settings, *, converged, finished, sampled_without_label):
    """Whether enough has come in to retrain (see :func:`resolved`). Counting sampled structures still needs a new
    converged label, so a round of only rejections or failures doesn't retrain on unchanged data."""
    if settings['counts'] == 'sampled' and converged < 1:
        return False
    count = resolved(settings, converged=converged, finished=finished, sampled_without_label=sampled_without_label)
    return count >= settings['after']


def describe(settings):
    """One line for the coordinator's startup report."""
    if settings['counts'] == 'labels':
        return f'retraining: once {settings["after"]:,} new labels are held'
    return (f'retraining: once {settings["after"]:,} sampled structures are resolved (labeled, or ended without a '
            'label), with at least one new label')
