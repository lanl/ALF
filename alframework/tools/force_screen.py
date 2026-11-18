"""Physical ensemble-force screening, independent of engines and propagation biases."""

import numpy as np

# Preserve the previous import path.
from alframework.tools.screen_audit import RuntimeScreenAudit

SCHEMA = 'alf.physical_force_screen.v1'
SOURCE = 'physical_ensemble_mean'


def validated_force_cutoff(value):
    if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0:
        raise ValueError('max_force_cutoff must be finite and positive')
    return float(value)


def physical_force_diagnostics(prediction, system=0):
    """Reduce unbiased predictions: mean over members, atom norms, max over states.

    Accept the existing NumPy observation contract. State forces, when present,
    are [members, states, atoms, 3]; ground-state forces are [members, atoms, 3].
    Artificial/propagation-force fields are deliberately not read.
    """
    if 'state_forces' in prediction:
        forces = np.asarray(prediction['state_forces'][system], dtype=float)
        energies = np.asarray(prediction['state_energies'])
        if forces.ndim != 4 or energies.ndim != 3 or forces.shape[:2] != energies.shape[:2]:
            raise ValueError('Physical state forces must have member/state/atom/component axes')
    else:
        forces = np.asarray(prediction['forces'][system], dtype=float)[:, None]
    if forces.ndim != 4 or forces.shape[0] < 1 or forces.shape[2] < 1 or forces.shape[-1] != 3:
        raise ValueError('Invalid physical member-force shape')
    means = forces.mean(0)
    maxima = np.linalg.norm(means, axis=-1).max(-1)
    maxima[~np.isfinite(forces).all(axis=(0, 2, 3))] = np.inf
    return dict(schema=SCHEMA, source=SOURCE, Fmeanmax=float(maxima.max()),
                physical_forces={f'F{s}': f.tolist() for s, f in enumerate(means)},
                **{f'physical_F{s}_max': float(value) for s, value in enumerate(maxima)})


def physical_force_exceeds(diagnostics, cutoff):
    if diagnostics.get('schema') != SCHEMA or diagnostics.get('source') != SOURCE:
        raise ValueError('Force cutoff requires physical-model schema/source markers')
    return not np.isfinite(diagnostics['Fmeanmax']) or diagnostics['Fmeanmax'] > validated_force_cutoff(cutoff)


def force_screen_metadata(diagnostics):
    return {key: value for key, value in diagnostics.items() if key == 'Fmeanmax' or key.startswith('physical_F')} | {
        'physical_force_schema': SCHEMA, 'physical_force_source': SOURCE}
