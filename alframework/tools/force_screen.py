"""Physical ensemble-force screening, independent of engines and propagation biases."""
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

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


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class RuntimeScreenAudit:
    """Bounded initial/last-valid/rejected evidence, never a full trajectory."""
    def __init__(self, molecules, config):
        directory = config.get('screen_audit_dir')
        self.path = Path(directory) / ('screen-'+uuid4().hex+'.json') if directory else None
        self.records = {m.get_moleculeid(): dict(initial=None, last_valid=None, rejected=None) for m in molecules}
        self.molecules = list(molecules)
        if self.path is not None:
            for m in molecules:
                m.update_metadata(dict(screen_audit_path=str(self.path)))

    def record(self, molecule, kind, atoms, step=0, time_ps=0., diagnostics=None):
        self.records[molecule.get_moleculeid()][kind] = dict(
            atomic_numbers=atoms.numbers.tolist(), positions=atoms.positions.tolist(),
            topology_atom_ids=None if 'topology_atom_ids' not in atoms.arrays else atoms.arrays['topology_atom_ids'].tolist(),
            step=int(step), time_ps=float(time_ps), diagnostics=diagnostics or {})

    def write(self):
        if self.path is None:
            for m in self.molecules:
                m.update_metadata(dict(screen_audit=_json_value(self.records[m.get_moleculeid()])))
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(_json_value(self.records), indent=2, allow_nan=False)+'\n')
            temp.replace(self.path)
