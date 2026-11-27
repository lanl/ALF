"""Record initial, last valid, and rejected frames from sampler screens."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np


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
            from alframework.tools.files import replaced_atomically
            with replaced_atomically(self.path, suffix='.tmp', durable=False) as temporary:
                temporary.write_text(json.dumps(_json_value(self.records), indent=2, allow_nan=False) + '\n')
