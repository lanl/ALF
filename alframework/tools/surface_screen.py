"""Connectivity and loss-of-contact diagnostics for ALF-built adslabs."""

from dataclasses import dataclass

import numpy as np
from ase.data import covalent_radii
from ase.geometry import find_mic
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class AnomalyCheckSettings:
    connectivity_scale: float
    desorption_distance_A: float


def anomaly_check_settings(sampler_config):
    raw = sampler_config.get('anomaly_check', {})
    if not isinstance(raw, dict) or type(raw.get('enabled', False)) is not bool:
        raise ValueError('anomaly_check must be an object with a Boolean enabled setting')
    unknown = sorted(set(raw) - {'enabled', 'connectivity_scale', 'desorption_distance_A'})
    if unknown:
        raise ValueError(f'Unknown anomaly_check settings: {", ".join(unknown)}')
    if not raw.get('enabled', False):
        return None
    limits = []
    for key, default in [('connectivity_scale', 1.5), ('desorption_distance_A', 4.0)]:
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'anomaly_check.{key} must be finite and positive')
        limits.append(float(value))
    return AnomalyCheckSettings(*limits)


def prepare_surface_anomaly_check(molecule, settings):
    """Resolve the original atom indices once, before modifying the trajectory or building its engine."""
    metadata = molecule.get_metadata().get('surface_builder')
    if settings is None or metadata is None:
        return None
    atoms = molecule.get_atoms()
    tags = atoms.get_tags()
    if not isinstance(metadata, dict) or not atoms.has('tags') or not np.isin(tags, (0, 1, 2)).all():
        raise ValueError('anomaly_check requires ALF slab tags 0, 1 and 2 on their respective regions')

    adsorbate_indices = np.flatnonzero(tags == 2).tolist()
    if 'adsorbates' in metadata:
        placements = metadata['adsorbates']
        if not isinstance(placements, list):
            raise ValueError('surface_builder.adsorbates must list the original ligand atom groups')
        groups = []
        for placement in placements:
            indices = placement.get('atom_indices') if isinstance(placement, dict) else None
            if not isinstance(indices, list) or not indices or any(type(index) is not int for index in indices):
                raise ValueError('Each adsorbate needs a nonempty list of integer atom_indices')
            groups.append(indices)
    elif 'adsorbate_smiles' in metadata or 'adsorbate_file' in metadata:
        groups = [adsorbate_indices] if adsorbate_indices else None
    else:
        groups = []
    if groups is None or sorted(index for group in groups for index in group) != adsorbate_indices:
        raise ValueError('Adsorbate atom groups must cover every atom tagged 2 exactly once')
    if not groups:
        return None
    surface_indices = np.flatnonzero(tags == 1)
    if not len(surface_indices):
        raise ValueError('Adsorbate slab has no exposed substrate atoms tagged 1')
    return SurfaceAnomalyCheck(atoms.numbers, groups, surface_indices, settings)


class SurfaceAnomalyCheck:
    """One trajectory's ligand membership, bond cutoffs and first loss-of-contact events."""

    def __init__(self, numbers, groups, surface_indices, settings):
        self._surface_indices = surface_indices
        self._desorption_distance = settings.desorption_distance_A
        self._initial_contact = None
        self._desorptions = [None] * len(groups)
        self._groups = []
        for group in groups:
            indices = np.asarray(group)
            left, right = np.triu_indices(len(indices), k=1)
            radii = covalent_radii[numbers[indices]]
            thresholds = settings.connectivity_scale * (radii[left] + radii[right])
            self._groups.append((indices, left, right, thresholds))

    def inspect(self, atoms, step, time_ps):
        positions = atoms.positions
        if not np.isfinite(positions).all() or not np.isfinite(atoms.cell.array).all():
            return {'surface_reject_reason': 'surface_nonfinite'}

        surface = positions[self._surface_indices]
        initial = self._initial_contact is None
        if initial:
            self._initial_contact = []
        reports = []
        reason = None
        for ligand, (indices, left, right, thresholds) in enumerate(self._groups):
            ligand_positions = positions[indices]
            offsets = ligand_positions[:, None, :] - surface
            distance = float(find_mic(offsets.reshape(-1, 3), atoms.cell, pbc=atoms.pbc)[1].min())
            in_contact = distance <= self._desorption_distance
            if initial:
                self._initial_contact.append(in_contact)

            report = {'ligand_index': ligand, 'min_surface_distance_A': distance,
                      'first_desorption': self._desorptions[ligand]}
            if len(indices) > 1:
                distances = find_mic(ligand_positions[left] - ligand_positions[right], atoms.cell, pbc=atoms.pbc)[1]
                graph = np.zeros((len(indices), len(indices)), dtype=bool)
                graph[left, right] = distances <= thresholds
                count, labels = connected_components(graph, directed=False)
                if count > 1:
                    report['fragments'] = [indices[labels == part].tolist() for part in range(count)]
                    reason = 'adsorbate_fragmentation'

            if ('fragments' not in report and self._initial_contact[ligand] and not in_contact
                    and self._desorptions[ligand] is None):
                event = {'step': int(step), 'time_ps': float(time_ps), 'min_surface_distance_A': distance}
                self._desorptions[ligand] = report['first_desorption'] = event
            reports.append(report)
        return {'surface_reject_reason': reason, 'surface_adsorbates': reports}
