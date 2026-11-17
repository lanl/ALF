"""Checks that a QM label can be stored and trained on, used when labels are held and again when they are stored."""
import re

import numpy as np
from ase import Atoms

# Energies are scalars and forces (atoms, 3), including the per-state names of excited-state labeling.
ENERGY_NAME = re.compile(r'(?:energy|free_energy|E\d+|dE\d+)')
FORCE_NAME = re.compile(r'(?:forces|F\d+)')
SCOPES = ('system', 'atomic', 'pair_atomic')


def validate_geometry(atoms):
    """Raise ``ValueError`` unless ``atoms`` has atoms, finite positions and cell, and real elements."""
    if not isinstance(atoms, Atoms) or not len(atoms):
        raise ValueError('Label geometry must contain atoms.')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError('Nonfinite coordinates or cell.')
    if np.any(atoms.numbers <= 0):
        raise ValueError('Label geometry contains invalid atomic numbers.')
    if atoms.pbc.all() and abs(atoms.cell.volume) == 0:
        raise ValueError('Periodic label geometry needs a nonsingular cell.')


def validate_values(results):
    """Raise ``ValueError`` if any numeric result holds NaN or infinity."""
    for key, value in results.items():
        array = np.asarray(value)
        if array.dtype.kind in 'biufc' and not np.isfinite(array).all():
            raise ValueError(f'Nonfinite label property {key!r}.')


def validate_label(molecule, properties):
    """Raise ``ValueError`` unless a converged label can be stored: valid geometry, and every requested property
    present, numeric, of the right shape and finite after unit conversion. Values and units are not changed."""
    atoms, results = molecule.get_atoms(), molecule.get_results()
    validate_geometry(atoms)
    if atoms.pbc.any() and not atoms.pbc.all():
        raise ValueError('Partly periodic labels cannot be stored: a shard holds fully periodic or nonperiodic frames.')
    validate_values(results)
    for key, (_, scope, multiplier, *_) in properties.items():
        scope = scope.lower()
        if scope not in SCOPES:
            raise ValueError(f'Unknown label property scope {scope!r}.')
        if key not in results:
            raise ValueError(f'Missing required label property {key!r}.')
        array = np.asarray(results[key])
        if array.dtype.kind not in 'biufc':
            raise ValueError(f'Label property {key!r} must be numeric.')
        if ENERGY_NAME.fullmatch(key) and array.size != 1:
            raise ValueError(f'Energy property {key!r} must be scalar.')
        if FORCE_NAME.fullmatch(key) and array.shape != (len(atoms), 3):
            raise ValueError(f'Force property {key!r} must have shape (atoms, 3).')
        if scope == 'atomic' and (array.ndim < 1 or array.shape[0] != len(atoms)):
            raise ValueError(f'Label property {key!r} has an invalid atom axis.')
        if scope == 'pair_atomic' and (array.ndim < 2 or array.shape[1] != len(atoms)):
            raise ValueError(f'Label property {key!r} has an invalid pair/atom axis.')
        with np.errstate(over='ignore', invalid='ignore'):
            finite = np.isfinite(array * float(multiplier)).all()
        if not finite:
            raise ValueError(f'Label property {key!r} is nonfinite after unit conversion.')


def screen_labels(molecules, properties):
    """Mark converged labels that can't be stored unconverged, with the reason in ``labeling_error``.

    This runs before converged labels are counted, so an invalid label never satisfies a label threshold, and its
    batch siblings are kept.
    """
    for molecule in molecules:
        if molecule.check_convergence():
            try:
                validate_label(molecule, properties)
            except (TypeError, ValueError, OverflowError) as error:
                reject_label(molecule, str(error))
    return molecules


def reject_label(molecule, reason):
    """Mark one label unconverged, recording why."""
    molecule.set_converged_flag(False)
    molecule.update_metadata({'labeling_error': f'Invalid label: {reason}'})
