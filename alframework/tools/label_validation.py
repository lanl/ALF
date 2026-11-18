"""Validate builder output and QM labels before accepting structures."""
import re

import numpy as np
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject

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


def system_checker(system, kill_on_fail=True, print_error=True):
    """Check the geometry and numeric properties of a system.

    Args:
        system (MoleculesObject/list/tuple): A MoleculesObject or three-element sequence whose first element
                       is a moleculeid string or a metadata dict containing a string 'moleculeid'.
                       The second element is an ASE Atoms object. The third element is a dict that stores the
                       desired properties from the QM calculation (e.g. forces and energies).
        kill_on_fail (bool): Raise ``RuntimeError`` when validation fails.
        print_error (bool): If True prints the error message if something goes wrong.

    Returns:
        (bool): True if 'system' meets all requirements and False otherwise.

    """
    try:
        if isinstance(system, MoleculesObject):
            moleculeid = system.get_moleculeid()
            atoms = system.get_atoms()
            results = system.get_results()
        else:
            assert isinstance(system, (list, tuple))
            assert len(system) == 3
            moleculeid = system[0]['moleculeid'] if isinstance(system[0], dict) else system[0]
            atoms = system[1]
            results = system[2]
        assert isinstance(moleculeid, str)
        assert isinstance(atoms, Atoms)
        assert isinstance(results, dict)

        validate_geometry(atoms)
        validate_values(results)

        return True

    except Exception as e:
        if print_error:
            print(e)
        if kill_on_fail:
            raise RuntimeError('Atomic system failed to meet requirements.')

        return False
