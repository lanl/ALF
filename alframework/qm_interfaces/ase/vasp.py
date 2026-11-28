"""Labeling with ASE's VASP calculator."""
import re

import numpy as np

from alframework.tools.geometry import periodic_gaps
from alframework.tools.label_sources import LABEL_SOURCE
from alframework.tools.molecules import MoleculesObject
from alframework.tools.tasks import alf_task
from .generic import ase_label_source


_CELL_SPACING = 'kpoint_spacing_A_inv'
_SLAB_SPACING = 'slab_kspacing_A_inv'
_CELL_AXES = ('a', 'b', 'c')


def _kpoint_spacing(config):
    setting = _CELL_SPACING
    spacing = config.get(setting)
    if spacing is None:
        setting = _SLAB_SPACING
        spacing = config.get(setting)
    elif config.get(_SLAB_SPACING) is not None:
        raise ValueError(f'{_CELL_SPACING} cannot be combined with {_SLAB_SPACING}.')
    if spacing is None:
        return None, None
    if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f'{setting} must be a finite positive number.')
    options = config['input']
    if any(options.get(key) is not None for key in ('kpts', 'kspacing')):
        raise ValueError(f'{setting} cannot be combined with input.kpts or input.kspacing.')
    if options.get('gamma') is False:
        raise ValueError(f'{setting} requires a Gamma-centered mesh.')
    return setting, spacing


def validate_vasp_config(QM_config, master_config=None):
    """Check optional per-structure VASP settings at startup."""
    _kpoint_spacing(QM_config)
    _slab_dipole_enabled(QM_config)


def _slab_dipole_enabled(config):
    enabled = config.get('slab_dipole_correction', False)
    if type(enabled) is not bool:
        raise ValueError('slab_dipole_correction must be Boolean.')
    if enabled:
        options = config['input']
        if any(options.get(key) is not None for key in ('ldipol', 'idipol', 'dipol')):
            raise ValueError('slab_dipole_correction sets LDIPOL, IDIPOL and DIPOL automatically.')
        if options.get('nelect') is not None or any(
                options.get(key) not in (None, 0) for key in ('charge', 'net_charge')):
            raise ValueError('slab_dipole_correction requires neutral slabs and cannot override electron counts.')
    return enabled


def _slab_dipole_center(atoms, metadata):
    cell = atoms.cell
    if (metadata.get('vacuum_axes') != ['c'] or not atoms.pbc.all()
            or not np.isfinite(cell.array).all() or cell.volume == 0
            or not np.allclose(cell.angles()[:2], 90, rtol=0, atol=1e-5)):
        raise ValueError('slab_dipole_correction requires a periodic slab with its vacuum along perpendicular c.')
    heights = atoms.get_scaled_positions(wrap=True)[:, 2]
    ordered, gaps = periodic_gaps(heights, 1)
    gap = np.argmax(gaps)
    # Measure positions from the middle of the vacuum so a wrapped slab stays contiguous.
    origin = ordered[gap] + gaps[gap] / 2
    center = origin + np.average((heights - origin) % 1, weights=atoms.get_masses())
    return [0.5, 0.5, float(center % 1)]


def _spacing_kpoints(atoms, setting, spacing, vacuum_axes):
    cell = atoms.cell
    if not atoms.pbc.all() or not np.isfinite(cell.array).all() or cell.volume == 0:
        raise ValueError(f'{setting} requires a full periodic cell.')
    if setting == _SLAB_SPACING and not np.allclose(cell.angles()[:2], 90, rtol=0, atol=1e-5):
        raise ValueError(f'{setting} requires c to be perpendicular to a and b.')
    if (not isinstance(vacuum_axes, (list, tuple))
            or any(not isinstance(axis, str) or axis not in _CELL_AXES for axis in vacuum_axes)
            or len(set(vacuum_axes)) != len(vacuum_axes)):
        raise ValueError("vacuum_axes must be a list of distinct lattice axes chosen from 'a', 'b', and 'c'.")
    reciprocal_lengths = 2 * np.pi * np.linalg.norm(cell.reciprocal(), axis=1)
    return [1 if axis in vacuum_axes else max(1, int(np.ceil(length / spacing)))
            for axis, length in zip(_CELL_AXES, reciprocal_lengths)]


def vasp_input_for_structure(atoms, qm_config, metadata):
    """Apply the labeler's per-structure reciprocal mesh and slab dipole settings."""
    inputs = qm_config['input'].copy()
    setting, spacing = _kpoint_spacing(qm_config)
    if spacing is not None:
        if 'kpoints' in metadata:
            raise ValueError(f'{setting} cannot be combined with molecule metadata kpoints.')
        vacuum_axes = ('c',) if setting == _SLAB_SPACING else metadata.get('vacuum_axes', ())
        inputs.update(kpts=_spacing_kpoints(atoms, setting, spacing, vacuum_axes), gamma=True)
    elif 'kpoints' in metadata:
        inputs['kpts'] = metadata['kpoints']
    if _slab_dipole_enabled(qm_config) and 'surface_builder' in metadata:
        inputs.update(ldipol=True, idipol=3, dipol=_slab_dipole_center(atoms, metadata))
    return inputs


@alf_task('QM', validate_config=validate_vasp_config)
def VASP_ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """ASE VASP calculator based on the parameters in qm_config.json.

    Args:
        molecule_object (MoleculesObject): An object from the class MoleculesObject.
        QM_config (dict): The QM configuration: ``QM_run_command``, the VASP calculator's ``input`` settings,
                          and optional ``kpoint_spacing_A_inv`` and ``slab_dipole_correction``.
        QM_scratch_dir (str): Directory under which each molecule gets its own calculation directory.
        properties_list (dict): Dict whose keys are the properties that we would like ASE to extract from the OUTCAR
                                file and that will be stored in the database.

    Returns:
        (MoleculesObject): A MoleculesObject containing the results of the QM calculations.

    """
    from ase.calculators.vasp import Vasp as calc_class

    assert isinstance(molecule_object, MoleculesObject), 'molecule_object must be an instance of MoleculesObject'

    command = QM_config['QM_run_command']
    directory = QM_scratch_dir + '/' + molecule_object.get_moleculeid()
    properties = list(properties_list.keys())
    atoms = molecule_object.get_atoms()

    metadata = molecule_object.get_metadata()
    calc_input = vasp_input_for_structure(atoms, QM_config, metadata)
    if metadata.get('fixed_substrate_indices') and calc_input.get('nsw') in (None, 0):
        # Fixed layers constrain sampling. A static label includes the forces on those sites.
        atoms = atoms.copy()
        atoms.set_constraint()
    calc = calc_class(directory=directory, command=command, **calc_input)

    # Run the calculation
    calc.calculate(atoms=atoms, properties=properties)

    molecule_object.store_results(calc.results)

    convergedre = re.compile('aborting loop because EDIFF is reached')
    txt = open(directory + '/OUTCAR','r').read()
    if len(convergedre.findall(txt)) == 1:
        source = ase_label_source('VASP_ase_calculator_task', 'ase.calculators.vasp.Vasp', QM_config['input'])
        setting, spacing = _kpoint_spacing(QM_config)
        if spacing is not None:
            source[setting] = spacing
        if _slab_dipole_enabled(QM_config):
            source['slab_dipole_correction'] = True
        molecule_object.update_metadata({LABEL_SOURCE: source})
        molecule_object.set_converged_flag(True)
    else:
        molecule_object.set_converged_flag(False)

    return molecule_object
