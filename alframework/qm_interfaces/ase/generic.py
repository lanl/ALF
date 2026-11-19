"""Label one structure with any ASE calculator, and the checks shared by the ASE tasks.

A QM configuration names the calculator class in ``ASE_calculator`` and gives its constructor options either in
``input`` or, as older configurations do, beside the ALF settings. ALF checks the configuration once at startup,
then gives every label a fresh calculator, a fresh scratch directory and a copy of the molecule's geometry.
"""
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import stat

import numpy as np
from ase.calculators.calculator import BaseCalculator, all_changes
from ase.data import chemical_symbols
from parsl import python_app

from alframework.tools.label_validation import validate_geometry, validate_label
from alframework.tools.molecules import MoleculesObject
from alframework.tools.task_timeline import timed
from alframework.tools.config import load_module_from_config

# QM_config keys that are ALF settings, never calculator options.
ALF_SETTINGS = {'ASE_calculator', 'QM_run_command', 'batching', 'input', 'initial_magnetic_moments',
                'remove_scratch_on_success'}
# Calculator options ALF sets itself for every label.
RESERVED_OPTIONS = ALF_SETTINGS | {'atoms', 'directory', 'command', 'profile'}
PROPERTY_SCOPES = ('system', 'atomic', 'pair_atomic')
# What a molecule's metadata can say about its electronic state.
STATE_KEYS = ('charge', 'mult')
ERROR_KEYS = ('qm_error', 'qm_error_type', 'qm_failed_stage')


def _integer(value, name, minimum=None):
    """Return ``value`` as an int, refusing booleans, floats and anything below ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer.')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be at least {minimum}.')
    return int(value)


def _state(atoms, charge, mult):
    """The charge and multiplicity for a molecular calculation: the atoms' own, else the configured ones."""
    if atoms.pbc.any():
        raise ValueError('The ORCA and Q-Chem tasks label nonperiodic molecules only.')
    return (_integer(atoms.info.get('charge', charge), 'charge'),
            _integer(atoms.info.get('mult', mult), 'mult', 1))


def _backend(calculator_class):
    """ALF's settings module for ASE's own ORCA or Q-Chem class, or None.

    Only those exact classes get ALF's settings; a subclass is someone's own calculator and is left alone.
    """
    if calculator_class.__module__ == 'ase.calculators.orca':
        from ase.calculators.orca import ORCA
        if calculator_class is ORCA:
            from . import orca
            return orca
    if calculator_class.__module__ == 'ase.calculators.qchem':
        from ase.calculators.qchem import QChem
        if calculator_class is QChem:
            from . import qchem
            return qchem
    return None


def _calculator_options(config):
    """The calculator's constructor options: ``input``, or every key that isn't an ALF setting."""
    flat = set(config) - ALF_SETTINGS
    if 'input' not in config:
        options = {key: config[key] for key in flat}
    elif not isinstance(config['input'], dict):
        raise TypeError('QM_config input must be a dictionary.')
    elif flat:
        raise ValueError(f'Do not mix input with flat calculator options: {sorted(flat)}.')
    else:
        options = config['input']
    reserved = RESERVED_OPTIONS.intersection(options)
    if reserved:
        raise ValueError(f'Reserved calculator options: {sorted(reserved)}.')
    label = options.get('label')
    if label is not None and (not isinstance(label, str) or not label or Path(label).name != label):
        raise ValueError('Calculator label must be a filename without a directory.')
    return options


def _run_command(config):
    """``QM_run_command``: a nonempty string, or None for calculators that run in this process."""
    command = config.get('QM_run_command')
    if command is not None and (not isinstance(command, str) or not command.strip()):
        raise ValueError('QM_run_command must be a nonempty string or null.')
    return command


def _magnetic_moments(config):
    """``initial_magnetic_moments``: a starting moment in Bohr magnetons for each listed element, or None.

    ASE calculators take initial moments per atom, so a fixed list only fits one structure size. ALF sets each
    atom's moment from its element instead, and elements left out start at zero.
    """
    moments = config.get('initial_magnetic_moments')
    if moments is None:
        return None
    if not isinstance(moments, dict) or not moments:
        raise ValueError('initial_magnetic_moments must map element symbols to moments.')
    for symbol, moment in moments.items():
        if symbol not in chemical_symbols[1:]:
            raise ValueError(f'initial_magnetic_moments has an unknown element: {symbol!r}.')
        if isinstance(moment, bool) or not isinstance(moment, (int, float)) or not np.isfinite(moment):
            raise ValueError(f'The initial magnetic moment for {symbol} must be a finite number.')
    return {symbol: float(moment) for symbol, moment in moments.items()}


def _remove_scratch(config):
    """``remove_scratch_on_success``: delete a molecule's scratch directory once its label is good."""
    remove = config.get('remove_scratch_on_success', False)
    if not isinstance(remove, bool):
        raise ValueError('remove_scratch_on_success must be true or false.')
    return remove


def _check_properties(properties, calculator_class):
    """Each property needs ``[dataset, scope, multiplier]``, and the calculator must say it computes it."""
    if not isinstance(properties, dict) or not properties:
        raise ValueError('properties_list must be a nonempty dictionary.')
    for key, schema in properties.items():
        if not isinstance(key, str) or not key or not isinstance(schema, (list, tuple)) or len(schema) != 3:
            raise ValueError('Each property needs a name and [dataset, scope, multiplier].')
        dataset, scope, multiplier = schema
        if (not isinstance(dataset, str) or not dataset or not isinstance(scope, str)
                or scope.lower() not in PROPERTY_SCOPES or not np.isfinite(float(multiplier))):
            raise ValueError(f'Invalid property schema for {key!r}.')
    advertised = getattr(calculator_class, 'implemented_properties', None)
    if isinstance(advertised, (list, tuple, set, frozenset)):
        missing = set(properties) - set(advertised)
        if missing:
            raise ValueError(f'{calculator_class.__name__} does not implement {sorted(missing)}.')


def _configuration(QM_config, properties=None):
    """Check a QM configuration and return ``(calculator class, options, command, moments)``; nothing runs."""
    if not isinstance(QM_config, dict):
        raise TypeError('QM_config must be a dictionary.')
    config = deepcopy(QM_config)
    if not isinstance(config.get('ASE_calculator'), str) or not config['ASE_calculator'].strip():
        raise ValueError('ASE_calculator must name a calculator class.')
    options = _calculator_options(config)
    command = _run_command(config)
    moments = _magnetic_moments(config)
    _remove_scratch(config)
    if moments is not None and 'magmom' in options:
        raise ValueError('Give initial magnetic moments per element in initial_magnetic_moments, '
                         'or per atom in input magmom, not both.')
    calculator_class = load_module_from_config(config, 'ASE_calculator')
    if not isinstance(calculator_class, type) or not issubclass(calculator_class, BaseCalculator):
        raise TypeError('ASE_calculator must be an ASE BaseCalculator subclass.')
    if properties is not None:
        _check_properties(properties, calculator_class)
    # ORCA and Q-Chem get ALF's checks; any other class may provide its own.
    backend = _backend(calculator_class)
    if backend and moments is not None:
        raise ValueError('ORCA and Q-Chem take the spin state from the multiplicity, not initial_magnetic_moments.')
    validate = backend.validate_config if backend else getattr(calculator_class, 'validate_alf_config', None)
    if validate is not None:
        validate(options, command, properties)
    return calculator_class, options, command, moments


def validate_ase_config(QM_config, master_config=None):
    """Check generic ASE configuration at coordinator startup, without executing QM."""
    _configuration(QM_config, (master_config or {}).get('properties_list'))


def _calculator(calculator_class, atoms, directory, options, command):
    """A fresh calculator writing into ``directory``."""
    backend = _backend(calculator_class)
    if backend:
        return backend.create_calculator(atoms, directory, options, command)
    kwargs = dict(options, directory=directory)
    # Newer profile-based calculators don't take atoms in their constructor.
    if 'atoms' in inspect.signature(calculator_class).parameters:
        kwargs['atoms'] = atoms
    if command is not None:
        try:
            from ase.calculators.genericfileio import GenericFileIOCalculator
        except ImportError:
            GenericFileIOCalculator = ()
        if issubclass(calculator_class, GenericFileIOCalculator):
            raise ValueError('This profile-based calculator needs an adapter for QM_run_command.')
        kwargs['command'] = command
    return calculator_class(**kwargs)


def fresh_scratch_directory(directory):
    """Create an empty scratch directory, moving aside one left by an earlier attempt at this structure.

    A restarted finite bootstrap rebuilds the IDs it didn't finish, and a crashed run may have
    started their calculations. Their files are kept, beside the new directory.
    """
    if os.path.isdir(directory):
        attempt = 0
        while os.path.exists(f'{directory}.previous-{attempt}'):
            attempt += 1
        os.rename(directory, f'{directory}.previous-{attempt}')
        print(f'Warning: {directory} already existed, from an earlier attempt at this structure; '
              f'moved it to {directory}.previous-{attempt}.', flush=True)
    os.makedirs(directory)


def molecule_directory(molecule_object, scratch):
    """Resolve a molecule's scratch directory without allowing path traversal."""
    moleculeid = molecule_object.get_moleculeid()
    if not moleculeid or moleculeid in ('.', '..') or Path(moleculeid).name != moleculeid:
        raise ValueError('Molecule ID must be a filename without a directory.')
    return str(Path(scratch) / moleculeid)


def _working_atoms(molecule_object, moments=None):
    """A copy of the geometry to calculate on, carrying the molecule's charge, multiplicity and initial moments."""
    atoms = molecule_object.get_atoms().copy()
    atoms.info = deepcopy(atoms.info)
    for key in STATE_KEYS:
        if key in molecule_object.metadata:
            atoms.info[key] = deepcopy(molecule_object.metadata[key])
    if moments is not None:
        atoms.set_initial_magnetic_moments([moments.get(symbol, 0.0) for symbol in atoms.get_chemical_symbols()])
    return atoms


def _place_input_files(input_files, directory):
    """Put earlier stages' files in this stage's directory: moved on their last use, copied before that."""
    for source, destination, move in input_files:
        info = source.lstat()
        if source.parent.is_symlink() or not stat.S_ISREG(info.st_mode) or not info.st_size:
            raise ValueError(f'Expected a nonempty regular stage input file at {source}.')
        target = Path(directory) / destination
        if move:
            # Stages share the molecule's directory, so this is a rename on one file system, not a copy.
            os.replace(source, target)
        else:
            shutil.copyfile(source, target)


def remove_scratch_directory(directory):
    """Delete a finished label's scratch directory; a failure to delete leaves the label as it is."""
    try:
        shutil.rmtree(directory)
    except OSError as error:
        print(f'Warning: could not remove scratch directory {directory}: {error}', flush=True)


def _check_converged(calc):
    """A calculator that reports convergence must report exactly True."""
    if hasattr(calc, 'converged'):
        flag = calc.converged
        if not isinstance(flag, (bool, np.bool_)) or not bool(flag):
            raise ValueError(f'Calculator did not report convergence: {flag!r}.')


def _check_same_geometry(atoms, original):
    """The label belongs to the sampled geometry, so the calculator must not have moved it."""
    if (not np.array_equal(atoms.numbers, original.numbers)
            or not np.array_equal(atoms.pbc, original.pbc)
            or not np.array_equal(atoms.positions, original.positions)
            or not np.array_equal(atoms.cell.array, original.cell.array)):
        raise ValueError('Calculator changed the labeling geometry.')


def _checked_results(calc, atoms, molecule_object, properties_list):
    """This calculation's results, as a label ALF can store; earlier results never fill a gap."""
    candidate = MoleculesObject(atoms, molecule_object.get_moleculeid())
    candidate.store_results(deepcopy(dict(calc.results)))
    for key in properties_list:
        if key in calc.results and np.asarray(calc.results[key]).dtype.kind not in 'iuf':
            raise ValueError(f'ASE property {key!r} must be real numeric data.')
    if 'dipole' in properties_list and 'dipole' in calc.results and np.shape(calc.results['dipole']) != (3,):
        raise ValueError('dipole must have shape (3,).')
    validate_label(candidate, properties_list)
    return candidate.get_results()


def label_one(molecule_object, QM_config, QM_scratch_dir, properties_list, *, directory=None, input_files=()):
    """Run one fresh ASE calculation in this process; the workflow task calls it once per stage.

    A failure at any stage leaves the molecule unconverged, with the error and the stage it happened at in
    ``qm_error``, ``qm_error_type`` and ``qm_failed_stage``.

    The workflow task supplies ``input_files`` as ``(source path, destination filename, move)``.
    """
    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError('molecule_object must be a MoleculesObject.')
    molecule_object.set_converged_flag(False)
    stage = 'configuration'
    try:
        calc_class, options, command, moments = _configuration(QM_config, properties_list)
        atoms = _working_atoms(molecule_object, moments)
        validate_geometry(atoms)
        default_directory = molecule_directory(molecule_object, QM_scratch_dir)  # Also checks the molecule ID.
        directory = default_directory if directory is None else directory
        stage = 'scratch'
        fresh_scratch_directory(directory)
        stage = 'input_files'
        _place_input_files(input_files, directory)
        stage = 'construction'
        calc = _calculator(calc_class, atoms, directory, options, command)
        missing = set(properties_list) - set(calc.implemented_properties)
        if missing:
            raise ValueError(f'Calculator does not implement {sorted(missing)}.')
        stage = 'calculation'
        calc.calculate(atoms=atoms, properties=list(properties_list), system_changes=all_changes)
        backend = _backend(calc_class)
        if backend and hasattr(backend, 'check_completed'):
            backend.check_completed(calc)
        stage = 'validation'
        _check_converged(calc)
        _check_same_geometry(atoms, molecule_object.get_atoms())
        results = _checked_results(calc, atoms, molecule_object, properties_list)
    except Exception as error:
        molecule_object.update_metadata({'qm_error': f'{type(error).__name__}: {error}',
                                         'qm_error_type': type(error).__name__, 'qm_failed_stage': stage})
        return molecule_object
    molecule_object.store_results(results)
    # ORCA and Q-Chem record the charge and multiplicity they used, so the shard stores the same state.
    metadata = getattr(calc, 'metadata', None)
    if isinstance(metadata, dict):
        molecule_object.update_metadata(deepcopy(metadata))
    molecule_object.set_converged_flag(True)
    for key in ERROR_KEYS:
        molecule_object.metadata.pop(key, None)
    if _remove_scratch(QM_config):
        remove_scratch_directory(directory)
    return molecule_object


@python_app(executors=['alf_QM_executor'])
@timed('QM')
def ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """Label one molecule through upstream ASE and ALF's shared validation."""
    return label_one(molecule_object, QM_config, QM_scratch_dir, properties_list)


# The coordinator calls this at startup. Without Parsl's decorator (as in the docs build), the task is the function.
getattr(ase_calculator_task, 'func', ase_calculator_task).validate_config = validate_ase_config
