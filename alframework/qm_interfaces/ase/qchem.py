"""ALF's checks and settings for ASE's own Q-Chem calculator.

ALF labels single points, so only Q-Chem's single-point and force jobs are accepted. ASE writes the input and
reads the output.
"""
from copy import deepcopy

from ase.calculators.qchem import QChem

from .generic import _integer, _state

JOB_TYPES = ('SP', 'FORCE')


def validate_config(options, command, properties=None):
    """Check Q-Chem settings at coordinator startup, without running Q-Chem."""
    _integer(options.get('charge', 0), 'charge')
    _integer(options.get('multiplicity', 1), 'multiplicity', 1)
    for key in ('np', 'nt'):
        _integer(options.get(key, 1), key, 1)
    jobtype = options.get('jobtype')
    if jobtype is None:
        return  # ASE chooses FORCE when forces are requested.
    if str(jobtype).upper() not in JOB_TYPES:
        raise ValueError('ASE Q-Chem supports jobtype SP or FORCE only.')
    if properties and 'forces' in properties and str(jobtype).upper() != 'FORCE':
        raise ValueError('Q-Chem forces require jobtype FORCE (or leave jobtype out).')


def create_calculator(atoms, directory, options, command):
    """ASE's Q-Chem calculator for one label, with the molecule's charge and multiplicity."""
    options = deepcopy(options)
    charge, mult = _state(atoms, options.get('charge', 0), options.get('multiplicity', 1))
    options.update(charge=charge, multiplicity=mult)
    calc = QChem(atoms=atoms, directory=directory, command=command, **options)
    # ASE's constructor rebuilds the command to add its CPU flags; a configured command is used as written.
    if command is not None:
        calc.command = command
    calc.metadata = {'qm_backend': 'qchem', 'charge': charge, 'mult': mult}
    return calc
