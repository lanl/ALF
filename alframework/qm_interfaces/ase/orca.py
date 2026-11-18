"""ALF's checks and settings for ASE's own ORCA calculator.

ALF labels single points, so only settings for an energy and gradient calculation on the given geometry are
accepted. ASE writes the input and reads the output.
"""
from copy import deepcopy
from pathlib import Path
import re

from ase.calculators.calculator import CalculationFailed

from .generic import _integer, _state

OPTIONS = {'orcasimpleinput', 'orcablocks', 'charge', 'mult', 'label'}
# Simple-input keywords that move atoms or scan geometries, which a single-point label can't use.
GEOMETRY_KEYWORDS = re.compile(r'c?opt|optts|neb.*|md|scan|irc|freq|numfreq')
COORDINATE_BLOCK = re.compile(r'\bcoords\b', re.IGNORECASE)


def _orca_api():
    try:
        from ase.calculators.orca import ORCA, OrcaProfile
    except ImportError as error:
        raise ImportError('ASE ORCA needs an ASE version that provides OrcaProfile.') from error
    return ORCA, OrcaProfile


def validate_config(options, command, properties=None):
    """Check ORCA settings at coordinator startup, without running ORCA."""
    _orca_api()
    from ase.calculators.orca import OrcaTemplate
    if properties and set(properties) - set(OrcaTemplate().implemented_properties):
        raise ValueError('Requested properties are not supported by ASE ORCA.')
    unknown = set(options) - OPTIONS
    if unknown:
        raise ValueError(f'Unknown ORCA options: {sorted(unknown)}. Set cores with %pal in orcablocks, '
                         'and the environment in the worker or a wrapper script.')
    if not isinstance(command, str) or not command.strip():
        raise ValueError('ORCA needs QM_run_command naming the ORCA executable or a wrapper.')
    simple = options.get('orcasimpleinput')
    if not isinstance(simple, str) or not simple.strip():
        raise ValueError('orcasimpleinput must name the ORCA method and basis.')
    keywords = {keyword.lower() for keyword in simple.split()}
    if any(GEOMETRY_KEYWORDS.fullmatch(keyword) for keyword in keywords):
        raise ValueError('ORCA labels are single points (with EnGrad for forces): no optimization, MD or scans.')
    if properties and 'forces' in properties and 'engrad' not in keywords:
        raise ValueError('ORCA forces require EnGrad in orcasimpleinput.')
    blocks = options.get('orcablocks', '')
    if not isinstance(blocks, str):
        raise ValueError('orcablocks must be a string.')
    if COORDINATE_BLOCK.search(blocks):
        raise ValueError('ORCA coordinates come from the molecule, not a coords block in orcablocks.')
    _integer(options.get('charge', 0), 'charge')
    _integer(options.get('mult', 1), 'mult', 1)


def create_calculator(atoms, directory, options, command):
    """ASE's ORCA calculator for one label, with the molecule's charge and multiplicity."""
    ORCA, OrcaProfile = _orca_api()
    options = deepcopy(options)
    options.pop('label', None)  # ASE's ORCA uses fixed file names.
    charge, mult = _state(atoms, options.get('charge', 0), options.get('mult', 1))
    options.update(charge=charge, mult=mult)
    calc = ORCA(directory=directory, profile=OrcaProfile(command=command), **options)
    calc.metadata = {'qm_backend': 'orca', 'charge': charge, 'mult': mult}
    return calc


def check_completed(calc):
    """Require normal termination: ASE's gradient reader only warns about an incomplete run."""
    if 'ORCA TERMINATED NORMALLY' not in Path(calc.directory, 'orca.out').read_text():
        raise CalculationFailed('ORCA output lacks normal termination.')
