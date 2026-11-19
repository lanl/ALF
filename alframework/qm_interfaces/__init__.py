"""QM labeling interfaces.

ASE labeling lives in ``ase``, the older ORCA, Q-Chem and VASP interfaces in ``legacy``. Modules that used
to sit here still import, from the table below, with a ``FutureWarning`` naming their new location, so
existing configurations, scripts and pickled objects keep working. No files are kept for them.
"""
from alframework._moved_modules import redirect_moved_modules

# Old module name -> its new module, or, for a module that was split, each of its names -> its new module.
MOVED_MODULES = {
    'ase_calculator_interface': {
        'ase_calculator_task': 'alframework.qm_interfaces.ase.generic',
        'fresh_scratch_directory': 'alframework.qm_interfaces.ase.generic',
        'VASP_ase_calculator_task': 'alframework.qm_interfaces.ase.vasp',
    },
    'orca5_interface': 'alframework.qm_interfaces.legacy.orca5_interface',
    'qchem_DFT_interface': 'alframework.qm_interfaces.legacy.qchem_DFT_interface',
    'vaspase_interface': 'alframework.qm_interfaces.legacy.vaspase_interface',
}

redirect_moved_modules(__name__, MOVED_MODULES)
