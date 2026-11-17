"""QM labeling interfaces.

ASE labeling lives in ``ase``, the older ORCA, Q-Chem and VASP interfaces in ``legacy``. Modules that used
to sit here still import, from the table below, with a ``FutureWarning`` naming their new location, so
existing configurations, scripts and pickled objects keep working. No files are kept for them.
"""
import importlib
import importlib.abc
import importlib.util
import sys
import types
import warnings

# Old module name -> its new module, or, for a module that was split, each of its names -> its new module.
MOVED_MODULES = {
    'ase_calculator_interface': {
        'ase_calculator_task': 'ase.generic',
        'fresh_scratch_directory': 'ase.generic',
        'VASP_ase_calculator_task': 'ase.vasp',
    },
    'orca5_interface': 'legacy.orca5_interface',
    'qchem_DFT_interface': 'legacy.qchem_DFT_interface',
    'vaspase_interface': 'legacy.vaspase_interface',
}


def _warn_moved(old, new):
    warnings.warn(f'{old} has moved to {new}; update configurations and imports to the new path.',
                  FutureWarning, stacklevel=3)


def _split_module(name, names):
    """A module whose names now live in different modules, each imported when first used."""
    module = types.ModuleType(name, f'Moved: see {__name__}.MOVED_MODULES.')
    module.__all__ = sorted(names)

    def __getattr__(attribute):
        if attribute not in names:
            raise AttributeError(f'module {name!r} has no attribute {attribute!r}')
        new = f'{__name__}.{names[attribute]}'
        _warn_moved(f'{name}.{attribute}', f'{new}.{attribute}')
        return getattr(importlib.import_module(new), attribute)

    module.__getattr__ = __getattr__
    return module


class _MovedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves the old module paths in ``MOVED_MODULES``; real modules are always found first."""

    def find_spec(self, fullname, path=None, target=None):
        package, _, name = fullname.rpartition('.')
        if package == __name__ and name in MOVED_MODULES:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        target = MOVED_MODULES[spec.name.rpartition('.')[2]]
        if isinstance(target, dict):
            return _split_module(spec.name, target)
        new = f'{__name__}.{target}'
        _warn_moved(spec.name, new)
        return importlib.import_module(new)

    def exec_module(self, module):
        pass


if not any(isinstance(finder, _MovedModules) for finder in sys.meta_path):
    sys.meta_path.append(_MovedModules())
