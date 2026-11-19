"""Keep old module paths importable after a module moves, with a ``FutureWarning`` naming the new path.

A package lists its moved modules and calls :func:`redirect_moved_modules`. Configurations, scripts and
pickled objects that name an old path then get the module at its new path; no files are kept at the old
paths, and a real module of the same name is always found first.
"""
import importlib
import importlib.abc
import importlib.util
import sys
import types
import warnings


def _warn_moved(old, new):
    warnings.warn(f'{old} has moved to {new}; update configurations and imports to the new path.',
                  FutureWarning, stacklevel=3)


def _split_module(name, names):
    """A module whose names now live in different modules, each imported when first used."""
    module = types.ModuleType(name, f'Moved: its names now live in {sorted(set(names.values()))}.')
    module.__all__ = sorted(names)

    def __getattr__(attribute):
        if attribute not in names:
            raise AttributeError(f'module {name!r} has no attribute {attribute!r}')
        new = names[attribute]
        _warn_moved(f'{name}.{attribute}', f'{new}.{attribute}')
        return getattr(importlib.import_module(new), attribute)

    module.__getattr__ = __getattr__
    return module


class _MovedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves one package's old module paths."""

    def __init__(self, package, moved):
        self.package = package
        self.moved = moved

    def find_spec(self, fullname, path=None, target=None):
        package, _, name = fullname.rpartition('.')
        if package == self.package and name in self.moved:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        target = self.moved[spec.name.rpartition('.')[2]]
        if isinstance(target, dict):
            return _split_module(spec.name, target)
        # A placeholder; exec_module swaps in the real module. Returning the real module here would let the
        # import system overwrite its __spec__ with the old name's.
        return None

    def exec_module(self, module):
        target = self.moved[module.__name__.rpartition('.')[2]]
        if isinstance(target, dict):
            return
        _warn_moved(module.__name__, target)
        # The module at its new path, under both names, keeping its own spec.
        sys.modules[module.__name__] = importlib.import_module(target)


def redirect_moved_modules(package, moved):
    """Resolve ``package.<old>`` for each ``old`` in ``moved``.

    ``moved`` maps an old module name to the full path of its new module, or, for a module that was split, to
    a dictionary of each of its names and the full path of the module that now holds it.
    """
    if not any(isinstance(finder, _MovedModules) and finder.package == package for finder in sys.meta_path):
        sys.meta_path.append(_MovedModules(package, moved))
