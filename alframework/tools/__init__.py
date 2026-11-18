
"""Shared workflow tools and lazy aliases for renamed modules."""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings


MOVED_MODULES = {
    'excited_state_tools': 'alframework.tools.excited_state_schema',
    'failed_tasks': 'alframework.tools.task_failure_cleanup',
    'labeling': 'alframework.tools.labeling_coordinator',
    'molecules_class': 'alframework.tools.molecules',
    'node_setup': 'alframework.tools.worker_setup',
}


class _MovedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve previous module paths on demand, after checking for real modules."""

    def find_spec(self, fullname, path=None, target=None):
        package, _, name = fullname.rpartition('.')
        if package == __name__ and name in MOVED_MODULES:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        old = module.__name__
        new = MOVED_MODULES[old.rpartition('.')[2]]
        warnings.warn(f'{old} has moved to {new}; update imports to the new path.',
                      FutureWarning, stacklevel=2)
        # Keep the canonical module's spec and share its globals under both names.
        sys.modules[old] = importlib.import_module(new)


if not any(isinstance(finder, _MovedModules) for finder in sys.meta_path):
    sys.meta_path.append(_MovedModules())
