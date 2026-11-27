"""Workflow tools shared by ALF's builders, samplers, QM and ML interfaces.

``molecules_class``, renamed ``molecules``, still imports under its old name, with a ``FutureWarning`` naming the
new one, so pickled molecules keep loading. ``tools.tools`` keeps the names it used to define, for existing
scripts; new code imports them from the modules that define them.
"""
from alframework._moved_modules import redirect_moved_modules

MOVED_MODULES = {
    'molecules_class': 'alframework.tools.molecules',
}

redirect_moved_modules(__name__, MOVED_MODULES)
