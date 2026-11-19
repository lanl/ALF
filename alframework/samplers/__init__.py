"""Sampler tasks.

The dynamics engines live in ``engines``, and helpers shared with other parts of ALF in
``alframework.tools``. Modules that used to sit here still import, from the table below, with a
``FutureWarning`` naming their new location.
"""
from alframework._moved_modules import redirect_moved_modules

MOVED_MODULES = {
    'alchemi_engine': 'alframework.samplers.engines.alchemi',
    'packed_engine': 'alframework.samplers.engines.packed',
    'torchsim_engine': 'alframework.samplers.engines.torchsim',
    'geometry': 'alframework.tools.geometry',
}

redirect_moved_modules(__name__, MOVED_MODULES)
