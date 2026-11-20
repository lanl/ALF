"""ML interfaces: training tasks and the models samplers load, by package (``hippynn``, ``fine_tuning``).

Modules that used to sit here still import, from the table below, with a ``FutureWarning`` naming their new
location, so configurations, scripts and checkpoints that name them keep working.
"""
from alframework._moved_modules import redirect_moved_modules

MOVED_MODULES = {
    'hippynn_interface': 'alframework.ml_interfaces.hippynn.interface',
    'hippynn_lightning': 'alframework.ml_interfaces.hippynn.lightning',
    'hippynn_final': 'alframework.ml_interfaces.hippynn.final_evaluation',
    '_hippynn_compat': 'alframework.ml_interfaces.hippynn.compat',
}

redirect_moved_modules(__name__, MOVED_MODULES)
