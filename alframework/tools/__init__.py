"""Workflow tools shared by ALF's builders, samplers, QM and ML interfaces.

Modules renamed for what they do still import under their old names, from the table below, with a
``FutureWarning`` naming the new one. ``tools.tools`` keeps the names it used to define, for existing
scripts; new code imports them from the modules that define them.
"""
from alframework._moved_modules import redirect_moved_modules

MOVED_MODULES = {
    'excited_state_tools': 'alframework.tools.excited_state_schema',
    'failed_tasks': 'alframework.tools.task_failure_cleanup',
    'labeling': 'alframework.tools.labeling_coordinator',
    'molecules_class': 'alframework.tools.molecules',
    'node_setup': 'alframework.tools.worker_setup',
}

redirect_moved_modules(__name__, MOVED_MODULES)
