"""Declare an ALF task by its role: the Parsl executor it runs on and the decorators every task wears.

Each role (:data:`ROLES`) has three names, kept apart so that one can change without the others: the role a task
declares, the stage its calls are recorded under in the task timeline, and the label of the Parsl executor it runs on
(``alf_sampler_executor`` for builders and samplers, ``alf_QM_executor`` for labelers, ``alf_ML_executor`` for
trainers). ``@alf_task(role)`` makes a function a task of that role::

    from alframework.tools.tasks import alf_task

    @alf_task('QM', validate_config=validate_my_labeler)
    def my_labeler_task(molecule_object, QM_config, properties_list):
        ...

is the same as::

    @python_app(executors=['alf_QM_executor'])
    @timed('QM')
    @releases_memory_on_failure
    def my_labeler_task(molecule_object, QM_config, properties_list):
        ...
    my_labeler_task.func.validate_config = validate_my_labeler

so each call is recorded in the task timeline (:func:`alframework.tools.task_timeline.timed`), and a call that fails
frees what it holds before its worker takes the next task
(:func:`alframework.tools.task_failure_cleanup.releases_memory_on_failure`). The result is an ordinary Parsl app:
``my_labeler_task.func`` is the decorated function, and a configured standby executor
(``alf_<role>_standby_executor``) joins the task's executor at startup.

``@alf_join_task(role)`` does the same for a training task whose body runs on the coordinator and submits one app
per ensemble member to the role's executor (a Parsl ``join_app``, not timed itself).

A task of your own, outside ALF, can also be declared with ``@python_app`` directly; ALF needs only a Parsl app.

Renaming a role or its executor keeps older names working. :data:`ROLE_ALIASES` maps a former role name to the
current one. A role's ``accepts`` lists former executor labels, which ALF relabels to the current one when it loads
a Parsl configuration (:func:`relabel_executors`), so existing configurations still run. Keep a role's stage name,
which run directories already record, unless their readers accept both.
"""
from typing import NamedTuple

from alframework.tools.capabilities import declares as _declares
from alframework.tools.task_failure_cleanup import releases_memory_on_failure
from alframework.tools.task_timeline import timed


class Role(NamedTuple):
    """A task role's stage name, its executor's label, and the former labels of that executor."""
    stage: str
    executor: str
    accepts: tuple = ()


#: ALF's task roles.
ROLES = {'builder': Role('builder', 'alf_sampler_executor'), 'sampler': Role('sampler', 'alf_sampler_executor'),
         'QM': Role('QM', 'alf_QM_executor'), 'ML': Role('ML', 'alf_ML_executor')}
#: Former role names, each mapped to the current one.
ROLE_ALIASES = {}


def role(name):
    """The :class:`Role` called ``name``, or a former name for it."""
    try:
        return ROLES[ROLE_ALIASES.get(name, name)]
    except KeyError:
        raise ValueError(f'An ALF task role is one of {sorted(ROLES)}, not {name!r}.') from None


def executor_label(name):
    """The label of the Parsl executor role ``name``'s tasks run on."""
    return role(name).executor


def standby_label(label):
    """The label of the standby executor that joins executor ``label`` when it is configured."""
    return label.replace('_executor', '_standby_executor')


def executor_stages():
    """Each executor label, current or former, mapped to the stages of the roles that run on it."""
    stages = {}
    for current in ROLES.values():
        for label in (current.executor, *current.accepts):
            names = stages.setdefault(label, [])
            if current.stage not in names:
                names.append(current.stage)
    return {label: tuple(names) for label, names in stages.items()}


def relabel_executors(executors):
    """Give each executor configured under a former label (a role's ``accepts``, or its standby form) the current one.

    Changes the executors in place, before Parsl loads them, and returns ``(former, current)`` for each. Raises
    ``ValueError`` for a configuration with an executor under both labels.
    """
    renames = {}
    for current in ROLES.values():
        for former in current.accepts:
            renames[former] = current.executor
            renames[standby_label(former)] = standby_label(current.executor)
    labels = {executor.label for executor in executors}
    changed = []
    for executor in executors:
        new = renames.get(executor.label)
        if new is None:
            continue
        if new in labels:
            raise ValueError(f'The Parsl configuration has executors labeled both {executor.label!r} and {new!r}, '
                             f'the current name for it. Keep one.')
        changed.append((executor.label, new))
        executor.label = new
    return changed


def configured_executor(name):
    """The running Parsl configuration's executor for role ``name``, or ``None`` when it has none or Parsl isn't
    loaded."""
    try:
        import parsl
        return parsl.dfk().executors.get(executor_label(name))
    except Exception:
        return None


def role_app(function, executors):
    """A Parsl app running ``function`` on ``executors``: for an app the coordinator makes at startup around a
    configured task, which runs where that task does."""
    import parsl
    return parsl.python_app(executors=list(executors))(function)


def _annotate(function, declares, validate_config):
    if declares:
        function = _declares(*declares)(function)
    if validate_config is not None:
        function.validate_config = validate_config
    return function


def alf_task(name, *, declares=(), validate_config=None):
    """Make the decorated function a Parsl app of role ``name`` (see the module description).

    ``declares`` names the settings the task reads (:func:`alframework.tools.capabilities.declares`), and
    ``validate_config(config, master_config)`` checks its configuration at startup.
    """
    current = role(name)

    def decorate(function):
        import parsl  # Looked up when decorating, so that the docs build can replace python_app.
        task = _annotate(timed(current.stage)(releases_memory_on_failure(function)), declares, validate_config)
        return parsl.python_app(executors=[current.executor])(task)
    return decorate


def alf_join_task(name, *, declares=(), validate_config=None, member_layout=None):
    """Make the decorated function a Parsl join app that submits its members to role ``name``'s executor.

    ``member_layout(config)`` returns the device, GPUs per member and nodes per member, which ALF checks against the
    executor at startup.
    """
    label = executor_label(name)

    def decorate(function):
        import parsl
        task = _annotate(function, declares, validate_config)
        task.member_executors = (label,)
        if member_layout is not None:
            task.member_layout = member_layout
        return parsl.join_app(task)
    return decorate
