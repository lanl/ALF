"""What a task declares it supports, and the configuration check every task may provide.

Some settings are read by only some tasks: ``returned_metadata`` by the batched MLMD sampler, for example, or
``shared_training_data`` by HIPPYNN training. ALF checks such a setting against the task that would read it before
any task runs, so a setting a task would ignore stops the run at startup instead. A task declares what it reads with
``@declares``, under ``@python_app`` (or ``@join_app``)::

    @python_app(executors=['alf_sampler_executor'])
    @declares('returned_metadata', 'returned_momenta')
    def my_sampler_task(molecule_objects, sampler_config, ...):
        ...

``functools.wraps`` copies the declaration, so a task that wraps a built-in one with ``wraps`` supports what it does;
one that calls it otherwise declares the same features itself.

A task of any role can also check its configuration at startup: give its function a ``validate_config(config,
master_config)`` attribute, which raises ``ValueError`` for a configuration it can't run. ALF calls it with the
builder, sampler, QM or ML configuration, before the bootstrap.
"""

SUPPORTS = 'alf_supports'


def declares(*features):
    """Decorate a task function to declare the settings it reads (see the module description)."""
    for feature in features:
        if not isinstance(feature, str) or not feature:
            raise ValueError(f'A declared feature is a setting name, not {feature!r}.')

    def decorate(function):
        setattr(function, SUPPORTS, frozenset(getattr(function, SUPPORTS, ())) | frozenset(features))
        return function
    return decorate


def declared(task):
    """Every feature ``task`` (a Parsl app or a plain function) declares."""
    return frozenset(getattr(getattr(task, 'func', task), SUPPORTS, ()))


def supports(task, feature):
    return feature in declared(task)


def require(task, task_path, feature, setting=None):
    """Raise ``ValueError`` unless ``task`` declares ``feature``; ``setting`` names what asked for it."""
    if not supports(task, feature):
        raise ValueError(f'{setting or feature} is read only by tasks that declare {feature!r}, and {task_path} does '
                         f'not. A task of your own declares it with '
                         f'@alframework.tools.capabilities.declares({feature!r}) under @python_app.')


def validate_task_config(task, config, master_config):
    """Call the task's ``validate_config(config, master_config)``, if it has one."""
    validate = getattr(getattr(task, 'func', task), 'validate_config', None)
    if validate is not None:
        validate(config, master_config)
