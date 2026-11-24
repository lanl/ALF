"""The builder contract, used the same way in preflight, the bootstrap, active learning and --test_builder.

A builder task takes either ``moleculeid``, one ID, and builds one structure, or ``moleculeids``, a list of IDs, and
builds up to that many (``maximum_builder_structures``). :func:`builder_width` says how many IDs a task gets, and
:func:`builder_outputs` checks what it returns.
"""
import inspect

from alframework.tools.molecules import MoleculesObject


def builds_batches(builder_task):
    """Whether ``builder_task`` takes a list of IDs (``moleculeids``) rather than one (``moleculeid``)."""
    return 'moleculeids' in inspect.signature(getattr(builder_task, 'func', builder_task)).parameters


def builder_width(builder_task, master_config):
    """How many molecule IDs each ``builder_task`` call gets: ``maximum_builder_structures`` for a builder that
    takes ``moleculeids``, and 1 for one that takes a single ``moleculeid``, whatever the setting says."""
    width = master_config.get('maximum_builder_structures', 1)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError(f'maximum_builder_structures must be a positive integer; received {width!r}.')
    return width if builds_batches(builder_task) else 1


def builder_outputs(output, builder_name):
    """Return a builder task's structures as a list: it must return one MoleculesObject or a list of them.

    Anything else raises ``TypeError`` naming the builder, instead of being dropped or failing later on a
    field of an older record format.
    """
    if isinstance(output, MoleculesObject):
        return [output]
    if isinstance(output, list) and all(isinstance(item, MoleculesObject) for item in output):
        return output
    raise TypeError(f'Builder {builder_name} returned unsupported {type(output).__name__}; expected '
                    'a MoleculesObject or a list of MoleculesObjects. Migrate legacy records to MoleculesObject.')
