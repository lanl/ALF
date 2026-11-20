"""Names that differ between hippynn versions ALF supports.

hippynn renamed ``hippynn.experiment.serialization`` to ``hippynn.experiment.checkpoints`` (hippynn #206,
November 2026) without keeping the old name, so ALF looks up whichever one the installed hippynn has.
It also renamed a graph's ``node_from_name`` to ``unique_node_from_name`` and an energy target's
``mol_energy`` to ``system_energy``, keeping the old names with a deprecation warning; release 0.1.3 has
only the old ones.
"""
import importlib

CHECKPOINT_MODULES = ('hippynn.experiment.checkpoints', 'hippynn.experiment.serialization')


def node_from_name(graph, name):
    """The node of ``graph`` named ``name``, with whichever method the installed hippynn has."""
    find = getattr(graph, 'unique_node_from_name', None)
    return find(name) if find is not None else graph.node_from_name(name)


def system_energy(energy_target):
    """An energy target's system energy node: ``system_energy``, or ``mol_energy`` in older hippynn."""
    kind = type(energy_target)
    outputs = getattr(kind, 'output_names', None) or getattr(kind, '_output_names', ())  # 0.1.3: _output_names
    return energy_target.system_energy if 'system_energy' in outputs else energy_target.mol_energy


def checkpoints():
    """Return hippynn's checkpoint module (``load_checkpoint``, ``load_saved_tensors``, ``create_state``, ...).

    Only the module's own absence moves on to the older name; any other import error, such as a missing
    dependency inside it or no hippynn at all, is raised as it is.
    """
    for name in CHECKPOINT_MODULES:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    raise ModuleNotFoundError(f'hippynn provides none of {", ".join(CHECKPOINT_MODULES)}', name=CHECKPOINT_MODULES[0])
