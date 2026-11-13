"""Names that differ between hippynn versions ALF supports.

hippynn renamed ``hippynn.experiment.serialization`` to ``hippynn.experiment.checkpoints`` (hippynn #206,
November 2026) without keeping the old name, so ALF looks up whichever one the installed hippynn has.
"""
import importlib

CHECKPOINT_MODULES = ('hippynn.experiment.checkpoints', 'hippynn.experiment.serialization')


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
