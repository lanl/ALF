"""The builder return contract, checked the same way in the bootstrap, active learning and --test_builder."""
from alframework.tools.molecules_class import MoleculesObject


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
