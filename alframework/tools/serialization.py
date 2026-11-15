"""Opt-in faster serialization of task data: plain pickle first, Parsl's dill as the fallback.

Parsl serializes every task's arguments in the coordinator, and for data it uses dill, a pure-Python pickler.
For ALF's structures that is the coordinator's main cost at scale: one QM task's 256 structures took about
1.4 s with dill and 70 ms with pickle. With ``parsl_data_serializer: "pickle"`` in the master config, the
coordinator tries pickle first; anything pickle can't serialize, or that refers to the coordinator's own
``__main__`` (which a worker couldn't import), still goes through dill. Workers need no setup: Parsl loads a
deserializer it doesn't know from the payload's header, ``alframework.tools.serialization PickleSerializer``.
Results from workers are serialized there, as before.
"""
import pickle

from parsl.serialize.base import SerializerBase

DATA_SERIALIZERS = ('dill', 'pickle')


class PickleSerializer(SerializerBase):
    """pickle, for data a worker can import; refuses (so the next method is tried) what it can't do safely."""

    def serialize(self, data):
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        # Classes and functions are pickled by reference; one from the coordinator's __main__ would fail on the
        # worker, after serialization succeeded here, so leave those to dill, which pickles them by value.
        if b'__main__' in payload:
            raise pickle.PicklingError('refers to __main__; leaving it to dill')
        return payload

    def deserialize(self, payload):
        return pickle.loads(payload)


def data_serializer(master_config):
    """``parsl_data_serializer`` from the master config, checked: ``"dill"`` (default, Parsl's own) or ``"pickle"``."""
    choice = master_config.get('parsl_data_serializer', 'dill')
    if choice not in DATA_SERIALIZERS:
        raise ValueError(f'parsl_data_serializer must be one of {DATA_SERIALIZERS}, not {choice!r}')
    return choice


def use_data_serializer(choice):
    """Make Parsl try pickle before its other data serializers (``"pickle"``), or leave Parsl as it is (``"dill"``)."""
    if choice == 'dill':
        return
    from parsl.serialize import facade
    first = PickleSerializer()
    others = {key: value for key, value in facade.methods_for_data.items() if key != first.identifier}
    facade.methods_for_data.clear()
    facade.methods_for_data[first.identifier] = first
    facade.methods_for_data.update(others)
