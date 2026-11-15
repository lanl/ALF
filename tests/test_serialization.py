"""parsl_data_serializer "pickle": task data serialized with pickle first, dill as the fallback."""

import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from parsl.serialize import facade

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.serialization import PickleSerializer, data_serializer, use_data_serializer

PICKLE = PickleSerializer().identifier


@pytest.fixture
def pickle_first():
    saved = dict(facade.methods_for_data)
    use_data_serializer("pickle")
    yield
    facade.methods_for_data.clear()
    facade.methods_for_data.update(saved)


def molecules(count=3):
    out = []
    for i in range(count):
        molecule = MoleculesObject(Atoms("H2O", positions=np.eye(3) * (1 + i)), f"mol-{i}")
        molecule.update_metadata({"temps": [300.0] * 5, "step": i})
        out.append(molecule)
    return out


def test_dill_stays_the_default():
    saved = dict(facade.methods_for_data)
    use_data_serializer("dill")
    assert facade.methods_for_data == saved
    assert data_serializer({}) == "dill"
    with pytest.raises(ValueError, match="parsl_data_serializer"):
        data_serializer({"parsl_data_serializer": "json"})


def test_structures_go_through_pickle_and_come_back_the_same(pickle_first):
    payload = facade.serialize(molecules())
    assert payload.startswith(PICKLE + b"\n")
    back = facade.deserialize(payload)
    assert [m.get_moleculeid() for m in back] == ["mol-0", "mol-1", "mol-2"]
    assert np.array_equal(back[2].get_atoms().get_positions(), np.eye(3) * 3)
    assert back[1].get_metadata() == {"temps": [300.0] * 5, "step": 1}


def test_what_pickle_cannot_do_falls_back_to_dill(pickle_first):
    payload = facade.serialize({"transform": lambda x: x + 1})
    assert not payload.startswith(PICKLE)
    assert facade.deserialize(payload)["transform"](1) == 2


def test_references_to_the_coordinators_main_go_to_dill(pickle_first, monkeypatch):
    """A class from __main__ pickles by reference and would fail on the worker, so it isn't left to pickle."""
    import __main__

    class Local:
        pass

    Local.__module__ = "__main__"
    Local.__qualname__ = "Local"
    monkeypatch.setattr(__main__, "Local", Local, raising=False)
    assert not facade.serialize(Local()).startswith(PICKLE)


def test_enabling_twice_keeps_one_pickle_serializer_first(pickle_first):
    use_data_serializer("pickle")
    keys = list(facade.methods_for_data)
    assert keys[0] == PICKLE and keys.count(PICKLE) == 1 and len(keys) >= 2


def test_a_worker_with_parsls_defaults_reads_the_payload(pickle_first, tmp_path):
    """Workers don't register anything: Parsl loads the deserializer named in the payload's header."""
    path = tmp_path / "payload.bin"
    path.write_bytes(facade.serialize(molecules(2)))
    reader = ("import sys; from parsl.serialize import facade; "
              "assert all(b'alframework' not in k for k in facade.methods_for_data); "
              "out = facade.deserialize(open(sys.argv[1], 'rb').read()); "
              "print([m.get_moleculeid() for m in out])")
    result = subprocess.run([sys.executable, "-c", reader, str(path)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "['mol-0', 'mol-1']"


def test_the_coordinator_applies_the_setting_before_parsl_loads():
    from pathlib import Path

    main = (Path(__file__).resolve().parents[1] / "alframework" / "__main__.py").read_text()
    assert "use_data_serializer(data_serializer(master_config))" in main
    assert main.index("use_data_serializer(data_serializer(master_config))") < main.index("parsl.load(")
