import numpy as np
import pytest
from ase import Atoms

from alframework.tools.batching import (
    BatchBuffer,
    TaskWidths,
    batching_settings,
    default_batch_key,
    flatten_molecule_output,
    poll_delay,
)
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.fakes import FakeTask


def molecule(name, symbols="OH2", pbc=False):
    atoms = Atoms(symbols, positions=np.random.default_rng(0).normal(size=(len(Atoms(symbols)), 3)))
    if pbc:
        atoms.set_cell(np.eye(3) * 8.0)
        atoms.set_pbc(True)
    return MoleculesObject(atoms, name)


def ids(batches):
    return [[mol.get_moleculeid() for mol in batch] for batch in batches]


def test_batching_is_not_selected_without_block():
    assert batching_settings({}) is None
    assert batching_settings(None) is None
    assert BatchBuffer.from_config({"md_settings": []}) is None


def test_batching_settings_defaults():
    assert batching_settings({"batching": {"batch_size": 4}}) == {
        "batch_size": 4, "partial_policy": "timeout", "max_wait_seconds": 5.0}


@pytest.mark.parametrize("block, reason", [
    ({}, "batch_size is required"),
    ({"batch_size": 0}, "positive integer"),
    ({"batch_size": 2.5}, "positive integer"),
    ({"batch_size": True}, "positive integer"),
    ({"batch_size": 2, "partial_policy": "eventually"}, "partial_policy"),
    ({"batch_size": 2, "max_wait_seconds": -1}, "max_wait_seconds"),
    ({"batch_size": 2, "unknown": 1}, "Unknown"),
], ids=["no_size", "zero_size", "fractional_size", "boolean_size", "unknown_policy", "negative_wait", "unknown_key"])
def test_invalid_batching_settings_are_rejected(block, reason):
    with pytest.raises(ValueError, match=reason):
        batching_settings({"batching": block})


def test_flatten_preserves_order_and_identity():
    a, b, c = molecule("a"), molecule("b"), molecule("c")
    flat = flatten_molecule_output([a, [b, (c,)], None, []])
    assert [m.get_moleculeid() for m in flat] == ["a", "b", "c"]
    assert flat[1] is b
    assert flatten_molecule_output(None) == []
    with pytest.raises(TypeError, match="MoleculesObject"):
        flatten_molecule_output([a, "not a molecule"])


def test_full_batches_release_immediately_and_mix_sizes_and_compositions():
    buffer = BatchBuffer(3)
    buffer.extend([molecule("w0"), molecule("h0", "H2"), molecule("w1", "OH2OH2")], now=0.0)
    buffer.add(molecule("w2"), now=0.0)
    assert ids(buffer.pop_ready(now=0.0)) == [["w0", "h0", "w1"]]
    assert len(buffer) == 1


def test_periodic_and_nonperiodic_inputs_are_not_grouped():
    buffer = BatchBuffer(2)
    buffer.extend([molecule("gas0"), molecule("pbc0", pbc=True),
                   molecule("gas1"), molecule("pbc1", pbc=True)], now=0.0)
    assert ids(buffer.pop_ready(now=0.0)) == [["gas0", "gas1"], ["pbc0", "pbc1"]]


def test_partial_batch_times_out_without_new_input():
    buffer = BatchBuffer(4, max_wait_seconds=5.0)
    buffer.extend([molecule("a"), molecule("b")], now=100.0)
    assert buffer.pop_ready(now=104.9) == []
    assert buffer.seconds_until_ready(now=103.0) == pytest.approx(2.0)
    assert ids(buffer.pop_ready(now=105.0)) == [["a", "b"]]
    assert len(buffer) == 0
    assert buffer.seconds_until_ready() is None


def test_full_only_policy_waits_until_flush():
    buffer = BatchBuffer(4, partial_policy="full_only")
    buffer.add(molecule("a"), now=0.0)
    assert buffer.pop_ready(now=1e9) == []
    assert buffer.seconds_until_ready(now=1e9) is None
    assert ids(buffer.pop_ready(flush=True)) == [["a"]]


def test_remaining_members_keep_their_original_deadline():
    buffer = BatchBuffer(2, max_wait_seconds=5.0)
    buffer.add(molecule("a"), now=0.0)
    buffer.add(molecule("b"), now=1.0)
    buffer.add(molecule("c"), now=4.0)
    assert ids(buffer.pop_ready(now=4.0)) == [["a", "b"]]
    assert buffer.pop_ready(now=8.9) == []
    assert ids(buffer.pop_ready(now=9.0)) == [["c"]]


def test_custom_key_function_extends_compatibility():
    def key(mol):
        return default_batch_key(mol) + (mol.get_metadata().get("selected_state"),)

    buffer = BatchBuffer(2, key_function=key)
    for name, state in (("a", 0), ("b", 1), ("c", 0), ("d", 1)):
        mol = molecule(name)
        mol.update_metadata({"selected_state": state})
        buffer.add(mol, now=0.0)
    assert ids(buffer.pop_ready(now=0.0)) == [["a", "c"], ["b", "d"]]


def test_batch_builder_ids_survive_regrouping():
    builder_outputs = [[molecule(f"mol-{task}-{i}") for i in range(5)] for task in range(2)]
    expected = [m.get_moleculeid() for out in builder_outputs for m in out]
    buffer = BatchBuffer(3)
    for output in builder_outputs:
        buffer.extend(flatten_molecule_output(output), now=0.0)
    batches = buffer.pop_ready(now=0.0) + buffer.pop_ready(flush=True)
    assert [len(batch) for batch in batches] == [3, 3, 3, 1]
    assert [m.get_moleculeid() for batch in batches for m in batch] == expected


def test_buffer_status_is_json_safe():
    import json
    buffer = BatchBuffer(2)
    buffer.add(molecule("a", pbc=True), now=0.0)
    status = buffer.status(now=3.0)
    json.dumps(status)
    assert status["waiting"] == 1 and status["buckets"][0]["oldest_age_seconds"] == 3.0


def test_task_widths_count_actual_inputs_and_default_to_one():
    widths = TaskWidths()
    full, partial, legacy = (FakeTask("running", running=True, done=False) for _ in range(3))
    widths.record(full, 8)
    widths.record(partial, 3)
    assert widths.total([full, partial, legacy]) == 12
    widths.prune([partial])
    assert widths.total([partial]) == 3
    assert widths.width(full) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        widths.record(full, -1)


def test_poll_delay_honors_nearest_timeout():
    a, b = BatchBuffer(4, max_wait_seconds=5.0), BatchBuffer(4, max_wait_seconds=30.0)
    assert poll_delay(60, a, b, None) == 60
    a.add(molecule("x"), now=0.0)
    b.add(molecule("y"), now=0.0)
    assert poll_delay(60, a, b, now=2.0) == pytest.approx(3.0)
    assert poll_delay(1, a, now=0.0) == 1


def test_finite_bootstrap_ids_never_wrap_and_preserve_ordinary_width():
    from alframework.tools.batching import bootstrap_molecule_ids
    assert bootstrap_molecule_ids(8,10,4,True)==['mol-boot-0000000008','mol-boot-0000000009']
    assert bootstrap_molecule_ids(10,10,4,True)==[]
    assert len(bootstrap_molecule_ids(8,10,4,False))==4
