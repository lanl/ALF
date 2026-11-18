"""Previous sampler helper paths still resolve in a fresh Python process."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("canonical_first", [True, False])
def test_moved_sampler_modules(canonical_first):
    script = f"""
import importlib
import pickle
import sys
import warnings

from alframework.tools.tools import load_module_from_string

moved = [
    ("packed_engine", "samplers.engines.packed", "PackedState"),
    ("torchsim_engine", "samplers.engines.torchsim", "TorchSimEngine"),
    ("alchemi_engine", "samplers.engines.alchemi", "AlchemiEngine"),
    ("geometry", "tools.geometry", "minimum_distance"),
    ("velocities", "tools.velocities", "initialize_velocities"),
]
for old, new, name in moved:
    old = "alframework.samplers." + old
    new = "alframework." + new
    if {canonical_first!r}:
        importlib.import_module(new)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = pickle.loads(f"c{{old}}\\n{{name}}\\n.".encode())
    assert any(
        issubclass(w.category, FutureWarning) and old in str(w.message) and new in str(w.message)
        for w in caught
    ), caught
    canonical = importlib.import_module(new)
    assert canonical is importlib.import_module(old)
    assert canonical.__spec__.name == new
    assert loaded is load_module_from_string(f"{{old}}.{{name}}") is getattr(canonical, name)

from alframework.samplers import geometry
assert geometry is importlib.import_module("alframework.tools.geometry")
assert not ({{"torch", "torch_sim", "nvalchemi", "nvalchemiops", "warp"}} & sys.modules.keys())
"""
    subprocess.run([sys.executable, "-B", "-c", script], check=True, timeout=60)


def test_batched_mlmd_engine_reexports():
    from alframework.samplers import batched_mlmd
    from alframework.samplers.engines import ase, factory

    for name in ("ASEEngine", "OBSERVED_PROPERTIES", "_PredictedMember", "_StateMeanCalculator"):
        assert getattr(batched_mlmd, name) is getattr(ase, name)
    assert batched_mlmd.make_engine is factory.make_engine
