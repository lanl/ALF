"""ALF's workflow layer imports without any model library.

Sizing, capacity, preflight, HDF5 storage and builder outputs belong to ALF; model adapters such as HIPPYNN's
and the legacy NeuroChem path import their libraries themselves. This keeps that boundary from eroding.
"""

import os
import subprocess
import sys
from pathlib import Path

WORKFLOW_MODULES = (
    'alframework.tools.sizing', 'alframework.tools.capacity', 'alframework.tools.preflight',
    'alframework.tools.label_hdf5', 'alframework.tools.h5_parallel', 'alframework.tools.label_parts',
    'alframework.tools.data_shards', 'alframework.tools.label_sources', 'alframework.tools.ensemble_training',
    'alframework.tools.builder_outputs', 'alframework.tools.retraining',
)
MODEL_LIBRARIES = ('hippynn', 'pyanitools', 'anitraintools', 'ase_interface')
# The coordinator imports a run's labeler to read its signature; ASE labeling needs no ML or GPU library.
ASE_LABELING_MODULES = tuple(f'alframework.qm_interfaces.ase.{name}'
                             for name in ('generic', 'orca', 'qchem', 'vasp', 'workflow'))


def test_the_workflow_layer_imports_without_model_libraries():
    script = f"""
import importlib, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {MODEL_LIBRARIES!r}:
            raise ImportError('Model library blocked: ' + fullname)
sys.meta_path.insert(0, Block())
for module in {WORKFLOW_MODULES!r}:
    importlib.import_module(module)
assert not set({MODEL_LIBRARIES!r}) & set(sys.modules)
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=root,
                            env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_ase_labeling_imports_without_torch_or_hippynn():
    script = f"""
import importlib, sys
for module in {ASE_LABELING_MODULES!r}:
    importlib.import_module(module)
assert not {{'torch', 'hippynn'}} & set(sys.modules), sorted({{'torch', 'hippynn'}} & set(sys.modules))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
