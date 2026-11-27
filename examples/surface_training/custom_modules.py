"""Slightly displace each example geometry before evaluating its DFT forces."""

import hashlib

import numpy as np
from parsl import python_app

from alframework.builders.surface import build_surface
from alframework.tools.task_timeline import timed


@python_app(executors=['alf_sampler_executor'])
@timed('builder')
def surface_training_builder_task(moleculeid, builder_config, master_directory=None):
    """Use the ALF surface builder and perturb coordinates in the fixed cell."""
    molecule = build_surface(moleculeid, builder_config, master_directory)
    scale = float(builder_config.get('rattle_A', 0.025))
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('rattle_A must be finite and nonnegative')
    source = f'{builder_config.get("seed", 0)}|{moleculeid}|surface_training_rattle'
    seed = int.from_bytes(hashlib.sha256(source.encode()).digest()[:16], 'big')
    atoms = molecule.get_atoms()
    atoms.rattle(stdev=scale, rng=np.random.default_rng(seed))
    molecule.update_metadata({'rattle_A': scale})
    return molecule
