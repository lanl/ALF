"""Slightly displace each example geometry before evaluating its DFT forces."""

import hashlib
import re

import numpy as np
from parsl import python_app

from alframework.builders.surface import build_surface
from alframework.tools.molecules import MoleculesObject
from alframework.tools.task_timeline import timed


def rattle_surface(molecule, builder_config):
    scale = float(builder_config.get('rattle_A', 0.025))
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('rattle_A must be finite and nonnegative')
    source = f'{builder_config.get("seed", 0)}|{molecule.get_moleculeid()}|surface_training_rattle'
    seed = int.from_bytes(hashlib.sha256(source.encode()).digest()[:16], 'big')
    atoms = molecule.get_atoms()
    atoms.rattle(stdev=scale, rng=np.random.default_rng(seed))
    molecule.update_metadata({'rattle_A': scale})
    return molecule


@python_app(executors=['alf_sampler_executor'])
@timed('builder')
def surface_training_builder_task(moleculeid, builder_config, master_directory=None):
    """Use the ALF surface builder and perturb coordinates in the fixed cell."""
    return rattle_surface(build_surface(moleculeid, builder_config, master_directory), builder_config)


@python_app(executors=['alf_sampler_executor'])
@timed('builder')
def perovskite_training_builder_task(moleculeid, builder_config, master_directory=None, current_h5_id=0):
    """Cycle through all material, ligand and termination groups during bootstrap."""
    if __package__:
        from .perovskite_benchmark import termination
    else:
        from perovskite_benchmark import termination

    index = re.search(r'(\d+)$', moleculeid)
    if current_h5_id or index is None:
        return rattle_surface(build_surface(moleculeid, builder_config, master_directory), builder_config)
    materials = builder_config['materials']
    material, face = divmod(int(index.group(1)) % (2 * len(materials)), 2)
    target = ('CsX', 'MX2')[face]
    selected = {**builder_config, 'materials': [materials[material]]}
    for attempt in range(32):
        pose = build_surface(f'{moleculeid}-pose-{attempt:02d}', selected, master_directory)
        if termination(pose.get_atoms()) == target:
            frame = MoleculesObject(pose.get_atoms(), moleculeid)
            frame.update_metadata(pose.get_metadata())
            return rattle_surface(frame, builder_config)
    raise RuntimeError(f'Could not prepare {target} termination for {materials[material]["id"]}')
