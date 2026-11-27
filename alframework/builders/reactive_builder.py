# The builder code goes here
import random
import numpy as np

import ase

from alframework.tools.molecules import MoleculesObject
from alframework.tools.tasks import alf_task



def _rattle_enabled(value):
    """``rattle`` as a boolean: True, False, or the string true or false in any case."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        return value.strip().lower() == 'true'
    raise ValueError('rattle must be a boolean or the string true/false.')


@alf_task('builder')
def load_reactive_task(moleculeid, builder_config, rattle=True):
    """Loads atomic configurations for reactions from files to be used by the sampler.

    Args:
        moleculeid (str): System unique identifier in the database.
        builder_config (dict): Dictionary containing the builder parameters.
        ratttle(bool): If structures are perturbed slightly

    Returns:
        (MoleculesObject): A MoleculesObject representing the system.
    """

    rattle = _rattle_enabled(rattle)
    N = int(builder_config['N_reactions']) - 1

    # Finds random structure and random r/ts or product
    index_mol = random.randint(0,N) # Index for molecules
    index_r_ts_p = random.randint(0,2) # Index for r/ts/p

    r = ase.io.read(builder_config['molecule_library_dir'] + '/' + builder_config['r_files'], index_mol)
    ts = ase.io.read(builder_config['molecule_library_dir'] + '/' + builder_config['ts_files'], index_mol)
    p = ase.io.read(builder_config['molecule_library_dir'] + '/' + builder_config['p_files'], index_mol)

    if rattle:
        r.rattle()
        ts.rattle()
        p.rattle()

    atoms = [r, ts, p][index_r_ts_p]
        
    molecule_object = MoleculesObject(atoms, moleculeid)

    # The metadata contains the other structures for the reaction, these are accessed by the sampler
    molecule_object.metadata['reactant'] = r
    molecule_object.metadata['ts'] = ts
    molecule_object.metadata['product'] = p
    molecule_object.metadata['index_mol'] = index_mol
    
    return molecule_object
