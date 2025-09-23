# The builder code goes here
import glob
import random
import os
import numpy as np
from parsl import python_app, bash_app
import json

import ase
from ase import Atoms
from ase import neighborlist
from ase.geometry.cell import complete_cell
from ase.io import cfg
from ase.io import read, write

from alframework.tools.tools import random_rotation_matrix
from alframework.tools.tools import system_checker
from alframework.tools.tools import build_input_dict
from alframework.tools.molecules_class import MoleculesObject

import random
from copy import deepcopy

@python_app(executors=['alf_sampler_executor'])
def load_reactive_task(moleculeid, builder_config, rattle=True):
    """Loads atomic configurations for reactions from files to be used by the sampler.

    Args:
        moleculeid (str): System unique identifier in the database.
        builder_config (dict): Dictionary containing the builder parameters.
        ratttle(bool): If structures are perturbed slightly

    Returns:
        (MoleculesObject): A MoleculesObject representing the system.
    """

    N = int(builder_config['N_reactions']) - 1

    # Finds random structure and random r/ts or product
    index_mol = random.randint(0,N) # Index for molecules
    index_r_ts_p = random.randint(0,2) # Index for r/ts/p

    r = ase.io.read(builder_config['molecule_library_dir'] + '/' + builder_config['r_files'], index_mol)
    ts = ase.io.read(builder_config['molecule_library_dir'] + '/' + builder_config['ts_files'], index_mol)
    p = ase.io.read(builder_config['molecule_library_dir'] + '/' + builder_config['p_files'], index_mol)

    if rattle or rattle.lower() == 'true':
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
