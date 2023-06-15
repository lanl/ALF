import os
import random
import subprocess
import tempfile
from ase import Atoms
from ase.io import read, write
from parsl import python_app, bash_app
from alframework.tools.tools import system_checker


def qcg_grow(start_system, molecule_library_dir, solute_molecules=[], solvent_molecules=[], nsolv=[1,2]):

    #ensure system adhears to formating convention
    system_checker(start_system)
    curr_sys = start_system[1]
    cwd = os.getcwd()
  
    #path to solute/solvent xyz
    solute_xyz = os.path.join(cwd, molecule_library_dir, solute_molecules[0])
    solvent_xyz = os.path.join(cwd, molecule_library_dir, solvent_molecules[0])
    nsolv_start, nsolv_stop = nsolv
    nsolv = random.choice(range(int(nsolv_start), int(nsolv_stop)))
    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        #run crest
        runcmd = ['crest', solute_xyz, '-qcg', solvent_xyz, '--nsolv', str(nsolv)]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        #load cluster
        cluster_xyz = os.path.join(tmpdirname, 'grow', 'qcg_grow.xyz') 
        start_system[1] = read(cluster_xyz)

    return(start_system)


@python_app(executors=['alf_sampler_executor'])
def qcg_grow_task(moleculeid, builder_config):
    """
    Elements in builder params
        molecule_library_dir: path to library of molecular fragments to read in
        solute_molecule_options: listof lists detailing sets of solutes
        solvent_molecules: list or dictionary of solvent molecules. If dictionary, corresponding value is relative weight of solvent
        nsolv: int. number of solvent molecules
    """
    empty_system = [{'moleculeid':moleculeid}, Atoms(), {}]
    system = qcg_grow(empty_system, **builder_config)
    system_checker(system)
    return(system)

