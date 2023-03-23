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
import random
from copy import deepcopy

#def cfg_loader(model_string,start_molecule_index,cfg_directory,number=1):
#    #TODO: add assertion that cfg_directory exists and has cfg files
#    #TODO: Add assertion that number is a number
#    cfg_list = glob.glob(cfg_directory+'/*.cfg')
#    atom_list = [['mol-{:s}-{:010d}'.format(model_string,start_molecule_index+i),cfg.read_cfg(random.choice(cfg_list))] for i in range(number)]
#    return(atom_list)
    
@python_app(executors=['alf_sampler_executor'])
def simple_cfg_loader_task(moleculeid,builder_params):
    """
    builder_params:
    molecule_library_dir: path to molecule library
    shake: amount to displace each atom
    """
    cfg_list = glob.glob(builder_params['molecule_library_dir']+'/*.cfg')
    cfg_choice = random.choice(cfg_list)
    atom_object = cfg.read_cfg(cfg_choice)
    atom_object.set_positions(atom_object.get_positions() + np.random.uniform(-1*builder_params['shake'],builder_params['shake'],size=atom_object.get_positions().shape))
    return([{'moleculeid':moleculeid,'molecule_library_file':cfg_choice},atom_object,{}])
    
def readMolFiles(molecule_sample_path):
    mol_files = os.listdir(molecule_sample_path)
    mols = []
    moldict = {}
    
    for f in mol_files:
        try:
            moldict[f] = read(molecule_sample_path + f,parallel=False)
            mols.append(read(molecule_sample_path + f,parallel=False))
        except: 
            print('Error loading file:',molecule_sample_path + f)
            print('Skipping...')
            continue
    return(moldict,mols)

def condensed_phase_builder(start_system, molecule_library, solute_molecules=[], solvent_molecules=[], density=1.0, min_dist=1.2, max_patience=200, center_first_molecule=False, shake = 0.05, print_attempt=False):
    #ensure system adhears to formating convention
    system_checker(start_system)
    curr_sys = start_system[1]
    #Make sure we know what all of the molecules are
    #shake is amount of random perturbation added to each molecule x y and z independantly
    #center_first_molecule: First molecule will be placed with no rotation or translation
    for curN in solute_molecules:
        if curN not in list(molecule_library):
            raise RuntimeError("Solute {:s} not in molecule_library.".format(curN))
    for curN in solvent_molecules:
        if curN not in list(molecule_library):
            raise RuntimeError("Solvent {:s} not in molecule_library.".format(curN))
    
    #solvent_molecules can be a dictionary, where the lookup value is a relative probability of selecting that molecule
    
    solvent_weights = np.ones(len(solvent_molecules))
    if solvent_molecules is dict:
        solvent_list = list(solvent_molecules)
        for ind,solvent in  enumerate(solvent_list):
            solvent_weights[ind] = solvent_molecules[solvent]
    else:
        solvent_list = solvent_molecules
    
    actual_dens = 0.0
    attempts_total = 0
    attempts_lsucc = 0
    
    while actual_dens < density or len(solute_molecules)>1:
        if len(solute_molecules)>0:
            new_mol_name = solute_molecules.pop(0)
            new_mol_solute = True
        else:
            if len(solvent_list) == 0:
                break
            new_mol_name = random.choices(solvent_list,weights=solvent_weights)[0]
            new_mol_solute = False
        new_mol = molecule_library[new_mol_name].copy()
        
        if center_first_molecule:
            center_first_molecule=False
            new_mol.set_positions(new_mol.get_positions()-new_mol.get_center_of_mass()+np.diag(complete_cell(curr_sys.get_cell()))[np.newaxis,:]/2+np.random.uniform(-shake,shake,size=new_mol.get_positions().shape))
        else:
            T = np.dot(complete_cell(curr_sys.get_cell()),np.random.uniform(0.0, 1.0, size=3))
            M = random_rotation_matrix() # Rotation
            new_mol.set_positions(new_mol.get_positions()-new_mol.get_center_of_mass())
            xyz = new_mol.get_positions()+np.random.uniform(-shake,shake,size=new_mol.get_positions().shape)
            new_mol.set_positions(np.dot(xyz, M.T)+T)
        
        prop_sys = Atoms(np.concatenate([curr_sys.get_chemical_symbols(),new_mol.get_chemical_symbols()]),
                             positions=np.vstack([curr_sys.get_positions(),new_mol.get_positions()]),
                             cell = curr_sys.get_cell(),
                             pbc = curr_sys.get_pbc())
        
        #Neighborlist is set up so that If there are any neighbors, we have close contacts
        nl = neighborlist.NeighborList(min_dist, skin=0.0, self_interaction=False, bothways=True,primitive=ase.neighborlist.NewPrimitiveNeighborList)
        nl.update(prop_sys)
        failed = False
        #This should only wrap if pbc is True
        #The goal of this code is to determine close contacts, but only between the new frament and existing atoms
        xyz = prop_sys.get_positions(wrap=True)
        for i in range(len(curr_sys), len(prop_sys)):
            indices, offsets = nl.get_neighbors(i)
            for j in indices[indices<len(curr_sys)]:
                #This is a clumsy way of fixing pbcs and basically auto-wraps
                dij = prop_sys.get_distance(i,j,mic=True)
                #dij = np.linalg.norm(xyz[i] - (xyz[j] + np.dot(off, prop_sys.get_cell())))
                if dij < min_dist:
                    failed = True
                    break
            if failed is True:
                break
        
        attempts_total += 1
        attempts_lsucc += 1
        if attempts_lsucc > max_patience:
           break
        
        if not failed:
            attempts_lsucc = 0
            curr_sys = prop_sys.copy()
            actual_dens = (1.66054e-24)*np.sum(curr_sys.get_masses())/((1.0e-24)*curr_sys.get_volume()) # units of grams per cm^3
        elif new_mol_solute:
            #If the new molecule failed, and we are attempting to add solute (definite molecules) re add to list to try again. 
            solute_molecules.append(new_mol_name)
    start_system[0]['target_density'] = density
    start_system[0]['actual_density'] = actual_dens
    start_system[1] = curr_sys
    
    return(start_system)

@python_app(executors=['alf_sampler_executor'])
def simple_condensed_phase_builder_task(moleculeid,builder_params):
    """
    Elements in  builder parameters
        molecule_library_path: path to library of molecular fragments to read in
        solute_molecule_options: listof lists detailing sets of solutes
        solvent_molecules: list or dictionary of solvent molecules. If dictionary, corresponding value is relative weight of solvent
        cell_range: 3X2 list with x, y, and z ranges for cell size 
        Rrange: density range
        min_dist: minimum contact distance between fragments
        max_patience: How many attempts to  make before giving up on build
        center_first_molecule: Boolian,  if true first solute is centered in box and not rotated (useful for large molecules)
        shake: Distance to displace initial configurations
        print_attempt: Boolian,controls printing (set to False)
    """
#    import glob
#    import random
#    import os
#    import numpy as np
#    import json
#    
#    import ase
#    from ase import Atoms
#    from ase import neighborlist
#    from ase.geometry.cell import complete_cell
#    from ase.io import read, write
#    
#    from alframework.tools.tools import random_rotation_matrix
#    import random
#    from copy import deepcopy
    
    cell_shape = [np.random.uniform(dim[0],dim[1]) for dim in builder_params['cell_range']]
    
    empty_system = [{'moleculeid':moleculeid},Atoms(cell=cell_shape),{}]
    
    molecule_library,mols = readMolFiles(builder_params['molecule_library_dir'])
    
    solute_molecules = random.choice(builder_params['solute_molecule_options'])

    feed_parameters = {}
    for arg in builder_params:
        if not arg in ['solute_molecule_options','cell_range','Rrange','molecule_library_path','molecule_library_dir']:
            feed_parameters[arg] = builder_params[arg]
    
    feed_parameters['solute_molecules'] = solute_molecules
    feed_parameters['density'] = np.random.uniform(builder_params['Rrange'][0],builder_params['Rrange'][1])
    
    #with open('/users/bnebgen/ml4chem/Programs/parsl-alf2/examples/IL/temp.txt',"w") as outfile:
    #    outfile.write(json.dumps(feed_parameters))
       
    system = condensed_phase_builder(empty_system,molecule_library,**feed_parameters)
    
    return(system)
    
    
   
   