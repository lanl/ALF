import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SimpleLauncher

from parsl import python_app, bash_app

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
import random
from copy import deepcopy


config_running = Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',
            #This executor is kinda strange. We manually increase the node count so that 
            #our parallel qm job gets multiple nodes, but leave nodes_per_block=1 so 
            #that parsl  doesn't assign multiple tasks

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'standard',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 50,

                nodes_per_block=1, # I changed from 2 to 1
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=128 -A w23_ml4chem',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='3:00:00', # I changed from '6:00:00'

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_QM_standby_executor',
            #This executor is kinda strange. We manually increase the node count so that 
            #our parallel qm job gets multiple nodes, but leave nodes_per_block=1 so 
            #that parsl  doesn't assign multiple tasks

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'standard',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 50,

                nodes_per_block=1, # Changed from 2
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=128 -A w23_ml4chem --qos=standby --time-min=00:15:00',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='6:00:00', # Changed from '12:00:00'

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_ML_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -p gpu -A w23_ml4chem_g',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='16:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=8,
            #cpu_affinity='alternating',

            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 5,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -A w23_ml4chem_g ',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='8:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_standby_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=8,
            #cpu_affinity='alternating',

            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 5,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -A w23_ml4chem_g  --time-min=00:10:00 --qos=standby',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='8:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        )
    ]
)

config_debug = Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'debug',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1, # Changed from 2
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=128 --qos debug --reservation debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='1:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_ML_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu_debug',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='1:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=8,
            #cpu_affinity='alternating',

            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu_debug',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='1:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        )
    ]
)

# import glob
# from importlib import import_module
# import os
# import json
# import pickle
# import ase
# import re
# from ase import Atoms
# from ase.io import vasp as vasp_io
   
# @python_app(executors=['alf_QM_executor','alf_QM_standby_executor'])
# def ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
#     """
#         Args:
#             molecule_object (list): List containing the return of the builder task.
#             QM_config (dict): json file containing the parameters for the QM calculation.
#             QM_scratch_dir (str): Scratch path for the QM calculation.
#             properties_list: The property list defined on master_config.json. The properties defined here 
#                              will be extracted from the QM calculation and stored in the database for 
#                              training the NN. We can also specify an unit conversion factor. 
        
#         Returns:
#             (list): A list representing the updated molecule_object received that now stores the QM 
#                     calculation results.
#     """
    
#     import glob
    
#     from ase.calculators.vasp import Vasp as calc_class
    
#     directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
#     properties = list(properties_list)
    
#     command = QM_config['QM_run_command']

#     # Define the calculator with desired pseudopotentials
#     calc = calc_class(directory=directory, command=command, **QM_config['input_list'])
    
#     # Run the calculation
#     calc.calculate(atoms=molecule_object[1], properties=properties)
    
#     molecule_object[2] = calc.results
    
#     convergedre = re.compile('aborting loop because EDIFF is reached')
#     txt = open(directory + '/OUTCAR','r').read()
#     if len(convergedre.findall(txt)) == 1:
#         molecule_object[2]['converged'] = True
#     else:
#         molecule_object[2]['converged'] = False
        
#     return molecule_object

#####################################################################################################
from collections import Counter
from itertools import product

MM_of_Elements = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107,
                  'N': 14.0067, 'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305,
                  'Al': 26.9815386, 'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
                  'K': 39.0983, 'Ca': 40.078, 'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961,
                  'Mn': 54.938045, 'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409,
                  'Ga': 69.723, 'Ge': 72.64, 'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678,
                  'Sr': 87.62, 'Y': 88.90585, 'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063,
                  'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818,
                  'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6, 'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519,
                  'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116, 'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151,
                  'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032,
                  'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04, 'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479,
                  'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.966569,
                  'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804, 'Po': 208.9824, 'At': 209.9871,
                  'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278, 'Th': 232.03806, 'Pa': 231.03588,
                  'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614, 'Cm': 247.0703, 'Bk': 247.0703,
                  'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951, 'No': 259.1009, 'Lr': 262,
                  'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278, 'Ds': 281, 'Rg': 281, 'Cn': 285,
                  'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294}


def create_atomic_system(atom_charges, target_num_atoms=100):
    """
    Create a neutral atomic system

    Args:
      atom_charges (dict): Contains the atom type as key and its valence as values
      target_num_atoms (int): Targeted number of atoms to have in the system

    Returns:
      (dict): A dictionary where the keys are the atom types in the system and the corresponding values the number of
              each atom type in the system.

    # OBS: Sometimes it will be impossible to have the exact number 'taget_num_atoms' in the system due to the random
    #      way in which the atomic system was created. This isn't a problem because the box volume is adjusted based
    #      on the atomic system to yield the exact density that we want (which is what matters).

    """
    total_charge = 0
    atom_types = set(atom_charges.keys())
    atomic_system = {el: 0 for el in atom_types}

    anions = list(filter(lambda an: atom_charges[an] < 0, atom_types))
    cations = list(atom_types.difference(anions))
    atom_types = list(atom_types)

    max_anion_charge = min({atom_charges[anion] for anion in anions}) # anions are negative, so we use min
    max_cation_charge = max({atom_charges[cation] for cation in cations})


    while True:
        # Picking a random atom
        if total_charge == 0:
            atom = random.choice(atom_types)
        else:
            atom = random.choice(anions) if total_charge > 0 else random.choice(cations)
        atomic_system[atom] += 1
        total_charge += atom_charges[atom]

        # We want the charges to always be at most as negative as the most positive cation or as positive as the most
        # negative anion. The reason for this is because when the number of atoms in atomic_system is close to the
        # targeted number of atoms, we can always fix the total charge without having to put a lot of atoms.
        if total_charge > 0 and total_charge > abs(max_anion_charge):
            while total_charge > abs(max_anion_charge):
                balance_atom = random.choice(anions)
                atomic_system[balance_atom] += 1
                total_charge += atom_charges[balance_atom]

        elif total_charge < 0 and abs(total_charge) > max_cation_charge:
            while abs(total_charge) > max_cation_charge:
                balance_atom = random.choice(cations)
                atomic_system[balance_atom] += 1
                total_charge += atom_charges[balance_atom]

        # Checking if the number of atoms in the system is close to the targeted number of atoms.
        current_num_atoms = sum(atomic_system.values())
        if current_num_atoms >= (target_num_atoms - 1):
            if total_charge != 0:
                candidates = list(filter(lambda at: atom_charges[at] == -total_charge, atom_types))
                # If the anion/cation charges spectrum us "gapped" e.g. {-1,-3} then there is a chance that the total
                # charge will be, for instance, +2 and no single atom will neutralize it, so candidates will be empty.
                if candidates:
                    balance_atom = random.choice(candidates)
                    atomic_system[balance_atom] += 1
                    total_charge += atom_charges[balance_atom]

        if total_charge == 0 and current_num_atoms >= (target_num_atoms-1):
            break

    # Making sure that atomic system has the same "order" of keys as atom_charges since it matters for VASP
    atomic_system = {k: atomic_system[k] for k in atom_charges.keys()}

    # Sorting according to MM:
    # atom_types.sort(key=lambda at: MM_of_Elements[at])
    # atomic_system = {at: atomic_system[at] for at in atom_types}

    return atomic_system


def to_mic(r_ij, box_length):
    """
    Impose minimum image convention (MIC) on the array r_ij = r_i - r containing the distances between atom 'i' and all
    other atoms in the system. This is needed to properly calculate the distances when using PBC.

    Args:
      r_ij (np.array): A (n_atoms, 3) matrix r_ij = r_i - r, or a (n_atoms, n_atoms, 3) array containing all r_i - r.
      box_length (float): length of cubic cell

    Returns:
      np.array: r_ij under MIC
    """
    # If r_ij/L is < 0.5 then it is already the minimum distance, but if it is > 0.5 we subtract a box length from it.
    # This works because np.round() will round x > 0.5 to 1 and x <= 0.5 to 0. The idea is that the maximum allowed
    # distance along each dimension has to be at most L/2.
    return r_ij - box_length * np.round(r_ij/box_length)


def construct_simulation_box(atomic_system, min_distance, box_length=None, density=None, scale_coords=True,
                             max_iter=30, max_tries=25):
    """
    Constructs a simulation box given atomic species, their charges, desired density (g/cm^3), and the minimum distance
    between atoms. The atoms in the atomic system are randomly placed in the simulation box.

    Args:
        atomic_system (dict): A dictionary where the keys are the atom types and the values the corresponding number
                              of this atom type to be placed in the box.
        min_distance (float): Minimum absolute distance between any two atoms in Angstroms.
        box_length (float): Length of the cubic box.
        density (float): Density of the atomic system in g/cm^3.
        scale_coords (bool): Scales distances by the box length such that all coordinates are now between [0,1].
        max_iter (int): Maximum number of tries to place an atom in a specific grid.
        max_tries (int): Maximum number of tries to rerun the algorithm and try to fit the atoms that are missing to
                         reach the desired number of atoms.

    Returns:
        (np.array): A (N,3) np.array where 'N' is the total number of atoms in the system.
    """
    # The box volume is defined by the number of atoms, their type, and the desired density.
    if box_length is None and density is None:
        raise Exception('Please specify either the box length or the density')

    if box_length is None: # Estimate box length from the desired density
        mass_system = sum([MM_of_Elements[atom_type] * atom_num / 6.022e23 for atom_type, atom_num in atomic_system.items()]) # [g]
        box_volume = (mass_system / density * 1e24) # [Angstrom^3]
        box_length = box_volume ** (1/3)

    n_atoms = sum(atomic_system.values())

    N = int(box_length // min_distance)
    grid_spacing = box_length / N
    grid = np.arange(0, box_length, grid_spacing)

    if N**3 <= (n_atoms * 1.1):
        raise ValueError(f'Density is too high, there is no way or it will take a very long time to place {n_atoms} atoms '
                         f'in a box length of {box_length:.2f} given that each atom must be {min_distance} apart')

    # Creating a grid that when indexed by (i,j,k) returns the base coordinate at that point
    mesh_grid = np.array(np.meshgrid(grid, grid, grid)).T
    # Swapping axes such that meshgrid[a,b,c] corresponds to gridspacing * [a,b,c]
    mesh_grid = np.swapaxes(mesh_grid, 0, 2)
    mesh_grid = np.swapaxes(mesh_grid, 0, 1)
    occupied_mesh_grid = np.zeros_like(mesh_grid) # Will store the absolute position of each atom in the grid

    all_indices = np.indices((N,N,N)).T.reshape(-1,3)
    np.random.shuffle(all_indices)
    available_indices = list(map(tuple, all_indices))
    occupied_indices = []

    aux_neighbors_idx = list(product((0,-1,1), (0,-1,1), (0,-1,1)))
    del aux_neighbors_idx[0] # Removing (0,0,0) since it is useless
    aux_neighbors_idx = np.array(aux_neighbors_idx)

    rng = np.random.default_rng()
    total_tries = 0

    while len(occupied_indices) < n_atoms:
        if not available_indices:
            # Refill available_indices with the indices that are not occupied after a full sweep over all grid points.
            available_indices = set(map(tuple, all_indices.tolist())).difference(occupied_indices)
            total_tries += 1
            # This increase of max_iter speeds the code because we are now trying harder to fit the remaining atoms
            # in the box for a given index, which may avoid calculating the neighbors and redoing the whole cycle.
            max_iter += 2
            if total_tries > max_tries:
                raise Exception(f'Exceeded maximum iterations, only {len(occupied_indices)} were placed')

        idx = available_indices.pop()
        neighbor_list = np.array(idx) + aux_neighbors_idx

        # Applying PBC to the neighbors
        neighbor_list[neighbor_list == -1] = N-1
        neighbor_list[neighbor_list == N] = 0

        # Getting only the coordinates of the occupied neighbors from the neighbor list since it suffices to check
        # distances for them (its redundant to check distances for a empty cells).
        i_neighbors, j_neighbors, k_neighbors = neighbor_list[:, 0], neighbor_list[:, 1], neighbor_list[:, 2]
        neighbors_coords = occupied_mesh_grid[tuple(i_neighbors), tuple(j_neighbors), tuple(k_neighbors)]
        occupied_neighbors_coords = neighbors_coords[np.any(neighbors_coords, axis=1)] # Empty points have coords (0,0,0)

        # Trying to fit an atom at a random position in an available grid position
        for _ in range(max_iter):
            candidate_coord = mesh_grid[idx] + rng.uniform(0, grid_spacing, size=3)
            distances = np.linalg.norm(to_mic(candidate_coord - occupied_neighbors_coords, box_length), axis=1)
            if np.all(distances > min_distance):
                occupied_mesh_grid[idx] = candidate_coord
                occupied_indices.append(tuple(idx))
                break

    occupied_mesh_grid = occupied_mesh_grid.reshape(-1,3)
    coords = occupied_mesh_grid[np.any(occupied_mesh_grid, axis=1)]
    np.random.shuffle(coords)

    if scale_coords:
        coords /= box_length

    return coords

@python_app(executors=['alf_sampler_executor'])
def atomic_system_task(atom_charges, target_num_atoms, min_distance, box_length_range, moleculeid, max_tries=25, scale_coords=False):
    """
    Builds the atomic system

    Args:
        atom_charges (dict): A dictionary where the keys are the atom types in the system and the corresponding values
                             the charge of each atom type.
        target_num_atoms (int): Targeted total number of atoms in the system.
        min_distance (float): Minimum absolute distance between any two atoms in Angstroms.
        box_length_range (list): A list containing the minimum and maximum simulation box length [L_min, L_max].
        moleculeid (int): Unique identifier of the system.
        max_tries (int): Maximum number of cycles in the atom placing algorithm
        scale_coords (bool): Determines whether to scale the coordinates by the box length or not.

    Returns:
        (list): A list that representing a system/molecule. It contains dict indicating the unique identifier 
                of the system, an ASE Atoms object, and an empty dict that will be used to stored the results of 
                the QM calculations on that system/molecule.
    """
    rng = np.random.default_rng()
    box_length = rng.uniform(min(box_length_range), max(box_length_range))
    atomic_system = create_atomic_system(atom_charges, target_num_atoms)
    coords = construct_simulation_box(atomic_system, min_distance, box_length, max_tries=max_tries, scale_coords=scale_coords)

    atoms_type_list = []
    for k, v in atomic_system.items():
        atoms_type_list.extend([k] * v)

    atoms_object = ase.Atoms(atoms_type_list, coords, pbc=True, cell=np.diag([box_length]*3))

    return [{'moleculeid': moleculeid}, atoms_object, {}]


   
