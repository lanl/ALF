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
from alframework.tools.tools import build_input_dict
from alframework.tools.molecules_class import MoleculesObject
import random
from copy import deepcopy


# def cfg_loader(model_string,start_molecule_index,cfg_directory,number=1):
#    #TODO: add assertion that cfg_directory exists and has cfg files
#    #TODO: Add assertion that number is a number
#    cfg_list = glob.glob(cfg_directory+'/*.cfg')
#    atom_list = [['mol-{:s}-{:010d}'.format(model_string,start_molecule_index+i),cfg.read_cfg(random.choice(cfg_list))] for i in range(number)]
#    return(atom_list)

@python_app(executors=['alf_sampler_executor'])
def simple_cfg_loader_task(moleculeid, builder_config, shake=0.05):
    """Loads an atomic configuration from cfg file to be used by the sampler.

    Args:
        moleculeid (str): System unique identifier in the database.
        builder_config (dict): Dictionary containing the builder parameters.
        shake (float): Amount of random pertubation added to each atom coordinate.

    Returns:
        (MoleculesObject): A MoleculesObject representing the system.
    """
    cfg_list = glob.glob(builder_config['molecule_library_dir'] + '/*.cfg')
    cfg_choice = random.choice(cfg_list)
    ase_atoms = cfg.read_cfg(cfg_choice)
    ase_atoms.rattle(shake)

    molecule_object = MoleculesObject(ase_atoms, moleculeid)
    molecule_object.update_metadata({'molecule_library_file': cfg_choice})

    return molecule_object


def readMolFiles(molecule_sample_path):
    """Reads MOL files.

    Args:
        molecule_sample_path (str): Path of the directory that stores the MOL files to be read.

    Returns:
        (tuple): Tuple (moldict, mols) where the first element stores the MOL files as a dict and the second as a list.

    """
    mol_files = os.listdir(molecule_sample_path)
    mols = []
    moldict = {}

    for f in mol_files:
        try:
            moldict[f] = read(molecule_sample_path + f, parallel=False)
            mols.append(read(molecule_sample_path + f, parallel=False))
        except:
            print('Error loading file:', molecule_sample_path + f)
            print('Skipping...')
            continue

    return moldict, mols


def condensed_phase_builder(start_system, molecule_library, solute_molecules=[], solvent_molecules=[], density=1.0,
                            min_dist=1.2, max_patience=200, max_atoms=None, center_first_molecule=False, shake=0.05, early_stop_probability=0.0):
    """Builder for condensed phase systems.

        Args:
            start_system (MoleculesObject): An object from the class MoleculesObject.
            molecule_library (dict): Dictionary whose keys are the molecules present in the system and values are
                                     ase.Atoms objects representing that molecule.
            solute_molecules (list): List of the solute molecules.
            solvent_molecules (list or dict): List of dict storing the solvent molecules. If it is a dict, then the
                                              keys are the molecules and the vales the relative probability of
                                              selecting that molecule.
            density (float): Density of the system.
            min_dist (float): Minimum distance between atoms.
            max_patience (int): Number of max tries before giving up on a given system.
            max_atoms (int): Maximum number of atoms allowed.
            center_first_molecule (bool): If True the first molecules will be placed with no rotation nor translation.
            shake (float): Amount of random perturbation added to each molecule on the x-, y-, z-axes independently.
            early_stop_probability (float): After successfully adding a molecule, prabability of stopping. For generating more smaller systems.

        Returns:
            (MoleculesObject): An updated MoleculesObject representing the system.

    """
    # ensure system adhears to formating convention
    assert isinstance(start_system, MoleculesObject), 'start_system must be an instance of MoleculesObject'
    curr_sys = start_system.get_atoms()

    # Make sure we know what all of the molecules are
    for curN in solute_molecules:
        if curN not in list(molecule_library):
            raise RuntimeError("Solute {:s} not in molecule_library.".format(curN))
    for curN in solvent_molecules:
        if curN not in list(molecule_library):
            raise RuntimeError("Solvent {:s} not in molecule_library.".format(curN))

    # solvent_molecules can be a dictionary, where the lookup value is a relative probability of selecting that molecule
    solvent_weights = np.ones(len(solvent_molecules))
    if isinstance(solvent_molecules, dict):
        # solvent_list, solvent_weights = map(list, zip(*solvent_molecules.items())) # faster and easier way to do the code below
        solvent_list = list(solvent_molecules.keys())
        for idx, solvent in enumerate(solvent_list):
            solvent_weights[idx] = solvent_molecules[solvent]
    else:
        solvent_list = solvent_molecules

    actual_dens = 0.0
    attempts_total = 0
    attempts_lsucc = 0

    while actual_dens < density or len(solute_molecules) > 1:
        if max_atoms is not None:
            if len(curr_sys) > max_atoms:
                break
        if len(solute_molecules) > 0:
            new_mol_name = solute_molecules.pop(0)
            new_mol_solute = True
        else:
            if len(solvent_list) == 0:
                break
            new_mol_name = random.choices(solvent_list, weights=solvent_weights)[0]
            new_mol_solute = False
        new_mol = molecule_library[new_mol_name].copy()

        if center_first_molecule:
            center_first_molecule = False
            new_mol.set_positions(new_mol.get_positions() - new_mol.get_center_of_mass() +
                                  np.diag(complete_cell(curr_sys.get_cell()))[np.newaxis, :] / 2 +
                                  np.random.uniform(-shake, shake, size=new_mol.get_positions().shape)
                                  )

        else:
            T = np.dot(complete_cell(curr_sys.get_cell()), np.random.uniform(0.0, 1.0, size=3))
            M = random_rotation_matrix()
            new_mol.set_positions(new_mol.get_positions() - new_mol.get_center_of_mass())
            xyz = new_mol.get_positions() + np.random.uniform(-shake, shake, size=new_mol.get_positions().shape)
            new_mol.set_positions(np.dot(xyz, M.T) + T)

        prop_sys = Atoms(np.concatenate([curr_sys.get_chemical_symbols(), new_mol.get_chemical_symbols()]),
                         positions=np.vstack([curr_sys.get_positions(), new_mol.get_positions()]),
                         cell=curr_sys.get_cell(),
                         pbc=curr_sys.get_pbc())

        # Neighborlist is set up so that If there are any neighbors, we have close contacts
        nl = neighborlist.NeighborList(min_dist, skin=0.0, self_interaction=False, bothways=True,
                                       primitive=ase.neighborlist.NewPrimitiveNeighborList)
        nl.update(prop_sys)
        failed = False
        # This should only wrap if pbc is True
        # The goal of this code is to determine close contacts, but only between the new frament and existing atoms
        for i in range(len(curr_sys), len(prop_sys)):
            indices, offsets = nl.get_neighbors(i)
            for j in indices[indices < len(curr_sys)]:
                # This is a clumsy way of fixing pbcs and basically auto-wraps
                dij = prop_sys.get_distance(i, j, mic=True)
                # dij = np.linalg.norm(xyz[i] - (xyz[j] + np.dot(off, prop_sys.get_cell())))
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
            actual_dens = 1.66054e-24 * np.sum(curr_sys.get_masses()) / (1.0e-24 * curr_sys.get_volume()) # [g/cm^3]
                if early_stop_probability > random.random():
                    break
        elif new_mol_solute:
            # If the new molecule failed, and we are attempting to add solute (definite molecules) re add to list to try again.
            solute_molecules.append(new_mol_name)

    start_system.update_metadata({'target_density': density, 'actual_density': actual_dens})
    start_system.update_atoms(curr_sys)

    return start_system


@python_app(executors=['alf_sampler_executor'])
def simple_condensed_phase_builder_task(moleculeid, builder_config, molecule_library_dir, cell_range,
                                        solute_molecule_options, Rrange):
    """Condensed phase builder task that will be fed to the sampler.

    Args:
        moleculeid (str): String that uniquely identifies a system in the database.
        builder_config (dict): Dictionary that stores the config parameters of the builder.
        molecule_library_dir (str): Path of the molecule library directory.
        cell_range (list): A (3,2) list with the x, y, and z ranges for the cell size [min, max].
        solute_molecule_options (list): List of lists detailing sets of solutes.
        Rrange (list): Density range [density_min, density_max].

    Returns:
        (MoleculesObject): A MoleculesObject representing the system.
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

    cell_shape = [np.random.uniform(dim[0], dim[1]) for dim in cell_range]

    empty_system = MoleculesObject(Atoms(cell=cell_shape), moleculeid)

    molecule_library, mols = readMolFiles(molecule_library_dir)

    solute_molecules = random.choice(solute_molecule_options)

    feed_parameters = {'solute_molecules': solute_molecules, 'density': np.random.uniform(Rrange[0], Rrange[1])}

    input_parameters = build_input_dict(condensed_phase_builder,
                                        [{"start_system": empty_system, "molecule_library": molecule_library},
                                         feed_parameters,
                                         builder_config]
                                        )

    system = condensed_phase_builder(**input_parameters)
    assert isinstance(system, MoleculesObject), 'system must be an instance of MoleculesObject'

    return system


@python_app(executors=['alf_sampler_executor'])
def simple_multi_condensed_phase_builder_task(moleculeids, builder_config, molecule_library_dir, cell_range,
                                              solute_molecule_options, Rrange):
    """Multi condensed phase builder task that will be fed to the sampler.

    Args:
        moleculeids (str): List of strings that uniquely identifies a system in the database.
        builder_config (dict): Dictionary that stores the config parameters of the builder.
        molecule_library_dir (str): Path of the molecule library directory.
        cell_range (list): A (3,2) list with the x, y, and z ranges for the cell size [min, max].
        solute_molecule_options (list): List of lists detailing sets of solutes.
        Rrange (list): Density range [density_min, density_max].

    Returns:
        (list): A list of objects from MoleculesObject.
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

    molecule_library, mols = readMolFiles(molecule_library_dir)
    system_list = []

    for moleculeid in moleculeids:
        cell_shape = [np.random.uniform(dim[0], dim[1]) for dim in cell_range]

        empty_system = MoleculesObject(Atoms(cell=cell_shape), moleculeid)

        feed_parameters = {'solute_molecules': random.choice(solute_molecule_options),
                           'density': np.random.uniform(Rrange[0], Rrange[1])}

        input_parameters = build_input_dict(condensed_phase_builder,
                                            [{"start_system": empty_system, "molecule_library": molecule_library},
                                             feed_parameters, builder_config])

        system = condensed_phase_builder(**input_parameters)
        assert isinstance(system, MoleculesObject), 'system must be an instance of MoleculesObject'
        system_list.append(system)

    return system_list

#####################################################################################################
from collections import Counter
from ase.data import atomic_masses, atomic_numbers
from itertools import product

def create_atomic_system(atom_charges, target_num_atoms):
    """Create a neutral atomic system.

    Sometimes it will be impossible to have the exact number 'taget_num_atoms' in the system due to the random
    way in which the atoms were chosen. This isn't a problem because the box volume can be adjusted to yield the
    desired density.

    Args:
      atom_charges (dict): Contains the atom type as key and its valence as values.
      target_num_atoms (int): Targeted number of atoms to have in the system.

    Returns:
      (dict): A dictionary where the keys are the atom types in the system and the corresponding values the number of
              each atom type in the system.

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
        # negative anion. The reason for this is because when the number of atoms in 'atomic_system' is close to the
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

    # Keeping only the elements whose frequency is greater than zero
    atomic_system = {atom_type: atomic_system[atom_type]
                     for atom_type in atom_charges.keys() if atomic_system[atom_type] > 0}

    return atomic_system


def to_mic(r_ij, box_length):
    """Impose minimum image convention (MIC) on the array containing the distances between atoms.

    Args:
        r_ij (ndarray): A (n_atoms, 3) matrix r_ij = r_i - r, or a (n_atoms, n_atoms, 3) array containing all r_i - r.
        box_length (float): Length of cubic cell.

    Returns:
        (ndarray): Distances under MIC.
    """
    return r_ij - box_length * np.round(r_ij/box_length)


def construct_simulation_box(atomic_system, min_distance, box_length=None, density=None, scale_coords=True,
                             max_iter=30, max_tries=25):
    """Constructs a initial configuration for the given atomic system respecting the imposed constraints.

    Given a dictionary containing the atomic species and their charges, a random configuration of the system is
    constructed respecting the constraints (desired density (g/cm^3), box length, minimum distance between atoms).
    The atoms are randomly placed in the simulation box.

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
        (ndarray): A (N,3) array where 'N' is the total number of atoms in the system.
    """
    # The box volume is defined by the number of atoms, their type, and the desired density.
    if box_length is None and density is None:
        raise Exception('Please specify either the box length or the density')

    if box_length is None: # Estimate box length from the desired density
        mass_system = sum([atomic_masses[atomic_numbers[atom_type]] * n_atoms / 6.022e23
                           for atom_type, n_atoms in atomic_system.items()]) # [g]
        box_volume = (mass_system / density * 1e24) # [Angstrom^3]
        box_length = box_volume ** (1/3) # [Angstrom]

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
            # Refill 'available_indices' with the indices that are not occupied after a full sweep over all grid points.
            available_indices = set(map(tuple, all_indices.tolist())).difference(occupied_indices)
            total_tries += 1
            # This increase of 'max_iter' speeds the code because we are now trying harder to fit the remaining atoms
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
        occupied_neighbors_coords = neighbors_coords[np.any(neighbors_coords, axis=1)] # Empty grids have coords (0,0,0)

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


def atomic_system_builder(atom_charges, target_num_atoms, min_distance, box_length, max_tries=25, scale_coords=False):
    """Builds the atomic system

    Args:
        atom_charges (dict): A dictionary where the keys are the atom types in the system and the corresponding values
                             the charge of each atom type.
        target_num_atoms (int): Targeted total number of atoms in the system.
        min_distance (float): Minimum absolute distance between any two atoms [Angstroms].
        box_length (float): Length of the box [Angstroms]
        max_tries (int): Maximum number of cycles in the atom placing algorithm
        scale_coords (bool): Determines whether to scale the coordinates by the box length or not.

    Returns:
        (ase.Atoms): ASE atoms object representing a random configuration of the atomic system.

    """
    atomic_system = create_atomic_system(atom_charges, target_num_atoms)
    coords = construct_simulation_box(atomic_system, min_distance, box_length, max_tries=max_tries, scale_coords=scale_coords)

    atoms_type_list = []
    for k, v in atomic_system.items():
        atoms_type_list.extend([k] * v)

    return ase.Atoms(atoms_type_list, coords, pbc=True, cell=np.diag([box_length]*3))


@python_app(executors=['alf_sampler_executor'])
def atomic_system_task(moleculeid, atom_charges, target_num_atoms, min_distance, box_length_range):
    """Atomic system task that will be fed to the sampler.

    Args:
        moleculeid (str): Unique identifier of the atomic system in the database.
        atom_charges (dict): A dictionary where the keys are the atom types in the system and the corresponding 
                             values are the charge of each atom type.
        target_num_atoms (int): Targeted total number of atoms in the system.
        min_distance (float): Minimum absolute distance between any two atoms [Angstroms].
        box_length_range (list): A list containing the minimum and maximum simulation box length [Angstroms].

    Returns:
        (MoleculesObject): A MoleculesObject representing the system.

    """
    rng = np.random.default_rng()
    box_length = rng.uniform(min(box_length_range), max(box_length_range))

    ase_atoms = atomic_system_builder(atom_charges, target_num_atoms, min_distance, box_length)

    return MoleculesObject(ase_atoms, moleculeid)


@python_app(executors=['alf_sampler_executor'])
def TiAl_builder_task(moleculeid, target_num_atoms, min_distance, box_length_range):
    """Atomic system task that will be fed to the sampler.

    Args:
        moleculeid (str): Unique identifier of the atomic system in the database.
        target_num_atoms (int): Targeted total number of atoms in the system.
        min_distance (float): Minimum absolute distance between any two atoms [Angstroms].
        box_length_range (list): A list containing the minimum and maximum simulation box length [Angstroms].

    Returns:
        (MoleculesObject): A MoleculesObject representing the system.

    """
    box_length = np.random.uniform(min(box_length_range), max(box_length_range))
    num_Ti = np.random.randint(0, target_num_atoms+1)

    atomic_system = {'Ti': num_Ti, 'Al': target_num_atoms - num_Ti}
    coords = construct_simulation_box(atomic_system, min_distance, box_length, max_tries=25, scale_coords=False)

    atoms_type_list = []
    for k, v in atomic_system.items():
        atoms_type_list.extend([k] * v)

    ase_atoms = ase.Atoms(atoms_type_list, coords, pbc=True, cell=np.diag([box_length] * 3))

    return MoleculesObject(ase_atoms, moleculeid)
