import random
import numpy as np
import ase
from tqdm import tqdm
from collections import Counter
from itertools import product
from ase.data import atomic_masses, atomic_numbers

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

    # Keeping only the elements whose frequency is greater than zero
    atomic_system = {atom_type: atomic_system[atom_type]
                     for atom_type in atom_charges.keys() if atomic_system[atom_type] > 0}

    # "Sorting" (dicts are unordered) according to atomic number
    # atom_types.sort(key=lambda at: atomic_numbers[at])
    # atomic_system = {at: atomic_system[at] for at in atom_types}

    return atomic_system


def to_mic(r_ij, box_length):
    """Impose minimum image convention (MIC) on the array containing the distances between atoms.

    Args:
        r_ij (ndarray): A (n_atoms, 3) matrix r_ij = r_i - r, or a (n_atoms, n_atoms, 3) array containing all r_i - r.
        box_length (float): Length of cubic cell.

    Returns:
        (ndarray): Distances under MIC
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
                           for atom_type, n_atoms in atomic_system.items()])  # [g]
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


def atomic_system_builder(atom_charges, target_num_atoms, min_distance, box_length_range, max_tries=25, scale_coords=False):
    """Builds the atomic system

    Args:
        atom_charges (dict): A dictionary where the keys are the atom types in the system and the corresponding values
                             the charge of each atom type.
        target_num_atoms (int): Targeted total number of atoms in the system.
        min_distance (float): Minimum absolute distance between any two atoms in Angstroms.
        box_length_range (list): A list containing the minimum and maximum simulation box length [L_min, L_max].
        max_tries (int): Maximum number of cycles in the atom placing algorithm
        scale_coords (bool): Determines whether to scale the coordinates by the box length or not.

    Returns:
        (ase.Atoms): ASE atoms object representing a random configuration of the atomic system.

    """
    rng = np.random.default_rng()
    box_length = rng.uniform(min(box_length_range), max(box_length_range))
    atomic_system = create_atomic_system(atom_charges, target_num_atoms)
    coords = construct_simulation_box(atomic_system, min_distance, box_length, max_tries=max_tries, scale_coords=scale_coords)

    atoms_type_list = []
    for k, v in atomic_system.items():
        atoms_type_list.extend([k] * v)

    return ase.Atoms(atoms_type_list, coords, pbc=True, cell=np.diag([box_length]*3))


#----------------------------------------------------------------------------------------------------------------------#
## Tests
def test_atomic_system(charges, target_num_ats, iterations=10000):
    freq_natoms = {}
    total_charges = []
    freq_type = Counter({k: 0 for k in charges.keys()})

    for _ in tqdm(range(iterations)):
        at_sys = Counter(create_atomic_system(charges, target_num_ats))
        freq_type += at_sys
        n_atoms = sum(at_sys.values())

        if freq_natoms.get(n_atoms) is None:
            freq_natoms[n_atoms] = 1
        else:
            freq_natoms[n_atoms] += 1

        total_charges.append(sum([at_sys[at] * charges[at] for at in charges.keys()]))

    total = sum(freq_type.values())
    prob = {k: round(v/total, 3) for k,v in freq_type.items()}

    if np.all(np.array(total_charges) == 0):
        print('All systems were charge neutral')
    else:
        print('Error, some systems were not charge neutral')

    print(f'Number of atoms in the atomic system: {freq_natoms}')
    print(f'Frequency of each atomic species: {prob}')


def test_distances(coords, box_length):
    d_ijk = to_mic(coords[:,np.newaxis,:] - coords[np.newaxis,:,:], box_length)
    dists = np.linalg.norm(d_ijk, axis=2)
    min_distances = np.min(dists[dists != 0].reshape(d_ijk.shape[0], d_ijk.shape[0]-1), axis=-1)
    print(f'The minimum distance for each atom is {min_distances}')
    print(f'The minimum distance between any two atoms assuming PBC is {min(min_distances):.6f}')


## Testing the charge neutrality of the system and if the total number of atoms in it is close to what we want.
# test_atomic_system({'F': -1, 'Li': 1, 'Na': 1, 'K': 1, 'Cl': -2, 'Mg': 2, 'Be': 2}, 100)
# print(create_atomic_system({'F': -1, 'Li': 1, 'Na': 1, 'K': 1, 'Cl': -2, 'Mg': 2, 'Be': 2}, 100))
# quit()


## Testing the minimum distance between any two atoms considering PBC to check if it respects the 'min_distance'.
# L = 10
# coords = construct_simulation_box(create_atomic_system({'F': -1, 'Li': 1, 'Na': 1, 'K': 1, 'Cl': -2, 'Mg': 2, 'Be': 2}, 100),
#                                   1.5, box_length=L, scale_coords=False, max_tries=20, max_iter=30)
# test_distances(coords, L)
# quit()


## Testing algorithm's stability for very small box sizes (high densities)
# L = 8.5
# for i in tqdm(range(5000)):
#     coords = construct_simulation_box(create_atomic_system({'F': -1, 'Li': 1, 'Na': 1, 'K': 1, 'Cl': -2, 'Mg': 2, 'Be': 2}, 100),
#                                         1.5, box_length=L, scale_coords=False, max_tries=20, max_iter=30)
# quit()


## Checking the return from the builder
# atoms = atomic_system_builder({'F': -1, 'Li': 1, 'Na': 1, 'K': 1, 'Cl': -2, 'Mg': 2, 'Be': 2}, 100, 1.5, [8.5,13])
# print(atoms.get_atomic_numbers())
# print(atoms.cell.cellpar())
# print(atoms.get_all_distances()[atoms.get_all_distances() != 0].min())
# print(atoms.get_positions().tolist())
# print(atoms.get_chemical_symbols())
# quit()


## Testing the number of atoms inside the atomic environment of a given atom and its minimum distance
# N = 100
# min_dist = np.empty(N)
# n_neigh = np.empty(N)
# L = 12
# for i in range(N):
#     coords = construct_simulation_box({'F': 17, 'Li': 14, 'Na': 9, 'K': 12, 'Cl': 28, 'Mg': 9, 'Be': 10}, 1.5, box_length=L, scale_coords=False)
#     d_1 = np.linalg.norm(to_mic(coords[0] - coords[1:], L), axis=1) # Using the first atom to find the distances
#     neigh = d_1[d_1 <= 7]
#     min_dist[i] = neigh.min()
#     n_neigh[i] = neigh.shape[0]
#
# hist_dist = np.histogram(min_dist, np.arange(1.5, 3.1, 0.1))
# hist_neigh = np.histogram(n_neigh, np.arange(1.5, 3.1, 0.1))
#
# print(np.sort(min_dist))
# print(np.sort(n_neigh))
# print(min_dist[min_dist>2].size)
# print(f'Minimum avg distance in the AEV: {np.mean(min_dist):.2f}, avg number of atoms in the AEV: {np.mean(n_neigh):.2f}')
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.hist(min_dist, bins=hist_dist[1], rwidth=0.95, color='navy')
#
# for i, rect in enumerate(ax.patches):
#     height = rect.get_height()
#     ax.annotate(f'{int(hist_dist[0][i])}', xy=(rect.get_x()+rect.get_width()/2, height),
#                 xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=15, color='orangered')
# plt.show()


