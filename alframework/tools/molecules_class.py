from ase import Atoms
import numpy as np
import warnings
# from molecules_class import MoleculesObject
# assert isinstance(molecule_object, MoleculesObject), 'molecule_object must be an instance of MoleculesObject'

class MoleculesObject:
    """Class to create molecule objects"""
    def __init__(self, atoms, moleculeid):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object defining the chemical system of interest.
            moleculeid (str): Unique identifier of the system in the database.
        """
        assert isinstance(atoms, Atoms), 'The parameter atoms must be an ase.Atoms object'
        assert isinstance(moleculeid, str), 'The molecule id key must be a string.'

        self.atoms = atoms
        self._moleculeid = moleculeid
        self.converged = None # Tells whether the QM calculation converged
        self.qm_results = {} # Dict to store the QM calculation results
        self.metadata = {} # General metadata

    def __len__(self):
        """Length method, returns the number of atoms in the system."""
        return len(self.atoms)

    def __str__(self):
        """String method"""
        if self.atoms is not None:
            return f'System: {self.atoms.get_chemical_formula()}, Cell: {self.atoms.get_cell()}, ' \
                   f'QM results: {self.qm_results}, Converged: {self.converged}'
        else:
            return f'System: None, QM results: {self.qm_results}, Converged: {self.converged}'

    def __repr__(self):
        """Representation method"""
        return self._moleculeid

    def __eq__(self, other):
        """Equality method, objects are equal if they have the same atoms and coordinates"""
        assert isinstance(other, MoleculesObject), 'The given object must be an instance of MoleculesObject'

        # Checking if the two systems have the same chemical composition
        atomic_nums1 = np.sort(self.atoms.get_atomic_numbers())
        atomic_nums2 = np.sort(other.atoms.get_atomic_numbers())
        same_atoms = np.array_equal(atomic_nums1, atomic_nums2)

        # Checking if the atoms have equivalent coordinates
        coords1 = self.atoms.get_positions()
        coords2 = self.atoms.get_positions()
        same_coords = np.allclose(coords1, coords2, atol=0, rtol=1e-6)

        if same_atoms and same_coords:
            return True
        else:
            return False

    # def __hash__(self):
    #     """Hash method"""
    #     return hash(self._moleculeid)

    def __getitem__(self, item):
        """Get item method, allows indexing of the molecule object to resemble the old implementation"""
        warnings.warn('Warning: molecule object indexing is being deprecated. Replace with get methods.',
                      DeprecationWarning)
        if item == 0:
            return self._moleculeid
        elif item == 1:
            return self.atoms
        elif item == 2:
            return self.qm_results
        else:
            raise IndexError('Index must be either 0, 1, or 2.')

    def get_moleculeid(self):
        """Get the moleculeid of the system"""
        return self._moleculeid

    def get_atoms(self):
        """Get the ase.Atoms object"""
        return self.atoms

    def update_atoms(self, new_atoms):
        """Update the atoms attribute

        This is necessary because when running MLMD we may find a new configuration of the system that the NNs
        disagree. Thus, we want to store this configuration to run QM. When we update to None it means that
        there was no configuration during the MLMD in which the ensemble of NNs disagreed.

        Args:
              new_atoms (ase.Atoms): Atoms object representing the new configuration or system.

        """
        assert isinstance(new_atoms, Atoms) or new_atoms is None, \
            'Parameter must be either an ase Atoms object or None'
        self.atoms = new_atoms

    def append_atoms(self, new_atoms):
        """Append atoms to the existing atomic system.

        Args:
            new_atoms (ase.Atoms): ase.Atoms object to append to the current system.

        """
        self.atoms.extend(new_atoms)

    def get_results(self):
        """Get the QM results"""
        return self.qm_results

    def store_results(self, results, replace=True):
        """Stores the QM results

        Args:
            results (dict): Dictionary containing the results of the QM calculations that we wish to store
            replace (bool): If True completely update self.qm_results including values of common keys, but if False only
                            the keys in 'results' that doesn't exist in self.qm_results are added.

        """
        assert isinstance(results, dict), 'The results must be stored as a dictionary'

        if replace:
            self.qm_results.update(results)
        else:
            new_keys = set(results).difference(self.qm_results)
            self.qm_results.update({k: results[k] for k in new_keys})

    def check_stored_results(self):
        """Checks if there is any QM result stored"""
        if self.qm_results:
            return True
        else:
            return False

    def get_metadata(self):
        """Get metadata"""
        return self.metadata

    def update_metadata(self, new_metadata, replace=True):
        """Updates the object metadata

        Args:
            new_metadata (dict): New metadata to be added.
            replace (bool): If True completely update self.metadata including values of common keys, but if False only
                            the keys in 'new_metadata' that doesn't exist in self.metadata are added.

        """
        assert isinstance(new_metadata, dict), 'The new metadata must be a dict'

        if replace:
            self.metadata.update(new_metadata)
        else:
            new_keys = set(new_metadata).difference(self.metadata)
            self.metadata.update({k: new_metadata[k] for k in new_keys})

    def check_convergence(self):
        """True if converged, False otherwise, and None if no QM calculation was performed yet"""
        return self.converged

    def set_converged_flag(self, convergence_flag):
        """Set the convergence flag.

        Args:
            convergence_flag (bool): Bool indicating whether the calculation converged or not.
        """
        assert isinstance(convergence_flag, bool), 'The convergence flag must be either True or False'
        self.converged = convergence_flag

    def system_signature(self):
        """Returns the system signature as a string.

        #TODO: Compute system moments of inertia and rotate the system so that the highest order moment points along
               the x-axis, midde moment point along y-axis and last moment points along z-axis.
        """
        atomic_numbers = self.atoms.get_atomic_numbers()
        coords = self.atoms.get_positions()
        atoms_list = self.atoms.get_chemical_symbols()
        idx_sorted = np.argsort(atomic_numbers)

        signature_cell = "".join([f'{el:.6f}-'for el in self.atoms.get_cell().flatten()])
        signature_atoms = "".join([atoms_list[i] + f'{coords[i,0]:.6f}{coords[i,1]:.6f}{coords[i,2]:.6f}'
                                   for i in idx_sorted])

        return signature_cell + signature_atoms


def compare_molecule_objects(system1, system2):
    """Check if two MoleculesObject instances have the same chemical composition

    Args:
        system1 (MoleculesObject): First object to be compared
        system2 (MoleculesObject): Second object to be compared
    """
    assert isinstance(system1, MoleculesObject) and isinstance(system2, MoleculesObject), \
        'The given objects must be an instance of MoleculesObject'

    atomic_nums1 = np.sort(system1.atoms.get_atomic_numbers())
    atomic_nums2 = np.sort(system2.atoms.get_atomic_numbers())

    return np.array_equal(atomic_nums1, atomic_nums2)


########################################################################################################################
atoms1 = Atoms('H20')
atoms2 = Atoms('F2Li4Na3')
atoms3 = Atoms('F2Na3Li4')

sys1 = MoleculesObject(atoms1, 'abc')
sys2 = MoleculesObject(atoms2, 'def')
sys3 = MoleculesObject(atoms3, 'ghi')

# # When print a molecule object it tell us the molecule id, atoms object, qm results and convergence flag.
# print(sys2) # Moleculeid: def, Atoms(symbols='F2Li4Na3', pbc=False), QM results: {}, converged: []
#

# # Length of a molecule object is its number of atoms, allowed because __len__ is implemented.
# print(len(sys3)) # 9
#
# # Two molecule objects are equal when they have exactly they are exactly the same chemical system. They can still
# # differ by their configurations, but the elements and the number of each element must be the same for them to be
# # considered the same system.
# print(sys2 == sys3) # True
# print(sys1 == sys2) # False
