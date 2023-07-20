from ase import Atoms
import numpy as np

class MoleculeObject:
    """Class to create molecule objects"""
    def __init__(self, atoms):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object defining the system.
        """
        assert isinstance(atoms, Atoms), 'The parameter atoms is not an ase.Atoms object'

        self.atoms = atoms
        self.moleculeid = None # Unique identifier of the system
        self.converged = [] # Tells if the QM calculation on the system converged
        self.qm_results = {} # Dict to store the QM calculation results
        self.metadata = {} # General metadata of the system

    def __len__(self):
        """Length method, returns the number of atoms in the system."""
        return len(self.atoms)

    def __str__(self):
        """String method"""
        return f'Moleculeid: {self.moleculeid}, {self.atoms}, QM results: {self.qm_results}, converged: {self.converged}'

    def __repr__(self):
        """Representation method"""
        return self.moleculeid

    def __eq__(self, other):
        """Equality method, two objects are considered equal when they have exactly the same elements"""
        assert isinstance(other, MoleculeObject), 'The given object must be an instance of MoleculeObject'
        atomic_nums1 = np.sort(self.atoms.get_atomic_numbers())
        atomic_nums2 = np.sort(other.atoms.get_atomic_numbers())
        return np.array_equal(atomic_nums1, atomic_nums2)

    def __hash__(self):
        """Hash method"""
        return hash(self.moleculeid)

    def get_moleculeid(self):
        """Get the moleculeid of the system"""
        return self.moleculeid

    def set_moleculeid(self, moleculeid):
        """Set a molecule id.

        Args:
            moleculeid (str): String that uniquely identifies the system in the database.

        """
        assert isinstance(moleculeid, str), 'The molecule id key must be a string'

        if self.moleculeid is not None:
            raise Exception('The system already has a molecule id')
        else:
            self.moleculeid = moleculeid

    def get_results(self):
        """Get the QM results"""
        return self.qm_results

    def store_results(self, results, replace):
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

    def set_metadata(self, metadata):
        """Set the object metadata

        Args:
            metadata (dict): Dictionary that will be the molecule object metadata.

        """
        assert isinstance(metadata, dict), 'Metadata must be a dict'
        if self.metadata:
            raise Exception('This object already has a metadata, please use update_metadata() method instead')
        self.metadata = metadata

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
        """True if converged, False otherwise, and None if no QM calcultion was performed yet"""
        return self.converged

    def set_convergence_flag(self, convergence_flag):
        """Set the convergence flag.

        Args:
            convergence_flag (bool): Bool indicating whether the calculation converged or not.
        """
        self.converged = convergence_flag



########################################################################################################################
atoms1 = Atoms('H20')
atoms2 = Atoms('F2Li4Na3')
atoms3 = Atoms('F2Na3Li4')

sys1 = MoleculeObject(atoms1)
sys2 = MoleculeObject(atoms2)
sys3 = MoleculeObject(atoms3)

sys1.set_moleculeid('abc')
sys2.set_moleculeid('def')
sys3.set_moleculeid('ghi')

# When print a molecule object it tell us the molecule id, atoms object, qm results and convergence flag.
print(sys2) # Moleculeid: def, Atoms(symbols='F2Li4Na3', pbc=False), QM results: {}, converged: []

# Dict using MoleculeObjects can be created because __hash__ method is implemented. The keys will be the 'moleculeid'.
d = {sys1: 1, sys2: 3, sys3: 4}
print(d) # {abc: 1, def: 3, ghi: 4}

# Length of a molecule object is its number of atoms, allowed because __len__ is implemented.
print(len(sys3)) # 9

# Two molecule objects are equal when they have exactly they are exactly the same chemical system. They can still
# differ by their configurations, but the elements and the number of each element must be the same for them to be
# considered the same system.
print(sys2 == sys3) # True
print(sys1 == sys2) # False


