import numpy as np

from ase.calculators.calculator import Calculator,all_changes

class Well_Potential(Calculator):
    """
    This calculator applies a restoring force to constrain a series of 
    This is useful for cluster simulations
    r_start: distance from origin where potential starts, in Angstrom
    force: magnitudue of resotring force in eV/Angstrom
    origin: The center point of the spherical well potential
    zero_properties: In order to mix this calculator with other calculators, it must have the same implemented  properties list. this calculator will spoof those properties and return np.array(0) which should not change other propertys mixed with the mixer.
    mass_weighted: Flag to configure if forces applied to atoms should be mass weighted. Default is true to make each atom experience the same acceleration (keeping molecules together)
    #Todo: Add cos()**2 switching function.
    #TODO: Add drag term when in the potential to prevent orbits. 
    """
    def __init__(self,r_start,force,origin=[0,0,0],zero_properties=[],mass_weighted=True):
        super().__init__()
        self.r_start=r_start
        self.force=force
        self.origin=np.array(origin)
        self.mass_weighted = mass_weighted
        self.zero_properties = zero_properties
        if 'energy' in self.zero_properties:
            self.zero_properties.remove('energy')
        if 'forces' in self.zero_properties:
            self.zero_properties.remove('forces')
        self.implemented_properties = ['energy', 'forces'] + zero_properties
    
    def calculate(self, atoms=None, properties=['energy'],system_changes=all_changes):
        super().calculate(self, atoms, properties, system_changes)
        self.atoms = atoms.copy()
        relative_positions = self.atoms.get_positions() - self.origin[np.newaxis]
        
        if self.mass_weighted:
            mass_vector = self.atoms.get_masses()
        else:
            mass_vector = np.ones(len(self.atoms))
            
        depth = np.linalg.norm(relative_positions,axis=1) - self.r_start
        in_potential = depth>0
        in_hole = depth<0
        depth[depth<0] = 0
        unit_vectors = (relative_positions.T/(1e-4*in_hole+np.linalg.norm(relative_positions,axis=1))).T #Unit vectors not perfectly normalized for atoms in the flat piece of well
        self.results['energy'] = np.sum(mass_vector*depth*self.force)
        if 'forces' in properties:
            self.results['forces'] = -1*(unit_vectors.T*in_potential*mass_vector).T
        for current_property in properties:
            if current_property in self.zero_properties:
                self.results[current_property] = np.array(0)
        
class MLMD_calculator(Calculator):
    """
    This is a calculator that enables MLMD sampling with an ensemble of ML potentials 
    models: a list of ASE calculators that are joined to perform MD. 
    well_params: Inputs for the well potential above, which enables constrained MD simulations
    """
    def __init__(self,models,well_params=None):
        super().__init__()
        self.N_models = len(models)
        self.models = models
        self.weights = [1/self.N_models for i in range(self.N_models)]
        self.calculator_properties = list(set.intersection(*(set(calc.implemented_properties) for calc in self.models)))
        self.implemented_properties = self.calculator_properties.copy()
        if 'energy' in self.implemented_properties:
            self.implemented_properties.append('energy_stdev')
        if 'forces' in self.implemented_properties:
            self.implemented_properties.append('forces_stdev_mean')
            self.implemented_properties.append('forces_stdev_max')
        
        if not(well_params is None):
            if 'zero_properties' in well_params:
                well_params['zero_properties'] = list(set.intersection(set(well_params['zero_properties']),set(self.implemented_properties)))
            else: 
                well_params['zero_properties'] = self.implemented_properties
            self.models.append[Well_Potential(**well_params)]
            self.weights.append(1.0)
        
        self.mixer = Mixer(self.models,np.array(self.weights))
    def calculate(self,atoms,properties,system_changes=all_changes):
        super().calculate(self, atoms, properties, system_changes)
        run_properties = set.intersection(set(properties),set(self.calculator_properties))
        self.atoms = atoms.copy()
        self.results = self.mixer.get_properties(run_properties,atoms)
        if 'energy_stdev' in properties:
            self.results['energy_stdev'] = np.stdev(self.results['energy_contributions'][:self.N_models])
        if 'forces_stdev' in properties or 'forces_stdev_max' in properties:
            force_stdev = np.std(np.array(self.results['forces'][:self.N_models]),axis=0)
            self.results['forces_stdev_mean'] = np.mean(np.abs(force_stdev))
            self.results['forces_stdev_max'] = np.max(np.abs(force_stdev))
        