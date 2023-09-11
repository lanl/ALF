import numpy as np
import sys

from ase.calculators.calculator import Calculator,all_changes
from ase.calculators.mixing import Mixer

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
        Calculator.__init__(self)
        self.r_start=r_start
        self.force=force
        self.origin=np.array(origin)
        self.mass_weighted = mass_weighted
        self.zero_properties = zero_properties
        if 'energy' in self.zero_properties:
            self.zero_properties.remove('energy')
        if 'forces' in self.zero_properties:
            self.zero_properties.remove('potential_energy')
        if 'forces' in self.zero_properties:
            self.zero_properties.remove('forces')
        self.implemented_properties = ['energy', 'forces', 'potential_energy'] + zero_properties
    
    def calculate(self, atoms=None, properties=['energy'],system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
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
    def __init__(self,models,well_params=None,debug_print=False):
        Calculator.__init__(self)
        self.debug_print = debug_print
        self.N_models = len(models)
        self.models = models
        self.weights = [1/self.N_models for i in range(self.N_models)]
        self.calculator_properties = list(set.intersection(*(set(calc.implemented_properties) for calc in self.models)))

        self.use_bias = False

        if debug_print:
            print("calculator properties:" + str(self.calculator_properties))
            sys.stdout.flush()
        #self.implemented_properties = self.calculator_properties.copy()
        
        if not(well_params is None):
            if 'zero_properties' in well_params:
                well_params['zero_properties'] = list(set.union(set(well_params['zero_properties']),set(self.calculator_properties)))
            else: 
                well_params['zero_properties'] = self.calculator_properties.copy()
            self.models.append(Well_Potential(**well_params))
            self.weights.append(1.0)
        
        self.mixer = Mixer(self.models,np.array(self.weights))
        self.implemented_properties = self.mixer.implemented_properties.copy()
        
            
        if 'energy' in self.implemented_properties:
            self.implemented_properties.append('energy_stdev')
        if 'forces' in self.implemented_properties:
            self.implemented_properties.append('forces_stdev_mean')
            self.implemented_properties.append('forces_stdev_max')
    def calculate(self,atoms,properties,system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        add_properties = properties.copy()
        if 'energy_stdev' in add_properties:
            add_properties.append("energy")
        if 'forces_stdev_max' in add_properties or 'forces_stdev_mean' in add_properties:
            add_properties.append("forces")
        run_properties = list(set.intersection(set(self.calculator_properties),set(add_properties)))
        self.atoms = atoms.copy()
        self.results = self.mixer.get_properties(run_properties,atoms)
        if 'energy_stdev' in properties or self.use_bias:
            self.results['energy_stdev'] = np.std(self.results['energy_contributions'][:self.N_models])

        if 'forces_stdev_mean' in properties or 'forces_stdev_max' in properties:
            force_stdev = np.std(np.array(self.results['forces_contributions'][:self.N_models]),axis=0)
            self.results['forces_stdev_mean'] = np.mean(np.abs(force_stdev))
            self.results['forces_stdev_max'] = np.max(np.abs(force_stdev))

        if self.use_bias: # uncertainty driven dynamics (UDD) 
            self.results['E_en_bias']= self.E_en_bias_weight * self.results['energy_stdev']

            energy_stdev_dx = (1/np.sqrt(self.N_models)) * \
                            np.power( np.sum(np.power( (self.results['potential_energy_contributions'][:self.N_models] - self.results['potential_energy']), 2)), -0.5) * \
                            np.sum(np.expand_dims(np.expand_dims(self.results['potential_energy_contributions'] - self.results['potential_energy'], axis=1), axis=1 ) * \
                                                                (self.results['forces_contributions']           - self.results['forces']), axis=0)

            self.results['F_en_bias'] = -self.E_en_bias_weight * energy_stdev_dx
            self.results['energy'] += self.results['E_en_bias']
            self.results['forces'] += self.results['F_en_bias']


