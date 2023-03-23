from parsl import python_app, bash_app
from importlib import import_module
import os
import ase
from ase import Atoms
from alframework.tools.tools import system_checker

@python_app(executors=['alf_QM_executor'])
def ase_calculator_task(input_system,configuration,directory,properties=['energy','forces']):
    system_checker(input_system)
    molecule_id = input_system[0]['moleculeid']
    atoms = input_system[1]
    
    if os.path.isdir(directory):
        raise RuntimeError('Scratch directory exists: ' + directory)
    else:
       os.makedirs(directory)
    #This isn't the prettist thing
    #ase calculators are classes residing within a similarly named module
    #Pass in the full class path, This code seperates the two
    module_string = '.'.join(configuration['ASE_calculator'].split('.')[:-1])
    class_string = configuration['ASE_calculator'].split('.')[-1]
    calc_class = getattr(import_module(module_string),class_string)
    
    
    #Define the calculator
    calc = calc_class(atoms=atoms, directory=directory, command=command, **configuration)
    
    #Run the calculation
    calc.calculate(atoms=atoms, properties=properties)
    
    return_results = calc.results
    return_results['converged'] = calc.converged
    
    return([input_system[0],input_system[1],return_results])

