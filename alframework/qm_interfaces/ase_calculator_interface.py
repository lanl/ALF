import os
import json
import pickle
import re

import parsl
from parsl import python_app, bash_app

from importlib import import_module
import os
import ase
from ase import Atoms
from ase.io import vasp as vasp_io
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
    
    return_system = [input_system[0],input_system[1],return_results]
    system_checker(return_system)
    
    return(return_system)

   
@python_app(executors=['alf_QM_executor'])
def VASP_ase_calculator_task(input_system,configuration,directory,properties=['energy','forces']):
    
    import glob
    
    from ase.calculators.vasp import Vasp as calc_class
    command = configuration_list['QM_run_command']
    #Define the calculator

    calc_input = configuration['input'].copy()
    if 'kpoints' in input_system[0]:
        calc_input['kpts'] = input_system[0]['kpoints']
    calc = calc_class(directory=directory, command=command, **calc_input)
    
    #Run the calculation
    calc.calculate(atoms=input_system[1], properties=properties)

    input_system[2] = calc.results

    convergedre = re.compile('aborting loop because EDIFF is reached')
    txt = open(directory + '/OUTCAR','r').read()
    if len(convergedre.findall(txt)) == 1:
        input_system[2]['converged'] = True
    else:
        input_system[2]['converged'] = False

    return(input_system)

