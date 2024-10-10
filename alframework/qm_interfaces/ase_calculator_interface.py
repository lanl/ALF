import os
import json
import pickle
import re

import parsl
from parsl import python_app, bash_app

import os
import ase
from ase import Atoms
from ase.io import vasp as vasp_io
from alframework.tools.tools import system_checker
from alframework.tools.tools import load_module_from_config

@python_app(executors=['alf_QM_executor'])
def ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """
    Run an ASE calculation on a molecule object.

    Parameters:
    molecule_object (list): A list of length three containing a dictionary with the molecule id, an ASE Atoms object, and a dictionary of results from previous calculations.
    QM_config (dict): A dictionary of configuration options for the ASE calculator.
    QM_scratch_dir (str): The directory where ASE should write its scratch files.
    properties_list (list): A list of the properties to be calculated.

    Returns:
    A list of length three containing a dictionary with the molecule id, an ASE Atoms object, and a dictionary of results from the ASE calculation.
    """
    system_checker(molecule_object)
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    properties = list(properties_list)
    molecule_id = molecule_object[0]['moleculeid']
    atoms = molecule_object[1]
    
    if os.path.isdir(directory):
        raise RuntimeError('Scratch directory exists: ' + directory)
    else:
       os.makedirs(directory)
    #This isn't the prettist thing
    #ase calculators are classes residing within a similarly named module
    #Pass in the full class path, This code seperates the two
    calc_class = load_module_from_config(QM_config, 'ASE_calculator')
    
    
    #Define the calculator
    calc = calc_class(atoms=atoms, directory=directory, command=command, **QM_config)
    
    #Run the calculation
    calc.calculate(atoms=atoms, properties=properties)
    
    return_results = calc.results
    return_results['converged'] = calc.converged
    
    return_system = [molecule_object[0],molecule_object[1],return_results]
    system_checker(return_system)
    
    return(return_system)

   
@python_app(executors=['alf_QM_executor'])
def VASP_ase_calculator_task(molecule_object,QM_config,QM_scratch_dir,properties_list):
    
    import glob
    
    from ase.calculators.vasp import Vasp as calc_class
    command = QM_config['QM_run_command']
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    properties = list(properties_list)
    #Define the calculator

    calc_input = QM_config['input'].copy()
    if 'kpoints' in molecule_object[0]:
        calc_input['kpts'] = molecule_object[0]['kpoints']
    calc = calc_class(directory=directory, command=command, **calc_input)
    
    #Run the calculation
    calc.calculate(atoms=molecule_object[1], properties=properties)

    molecule_object[2] = calc.results

    convergedre = re.compile('aborting loop because EDIFF is reached')
    txt = open(directory + '/OUTCAR','r').read()
    if len(convergedre.findall(txt)) == 1:
        molecule_object[2]['converged'] = True
    else:
        molecule_object[2]['converged'] = False

    return(molecule_object)

