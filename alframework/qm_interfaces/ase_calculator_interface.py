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
    """Creates an ASE calculator based on the parameters in qm_config.json.

    Args:
        molecule_object (MoleculeObject): An object from the class MoleculeObject.
        QM_config (str): json file containing the parameters needed to run the QM code.
        QM_scratch_dir (str): Directory to store the QM calculations.
        properties_list (dict): Dict whose keys are the properties that we would like ASE to extract from the QM
                                calculation output file.
    Returns:
        (list): A modified molecule_object that now stores the results of the QM calculations in molecules_object[2].

    """
    system_checker(molecule_object)
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    command = QM_config['QM_run_command']
    properties = list(properties_list.keys())
    atoms = molecule_object[1]
    
    if os.path.isdir(directory):
        raise RuntimeError('Scratch directory exists: ' + directory)
    else:
       os.makedirs(directory)
    # This isn't the prettist thing
    # ase calculators are classes residing within a similarly named module
    # Pass in the full class path, This code seperates the two
    calc_class = load_module_from_config(QM_config, 'ASE_calculator')

    # Define the calculator
    calc = calc_class(atoms=atoms, directory=directory, command=command, **QM_config)
    
    # Run the calculation
    calc.calculate(atoms=atoms, properties=properties)
    
    return_results = calc.results
    return_results['converged'] = calc.converged
    
    return_system = [molecule_object[0], molecule_object[1], return_results]
    system_checker(return_system)
    
    return return_system

   
@python_app(executors=['alf_QM_executor'])
def VASP_ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """ASE VASP calculator based on the parameters in qm_config.json.

    Args:
        molecule_object(list): A list [metadata_dict, ase.Atoms, QM_results_dict] that uniquely represents a system.
                               The first element is a dictionary contaning metadata (one of which must be 'moleculeid'),
                               the second is an ase.Atoms object, and the third an empty dictionary that will
                               store the results of the QM calculation.
        QM_config (str): Relative path of the QM_config.json file that contains the parameters needed to run VASP.
        QM_scratch_dir (str): Relative path of the directory to store the QM calculations.
        properties_list (dict): Dict whose keys are the properties that we would like ASE to extract from the OUTCAR
                                file and that will be stored in the database.

    Returns:
        (list): A modified molecule_object that now stores the results of the QM calculations as a dictionary in
                molecule_object[2].

    """
    from ase.calculators.vasp import Vasp as calc_class

    command = QM_config['QM_run_command']
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    properties = list(properties_list.keys())

    # Define the calculator
    calc_input = QM_config['input'].copy()
    if 'kpoints' in molecule_object[0]:
        calc_input['kpts'] = molecule_object[0]['kpoints']
    calc = calc_class(directory=directory, command=command, **calc_input)
    
    # Run the calculation
    calc.calculate(atoms=molecule_object[1], properties=properties)

    molecule_object[2] = calc.results

    convergedre = re.compile('aborting loop because EDIFF is reached')
    txt = open(directory + '/OUTCAR','r').read()
    if len(convergedre.findall(txt)) == 1:
        molecule_object[2]['converged'] = True
    else:
        molecule_object[2]['converged'] = False

    return molecule_object

