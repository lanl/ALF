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

from alframework.tools.tools import load_module_from_config
from alframework.tools.molecules_class import MoleculesObject


def fresh_scratch_directory(directory):
    """Create an empty scratch directory, moving aside one left by an earlier attempt at this structure.

    A restarted finite bootstrap rebuilds the IDs it didn't finish, and a crashed run may have
    started their calculations. Their files are kept, beside the new directory.
    """
    if os.path.isdir(directory):
        attempt = 0
        while os.path.exists(f'{directory}.previous-{attempt}'):
            attempt += 1
        os.rename(directory, f'{directory}.previous-{attempt}')
        print(f'Warning: {directory} already existed, from an earlier attempt at this structure; '
              f'moved it to {directory}.previous-{attempt}.', flush=True)
    os.makedirs(directory)


@python_app(executors=['alf_QM_executor'])
def ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """Creates an ASE calculator based on the parameters in qm_config.json.

    Args:
        molecule_object (MoleculesObject): An object from the class MoleculesObject.
        QM_config (dict): json file containing the parameters needed to run the QM code.
        QM_scratch_dir (str): Directory to store the QM calculations.
        properties_list (dict): Dict whose keys are the properties that we would like ASE to extract from the QM
                                calculation output file.
    Returns:
        (MoleculesObject): A MoleculesObject containing the results of the QM calculations.

    """
    assert isinstance(molecule_object, MoleculesObject), 'molecule_object must be an instance of MoleculesObject'

    directory = QM_scratch_dir + '/' + molecule_object.get_moleculeid()
    command = QM_config['QM_run_command']
    properties = list(properties_list.keys())
    atoms = molecule_object.get_atoms()

    fresh_scratch_directory(directory)
    # This isn't the prettist thing
    # ase calculators are classes residing within a similarly named module
    # Pass in the full class path, This code seperates the two
    calc_class = load_module_from_config(QM_config, 'ASE_calculator')

    # Define the calculator
    calc = calc_class(atoms=atoms, directory=directory, command=command, **QM_config)
    
    # Run the calculation
    calc.calculate(atoms=atoms, properties=properties)

    # Stores results
    molecule_object.store_results(calc.results)
    molecule_object.set_converged_flag(calc.converged)
    
    return molecule_object

   
@python_app(executors=['alf_QM_executor'])
def VASP_ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """ASE VASP calculator based on the parameters in qm_config.json.

    Args:
        molecule_object (MoleculesObject): An object from the class MoleculesObject.
        QM_config (str): Relative path of the QM_config.json file that contains the parameters needed to run VASP.
        QM_scratch_dir (str): Relative path of the directory to store the QM calculations.
        properties_list (dict): Dict whose keys are the properties that we would like ASE to extract from the OUTCAR
                                file and that will be stored in the database.

    Returns:
        (MoleculesObject): A MoleculesObject containing the results of the QM calculations.

    """
    from ase.calculators.vasp import Vasp as calc_class

    assert isinstance(molecule_object, MoleculesObject), 'molecule_object must be an instance of MoleculesObject'

    command = QM_config['QM_run_command']
    directory = QM_scratch_dir + '/' + molecule_object.get_moleculeid()
    properties = list(properties_list.keys())

    # Define the calculator
    calc_input = QM_config['input'].copy()
    if 'kpoints' in molecule_object.metadata.keys(): # I don't think this is necessary, kpts should be defined in QM_config.
        calc_input['kpts'] = molecule_object.metadata['kpoints']
    calc = calc_class(directory=directory, command=command, **calc_input)
    
    # Run the calculation
    calc.calculate(atoms=molecule_object.get_atoms(), properties=properties)

    molecule_object.store_results(calc.results)

    convergedre = re.compile('aborting loop because EDIFF is reached')
    txt = open(directory + '/OUTCAR','r').read()
    if len(convergedre.findall(txt)) == 1:
        molecule_object.set_converged_flag(True)
    else:
        molecule_object.set_converged_flag(False)

    return molecule_object

