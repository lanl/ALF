"""Labeling with ASE's VASP calculator."""
import re
from parsl import python_app
from alframework.tools.label_sources import LABEL_SOURCE
from alframework.tools.task_timeline import timed
from alframework.tools.molecules import MoleculesObject
from .generic import ase_label_source

@python_app(executors=['alf_QM_executor'])
@timed('QM')
def VASP_ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """ASE VASP calculator based on the parameters in qm_config.json.

    Args:
        molecule_object (MoleculesObject): An object from the class MoleculesObject.
        QM_config (dict): The QM configuration: ``QM_run_command`` and the VASP calculator's ``input`` settings.
        QM_scratch_dir (str): Directory under which each molecule gets its own calculation directory.
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
        molecule_object.update_metadata({LABEL_SOURCE: ase_label_source(
            'VASP_ase_calculator_task', 'ase.calculators.vasp.Vasp', QM_config['input'])})
        molecule_object.set_converged_flag(True)
    else:
        molecule_object.set_converged_flag(False)

    return molecule_object
