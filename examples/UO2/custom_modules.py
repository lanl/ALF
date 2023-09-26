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
   
@python_app(executors=['alf_QM_executor','alf_QM_standby_executor'])
def ase_calculator_task(molecule_object,QM_config,QM_scratch_dir,properties_list):
    
    import glob
    
    from ase.calculators.vasp import Vasp as calc_class
    
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    properties = list(properties_list)
    
    command = QM_config['QM_run_command']
    #Define the calculator
    calc = calc_class(directory=directory, command=command, **QM_config['input_list'][0])
    
    #Run the calculation
    calc.calculate(atoms=molecule_object[1], properties=properties)
    #atoms.calc(properties=properties)
    for curF in glob.glob(directory+'/WAVECAR*'):
        os.remove(curF)
    
    calc2 = calc_class(directory=directory, command=command, **QM_config['input_list'][1])
    
    calc2.calculate(atoms=molecule_object[1], properties=properties)
    
    molecule_object[2] = calc2.results
    
    convergedre = re.compile('aborting loop because EDIFF is reached')
    txt = open(directory + '/OUTCAR','r').read()
    if len(convergedre.findall(txt)) == 1:
        molecule_object[2]['converged'] = True
    else:
        molecule_object[2]['converged'] = False
        
    return(molecule_object)
