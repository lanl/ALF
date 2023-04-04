import os
import json
import pickle

import parsl
from parsl import python_app, bash_app

from importlib import import_module
import os
import ase
from ase import Atoms
from ase.io import vasp as vasp_io
   
@python_app(executors=['alf_QM_executor'])
def ase_calculator_task(input_system,configuration_list,directory,properties=['energy','forces']):
    
    import glob
    
    from ase.calculators.vasp import Vasp as calc_class
    command = configuration_list['QM_run_command']
    #Define the calculator
    calc = calc_class(directory=directory, command=command, **configuration_list['input_list'][0])
    
    #Run the calculation
    calc.calculate(atoms=input_system[1], properties=properties)
    #atoms.calc(properties=properties)
    for curF in glob.glob(directory+'/WAVECAR*'):
        os.remove(curF)
    
    calc2 = calc_class(directory=directory, command=command, **configuration_list['input_list'][1])
    
    calc2.calculate(atoms=input_system[1], properties=properties)
    
    input_system[2] = calc2.results
    
    convergedre = re.compile('aborting loop because EDIFF is reached')
    txt = open(directory + '/OUTCAR','r').read()
    if len(convergedre.findall(txt)) == 1:
        input_system[2]['converged'] = True
    else:
        input_system[2]['converged'] = False
        
    return(input_system)
