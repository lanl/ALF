import os
import json
import pickle

from alframework.samplers.builders import cfg_loader
from alframework.qm_interfaces.ase_calculator_interface import ase_calculator_task

import parsl
from parsl import python_app, bash_app

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SimpleLauncher

from importlib import import_module
import os
import ase
from ase import Atoms
from ase.io import vasp as vasp_io

#from alframework.parsl_resource_configs.chicoma import config_standard

config_standard = Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'standard',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 50,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=32 --nodes=4 -A w23_ml4chem',

                # Command to be run before starting a worker
                worker_init='module load python/3.8-anaconda-2020.07; source /usr/projects/ml4chem/envs/p38-parsl-ani-load.bash',

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='16:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)

parsl.load(config_standard)

# Load the VASP options file:
with open('./vasp_2step_config.json','r') as input_file:
   VASP_config_list = json.load(input_file)
   
@python_app(executors=['alf_QM_executor'])
def ase_calculator_task(configuration_list,molecule_id,atoms,directory,command,properties=['energy','forces']):
    
    import glob
    
    from ase.calculators.vasp import Vasp as calc_class
    
    #Define the calculator
    calc = calc_class(directory=directory, command=command, **configuration_list[0])
    
    #Run the calculation
    calc.calculate(atoms=atoms, properties=properties)
    #atoms.calc(properties=properties)
    for curF in glob.glob(directory+'/WAVECAR*'):
        os.remove(curF)
    
    calc2 = calc_class(directory=directory, command=command, **configuration_list[1])
    
    calc2.calculate(atoms=atoms, properties=properties)
    
    return_results = calc2.results
    return_results['converged'] = calc.converged
        
    return(molecule_id,atoms,return_results)

atoms = vasp_io.read_vasp('POSCAR')

#configuration,calculator,directory,command,properties=('energy','forces')
task1 = ase_calculator_task(VASP_config_list,'mol_test',atoms,'/usr/projects/ml4chem/testing_vasp/parsl-test1/','module purge; module load PrgEnv-cray; source /usr/projects/icapt/VASP/vasp.6.3.2-aocc-mkl/setenv_chicoma.sh; /usr/bin/srun /usr/projects/icapt/VASP/vasp.6.3.2-aocc-mkl/bin/vasp_std > vasp_out.txt 2>&1',properties=['energy', 'free_energy', 'forces'])
task2 = ase_calculator_task(VASP_config_list,'mol_test',atoms,'/usr/projects/ml4chem/testing_vasp/parsl-test2/','module purge; module load PrgEnv-cray; source /usr/projects/icapt/VASP/vasp.6.3.2-aocc-mkl/setenv_chicoma.sh; /usr/bin/srun /usr/projects/icapt/VASP/vasp.6.3.2-aocc-mkl/bin/vasp_std > vasp_out.txt 2>&1',properties=['energy', 'free_energy', 'forces'])

output  = task1.result()

print(output)

#with open('./atoms_pickle.pkl','wb') as outfile:
#    pickle.dump(outfile,results)