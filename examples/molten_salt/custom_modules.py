import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SimpleLauncher

from parsl import python_app, bash_app


config_running = Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',
            #This executor is kinda strange. We manually increase the node count so that 
            #our parallel qm job gets multiple nodes, but leave nodes_per_block=1 so 
            #that parsl  doesn't assign multiple tasks

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

                nodes_per_block=2,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=128 -A w23_ml4chem',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='3:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_QM_standby_executor',
            #This executor is kinda strange. We manually increase the node count so that 
            #our parallel qm job gets multiple nodes, but leave nodes_per_block=1 so 
            #that parsl  doesn't assign multiple tasks

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

                nodes_per_block=2,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=128 -A w23_ml4chem --qos=standby --time-min=00:15:00',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='6:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_ML_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -p gpu -A w23_ml4chem_g',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='16:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=8,
            #cpu_affinity='alternating',

            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 5,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -A w23_ml4chem_g ',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='8:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_standby_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=8,
            #cpu_affinity='alternating',

            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 5,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -A w23_ml4chem_g  --time-min=00:10:00 --qos=standby',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='8:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        )
    ]
)

config_debug = Config(
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
                'debug',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=2,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=128 -A w23_ml4chem --qos debug --reservation debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='1:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_ML_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu_debug',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='1:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=8,
            #cpu_affinity='alternating',

            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu_debug',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='1:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=30,
            ),
        )
    ]
)

import glob
from importlib import import_module
import os
import json
import pickle
import ase
import re
from ase import Atoms
from ase.io import vasp as vasp_io
   
@python_app(executors=['alf_QM_executor','alf_QM_standby_executor'])
def ase_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """
        Args:
            molecule_object (list): List containing the return of the builder task.
            QM_config (dict): json file containing the parameters for the QM calculation.
            QM_scratch_dir (str): Scratch path for the QM calculation.
            properties_list: The property list defined on master_config.json. The properties defined here 
                             will be extracted from the QM calculation and stored in the database for 
                             training the NN. We can also specify an unit conversion factor. 
        
        Returns:
            (list): A list representing the updated molecule_object received that now stores the QM 
                    calculation results.
    """
    
    import glob
    
    from ase.calculators.vasp import Vasp as calc_class
    
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    properties = list(properties_list)
    
    command = QM_config['QM_run_command']

    # Define the calculator with desired pseudopotentials
    calc = calc_class(directory=directory, command=command, **QM_config['input_list'], 
                setups={'F': '', 'Li': '_sv', 'Na': '_pv', 'K': '_sv', 'Cl': '', 'Mg': '_pv', 'Be': '_sv'})
    
    # Run the calculation
    calc.calculate(atoms=molecule_object[1], properties=properties)

    for curF in glob.glob(directory+'/WAVECAR*'):
        os.remove(curF)
    
    molecule_object[2] = calc.results
    
    convergedre = re.compile('aborting loop because EDIFF is reached')
    txt = open(directory + '/OUTCAR','r').read()
    if len(convergedre.findall(txt)) == 1:
        molecule_object[2]['converged'] = True
    else:
        molecule_object[2]['converged'] = False
        
    return molecule_object

