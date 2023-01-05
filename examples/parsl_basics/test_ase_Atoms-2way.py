###################################
# Step 1: Specify a configuration #
###################################

import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher

from ase import Atoms

# Uncomment this to see logging info
#import logging
#logging.basicConfig(level=logging.DEBUG)

config_darwin = Config(
    executors=[
        HighThroughputExecutor(
            label='Darwin_singlenode',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),

            cores_per_worker=2,

            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                'atdm-ml',

                nodes_per_block=1,

                # String to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --qos=debug',

                # Command to be run before starting a worker
                worker_init='module load miniconda3; source activate parsl',

                # We request all hyperthreads on a node.
                launcher=SrunLauncher(overrides='-c 64'),
                walltime='00:10:00',

                # Slurm scheduler can be slow at times,  increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)

# Load the Parsl config
parsl.load(config_darwin)


#############################
# Step 2: Define Parsl apps #
#############################

from parsl import python_app, bash_app

# This is just dummy code to make an atoms object.
# We will replace it with both builder and ML driven MD code. Atoms objects will be passed between them.
@python_app
def builder(input_atoms):
    from ase import Atoms
    print(input_atoms.get_positions())
    d = 2.9
    L = 10.0
    wire = Atoms('Au',
                 positions=[[0.0, L/2.0, L/2.0]],
                 cell=[d, L, L],
                 pbc=[1, 0, 0])
    return(wire)
 

###################################
# Step 3: Create a Parsl workflow #
###################################

# Load ASE library
from ase import Atoms

d = 2.9
L = 10.0
wire = Atoms('Au',
    positions=[[0.0, L/2.0, L/2.0]],
    cell=[d, L, L],
    pbc=[1, 0, 0])

# Define a Parsl task
task1 = builder(wire)

# Call and execute the task through Slurm; return the results
ase_atom_object = task1.result()

# Check if the Atoms object is passed correctly 
print(ase_atom_object.get_positions())    # get_positions is a function of Atoms objects 

