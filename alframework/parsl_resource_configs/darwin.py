import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SimpleLauncher

# The following configuration requests 10 hours of allocation on Darwin.
config_atdm_ml = Config(
    executors=[

        HighThroughputExecutor(

            label='alf_QM_executor',

            max_workers=1,

            provider=SlurmProvider(
                # Partition / QOS
                # NOTE: replace node partition before use
                #'regular',
                #'ml4chem',
                'atdm-ml',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # String to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=8',

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load miniconda3; source /usr/projects/ml4chem/envs/p38-parsl-ani.bash',

                launcher=SimpleLauncher(),
                walltime='10:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        ),

        HighThroughputExecutor(
        
            label='alf_ML_executor',

            max_workers=1,

            provider=SlurmProvider(
                # Partition / QOS
                # NOTE: replace node partition before use
                #'regular',
                #'ml4chem',
                'atdm-ml',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load miniconda3; source /usr/projects/ml4chem/envs/p38-parsl-ani.bash',

                launcher=SingleNodeLauncher(),
                walltime='10:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        ),

        HighThroughputExecutor(
        
            label='alf_sampler_executor',

            max_workers=8,

            provider=SlurmProvider(
                # Partition / QOS
                # NOTE: replace node partition before use
                #'regular',
                #'ml4chem',
                'atdm-ml',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load miniconda3; source /usr/projects/ml4chem/envs/p38-parsl-ani.bash',

                launcher=SingleNodeLauncher(),
                walltime='10:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)

# The following configuration requests only 2 hours of allocation instead of 10 hours.
config_atdm_ml_short = Config(

    executors=[
    
        HighThroughputExecutor(

            label='alf_QM_executor',

            max_workers=1,

            provider=SlurmProvider(
                # Partition / QOS
                # NOTE: replace node partition before use
                #'regular',
                #'ml4chem',
                'atdm-ml',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # String to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=8',

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load miniconda3; source /usr/projects/ml4chem/envs/p38-parsl-ani.bash',

                launcher=SimpleLauncher(),
                walltime='00:02:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        ),

        HighThroughputExecutor(

            label='alf_ML_executor',

            max_workers=1,

            provider=SlurmProvider(
                # Partition / QOS
                # NOTE: replace node partition before use
                #'regular',
                #'ml4chem',
                'atdm-ml',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load miniconda3; source /usr/projects/ml4chem/envs/p38-parsl-ani.bash',

                launcher=SingleNodeLauncher(),
                walltime='00:02:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        ),

        HighThroughputExecutor(

            label='alf_sampler_executor',

            max_workers=8,

            provider=SlurmProvider(
                # Partition / QOS
                # NOTE: replace node partition before use
                #'regular',
                #'ml4chem',
                'atdm-ml',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load miniconda3; source /usr/projects/ml4chem/envs/p38-parsl-ani.bash',

                launcher=SingleNodeLauncher(),
                walltime='00:02:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)
