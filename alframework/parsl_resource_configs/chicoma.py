import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SimpleLauncher

# The following configuration can be used for submitting jobs to a standard job queue
config_standard = Config(

    executors=[
        # TODO: Double check the setting in this executor. 
        # We manually increase the node count so that our parallel qm job gets multiple nodes, 
        # but leave nodes_per_block=1 so that Parsl doesn't assign multiple tasks.
        HighThroughputExecutor(

            label='alf_QM_executor',

            max_workers=1,

            provider=SlurmProvider(
                # Partition / QOS
                'standard',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 50,

                nodes_per_block=1,
                #workers_per_node=1,

                # String to prepend to #SBATCH blocks in the submit
                # NOTE: replace project code
                scheduler_options='#SBATCH --ntasks-per-node=32 --nodes=4 -A w23_ml4chem',

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load python/3.8-anaconda-2020.07; source /usr/projects/ml4chem/envs/p38-parsl-ani-load.bash',

                launcher=SimpleLauncher(),
                walltime='16:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        ),

        HighThroughputExecutor(

            label='alf_ML_executor',

            max_workers=1,

            provider=SlurmProvider(
                # Partition / QOS
                'gpu',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # String to prepend to #SBATCH blocks in the submit
                # NOTE: replace project code
                #scheduler_options='#SBATCH --qos=debug',
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g',

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load python/3.8-anaconda-2020.07; source /usr/projects/ml4chem/envs/p38-parsl-ani-load.bash',

                launcher=SingleNodeLauncher(),
                walltime='16:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        ),

        HighThroughputExecutor(

            label='alf_sampler_executor',

            max_workers=4,

            provider=SlurmProvider(
                # Partition / QOS
                'gpu',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # String to prepend to #SBATCH blocks in the submit
                # NOTE: replace project code
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g',

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load python/3.8-anaconda-2020.07; source /usr/projects/ml4chem/envs/p38-parsl-ani-load.bash',

                launcher=SingleNodeLauncher(),
                walltime='16:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)


# The following configuration can be used for submitting jobs to the debug queue
config_debug = Config(
    executors=[

        HighThroughputExecutor(

            label='alf_QM_executor',

            max_workers=1,

            provider=SlurmProvider(
                # Partition / QOS
                'standard',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # String to prepend to #SBATCH blocks in the submit
                # NOTE: replace project code
                scheduler_options='#SBATCH --ntasks-per-node=32 --nodes=4 -A w23_ml4chem --qos=debug --reservation=debug',

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load python/3.8-anaconda-2020.07; source /usr/projects/ml4chem/envs/p38-parsl-ani-load.bash',

                launcher=SimpleLauncher(),
                walltime='2:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        ),

        HighThroughputExecutor(

            label='alf_ML_executor',

            max_workers=1,

            provider=SlurmProvider(
                # Partition / QOS
                'gpu',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # String to prepend to #SBATCH blocks in the submit
                # NOTE: replace project code
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load python/3.8-anaconda-2020.07; source /usr/projects/ml4chem/envs/p38-parsl-ani-load.bash',

                launcher=SingleNodeLauncher(),
                walltime='2:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        ),

        HighThroughputExecutor(

            label='alf_sampler_executor',

            max_workers=4,

            provider=SlurmProvider(
                # Partition / QOS
                'gpu',

                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # String to prepend to #SBATCH blocks in the submit
                # NOTE: replace project code
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                # NOTE: replace modules and source path
                worker_init='module load python/3.8-anaconda-2020.07; source /usr/projects/ml4chem/envs/p38-parsl-ani-load.bash',

                launcher=SingleNodeLauncher(),
                walltime='2:00:00',

                # Slurm scheduler can be slow at times, increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)