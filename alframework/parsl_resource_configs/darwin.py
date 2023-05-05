import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SimpleLauncher

config_atdm_ml = Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'atdm-ml',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=8',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='10:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        ),
        HighThroughputExecutor(
            label='alf_QM_standby_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'atdm-ml',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 0,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=8 --qos=standby',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='10:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
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
                'atdm-ml',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='10:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
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
                'atdm-ml',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='10:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)

config_atdm_ml_short = Config(
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
                'atdm-ml',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=8',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='00:02:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        ),
        HighThroughputExecutor(
            label='alf_QM_standby_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'atdm-ml',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=8 --qos=standby',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='00:02:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        ),
        HighThroughputExecutor(
            label='alf_ML_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=1,


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'atdm-ml',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='00:02:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers=8,

            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'atdm-ml',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                #scheduler_options='#SBATCH --qos=debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='00:02:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)
