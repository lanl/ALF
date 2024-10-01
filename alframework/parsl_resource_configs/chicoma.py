import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SimpleLauncher

config_standard = Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',
            #This executor is kinda strange. We manually increase the node count so that 
            #our parallel qm job gets multiple nodes, but leave nodes_per_block=1 so 
            #that parsl  doesn't assign multiple tasks

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers_per_node=1,
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
                scheduler_options='#SBATCH --ntasks-per-node=36 --nodes=1 -A w23_ml4chem',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='16:00:00',

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
            max_workers_per_node=1,
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
                scheduler_options='#SBATCH --ntasks-per-node=36 --nodes=1 -A w23_ml4chem --qos standby',

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
            max_workers_per_node=1,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g',

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
            max_workers_per_node=4,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g --qos=standby ---time-min=00:10:00',

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
            label='alf_sampler_standby_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers_per_node=4,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g',

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
        )
    ]
)

config_gpu = Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',
            #This executor is kinda strange. We manually increase the node count so that 
            #our parallel qm job gets multiple nodes, but leave nodes_per_block=1 so 
            #that parsl  doesn't assign multiple tasks

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers_per_node=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 20,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=32 --nodes=4 -A w23_ml4chem_g',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='16:00:00',

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
            max_workers_per_node=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'gpu',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 20,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=32 --nodes=4 -A w23_ml4chem_g --qos=standby ---time-min=01:00:00',

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
            max_workers_per_node=1,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g',

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
            max_workers_per_node=4,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g',

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
            max_workers_per_node=1,
            #cpu_affinity='alternating',


            provider=SlurmProvider(
                # Partition / QOS
                #'regular',
                #'ml4chem',
                'debug',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 2,

                nodes_per_block=1,
                #workers_per_node=1,

                # string to prepend to #SBATCH blocks in the submit
                scheduler_options='#SBATCH --ntasks-per-node=36 --nodes=1 -A w23_ml4chem --qos=debug --reservation=debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='2:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=5,
            ),
        ),
        HighThroughputExecutor(
            label='alf_ML_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers_per_node=1,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='2:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=5,
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers_per_node=4,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='2:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=5,
            ),
        )
    ]
)

config_debug_hybrid = Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',

            # Optional: the network interface on the login node to
            # which compute nodes can communicate
            #address=address_by_interface('bond0.144'),
            max_workers_per_node=1,
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
                scheduler_options='#SBATCH --ntasks-per-node=36 --nodes=4 -A w23_ml4chem',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SimpleLauncher(),
                walltime='16:00:00',

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
            max_workers_per_node=1,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='1:00:00',

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
            max_workers_per_node=4,
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
                scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',

                # Command to be run before starting a worker
                #worker_init=

                # We request all hyperthreads on a node.
                #launcher=SrunLauncher(overrides='-c 64'),
                launcher=SingleNodeLauncher(),
                walltime='1:00:00',

                # Slurm scheduler on Cori can be slow at times,
                # increase the command timeouts
                cmd_timeout=120,
            ),
        )
    ]
)
