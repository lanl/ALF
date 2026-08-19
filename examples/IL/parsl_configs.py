"""Example Parsl resource configurations for the simple water workflow.

This file controls only runtime resources: scheduler partitions, allocations,
accounts, QoS, walltimes, launchers, and worker counts. It does not change the
water system, QM method, ML model, or sampling behavior. Replace the
placeholders below with values for your site before running.
"""

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher, SingleNodeLauncher
from parsl.providers import SlurmProvider


# Replace these placeholders with names from your Slurm cluster.
CPU_PARTITION = "your_cpu_partition"
GPU_PARTITION = "your_gpu_partition"
DEBUG_PARTITION = "your_debug_partition"

# Set an account to None when the matching partition does not require one.
CPU_ACCOUNT = None
GPU_ACCOUNT = "your_gpu_account"
DEBUG_ACCOUNT = CPU_ACCOUNT

# Set QoS values to None unless your site requires #SBATCH --qos=...
CPU_QOS = None
GPU_QOS = None
DEBUG_QOS = None

# Scheduler resource lines are site-specific. Some clusters use --gres, some
# use --gpus, and some use custom constraints. Replace these examples with the
# exact #SBATCH lines required by your scheduler.
CPU_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=36"
GPU_TRAINING_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4"
GPU_SAMPLER_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4"
DEBUG_CPU_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=36"
DEBUG_GPU_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4"

# Replace these with commands needed before workers start at your site.
# CPU workers usually load QM software and MPI modules. GPU workers usually
# activate the Python environment and any CUDA/ROCm modules needed by ML tasks.
CPU_WORKER_INIT = ""
GPU_WORKER_INIT = ""


# Parsl block terminology:
# - A block is one scheduler allocation requested from Slurm.
# - nodes_per_block is the number of nodes in each allocation.
# - init_blocks is the number of allocations requested when Parsl starts.
# - min_blocks is the lower bound Parsl tries to keep available.
# - max_blocks is the upper bound on simultaneous allocations for an executor.
# - max_workers_per_node is how many ALF tasks may run on each allocated node.
#
# ALF queue-depth knobs live in master_config.json. For example,
# parallel_samplers is the number of builder/sampler candidates ALF tries to
# keep active, while the Parsl executor settings below determine how many of
# those tasks can actually run at once.


def slurm_options(extra_options="", account=None, qos=None):
    """Return optional #SBATCH lines for a SlurmProvider."""
    lines = []
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if extra_options:
        lines.append(extra_options)
    return "\n".join(lines)


config_1node = Config(
    executors=[
        # QM labeling tasks, such as ORCA single-point calculations.
        # Use one ALF worker per node when each QM task expects the whole node.
        HighThroughputExecutor(
            label="alf_QM_executor",
            max_workers_per_node=1,
            provider=SlurmProvider(
                partition=CPU_PARTITION,
                # Start with no allocation and scale up on demand. With
                # nodes_per_block=1 and max_blocks=1, this executor requests
                # at most one CPU node at a time.
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    CPU_RESOURCE_OPTIONS,
                    account=CPU_ACCOUNT,
                    qos=CPU_QOS,
                ),
                worker_init=CPU_WORKER_INIT,
                launcher=SimpleLauncher(),
                walltime="6:00:00",
                cmd_timeout=30,
            ),
        ),
        # Optional lower-priority QM executor. ALF automatically adds this label
        # when it exists, so remove this executor if your cluster has no
        # standby/preemptible queue.
        HighThroughputExecutor(
            label="alf_QM_standby_executor",
            max_workers_per_node=1,
            provider=SlurmProvider(
                partition=CPU_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    CPU_RESOURCE_OPTIONS,
                    account=CPU_ACCOUNT,
                    qos=CPU_QOS,
                ),
                worker_init=CPU_WORKER_INIT,
                launcher=SimpleLauncher(),
                walltime="1:00:00",
                cmd_timeout=30,
            ),
        ),
        # ML training tasks, such as HIPPYNN ensemble training. This template
        # requests one GPU allocation and runs one ALF training task in it.
        HighThroughputExecutor(
            label="alf_ML_executor",
            max_workers_per_node=1,
            provider=SlurmProvider(
                partition=GPU_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    GPU_TRAINING_RESOURCE_OPTIONS,
                    account=GPU_ACCOUNT,
                    qos=GPU_QOS,
                ),
                worker_init=GPU_WORKER_INIT,
                launcher=SingleNodeLauncher(),
                walltime="16:00:00",
                cmd_timeout=30,
            ),
        ),
        # ML-driven molecular dynamics sampling tasks. The master config's
        # gpus_per_node value controls GPU assignment inside sampler workers.
        # With max_workers_per_node=4 and gpus_per_node=4, ALF can run up to
        # four sampler tasks on one GPU node, one per GPU.
        HighThroughputExecutor(
            label="alf_sampler_executor",
            max_workers_per_node=4,
            provider=SlurmProvider(
                partition=GPU_PARTITION,
                init_blocks=0,
                min_blocks=0,
                # Increase max_blocks only when you want Parsl to request more
                # simultaneous GPU allocations for sampling.
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    GPU_SAMPLER_RESOURCE_OPTIONS,
                    account=GPU_ACCOUNT,
                    qos=GPU_QOS,
                ),
                worker_init=GPU_WORKER_INIT,
                launcher=SingleNodeLauncher(),
                walltime="4:00:00",
                cmd_timeout=30,
            ),
        ),
        # Optional lower-priority sampler executor. Remove this executor if your
        # cluster does not provide a suitable standby/preemptible GPU queue.
        HighThroughputExecutor(
            label="alf_sampler_standby_executor",
            max_workers_per_node=4,
            provider=SlurmProvider(
                partition=GPU_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    GPU_SAMPLER_RESOURCE_OPTIONS,
                    account=GPU_ACCOUNT,
                    qos=GPU_QOS,
                ),
                worker_init=GPU_WORKER_INIT,
                launcher=SingleNodeLauncher(),
                walltime="4:00:00",
                cmd_timeout=30,
            ),
        ),
    ]
)


# The --test_builder, --test_qm, --test_ml, and --test_sampler commands use
# this smaller debug config when parsl_debug_configuration is set in
# master_config.json.
config_debug = Config(
    executors=[
        HighThroughputExecutor(
            label="alf_QM_executor",
            max_workers_per_node=1,
            provider=SlurmProvider(
                partition=DEBUG_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    DEBUG_CPU_RESOURCE_OPTIONS,
                    account=DEBUG_ACCOUNT,
                    qos=DEBUG_QOS,
                ),
                worker_init=CPU_WORKER_INIT,
                launcher=SimpleLauncher(),
                walltime="2:00:00",
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label="alf_ML_executor",
            max_workers_per_node=1,
            provider=SlurmProvider(
                partition=DEBUG_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    DEBUG_GPU_RESOURCE_OPTIONS,
                    account=GPU_ACCOUNT,
                    qos=DEBUG_QOS,
                ),
                worker_init=GPU_WORKER_INIT,
                launcher=SingleNodeLauncher(),
                walltime="1:00:00",
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label="alf_sampler_executor",
            max_workers_per_node=4,
            provider=SlurmProvider(
                partition=DEBUG_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    DEBUG_GPU_RESOURCE_OPTIONS,
                    account=GPU_ACCOUNT,
                    qos=DEBUG_QOS,
                ),
                worker_init=GPU_WORKER_INIT,
                launcher=SingleNodeLauncher(),
                walltime="1:00:00",
                cmd_timeout=30,
            ),
        ),
    ]
)
