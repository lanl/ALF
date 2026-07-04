"""Example Parsl resource configurations for the simple water workflow.

This file controls where ALF tasks run: scheduler partitions, allocations,
walltimes, launchers, and worker counts. It does not change the water system,
QM method, ML model, or sampling behavior. Replace the placeholder partition,
account, and module settings with values for your site before running.
"""

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher, SingleNodeLauncher
from parsl.providers import SlurmProvider


# Replace these placeholders with names from your Slurm cluster.
CPU_PARTITION = "your_cpu_partition"
GPU_PARTITION = "your_gpu_partition"
DEBUG_PARTITION = "your_debug_partition"
ACCOUNT = "your_account"
GPU_ACCOUNT = "your_gpu_account"

# Replace these with commands needed before workers start at your site.
# Examples: "module load orca", "module load cuda", or "source activate alf".
QM_WORKER_INIT = ""
GPU_WORKER_INIT = ""


# Parsl block terminology:
# - A block is one scheduler allocation requested from Slurm.
# - nodes_per_block is the number of nodes in each allocation.
# - init_blocks is the number of allocations requested when Parsl starts.
# - min_blocks is the lower bound Parsl tries to keep available.
# - max_blocks is the upper bound on simultaneous allocations for an executor.
# - max_workers is the number of ALF tasks that may run in those allocations.


def slurm_options(account, extra_options=""):
    """Return generic #SBATCH lines for a SlurmProvider."""
    lines = [f"#SBATCH --account={account}"]
    if extra_options:
        lines.append(extra_options)
    return "\n".join(lines)


config_1node = Config(
    executors=[
        # QM labeling tasks, such as ORCA single-point calculations.
        # Use one worker per node when each QM task expects the whole node.
        HighThroughputExecutor(
            label="alf_QM_executor",
            # max_workers limits concurrent ALF QM tasks for this executor.
            max_workers=36,
            provider=SlurmProvider(
                partition=CPU_PARTITION,
                # Start with no allocation, scale up on demand, and allow at
                # most two one-node allocations for this executor.
                init_blocks=0,
                min_blocks=0,
                max_blocks=2,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    ACCOUNT,
                    "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=36",
                ),
                worker_init=QM_WORKER_INIT,
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
            max_workers=36,
            provider=SlurmProvider(
                partition=CPU_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=2,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    ACCOUNT,
                    "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=36",
                ),
                worker_init=QM_WORKER_INIT,
                launcher=SimpleLauncher(),
                walltime="1:00:00",
                cmd_timeout=30,
            ),
        ),
        # ML training tasks, such as HIPPYNN ensemble training.
        HighThroughputExecutor(
            label="alf_ML_executor",
            max_workers=1,
            provider=SlurmProvider(
                partition=GPU_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    GPU_ACCOUNT,
                    "#SBATCH --nodes=1\n#SBATCH --gres=gpu:1",
                ),
                worker_init=GPU_WORKER_INIT,
                launcher=SingleNodeLauncher(),
                walltime="16:00:00",
                cmd_timeout=30,
            ),
        ),
        # ML-driven molecular dynamics sampling tasks. The master config's
        # gpus_per_node value controls GPU assignment inside sampler workers.
        HighThroughputExecutor(
            label="alf_sampler_executor",
            max_workers=4,
            provider=SlurmProvider(
                partition=GPU_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=2,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    GPU_ACCOUNT,
                    "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4",
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
            max_workers=4,
            provider=SlurmProvider(
                partition=GPU_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=2,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    GPU_ACCOUNT,
                    "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4",
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
            max_workers=36,
            provider=SlurmProvider(
                partition=DEBUG_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    ACCOUNT,
                    "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=36",
                ),
                worker_init=QM_WORKER_INIT,
                launcher=SimpleLauncher(),
                walltime="2:00:00",
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label="alf_ML_executor",
            max_workers=1,
            provider=SlurmProvider(
                partition=DEBUG_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    GPU_ACCOUNT,
                    "#SBATCH --nodes=1\n#SBATCH --gres=gpu:1",
                ),
                worker_init=GPU_WORKER_INIT,
                launcher=SingleNodeLauncher(),
                walltime="1:00:00",
                cmd_timeout=30,
            ),
        ),
        HighThroughputExecutor(
            label="alf_sampler_executor",
            max_workers=4,
            provider=SlurmProvider(
                partition=DEBUG_PARTITION,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=1,
                scheduler_options=slurm_options(
                    GPU_ACCOUNT,
                    "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4",
                ),
                worker_init=GPU_WORKER_INIT,
                launcher=SingleNodeLauncher(),
                walltime="1:00:00",
                cmd_timeout=30,
            ),
        ),
    ]
)
