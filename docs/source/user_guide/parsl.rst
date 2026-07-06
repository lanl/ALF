Parsl And HPC Execution
=======================

ALF uses `Parsl <https://parsl-project.org/>`_ to submit and track workflow
tasks. Parsl does not change the scientific work ALF performs; it decides where
builder, sampler, QM, and ML tasks run, how many tasks may run concurrently, and
which scheduler allocation is requested for each task type.

This page is a practical guide for adapting ALF to a new local workstation or
HPC cluster.

How ALF Uses Parsl
------------------

ALF task functions are decorated with Parsl app definitions. Each app has one
or more executor labels. The Parsl configuration provides executors with those
labels and maps them onto local workers or scheduler allocations.

The common executor labels are:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Executor label
     - Typical role
   * - ``alf_QM_executor``
     - Runs QM labeling tasks, such as ORCA, QChem, or VASP jobs.
   * - ``alf_QM_standby_executor``
     - Optional lower-priority or shorter-walltime QM executor. ALF adds this
       standby label automatically when it exists.
   * - ``alf_ML_executor``
     - Runs model training tasks.
   * - ``alf_sampler_executor``
     - Runs ML-driven sampling tasks.
   * - ``alf_sampler_standby_executor``
     - Optional standby sampler executor.

In ``master_config.json``, these fields select the Parsl configuration:

.. code-block:: json

   {
     "parsl_configuration": "parsl_configs.config_1node",
     "parsl_debug_configuration": "parsl_configs.config_debug"
   }

Normal runs use ``parsl_configuration``. Stage checks such as ``--test_builder``
and ``--test_qm`` use ``parsl_debug_configuration`` when it is present.
In the example directories, ``parsl_configs.py`` is a template for Parsl
executor, Slurm partition, account, walltime, module, and launcher settings.
The placeholders in those files must be replaced for your cluster before a
full run.

Minimal Single-Node Debug Config
--------------------------------

Use a small debug config first. It should request short walltime, few blocks,
and enough workers to run one or two tasks while you validate imports, paths,
and external executables.

.. code-block:: python

   from parsl.config import Config
   from parsl.executors import HighThroughputExecutor
   from parsl.providers import SlurmProvider
   from parsl.launchers import SimpleLauncher

   config_debug = Config(
       executors=[
           HighThroughputExecutor(
               label="alf_QM_executor",
               max_workers_per_node=1,
               provider=SlurmProvider(
                   partition="debug",
                   account="your_account",
                   nodes_per_block=1,
                   init_blocks=0,
                   min_blocks=0,
                   max_blocks=1,
                   walltime="00:30:00",
                   scheduler_options="#SBATCH --ntasks-per-node=1",
                   worker_init="module load orca",
                   launcher=SimpleLauncher(),
               ),
           ),
       ]
   )

This pattern is useful for ``--test_qm`` and other bring-up checks. Replace the
partition, account, module commands, and walltime for your cluster.

One QM Job Per Allocation
-------------------------

Many QM engines are themselves parallel programs. In that case, configure Parsl
to run one ALF QM task per allocation and let the QM command use the cores or
MPI ranks inside that allocation.

.. code-block:: python

   HighThroughputExecutor(
       label="alf_QM_executor",
       max_workers_per_node=1,
       provider=SlurmProvider(
           partition="standard",
           account="your_account",
           nodes_per_block=1,
           init_blocks=0,
           min_blocks=0,
           max_blocks=8,
           walltime="06:00:00",
           scheduler_options="#SBATCH --nodes=1 --ntasks-per-node=36",
           worker_init="module load orca",
           launcher=SimpleLauncher(),
       ),
   )

The important point is that ``max_workers_per_node=1`` prevents multiple QM
tasks from landing on the same node when each QM task expects the whole node.

MPI-Based DFT Codes
-------------------

For MPI-based DFT codes, the launch command usually belongs in the QM config
that ALF passes to the QM interface, while Parsl requests the allocation.

For example, a VASP-style QM config may use a command like:

.. code-block:: json

   {
     "QM_run_command": "srun -n 128 vasp_std"
   }

The matching Parsl executor should request resources consistent with that
command:

.. code-block:: python

   HighThroughputExecutor(
       label="alf_QM_executor",
       max_workers_per_node=1,
       provider=SlurmProvider(
           partition="standard",
           account="your_account",
           nodes_per_block=4,
           init_blocks=0,
           min_blocks=0,
           max_blocks=4,
           walltime="12:00:00",
           scheduler_options="#SBATCH --nodes=4 --ntasks-per-node=32",
           worker_init="module load vasp",
           launcher=SimpleLauncher(),
       ),
   )

Different clusters use different launchers and MPI integration. Some sites
prefer ``srun`` inside ``QM_run_command``; others require a launcher override or
site-specific wrapper script. The resource request and QM command should agree
on node count, rank count, thread count, and environment modules.

GPU Executors For ML And Sampling
---------------------------------

ML training and MLMD sampling commonly need GPUs. Use separate executors so
these tasks do not compete with CPU-only QM jobs.
GPU scheduler syntax is site-specific: one cluster may use
``#SBATCH --gres=gpu:N``, another may use ``#SBATCH --gpus=N``, and another may
require constraints or site wrappers. Treat the GPU lines below as examples to
replace, not portable defaults.

.. code-block:: python

   from parsl.launchers import SingleNodeLauncher

   HighThroughputExecutor(
       label="alf_ML_executor",
       max_workers_per_node=1,
       provider=SlurmProvider(
           partition="gpu",
           account="your_gpu_account",
           nodes_per_block=1,
           init_blocks=0,
           min_blocks=0,
           max_blocks=1,
           walltime="04:00:00",
           scheduler_options="#SBATCH --nodes=1 --gres=gpu:1",
           worker_init="module load cuda",
           launcher=SingleNodeLauncher(),
       ),
   )

   HighThroughputExecutor(
       label="alf_sampler_executor",
       max_workers_per_node=4,
       provider=SlurmProvider(
           partition="gpu",
           account="your_gpu_account",
           nodes_per_block=1,
           init_blocks=0,
           min_blocks=0,
           max_blocks=2,
           walltime="04:00:00",
           scheduler_options="#SBATCH --nodes=1 --gres=gpu:4",
           worker_init="module load cuda",
           launcher=SingleNodeLauncher(),
       ),
   )

ALF also uses ``gpus_per_node`` from the master configuration to assign visible
GPU ids inside sampler tasks.

Generic CPU/GPU Template Pattern
--------------------------------

For portable examples, keep CPU and GPU resource placeholders separate. Account
and QoS lines can be optional when a partition does not require them.

.. code-block:: python

   CPU_PARTITION = "your_cpu_partition"
   GPU_PARTITION = "your_gpu_partition"
   CPU_ACCOUNT = None
   GPU_ACCOUNT = "your_gpu_account"
   CPU_QOS = None
   GPU_QOS = None

   CPU_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=36"
   GPU_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4"

   def slurm_options(extra_options="", account=None, qos=None):
       lines = []
       if account:
           lines.append(f"#SBATCH --account={account}")
       if qos:
           lines.append(f"#SBATCH --qos={qos}")
       if extra_options:
           lines.append(extra_options)
       return "\n".join(lines)

   HighThroughputExecutor(
       label="alf_QM_executor",
       max_workers_per_node=1,
       provider=SlurmProvider(
           partition=CPU_PARTITION,
           nodes_per_block=1,
           init_blocks=0,
           min_blocks=0,
           max_blocks=8,
           scheduler_options=slurm_options(
               CPU_RESOURCE_OPTIONS,
               account=CPU_ACCOUNT,
               qos=CPU_QOS,
           ),
           worker_init="module load orca",
           launcher=SimpleLauncher(),
       ),
   )

   HighThroughputExecutor(
       label="alf_sampler_executor",
       max_workers_per_node=4,
       provider=SlurmProvider(
           partition=GPU_PARTITION,
           nodes_per_block=1,
           init_blocks=0,
           min_blocks=0,
           max_blocks=1,
           scheduler_options=slurm_options(
               GPU_RESOURCE_OPTIONS,
               account=GPU_ACCOUNT,
               qos=GPU_QOS,
           ),
           worker_init="module load cuda",
           launcher=SingleNodeLauncher(),
       ),
   )

In this pattern, ``max_blocks`` controls how many scheduler allocations Parsl
may request for that executor, and ``max_workers_per_node`` controls how many
ALF tasks may run on each allocated node.

Configuration Knobs
-------------------

These are the Parsl settings most users need to edit.

In Parsl, a block is one scheduler allocation. For one executor, the largest
node footprint Parsl may request is ``max_blocks * nodes_per_block``. The
worker setting then controls how many ALF tasks can run inside those allocated
resources.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - What to check
   * - ``partition`` / ``account``
     - Queue, partition, project, or allocation name required by the scheduler.
   * - ``walltime``
     - Maximum runtime for the scheduler allocation.
   * - ``nodes_per_block``
     - Number of nodes requested for one Parsl block.
   * - ``max_blocks``
     - Maximum number of scheduler allocations Parsl may request for that
       executor.
   * - ``max_workers_per_node`` or ``max_workers``
     - How many ALF tasks may execute on the allocated resources.
   * - ``scheduler_options``
     - Extra ``#SBATCH`` lines, such as node counts, task counts, GPUs, QoS, or
       account overrides.
   * - ``worker_init``
     - Commands run before Parsl workers start, such as ``module load`` or
       environment activation.
   * - ``launcher``
     - How Parsl starts workers inside an allocation. Common choices in ALF
       examples include ``SimpleLauncher`` and ``SingleNodeLauncher``.
   * - ``address``
     - Network interface or hostname used when compute nodes must connect back
       to the submit process.

Bring-Up Checklist
------------------

1. Start with a debug Parsl config and run ``--test_builder``.
2. Run ``--test_qm`` with a tiny system and inspect the QM scratch directory.
3. Confirm parsed values are written to ``qm_test.h5``.
4. Run ``--test_ml`` after a small HDF5 data file exists.
5. Run ``--test_sampler`` only after a trained model is available.
6. Increase ``max_blocks``, walltime, and queue thresholds only after individual
   stages work.

The existing files in ``alframework/parsl_resource_configs`` and
``examples/*/parsl_configs.py`` show the expected structure, but cluster
partition names, accounts, modules, walltimes, and launcher choices must be
adapted for the resources you plan to use.
