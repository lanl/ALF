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

Different clusters use different launchers and MPI integration. The resource request and QM command should agree
on node count, rank count, thread count, and environment modules.

Separate GPU Executors For ML And Sampling
------------------------------------------

ML training and MLMD sampling tasks typically need GPUs. Use separate executors so
these tasks do not compete with CPU-only (e.g., QM) jobs. Below is an example parsl config 
for GPU execution of ML and sampling tasks.

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

Globus Compute Endpoints
------------------------

ALF can also target `Globus Compute <https://www.globus.org/compute>`_
endpoints through Parsl. In this pattern, ``parsl_configs.py`` uses Parsl's
``GlobusComputeExecutor`` instead of a scheduler-backed executor, while keeping
the ALF executor labels such as ``alf_QM_executor``, ``alf_ML_executor``, or
``alf_sampler_executor``. This can useful for multisite runs where different endpoints 
provide different resources, or for users who do not have a local scheduler.

The Parsl documentation includes a
`Globus Compute multisite example <https://parsl.readthedocs.io/en/stable/userguide/configuration/examples.html#globus-compute-multisite>`_
with additional endpoint setup details. A compact ALF-style configuration can
look like this:

.. code-block:: python

   from globus_compute_sdk import Executor

   from parsl.config import Config
   from parsl.executors import GlobusComputeExecutor

   ml_endpoint = "YOUR_GLOBUS_COMPUTE_ENDPOINT_UUID"

   config_globus = Config(
       executors=[
           GlobusComputeExecutor(
               executor=Executor(endpoint_id=ml_endpoint),
               label="alf_ML_executor",
           ),
       ],
   )

For multisite runs, add more ``GlobusComputeExecutor`` entries with different
endpoint UUIDs and the ALF labels for the work each endpoint should receive.
For example, one endpoint could provide ``alf_QM_executor`` while another
provides ``alf_ML_executor`` or ``alf_sampler_executor``. Node that the Globus Compute 
SDK must be installed in the Python environment that runs ALF to use this pattern functionality.

Generic CPU/GPU Template Pattern
--------------------------------

The simple-water example includes a generic Slurm template in
``examples/simple_water/parsl_configs.py``. The pattern below mirrors that file:
CPU resources are used for QM labeling, GPU resources are used for ML training
and sampling, and a smaller debug configuration is available for staged test
commands.

This is still a template. Replace partition names, accounts, QoS values,
``#SBATCH`` resource lines, module loads, launchers, and walltimes with values
that match the cluster.

.. code-block:: python

   from parsl.config import Config
   from parsl.executors import HighThroughputExecutor
   from parsl.launchers import SimpleLauncher, SingleNodeLauncher
   from parsl.providers import SlurmProvider

   CPU_PARTITION = "your_cpu_partition"
   GPU_PARTITION = "your_gpu_partition"
   DEBUG_PARTITION = "your_debug_partition"

   CPU_ACCOUNT = "your_cpu_account"
   GPU_ACCOUNT = "your_gpu_account"
   DEBUG_ACCOUNT = "your_debug_account"

   CPU_QOS = None
   GPU_QOS = None
   DEBUG_QOS = None

   CPU_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=36"
   GPU_TRAINING_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4"
   GPU_SAMPLER_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4"
   DEBUG_CPU_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=36"
   DEBUG_GPU_RESOURCE_OPTIONS = "#SBATCH --nodes=1\n#SBATCH --gres=gpu:4"

   CPU_WORKER_INIT = ""
   GPU_WORKER_INIT = ""

   def slurm_options(extra_options="", account=None):
       lines = []
       if account:
           lines.append(f"#SBATCH --account={account}")
       if extra_options:
           lines.append(extra_options)
       return "\n".join(lines)

   config_1node = Config(
       executors=[
           HighThroughputExecutor(
               label="alf_QM_executor",
               max_workers_per_node=1,
               provider=SlurmProvider(
                   partition=CPU_PARTITION,
                   init_blocks=0,
                   min_blocks=0,
                   max_blocks=2,
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
           HighThroughputExecutor(
               label="alf_QM_standby_executor",
               max_workers_per_node=1,
               provider=SlurmProvider(
                   partition=CPU_PARTITION,
                   init_blocks=0,
                   min_blocks=0,
                   max_blocks=2,
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
           HighThroughputExecutor(
               label="alf_sampler_executor",
               max_workers_per_node=4,
               provider=SlurmProvider(
                   partition=GPU_PARTITION,
                   init_blocks=0,
                   min_blocks=0,
                   max_blocks=2,
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
           HighThroughputExecutor(
               label="alf_sampler_standby_executor",
               max_workers_per_node=4,
               provider=SlurmProvider(
                   partition=GPU_PARTITION,
                   init_blocks=0,
                   min_blocks=0,
                   max_blocks=2,
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

Production executors in this template have separate roles. ``alf_QM_executor``
runs CPU QM labeling tasks with one ALF worker per node. The optional
``alf_QM_standby_executor`` can target lower-priority or shorter-walltime CPU
resources. ``alf_ML_executor`` runs one ML training task per GPU allocation.
``alf_sampler_executor`` runs MLMD or reactive sampling tasks on GPU resources,
and ``alf_sampler_standby_executor`` is the optional standby version of that
sampler pool.

The debug configuration should be smaller and faster. The ``--test_builder``,
``--test_qm``, ``--test_ml``, and ``--test_sampler`` commands use it when
``parsl_debug_configuration`` is set in ``master_config.json``.

.. code-block:: python

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

In these patterns, a Parsl block is one scheduler allocation. For each
executor, ``nodes_per_block * max_blocks`` controls the maximum node footprint
Parsl may request, while ``max_workers_per_node`` controls how many ALF tasks
can run on each allocated node. ALF settings such as ``parallel_samplers`` and
``target_queued_QM`` are queue-depth targets; they do not directly request
nodes. For sampler tasks, make sure ``gpus_per_node`` matches the number of
sampler workers you expect to place on each GPU node.

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


The existing files in ``alframework/parsl_resource_configs`` and
``examples/*/parsl_configs.py`` show the expected structure, but cluster
partition names, accounts, modules, walltimes, and launcher choices must be
adapted for the resources you plan to use.
