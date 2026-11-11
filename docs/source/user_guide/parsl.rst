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
these tasks do not compete with CPU-only (e.g., QM) jobs. Below is an example Parsl config
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

ALF's MLMD sampler workers preserve scheduler/Parsl GPU visibility and select
local device ordinals automatically by worker rank. ``gpus_per_node`` caps
assignment when no visibility mask is set; an existing allocation takes
precedence. GPU numbering inside a worker is relative to that allocation.
See :doc:`samplers` for explicit device overrides and legacy defaults.

.. _parsl-multinode-training:

Multi-Node ML Training
^^^^^^^^^^^^^^^^^^^^^^

The multi-node training tasks (see :ref:`ml-multinode-training`) run one Parsl task per ensemble
member, so ``alf_ML_executor`` needs one worker per GPU. For any number of nodes, use one-node blocks
and set ``max_blocks`` to the number of nodes:

.. code-block:: python

   HighThroughputExecutor(
       label="alf_ML_executor",
       max_workers_per_node=4,
       available_accelerators=4,
       provider=SlurmProvider(
           partition="gpu",
           account="your_gpu_account",
           nodes_per_block=1,
           init_blocks=0,
           min_blocks=0,
           max_blocks=10,
           walltime="04:00:00",
           scheduler_options="#SBATCH --nodes=1 --gres=gpu:4",
           worker_init="module load cuda",
           launcher=SingleNodeLauncher(),
       ),
   )

With this setup, 10 nodes give 40 slots. Parsl only requests the blocks that waiting members need, so
an ensemble of 8 uses 2 nodes. ALF prints a warning if there are fewer slots than members; they still
all train, but take longer. One-node blocks have these benefits:

- A lost node only affects its own members, and Parsl can request a replacement block. Slurm normally
  ends a whole job when one of its nodes fails, so losing a node from a 10-node block would stop every
  member in it.
- Members start on whichever nodes are ready, and the rest take free slots as they open. Training
  finishes even if some blocks never start, but more slowly.

All blocks stay allocated until the slowest member finishes, and then for ``max_idletime``, because
Parsl's default ``simple`` strategy only scales in when no tasks remain. ``strategy="htex_auto_scale"``
in the Parsl ``Config`` can release idle blocks sooner, but it applies to every executor, and it hasn't
been qualified with ALF.

Each block's walltime must cover a member's training, including any retry, and blocks may start at
different times. A member that takes longer than the walltime fails on every attempt.

By default, Parsl permanently stops an executor once every block it has started so far has failed, for
example when the only block hits a node failure. ALF then stops; see :ref:`parsl-releasing-allocations`.
To survive isolated failures, use ``windowed_error_handler``, which only gives up after the last three
blocks have all failed. Cancelled and timed-out blocks don't count:

.. code-block:: python

   from parsl.jobs.error_handlers import windowed_error_handler

   HighThroughputExecutor(
       label="alf_ML_executor",
       block_error_handler=windowed_error_handler,
       ...
   )

A single multi-node block (``nodes_per_block=10``, ``max_blocks=1``) starts every node together, but
may queue for longer. It needs ``SrunLauncher``, because ``SingleNodeLauncher`` starts workers on only
the first node, and members would run a few at a time.

To train each member on :ref:`several GPUs <ml-multigpu-members>`, give each worker that many GPUs. With
``gpus_per_member`` 4 on 4-GPU nodes, use one worker per node and no ``available_accelerators``, so
the worker sees the whole node, and set ``max_blocks`` to the number of members:

.. code-block:: python

   HighThroughputExecutor(
       label="alf_ML_executor",
       max_workers_per_node=1,
       provider=SlurmProvider(nodes_per_block=1, max_blocks=4, ...),
   )

With ``gpus_per_member`` 2, use two workers per node and list each worker's GPUs, with
``max_workers_per_node=2`` and ``available_accelerators=["0,1", "2,3"]``. Parsl sets each worker's
``CUDA_VISIBLE_DEVICES`` to its entry. ALF stops with an error if a worker would get fewer GPUs than
``gpus_per_member``.

.. _parsl-running-at-scale:

Running at Scale
^^^^^^^^^^^^^^^^

When many nodes start at once, every worker importing PyTorch, CuPy, and their libraries from shared
storage opens thousands of small files, which slows each start and loads the filesystem's metadata
servers. Package caches that default to your home directory are written by every node at once. The
CPP and AcAc Parsl templates can avoid both; everything below is off unless you set it.

**A node-local environment.** Pack the environment once into an archive on shared storage, and set
``ALF_ENV_TARBALL`` to it (``ALF_QM_ENV_TARBALL``, ``ALF_ML_ENV_TARBALL``, and
``ALF_SAMPLER_ENV_TARBALL`` override it for one role). Each block's setup unpacks it into
``$ALF_NODE_LOCAL_DIR`` (by default ``$TMPDIR``, or ``/tmp``) and starts its workers from that copy. A
node unpacks a given archive once: blocks landing on the same node take turns under a lock, and later
blocks reuse the copy. A rebuilt archive, with a different size or modification time, unpacks into a
new directory. If unpacking fails, the block stops with a message instead of starting workers. The
coordinator and Parsl's interchange keep using ``ALF_<ROLE>_PYTHON``, and ALF's own source still comes
from ``ALF_SOURCE``, so a code change needs no new archive.

The environment in the archive must still work after being moved:

- **conda:** ``conda pack -n alf -o alf-env.tar.gz``, with ``conda-pack`` installed in another
  environment. Its ``bin/conda-unpack`` runs after unpacking.
- **uv:** a plain virtual environment can't be moved, since it refers to its base interpreter by an
  absolute path, which would still load the standard library from shared storage. Instead, install a
  uv-managed Python into the directory you will archive, with ``UV_PYTHON_INSTALL_DIR``, install
  ALF's dependencies into that interpreter with ``uv pip install --python <it>``, and archive the
  directory. Set ``ALF_ENV_PYTHON`` to the interpreter's path inside the archive, such as
  ``cpython-3.12.12-linux-x86_64-gnu/bin/python3``.

Unpack the archive on the node, not onto shared storage, and check that its interpreter imports ALF's
dependencies there before a long run. For multi-node blocks with ``SrunLauncher``, add an ``sbcast`` of
the archive to ``ALF_<ROLE>_WORKER_INIT`` and point ``ALF_ENV_TARBALL`` at the copy, so each node reads
its own.

**Node-local caches.** ``ALF_NODE_LOCAL_CACHES=1`` points ``XDG_CACHE_HOME``, ``MPLCONFIGDIR``,
``NUMBA_CACHE_DIR``, ``TRITON_CACHE_DIR``, and ``CUPY_CACHE_DIR`` at node-local storage. Each new node
then compiles GPU4PySCF's CuPy kernels once; to reuse compiled kernels, set ``ALF_CUPY_CACHE_DIR`` to
a cache filled beforehand. It also sets ``HDF5_USE_FILE_LOCKING=FALSE``, which is safe for ALF, since
it writes each shard once and renames it into place, and readers never see a partial one.

.. _parsl-gpu4pyscf-workers:

GPU4PySCF Workers
^^^^^^^^^^^^^^^^^

GPU4PySCF is different: each QM worker must see exactly one GPU, as its local device
0, before CUDA starts. This applies to ``gpu4pyscf_excited_state_task`` and to ``pyscf_task`` with
``"backend": "gpu"``. A worker that can see several GPUs marks every label as failed
with a device error, whatever its rank, including when only one worker is running.

When ALF has the whole node, use Parsl's ``available_accelerators``, which pins each
worker to one GPU. ``gpus_per_node`` has no effect on this:

.. code-block:: python

   HighThroughputExecutor(
       label="alf_QM_executor",
       max_workers_per_node=4,
       available_accelerators=4,
       provider=provider,
   )

A number names GPUs from 0 upward, which can be wrong when the scheduler gave you
only some of a node's GPUs. In that case, start workers with
``examples/CPP_excited_gpu4pyscf/qm_worker_pool.py``, which looks up the allocated GPUs
and binds one to each worker; see :doc:`../examples/cpp_excited_gpu4pyscf`.

Globus Compute Endpoints
------------------------

ALF can also target `Globus Compute <https://www.globus.org/compute>`_
endpoints through Parsl. In this pattern, ``parsl_configs.py`` uses Parsl's
``GlobusComputeExecutor`` instead of a scheduler-backed executor, while keeping
the ALF executor labels such as ``alf_QM_executor``, ``alf_ML_executor``, or
``alf_sampler_executor``. This can be useful for multisite runs where different endpoints
provide different resources, or for users who do not have a local scheduler.

The Parsl documentation includes a
`Globus Compute multisite example <https://parsl.readthedocs.io/en/stable/userguide/configuration/examples.html#globus-compute-multisite>`_
with additional endpoint setup details. Below is a bare-bones Parsl configuration for using ALF with Globus Compute.

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
provides ``alf_ML_executor`` or ``alf_sampler_executor``. Note that the Globus Compute
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
resources. ``alf_ML_executor`` runs one ML training task per GPU allocation, or one
task per ensemble member with :ref:`multi-node training <parsl-multinode-training>`.
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
nodes. For unmasked sampler workers, ``gpus_per_node`` specifies the number of
GPUs to distribute over, not the worker count. More workers than visible GPUs
share devices by rank; scheduler/Parsl visibility takes precedence.

.. _parsl-releasing-allocations:

Releasing Allocations
---------------------

When ALF exits, it stops Parsl's interchange and cancels its blocks. This includes the end of a
stage-test command such as ``--test_ml``, an error, Ctrl-C, and ``SIGTERM`` from ``scancel`` or a job's
time limit. ``SIGKILL`` can't be handled, and neither can a kill that arrives before cleanup finishes.
After either one, the interchange can keep running and keep nodes allocated until walltime, so check
``squeue`` and cancel any leftover blocks yourself.

ALF also stops with an error if Parsl permanently stops the builder, sampler, QM, or ML executor, because
every task on it would fail from then on. The error includes Parsl's reason. Status is saved first, but
labels that finished since the last store are lost. Restart ALF once the cause is fixed. A failed standby
executor doesn't stop ALF. Blocks still running on the failed executor stop at its heartbeat limit, not
immediately.

Sizing Executors For Batching
-----------------------------

A Parsl worker runs one ALF task, whatever its width. With batching, one task can
hold many structures, so size executors by tasks rather than structures:

* **Sampler:** with the ``torchsim`` or ``alchemi`` engines, one worker per GPU is
  a natural starting point, since each task packs its trajectories onto one device.
* **QM, native batches:** a PySEQM batch runs as one task, so on GPU nodes it also
  usually needs one worker per GPU.
* **QM, sequential batches:** members run one after another on one worker, so
  walltime must cover ``batch_size`` calculations.

Each worker imports its task's backend, so ``worker_init`` must set up an environment
that has it, such as TorchSim for sampler workers. Different executors can use
different environments. See :doc:`batching` for how ALF's queue limits relate to
batch sizes.

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


The master configuration selects a Parsl configuration by import string, usually
``parsl_configs.config_1node`` from the ``parsl_configs.py`` file in the run
directory. The ``examples/*/parsl_configs.py`` templates show the expected structure,
and the older profiles in ``alframework/parsl_resource_configs`` are kept for
reference only, with out-of-date accounts and executor layouts. In either case, cluster
partition names, accounts, modules, walltimes, and launcher choices must be
adapted for the resources you plan to use.
