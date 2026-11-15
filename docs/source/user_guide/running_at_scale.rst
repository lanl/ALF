.. _running-at-scale:

Running at Scale
================

When many nodes start at once, every worker importing PyTorch, CuPy, and their libraries from shared
storage opens thousands of small files, which slows each start and loads the filesystem's metadata
servers. Package caches that default to your home directory are written by every node at once. The
CPP and AcAc Parsl templates can avoid both; everything below is off unless you set it.

Node-Local Environment
----------------------

Pack the environment once into an archive on shared storage, and set
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
dependencies there before a long run.

A provider's ``worker_init`` runs once per block, on its first node. For blocks of several nodes whose
launcher starts workers on all of them (``SrunLauncher``), pass ``all_nodes=True`` to ``worker_setup`` (or
``setup_from_environment``): the setup then runs on every node of the block as one ``srun`` step, and a
node that cannot unpack stops the block. ``all_nodes`` can also be the command that runs one task per
node, such as ``mpiexec -ppn 1``. Reading the archive from shared storage is usually fine: on Frontier,
unpacking a 6.5 GB conda-pack archive took about 72 s per node, whether it was read from Lustre or from a
node-local ``sbcast`` copy, since decompressing dominates.

Node-Local Caches and HDF5
--------------------------

``ALF_NODE_LOCAL_CACHES=1`` points ``XDG_CACHE_HOME``, ``MPLCONFIGDIR``,
``NUMBA_CACHE_DIR``, ``TRITON_CACHE_DIR``, and ``CUPY_CACHE_DIR`` at node-local storage. Each new node
then compiles GPU4PySCF's CuPy kernels once; to reuse compiled kernels, set ``ALF_CUPY_CACHE_DIR`` to
a cache filled beforehand. It also sets ``HDF5_USE_FILE_LOCKING=FALSE``, which is safe for ALF, since
it writes each shard once and renames it into place, and readers never see a partial one.

.. _running-at-scale-small-files:

Fewer Small Files
-----------------

A parallel filesystem handles every file creation on a metadata server, and a run that writes one file
per structure sends all of them there at once. On Lustre, a directory's files and subdirectories stay
on that directory's metadata server, and small files may be stored on it too (Data-on-MDT), so giving
each task its own subdirectory still loads the same server. On Frontier's Lustre, single file writes
during sampling stalled for 7 to 25 minutes, at 2, 8 and 16 nodes, with and without per-task
subdirectories. A stage waits for its slowest task, so one stall delays the whole stage, and stalls
become more likely as the number of nodes grows.

- Set the sampler's ``metadata_files`` to ``"per_task"``, one file per task instead of one per
  trajectory, or to ``"none"``. The metadata still returns to the coordinator with each structure
  (see :doc:`samplers`).
- Set ``trajectory_summary_path`` so progress plots read one summary file, not every metadata file
  (see :doc:`monitoring`).
- On Lustre, create the run's output directories, such as ``meta_dir`` and
  ``ALF_TASK_TIMELINE_DIR``, before the run, spread over several metadata servers:
  ``lfs mkdir -c 8 <dir>``. ALF keeps a directory that already exists, and new files and
  subdirectories are then spread over those servers. ``lfs getdirstripe <dir>`` shows the layout.

.. _running-at-scale-coordinator-memory:

Coordinator Memory
------------------

The coordinator keeps every sampled structure in memory until its QM task finishes, so its memory
grows with the number of structures in a cycle: on the order of 100 KB per structure with the default
settings, mostly ASE atoms and sampler metadata, at its peak. With thousands of samplers that reaches
tens of gigabytes, on what is often a shared login node. Finished labels don't add to it: each pass of
the loop moves them to ``held_labels_dir`` on disk (see :ref:`configuration-restarts`), and a store
reads them back one held piece, at most 1,000 labels, at a time.

- Set the sampler's ``returned_metadata`` to ``"compact"`` with ``metadata_files`` ``"per_task"``:
  structures return without the per-trajectory arrays the metadata file already keeps (see
  :doc:`samplers`).
- Watch the coordinator's resident memory over a smaller run and scale it by the number of
  structures per cycle before going larger.

.. _running-at-scale-coordinator-cpu:

Coordinator CPU
---------------

The coordinator is one Python process, and it serializes every task's input structures before Parsl sends
them. Parsl does this with dill, a pure-Python pickler: one QM task of 256 sampled structures took about
1.4 s, so with thousands of such tasks the coordinator spends most of its time serializing, and the
last tasks of a stage start minutes after the first. Two settings cut this, together by about 200 times
for that task:

- The sampler's ``returned_metadata`` ``"compact"`` (see :ref:`running-at-scale-coordinator-memory`):
  fewer objects per structure, about 10 times faster to serialize.
- The master's ``parsl_data_serializer`` ``"pickle"`` (see :doc:`configuration`): plain pickle first, dill
  only as the fallback, about 20 times faster again.

.. _running-at-scale-huge-pages:

Transparent Huge Pages on the Coordinator's Host
------------------------------------------------

At scale the coordinator allocates large blocks of memory: harvested results, held-label pieces, and the
arrays of each stored shard. On a host whose transparent huge pages are always on
(``cat /sys/kernel/mm/transparent_hugepage/enabled`` shows ``[always]``), the kernel backs these with huge
pages, and on a busy shared login node that made large allocations 30 to 60 times slower: filling a new
1.3 GB array took 39 s instead of 0.7 s, and a 400 MB one 4.4 s instead of 0.2 s. Allocations of about
100 MB were not affected, so small runs don't show it.

A process can opt out for itself and every process it starts, without administrator rights: call
``prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)`` and then ``exec`` the coordinator. The setting is kept across
``exec`` and inherited by child processes, such as Parsl's interchange. For example, a wrapper
``thp_off.py``:

.. code-block:: python

   import ctypes, os, sys
   ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # 41 = PR_SET_THP_DISABLE
   os.execvp(sys.argv[1], sys.argv[1:])

started as ``python thp_off.py python -m alframework --master master_config.json``. ``grep THP_enabled
/proc/<pid>/status`` shows ``0`` for the coordinator and its children once it applies.

.. _running-at-scale-batched-builders:

Batched Builders
----------------

Each builder task is a Parsl task that the coordinator creates, serializes and tracks. With one structure
per task, a run with 262,144 trajectories submitted 262,144 builder tasks: the coordinator spent about 5
minutes of CPU creating them, and sampling started about 2 minutes after its workers did, while they worked
through the builders. A builder that takes ``moleculeids`` and returns a list, with
``maximum_builder_structures`` in the master configuration (see :doc:`builders`), divides that by the batch
size. Matching it to the sampler's batch size (see :doc:`batching`) gives one builder task per sampler task.
A failed builder task loses its whole batch, so a batched builder should draw a structure that fails again
rather than raise.

.. _running-at-scale-storing-labels:

Storing Labels
--------------

Before each training, the coordinator stores the labels held since the last one as a new HDF5 shard, on its
main thread, and then submits the training. The time grows with the number of labels: about 7 minutes for
262,144 labels of 201 atoms. Two things dominated it:

- Compression: shards are written with gzip level 6 on one thread. For coordinates and forces it saved only
  15 to 25% of the space, at about three times the time of writing them uncompressed.
- Large allocations: storing builds one array per formula and property, which is slow on a host with
  transparent huge pages always on (see :ref:`running-at-scale-huge-pages`). Without them, storing 65,536
  labels went from 69 to 51 s.

Plan for this pause between the end of labeling and the start of training in large runs.

.. _running-at-scale-training:

Training With Many Labels
-------------------------

Two parts of each training grow with the number of labels, whatever the number of GPUs: every training rank
loads the whole HDF5 database before the distributed sampler gives it its share, and the final test after
training runs on one GPU over the whole test split. With 8 GPUs per member, the time outside the epochs grew
roughly linearly, from about 1.3 minutes with 2,048 labels to about 21 minutes with 262,144.

Data parallelism (see :ref:`ml-multigpu-members`) keeps the time per epoch flat only while a member's GPUs
grow with the labels. A member's ranks start on one node, so a member uses at most one node's GPUs; beyond
that, each epoch takes longer in proportion to the labels per GPU (four times the labels per GPU, four times
the epoch). Size the ML job's walltime for both.

Related
-------

- :ref:`parsl-multinode-training`: one-node blocks, so a lost node only affects its own ensemble members.
- :ref:`ml-multinode-training` and :ref:`ml-multigpu-members`: spreading an ensemble over nodes, and
  training one member on several GPUs.
- :doc:`batching`: several structures per task, so fewer, larger tasks.
- :doc:`gpu_sizing`: measuring batch sizes and GPUs per member on your own hardware.
