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
node, such as ``mpiexec -ppn 1``. Reading the archive from shared storage is usually fine: on one HPC system,
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
each task its own subdirectory still loads the same server. On one large Lustre filesystem, single file writes
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
- Set the sampler's ``returned_momenta`` to ``false``: structures return without their MD momenta.
  Serialized for a QM task, a returned 201-atom structure with compact metadata took 12.0 KiB, 4.7 KiB of
  it momenta; Parsl's interchange holds a serialized copy of every queued QM task.
- Progress plots don't add to it: each update runs in its own new process rather than a fork of the
  coordinator, which would copy much of a large coordinator's memory (see :ref:`samplers-progress-plots`).
- Watch the coordinator's resident memory over a smaller run and scale it by the number of
  structures per cycle before going larger. Setting ``coordinator_memory_sample`` (see :doc:`configuration`)
  records it, with that of Parsl's interchanges, throughout the run.

.. _running-at-scale-data-by-reference:

Finite workflows using the batched MLMD sampler, immediate frame selection and native batched QM
can hand sampled frames to QM by reference, through files on a shared filesystem, instead of through
the coordinator. This is off unless the master configuration turns it on:

.. code-block:: json

   "data_by_reference": {"directory": "sampling-frames", "sampler_frames": true}

``directory`` is relative to ``master_directory`` unless absolute. The sampler
writes one pickle file per batch with eligible frames and returns trajectory summaries and frame
references. The coordinator counts completed frames toward the normal sampler wave, and QM workers
read the original frames once the wave opens. Both worker roles need access to the same absolute paths.
Set ``maximum_stored_labels`` for a finite workflow and size the spool for the largest in-flight
wave, queued cleanup, and any abandoned files. A receipt publishes a closed, flushed batch file.
ALF keeps each file through QM retries and until its last frame's outcome is recorded. Bounded
HDF5 storage makes successful labels durable as held parts. With legacy storage their input files
stay until the training shard is published. The coordinator retires files in its normal polling loop
and cleans the invocation directory after workers exit. Restart starts a fresh spool and resumes
from persisted status and labels. Remove abandoned directories after the workers have exited.
For debugging exact QM inputs, add ``"keep_files": true`` to ``data_by_reference``. The coordinator
preserves all staged batches and prints the path, file count, and bytes at normal exit. Size a retained
spool for the full debug run. The earlier names of these settings, ``sampler_frame_spool`` and
``keep_sampler_frames``, are refused with a message naming ``data_by_reference``.

Durable labels become visible through the atomically published HDF5 snapshot. HIPPYNN training and
fine-tuning read its backing parts directly in serial and parallel, keeping at most one part open per
group reader, and stop with an error if a part is missing, rather than reading HDF5's fill values.
``ALF_H5_READ_WORKERS`` controls group parallelism. Keep the snapshot and its referenced parts
together when moving a run.

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

Training members allocate the same way: reading the database pads every group into arrays of the full
dataset, then converts them, and the ranks load their setup. Members are started by Parsl workers, not by
the coordinator, so they don't inherit its setting. Set ``ALF_THP_DISABLE=1`` in the ML workers'
environment (``worker_init``): each member's process, and each node process of a member on several nodes,
then opts out for itself and its ranks. On the shared login node above, reading 2,097,352 structures of
up to 201 atoms (a 15 GB shard) took 823 s with huge pages and 133 s without.

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

``label_storage_batch_size`` (see :doc:`configuration`) removes most of it: finished labels are written as
HDF5 parts of at most that many structures while QM runs, and at the training boundary ALF publishes one
virtual shard over them. ``label_h5_layout`` chooses a lossless layout for those writes: lzf or a lower gzip
level, HDF5's byte-shuffle filter, and chunks of whole frames instead of h5py's guess, which for 201-atom
structures was 1,376 frames by 7 atoms by 1 coordinate. Replaying the 2,097,152 labels of a large run on one
node, with transparent huge pages off and every arm giving the same stored values:

.. list-table::
   :header-rows: 1

   * - Storage
     - Writing while QM runs
     - At the training boundary
     - Peak memory
     - Size
     - Read back
   * - Held pickles, then one shard (the default)
     - 414 s
     - 1,605 s
     - 39.3 GiB
     - 22.7 GiB held
     - 95 s
   * - Parts of 8,192, gzip level 6
     - 1,328 s
     - 7 s
     - 1.4 GiB
     - 14.5 GiB
     - 183 s
   * - Parts of 8,192, lzf, shuffle, chunks of 1,024 frames
     - 578 s
     - 7 s
     - 1.3 GiB
     - 14.0 GiB
     - 47 s
   * - Parts of 8,192, gzip level 1, shuffle, chunks of 1,024 frames
     - 821 s
     - 7 s
     - 1.3 GiB
     - 12.6 GiB
     - 64 s

The shards stay HDF5 with the same datasets, dtypes and values, so trainers and other readers read them
unchanged. Writing parts happens on the coordinator as results arrive, so choose a layout whose writing keeps
pace with your QM wave; keep the default where other tools expect gzip level 6.

Each pass of the coordinator's loop writes one part, then more while a full part's worth of finished labels is
waiting, for up to 30 seconds, so the loop keeps answering while labels don't pile up behind it. Writing only one
part per pass, as ALF did before, fell behind at scale: with 1,024 nodes a pass took about 5 s, and the 257 parts
of 2,097,152 labels were still being written 24 minutes after QM finished. A part of 8,192 structures takes about
1.7 s to write with lzf, most of it building the arrays.

.. _running-at-scale-training:

Training With Many Labels
-------------------------

Every member reads the whole HDF5 database and screens every structure for close atom pairs before it
trains, and tests the best model on every split afterwards, so these grow with the number of labels. The
screen uses ALF's batched neighbor search, once per training, and the final test is split over the member's
GPUs (see :ref:`ml-multigpu-members`). For 262,344 structures of 201 atoms on 8 GPUs per member, a member
had its data loaded, screened and set up for training 70 s after it started, and its final test, plots and
saving took 1.6 minutes; it trained in 6.75 minutes with 2 epochs. Before these changes the screen ran twice, about 6
minutes each, and the final test ran on one GPU, so the same member took 23.5 minutes.

The training log reports how long each step of this preparation took, in lines starting with
``Preparation:``: reading the HDF5 data, screening close atoms, conversion and outlier removal, splitting,
E0 values, and the setup written for the ranks. Reading a large shard decompresses each dataset on one
thread, so ALF reads the groups of files of 256 MiB or more in parallel processes, one per available core
up to one per group; the batches are the same, in the same order. The ML configuration's ``h5_read_workers``
sets the number (``1`` reads serially), or ``ALF_H5_READ_WORKERS`` without it. For the 15 GB shard above, with huge pages off, 8 processes took 93 s and one 133 s,
and every array was identical. Turning off huge pages (see :ref:`running-at-scale-huge-pages`) mattered
more.

A shard published over parts (``label_storage_batch_size``) is a small file of virtual datasets: ALF counts the
parts it references when deciding to read in parallel, and each process reads the parts' datasets directly
rather than through HDF5's virtual layer, which was about 6 times slower. The arrays are the same. Before this,
a member of a 1,024-node run read its 2,097,152 labels serially through the virtual layer for 17.5 minutes;
reading 16 of those parts (128,000 structures) went from 176 s to between 4 and 25 s, depending on the load of
the node.

Data parallelism (see :ref:`ml-multigpu-members`) keeps the time per epoch flat only while a member's GPUs
grow with the labels. On one node, each epoch takes longer in proportion to the labels per GPU (four times
the labels per GPU, four times the epoch). ``member_nodes`` spreads a member over several nodes. For one
member with 1,468,385 training structures of up to 201 atoms, on nodes of 8 GPUs:

.. list-table::
   :header-rows: 1

   * - Nodes
     - GPUs
     - Batch, total (per GPU)
     - Steps per epoch
     - Epoch
   * - 1
     - 8
     - 512 (64)
     - 2,868
     - 721 s
   * - 2
     - 16
     - 1,024 (64)
     - 1,434
     - 371 s
   * - 4
     - 32
     - 2,048 (64)
     - 717
     - 191 s
   * - 2
     - 16
     - 512 (32)
     - 2,868
     - 421 s
   * - 4
     - 32
     - 512 (16)
     - 2,868
     - 275 s

Keeping the batch per GPU gave 97% and 94% of a linear speedup; keeping the total batch, 86% and 66%,
because each GPU processed fewer structures per second at smaller batches (270, 233 and 180 per GPU per
second). Every rank's median step time agreed within 0.3%, so no rank held the others back, and the
gradient all-reduce wasn't the limit. A larger total batch means fewer optimizer steps per epoch, so check
accuracy before changing it. Reading and preparing the data (about 7.5 minutes here, with huge pages on) and the final test and
plots (about 6 to 11 minutes) don't get faster with more GPUs, and each rank held a full copy of the data
(about 24 GB) unless the ML configuration's ``shared_training_data`` is ``true``. Size the ML job's walltime for
all of these.

For ALF's standard Lightning loss and plot graphs, set the ML configuration's ``stream_final_evaluation`` to
``true`` (or ``ALF_LIGHTNING_STREAM_FINAL=1`` in the ML workers' environment) to bound memory during final
evaluation. Each rank accumulates sums and element counts so RMSE
and combined losses use the whole split. For histogram plots, ranks save only the plotted columns temporarily
to ``ALF_NODE_LOCAL_DIR`` or the system temporary directory, then use common bin edges and return bin counts.
Unrecognized loss and plot graphs use the regular whole-split evaluator. The option applies after training
has finished and the ranks have loaded the best checkpoint.

.. _running-at-scale-failed-gpu-tasks:

Failed GPU Tasks
----------------

A Parsl worker runs many tasks, one after another. ALF's tasks that use a GPU in the worker process (the MD
samplers, the teacher, PySEQM, PySCF and GPU4PySCF, and the reactive sampler) free what they held when they
fail, including PyTorch's and CuPy's cached GPU memory, before the worker takes its next task, and a trajectory
that fails keeps only its error message (``sampling_error``), not the exception. Ensemble members train in a
child process, so their memory is freed when it exits. Python otherwise keeps a failed task's objects, such as its MD engine
and the tensors of the evaluation that failed, until its cycle collector happens to run. In one 128-node run,
one batch of 256 trajectories ran its GPU out of memory, and its 63 GB stayed allocated: that worker then
failed its next five tasks, four within 0.04 s each, and, being free again first, took the tasks resubmitted
to replace the lost ones. Your own GPU tasks can do the same with
``alframework.tools.task_failure_cleanup.releases_memory_on_failure``, under ``@python_app`` and ``timed``.

The task timeline (see :doc:`monitoring`) records each task's peak GPU memory, to find what grew.

Related
-------

- :ref:`parsl-multinode-training`: one-node blocks, so a lost node only affects its own ensemble members.
- :ref:`ml-multinode-training` and :ref:`ml-multigpu-members`: spreading an ensemble over nodes, and
  training one member on several GPUs.
- :doc:`batching`: several structures per task, so fewer, larger tasks.
- :doc:`gpu_sizing`: measuring batch sizes and GPUs per member on your own hardware.
