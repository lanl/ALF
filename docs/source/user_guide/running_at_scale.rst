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

Related
-------

- :ref:`parsl-multinode-training`: one-node blocks, so a lost node only affects its own ensemble members.
- :ref:`ml-multinode-training` and :ref:`ml-multigpu-members`: spreading an ensemble over nodes, and
  training one member on several GPUs.
- :doc:`batching`: several structures per task, so fewer, larger tasks.
- :doc:`gpu_sizing`: measuring batch sizes and GPUs per member on your own hardware.
