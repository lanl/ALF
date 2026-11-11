Sizing GPU Use
==============

How large a training batch fits on your GPUs, how many GPUs each ensemble member needs, and how many
trajectories each sampler GPU should run depend on the network, the system size, and the GPU. ALF
includes a benchmark that measures these on the machine you'll run on and suggests settings:

.. code-block:: bash

   python -m alframework.benchmark

Run it on a GPU node, from an ALF clone or an install, in the environment ALF's ML and sampler workers
use. With no options it measures two HIP-NN networks on a 96-atom water cluster, which takes a few
minutes. ``--device cpu`` checks that it works anywhere, but CPU numbers mean nothing for GPU settings.

For Your Own Run
----------------

Point it at a run directory to measure that run's network on its structure:

.. code-block:: bash

   python -m alframework.benchmark --run-dir /path/to/run

It reads ``master_config.json`` and the ML and sampler configurations it names: the network settings,
the number of electronic states (from the ``E0``, ``E1``, ... properties), the ensemble size, and the
training and sampler batch sizes, which it adds to the sizes it tries. It uses the run's
``reference.xyz`` when there is one; ``--system`` chooses another structure: ``water`` (with
``--atoms``), ``cpp6`` to ``cpp12`` from a clone, or any file ASE reads. ``--profiles`` compares
networks, for example ``--profiles run hipnn100 hiphop64``.

What It Measures
----------------

Each measurement runs in its own process, so one that runs out of GPU memory is recorded as ``oom``
and the others continue.

Training
   For each network and batch size, the time of one training step and one evaluation step of the
   graph ALF trains, and the peak GPU memory. The labels are random, since time and memory don't
   depend on them.

Sampling
   For each number of trajectories, the time for the whole ensemble to compute energies and forces on
   that many structures, which is the model's share of one MD step on a sampler GPU, and the peak
   memory with every member loaded. Members are written untrained and loaded with ALF's sampler
   loaders; an untrained model costs the same as a trained one.

Reading The Suggestions
-----------------------

The benchmark prints its measurements, then suggested settings, and writes everything to
``gpu_sizing.json`` (``--output``).

``batch_size``
   The smallest training batch whose throughput is within 10% of the best that fits in 80% of GPU
   memory (``--memory-fraction``). Larger batches gain little and leave less room. It also gives the
   epoch time for ``--train-rows`` structures, and whether ``--epochs`` epochs fit a
   ``--walltime-hours`` allocation; a training that runs out of time starts over.

``gpus_per_member``
   With ``--run-dir``, when the run's batch size doesn't fit on one GPU, the fewest GPUs that split it
   into shares that did fit. See :ref:`ml-multigpu-members`. This is an estimate: GPUs also spend time
   exchanging gradients, which a single-GPU measurement can't show.

Trajectories per sampler GPU
   The smallest number whose throughput is within 10% of the best that fits, as
   ``"batching": {"batch_size": ...}``, and ``parallel_samplers`` for the sampler GPUs
   (``--sampler-gpus``, by default the GPUs visible). See :doc:`batching`.

What It Doesn't Measure
-----------------------

* QM labeling, whose cost depends on the QM code and its settings.
* Communication between GPUs when one member trains on several.
* The integrator and screens of a full MD step, which cost little next to the model.
* Accuracy: a faster network or a larger batch may train differently.
