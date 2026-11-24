Batching
========

By default, ALF runs one structure per task: a builder task makes one structure,
a sampler task runs one trajectory, and a QM task labels one structure. Batching
puts several structures into one task. It is useful when:

* a dynamics engine can advance many trajectories together on one GPU,
* a QM code can solve many molecules in one call, or
* many short jobs would spend too long waiting on the scheduler.

Batching is opt-in. Without the settings on this page, ALF submits one structure
per task.

To measure how many structures a GPU should take, run the benchmark in :doc:`gpu_sizing`.

.. _batching-sizing:

Sizing a Run
------------

A run's size is several settings that have to agree with each other and with the
Parsl configuration: the sampler and QM batch sizes, the builder width, and the queue
limits below. The master configuration's ``sizing`` block states the intent instead,
and ALF fills them in:

.. code-block:: json

   "sizing": {
     "trajectories_per_sampler_task": 256,
     "structures_per_qm_task": 256,
     "labels_per_round": 16384
   }

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Setting
     - Filled in as
   * - Sampler ``batching.batch_size``
     - ``trajectories_per_sampler_task``. A ``batching`` block without ``batch_size``
       keeps its other fields.
   * - ``maximum_builder_structures``
     - ``trajectories_per_sampler_task``, when the builder takes ``moleculeids``, so one
       builder task fills one sampler task.
   * - ``parallel_samplers``
     - The sampler tasks the Parsl configuration runs at once (see
       :ref:`the capacity report <batching-capacity>`) times ``trajectories_per_sampler_task``, which keeps
       every sampler slot busy. ``sampler_tasks_at_once`` replaces the Parsl count, and
       is needed when the executors don't say.
   * - QM ``batching.batch_size``
     - ``structures_per_qm_task``. The mode comes from the QM task unless given.
   * - ``target_queued_QM``
     - ``qm_tasks_at_once`` times ``structures_per_qm_task``, which keeps that many QM
       tasks queued or running. Without ``qm_tasks_at_once``, ``labels_per_round``.
   * - ``save_h5_threshold``
     - ``labels_per_round`` minus one: a retraining starts once more than
       ``save_h5_threshold`` labels are held, so once ``labels_per_round`` are.

Each key is optional and fills in only its own settings. ``trajectories_per_sampler_task``
needs a sampler task that takes ``molecule_objects``, such as ``mlmd_trajectories_task``.
``qm_tasks_at_once`` needs ``structures_per_qm_task``. Set it to size the QM queue for the
QM hardware, for example the QM task slots of the capacity report, apart from how often
the run retrains: ``labels_per_round`` then sets only ``save_h5_threshold``.
A setting also given in its own place must have the same value; a different one stops
the coordinator, naming both. ``minimum_QM``, ``bootstrap_set``,
``maximum_completed_QM`` and ``maximum_stored_labels`` stay yours. The coordinator
prints what it derived, for example:

.. code-block:: text

   Sizing: sampler batching.batch_size 256, maximum_builder_structures 256, parallel_samplers 16,384
   (64 sampler tasks at once), QM batching.batch_size 256, target_queued_QM 16,384, save_h5_threshold 16,383
   (retrain once 16,384 new labels are held)

The block is read once, when the coordinator starts; its values apply to every reload of
the configuration files, and an edit to it waits for a restart.

Choosing Widths
---------------

Without the ``sizing`` block, you choose each width separately, and changing one does not change the others.
Splitting or combining structures into tasks does not change their molecule IDs.

.. list-table::
   :header-rows: 1
   :widths: 36 36 28

   * - Width
     - Setting
     - File
   * - Structures per builder task
     - ``maximum_builder_structures``
     - Master config. The builder must return several structures; see
       :doc:`builders`.
   * - Trajectories per sampler task
     - ``batching.batch_size``
     - Sampler config. See :doc:`samplers`.
   * - Structures per QM task
     - ``batching.batch_size``
     - QM config. See `Labeling Batches`_.
   * - Trajectories compared when choosing frames to label
     - ``acquisition.selection.cohort_size``
     - Sampler config. See :doc:`acquisition`.

The ``batching`` Block
----------------------

Sampler and QM configs use the same block:

.. code-block:: json

   {
     "batching": {
       "batch_size": 32,
       "partial_policy": "timeout",
       "max_wait_seconds": 5.0
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``batch_size``
     - Required. The most structures one task can hold.
   * - ``partial_policy``
     - ``"timeout"`` (default) also submits partly filled batches once their
       oldest structure has waited ``max_wait_seconds``. ``"full_only"`` submits
       full batches only.
   * - ``max_wait_seconds``
     - How long a partly filled batch can wait under ``"timeout"``. The default
       is ``5.0``.

Full batches are submitted right away. Structures only share a batch if they are
compatible:

* All batches group structures with the same periodic boundary conditions.
* Sampler batches also share one model generation and, for excited-state
  sampling, one selected state.

Molecules of different sizes and compositions can otherwise share a batch.

Labeling Batches
----------------

A QM config's ``batching`` block can also give a ``mode``, which says how the batch
is labeled. Without one, ALF chooses it from the QM task: ``native`` for a task that
takes ``molecule_objects``, ``sequential`` for one that takes ``molecule_object``. The
startup line then says ``(mode from the QM task)``:

.. code-block:: json

   {
     "batching": {"mode": "native", "batch_size": 64}
   }

.. list-table::
   :header-rows: 1
   :widths: 16 44 40

   * - Mode
     - What happens
     - QM tasks
   * - ``native``
     - The QM code evaluates the whole batch in one call.
     - Tasks that accept ``molecule_objects``: ``pyseqm_task``,
       ``pyseqm_excited_state_batch_task``, and ``teacher_batch_task``.
   * - ``sequential``
     - One Parsl task calls a single-structure QM task on each member in turn.
       This is job packing: it reduces the number of scheduler jobs, but members
       still run one after another.
     - Tasks that accept ``molecule_object``: ORCA, Q-Chem, VASP, ASE calculators,
       GPU4PySCF, ``pyscf_task``, ``pyseqm_excited_state_task``, and ``teacher_task``.

ALF stops at startup if a given mode doesn't match the QM task. With ``sequential``,
make sure ``batch_size`` times the runtime of one calculation fits in your
executor's walltime.

If one member fails, ALF keeps the labels from the others. The failed member is
marked unconverged and isn't stored. ALF doesn't retry it, except that the PySEQM
tasks fall back to solving members alone when a whole batch solve fails.

Sampler Batches
---------------

Sampler batching needs ``mlmd_trajectories_task``. A ``batching`` block in its config
selects ``"driver": {"strategy": "batched"}``, which can also be written out; an
explicit ``"single"`` with a ``batching`` block is an error. ALF stops at startup if a
single-structure sampler task is given these settings. See
:ref:`samplers-multi-trajectory`.

.. _batching-excited-state-batches:

Excited-State Batches
~~~~~~~~~~~~~~~~~~~~~

An excited-state sampler batch holds trajectories on one selected state. By default,
``"state_selection": {"mode": "batch_cycle", ...}`` changes state with every structure,
so a refill after a batch finishes adds one structure to each state's batch, and those
batches leave partly filled when ``max_wait_seconds`` runs out. Give each state a block
of structures instead:

.. code-block:: json

   {
     "state_selection": {"mode": "batch_cycle", "states": [0, 1, 2, 3], "block_size": 16},
     "batching": {"batch_size": 16, "partial_policy": "timeout", "max_wait_seconds": 10}
   }

Set ``block_size`` to ``batch_size``, and ``maximum_builder_structures`` in the master
config to the same number, so that each builder task fills one batch. Keep
``parallel_samplers`` a multiple of it.

Queue Limits And Workers
------------------------

ALF's queue limits count structures, not tasks. This covers
``parallel_samplers``, ``target_queued_QM``, ``maximum_completed_QM``,
``bootstrap_set``, and ``save_h5_threshold``, so they keep the same meaning when
you turn batching on. Structures waiting for a batch to fill count as queued.

Parsl workers count tasks. With ``"parallel_samplers": 100`` and a sampler
``batch_size`` of 32, three or four sampler tasks run at once, so four sampler
workers are enough where one-per-task sampling could use up to 100. See
:doc:`parsl` for how to size executors.

With ``"partial_policy": "full_only"``, keep the matching queue limit at least as
large as ``batch_size``: ``target_queued_QM`` for QM batches, ``parallel_samplers`` for
sampler batches. Otherwise ALF would stop building structures before a batch could
fill, so it stops at startup instead.

Checking Your Settings
----------------------

ALF prints the batching it will use at startup, for example:

.. code-block:: text

   Sampling: torchsim engine, batches of up to 32 structures (timeout, 5 s)
   QM labeling: native batches of up to 64 structures (timeout, 5 s)

.. _batching-capacity:

It then sets what the run keeps in flight beside what its Parsl executors can run: each
stage's tasks at once (workers per node, times the worker pools the launcher starts on
each node, times nodes per block, times ``max_blocks``), times structures per task,
against its queue setting. Parsl itself counts one pool per node; ALF reads more from
``SrunLauncher``'s ``--ntasks-per-node`` (or ``--ntasks`` over the block's nodes),
``MpiExecLauncher``'s ``-ppn`` and ``AprunLauncher``'s ``-N``, as a layout with one pool
per GPU sets them, and from a ``pools_per_node`` attribute on a launcher of your own.
The same count of workers per node decides how many ensemble members train at once and
whether they would share a GPU (see :ref:`ml-multinode-training`), and a builder that
takes one ``moleculeid`` counts as one structure per task. For example:

.. code-block:: text

   Capacity:
     sampling: 16 tasks at once of up to 256 structures = 4,096; parallel_samplers 4,096
     builders: 256 structures per task, 16 tasks to fill sampling
     QM: 8 tasks at once of up to 256 structures = 2,048; target_queued_QM 2,048
     retraining: once more than 4,095 new labels are held
     ML: 8 ensemble members, 8 trained at once

A ``Warning:`` line follows for each mismatch it finds:

- sampler or QM task slots that ``parallel_samplers`` or ``target_queued_QM`` leaves idle;
- a ``parallel_samplers`` that isn't a multiple of the sampler batch size, so one batch
  at a time waits out ``max_wait_seconds``;
- builder tasks whose width differs from the sampler batch size;
- more ensemble members than ML slots, for tasks that train each member as its own app.

The report changes nothing; a stage whose executors don't say how many tasks they run
is reported as running an unknown number.

While running, the queue status for batched stages shows how many structures are
in tasks and how many are waiting:

.. code-block:: text

   Total Structures: 100 (3 tasks, 4 waiting for a batch)

The ``--test_sampler`` and ``--test_qm`` stage checks submit their one structure
immediately, so they work with any ``batch_size``.

Limits
------

* ``batching`` is read once at startup, so changing it requires a restart. Most
  other settings are re-read while ALF runs; see :doc:`configuration`.
* Structures waiting for a batch are held in memory. They aren't restored if the
  ALF process restarts.

Batching API Links
------------------

* :doc:`Batching module <../api_documentation/alframework.tools.batching>`
* :doc:`Labeling coordinator module <../api_documentation/alframework.tools.labeling_coordinator>`
* :doc:`Sampler coordinator module <../api_documentation/alframework.tools.sampler_coordinator>`
