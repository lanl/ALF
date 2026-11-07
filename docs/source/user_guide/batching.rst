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

Choosing Widths
---------------

You choose each width separately, and changing one does not change the others.
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

A QM config's ``batching`` block also needs a ``mode``, which says how the batch
is labeled:

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
       GPU4PySCF, ``pyseqm_excited_state_task``, and ``teacher_task``.

ALF stops at startup if the mode doesn't match the QM task. With ``sequential``,
make sure ``batch_size`` times the runtime of one calculation fits in your
executor's walltime.

If one member fails, ALF keeps the labels from the others. The failed member is
marked unconverged and isn't stored. ALF doesn't retry it, except that the PySEQM
tasks fall back to solving members alone when a whole batch solve fails.

Sampler Batches
---------------

Sampler batching needs ``mlmd_trajectories_task`` and
``"driver": {"strategy": "batched"}`` alongside the ``batching`` block. ALF stops
at startup if a single-structure sampler task is given these settings. See
:ref:`samplers-multi-trajectory`.

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
large as ``batch_size``. Otherwise ALF stops building structures before a batch
can fill:

* ``target_queued_QM`` for QM batches. ALF checks this at startup.
* ``parallel_samplers`` for sampler batches. ALF doesn't check this, so the run
  stalls without an error.

Checking Your Settings
----------------------

ALF prints the batching it will use at startup, for example:

.. code-block:: text

   Sampling: torchsim engine, batches of up to 32 structures (timeout, 5 s)
   QM labeling: native batches of up to 64 structures (timeout, 5 s)

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
* :doc:`Labeling module <../api_documentation/alframework.tools.labeling>`
* :doc:`Sampler coordinator module <../api_documentation/alframework.tools.sampler_coordinator>`
