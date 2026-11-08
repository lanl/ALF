Configuration
=============

ALF runs are configured with a small set of JSON files plus a Python Parsl
resource file. The master configuration is the entry point: it names the task
functions, points to the stage-specific config files, and defines the output
paths ALF uses while the active-learning loop runs.

The master configuration gives the name of each stage file, so you can name them as
you like. The examples often use backend names, such as ``mlmd_config.json``,
``hippynn_config.json``, and ``orca_config.json``. The roles are:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Purpose
   * - ``master_config.json``
     - Connects all workflow stages, output paths, queue thresholds, and Parsl
       resource configurations.
   * - ``builder_config.json``
     - Defines how initial structures are loaded or constructed.
   * - ``sampler_config.json``
     - Defines ML-driven sampling behavior and uncertainty thresholds.
   * - ``ml_config.json``
     - Defines model training, model loading, architecture, and data keys.
   * - ``qm_config.json``
     - Defines the external electronic-structure calculation.
   * - ``parsl_configs.py``
     - Defines local or cluster executors used by ALF tasks.

How ALF Uses These Files
------------------------

When you run ALF, pass the master configuration explicitly:

.. code-block:: bash

   python -m alframework --master master_config.json

ALF loads the master config first, then loads the builder, sampler, ML, and QM
configs named by ``*_config_path`` fields. The combined settings are used to
initialize Parsl tasks, queues, restart state, and output locations.

In practice, this means most workflow changes can be made by editing config
files rather than Python source code. Python changes are needed only when you
add a new task implementation, parser, builder, sampler, or model backend.

During the active-learning loop, ALF re-reads the master and stage configs on every
iteration, so most edits apply to tasks submitted afterward without a restart. A
config that fails to load is reported, and ALF keeps using the previous one.

These are read once at startup, so changing them requires a restart:

* The Parsl configuration, the task import strings, ``maximum_stored_labels``, and
  ``trajectory_summary_path``.
* ``batching`` in the sampler and QM configs.
* In the sampler config: ``driver``, ``acquisition``, ``model_mode``, ``state_selection``,
  ``gap_uncertainty``, ``gap_seeking``, and ``sampling_windows``.
* With cohort selection, also ``Escut``, ``Fscut``, and ``gap_std_cut``.

If you change one of the config settings during a run, ALF warns once and keeps the value it
started with. To change sampling partway through a run on purpose, use
:ref:`sampling windows <samplers-sampling-windows>`.

Master Configuration
--------------------

The master config controls the workflow wiring. Common fields include:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Meaning
   * - ``master_directory``
     - Base directory for relative paths. ``"pwd"`` means the directory where
       ALF is launched.
   * - ``builder_config_path``, ``sampler_config_path``, ``ML_config_path``,
       ``QM_config_path``
     - Stage-specific config files loaded after the master config.
   * - ``builder_task``, ``sampler_task``, ``ML_task``, ``QM_task``
     - Import strings for the Parsl task functions ALF will submit.
   * - ``h5_path``
     - Output pattern for HDF5 training batches, commonly
       ``h5store/data-{:04d}.h5``.
   * - ``model_path``
     - Output pattern for model directories, commonly
       ``models/model-{:04d}``.
   * - ``status_path``
     - Restart/progress state file, commonly ``status.txt``.
   * - ``QM_scratch_dir``
     - Directory where QM tasks write per-structure input, output, and scratch
       files.
   * - ``properties_list``
     - Mapping from ALF property names to HDF5 dataset names, property scope,
       and unit conversion factors. See :ref:`configuration-property-names`, and
       :doc:`units` for the conversion direction and common factors.
   * - ``parsl_configuration``
     - Parsl resource configuration used for normal runs.
   * - ``parsl_debug_configuration``
     - Optional smaller Parsl configuration used by stage-test commands.
   * - ``gpus_per_node``
     - Number of GPUs on each ML or sampler node. The two share this value, and a GPU
       mask on the worker takes precedence over it. See :doc:`parsl`.
   * - ``target_queued_QM``, ``parallel_samplers``, ``minimum_QM``,
       ``save_h5_threshold``, ``bootstrap_set``
     - Queue-depth and retraining controls for the active-learning loop. These
       count structures, including when tasks are batched. See :doc:`batching`.
   * - ``finite_bootstrap``
     - Optional. ``true`` builds exactly ``bootstrap_set`` structures, for a builder
       reading a fixed list of seeds. ALF stops with an error if builders run out
       before that many are labeled. The default is ``false``, which keeps building
       until enough labeling tasks have finished.
   * - ``maximum_stored_labels``
     - Optional. ALF stops after storing at least this many converged labels. It
       starts no new work, lets any running training finish and be promoted, reports
       what it dropped, releases its allocations, and exits. Labels that finished
       but weren't stored yet are stored in one last shard, which no model trained
       on. Sampler and QM work still running isn't stored. Labels are counted when a
       batch is stored, so a run can overshoot by up to ``save_h5_threshold``, plus
       that last shard. ALF reads this once, at startup.
       The count is kept in ``status_path``. By default, a run continues until you
       stop it.
   * - ``member_retries``
     - Optional, for the multi-node training tasks. How many more times to run a failed
       ensemble member. The default is ``1``. See :ref:`ml-multinode-training`.
   * - ``maximum_builder_structures``
     - Optional number of structures one builder task returns. The default is
       ``1``. See :doc:`builders`.
   * - ``maximum_completed_QM``
     - Optional limit on structures in the QM queue plus finished labels waiting to
       be stored. ALF stops starting builders at this limit until results are
       stored. By default, there is no limit.
   * - ``held_labels_dir``
     - Optional. Where finished labels wait until they're stored; see
       :ref:`configuration-restarts`. The default is ``h5_dir`` with ``_held``
       added, such as ``h5store_held/``. It must be outside ``h5_dir``.
   * - ``plotting_utility``, ``plot_dir``, ``update_plots_every``
     - Optional progress plots. ``plotting_utility`` is an import string such as
       ``alframework.tools.plotting.analysis_plot``, and ``plot_dir`` is where it
       writes. ``update_plots_every`` is how many loop iterations separate plot
       updates. Its default of ``1000`` can mean many hours, because a loop can take
       up to a minute. See :ref:`samplers-progress-plots`.
   * - ``trajectory_summary_path``
     - Optional file where ALF appends one row per trajectory, for progress plots.
       See :ref:`samplers-progress-plots`.

.. _configuration-property-names:

Property Names
~~~~~~~~~~~~~~

Each ``properties_list`` entry is ``"key": [dataset, scope, factor]``. The key is what the
labeler computes, and ``dataset`` is the name it gets in the HDF5 shards, which trainers and
fine-tuning read.

* **Ground-state labelers,** such as ORCA, Q-Chem, VASP through ASE, the teacher, and
  ``pyseqm_task``, use ``energy`` and ``forces``, and ground-state trainers and fine-tuning read
  those dataset names.
* **State-aware labelers,** ``pyscf_task``, ``gpu4pyscf_excited_state_task``, and PySEQM's excited-state tasks,
  name states ``E0``, ``E1``, … and their forces ``F0``, ``F1``, …, with gaps ``dE01``, and so
  on. State 0 is the ground state. See :doc:`excited_states`.

A state-aware labeler can also run the ground state alone, for example GPU4PySCF with
``nroots: 0``. To make its labels readable by ground-state trainers and fine-tuning, give the
states those dataset names:

.. code-block:: json

   "properties_list": {
     "E0": ["energy", "system", 1.0],
     "F0": ["forces", "atomic", 1.0]
   }

Path Conventions
----------------

Relative paths in config files are resolved from ``master_directory``. In most
examples, ``"master_directory": "pwd"`` means ALF treats the current working
directory as the run directory. This is why examples should usually be launched
from inside their own directory.

Fields ending in ``_path`` usually name a file or file pattern. ALF also derives
the matching directory from those paths when it needs to create output
directories. Fields ending in ``_dir`` are treated as directories and are
created if needed.

Output And Restart Files
------------------------

During a run, ALF writes a standard set of outputs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Purpose
   * - ``status.txt``
     - Restart and progress state: the current HDF5, model, and molecule ids, the
       number of labels stored (``lifetime_stored_labels``), and counts of failed tasks
       and failed structures (for example, ``lifetime_failed_QM_structures``).
   * - ``status.txt.lock``
     - Held by the running coordinator, so a second one can't start on the same run.
   * - ``h5store/data-*.h5``
     - Labeled data batches written from converged QM results. When any structure's metadata has
       ``charge`` or ``mult``, the shard also stores ``total_charge`` and ``spin_multiplicity`` for every
       structure, defaulting to 0 and 1; UMA fine-tuning reads them, and HIPPYNN training ignores them.
   * - ``h5store_held/held-*.pkl``
     - Finished labels waiting for the next store. They're removed once stored.
   * - ``models/model-*``
     - Model ensembles trained or fine-tuned during the ALF loop.
   * - ``sampling/metadata-*.p``
     - Sampler metadata, uncertainty diagnostics, and selected-structure
       records.
   * - QM scratch directory
     - Engine-specific input, output, and scratch files for selected
       structures.
   * - ``status_plots/``
     - Optional progress plots when a plotting utility is configured.

If ``status.txt`` exists, ALF restarts from it. If no status file exists, ALF
checks existing model and HDF5 paths to decide whether to bootstrap, start from
provided HDF5 data, or use an existing seed model.

.. _configuration-restarts:

Restarting After A Crash
~~~~~~~~~~~~~~~~~~~~~~~~

To continue a run that stopped, run the same command again. No files need
editing.

* **Finished labels are kept.** ALF stores labels into ``h5_dir`` only between
  trainings, so that every ensemble member trains on the same data. Until then,
  each finished label is written to ``held_labels_dir``. A restart stores those
  labels with the next shard, including labels that finished during a bootstrap
  or during training.
* **Status catches up with disk.** A shard stored just before the crash is
  counted and kept, a model ID that started training is never reused, and
  molecule IDs skip ahead past any used after status was last saved. ALF prints
  each change it makes, starting with ``Restart:``.
* **A finite bootstrap rebuilds only what it lacks.** It labels exactly the
  bootstrap IDs it doesn't hold, so the bootstrap holds the same structures as a
  run that never stopped. The ASE calculator task moves an old scratch directory
  for such an ID to ``<directory>.previous-N`` instead of reusing it.
* **Nothing is half-written.** ALF writes the status file, each shard, and each
  held label to a temporary file, such as ``status.txt.tmp`` or
  ``data-0003.h5.partial``, and renames it when it's complete. Readers ignore a
  leftover ``.partial`` file, and the next store overwrites it.
* **One coordinator per run.** The running coordinator holds a lock on
  ``status.txt.lock``, in every mode, including the stage tests. A second one
  stops at startup and names the holder. The lock is released when the
  coordinator exits or is killed, so it never needs removing by hand.

What a crash still costs: a training that was running is lost, and a later
training replaces it. Work on builders and samplers, and QM calculations that
hadn't finished, start again. After ``SIGKILL`` or a lost node, ALF can't release
its allocations itself; the scheduler ends them, or cancel them with
``scancel``. A ``SIGTERM`` lets ALF release them before it exits.

Common Run Lifecycle
--------------------

A typical ALF run follows this lifecycle:

1. Build or load initial structures with the configured builder task.
2. Bootstrap with QM labels, or start from seeded data/model files already in
   the run directory.
3. Train or load an initial ML model ensemble.
4. Sample configurations with the current ML model.
5. Send selected high-uncertainty structures to QM.
6. Store converged QM labels in a new HDF5 batch.
7. Retrain or fine-tune the model ensemble and update the current model id.
8. Repeat sampling, QM labeling, and retraining until the run is stopped, either by
   ``maximum_stored_labels`` or by you. ALF releases its allocations when it stops;
   see :ref:`parsl-releasing-allocations`.

Stage Config Files
------------------

Builder config
   Controls initial structure generation or loading. Examples include molecule
   libraries, fragment choices, cell ranges, minimum distances, seed structure
   directories, and coordinate perturbation settings.

Sampler config
   Controls configuration-space exploration. MLMD examples define time step,
   maximum time, temperature and density schedules, uncertainty cutoffs,
   metadata output, and the ASE model-loader function. The multi-trajectory sampler
   adds ``driver``, ``batching``, and ``acquisition`` blocks. See :doc:`samplers`,
   :doc:`batching`, and :doc:`acquisition`.

ML config
   Controls model training and model loading. HIPPYNN examples define ensemble
   size, species, data keys, architecture settings, loss weights, learning rate,
   scheduler options, controller options, and device behavior. See
   :doc:`ml_interfaces` for ML-backend details.

QM config
   Controls electronic-structure labeling. QM config files usually define the
   executable or launch command, CPU count, method keywords, input blocks, and
   engine-specific calculator options. An optional ``batching`` block labels several
   structures in one task. See :doc:`qm_interfaces` for supported QM interfaces and
   :doc:`batching` for batching.

Parsl resource config
   Controls where tasks run. The master config selects ``parsl_configuration``
   for normal runs and optionally ``parsl_debug_configuration`` for stage
   checks. See :doc:`parsl` for executor labels and cluster templates.

Stage Test Commands
-------------------

The ``--test_*`` flags run selected workflow stages through the normal ALF entry
point. They are useful during bring-up and for deliberate one-stage work.

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Command
     - What it runs
     - Common use
   * - ``--test_builder``
     - Runs the configured builder task once and prints the returned
       ``MoleculesObject``.
     - Check structure loading/building, or generate one builder output for
       inspection.
   * - ``--test_sampler``
     - Runs the builder once, then runs the configured sampler task once using
       the current model id.
     - Check ML model loading and sampler behavior, or run one sampler job
       without entering the full loop.
   * - ``--test_qm``
     - Runs the builder once, then runs the configured QM task once and writes
       ``qm_test.h5``.
     - Check QM execution, parsing, unit conversion, and HDF5 property names on
       one structure.
   * - ``--test_ml``
     - Runs the configured ML task once using the current HDF5/model state.
     - Train or fine-tune one model ensemble job, or check ML data/model
       settings before a full run.

All stage-test commands create or update ``status.txt`` because they enter
through the main ALF process. ``--test_ml`` can also create a real model
directory and advance the current training/model ids when training succeeds.
When ``parsl_debug_configuration`` is present in ``master_config.json``, these
commands use it instead of the normal ``parsl_configuration``. ALF releases its
allocations when these commands finish; see :ref:`parsl-releasing-allocations`.

Practical Tips
--------------

* Keep all config files for a run in one directory for reproducibility.
* Launch examples from their own directory unless ``master_directory`` points
  somewhere else intentionally.
* Run stage checks before long production runs:

  .. code-block:: bash

     python -m alframework --master master_config.json --test_builder
     python -m alframework --master master_config.json --test_qm
     python -m alframework --master master_config.json --test_ml
     python -m alframework --master master_config.json --test_sampler

  These commands run individual workflow stages through the main ALF entry
  point; see `Stage Test Commands`_ for behavior and side effects.

* Archive or remove old ``status.txt``, ``h5store/``, ``models/``, sampler
  metadata, and QM scratch directories before starting a new independent run.

Related Examples
----------------

Start with :doc:`../examples/simple_water` for a complete first workflow. Use
:doc:`../examples/seeded_active_learning` when starting from existing
structures or HDF5 data, and :doc:`../examples/fine_tuning` when starting from
existing model weights. For several electronic states, see :doc:`excited_states`.
