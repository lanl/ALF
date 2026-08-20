Configuration
=============

ALF runs are configured with a small set of JSON files plus a Python Parsl
resource file. The master configuration is the entry point: it names the task
functions, points to the stage-specific config files, and defines the output
paths ALF uses while the active-learning loop runs.

Most examples use these files:

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
       and unit conversion factors. See :doc:`units` for the conversion
       direction and common factors.
   * - ``parsl_configuration``
     - Parsl resource configuration used for normal runs.
   * - ``parsl_debug_configuration``
     - Optional smaller Parsl configuration used by stage-test commands.
   * - ``gpus_per_node``
     - Number of GPUs ALF assumes each ML or sampler worker can see.
   * - ``target_queued_QM``, ``parallel_samplers``, ``minimum_QM``,
       ``save_h5_threshold``, ``bootstrap_set``
     - Queue-depth and retraining controls for the active-learning loop.

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
     - Restart and progress state, including current HDF5, model, and molecule
       ids plus failed-task counters.
   * - ``h5store/data-*.h5``
     - Labeled data batches written from converged QM results.
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
8. Repeat sampling, QM labeling, and retraining until the run is stopped.

Stage Config Files
------------------

Builder config
   Controls initial structure generation or loading. Examples include molecule
   libraries, fragment choices, cell ranges, minimum distances, seed structure
   directories, and coordinate perturbation settings.

Sampler config
   Controls configuration-space exploration. MLMD examples define time step,
   maximum time, temperature and density schedules, uncertainty cutoffs,
   metadata output, and the ASE model-loader function.

ML config
   Controls model training and model loading. HIPPYNN examples define ensemble
   size, species, data keys, architecture settings, loss weights, learning rate,
   scheduler options, controller options, and device behavior. See
   :doc:`ml_interfaces` for ML-backend details.

QM config
   Controls electronic-structure labeling. QM config files usually define the
   executable or launch command, CPU count, method keywords, input blocks, and
   engine-specific calculator options. See :doc:`qm_interfaces` for supported
   QM interfaces.

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
commands use it instead of the normal ``parsl_configuration``.

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
existing model weights.
