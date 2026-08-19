Seeded Active Learning
======================

The seeded active learning example shows how to start ALF from user-provided
structures and optional existing labeled data. Use this when you already have
reasonable starting configurations, a small labeled dataset, or both, and want
to avoid a random bootstrap stage.

The files needed to run the example are located in
``examples/seeded_active_learning``.

The checked-in configuration is intentionally modest for initial bring-up: it
uses four ensemble members, caps training at 200 epochs, and limits the active
learning queues to 25 items. Increase these values only after the staged checks
work on the target machine.

Before You Run
--------------

This example assumes you can provide starting structures, prior HDF5 data, or
both.

* Convert starting structures to ASE-readable ``.cfg`` files and place them in
  ``fragment_library``.
* Place optional prior labeled data in ``h5store`` using ALF/HIPPYNN HDF5
  format.
* Edit ``hippynn_config.json`` so architecture, species, and data keys match
  the provided HDF5 data.
* Edit ``orca_config.json`` so ``QM_run_command`` points to the local ORCA
  executable.
* Edit ``parsl_configs.py`` for the target machine.

Workflow Pattern
----------------

This example uses ALF's standard tasks:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Stage
     - Task
   * - Builder
     - ``alframework.builders.builders.simple_cfg_loader_task``
   * - Sampler
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
   * - QM
     - ``alframework.qm_interfaces.orca5_interface.orca_calculator_task``
   * - ML
     - ``alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_task``

The builder reads ``.cfg`` structures from ``fragment_library``. The ML task is
the normal HIPPYNN ensemble training task, so this example does not require a
pretrained model or a custom task.

Seed The Run
------------

Place starting structures in the fragment library:

.. code-block:: text

   fragment_library/water_seed.cfg
   fragment_library/your_structure.cfg
   ...

Set ``shake`` in ``builder_config.json`` to ``0.0`` to reuse those structures
exactly, or to a small value such as ``0.05`` to perturb each loaded structure
before sampling.

If you already have labeled data in ALF/HIPPYNN HDF5 format, place it in the
run's HDF5 store:

.. code-block:: text

   h5store/data-0000.h5
   h5store/data-0001.h5
   ...

When ``status.txt`` is absent and ``h5store/data-0000.h5`` exists, ALF detects
the existing HDF5 store. Because labeled data already exists, ALF skips the
random bootstrap-labeling stage and trains ``models/model-0000`` from the
provided data before starting ML-driven sampling.

Recommended Bring-Up Commands
-----------------------------

From ``examples/seeded_active_learning``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_ml
   python -m alframework --master master_config.json --test_sampler
   python -m alframework --master master_config.json --test_qm

Run ``--test_ml`` only after at least one HDF5 file exists in ``h5store``. Run
``--test_sampler`` after ``models/model-0000`` has been trained or placed under
``models/``.

After the stage checks pass, start ALF:

.. code-block:: bash

   python -m alframework --master master_config.json

Expected Outputs
----------------

During a seeded run, ALF writes the standard outputs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Purpose
   * - ``status.txt``
     - Restart state, including current HDF5 and model ids.
   * - ``models/model-0000``
     - First ensemble trained from the provided HDF5 data.
   * - ``h5store/data-*.h5``
     - Existing and newly labeled data batches.
   * - ``sampling/metadata-*.p``
     - MLMD sampling metadata and uncertainty diagnostics.
   * - ``qm_scratch/``
     - ORCA input, output, and scratch directories for selected structures.

Use this example when you want ALF to start from your own structures or data.
Use the fine-tuning example when you also want to initialize training from
existing model weights.
