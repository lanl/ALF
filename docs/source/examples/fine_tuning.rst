Fine-Tuning Existing Model Example
==================================

The fine-tuning example demonstrates how to start ALF from an existing HIPPYNN
ensemble and adapt that ensemble to new ALF data. It combines seeded active
learning with weights-based fine-tuning: ALF loads a pretrained ensemble for
MLMD sampling, labels new high-uncertainty structures, then continues
supervised training from the pretrained HIPPYNN weights.

This matches the common MLIP fine-tuning pattern: initialize from pretrained
weights, train on smaller target data with a lower learning rate, and write an
adapted model. It does not reuse the old optimizer state.

The files needed to run the example are located in
``examples/fine_tuning``. If you only want to bring your own
structures or HDF5 data without starting from model weights, use
:doc:`seeded_active_learning`.

Before You Run
--------------

This example assumes you already have a trained HIPPYNN ensemble that can be
used for sampling and weight initialization.

* Put starting structures in ``fragment_library`` as ASE-readable ``.cfg``
  files. The included ``water_seed.cfg`` is a minimal builder-test structure.
* Set ``shake`` in ``builder_config.json`` to ``0.0`` for exact reuse or to a
  small value such as ``0.05`` to perturb loaded structures.
* Edit ``orca_config.json`` so ``QM_run_command`` points to the local ORCA
  executable.
* Edit ``hippynn_config.json`` so species and data keys match the pretrained
  ensemble and any prior HDF5 data. The model graph itself is loaded from the
  HIPPYNN checkpoint files.
* Keep ``learning_rate`` lower than the original pretraining run unless you have
  a reason to aggressively adapt the model.
* Edit ``parsl_configs.py`` for the target machine.

Workflow Pattern
----------------

This example uses existing ALF hooks:

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
     - ``fine_tune_ml_task.fine_tune_HIPPYNN_ensemble_task``

The builder reads ``.cfg`` structures from ``fragment_library``. The sampler
loads the seed HIPPYNN ensemble through ``HIPNN_ASE_load_ensemble``. The ML
stage uses the local ``fine_tune_ml_task.py`` module in this example directory,
so no changes are required in ALF's main program. Users can adapt this local
task pattern for other model architectures.

Seed The Model
--------------

Place a trained HIPPYNN ensemble in the first ALF model slot:

.. code-block:: text

   models/model-0000/model-00/
   models/model-0000/model-01/
   ...

Do not create ``status.txt`` before the first run. When ALF starts without a
status file, it checks ``model_path``. If ``models/model-0000`` already exists,
ALF sets ``current_model_id`` to ``0`` and uses that model for sampling instead
of building a bootstrap set. The local fine-tuning task then uses the same model
id as the source weights for the next training job.

If you have previous labeled data in ALF/HIPPYNN HDF5 format, place it in the
run's HDF5 store:

.. code-block:: text

   h5store/data-0000.h5
   h5store/data-0001.h5
   ...

The fine-tuning task reads the HDF5 directory, so those batches can be included
when ALF trains the next ensemble.

Fine-Tuning Task
----------------

The example-local ``fine_tune_ml_task.py`` is loaded through ``ML_task`` in
``master_config.json``:

.. code-block:: json

   "ML_task": "fine_tune_ml_task.fine_tune_HIPPYNN_ensemble_task"

Each ensemble member is loaded from the current ALF model id:

.. code-block:: text

   models/model-0000/model-00/experiment_structure.pt
   models/model-0000/model-00/best_checkpoint.pt
   models/model-0000/model-01/experiment_structure.pt
   models/model-0000/model-01/best_checkpoint.pt
   ...

The task creates a fresh Adam optimizer for the target data and writes the
adapted ensemble to ``models/model-0001``. Later active-learning rounds
fine-tune from the latest accepted model and write the next model id.

Recommended Bring-Up Commands
-----------------------------

From ``examples/fine_tuning``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_sampler
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json --test_ml

Run ``--test_sampler`` only after the seed ensemble has been placed under
``models/model-0000``. Run ``--test_ml`` when prior HDF5 data exists and the
HIPPYNN data keys match that data.

After the stage checks pass, start ALF:

.. code-block:: bash

   python -m alframework --master master_config.json

Expected Outputs
----------------

During a fine-tuning run, ALF writes the standard outputs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Purpose
   * - ``status.txt``
     - Restart state, including the current seed model id and later model ids.
   * - ``h5store/data-*.h5``
     - Newly labeled data batches, plus any optional prior data you provided.
   * - ``models/model-0001``
     - First ensemble fine-tuned from the seed model.
   * - ``sampling/metadata-*.p``
     - MLMD sampling metadata and uncertainty diagnostics.
   * - ``orca_scratch/``
     - ORCA input, output, and scratch directories for selected structures.

Use this example when you already have a model that can guide sampling and want
ALF to collect additional labels around its uncertainty, then adapt the model
weights to the new data.
