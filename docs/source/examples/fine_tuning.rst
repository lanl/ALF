Fine-Tuning Existing Models
===========================

This fine-tuning example demonstrates how to start ALF from existing HIPPYNN
weights and adapt them to new ALF data. The cleanest starting point is a
pretrained ensemble, but a common fine-tuning workflow starts from only
one model. In this case, copy the same model into multiple ensemble-member
directories, fine-tune the copies independently on seed target-domain labels,
and then continue active learning.

The files needed to run the example are located in
``examples/fine_tuning``. If you only want to bring your own
structures or HDF5 data without starting from model weights, use
:doc:`seeded_active_learning`.

Before You Run
--------------

This example assumes you already have trained HIPPYNN weights that can be used
for sampling and weight initialization. A true pretrained ensemble is ideal. A
single pretrained model can also be used, but copied ensemble members have zero
or near-zero disagreement until they are fine-tuned differently.

* Put starting structures in ``fragment_library`` as ASE-readable ``.cfg``
  files. The included ``water_seed.cfg`` is a minimal builder-test structure.
* Set ``shake`` in ``builder_config.json`` to ``0.0`` for exact reuse or to a
  small value such as ``0.05`` to perturb loaded structures.
* Edit ``orca_config.json`` so ``QM_run_command`` points to the local ORCA
  executable.
* Edit ``hippynn_config.json`` so species and data keys match the pretrained
  ensemble and any prior HDF5 data. The model graph itself is loaded from the
  HIPPYNN checkpoint files.
* Set ``n_models`` to the number of ensemble-member directories you provide.
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

Place trained HIPPYNN weights in the first ALF model slot. If you have a true
ensemble, each ``model-XX`` directory should contain a different ensemble
member:

.. code-block:: text

   models/model-0000/model-00/
   models/model-0000/model-01/
   ...

If you have only one foundation model, copy that same model into each
``model-XX`` directory and set ``n_models`` in ``hippynn_config.json`` to the
number of copied members:

.. code-block:: text

   models/model-0000/model-00/experiment_structure.pt
   models/model-0000/model-00/best_checkpoint.pt
   models/model-0000/model-01/experiment_structure.pt
   models/model-0000/model-01/best_checkpoint.pt
   ...

Do not create ``status.txt`` before the first run. When ALF starts without a
status file, it checks ``model_path``. If ``models/model-0000`` already exists,
ALF sets ``current_model_id`` to ``0`` and uses that model for sampling instead
of building a bootstrap set. The local fine-tuning task then uses the same model
id as the source weights for the next training job.

Initial Ensemble Disagreement
-----------------------------

Copying one model into multiple ensemble slots does not by itself create useful
uncertainty. Before the copied members are fine-tuned, they make the same or
nearly the same predictions, so ``energy_stdev``, ``forces_stdev_mean``, and
``forces_stdev_max`` will be zero or very small. UDD bias also has no useful
effect when the ensemble energy standard deviation is zero.

The disagreement appears only after the copied members are trained differently.
The example-local fine-tuning task creates a fresh optimizer for each member
and uses randomized train/validation/test splits, so copied members can diverge
once they are fine-tuned on target-domain HDF5 data. The first useful ensemble
for uncertainty-driven active learning is usually the fine-tuned output, such
as ``models/model-0001``.

If you have previous labeled data in ALF/HIPPYNN HDF5 format, place it in the
run's HDF5 store:

.. code-block:: text

   h5store/data-0000.h5
   h5store/data-0001.h5
   ...

The fine-tuning task reads the HDF5 directory, so those batches can be included
when ALF trains the next ensemble.

Fine-Tune Only From Existing Data
---------------------------------

You can fine-tune existing checkpoints on existing labeled data without running
MLMD. Place the seed checkpoints under ``models/model-0000``, place ALF/HIPPYNN
HDF5 batches under ``h5store/``, start without a pre-existing ``status.txt``,
and run only the ML stage:

.. code-block:: bash

   python -m alframework --master master_config.json --test_ml

This supervised fine-tuning step loads the current seed model, reads the HDF5
store, and writes the adapted ensemble to the next model slot, usually
``models/model-0001``. It does not launch builders, MLMD samplers, or QM
labeling. After that model exists, you can either inspect it as a standalone
fine-tuned ensemble or start the full ALF loop to continue MLMD-driven active
learning:

.. code-block:: bash

   python -m alframework --master master_config.json

If you do not have seed labels, the copied model can still run deterministic
MLMD, but uncertainty-triggered QM selection may not fire. In that case, first
collect a seed set using bootstrap QM labels, MD snapshots, or manual structure selection. After
those labels are written to HDF5, run the fine-tuning task to create a
diversified ensemble for continued ALF sampling.

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

For a single-model starting point, the recommended sequence is:

1. Copy the model into ``models/model-0000/model-00`` through
   ``model-NN``.
2. Provide or collect a small target-domain HDF5 seed set.
3. Fine-tune the copied members independently to produce ``models/model-0001``.
4. Continue active learning from ``models/model-0001``, where ensemble
   disagreement can guide QM selection.

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
   * - ``qm_scratch/``
     - ORCA input, output, and scratch directories for selected structures.

Use this example when you already have a model that can guide sampling and want
ALF to collect additional labels around its uncertainty, then adapt the model
weights to the new data.
