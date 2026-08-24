Quickstart
==========

This quickstart helps get ALF up and running using ``examples/simple_water``.
It is a checklist for bringing up the workflow; the full walkthrough lives in
:doc:`../examples/simple_water`.

Prerequisites
-------------

Before running ALF, make sure:

* ALF is installed in the Python environment you will use for the run.
* You are working from an example or run directory that contains the ALF config
  files.
* The external QM and ML software needed by the chosen example is available.
  For ``examples/simple_water``, that means an ORCA executable and the HIPPYNN
  environment needed for training and sampling.
* The Parsl resource configuration matches the machine where you will run ALF.

Pick An Example
---------------

Start with the simple water example:

.. code-block:: bash

   cd examples/simple_water

This example is small enough to inspect quickly, but it exercises all major ALF
stages: builder, sampler, QM labeling, ML training, data storage, and retraining.
After it is working, use the other example pages as templates for seeded,
periodic, reactive-sampling, and fine-tuning workflows.

Which Example Should I Use?
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Example
     - Use it when...
   * - :doc:`../examples/simple_water`
     - You want the recommended first full ALF run.
   * - :doc:`../examples/seeded_active_learning`
     - You want to bring existing structures, existing ALF/HIPPYNN HDF5 data,
       or both.
   * - :doc:`../examples/fine_tuning`
     - You want to start from existing model weights and adapt the model to new
       ALF data.
   * - :doc:`../examples/molten_salt`
     - You need a periodic VASP workflow.
   * - :doc:`../examples/il`
     - You need molecular mixtures or richer fragment libraries.
   * - :doc:`../examples/reactive_sampling`
     - You want reaction-path sampling around reactant, transition-state, and
       product structures.
   * - :doc:`../examples/uo2`
     - You want a NeuroChem/ANI-style workflow.

Seeded active learning and fine-tuning solve different setup problems. Seeded
active learning starts from structures or labeled HDF5 data and trains with
ALF's normal ML task. Fine-tuning starts from existing model weights, then
continues supervised training on ALF data with an example-local ML task.

Check The Config Files
----------------------

The simple water directory contains the files ALF needs for a complete run:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Purpose
   * - ``master_config.json``
     - Connects all stages, task functions, output paths, queue thresholds, and
       Parsl configs.
   * - ``builder_config.json``
     - Defines how initial water structures are built.
   * - ``mlmd_config.json``
     - Defines ML-driven MD sampling and uncertainty thresholds.
   * - ``orca_config.json``
     - Defines ORCA command, CPU count, method keywords, and input blocks.
   * - ``hippynn_config.json``
     - Defines HIPPYNN ensemble, network, training, and device settings.
   * - ``parsl_configs.py``
     - Defines local or cluster executors used by ALF tasks.

Edit Machine-Specific Settings
------------------------------

Before launching a run, update the settings that depend on your machine:

* In ``orca_config.json``, set ``QM_run_command`` to the ORCA executable or
  launch command available on your system.
* In ``parsl_configs.py``, set partitions, accounts, walltimes, worker
  initialization commands, and CPU/GPU executor sizes for your local or cluster
  environment.
* In ``hippynn_config.json``, check device and training settings before running
  ML training or MLMD sampling.
* In ``master_config.json``, confirm output paths such as ``h5_path``,
  ``model_path``, ``QM_scratch_dir``, and ``status_path``.

Run Stage Checks
----------------

Run the stages one at a time before starting the full active-learning loop:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json --test_ml
   python -m alframework --master master_config.json --test_sampler

These stage commands make it easier to isolate setup problems in structure
building, ORCA execution, HIPPYNN training, or MLMD sampling before a long run
is queued. They can also be useful outside initial bring-up when you want to run
one stage deliberately. For example, ``--test_ml`` submits the configured ML
task once against the current HDF5/model state, so for ensemble ML tasks it can
complete one ensemble training or fine-tuning job without starting the full
active-learning loop.

Start The Workflow
------------------

After the stage checks pass, start the active-learning loop:

.. code-block:: bash

   python -m alframework --master master_config.json

ALF writes progress to ``status.txt`` and prints queue status for builder,
sampler, QM, and ML tasks while the loop is running.

Know The Outputs
----------------

During a run, expect these paths to appear in the working directory:

* ``status.txt``: restart and progress state.
* ``h5store/``: HDF5 training-data batches.
* ``models/``: trained model directories.
* ``sampling/``: sampler metadata.
* ``orca_scratch/``: per-structure ORCA input, output, and scratch directories.

What To Read Next
-----------------

* :doc:`../examples/simple_water` for the full simple-water walkthrough.
* :doc:`../user_guide/configuration` for the overall configuration model.
* :doc:`../user_guide/parsl` for adapting executors and cluster resources.
* :doc:`../user_guide/qm_interfaces` for ORCA, VASP, Q-Chem, and ASE-backed QM
  interfaces.
* :doc:`../user_guide/ml_interfaces` for HIPPYNN and new ML backend setup.
