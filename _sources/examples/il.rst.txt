Ionic Liquid Example
====================

The ionic liquid example demonstrates a molecular condensed-phase ALF workflow
with a richer fragment library than the simple water example. It is intended as
a starting point for mixtures containing solvent molecules, cations, anions,
and small organic fragments.

The files needed to run the example are located in ``examples/IL``.

Before You Run
--------------

This example is a template, not a ready-to-run cluster job.

* Edit ``parsl_configs.py`` for the target Slurm partitions, accounts, QoS,
  walltimes, worker initialization commands, and GPU resource requests.
* Edit ``orca_config.json`` so ``QM_run_command`` points to the local ORCA
  executable and requested output includes the properties in
  ``properties_list``.
* Edit ``hippynn_config.json`` for the desired species, network, training, and
  device settings.
* Review ``builder_config.json`` before changing the fragment library. The
  builder expects fragment names to match files in ``fragment_library``.

Workflow Pattern
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Stage
     - Task
   * - Builder
     - ``alframework.builders.builders.simple_condensed_phase_builder_task``
   * - Sampler
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
   * - QM
     - ``alframework.qm_interfaces.orca5_interface.orca_double_calculator_task``
   * - ML
     - ``alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_task``

The builder assembles systems from ``fragment_library`` using
``solute_molecule_options`` and ``solvent_molecules``. This is useful when the
active-learning space includes multiple possible cation/anion combinations or
solvent environments.

The sampler uses HIPPYNN models through ``HIPNN_ASE_load_ensemble``. Unlike the
simple water example, this sampler config includes temperature and density
fluctuation ranges, so ALF can explore both thermal and box-size variation
during MLMD.

The QM task uses the ORCA double-check interface, which is useful when the
workflow should validate a first calculation with a second ORCA input before
accepting the label.

Recommended Bring-Up Commands
-----------------------------

From ``examples/IL``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json --test_ml
   python -m alframework --master master_config.json --test_sampler
   python -m alframework --master master_config.json

Configuration Files
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - ``master_config.json``
     - Connects the condensed-phase builder, MLMD sampler, ORCA double-check
       task, HIPPYNN training task, queue controls, outputs, and Parsl configs.
   * - ``builder_config.json``
     - Defines fragment-library choices, solvent probabilities, cell ranges,
       radius scaling, minimum distance, maximum atom count, and coordinate
       shaking.
   * - ``mlmd_config.json``
     - Defines MD time step, maximum time, uncertainty thresholds, temperature
       and density schedules, trajectory output, and ML calculator options.
   * - ``orca_config.json``
     - Defines ORCA commands and input blocks.
   * - ``hippynn_config.json``
     - Defines HIPPYNN ensemble and training settings.
   * - ``parsl_configs.py``
     - Generic CPU/GPU Slurm template copied from the water example.

Expected Outputs
----------------

The run writes the standard ALF outputs: ``status.txt``, ``h5store/data-*.h5``,
``models/model-*``, ``sampling/metadata-*.p``, ORCA scratch directories under
``QM_scratch_dir``, and optional ``status_plots/`` output.

See :doc:`../user_guide/builders` for condensed-phase builder settings,
:doc:`../user_guide/samplers` for MLMD and density fluctuation controls, and
:doc:`../user_guide/parsl` for adapting the copied Parsl template.
