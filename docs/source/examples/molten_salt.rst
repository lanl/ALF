Molten Salt Example
===================

The molten salt example demonstrates an ALF workflow for periodic ionic
systems. It is intended for workflows where the builder constructs
charge-balanced atomic systems and the QM stage labels selected configurations
with VASP.

The files needed to run the example are located in ``examples/molten_salt``.

Before You Run
--------------

This example is a template for a periodic VASP + HIPPYNN workflow. Before
running it on a new machine:

* Edit ``parsl_configs.py`` for the target Slurm partition, account, QoS,
  walltime, worker initialization, and CPU/GPU resource requests. See
  :doc:`../user_guide/parsl`.
* Edit ``vasp_config.json`` so ``QM_run_command`` and VASP input settings match
  the local installation and intended calculation.
* Edit ``hippynn_config.json`` for the elements, network size, training
  controls, and device settings for the run. See
  :doc:`../user_guide/ml_interfaces`.
* Confirm that the requested properties in ``master_config.json`` match the
  VASP calculator results and HDF5 keys you want to train on.

Workflow Stages
---------------

This example uses:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Stage
     - Task
   * - Builder
     - ``alframework.builders.builders.atomic_system_task``
   * - Sampler
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
   * - QM
     - ``alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task``
   * - ML
     - ``alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_task``

The builder reads ``builder_config.json`` and creates charge-neutral random
atomic systems from the configured ``atom_charges``. The provided configuration
targets 100 atoms, enforces a minimum interatomic distance, and samples a box
length from ``box_length_range``. This is the right builder family for molten
salts or other ionic atomic systems that are not assembled from molecular
fragments.

The sampler reads ``mlmd_config.json`` and runs ML-driven MD with a HIPPYNN ASE
calculator ensemble. In this example, density fluctuation fields are set to
``null`` so the sampler does not rescale the box during sampling.

The QM stage uses ASE's VASP calculator wrapper. This is the preferred path for
periodic DFT engines that ASE already supports. See
:doc:`../user_guide/qm_interfaces` for the generic ASE calculator pattern.

Recommended Bring-Up Commands
-----------------------------

From ``examples/molten_salt``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json --test_ml
   python -m alframework --master master_config.json --test_sampler
   python -m alframework --master master_config.json

The test modes use ``parsl_debug_configuration`` from ``master_config.json``.
Use them to verify the builder, VASP command, HIPPYNN environment, and sampler
resource mapping before launching a long active-learning run.

Expected Outputs
----------------

During a run, expect the usual ALF output structure:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Purpose
   * - ``status.txt``
     - Restart and progress state.
   * - ``h5store/data-*.h5``
     - VASP-labeled training batches.
   * - ``models/model-*``
     - HIPPYNN model ensemble directories.
   * - ``sampling/metadata-*.p``
     - Sampler metadata and uncertainty records.
   * - ``vasp_calculations/``
     - Per-structure VASP scratch and output directories.
   * - ``status_plots/``
     - Optional progress plots from the configured plotting utility.

Configuration Files
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - ``master_config.json``
     - Connects the builder, sampler, VASP QM task, HIPPYNN training task,
       output paths, queue-depth controls, and Parsl configs.
   * - ``builder_config.json``
     - Defines element charges, target atom count, minimum distance, and box
       length range for charge-neutral atomic systems.
   * - ``mlmd_config.json``
     - Defines MD time step, maximum time, uncertainty thresholds, temperature
       schedule, optional density schedule, metadata path, and ASE model loader.
   * - ``vasp_config.json``
     - Defines the VASP command and ASE VASP calculator options.
   * - ``hippynn_config.json``
     - Defines HIPPYNN ensemble, network, species, optimizer, and training
       options.
   * - ``parsl_configs.py``
     - Generic CPU/GPU Slurm template copied from the water example.

Environment setup for this example now lives in the command strings and worker
initialization hooks that users adapt in ``vasp_config.json`` and
``parsl_configs.py``.
