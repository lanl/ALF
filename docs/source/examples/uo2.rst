UO2 Example
===========

The UO2 example demonstrates a periodic solid workflow built from pre-existing
CFG structures. It also uses NeuroChem/ANI for ML and a custom
two-step VASP QM wrapper.

The files needed to run the example are located in ``examples/UO2``.

Before You Run
--------------

This example is more specialized than the HIPPYNN examples.

* Edit ``parsl_configs.py`` for the target cluster. The copied file is a generic
  CPU/GPU Slurm template and is not expected to run unchanged.
* Keep ``custom_modules.py`` unless you replace the UO2 QM task. It defines the
  custom two-step VASP wrapper used by ``master_config.json``.
* Edit ``vasp_2step_config.json`` so both VASP input blocks and
  ``QM_run_command`` match the local VASP installation.
* Edit ``ani_config.json`` for the NeuroChem/ANI training setup and make sure
  the NeuroChem dependencies are available.
* Review the CFG files in ``fragment_library``. These are pre-generated
  structures loaded by the builder rather than molecules assembled by ALF.

Workflow Pattern
----------------

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
     - ``custom_modules.ase_calculator_task``
   * - ML
     - ``alframework.ml_interfaces.neurochem_interface.train_ANI_model_task``

The builder loads ``.cfg`` structures from ``fragment_library`` and applies the
configured coordinate shake. Use this path when you already have periodic
starting structures from another simulation or crystal-generation workflow.

The sampler uses ``use_potential_specific_code: "neurochem"`` and
``alframework.ml_interfaces.neurochem_interface.NeuroChemCalculator``. It
includes temperature and density fluctuation schedules for exploring the solid
around the loaded configurations.

The QM task is intentionally custom. It runs one VASP input, removes WAVECAR
files, runs a second VASP input, and records convergence from ``OUTCAR``.

Recommended Bring-Up Commands
-----------------------------

From ``examples/UO2``:

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
     - Connects the CFG loader, MLMD sampler, custom VASP task, ANI training
       task, output paths, queue controls, and Parsl configs.
   * - ``builder_config.json``
     - Points to the CFG structure library and coordinate-shake value.
   * - ``mlmd_config.json``
     - Defines MLMD schedule, density fluctuation, uncertainty thresholds,
       NeuroChem calculator hook, and metadata path.
   * - ``vasp_2step_config.json``
     - Defines the two VASP input stages used by ``custom_modules.py``.
   * - ``ani_config.json``
     - Defines NeuroChem/ANI model and training settings.
   * - ``custom_modules.py``
     - Defines the custom two-step VASP QM task. This file is still required by
       the provided master config.
   * - ``parsl_configs.py``
     - Generic CPU/GPU Slurm template copied from the water example.

Expected Outputs
----------------

The run writes ``status.txt``, ``h5store/data-*.h5``, ``models/model-*``,
``sampling/metadata-*.p``, VASP scratch directories under ``QM_scratch_dir``,
and any optional plotting output configured in the master config.

See :doc:`../user_guide/builders` for CFG loader details,
:doc:`../user_guide/ml_interfaces` for NeuroChem/ANI notes, and
:doc:`../user_guide/qm_interfaces` for the ASE/VASP integration pattern.
