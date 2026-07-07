Reactive Sampling Example
=========================

The reactive sampling example demonstrates an ALF workflow for reaction-pathway
sampling. It is intended for systems where reactant, transition-state, and
product structures are used to guide sampling toward chemically relevant
configurations.

The files needed to run the example are located in
``examples/reactive_sampling``.

Before You Run
--------------

This example is more specialized than the water and molten-salt examples. It
assumes you already have reactant, transition-state, and product structures for
the reaction family you want to explore.

* Edit ``parsl_configs.py`` for the target cluster. Reactive sampling and ML
  model loading should run on ``alf_sampler_executor``; ORCA labeling should
  run on ``alf_QM_executor``.
* Edit ``orca_config.json`` so ``QM_run_command`` points to the local ORCA
  executable and the requested output includes the properties in
  ``properties_list``.
* Edit ``builder_config.json`` if you replace the included ``r.xyz``,
  ``ts.xyz``, and ``p.xyz`` structures with a larger reaction library.
* Edit ``react_sampler_config.json`` to choose the reactive search method,
  number of NEB images, optimization limits, and uncertainty thresholds.

Workflow Pattern
----------------

This example uses:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Stage
     - Task
   * - Builder
     - ``alframework.builders.reactive_builder.load_reactive_task``
   * - Sampler
     - ``alframework.samplers.reactive_sampler.reactive_sampling``
   * - QM
     - ``alframework.qm_interfaces.orca5_interface.orca_calculator_task``
   * - ML
     - ``alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_task``

The builder reads ``builder_config.json`` and loads reaction metadata from the
fragment library. The provided example contains one reactant file, one
transition-state file, and one product file.

The sampler reads ``react_sampler_config.json``. The provided configuration
uses ``NEB_DIMER``, builds 20 NEB images, allows 20 outer iterations, and uses
energy and force uncertainty thresholds to decide which structures should be
sent to QM.

The QM and ML stages follow the same ORCA + HIPPYNN pattern as the simple water
example. See :doc:`../user_guide/qm_interfaces` and
:doc:`../user_guide/ml_interfaces` for the general interface details.

Recommended Bring-Up Commands
-----------------------------

From ``examples/reactive_sampling``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json --test_ml
   python -m alframework --master master_config.json --test_sampler
   python -m alframework --master master_config.json

The staged checks are especially useful here because failures can come from
several independent sources: malformed reaction structures, ORCA setup,
HIPPYNN environment setup, or sampler resource placement.

Expected Outputs
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Purpose
   * - ``status.txt``
     - Restart and progress state.
   * - ``h5store/data-*.h5``
     - ORCA-labeled reaction-pathway structures.
   * - ``models/model-*``
     - HIPPYNN model ensemble directories.
   * - ``react-test-running/``
     - ORCA scratch directories for selected structures.
   * - ``status_plots/``
     - Optional progress plots.

Configuration Files
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - ``master_config.json``
     - Connects reactive builder, reactive sampler, ORCA labeling, HIPPYNN
       training, output paths, and Parsl configs.
   * - ``builder_config.json``
     - Lists reaction-library files with reactant, transition-state, and
       product structures.
   * - ``react_sampler_config.json``
     - Defines the reactive sampling method, NEB image count, iteration limits,
       and uncertainty thresholds.
   * - ``orca_config.json``
     - Defines the ORCA command and input blocks.
   * - ``hippynn_config.json``
     - Defines HIPPYNN ensemble and training settings.
   * - ``parsl_configs.py``
     - Generic CPU/GPU Slurm template copied from the water example.

Use this example when the active-learning target is a reaction coordinate or a
set of related reaction pathways rather than broad molecular dynamics sampling.
For broader sampling around a stable phase, start with
:doc:`simple_water` or :doc:`molten_salt` instead.
