Simple Water Example
====================

The simple water example is a compact ALF workflow for a small molecular
system. It is a useful starting point for checking that the builder, sampler,
QM interface, ML interface, and Parsl resource configuration are connected
correctly.

What it demonstrates
--------------------

This example uses:

* ``simple_condensed_phase_builder_task`` for structure construction
* ``simple_mlmd_sampling_task`` for uncertainty-driven MD sampling
* ``orca_calculator_task`` for QM labeling
* ``train_HIPPYNN_ensemble_task`` for model retraining

Relevant files
--------------

The example lives in ``examples/simple_water``.

* `master_config.json <https://github.com/lanl/ALF/blob/main/examples/simple_water/master_config.json>`_
* `builder_config.json <https://github.com/lanl/ALF/blob/main/examples/simple_water/builder_config.json>`_
* `mlmd_config.json <https://github.com/lanl/ALF/blob/main/examples/simple_water/mlmd_config.json>`_
* `orca_config.json <https://github.com/lanl/ALF/blob/main/examples/simple_water/orca_config.json>`_
* `hippynn_config.json <https://github.com/lanl/ALF/blob/main/examples/simple_water/hippynn_config.json>`_

Use this example when you want a small end-to-end workflow to adapt before
moving to larger systems.
