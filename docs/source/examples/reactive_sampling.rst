Reactive Sampling Example
=========================

The reactive sampling example demonstrates an ALF workflow for reaction-pathway
sampling. It is intended for systems where reactant, transition-state, and
product structures are used to guide sampling toward chemically relevant
configurations.

What it demonstrates
--------------------

This example uses:

* ``load_reactive_task`` for loading reaction structures
* ``reactive_sampling`` for NEB- and dimer-style reactive sampling
* ``orca_calculator_task`` for QM labeling
* ``train_HIPPYNN_ensemble_task`` for model retraining

Relevant files
--------------

The example lives in ``examples/reactive_sampling``.

* `master_config.json <https://github.com/lanl/ALF/blob/main/examples/reactive_sampling/master_config.json>`_
* `builder_config.json <https://github.com/lanl/ALF/blob/main/examples/reactive_sampling/builder_config.json>`_
* `react_sampler_config.json <https://github.com/lanl/ALF/blob/main/examples/reactive_sampling/react_sampler_config.json>`_
* `orca_config.json <https://github.com/lanl/ALF/blob/main/examples/reactive_sampling/orca_config.json>`_
* `hippynn_config.json <https://github.com/lanl/ALF/blob/main/examples/reactive_sampling/hippynn_config.json>`_

Use this example when the active-learning target is a reaction coordinate or a
set of related reaction pathways rather than broad molecular dynamics sampling.
See :doc:`../user_guide/ml_interfaces` for the HIPPYNN training task and ASE
calculator loader pattern used by the reactive sampler.
