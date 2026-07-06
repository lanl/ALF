Molten Salt Example
===================

The molten salt example demonstrates an ALF workflow for periodic ionic
systems. It is intended for workflows where the builder constructs
charge-balanced atomic systems and the QM stage labels selected configurations
with VASP.

What it demonstrates
--------------------

This example uses:

* ``atomic_system_task`` for charge-balanced structure generation
* ``simple_mlmd_sampling_task`` for uncertainty-driven MD sampling
* ``VASP_ase_calculator_task`` for QM labeling
* ``train_HIPPYNN_ensemble_task`` for model retraining

Relevant files
--------------

The example lives in ``examples/molten_salt``.

* `master_config.json <https://github.com/lanl/ALF/blob/main/examples/molten_salt/master_config.json>`_
* `builder_config.json <https://github.com/lanl/ALF/blob/main/examples/molten_salt/builder_config.json>`_
* `mlmd_config.json <https://github.com/lanl/ALF/blob/main/examples/molten_salt/mlmd_config.json>`_
* `vasp_config.json <https://github.com/lanl/ALF/blob/main/examples/molten_salt/vasp_config.json>`_
* `hippynn_config.json <https://github.com/lanl/ALF/blob/main/examples/molten_salt/hippynn_config.json>`_
* `p310-alf-load.bash <https://github.com/lanl/ALF/blob/main/examples/molten_salt/p310-alf-load.bash>`_

Adapt the VASP command, Parsl resource configuration, scratch paths, and module
load script to match the target compute environment. See
:doc:`../user_guide/ml_interfaces` for the HIPPYNN configuration pattern and
the ML sampler calculator hook used by this example.
