What Is ALF
===========

The Active Learning Framework (ALF) automates the construction of structurally diverse datasets for
machine learning interatomic potentials (MLIPs) using a query by committee (QBC) active-learning approach.

At a high level, ALF coordinates four core stages:

1. System construction (builders)
2. Sampling (samplers)
3. ML model training (ML interfaces)
4. Electronic structure calculations (QM interfaces)

Overview of the workflow
------------------------

ALF uses a master process to orchestrate tasks and data flow between these
stages. The process is typically launched with:

.. code-block:: bash

   python -m alframework --master master.json

The ``master.json`` file points to the other configuration files and task
definitions needed for each stage. In practice, ALF uses five JSON files:

1. Master configuration
2. Builder configuration
3. Sampler configuration
4. ML configuration
5. QM configuration

How each iteration works
-----------------------

During an iteration, ALF builds candidate structures, samples configurations
using the current ML model ensemble, evaluates selected structures with QM,
and stores the resulting data for retraining. The newly trained model is then
used in the next sampling cycle.

This loop is designed to run for many iterations with minimal user
intervention.

Execution model
---------------

ALF is integrated with Parsl for task execution, so jobs can be launched on
local resources or queued cluster resources depending on your Parsl config.
Resource profiles are defined in ``alframework/parsl_resource_configs`` and
can be customized for your system.

Testing individual stages
-------------------------

Before long runs, ALF supports stage-level checks:

.. code-block:: bash

   python -m alframework --master master.json --test_builder
   python -m alframework --master master.json --test_sampler
   python -m alframework --master master.json --test_ml
   python -m alframework --master master.json --test_qm

These tests validate each stage independently before running the full active
learning workflow.
