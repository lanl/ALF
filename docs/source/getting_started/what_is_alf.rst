What Is ALF
===========

The **Active Learning Framework (ALF)** automates the construction of structurally diverse datasets for
machine learning interatomic potentials (MLIPs) using a query by committee (QBC) active-learning approach.

At a high level, ALF coordinates four core modules:

* **System construction (Builders)**
* **Sampling (Samplers)**
* **ML model training (ML interfaces)**
* **Electronic structure calculations (QM interfaces)**

**These modules are used to perform the main ALF workflow:**

1. Build initial structures (bootstrap) and label with a QM engine of choice. This step accomplishes initial sampling of the conformational/chemical space.
2. Train an initial MLIP ensemble from the labeled data.
3. Sample new structures with ML-driven molecular dynamics.
4. Send selected high-uncertainty configurations to QM engine for labeling.
5. Store labeled data in HDF5 files.
6. Retrain ML ensemble once the number of high-uncertainty configurations exceeds a user-defined threshold. Repeat until no high-uncertainty
   frames are identified, or the number of ALF iterations is exceeded. Use final MLIP to perform down-stream production task (Not part of ALF).



.. figure:: ../_static/alf.png
   :alt: Core ALF workflow from initial sampling (bootstrapping) to production-ready MLIP.
   :align: center
   :width: 85%

   Core ALF workflow from initial sampling (bootstrapping) to production-ready MLIP.



During an iteration, ALF builds candidate structures, samples configurations
using the current ML model ensemble, evaluates selected structures with QM,
and stores the resulting data for retraining. The newly trained models are then
used in the next sampling cycle.

This loop is designed to run for many iterations with minimal user
intervention.

Overview of the workflow
------------------------

ALF uses a master process to orchestrate tasks and data flow between the core
stages. The process is typically launched with:

.. code-block:: bash

   python -m alframework --master master_config.json

The ``master_config.json`` file points to the other configuration files and task
definitions needed for each stage, so you can choose their names. In practice,
ALF uses five JSON files. The examples name them like this:

1. Master configuration --- ``master_config.json``
2. Builder configuration --- ``builder_config.json``
3. Sampler configuration --- ``mlmd_config.json``
4. ML configuration --- ``hippynn_config.json``
5. QM configuration --- ``orca_config.json``

.. figure:: ../_static/alf_architecture.png
   :alt: Overview of ALF's code structure.
   :align: center
   :width: 85%

   Overview of ALF's code structure.



Execution model
---------------

ALF is integrated with Parsl for task execution, so jobs can be launched on
local resources or queued cluster resources depending on your Parsl config.
The master configuration names the Parsl config to use. Each example includes a
``parsl_configs.py`` template to edit for your system, and
``alframework/parsl_resource_configs`` holds profiles for the Darwin and Chicoma
clusters.

By default each task handles one structure. When a dynamics engine or QM code
can process many structures together, :doc:`../user_guide/batching` groups them
into wider tasks.

Testing individual stages
-------------------------

Before long runs, ALF supports stage-level checks:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_sampler
   python -m alframework --master master_config.json --test_ml
   python -m alframework --master master_config.json --test_qm

These tests validate each stage independently before running the full active
learning workflow.
