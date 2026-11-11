What Is ALF
===========

The **Active Learning Framework (ALF)** builds training datasets for machine-learned interatomic
potentials (MLIPs). It runs an active-learning loop: an ensemble of models drives sampling, frames
where the models disagree are labeled with quantum chemistry, and the ensemble is retrained on the
growing dataset, without a person in the loop.

ALF coordinates four kinds of task:

* **Builders** make starting structures.
* **Samplers** run ML-driven dynamics and pick frames to label.
* **QM interfaces** compute reference energies, forces, and other properties.
* **ML interfaces** train the model ensemble.

**A run goes like this:**

1. Build starting structures (the bootstrap) and label them with the QM code of your choice.
2. Train an initial MLIP ensemble on those labels.
3. Sample new structures with molecular dynamics driven by the current ensemble.
4. Send frames whose uncertainty passes the acquisition criteria to QM for labeling.
5. Store converged labels in HDF5 shards.
6. Retrain once enough new converged labels have arrived (``save_h5_threshold``), then sample with
   the new ensemble. Sampling and labeling continue while a training runs.
7. Repeat until ``maximum_stored_labels`` is reached, or until you stop the run. The final models
   are used for production simulations outside ALF.

What ALF Can Do
---------------

Beyond the basic loop:

* **Choose what to label:** uncertainty thresholds, ranking, per-trajectory policies, and
  selection across groups of trajectories; see :doc:`../user_guide/acquisition`.
* **Steer sampling:** uncertainty-driven dynamics, confining potentials, gap-seeking for excited
  states, and sampling windows that switch methods as the dataset grows; see
  :doc:`../user_guide/samplers`.
* **Excited states:** multi-state models with energy gaps and couplings, trained on TDDFT or
  semi-empirical labels; see :doc:`../user_guide/excited_states`.
* **Start from existing models:** fine-tune HIPPYNN ensembles or foundation models such as MACE and
  UMA, or label with a foundation model as a teacher; see :doc:`../user_guide/fine_tuning` and
  :doc:`../user_guide/teacher_student`.
* **Run on GPUs in bulk:** batched dynamics engines that advance many trajectories per GPU, and
  batched QM labeling; see :doc:`../user_guide/batching`.
* **Run at HPC scale:** Parsl executors on Slurm or Globus Compute, ensembles spread across nodes,
  members trained on several GPUs, restarts after a crash, and node-local environments; see
  :doc:`../user_guide/parsl` and :doc:`../user_guide/running_at_scale`.

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
``parsl_configs.py`` template to edit for your system; start from one of these.
``alframework/parsl_resource_configs`` holds older Darwin and Chicoma profiles,
kept for reference, whose accounts and executor layouts are out of date.

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
