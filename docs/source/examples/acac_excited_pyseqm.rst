Excited-State Acetylacetone With PySEQM
=======================================

This example runs active learning over the four lowest singlet states (S0-S3) of
keto acetylacetone (AcAc), a 15-atom molecule. It is the smallest complete
excited-state workflow in ALF, so start here before moving to larger molecules. The
pieces are explained in :doc:`../user_guide/excited_states`.

The files needed to run the example are located in
``examples/AcAc_excited_pyseqm``.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Stage
     - What this example uses
   * - Initial geometries
     - ``rdkit_molecule_builder_task``, replaying seeds prepared with RDKit.
   * - QM labels
     - ``pyseqm_excited_state_batch_task``: AM1 with CIS, in native batches of 4.
   * - Training
     - ``train_excited_state_HIPPYNN_ensemble_task``: a four-member ensemble with one
       head per state.
   * - Sampling
     - ``mlmd_trajectories_task`` on the ALCHEMI engine in batches of 50, cycling
       through S0-S3.
   * - Acquisition
     - Summed ranking with ``cohort_top_k``, keeping a separate cohort for each state.

Before You Run
--------------

* Install HIPPYNN, RDKit, and ALCHEMI (``.[full,topology,alchemi]``) and PySEQM; see
  :doc:`../getting_started/installation`. Preparation only needs RDKit.
* The configuration runs training, sampling, and QM labeling on NVIDIA GPUs, and its
  Parsl template requests four GPUs per node. Edit the files if your layout differs.

Prepare A Run
-------------

``prepare.py`` copies the example into a new directory, so the tracked files stay
unchanged. It runs on a CPU and submits nothing.

.. code-block:: bash

   cd examples/AcAc_excited_pyseqm
   python prepare.py --output-dir /scratch/acac --generate-seeds

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Option
     - Default
     - Meaning
   * - ``--output-dir``
     - Required
     - The new run directory. An existing directory is never overwritten.
   * - ``--generate-seeds``
     - Off
     - Prepare the RDKit seeds now. Without it, you need to prepare ``seeds/`` yourself
       before running.
   * - ``--pool-size``, ``--seed-size``
     - ``50000``, ``5000``
     - Geometries in the pool, and how many become seeds. ``bootstrap_set`` is set to
       the seed size.
   * - ``--model``
     - ``hiphop145``
     - Network size: ``hiphop145``, ``hiphop64``, or ``hipnn64``.

For a quick trial, use something like ``--pool-size 300 --seed-size 30``.

The seeds go in ``seeds/``. The sampler's ``topology_check`` reads
``seeds/reference.xyz``, so a run needs that directory even with your own seeds. See
:ref:`builders-rdkit-seeds` for how they are made.

``prepare.py`` always uses ``seed_keto.json``. The ``seed_enol.json`` and ``enol.xyz``
files prepare the enol form the same way, with
``python -m alframework.builders.rdkit_molecule``.

Configure Parsl
---------------

``parsl_configs.py`` is a Slurm template that reads its settings from the environment.
Importing it submits no jobs. Set these before starting ALF:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Meaning
   * - ``ALF_PARTITION``, ``ALF_ACCOUNT``
     - Slurm partition and account. ``ALF_QM_PARTITION``, ``ALF_ML_PARTITION``, and
       ``ALF_SAMPLER_PARTITION`` override the partition for one role.
   * - ``ALF_QM_PYTHON``, ``ALF_ML_PYTHON``, ``ALF_SAMPLER_PYTHON``
     - Python interpreter for each role, for when they use separate environments.
   * - ``ALF_QM_WORKER_INIT``, ``ALF_ML_WORKER_INIT``, ``ALF_SAMPLER_WORKER_INIT``
     - Shell commands run when workers start, such as loading CUDA modules.
   * - ``ALF_QM_WALLTIME``, ``ALF_ML_WALLTIME``, ``ALF_SAMPLER_WALLTIME``
     - Allocation time limits. The defaults are 30 minutes for QM and sampling, and
       one hour for training.
   * - ``ALF_QM_RESOURCE_OPTIONS``, ``ALF_ML_RESOURCE_OPTIONS``,
       ``ALF_SAMPLER_RESOURCE_OPTIONS``
     - Scheduler options. The default is ``#SBATCH --gres=gpu:4``.
   * - ``ALF_SOURCE``
     - Optional ALF checkout to use instead of the installed package.
   * - ``ALF_ENV_TARBALL``, ``ALF_QM_ENV_TARBALL``, ``ALF_ML_ENV_TARBALL``,
       ``ALF_SAMPLER_ENV_TARBALL``
     - Optional archive of the Python environment, unpacked once on each node; the role variables
       override it for one role. ``ALF_ENV_PYTHON`` is its interpreter inside the archive, by default
       ``bin/python``. See :doc:`../user_guide/running_at_scale`.
   * - ``ALF_NODE_LOCAL_CACHES``, ``ALF_NODE_LOCAL_DIR``, ``ALF_CUPY_CACHE_DIR``
     - Optional. ``ALF_NODE_LOCAL_CACHES=1`` moves package caches to node-local storage and turns off
       HDF5 file locking. See :doc:`../user_guide/running_at_scale`.

See :doc:`../user_guide/parsl` for how executors are laid out.

Run The Example
---------------

From the prepared directory:

.. code-block:: bash

   python -m alframework --master master_config.json

ALF labels all the seeds, then trains the first ensemble. After that, it samples from
each state, labels the selected frames, and retrains. ``finite_bootstrap`` is on and
there are no spare seeds, so if a seed's label fails, the first ensemble trains on the
labels that converged. To replace failed labels instead, lower ``bootstrap_set`` below
the seed size and set ``bootstrap_spares`` to the difference; see
:doc:`../user_guide/configuration`. Look for failed labels in the logs and metadata.

Settings To Know
----------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - In this example
   * - ``state_selection``
     - ``batch_cycle`` over states 0-3. Each batch of trajectories stays on one state.
   * - ``selection``
     - ``cohort_size`` 50 and ``k`` 500, with a separate cohort for each state. With
       ``trajectory_policy`` set to ``continue``, a trajectory can add several frames.
   * - ``topology_check``
     - Ends trajectories that break the keto form's bonds. See
       :ref:`samplers-runtime-screens`.
   * - ``max_force_cutoff``
     - 16 eV/Angstrom on the unbiased forces of any state.
   * - ``screen_audit_dir``
     - ``sampling/screen_audit``. Look here to see which trajectories were rejected.
   * - ``energy_offset_eV``
     - -1391.45 eV, subtracted from every state energy in ``QM_config.json``. Set it
       for your own molecule.
   * - ``batching``
     - QM labels are solved in native batches of 4 and released after 60 seconds if a
       batch hasn't filled. See :doc:`../user_guide/batching`.

Expected Outputs
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Purpose
   * - ``status.json``
     - Restart state, including current HDF5 and model ids.
   * - ``seeds/``
     - The geometry pool, selected seeds, reference, and provenance.
   * - ``h5store/data-*.h5``
     - Labeled batches.
   * - ``models/model-*``
     - Trained four-member ensembles.
   * - ``sampling/screen_audit/``
     - Initial, last valid, and rejected frames from the runtime screens.

The example's settings are a starting point. A run shows that the workflow operates on
your machine, not that the trained potential is accurate, so check the model against
reference data before relying on it.
