Excited-State Cycloparaphenylene With GPU4PySCF
===============================================

This example runs active learning over the S0-S3 states of a cycloparaphenylene
(CPP) ring with TDDFT labels, in the gas phase or in implicit dichloromethane (DCM).
It includes minimized references for CPP6 (60 atoms) and CPP9 (90 atoms), and can
prepare other para-linked ring sizes. For a smaller first run, see
:doc:`acac_excited_pyseqm`. The pieces are explained in
:doc:`../user_guide/excited_states`.

The files needed to run the example are located in
``examples/CPP_excited_gpu4pyscf``.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Stage
     - What this example uses
   * - Initial geometries
     - ``custom_modules.ring_molecule_builder_task``, replaying phenyl rotations and
       rattles of the minimized reference.
   * - QM labels
     - ``custom_modules.gpu4pyscf_label_task``: CAM-B3LYP/6-31G with TDA, forces for
       every state, and coupling vectors. It wraps
       ``gpu4pyscf_excited_state_task``.
   * - Training
     - ``train_excited_state_HIPPYNN_ensemble_task``: a four-member ensemble with one
       head per state. Couplings and optical data are stored but not trained on.
   * - Sampling
     - ``mlmd_trajectories_task`` on the ALCHEMI engine in batches of 4, cycling through
       S0-S3.
   * - Acquisition
     - Summed ranking with ``cohort_top_k``, keeping a separate cohort for each state.

Before You Run
--------------

* Install HIPPYNN, RDKit, and ALCHEMI (``.[full,topology,alchemi]``) for training and
  sampling, and GPU4PySCF, PySCF, and CuPy for labeling; see
  :doc:`../getting_started/installation`. Preparation only needs ALF and ASE.
* **QM workers need native cuTENSOR.** Install a wheel matching your CUDA runtime, and
  put the wheel's ``cutensor/lib`` directory on ``LD_LIBRARY_PATH`` through
  ``ALF_QM_WORKER_INIT``. ``gpu4pyscf_label_task`` fails before calculating if it's
  missing, because CPP9 coupling calculations run out of memory on a 40 GB GPU without
  it. This example has been run with GPU4PySCF 1.9.1, PySCF 2.15.0, CuPy 14.2.0, and
  ``cutensor-cu13`` 2.8.1.
* Plan for a large GPU. See the memory notes below.

Prepare A Run
-------------

``prepare.py`` copies the example into a new directory, so the tracked files stay
unchanged. It runs on a CPU and submits nothing.

.. code-block:: bash

   cd examples/CPP_excited_gpu4pyscf
   python prepare.py --ring-size 9 --solvent dcm --output-dir /scratch/cpp9 --generate-seeds

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Option
     - Default
     - Meaning
   * - ``--output-dir``
     - Required
     - The new run directory. An existing directory is never overwritten.
   * - ``--ring-size``
     - ``9``
     - Number of phenylene rings. Sizes 6 and 9 use the included references.
   * - ``--solvent``
     - ``gas``
     - ``gas`` or ``dcm``. ``dcm`` selects ``QM_config_dcm.json`` and adds the
       ``vertical_*`` properties to ``properties_list``.
   * - ``--count``
     - ``1000``
     - Number of seed geometries for the bootstrap. Sets ``bootstrap_set``.
   * - ``--spares``
     - 5% of ``--count``
     - Extra seeds, each labeled only in place of one whose label failed. Sets
       ``bootstrap_spares``.
   * - ``--generate-seeds``
     - Off
     - Prepare the seeds now. Otherwise, prepare them yourself with
       ``python initial_sampling.py --config seed_config.json --output-dir seeds``.
   * - ``--model``
     - ``hiphop145``
     - Network size: ``hiphop145``, ``hiphop100``, ``hiphop64``, or ``hipnn64``.
   * - ``--reference``, ``--energy-offset-ev``
     - None
     - For another ring size, a minimized S0 XYZ file and its energy offset. Don't reuse
       the offset from another size. The method only supports para-linked CPP rings.

The included references are gas-phase minima, and ``--solvent dcm`` uses them too.
Seed preparation applies phenyl rotations and small rattles to the reference, half in
a narrow range and half in a broad one. It runs no QM or optimization. Edit
``seed_config.json`` to change those ranges.

Configure Parsl
---------------

``parsl_configs.py`` is a Slurm template that reads its settings from the environment,
with the variables listed in :doc:`acac_excited_pyseqm`. Importing it submits no jobs.
It adds ``ALF_QM_MAX_BLOCKS``, ``ALF_ML_MAX_BLOCKS``, and ``ALF_SAMPLER_MAX_BLOCKS``.
Unlike that template, every role's allocation lasts 16 hours by default, because labels
and trainings still running at the time limit are lost. If your partition's limit is
shorter, set ``ALF_QM_WALLTIME``, ``ALF_ML_WALLTIME``, and ``ALF_SAMPLER_WALLTIME`` to it.

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Role
     - Most nodes
     - Layout on each node
   * - QM
     - 8
     - Four workers, each with one GPU and eight CPU threads.
   * - Training
     - 1
     - One task that trains four members on four local GPUs.
   * - Sampling
     - 1
     - Four workers.

All three start with no nodes. Parsl requests nodes as work arrives, and releases them
when they are idle, so a smaller run can lower the limits instead of reserving nodes.
Training stays on one node here. For a larger ensemble, see :ref:`ml-multinode-training`.

**GPU binding.** GPU4PySCF needs each QM worker to see one GPU as its local device 0;
see :ref:`parsl-gpu4pyscf-workers`. ``parsl_configs.py`` handles this by launching QM
workers through ``qm_worker_pool.py``, which looks up the GPUs the scheduler allocated
and has Parsl bind one to each worker. Training and sampling assign their devices as
usual. To use this binding at another site, keep that launch command and copy
``qm_worker_pool.py`` along with it.

Run The Example
---------------

From the prepared directory:

.. code-block:: bash

   python -m alframework --master master_config.json

ALF labels all the seeds in order, then trains the first ensemble. After that, it
samples from each state, labels the selected frames, and retrains. New sampler starting
geometries come from the stored label files. Labels take minutes each at this size, so
run the coordinator somewhere that survives disconnecting, such as ``tmux``.

The bootstrap goes on until ``--count`` labels have converged. A seed whose label fails,
for example by running out of GPU memory or by losing its worker at the end of an
allocation, is replaced by the next spare seed. If the spares run out too, the first
ensemble trains on the labels that converged. Each bootstrap status report gives the
converged and failed counts and the spares used; look for the failed labels in the logs
and metadata.

Labels still running when a QM allocation reaches its time limit are lost, so keep
``ALF_QM_WALLTIME`` long; see `Configure Parsl`_.

If ALF Stops
~~~~~~~~~~~~

If the coordinator dies, for example because the login node rebooted, ``tmux`` was closed, or the
process was killed, restart it with the same command, from the same prepared directory:

1. **Cancel the run's old Slurm jobs first.** A coordinator killed outright can't release its
   allocations, and with 16-hour allocations they would sit idle for hours; the new coordinator can't
   use them. Parsl names each job ``parsl.alf_<role>_executor.block-<n>.<time>``:

   .. code-block:: bash

      squeue --me --format="%.12i %.50j %.10M"
      scancel JOBID [JOBID ...]   # this run's parsl.alf_ jobs

   If several ALF runs share your account, take this run's job IDs from the
   ``Launched block ... with job ID ...`` lines in its ``runinfo/*/parsl.log``. A coordinator stopped
   with ``SIGTERM``, or with Ctrl-C, releases its jobs itself.

2. **Start it again,** inside ``tmux``:

   .. code-block:: bash

      python -m alframework --master master_config.json

   The new coordinator prints a ``Restart:`` line for each thing it reconciles. Finished labels are
   kept, and a bootstrap labels only the seeds it doesn't hold yet. A seed whose label was running at
   the crash is labeled again, and isn't counted as a failure. A training that was running starts over.
   If the old coordinator is somehow still running, the new one refuses to start and names its host
   and process.

See :ref:`configuration-restarts` for what a restart keeps and what it costs.

Solvent
-------

``--solvent dcm`` turns on C-PCM with the settings in ``QM_config_dcm.json``:
dielectrics of 9.02 and 2.03, Lebedev order 29, and a radius scale of 1.2. See
:ref:`excited-states-solvent` for what each field does.

In DCM, state energies, forces, couplings, and the ordinary optical properties use
equilibrium linear-response PCM. The ``vertical_*`` properties come from a separate
nonequilibrium solve and are meant for vertical spectra. Each DCM label takes longer
than a gas-phase one, because of the solvent and the extra solve.

Memory
------

These are measurements on a 40 GB A100, for CPP9 with the default ``hiphop145`` network:

* **QM:** a gas-phase label reached about 39 GiB, close to the card's limit, so a harder
  geometry may not fit. A CPP6 label reached about 30 GiB.
* **Training:** ``hippynn_config.json`` uses a batch size of 8 for training, evaluation,
  and the scheduler limit, which peaked near 28 GiB. A batch of 16 reached about 37 GiB,
  and a batch of 32 ran out of memory. Dataset size mainly affects host memory.

Memory use for the smaller networks hasn't been measured.

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
     - Selected seeds, reference, and provenance.
   * - ``preparation.json``
     - The ring size, solvent, model, and energy offset used to prepare this run.
   * - ``h5store/data-*.h5``
     - Labeled batches, including coupling vectors and optical data.
   * - ``models/model-*``
     - Trained four-member ensembles.
   * - ``sampling/screen_audit/``
     - Initial, last valid, and rejected frames from the runtime screens.

The example's settings are a starting point. A run shows that the workflow operates on
your machine, not that the trained potential is accurate or stable over long dynamics,
so check the model against reference data before relying on it.
