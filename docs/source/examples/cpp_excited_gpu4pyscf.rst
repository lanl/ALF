Excited-State Cycloparaphenylene With GPU4PySCF
===============================================

This example runs active learning over the S0-S3 states of a cycloparaphenylene
(CPP) ring with TDDFT labels, in the gas phase or in implicit dichloromethane (DCM).
It includes minimized references for CPP6 through CPP12 (60 to 120 atoms), and can
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
     - ``train_excited_state_HIPPYNN_ensemble_task``: a four-member HIP-NN ensemble
       (``hipnn64``) with one head per state, at batch 64. Couplings and optical data are
       stored but not trained on.
   * - Sampling
     - ``mlmd_trajectories_task`` on the ALCHEMI engine in batches of 16, cycling through
       S0-S3 one batch at a time.
   * - Acquisition
     - First trigger: each trajectory stops at its first uncertain frame, which is
       labeled. Energy, force, and adjacent-gap uncertainty can each trigger it.

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
     - Number of phenylene rings. Sizes 6 through 12 use the included references.
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
     - ``hipnn64``
     - Network: ``hipnn64``, or the larger HIP-HOP networks ``hiphop64``, ``hiphop100``, or
       ``hiphop145``. See the timings below before choosing a HIP-HOP network for a large
       dataset.
   * - ``--reference``, ``--energy-offset-ev``
     - None
     - For another ring size, a minimized S0 XYZ file and its energy offset. Don't reuse
       the offset from another size. The method only supports para-linked CPP rings.

The included references are gas-phase minima, and ``--solvent dcm`` uses them too.
All are CAM-B3LYP/6-31G, density-fitted S0 optimizations with the theory in
``QM_config.json``. Each ``references/cppN_s0.json`` records its source, energy,
residual gradient and the ``energy_offset_eV`` that ``prepare.py`` writes into
``QM_config.json``. To add a size, optimize its S0 geometry with the same theory, then
bundle it with ``python references/import_s0_reference.py /path/to/campaign/cppN``. The
script checks convergence, theory and topology, and refuses to overwrite a reference.
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
     - Four workers, one GPU each. Each task runs 16 trajectories on one state.

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

Sampling Settings
-----------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - In this example
   * - ``dt``, ``maxt``, ``min_time``
     - 0.5 fs steps for up to 1 ps at 300 K. Frames before 0.1 ps aren't labeled, since
       each trajectory starts from a stored label.
   * - ``Escut``, ``Fscut``
     - 0.10 eV on the ensemble spread of the selected state's total energy, about 1.1 meV
       per atom for CPP9, and 0.05 eV/Angstrom on the mean force spread. The largest force
       spread of any atom also triggers above 0.15 eV/Angstrom, and on a 90-atom ring it
       usually fires before the mean does.
       ``Escut`` applies to the whole molecule, so it is stricter per atom on larger
       rings. Consider scaling it with ring size.
   * - ``gap_uncertainty``, ``gap_std_cut``
     - The S0-S1, S1-S2, and S2-S3 gaps, with one cutoff of 0.05 eV on the largest of
       their spreads. In CPP rings, S2 and S3 are the bright states and S1 is dark, and
       excitation relaxes from S2 and S3 into S1. S2 and S3 are nearly degenerate near
       the minimum, so their gap's spread can dominate; if ``gap_lower_state`` in the
       trajectory summary is mostly 2, use ``"pair": [1, 2]`` instead.
   * - ``state_selection``
     - ``batch_cycle`` over S0-S3 with ``block_size`` 16. Each state is given to 16
       structures in a row, so each builder task (``maximum_builder_structures`` 16)
       fills one sampler batch.
   * - ``parallel_samplers``
     - 64: one batch of 16 for each of the four sampler GPUs.
   * - ``topology_check``
     - Ends trajectories whose bonds stretch below 0.85 or above 1.25 times the
       reference, or that form new bonds. See :ref:`samplers-runtime-screens`.

The cutoffs are starting values. Without cohorts, ALF rereads them while it runs, so
you can tune them in ``sampler_config.json`` without a restart. The master config
writes one row per trajectory to ``status_plots/trajectories.jsonl``, with the peak
``Es``, ``Fs``, ``Fsmax``, and ``gap_std`` of each; see the progress plots in
:doc:`../user_guide/samplers`. If almost every trajectory triggers at 0.1 ps, raise the
cutoffs. If many run to 1 ps without triggering, lower them, or the QM workers run
short of work.

Solvent
-------

``--solvent dcm`` turns on C-PCM with the settings in ``QM_config_dcm.json``:
dielectrics of 9.02 and 2.03, Lebedev order 29, and a radius scale of 1.2. See
:ref:`excited-states-solvent` for what each field does.

The gas-phase ``QM_config.json`` holds the same ``solvent`` block with ``"enabled": false``.
To try PCM on a few labels, for example with ``--test_qm``, set it to ``true``. That
labels with equilibrium PCM but adds no ``vertical_*`` properties, so prepare a DCM
campaign with ``--solvent dcm``.

In DCM, state energies, forces, couplings, and the ordinary optical properties use
equilibrium linear-response PCM. The ``vertical_*`` properties come from a separate
nonequilibrium solve and are meant for vertical spectra. Each DCM label takes longer
than a gas-phase one, because of the solvent and the extra solve.

Memory
------

These are measurements on a 40 GB A100, for CPP9:

* **QM:** a gas-phase label reached about 39 GiB, close to the card's limit, so a harder
  geometry may not fit. A CPP6 label reached about 30 GiB.
* **Larger rings:** in an earlier TDDFT benchmark on the same card, a CPP11 label ran out
  of memory when the three coupling pairs were solved together. It only fit when the
  pairs were solved one after another, which this example's labeler does not do. Before a
  CPP10 or larger campaign, label one seed as a test, or use a larger GPU.
* **Training with the default** ``hipnn64``: ``hippynn_config.json`` uses batch 64 for
  training, evaluation, and the scheduler limit. With 40,000 training structures, each of
  the four members took about 167 s per epoch and peaked at about 4 GiB allocated and
  12.5 GiB reserved. Batch 128 was barely faster and reserved about 32 GiB. At batch 64,
  the 300-epoch limit comes to about 14 hours, just within a 16-hour allocation.
* **Training with** ``hiphop145``: at batch 8 it peaked near 28 GiB, batch 16 reached about
  37 GiB, and batch 32 ran out of memory. An epoch at 40,000 training structures takes
  roughly an hour, so a 16-hour allocation fits only about 14 epochs.

Dataset size mainly affects host memory. These numbers come from repeated copies of the
same structures, so they measure capacity and speed, not accuracy. The batch-64 learning
rate hasn't been tuned; check the first trainings' validation errors.

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
