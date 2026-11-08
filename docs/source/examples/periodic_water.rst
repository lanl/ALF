Periodic Water With PySCF
=========================

This example learns liquid water in periodic boxes, labeled with periodic DFT from PySCF on the CPU or
GPU4PySCF on a GPU, and trains a periodic HIPPYNN ensemble. Both QM packages are free and
pip-installable, with ALF's ``pyscf`` or ``gpu4pyscf`` extra. The files are in ``examples/periodic_water``.

How It Runs
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - Task and settings
   * - Builder
     - ``simple_condensed_phase_builder_task`` with ``"periodic": true``: boxes of 7–8 Å at 0.9–1.0 g/cm³,
       with ``min_dist: 1.6`` and ``shake: 0.02``, so the starting structures are realistic enough for
       periodic DFT to converge.
   * - Sampler
     - ``simple_mlmd_sampling_task`` with the HIPPYNN ensemble, sampling densities of 0.9–1.1.
   * - QM
     - ``pyscf_task``: PBE, GTH-DZVP with GTH pseudopotentials, at the Γ point.
   * - ML
     - ``train_HIPPYNN_ensemble_task`` with ``cell_key: "cell"``, so the model is periodic.

Two QM configurations are provided:

``qm_config.json``
   ``"backend": "gpu"`` with ``"df": "gdf"``, for production. It needs GPU4PySCF; see
   :ref:`qm-interfaces-pyscf`.
``qm_config_cpu.json``
   ``"backend": "cpu"`` with ``"df": "multigrid"``, for bring-up without a GPU. Set it as
   ``QM_config_path`` in ``master_config.json``.

Both store the ground state as ``energy`` and ``forces``, which the trainer reads, and the shards record
each box's ``cell``. To also store the stress, add ``"stress": ["stress", "system", 1.0]`` to
``properties_list``; on the CPU, that needs ``"df": "fftdf"``.

Running
-------

From ``examples/periodic_water``, after setting the machine in ``parsl_configs.py``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json

With ``qm_config_cpu.json``, ``--test_qm`` labeled a 15-water box of 7.7 Å in about a minute on a laptop,
abridged:

.. code-block:: text

   System: H30O15, Cell: Cell([7.975659559937997, 7.562146839600099, 7.656165091839759])
   PBCs: [ True  True  True]
   E0: -7018.955660661688
   ...
   Data written to qm_test.h5:
   energy:
   [-7018.9557]

If many structures fail with ``qm_failed_stage: "scf"``, the builder's boxes are probably too strained:
raise ``min_dist`` or lower the density range before turning to ``"smearing_eV"``.
