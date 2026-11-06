Excited States
==============

ALF can run active learning over several electronic states of a molecule: it
labels every state with QM, trains one HIPPYNN ensemble with a head per state, and
runs dynamics on a chosen state.

Excited-state workflows are limited to nonperiodic molecules, and training needs
every structure in the dataset to have the same atoms in the same order. The
stages fit together like this:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - Task
   * - Initial geometries
     - Any builder. ``alframework.builders.external_geometry.external_geometry_builder_task``
       imports existing geometries without their old labels; see :doc:`builders`.
   * - QM labels
     - ``alframework.qm_interfaces.gpu4pyscf_interface.gpu4pyscf_excited_state_task``,
       ``alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_task``, or
       ``alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_batch_task``
   * - Training
     - ``alframework.ml_interfaces.hippynn_interface.train_excited_state_HIPPYNN_ensemble_task``
   * - Sampling
     - ``alframework.samplers.batched_mlmd.mlmd_trajectories_task``

There is no worked excited-state example in this guide yet, so treat the snippets
below as fragments to combine with a full configuration, such as the one in
:doc:`../examples/simple_water`.

Property Names
--------------

``properties_list`` in the master config declares the states, using these names:

.. code-block:: json

   {
     "properties_list": {
       "sE0": ["sE0", "system", 1.0],
       "sE1": ["sE1", "system", 1.0],
       "F0": ["F0", "atomic", 1.0],
       "F1": ["F1", "atomic", 1.0],
       "dE01": ["dE01", "system", 1.0]
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Key
     - Scope
     - Meaning
   * - ``sE0``, ``sE1``, ...
     - ``system``
     - Total energy of each state. Numbering must start at 0 with no gaps, and all
       states must use the same conversion factor.
   * - ``F0``, ``F1``, ...
     - ``atomic``
     - Forces on each state. QM labeling, training, and sampling all need one for
       every state.
   * - ``dE01``, ``dE12``, ...
     - ``system``
     - Optional energy gap from the lower state to the upper state. Use the form
       ``dE0_10`` when a state number has two digits. Gaps must use the same
       conversion factor as the state energies.
   * - ``nacr``
     - ``pair_atomic``
     - Optional nonadiabatic coupling vectors, from GPU4PySCF only. They are stored
       but not used in training.

Both labelers report eV and eV/Angstrom, so the conversion factors above are
``1.0``. See :doc:`units` for how those factors work.

Labeling With GPU4PySCF
-----------------------

``gpu4pyscf_excited_state_task`` runs singlet TDDFT within the Tamm-Dancoff
approximation (TDA) for one molecule per task, on a GPU. It needs a visible CUDA
device on the QM worker and a CuPy build matching the worker's CUDA runtime. To
pack several molecules into a job, use ``"mode": "sequential"``; see
:doc:`batching`.

The number of states in ``properties_list`` must equal ``nroots + 1``.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Default
     - Meaning
   * - ``xc``, ``basis``
     - ``cam-b3lyp``, ``6-31g*``
     - Functional and basis set.
   * - ``charge``, ``multiplicity``
     - ``0``, ``1``
     - Molecular charge. Multiplicity must be ``1``.
   * - ``nroots``
     - ``5``
     - Number of excited states.
   * - ``density_fit``, ``auxbasis``
     - ``true``, ``null``
     - Density fitting and its auxiliary basis.
   * - ``grids_level``
     - ``3``
     - DFT integration grid level.
   * - ``scf_conv_tol``, ``scf_max_cycle``
     - ``1e-10``, ``100``
     - SCF convergence.
   * - ``tda_conv_tol``, ``tda_max_cycle``, ``tda_lindep``
     - ``1e-8``, ``200``, ``1e-20``
     - TDA convergence.
   * - ``cphf_conv_tol``
     - ``1e-8``
     - Convergence of the coupled-perturbed equations for gradients.
   * - ``num_threads``, ``max_memory_mb``
     - ``8``, ``null``
     - CPU threads and memory limit.
   * - ``energy_offset_eV``
     - ``0.0``
     - Value subtracted from every state energy.
   * - ``compute_nacr``
     - ``false``
     - Also compute excited-excited coupling vectors. Requires ``nroots`` of at
       least 2 and a ``nacr`` entry in ``properties_list``. ALF records the state
       pair order in an HDF5 dataset named ``nac_pairs``.

Labeling With PySEQM
--------------------

The PySEQM tasks run AM1 with configuration interaction singles (CIS). They support
neutral, closed-shell molecules only.

* ``pyseqm_excited_state_batch_task`` solves several molecules in one call. Use it
  with ``"mode": "native"``; see :doc:`batching`.
* ``pyseqm_excited_state_task`` labels one molecule per task.

.. code-block:: json

   {
     "method": "AM1",
     "device": "cuda",
     "cis_tol": 1e-8,
     "batching": {"mode": "native", "batch_size": 64}
   }

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Default
     - Meaning
   * - ``method``
     - ``AM1``
     - Must be ``AM1``.
   * - ``device``
     - ``cpu``
     - Torch device. For GPU labeling, place ``alf_QM_executor`` on GPU nodes.
   * - ``cis_tol``, ``scf_eps``
     - ``1e-8``, ``1e-10``
     - CIS and SCF tolerances. The SCF tolerance used is at most a tenth of
       ``cis_tol``.
   * - ``timeout_seconds``
     - ``300``
     - Time limit for one PySEQM solve.
   * - ``num_threads``
     - ``1``
     - CPU threads.
   * - ``energy_offset_eV``
     - ``0.0``
     - Value subtracted from every state energy.

PySEQM needs at least two states and only accepts ``sE#``, ``F#``, and ``dE#``
properties. Within a batch, ALF only solves molecules with the same atoms together,
so a mixed batch runs as several smaller solves. If a batch solve fails, ALF retries
each member alone, and only converged members are stored.

Training
--------

Point the master config at the excited-state trainer. It reads the states from
``properties_list``:

.. code-block:: json

   {
     "ML_task": "alframework.ml_interfaces.hippynn_interface.train_excited_state_HIPPYNN_ensemble_task",
     "ML_config_path": "hippynn_config.json"
   }

Its ML config follows the ground-state options in :doc:`ml_interfaces`, with these
differences:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``cell_key``
     - Must be ``null``, since training is nonperiodic.
   * - ``train_forces``, ``export_force_gradients``
     - Must be ``true`` (the default), so that sampling can use forces from every
       state.
   * - ``network_choice``
     - ``0`` for HipNN or ``1`` for HipHopNN (the default).
   * - ``energy_weight``, ``force_weight``, ``l2_weight``
     - Loss weights.
   * - ``gap_targets``
     - Optional gap training, such as ``{"enabled": true, "weight": 1.0, "pairs":
       [[0, 1]]}``. Each gap needs a ``dE##`` property, and gaps are derived from
       the state heads rather than trained as separate heads.
   * - ``optimizer``
     - Must be ``AdamW``.
   * - ``batch_size``, ``eval_batch_size``, ``max_epochs``, ``termination_patience``
     - Training loop settings, set directly rather than through
       ``controller_options``.
   * - ``exports``
     - Optional plot output, such as ``{"pdf": true, "png": true}``. CSV export is
       not supported.
   * - ``hippynn_kernel``
     - HIPPYNN kernel implementation. The default is ``pytorch``.

The trainer never deletes or overwrites model data, so it doesn't support
``remove_existing``. It also doesn't support ``h5_test_dir``; use ``valid_size`` and
``test_size`` to hold out data.

Sampling On A Selected State
----------------------------

Excited-state sampling uses ``mlmd_trajectories_task`` with
``"model_mode": "excited_state"``. A trajectory moves on the ensemble-mean forces of
its selected state, and ``Es`` and ``Fs`` uncertainty is measured on that state.

.. code-block:: json

   {
     "model_mode": "excited_state",
     "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
     "gap_uncertainty": {"pair": [0, 1]},
     "gap_std_cut": 0.1,
     "batched_calculator": "alframework.ml_interfaces.hippynn_interface.load_hippynn_excited_state_ensemble",
     "batched_calculator_options": {
       "properties_list": {
         "sE0": ["sE0", "system", 1.0],
         "sE1": ["sE1", "system", 1.0],
         "F0": ["F0", "atomic", 1.0],
         "F1": ["F1", "atomic", 1.0],
         "dE01": ["dE01", "system", 1.0]
       }
     },
     "driver": {"strategy": "batched", "engine": "torchsim"},
     "batching": {"batch_size": 32},
     "acquisition": {"eligibility": ["energy", "force_mean", "force_max", "gap"]}
   }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``model_mode``
     - ``"excited_state"`` turns this on. The default is ``"ground_state"``.
   * - ``state_selection``
     - Which state each trajectory uses. ``{"mode": "fixed", "state": 1}`` uses one
       state. ``{"mode": "batch_cycle", "states": [0, 1]}`` assigns the listed
       states in turn as structures are admitted. A ``selected_state`` value in a
       molecule's metadata takes priority over either.
   * - ``batched_calculator``, ``batched_calculator_options``
     - The model loader, which must provide every state. Options must include the
       ``properties_list`` used in training, and can include ``hippynn_kernel``.
   * - ``gap_uncertainty``
     - Optional. ``pair`` names one lower and one upper state, and ALF records the
       standard deviation of their gap for each frame.
   * - ``gap_std_cut``
     - Cutoff for the ``gap`` acquisition term. To trigger labeling on gap
       uncertainty, also add ``"gap"`` to ``acquisition.eligibility``. See
       :doc:`acquisition`.

Limits:

* UDD and well potentials are not available in excited-state mode. ALF stops at
  startup if ``MLMD_calculator_options`` sets either one.
* Each sampler task holds one state, so ALF batches trajectories by state.
* Structures must be nonperiodic.

Excited-State API Links
-----------------------

* :doc:`Excited-state tools module <../api_documentation/alframework.tools.excited_state_tools>`
* :doc:`GPU4PySCF interface module <../api_documentation/alframework.qm_interfaces.gpu4pyscf_interface>`
* :doc:`PySEQM interface module <../api_documentation/alframework.qm_interfaces.pyseqm_interface>`
* :doc:`HIPPYNN interface module <../api_documentation/alframework.ml_interfaces.hippynn_interface>`
