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

The snippets below are fragments. For complete runs, see
:doc:`../examples/acac_excited_pyseqm`, which uses PySEQM, and
:doc:`../examples/cpp_excited_gpu4pyscf`, which uses GPU4PySCF with an optional
solvent.

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
   * - ``transition_dipoles``, ``oscillator_strengths``, ``absolute_state_energies``
     - ``system``
     - Optional optical outputs, from GPU4PySCF only. Dipoles are in atomic units, and
       absolute energies are in eV with no offset removed. They are stored but not
       used in training.
   * - ``vertical_excitation_energies``, ``vertical_transition_dipoles``,
       ``vertical_oscillator_strengths``
     - ``system``
     - Optional vertical spectrum, from GPU4PySCF only; see
       :ref:`excited-states-solvent`. The energies must use the same conversion factor
       as the state energies. They are stored but not used in training.

Both labelers report eV and eV/Angstrom, so the conversion factors above are
``1.0``. See :doc:`units` for how those factors work.

Labeling With GPU4PySCF
-----------------------

``gpu4pyscf_excited_state_task`` runs singlet TDDFT within the Tamm-Dancoff
approximation (TDA) for one molecule per task, on a GPU. It needs a CuPy build
matching the worker's CUDA runtime, and each QM worker must see exactly one GPU, as
its local device 0; see :ref:`parsl-gpu4pyscf-workers`. To pack several molecules into
a job, use ``"mode": "sequential"``; see :doc:`batching`.

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
   * - ``solvent``
     - ``null``
     - Optional implicit solvent; see :ref:`excited-states-solvent`.

Large molecules need a lot of GPU memory, and native cuTENSOR to stay within it; see
:doc:`../getting_started/installation`. As a guide, a CPP9 ring (90 atoms at
CAM-B3LYP/6-31G with four states and couplings) reaches about 39 GiB on a 40 GB A100
in the gas phase, which leaves no margin for larger or harder geometries.

.. _excited-states-solvent:

Implicit Solvent
^^^^^^^^^^^^^^^^

``solvent`` adds a C-PCM solvent to GPU4PySCF labels. Setting ``{"enabled": true}``
alone gives dichloromethane:

.. code-block:: json

   {
     "solvent": {
       "enabled": true,
       "name": "dichloromethane",
       "static_dielectric": 9.02,
       "optical_dielectric": 2.03
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 36 20 44

   * - Field
     - Default
     - Meaning
   * - ``enabled``
     - ``true``
     - ``false`` runs in the gas phase. ALF rejects an empty ``solvent`` block, so a
       mistake can't silently label in vacuum.
   * - ``name``
     - ``dichloromethane``
     - Descriptive label, recorded in each molecule's metadata. It doesn't set any
       physical values, so also set the dielectrics for another solvent.
   * - ``static_dielectric``, ``optical_dielectric``
     - ``9.02``, ``2.03``
     - Must satisfy ``static_dielectric >= optical_dielectric >= 1``.
   * - ``method``, ``surface_discretization_method``
     - ``C-PCM``, ``SWIG``
     - The only supported values.
   * - ``lebedev_order``
     - ``29``
     - Cavity surface grid. Must be an order that PySCF supports.
   * - ``vdw_scale``, ``pcm_conv_tol``
     - ``1.2``, ``1e-7``
     - Cavity radius scale and PCM tolerance.

Two kinds of results come from a solvent calculation:

* **State energies, forces, couplings, and the ordinary optical properties** use
  equilibrium linear-response PCM. These are linear-response energies, not
  state-specific PCM energies. Each coupling pair is solved separately.
* **The ``vertical_*`` properties** come from a separate nonequilibrium TDA solve using
  the optical dielectric. Their roots are ordered on their own, so a vertical root
  isn't guaranteed to be the same state as the equilibrium root with that number.
  ``energy_offset_eV`` doesn't apply to excitation energies.

ALF computes the extra solve only if ``properties_list`` asks for a ``vertical_*``
property. In the gas phase, these properties come from the same TDA solve as the
state energies. Solvent calculations take longer than gas-phase ones.

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
   * - ``MLMD_calculator_options``
     - Optional ``well_params`` and ``udd_bias_weight``, as in :doc:`samplers`. They
       act on the selected state's member energies and forces. Screening and
       uncertainty still read unbiased predictions.

Limits:

* Each sampler task holds one state, so ALF batches trajectories by state.
* Structures must be nonperiodic.

Cohorts can also be kept separate for each state, with
``selection.group_by_selected_state``; see :doc:`acquisition`. To end trajectories that
break a bond or develop unphysical forces on any state, see
:ref:`samplers-runtime-screens`.

Gap-Seeking
-----------

Gap-seeking samples geometries near a state crossing. Once a trajectory's gap to an
adjacent state is small enough, it permanently leaves its selected state and moves on a
penalty surface for that pair, which pushes it toward the crossing. The surface is the
one from Levine, Coe, and Martinez.

.. code-block:: json

   {
     "gap_seeking": {
       "enabled": true,
       "trigger_gap_threshold_eV": 0.05,
       "sigma": 3.5,
       "alpha_eV": 0.05
     }
   }

For states :math:`i` and :math:`j`, with ensemble-mean energies :math:`E_i` and
:math:`E_j` and :math:`\Delta E = E_j - E_i`, the surface is:

.. math::

   E_\mathrm{gap} = \frac{E_i + E_j}{2}
   + \sigma \frac{\Delta E^2}{\sqrt{\Delta E^2 + 10^{-12}} + \alpha}

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Meaning
   * - ``enabled``
     - ``true`` turns this on. The default is ``false``. It requires
       ``"model_mode": "excited_state"``.
   * - ``trigger_gap_threshold_eV``
     - Required. A trajectory switches when its gap, in eV, is at or below this value.
   * - ``sigma``, ``alpha_eV``
     - Required. :math:`\sigma` and :math:`\alpha` in the surface above.
   * - ``mode``, ``switch_policy``, ``candidate_pairs``
     - Optional. The only supported values are ``levine_coe_martinez_switch``,
       ``stay_fixed``, and ``adjacent``.

How it runs:

* **Check:** at each ``Ncheck``, after the usual screening and uncertainty checks, each
  trajectory still running looks at the adjacent pairs containing its selected state,
  and takes the smallest absolute gap. On a tie, it takes the lower pair.
* **Switch:** only that trajectory switches, and it never switches back, even if the gap
  grows again. Each trajectory can pick its own pair.
* **What changes:** only the forces that move the trajectory. Screening, ``Es``, ``Fs``,
  and the rest of acquisition still read unbiased predictions on the selected state.
* **Wells:** ``well_params`` adds on top of the surface. ALF stops at startup if
  ``udd_bias_weight`` is also set, because UDD's bias describes the selected state.
* **Records:** each molecule's metadata has ``gap_seeking_current_mode`` and
  ``gap_seeking_switched``, plus ``gap_seeking_pair`` and the trigger gap, step, and time
  once it has switched. Frames sampled after a switch carry the same fields.

Excited-State API Links
-----------------------

* :doc:`Excited-state tools module <../api_documentation/alframework.tools.excited_state_tools>`
* :doc:`GPU4PySCF interface module <../api_documentation/alframework.qm_interfaces.gpu4pyscf_interface>`
* :doc:`PySEQM interface module <../api_documentation/alframework.qm_interfaces.pyseqm_interface>`
* :doc:`HIPPYNN interface module <../api_documentation/alframework.ml_interfaces.hippynn_interface>`
