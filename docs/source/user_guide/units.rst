Units And Property Conversion
=============================

ALF has no global unit system, and it does not record or check units at
runtime. Each run has to keep its QM results, HDF5 training data, ML model, and
sampler settings consistent. This page describes where the one conversion
happens and which units each part of ALF expects.

ALF Data Path
-------------

For a normal active-learning iteration:

1. A QM task stores values in ``MoleculesObject.results``, in whatever units
   its engine reports.
2. ``properties_list`` in ``master_config.json`` selects which result keys are
   written to HDF5.
3. ``store_current_data(...)`` multiplies each saved property by the conversion
   factor in ``properties_list``. This is the only conversion ALF applies.
4. The trainers read the HDF5 values as they are. Training units are whatever
   the shards hold.
5. Samplers run dynamics in ASE units, using the trained model's energies and
   forces.

``properties_list`` Format
--------------------------

A typical molecular example, labeled with ORCA, uses:

.. code-block:: json

   "properties_list": {
     "energy": ["energy", "system", 27.211386024367243],
     "forces": ["forces", "atomic", 51.422067090480645]
   }

Each property entry maps a parsed result to an HDF5 dataset:

.. list-table::
   :header-rows: 1
   :widths: 30 24 46

   * - JSON location
     - Example value
     - Meaning
   * - Property key
     - ``"energy"``
     - Property name expected in ``MoleculesObject.results``.
   * - First list entry
     - ``"energy"``
     - HDF5 dataset name written for ML training.
   * - Second list entry
     - ``"system"``
     - Property scope; see below.
   * - Third list entry
     - ``27.211386024367243``
     - Multiplicative conversion factor applied when writing the HDF5 dataset.

The property key and HDF5 dataset name are often the same, as in
``"energy": ["energy", ...]``, but they do not have to be. For example, a QM
task could store ``"SCF_energy"`` in ``MoleculesObject.results`` and write it
to an HDF5 dataset named ``"energy"``.

There are three scopes:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Scope
     - Shape
   * - ``system``
     - One value per structure, such as an energy or a dipole.
   * - ``atomic``
     - One value per atom, such as forces or charges.
   * - ``pair_atomic``
     - ``[pairs, atoms, 3]``, such as the nonadiabatic coupling vectors in
       :doc:`excited_states`.

ALF sorts atoms by atomic number before writing a shard, to match the sorted
``species`` and ``coordinates`` arrays. Atom-level values are reordered to
match, along the atom axis for both ``atomic`` and ``pair_atomic``.

Units Each Part Of ALF Uses
---------------------------

**QM tasks** store what their engine reports:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - QM task
     - Energies and forces
     - Usual factors
   * - ORCA
     - Hartree and Hartree/Bohr
     - ``27.211386024367243`` and ``51.422067090480645``, giving eV and eV/Å
   * - ASE calculators, including VASP
     - eV and eV/Å
     - ``1.0``
   * - PySEQM
     - eV and eV/Å
     - ``1.0``
   * - GPU4PySCF excited states
     - Converted to eV and eV/Å by the task. Dipoles stay in atomic units.
     - ``1.0``
   * - Foundation-model teachers
     - eV and eV/Å
     - Must be ``1.0``. ALF stops at startup otherwise; see
       :ref:`qm-teacher-labeling`.

**HIPPYNN models should be trained on eV and eV/Å.** ALF's HIPPYNN ASE
calculator declares its energies to be eV, and the batched sampling engines
take energies in eV and forces in eV/Å. A model trained on other units still
trains without error. But the samplers then run dynamics with every energy and
force scaled by the wrong factor.

**ANI models use Hartree.** The UO2 example trains ANI, so its factor of
``0.0367`` converts VASP's eV *to* Hartree.

**Samplers** use ASE units: Ångström, eV, and time through ``ase.units.fs``.
``Escut`` and ``Fscut`` are compared with the model's predictions, so they are
in the training units, usually eV and eV/Å.

What ALF Does Not Check
-----------------------

* Shards don't record their units. The one exception is teacher provenance,
  which records eV and eV/Å with the teacher.
* A seeded run trains on whatever shards are in ``h5_dir``. Seed shards must
  already be in the run's units, or the model trains on mixed units with no
  warning.
* Apart from teachers, nothing compares a factor with the engine's units.
  Copying a QM config from an example that uses a different engine is the usual
  way to get this wrong.

Practical Checks
----------------

Use the QM test mode before launching a long run:

.. code-block:: bash

   python -m alframework --master master_config.json --test_qm

This writes ``qm_test.h5`` and prints both parsed QM values and HDF5 values
after conversion. Compare those numbers to the external QM output and the
units your ML model expects.

Before production runs, also check:

* ML config keys match the HDF5 dataset names in ``properties_list``.
* ``Escut`` and ``Fscut`` in sampler configs are in the training units.
* Custom ML calculators return eV and eV/Å, as ASE expects.
* Custom QM tasks document the units they place in
  ``MoleculesObject.results``.

Related Pages
-------------

* :doc:`configuration` for the master config structure.
* :doc:`qm_interfaces` for QM parser and task setup.
* :doc:`ml_interfaces` for ML data keys and sampler calculator setup.
* :doc:`samplers` for MLMD and reactive sampler thresholds.
