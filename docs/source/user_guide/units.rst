Units And Property Conversion
=============================

ALF does not enforce one global unit system at runtime. Instead, each workflow
must keep the QM parser, HDF5 training data, ML configuration, and sampler
calculator units consistent. This page describes where conversion happens and
where users should check their configs.

ALF Data Path
-------------

For a normal active-learning iteration:

1. A QM task parses engine output and stores values in
   ``MoleculesObject.results``.
2. ``properties_list`` in ``master_config.json`` selects which result keys are
   written to HDF5.
3. ``store_current_data(...)`` multiplies each saved property by the conversion
   factor in ``properties_list``.
4. ML config files read the HDF5 dataset names written by ``properties_list``.
5. Sampler calculators return energies and forces to ASE during MLMD or
   reactive sampling.

The important point is that the third entry in each ``properties_list`` record
is a multiplicative conversion applied when ALF writes HDF5 training data.

``properties_list`` Format
--------------------------

A typical molecular example uses:

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
     - Property scope. Use ``"system"`` for one value per structure and
       ``"atomic"`` for one value per atom.
   * - Third list entry
     - ``27.211386024367243``
     - Multiplicative conversion factor applied when writing the HDF5 dataset.

The property key and HDF5 dataset name are often the same, as in
``"energy": ["energy", ...]``, but they do not have to be. For example, a QM
task could store ``"SCF_energy"`` in ``MoleculesObject.results`` and write it
to an HDF5 dataset named ``"energy"``.

For atom-level properties, ALF sorts atoms by atomic number before writing the
dataset, matching the sorted ``species`` and ``coordinates`` arrays.

Custom QM tasks should document the units they place in
``MoleculesObject.results``. Then choose the ``properties_list`` multiplier so
the HDF5 data match the ML configuration.

Practical Checks
----------------

Use the QM test mode before launching a long run:

.. code-block:: bash

   python -m alframework --master master_config.json --test_qm

This writes ``qm_test.h5`` and prints both parsed QM values and HDF5 values
after conversion. Compare those numbers to the external QM output and the
expected ML training units.

Before production runs, also check:

* ML config keys match the HDF5 dataset names in ``properties_list``.
* ``Escut`` and ``Fscut`` in sampler configs use units compatible with the
  sampler calculator results.
* Custom ML calculators return ASE-compatible units unless the sampler path
  explicitly handles another convention.

Related Pages
-------------

* :doc:`configuration` for the master config structure.
* :doc:`qm_interfaces` for QM parser and task setup.
* :doc:`ml_interfaces` for ML data keys and sampler calculator setup.
* :doc:`samplers` for MLMD and reactive sampler thresholds.
