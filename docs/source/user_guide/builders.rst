Builders
========

Builders construct the initial structures that enter an ALF active-learning
loop. A builder task typically returns a ``MoleculesObject`` containing an
``ase.Atoms`` object, a molecule id, and any metadata the sampler needs later.

In a master configuration, the selected builder is specified with the
``builder_task`` import string:

.. code-block:: json

   {
     "builder_task": "alframework.builders.builders.simple_condensed_phase_builder_task",
     "builder_config_path": "builder_config.json"
   }

Most builder tasks run on the ``alf_sampler_executor`` Parsl executor because
they feed candidate structures into the sampler queue.

Supported Builder Tasks
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 36 34 30

   * - Builder task
     - What it builds
     - Typical use
   * - ``alframework.builders.builders.simple_cfg_loader_task``
     - Loads a random ``.cfg`` structure from a configured library and applies
       a small coordinate perturbation.
     - Reusing pre-generated atomic configurations, such as the UO2 example.
   * - ``alframework.builders.builders.simple_condensed_phase_builder_task``
     - Places fragment-library molecules into a simulation cell with density,
       minimum-distance, and placement controls.
     - Standard molecular or condensed-phase workflows, including simple water
       and ionic-liquid examples.
   * - ``alframework.builders.builders.simple_multi_condensed_phase_builder_task``
     - Batched condensed-phase builder that returns multiple
       ``MoleculesObject`` instances from one builder task.
     - Higher builder throughput when ``maximum_builder_structures`` is set in
       the master config.
   * - ``alframework.builders.builders.atomic_system_task``
     - Creates a charge-neutral random atomic system from element charge rules,
       then places atoms in a cubic periodic box.
     - Molten salts or other ionic atomic systems.
   * - ``alframework.builders.builders.TiAl_builder_task``
     - Creates a specialized binary Ti/Al random atomic system.
     - Ti/Al alloy-style test or demonstration systems.
   * - ``alframework.builders.reactive_builder.load_reactive_task``
     - Loads reactant, transition-state, and product structures and stores the
       reaction set in metadata.
     - Reactive sampling workflows.
   * - ``alframework.builders.rdkit_condensed_phase.rdkit_condensed_phase_builder_task``
     - Builds molecular fragments from SMILES with RDKit, then uses
       condensed-phase placement logic.
     - Optional SMILES-based condensed-phase workflows when RDKit is installed.
   * - ``alframework.builders.external_geometry.external_geometry_builder_task``
     - Imports geometries from XYZ, extended XYZ, or HDF5 files, without any labels
       stored with them.
     - Starting from structures you already have, including frames from an existing
       dataset that need relabeling.
   * - ``alframework.builders.rdkit_molecule.rdkit_molecule_builder_task``
     - Replays molecular seed geometries prepared ahead of time with RDKit, then
       geometries from stored labels.
     - Single molecules that need varied starting geometries; see
       :ref:`builders-rdkit-seeds`.
   * - ``alframework.builders.defect_crystal_builder.defect_cell_builder_task``
     - Loads a random ``.cfg`` crystal structure, then applies random vacancies,
       interstitials, replacements, atom displacements, and cell scaling.
     - Crystals with point defects.

Choosing A Builder
------------------

Use ``simple_condensed_phase_builder_task`` for most workflows that start from
fragment files such as ``.xyz`` or other ASE-readable molecule files. This is
the standard choice for molecular liquid, solvent, and mixture examples.

Use ``simple_multi_condensed_phase_builder_task`` when builder submission
overhead is significant or when one builder task should produce several
independent candidates. Pair this with ``maximum_builder_structures`` in the
master configuration so ALF accounts for the number of structures returned by
each builder task.

Use ``atomic_system_task`` when the system is described by atomic species and
formal charges rather than molecular fragments. The task chooses a neutral
composition near ``target_num_atoms`` and enforces minimum-image placement
constraints in a cubic periodic box.

Use ``simple_cfg_loader_task`` when useful initial structures already exist as
``.cfg`` files. This is a lightweight loader rather than a packing builder.

Use ``load_reactive_task`` for reaction-path workflows. It loads one reactant,
transition-state, and product triplet, chooses one structure as the current
configuration, and stores the rest of the reaction context in metadata for the
reactive sampler.

Use ``rdkit_condensed_phase_builder_task`` only when RDKit is installed and
SMILES strings are the desired source of molecular fragments. It is optional
because RDKit is not part of the lightest ALF setup.

Use ``external_geometry_builder_task`` when you already have the starting
structures and want each one labeled again from scratch. It copies only the
geometry, so it never inherits results or calculator settings from the source file.

Common Configuration Fields
---------------------------

Fragment and condensed-phase builders
   These fields are used by ``simple_condensed_phase_builder_task``,
   ``simple_multi_condensed_phase_builder_task``, and related condensed-phase
   builders.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``molecule_library_dir`` or ``molecule_library_path``
        - Directory containing molecular fragment files. Older examples may use
          either spelling depending on the builder task.
      * - ``solute_molecule_options``
        - List of possible solute fragment sets. A builder task chooses one set
          for a generated structure.
      * - ``solvent_molecules``
        - Solvent fragments, optionally with relative sampling weights.
      * - ``cell_range``
        - Per-axis simulation-cell size ranges.
      * - ``Rrange``
        - Density or packing range sampled for generated structures.
      * - ``min_dist``
        - Minimum allowed interatomic distance during placement.
      * - ``max_patience``
        - Number of failed placement attempts before giving up on a structure.
      * - ``center_first_molecule``
        - Whether to center the first solute without random rotation.
      * - ``max_atoms``
        - Optional atom-count cap for generated systems.
      * - ``shake``
        - Random coordinate perturbation applied during placement or loading.

Atomic-system builders
   These fields are used by ``atomic_system_task`` and related atomic box
   builders.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``atom_charges``
        - Mapping from element symbol to formal charge used to choose a neutral
          composition.
      * - ``target_num_atoms``
        - Target system size. The final count may differ slightly so the system
          can remain neutral.
      * - ``min_distance``
        - Minimum atom-atom distance in Angstroms.
      * - ``box_length_range``
        - Range from which the cubic simulation-box length is sampled.

Reactive builders
   These fields are used by ``load_reactive_task``.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``molecule_library_dir``
        - Directory containing reaction structure files.
      * - ``r_files``
        - Reactant structure file or file pattern read by ASE.
      * - ``ts_files``
        - Transition-state structure file or file pattern read by ASE.
      * - ``p_files``
        - Product structure file or file pattern read by ASE.
      * - ``N_reactions``
        - Number of reaction entries available in the files.

External geometry builder
   These fields are used by ``external_geometry_builder_task``.

   .. code-block:: json

      {
        "sources": ["geometries/start.xyz", "geometries/more.h5"],
        "selection_mode": "deterministic_hash",
        "selection_seed": 7
      }

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``sources``
        - Required list of ``.xyz``, ``.extxyz``, ``.h5``, or ``.hdf5`` files.
          Relative paths start from ``master_directory``. HDF5 groups need
          ``coordinates`` shaped ``[frames, atoms, 3]`` and ``species``, and can also
          provide ``cell`` and ``pbc``.
      * - ``selection_mode``
        - ``"deterministic_hash"`` (default) picks a frame from the molecule ID and
          seed, and can pick the same frame more than once. ``"sequential"`` uses the
          trailing number in the molecule ID as the frame number, and fails once the
          sources run out of frames.
      * - ``selection_seed``
        - Seed for ``"deterministic_hash"``. The default is ``0``.

   Molecule metadata records where each frame came from. To return several
   structures from one task, set ``maximum_builder_structures``.

Defect crystal builder
   ``defect_cell_builder_task`` reads these fields:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``cfg_dir``
        - Directory containing ``.cfg`` structure files. Include the trailing ``/``.
      * - ``maxVac``, ``maxInt``, ``maxRep``
        - Most vacancies, interstitials, and replacements to apply. Each task picks a
          random count up to this value. The defaults are ``0``.
      * - ``typInt``, ``typRep``
        - Lists of element types for interstitials and replacements.
      * - ``rattle``, ``cRattle``
        - Standard deviations of random atom displacements and random changes to the
          cell lengths, in Angstroms. The defaults are ``0.0``.
      * - ``cScaleRange``
        - Range from which a cell scaling factor is drawn.
      * - ``maxAt``
        - The structure is repeated into a supercell with fewer than this many atoms.
          The default is ``100``.

Batched builders
   ``maximum_builder_structures`` belongs in the master configuration, not the
   builder configuration. It tells ALF how many candidate structures one builder
   task may return. Use it with
   ``simple_multi_condensed_phase_builder_task`` or
   ``external_geometry_builder_task``. See :doc:`batching` for how builder width
   relates to sampler and QM batches.

.. _builders-rdkit-seeds:

Preparing Molecular Seeds With RDKit
------------------------------------

``rdkit_molecule_builder_task`` doesn't generate geometries during a run. You prepare
them first on a CPU, which needs the ``topology`` extra:

.. code-block:: bash

   python -m alframework.builders.rdkit_molecule --config seed_config.json --output-dir seeds

Preparation embeds a large pool of geometries around your reference at increasing
levels of distortion, screens them, and uses farthest-point sampling (FPS) to pick a
varied subset. It runs no QM, dynamics, or force-field optimization, and it won't
overwrite an existing directory.

.. code-block:: json

   {
     "smiles": "CC(=O)CC(C)=O",
     "reference_geometry": "keto.xyz",
     "reference_charge": 0,
     "pool_size": 50000,
     "seed_size": 5000
   }

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Field
     - Default
     - Meaning
   * - ``smiles``, ``reference_geometry``
     - Required
     - The molecule, and its explicit-hydrogen reference geometry. A relative path
       starts from the seed configuration file.
   * - ``reference_charge``
     - From the SMILES
     - Molecular charge of the reference. It must match the SMILES formal charge.
   * - ``pool_size``, ``seed_size``
     - ``50000``, ``5000``
     - Geometries to accept into the pool, and how many to select from it.
   * - ``seed``, ``threads``
     - ``42``, ``1``
     - Random seed and RDKit embedding threads.
   * - ``box_size_mult``, ``max_embedding_rounds``, ``polar_h``
     - ``4.0``, ``5``, ``true``
     - Embedding box size, embedding attempts at each distortion level, and whether
       hydrogens on N, O, or S use the heavy-atom scales. Preparation fails if a level
       can't fill its share of the pool.
   * - ``bond_scales``, ``angle_scales``, ``xh_bond_scales``, ``xh_angle_scales``,
       ``torsion_scales``
     - Six levels each
     - How far each distortion level widens the bounds around the reference. All five
       lists must be the same length, one entry per level.
   * - ``geometry_screen``
     - See below
     - Rejects bad pool geometries, using ``bond_min_scale`` (``0.85``),
       ``bond_max_scale`` (``1.5``), ``nonbonded_min_scale`` (``1.4``),
       ``connectivity_scale`` (``1.25``), and ``min_distance_A`` (``0.7``).

The output directory holds ``pool.h5``, ``seed.h5``, a copy of the reference,
``builder_config.json``, and ``provenance.json``. The last file records every setting
and file fingerprint, and suggests the matching settings under ``alf_configuration``.
Add them to your run:

.. code-block:: json

   {
     "builder_task": "alframework.builders.rdkit_molecule.rdkit_molecule_builder_task",
     "builder_config_path": "seeds/builder_config.json",
     "bootstrap_set": 5000,
     "finite_bootstrap": true
   }

Set ``bootstrap_set`` to ``seed_size``. The builder config needs ``seed_manifest_path``,
pointing at ``provenance.json``, and can set ``selection_seed`` (default ``42``).

The builder stops if the prepared files have changed. It labels ``seed.h5`` in order
during the bootstrap. After that, it picks geometries from the stored label files,
keeping only the geometry and atom identities. Its atoms carry the
``topology_atom_ids`` that the sampler's ``topology_check`` needs; see
:ref:`samplers-runtime-screens`.

Related Examples
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Example
     - Builder task
     - Notes
   * - ``examples/simple_water``
     - ``alframework.builders.builders.simple_condensed_phase_builder_task``
     - Single condensed-phase water candidate per builder task.
   * - ``examples/simple_water_multi_builder``
     - ``alframework.builders.builders.simple_multi_condensed_phase_builder_task``
     - Batched version of the simple water builder.
   * - ``examples/molten_salt``
     - ``alframework.builders.builders.atomic_system_task``
     - Charge-neutral random ionic atomic systems.
   * - ``examples/UO2``
     - ``alframework.builders.builders.simple_cfg_loader_task``
     - Loads pre-generated ``.cfg`` structures.
   * - ``examples/reactive_sampling``
     - ``alframework.builders.reactive_builder.load_reactive_task``
     - Loads reactant, transition-state, and product structures.
   * - ``examples/IL``
     - ``alframework.builders.builders.simple_condensed_phase_builder_task``
     - Condensed-phase ionic-liquid style mixtures.
   * - ``examples/AcAc_excited_pyseqm``
     - ``alframework.builders.rdkit_molecule.rdkit_molecule_builder_task``
     - RDKit seeds for one molecule.

Builder API Links
-----------------

You can link from this guide directly to API pages:

* :doc:`Builders package API <../api_documentation/alframework.builders>`
* :doc:`Core builders module <../api_documentation/alframework.builders.builders>`
* :doc:`Molten-salt builder module <../api_documentation/alframework.builders.moltensalt_builder>`
* :doc:`Reactive builder module <../api_documentation/alframework.builders.reactive_builder>`
* :doc:`External geometry module <../api_documentation/alframework.builders.external_geometry>`
* :doc:`RDKit molecule module <../api_documentation/alframework.builders.rdkit_molecule>`
* :doc:`Defect crystal module <../api_documentation/alframework.builders.defect_crystal_builder>`