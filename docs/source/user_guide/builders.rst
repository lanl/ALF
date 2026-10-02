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

Batched builders
   ``maximum_builder_structures`` belongs in the master configuration, not the
   builder configuration. It tells ALF how many candidate structures one builder
   task may return. Use it with
   ``simple_multi_condensed_phase_builder_task``.

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

Builder API Links
-----------------

You can link from this guide directly to API pages:

* :doc:`Builders package API <../api_documentation/alframework.builders>`
* :doc:`Core builders module <../api_documentation/alframework.builders.builders>`
* :doc:`Molten-salt builder module <../api_documentation/alframework.builders.moltensalt_builder>`
* :doc:`Reactive builder module <../api_documentation/alframework.builders.reactive_builder>`