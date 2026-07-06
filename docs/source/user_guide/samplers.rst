Samplers
========

Samplers explore configuration space and decide whether a structure should be
sent to QM labeling. In most ALF workflows, the sampler receives a
``MoleculesObject`` from the builder, evaluates it with an ensemble ML
calculator, and returns either a selected high-uncertainty configuration or a
marker that no QM calculation is needed.

In a master configuration, the selected sampler is specified with the
``sampler_task`` import string:

.. code-block:: json

   {
     "sampler_task": "alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task",
     "sampler_config_path": "mlmd_config.json"
   }

Sampler tasks usually run on the ``alf_sampler_executor`` Parsl executor. GPU
assignment is controlled by ``gpus_per_node`` in the master configuration and
the worker layout in the Parsl resource configuration.

Supported Sampler Tasks
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 34 26

   * - Sampler task
     - What it does
     - Typical use
   * - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - Runs Langevin molecular dynamics with an ensemble ML calculator,
       monitors energy and force uncertainty, and returns configurations that
       exceed uncertainty or distance criteria.
     - General active-learning loops based on MD exploration.
   * - ``alframework.samplers.reactive_sampler.reactive_sampling``
     - Uses NEB and/or dimer-style searches from reaction metadata, monitors
       ensemble uncertainty along reaction pathways, and returns uncertain
       structures for QM labeling.
     - Reaction-pathway active learning with reactant, transition-state, and
       product structures.

Choosing A Sampler
------------------

Use ``simple_mlmd_sampling_task`` for most ALF workflows. It is designed for
active-learning loops where an MLIP ensemble explores configuration space and
uncertainty thresholds decide which structures need QM labels.

Use ``reactive_sampling`` when the builder provides reaction metadata. The
input ``MoleculesObject`` must contain reactant, transition-state, and product
structures, such as those returned by
``alframework.builders.reactive_builder.load_reactive_task``.

Use HIPPYNN-style MLMD sampling when ``ase_calculator`` points to a model
loader such as ``alframework.ml_interfaces.hippynn_interface.HIPNN_ASE_load_ensemble``.
Use NeuroChem-specific sampling behavior when ``use_potential_specific_code``
is set to ``"neurochem"`` and ``ase_calculator`` points to the NeuroChem
calculator loader.

Keep sampler resource expectations in sync with Parsl. MLMD and reactive
sampling often need GPUs, so ``alf_sampler_executor``, ``gpus_per_node``, and
cluster GPU scheduler options should agree. See :doc:`parsl` for resource
configuration guidance.

Common Configuration Fields
---------------------------

MLMD dynamics fields
   These fields control the length and cadence of MD sampling.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``dt``
        - MD timestep in femtoseconds.
      * - ``maxt``
        - Maximum simulation time in picoseconds.
      * - ``Ncheck``
        - Number of MD steps between uncertainty checks.
      * - ``min_time``
        - Optional minimum simulation time before uncertainty checks can select
          a structure.
      * - ``distcut``
        - Minimum allowed interatomic distance. Structures can be rejected if
          atoms come too close.

Uncertainty thresholds
   These fields decide when a sampled configuration should be sent to QM.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``Escut``
        - Energy standard-deviation threshold.
      * - ``Fscut``
        - Mean force standard-deviation threshold.
      * - ``forces_stdev_max``
        - The MLMD sampler also checks the maximum per-force uncertainty using
          an internal threshold of ``3 * Fscut``.

Temperature schedule fields
   The sampler randomly chooses temperature schedule parameters from configured
   ranges for each task.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``srt_temp``
        - Range for the starting temperature.
      * - ``end_temp``
        - Range for the ending temperature.
      * - ``amp_temp``
        - Range for sinusoidal temperature fluctuation amplitude.
      * - ``per_temp``
        - Range for temperature fluctuation period.

Density schedule fields
   Density changes are optional. Use ``null`` for ``end_dens``, ``amp_dens``,
   and ``per_dens`` to disable density changes.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``end_dens``
        - Target final density range. ``null`` disables cell rescaling.
      * - ``amp_dens``
        - Range for density fluctuation amplitude.
      * - ``per_dens``
        - Range for density fluctuation period.

ML calculator fields
   These fields control how sampler tasks load and wrap ML models.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``ase_calculator``
        - Import string for the ML calculator or model-loader function.
      * - ``MLMD_calculator_options``
        - Options passed to ``MLMD_calculator``, such as ``well_params``.
      * - ``translate_to_center``
        - Whether to translate the incoming structure so its center of mass is
          near the origin before sampling.

Output and trajectory fields
   These fields control sampler metadata and optional trajectory output.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``meta_dir`` or ``meta_path``
        - Directory for sampler metadata files. Examples use both names; the
          task input builder maps compatible names to the sampler function.
      * - ``trajectory_frequency``
        - Probability of writing a trajectory for a sampling task.
      * - ``trajectory_interval``
        - MD-step interval for trajectory writes when trajectory output is
          enabled.

Reactive sampler fields
   These fields are used by ``reactive_sampling``.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``reactive_sampling_method``
        - Reactive search mode. Values containing ``NEB`` run NEB; values
          containing ``DIMER`` run dimer sampling.
      * - ``N_neb``
        - Number of intermediate NEB images.
      * - ``max_iter``
        - Maximum number of repeated reactive-sampling attempts.
      * - ``neb_steps``
        - Optimizer steps for each NEB or dimer call.
      * - ``Escut``, ``Fscut``
        - Energy and force uncertainty thresholds.
      * - ``Fmmult``
        - Multiplier applied to ``Fscut`` for maximum force uncertainty. The
          default is ``3.0``.

What The Sampler Returns
------------------------

If uncertainty or another selection criterion triggers, the returned
``MoleculesObject`` keeps its atoms. ALF then passes that structure to the QM
task queue for labeling.

If sampling completes without selecting a configuration, the sampler sets the
atoms to ``None`` on the returned ``MoleculesObject``. ALF treats this as a
successful sampling attempt that does not need QM labeling.

Sampler metadata records the relevant diagnostics, including uncertainty
values, temperature and density schedule history, distance checks, selection
flags, final positions, and cell information. MLMD metadata is commonly written
to files such as ``sampling/metadata-<moleculeid>.p`` when ``meta_dir`` or a
compatible metadata path is configured.

Related Examples
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Example
     - Sampler task
     - Notes
   * - ``examples/simple_water``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - HIPPYNN ensemble MLMD sampling.
   * - ``examples/simple_water_multi_builder``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - Same sampler as simple water; builder throughput changes.
   * - ``examples/molten_salt``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - MLMD sampling for periodic ionic systems.
   * - ``examples/UO2``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - Uses ``use_potential_specific_code: "neurochem"``.
   * - ``examples/IL``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - MLMD sampling for ionic-liquid style mixtures.
   * - ``examples/reactive_sampling``
     - ``alframework.samplers.reactive_sampler.reactive_sampling``
     - NEB/dimer-style reaction-pathway sampling.

Sampler API Links
-----------------

You can link from this guide directly to API pages:

* :doc:`Samplers package API <../api_documentation/alframework.samplers>`
* :doc:`MLMD sampling module <../api_documentation/alframework.samplers.mlmd_sampling>`
* :doc:`Reactive sampler module <../api_documentation/alframework.samplers.reactive_sampler>`
* :doc:`ASE ensemble calculator module <../api_documentation/alframework.samplers.ASE_ensemble_constructor>`

