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

Sampler tasks usually run on the ``alf_sampler_executor`` Parsl executor.

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
   * - ``alframework.samplers.udd_sampling.simple_udd_sampling_task``
     - Runs MLMD with an uncertainty-driven dynamics bias that steers
       trajectories toward high ensemble energy uncertainty while retaining the
       usual QM-selection threshold checks.
     - Exploration runs where biasing MD toward model disagreement is desired.
   * - ``alframework.samplers.batched_mlmd.mlmd_trajectories_task``
     - Runs one or more MLMD trajectories in one task, with a choice of dynamics
       engine and of how frames are selected for QM. See
       `Multi-Trajectory Sampling`_.
     - Running many trajectories on one GPU, ranking frames before labeling, or
       excited-state sampling.
   * - ``alframework.samplers.reactive_sampler.reactive_sampling``
     - Uses NEB and/or dimer-style searches from reaction metadata, monitors
       ensemble uncertainty along reaction pathways, and returns uncertain
       structures for QM labeling.
     - Reaction-pathway active learning with reactant, transition-state, and
       product structures.

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
   See :ref:`ml-interface-extension-points` and
   :ref:`ml-interface-new-architecture-template` for the corresponding ML
   interface requirements.

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

Uncertainty-driven dynamics
   Uncertainty-driven dynamics (UDD) is a biased MLMD sampler that steers
   dynamics toward regions where the model committee has high ensemble energy
   uncertainty. The usual ``Escut`` and ``Fscut`` threshold logic still decides
   whether a sampled configuration is returned for QM labeling.

   Select UDD in ``master_config.json`` with a separate sampler task:

   .. code-block:: json

      {
        "sampler_task": "alframework.samplers.udd_sampling.simple_udd_sampling_task"
      }

   Set the bias strength in the sampler config:

   .. code-block:: json

      {
        "udd_bias_weight": 0.45,
        "MLMD_calculator_options": {
          "well_params": null
        }
      }

   ``udd_bias_weight`` controls the strength of the energy-uncertainty bias.
   Use ``simple_mlmd_sampling_task`` instead when no UDD bias is desired.

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

.. _samplers-multi-trajectory:

Multi-Trajectory Sampling
-------------------------

``mlmd_trajectories_task`` uses the same MLMD settings as
``simple_mlmd_sampling_task``, listed above, and takes several trajectories in one
task. Each trajectory stops on its own, and the task returns one result per input
once its last trajectory finishes.

Select it in ``master_config.json``, then add its settings to the sampler config:

.. code-block:: json

   {
     "sampler_task": "alframework.samplers.batched_mlmd.mlmd_trajectories_task"
   }

.. code-block:: json

   {
     "driver": {"strategy": "batched", "engine": "torchsim"},
     "batching": {"batch_size": 32, "partial_policy": "timeout", "max_wait_seconds": 5.0},
     "batched_calculator": "alframework.ml_interfaces.hippynn_interface.load_hippynn_batched_ensemble",
     "random_seed": 1234,
     "device": "cuda"
   }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``driver.strategy``
     - ``"single"`` (default) runs one trajectory per task. ``"batched"`` groups
       trajectories using the required ``batching`` block; see :doc:`batching`.
   * - ``driver.engine``
     - ``"ase"`` (default), ``"torchsim"``, or ``"alchemi"``. See below.
   * - ``batched_calculator``
     - Import string for a loader that evaluates many structures at once, used by
       the ``torchsim`` and ``alchemi`` engines in place of ``ase_calculator``. For
       HIPPYNN, use
       ``alframework.ml_interfaces.hippynn_interface.load_hippynn_batched_ensemble``.
       Without it, these engines evaluate the ``ase_calculator`` ensemble one
       structure at a time.
   * - ``batched_calculator_options``
     - Optional keyword arguments for that loader.
   * - ``random_seed``
     - Optional. Derives each trajectory's random numbers from this seed and its
       molecule ID, so results don't depend on how trajectories are grouped.
   * - ``acquisition``
     - Optional. How frames are chosen for QM; see :doc:`acquisition`.
   * - ``MLMD_calculator_options``
     - As above. Set ``udd_bias_weight`` here for uncertainty-driven dynamics.

Dynamics engines
   All three engines use the same checks, schedules, stopping rules, and
   acquisition. They differ in how trajectories are advanced:

   .. list-table::
      :header-rows: 1
      :widths: 16 34 24 26

      * - Engine
        - How it runs
        - Hardware
        - Install
      * - ``ase``
        - One ASE Langevin integrator per trajectory.
        - Whatever the calculator supports.
        - Included.
      * - ``torchsim``
        - All trajectories in one packed TorchSim state.
        - CPU, NVIDIA, and AMD.
        - ``torchsim`` extra, Python 3.12+.
      * - ``alchemi``
        - All trajectories in one packed state, using ALCHEMI kernels.
        - CPU and NVIDIA. For ROCm, use TorchSim.
        - ``alchemi`` extra, Python 3.11+. Versions 0.2.0 and 0.3.0 are supported.

   The ``torchsim`` and ``alchemi`` engines must be installed in the sampler worker
   environment; see :doc:`../getting_started/installation`. They don't support ASE
   constraints, and their ``MLMD_calculator_options`` are limited to
   ``well_params``, ``udd_bias_weight``, ``udd_bias_eps``, and ``debug_print``.

   ``examples/simple_water`` includes ``mlmd_config_torchsim.json`` and
   ``mlmd_config_alchemi.json``; see :doc:`../examples/simple_water`.

Other differences from ``simple_mlmd_sampling_task``:

* ``use_potential_specific_code`` isn't supported, so NeuroChem workflows need
  ``simple_mlmd_sampling_task``.
* ALF stops at startup if a single-structure sampler task is given ``driver``,
  ``batching``, or ``acquisition.selection``.
* For excited-state sampling, see :doc:`excited_states`.

GPU Selection
-------------

Legacy MLMD, UDD, and the multi-trajectory sampler select GPUs automatically
within the worker's scheduler/Parsl allocation. ALF doesn't change the CUDA, HIP,
or ROCr visibility variables it finds.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``device``
     - Behavior
   * - not set
     - ``simple_mlmd_sampling_task`` and ``simple_udd_sampling_task`` use a GPU
       automatically and fail if none is available.
       ``mlmd_trajectories_task`` uses a GPU automatically and falls back to CPU if
       none is available.
   * - ``"cuda"``
     - Assign a GPU automatically. Fails if none is available.
   * - ``"cuda:N"``
     - Use visible device ``N``. Fails if it isn't available. With
       ``CUDA_VISIBLE_DEVICES=2,5``, ``cuda:1`` is the second exposed GPU, not
       physical GPU 1.
   * - ``"cpu"``
     - Use the CPU, without looking for a GPU.

Automatic assignment spreads workers over the visible devices by
``PARSL_WORKER_RANK``, so a worker restricted to one GPU uses that GPU. Without a
visibility variable, ``gpus_per_node`` limits how many devices are used. A worker
with no rank uses the first device.

NeuroChem inference supports CUDA only, and its ``model_details['gpu']`` value is a
local device number passed to its constructor. Training pools and reactive samplers
handle devices separately.

What The Sampler Returns
------------------------

If uncertainty or another selection criterion triggers, the returned
``MoleculesObject`` keeps its atoms. ALF then passes that structure to the QM
task queue for labeling.

If sampling completes without selecting a configuration, the sampler sets the
atoms to ``None`` on the returned ``MoleculesObject``. ALF treats this as a
successful sampling attempt that does not need QM labeling.

With ``cohort_top_k`` selection, a trajectory's frames wait in a pool, and ALF sends
only the selected frames to QM once the cohort finishes; see :doc:`acquisition`.

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
   * - UDD workflows
     - ``alframework.samplers.udd_sampling.simple_udd_sampling_task``
     - Biased MLMD sampling toward high ensemble energy uncertainty.
   * - ``examples/simple_water``, TorchSim and ALCHEMI configs
     - ``alframework.samplers.batched_mlmd.mlmd_trajectories_task``
     - Batched sampling with ``mlmd_config_torchsim.json`` or
       ``mlmd_config_alchemi.json``.

Sampler API Links
-----------------

You can link from this guide directly to API pages:

* :doc:`Samplers package API <../api_documentation/alframework.samplers>`
* :doc:`MLMD sampling module <../api_documentation/alframework.samplers.mlmd_sampling>`
* :doc:`Multi-trajectory MLMD module <../api_documentation/alframework.samplers.batched_mlmd>`
* :doc:`TorchSim engine module <../api_documentation/alframework.samplers.torchsim_engine>`
* :doc:`ALCHEMI engine module <../api_documentation/alframework.samplers.alchemi_engine>`
* :doc:`Reactive sampler module <../api_documentation/alframework.samplers.reactive_sampler>`
* :doc:`ASE ensemble calculator module <../api_documentation/alframework.samplers.ASE_ensemble_constructor>`
