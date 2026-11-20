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

Each task's settings come from the sampler configuration. The fields most of them use are listed in
`Configuration Reference`_ at the end of this page.

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
     "batched_calculator": "alframework.ml_interfaces.hippynn.load_hippynn_batched_ensemble",
     "random_seed": 1234,
     "device": "cuda"
   }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``driver.strategy``
     - ``"single"`` runs one trajectory per task. ``"batched"`` groups
       trajectories using the required ``batching`` block; see :doc:`batching`. The
       default is ``"batched"`` when the config has a ``batching`` block, else
       ``"single"``.
   * - ``driver.engine``
     - Chooses how trajectories advance. Use ``"ase"`` (default), ``"torchsim"``,
       or ``"alchemi"``, or the import path of an engine class of your own, such as
       ``"my_package.engines.MyEngine"`` (see :mod:`alframework.samplers.engines.factory`
       for what the class provides). See below.
   * - ``batched_calculator``
     - Import string for a loader that evaluates many structures at once, used by
       the ``torchsim`` and ``alchemi`` engines in place of ``ase_calculator``. For
       HIPPYNN, use
       ``alframework.ml_interfaces.hippynn.load_hippynn_batched_ensemble``.
       The MACE and UMA examples use model loaders backed by
       ``alframework.ml_interfaces.torchsim_ensemble``.
       With only ``ase_calculator`` configured, these engines evaluate the ensemble
       one structure at a time.
   * - ``batched_calculator_options``
     - Optional keyword arguments for that loader.
   * - ``random_seed``
     - Optional. Derives each trajectory's random numbers from this seed and its
       molecule ID, so results don't depend on how trajectories are grouped.
   * - ``acquisition``
     - Optional. How frames are chosen for QM; see :doc:`acquisition`.
   * - ``MLMD_calculator_options``
     - As above. Set ``udd_bias_weight`` here for uncertainty-driven dynamics.
   * - ``friction_per_fs``
     - Optional. Langevin friction per femtosecond, for all three engines. It must
       be zero or more. Without it, the engines use a friction of ``0.02`` in ASE
       units.
   * - ``topology_check``, ``max_force_cutoff``, ``screen_audit_dir``
     - Optional. End trajectories that break or develop unphysical forces; see
       :ref:`samplers-runtime-screens`.

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

   The ``ase`` engine and ``simple_mlmd_sampling_task`` integrate with ASE's ``Langevin``; ``torchsim`` and
   ``alchemi`` with a BAOAB splitting (Leimkuhler and Matthews, 2013). All of them sample the same canonical
   distribution with each system's center of mass at rest, as ASE's ``FixCom`` constraint holds it: the 3N-3
   remaining degrees of freedom are at the thermostat's temperature, and the recorded
   ``kinetic_temperature_K`` counts them. The ASE integrators run with ``fixcm=False`` and ``FixCom`` attached
   only during dynamics; a structure returns with the constraints it came with, so labels computed through
   it keep their net force.

   ``examples/simple_water`` includes ``mlmd_config_torchsim.json`` and
   ``mlmd_config_alchemi.json``; see :doc:`../examples/simple_water`.

Other differences from ``simple_mlmd_sampling_task``:

* ``use_potential_specific_code`` isn't supported, so NeuroChem workflows need
  ``simple_mlmd_sampling_task``.
* ALF stops at startup if a single-structure sampler task is given ``driver``,
  ``batching``, or ``acquisition.selection``.
* For excited-state sampling, see :doc:`excited_states`.

.. _samplers-initial-velocities:

Initial Velocities And Temperature
----------------------------------

``velocity_initialization`` is a top-level sampler setting for
``simple_mlmd_sampling_task``, ``simple_udd_sampling_task``, and
``mlmd_trajectories_task``. UDD uses the same scalar MLMD sampler. The batched
sampler initializes velocities before creating its ASE, TorchSim or ALCHEMI
engine, subject to that engine's hardware and constraint requirements.
Reactive NEB and dimer searches are geometry optimizations and ignore this
setting. Custom sampler tasks need their own support for it.

The MLMD tasks preserve their input momenta by default. A positions-only
structure starts at rest. The thermostat exchanges energy with the system
during dynamics, so setting its temperature alone leaves the initial momenta
unchanged.

This is valid Langevin dynamics. Preserving input momenta is useful for
continuations, directed motion within a system (motion of the whole system is
removed, since its center of mass stays at rest), deliberately cold starts, gradual heating, and
existing workflows whose warm-up is long enough. Starting from rest can also
give gentle local perturbations of seed structures. An active-learning sample
need not be an equilibrium sample.

For a 600 K initial draw with a 400 K thermostat, use these sampler settings:

.. code-block:: json

   {
     "velocity_initialization": "maxwell_boltzmann_if_missing",
     "initial_velocity_temperature_K": 600.0,
     "srt_temp": [400.0, 400.0],
     "end_temp": [400.0, 400.0],
     "amp_temp": [0.0, 0.0],
     "random_seed": 42
   }

``velocity_initialization`` accepts two values:

* ``"preserve"`` (default): never change the input momenta. This consumes no
  random numbers and retains the existing dynamics.
* ``"maxwell_boltzmann_if_missing"``: when the input has no ``momenta`` array,
  draw Gaussian momentum components with variance :math:`m k_B T_0`, where
  :math:`T_0` is ``initial_velocity_temperature_K`` when supplied and the
  initial thermostat target otherwise. The override accepts a finite,
  nonnegative number in Kelvin and requires the thermal policy. It changes
  the initial draw only. The temperature schedule still controls the
  thermostat from the first step. Explicit momenta, including an array of
  zeros, are preserved.

The sampler calls ``alframework.tools.velocities`` to initialize momenta before
constructing an engine. The draw uses an independent molecule-ID random stream,
so ``random_seed`` reproduces it without changing the temperature-schedule or
Langevin noise streams. Without a seed, the generated seed is recorded as
``initial_velocity_seed``. The scalar sampler still draws Langevin noise from
the legacy global random stream.

The draw does not rescale to an exact kinetic temperature or remove center-of-mass
or angular momentum. This retains the Maxwell distribution and its finite-system
temperature fluctuations. ASE constraints are applied when momenta are assigned;
the packed engines' existing constraint restrictions still apply. For prescribed
translation or rotation, supply the desired momenta instead. Thermal velocities
alone do not equilibrate the starting positions.

Short sampling runs can end well before a cold start heats to its target. For a
free particle in a constant bath, the estimate is

.. math::

   \langle T_{\mathrm{kin}}(t)\rangle
      = T_{\mathrm{bath}}\left(1-\exp(-2\gamma t)\right).

At ``friction_per_fs=0.01``, a cold start in a 400 K bath reaches about 42 K
after 5.5 fs by this estimate. Molecular forces also affect the transient.
For degrees of freedom that the thermostat equilibrates, cold and thermal starts
approach the same stationary distribution, but explore differently before then. Temperature
cycling is useful only if trajectories last long enough for the intended
heating and cooling to act.

Trajectory metadata and the coordinator's ``trajectory_summary_path`` record
``initial_kinetic_temperature_K``, ``kinetic_temperature_K`` (at the returned
frame), and ``thermostat_temperature_K`` (the last applied target). Cohort
candidates record their own instantaneous temperature. Metadata also records
``velocity_initialization``, ``velocities_initialized``, and
``initial_velocity_temperature_K`` (the draw target, or ``null`` when no draw
was made). The temperature
scalars survive compact returned metadata. Undefined kinetic temperatures are
``null``. Older trajectory-summary rows have no such measurements, and HDF5
label shards do not automatically store momenta or these temperature fields.

.. _samplers-runtime-screens:

Runtime Screens
---------------

``mlmd_trajectories_task`` can end a trajectory that has left the chemistry you
meant to sample, so that its broken geometries never reach QM. Both screens are off
by default, work with all three engines, and run when a trajectory starts and at each
uncertainty check, every ``Ncheck`` steps.

.. code-block:: json

   {
     "topology_check": {
       "enabled": true,
       "reference_conformer_path": "seeds/reference.xyz",
       "reference_charge": 0,
       "bond_min_scale": 0.85,
       "bond_max_scale": 1.5,
       "connectivity_scale": 1.25
     },
     "max_force_cutoff": 16.0,
     "screen_audit_dir": "sampling/screen_audit"
   }

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Meaning
   * - ``topology_check.enabled``
     - Turns the bond check on. The default is ``false``.
   * - ``topology_check.reference_conformer_path``
     - Required. XYZ, MOL, or SDF file that defines the expected bonds. A relative
       path starts from the master directory.
   * - ``topology_check.reference_format``
     - ``"xyz"``, ``"mol"``, ``"mdl"``, or ``"sdf"``. The default, ``"auto"``, uses
       the file extension.
   * - ``topology_check.reference_charge``
     - Molecular charge, used to work out bonds from an XYZ file. The default is
       ``0``.
   * - ``topology_check.bond_min_scale``, ``topology_check.bond_max_scale``
     - Required. Limits on each bond's length as a fraction of its length in the
       reference. They must satisfy ``0 < bond_min_scale <= 1 <= bond_max_scale``.
   * - ``topology_check.connectivity_scale``
     - Required. Two atoms that aren't bonded in the reference count as a new bond
       when they come within this multiple of the sum of their covalent radii.
   * - ``max_force_cutoff``
     - Largest allowed force on any atom, in eV/Angstrom, using the ensemble-mean
       forces from unbiased predictions. In excited-state mode, it applies to every
       state, not only the selected one. Forces that aren't finite always fail.
   * - ``screen_audit_dir``
     - Where each task writes one JSON file with the initial, last valid, and rejected
       frames of its trajectories. It doesn't record whole trajectories. Without it,
       ALF stores the same record in each molecule's metadata.

When a frame fails, ALF ends that trajectory, records why in the molecule's metadata
as ``screen_reject_reason``, and doesn't label the frame. Earlier frames from the
same trajectory stay eligible if ``"trajectory_policy": "continue"`` kept them in a
cohort; see :doc:`acquisition`.

The bond check has these requirements:

* Structures must be nonperiodic.
* Each structure's atoms must carry ``topology_atom_ids``, which map them to the
  reference. ``rdkit_molecule_builder_task`` provides them, and
  ``external_geometry_builder_task`` keeps them when its source files have them. Without
  them, every trajectory fails with ``topology_identity``. See :doc:`builders`.
* Sampler workers need the ``topology`` extra, because RDKit reads the reference; see
  :doc:`../getting_started/installation`.

.. _samplers-sampling-windows:

Sampling Windows
----------------

A run can start on ordinary sampling and move to UDD or gap-seeking later, once it has
enough labels. It stays one run: the same dataset, with the ensemble continuing to
retrain. ``sampling_windows`` goes in the sampler config, and requires
``mlmd_trajectories_task``.

.. code-block:: json

   {
     "sampling_windows": [
       {"until_stored_labels": 40000},
       {
         "until_stored_labels": 50000,
         "MLMD_calculator_options": {"udd_bias_weight": 0.1}
       },
       {
         "gap_seeking": {
           "enabled": true,
           "trigger_gap_threshold_eV": 0.1,
           "sigma": 3.5,
           "alpha_eV": 0.05
         }
       }
     ]
   }

Here, sampling is unchanged up to 40,000 stored labels, uses UDD up to 50,000, and then
uses gap-seeking.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Meaning
   * - ``until_stored_labels``
     - The window applies while the stored label count is below this value. Limits must
       increase. The last window must not set one. The count includes bootstrap labels,
       and is kept in ``status_path`` as ``lifetime_stored_labels``.
   * - ``MLMD_calculator_options``, ``gap_seeking``
     - Optional. These replace the same settings in the rest of the sampler config for
       this window. Windows don't build on each other, so a setting from one window
       doesn't carry into the next.

Things to know:

* ALF checks every window at startup, so a window that combines gap-seeking with UDD,
  for example, stops the run before it starts. The list can't change during a run.
* A switch happens on the first loop after stored labels reach the limit, so a run can
  go past the limit by up to one batch. A batch holds more than ``save_h5_threshold``
  labels, and more still when labels converged while a training ran.
* Tasks already running finish under the old window. A batch uses the window current when
  it's submitted, and each molecule's metadata records its ``sampling_window``.
* A cohort that spans a switch pools frames from both windows. Selection still works,
  because scores use unbiased predictions.
* Gap-seeking also needs ``"model_mode": "excited_state"``; see :doc:`excited_states`.

GPU Selection
-------------

Legacy MLMD, UDD, reactive sampling, and the multi-trajectory sampler select GPUs
automatically within the worker's scheduler/Parsl allocation. ALF doesn't change the CUDA, HIP,
or ROCr visibility variables it finds.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``device``
     - Behavior
   * - not set
     - ``simple_mlmd_sampling_task``, ``simple_udd_sampling_task``, and
       ``reactive_sampling`` use a GPU automatically and fail if none is available.
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
local device number passed to its constructor. HIPPYNN training with
``"device_string": "from_multiprocessing"`` puts each ensemble member on its own
visible device, also without changing those variables. To train across several nodes,
see :ref:`ml-multinode-training`.

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
compatible metadata path is configured; the multi-trajectory sampler puts them in
``sampling/task-<first moleculeid>/`` by default (``metadata_files``, ``meta_dir_layout``).

With the multi-trajectory sampler:

* ``Es``, ``Fs``, and ``Fsmax`` are from a trajectory's last valid check. If a screen
  rejects a trajectory, they come from the check before, because the rejected frame is
  physically suspect. A trajectory rejected on its first check has none.
* ``peak_Es``, ``peak_Fs``, and ``peak_Fsmax`` are the highest values across its valid
  checks.
* ``realtime_simulation`` is the time from the start of the task until that trajectory
  finished.

.. _samplers-progress-plots:

Progress Plots
--------------

Progress plots show how sampling changes as the model improves, with one box per model.
They work with all samplers, in batches, and with several electronic states. To turn them
on, add this to the master config:

.. code-block:: json

   {
     "plotting_utility": "alframework.tools.plotting.analysis_plot",
     "plot_dir": "status_plots/",
     "trajectory_summary_path": "status_plots/trajectories.jsonl",
     "update_plots_every": 30
   }

With ``trajectory_summary_path``, the coordinator appends one small row per trajectory to
that file, from metadata it already receives, and plots read only that file. Each row has the
trajectory's outcome, state, times, and its last and peak ``Es``, ``Fs``, and ``Fsmax``, plus
``gap_std`` and ``peak_gap_std`` when ``gap_uncertainty`` is set, and with adjacent pairs,
``gap_lower_state`` for the pair whose spread was largest on the last frame. Those peaks are what to
compare with the cutoffs when tuning them. Prefer this to
``meta_dir``, particularly in large runs. ``meta_dir`` writes one file per trajectory into one
directory per task by default (see ``metadata_files`` and ``meta_dir_layout``), and plots then
reread all of them on each update. Without ``trajectory_summary_path``, plots read ``meta_dir`` if
it's set.

ALF writes these to ``plot_dir``, as ``analysis-plot-<name>.png``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Name
     - Shows
   * - ``outcomes``
     - The share of each model's trajectories that were completed (ran to the end), eligible (gave QM
       a candidate), rejected, or failed. It's only available from samplers that record an outcome,
       which includes the multi-trajectory sampler. ``task_failed`` means a whole batched task failed.
   * - ``simtime``, ``log_simtime``
     - Simulated time per trajectory. This should grow as the model improves.
   * - ``realtime``, ``log_realtime``
     - Wall time per trajectory.
   * - ``Es``, ``Fs``, ``Fsmax``
     - Energy and force uncertainty, using each trajectory's peak when it's recorded. The title
       says which. With the sampler's cutoffs, a line marks each trigger threshold (``Escut``,
       ``Fscut``, and ``3 * Fscut`` for ``Fsmax``), and the title says what share of the latest
       model's trajectories passed it.
   * - ``gap_std``
     - Excited states with ``gap_uncertainty``: the spread of the gap, with ``gap_std_cut``.
   * - ``<name>-state<N>``
     - The same box plots for each selected state, when a run has more than one, however many.
   * - ``sampling-cost``
     - Simulated time spent per QM candidate, and the share of simulated time in trajectories
       that gave none. It shows what a cutoff costs in sampling, mostly with
       ``"trajectory_policy": "stop"``.
   * - ``rejections``
     - Each model's rejected trajectories by the reason recorded for them.
   * - ``cutoff-tuning``
     - With the cutoffs: for the latest four models, the share of trajectories whose peak would pass
       the thresholds if every cutoff were multiplied by a factor from 0.25 to 4, for the signals
       that trigger QM together and one by one. Trajectories stopped at their first trigger have
       peaks cut short, so above a factor of 1 the shares are lower bounds.
   * - ``trigger-time``
     - With the cutoffs: for the latest model, the simulated time when each trajectory stopped
       against its peak uncertainty over the threshold. Early triggers mean the model fails fast.

Plots are by ALF iteration (the model in use). Box plots show the middle half of each model's trajectories
with the median, whiskers from the 5th to
the 95th percentile, and a tick at the largest value, so a long tail doesn't hide the rest. The
cutoffs, ``Escut``, ``Fscut`` and ``gap_std_cut``, are read from the sampler config. Which signals
trigger QM follows the sampler's ``acquisition.eligibility``: energy and both force signals by default,
and the gap only when ``"gap"`` is listed (see :doc:`acquisition`). With a cutoff of ``0`` or none,
its line and the cutoff-tuning and trigger-time plots are left out.

A dashed line marks the first model in each sampling window. A plot with no data is skipped, and
ALF prints which ones it skipped.

What ``simtime`` means depends on ``trajectory_policy``. With ``stop``, trajectories stop at the
first frame that triggers, so it's the time until uncertainty got too high. With ``continue``, they
keep going, so it's the time until a screen rejected them or ``maxt``, and it can't exceed ``maxt``.

At 1 million trajectories, the summary is about 350 MB, and one plot update takes several seconds
and about 1.3 GB of memory on the coordinator's node. Each update rereads the whole file.

Each update runs in a new Python process (``python -m alframework.tools.progress_plots``), not in a
fork of the coordinator, so its memory doesn't grow with the labels the coordinator holds. That
process imports ``plotting_utility`` by its dotted name, using the coordinator's import path, and
receives the plot settings as JSON. A custom plotting utility must therefore be a function in an
importable module, not one defined in the script that starts ALF. If the previous update is still
running when the next one is due, ALF skips the new one and prints that it did.

Configuration Reference
-----------------------

The fields that most sampler configurations set, by topic.

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
      * - ``velocity_initialization``
        - Optional MLMD starting-momenta policy. Defaults to ``"preserve"``.
          See :ref:`samplers-initial-velocities` for the thermal option and
          supported tasks.
      * - ``initial_velocity_temperature_K``
        - Optional Kelvin target for the starting-momenta draw. Requires
          ``"maxwell_boltzmann_if_missing"``. Defaults to the initial thermostat
          target and leaves the thermostat schedule independent.

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
      * - ``metadata_files``
        - Multi-trajectory sampler only. What a task writes to ``meta_dir``: ``"per_trajectory"``
          (default) writes ``metadata-<moleculeid>.p`` per trajectory; ``"per_task"`` writes one
          ``metadata-task-<first moleculeid>.p`` per task, a list of the same dicts, each with a
          ``moleculeid`` key; ``"none"`` writes no metadata file. Either way the metadata returns with
          each structure, and ``trajectory_summary_path`` still records it.
      * - ``returned_metadata``
        - Multi-trajectory sampler only. What metadata each returned structure carries: ``"full"``
          (default) or ``"compact"``, which leaves out the per-trajectory arrays and lists the metadata
          file keeps (``positions``, ``chemical_symbols``, ``temps``, ``denss``, ``system_comp``), about two
          thirds of a trajectory's metadata in the coordinator's memory. Compact needs ``meta_dir`` and
          ``metadata_files`` ``"per_trajectory"`` or ``"per_task"``; the coordinator refuses it otherwise,
          and for other sampler tasks. See :ref:`running-at-scale-coordinator-memory`.
      * - ``returned_momenta``
        - Multi-trajectory sampler only. ``false`` returns each structure's final positions and cell without
          its MD momenta, which labeling and storage don't read; the default ``true`` keeps them. For
          201-atom structures they are about 40% of what each returned structure costs the coordinator and
          Parsl's interchange. The coordinator refuses it for other sampler tasks.
      * - ``meta_dir_layout``
        - Multi-trajectory sampler only. Where per-trajectory files (metadata and trajectories) go:
          ``"per_task"`` (default) gives each task its own subdirectory,
          ``meta_dir/task-<first moleculeid>/``; ``"flat"`` writes them into ``meta_dir`` itself, for
          readers that expect one directory. Progress plots read every layout. Subdirectories don't
          reduce the number of files: at scale, write fewer of them with ``metadata_files``, as
          described in :ref:`running-at-scale-small-files`.
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

Sampler task paths in ``sampler_task`` stay the same. Python code that imports
engines or shared helpers directly should use the paths below. Previous module
paths still import and emit a warning with the new path.

* :doc:`Samplers package API <../api_documentation/alframework.samplers>`
* :doc:`MLMD sampling module <../api_documentation/alframework.samplers.mlmd_sampling>`
* :doc:`Multi-trajectory MLMD module <../api_documentation/alframework.samplers.batched_mlmd>`
* :doc:`ASE engine module <../api_documentation/alframework.samplers.engines.ase>`
* :doc:`TorchSim engine module <../api_documentation/alframework.samplers.engines.torchsim>`
* :doc:`ALCHEMI engine module <../api_documentation/alframework.samplers.engines.alchemi>`
* :doc:`Shared packed engine module <../api_documentation/alframework.samplers.engines.packed>`
* :doc:`Velocity initialization module <../api_documentation/alframework.tools.velocities>`
* :doc:`Geometry module <../api_documentation/alframework.tools.geometry>`
* :doc:`Reactive sampler module <../api_documentation/alframework.samplers.reactive_sampler>`
* :doc:`ASE ensemble calculator module <../api_documentation/alframework.samplers.ASE_ensemble_constructor>`
