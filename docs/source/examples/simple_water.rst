Simple Water Example
====================

The simple water example is a compact ALF workflow for a small molecular
system. It is intended as the first example to copy and adapt when setting up a
new active-learning run. The example connects all major ALF stages:

1. Build initial water-containing structures.
2. Sample those structures with ML-driven molecular dynamics.
3. Send selected high-uncertainty configurations to ORCA for QM labeling.
4. Store labeled data in HDF5 files.
5. Retrain a HIPPYNN ensemble from the accumulated data.

The files live in ``examples/simple_water``. The JSON files are deliberately
small, but each one controls a different part of the workflow.

Before You Run
--------------

The example is not a machine-independent benchmark. Before running it on a new
system, check the following items.

* ``orca_config.json`` contains an ORCA executable path. Replace
  ``QM_run_command`` with the ORCA command available on your machine.
* ``master_config.json`` points to a Parsl configuration named
  ``custom_modules.config_1node``. Edit ``examples/simple_water/custom_modules.py``
  or point ``parsl_configuration`` to a different resource config for your
  cluster.
* HIPPYNN and its ML dependencies must be installed if you run ML training or
  sampling. The builder and QM test modes can be used separately while bringing
  up the environment.
* Paths such as ``h5store/``, ``models/``, ``sampling/``, and
  ``water-test-running/`` are run outputs. Use a clean working directory or
  archive old output before starting a new production run.

See :doc:`../user_guide/parsl` for details on how Parsl resource configuration
maps ALF tasks onto local or HPC resources.

Workflow Stages
---------------

Builder stage
   ``simple_condensed_phase_builder_task`` reads ``builder_config.json`` and
   creates initial water structures from the fragment library. In this example,
   the only fragment is ``fragment_library/water.xyz``.

Sampler stage
   ``simple_mlmd_sampling_task`` reads ``mlmd_config.json`` and runs
   uncertainty-driven MLMD using a HIPPYNN ensemble. Configurations are selected
   for QM if energy or force uncertainty exceeds the configured cutoffs.

QM stage
   ``orca_calculator_task`` reads ``orca_config.json``, writes ORCA input files,
   runs ORCA, parses requested properties, and returns a labeled
   ``MoleculesObject``.

ML stage
   ``train_HIPPYNN_ensemble_task`` reads ``hippynn_config.json`` and trains a
   HIPPYNN model ensemble from the HDF5 data written by ALF.

Recommended Bring-Up Commands
-----------------------------

Run the stages one at a time before starting a full active-learning loop. From
the ``examples/simple_water`` directory:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json --test_ml
   python -m alframework --master master_config.json --test_sampler

The test modes use ``parsl_debug_configuration`` when it is provided in the
master config. This lets you keep a smaller, faster Parsl configuration for
environment checks.

After the individual stages work, start the active-learning loop:

.. code-block:: bash

   python -m alframework --master master_config.json

ALF writes ``status.txt`` and then continues running the active-learning loop,
periodically printing queue status for builder, sampler, QM, and ML tasks.

Expected Outputs
----------------

During staged checks and full runs, expect these files and directories:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Purpose
   * - ``status.txt``
     - Restart and progress state, including current model/data ids and failed
       task counters.
   * - ``qm_test.h5``
     - HDF5 file written by ``--test_qm`` so you can compare parsed ORCA output
       with stored training data.
   * - ``h5store/data-0000.h5``
     - First stored training-data batch from converged QM results.
   * - ``models/model-0000``
     - First trained HIPPYNN ensemble directory.
   * - ``sampling/metadata-*.p``
     - Per-sampling-task metadata, including uncertainty, temperature, density,
       and final structure information.
   * - ``water-test-running/``
     - QM scratch directory containing per-molecule ORCA run directories.
   * - ``status_plots/``
     - Optional plots created by the configured plotting utility.

Master Configuration
--------------------

``master_config.json`` connects the stages and controls the active-learning
loop.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``master_directory``
     - Base directory for relative paths. ``"pwd"`` means the directory
       containing the master config.
   * - ``*_config_path``
     - Paths to the builder, sampler, QM, and ML JSON files.
   * - ``builder_task``, ``sampler_task``, ``QM_task``, ``ML_task``
     - Import strings for the Python task functions ALF will submit through
       Parsl.
   * - ``properties_list``
     - Maps ALF result names to HDF5 dataset names, property scope, and unit
       conversion factor.
   * - ``h5_path``
     - Output pattern for stored training data, such as
       ``h5store/data-0000.h5``.
   * - ``model_path``
     - Output pattern for trained models, such as ``models/model-0000``.
   * - ``target_queued_QM``, ``minimum_QM``, ``save_h5_threshold``
     - Queue thresholds controlling when ALF launches QM tasks and stores new
       labeled data.
   * - ``parallel_samplers``
     - Target number of builder/sampler tasks ALF tries to keep active.
   * - ``bootstrap_set``
     - Number of initial QM labels to collect before the first model training.
   * - ``parsl_configuration``
     - Import string for the production Parsl config.
   * - ``parsl_debug_configuration``
     - Import string for the smaller config used by stage test modes.
   * - ``QM_scratch_dir``
     - Directory for QM input, output, and scratch subdirectories.

Builder Configuration
---------------------

``builder_config.json`` controls initial structure construction.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``molecule_library_path``
     - Directory containing fragment files, relative to the example directory.
   * - ``solute_molecule_options``
     - Fragment combinations that may be placed as solutes. The water example
       uses one water molecule.
   * - ``solvent_molecules``
     - Solvent fragment names and sampling weights.
   * - ``cell_range``
     - Minimum and maximum box lengths for each cell direction.
   * - ``Rrange``
     - Density or packing range used by the condensed-phase builder.
   * - ``min_dist``
     - Minimum allowed distance between generated structures.
   * - ``max_patience``
     - Number of placement attempts before the builder gives up.
   * - ``center_first_molecule``
     - Whether to center the first molecule, useful for solute-centered systems.
   * - ``shake``
     - Random coordinate perturbation applied to generated structures.

Sampler Configuration
---------------------

``mlmd_config.json`` controls uncertainty-driven MD sampling.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``dt``, ``maxt``
     - MD timestep in femtoseconds and maximum simulation time in picoseconds.
   * - ``Escut``, ``Fscut``
     - Energy and force uncertainty thresholds for selecting configurations for
       QM labeling.
   * - ``Ncheck``
     - Number of MD steps between uncertainty checks.
   * - ``srt_temp``, ``end_temp``
     - Ranges for starting and ending temperatures.
   * - ``amp_temp``, ``per_temp``
     - Ranges for temperature fluctuation amplitude and period.
   * - ``end_dens``, ``amp_dens``, ``per_dens``
     - Density target and fluctuation controls. ``null`` disables density
       changes in this example.
   * - ``meta_path``
     - Directory for sampler metadata files.
   * - ``ase_calculator``
     - Import string for the ML ensemble calculator loader.
   * - ``MLMD_calculator_options``
     - Extra calculator options, such as the soft wall potential.
   * - ``trajectory_frequency``, ``trajectory_interval``
     - Controls optional trajectory writing during sampling.

ORCA Configuration
------------------

``orca_config.json`` controls QM labeling through ORCA.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``QM_run_command``
     - ORCA executable or launch command. Replace this path for your system.
   * - ``ncpu``
     - CPU count passed to the ORCA interface when applicable.
   * - ``orca_env_file``
     - Optional shell environment file sourced before ORCA execution.
   * - ``orcasimpleinput``
     - ORCA method, basis, and job keywords.
   * - ``orcablocks``
     - Additional ORCA input blocks, such as memory, SCF, and property blocks.
   * - ``Ediff``, ``Fdiff``
     - Thresholds used by double-calculation checks when using the double ORCA
       task.

HIPPYNN Configuration
---------------------

``hippynn_config.json`` controls model architecture, training, and data
filtering.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``n_models``
     - Number of models in the ensemble.
   * - ``energy_key``, ``force_key``, ``coordinates_key``, ``species_key``
     - HDF5 dataset names used for training.
   * - ``network_params``
     - HIPPYNN architecture settings, including species, feature counts,
       distance cutoffs, and layer counts.
   * - ``*_loss_weight``
     - Relative weights for energy, force, and regularization losses.
   * - ``valid_size``, ``test_size``
     - Fractions of data reserved for validation and test splits.
   * - ``learning_rate``
     - Initial optimizer learning rate.
   * - ``scheduler_options``
     - Learning-rate scheduler behavior, such as patience and decay factor.
   * - ``controller_options``
     - Training loop options, including batch size, maximum epochs, and early
       termination patience.
   * - ``device_string``
     - Device selection behavior for training tasks.
   * - ``remove_high_*``
     - Optional data filtering thresholds before training.

Adapting The Example
--------------------

For a new molecular system, start by replacing the fragment library and builder
configuration. Then update the QM method and executable path. Once the builder
and QM tests work, tune the sampler thresholds and ML architecture for the
target chemistry. Finally, update the Parsl configuration so the QM, ML, and
sampler tasks request resources that match the software and hardware used on
your cluster.
