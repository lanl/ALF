ML Interfaces
=============

ML interfaces connect ALF's active-learning loop to machine-learned
interatomic potential (MLIP) packages. They are responsible for two related
jobs: training a model ensemble from ALF's HDF5 data and loading that ensemble
as an ASE calculator for sampling.


Supported MLIP Packages
-----------------------

ALF's current examples primarily use HIPPYNN, and the rest of this guide uses
HIPPYNN as the main worked configuration path. NeuroChem/ANI is also supported
through its own training task and sampler loading path.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Package
     - Associated training task
   * - HIPPYNN
     - ``alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_task``
   * - HIPPYNN, several electronic states
     - ``alframework.ml_interfaces.hippynn_interface.train_excited_state_HIPPYNN_ensemble_task``.
       See :doc:`excited_states`.
   * - HIPPYNN, across several nodes
     - ``train_HIPPYNN_ensemble_multinode_task`` and
       ``train_excited_state_HIPPYNN_ensemble_multinode_task``, in the same module. See
       :ref:`ml-multinode-training`.
   * - NeuroChem/ANI
     - ``alframework.ml_interfaces.neurochem_interface.train_ANI_model_task``
   * - Fine-tuning a pretrained HIPPYNN or MACE model
     - ``alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_task``, or
       ``fine_tune_ensemble_multinode_task`` across nodes. See :doc:`fine_tuning`.

.. note::

   NeuroChem is a legacy path. ALF's automated tests run it against a stand-in for
   its backend, and it wasn't part of the validation runs for batched sampling.
   ``mlmd_trajectories_task`` doesn't support it.

Training tasks run on ``alf_ML_executor``. The number of GPUs available to the
task is passed from the master configuration through ``gpus_per_node``. The
actual cluster resources are controlled by the Parsl configuration; see
:doc:`parsl`.

.. _ml-interface-extension-points:

How ALF Uses ML Interfaces
--------------------------

ALF uses ML code through two configuration hooks.

Training task
   The ``ML_task`` field in ``master_config.json`` is an import string for a
   Python function that trains or updates a model ensemble. ALF submits this
   task through Parsl on the ``alf_ML_executor`` executor. The multi-node tasks
   instead run one app per ensemble member on it; see :ref:`ml-multinode-training`.

Sampler calculator
   The ``ase_calculator`` field in sampler configuration files is an import
   string for a function or class that loads trained models for ML-driven
   sampling. For the default MLMD sampler, the loader usually returns a list of
   ASE calculators that ALF wraps with
   ``alframework.samplers.ASE_ensemble_constructor.MLMD_calculator``.

These two hooks are intentionally separate. A new ML architecture can use ALF's
standard training loop, standard MLMD sampler, or both, as long as it follows
the expected function signatures and ASE calculator behavior.

.. _ml-interface-common-configuration:

Common Configuration Fields
---------------------------

Master configuration
   These fields connect ALF's active-learning loop to the ML task.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``ML_task``
        - Import string for the model-training task.
      * - ``ML_config_path``
        - JSON file containing backend-specific training options.
      * - ``model_path``
        - Output pattern for model directories, commonly
          ``models/model-{:04d}``.
      * - ``h5_path``
        - Output pattern for ALF HDF5 training data, commonly
          ``h5store/data-{:04d}.h5``.
      * - ``gpus_per_node``
        - Number of GPUs on each ML or sampler node. The two share this value, and a
          GPU mask on the worker takes precedence over it.
      * - ``save_h5_threshold``
        - Number of newly labeled configurations needed before ALF writes a
          new HDF5 batch and starts another ML training cycle.

Data and property expectations
   The ML task reads ALF's HDF5 data. Dataset names in the ML configuration
   must match the names written by ``properties_list`` and ALF's HDF5 storage.
   Common keys are ``energy``, ``forces``, ``species``, and ``coordinates``.
   Periodic, electrostatic, or multipole training may also require keys such as
   ``cell``, ``charges``, ``dipole``, or ``quadrupole`` depending on the ML
   backend. See :doc:`units` for how QM results are converted before they reach
   the HDF5 training data.

Waiting for shards
   Before training, the HIPPYNN tasks wait until every shard ALF has stored
   appears in ``h5_dir`` on the ML node. Those are the ``h5_path`` names
   numbered below ``current_h5_id``. On shared filesystems, another node can
   briefly see an out-of-date directory listing, and training would otherwise
   leave out the newest shard without an error. After 120 s, training stops with
   an error naming the missing shard. Other files in ``h5_dir``, such as seed
   shards, are still used. So deleting a stored shard now stops training,
   instead of training without it.

HIPPYNN Interface
-----------------

Use the HIPPYNN interface when the workflow should train HIPPYNN model
ensembles during the ALF loop. A typical master configuration uses:

.. code-block:: json

   {
     "ML_task": "alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_task",
     "ML_config_path": "hippynn_config.json"
   }

The sampler usually loads the trained ensemble with:

.. code-block:: json

   {
     "ase_calculator": "alframework.ml_interfaces.hippynn_interface.HIPNN_ASE_load_ensemble",
     "MLMD_calculator_options": {"well_params": null}
   }

Important HIPPYNN configuration fields include:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``n_models``
     - Number of HIPPYNN models in the ensemble.
   * - ``energy_key``, ``force_key``, ``coordinates_key``, ``species_key``
     - HDF5 dataset names used for training.
   * - ``network_params``
     - HIPPYNN architecture settings, including ``possible_species``, feature
       counts, distance cutoffs, layer counts, sensitivity type, and residual
       options.
   * - ``cell_key``
     - HDF5 cell key for periodic training. Use ``null`` for nonperiodic data.
   * - ``*_loss_weight``
     - Relative weights for energy, force, regularization, and optional
       physics terms.
   * - ``valid_size``, ``test_size``
     - Fractions of the data reserved for validation and test splits.
   * - ``learning_rate``
     - Initial optimizer learning rate.
   * - ``scheduler_options``
     - Learning-rate scheduler settings such as maximum batch size, patience,
       and decay factor.
   * - ``controller_options``
     - Training loop settings such as batch size, evaluation batch size,
       maximum epochs, and termination patience.
   * - ``device_string``
     - Device selection behavior. Examples use ``from_multiprocessing`` so
       ensemble members can be distributed across visible GPUs, within any mask the
       scheduler set. Any other value overwrites ``CUDA_VISIBLE_DEVICES`` for that
       training process. Multi-node training accepts only ``from_multiprocessing``,
       ``cuda``, or ``cpu``; see :ref:`ml-multinode-training`.
   * - ``remove_high_energy_*``, ``remove_high_forces_*``
     - Optional filters for removing high-energy or high-force structures from
       the training set.

``HIPNN_ASE_load_ensemble(model_dir, device="cuda:0")`` loads each
``model-*`` subdirectory under the current model directory and returns a list
of ASE calculators. The generic MLMD sampler wraps that list with
``MLMD_calculator`` to compute mean predictions and ensemble uncertainties.

For the ``torchsim`` and ``alchemi`` dynamics engines, set ``batched_calculator`` in
the sampler config instead:

.. code-block:: json

   {
     "batched_calculator": "alframework.ml_interfaces.hippynn_interface.load_hippynn_batched_ensemble"
   }

``load_hippynn_batched_ensemble(model_directory, device="cuda:0", energy_key="energy",
force_key="forces")`` evaluates each ensemble member on a whole batch of structures
at once. It needs at least two ``model-*`` members, and a checkpoint trained with a
cell requires fully periodic systems. ``batched_calculator_options`` can set
``energy_key`` and ``force_key``. See :ref:`samplers-multi-trajectory`.

.. _ml-multinode-training:

Training Across Several Nodes
-----------------------------

The standard HIPPYNN tasks train on one node, so an ensemble of 8 on a 4-GPU node runs 4
members at a time. The multi-node tasks can spread an ensemble over any number of nodes,
with each member still training on one GPU. To use one, set ``ML_task``:

.. code-block:: json

   {
     "ML_task": "alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_multinode_task",
     "gpus_per_node": 4,
     "member_retries": 1
   }

For several electronic states, use ``train_excited_state_HIPPYNN_ensemble_multinode_task``.
Set the ensemble size with ``n_models`` in the ML configuration, and the nodes in the Parsl
configuration; see :ref:`parsl-multinode-training`. Models are written and loaded the same way
as with the standard tasks.

ALF waits for the whole ensemble
   ALF treats the ensemble as one training task. It promotes the new model only after every
   member has succeeded, and it doesn't store new data or start another training until then.
   Sampling and labeling continue on the current model in the meantime, and finished labels
   stay in the coordinator's memory until the next store. Progress appears in the coordinator's
   output, for example ``Training 0003: 35/40 members finished, 1 retrying, 0 failed``.

Failed members are retried
   A member that raises an error, loses its worker or node, or doesn't finish training is run
   again, up to ``member_retries`` more times. The default is ``1``, and ``0`` turns retries off.
   Members that succeeded aren't rerun. ALF renames a failed member's directory, for example to
   ``failed-model-03-attempt-1``, which ensemble loaders ignore. Nothing is deleted. If a member
   runs out of retries, the others still finish, but ALF doesn't promote that ensemble, and the
   next training starts from a new training ID. There's no time limit, so training waits for as
   long as Slurm keeps a block queued. Retries mainly help with lost nodes: a retried member keeps
   its seed, so a numerical failure is likely to repeat.

Each member uses one GPU
   Each member's GPU comes from its Parsl worker, within any mask set by the scheduler or
   ``available_accelerators``. So ``device_string`` must be ``from_multiprocessing``, ``cuda``, or
   ``cpu``. Without ``available_accelerators``, ALF stops with an error if a node has more workers than
   ``gpus_per_node``, since members would share a GPU.

Each attempt writes ``placement-model-03-attempt-1.json`` in the ensemble directory, with its
host, worker, device, and start and end times. The record stays even if the node is lost.

More members cost more than training time:

- More nodes make training faster, but not sampling. Every sampler task loads and evaluates every
  member, so MLMD with 40 members takes about 5 times as long per step as with 8.
- Every member reads all of the training data, so the load on shared storage grows with both the
  ensemble and the dataset.
- All members stay on the GPU during sampling, so check GPU memory before using a large ensemble.
- Uncertainty cutoffs may need adjusting. ALF uses the population standard deviation, so the same
  underlying spread reads about 14% higher with 40 members than with 4, and about 6% higher than with
  8. Cutoffs chosen for a smaller ensemble trigger earlier with a larger one.

.. _ml-interface-new-architecture-template:

Adding New ML Architectures
-----------------------------------------

New ML backends can be added as ordinary Python modules that ALF imports from
configuration strings. The module can live in ``alframework.ml_interfaces`` or
in a project-local module on ``PYTHONPATH``.

Training task signature
   Define a Parsl task with the same call shape used by ALF:

   .. code-block:: python

      from parsl import python_app


      @python_app(executors=["alf_ML_executor"])
      def train_MYMODEL_task(
          ML_config,
          h5_dir,
          model_path,
          current_training_id,
          gpus_per_node,
          remove_existing=False,
          h5_test_dir=None,
      ):
          output_dir = model_path.format(current_training_id)

          # 1. Read ALF HDF5 data from h5_dir.
          # 2. Train one or more models using options in ML_config.
          # 3. Write artifacts under output_dir.
          # 4. Decide whether each model completed successfully.

          completed = True
          return completed, current_training_id

   ``completed`` may be a single boolean or a list of booleans for ensemble
   members. ``current_training_id`` should be returned unchanged so ALF can
   associate the result with the expected model directory.

Sampler loader signature
   Provide a loader that the sampler can import through ``ase_calculator``:

   .. code-block:: python

      def MYMODEL_ASE_load_ensemble(model_dir, device="cuda:0"):
          calculators = []
          for member_dir in sorted_model_member_directories(model_dir):
              calculators.append(load_one_model_as_ase_calculator(member_dir, device))
          return calculators

For the generic MLMD path, each calculator should behave like an ASE
calculator and provide at least ``energy`` and ``forces``. ALF will wrap the
returned list with ``MLMD_calculator`` and add ``energy_stdev``,
``forces_stdev_mean``, and ``forces_stdev_max``.

Configuration snippets
   Point the master configuration at the training task:

   .. code-block:: json

      {
        "ML_task": "my_ml_interface.train_MYMODEL_task",
        "ML_config_path": "mymodel_config.json",
        "model_path": "models/model-{:04d}"
      }

   Point the sampler configuration at the ASE loader:

   .. code-block:: json

      {
        "ase_calculator": "my_ml_interface.MYMODEL_ASE_load_ensemble",
        "MLMD_calculator_options": {"well_params": null}
      }

Implementation checklist
   Before using a new ML backend in production, check that:

   * The training task can run on ``alf_ML_executor`` without importing GPU
     libraries on CPU-only workers.
   * The model output directory matches ``model_path.format(current_training_id)``.
   * The sampler loader can load the current model directory generated by the
     training task.
   * ASE calculator results use ALF's expected units, usually eV for energies
     and eV/Angstrom for forces.
   * Dataset names in the ML config match the HDF5 data written by
     ``properties_list`` and the units described in :doc:`units`.
   * External dependencies are installed through the environment loaded by the
     ML and sampler Parsl workers.

ML Interface API Links
----------------------

You can link from this guide directly to API pages:

* :doc:`ML interfaces package API <../api_documentation/alframework.ml_interfaces>`
* :doc:`HIPPYNN interface module <../api_documentation/alframework.ml_interfaces.hippynn_interface>`
* :doc:`NeuroChem interface module <../api_documentation/alframework.ml_interfaces.neurochem_interface>`
* :doc:`Fine-tuning module <../api_documentation/alframework.ml_interfaces.fine_tuning>`, with its
  :doc:`HIPPYNN <../api_documentation/alframework.ml_interfaces.hippynn_fine_tuning>` and
  :doc:`MACE <../api_documentation/alframework.ml_interfaces.mace_interface>` adapters
* :doc:`MLMD sampler guide <samplers>`
* :doc:`Parsl execution guide <parsl>`

Related Examples
----------------

See :doc:`../examples/simple_water`, :doc:`../examples/molten_salt`, and
:doc:`../examples/reactive_sampling` for HIPPYNN-based active-learning
workflows. See the UO2 example in the repository for a NeuroChem/ANI-style
workflow.
