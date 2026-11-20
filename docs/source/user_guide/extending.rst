.. _extending:

Extending ALF
=============

ALF runs four kinds of task, a builder, a sampler, a labeler and a trainer, and the sampler runs dynamics with an
engine on a model's evaluator. Each is code ALF imports from a string in a configuration, so code of your own fits
without changing ALF. This page lists what each piece is given and must return, the optional hooks every piece can
have, and the checks that tell you, in your own tests, that a piece fits.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Piece
     - Selected by
     - Contract
   * - Builder
     - ``builder_task`` (master)
     - Takes ``moleculeid`` (one structure) or ``moleculeids`` (several) and ``builder_config``; returns one
       ``MoleculesObject`` or a list of them, with the IDs it was given. See :doc:`builders`.
   * - Sampler
     - ``sampler_task`` (master)
     - Takes ``molecule_object`` (one trajectory) or ``molecule_objects`` (several), ``sampler_config``,
       ``model_path`` and ``current_model_id``; returns each structure, with its atoms set to the frame to label or
       to ``None``. See :doc:`samplers`.
   * - Dynamics engine
     - ``driver.engine`` (sampler)
     - A built-in name or the import path of a class; see :ref:`extending-engines`.
   * - Labeler
     - ``QM_task`` (master)
     - Takes ``molecule_objects`` (a batch) or ``molecule_object`` (one structure) and ``QM_config``; returns them
       with results stored and a converged flag, or a failed one marked unconverged with ``labeling_error`` or
       ``qm_error`` in its metadata. See :doc:`qm_interfaces`.
   * - Trainer
     - ``ML_task`` (master)
     - Takes ``ML_config``, ``h5_dir``, ``model_path`` and ``current_training_id``; returns
       ``(completed, current_training_id)``. See :ref:`extending-models`.
   * - Model for sampling
     - ``ase_calculator`` or ``batched_calculator`` (sampler)
     - A loader for the trained model; see :ref:`extending-models`.
   * - Uncertainty
     - ``acquisition.terms`` (sampler)
     - Named numbers per structure that a model reports itself; see :ref:`acquisition-own-uncertainty`.

ALF fills each task's arguments by name from the configurations (``build_input_dict``), so a task can also take any
other setting, such as ``master_directory`` or ``gpus_per_node``, by naming it as a parameter.

Optional Hooks
--------------

**Checking a configuration at startup.** Give a task function a ``validate_config(config, master_config)``
attribute that raises ``ValueError`` for a configuration it can't run. ALF calls it before the bootstrap, with the
builder, sampler, QM or ML configuration, so a mistake stops the run instead of failing tasks:

.. code-block:: python

   def validate_my_labeler(QM_config, master_config):
       if "method" not in QM_config:
           raise ValueError("my_labeler_task needs QM_config['method'].")

   getattr(my_labeler_task, "func", my_labeler_task).validate_config = validate_my_labeler

**Declaring the settings a task reads.** Some settings are read only by some tasks, and ALF refuses them for a task
that doesn't declare it reads them, at startup, rather than letting a task ignore them. Declare them with
``alframework.tools.capabilities.declares``, under ``@python_app``:

.. code-block:: python

   from parsl import python_app
   from alframework.tools.capabilities import declares

   @python_app(executors=["alf_sampler_executor"])
   @declares("returned_metadata", "returned_momenta", "metadata_files")
   def my_sampler_task(molecule_objects, sampler_config, model_path, current_model_id):
       ...

The settings that need a declaration are the sampler's ``returned_metadata``, ``returned_momenta``,
``metadata_files`` and ``frame_spool`` (for ``data_by_reference``), and the trainer's ``shared_training_data``,
``stream_final_evaluation`` and ``h5_read_workers``. A task that wraps a built-in one with ``functools.wraps``
inherits its declarations.

**Where your code runs.** The coordinator imports every task's module, and an engine's, to read its signature and
declarations, so import GPU libraries such as PyTorch inside functions. A task's body runs on the workers of the
executor its ``@python_app`` names: ``alf_sampler_executor`` for builders and samplers, ``alf_QM_executor`` for
labelers and ``alf_ML_executor`` for trainers. Decorating it with ``alframework.tools.task_timeline.timed(stage)``,
under ``@python_app``, records it in the task timeline (see :doc:`monitoring`).

.. _extending-engines:

Dynamics Engines
----------------

``driver.engine`` takes ``"ase"``, ``"torchsim"`` or ``"alchemi"``, or the import path of an engine class, such as
``"my_package.engines.MyEngine"``. The class takes ``(atoms_list, models, dt, temperatures_K, rngs,
calculator_options, device=..., friction=...)``, one entry per trajectory, and provides what the sampler calls:
``step(indices, n)``, returning ``{index: exception}`` for trajectories that failed; ``observe(index)``, returning
``{"prediction": ..., "min_distance": ...}`` with the members' unbiased prediction; ``atoms(index)``,
``set_temperature``, ``scale_cell``, ``finish``, ``attach_trajectory`` and the per-trajectory ``steps`` counts.
A packed engine can subclass :class:`~alframework.samplers.engines.packed.PackedMLMDEngine`, which does all of this
for trajectories packed on one device and needs only its integrator. Set ``accepts_batched_calculator = True`` on
a class that evaluates the model a ``batched_calculator`` loads; otherwise it gets the ``ase_calculator`` model.

.. _extending-models:

ML Architectures
----------------

A new architecture needs a trainer, and a loader that gives the sampler the trained model.

**Trainer.** A Parsl task on ``alf_ML_executor`` that reads the HDF5 shards in ``h5_dir``, trains, and writes the
model under ``model_path.format(current_training_id)``; :ref:`ml-interface-new-architecture-template` has a
template. It returns ``(completed, current_training_id)``, where ``completed`` is a boolean or one per ensemble
member. Read shards through :func:`alframework.tools.h5_parallel.read_shard`, which handles label storage in parts
and reads large files in parallel. ``n_models`` can be 1 for a run that chooses frames by the model's own
uncertainty.

**Loader for sampling.** One of:

- ``ase_calculator``: a function ``(model_directory, device)`` returning a list of ASE calculators, one per member,
  each giving ``energy`` and ``forces``. Both sampler families and every engine can use it.
- ``batched_calculator``: a function ``(model_directory, device, **batched_calculator_options)`` returning an
  evaluator for the ``torchsim`` and ``alchemi`` engines. It has ``n_models`` and ``evaluate(numbers, positions,
  cells, pbc, counts)``, which takes the batch's packed tensors and returns energies ``[n_models, structures]`` and
  forces ``[n_models, atoms, 3]``. It may also provide ``predict(atoms_list)``, in the layout
  :mod:`alframework.ml_interfaces.ensemble_inference` describes.

Either can report uncertainty of its own, an ASE calculator in ``results["alf_uncertainty"]``, an evaluator in
``predict``'s ``"uncertainty"`` or in ``last_uncertainty`` after ``evaluate``; see
:ref:`acquisition-own-uncertainty`. Excited-state models have more to provide; see :doc:`excited_states`.

Checking a Piece in Your Own Tests
----------------------------------

:mod:`alframework.testing` calls a piece the way ALF would, in the test's own process, and raises
``ContractError`` naming what doesn't fit:

.. code-block:: python

   from alframework.testing import check_builder_task, check_engine, check_evaluator, check_labeler_task

   def test_my_pieces_fit_alf(tmp_path):
       structures = check_builder_task(my_builder_task, builder_config={...})
       check_labeler_task(my_labeler_task, structures, QM_config={...}, QM_scratch_dir=str(tmp_path),
                          properties_list={"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]})
       check_evaluator(load_my_model("models/model-0000", device="cpu"), [atoms for atoms in my_structures])
       check_engine("my_package.engines.MyEngine", my_structures, my_model)

The checks cover what ALF relies on: IDs and periodicity, labels it can store and failures that say why, the
evaluator's layout and finite values, and the engine's methods and step counts. They don't check the science.
ALF's own tests run them on its built-in pieces.
