Fine-Tuning Existing Models With ALF
====================================

ALF can fine-tune a model you already have on structures that ALF itself selects. The model being
fine-tuned runs the MD that finds new structures. Structures where its copies disagree are labeled,
and each round adapts the model to everything labeled so far.

One task handles every model family, through a small adapter per family:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Family
     - Adapter
   * - HIPPYNN
     - ``alframework.ml_interfaces.hippynn_fine_tuning.HippynnFineTuner``, for a pretrained HIPPYNN
       model or ensemble.
   * - MACE
     - ``alframework.ml_interfaces.mace_interface.MaceFineTuner``, for a MACE foundation model such
       as MACE-MP.

UMA can't be fine-tuned this way yet, but it can already label structures as a teacher; see
:doc:`teacher_student`.

How A Round Works
-----------------

Fine-tuning uses ALF's ordinary loop. Only the ML task and the sampler's model loader change.

1. **Start.** ALF needs labels, or an ensemble to sample with, before the first round. See
   `Two Ways To Start`_.
2. **Fine-tune.** The ML task makes ``n_models`` copies of the pretrained model and fine-tunes each
   on every stored label. Each copy has its own seed, so it gets its own validation split and
   training order.
3. **Sample.** MD runs on the fine-tuned copies. Where they disagree, beyond ``Escut`` or
   ``Fscut``, the structure is sent for labeling, exactly as with a trained ensemble.
4. **Repeat.** After each store, the next round fine-tunes new copies, again from the pretrained
   model, on all labels so far. Its model replaces the last one when every copy finishes.

Starting every round from the pretrained model keeps rounds independent. A poor round can't carry
over into the next, errors don't build up from round to round, and each model depends only on the
pretrained model and the labels. The cost of a round grows with the number of labels.

The copies' disagreement is what selects new structures, so ``n_models`` must be at least 2.

Two Ways To Start
-----------------

**From a bootstrap.** ALF labels ``bootstrap_set`` structures from the builder, and the first
round fine-tunes on them. Use this when you start from a single model: copies of one model would
never disagree until they are fine-tuned apart.

.. warning::

   Copies fine-tuned from one model differ only through their seeds, splits, and training order,
   and they can stay very close. In a short CPU check with small HIPPYNN models, two copies
   fine-tuned for 3 epochs from one model differed in energy by 0.014 eV but in forces by only
   1e-6 eV/Å, far below any useful ``Fscut``. Copies started from the two members of a pretrained
   ensemble differed by 0.04 eV and 3e-3 eV/Å. Start from an ensemble when you can, and in early
   rounds compare the ``Es`` and ``Fsmax`` that sampling records with your cutoffs.

**From a pretrained ensemble in model slot 0.** Put the ensemble's members in
``models/model-0000/model-00``, ``model-01``, and so on, and point ``foundation_model`` at
``models/model-0000``. ALF finds that model at startup, skips the bootstrap, and samples with the
ensemble straight away, so its disagreement chooses the first labels. This is how
``examples/fine_tuning`` starts. It works for any family whose sampler loader reads that directory;
for HIPPYNN, any HIPPYNN loader does.

Either way, labels already in ``h5_dir`` are used from the first round.

Choosing Between Fine-Tuning And A Student
------------------------------------------

* **Fine-tune** when you want the pretrained model itself, made accurate for your system, and can
  afford to run it for MD.
* **Distill** into a HIPPYNN student, with :doc:`teacher_student`, when you need a much faster model
  and the foundation model is already accurate enough to learn from.

Common Setup
------------

In ``master_config.json``, set the ML task:

.. code-block:: json

   "ML_task": "alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_task"

The ML configuration holds one ``fine_tune`` block. These settings apply to every family, and the
sections below add each family's own:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Setting
     - Meaning
   * - ``adapter``
     - The family's adapter class, from the table above.
   * - ``foundation_model``
     - The pretrained model every round starts from: a file for MACE, a directory for HIPPYNN. A
       relative path is taken from the master directory, as are other settings ending in ``_file``.
   * - ``n_models``
     - Copies per round, at least 2. The default is 4.
   * - ``seed``
     - Copy ``i`` uses ``seed + i``. The default is 0.
   * - ``device``
     - ``cuda`` (the default) or ``cpu``. On one node, each GPU fine-tunes one copy at a time, up
       to ``gpus_per_node`` GPUs. On CPU, copies run one after another.
   * - ``E0s``
     - Required, because it depends on the labels' level of theory. ``foundation`` keeps the
       pretrained model's atomic energies, for labels at its level of theory. ``average`` refits them
       to the labels. Each family's section lists its other choices.

ALF checks the ``fine_tune`` block at startup, before anything is labeled.

**Units.** Labels must be in the units the pretrained model uses, which for ALF's samplers is eV and
eV/Å. ``properties_list`` converts QM output to those units; see :doc:`units`.

**Trust.** Pretrained models load through PyTorch, which can run code stored in the file. Fine-tune
only from models you trust.

HIPPYNN
-------

The source is a HIPPYNN model directory, holding ``experiment_structure.pt`` and
``best_checkpoint.pt`` as HIPPYNN writes them, or an ensemble directory of such ``model-*``
members. From an ensemble, copy ``i`` starts from member ``i``, cycling if there are fewer members
than copies. Each copy keeps the pretrained network and its weights, gets a fresh Adam optimizer,
and trains on the shards in place, without copying them.

.. code-block:: json

   {
     "fine_tune": {
       "adapter": "alframework.ml_interfaces.hippynn_fine_tuning.HippynnFineTuner",
       "foundation_model": "models/model-0000",
       "n_models": 4,
       "E0s": "foundation",
       "learning_rate": 1e-4,
       "freeze": []
     }
   }

The copies are written in HIPPYNN's usual layout, so the sampler loads them as it loads any trained
HIPPYNN ensemble, with ``"ase_calculator": "alframework.ml_interfaces.hippynn_interface.HIPNN_ASE_load_ensemble"``
or the batched ``load_hippynn_batched_ensemble``; see :doc:`ml_interfaces`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Meaning
   * - ``E0s``
     - ``foundation`` or ``average``. ``average`` refits the energy head's atomic energies to the
       training split before fine-tuning, as ALF's HIPPYNN trainer fits them.
   * - ``learning_rate``
     - Adam's learning rate. The default, 1e-4, is low, to adapt rather than retrain.
   * - ``scheduler_options``
     - For HIPPYNN's ``RaiseBatchSizeOnPlateau``. Defaults: ``max_batch_size`` 64, ``patience`` 20,
       ``factor`` 0.5.
   * - ``controller_options``
     - For HIPPYNN's ``PatienceController``. Defaults: ``batch_size`` 32, ``eval_batch_size`` 64,
       ``max_epochs`` 200, ``termination_patience`` 51.
   * - ``stopping_key``
     - The validation loss that decides the best model. By default, the one the pretrained model was
       trained with.
   * - ``freeze``
     - Graph nodes, or parameter-name prefixes, whose weights stay fixed. See below.
   * - ``valid_size``, ``test_size``
     - Fractions of the labels for validation and testing, 0.1 each. Each gets at least one structure,
       so an early round with few labels still trains. A round needs at least 3 labels.
   * - ``h5_test_dir``
     - Optional. A directory of shards to test on instead of a random test split, such as a fixed
       QM-labeled set.
   * - ``remove_high_energy_cut``, ``remove_high_energy_std``, ``remove_high_forces_cut``,
       ``remove_high_forces_std``
     - Leave out structures with extreme energies or forces, as in ALF's HIPPYNN trainer.
   * - ``energy_key``, ``force_key``, ``species_key``, ``coordinates_key``, ``quadrupole_key``
     - Dataset names in the shards. Defaults: ``energy``, ``forces``, ``species``, ``coordinates``, and
       none.
   * - ``structure_file``, ``checkpoint_file``
     - The source files. Defaults: ``experiment_structure.pt`` and ``best_checkpoint.pt``.

Freezing
~~~~~~~~

``freeze`` takes the names of graph nodes, such as the interaction network or the energy head, or
the start of parameter names. To see a model's node names, load it and list the nodes that have
parameters:

.. code-block:: python

   from hippynn.experiment.serialization import load_checkpoint

   model = load_checkpoint("model-00/experiment_structure.pt", "model-00/best_checkpoint.pt",
                           restart_db=False, map_location="cpu")["training_modules"].model
   names = [name for name, _ in model.named_parameters()]
   print([node.name for node, key in model.names_dict.items()
          if any(name.startswith(f"moddict.{key}.") for name in names)])

For a typical model this prints names such as ``hipnn`` and ``henergy``. ``"freeze": ["hipnn"]``
then fine-tunes only the energy head. A name that matches nothing stops the round with a list of
the names that would. ``training_log.txt`` reports how many parameters were frozen.

What a HIPPYNN round writes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ``models/model-NNNN/model-NN`` holds HIPPYNN's usual output: ``experiment_structure.pt``,
``best_checkpoint.pt``, ``best_model.pt``, ``training_log.txt``, and its metrics and plots. ALF adds
``fine_tune_args.json``, recording the copy's source, seed, shards, and settings, and
``fine_tune.log``, the copy's own output before HIPPYNN's log starts.

MACE
----

Install ``mace-torch`` in the ML workers' environment, and ``mace-torch`` with ALF's ``torchsim``
extra in the sampler workers' environment. The source is a MACE checkpoint file. The teacher
walkthrough shows one way to download one; see :doc:`teacher_student`.

.. code-block:: json

   {
     "fine_tune": {
       "adapter": "alframework.ml_interfaces.mace_interface.MaceFineTuner",
       "foundation_model": "foundation/mace-mp-0-small.model",
       "n_models": 4,
       "E0s": "average",
       "learning_rate": 0.001,
       "max_num_epochs": 50
     }
   }

The sampler loads the copies for batched MD, with ``mlmd_trajectories_task`` as the
``sampler_task``:

.. code-block:: json

   "batched_calculator": "alframework.ml_interfaces.mace_interface.load_mace_batched_ensemble",
   "batched_calculator_options": {"dtype": "float32"},
   "driver": {"strategy": "batched", "engine": "torchsim"},
   "batching": {"batch_size": 16, "partial_policy": "timeout", "max_wait_seconds": 5.0}

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Meaning
   * - ``E0s``
     - ``foundation``, ``average``, or ``estimated``, as MACE defines them; or a dictionary from
       atomic number to energy, or a ``.json`` file.
   * - ``learning_rate``, ``max_num_epochs``, ``batch_size``, ``valid_fraction``
     - MACE's ``lr``, ``max_num_epochs``, ``batch_size``, and ``valid_fraction``. Defaults: 0.001, 50,
       4, and 0.1.
   * - ``dtype``
     - Training precision, ``float64`` (the default) or ``float32``.
   * - ``freeze``
     - Optional. Freezes layers, as MACE's ``--freeze``.
   * - ``replay_file``
     - Optional. A local file with a sample of the foundation model's training data; see
       `Forgetting`_.
   * - ``energy_dataset``, ``forces_dataset``
     - The HDF5 datasets read from each shard. Defaults: ``energy`` and ``forces``, which must match
       ``properties_list``.
   * - ``mace_options``
     - Other ``mace_run_train`` arguments, such as ``{"swa": true, "patience": 10}``. Settings ALF
       manages, such as ``lr`` or ``seed``, are refused here.

Each ``models/model-NNNN/model-NN`` holds ``mace.model``, which sampling loads; ``train.xyz``, the
labels it trained on as extended XYZ; and MACE's own ``train.log``, ``logs/``, ``results/``, and
``checkpoints/``.

Forgetting
~~~~~~~~~~

Plain fine-tuning adapts a model to your structures, and it can lose accuracy elsewhere. With
MACE, a ``replay_file`` makes MACE's multihead fine-tuning also train on a sample of the foundation
model's data, which limits this. MACE's default is to download replay data for MACE-MP models,
which compute nodes usually can't do, so ALF turns that off unless a local file is configured. With
HIPPYNN, freezing the interaction network and fine-tuning only the energy head is the usual way to
keep more of the pretrained model.

The Lineage Record
------------------

Every round writes ``models/model-NNNN/fine_tune_lineage.json``, saying what it started from and
learned from, and ``label_sources.json``, the label sources of its shards. For a MACE round on a
CPU, abridged:

.. code-block:: text

   {
     "foundation_model": ".../mace-mp-0-small.model",
     "foundation_model_sha256": "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736",
     "adapter": "alframework.ml_interfaces.mace_interface.MaceFineTuner",
     "package": "mace-torch",
     "package_version": "0.3.17",
     "settings": {"n_models": 2, "seed": 7, "device": "cpu", "E0s": "average", "learning_rate": 0.01, ...},
     "shards": ["data-0000.h5", "data-0001.h5"],
     "seeds": [7, 8],
     "completed": [true, true]
   }

A HIPPYNN source is several files, so its record lists each one's SHA-256 under
``foundation_model_files``, such as ``model-00/best_checkpoint.pt``, in place of
``foundation_model_sha256``. ``settings`` includes every default the round used.

Walkthroughs
------------

HIPPYNN: a pretrained ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/fine_tuning`` fine-tunes a pretrained HIPPYNN ensemble on water labeled with ORCA. Run
each step from that directory.

1. Put the pretrained ensemble in ``models/model-0000/model-00``, ``model-01``, and so on, with
   ``experiment_structure.pt``, ``best_checkpoint.pt``, and ``best_model.pt`` in each.
2. Optionally, put earlier labels in ``h5store/`` as ALF shards.
3. Put starting structures in ``fragment_library/``, set ``QM_run_command`` in
   ``orca_config.json``, and the machine settings in ``parsl_configs.py``.
4. Check the stages, then start the run:

   .. code-block:: bash

      python -m alframework --master master_config.json --test_builder
      python -m alframework --master master_config.json --test_sampler
      python -m alframework --master master_config.json --test_qm
      python -m alframework --master master_config.json

ALF samples with the pretrained ensemble from the start. The first store starts round 1, written to
``models/model-0001``, which replaces the pretrained ensemble for sampling once every copy finishes.
To fine-tune once on labels already in ``h5store/``, without sampling, run ``--test_ml``.

The same loop was run end to end on a CPU through ALF's coordinator, with small HIPPYNN models,
single water molecules, and EMT standing in for ORCA. Started from a pretrained two-member ensemble
in ``models/model-0000``, ALF skipped the bootstrap, sampled with the ensemble, stored only sampled
structures, and promoted round 1. Started instead from one model and a bootstrap, round 0 was
promoted, sampling on its copies selected new structures, and round 1 fine-tuned from the original
model on both shards and was promoted. Those copies barely disagreed (see the warning in
`Two Ways To Start`_), so that run used cutoffs of zero.

MACE: a foundation model
~~~~~~~~~~~~~~~~~~~~~~~~

``examples/mace_fine_tuning`` fine-tunes MACE-MP small on water labeled with ORCA.

1. Copy the checkpoint to ``foundation/mace-mp-0-small.model``.
2. Set ``QM_run_command`` in ``orca_config.json``, and the machine settings in
   ``parsl_configs.py``. The ML executor needs GPUs, and so does the sampler executor for batched
   MD.
3. Check the builder and the QM labels, then start the run:

   .. code-block:: bash

      python -m alframework --master master_config.json --test_builder
      python -m alframework --master master_config.json --test_qm
      python -m alframework --master master_config.json

After the bootstrap, ``models/model-0000`` holds the first fine-tuned copies, and sampling starts
with them.

The same loop was run end to end on a CPU, with EMT standing in for ORCA so that the labels differed
from MACE's predictions. Two copies fine-tuned for a few epochs on the bootstrap labels. TorchSim MD
on those copies selected new water boxes, and the second round fine-tuned from the original
checkpoint on both shards and was promoted. In a separate check on held-out water, fine-tuning on
60 EMT-labeled frames cut the energy error against EMT from 0.94 eV for the foundation model to
0.04 and 0.10 eV for the two copies, once each model's constant offset was removed.

Running Across Nodes
--------------------

``alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_multinode_task`` fine-tunes each copy as
its own app, so a round can span any number of nodes, for any family. It retries failed copies with
``member_retries``, as the HIPPYNN multi-node tasks do; see :ref:`ml-multinode-training` for the
Parsl setup.

Restarts
--------

A round that is running when the coordinator stops is lost. After a restart, the next round uses a
new model ID and starts again from the pretrained model. See :ref:`configuration-restarts`.

Adding A Model Family
---------------------

A model family plugs in with an adapter class, named in ``fine_tune.adapter``. ALF constructs it
with the ``fine_tune`` settings, and it provides:

* ``validate(master_config)``, called at startup;
* ``prepare_data(shards, member_dir)``, which gives the family's training its input, from ALF's
  HDF5 shards;
* ``train_member(data, member_dir, member, seed, device, environment)``, which fine-tunes copy
  ``member``. Run the training in a child process with ``environment``, which selects the copy's
  GPU;
* ``member_complete(member_dir)``, whether that copy finished;
* ``identity()``, the package name and version, recorded in the lineage;
* optionally ``fingerprint()``, each source file's SHA-256, when the source is more than one file.

Sampling also needs a loader for the copies; see :doc:`ml_interfaces`.
