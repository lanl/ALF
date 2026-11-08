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
     - ``alframework.ml_interfaces.fine_tuning.hippynn.HippynnFineTuner``, for a pretrained HIPPYNN
       model or ensemble.
   * - MACE
     - ``alframework.ml_interfaces.fine_tuning.mace.MaceFineTuner``, for a MACE foundation model such
       as MACE-MP.
   * - UMA
     - ``alframework.ml_interfaces.fine_tuning.uma.UmaFineTuner``, for one task of a UMA checkpoint,
       such as ``omol`` or ``omat``.

What Fine-Tuning Means In ALF
-----------------------------

In ALF, **fine-tuning** means: each model in the ensemble starts from a pretrained model's weights,
not from random ones, and trains on labels that ALF selected, with a fresh optimizer. Every round
starts again from the same pretrained model, on all the labels so far. The copies in a round differ
only through their seeds, which give each its own validation split and training order, or, from a
pretrained ensemble, through starting from different members.

It is different from these:

.. list-table::
   :header-rows: 1
   :widths: 22 26 24 28

   * -
     - Starts from
     - Trains on
     - In ALF
   * - Training from scratch
     - Random weights
     - All stored labels
     - ``train_HIPPYNN_ensemble_task`` and the other trainers; see :doc:`ml_interfaces`
   * - Resuming a training
     - The interrupted run's checkpoint and optimizer state
     - The same data
     - Not part of ALF's loop. HIPPYNN calls this restarting.
   * - Distillation
     - Random weights, for a smaller student
     - A teacher model's labels
     - :doc:`teacher_student`
   * - Fine-tuning
     - A pretrained model's weights, every round
     - Labels ALF selected, all so far
     - This page

Each family's own training does the fitting. ALF adds what's around it: MD on the current copies to
choose what to label, several copies so their disagreement can choose, rounds that start from the
pretrained model each time, a record of what each round started from and learned from, restarts,
and placing copies across nodes.

Whose recipe ALF runs
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Family
     - Recipe
   * - MACE
     - MACE's own ``mace_run_train``, with defaults from MACE's fine-tuning guidance: plain ("naive")
       fine-tuning, or multihead fine-tuning with replay when you give a replay file. See MACE's
       `fine-tuning guide <https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html>`__ and
       `guidance <https://mace-docs.readthedocs.io/en/latest/guide/finetuning_guidance.html>`__.
   * - UMA
     - fairchem's own fine-tuning, with the defaults of fairchem's template. See fairchem's
       `fine-tuning page <https://facebookresearch.github.io/fairchem/fine-tuning>`__.
   * - HIPPYNN
     - ALF's own. HIPPYNN documents
       `restarting <https://lanl.github.io/hippynn/examples/restarting.html>`__ an interrupted
       training, but not fine-tuning. ALF loads a checkpoint's network and weights, gives them a fresh
       optimizer, and trains them with the same steps as ALF's HIPPYNN trainer.

.. _fine-tuning-licenses:

Pretrained Models And Licenses
------------------------------

**ALF ships no model weights.** You supply the pretrained model, and its license and terms apply to
it and to the copies you fine-tune from it. Check them before you start, and again before you share
fine-tuned copies.

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Family
     - Where models come from
     - Terms
   * - MACE
     - MACE's foundation models, from the MACE developers (see MACE's
       `foundation models page <https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html>`__).
       Download once on a machine with network access; the teacher guide shows one way.
     - Per model. MACE lists MACE-MP-0 and MACE-MPA-0 under the MIT license, and several others,
       including MACE-OMAT-0, MACE-OFF23, and MACE-MH, under the Academic Software License. MACE-MDP
       isn't meant for fine-tuning.
   * - UMA
     - ``facebook/UMA`` on Hugging Face, gated: accept the license there, then download with ``hf``.
       See :doc:`teacher_student` for the steps.
     - The FAIR Chemistry License, including its acceptable-use policy. Fine-tuned copies are
       derivative works, so sharing them is subject to it, and publications using UMA must
       acknowledge it.
   * - HIPPYNN
     - Your own trained models, or someone else's.
     - Whatever their owner set.

Pretrained models load through PyTorch, which can run code stored in the file. Fine-tune only from
files you trust.

How A Round Works
-----------------

Fine-tuning uses ALF's ordinary loop. Only the ML task and the sampler's model loader change.

1. **Start.** ALF needs labels, or an ensemble to sample with, before the first round. See
   `Two Ways To Start`_.
2. **Fine-tune.** The ML task makes ``n_models`` copies of the pretrained model and fine-tunes each
   on every stored label.
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
   ensemble differed by 0.04 eV and 3e-3 eV/Å. Copies of one MACE or UMA checkpoint disagreed more
   in similar checks. Start from an ensemble when you can, and in early rounds compare the ``Es`` and
   ``Fsmax`` that sampling records with your cutoffs.

**From a pretrained ensemble in model slot 0.** Put the ensemble's members in
``models/model-0000/model-00``, ``model-01``, and so on, and point ``foundation_model`` at
``models/model-0000``. ALF finds that model at startup, skips the bootstrap, and samples with the
ensemble straight away, so its disagreement chooses the first labels. This is how
``examples/fine_tuning`` starts, with HIPPYNN.

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
     - The pretrained model every round starts from: a checkpoint file for MACE and UMA, a directory
       for HIPPYNN. A relative path is taken from the master directory, as are other settings ending
       in ``_file``.
   * - ``n_models``
     - Copies per round, at least 2. The default is 4.
   * - ``seed``
     - Copy ``i`` uses ``seed + i``. The default is 0.
   * - ``device``
     - ``cuda`` (the default) or ``cpu``. On one node, each GPU fine-tunes one copy at a time, up
       to ``gpus_per_node`` GPUs. On CPU, copies run one after another.
   * - ``E0s``
     - Required, because the right choice depends on the labels' level of theory. It sets the
       per-element reference energies. Each family's section lists its choices and which to use.

ALF checks the ``fine_tune`` block at startup, before anything is labeled.

**Units.** Labels must be in the units the pretrained model uses, which for ALF's samplers is eV and
eV/Å. ``properties_list`` converts QM output to those units; see :doc:`units`.

HIPPYNN
-------

The source is a HIPPYNN model directory, holding ``experiment_structure.pt`` and
``best_checkpoint.pt`` as HIPPYNN writes them, or an ensemble directory of such ``model-*``
members. From an ensemble, copy ``i`` starts from member ``i``, cycling if there are fewer members
than copies. Each copy keeps the pretrained network and its weights, gets a fresh Adam optimizer,
and trains on the shards in place, without copying them. This recipe is ALF's own; see
`Whose recipe ALF runs`_.

.. code-block:: json

   {
     "fine_tune": {
       "adapter": "alframework.ml_interfaces.fine_tuning.hippynn.HippynnFineTuner",
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
     - ``foundation`` keeps the energy head's atomic energies, for labels at the pretrained model's
       level of theory. ``average`` refits them to the training split before fine-tuning, as ALF's
       HIPPYNN trainer fits them, for labels at another level of theory.
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
       so an early round with few labels still trains. A round needs at least 3 labels, or 2 with
       ``h5_test_dir``.
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
extra in the sampler workers' environment. The source is a MACE checkpoint file; see
`Pretrained Models And Licenses`_. ALF runs MACE's own ``mace_run_train`` for each copy, with
defaults from MACE's fine-tuning guidance, and any of them can be changed.

.. code-block:: json

   {
     "fine_tune": {
       "adapter": "alframework.ml_interfaces.fine_tuning.mace.MaceFineTuner",
       "foundation_model": "foundation/mace-mp-0-small.model",
       "n_models": 4,
       "E0s": "estimated"
     }
   }

The sampler loads the copies for batched MD, with ``mlmd_trajectories_task`` as the
``sampler_task``:

.. code-block:: json

   "batched_calculator": "alframework.ml_interfaces.fine_tuning.mace.load_mace_batched_ensemble",
   "batched_calculator_options": {"dtype": "float32"},
   "driver": {"strategy": "batched", "engine": "torchsim"},
   "batching": {"batch_size": 16, "partial_policy": "timeout", "max_wait_seconds": 5.0}

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Meaning
   * - ``E0s``
     - As MACE recommends, in order: isolated-atom energies at your labels' level of theory, as a
       dictionary from atomic number to energy or a ``.json`` file; else ``estimated``, which fits
       them with the foundation model's help. ``foundation`` keeps the foundation model's, only for
       labels at its level of theory. MACE advises against ``average`` for fine-tuning, and ALF warns
       at startup if you choose it.
   * - ``learning_rate``, ``max_num_epochs``
     - MACE's ``lr`` and ``max_num_epochs``. Defaults, from MACE's recipe: 1e-3 and 6 for plain
       fine-tuning; 1e-4 and 20 with a ``replay_file``.
   * - ``batch_size``, ``valid_fraction``
     - MACE's ``batch_size`` and ``valid_fraction``. Defaults: 2 and 0.1.
   * - ``dtype``
     - Training precision, ``float64`` (the default) or ``float32``.
   * - ``freeze``
     - Optional. Freezes layers, as MACE's ``--freeze``.
   * - ``replay_file``
     - Optional. A local file with a sample of the foundation model's training data, for MACE's
       multihead fine-tuning; see `Forgetting And Checking A Fine-Tuned Model`_.
   * - ``energy_dataset``, ``forces_dataset``
     - The HDF5 datasets read from each shard. Defaults: ``energy`` and ``forces``, which must match
       ``properties_list``.
   * - ``mace_options``
     - Other ``mace_run_train`` arguments. By default, MACE's recommended ``ema`` (decay 0.999, or
       0.9999 with replay), ``amsgrad``, ``clip_grad`` 1.0, ``weight_decay`` 0.0, ``energy_weight`` and
       ``forces_weight`` 10, and ``scaling`` ``rms_forces_scaling``. Values you give replace these;
       ``false`` turns a switch off. Settings ALF manages, such as ``lr`` or ``seed``, are refused here.

MACE's guidance on E0s also notes that a high initial error, hundreds of meV per atom, usually
means the reference energies don't match the labels. Models with several heads, such as MACE-MH,
haven't been checked with ALF.

Each ``models/model-NNNN/model-NN`` holds ``mace.model``, which sampling loads; ``train.xyz``, the
labels it trained on as extended XYZ; and MACE's own ``train.log``, ``logs/``, ``results/``, and
``checkpoints/``.

UMA
---

Install ``fairchem-core`` in the ML workers' environment, and ``fairchem-core`` with ALF's ``torchsim``
extra in the sampler workers' environment. The source is a UMA checkpoint file; see
`Pretrained Models And Licenses`_. Compute nodes read the local file and never download.

ALF runs fairchem's own fine-tuning for each copy, writing fairchem's training configuration itself,
so you don't need fairchem's git repository. UMA fine-tunes one task at a time, and the task is the
level of theory: labels must match it. A fine-tuned copy must be used with the task it was
fine-tuned on.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Task
     - What it was trained on
   * - ``omol``
     - Non-periodic molecules and clusters, at ωB97M-V/def2-TZVPD. Uses each structure's total
       charge and spin multiplicity.
   * - ``omat``
     - Inorganic materials, at a PBE-level of theory.
   * - ``oc20``, ``oc25``
     - Open Catalyst datasets, for catalysis.
   * - ``odac``
     - Metal-organic frameworks.
   * - ``omc``
     - Molecular crystals.

.. code-block:: json

   {
     "fine_tune": {
       "adapter": "alframework.ml_interfaces.fine_tuning.uma.UmaFineTuner",
       "foundation_model": "foundation/checkpoints/uma-s-1p2p1.pt",
       "task": "omol",
       "n_models": 4,
       "E0s": "average",
       "epochs": 10
     }
   }

The sampler loads the copies for batched MD, with ``mlmd_trajectories_task`` as the
``sampler_task``. Its ``task`` must be the one the copies were fine-tuned on:

.. code-block:: json

   "batched_calculator": "alframework.ml_interfaces.fine_tuning.uma.load_uma_batched_ensemble",
   "batched_calculator_options": {"task": "omol", "dtype": "float32", "charge": 0, "spin": 1},
   "driver": {"strategy": "batched", "engine": "torchsim"}

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Meaning
   * - ``task``
     - Required: one of the tasks above.
   * - ``E0s``
     - ``average`` or ``foundation``. See below.
   * - ``epochs``, ``steps``
     - How long each copy trains. Set exactly one; the other is ``null``. The default, 1 epoch, is
       fairchem's. An early ALF round has few labels, so 1 epoch is only a few steps;
       ``examples/uma_fine_tuning`` raises it to 10.
   * - ``batch_size``, ``lr``, ``weight_decay``, ``max_neighbors``
     - Defaults, from fairchem's template: 2, 4e-4, 1e-3, and 300. fairchem suggests the largest batch
       that fits, as long as a round still takes many steps, and notes that ``max_neighbors`` 100 is
       usually enough when memory is short.
   * - ``valid_fraction``
     - The fraction of labels each copy keeps aside for validation, split with its own seed, so
       copies see different data. The default is 0.1; each copy keeps at least one structure aside, and
       a round needs at least 2 labels.
   * - ``energy_dataset``, ``forces_dataset``
     - The HDF5 datasets read from each shard. Defaults: ``energy`` and ``forces``, which must match
       ``properties_list``.
   * - ``keep_training_checkpoints``
     - ``false`` by default: after training, fairchem's resume checkpoints, which hold optimizer state
       and several copies of the weights, are removed, keeping the copy itself.

Reference energies
~~~~~~~~~~~~~~~~~~

UMA computes each energy as a per-element reference energy for each atom, plus what the network
learns on top.

* ``average`` refits the references to your labels by least squares, as fairchem's own dataset
  script does. Use it when your labels come from a different level of theory than the task, such as
  ORCA labels on ``omol``: absolute energies then differ by large per-element amounts, which the
  refit absorbs.
* ``foundation`` keeps the checkpoint's own references for the task, and its energy scale. Use it
  when your labels share the task's level of theory, or when they cover too few compositions for a
  refit to separate the elements; with only water, for example, hydrogen and oxygen always appear
  together.

Charge and spin
~~~~~~~~~~~~~~~

``omol`` needs each structure's total charge and spin multiplicity. When a structure's metadata has
``charge`` or ``mult``, as the teacher and some builders set them, ALF stores them in the shard as
``total_charge`` and ``spin_multiplicity``, and fine-tuning reads them; structures without them are
neutral singlets. ALF's batched samplers don't pass charge or spin per structure, so the sampler's
``charge`` and ``spin`` apply to every structure sampled in a run. They default to 0 and 1.

What a UMA round writes
~~~~~~~~~~~~~~~~~~~~~~~

Each ``models/model-NNNN/model-NN`` holds ``uma.pt``, the fine-tuned checkpoint, which sampling
loads; ``fine_tune.yaml``, the training configuration ALF wrote; ``data/``, the copy's training and
validation sets; ``train.log``; and fairchem's logs under ``runs/``.

Each copy holds a full UMA model while sampling, so several copies of a large checkpoint need GPU
memory to match.

Forgetting And Checking A Fine-Tuned Model
------------------------------------------

Fine-tuning adapts a model to your structures, and it can lose accuracy elsewhere.

* **MACE:** a ``replay_file`` makes MACE's multihead fine-tuning also train on a sample of the
  foundation model's data, which MACE's guidance found preserves its accuracy outside your system.
  MACE provides replay data for its MP models, and its ``mp`` shortcut downloads it during training,
  which compute nodes usually can't do; ALF needs a local file, and otherwise turns replay off.
* **UMA:** fairchem's fine-tuning has no replay. ``E0s: "foundation"`` keeps the most of the
  pretrained model it can.
* **HIPPYNN:** freezing the interaction network and fine-tuning only the energy head keeps more of
  the pretrained model.

MACE's guidance also found that low test errors don't show that MD with a fine-tuned model is
stable. Before relying on the final copies, run MD with them on the structures you care about, and
check the properties you need, not only the errors in ``training_log.txt`` or ``train.log``.

The Lineage Record
------------------

Every round writes ``models/model-NNNN/fine_tune_lineage.json``, saying what it started from and
learned from, and ``label_sources.json``, the label sources of its shards. For a MACE round, abridged:

.. code-block:: text

   {
     "foundation_model": ".../mace-mp-0-small.model",
     "foundation_model_sha256": "...",
     "adapter": "alframework.ml_interfaces.fine_tuning.mace.MaceFineTuner",
     "package": "mace-torch",
     "package_version": "...",
     "settings": {"n_models": 2, "seed": 7, "device": "cpu", "E0s": "estimated", "learning_rate": 0.001,
                  "max_num_epochs": 6, "mace_options": {"ema": true, "ema_decay": 0.999, ...}, ...},
     "shards": ["data-0000.h5", "data-0001.h5"],
     "seeds": [7, 8],
     "completed": [true, true]
   }

A HIPPYNN source is several files, so its record lists each one's SHA-256 under
``foundation_model_files``, such as ``model-00/best_checkpoint.pt``, in place of
``foundation_model_sha256``. ``settings`` includes every default the round used.

Walkthroughs
------------

The examples name the checkpoints they expect, but don't include them; see
`Pretrained Models And Licenses`_.

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

1. Copy your MACE checkpoint to ``foundation/mace-mp-0-small.model``, or point
   ``foundation_model`` at it.
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
from MACE's predictions. TorchSim MD on the first round's copies selected new water boxes, and the
second round fine-tuned from the original checkpoint on both shards and was promoted. In a separate
check with ALF's defaults, which follow MACE's recipe, two copies fine-tuned on 60 EMT-labeled water
molecules cut the energy error on held-out water from 0.94 eV for the foundation model to 0.011 and
0.022 eV, once each model's constant offset was removed.

UMA: one task of a checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/uma_fine_tuning`` fine-tunes the ``omol`` task of ``uma-s-1p2p1`` on water labeled with
ORCA.

1. On a machine with network access, after accepting UMA's license:

   .. code-block:: bash

      hf auth login
      hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir foundation

2. Set ``QM_run_command`` in ``orca_config.json``, and the machine settings in
   ``parsl_configs.py``. The ML executor needs GPUs, and so does the sampler executor.
3. Check the builder and the QM labels, then start the run:

   .. code-block:: bash

      python -m alframework --master master_config.json --test_builder
      python -m alframework --master master_config.json --test_qm
      python -m alframework --master master_config.json

ORCA's level of theory isn't ``omol``'s, so the example refits the reference energies with
``E0s: "average"``. Its builder makes clusters (``"periodic": false``), because ORCA labels each
structure as a cluster and ``omol`` was trained only on non-periodic structures. A fine-tuned
model trained on periodic copies of ORCA's cluster labels would learn interactions with images that
ORCA never saw. The MACE example builds clusters for the same reason.

The same loop was run end to end on a CPU through ALF's coordinator, with ``uma-s-1p2p1``, single
water molecules, and EMT standing in for ORCA. Two copies fine-tuned for 4 steps on the bootstrap
labels and were promoted. TorchSim MD on them selected new structures, and the second round
fine-tuned from the original checkpoint on both shards and was promoted. Unlike small HIPPYNN
copies, these UMA copies of one checkpoint disagreed clearly: by 0.24 eV in energy and 0.22 eV/Å in
forces, in a separate two-copy round. Separate rounds also ran ``omol`` with ``foundation``
references, and ``omat`` on periodic copper, and an ion and a radical reached training with their
charge and spin.

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

Sampling also needs a loader for the copies; see :doc:`ml_interfaces`. Follow the family's own
fine-tuning advice for its defaults, and say where they come from.
