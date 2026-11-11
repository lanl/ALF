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

.. _fine-tuning-recipes:

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

.. _fine-tuning-starting:

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
eV/Å. ``properties_list`` converts QM output to those units; see :doc:`units`. Fine-tuning reads the
``energy`` and ``forces`` datasets. For ground-state DFT on a GPU, use GPU4PySCF with ``nroots: 0`` and
store ``E0`` and ``F0`` under those names; see :ref:`configuration-property-names`.

Model Families
--------------

Each family has its own page: settings, what a round writes, and a walkthrough. The examples name
the checkpoints they expect, but don't include them; see `Pretrained Models And Licenses`_.

.. toctree::
   :maxdepth: 1

   fine_tuning_hippynn
   fine_tuning_mace
   fine_tuning_uma

.. _fine-tuning-forgetting:

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
