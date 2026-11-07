Fine-Tuning Existing Models With ALF
====================================

ALF can fine-tune an existing model, such as a MACE foundation model, on structures that ALF
itself selects. The model being fine-tuned runs the MD that finds new structures, the structures
where its copies disagree are labeled, and each round adapts the model to everything labeled so
far. MACE is supported now. Other model families plug in through an adapter; see
`Other Model Families`_.

How A Round Works
-----------------

Fine-tuning uses ALF's ordinary loop. Only the ML task and the sampler's model loader change.

1. **Bootstrap.** ALF labels ``bootstrap_set`` structures from the builder, as in any run.
2. **Fine-tune.** The ML task makes ``n_models`` copies of the original checkpoint and fine-tunes
   each on every stored label. Each copy has its own seed, so it gets its own validation split and
   training order.
3. **Sample.** MD runs on the fine-tuned copies. Where they disagree, beyond ``Escut`` or
   ``Fscut``, the structure is sent for labeling, exactly as with a trained ensemble.
4. **Repeat.** After each store, the next round fine-tunes new copies, again from the original
   checkpoint, on all labels so far. Its model replaces the last one when every copy finishes.

Starting every round from the original checkpoint keeps rounds independent: a poor round can't
carry over into the next, and the result depends only on the labels. The cost of a round grows
with the number of labels.

The copies' disagreement is what selects new structures, so ``n_models`` must be at least 2.
Identical copies would never disagree, which is why each one is fine-tuned before ALF samples with
it.

Choosing Between Fine-Tuning And A Student
------------------------------------------

* **Fine-tune** when you want the foundation model itself, made accurate for your system, and can
  afford to run it for MD.
* **Distill** into a HIPPYNN student, with :doc:`teacher_student`, when you need a much faster model
  and the foundation model is already accurate enough to learn from.

Setting Up MACE Fine-Tuning
---------------------------

Install ``mace-torch`` in the ML workers' environment, and ``mace-torch`` with ALF's ``torchsim``
extra in the sampler workers' environment. Put the foundation checkpoint on a filesystem that
compute nodes can read: they often have no network, and ALF records the checkpoint's SHA-256 with
every round. The teacher walkthrough shows one way to download a MACE checkpoint; see
:doc:`teacher_student`.

In ``master_config.json``:

.. code-block:: json

   "ML_task": "alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_task",
   "ML_config_path": "fine_tune_config.json",
   "sampler_task": "alframework.samplers.batched_mlmd.mlmd_trajectories_task",
   "properties_list": {"energy": ["energy", "system", 27.211386024367243],
                       "forces": ["forces", "atomic", 51.422067090480645]}

MACE works in eV and eV/Å, so ``properties_list`` must store those units. The factors above
convert ORCA's Hartree and Hartree/Bohr; see :doc:`units`.

The ML configuration has one ``fine_tune`` block:

.. code-block:: json

   {
     "fine_tune": {
       "adapter": "alframework.ml_interfaces.mace_interface.MaceFineTuner",
       "foundation_model": "foundation/mace-mp-0-small.model",
       "n_models": 4,
       "device": "cuda",
       "E0s": "average",
       "learning_rate": 0.001,
       "max_num_epochs": 50
     }
   }

The sampler configuration loads the copies for batched MD:

.. code-block:: json

   "batched_calculator": "alframework.ml_interfaces.mace_interface.load_mace_batched_ensemble",
   "batched_calculator_options": {"dtype": "float32"},
   "driver": {"strategy": "batched", "engine": "torchsim"},
   "batching": {"batch_size": 16, "partial_policy": "timeout", "max_wait_seconds": 5.0}

ALF checks the ``fine_tune`` block at startup, before the bootstrap is labeled.

Settings
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Setting
     - Meaning
   * - ``adapter``
     - The model family's adapter class. ``alframework.ml_interfaces.mace_interface.MaceFineTuner``
       for MACE.
   * - ``foundation_model``
     - The checkpoint every round starts from. A relative path is taken from the master
       directory, as are other settings ending in ``_file``.
   * - ``n_models``
     - Copies per round, at least 2. The default is 4.
   * - ``seed``
     - Copy ``i`` uses ``seed + i``. The default is 0.
   * - ``device``
     - ``cuda`` (the default) or ``cpu``. On one node, each GPU fine-tunes one copy at a time, up
       to ``gpus_per_node`` GPUs. On CPU, copies run one after another.
   * - ``E0s``
     - Required for MACE, because it depends on the labels' level of theory. ``foundation`` keeps
       the foundation model's atomic energies, for labels at its level of theory. ``average`` and
       ``estimated`` fit new ones to the labels. A dictionary from atomic number to energy, or a
       ``.json`` file, sets them.
   * - ``learning_rate``, ``max_num_epochs``, ``batch_size``, ``valid_fraction``
     - MACE's ``lr``, ``max_num_epochs``, ``batch_size``, and ``valid_fraction``. Defaults: 0.001, 50,
       4, and 0.1.
   * - ``dtype``
     - Training precision, ``float64`` (the default) or ``float32``.
   * - ``freeze``
     - Optional. Freezes layers, as MACE's ``--freeze``.
   * - ``replay_file``
     - Optional. A local file with a sample of the foundation model's training data. With it, MACE's
       multihead fine-tuning also trains on that data, which limits forgetting outside your system.
       Without it, the copies are fine-tuned on ALF's labels alone.
   * - ``energy_dataset``, ``forces_dataset``
     - The HDF5 datasets read from each shard. Defaults: ``energy`` and ``forces``, which must match
       ``properties_list``.
   * - ``mace_options``
     - Other ``mace_run_train`` arguments, such as ``{"swa": true, "patience": 10}``. Settings ALF
       manages, such as ``lr`` or ``seed``, are refused here.

Forgetting
~~~~~~~~~~

Plain fine-tuning adapts the model to your structures, and it can lose accuracy elsewhere. If the
model must stay general, give it a ``replay_file``. MACE's default is to download replay data for
MACE-MP models, which compute nodes usually can't do, so ALF turns that off unless a local file is
configured.

What Each Round Writes
----------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Path
     - Contents
   * - ``models/model-NNNN/model-NN/mace.model``
     - A fine-tuned copy, which sampling loads.
   * - ``models/model-NNNN/model-NN/train.xyz``
     - The labels it trained on, as extended XYZ.
   * - ``models/model-NNNN/model-NN/train.log``, ``logs/``, ``results/``, ``checkpoints/``
     - MACE's own output.
   * - ``models/model-NNNN/fine_tune_lineage.json``
     - What the round started from and learned from.
   * - ``models/model-NNNN/label_sources.json``
     - The label sources of its shards; see :doc:`teacher_student`.

``fine_tune_lineage.json`` from a two-copy round on a CPU, abridged:

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

Walkthrough
-----------

``examples/mace_fine_tuning`` fine-tunes MACE-MP small on water labeled with ORCA. Run each step
from that directory.

1. Copy the checkpoint to ``foundation/mace-mp-0-small.model``.
2. Set ``QM_run_command`` in ``orca_config.json``, and the machine settings in
   ``parsl_configs.py``. The ML executor needs GPUs, and so does the sampler executor for batched
   MD.
3. Check the builder and the QM labels, then start the run:

   .. code-block:: bash

      python -m alframework --master master_config.json --test_builder
      python -m alframework --master master_config.json --test_qm
      python -m alframework --master master_config.json

4. After the bootstrap, ``models/model-0000`` holds the first fine-tuned copies, and sampling starts
   with them. Each later model directory is a new round.

The same loop was run end to end on a CPU, with EMT standing in for ORCA so that the labels differed
from MACE's predictions. Two copies fine-tuned for a few epochs on the bootstrap labels. TorchSim MD
on those copies selected new water boxes, and the second round fine-tuned from the original
checkpoint on both shards and was promoted. In a separate check on held-out water, fine-tuning on
60 EMT-labeled frames cut the energy error against EMT from 0.94 eV for the foundation model to
0.04 and 0.10 eV for the two copies, once each model's constant offset was removed.

Running Across Nodes
--------------------

``alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_multinode_task`` fine-tunes each copy as
its own app, so a round can span any number of nodes. It retries failed copies with
``member_retries``, as the HIPPYNN multi-node tasks do; see :ref:`ml-multinode-training` for the
Parsl setup.

Restarts
--------

A round that is running when the coordinator stops is lost. After a restart, the next round uses a
new model ID and starts again from the original checkpoint. See :ref:`configuration-restarts`.

Other Model Families
--------------------

A model family plugs in with an adapter class, named in ``fine_tune.adapter``. ALF constructs it
with the ``fine_tune`` settings, and it provides:

* ``validate(master_config)``, called at startup;
* ``prepare_data(shards, member_dir)``, which turns ALF's HDF5 shards into the family's training
  input;
* ``train_member(data, member_dir, seed, device, environment)``, which fine-tunes one copy. Run the
  training in a child process with ``environment``, which selects the copy's GPU;
* ``member_complete(member_dir)``, whether that copy finished;
* ``identity()``, the package name and version, recorded in the lineage.

Sampling also needs a ``batched_calculator`` loader for the copies; see :doc:`ml_interfaces`.
HIPPYNN is the next planned adapter. ``examples/fine_tuning`` shows an older, HIPPYNN-only approach;
see :doc:`../examples/fine_tuning`.
