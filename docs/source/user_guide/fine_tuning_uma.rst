Fine-Tuning UMA
===============

Install ``fairchem-core`` in the ML workers' environment, and ``fairchem-core`` with ALF's ``torchsim``
extra in the sampler workers' environment. The source is a UMA checkpoint file; see
:ref:`Pretrained Models And Licenses <fine-tuning-licenses>`. Compute nodes read the local file and never download.

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

As for the UMA teacher, fairchem-core 2.21 computes predictions in float32. ``dtype`` controls the
copies' input and output tensor precision; selecting ``float64`` does not increase prediction precision
(see :doc:`teacher_student`).

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
------------------

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
---------------

``omol`` needs each structure's total charge and spin multiplicity. When a structure's metadata has
``charge`` or ``mult``, as the teacher and some builders set them, ALF stores them in the shard as
``total_charge`` and ``spin_multiplicity``, and fine-tuning reads them; structures without them are
neutral singlets. ALF's batched samplers don't pass charge or spin per structure, so the sampler's
``charge`` and ``spin`` apply to every structure sampled in a run. They default to 0 and 1.

What a UMA round writes
-----------------------

Each ``models/model-NNNN/model-NN`` holds ``uma.pt``, the fine-tuned checkpoint, which sampling
loads; ``fine_tune.yaml``, the training configuration ALF wrote; ``data/``, the copy's training and
validation sets; ``train.log``; and fairchem's logs under ``runs/``.

Each copy holds a full UMA model while sampling, so several copies of a large checkpoint need GPU
memory to match.

Walkthrough
-----------

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

See :doc:`fine_tuning` for what fine-tuning means in ALF, the common setup, and the lineage record.
