Fine-Tuning MACE
================

Install ``mace-torch`` in the ML workers' environment, and ``mace-torch`` with ALF's ``torchsim``
extra in the sampler workers' environment. The source is a MACE checkpoint file; see
:ref:`Pretrained Models And Licenses <fine-tuning-licenses>`. ALF runs MACE's own ``mace_run_train`` for each copy, with
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
       multihead fine-tuning; see :ref:`Forgetting And Checking A Fine-Tuned Model <fine-tuning-forgetting>`.
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

Walkthrough
-----------

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

See :doc:`fine_tuning` for what fine-tuning means in ALF, the common setup, and the lineage record.
