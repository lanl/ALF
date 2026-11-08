Fine-Tuning A Pretrained HIPPYNN Ensemble
=========================================

This example starts ALF from a HIPPYNN ensemble you have already trained, and fine-tunes it on
structures that ALF selects and labels with ORCA. The files are in ``examples/fine_tuning``. The
fine-tuning settings and how a round works are covered in :doc:`../user_guide/fine_tuning`.

You supply the pretrained ensemble; ALF ships no model weights. HIPPYNN has no official
fine-tuning recipe, so the one here is ALF's own; see :ref:`fine-tuning-licenses` and the guide's
"Whose recipe ALF runs". HIPPYNN checkpoints load through PyTorch, which can run code stored in the
file, so use only files you trust.

If you want to start from your own structures or labels, without model weights, see
:doc:`seeded_active_learning`.
For foundation models, see :doc:`mace_fine_tuning` and :doc:`uma_fine_tuning`.

The checked-in configuration is deliberately small for bring-up: four copies, at most 200 epochs
per round, and queues of 25 structures. Increase these only after the stage checks work on your
machine.

How It Runs
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Stage
     - Task
   * - Builder
     - ``alframework.builders.builders.simple_cfg_loader_task``, reading ``fragment_library/``
   * - Sampler
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``, loading the current ensemble
       with ``HIPNN_ASE_load_ensemble``
   * - QM
     - ``alframework.qm_interfaces.orca5_interface.orca_calculator_task``
   * - ML
     - ``alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_task``, with the HIPPYNN adapter

The pretrained ensemble goes in model slot 0, ``models/model-0000``. With no status file yet, ALF
finds it at startup, skips the bootstrap, and samples with it straight away. ``hippynn_config.json``
names the same directory as the source:

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

Each round fine-tunes four copies, copy ``i`` from member ``i`` of the pretrained ensemble, on every
label stored so far, and writes them to the next model slot. Every round starts again from the
pretrained ensemble. ``E0s`` is ``foundation`` because the labels and the pretrained model are
assumed to share a level of theory; use ``average`` if they don't.

Before You Run
--------------

1. **The pretrained ensemble.** Copy each member to ``models/model-0000/model-00``, ``model-01``,
   and so on. Each needs ``experiment_structure.pt`` and ``best_checkpoint.pt``, which fine-tuning
   starts from, and ``best_model.pt``, which the sampler loads.

   .. code-block:: text

      models/model-0000/model-00/experiment_structure.pt
      models/model-0000/model-00/best_checkpoint.pt
      models/model-0000/model-00/best_model.pt
      models/model-0000/model-01/...

   Don't create ``status.txt``; ALF creates it on the first run.

2. **A single pretrained model instead.** Copies of one model never disagree, so ALF couldn't select
   anything with them. Put the model in its own directory, point ``foundation_model`` at it, and
   leave ``models/`` empty. ALF then labels a bootstrap set first, and the first round's copies,
   fine-tuned with different seeds and splits, disagree from then on.

3. **Earlier labels.** ALF shards in ``h5store/``, such as ``data-0000.h5``, are part of every
   round's training data. They must be in eV and eV/Å, like the pretrained model.

4. **Settings.** Put starting structures in ``fragment_library/`` as ASE-readable ``.cfg`` files.
   Set ``QM_run_command`` in ``orca_config.json``, and the machine in ``parsl_configs.py``.

Running
-------

From ``examples/fine_tuning``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_sampler
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json

``--test_sampler`` needs the pretrained ensemble in place. To fine-tune once on the labels in
``h5store/``, without sampling or labeling, run ``--test_ml``; it writes ``models/model-0001``.

Outputs
-------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Contents
   * - ``models/model-0001/model-NN/``
     - A fine-tuned copy in HIPPYNN's usual layout, with ``training_log.txt`` and
       ``fine_tune_args.json``.
   * - ``models/model-0001/fine_tune_lineage.json``
     - The source files' SHA-256 values, the settings, the shards, and each copy's seed.
   * - ``h5store/data-*.h5``
     - Labels from ORCA, and any you provided.
   * - ``status.txt``
     - Progress and restart state.
   * - ``sampling/``
     - Sampler metadata and uncertainty diagnostics.

Check a round's ``training_log.txt`` files for ``Training complete``. A copy that fails stops that
round from replacing the current model, and the next store starts a new round.
