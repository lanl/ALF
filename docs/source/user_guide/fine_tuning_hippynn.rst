Fine-Tuning HIPPYNN
===================

The source is a HIPPYNN model directory, holding ``experiment_structure.pt`` and
``best_checkpoint.pt`` as HIPPYNN writes them, or an ensemble directory of such ``model-*``
members. From an ensemble, copy ``i`` starts from member ``i``, cycling if there are fewer members
than copies. Each copy keeps the pretrained network and its weights, gets a fresh Adam optimizer,
and trains on the shards in place, without copying them. This recipe is ALF's own; see
:ref:`Whose recipe ALF runs <fine-tuning-recipes>`.

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
--------

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
---------------------------

Each ``models/model-NNNN/model-NN`` holds HIPPYNN's usual output: ``experiment_structure.pt``,
``best_checkpoint.pt``, ``best_model.pt``, ``training_log.txt``, and its metrics and plots. ALF adds
``fine_tune_args.json``, recording the copy's source, seed, shards, and settings, and
``fine_tune.log``, the copy's own output before HIPPYNN's log starts.

Walkthrough
-----------

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
:ref:`Two Ways To Start <fine-tuning-starting>`), so that run used cutoffs of zero.

See :doc:`fine_tuning` for what fine-tuning means in ALF, the common setup, and the lineage record.
