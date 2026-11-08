Fine-Tuning UMA
===============

This example fine-tunes copies of a UMA checkpoint's ``omol`` task on water clusters that ALF selects
and labels with ORCA. The files are in ``examples/uma_fine_tuning``. The fine-tuning settings and how a
round works are covered in :doc:`../user_guide/fine_tuning`.

ALF ships no model weights. UMA's weights are covered by the FAIR Chemistry License, which you accept
on Hugging Face. Fine-tuned copies are derivative works, so sharing them is subject to it, and
publications using UMA must acknowledge it; see :ref:`fine-tuning-licenses`.

How It Runs
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Stage
     - Task
   * - Builder
     - ``alframework.builders.builders.simple_condensed_phase_builder_task``, making water clusters
       (``"periodic": false``)
   * - Sampler
     - ``alframework.samplers.batched_mlmd.mlmd_trajectories_task`` on TorchSim, loading the copies
       with ``fine_tuning.uma.load_uma_batched_ensemble``
   * - QM
     - ``alframework.qm_interfaces.orca5_interface.orca_calculator_task``
   * - ML
     - ``alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_task``, with ``UmaFineTuner``

ALF first labels a bootstrap set. Each round then fine-tunes four copies, every one starting from the
original checkpoint, on all labels stored so far. The copies' disagreement during MD picks the next
structures to label.

ORCA's level of theory differs from ``omol``'s (ωB97M-V), so ``E0s`` is ``average``, which refits
UMA's per-element reference energies to the labels. ``epochs`` is 10, above fairchem's default of 1,
because early rounds have few labels; see fairchem's guide at
https://facebookresearch.github.io/fairchem/fine-tuning. Sampling treats every structure as a neutral
singlet (``charge`` 0 and ``spin`` 1 in ``mlmd_config.json``). The builder makes clusters because ORCA
labels each structure as a cluster, and ``omol`` was trained only on non-periodic structures.

Before You Run
--------------

1. On a machine with network access, accept UMA's license at https://huggingface.co/facebook/UMA,
   then download the checkpoint. Compute nodes only read the file.

   .. code-block:: bash

      hf auth login
      hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir foundation

2. Install ``fairchem-core`` and ALF's ``torchsim`` extra in the ML and sampler worker environments.
3. Set ``QM_run_command`` in ``orca_config.json``, and the machine in ``parsl_configs.py``.

Running
-------

From ``examples/uma_fine_tuning``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json

Each round writes its copies to the next ``models/model-NNNN/``, with ``fine_tune_lineage.json``
recording the checkpoint's SHA-256, the settings, the shards, and each copy's seed. Labels go to
``h5store/``.

See :doc:`mace_fine_tuning` for the same workflow with MACE, and :doc:`fine_tuning` for HIPPYNN.
