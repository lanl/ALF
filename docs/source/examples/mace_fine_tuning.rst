Fine-Tuning MACE
================

This example fine-tunes copies of a MACE foundation model on water clusters that ALF selects and
labels with ORCA. The files are in ``examples/mace_fine_tuning``. The fine-tuning settings and how a
round works are covered in :doc:`../user_guide/fine_tuning`.

ALF ships no model weights. You supply the checkpoint, and its license applies: MACE-MP-0 is MIT
licensed, while several other MACE foundation models use the Academic Software License; see
:ref:`fine-tuning-licenses`.

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
       with ``fine_tuning.mace.load_mace_batched_ensemble``
   * - QM
     - ``alframework.qm_interfaces.orca5_interface.orca_calculator_task``
   * - ML
     - ``alframework.ml_interfaces.fine_tuning.fine_tune_ensemble_task``, with ``MaceFineTuner``

ALF first labels a bootstrap set of 100 structures. Each round then fine-tunes four copies, every one
starting from the original checkpoint, on all labels stored so far. The copies' disagreement during MD
picks the next structures to label.

Settings in ``fine_tune_config.json`` follow MACE's naive fine-tuning recipe
(https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html). ``E0s`` is ``estimated``, as that
recipe recommends when isolated-atom energies aren't available; use ``foundation`` only when the labels
share the foundation model's level of theory. ORCA gives Hartree, so ``properties_list`` converts to
eV and eV/Å, as MACE expects. The builder makes clusters because ORCA labels each structure as a
cluster, and the fine-tuned model must learn the same thing.

Before You Run
--------------

1. Download a MACE checkpoint to ``foundation/mace-mp-0-small.model``. Compute nodes often have no
   network.
2. Install ``mace-torch`` and ALF's ``torchsim`` extra in the ML and sampler worker environments.
3. Set ``QM_run_command`` in ``orca_config.json``, and the machine in ``parsl_configs.py``.

Running
-------

From ``examples/mace_fine_tuning``:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm
   python -m alframework --master master_config.json

Each round writes its copies to the next ``models/model-NNNN/``, with ``fine_tune_lineage.json``
recording the checkpoint's SHA-256, the settings, the shards, and each copy's seed. Labels go to
``h5store/``.

See :doc:`uma_fine_tuning` for the same workflow with UMA, and :doc:`fine_tuning` for HIPPYNN.
