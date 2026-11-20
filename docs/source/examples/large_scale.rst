Running On Many Nodes
=====================

A template for an ALF run on many nodes, with every setting that :doc:`../user_guide/running_at_scale` measured to
matter at that size. The chemistry is the water cluster of :doc:`simple_water`, labeled by a foundation-model
teacher, so the example is about the workflow; replace the builder, sampler, labeler and trainer with your own and
the scaling settings carry over. The files are in ``examples/large_scale``.

How It Runs
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - Task and settings
   * - Builder
     - ``simple_multi_condensed_phase_builder_task``, 256 structures per task (``maximum_builder_structures``), so
       one builder task feeds one sampler task.
   * - Sampler
     - ``mlmd_trajectories_task`` with the ``torchsim`` engine, 256 trajectories per task on one GPU, returning
       compact metadata without momenta and writing one metadata file per task.
   * - QM
     - ``teacher_batch_task`` with a UMA checkpoint, labeling up to 256 structures per call (native batching).
   * - ML
     - ``train_HIPPYNN_ensemble_multinode_task``: eight members, each trained on one node's four GPUs, with shared
       training arrays and a streaming final test.

``master_config.json`` gathers the settings that change how ALF runs at scale in one ``scaling`` block (see
:ref:`running-at-scale-settings`), and ``maximum_stored_labels`` makes the run finite. ``parsl_configs.py`` gives
each role one multi-node block, with one Parsl worker pool per node, and reads the machine from environment
variables: the partition and account, GPUs and cores per node, nodes per role, and the workers' node environment
(a packed environment, node-local caches and ``ALF_THP_DISABLE``).

Sizing
------

Keep the structures in flight equal to the sampling capacity: ``parallel_samplers`` is the sampler nodes times the
GPUs per node times the sampler batch size, 16,384 for 16 sampler nodes of 4 GPUs. ``target_queued_QM`` and
``save_h5_threshold`` at the same number retrain after every wave of sampling.

Running
-------

Download the teacher checkpoint named in ``qm_config.json``, set the environment variables that the README in
``examples/large_scale`` lists, and start the coordinator on a login node:

.. code-block:: bash

   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json

The coordinator prints the scaling settings it applied. ``records/`` holds the trajectory summary and the
coordinator's memory and loop timing, and ``run_report/`` the figures made from them (see :doc:`../user_guide/monitoring`).
To hand sampled frames to QM through a shared filesystem instead of through the coordinator, add
``"data_by_reference": {"directory": "sampling-frames", "sampler_frames": true}`` to the ``scaling`` block.
