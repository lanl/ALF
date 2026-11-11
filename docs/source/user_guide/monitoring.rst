Monitoring A Run
================

An ALF run is a long-lived coordinator process and the Parsl tasks it starts. This page lists where to
look while it runs, and what each source tells you.

The Coordinator's Output
------------------------

Every loop, about once a minute, the coordinator prints the state of each task queue: builders,
samplers, QM, and ML, with counts of queued, running, and finished tasks. Lines starting
``Training 0003:`` report a training's members as they finish. When the run stops at its label budget,
it prints what it stored and what it dropped. Keep this output, for example by redirecting it to a file
or running the coordinator under ``nohup``.

The Status File
---------------

``status_path`` in the master configuration, ``status.json`` in most examples, is rewritten every loop.
It holds the current model, training, shard, and molecule IDs, the number of labels stored so far
(``lifetime_stored_labels``), and counts of failed tasks and structures by stage. A restart reads it,
so don't edit it while the coordinator runs. See :ref:`configuration-restarts`.

Rising failure counts are the first sign of trouble: ``lifetime_failed_QM_structures`` for labels that
didn't converge, and ``lifetime_failed_sampler_structures`` for trajectories that ended in error.

Sampling
--------

With ``trajectory_summary_path`` set in the master configuration, the coordinator appends one row per
trajectory: how it ended, when, its peak uncertainties, and its electronic state. This is the file to
tune uncertainty cutoffs from: if almost every trajectory triggers at once, the cutoffs are too low for
the current model; if most run to the end without triggering, QM runs short of work. The progress plots
in :ref:`samplers-progress-plots` are drawn from the same data.

Labels And Data
---------------

Stored labels are in the HDF5 shards under ``h5_dir``, one shard per retraining. Finished labels that
aren't stored yet wait in ``held_labels_dir``, so a crash doesn't lose them. QM tasks write each
structure's input, output, and scratch files under ``QM_scratch_dir``; look there when labels fail.

Training
--------

Each training writes an ensemble directory under ``model_path``, with one ``model-NN`` directory per
member. In each, ``training_log.txt`` has the epoch-by-epoch losses, and the final test errors, which are
the best quick check of model quality. Multi-node trainings also write a ``placement-*.json`` record per
member attempt: host, GPU, and start and end times. See :ref:`ml-multinode-training`.

Parsl
-----

Parsl's own logs are under the ``runinfo`` directory of the run, one directory per coordinator start. A
block that fails to start, a worker that dies, or a scheduler error shows up there first. See
:doc:`parsl`.
