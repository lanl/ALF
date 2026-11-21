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

``status_path`` in the master configuration, ``status.txt`` in most examples, is rewritten every loop.
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

The scalar and multi-trajectory MLMD samplers also record the initial and final
kinetic temperatures and the last thermostat target as
``initial_kinetic_temperature_K``, ``kinetic_temperature_K`` and
``thermostat_temperature_K``. Compare these when short trajectories stop before
warming up; the target is not a measurement of the actual kinetic temperature.
See :ref:`samplers-initial-velocities` for optional thermal starts.

Labels And Data
---------------

Stored labels are in the HDF5 shards under ``h5_dir``, one shard per retraining. Finished labels that
aren't stored yet wait in ``held_labels_dir``, so a crash doesn't lose them. QM tasks write each
structure's input, output, and scratch files under ``QM_scratch_dir``; look there when labels fail.

Training
--------

Each training writes an ensemble directory under ``model_path``, with one ``model-NN`` directory per
member. In each, ``training_log.txt`` has the epoch-by-epoch losses, and the final test errors, which are
the best quick check of model quality. HIPPYNN members also write ``training_summary.json`` with the same
numbers for reading by a program: ``final_metric`` (each split's metrics for the best model, from the final
test), ``metric`` (the best value of each metric), ``epoch_metric`` (each epoch's metrics), ``epoch_seconds``,
``epochs`` and ``split_counts``. Multi-node trainings also write a ``placement-*.json`` record per
member attempt: host, GPU, and start and end times. See :ref:`ml-multinode-training`.

Task Timeline
-------------

To measure how long each stage takes, set ``ALF_TASK_TIMELINE_DIR`` to a directory on shared storage in the
workers' environment (``worker_init``). ALF's builder, sampler, QM, and ML tasks then each append one JSON
line as they end, to ``<host>-<pid>.jsonl`` in that directory, so no two workers write to the same file.
Each line has ``stage``, ``task``, ``items`` (the number of structures), ``start`` and ``end`` (Unix
seconds), ``seconds``, ``outcome`` (``ok``, or the exception's class name), ``host``, ``pid``, the
Slurm job and Parsl block, pool and rank, and ``gpu_peak_gb``: the most GPU memory PyTorch had allocated
during the task, on the device that used the most, or ``null`` in a worker that doesn't use a GPU through
PyTorch. A task whose peak stands out from the others' shows what grew before a GPU ran out of memory. A stage's time is then its first task's start to its last
task's end, without the time a block spends idle. Tasks in your own modules are recorded if they are
decorated with ``alframework.tools.task_timeline.timed(stage)`` under ``@python_app``. With the variable
unset, nothing is written, and a line that can't be written is skipped without failing the task.

A task can also time its own parts with ``with alframework.tools.task_timeline.phase(name):``. Its line then
has ``phases``, seconds per part, and ``phase_imports``, the modules each part imported, which shows a part
spending its time on first imports. QM tasks that read sampled frames by reference (``data_by_reference``)
record ``read_frames``, ``load_labeler`` and ``label``, and ``write_labels`` when they also write their labels by
reference.

.. _monitoring-coordinator-loop:

Coordinator Loop
----------------

At scale the coordinator itself can be what a run waits for: it submits every task, collects every result
and writes the held labels, one pass of its loop after another. To see where its time goes, set
``coordinator_loop_path`` in the master configuration, such as ``"coordinator_loop.csv"``. Each pass then
appends one CSV row: its ``loop`` (``bootstrap`` or ``active``), ``start`` and ``seconds`` without its
sleep, the seconds in each phase (reloading the configuration files, submitting and collecting builders,
submitting and collecting sampling, submitting QM, holding finished labels, storing a shard, collecting
training results, progress plots, printing and writing status, checking the executors, and retiring
by-reference files), the ``sleep`` that followed, the QM results it held, and how much work was in flight
at its end. It is off by default, adds a few timer calls per pass when on, and a row that can't be written
stops the timing without stopping the run. A restarted coordinator appends to the same file.

When passes get longer and their sleeps no longer leave the loop idle, the coordinator limits the run, and
the busiest phase is where to look.

.. _monitoring-run-report:

Run Report
----------

A run report gathers these records into figures and one JSON summary. Make one for any run, finished or
still running, from its directory:

.. code-block:: bash

   python -m alframework.tools.run_report RUN_DIR [--out DIR] [--master master_config.json] [--timeline DIR]

It writes to ``RUN_DIR/run_report`` by default. To have the coordinator refresh it as the run goes, set
``"run_report": true`` in the master configuration (and ``run_report_dir`` to write elsewhere): each
progress-plot update (``update_plots_every``) then also remakes the report, in its own process, and skips an
update while the previous one still runs. It needs no plotting utility.

A *cycle* below is the model the run had when a task started, the same number as a trajectory's
``model``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Shows
   * - ``run-report-stages.png``
     - Tasks running in each stage over the run, with the waits for Parsl blocks shaded (from a block's
       submission to its stage's next task: the scheduler's queue and the block's startup) and a line where
       each model finished.
   * - ``run-report-task-times.png``
     - Each stage's task times per cycle, shortest to longest. A long top whisker is a straggler the whole
       stage waited for.
   * - ``run-report-spread.png``
     - The ensemble's energy, force and (excited states) gap spread per cycle, using each trajectory's peak:
       the median and 90th percentile, or one line per selected state when a run has several, and a panel for each
       uncertainty the model reports of its own (:ref:`acquisition-own-uncertainty`). Then the share
       of trajectories that gave QM a candidate, and the sampling GPU time per candidate, from the cycle's
       sampler tasks in the task timeline. All should fall as the models improve.
   * - ``run-report-labels.png``
     - Labels stored for each training round, and in total.
   * - ``run-report-training.png``
     - The last epoch's time and the final test errors per training round, one point per member, or for
       excited states, each state's energy and force errors as one line per state.
   * - ``run-report-health.png``
     - Each stage's peak GPU memory per task, the coordinator's memory over the run, and failed tasks.
   * - ``run-report-loop.png``
     - With ``coordinator_loop_path``: the share of the coordinator's loop time each phase took over the
       run, waiting included, and each pass's time without waiting.
   * - ``run_report.json``
     - The numbers behind the figures, plus each stage's task count, failures and time percentiles, the
       gaps between stages in each cycle, the block waits, and hosts whose median task time is at least
       1.5 times their stage's.

It reads the task timeline (``task_timeline_dir`` in the master configuration, else
``ALF_TASK_TIMELINE_DIR``, else ``RUN_DIR/task_timeline``), Parsl's ``runinfo`` submit scripts, the
trajectory summary, the label shards (``h5_path``), the members' ``training_summary.json`` and, with
``ALF_LIGHTNING_STATS=1``, their per-rank ``lightning_stats``, the coordinator's memory samples
(``coordinator_memory_sample``), and its loop timing (``coordinator_loop_path``). A figure whose records are missing is skipped, and ``run_report.json`` lists
it under ``skipped``.

Parsl
-----

Parsl's own logs are under the ``runinfo`` directory of the run, one directory per coordinator start. A
block that fails to start, a worker that dies, or a scheduler error shows up there first. See
:doc:`parsl`.
