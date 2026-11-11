Glossary
========

Terms used across this guide, with where to read more.

Acquisition
   Deciding which sampled frames to label: which uncertainty signals count, how frames are ranked, and
   what a trajectory does after one is chosen. See :doc:`acquisition`.

Block
   One Parsl scheduler allocation, such as one Slurm job. An executor's ``nodes_per_block`` and
   ``max_blocks`` bound how many nodes it may request. See :doc:`parsl`.

Bootstrap
   The first set of structures, labeled before any model exists, which the first ensemble trains on.
   Its size is ``bootstrap_set``; a finite bootstrap labels a fixed seed list. See :doc:`configuration`.

Cohort
   A group of trajectories whose candidate frames are pooled and ranked together, so the best few are
   labeled rather than the first few. See :doc:`acquisition`.

Coordinator
   The ``python -m alframework`` process. It decides what to run next, submits Parsl tasks, stores
   labels, and writes the status file. It runs no heavy computation itself.

Ensemble and member
   ALF trains several models, the members of an ensemble, on the same data with different seeds. Their
   disagreement is the uncertainty used to choose frames.

Executor and worker
   A Parsl executor runs one kind of ALF task, such as ``alf_QM_executor``. Each of its blocks starts
   workers, which run one task at a time. See :doc:`parsl`.

Gap-seeking
   Excited-state sampling that biases trajectories toward small gaps between electronic states, where
   models are least reliable. See :doc:`excited_states`.

Held labels
   Finished labels waiting to be stored in the next shard, kept on disk so a crash doesn't lose them.

Label budget
   ``maximum_stored_labels``: the run stops once this many converged labels are stored. See
   :doc:`configuration`.

Model ID and training ID
   Every training gets a new training ID. The model ID is the training whose ensemble samplers use now;
   it changes only when a training succeeds.

Sampling window
   A range of stored-label counts with its own sampler settings, so a run can change sampling method as
   the dataset grows. See :doc:`samplers`.

Shard
   One HDF5 file of stored labels, ``h5_path`` with the shard number. Every retraining stores one, and
   each training reads them all.

Trajectory policy
   What a trajectory does after a frame is chosen: ``continue`` keeps sampling, ``stop`` ends it. See
   :doc:`acquisition`.

UDD
   Uncertainty-driven dynamics: a bias that pushes trajectories toward configurations where the
   ensemble disagrees. See :doc:`samplers`.
