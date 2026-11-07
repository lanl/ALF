Acquisition
===========

Acquisition decides which sampled frames go to QM labeling. It answers three
questions, in this order:

Eligibility
   Is this frame uncertain enough to label at all?

Ranking
   How do eligible frames compare with each other?

Selection
   Which eligible frames are actually sent to QM?

The settings on this page go in an ``acquisition`` block in the sampler config.
They apply to ``alframework.samplers.batched_mlmd.mlmd_trajectories_task``; see
:ref:`samplers-multi-trajectory`.

``simple_mlmd_sampling_task`` and ``simple_udd_sampling_task`` always use the
default eligibility rule and return the first eligible frame. They stop at startup
if given ``acquisition.selection``, and they ignore the rest of the block.

Eligibility
-----------

A frame is eligible when any of its eligibility terms exceeds that term's cutoff.
All signals are standard deviations across the ensemble, computed from unbiased
predictions, so UDD and well-potential forces never affect them.

.. list-table::
   :header-rows: 1
   :widths: 16 40 18 26

   * - Term
     - Signal
     - Cutoff
     - Used by default
   * - ``energy``
     - Energy standard deviation
     - ``Escut``
     - Yes
   * - ``force_mean``
     - Mean per-component force standard deviation
     - ``Fscut``
     - Yes
   * - ``force_max``
     - Largest per-component force standard deviation
     - ``3 * Fscut``
     - Yes
   * - ``force_rms``
     - Root mean square per-component force standard deviation
     - ``Fscut``
     - No
   * - ``gap``
     - Standard deviation of an energy gap between two states
     - ``gap_std_cut``
     - No. See :doc:`excited_states`.

To choose different terms, list them in ``eligibility``. Each listed term then
needs a positive cutoff:

.. code-block:: json

   {
     "Escut": 0.5,
     "Fscut": 0.3,
     "acquisition": {"eligibility": ["energy", "force_mean"]}
   }

Two settings also apply, whichever terms you choose:

* ``min_time``: frames aren't checked until this simulation time, in picoseconds.
  The default is ``0``.
* ``distcut``: a frame with two atoms closer than this distance, in Angstroms, is
  rejected and ends its trajectory, however uncertain it is. The default for
  ``mlmd_trajectories_task`` is ``1.2``.

Ranking
-------

Each eligible frame gets a score built from normalized terms, where each term's
signal is divided by its cutoff.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``strategy``
     - Score
   * - ``"uncertainty"``
     - The default. The largest of ``energy``, ``force_mean``, and ``force_max``,
       meaning the worst violation of the default thresholds.
   * - ``"summed"``
     - A weighted sum of the terms named in ``weights``. Every term with a
       positive weight needs a positive cutoff.

.. code-block:: json

   {
     "acquisition": {
       "strategy": "summed",
       "weights": {"energy": 1.0, "force_rms": 1.0}
     }
   }

``w_energy`` and ``w_force`` can replace ``weights``, as shorthand for the
``energy`` and ``force_rms`` weights. Give both, and don't combine them with
``weights``.

Ranking only orders frames that are already eligible. A high summed score never
makes an ineligible frame eligible, so a term that only appears in ``weights``
affects ranking but doesn't trigger labeling.

Equal scores are ordered by trajectory ID and then step, so results don't depend
on the order in which tasks finish.

Selection
---------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``selection.mode``
     - Behavior
   * - ``"first_trigger"``
     - The default. Each trajectory stops at its first eligible frame and that
       frame is labeled. The score is only recorded.
   * - ``"cohort_top_k"``
     - Eligible frames from a group of trajectories, called a cohort, are pooled,
       and only the ``k`` highest-scoring frames are labeled.

Use ``cohort_top_k`` when QM is expensive compared with sampling, so that QM goes
to the most uncertain frames rather than to whichever frames triggered first:

.. code-block:: json

   {
     "acquisition": {
       "trajectory_policy": "continue",
       "selection": {
         "mode": "cohort_top_k",
         "cohort_size": 50,
         "k": 16,
         "max_candidates_per_trajectory": 4
       }
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Field
     - Meaning
   * - ``selection.cohort_size``
     - Required. Number of trajectories in each cohort.
   * - ``selection.k``
     - Required. Number of frames to label from each cohort.
   * - ``selection.max_candidates_per_trajectory``
     - Optional. The most frames one trajectory can keep in the pool. Only its
       highest-scoring frames are kept. Each trajectory is always limited to at
       most ``k``.
   * - ``selection.selector``
     - Optional. ``"top_k"`` (default) picks the highest-scoring frames. ``"fps"``
       picks a geometrically varied set; see below.
   * - ``selection.group_by_selected_state``
     - Optional. ``true`` keeps a separate cohort for each selected state, so each
       state gets its own ``k`` frames rather than competing with the others. It
       requires ``"model_mode": "excited_state"``; see :doc:`excited_states`.
   * - ``trajectory_policy``
     - ``"stop"`` (default) ends a trajectory at its first eligible frame, so it
       adds at most one frame. ``"continue"`` keeps sampling so it can add more,
       and requires ``cohort_top_k``.

How cohorts run:

* **Admission:** structures join the open cohort one at a time as builders produce
  them. A cohort can be larger than ``parallel_samplers``, because members leave
  the sampler queue as they finish.
* **Model:** a cohort uses the model that was current when it opened. If a new
  model arrives, the open cohort finishes with its original model, and the next
  cohort uses the new one.
* **Finishing:** once a full cohort's trajectories have all finished, its selector
  sends ``k`` frames to QM. A trajectory finishes as ``eligible``, ``completed``,
  ``rejected``, or ``failed``.
* **Failures:** frames a trajectory contributed before ending remain in the pool.
  Frames held inside a worker that crashes are lost.
* **Grouping:** selection depends only on the frames themselves, not on how
  trajectories were grouped into tasks. Sampler ``batch_size`` and ``cohort_size``
  are independent.

Diverse selection
   Top-K can spend all of a cohort's labels on near-identical frames from one
   uncertain region. ``"selector": "fps"`` instead uses farthest-point sampling on
   each frame's sorted pair distances, so it picks frames that differ from each
   other. Every frame in the pool is already eligible, but among those, scores
   don't affect the choice.

   .. code-block:: json

      {
        "acquisition": {
          "trajectory_policy": "continue",
          "selection": {
            "mode": "cohort_top_k",
            "cohort_size": 50,
            "k": 16,
            "selector": "fps",
            "max_candidates_per_trajectory": 8
          }
        }
      }

   It requires ``max_candidates_per_trajectory``, which can be larger than ``k``.
   Each trajectory keeps that many of its highest-scoring frames for the pool, so a
   larger value gives the selector more to choose from and uses more memory. ALF
   stops with an error if a cohort's frames mix compositions or are periodic.

Checking Results
----------------

With cohorts, ALF's status output includes a ``cohorts:`` line showing the open
cohorts, how many trajectories each has admitted and finished, and how many frames
have been selected.

A selected frame's molecule ID is its trajectory ID plus its step, as in
``mol-0001-0000000042-s000000120``. Its metadata records ``cohort_id``,
``frame_id``, ``ranking_score``, and ``cohort_rank``.

Limits
------

* **First-trigger results can wait on batch-mates.** A task returns once, when its
  last trajectory finishes, so a slow trajectory can delay labeling for the others
  in its batch.
* **A cohort only finishes when it is full.** There is no timeout for admission and
  no selection from a partly filled cohort, so builders must keep producing
  structures until the cohort has ``cohort_size`` members.
* **Cohorts are held in memory.** Unfinished cohorts are lost if the ALF process
  restarts.
* **Settings are fixed at startup.** Restart ALF to change ``selection``.

Acquisition API Links
---------------------

* :doc:`Acquisition module <../api_documentation/alframework.samplers.acquisition>`
* :doc:`Cohorts module <../api_documentation/alframework.tools.cohorts>`
