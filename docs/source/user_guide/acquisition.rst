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

``simple_mlmd_sampling_task`` and ``simple_udd_sampling_task`` run one trajectory per
task and return its first eligible frame. They make the same eligibility decision,
from ``eligibility`` and ``terms`` below, and stop at startup if given
``acquisition.selection``.

Eligibility
-----------

A frame is eligible when any of its eligibility terms exceeds that term's cutoff.
The built-in terms are standard deviations across the ensemble, computed from
unbiased predictions, so UDD and well-potential forces never affect them. Terms of
your own can read uncertainty the model reports itself (see
:ref:`acquisition-own-uncertainty`).

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
   * - ``"largest"``
     - The largest of the run's ``eligibility`` terms, which suits a run whose
       terms are its own.

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

.. _acquisition-own-uncertainty:

A Model's Own Uncertainty
-------------------------

An ensemble's spread is one measure of uncertainty. A model can also report
uncertainty of its own, as one number per structure: from MC dropout, an evidential
or mean-variance head, last-layer, Laplace or Gaussian-process variance, a neural
tangent kernel, a distance in its latent space to the training data, or a committee
of different architectures.

**What the model reports.** A batched evaluator (``batched_calculator``) adds
``"uncertainty"`` to what ``predict`` returns: one dict of named numbers per structure.
From ``evaluate``, which the packed engines call, it sets ``last_uncertainty`` to the
same list. An ASE calculator (``ase_calculator``) puts the dict in
``results["alf_uncertainty"]``; an ensemble of calculators averages what its members
report. Each value is one number per structure, so reduce per-atom values first, for
example to their maximum. See :mod:`alframework.ml_interfaces.ensemble_inference`.

**Atom scores.** For diagnostics, the calculator can also report
``results["alf_atomic_uncertainty"]``, a mapping from score name to a finite,
nonnegative array with one value per atom in input order. A batched evaluator
uses ``"atomic_uncertainty"`` from ``predict`` or ``last_atomic_uncertainty``
after ``evaluate``, as a list of mappings in structure order. Supply arrays
on the host and use an empty mapping for any structure with no atom scores.
ASE ensembles average each named array over the members that report it.

A provider can expose multiple scores, for example:

.. code-block:: python

   self.results["alf_atomic_uncertainty"] = {
       "force_spread": force_spread_by_atom,
       "embedding_distance": embedding_distance_by_atom,
   }
   self.results["alf_uncertainty"] = {
       "force_spread_max": float(max(force_spread_by_atom)),
       "embedding_distance_max": float(max(embedding_distance_by_atom)),
   }

Configure each scalar reduction you want to use as an acquisition term.
For ground-state MLMD, selected frames carry the score arrays under
``atomic_uncertainty`` in their sampling metadata. With cohort selection, each
retained candidate carries its own arrays and the selected frame inherits them.
Enable the sampling metadata writer to keep these records on disk.

Try the :download:`two-source CPU replay <../../../examples/atom_resolved_uncertainty/replay.py>`
from the repository root. The two score sources trigger independently on
synthetic structures.

.. code-block:: sh

   python -m pip install -e .
   python examples/atom_resolved_uncertainty/replay.py

**What the run reads.** ``terms`` names each one, with the cutoff that normalizes it
like the built-in terms, and ``eligibility`` and the ranking can then use it:

.. code-block:: json

   {
     "acquisition": {
       "terms": {"dropout": {"uncertainty": "dropout_std", "cutoff": 0.05}},
       "eligibility": ["dropout"],
       "strategy": "largest"
     }
   }

The terms mix freely with the built-in ones, such as
``"eligibility": ["energy", "force_mean", "dropout"]``. ``Escut`` and ``Fscut`` are
needed only when a built-in term is used, as it is by default.

**One model.** A run that uses only terms of its own doesn't need an ensemble:
``n_models`` (or ``fine_tune.n_models``) can be ``1``. A single member's spread is 0,
so a run whose eligibility or ranking reads the spread stops at startup with one
member, and its sampler tasks refuse a one-member model as well.

**Records.** Each trajectory records the largest value each of these reached during
the trajectory, as ``max_uncertainty`` in its metadata and ``max_uncertainty_<name>``
in the trajectory summary. The run report adds a panel for each beside the spread
(see :doc:`monitoring`).

This covers both sampler families, with every engine. Dynamics on a modified
potential, UDD and gap-seeking, use the ensemble's spread as their bias and still
need an ensemble.

HIPPYNN energy and force NTK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional ``alframework.ml_interfaces.hippynn.ntk`` module estimates uncertainty
from the parameter sensitivities of a fitted molecular energy graph. In your
``ase_calculator`` loader, load the graph and the geometries that supplied its
energy and force training labels, fit one estimator, and return a list containing
its calculator:

.. code-block:: python

   from hippynn import custom_kernels
   from alframework.ml_interfaces.hippynn.ntk import EnergyForceNTK, NTKASECalculator

   custom_kernels.set_custom_kernels(False)
   estimator = EnergyForceNTK(energy_graph, training_atoms,
                              energy_scale=0.1, force_scale=1.0, seed=7)
   calculators = [NTKASECalculator(estimator)]

Use a graph that takes atomic numbers and positions in Å and returns the
total energy in eV as its first output of shape ``[batch, 1]``. Training and
query structures are nonperiodic ASE ``Atoms``. The energy and force scales
weight training observations in eV and eV/Å. Fit a new estimator after
retraining the model. This implementation uses CPU float64 and native PyTorch
kernels. Force scores require a parameter gradient for each Cartesian force
component, which adds cost at every calculator evaluation.

``predict_energy_forces(atoms)`` returns the predicted energy and forces
along with scores for energy, force components, and atoms. The ASE adapter
reports ``ntk_energy`` and ``ntk_force_max`` as scalar signals, and
``ntk_force`` as an array. For example, configure a force trigger with
``"terms": {"force_ntk": {"uncertainty": "ntk_force_max", "cutoff": 0.2}}``,
``"eligibility": ["force_ntk"]``, and ``"strategy": "largest"`` inside
``acquisition``. The cutoff is illustrative. Choose thresholds using independent
reference energies or forces and check them on held-out structures.

The :download:`NTK replay example <../../../examples/atom_resolved_uncertainty/ntk_replay.py>`
generates an analytic bond dataset in memory, trains HIP-NN from scratch, and
compares standalone scores with ALF replay on six held-out structures. Run it
from the repository root after installing PyTorch and HIPPYNN.

.. code-block:: sh

   python examples/atom_resolved_uncertainty/ntk_replay.py

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
* **Settings are fixed at startup.** Restart ALF to change ``acquisition``, or
  ``Escut``, ``Fscut``, or ``gap_std_cut``. If you edit them during a run, ALF warns
  once and keeps the values it started with.

Acquisition API Links
---------------------

* :doc:`Acquisition module <../api_documentation/alframework.samplers.acquisition>`
* :doc:`Cohorts module <../api_documentation/alframework.tools.cohorts>`
