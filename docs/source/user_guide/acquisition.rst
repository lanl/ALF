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

Packed providers with expensive scoring can instead define
``observe_uncertainty(numbers, positions, cell, pbc)``. ALF calls it with the
current device tensors of one structure at an acquisition check. Return a
mapping with optional ``uncertainty`` and ``atomic_uncertainty`` mappings for
that structure. Use distinct names for any scores also returned from
``evaluate``.

To score the active structures together, a packed evaluator can instead implement
``observe_uncertainty_batch(numbers, positions, cells, pbc, counts, energies, forces)``.
The inputs use the same packed geometry as ``evaluate``, with unbiased energies
and forces on the engine device. Return one observation mapping per structure in
the same order. This hook takes precedence if both observation hooks are present.

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

Local environment novelty
~~~~~~~~~~~~~~~~~~~~~~~~~

Local novelty compares each atom's element-resolved radial descriptor with
reference environments of the same element. It reports the nearest distance
as ``local_novelty`` for each atom and ``local_novelty_max`` for the frame.
It needs geometry and an ASE-readable reference file containing structures from
the potential's training data. Include every element the model will sample.
The score is dimensionless and describes radial novelty. Choose a cutoff using
held-out QM force errors for the potential in use.

For TorchSim or ALCHEMI, wrap any single-member packed potential with the
loader below. It uses TorchSim's device neighbor search and scores only at
acquisition checks. The example uses a HIPPYNN loader for the underlying
potential. Preserve any options for that loader under ``base_options``.

.. code-block:: json

   {
     "batched_calculator": "alframework.ml_interfaces.local_novelty.load_local_novelty_batched",
     "batched_calculator_options": {
       "base_loader": "alframework.ml_interfaces.hippynn.load_hippynn_batched_ensemble",
       "reference_path": "/path/to/training-reference.xyz",
       "cutoff": 5.0,
       "radial_bins": 12
     },
     "acquisition": {
       "terms": {"novel": {"uncertainty": "local_novelty_max", "cutoff": 0.2}},
       "eligibility": ["novel"],
       "strategy": "largest"
     }
   }

``batched_calculator_options.cutoff`` is the descriptor radius in Å, and the
acquisition cutoff of 0.2 is illustrative. Packed periodic cells must have three
linearly independent cell vectors. Reference comparisons run in blocks to bound
temporary GPU memory. Start with a representative reference subset, since scoring
time and reference storage grow with the number of reference environments.

For ASE, a custom ``ase_calculator`` loader can return a list with one
``LocalNoveltyCalculator(base_calculator, novelty)``. Build ``novelty`` with
``LocalEnvironmentNovelty(reference_structures, device=device)`` from
``alframework.ml_interfaces.local_novelty``. The ASE wrapper uses the same radial
descriptor and nearest-reference scoring with ASE's neighbor search.

PNNL per-atom uncertainty with MACE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ground-state ASE MD, use the optional MACE loader
``alframework.ml_interfaces.uq_mlip.load_mace_uq`` as ``ase_calculator``.
Install the tested `PNNL UQ-MLIP revision
<https://github.com/pnnl/UQ-MLIP/tree/61ae6e171db4f64382b81df34673b93edc4cee42>`_
with its ``mace`` extra in the sampler environment. Put a single MACE ASE
checkpoint at ``mace.model`` in the ALF model-generation directory or in its
single ``model-*`` member directory. Before making that generation available
to workers, fit and save the quantile regressor using representative structures:

.. code-block:: python

   from ase.io import read
   from alframework.ml_interfaces.uq_mlip import fit_mace_uq

   fit_mace_uq(model_directory, read("validation.extxyz", index=":"))

Fit a fresh regressor for each new checkpoint. Keep the fitted directory with
that checkpoint. The loader checks both files against the manifest. MACE
inference and embedding extraction run on the sampler's device. The
regressor runs on the CPU and returns one interval half-width per atom under
``uq_mlip``. This is an internal atomic-energy score. The
`interface study <https://doi.org/10.1038/s41524-026-02051-8>`_ uses a
different normalization. Calibrate a cutoff on the score you actually deploy.

To acquire frames by their highest atom score, set ``acquisition.terms`` to
``{"uq": {"uncertainty": "uq_mlip_max", "cutoff": 0.2}}``, set
``eligibility`` to ``["uq"]``, and use the ``largest`` strategy. The cutoff
is illustrative. For a region-specific trigger, a provider can keep the full
atom array and add another named scalar computed over the chosen atoms. For
example, mask the array to sodium ions in the interface region, report their
maximum under ``uq_mlip_interface_max``, and define a term for that name.

The :download:`MACE and UQ-MLIP replay example <../../../examples/atom_resolved_uncertainty/uq_mlip_replay.py>`
checks molecular and periodic structures, first-trigger MD, and a fixed-score
cohort with a local MACE checkpoint.

HIPPYNN energy and force NTK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional ``alframework.ml_interfaces.hippynn.ntk`` module estimates uncertainty
from the parameter sensitivities of a fitted molecular energy graph. In your
``ase_calculator`` loader, load the graph and the geometries that supplied its
energy and force training labels, fit one estimator, and return a list containing
its calculator:

.. code-block:: python

   from hippynn import custom_kernels
   from alframework.ml_interfaces.hippynn.ntk import (
       EnergyForceNTK, NTKASECalculator, NTKPackedEvaluator,
   )

   custom_kernels.set_custom_kernels(False)
   estimator = EnergyForceNTK(energy_graph, training_atoms,
                              energy_scale=0.1, force_scale=1.0, seed=7,
                              device=device)
   calculators = [NTKASECalculator(estimator)]

Use a graph that takes atomic numbers and positions in Å and returns the
total energy in eV as its first output of shape ``[batch, 1]``. Training and
query structures are nonperiodic ASE ``Atoms``. The energy and force scales
weight training observations in eV and eV/Å. Fit a new estimator after
retraining the model. Pass the ``device`` argument supplied by ALF to the
loader. The estimator uses native PyTorch kernels and defaults to float64 on
CPU and float32 on GPU. Use ``dtype=torch.float64`` to request double precision
on CUDA or ROCm. MPS uses float32. Recalibrate cutoffs for the deployed model
and precision.

For TorchSim or ALCHEMI, return ``NTKPackedEvaluator(estimator)`` from a
``batched_calculator`` loader. It computes ordinary batched energy and forces
at each step and NTK scores only at acquisition checks. The graph and sketch
use the estimator's precision and the engines keep their integration
precision. Use TorchSim for ROCm, TorchSim or ALCHEMI for NVIDIA CUDA, and
ASE for Apple MPS. Force scores require a parameter gradient for each
Cartesian force component. The ASE adapter computes them on each evaluation.

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
compares standalone scores with ALF replay on six held-out structures. It
also runs two short trajectories and checks the saved atom scores at the
selected geometries. Run these checks from the repository root after
installing PyTorch and HIPPYNN.

.. code-block:: sh

   python examples/atom_resolved_uncertainty/ntk_replay.py
   PYTORCH_ENABLE_MPS_FALLBACK=0 python examples/atom_resolved_uncertainty/ntk_replay.py --device mps

For native GPU qualification on Linux, use a PyTorch build for the target
hardware and install ALF's ``torchsim`` extra, plus ``alchemi`` for NVIDIA
ALCHEMI checks. PyTorch uses the device spelling ``cuda`` for both CUDA and
ROCm. Run the first two commands on NVIDIA and AMD, and the third on NVIDIA.

.. code-block:: sh

   ALF_TEST_NTK_DEVICE=cuda:0 python -m pytest tests/test_hippynn_ntk.py
   python examples/atom_resolved_uncertainty/ntk_replay.py --device cuda:0 --engine torchsim
   python examples/atom_resolved_uncertainty/ntk_replay.py --device cuda:0 --engine alchemi

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
       and a selector chooses up to ``k`` frames to label.

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
     - The most frames one trajectory can keep in the pool. Required for
       ``fps`` and ``adaptive_ntk``. With ``top_k``, a trajectory keeps at most
       ``k`` frames and the cap can reduce that limit.
   * - ``selection.selector``
     - Optional. ``"top_k"`` (default) uses fixed scores. ``"fps"`` uses geometry
       to diversify the batch. ``"adaptive_ntk"`` updates the remaining energy
       NTK scores after each choice.
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
  sends up to ``k`` frames to QM. A trajectory finishes as ``eligible``, ``completed``,
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

Adaptive energy NTK selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``adaptive_ntk`` selector uses energy gradients with respect to model
parameters, as in the `AdaptNTK acquisition method
<https://arxiv.org/html/2609.00488v1#S4.SS2>`_. The first choice has the highest
initial energy score. After each choice, the coordinator updates the precision
matrix with that geometry's feature row and recomputes the scores of the remaining
candidates. Similar rows can drop in rank when they carry information already
represented by a selected frame. Selection runs before any of the cohort's QM
labels are available.

For each finished HIPPYNN checkpoint, fit and save a basis using the exact
geometries used to train that checkpoint:

.. code-block:: python

   from hippynn import custom_kernels
   from alframework.ml_interfaces.hippynn.ntk import EnergyNTK, NTKPackedEvaluator

   custom_kernels.set_custom_kernels(False)
   EnergyNTK(energy_graph, training_atoms, seed=7, device=device).save_basis(basis_path)

Publish the checkpoint and its basis together to the ALF workers. In the
``batched_calculator`` loader, load the same energy graph, then return
``NTKPackedEvaluator(EnergyNTK.from_basis(energy_graph, basis_path, device=device))``.
The loader checks the model and sketch against the saved basis. A cohort uses
one immutable basis. The basis stores the precision used for scoring, so fit
in the precision you intend to deploy. The graph contract and precision
defaults are the same as above. To retain per-atom force diagnostics, also
pass an ``EnergyForceNTK`` for the same checkpoint, device, and dtype as
``force_diagnostics`` to the evaluator.

Use a ground-state packed sampler with TorchSim (CUDA or ROCm) or ALCHEMI
(NVIDIA CUDA). For example, if your loader is ``load_adaptive_ntk`` in
``my_project.models``:

.. code-block:: json

   {
     "driver": {"engine": "torchsim"},
     "batched_calculator": "my_project.models.load_adaptive_ntk",
     "acquisition": {
       "terms": {"check": {"uncertainty": "ntk_energy", "cutoff": 0.2}},
       "eligibility": ["check"],
       "strategy": "largest",
       "trajectory_policy": "continue",
       "selection": {
         "mode": "cohort_top_k",
         "selector": "adaptive_ntk",
         "cohort_size": 50,
         "k": 16,
         "max_candidates_per_trajectory": 8
       }
     }
   }

The cutoff is illustrative. ALF first applies the configured eligibility checks.
Workers keep up to the explicit cap per trajectory, ranked by initial energy
NTK score. The coordinator fills the budget from the retained pool. A higher
cap gives it more alternatives and uses more memory. Each selected frame's
sampling metadata includes ``cohort_candidates_pruned`` and an ``adaptive_ntk``
record with the basis ID, initial score, score at selection, and zero-based
selection order. These two scores are variances in eV². The scalar signal
``ntk_energy`` is their initial square root in eV.

The :download:`adaptive NTK replay example <../../../examples/atom_resolved_uncertainty/adaptive_ntk_replay.py>`
fits a scratch-trained HIP-NN model and compares fixed top-k with adaptive
selection on the same full geometry pool. Use ``--pool`` to replay every frame
of a nonperiodic hydrogen geometry file readable by ASE, and ``--budget`` to set
the shared label budget. With an engine specified, the example also
checks which frames two MD trajectories contribute and which frames the
coordinator sends to labeling.

.. code-block:: sh

   python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py
   PYTORCH_ENABLE_MPS_FALLBACK=0 python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py --device mps
   python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py --device cpu --engine torchsim

For native GPU review, use a PyTorch build for the target hardware and install
the relevant engine extra. Run the first two commands on NVIDIA CUDA and AMD
ROCm. Run the ALCHEMI command on NVIDIA CUDA.

.. code-block:: sh

   ALF_TEST_NTK_DEVICE=cuda:0 python -m pytest tests/test_hippynn_ntk.py tests/test_adaptive_ntk.py
   python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py --device cuda:0 --engine torchsim
   python examples/atom_resolved_uncertainty/adaptive_ntk_replay.py --device cuda:0 --engine alchemi

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
