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
  ``mlmd_trajectories_task`` is ``1.2``. To label such frames instead, set
  ``acquisition.distcut_frames`` (see Selection below).

Selecting Atoms for Force Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``acquisition.force_atoms`` when a subset of atoms should drive force-based
acquisition. For a common atom order, give zero-based indices. This example
uses the maximum per-atom ensemble disagreement over atoms 12 and 47.

.. code-block:: json

   {
     "acquisition": {
       "force_atoms": {"indices": [12, 47]},
       "terms": {
         "site_force": {"uncertainty": "ensemble_force_selected_max", "cutoff": 0.20}
       },
       "eligibility": ["site_force"],
       "strategy": "largest"
     }
   }

When the selected atoms vary by structure, supply a one-dimensional boolean
ASE atom array with one entry per atom. For an array named ``active_site``, use
``"force_atoms": {"array": "active_site"}``. Each structure must select at least
one atom. The ensemble score for one atom is the norm of the three force
component standard deviations across the unbiased member predictions.
The cutoff above is illustrative and has units of eV/Å.

For NTK, use ``ntk_force_selected_max`` with the same selector and an exact atom
score provider. The prepared HIPPYNN sampler has a mode that computes only
the selected atoms' sensitivity rows. Both methods evaluate the full structure
for MD and QM labeling. The existing whole-frame force and energy terms retain
their definitions. For selected frames, ``force_atom_indices`` lists the resolved
indices, and ``selected_force_uncertainty`` maps each selected-force signal name to
its per-atom scores in that order.

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

Energy, force, and Hessian NTK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional ``alframework.ml_interfaces.ntk`` package estimates the uncertainty of one frozen
potential. It linearizes the model in its weights and fits a posterior on the training geometries. The
score of any quantity that is linear in the energies is its posterior standard deviation. Those
quantities are energies, forces, Hessian-vector products, and energy gaps. Adapters connect models:

- ``alframework.ml_interfaces.hippynn.ntk`` takes HIPPYNN graphs, molecular or periodic (with a
  ``CellNode`` input), with one ``[batch, 1]`` energy output per state.
- ``alframework.ml_interfaces.ntk.mace`` takes MACE models, molecular or periodic. It needs
  ``mace-torch``.
- ``alframework.ml_interfaces.ntk.xmace`` takes `X-MACE <https://github.com/rhyan10/X-MACE>`_
  excited-state models (``ExcitedMACE``, ``AutoencoderExcitedMACE``, ``LEHExcitedMACE``), including
  models fine-tuned from a ground-state foundation backbone. X-MACE installs as the package ``mace`` in
  place of mace-torch, so use a separate environment.

The posterior covers the model's trainable parameters. For a model fine-tuned with frozen layers or LoRA,
the frozen parameters stay out, and the training geometries are the fine-tuning data. The MACE and X-MACE
adapters also take ``parameters="all"``, or a function of each parameter's name.

In your ``ase_calculator`` loader, load the model and the geometries that supplied its training labels,
fit one estimator, and return a list containing its calculator:

.. code-block:: python

   from alframework.ml_interfaces.hippynn.ntk import (
       EnergyForceNTK, NTKASECalculator, NTKPackedEvaluator,
   )

   estimator = EnergyForceNTK(energy_graph, training_atoms,
                              energy_scale=0.1, force_scale=1.0, seed=7,
                              device=device)
   calculators = [NTKASECalculator(estimator, hessian_probes=4)]

For MACE, use ``NTKEstimator(MACENTKAdapter.from_checkpoint(path, device=device), training_atoms, ...)``
from ``alframework.ml_interfaces.ntk``. Each training geometry adds the rows its loss supervised, each
divided by its scale. The scales are an energy scale in eV and an optional force scale in eV/Å. An
optional ``hessian_scale`` in eV/Å² applies to geometries trained on Hessians:
``atoms.info["hessian_probes"]`` (``[k, N, 3]``) holds the fixed probes of Projected Hessian Learning,
and ``atoms.info["hessian"]`` marks a stored full Hessian. Use the model's validation RMSE of each
quantity as its scale.

Fit a new estimator after retraining the model. Pass the ``device`` argument supplied by ALF to the loader.
The estimator defaults to float64 on CPU and float32 on GPU. Use ``dtype=torch.float64`` to request
double precision on CUDA or ROCm; MPS uses float32. HIPPYNN scoring switches to plain-PyTorch
message-passing kernels and then restores the active ones, so MD forces can keep using Triton.
Recalibrate cutoffs for the deployed model and precision.

For TorchSim or ALCHEMI, return ``NTKPackedEvaluator(estimator)`` from a ``batched_calculator``
loader. It computes ordinary batched energies and forces at each step, and NTK scores only at
acquisition checks. The graph and sketch use the estimator's precision; the engines keep their
integration precision. Use TorchSim for ROCm, TorchSim or ALCHEMI for NVIDIA CUDA, and ASE for Apple MPS.

The calculator and the packed evaluator take the scores to compute:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Option
     - Reports
     - Meaning and cost
   * - always
     - ``ntk_energy``
     - Energy standard deviation in eV. Periodic cells also report ``ntk_energy_per_atom``.
   * - ``forces=True`` (default)
     - ``ntk_force_max``, ``ntk_force_rms``, ``ntk_force`` (per atom)
     - Exact scores from every force component's row: 3N rows.
   * - ``force_probes=k``
     - ``ntk_force_rms``
     - The RMS force-component uncertainty from k random projections wᵀF, using E‖δF‖² = E[(wᵀδF)²]: k rows.
   * - ``hessian_probes=k``
     - ``ntk_hessian``
     - The RMS Hessian-element uncertainty in eV/Å² from k projections uᵀHv with independent Gaussian u
       and v, using E[(uᵀδHv)²] = ‖δH‖_F²: k rows. The full Hessian is never formed.
   * - ``hvp_probes=k``
     - ``ntk_hvp`` (per atom)
     - The per-atom Hessian uncertainty from k Hessian-vector products: 3N rows per probe.

A model with several states adds the state name to each score (``ntk_energy_E1``, ``ntk_hessian_E1``).
It reports adjacent gaps by default. To score specific pairs, give ``gap_pairs`` to ``predict`` or in
``batched_calculator_options.scores``. For example, ``{"gap_pairs": [[1, 2], [1, 3]], "forces": false}``
reports ``ntk_dE12`` and ``ntk_dE13`` for a four-state model. The ``ntk_near_crossing_E1`` flag still uses
adjacent gaps. It is 1.0 when a predicted gap
next to that state is below ``crossing_gap_eV``, an option of the calculator and evaluator (default
0.1 eV). Adiabatic energies are not smooth where states cross. Their Hessians and Hessian scores grow
like 1/gap there, so don't use per-state Hessian scores inside that window. The gap scores and the scores
of smooth model outputs stay meaningful through a crossing. Request them with ``outputs=(...)``; each is
reported as ``ntk_<name>_max``, and elementwise for short outputs. For X-MACE they are:

- ``ground_energy``, the backbone surface;
- ``state_offsets``, each state relative to the backbone;
- ``latent`` and ``state_matrix``: for ``AutoencoderExcitedMACE``, the permutation-invariant vector and
  the tridiagonal matrix it diagonalizes, in eV;
- ``smooth_nacs``: couplings scaled by the gap, at 3N rows per state pair.

``predict(atoms, **options)`` returns the same scores as tensors, and
``predict_energy_forces(atoms)`` returns the force-component scores. For example, configure a force
trigger with ``"terms": {"force_ntk": {"uncertainty": "ntk_force_max", "cutoff": 0.2}}``,
``"eligibility": ["force_ntk"]`` and ``"strategy": "largest"`` inside ``acquisition``. The cutoff is
illustrative. Choose thresholds using independent reference energies, forces, or Hessians, and check them
on held-out structures.

Rows come from forward-mode differentiation (``torch.func``), which costs about one evaluation per sketch
dimension whatever the atom count, or from batched reverse mode, which costs about one backward pass per
row. HIPPYNN uses forward mode when it has many rows to compute. MACE uses reverse mode, because MACE's
forward pass can't run inside ``torch.func``.

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

   ALF_TEST_NTK_DEVICE=cuda:0 python -m pytest tests/test_ntk.py -k hippynn
   python examples/atom_resolved_uncertainty/ntk_replay.py --device cuda:0 --engine torchsim
   python examples/atom_resolved_uncertainty/ntk_replay.py --device cuda:0 --engine alchemi

.. _prepared-ground-state-hippynn-force-ntk:

Prepared ground-state HIPPYNN force NTK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For packed ground-state HIPPYNN, set ``record_ntk_training_split`` to ``true`` in
the ML configuration before training. Each member records its training rows
after filtering and splitting. Prepare the basis once after training, using
energy and force scales in eV and eV/Å chosen from validation data. The numbers
below are illustrative.

.. code-block:: sh

   python -m alframework.ml_interfaces.hippynn.ntk_basis prepare models/model-0000 \
       --member 0 --device cuda:0 --energy-scale 0.05 --force-scale 0.10

Preparation writes ``ntk_basis.npz`` beside that member's checkpoint and
``ntk_training_split.json``. Keep both files with the checkpoint. The
default device is CPU, sketch width is 64, and seed is 0. Use the target GPU when
preparing a basis in GPU precision. If the training shards moved, specify their
paths in manifest order with ``--shards`` and keep any backing parts alongside
them in their original relative layout. Use ``--basis-file`` to prepare a separate
artifact for different fit settings.

Configure the sampler with an illustrative cutoff for the projected score.

.. code-block:: json

   {
     "driver": {"strategy": "batched", "engine": "torchsim"},
     "batching": {"batch_size": 8},
     "batched_calculator": "alframework.ml_interfaces.hippynn.ntk_basis.load_hippynn_ground_state_ntk",
     "batched_calculator_options": {"member": 0, "force_score": "projected", "force_probes": 8},
     "acquisition": {
       "terms": {"force_ntk": {"uncertainty": "ntk_force_rms", "cutoff": 0.05}},
       "eligibility": ["force_ntk"],
       "strategy": "largest"
     }
   }

The same loader works with ALCHEMI. One chosen member evaluates the physical
energies and forces during packed MD. At acquisition checks, ``projected`` uses
eight force probes by default and reports only frame-level RMS uncertainty. For
exact atom scores, set ``force_score`` to ``"exact"`` and use ``ntk_force_max``
as the acquisition signal. Both modes use the same basis. Replay can load that
basis directly with ``load_hippynn_ground_state_ntk_estimator`` from the same
module. Loading verifies the checkpoint and split without reopening label shards
or fitting in a worker. Prepare a basis for each new checkpoint before sampling.
To compare derivative performance on the same basis, set ``ntk_engine`` in
``batched_calculator_options`` to ``"reverse"`` or ``"forward"``. The default
``"auto"`` chooses the differentiation backend from the number of requested rows.

To combine exact selected-atom forces and total-energy uncertainty, use the
same prepared basis and change the calculator options and acquisition block.

.. code-block:: json

   {
     "batched_calculator_options": {"member": 0, "force_score": "selected", "energy_score": true},
     "acquisition": {
       "force_atoms": {"indices": [12, 47]},
       "terms": {
         "site_force": {"uncertainty": "ntk_force_selected_max", "cutoff": 0.20},
         "total_energy": {"uncertainty": "ntk_energy", "cutoff": 0.10}
       },
       "eligibility": ["site_force", "total_energy"],
       "strategy": "largest"
     }
   }

At each check for this example, two selected atoms and the total energy need
seven sensitivity rows. Either term can trigger labeling. The energy cutoff
is in eV, and periodic frames also report ``ntk_energy_per_atom`` when energy
scoring is enabled. Use ``energy_score`` with projected or exact force modes as
well. It defaults to false. Projected Hessian scoring is available through the
general NTK evaluator above.

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

Use ``acquisition.selection.mode: final_frame`` for fixed-duration benchmarks and
debugging with the batched sampler. Each trajectory that finishes successfully
sends its final geometry to QM. The metadata records the first checked uncertainty
or distance violation after ``min_time``, with its geometry under
``first_threshold_violation``. Enable sampler metadata files to retain this field
when compact return metadata is enabled. The sampler always checks the final step,
even when ``Ncheck`` skips intermediate steps. Explicit topology and physical-force
screens still apply.

``acquisition.distcut_frames`` says what a ``distcut`` violation does. With
``"reject"``, the default, it ends the trajectory without a label as described
above. With ``"label"``, the frame is labeled instead, since a close contact the
model allowed is often worth learning:

.. code-block:: json

   "acquisition": {"selection": {"mode": "final_frame"}, "distcut_frames": "label"}

* With ``first_trigger``, a trajectory stops at its first frame that is either
  uncertain or closer than ``distcut``, and that frame is labeled.
* With ``final_frame``, every trajectory still runs its full duration, so step
  counts and step timings are those of ``final_frame`` alone. A trajectory that
  came closer than ``distcut`` is labeled at the first such frame rather than at
  its final frame, and one that never did is labeled at its final frame.
  Uncertainty violations stay diagnostics. The metadata's ``step``,
  ``frame_id``, ``simulationtime``, ``distmin``, geometry and uncertainty describe
  the labeled frame; ``trajectory_final_step`` records the last step run.

Cohort selection doesn't take ``"label"``. ``final_frame`` needs the batched
sampler, ``mlmd_trajectories_task``.

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

   ALF_TEST_NTK_DEVICE=cuda:0 python -m pytest tests/test_ntk.py tests/test_acquisition.py -k 'hippynn or adaptive_ntk'
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
