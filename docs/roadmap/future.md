# Future roadmap

Capabilities not yet started, their decided order, and the checks they will need. See the [roadmap index](README.md) for current status and standing rules.

## Subsequent capability increments

These intended development capabilities remain on the roadmap. Row numbers
are stable identifiers; the working order is set in [Decided priority order](#decided-priority-order-2026-11-03)
below. Independent initialization/replay contributions may proceed
once their helper and identity requirements are available. These are future
capabilities, not features already supplied by Increment 6 or promises that every
existing sampler can already run through the new engines.

| Increment | Capabilities to integrate | Important boundary |
| --- | --- | --- |
| 1. Composable sampling | Separate propagation from UDD, confinement, and other bias/force controls; support callable extensions and grouped non-MD evaluations. | Preserve physical predictions, formulas, switching order, and replica-local state. Folded into the batched UDD/gap ALF step (see [Decided priority order](#decided-priority-order-2026-11-03)). |
| 2. Additional acquisition strategies | Custom import-string strategies and geometry-based demonstrations, followed by production distance-based acquisition and diversity-aware batch selection within cohorts (and later against the existing training set). | Eligibility, ranking, trajectory policy, and selection remain separate; future selectors declare their retention needs. The selector/retention interface follows E5 (see [Decided priority order](#decided-priority-order-2026-11-03)). |
| 3. Model/training handoffs and distillation | Opaque model-artifact references, model-independent inference adapters, and foundation-MLIP reference labeling. | Record teacher identity, version, units, and label provenance separately from the student sampling model. Teacher labeling (3a) is a parallel track after Increment 7 (see [Decided priority order](#decided-priority-order-2026-11-03)); handoffs and distillation (3b) remain later. Fine-tuning tiers F1–F3 are planned in [Later steps](#later-steps-outline-detail-after-e5). |
| 4. Molecular excited states | PySEQM/GPU4PySCF labeling, multi-state HIPPYNN training, derived gap targets, state selection, gap diagnostics, and LCM. | Preserve per-state convergence, state/atom mappings, physical screening, and the distinction between predictions and propagation controls. |
| 5. RDKit initialization | Reference-centered KDG geometry generation, seed-specific screening, and FPS selection, with reproducible settings/provenance. | Optional preprocessing/builders; unlabeled seed geometries are not labeled training rows. |
| 6. Geometry-only HDF5 replay | Reuse stored geometries for sampling or relabeling, with original identities and source provenance. | Do not accidentally carry old labels into a newly requested labeling operation. |
| 7. Optional additional labels | Preserve optional nonadiabatic-coupling labeling and storage, including state-pair and atom-axis mappings. | Computing/storing NACs does not automatically make them energy/force training targets. |
| 8. Durable recovery and staged workflows | Persist batching/cohort handoffs; add retained-label budgets, ALF-to-GALF/configuration stages, exact dataset checkpoints, and endpoint training. | Preserve ordinary runs as the default; budgets count retained labeled frames, not candidates or submitted jobs. |
| 9. Further model and sampling extensions | Additional production model families, reactive batching, and separately planned methods. | Follow the same minimal-core-change and existing-workflow compatibility gates. |
| 10. Multi-node ML training | Train ensemble members across nodes, then optionally data-parallel training of one member across GPUs/nodes. | Keep the `ML_task` entrypoint, `(completed, training_id)` return, member directory layout, and model-promotion semantics; single-node training remains the default. Parallel track, independent of the excited-state sequence (see [Decided priority order](#decided-priority-order-2026-11-03)). |
| 11. Non-ensemble uncertainty | Pluggable uncertainty estimators, including single-network methods, alongside the existing ensemble disagreement. | New signals carry their own names, units, and cutoffs; `Es`/`Fs`/`Fsmax` and `Escut`/`Fscut` keep their ensemble-std meaning. First planned use: foundation-model fine-tuning (F3). |

Items 10 and 11 and the diversity part of item 2 were added at the user's
request on November 2, 2026; their order relative to the other rows is not yet
decided. Readiness notes from a review of development at `24028da`:

- **Multi-node training (10).** `train_HIPPYNN_ensemble_task` is one Parsl
  app on `alf_ML_executor` that runs `multiprocessing.Pool(gpus_per_node)` on
  one node, one member per GPU. Each member sets `CUDA_VISIBLE_DEVICES` itself,
  which is the "training pools" GPU-visibility follow-up in [`batching.md`](batching.md#known-implementation-follow-ups). The tracked
  Darwin/Chicoma configs give the ML executor one node per block. `__main__.py`
  waits for a single ML task (`ML_task_queue.get_number() < 1`) and promotes the
  model after it returns.
  - The first step is to run members across nodes: they train independently,
    so this is task placement, not distributed numerics. A Parsl `join_app`
    that fans out one app per member (or per node) and returns the existing
    `(completed, training_id)` keeps `__main__.py` unchanged. A multi-node
    block with `srun` placement is the alternative. Both need the
    local-ordinal GPU assignment already used by the samplers, and the
    per-member `completed` handling for partial failures must be kept.
  - A later step is data-parallel training of one member, for datasets or
    models too large for one GPU. HIPPYNN's Lightning trainer
    (`hippynn.experiment.lightning_trainer`) is marked experimental and lacks
    some features. Adopting it changes checkpoint layout and the batched
    loader's `best_model.pt` assumption, so it is a separate decision.
- **Non-ensemble uncertainty (11).** This is partly ready.
  - Ready: acquisition reads a signal dictionary, and ranking terms are a
    table (`SCORE_TERMS`). Packed engines keep physical predictions separate
    from propagation.
  - Ensemble assumptions to factor out:
    - both evaluators require at least two members, and the evaluator contract
      returns per-member `[models, ...]` arrays;
    - `run_mlmd_trajectories` calls `ensemble_statistics` directly;
    - eligibility (`uncertainty_flags`) hard-codes `Es`/`Fs`/`Fsmax` against
      `Escut`/`Fscut`/`3*Fscut`;
    - propagation uses the member mean;
    - UDD needs member deviations;
    - legacy samplers read `MLMD_calculator`'s `energy_stdev` results;
    - training assumes `n_models` member directories.
  - Proposed contract: the evaluator returns a mean prediction plus named
    uncertainty signals, with ensemble statistics as the default estimator.
    Eligibility then reads configurable signal/cutoff pairs. Methods that need
    repeated stochastic passes (for example MC dropout) can still use the
    member layout.
  - A single-network estimator may provide only energy uncertainty or no
    per-atom force uncertainty, so eligibility must allow a subset of signals.
  - Calibration and cutoffs need per-method validation; ensemble cutoffs must
    not be reused.
  - UDD-style biasing needs a differentiable uncertainty, which ties this to
    composable sampling (item 1).
- **Diversity-aware acquisition (2).** The cohort pool is the natural place
  for it.
  - Ready: `CohortPool.ready()` selects with `top_k(cohort.candidates, k)`, and
    candidates already carry numbers, positions, cell, PBC, and signals. A
    selector interface, such as score-weighted farthest-point or k-center
    selection on geometric descriptors, can replace that call.
  - Constraints:
    - `_prune` and `max_candidates_per_trajectory` drop candidates by score
      inside the worker before the pool sees them, so a diversity selector must
      declare its own retention (as noted under Known implementation
      follow-ups).
    - First-trigger mode has no pool.
    - Selection must stay invariant to grouping and completion order:
      canonicalize candidates by `(trajectory_id, step)` first and break ties
      deterministically.
    - Model-latent descriptors would require evaluators to return features
      and candidates to carry them, which costs memory.
    - Diversity against the existing training set needs read access to the
      HDF5 store; it is a separate step from within-cohort diversity.
  - The `acquisition.selection` keys are validated against a fixed set, so new
    selectors need a `mode` or import-string extension.

## Decided priority order (2026-11-03)

After Increment 7 closes, the user set this order. Each step will take many
small, separately reviewable changes over time.

1. **Excited-state ALF** (row 4, pulling forward the parts of rows 6 and 7 that
   it needs). The target is the excited-state work on the fork's
   `production_excited_states`. The headline example is the
   cycloparaphenylene ring (cpp) run as a full ALF active-learning example.
   Steps are in [the E1–E6 plan](excited_states.md#excited-state-alf-plan-e1e6).
2. **Composable sampling with batched UDD and gap ALF** (row 1 and the
   sampling half of row 4). The bias/control interface comes first. UDD,
   wells, LCM gap seeking, and the deferred physical-force screen are built on
   it, with gap-aware acquisition.
3. **Staged campaigns across engines** (row 8): durable handoffs, staged
   configurations, and budgets.

Two further items sit outside that sequence:

- **Multi-node training (row 10), a parallel track.** It is independent of
  steps 1–3 and can start whenever there is capacity. The first step places
  members across nodes. Data-parallel training of one member is a later,
  separate decision.
- **Acquisition selector/retention interface (first part of row 2), after
  E5.** A pluggable cohort selector that declares its candidate-retention
  needs replaces the hard-wired `top_k` in `CohortPool.ready()` and the
  score-only `_prune` in `batched_mlmd.py`. Top-K remains the default with
  identical selections.
- **Foundation-model teacher labeling (row 3a), a parallel track.** A
  foundation MLIP supplies reference labels instead of QM. It is independent of
  steps 1–3 and can start after the Increment 7 close-out. Ground-state
  teachers come first; excited-state teachers wait until a suitable model
  exists.
- **Fine-tuning with ALF (rows 3b and 11), a later goal in three tiers.** ALF
  already fine-tunes ALF-trained HIPPYNN ensembles through the example-local
  task in `examples/fine_tuning/`.
  - **F1, library HIPPYNN fine-tuning.** It can start any time after the
    current-release compatibility gate.
  - **F2, multi-state fine-tuning.** It follows E3.
  - **F3, foundation-model fine-tuning under active learning.** It follows E5,
    teacher labeling (3a), and multi-node training.

  See [Later steps](#later-steps-outline-detail-after-e5).

Rationale (2026-11-03): the user set composable sampling and multi-node
training as the priorities after excited states, to make other uncertainty and
acquisition methods easier.
- **Composable sampling is folded into step 2** because that step already
  rebuilds UDD and LCM in the packed engines. Porting them ad hoc first would
  mean writing them twice.
- **Multi-node training runs in parallel** because it touches only training
  placement. It buys throughput: larger ensembles and multi-state ring
  training. It does not add method extensibility.
- **Method extensibility comes from three seams:**
  1. named uncertainty signals from inference (E5);
  2. eligibility from configured signal/cutoff pairs (E5);
  3. the selector/retention interface above.
- **How later methods attach:** single-network estimators (row 11) plug into
  seams 1–2. Differentiable estimators used for biasing plug into the
  composable-sampling interface. Diversity selection (row 2) plugs into seam 3.

Rows 2 (remaining strategies), 3, 5, 9, and 11 follow later. While
the work is ported, the fork remains the production environment for excited
states; `production_excited_states` (`c364fa8`) is the source material. Its
excited-state sources match `feature/unified-sampling-drivers` byte for byte.

## Later steps (outline; detail after E5)

- **Single-member excited-state sampling (user addition, November 5).**
  - Allow selection/loading of one ensemble member for selected-state
    propagation, at trajectory widths one/four across ASE, TorchSim, and
    ALCHEMI. Keep full-ensemble sampling as the default.
  - Define the acquisition policy explicitly when committee uncertainty is
    unavailable: support sampling without committee acquisition or a separately
    qualified uncertainty signal. Specify any optional full-ensemble checks
    separately from the single-member propagation path.
  - Gate: only the requested checkpoint is loaded for single-member execution;
    propagation matches that member's selected-state forces, single/batched
    observations agree, and existing ensemble/ground-state paths remain valid.
    This is future work, not an additional cpp-ring prerequisite.
- **Composable sampling with batched UDD and gap ALF.**
  - Start with the bias/control interface in the packed engines (row 1). Each
    control:
    - receives physical predictions and returns propagation forces;
    - keeps replica-local state (for example, the permanent LCM switch);
    - never changes the unbiased predictions used for screening and
      uncertainty;
    - applies in a fixed, documented switching order.
  - First plugins:
    - UDD on the selected state;
    - wells;
    - LCM gap seeking as a per-replica permanent mode switch.
  - The deferred physical-force screen (`max_force_cutoff`) excludes bias and
    LCM forces.
  - Gap-aware summed acquisition.
  - Gate: parity with the current `MLMD_calculator` UDD/well formulas and with
    the fork's ALCHEMI LCM behavior, across ASE, TorchSim, and ALCHEMI.
- **Multi-node training (parallel track).**
  - Today `train_HIPPYNN_ensemble_task` runs `multiprocessing.Pool` on one
    node. Fan members out across `alf_ML_executor` blocks that span nodes,
    via a `join_app` or per-member apps, returning the existing
    `(completed, training_id)`. `__main__.py` is unchanged.
  - Use local-ordinal GPU assignment, as for samplers (`tools/gpu.py`); keep
    per-member `completed` handling and the existing member directory layout.
  - Gate: a real two-node Darwin training of four members, then promotion and
    reload.
  - Data-parallel training of one member (HIPPYNN's experimental Lightning
    trainer, with a checkpoint-layout change) is a separate decision.
- **Acquisition selector/retention interface (after E5).**
  - The selector receives canonicalized candidates and declares what workers
    must retain.
  - Top-K is the default.
  - Gate: identical selections in the existing `tests/test_cohorts.py`
    width/completion-order replay; online scalar top-K retention closes the
    candidate-memory follow-up.
- **Foundation-model teacher labeling (row 3a, parallel track).** The
  labeling boundary already supports this: `LabelingCoordinator`
  (`tools/labeling.py`) packs `native` batches for any `QM_task` that accepts
  `molecule_objects`, as PySEQM does. Storage, retraining, and promotion see
  only labels and a converged flag.
  - **Adapter:** an ordinary `QM_task`, native over `molecule_objects`, with a
    single-structure form for sequential packing. It:
    - loads the teacher once per worker;
    - evaluates with packed batched inference, reusing the evaluator's padding
      and atom-order conventions, or with a per-structure ASE calculator;
    - converts outputs to ALF's eV and eV/Å;
    - selects its GPU through `tools/gpu.py` local ordinals.
  - **Why `ase_calculator_task` is not enough:** it passes `QM_run_command`
    and the whole `QM_config` to the calculator constructor, and it requires
    `calc.converged`.
  - **Converged rule:** a teacher needs its own rule, such as finite outputs
    plus optional domain checks.
  - **Provenance:** record the teacher's name, version, checkpoint hash, units,
    and adapter settings with its labels, separately from the student model.
    `store_current_data` writes only `properties_list` keys, so this needs an
    explicit metadata path.
  - **Site:** a GPU-configured `alf_QM_executor`. This is configuration only,
    as in the PySEQM GPU campaign.
  - **Gate:**
    - single and native-batch parity at widths 1/2/4/8;
    - a bounded water-style active-learning campaign in which teacher labels
      are stored, used for real retraining, and promoted;
    - provenance present in the stored data.
  - **Out of scope for 3a; remains in 3b or row 9:** mixed teacher/QM labels
    (multi-fidelity or delta learning), teacher-driven dynamics,
    fine-tuning/distillation handoffs, and opaque model-artifact references.
- **Fine-tuning with ALF (later goal; added 2026-11-03).**
  - **What exists.** `examples/fine_tuning/` (and its Darwin variant) uses a
    local `ML_task`, `fine_tune_ml_task.fine_tune_HIPPYNN_ensemble_task`. It:
    - loads a trained HIPPYNN seed ensemble from `models/model-0000/`;
    - keeps the checkpoint graph and weights, with a fresh optimizer and
      controller at a lower learning rate;
    - rebuilds the training database from `h5store/`, filtered to the
      checkpoint's inputs and targets;
    - writes each adapted ensemble to the next model slot.

    `__main__.py` is unchanged and `--test_ml` runs fine-tuning only. The
    example is part of the compatibility baseline. Fine-tuning is example
    code, not library code, and it covers HIPPYNN ensembles in ALF's own
    layout only.
  - **F1, library HIPPYNN fine-tuning (row 3b).**
    - Move the task beside `train_HIPPYNN_ensemble_task` as an ordinary
      `ML_task`.
    - Make its settings explicit: source checkpoint, optimizer reset, learning
      rate, frozen layers, and training-data filter.
    - Record lineage: the parent checkpoint hash and generation.
    - Keep the example working by pointing it at the library task, with its
      results unchanged.
    - Inherit multi-node placement, and do not conflict with it.
    - Gate: the existing fine-tuning example gives equivalent results through
      the library task, including a real Darwin fine-tuning round.
  - **F2, multi-state fine-tuning (after E3).** Start a multi-state model from
    a pretrained ground-state or earlier multi-state HIPPYNN; for example,
    seed the cpp ring model from a ground-state model. Use the F1 lineage and
    settings, with an explicit mapping from source to new heads and targets.
  - **F3, foundation-model fine-tuning under active learning (rows 3b and
    11).** ALF samples with the model being fine-tuned and labels with QM or a
    teacher (3a). It needs:
    - **Uncertainty.** Foundation models are usually single networks.
      Options: fine-tuned ensembles (expensive), heads on a shared backbone,
      or single-network estimators (row 11) through the E5 named-signal
      contract. F3 is the natural first use of row 11.
    - **Model-family adapters** for training (behind `ML_task`) and inference
      (behind the evaluator contract).
    - **Opaque model-artifact references,** replacing the batched loader's
      `best_model.pt` and the trainers' `model-NN` assumptions. This must
      preserve the existing HIPPYNN layout.
    - **Provenance:** the base model, version, license, and which weights
      changed.
    - **Engines:** TorchSim and ALCHEMI already host foundation-model
      calculators, so the engine side may need little new work. Each model
      still needs its own qualification.

    Gate: a bounded active-learning campaign in which a fine-tuned foundation
    model's uncertainty selects frames that are labeled, stored, used for
    fine-tuning, and promoted, with lineage and provenance recorded.
  - **First F3 targets: MACE and UMA (notes from 2026-11-03).** These
    describe the packages found installed, not an acceptance claim. Each
    package must still pass the clean current-release qualification.
    - **Availability.** `alf_alchemi_02` contains `mace-torch` 0.3.16 and
      `fairchem-core` 2.23.0. The batched engines already provide calculators:
      - TorchSim: `torch_sim/models/mace.py` and `fairchem.py`;
      - ALCHEMI 0.2: `nvalchemi/models/mace.py` and `uma.py`.
    - **Inference.** ALF needs an evaluator adapter, the counterpart of
      `load_hippynn_batched_ensemble`.
      - It is selected through `batched_calculator` and returns per-member
        energies `[M, S]` and forces `[M, atoms, 3]` in eV and eV/Å. Engine
        changes are not expected.
      - The legacy sampler could use an `ase_calculator` loader returning one
        ASE calculator per member. This is untested.
    - **MACE training adapter.** `mace_run_train` provides:
      - `--foundation_model`, and `--freeze` for frozen layers;
      - multihead fine-tuning (`--multihead_finetuning`, `--pt_train_file`),
        which replays pretraining data to limit forgetting.

      The adapter exports `h5store/` to extxyz in eV and eV/Å and trains each
      member. It keeps outputs in ALF's `model-NN` layout and returns
      `(completed, training_id)`.
    - **UMA training adapter.** `create_uma_finetune_dataset.py` builds the
      dataset, fits normalizers and linear references, and writes a config
      from fairchem's fine-tuning template. The fairchem CLI then trains from
      the UMA checkpoint. The adapter must select and record the task head
      (`omol` for molecules, `omat` for materials).
    - **Reference level and energy offsets.** Fine-tuning on ALF's labels
      (for example ORCA PBE/def2-SVP) moves the model away from its pretraining
      level of theory. Make these explicit and record them:
      - MACE E0s;
      - UMA linear references;
      - `omol` charge and spin inputs.
    - **Training placement.** Both trainers support multi-GPU training
      natively (MACE `--distributed`, fairchem DDP). For these families,
      multi-node training means placing members and then delegating to the
      native distributed training, not building a new data-parallel trainer.
    - **Uncertainty.** Members fine-tuned from one foundation checkpoint tend
      to agree too closely, so their disagreement underestimates error. MACE
      copies are affordable but share this problem; several UMA copies are
      also expensive. Options, roughly cheapest first:
      1. a heterogeneous committee (for example MACE, UMA, and HIPPYNN), which
         the per-member evaluator contract already allows;
      2. multiple heads on a shared backbone;
      3. single-network estimators (row 11) through E5's named signals.

      Calibrate cutoffs for each method; do not reuse ensemble `Escut`/`Fscut`.
    - **Access, license, and environments.**
      - UMA weights are gated on Hugging Face: workers need an accepted
        license and a token.
      - Record each base checkpoint, version, and license: UMA's FAIR
        Chemistry license and MACE's model licenses.
      - Each family pins its own torch/e3nn range, so each gets its own
        qualified environment, separate from the HIPPYNN environment.
    - **Suggested order.** Each step has a bounded campaign gate like teacher
      labeling. The first two need no fine-tuning.
      1. UMA or MACE as a teacher (3a), labeling for HIPPYNN students.
      2. A frozen committee as the sampling model: the evaluator adapter
         only, to prove the engine, evaluator, and uncertainty plumbing.
      3. MACE fine-tuning (more mature entrypoints and smaller models).
      4. UMA fine-tuning.
- **Staged campaigns across engines** (row 8): durable cohort/handoff
  persistence, a run-control equivalent, staged configurations, and
  retained-label budgets. The fork's `run_control`, `workflow_io`,
  `staged_workflow`, and `dataset_screening` are source material.

Current excited-state coverage includes atom-identity validation and
topology-aware replay/storage. Chemical topology screening is the explicit
Increment 11 prerequisite above. Existing ordinary workflows must not acquire
a new fixed-topology requirement when screening is disabled.

Budgeted workflows must distinguish selected candidates, labeling attempts,
successful labels, stored/retained frames, and rows surviving trainer filters.
Report these stages separately. Exact checkpoints and final endpoint training
refer to the chosen retained-data lineage; routine retraining and raw-store
overshoot are separate concepts.

## Existing-example compatibility (planned, 2026-11-03)

No planned increment is meant to make the original examples unusable; every
new capability stays opt-in. The evidence is narrower than that intent, so
these checks are part of the future plan.

**Current state.**
- All ten tracked examples still select the original tasks:
  - `simple_mlmd_sampling_task`, or `reactive_sampling`;
  - the ORCA, VASP, and ASE labelers;
  - `train_HIPPYNN_ensemble_task`, `fine_tune_HIPPYNN_ensemble_task`, or
    `train_ANI_model_task`;
  - the original builders.
- The only deliberate config change is `"distcut": 0.7` in four water
  examples, made at the user's direction; the code default stays 1.2 Å.
- Since these changes, only the water family has run through the real
  coordinator: the legacy ASE width-one cases of the campaign matrix.
- IL, molten salt (VASP), UO2 (ANI/NeuroChem), reactive sampling,
  fine-tuning, and seeded active learning have CPU tests and unchanged code
  paths. They have not been re-run end to end.

**Risk by planned step.**
- **Low:** E1–E4, teacher labeling, and the selector interface. They add new
  tasks or opt-in branches. Watch the shared helpers:
  - `store_current_data`, where E1 adds the NAC scope;
  - `hippynn_interface.py`, where E3 must leave the ground-state trainer
    unchanged.
- **Medium:**
  - E5's evaluator and eligibility generalization: ground-state shapes and
    `Es`/`Fs`/`Fsmax` must remain the default.
  - Multi-node training: every HIPPYNN example uses
    `train_HIPPYNN_ensemble_task`, so single-node placement stays the default
    with identical behavior.
- **Highest: composable sampling with UDD and wells.** Five examples set
  `well_params`: IL, molten salt, simple water, fine-tuning, and seeded active
  learning. Build the plugins for the packed engines. Leave `MLMD_calculator`
  and `mlmd_sampling` unchanged, or prove formula parity if code must be
  shared.
- **Staged campaigns and recovery:** keep ordinary runs as the default; do not
  port the fork's `run_control` changes to `__main__.py` wholesale.
- **Fresh-install policy:** newer dependencies can break examples without any
  ALF change. ANI/NeuroChem and VASP paths are the least exercised.

**Planned checks.**
1. **Compatibility smoke matrix.** One bounded real-coordinator run per
   tracked example family. Run it:
   - at the first fresh-environment build;
   - after E5, after composable sampling, and after multi-node training;
   - at each release point.

   Water, fine-tuning, and seeded active learning run with ORCA on Darwin.
   IL, molten salt (VASP), UO2 (ANI), and reactive sampling need their
   backends. Record a missing backend as "unverified, backend unavailable",
   never as passed.
2. **Config-load test.** A CPU test that loads every tracked example's JSON
   configurations and validates them against current code, including
   `acquisition_settings`, `driver_settings`, labeling batch settings, and
   task import strings. It is cheap and suitable for CI.
3. **UDD/well parity.** In the composable-sampling step, test the new plugins
   against the legacy `MLMD_calculator` well and UDD forces, using the
   examples' actual `well_params` values (`r_start` 3.0 and 10).

**Breaking changes need their own increment.** Examples would stop working only
through a deliberate decision, which no current increment makes. Examples of
such decisions:
- removing the legacy sampler or `MLMD_calculator`;
- changing code defaults such as `distcut` or the uncertainty cutoffs;
- adopting the fork's `sampling`/`inference` configuration dialect, already
  rejected in Increment 4;
- dropping a dependency such as NeuroChem.

Any such retirement is its own increment: an explicit decision, a deprecation
period, and migrated example configurations. It is never a side effect of
another change.
