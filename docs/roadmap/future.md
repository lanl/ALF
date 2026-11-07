# Future roadmap

What's left, in what order, and what's easy. See the [roadmap index](README.md) for current status and standing rules.

This file lists remaining work and its completed prerequisites. Delivered work is recorded in [`batching.md`](batching.md) and [`excited_states.md`](excited_states.md).

## Order of work

Updated November 6, 2026, after E6 passed. The user set the sequence on November 3, and revised the bias work on November 6.

E6, the bounded quick-win hardware round, and gap-seeking/sampling-window
mechanics have passed; see the [E6 record](excited_states.md),
[quick-win qualification](README.md#quick-win-hardware-qualification--passed-2026-11-06),
and [gap/window qualification](README.md#gap-seeking-and-sampling-window-hardware-qualification--passed-2026-11-06).

1. **Safe restarts and disk races** ([below](#safe-restarts-and-disk-races), with priorities in [`exascale.md`](exascale.md#priority-list)). Tabled for now (user decision, November 6), and next when picked up.
2. **Placement and memory** ([Group C](#group-c-placement-and-memory)). Multi-node independent-member training passed its bounded Darwin gate on November 7; larger deployments remain unqualified. It buys throughput for larger ensembles, not new methods.
3. **Teacher labeling, then F1 and F2 fine-tuning** ([Group D](#group-d-model-families-and-labels)). Teacher labeling is built, and awaits its Darwin gate.
4. **New uncertainty and acquisition methods** ([Group G](#group-g-new-uncertainty-and-acquisition-methods)).
5. **Long runs and staged campaigns** ([Group E](#group-e-long-runs)), last: very low priority, and may be removed (user decision, November 6).

E6 passed on November 6, within its bounded scope. CPP scale-up ([Group F](#group-f-cpp-scale-up)) now depends on the science goals.

**No merge to `main` is planned (user decision, November 6).** `main` and the public documentation site, which deploys from it, stay as they are.

**Why the bias work changed (user decision, November 6).** The November 3 plan built a general bias/control interface first, then moved UDD, wells, and gap-seeking onto it. But the packed engines already apply UDD and wells, and `SelectedStateEvaluator.evaluate` already gives them the selected state's member arrays. So the direct path comes first, and a general interface waits until a further method needs one.

The fork remains source material; `production_excited_states` (`c364fa8`) holds the gap-seeking implementation.

## Quick wins

All landed on November 6; see the [record](README.md#quick-wins-2026-11-06). The guide publishes only from `main` for now (user decision), so publishing it from `development` was dropped.

**Real hardware gates passed November 6:** all 24 selected-state UDD/well cases
across three engines and widths one/four; ground-state training with reordered
four-GPU and restricted two-GPU masks; real NEB/dimer device checks; and the
GPU4PySCF guard/response checks. A small ordinary ALF loop also qualified FPS
selection, native width-four labeling, retraining/promotion and the production
label-budget stop. See the [evidence](README.md#quick-win-hardware-qualification--passed-2026-11-06).

**Subsequent real mechanics gates passed November 6.** These use the later
ALF `1055d9c` source boundary; see the [evidence](README.md#gap-seeking-and-sampling-window-hardware-qualification--passed-2026-11-06).

- Gap-seeking: real width-one/four runs with and without wells on all three
  engines, permanent switching and no-switch controls, independent surface-force
  checks and preservation of unbiased screening. The frozen 20 eV trigger tests
  mechanics; near-crossing accuracy remains unqualified.
- Sampling windows: an ordinary ALF loop switched at 40 stored labels, stored
  eight new propagated labels in each window, retrained/promoted twice, and exited
  normally at its 48-label budget with owned-job termination verified.

**Look easy, but aren't:**

- **Single-member sampling.** With no ensemble spread, it first needs a decision on what triggers labeling. See [Group G](#group-g-new-uncertainty-and-acquisition-methods).
- **F1 fine-tuning.** It moves 353 lines of example code into the library, and its gate is a real fine-tuning round.
- **Teacher labeling.** The adapter is small, but it needs a provenance path and a campaign gate.

## Group B: Biased dynamics on excited states

| Capability | What it covers | Important boundary |
| --- | --- | --- |
| Composable sampling | Separate propagation from UDD, confinement, and other bias/force controls; support callable extensions and grouped non-MD evaluations. | Preserve physical predictions, formulas, switching order, and replica-local state. |

1. **Selected-state UDD and wells — qualified November 6.** Real width-one/four
   runs passed across ASE, TorchSim and ALCHEMI. Common-frame forces match the
   `MLMD_calculator` formulas, including active example-radius wells, and screening
   still reads unbiased predictions. This gate is closed.
2. **Gap-seeking — mechanics qualified November 6.** Real width-one/four runs
   passed on all three engines, including wells, switch/no-switch controls and
   independent checks of the fork's surface formula; see the
   [hardware record](README.md#gap-seeking-and-sampling-window-hardware-qualification--passed-2026-11-06).
   Physically small-gap acquisition and long-time behavior remain unqualified.
3. **General bias/control interface, only when a further method needs one.** Each control:
    - receives physical predictions and returns propagation forces;
    - keeps replica-local state (for example, the permanent gap-seeking switch);
    - never changes the unbiased predictions used for screening and
      uncertainty;
    - applies in a fixed, documented switching order.

## Safe restarts and disk races

Tabled on November 6, before any restart code changed. ALF has since gained exit handling that releases allocations, but none of the restart fixes below. An audit of ALF's disk writes read all of `alframework/` and traced real coordinator and training test runs on one machine. The November 7 bounded Darwin round subsequently measured cross-node locks, shard visibility, five-node member placement and interruption/restart behavior; see [the measurements](exascale.md#darwin-measurements-2026-11-07). It did not implement crash-safe recovery or establish large-campaign performance.

**ALF has one writer for shared state.** The coordinator alone writes `status.json` and the HDF5 shards. Workers write only to paths unique to a structure, member, or task, stores can't overlap training because one `ML_task_queue` check gates both, and models are read only after promotion. So there's no worker-against-worker race in normal operation.

**Race risks:**

- **Stale directory views.** A trainer lists `h5_dir` right after a store. With attribute caching, another node might not see the new shard, so either trainer could silently train without it. Replay looks up each shard by name and raises `FileNotFoundError`.
- **Two coordinators in one directory.** There's no lock. The fork uses `fcntl.flock`.
- **ID reuse after a crash.** `status.json` can lag one loop. `ase_calculator_task` then raises "Scratch directory exists", while ORCA, Q-Chem, VASP, and metadata pickles silently overwrite.

**Crash risks:**

- A crash after a store, before status is saved: the shard name is reused, and `datapacker`'s `w-` mode makes the next store fail.
- A crash during a store leaves a partial `*.h5` where training reads it.
- A crash during a status write leaves a truncated `status.json`.
- A crash after starting training can reuse a training ID, and the excited-state trainer refuses a nonempty directory.
- A restart during a `finite_bootstrap` fails with "Finite bootstrap exhausted", because `current_molecule_id` has advanced with nothing stored.
- An interrupted bootstrap loses all its labels, which are stored only at the end. This may be the most expensive risk for large CPP bootstraps.
- A crash during training loses every label finished since the last store, because stores wait for training to finish.
- `maximum_stored_labels` drops labels already finished but not yet stored.

**File I/O at exascale, and what to fix first.** See [`exascale.md`](exascale.md) for every operation ranked by severity, the [priority list](exascale.md#priority-list), the measurements needed on Darwin, and where node-local storage fits.

## Group C: Placement and memory

| Capability | What it covers | Important boundary |
| --- | --- | --- |
| Multi-node ML training | Independent-member training passed on two and five real GPU nodes; larger deployments remain unqualified. Data-parallel training of one member across GPUs or nodes is a separate decision. | Keep the `ML_task` entrypoint, `(completed, training_id)` return, member directory layout, and model-promotion semantics; single-node training remains the default. |

| Area | Current behavior or gap | Follow-up and boundary |
| --- | --- | --- |
| Packed inference memory | HIPPYNN pads to the largest system in a task. Capacity is specified in systems, without an atom/pair-memory budget or automatic OOM splitting. A failed batched evaluator call can fail the whole active group. | Profile heterogeneous sizes before increasing widths. Consider adapter-declared memory limits and explicit retry/splitting later, preserving atom order, model pinning, identities, and per-trajectory random streams. This is not a requirement to add every OOM recovery path now. |

- **Multi-node training: bounded Darwin qualification passed (November 7).** `train_HIPPYNN_ensemble_multinode_task` and `train_excited_state_HIPPYNN_ensemble_multinode_task` train one member per GPU on any number of nodes, using `tools/ensemble_training.py`. `__main__.py`, promotion, and the member layout are unchanged, and single-node training stays the default. See the [record](README.md#multi-node-training-2026-11-06).
  - **Result:** both trainers passed eight-member/two-node clean and cancelled-block
    retry runs; excited-state training passed 20 members on five nodes. The ordinary
    ALF loop verified zero-retry failure isolation and a later eight-member promotion.
    Real sampling, composition-only loading, `--test_ml` cleanup, coordinator `SIGTERM`
    and dead-executor handling passed. See the
    [hardware record](README.md#current-development-hardware-qualification--passed-with-restart-limits-2026-11-07).
    The recorded cleanup signal was coordinator `SIGTERM`, not literal `scancel` of
    a batch-hosted coordinator; `SIGKILL` still required supervisor cleanup.
  - **Original Darwin gates and boundaries:**
    1. 8 members on 2 nodes × 4 GPUs, for both trainers. Placement records show 8 distinct host and GPU pairs, with members overlapping in time. All 8 reload, the model promotes, and sampling runs on it.
    2. Every member's training records list the same shards.
    3. Cancel one block mid-training. Only its members are retried, failed directories stay, and the ensemble promotes and reloads.
    4. With `member_retries: 0`, the same cancellation leaves the coordinator running, promotes nothing, and the next training uses a new ID.
    5. For one trainer, the largest allocation available, ideally 40 members on 10 nodes. Record wall time against the 2-node run, how long members take to load data, and whether late nodes still get work.
    6. After `--test_ml`, `squeue` empties shortly after ALF exits. Only a real allocation can confirm this.
    7. `scancel` of the coordinator during multi-node training releases the ML blocks.
    8. Excited-state sampler tasks load members without reading `training_summary.json`.
    9. A deliberately bad `worker_init` on the ML executor makes ALF stop with a clear message and releases the other executors' blocks. Real node failures can't be triggered on demand, and `scancel` counts as cancelled, not failed.
  - **Known limits, for Darwin to watch:** uncertainty cutoffs shift with ensemble size. GPU memory for 40 sampling members is untested. `strategy="htex_auto_scale"` could release nodes sooner, but hasn't been tried.
  - **Not included:** promoting a partial ensemble, a training deadline, and restart recovery of a partly trained ensemble after a coordinator crash.
- **Data-parallel training of one member (separate decision).** For datasets or models too large for one GPU. HIPPYNN's Lightning trainer (`hippynn.experiment.lightning_trainer`) is marked experimental and lacks some features. Adopting it changes the checkpoint layout and the batched loader's `best_model.pt` assumption.

## Group D: Model families and labels

| Capability | What it covers | Important boundary |
| --- | --- | --- |
| Model/training handoffs and distillation | Opaque model-artifact references, model-independent inference adapters, and foundation-MLIP reference labeling. | Record teacher identity, version, units, and label provenance separately from the student sampling model. Teacher labeling is an independent track; handoffs and distillation come later. Fine-tuning tiers F1–F3 are planned below. |
| Further model and sampling extensions | Additional production model families, reactive batching, and separately planned methods. | Follow the same minimal-core-change and existing-workflow compatibility gates. |

- **Foundation-model teacher labeling: built November 7, Darwin gate pending.** See the [record](README.md#teacher-labeling-2026-11-07) and the [guide](../source/user_guide/qm_interfaces.rst). The original plan follows. The
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
  - **Out of scope for teacher labeling; remains with handoffs or further extensions:** mixed teacher/QM labels
    (multi-fidelity or delta learning), teacher-driven dynamics,
    fine-tuning/distillation handoffs, and opaque model-artifact references.
  - **Order:** ground-state teachers come first; excited-state teachers wait until a
    suitable model exists. UMA or MACE are the first candidates; see the notes in
    [Group G](#group-g-new-uncertainty-and-acquisition-methods).
- **Fine-tuning with ALF (later goal; added 2026-11-03).**
  - **Prerequisites.** F1 and F2 can start at any time. F3 is in
    [Group G](#group-g-new-uncertainty-and-acquisition-methods).
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
  - **F1, library HIPPYNN fine-tuning.**
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
  - **F2, multi-state fine-tuning.** Start a multi-state model from
    a pretrained ground-state or earlier multi-state HIPPYNN; for example,
    seed the cpp ring model from a ground-state model. Use the F1 lineage and
    settings, with an explicit mapping from source to new heads and targets.
The model/training-boundary work must also address two porting requirements recorded during the experimental-fork review:

- The training handoff must tolerate valid trainer callables whose signature
  omits `ML_config`.
- Move model-family artifact validation behind adapters; common final-training
  validation must not assume HIPPYNN-style `model-NN` member directories.

## Group F: CPP scale-up

E6 passed within a bounded pilot, and none of these has been shown so far; see [`excited_states.md`](excited_states.md).

- Ten-node scaling.
- Production width 50, and the 50,000-to-5,000 preparation.
- Predictive accuracy and long-time stability.
- GPU capacity for the smaller HIP-HOP100, HIP-HOP64, and HIP-NN64 models.
- Full-epoch validation and checkpoint export at the recommended batch size of 8, which the calibration didn't run.

## Group G: New uncertainty and acquisition methods

These come last (user decision, November 6). They add new ways to measure uncertainty or choose frames, so they wait until the other groups are done.

Method extensibility comes from three seams:

1. named uncertainty signals from inference (done);
2. eligibility from configured signal/cutoff pairs (done);
3. the selector/retention interface below.

Single-network estimators plug into seams 1-2, and diversity selection plugs into seam 3. Differentiable estimators used for biasing would need the general interface in [Group B](#group-b-biased-dynamics-on-excited-states).

### Selection and retention

| Capability | What it covers | Important boundary |
| --- | --- | --- |
| Additional acquisition strategies | Custom import-string strategies and geometry-based demonstrations, followed by production distance-based acquisition and diversity-aware batch selection within cohorts (and later against the existing training set). | Eligibility, ranking, trajectory policy, and selection remain separate; future selectors declare their retention needs. |

`selection.selector` now chooses between `top_k` and `fps`, and `worker_retention_limit` says what workers keep for each. What's left:

- Custom selectors and strategies through import strings. The `acquisition.selection` keys are validated against a fixed set, so these need an extension.
- Selectors that use scores as well as geometry, such as score-weighted farthest-point or k-center. `fps` ignores scores among eligible frames, apart from which frames a worker keeps.
- Keeping frames for `fps` by something other than score. A worker's cap still drops candidates by score before the pool sees them.
- Diversity against the existing training set. It needs read access to the HDF5 store, and is a separate step from within-cohort diversity.
- Model-latent descriptors, which would require evaluators to return features and candidates to carry them, which costs memory.

Any new selector must stay invariant to grouping and completion order: canonicalize candidates by `(trajectory_id, step)` first and break ties deterministically. First-trigger mode has no pool. A `max_candidates_per_trajectory` below `k` can still change the unrestricted global top-K.

### Non-ensemble uncertainty

| Capability | What it covers | Important boundary |
| --- | --- | --- |
| Non-ensemble uncertainty | Pluggable uncertainty estimators, including single-network methods, alongside the existing ensemble disagreement. | New signals carry their own names, units, and cutoffs; `Es`/`Fs`/`Fsmax` and `Escut`/`Fscut` keep their ensemble-std meaning. First planned use: foundation-model fine-tuning (F3). |

Readiness notes from a review of development at `24028da`, with the finished eligibility item removed.

- **Non-ensemble uncertainty.** This is partly ready.
  - Ready: acquisition reads a signal dictionary, and ranking terms are a
    table (`SCORE_TERMS`). Packed engines keep physical predictions separate
    from propagation.
  - Ensemble assumptions to factor out:
    - both evaluators require at least two members, and the evaluator contract
      returns per-member `[models, ...]` arrays;
    - `run_mlmd_trajectories` calls `ensemble_statistics` directly;
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
    the general interface in Group B.

### Single-member sampling

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

### Foundation-model fine-tuning

- **Fine-tuning with ALF, tier F3.** It needs teacher labeling and multi-node training, and one of the uncertainty options below. Multi-node training exists for HIPPYNN only; MACE and fairchem would still need theirs.
  - **F3, foundation-model fine-tuning under active learning.** ALF samples with the model being fine-tuned and labels with QM or a
    teacher. It needs:
    - **Uncertainty.** Foundation models are usually single networks.
      Options: fine-tuned ensembles (expensive), heads on a shared backbone,
      or single-network estimators through the named-signal
      contract. F3 is the natural first use of non-ensemble uncertainty.
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
      3. single-network estimators through the named signals.

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
      1. UMA or MACE as a teacher, labeling for HIPPYNN students.
      2. A frozen committee as the sampling model: the evaluator adapter
         only, to prove the engine, evaluator, and uncertainty plumbing.
      3. MACE fine-tuning (more mature entrypoints and smaller models).
      4. UMA fine-tuning.

## Group E: Long runs

Very low priority, and may be removed (user decision, November 6).

| Capability | What it covers | Important boundary |
| --- | --- | --- |
| Durable recovery and staged workflows | Persist batching/cohort handoffs; add retained-label budgets, ALF-to-GALF/configuration stages, exact dataset checkpoints, and endpoint training. | Preserve ordinary runs as the default; budgets count retained labeled frames, not candidates or submitted jobs. |

| Area | Current behavior or gap | Follow-up and boundary |
| --- | --- | --- |
| Cohort admission | A cohort requires full membership and all terminal reports, with no admission timeout or partial selection. A permanently undersupplied builder leaves it unfinished. | Keep incremental admission for cohorts larger than worker capacity; do not introduce an automatic partial-cohort policy without a separate decision. |
| Queue thresholds and strict batches | `full_only` can leave a PBC/model group waiting indefinitely when it cannot fill within available capacity. The inherited `minimum_QM` harvest gate can also hold a small number of finished sampler results. | Exercise small capacities and fragmented groups with the existing timeout policy. Check that configured completion thresholds are reachable. Add focused diagnostics/validation or drain behavior only as a separate compatibility change. Strict QM and sampler batches are already checked at startup against `target_queued_QM` and `parallel_samplers`. |
| Restart and failed work | Buffers, cohorts, model admission sequence, and handoffs are in memory. No new retry journal or exactly-once QM handoff exists. Candidates survive a later isolated failure only if the task returns them; crashed-worker-only frames are lost. | Keep this limitation visible for ordinary screen/tmux runs as well as staged workflows. Add durable snapshots, logical IDs, replay, and handoff recovery in the later recovery increment, without requiring new pause controls now. |

- **Staged campaigns across engines:** durable cohort/handoff
  persistence, a run-control equivalent, staged configurations, and
  retained-label budgets. The fork's `run_control`, `workflow_io`,
  `staged_workflow`, and `dataset_screening` are source material.
- **Sampling windows.** Landed November 6: `sampling_windows` moves one run from base sampling to UDD or gap-seeking at stored-label counts; see the [record](README.md#sampling-windows-2026-11-06). A bounded real base-to-gap-seeking loop has passed. The staged runner above would add fresh ensembles for each stage, exact dataset checkpoints, inherited-data compatibility checks, and a task journal with resume.
- **Budgets beyond a stop rule.** `maximum_stored_labels` ends a run after enough converged labels are stored. Retained-label budgets, draining running work, and the other counts below are still open.
- **Filtering converged labels after QM.** Runtime force and topology screens run on sampled frames, and don't filter stored QM labels. Reuse `tools/molecular_topology.py` where possible, and keep it opt-in, because it changes which data is stored.

Budgeted workflows must distinguish selected candidates, labeling attempts,
successful labels, stored/retained frames, and rows surviving trainer filters.
Report these stages separately. Exact checkpoints and final endpoint training
refer to the chosen retained-data lineage; routine retraining and raw-store
overshoot are separate concepts.

## Parking lot

No plan yet. Each needs its own decision.

- NPT sampling, and periodic excited states.
- Surface hopping.
- True mixed-composition native CIS forces.
- Grouped non-MD evaluations, and reactive batching.
- Data-parallel training of one member.
- MI350 hardware, deferred by the user. Legacy NeuroChem stays excluded from current acceptance.
- Streaming results and replacing finished trajectories in flight, which needs a task/result protocol change:

| Area | Current behavior or gap | Follow-up and boundary |
| --- | --- | --- |
| Capacity and result latency | Finished members leave GPU packed state, but their queue width is released only when the whole task completes and is harvested. Earlier first-trigger frames wait for slower batch-mates. There is no inflight replacement or result streaming. | Keep physical capacity, candidate storage, and pending QM work distinct. Streaming or replacement would need an explicit task/result protocol change; fixed task membership remains current behavior. |

## Existing-example compatibility (planned, 2026-11-03)

No planned increment is meant to make the original examples unusable; every
new capability stays opt-in. The evidence is narrower than that intent, so
these checks are part of the future plan.

**Current state.**
- All ten original example configurations still select the original tasks. `examples/AcAc_excited_pyseqm` and `examples/CPP_excited_gpu4pyscf` use the newer ones.
  - `simple_mlmd_sampling_task`, or `reactive_sampling`;
  - the ORCA, VASP, and ASE labelers;
  - `train_HIPPYNN_ensemble_task`, `fine_tune_HIPPYNN_ensemble_task`, or
    `train_ANI_model_task`;
  - the original builders.
- The only deliberate config change is `"distcut": 0.7` in four water
  examples and IL, made at the user's direction; the code default stays 1.2 Å.
  Without it, every trajectory stops at its first check, because these systems
  have bonds shorter than 1.2 Å.
- The first smoke matrix ran during Increment 7. Simple water and fine-tuning passed. The user accepted the other cases as validation gaps at the close-out, not as passes; see [`batching.md`](batching.md#increment-7--closed-by-user-acceptance-2026-11-05). It hasn't run since, including the planned run after E5.
- `tests/test_examples.py` loads every tracked example's configurations and checks their tasks on each push.

**Risk by planned step.**
- **Low:** teacher labeling and the selector interface. They add new tasks or opt-in branches.
- **Multi-node training:** built and opt-in, so the risk is low. Every HIPPYNN example still uses a single-node task, whose behavior didn't change, except that ALF now releases allocations on exit.
- **Selected-state UDD and wells:** low for the original examples, because they only lift a check in excited-state mode. Five examples set `well_params`: IL, molten salt, simple water, fine-tuning, and seeded active learning. Leave `MLMD_calculator` and `mlmd_sampling` unchanged, or prove formula parity if code must be shared.
- **Staged campaigns and recovery:** keep ordinary runs as the default; do not
  port the fork's `run_control` changes to `__main__.py` wholesale.
- **Fresh-install policy:** newer dependencies can break examples without any
  ALF change. ANI/NeuroChem and VASP paths are the least exercised.

**Planned checks.**
1. **Compatibility smoke matrix.** One bounded real-coordinator run per tracked example family, after Group B and after multi-node training.

   Water, fine-tuning, and seeded active learning run with ORCA on Darwin.
   IL, molten salt (VASP), UO2 (ANI), and reactive sampling need their
   backends. Record a missing backend as "unverified, backend unavailable",
   never as passed.
2. **UDD/well parity — completed November 6.** Selected-state forces matched the
   legacy `MLMD_calculator` well and UDD forces, including active translated
   geometries with the examples' `r_start` values 3.0 and 10. New bias methods
   still require their own checks.

Existing ordinary workflows must not acquire a new fixed-topology requirement when screening is disabled.

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
