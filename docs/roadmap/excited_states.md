# Excited-state roadmap

The E1-E6 plan and Increments 8-11B: excited-state contracts, labeling, training, sampling, and the AcAc and CPP work. See the [roadmap index](README.md) for current status and standing rules.

## Delivered increments

Moved from [`README.md`](README.md) on November 6, 2026. **Merged** means in official `lanl/ALF:development`. Increment 11B is still in progress; see [Increment status](README.md#increment-status).

| Increment | Status | Deliverable | Acceptance gate |
| --- | --- | --- | --- |
| 8. Excited-state contracts and labeling (E1–E2) | merged (`401fc89`) | State/gap contracts, topology-aware NAC storage, GPU4PySCF TDA labels, and native PySEQM CIS batching. | Backend qualification gates passed; see [Increment 8](#increment-8--completed-excited-state-contracts-and-labeling-2026-11-05). |
| 9. Multi-state training and external geometries (E3–E4) | merged (`7c24cb2`, live remote verified 2026-11-05) | Shared-trunk state heads and force gradients in the existing HIPPYNN interface; unlabeled XYZ/HDF5 geometry builder. | 402 library tests pass; real 32-frame import/label/store/four-member GPU training and reload passed. |
| 10. Selected-state sampling and acquisition (E5) | merged (`a5090e0`, `6c2751d`, `1c367ba`) | Multi-state inference, state selection and grouping, and named gap uncertainty across ASE, TorchSim CUDA, and ALCHEMI. | Focused compatibility/parity tests; real widths one/four on each engine; one shared small acquisition/label/store/retraining qualification. |
| 11A. Chemical topology screening and fixed-keto AcAc ALF | merged (`4a1caba`) | Opt-in reference-based chemical topology screening across ASE, TorchSim, and ALCHEMI; an ordinary AcAc ALF loop. | Screens passed on all three engines; newly sampled labels entered a promoted four-member ensemble. |

## Delivered capabilities

Moved from [`future.md`](future.md) on November 6, 2026. LCM gap seeking is the one part still open; see [Group B](future.md#group-b-biased-dynamics-on-excited-states).

| Capability | What it covers | Important boundary | Where it landed |
| --- | --- | --- | --- |
| Molecular excited states | PySEQM/GPU4PySCF labeling, multi-state HIPPYNN training, derived gap targets, state selection, gap diagnostics, and LCM. | Preserve per-state convergence, state/atom mappings, physical screening, and the distinction between predictions and propagation controls. | Increments 8-10 |
| RDKit initialization | Reference-centered KDG geometry generation, seed-specific screening, and FPS selection, with reproducible settings/provenance. | Optional preprocessing/builders; unlabeled seed geometries are not labeled training rows. | Increment 11A, `builders/rdkit_molecule.py` |
| Geometry-only HDF5 replay | Reuse stored geometries for sampling or relabeling, with original identities and source provenance. | Do not accidentally carry old labels into a newly requested labeling operation. | Increment 9, `builders/external_geometry.py` |
| Optional additional labels | Preserve optional nonadiabatic-coupling labeling and storage, including state-pair and atom-axis mappings. | Computing/storing NACs does not automatically make them energy/force training targets. | Increment 8 |

Also delivered from the later plan:

- Physical-force and chemical topology screening: Increment 11A, `tools/force_screen.py` and `tools/molecular_topology.py`.
- Gap-aware summed acquisition, named signals, and configured eligibility: Increment 10, `SCORE_TERMS` and `acquisition.eligibility`.

## Excited-state ALF plan (E1–E6)

Each step is a separate local change with its own CPU tests. Existing
configurations, task signatures, and ground-state results stay unchanged.
E1–E5 are complete, and Increment 11A qualified topology/RDKit and keto AcAc ALF.
Increment 11B is working on E6.

| Step | Deliverable | Acceptance gate |
| --- | --- | --- |
| E1. Contract and storage — complete in Increment 8 | Port the state/gap key contract, `atom_order`, and the `pair_atomic`/`nac_pairs` storage scope. | Multi-state and NAC HDF5 round-trip tests; atom order restored; existing `properties_list` behavior unchanged. |
| E2. Excited-state labelers — complete in Increment 8 | GPU4PySCF TDA as an ordinary `QM_task`, with single and sequential packing (no native batch). Promote cpp9's transition dipoles and oscillator strengths into the interface. PySEQM CIS single and composition-grouped native batches. | Mocked CPU tests and a CPU PySCF comparison; then real A100 GPU4PySCF labels with the cpp9 label checks (shapes, finiteness, offset applied once, oscillator strengths ≥ 0, NAC pair order). |
| E3. Multi-state training — complete in Increment 9 | Separate excited-state `ML_task` in the existing `hippynn_interface.py`, sharing its opt-in `ALFPyAniDirectoryDB`; derived gap losses and fixed-composition nonperiodic data. | Four GPU-trained members reload with exact input/split manifests; ground-state trainer unchanged. |
| E4. External initial geometries — complete in Increment 9 | `external_geometry_builder_task` imports unlabeled XYZ/extxyz or geometry-only HDF5 with sequential or deterministic-hash selection. The random ring generator stays outside ALF. | Identities, atom order, and provenance survived real labeling inputs; source labels/calculators are stripped. |
| E5. Sampling on a chosen state — complete in Increment 10 | Multi-state evaluator output (per-state member arrays; existing ground-state shapes remain the default). A `selected_state` from metadata or `fixed`/`batch_cycle`, with a selected-state batch key. Per-state diagnostics and a named gap signal with its own cutoff as a `SCORE_TERMS` entry. Eligibility reads configured signal/cutoff pairs; `Es`/`Fs`/`Fsmax` keep their meaning. | CPU parity with fakes across ASE, TorchSim, and ALCHEMI; a bounded molecular excited-state loop through the real coordinator. |
| E6. cpp ring full ALF example | After the chemical topology prerequisite passes, run an ordinary master configuration through `alframework.__main__`: external ring bootstrap, GPU4PySCF labels with stored NACs, multi-state training, selected-state MLMD/acquisition, and iteration. Reuse the fork's scientific settings rather than its standalone driver. | Opt-in chemical topology screening is enabled and qualified; a bounded real campaign selects new ring structures, labels/stores them, and promotes a retrained, loadable four-member ensemble. |

E6 is the first point where dev can do the cpp ring work, so it follows
sampling (E5) rather than preceding it. Open decisions:
- resolved in Increment 8: PySEQM CIS uses native composition-grouped solves,
  each isolated in a managed child process;
- answered by the E6 pilot configuration: the ring campaign cycles through S0–S3;
- answered by the E6 pilot configuration: its first campaign leaves gap-uncertainty
  ranking and gap loss off.

## Fork source audit (2026-11-03)

This records the source audit before Increments 8 and 9. The dev gaps listed
here are historical; current completion and remaining work are recorded in
the increment sections and the E1–E6 table above.

- **The cpp9 example (`examples/cpp9_gpu4pyscf`) is a standalone trial, not
  an active-learning loop.**
  - `run_trial.py` and `cpp_trial/` drive two Parsl sessions:
    1. label a fixed pool of 1,000 perturbed geometries with GPU4PySCF
       (CAM-B3LYP/6-31G, RKS/TDA, `nroots=3` giving S0–S3, analytic
       gradients, and NACs for pairs (1,2), (1,3), (2,3));
    2. train four multi-state HIPPYNN members, then exit.
  - There is no sampler, no acquisition, and no `alframework.__main__`.
  - `cpp_trial/qm.py` is a local copy of the core QM interface that adds
    transition dipoles and oscillator strengths.
  - Its audit (`docs/cpp9_example_audit_2026-10-21.md` on the fork) records
    CPU tests with mocks and one CPU PySCF comparison only. Real GPU QM and
    training were not validated there, and rejected frames are not refilled.
- **Self-contained excited-state core.** These depend on ordinary task
  entrypoints, `molecules_class`, `pyanitools`, and each other, not on the
  fork's staged workflow, run control, or sampling dialect:
  - `tools/excited_state_tools.py`: the `sE#`/`F#`/`dE#` key contract;
    the state count is inferred from contiguous keys in `properties_list`.
  - `tools/atom_order.py`.
  - The `pair_atomic` storage scope and `nac_pairs` index in
    `tools/tools.py::store_current_data`.
  - `qm_interfaces/gpu4pyscf_interface.py`: `run_gpu4pyscf_states`,
    `gpu4pyscf_excited_state_task`, singlet RKS/TDA, and optional NACs.
  - PySEQM AM1 CIS through `pyseqm_excited_state_task`, run as one molecule
    per child process.
  - `ml_interfaces/excited_state_hippynn_interface.py`:
    - one shared trunk with an energy head and force gradient per state;
    - optional derived gap targets;
    - its own explicit HDF5 loader;
    - `train_excited_state_HIPPYNN_ensemble_task`;
    - nonperiodic only.
  - `builders/h5_replay_builder.py`.
- **Excited-state sampling exists only in the fork's ALCHEMI sampler**
  (`samplers/alchemi_sampling.py`, `model_mode='excited_state'`).
  - The propagated state comes from `selected_state` metadata or from
    `state_selection` `fixed`/`batch_cycle` (`tools/sampler_batching.py`).
    Every member of a batch shares one state.
  - Uncertainty uses `Es`/`Fs`/`Fsmax` on the propagated state, with
    per-state and `dE##_stdev` diagnostics.
  - Summed scoring has an optional gap-uncertainty term.
  - LCM `gap_seeking` permanently switches a triggered replica to the
    smoothed surface.
  - The fork's ASE sampler has no excited-state path. Its full-loop examples
    (`excited_state_gpu4pyscf`, `_nacr`, `excited_state_pyseqm`) also rely on
    the fork's `run_control` and staged workflow.
- **What dev lacks.**
  - There is no GPU4PySCF interface.
  - `pyseqm_parameters` rejects `excited_states`.
  - `train_HIPNN_model` fits one energy.
  - The evaluator contract assumes one energy per system: `energies [M, S]`,
    `HippynnBatchedEnsemble`, `PackedMLMDEngine._evaluate`/`observe`,
    `ensemble_statistics`, and `uncertainty_flags`.
  - Existing hooks:
    - the `SCORE_TERMS` docstring anticipates a gap term;
    - `default_batch_key` anticipates a selected-state key;
    - storage writes only keys listed in `properties_list`, so new keys are
      opt-in.
  - Per-state keys are atom-first (`F#` is `[atoms, 3]` per frame), so the
    existing `"atomic"` sorting handles them. Only NACs need the new scope.

## Increment 8 — completed: excited-state contracts and labeling (2026-11-05)

The user approved E1–E2, including native PySEQM CIS batching. Both backend
qualification gates passed. E3–E6, training, dynamics, cpp9 execution, and
user-guide updates remain outside this increment. Increment 7 acceptance is
preserved; its campaigns were not repeated.

CI follow-up (2026-11-05): the first `development` run of the fresh-install
workflow failed one test in the `base` (torch-free) job.
`test_fresh_worker_initializes_selected_gpu_before_resetting_allocator_stats`
imported `torch` directly; it now uses `pytest.importorskip('torch')` like the
other torch-dependent tests (`2f25c51`, local). A fresh torch-free venv runs
378 passed and 19 skipped; the test still passes in `atomistic`. This is a test
gating fix, not a behavior change.

- **E1:** dependency-free contiguous state/gap contracts, explicit topology
  identity helpers, and opt-in `pair_atomic` NAC storage with a constant
  `nac_pairs` index. Tagged frames preserve same-element identities through
  species sorting; ordinary untagged ordering is unchanged. Invalid or
  inconsistent pair mappings are rejected before writing.
- **GPU4PySCF:** an ordinary single-molecule QM task with sequential packing,
  RKS/TDA energies and analytic forces, optional gaps, excited–excited NACs,
  transition dipoles, and oscillator strengths. Offsets apply once. Backend
  execution uses ALF's selected local device via upstream's single-device
  configuration hook, without rewriting visibility masks. Previously imported
  multi-GPU mode is rejected. The new `tda_lindep` setting defaults to 1e-20
  so corrections needed for the 1e-8 residual tolerance are retained.
- **PySEQM:** separate single/native AM1/CIS tasks, managed child-process
  timeouts and reaping, explicit SCF/CIS convergence, atom/order restoration,
  and per-member exception retries. Existing ground-state tasks are unchanged.
  Both old `52850b45` and current `66021961` reject all-state analytic CIS
  forces for mixed compositions. The user approved composition-grouped native
  solves with original submission order restored. Reports distinguish submitted
  width, compatibility groups, actual backend widths, retries, and failed IDs.
  True mixed-composition native CIS forces remain deferred.

**Accepted qualification.** The final focused suite passed 124 tests, including
existing storage/ground-state labeling regressions, state/atom/pair mappings,
optional-dependency errors, child cleanup, and qualification deadlines/owned-job
cleanup. Real NVIDIA checks passed on the fixed small formaldehyde/ethene
fixtures, requesting S0–S3:

- PySEQM homogeneous native widths 1/2/4/8, and a mixed request split into two
  width-four solves, matched singles within 2.2e-13 eV and eV/Angstrom.
  Maximum finite-difference error was 1.44e-6 eV/Angstrom. Single,
  sequential-four, and native-four Parsl labeling through HDF5 storage passed;
  a current-version ground-state compatibility check also passed.
- GPU4PySCF CPU/GPU errors were 9.82e-7 eV and 6.23e-5 eV/Angstrom, below
  the required 1e-5 and 1e-4 limits. Maximum representative finite-difference
  error was 1.34e-5 eV/Angstrom, below 1e-3. NAC pair order, finite optical
  outputs, nonnegative oscillator strengths, and actual single/sequential-four
  Parsl labeling through HDF5 storage passed. Each solve recorded single-device
  backend execution with four visible GPUs.

CPU PySCF lacks analytic DF-TDDFT gradients. Its independent force reference
therefore central-differences energies from the complete small TDA response
matrix, with the same functional, basis, density fitting, and numerical grid.
Qualification uses grid level 5 on CPU and GPU to reduce quadrature-response
error; the production grid default remains unchanged. No acceptance tolerance
was relaxed. Earlier incomplete attempts remain preserved as incomplete.

New isolated environments are under `ALF_LANL/environments/20261105/inc8/`:
GPU4PySCF 1.9.1, PySCF 2.15.0, CuPy 14.2.0, Torch 2.15.0 CUDA, and official
PySEQM master `66021961`. Both environments passed dependency consistency
checks and retain exact resolutions. Existing environments/checkouts were not
modified. Core implementation commits are `49f6c6e`, `fc7118e`, and `f89b722`;
focused verification fixes are recorded in subsequent local commits.

The consolidated accepted evidence is
`/vast/home/pjlohr/ALF_LANL/validation/inc8_20261105/qualification-result.json`.
It references the accepted PySEQM result in `run-001/`, GPU4PySCF result in
`run-005/`, exact source/dependency fingerprints, numerical errors, HDF5 hashes,
and all six verified owned-job cleanup reports. Qualification including retries
finished within 27 minutes of first preparation, inside the original two-hour
queue-inclusive deadline. Each allocation retained its 30-minute limit.

Increment 9 completed E3 multi-state training and E4 external initial geometries;
the E1–E6 sequence above remains the feature roadmap. No increment 8 training,
dynamics, cpp9 campaign, NeuroChem testing, or user-guide update is required.

## Increment 9 — completed: multi-state training and external geometries (2026-11-05)

User-approved boundary: fixed-composition nonperiodic molecular training and
a small integrated PySEQM CIS qualification. E5 dynamics/acquisition and E6 cpp
remain later work. All excited-state training code lives in the existing
`ml_interfaces/hippynn_interface.py`, as requested; there is no separate
excited-state interface or database module.

- **E4:** `builders/external_geometry.py` adds the single/multi-structure
  `external_geometry_builder_task`. Ordered XYZ/extxyz and geometry-only HDF5
  sources support stateless deterministic-hash selection or exhaustive
  sequential selection. It preserves geometry, cells/PBC, supplied topology
  IDs, and source provenance; it carries no source labels/calculators.
  Sequential exhaustion is explicit. Builder acceptance: 19 CPU tests passed.
- **E3:** `train_excited_state_HIPPYNN_ensemble_task` preserves the ordinary
  ML task arguments, adds `properties_list`, and returns the standard
  `(completed, training_id)` with `model-XX` directories. HipNN/HipHopNN trunks
  share per-state energy heads and analytic force gradients. Gap losses are
  derived upper-minus-lower differences. NAC/optical properties are not trained.
- The existing `ALFPyAniDirectoryDB` gains opt-in schema-directed reading,
  explicit topology canonicalization, and retained-row checkpoint restarts.
  Its ground-state defaults and trainer are unchanged. Requested state/gap
  datasets are validated before float32 conversion; unrelated NAC datasets
  are never read. Manifests retain file hashes, original IDs, filtering, and
  train/valid/test membership.
- The new trainer uses spawn workers and visible local GPU ordinals by member
  ID without rewriting scheduler masks. Member failures remain unsuccessful
  completion flags; owned children are joined/terminated on error. It refuses
  nonempty output directories.
- Focused implementation checks: **123 passed**, including actual CPU HipNN
  and HipHopNN training, checkpoint/database reload with identical filtering
  and splits, force finite differences, gap graphs, topology permutations,
  upstream filtering parity, and ground-state loader/storage regressions.
- Qualification lives exclusively in the separate `alf_validation` repository,
  at `examples/excited_state_training/`. It imports 32 formaldehydes from XYZ
  and geometry-only HDF5, labels S0–S3 and adjacent gaps with eight native
  width-four PySEQM submissions, stores all IDs, then trains four HipHopNN
  members for five epochs with 24/4/4 splits. Completion requires four loadable
  members on distinct local GPU ordinals, finite metrics/predictions, derived
  gaps, force finite differences <= 1e-3 eV/Angstrom, and exact ID traceability.
- A new frozen training environment was installed at
  `/vast/home/pjlohr/ALF_LANL/environments/20261105/inc9-training-cuda`:
  HIPPYNN 0.1.3, Torch 2.15.0+cu132, NumPy 2.5.4, Parsl 2026.11.2;
  dependency compatibility passes. The accepted Increment 8 PySEQM environment
  and official revision `66021961` remain frozen and are reused.
- Real qualification **passed** on ml4chem cn4072, with four NVIDIA RTX A6000
  GPUs, explicit interpreters and GCC 11. Two sequential stages, including a
  training-only retry, completed within the original two-hour queue-inclusive
  deadline. Each allocation retained its 30-minute limit. Generated evidence
  stays under `validation/inc9_20261105/`, outside Git.

First real attempt: all 32 geometries converged and stored, with eight actual
native width-four solves on ml4chem cn4072 (job 17378861). Training job 17378867
failed before the first optimizer step: HIPPYNN 0.1.3's Triton config pruner
returns a generator, while Triton 3.9.0 expects a sized list. Both jobs are
verified terminated. The new trainer now explicitly defaults to HIPPYNN's
public `pytorch` kernels, records the active implementation, and permits an
explicit `hippynn_kernel` override; ground-state training is unchanged.
No upstream source or frozen dependency was patched or downgraded. All 67
HIPPYNN tests pass with GCC 11 (PyTorch's compiled path requires C++20).
A training-only retry reused fingerprint-verified accepted labels and
the original queue-inclusive deadline; the failed attempt remains incomplete.

Accepted results:

- **Import/label/store:** `run-001/label/result.json`, job **17378861**:
  all 32 distinct input geometries (16 XYZ and 16 geometry-only HDF5) passed
  identity/order checks, converged, and stored with S0–S3 forces/energies and
  adjacent gaps. Eight submitted tasks and all actual native solves had width
  four; there were no labeling retries. Stored HDF5 SHA256 is
  `ee74b7719db64279dbade4e895121c20622eac6bae53dd9b28905a7591722641`.
- **Train/reload:** `run-003/train/result.json`, job **17378910**:
  the production Parsl ML task trained four HipHopNN members for **five epochs**
  on **cuda:0–3**, with changed parameters and finite metrics. Each member's
  input manifest traces all 32 stored IDs into **24/4/4** train/valid/test
  splits. All four checkpoint/database reloads reproduce those splits and
  produce finite state forces/energies and consistent derived gaps. Maximum
  representative force finite-difference error across the four states/members
  is **1.65e-6 eV/Angstrom**, below **1e-3**.
- **Regression checks:** final fresh training-environment suite **402 passed,
  42 skipped**; external qualification tests **12 passed**. The new trainer's
  explicit PyTorch kernels and C++20 compiler were used for final CPU checks.
  Skips cover unavailable optional dependencies/hardware; they are not passes.
- **Source/resources:** accepted ALF code is `89aeff2`, assembled with
  `alf_validation` `74a4a99`; all current `alframework` source hashes match the
  accepted frozen snapshot. The accepted labeling source remains `7885ee3`.
  Source/config/environment fingerprints and the original failed attempt are
  preserved. All three owned jobs (**17378861**, **17378867**, **17378910**)
  are verified terminated; only those IDs were queried/cancelled by cleanup.
- The consolidated report is
  `/vast/home/pjlohr/ALF_LANL/validation/inc9_20261105/qualification-result.json`.
  Qualification including the repair/retry took **16.4 minutes** from first
  submission, including queue waits, within the original two-hour deadline.
  `run-002` was an unsubmitted preparation and contributed no acceptance gate.

Increment 9 is complete. This qualifies fixed-composition nonperiodic training
with HIPPYNN's public PyTorch kernel path, not HIPPYNN's incompatible automatic
Triton adapter. It adds no potential-accuracy, dynamics, acquisition, cpp-ring,
MI350, NeuroChem, multi-node training, or user-guide gate. Next work is E5;
E6 follows once selected-state sampling and acquisition are qualified.

## Increment 10 — completed: selected-state sampling and acquisition (E5, 2026-11-05)

User decision, November 5: qualify ASE, TorchSim CUDA, and ALCHEMI with short
real sampling checks at widths one and four, followed by one shared small
end-to-end qualification. This extends the accepted Increment 9 models and
data; it does not repeat the Increment 7 matrices or Increment 8 labeler
qualification. The initial excited-state scope remains fixed-composition,
nonperiodic molecules. The implementation and acceptance boundary are recorded below.

Implementation and real qualification are complete. The three main code
commits are `a5090e0` (multi-state inference), `6c2751d` (state propagation and
identity preservation), and `1c367ba` (gap eligibility and audited acquisition).
All HIPPYNN-specific code remains in the existing `hippynn_interface.py`.
Ground-state shapes/defaults and production task signatures remain compatible;
state metadata and topology IDs now survive cohort reconstruction.

Accepted evidence, outside Git:

- Consolidated report:
  `/vast/home/pjlohr/ALF_LANL/validation/inc10_20261105/qualification-result.json`.
  Real artifacts are under `run-002/`; source/library fingerprints match
  `1c367ba`. The accepted external harness is `a74754d`, with delivery at
  `fb57258` adding stricter retry-environment checks and tests; its scientific
  functions and the shared training qualifier match the accepted snapshot.
- **Six sampling cases passed:** ASE, TorchSim CUDA, and ALCHEMI each executed
  four width-one tasks and one width-four task on ml4chem cn4073 (four NVIDIA
  RTX A5000 GPUs). Each four-trajectory cohort selected two frames: **twelve
  new frames**, four per engine, acquired after step 21 (5.25 fs physical
  propagation; the acquisition check grid reports 0.005 ps). Gap uncertainty
  exceeded its fixed cutoff and contributed positively to every selected
  frame's summed score. No cutoff was tuned after execution.
- **Label/store passed:** job **17379064** made three actual native width-four
  PySEQM CIS solves; all twelve SCF/CIS calculations converged, with no retries
  or failed IDs. All twelve new labels stored alongside the unchanged 32-frame
  bootstrap dataset, giving **44 retained frames**.
- **Retrain/promote passed:** job **17379065** trained four members for **five
  epochs on cuda:0–3**, with changed parameters and finite metrics. Every
  member retained all 44 frames, split **34 train / 5 valid / 5 test**. All
  twelve sampled IDs occur across the members' actual training splits. Four
  checkpoint/database reloads reproduced their splits; the promoted ensemble
  reloads through the new inference loader. Maximum representative state-force
  finite-difference error is **3.70e-6 eV/Angstrom**, below 1e-3. The coordinator
  status records promoted model generation **1**.
- **Parity/regressions passed:** common recorded predictions agree between
  single and packed inference, and across all three engines, at the approved
  raw-member tolerances with measured state/gap-spread roundoff bounds. Raw
  arrays and `cross-engine-parity.json` are retained. Final library suite:
  **409 passed / 44 skipped**. External qualifier tests: **22 passed**.
  Focused packed-state CPU observation/compaction checks passed separately in
  the frozen TorchSim CUDA and ALCHEMI environments. Optional skips are not
  hardware passes.
- Accepted sampling jobs **17379053**, **17379056**, **17379060**, labeling job
  **17379064**, and training job **17379065** all completed and are verified
  terminated. Sampling/labeling used local GPU zero; training used all four
  local ordinals, preserving inherited masks. ml4chem declares these GPUs as
  scheduler features rather than GRES, so each stage reserved one exclusive
  four-GPU node; allocation stages ran sequentially.
- An initial harness source audit counted generated supervisor bytecode as
  source changes. Jobs **17379040**, **17379042**, **17379044**, **17379046**
  failed before scientific work; all are verified terminated. The corrected
  harness ignores generated bytecode, still fingerprints source, and preserves
  accepted stages/environments and the first deadline on verified retries.
  The unsuccessful attempt remains under `run-001/`.
- Total first-execution-to-cleanup time, including the repair/retry and queue
  waits: **609.5 seconds (10.2 minutes)**, within the original two-hour limit.
  Slurm accounting records all nine owned jobs; `squeue` has no remaining rows
  for those IDs. Only owned resources were queried/cancelled.

This completes E5 workflow/mechanics qualification for fixed-composition,
nonperiodic molecules. It does not establish potential accuracy or long-time
stability. No cpp-ring campaign, GPU4PySCF requalification, surface hopping,
UDD/wells/LCM extension, multi-node training, or user-guide work was required.
Increment 11 (E6) is next.

The qualification profile is fixed by user decisions: molecular cutoffs
`Escut=0.001 eV`, `Fscut=0.01 eV/Angstrom`, and S1/S2 gap-spread cutoff
`0.01 eV`; gap uncertainty triggers and contributes to summed ranking. Require
sampled labels from every engine. State cycling rotates per admitted trajectory,
independently of execution width; metadata overrides consume an ordinal.
Six sampling cases use four-trajectory cohorts with top-two selection. Their
combined selections feed one native PySEQM label/store/four-member retraining
stage. This gives at most twelve new labels; at least eight must pass, with
all engines represented in retained inputs and across actual training splits.

The implementation follows these three approved changes:

1. **Multi-state physical inference.** Extend the existing
   `ml_interfaces/hippynn_interface.py`, with shared prediction/statistics
   helpers where appropriate. Load the trained state heads and expose ordered
   per-state member energies and analytic forces. Validate state availability,
   shapes, finiteness, member order, and supplied topology identity. Preserve
   the existing ground-state return shapes and default behavior. Derive gaps
   from each member's upper-state minus lower-state energies, then compute
   their ensemble spread; do not combine independent state spreads to estimate
   gap uncertainty. Select and record the qualified public PyTorch kernel path
   explicitly for the excited-state loader, including inference workers.
2. **Selected-state propagation.** Add opt-in state selection to the shared
   multi-trajectory path and its ASE, TorchSim, and ALCHEMI adapters. Explicit
   valid `selected_state` metadata takes precedence over configured `fixed` or
   `batch_cycle` selection. Resolve the state before buffering/submission and
   preserve it in trajectory/candidate metadata. Cycling uses a recorded
   admission sequence, independent of task packing or completion order. Each
   task shares a selected state; the sampling batch key includes that state
   and the existing PBC/model-generation grouping. Propagate the selected
   state's mean physical force and retain per-state diagnostics. Invalid or
   missing state heads fail explicitly. Keep legacy ground-state entrypoints
   and task signatures compatible. Surface hopping and biased propagation
   belong to later work.
3. **Named acquisition signals.** Keep `Es`, `Fs`, and `Fsmax` as diagnostics
   on the propagated state, with unchanged default cutoffs/flags. Add explicit
   opt-in signal/cutoff eligibility and a named gap-uncertainty `SCORE_TERMS`
   entry with its own positive cutoff and explicit state pair. Unconfigured
   ground-state eligibility and ranking remain identical. Gap-aware ranking
   alone must not change eligibility. Record the configured pair, signal,
   cutoff, eligibility flags, and ranking contribution on selected frames.

Focused checks belong in the existing ALF test files for the affected modules.
Cover actual trained-model single/packed prediction parity, representative
force finite differences, state/gap axes, identical-element topology
permutations, fixed/cycled/metadata selection, state-aware buffering,
independent stopping and packed compaction, GPU visibility, named signal
validation, and unchanged ground-state statistics/selections. Use fake
predictions for deterministic engine/threshold/selection tests. Compare common
recorded structures using the approved member tolerances and uncertainty
roundoff bounds; identical stochastic trajectories or near-cutoff decisions
across engines are not acceptance requirements.

Real qualification remains in `alf_validation`, with preparation-only support
and separate engine selection:

- Reuse fingerprint-verified copies of the accepted Increment 9 closed HDF5
  inputs and four-member model. Keep originals frozen. Start with clean
  coordinator state and no inherited pending work.
- Run six short sampling checks: ASE, TorchSim CUDA, and ALCHEMI, each with
  `driver.strategy="single"` at width one and `"batched"` at width four.
  Record actual executed widths. Require a successful four-trajectory task
  for each engine. Recommended small-molecule profile: S1 propagation and
  S1/S2 gap diagnostics, with fixed/cycled state grouping also exercised by
  focused tests. Freeze trajectory bounds and scientific settings before
  submission; short-run mechanics do not establish potential accuracy.
- Follow with one shared small qualification through the production
  `SamplerCoordinator`, `LabelingCoordinator`, storage, and excited-state ML
  task. Use PySEQM CIS native width-four labeling and one four-member
  retraining. Target at least eight distinct genuinely sampled IDs, traceable
  through converged labels and stored frames into the promoted ensemble's
  retained training-input manifests. Require real training epochs, changed
  parameters, and four valid loadable output members. Bootstrap/replayed input
  IDs cannot satisfy the sampled-ID target. No separate retraining per engine
  is required. Record the engines contributing the sampled IDs.
- Declare acquisition cutoffs before launch; do not tune them after seeing
  results to force the acquisition target. Missing natural acquisition,
  missing executed widths, backend failures, and deadline exhaustion remain
  explicit incomplete gates. Retain passed stages on fingerprint-verified
  retries under the original deadline.
- Reuse qualified frozen environments when compatible; any additional
  environment is isolated under `ALF_LANL/environments/`, resolved and frozen
  before execution. Use explicit interpreters, pinned source snapshots, the
  qualified kernel selection, and GCC 11 where required. Keep personal
  environments and upstream checkouts unchanged.
- Run allocation stages sequentially, with 30-minute GPU allocations and a
  two-hour overall queue-inclusive deadline including cleanup. Training may
  use one four-GPU allocation; other stages need only their bounded sampler
  or labeler resources. Supervise from a persistent Darwin session; journal
  owned job IDs, clean up only those jobs, and verify termination.

Store generated configurations, labels, logs, models, timings, and fingerprints
outside Git. Commit production code/tests and roadmap updates locally on
`development`; commit the harness separately in `alf_validation`. Never push.
Increment 10 completes only when the focused checks, all six real sampling
checks, and the shared sampled-label-retraining qualification pass. It has no
cpp-ring, repeated water campaign, GPU4PySCF requalification, MI350, NeuroChem,
long-time stability, multi-node training, or user-guide requirement.

## Increment 11A — completed: keto AcAc ALF (qualified November 5; reviewed November 6)

Accepted evidence: `/vast/home/pjlohr/ALF_LANL/validation/inc11a_20261105/run-003/`,
particularly `result.json`, `pilot/verification.json`, `pilot/accepted.json`,
and the model training/input manifests. The frozen pilot used library revision
`e919472` and external harness `a244a15`; prerequisite engine/seed evidence was
retained from `run-002` after fingerprint verification. No cutoffs were tuned.

- ASE, TorchSim CUDA, and ALCHEMI each passed real widths one/four and
  invalid-input isolation. RDKit produced 1,920 accepted geometries and 192 FPS seeds.
- All 192 bootstrap frames converged and were stored. Model generation 3 retained
  26 distinct propagated sampled IDs across all four members: S0/S1/S2/S3 counts
  2/8/8/8. All four states contributed to the ensemble's actual training splits;
  this does not require every member's split to contain every state.
- All four initial and promoted members completed real training, changed parameters,
  and reloaded. The accepted generation ran 300/300/186/279 epochs. Representative
  state-force finite differences had maximum error 5.846875e-7 eV/Å (limit 1e-3).
- All 230 completed labeling evaluations converged; 224 frames came from actual
  native-width-four solves, and six from partial width-two solves. Stored frames
  totaled 218 (192 bootstrap + 26 sampled); 12 additional converged sampled labels
  were left unstored at shutdown and do not count toward acceptance.
- Runtime screening rejected 65 completed trajectories: 57 topology violations
  and eight physical-force violations. Previously valid candidates remained eligible
  under the cohort policy. Four sampler tasks failed when allocation `17379369`
  reached its 30-minute limit; subsequent work recovered and all gates passed.
- The coordinator reached acceptance November 5 at 18:19 Mountain, about 76.6 minutes
  after the first submission, within the original four-hour queue-inclusive budget.
  All owned jobs from the pilot and earlier launch attempts are verified terminated;
  a November 6 live-queue check found no remaining qualification allocations.
- After merging upstream documentation/test cleanup through `b995c15` as `fea5f5a`,
  the available full suite passed: 491 passed, 42 skipped (optional packed engines
  absent from the CPU test environment). Executable library ASTs matched the
  qualified snapshot. Log: `upstream-development-20261106-pytest.log` in the evidence root.

Increment 11A acceptance is complete. The launch/progress entries below are
historical records. Production width 50, 50,000-to-5,000 preparation, potential
accuracy, and long-time stability are not qualified by this bounded pilot.

User-approved scope, November 5: place a fixed-keto AcAc stage before cpp.
Shared runtime screens (`0844678`), the prepared RDKit/FPS replay builder and
finite bootstrap (`49705b0`), and state-specific cohorts/friction/production
profiles (`d241a1b`) are implemented. The opt-in harness lives separately in
`alf_validation/examples/acac_qualification/`, not in ALF.

The topology checker requires explicit atom identities, preserves caller atom
order, and checks reference bond windows 0.85–1.50 and unexpected connectivity
at 1.25 times summed covalent radii. The independent physical-force cutoff
uses the maximum atom-vector norm of the ensemble-mean unbiased force over
all available states. Both screens operate before integration and at existing
observation points. Earlier valid candidates retain their own check metadata.
Existing `distcut` timing remains unchanged. Runtime screens are opt-in and
nonperiodic fixed-connectivity chemistry is required for topology checking.

`examples/keto_acac/` ships the fork's six-level reference-centred KDG/FPS
50,000-to-5,000 preset, keto/enol references, AM1/CIS labels and offset,
four-member HIP-HOP145 architecture, fork training settings (300 epochs,
patience 50), and AcAc dynamics/scientific cutoffs. Production sampling width
50, capacity 200, per-state cohort size 50 and top-K 500 are shipped settings;
width-four qualification does not establish production-scale coverage.
Builder tasks replay prepared seeds and subsequently closed labeled shards;
they never regenerate or rerun FPS. `finite_bootstrap` caps admitted IDs and
reports exhaustion with failed work instead of requesting nonexistent seeds.

Local evidence under `validation/inc11a_20261105/`: the available full library
suite passed (456 passed, 46 skipped), with a subsequent additional candidate
retention test passing; focused builder/finite-driver and cross-engine fake
screen tests passed. Reduced keto and enol preparations matched the fork
exactly in RDKit 2026.9.1, including coordinates, identities, strata, selected
indices and FPS ranks. Four isolated RDKit-enabled CPU/sampling environments
are frozen under `environments/20261105/inc11a/`; existing environments remain
unchanged. The external runner's five focused tests passed.

Qualification launch, November 5 at 17:04 Mountain: persistent session
`alf-inc11a-20261105` on `darwin-fe2` supervises
`validation/inc11a_20261105/run-002/`. Frozen library revision `976d323` and
external harness revision `d3f059d` are assembled in its source snapshot.
The first corrected ASE allocation `17379283` entered Python/Parsl; later
engine checks, preparation, and the ordinary AcAc pilot run sequentially.
The original four-hour deadline is retained: November 5, 21:02:30 Mountain,
with the last five minutes reserved for cleanup. The initial launch attempt
failed before Python because the harness sourced the Darwin login profile
twice under strict shell settings; this was reproduced and corrected in the
external harness. All four jobs from that attempt are verified terminated.
No real acceptance gate was claimed at launch. The user may disconnect while tmux runs.

Qualification progress, November 5 at 17:17 Mountain: all three engines passed
single/four-trajectory screening and invalid-input isolation. RDKit generated
all 1,920 accepted geometries and 192 FPS seeds. These accepted prerequisites
are retained, with code/environment/model/artifact fingerprints verified, in
`validation/inc11a_20261105/run-003/`. The first pilot stopped before QM:
a missing trailing newline concatenated `ALF_SOURCE` with Parsl's following
launcher export, so the provenance guard rejected its malformed expected
path. The actual imported ALF path was pinned. All five owned jobs in
`run-002` terminated cleanly. The production template (`e919472`) and external
harness (`a244a15`) now preserve shell export boundaries; six focused harness
tests passed. Only the fresh AcAc pilot is retried in persistent tmux, using
the original 21:02:30 Mountain deadline and unchanged scientific cutoffs.
The snapshot also includes the user's committed database-reader compatibility
fix `1db12ef`; reused inference definitions and all other library scientific
sources were verified unchanged. Job `17379315` is the retry's sampling/builder
allocation. Consult `run-003/result.json` and `pilot/owned_jobs.json` for the
historical attempt; final acceptance is recorded above.

Accepted gate, passed November 5: all three engines must execute valid widths
one/four and isolate an invalid input; prepare 1,920 accepted keto geometries
and 192 FPS seeds; use ordinary `alframework.__main__` with native-four CIS
labeling and fresh four-GPU training. Require at least 128 converged stored
bootstrap frames, eight genuinely propagated sampled IDs covering S0–S3,
and a promoted four-member ensemble retaining them with actual training-split
contributions from every state. Pilot training batch/scheduler maximum is 16;
ALCHEMI width four, capacity 16, per-state cohorts four/top-two, continue
policy, 0.2 ps maximum and 0.1 ps acquisition minimum. Cutoffs are frozen.
Require reloads, changed parameters, real epochs, and representative force
finite differences within 1e-3 eV/Å. No acceptance is inferred from CPU tests.

The first qualification submission starts one four-hour queue-inclusive budget
with five minutes reserved for owned-job cleanup. Supervise in persistent
Darwin tmux, journal exact job IDs, and verify termination. QM/sampling
allocations are capped at 30 minutes and training at one hour, at most one
allocation per executor; existing qualified Torch/HIPPYNN/PySEQM versions,
explicit interpreters, pinned source, GCC 11 and public PyTorch kernels apply.
Generated inputs/models/logs remain outside Git. Commit locally; never push.

Deferred: post-QM force/distance/topology filtering of converged labels,
GALF/LCM, staged handoffs, single-member excited-state sampling, full-scale
production qualification, and user-guide updates. Runtime force cutoffs do
not filter stored QM labels in this stage.

Documentation follow-up, done November 6: the guide now covers the held-back features,
with `examples/AcAc_excited_pyseqm` as the worked excited-state example; see
[User documentation pass](README.md#user-documentation-pass-2026-11-06).

## Increment 11B — calibration and DCM labeling qualified; cpp-ring ALF (E6) planned

Calibration scope approved November 6: brief A100 40 GB measurements before sizing the
full CPP9 ALF loop. The fork production trial accepted 1,000/1,000 CPP9 labels
(median full calculation 185.6 seconds; cold smoke 234.5 seconds), and completed
four HIP-HOP145 members. Preserve it as read-only historical evidence.

The external harness will evaluate the same smoke geometry twice through the
production GPU4PySCF task, including S0–S3 energies/forces, optics and all three
excited–excited NAC pairs, and verify an HDF5 round trip. It will also build
1,000-row and explicitly repeated 50,000-row capacity fixtures from the converged
historical labels. Measure the real 40,000/5,000/5,000 split and production
training steps at batches 1/2/4/8/16/32/64, plus evaluation at 32, on one A100.
Use two warm-up and five measured steps per case; report host RAM, GPU allocated/
reserved/device peaks, OOM limits, and epoch-time projections. Repeated capacity
rows are not new scientific labels and benchmark models are not promoted.
The current trainer streams CPU-held minibatches to the GPU; dataset-size and
batch-size effects must be reported separately.

Use frozen QM/training environments, pinned source, explicit interpreters, public
PyTorch kernels and GCC 11. Run sequential QM/training allocations under a new
two-hour queue-inclusive deadline, with 30-minute QM and one-hour training caps
and two minutes for owned-job cleanup. User steering selected `shared-redstone`
instead of `shared-gpu-ampere`; its A100-SXM4-40GB nodes match the memory target.

Calibration launch, November 6 at 09:38 Mountain: persistent frontend session
`alf-cpp9-cal-20261106` supervises
`validation/cpp9_calibration_20261106/run-001/`. Frozen library `4a1caba` and
external harness `b3cd36b` are assembled in the run snapshot. All 14 focused
harness tests passed before submission, including the exact 40k/5k/5k split
and a real coordinator/storage test with a mocked backend. The 50,000-row
fixture is 581,460,752 bytes and records original source IDs and repeat ordinals;
no new QM labels were invented by repetition. First owned QM job: `17381055`.
The following results supersede the launch status. Calibration code/tests are
in `alf_validation`; generated data/logs remain outside Git. This calibration
neither launches nor completes E6. User guides remain user-owned and untouched.


### CPP9 A100 calibration results (2026-11-06)

Calibration completed with measured capacity limits. Evidence:
`validation/cpp9_calibration_20261106/run-003/result.json`,
`calibration-summary.json`, `training-memory.csv`, and `qm/labels.h5`.
The final QM source used library `aef5d24` and external harness `5b93641`;
postprocessing is committed separately in the validation repository. Inputs,
closed data, source fingerprints and retained training evidence were verified.
All 17 focused harness tests passed; no ALF public API or user-guide change
was needed.

**Original-fork clue and dependency correction.** The fork used GPU4PySCF
1.8.0/PySCF 2.14.0/CuPy 13.4.1 with cuTENSOR 2.2.0. The accepted Increment 8
stack had GPU4PySCF 1.9.1/PySCF 2.15.0/CuPy 14.2.0 but lacked cuTENSOR.
Its CuPy fallback converged SCF/TDA and then exhausted A100 memory in the NAC
stage (178.45 seconds). The original adapter had no relaxed method, different
NAC pairs, or special force omission; its staged optical contraction was
already present in development.

An isolated environment at
`environments/20261106/cpp9-gpu4pyscf-cutensor/` adds only `cutensor-cu13==2.8.1`
to the frozen QM package set; all earlier environments/checkouts are unchanged.
Native loading additionally requires the wheel's `cutensor/lib` directory in
worker `LD_LIBRARY_PATH`: the first installation-only retry was rejected by
the strict backend guard, before calculation, because the shared libraries
were undiscoverable. The final worker verifies native cuTENSOR is active;
no fallback, scientific tolerance change, visibility-mask change, or density-
fitting offload was used. Dependency sources and comparisons are recorded in
`dependency-comparison.json` and `cutensor-environment.json` in the evidence root.

**Real QM and storage.** Two sequential evaluations of the historical smoke
geometry succeeded in one worker on `shared-redstone` (A100-SXM4-40GB):
185.03 seconds for the first call and 137.63 seconds for the repeat. These
are first/repeat calls after backend preflight; shared CUDA/compiler disk
caches were retained. All S0–S3 energies/analytic forces, absolute energies,
transition dipoles, oscillator strengths and NAC pairs (1,2)/(1,3)/(2,3) passed
finiteness/convergence and HDF5 round-trip checks, including identity and
pair/atom-axis ordering. The offset was applied once. Repeat errors were
1.53e-10 eV and 4.34e-9 eV/Å (limits 1e-5 and 1e-4). Differences from the
historical first label were 9.36e-9 eV and 1.46e-6 eV/Å. NAC signs are not
forced to agree. Sampled QM device peaks were 38.985/38.983 GiB, close to the
40 GB card's limit; this is representative-geometry coverage, not a claim
that every future geometry fits.

**Training capacity.** The 50,000-row fixture repeats 1,000 converged historical
CPP9 geometries, with unique capacity IDs and source provenance. The real
split is 40,000 train / 5,000 validation / 5,000 test. These are not 50,000 new
scientific labels, and no benchmark model was promoted. The production
HIP-HOP145 S0–S3 energy/force graph, AdamW, float32 arithmetic and public
PyTorch kernels were exercised in fresh children with two warm-up and five
measured updates. Successful training probes had finite losses and changed
parameters. NACs/optics were excluded from training targets.

| Rows | Mode | Batch | Result | Sampled GPU peak (GiB) | Median step (s) |
| ---: | --- | ---: | --- | ---: | ---: |
| 1,000 | train | 1 | passed | 5.18 | 0.1329 |
| 50,000 | train | 1 | passed | 3.01 | 0.1318 |
| 50,000 | train | 2 | passed | 7.48 | 0.1911 |
| 50,000 | train | 4 | passed | 18.57 | 0.3381 |
| 50,000 | train | 8 | passed | 27.76 | 0.7564 |
| 50,000 | train | 16 | passed | 37.39 | 1.4844 |
| 50,000 | train | 32 | oom | — | — |
| 50,000 | eval | 32 | oom | — | — |
| 50,000 | eval | 8 | passed | 27.75 | 0.1657 |

Batch 64 was deliberately not attempted after the first OOM at 32. Evaluation
at the production batch 32 also OOMed; the separate batch-eight evaluation
probe above was added to measure a feasible evaluation rate. Preserve those
OOMs as capacity findings, not harness failures. For this A100 40 GB/public-
PyTorch profile, recommend training batch eight, scheduler maximum eight,
and evaluation batch eight. Batch 16 ran but left little device headroom;
there is no production-width or long-training qualification here.

The CPU split tensors are 307,600,000 bytes (about 293 MiB), versus 6,152,000
bytes at 1,000 rows. Peak host RSS at batch one grew from 2.07 to 2.69 GiB;
the recommended batch-eight probe peaked at 2.80 GiB host RSS. Its CUDA peak
allocated/reserved values were 11.22/27.25 GiB; sampled device usage was
27.76 GiB. Thus dataset size primarily affects host RAM, while force-gradient
workspaces and allocator reservations dominate per-GPU capacity. Full-epoch
validation accumulation and checkpoint exports were not run.

At batch eight, the measured step rate projects about 63.0 minutes for 5,000
training updates plus 3.1 minutes for validation and the default 10% training
subset evaluation: approximately 66.1 minutes per 50,000-row epoch. A full
300-epoch ceiling projects about 330.7 hours (13.8 days) per member; this is
an extrapolation, not a convergence prediction. Four independent members on
four GPUs have a projected summed process RSS of about 11.2 GiB; the same
per-epoch wall-time estimate assumes no contention and concurrent scaling
was not measured. Plots/checkpoints, final testing and scheduling overhead
are outside the projection.

**Execution and cleanup.** All stages/retries used the original November 6
09:38:43–11:38:43 Mountain deadline. Final job `17381111` completed at 10:09:06,
about 30.4 minutes after the first submission. All five owned allocations
(`17381055`, `17381062`, `17381096`, `17381099`, `17381111`) have terminal
Slurm accounting and verified cleanup; the persistent session exited. Failed
QM attempts and passed capacity measurements were retained under fingerprints.
Other users' or unrelated jobs were not cancelled.

### Portable examples and DCM C-PCM qualification (2026-11-06)

**Passed labeling/storage prerequisite; E6 active learning remains planned.**
The final evidence is
`validation/dcm_pcm_20261106/run-002/result.json`,
`qualification-summary.json`, and `qualification-report.md`. The qualified
library is `f90de02`; the corrected external harness is `bf1af78`. Frozen
source/input fingerprints, retained ring evidence, and package versions were
verified after completion. The final library suite passed **507 tests, with
42 optional-engine skips**; focused GPU4PySCF checks passed 68 tests and the
external example/qualification checks passed 33 tests.

**Example and test placement.** The transferable example directories are
`examples/AcAc_excited_pyseqm` (renamed from `keto_acac`) and
`examples/CPP_excited_gpu4pyscf`. The CPP example includes minimized gas-phase
S0 references for CPP6/CPP9, their provenance and energy offsets, and CPU
preparation for other para-linked CPP sizes when a minimized reference and
explicit offset are supplied. Ring-specific preparation and bootstrap/runtime
replay policy live in that example. Its thin builder delegates geometry
reading, label stripping and atom-identity preservation to the existing
`external_geometry.py`; the recently introduced library ring builder was
removed. Sequential finite bootstrap and deterministic replay from closed
labeled shards are preserved. Newly added example/profile/portability tests
now live in `alf_validation`; the HIPPYNN four-state test uses a small
independent fixture. Earlier RDKit coverage is unchanged. HIP-HOP145,
HIP-HOP64 and HIP-NN64 are configuration choices; smaller-model GPU capacity
and predictive accuracy have not been qualified.

**Scientific interface.** The existing production GPU4PySCF task now supports
opt-in DCM C-PCM with static/optical dielectrics 9.02/2.03, Lebedev order 29,
radius scale 1.2, SWIG surface and PCM tolerance 1e-7. Qualification retained
neutral-singlet CAM-B3LYP/6-31G, density fitting, DFT grid level 3, SCF 1e-10,
TDA 1e-6 with at most 300 cycles, CPHF 1e-8 and eight CPU threads.
Equilibrium LR-C-PCM supplies S0–S3 energies/analytic forces, gaps and the
native solvent-aware pairwise NACs (1,2)/(1,3)/(2,3). These are LR-PCM labels,
not state-specific PCM energies. Existing dipole/oscillator-strength fields
remain equilibrium outputs. Separate optional `vertical_excitation_energies`,
`vertical_transition_dipoles` and `vertical_oscillator_strengths` come from a
nonequilibrium TDA solve with explicit optical dielectric and independent
vertical root ordering. The offset is applied once to state energies, never
to gaps or vertical excitation energies. Gas defaults and task signatures
are preserved. CPP preparation selects `--solvent gas|dcm` into fresh isolated
run directories; the included topology references remain gas-phase minima.
NACs and optical arrays remain storage properties rather than training heads.

**Real GPU evidence.** All six final fixture/environment cases converged
through production `LabelingCoordinator` calls and passed HDF5 round trips:

| Fixture | Medium | Calculation time (s) | Sampled GPU peak (GiB) |
| --- | --- | ---: | ---: |
| Perturbed formaldehyde | gas | 7.89 | 0.70 |
| Perturbed formaldehyde | DCM | 13.60 | 0.72 |
| CPP6 minimized S0 reference | gas | 146.33 | 29.91 |
| CPP6 minimized S0 reference | DCM | 261.81 | 14.84 |
| CPP9 minimized S0 reference | gas | 404.60 | 39.02 |
| CPP9 minimized S0 reference | DCM | 698.40 | 33.27 |

These are per-fixture measurements, not a universal labeling rate. The earlier
185.03-second CPP9 calibration used a perturbed smoke geometry on `cn0`;
this minimized-reference check ran on `cn1`. The extra gas vertical-spectrum
handling took only 0.02 seconds. Geometry/node and cache differences prevent
direct attribution of the slowdown without a matched profiling comparison.
The approximately 39 GiB gas peak remains close to card capacity; this does
not establish that every future sampled geometry will fit.

Maximum small-molecule CPU/GPU differences were **7.1e-7 eV** for equilibrium
and vertical energies (limit 1e-5), **2.09e-6 eV/Å** for forces (limit 1e-4),
and **2.10e-12 Å⁻¹** for phase-aligned pairwise NACs (comparison limit 1e-5).
Representative moving-grid GPU force finite differences differed by
**7.32e-4 eV/Å** (limit 1e-3). The gas/DCM energy change was 0.18277 eV.
Requested optical arrays were finite, oscillator strengths nonnegative, and
storage checks verified atom identities, offsets, scales, NAC pair order and
pair/atom axes. Each store retains its frozen solvent/response manifest.

The first small-molecule attempt failed reference comparisons, not labeling
or GPU memory allocation. The CPU force reference initially moved the DFT
quadrature, whereas upstream analytic gradients omit quadrature-grid response.
The corrected CPU derivative fixes DFT quadrature while rebuilding the PCM
cavity; the independent moving-grid finite-difference gate is retained.
The independent NAC reference also needed ALF's Davidson `lindep=1e-20`
instead of the upstream default. Matching it removed the discrepancy.
Scientific inputs and acceptance limits were unchanged. Both passed ring
stages were retained under verified fingerprints; only the affected small
stage was retried. Initial evidence remains under `run-001/`.

**Runtime dependency and CI boundary.** Real checks reused frozen GPU4PySCF
1.9.1/PySCF 2.15.0/CuPy 14.2.0 with `cutensor-cu13==2.8.1` and the wheel's
`cutensor/lib` directory on worker `LD_LIBRARY_PATH`. Preflight verified native
cuTENSOR before calculation and would reject fallback; no CuPy contraction fallback,
density-fitting offload, visibility-mask change or OOM failure occurred in
this qualification. cuTENSOR is required for this qualified A100 CPP/PCM
profile, not for all ALF installations or GitHub CPU unit-test jobs.
`tests/test_gpu4pyscf.py` fakes GPU4PySCF/PySCF/CuPy; its CPU results establish
interface behavior, not real GPU numerics or memory capacity. Any future GPU
CI job must install a CUDA-compatible cuTENSOR wheel, expose its libraries,
and run the external native-backend preflight. Do not add CUDA/cuTENSOR to
the current CPU CI matrix solely for these mocked tests.

**Execution and cleanup.** Sequential one-worker allocations used
`shared-redstone` A100-SXM4-40GB nodes, explicit interpreters and GCC 11,
30-minute allocation caps, and the original November 6
11:07:42–13:07:42 Mountain queue-inclusive deadline. Qualification and cleanup
completed in **36.32 minutes**, reserving two minutes for cleanup. All four
owned jobs (`17381269`, `17381278`, `17381297`, `17381353`) have terminal
accounting and verified termination; both persistent Darwin supervisors
exited. The first job's failure records the corrected reference mismatch;
the ring jobs and retry completed successfully. No environment/upstream
installation was changed, and no training, dynamics or full ALF campaign was
launched. Generated artifacts remain outside Git.

### CPP9 E6 pilot — implementation and launch preparation (2026-11-06)

The approved bounded ordinary ALF pilot uses 32 fresh seed-42 CPP9
rotation/rattle geometries, DCM C-PCM with the complete qualified output
schema, and four HIP-HOP100 members. Training is limited to 50 epochs,
patience 50, and training/evaluation/scheduler batch eight. ALCHEMI runs
state-specific width-four groups cycling S0–S3, with four-member cohorts,
top-two selection and continue policy. Dynamics use 300 K, 0.1 fs steps,
ten-step observations, friction 0.1 fs⁻¹, a 5 fs acquisition minimum and
25 fs maximum. Topology and the existing uncertainty, distance and 16 eV/Å
physical-force cutoffs remain enabled; gap loss/acquisition remain disabled.
The storage trigger is seven, and acceptance requires all 32 fresh bootstrap
labels plus at least eight propagated, converged/stored new IDs covering all
states, retained by every promoted member and represented across actual
training splits. Real training, reload, finite predictions/gaps and model-force
finite differences within 1e-3 eV/Å are required.

Darwin uses three independent single-node allocations: four A100 40 GB QM
workers on shared-redstone, four local A6000 GPUs for ensemble training, and
four A6000 sampling workers on ml4chem. The queue-inclusive budget is four
hours, reserving five minutes for cleanup; allocation caps are 3.5 hours for
QM and one hour for training/sampling. The external `alf_validation`
CPP9 E6 supervisor instruments production tasks and ordinary
`alframework.__main__`, journals owned jobs, freezes inputs/source/environments,
and requires four overlapping successful bootstrap calculations on distinct
physical GPUs before reporting that disconnection is safe. New sampling
admissions stop after sufficient stored state coverage; overshoot is reported.

The transferable CPP example adds opt-in HIP-HOP100 without changing model
defaults or the production 300-epoch preset. Its ordinary coordinator runs in
`tmux` with `python -m alframework --master master_config.json`.
`parsl_configs.py` controls demand-driven node allocation: QM max_blocks=8,
training=1, sampling=1, one node per block, init_blocks=min_blocks=0, four QM
and sampling workers per node, and one local four-GPU training task. The QM
queue target is 64. Idle nodes are released by Parsl; no campaign controller
or ten-node reservation is required on the destination cluster. Native
cuTENSOR preflight follows worker device selection. Parsl binds each QM
worker to one allocated GPU UUID before CUDA imports; calculation tasks leave
the worker mask unchanged.

Launch evidence: ALF `30ba5ea`, external harness `ac4c0cb`; generated run
`validation/cpp9_e6_20261106/run-001/`. The focused external suite passed
**51 tests**; the full library suite passed **507 tests, 42 skipped**. The
persistent `alf-cpp9-e6` tmux supervisor and ordinary coordinator are running
on `darwin-fe2`; sampling allocation `17381481` and QM allocation `17381482`
started on `cn4072` and `cn0`. Training is queued by Parsl when bootstrap
labeling/storage finishes. The original queue-inclusive window is November 6,
12:33–16:33 Mountain, with sampling/QM acceptance still pending. The user
chose to disconnect after launch rather than wait for four bootstrap successes;
the four-GPU overlap requirement remains an automatic acceptance gate.

The first attempt stopped after 1/32 converged/stored bootstrap labels.
GPU 0 succeeded; workers selecting local GPU indices 1–3 failed during SCF
with 21 negative-dimension and ten indexing errors. No GPU OOM was reported.
All three owned allocations terminated. With only one stored frame, initial
training could not create nonempty train/validation/test splits; no sampling
or promotion occurred. This does not invalidate the prior single-worker DCM
labeling qualification.

The correction (`6f69486`, external harness `941f202`) binds QM workers using
Parsl's ordinary accelerator mechanism. A short-lived CUDA query resolves
UUIDs in the scheduler-visible order; the manager retains its allocation
mask, and each fresh QM worker sees its assigned GPU as local CUDA device 0.
This is required because upstream single-device kernels explicitly index
GPU-zero buffers. ALF now rejects nonzero backend device indices before
backend imports instead of allowing those kernels to fail during SCF. No
upstream installation, scientific setting or training GPU assignment changed.
The full library suite passed **510 tests, 42 skipped**; the affected combined
suite passed **126 tests**, and the latest CPP9 harness/budget suite passed
**15 tests**.

An initial corrected launch demonstrated four distinct GPU UUIDs/PCI buses
with local device zero. It was stopped to honor the user's explicit decision
to give the corrected attempt a fresh four-hour budget; both owned resources
(`17381880`, `17381882`) have verified termination. `run-002` was preparation
only. The fresh attempt is `validation/cpp9_e6_20261106/run-004/`, using
identical verified seed geometries/configuration/environments and no reused
labels. It runs under persistent frontend `alf-cpp9-e6` tmux; initial owned
jobs are `17381894` (sampling) and `17381895` (QM). Its queue-inclusive window
is **15:19:44–19:19:44 Mountain on November 6**, reserving five minutes for
cleanup. The normal retry path still preserves deadlines; budget renewal is
an explicit preparation option recorded in the frozen manifest. Native
convergence, training, acquisition and promotion acceptance remain pending.

**E6 remains incomplete until the pilot passes and owned-job termination is
verified.** Ten-node scaling, predictive accuracy, long-time stability,
production-scale qualification and post-QM filtering remain unclaimed.
CPP6 remains an additional transferable example. User guides remain untouched.

User decision, November 6: the guide's page for `examples/CPP_excited_gpu4pyscf`
documents the whole example now, ahead of this result, and says nothing about E6. This
section is where E6 status is recorded. If the pilot changes the example's settings or
shows a limit users need, update that page too.

GPU4PySCF library fixes after the test review, November 6 (user decision). CPU tests
with fakes only; the running pilot's frozen snapshot doesn't include them.

- `select_gpu4pyscf_device` now rejects any worker that sees more than one GPU, at any
  rank, and no longer selects by rank. Before this, rank 0 labeled normally while other
  ranks failed every label, the pattern in the first pilot attempt. Workers bound by
  `qm_worker_pool.py` see one GPU, so they take the same path as before. A single rank-0
  worker on a node exposing several GPUs now needs `available_accelerators=1` or a
  one-GPU allocation. `gpus_per_node` remains in the signatures but is unused here.
- `qm_optical_response`, `qm_vertical_response`, and `qm_vertical_root_order` now record
  `none` when that calculation didn't run. Gas-phase vertical root order is now
  `same as state TDA roots`, because those energies come from the state TDA solve.

Still to check on Darwin: a short real run confirming that a bound worker labels and an
unbound multi-GPU worker fails with the new message, and that `alf_validation` has
nothing expecting several visible GPUs at rank 0 or reading these fields on runs without
those properties.



User decision, November 5: add the opt-in chemical topology checker across
ASE, TorchSim, and ALCHEMI before launching the cpp-ring campaign. Increment
10's accepted topology coverage validates atom identities/order and composition;
its minimum-distance screen rejects close contacts. It does not check chemical
connectivity, bond breaking/formation, or ring opening. Increment 10 remains
accepted within that boundary.

Prerequisite implementation and gate (already passed in Increment 11A):

- Adapt the fork's reference-based `tools/molecular_topology.py` checker into
  shared development helpers, with one opt-in `topology_check` configuration
  used by all three engines. Keep reference bond/connectivity inference and
  optional dependencies outside the coordinator and behind the opt-in path.
- Validate the reference structure and supplied atom identities. Check expected
  bond-length bounds and unexpected connectivity against that reference;
  preserve identical-element identities through permutations. This adds
  chemical geometry screening, without changing propagation forces.
- Evaluate the screen at trajectory initialization and existing observation
  points, before candidate eligibility/ranking. A violation ends the affected
  trajectory, records a structured rejection reason/violations, and excludes
  the invalid frame from labeling. Previously valid candidates remain governed
  by the existing cohort policy. Keep screen behavior opt-in for both excited-
  and ground-state multi-trajectory sampling.
- Require common-fixture tests across ASE, TorchSim, and ALCHEMI for valid
  geometry, stretched/compressed reference bonds, new connectivity, ring
  opening, invalid reference/identity mappings, and atom permutations. Verify
  identical decisions on common recorded structures, independent stopping in
  packed tasks, and unchanged behavior with the checker disabled.
- Complete bounded real width-one/four screening checks across all three
  engines using the external validation harness before the ring campaign.
  Preserve the accepted Increment 10 evidence; a full campaign rerun is not
  required for this prerequisite. Fix the exact screen configuration and
  rejection/candidate semantics in the implementation plan before coding.

After this prerequisite passes, use the existing E6 gate in the plan above: externally generated ring
geometries, GPU4PySCF labels including stored NACs/optical properties as
requested, four-member multi-state training, selected-state sampling, and at
least one acquisition/label/store/retraining promotion through ordinary
`alframework.__main__`. NACs remain storage targets rather than training
targets. Reuse the accepted interface checks; choose bounded campaign sizes,
propagated states, and gap-acquisition settings before implementation. The
fork's standalone cpp trial remains source material, not the coordinator.

The following increment is composable sampling with selected-state UDD,
wells, and LCM gap seeking, detailed after E5. Multi-node member placement and
teacher labeling remain independent tracks rather than E5/E6 prerequisites.
