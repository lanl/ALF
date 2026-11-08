# Darwin qualification round: teacher labeling, fine-tuning, and restarts

Planned November 7, 2026. These are instructions for the Darwin round that qualifies the November 7 work on GPUs. Most of that work has run only on CPU. Results go in a new record in [the roadmap index](README.md), and in [`exascale.md`](exascale.md) for B2 and B6.

| Section | What it shows | Kind |
| --- | --- | --- |
| [B1](#b1-gpu4pyscf-under-the-new-state-names) | GPU4PySCF ground-state DFT (`nroots: 0`) and the excited path under `E#` names | Pass/fail |
| [B2](#b2-restart-acceptance) | Restarts keep their labels | Pass/fail |
| [B3](#b3-teacher-labeling-and-a-real-student) | MACE and UMA teachers on GPU, and a real HIPPYNN student | Pass/fail, plus measurements |
| [B4](#b4-fine-tuning-three-families-on-dft-labels) | Fine-tuning HIPPYNN, MACE, and UMA on DFT labels | Pass/fail, plus measurements |
| [B5](#b5-multi-node-fine-tuning) | Multi-node fine-tuning with real models | Pass/fail |
| [B6](#b6-tmp-and-settings) | `/tmp` and settings, for running at scale (priority items 7–8) | Measurements only |
| [B7](#b7-pyscf_task-on-the-gpu-and-periodic-water) | `pyscf_task` on the GPU, and the periodic-water example with a real periodic student | Pass/fail, plus measurements |

## B0. Setup and rules

**Source.**
- Record the ALF commit this round runs at: the commit that adds this file or a later one.
- Record the `alf_validation` harness revisions.
- Use only `development`, never `main`.

**Environments.**
- Use the qualified ALF and GPU4PySCF environments, unchanged.
- Build two more, each layered on the ALF environment, and record `pip freeze` for each:
  - `mace`: `mace-torch` 0.3.17 and TorchSim;
  - `uma`: `fairchem-core` 2.25.0 and TorchSim.

  They're separate because the two packages have needed different `e3nn` versions. These are the versions checked locally.
- Use each executor's `worker_init` to choose its environment: QM in the GPU4PySCF environment, and ML and sampling in the model's environment. The coordinator must be able to import every task module.

**Checkpoints.**
- **MACE-MP-0 small:** on a login node, run `python -c "from mace.calculators import mace_mp; mace_mp(model='small')"`, then copy the file from `~/.cache/mace/`.
- **UMA uma-s-1p2p1:** the user downloads it with their own account, on a login node:
  ```
  hf auth login
  hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir <dir>
  ```
  Never read or print the token.
- **Record** each checkpoint's SHA-256. UMA's should start `b2673b85` and end `d0d5ce`.
- **Compute nodes** read local files only, with `HF_HUB_OFFLINE=1`.

**Paths.** The GPU4PySCF task moved: use `alframework.qm_interfaces.pyscf.gpu4pyscf.gpu4pyscf_excited_state_task`, and `alframework.qm_interfaces.pyscf.pyscf_task` for B7. The old `gpu4pyscf_interface` path is gone.

**Names.** State-aware code now names states `E0, E1, …` and forces `F0, F1, …`, and `sE#` is gone; see the configuration guide's "Property names". There is no converter, so any accepted inputs that use `sE#` are regenerated or relabeled. Ground-state configurations are unchanged.

**System and labels.**
- Formaldehyde single molecules, non-periodic, from the accepted Increment 9 inputs.
- One DFT level throughout: GPU4PySCF's default, cam-b3lyp/6-31g\*.
- Ground-state labels use `gpu4pyscf_excited_state_task` with `nroots: 0` and
  ```json
  "properties_list": {"E0": ["energy", "system", 1.0], "F0": ["forces", "atomic", 1.0]}
  ```
  This stores the shards as `energy`/`forces`, which the student and fine-tuning read.

**DFT test set.** In B1, label about 64 formaldehyde frames once, sampled more widely than the bootstrap. Keep them in their own directory, outside every `h5_dir`. B3 and B4 measure against them.

**Rules.**
- Keep evidence outside Git, under `<validation root>/development_<date>/`.
- **No library edits during the round.** A defect stops its section; record the evidence and report it. Fixes are made on `development`, with tests, after the round.
- No dependency upgrades beyond the two new environments.
- Supervise with persistent `tmux`, and show with `squeue` that every owned job ended.
- If a tool or permission is missing, record that and move on; don't substitute an estimate.

**Order.**
- Run B1 → B6.
- B3 needs B1's test set, and B4's HIPPYNN model is B3's student.
- Sections may span allocations. Record each section's queue-inclusive window.

## B1. GPU4PySCF under the new state names

**Ground state, `nroots: 0`,** on 8 formaldehyde geometries:

| Check | Pass |
| --- | --- |
| `energy`/`forces` against a `nroots: 1` run's `E0`/`F0`, same geometries and settings | Within 1e-8 eV and 1e-6 eV/Å |
| Forces against a central finite difference of the energy, for one geometry | Within 1e-4 eV/Å |
| One geometry against CPU PySCF at the same settings | Energy within 1e-6 eV |
| An SCF forced not to converge, with `scf_max_cycle: 1` | The structure is marked unconverged, with `qm_failed_stage`, and isn't stored |
| `"mode": "sequential"` batches of 4 | The same labels as one at a time |
| The shards | `energy`, `forces`, and the `total_charge`/`spin_multiplicity` datasets; no `E0` dataset |

**The excited path under `E#`.** Rerun the November 6 bound gas and DCM evaluations, full and energy/force-only, and compare values with the frozen evidence. Pass: the same energies and forces to 1e-10 eV and 1e-6 eV/Å.

**Then** label the DFT test set.

## B2. Restart acceptance

This reruns [section 3 of the restart profile](exascale.md#3-kill-and-restart-a-small-real-loop-sets-the-baseline-that-restart-fixes-are-measured-against). On November 7, at `ce711d8` and before the fixes, it was only a baseline. Now it is pass/fail.

**Loops.**
- A small MACE-teacher loop with a 4-member HIPPYNN student on formaldehyde. Its labels are cheap, so the store and training rows are easy to hit.
- For the finite-bootstrap rows, the November 7 configuration whose restart failed, under the new names.

**Rows.** Run each row at least twice. Restart with the same command, and change nothing by hand in between.

| When to kill | Signals |
| --- | --- |
| During a bootstrap | SIGTERM, SIGKILL |
| During a finite bootstrap | SIGTERM, SIGKILL |
| During a finite bootstrap with `bootstrap_spares: 2`, with one seed forced to fail (`scf_max_cycle: 1` for its ID, or a cancelled QM block) | SIGKILL |
| Between stores, with training idle | SIGKILL |
| During training | SIGTERM, SIGKILL |
| During a store (best effort) | SIGKILL |

**Pass when, for every row:**
- The restart takes the coordinator lock, prints its `Restart:` lines, and reaches the next promotion.
- No finished label is lost: finished QM results equal stored labels plus those in `<h5_dir>_held/`. Stored IDs have no duplicates.
- No training ID is reused, and training never reads a `.partial` shard.
- After a store interruption, the next store succeeds.
- A finite bootstrap ends with `bootstrap_set` converged labels, each once: the seeds in order, plus one spare for each failed seed, and no spare when nothing failed.
- With the first coordinator running, a second in the same directory is refused, with the lock's host and process.

**Record only:** how long allocations survive a SIGKILL. Cleaning them up is the scheduler's job, not ALF's.

**Also:** rerun [section 5](exascale.md#5-replay-manifest-cost-decides-how-urgent-the-manifest-cache-is), the replay-manifest timing, on these shards.

## B3. Teacher labeling and a real student

**GPU measurements,** for MACE-MP-0 small and UMA `omol`, on formaldehyde:
- throughput and peak GPU memory at batch sizes 1, 8, 32, and 128;
- label differences between float32 and float64;
- one batch on GPU against CPU;
- batched labels against one at a time, which should agree to float precision.

**A loop per teacher,** each in its own directories:
- `teacher_batch_task` with native batching.
- A 4-member `train_HIPPYNN_ensemble_task` student on GPU, with `cell_key: null` and `h5_test_dir` set to the DFT test set.
- Run to two promotions, with sampled labels in the second.

**Pass when:**
- both promotions happen, and every member reloads and predicts finite values;
- every shard carries `label_provenance` naming the teacher and its checkpoint's SHA-256, and each model's `label_sources.json` names the teacher;
- pointing a teacher run at an `h5_dir` holding the DFT labels fails at startup with the mixed-label message;
- the student's `training_log.txt` has test metrics against the DFT test set.

**Report:** each student's errors against its teacher, on held-out teacher labels, and against DFT. Keep the UMA-teacher student, which B4 fine-tunes.

## B4. Fine-tuning three families on DFT labels

| Family | Pretrained model | Settings |
| --- | --- | --- |
| HIPPYNN | B3's UMA-teacher student | `E0s: average` |
| MACE | MACE-MP-0 small | MACE's naive recipe (ALF's default), `E0s: estimated` |
| UMA | uma-s-1p2p1, `omol` | `E0s: average`, `epochs: 10` |

**Each family:**
- a GPU4PySCF `nroots: 0` bootstrap of about 32 labels, then two rounds;
- four copies on one node with four GPUs;
- sampling on the copies: the HIPPYNN sampler, or TorchSim on CUDA for MACE and UMA, with `load_mace_batched_ensemble` or `load_uma_batched_ensemble`.

**Pass when:**
- each round's `fine_tune_lineage.json` is complete, with four distinct seeds;
- each copy ran on its own GPU: the copy's `CUDA_VISIBLE_DEVICES` and the device it reports;
- every copy loads and predicts finite values;
- the second round's shards include sampled labels;
- on the DFT test set, after removing each model's constant energy offset, the fine-tuned copies' mean force MAE and energy MAE are below the pretrained model's.

**Record:**
- **Copy disagreement on the test set,** with the CPU numbers for comparison:
  - HIPPYNN copies of ensemble members, 3.1e-3 eV/Å;
  - MACE, 0.067 eV/Å;
  - UMA, 0.22 eV/Å.
- Peak GPU memory per copy, and wall time per round.

## B5. Multi-node fine-tuning

Use `fine_tune_ensemble_multinode_task` with real MACE and real UMA: four copies on two nodes, one clean run and one where a block is cancelled mid-round, as in the [November 7 trainer checks](README.md#current-development-hardware-qualification--passed-with-restart-limits-2026-11-07).

**Pass when:**
- the copies ran on four distinct host/GPU pairs;
- only the cancelled block's copies were retried;
- the lineage is complete;
- the round is promoted once.

## B6. `/tmp` and settings

These are measurements only, for priority items 7 and 8 in [`exascale.md`](exascale.md#tier-2-restarts-that-keep-labels). On November 7, the two profiled GPU nodes had `/tmp` on a 63 GB tmpfs; other partitions weren't checked.

**For each partition ALF uses,** including CPU partitions, from inside a job:
- `df -h /tmp` and `findmnt -T /tmp`;
- `$TMPDIR`;
- whether nodes are shared with other jobs;
- whether files in `/tmp` survive after the job ends.

**Defaults in a job:**
- `HDF5_USE_FILE_LOCKING`, `PYSCF_TMPDIR`, and `MPLCONFIGDIR`;
- where Parsl's `run_dir` and `worker_logdir_root` land.

## B7. `pyscf_task` on the GPU, and periodic water

`pyscf_task` and `PySCFCalculator` were checked on the CPU only, against PySCF itself and finite differences. Their GPU path moves each PySCF method to GPU4PySCF with `to_gpu()`. Use the GPU4PySCF environment, and record its `gpu4pyscf` version, which should be 1.9 or later for periodic gradients and stress.

**CPU against GPU,** with the same settings on the same structures:

| Structures | Settings | Pass |
| --- | --- | --- |
| 4 water clusters from `simple_water` | `qm_config_pyscf.json`, at def2-SVP | Energies within 1e-5 eV and forces within 1e-4 eV/Å |
| 2 periodic water boxes from `periodic_water` | `periodic.df` `multigrid`, then `fftdf` | The same, for each `df` that the GPU accepts |
| One formaldehyde | `nroots: 2`, `density_fit: false` | State energies within 1e-5 eV and state forces within 1e-4 eV/Å |

**GPU only:**
- Periodic forces with `df: gdf` against a central finite difference of the energy, on one box: within 1e-3 eV/Å.
- Periodic stress against ASE's `calculate_numerical_stress`: within 1e-4 eV/Å³.
- A k-point mesh of `[2, 2, 2]` with `fftdf` or `gdf`, on one small box.

With `df: multigrid` on the GPU, confirm that the method's integrator is GPU4PySCF's own (`calc.metadata` and the PySCF log), since ALF switches it on after `to_gpu()`.

Record each combination GPU4PySCF refuses, with its error. These are answers, not failures.

**The `periodic_water` example,** with `qm_config.json` on GPUs:
- `--test_builder` and `--test_qm`;
- then a loop to two promotions, with a real 4-member periodic HIPPYNN student (`cell_key: "cell"`) and sampled labels in the second.

**Pass when:**
- the shards hold `energy`, `forces`, and `cell`;
- every member reloads and predicts finite values on a periodic box;
- fewer than 5 % of the labels fail; record each failure's `qm_failed_stage`.

**Record:** wall time per label against box size, peak GPU memory, and how often the SCF failed.

## Report back

For each section: a short result, pass or fail where it has one, and the evidence path. Also answer:
- Did any section find an ALF defect? Give what, where, and the evidence.
- Can QM scratch move to node-local `/tmp` on each partition? (B6)
- Do copies fine-tuned from one pretrained model disagree enough on GPU to drive selection? (B4)
- Which periodic `df` choices does GPU4PySCF accept for forces and stress? (B7)
