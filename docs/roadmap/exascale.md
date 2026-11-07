# File I/O at exascale

An audit of where ALF's file operations could become a problem in campaigns of thousands of nodes, with a priority list for fixing them. It was done on November 6, 2026, by reading the file operations in `alframework/` and the CPP example, and the defaults in HIPPYNN, PySCF, and Parsl.

**Where this stands:** nothing has been changed in the code. The wider restart plan is in [Safe restarts and disk races](future.md#safe-restarts-and-disk-races).

**What was and wasn't measured:** nothing was measured on a shared filesystem or at scale, so severities are estimates from reading the code. The only timings are in-memory ones on a Mac, using random floats. GPU4PySCF wasn't installed there, so whether it inherits PySCF's temp-file behavior is unconfirmed.

## Main findings

- **The largest cost is the replay manifest.** Every builder task reads and hashes every shard in full. By estimate, 1M CPP9 labels is about 8 GB per task, so 1,000 builder tasks would read about 8 TB.
- **The largest data-loss risk is labels held in memory.** Stores only happen when no training is running, so a crash loses every label finished since the last store. Bootstraps have the same risk.
- **Both trainers can silently miss a new shard.** They list `h5_dir`, so a stale directory view on another node means training without that shard.
- **The HDF5 store isn't the bottleneck it looked like.** A gzip-6 store of 10,000 CPP9-sized labels (90 atoms, 3 states, 87 MB) took 1.7 s in memory.
- **The coordinator is the only writer of shared state.** Workers write to paths unique to a structure, member, or task, so there's no worker-against-worker race in normal operation.

## Priority list

Order weighs data loss first, then cost at scale, then effort.

**Multi-node training changes two items (November 6).** Its tasks already wait for shards the coordinator listed, which covers item 2 for them; single-node training still needs it. Larger ensembles also raise the value of item 9, because every sampler task loads every member.

### Tier 1: small fixes that protect runs

Worth doing whatever a profile shows, except that item 4 needs a `flock` test first.

| # | Work | Size | Needs from Darwin |
| --- | --- | --- | --- |
| 1 | Write status and shards to a temporary file, then rename. `force_screen.py:89` already does this. | Small | Nothing |
| 2 | Check for every expected shard by name in both trainers, with a bounded wait, then raise. Extra files stay allowed. | Small | Nothing, though a visibility test would show how long to wait |
| 3 | Cache each shard's manifest per worker, keyed by path, size, and modified time (the `_load_reference_cached` pattern in `molecular_topology.py`), and open the seed once per task. | Small to medium | Nothing |
| 4 | Lock the coordinator on `<status_path>.lock`, and refuse to start if it's held. | Small | Whether `flock` works on `/vast` |

### Tier 2: restarts that keep labels

| # | Work | Size | Needs from Darwin |
| --- | --- | --- | --- |
| 5 | Save finished labels during a bootstrap and during training, to a holding area outside `h5_dir`, and move them in at the next training start. Store finished labels at a budget stop too. | Medium, needs design | A kill and restart test |
| 6 | Reconcile status with disk on restart: shard and training IDs, a molecule-ID margin, the bootstrap reset, and pickup from the holding area. Add `tests/test_restarts.py`. | Medium | A kill and restart test |
| 7 | Add a "running at scale" guide section and Darwin settings for `HDF5_USE_FILE_LOCKING`, `worker_logdir_root`, `run_dir`, `MPLCONFIGDIR`, `TMPDIR`, and `PYSCF_TMPDIR`. | Small | The `/tmp` checks below |
| 8 | Run QM in node-local scratch: remove each directory after its results return, and copy failed calculations to shared storage. | Small | The `/tmp` checks below |

Items 5 and 6 ship together, because holding-area labels are only useful if a restart picks them up. The holding area is needed because a store inside `h5_dir` during training could give later ensemble members more data than earlier ones.

### Tier 3: after a Darwin profile

| # | Work |
| --- | --- |
| 9 | Cache loaded models per worker by model ID, and cache the topology hash. First check that nothing mutates a cached evaluator. If reads still hurt, copy each promoted model to node-local storage once per node. |
| 10 | Write metadata and audit records as one file per task, and write trajectories locally, then copy them to shared storage when the task ends. |
| 11 | Cache builder library reads per worker. |
| 12 | Unpack the Python environment to node-local storage at block start, or use a container. |

### Only if a profile shows a problem

- A consolidated training file, and using a recorded hash in the trainer.
- Training on node-local copies of the shards, then copying the finished model back before it's promoted.
- A background store writer.
- Limiting the plot scan.

### Darwin profile

This can run alongside tier 1. It should cover:

- Write and metadata counts per process, using Darshan or `strace`.
- Import time at worker startup.
- Shard visibility across nodes on `/vast`.
- `flock` on `/vast`.
- A kill and restart of a small real loop.
- For `/tmp` on each partition: size, whether it's disk or RAM-backed, whether nodes are shared with other jobs, and whether it's purged when a job ends.

## All operations, by severity

The last column gives the priority item. "Profile" means only if a profile shows a problem.

### High

| Operation | Where | Scales with | Why it matters | Item |
| --- | --- | --- | --- | --- |
| Replay builder reads and hashes every shard, then scans its groups | `geometry_manifest` in `external_geometry.py:32`; RDKit and CPP ring builders | Builder tasks × total HDF5 bytes | Each task holds each whole shard in memory while hashing it, builds one tuple per frame, and reopens the shard per molecule (`:117`). SHA-256 ran at about 3 GB/s, so the read probably dominates. | 3 |
| Finished labels stay in memory while training runs | `__main__.py:433` | Labels finished during training | A crash loses every label since the last store, on every cycle. | 5 |
| Bootstrap holds all labels until it finishes | `__main__.py:273-327` | `bootstrap_set` | A crash loses the whole bootstrap, which may be the most expensive single loss for large CPP runs. | 5 |
| Trainer lists `h5_dir` right after a store | `hippynn_interface.py:574` (`os.listdir`, `allow_unfound=True`) and `:1234` (`glob`) | Each store | A stale view silently trains without the new shard. Replay is safer, because it looks up shards by name and raises (`rdkit_molecule.py:425`). | 2 |
| One directory per label, under one parent | `QM_scratch_dir/<id>/`; ASE, ORCA, Q-Chem, and VASP only | Labels | Heavy metadata load, and ASE and ORCA never remove them by default. GPU4PySCF and PySeQM create none, so CPP and AcAc are unaffected. | 8 |
| One metadata pickle per trajectory, in one flat directory | `batched_mlmd.py:437`, `mlmd_sampling.py:165` | Trajectories | Only when `meta_dir` is set. CPP and AcAc don't set it, but the ground-state examples checked do. | 10 |

### Medium

| Operation | Where | Scales with | Why it matters | Item |
| --- | --- | --- | --- | --- |
| Non-atomic status and shard writes | 6 status write sites in `__main__.py`; `datapacker` | Each store and loop | A crash can leave truncated status, which stops a restart, or a partial shard that `w-` mode then refuses to replace. | 1 |
| Two coordinators in one directory | `__main__.py`; no lock anywhere | Operator error | Both write status and collide on shards. | 4 |
| Loads every ensemble member per sampler task | `_load_member`, `hippynn_interface.py:836` | Sampler tasks × members × 2 or 3 | Happens on every task, not only after promotion. Excited-state ensembles also read `training_summary.json`, and the legacy ASE path loads per trajectory. | 9 |
| Opens `seed.h5` and rehashes inputs | `rdkit_molecule.py:415-435`, CPP `custom_modules.py:24-45` | Molecules, plus builder tasks × seed size | The seed opens twice per molecule during a bootstrap. The seed and reference are hashed on every builder task for the whole campaign, and the manifest is hashed per molecule. | 3 |
| HDF5 file locking on reads | Every `h5py.File` read; nothing sets it | Concurrent readers | Can fail or contend on shared filesystems. Shards are never modified once closed, so reads don't need the lock. | 7 |
| Parsl logs | `runinfo/`; `worker_N.log` per worker and `manager.log` per pool | Nodes × workers | The logs are unconditional, and no ALF config sets `worker_logdir_root`. | 7 |
| Trajectory files, as often as every step | `Trajectory`; `simple_water` uses 5% of trajectories at interval 1 | Trajectories × frequency × steps | Off by default and in both excited-state examples, so only configs that opt in pay this cost. They share the pickles' flat directory. | 10 |
| One screen-audit JSON per sampler task, in one flat directory | `force_screen.py:69`; enabled in CPP | Sampler tasks | Its writes are already atomic. | 10 |
| Builder lists and reads its library per task | `builders.py:42`, `:63-70`, `:238` | Builder tasks × library files | `readMolFiles` reads every file twice. | 11 |
| Python imports when workers start | All workers, and the `spawn` pool in the excited-state trainer | Workers | Unmeasured, and based on general HPC experience rather than ALF's code. It matters when the environment is on shared storage. | 12 |
| Trainer reads every shard, and excited-state training also hashes them | `hippynn_interface.py:574`, `:1283` | Members × total HDF5 bytes | Only one training task runs at a time, so this is usually small next to hours of training. Multi-node ensembles raise it: 40 members would read about 320 GB per retrain at 1M CPP9 labels. | Profile |

### Low

| Operation | Where | Scales with | Why it matters | Item |
| --- | --- | --- | --- | --- |
| Writes every training frame into each member's summary | `training_summary.json` and `training_input_manifest.json` in `hippynn_interface.py`; excited-state training only | Members × labels | About 200 MB per file at 1M labels, for every retrain. The loader no longer reads this file on each sampler task; it reads `member_composition.json` instead. The summary is kept whole because it's an important record, so the write remains. | None |
| Reads and hashes the topology reference | `molecular_topology.py:340` | Sampler tasks | A small file, read before the cache lookup. | 9 |
| Library temp files and config | `~/.matplotlib`, `$TMPDIR/torchinductor_*`, PySCF's per-SCF checkpoint (`pyscf/scf/hf.py:1814`) | Workers or SCF runs | Only an issue if those paths are shared. | 7 |
| HDF5 store blocks the main loop | `store_current_data` | Labels per store | About 17 s per 100,000 CPP9 labels at the measured rate, before any network cost. | Profile |
| Plot process scans all pickles | `plotting.py` | Total pickles | Off unless `plotting_utility` is set, and every 1000 loops by default. Partial pickles are caught and only lose a plot point. | Profile |
| Reads five config files per loop | `__main__.py:357-371` | Loops | One process. Pending batch timeouts can shorten the 60 s sleep (`batching.py:318`). | None |
| Best-model and checkpoint writes, plus plots every 50 epochs | HIPPYNN `routines.py:548-555` | Members × improved epochs | Not atomic, but a failed member isn't promoted. | None |

## Crash and race notes

- **Crash after a store, before status is saved:** the shard name is reused, and `w-` mode makes the next store fail.
- **Partial shard:** it can reach a trainer but not replay, because replay reads only IDs below `current_h5_id`.
- **Molecule ID reuse:** only `ase_calculator_task` raises. ORCA, Q-Chem, VASP, and metadata pickles silently overwrite.
- **Training ID reuse:** the excited-state trainer refuses a nonempty directory.
- **Restart during a `finite_bootstrap`:** it fails with "Finite bootstrap exhausted", because `current_molecule_id` has advanced with nothing stored.
- **`maximum_stored_labels` stop:** it drops labels that are finished but not yet stored.
- **Leaked allocations: fixed for most exits (November 6).** ALF used to call Parsl's `cleanup` only at a `maximum_stored_labels` stop. On any other exit, the interchange outlived the coordinator and kept answering worker heartbeats, so nodes stayed held until walltime. This was seen locally after `--test_ml`, and interchanges from one crash test were still running 21 minutes later. `SIGKILL` can still leak them.

## Node-local storage

**Rule:** anything a restart or another node needs stays on shared storage. That means `status.json` and its lock, HDF5 shards, the label holding area, promoted models, and provenance.

| Data | Node-local? | Notes |
| --- | --- | --- |
| QM scratch | **Yes, the best fit** | Results return through Parsl, not files. An absolute `QM_scratch_dir` already works, because the tasks create their own directories. Cleanup is needed, because ASE and ORCA leave theirs behind. |
| Library temp files, `matplotlib` config | Yes | Environment settings only. |
| Parsl worker logs | Yes, with a tradeoff | Logs vanish with the node, which makes failed blocks harder to debug. Keep them shared while a setup is new. |
| Python environment | Likely | `ALF_<ROLE>_WORKER_INIT` in the CPP Parsl config can already run the unpack step. |
| Promoted models | As a read-only copy | Try the per-worker cache first. Workers sharing a node would need an atomic copy. |
| Replay shards | Probably not | Copying the dataset to every node is the same volume problem. The manifest cache removes the full reads, leaving small frame reads. |
| Training | Only if profiled | One node reads the shards per member, so a local copy helps, but the model must be back on shared storage before promotion. |
| Metadata and trajectories | Needs code | Tasks don't create `meta_dir`; the coordinator creates it on its own node only (`__main__.py:92-96`), so a local path would raise `FileNotFoundError`. The plot process also couldn't see local files. |

**Limits to design around:**

- `/tmp` may be RAM-backed, where it would compete with the calculation for memory.
- Config values don't expand `$TMPDIR`, and a path not starting with `/` gets the master directory prefixed (`tools.py:526`). Use a literal path with a run ID, so runs and users can't collide.
- Files are lost when a node fails or a job ends, so anything worth keeping must be copied out first. A walltime kill can skip that copy.
