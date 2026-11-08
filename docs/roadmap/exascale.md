# File I/O at exascale

An audit of where ALF's file operations could become a problem in campaigns of thousands of nodes, with a priority list for fixing them. It was done on November 6, 2026, by reading the file operations in `alframework/` and the CPP example, and the defaults in HIPPYNN, PySCF, and Parsl.

**Where this stands:** items 1-6 of the priority list were built on November 7, after the Darwin round had measured `ce711d8`: atomic writes, the shard check in single-node training, and the replay manifest cache; then the coordinator lock, the label holding area, and restart reconciliation. Items 7 and 8 are open, and the wider restart plan is in [Safe restarts and disk races](future.md#safe-restarts-and-disk-races). Three related changes landed earlier: ALF now releases allocations on exit, the excited-state loader no longer reads `training_summary.json` on each sampler task, and the multi-node training tasks wait for shards. Bounded measurements now cover two-node operation and 20 independent training members on five nodes; they do not establish exascale performance.

**What was and wasn't measured:** the original audit used code inspection and
in-memory Mac timings. The [November 7 Darwin measurements](#darwin-measurements-2026-11-07)
add real shared-filesystem and bounded multi-node evidence. Large-dataset costs
remain estimates; GPU4PySCF temp-file behavior was not measured in this round.

## Darwin measurements (2026-11-07)

ALF `ce711d8`; frozen environments; evidence under
`validation/development_20261107/`. Use `qualification_summary.json`,
`run-003/io-profile/result.json`, `run-003/restart-profile/result.json` and their
referenced original observations. Functional qualification is recorded in the
[roadmap index](README.md#current-development-hardware-qualification--passed-with-restart-limits-2026-11-07).
These are measurements, not implementation of the priority list below.

**Locks:** the shared home filesystem was NFSv3. `flock` refused a second holder on the same
node and across nodes. After normal release, the other node acquired in 0.00242 s;
after killing the holder child, in 0.00275 s. Cancelled-node observations in the
retry run also acquired the lock (0.00216–0.00363 s). This supports the proposed
coordinator lock; ALF still has no coordinator lock.

**Visibility:** 50 direct and 50 temporary-file/rename publications were checked
from another node, with all listing, named lookup and readable-content measurements
present and all content hashes matching. Observed median/maximum delays, seconds:

| Publication | Listing | Named lookup | Readable, matching contents |
| --- | --- | --- | --- |
| Direct | 0.108 / 0.198 | 0.108 / 0.198 | 0.113 / 0.202 |
| Rename | 0.103 / 0.240 | 0.097 / 0.199 | 0.101 / 0.204 |

Polling was about 0.2 s, so these are observation bounds, not precise filesystem
latencies. The longest listing delay was 0.240 s in this corrected probe. Earlier
harness file barriers showed delayed visibility; the rerun used a live control
barrier rather than weakening concurrency assertions. No mismatched completed
content was accepted. This small probe does not prove crash-atomic publication or
justify shortening production shard waits.

**Restart baseline:** 11 interruptions were observed; the second best-effort
store-interruption trigger was not hit. Both finite-bootstrap `SIGTERM` and
`SIGKILL` restart attempts exited unsuccessfully. One store-interruption restart
also exited unsuccessfully. Idle/training restarts remained alive during bounded
observation; that is not a verified complete recovery. `SIGKILL` left one to three
worker allocations alive after approximately 121 s; the supervisor then cleaned
up owned jobs. The short `SIGTERM` baseline observed allocations still settling
after about 6.5 s; dedicated cleanup gates separately verified automatic release
with a longer wait. No status or shard repair was performed by hand. All owned
jobs ultimately terminated. Interrupted-bootstrap recovery, durable labels and
status/disk reconciliation remain open.

**Small-fixture costs and local storage:** the replay manifest for a 61,260-byte,
32-frame fixture took 0.0096–0.0102 s; this cannot establish large-dataset cost.
Fresh-process imports took 9.62–10.29 s and repeats 8.30–8.70 s. On the two profiled
GPU nodes, `/tmp` resolved to `/ram`, a roughly 63 GB tmpfs, with `TMPDIR=/tmp`.
It competes with host memory. Other partitions and post-job file survival were
not qualified. Darshan was unavailable; no complete per-role file-operation or
large-dataset contention profile was established.

**Priority impact:** retain items 1–6 and their order. Cross-node lock mechanics
support item 4; observed restart failures reinforce items 5–6. These small-fixture
measurements do not justify lowering replay/training I/O severity, claiming
exascale throughput, or enabling node-local environments/scratch across partitions.
The restart implementation remains tabled by the existing user decision.

## Main findings

- **The largest cost is the replay manifest.** Every builder task reads and hashes every shard in full. By estimate, 1M CPP9 labels is about 8 GB per task, so 1,000 builder tasks would read about 8 TB. *Built November 7:* each worker now reads each shard once.
- **The largest data-loss risk is labels held in memory.** Stores only happen when no training is running, so a crash loses every label finished since the last store. Bootstraps have the same risk.
- **Both trainers can silently miss a new shard.** They list `h5_dir`, so a stale directory view on another node means training without that shard. *Built November 7:* every HIPPYNN training task now waits for the shards the coordinator stored.
- **The HDF5 store isn't the bottleneck it looked like.** A gzip-6 store of 10,000 CPP9-sized labels (90 atoms, 3 states, 87 MB) took 1.7 s in memory.
- **The coordinator is the only writer of shared state.** Workers write to paths unique to a structure, member, or task, so there's no worker-against-worker race in normal operation.

## Priority list

Order weighs data loss first, then cost at scale, then effort.

**Multi-node training changes two items (November 6).** Its tasks already wait for shards the coordinator listed, which covers item 2 for them; single-node training got the same check on November 7. Larger ensembles also raise the value of item 9, because every sampler task loads every member.

### Tier 1: small fixes that protect runs

Worth doing whatever a profile shows, except that item 4 needs a `flock` test first.

| # | Work | Size | Needs from Darwin |
| --- | --- | --- | --- |
| 1 | **Built November 7.** Write status and shards to a temporary file, then rename. `force_screen.py:89` already does this. | Small | Nothing |
| 2 | **Built November 7.** Check for every expected shard by name in both trainers, with a bounded wait, then raise. Extra files stay allowed. | Small | Nothing. On Darwin the longest listing delay was 0.240 s, well inside the 120 s wait |
| 3 | **Built November 7.** Cache each shard's manifest per worker, keyed by path, size, and modified time (the `_load_reference_cached` pattern in `molecular_topology.py`), and open the seed once per task. | Small to medium | Nothing |
| 4 | **Built November 7.** Lock the coordinator on `<status_path>.lock`, and refuse to start if it's held. | Small | Whether `flock` works on the shared filesystem: it does, across nodes |

### Tier 2: restarts that keep labels

| # | Work | Size | Needs from Darwin |
| --- | --- | --- | --- |
| 5 | **Built November 7.** Save finished labels during a bootstrap and during training, to a holding area outside `h5_dir`, and move them in at the next training start. Store finished labels at a budget stop too. | Medium, needs design | A kill and restart test |
| 6 | **Built November 7.** Reconcile status with disk on restart: shard and training IDs, a molecule-ID margin, the bootstrap reset, and pickup from the holding area. Add `tests/test_restarts.py`. | Medium | A kill and restart test |
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

### Darwin profile for restart work

**Ran on November 7, at ALF `ce711d8`;** results are in [Darwin measurements](#darwin-measurements-2026-11-07). The plan is kept below as it was written. Measure these on Darwin, alongside the multi-node round. Nothing here is pass or fail, and no library code should change. Keep evidence outside Git under `validation/`, and record results in `exascale.md`.

If a tool or permission isn't available, record that and move on. Don't substitute an estimate.

**Since it ran,** priority items 1-3 were built on top of the measured baseline. One store-interruption restart failed at `ce711d8`, which is the case item 1 targets, and the record doesn't say why. Repeating the "during a store" row of section 3, and section 5, at the newer commit would show what changed.

#### 1. `flock` on the shared filesystem (decides how the coordinator lock works)

Use a file on the shared filesystem, with two allocated nodes, A and B.

```python
import fcntl, sys, time
handle = open(sys.argv[1], "w")
fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Raises BlockingIOError if it's held.
print("locked", flush=True)
time.sleep(float(sys.argv[2]))
```

| Check | Record |
| --- | --- |
| Hold on A, try on A | Refused? This is the control. |
| Hold on A, try on B | Refused? This is the question that matters. |
| Release on A, then try on B | Acquired, and how quickly? |
| `kill -9` the holder on A, then try on B | Acquired, and how quickly? A lock that outlives its holder would block restarts. |
| Hold on A, lose or cancel node A's job, then try on B | Acquired, and how quickly? |

Report `findmnt -T <path>` for the mount type and options.

#### 2. Shard visibility across nodes (decides the trainers' shard check, and whether the 120 s wait is right)

Node A writes HDF5 files into one directory, using `store_current_data` or a same-sized file. Node B polls that directory, so that its view is cached. Repeat at least 50 times, with a few seconds between writes.

Record the delay on node B until the new file:

- appears in `os.listdir`, which the trainers use
- is found by name with `os.path.exists` and without a prior listing, which replay uses
- opens with `h5py` and reads the same hash as on node A

Then repeat, with node A writing to a temporary name in that directory and renaming when complete. Record whether node B ever sees the final name with incomplete contents.

Report the median, the maximum, and any failures. Run it on an otherwise idle directory, and again during a real training or sampling load if that's practical.

#### 3. Kill and restart a small real loop (sets the baseline that restart fixes are measured against)

Use a small real ALF loop with the current code. For each row, kill the coordinator, restart it with the same command, and record what happens. Run each at least twice, because the timing varies.

| When to kill | Signal | Record |
| --- | --- | --- |
| During a bootstrap | `SIGTERM` | Does it restart? How many finished labels were lost? |
| During a bootstrap | `SIGKILL` | The same, plus how long blocks and the interchange survive. |
| Between stores, with training idle | `SIGKILL` | Does it restart? How many finished labels were lost? |
| During training | `SIGTERM` | Are all blocks released? Does the next training start cleanly? |
| During training | `SIGKILL` | How long do blocks survive? Does a training ID get reused, and does that fail? |
| During a store, if the timing can be hit | `SIGKILL` | Is a partial shard left? Does the next store fail? Does training read the partial file? |

Expected failures are listed under "Crash and race notes" in `exascale.md`. Report which ones actually happen, and anything that isn't listed.

Do not edit status or shards by hand between kill and restart, and record any cleanup that was needed to continue. Verify with `squeue` that every owned job ended.

#### 4. Who touches the filesystem (checks the severity rankings)

Run one small real loop, ideally the CPP or AcAc configuration, and one ground-state example if ORCA is available. Capture each of these separately: the coordinator, one sampler worker, one QM worker, and one training task.

Use Darshan if it's available, or otherwise `strace -f -c -e trace=%file`. Record for each:

- opens, creates, stats, and renames
- bytes read and written, if the tool gives them
- the paths with the most operations
- operations per trajectory or per label, so results can be scaled

Compare these to the tables in `exascale.md`, and list any row that looks wrong, plus any heavy operation that's missing.

#### 5. Replay manifest cost (decides how urgent the manifest cache is)

Time `geometry_manifest` on a real set of shards, recording the total HDF5 size. Run it once alone, then from several workers at once, if that's practical. Also record whether most of the time goes to reading, hashing, or scanning groups.

#### 6. Worker startup and imports

In a fresh process on a compute node, run `python -X importtime -c "import torch, hippynn, alframework"` with the environment where it's normally installed. Record the total time on a cold start and a repeat. If it's practical, also run from a node-local copy of the environment, and with several nodes starting at once.

#### 7. `/tmp` on each partition (decides whether QM scratch can move to node-local storage)

For each partition ALF uses, from inside a job:

- `df -h /tmp` and `findmnt -T /tmp`: size, and whether it's disk or RAM-backed
- the value of `$TMPDIR`
- whether nodes are shared with other jobs
- whether files in `/tmp` survive after the job ends

#### Report back

For each section, give a short result, and the evidence path. In addition, answer these directly:

1. Does `flock` work across nodes on the shared filesystem?
2. What's the longest delay before a new shard appears in another node's listing?
3. Is a rename ever visible before its contents are?
4. Which restart failures happened, and how many labels did each lose?
5. How long does `SIGKILL` leave blocks allocated?
6. Which `exascale.md` rows should change severity?

#### What the results decide

| Result | Effect on the restart plan |
| --- | --- |
| `flock` refused across nodes | The lock is as simple as planned. Otherwise it needs a lock file holding host and process ID, plus stale-lock detection. |
| Longest listing delay | Whether the 120 s shard wait, now used by every HIPPYNN training task, is long enough. |
| Rename visibility | Confirms that atomic shard writes are safe on the shared filesystem. |
| Kill and restart failures | The acceptance tests for items 5 and 6. |
| File operation counts | Reorders tier 3 in `exascale.md`. |
| Manifest timing | The cost the manifest cache removes, and, if repeated at the newer commit, what's left. |
| `/tmp` details | Whether item 8 can go ahead on each partition. |

## All operations, by severity

The last column gives the priority item. "Profile" means only if a profile shows a problem.

### High

| Operation | Where | Scales with | Why it matters | Item |
| --- | --- | --- | --- | --- |
| Replay builder reads and hashes every shard, then scans its groups | `geometry_manifest` in `external_geometry.py:32`; RDKit and CPP ring builders | Builder tasks × total HDF5 bytes | Each task holds each whole shard in memory while hashing it, builds one tuple per frame, and reopens the shard per molecule (`:117`). SHA-256 ran at about 3 GB/s, so the read probably dominates. *Built November 7:* indexed once per worker per shard version, with one count per group, and each source opened once per task. | 3 |
| Finished labels stay in memory while training runs | `__main__.py:435` | Labels finished during training | A crash loses every label since the last store, on every cycle. | 5 |
| Bootstrap holds all labels until it finishes | `__main__.py:274-329` | `bootstrap_set` | A crash loses the whole bootstrap, which may be the most expensive single loss for large CPP runs. | 5 |
| Trainer lists `h5_dir` right after a store | `hippynn_interface.py:574` (`os.listdir`, `allow_unfound=True`) and `:1234` (`glob`) | Each store | A stale view silently trains without the new shard. Replay is safer, because it looks up shards by name and raises (`rdkit_molecule.py:425`). The multi-node tasks already wait for the shards the coordinator listed. *Built November 7:* the single-node tasks wait for `h5_path` shards numbered below `current_h5_id`. | 2 |
| One directory per label, under one parent | `QM_scratch_dir/<id>/`; ASE, ORCA, Q-Chem, and VASP only | Labels | Heavy metadata load, and ASE and ORCA never remove them by default. GPU4PySCF and PySeQM create none, so CPP and AcAc are unaffected. | 8 |
| One metadata pickle per trajectory, in one flat directory | `batched_mlmd.py:437`, `mlmd_sampling.py:165` | Trajectories | Only when `meta_dir` is set. CPP and AcAc don't set it, but the ground-state examples checked do. Progress plots no longer need it, with `trajectory_summary_path`. | 10 |

### Medium

| Operation | Where | Scales with | Why it matters | Item |
| --- | --- | --- | --- | --- |
| Non-atomic status and shard writes | 6 status write sites in `__main__.py`; `datapacker` | Each store and loop | A crash can leave truncated status, which stops a restart, or a partial shard that `w-` mode then refuses to replace. *Built November 7:* status and shards are renamed into place, and a leftover `.partial` is ignored and overwritten. A crash between the rename and the status write still stops a restart, now with a clear message; that's for item 6. | 1 |
| Two coordinators in one directory | `__main__.py`; no lock anywhere | Operator error | Both write status and collide on shards. | 4 |
| Loads every ensemble member per sampler task | `_load_member`, `hippynn_interface.py:836` | Sampler tasks × members × 2 or 3 | Happens on every task, not only after promotion. Excited-state ensembles also read the small `member_composition.json`, and the legacy ASE path loads per trajectory. | 9 |
| Opens `seed.h5` and rehashes inputs | `rdkit_molecule.py:415-435`, CPP `custom_modules.py:24-45` | Molecules, plus builder tasks × seed size | The seed opens twice per molecule during a bootstrap. The seed and reference are hashed on every builder task for the whole campaign, and the manifest is hashed per molecule. *Built November 7:* the seed and reference are hashed once per worker per version, the manifest once per task, and the seed opened twice per task. | 3 |
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
| Plot process scans all pickles | `plotting.py` | Total pickles | Off unless `plotting_utility` is set, and every 1000 loops by default. Partial pickles are caught and only lose a plot point. `trajectory_summary_path` avoids this: one file, written by the coordinator. | Profile |
| Reads five config files per loop | `__main__.py:359-373` | Loops | One process. Pending batch timeouts can shorten the 60 s sleep (`batching.py:318`). | None |
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
| Metadata and trajectories | Needs code | Tasks don't create `meta_dir`; the coordinator creates it on its own node only (`__main__.py:93-97`), so a local path would raise `FileNotFoundError`. The plot process also couldn't see local files. |

**Limits to design around:**

- `/tmp` may be RAM-backed, where it would compete with the calculation for memory.
- Config values don't expand `$TMPDIR`, and a path not starting with `/` gets the master directory prefixed (`tools.py:526`). Use a literal path with a run ID, so runs and users can't collide.
- Files are lost when a node fails or a job ends, so anything worth keeping must be copied out first. A walltime kill can skip that copy.
