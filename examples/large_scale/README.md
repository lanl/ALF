# Running on many nodes

A template for an ALF run on many nodes, with every setting that the Running at Scale guide measured to matter at
that size. The chemistry is the simple water cluster of `simple_water`, labeled by a foundation-model teacher, so
that the example is about the workflow rather than the system. Replace the builder, sampler, labeler and trainer
settings with your own; the scaling settings carry over.

## What it sets

- **Fewer, larger tasks.** Each builder task builds 256 structures (`maximum_builder_structures`), each sampler task
  runs 256 trajectories on one GPU (`batching` in `mlmd_config.json`), and each QM task labels up to 256 structures
  in one teacher call (`batching` in `qm_config.json`).
- **The `scaling` block in `master_config.json`.** It gathers the settings that change how ALF runs at scale:
  pickle serialization, a short poll, releasing the coordinator's freed memory after each shard, turning off
  transparent huge pages for the coordinator, label storage in lossless HDF5 parts, compact sampler returns
  with one metadata file per task, shared training arrays and a streaming final test, and records for watching the
  run (trajectory summary, coordinator memory and loop timing, read by `run_report`). See "Scaling Settings in
  One Place" in the Running at Scale guide for what each does.
- **A label budget.** `maximum_stored_labels` makes the run finite: it stores at least that many labels, lets the
  last training finish, and exits.
- **One Parsl worker pool per node** (`parsl_configs.py`): sampler and QM workers each pinned to one GPU, and one
  training worker per node, with each ensemble member training on that node's GPUs (`gpus_per_member`).

## Sizing

Keep the structures in flight equal to the sampling capacity: `parallel_samplers` is the sampler nodes times
`ALF_GPUS_PER_NODE` times the sampler batch size. The defaults, 16,384, fit 16 sampler nodes of 4 GPUs at 256
trajectories each; with `ALF_SAMPLER_NODES=4` set `parallel_samplers` to 4,096. Set `target_queued_QM` and
`save_h5_threshold` to the same number to retrain after every wave of sampling. A member on several nodes needs
`member_nodes`, a `member_launcher` and a Parsl configuration that gives each ML worker its group of nodes; see the
ML interfaces guide.

## Running it

Download the teacher checkpoint named in `qm_config.json`. Then describe the machine and start the coordinator on
a login node, inside `tmux` or `screen`:

```bash
export ALF_PARTITION=<partition> ALF_ACCOUNT=<account>
export ALF_GPUS_PER_NODE=4 ALF_CORES_PER_NODE=32
export ALF_SAMPLER_NODES=16 ALF_QM_NODES=4 ALF_ML_NODES=8
export ALF_SAMPLER_RESOURCE_OPTIONS='#SBATCH --gpus-per-node=4' ALF_QM_RESOURCE_OPTIONS='#SBATCH --gpus-per-node=4'
export ALF_ML_RESOURCE_OPTIONS='#SBATCH --gpus-per-node=4'
# The workers' node environment: a packed environment unpacked once per node, node-local caches, no huge pages.
export ALF_ENV_TARBALL=/shared/envs/alf.tar.gz ALF_NODE_LOCAL_CACHES=1 ALF_THP_DISABLE=1
python -m alframework --master master_config.json --test_builder
python -m alframework --master master_config.json
```

The coordinator prints the scaling settings it applied when it starts. `records/` and `run_report/` show the run
as it goes: `python -m alframework.tools.run_report .` remakes the report at any time.

## Handing frames to QM by reference

On a filesystem that the sampler and QM nodes share, the sampled frames can skip the coordinator: add
`"data_by_reference": {"directory": "sampling-frames", "sampler_frames": true}` to the `scaling` block. It is off
unless set, needs the batched MLMD sampler, native QM batching and `maximum_stored_labels`, all of which this
example has, and is described under "Coordinator Memory" in the Running at Scale guide.
