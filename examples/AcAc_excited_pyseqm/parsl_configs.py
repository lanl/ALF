"""Portable Slurm template; set ALF_PARTITION and site worker initialization.

ALF_{ML,SAMPLER,QM}_PYTHON select interpreters. ALF_{ML,SAMPLER,QM}_WORKER_INIT
can load CUDA/compiler modules and library paths. ALF_SOURCE optionally pins
an ALF checkout; otherwise use the installed package. Import submits no jobs.
No CUDA visibility mask is rewritten.

At scale, ALF_ENV_TARBALL (or ALF_{ML,SAMPLER,QM}_ENV_TARBALL) names an archive of a relocatable
environment (conda-pack, or uv; ALF_ENV_PYTHON is its interpreter) that each node unpacks once
into node-local storage (ALF_NODE_LOCAL_DIR, default $TMPDIR or /tmp), and
ALF_NODE_LOCAL_CACHES=1 moves package caches there and turns off HDF5 file locking. Both are off
by default; see the Parsl guide's Running at Scale section.
"""
import os
from pathlib import Path
import shlex
import sys

import alframework
from alframework.tools.node_setup import setup_from_environment
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.executors.high_throughput.executor import DEFAULT_LAUNCH_CMD
from parsl.launchers import SingleNodeLauncher
from parsl.providers import SlurmProvider


def make_config():
    source = Path(os.environ.get('ALF_SOURCE', str(Path(alframework.__file__).resolve().parent.parent))).resolve()
    if not (source/'alframework').is_dir():
        raise ValueError('ALF_SOURCE must identify the pinned ALF source checkout')
    run = Path(os.environ.get('ALF_RUN_ROOT', os.getcwd())).resolve()
    executors = []
    for role, workers, cores in [('ML', 1, 4), ('sampler', 4, 1), ('QM', 1, 8)]:
        prefix = 'ALF_'+role.upper()
        python = os.environ.get(prefix+'_PYTHON', sys.executable)
        # A node-local environment changes only the workers' interpreter: the interchange runs here.
        setup, worker_python = setup_from_environment(prefix)
        init = [os.environ.get(prefix+'_WORKER_INIT', ''), *setup,
                'export PATH='+('"$ALF_ENV_BIN"' if worker_python else shlex.quote(str(Path(python).parent)))+':"$PATH"',
                'export PYTHONPATH='+shlex.quote(str(source)+':'+str(run)),
                'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1']
        provider = SlurmProvider(
            partition=os.environ.get(prefix+'_PARTITION', os.environ.get('ALF_PARTITION', 'your_gpu_partition')),
            account=os.environ.get('ALF_ACCOUNT') or None,
            nodes_per_block=1, init_blocks=0, min_blocks=0, max_blocks=1,
            cores_per_node=max(4, workers*cores),
            scheduler_options=os.environ.get(prefix+'_RESOURCE_OPTIONS', '#SBATCH --gres=gpu:4'),
            launcher=SingleNodeLauncher(),
            walltime=os.environ.get(prefix+'_WALLTIME', '01:00:00' if role=='ML' else '00:30:00'),
            worker_init='\n'.join(init)+'\n', cmd_timeout=60)
        executors.append(HighThroughputExecutor(
            label=f'alf_{role}_executor', max_workers_per_node=workers, cores_per_worker=cores,
            cpu_affinity='none', address=address_by_hostname(), provider=provider,
            heartbeat_period=10, heartbeat_threshold=180,
            interchange_launch_cmd=[python, '-m', 'parsl.executors.high_throughput.interchange'],
            launch_cmd=(worker_python or shlex.quote(python))+' -m parsl.executors.high_throughput.process_worker_pool'+
                       DEFAULT_LAUNCH_CMD.removeprefix('process_worker_pool.py')))
    return Config(executors=executors, run_dir=str(run/'runinfo'), retries=0,
                  max_idletime=60, strategy_period=5)


config = make_config()
