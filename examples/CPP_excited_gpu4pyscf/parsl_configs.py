"""Portable Slurm template; set ALF_PARTITION and site worker initialization.

ALF_{ML,SAMPLER,QM}_PYTHON select interpreters. ALF_{ML,SAMPLER,QM}_WORKER_INIT
can load CUDA/compiler modules and library paths. ALF_SOURCE optionally pins
an ALF checkout; otherwise use the installed package. Import submits no jobs.
Default limits are eight QM nodes, one training node and one sampling node.
All start empty; ordinary Parsl scaling queues nodes when work is available
and releases idle nodes. Override ALF_{QM,ML,SAMPLER}_MAX_BLOCKS for smaller runs.
One training task launches four ensemble members on four local GPUs.
QM workers are bound by Parsl to one allocated GPU UUID each, making its
local CUDA index zero. The manager retains scheduler visibility; training
and sampling retain their existing device assignment.
"""
import os
from pathlib import Path
import shlex
import sys

import alframework
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
    for role, workers, cores in [('ML', 1, 4), ('sampler', 4, 1), ('QM', 4, 8)]:
        prefix = 'ALF_'+role.upper()
        python = os.environ.get(prefix+'_PYTHON', sys.executable)
        init = [os.environ.get(prefix+'_WORKER_INIT', ''),
                'export PATH='+shlex.quote(str(Path(python).parent))+':"$PATH"',
                'export PYTHONPATH='+shlex.quote(str(source)+':'+str(run)),
                'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1']
        provider = SlurmProvider(
            partition=os.environ.get(prefix+'_PARTITION', os.environ.get('ALF_PARTITION', 'your_gpu_partition')),
            account=os.environ.get('ALF_ACCOUNT') or None,
            nodes_per_block=1, init_blocks=0, min_blocks=0,
            max_blocks=int(os.environ.get(prefix+'_MAX_BLOCKS', '8' if role=='QM' else '1')),
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
            launch_cmd=shlex.quote(python)+(' '+shlex.quote(str(run/'qm_worker_pool.py')) if role=='QM'
                       else ' -m parsl.executors.high_throughput.process_worker_pool')+
                       DEFAULT_LAUNCH_CMD.removeprefix('process_worker_pool.py')))
    return Config(executors=executors, run_dir=str(run/'runinfo'), retries=0,
                  max_idletime=60, strategy_period=5)


config = make_config()
