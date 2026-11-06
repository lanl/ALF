"""Darwin template: set ALF_{ML,SAMPLER,QM}_PYTHON and ALF_SOURCE explicitly.

Import creates no allocations. The coordinator must outlive its workers.
"""
import os
from pathlib import Path
import shlex
import sys

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.executors.high_throughput.executor import DEFAULT_LAUNCH_CMD
from parsl.launchers import SingleNodeLauncher
from parsl.providers import SlurmProvider


def make_config():
    source=Path(os.environ.get('ALF_SOURCE',str(Path(__file__).resolve().parents[2]))).resolve()
    if not (source/'alframework').is_dir():
        raise ValueError('ALF_SOURCE must identify the pinned ALF source checkout')
    run=Path(os.environ.get('ALF_RUN_ROOT',os.getcwd())).resolve()
    executors=[]
    for role,workers,cores in [('ML',1,4),('sampler',4,1),('QM',1,4)]:
        python=os.environ.get('ALF_'+role.upper()+'_PYTHON',sys.executable)
        init=['source /etc/profile', 'module load '+shlex.quote(os.environ.get('ALF_CUDA_MODULE','cuda/13.2.1')),
              'export PATH='+shlex.quote(str(Path(python).parent))+':"$PATH"',
              'export PYTHONPATH='+shlex.quote(str(source)+':'+str(run)),
              'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
              'export CXX=/projects/opt/centos8/x86_64/gcc/11.2.0/bin/g++',
              'export CC=/projects/opt/centos8/x86_64/gcc/11.2.0/bin/gcc']
        provider=SlurmProvider(partition=os.environ.get('ALF_PARTITION','ml4chem'),account=os.environ.get('ALF_ACCOUNT','y2020-bf'),
            nodes_per_block=1,init_blocks=0,min_blocks=0,max_blocks=1,cores_per_node=4,
            scheduler_options='#SBATCH --exclusive\n#SBATCH --mem=0\n#SBATCH --constraint=gpu_vendor:nvidia&gpu_count:4',
            launcher=SingleNodeLauncher(),walltime='01:00:00' if role=='ML' else '00:30:00',
            worker_init='\n'.join(init)+'\n',cmd_timeout=60)
        executors.append(HighThroughputExecutor(label=f'alf_{role}_executor',max_workers_per_node=workers,
            cores_per_worker=cores,cpu_affinity='none',address=address_by_hostname(),provider=provider,
            heartbeat_period=10,heartbeat_threshold=180,
            interchange_launch_cmd=[python,'-m','parsl.executors.high_throughput.interchange'],
            launch_cmd=shlex.quote(python)+' -m parsl.executors.high_throughput.process_worker_pool'+
                       DEFAULT_LAUNCH_CMD.removeprefix('process_worker_pool.py')))
    return Config(executors=executors,run_dir=str(run/'runinfo'),retries=0,
                  max_idletime=60,strategy_period=5)


config=make_config()
