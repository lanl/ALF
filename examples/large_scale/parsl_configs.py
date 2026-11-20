"""A Slurm template for runs on many nodes: one Parsl worker pool per node, multi-node blocks.

Set the machine through environment variables before starting the coordinator; importing this submits nothing.

- ``ALF_PARTITION`` and ``ALF_ACCOUNT``; ``ALF_<ROLE>_RESOURCE_OPTIONS`` for extra ``#SBATCH`` lines, such as a
  GPU request, where ``<ROLE>`` is ``SAMPLER``, ``QM`` or ``ML``.
- ``ALF_GPUS_PER_NODE`` (default 4) and ``ALF_CORES_PER_NODE`` (default 32): what each node offers ALF.
- ``ALF_<ROLE>_NODES`` (default 4, 4 and 8): each role's nodes, as one block; ``ALF_<ROLE>_WALLTIME`` (default
  ``02:00:00``). A role starts its block when it first has work.
- ``ALF_INTERFACE``: the network interface that compute nodes reach the coordinator through, if the host name
  doesn't resolve to it.
- ``ALF_<ROLE>_WORKER_INIT``: site lines, such as loading modules.
- The node environment, read by ``alframework.tools.worker_setup.setup_from_environment``: ``ALF_ENV_TARBALL``
  (a packed environment each node unpacks once), ``ALF_NODE_LOCAL_DIR``, ``ALF_NODE_LOCAL_CACHES=1`` and
  ``ALF_THP_DISABLE=1``. See the Running at Scale guide.

Each block starts one worker pool per node (``SrunLauncher``), not one per GPU: the interchange then tracks one
manager per node, which is what kept dispatch working on 1,024 nodes. Sampler and QM pools run one worker per GPU,
each pinned to its own GPU (``available_accelerators``). The ML pool runs one worker per node, and each ensemble
member trains on that node's GPUs (``gpus_per_member`` in ``hippynn_config.json``).
"""
import os
import shlex
import sys

from alframework.tools.worker_setup import setup_from_environment
from parsl.addresses import address_by_hostname, address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.executors.high_throughput.executor import DEFAULT_LAUNCH_CMD
from parsl.launchers import SingleNodeLauncher, SrunLauncher
from parsl.providers import LocalProvider, SlurmProvider

GPUS_PER_NODE = int(os.environ.get('ALF_GPUS_PER_NODE', '4'))
CORES_PER_NODE = int(os.environ.get('ALF_CORES_PER_NODE', '32'))
# Each role's nodes by default, and whether its workers each take one GPU.
ROLES = {'sampler': (4, True), 'QM': (4, True), 'ML': (8, False)}


def _address():
    interface = os.environ.get('ALF_INTERFACE')
    return address_by_interface(interface) if interface else address_by_hostname()


def _executor(role, nodes, per_gpu):
    prefix = 'ALF_' + role.upper()
    setup, worker_python = setup_from_environment(prefix, all_nodes=nodes > 1)
    python = worker_python or shlex.quote(sys.executable)
    init = [os.environ.get(prefix + '_WORKER_INIT', ''), *setup,
            'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1']
    workers = GPUS_PER_NODE if per_gpu else 1
    provider = SlurmProvider(
        partition=os.environ.get(prefix + '_PARTITION', os.environ.get('ALF_PARTITION', 'your_gpu_partition')),
        account=os.environ.get('ALF_ACCOUNT') or None,
        nodes_per_block=nodes, init_blocks=0, min_blocks=0, max_blocks=1,
        cores_per_node=CORES_PER_NODE,
        scheduler_options=os.environ.get(prefix + '_RESOURCE_OPTIONS', ''),
        launcher=SrunLauncher() if nodes > 1 else SingleNodeLauncher(),
        walltime=os.environ.get(prefix + '_WALLTIME', '02:00:00'),
        worker_init='\n'.join(init) + '\n', cmd_timeout=120)
    return HighThroughputExecutor(
        label=f'alf_{role}_executor', max_workers_per_node=workers,
        cores_per_worker=max(1, CORES_PER_NODE // workers),
        available_accelerators=GPUS_PER_NODE if per_gpu else [],
        address=_address(), provider=provider, heartbeat_period=30, heartbeat_threshold=300,
        interchange_launch_cmd=[sys.executable, '-m', 'parsl.executors.high_throughput.interchange'],
        # Workers start from the node-local copy of the environment when there is one; the interchange runs here.
        launch_cmd=python + ' -m parsl.executors.high_throughput.process_worker_pool'
                   + DEFAULT_LAUNCH_CMD.removeprefix('process_worker_pool.py'))


def make_config():
    executors = [_executor(role, int(os.environ.get(f'ALF_{role.upper()}_NODES', nodes)), per_gpu)
                 for role, (nodes, per_gpu) in ROLES.items()]
    return Config(executors=executors, run_dir=os.path.join(os.getcwd(), 'runinfo'), retries=0,
                  max_idletime=600, strategy_period=5)


def make_debug_config():
    """Everything on the node the coordinator runs on, for ``--test_builder``, ``--test_qm`` and the like."""
    return Config(executors=[
        HighThroughputExecutor(label=f'alf_{role}_executor', max_workers_per_node=1, address='127.0.0.1',
                               provider=LocalProvider(init_blocks=1, max_blocks=1))
        for role in ROLES], run_dir=os.path.join(os.getcwd(), 'runinfo'), retries=0)


config = make_config()
config_debug = make_debug_config()
