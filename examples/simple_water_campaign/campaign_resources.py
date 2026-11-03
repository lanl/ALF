"""Darwin-only resource profile; imported by ALF from a generated run directory."""
import os
from pathlib import Path
import shlex

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.executors.high_throughput.executor import DEFAULT_LAUNCH_CMD
from parsl.launchers import SingleNodeLauncher
from parsl.providers import SlurmProvider

from campaign_support import event, read_json


class CampaignSlurmProvider(SlurmProvider):
    """Journal exact job IDs so supervisor cleanup cannot affect other work."""
    def submit(self, command, tasks_per_node, job_name='parsl'):
        job_id = super().submit(command, tasks_per_node, job_name)
        if job_id:
            event('job_submitted', job_id=str(job_id))
        return job_id


def make_config(run):
    spec = read_json(Path(run) / 'run.json')
    site, profile = spec['site'], spec['profile']
    source = Path(spec['source'])
    engine = (spec['case'] or {}).get('engine', 'ase')
    sampler_python = site['base_python'] if engine == 'ase' else site[engine + '_python']
    # Parsl/dill code objects cannot safely cross Python 3.12 -> 3.11.
    # ALCHEMI's physics executes in a qualified 3.11 subprocess of this worker.
    transport_python = site['base_python'] if engine == 'alchemi' else sampler_python

    def executor(role, python, workers, cores, cpu=False):
        initialization = [
            'source /etc/profile',
            'export PATH=' + shlex.quote(str(Path(python).parent)) + ':"$PATH"',
            'export PYTHONPATH=' + shlex.quote(str(source) + ':' + str(source / 'examples/simple_water_campaign')),
            'export ALF_CAMPAIGN_RUN=' + shlex.quote(str(run)),
            'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HIPPYNN_USE_CUSTOM_KERNELS=False',
        ]
        if cpu:
            initialization.append('module load ' + shlex.quote(site['mpi_module']))
        provider = CampaignSlurmProvider(
            partition=site['cpu_partition'] if cpu else site['gpu_partition'], account=site['account'],
            init_blocks=0, min_blocks=0, max_blocks=1, nodes_per_block=1,
            cores_per_node=16 if cpu else 4,
            scheduler_options='#SBATCH --mem=40G' if cpu else '#SBATCH --mem=0',
            worker_init='\n'.join(initialization), launcher=SingleNodeLauncher(), walltime='04:00:00',
            cmd_timeout=60)
        return HighThroughputExecutor(
            label=f'alf_{role}_executor', max_workers_per_node=workers, cores_per_worker=cores,
            address=address_by_hostname(), provider=provider, heartbeat_period=10, heartbeat_threshold=120,
            interchange_launch_cmd=[python, '-m', 'parsl.executors.high_throughput.interchange'],
            launch_cmd=shlex.quote(python) + ' -m parsl.executors.high_throughput.process_worker_pool' +
                       DEFAULT_LAUNCH_CMD.removeprefix('process_worker_pool.py'))

    return Config(executors=[executor('QM', site['base_python'], 2, profile['orca_cores'], cpu=True),
                             executor('ML', site['base_python'], 1, 4),
                             executor('sampler', transport_python, 4, 1)],
                  run_dir=str(Path(run) / 'runinfo'), max_idletime=60, retries=0, strategy_period=5)


config = make_config(Path(os.environ['ALF_CAMPAIGN_RUN']))
