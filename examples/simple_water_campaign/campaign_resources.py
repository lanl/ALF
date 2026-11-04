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
    campaign = profile.get('campaign', 'core')
    sampler_python = site['base_python'] if engine == 'ase' else site[engine + '_python']
    if campaign == 'mi250':
        sampler_python = site['rocm_python']
    # Parsl/dill code objects cannot safely cross Python 3.12 -> 3.11.
    # ALCHEMI's physics executes in a qualified 3.11 subprocess of this worker.
    transport_python = site['base_python'] if engine == 'alchemi' else sampler_python

    def executor(role, python, workers, cores, cpu=False):
        amd = campaign == 'mi250' and role == 'sampler'
        initialization = [
            'source /etc/profile',
            'export PATH=' + shlex.quote(str(Path(python).parent)) + ':"$PATH"',
            'export PYTHONPATH=' + shlex.quote(str(source) + ':' + str(source / 'examples/simple_water_campaign')),
            'export ALF_CAMPAIGN_RUN=' + shlex.quote(str(run)),
            'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HIPPYNN_USE_CUSTOM_KERNELS=False',
        ]
        for variable, key in (('CXX', 'cxx'), ('CC', 'cc')):
            if site.get(key):
                initialization.append('export ' + variable + '=' + shlex.quote(site[key]))
        if not cpu and not amd and site.get('cuda_module'):
            initialization.append('module load ' + shlex.quote(site['cuda_module']))
        if campaign == 'pyseqm':
            initialization.append('export PYTHONPATH=' + shlex.quote(site['pyseqm_source']) + ':"$PYTHONPATH"')
        if amd:
            initialization.append('module load ' + shlex.quote(site['rocm_module']))
        if cpu:
            initialization.append('module load ' + shlex.quote(site['mpi_module']))
            initialization += ['export PARSL_CORES=16',
                               'export OMPI_MCA_hwloc_base_binding_policy=none']
        provider = CampaignSlurmProvider(
            partition=site['cpu_partition'] if cpu else site['rocm_partition'] if amd else site['gpu_partition'], account=site['account'],
            init_blocks=0, min_blocks=0, max_blocks=1, nodes_per_block=1,
            cores_per_node=None if cpu else 4,
            # Open MPI counts Slurm tasks, not cpus-per-task, as MPI slots.
            scheduler_options=('#SBATCH --ntasks-per-node=16\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=40G'
                               if cpu else '#SBATCH --mem=0'),
            worker_init='\n'.join(initialization), launcher=SingleNodeLauncher(), walltime='04:00:00',
            cmd_timeout=60)
        return HighThroughputExecutor(
            label=f'alf_{role}_executor', max_workers_per_node=workers, cores_per_worker=cores,
            cpu_affinity='block' if cpu else 'none',
            address=address_by_hostname(), provider=provider, heartbeat_period=10, heartbeat_threshold=120,
            interchange_launch_cmd=[python, '-m', 'parsl.executors.high_throughput.interchange'],
            launch_cmd=shlex.quote(python) + ' -m parsl.executors.high_throughput.process_worker_pool' +
                       DEFAULT_LAUNCH_CMD.removeprefix('process_worker_pool.py'))

    executors = [executor('ML', site['base_python'], 1, 4),
                 executor('sampler', transport_python, 1 if campaign == 'pyseqm' else 4, 1)]
    if campaign != 'pyseqm':
        executors.insert(0, executor('QM', site['base_python'], 2, profile['orca_cores'], cpu=True))
    return Config(executors=executors,
                  run_dir=str(Path(run) / 'runinfo'), max_idletime=60, retries=0, strategy_period=5)


config = make_config(Path(os.environ['ALF_CAMPAIGN_RUN']))
