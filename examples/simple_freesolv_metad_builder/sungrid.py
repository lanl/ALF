import os
import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface
from parsl.providers import GridEngineProvider, LocalProvider
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SimpleLauncher
from parsl.channels import SSHChannel, LocalChannel

userhome = os.path.expanduser("~")
config_1node= Config(
    executors=[
        HighThroughputExecutor(
            label='alf_QM_executor',
            max_workers=80,
            provider=GridEngineProvider(
                channel=LocalChannel(userhome=userhome),
                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                parallelism=1,
                walltime="02:00:00",
                scheduler_options='#$ -pe smp 80',
                worker_init='source $HOME/.alfrc',
                launcher=SingleNodeLauncher(),
                cmd_timeout=30,
                queue='JG'
            ),
        ),
        HighThroughputExecutor(
            label='alf_ML_executor',
            max_workers=1,
            #provider=LocalProvider(
            #    channel=LocalChannel(userhome=userhome),
            #    nodes_per_block=1,
            #    init_blocks=0,
            #    min_blocks=0,
            #    max_blocks=1,
            #    parallelism=1,
            #    worker_init='source $HOME/.alfrc',
            #    cmd_timeout=30,
            #),
            provider=GridEngineProvider(
                channel=LocalChannel(userhome=userhome),
                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                parallelism=1,
                walltime="01:00:00",
                scheduler_options='#$ -pe smp 80\n#$ -l ngpus=4',
                worker_init='source $HOME/.alfrc',
                launcher=SingleNodeLauncher(),
                cmd_timeout=30,
                queue='UI-GPU'
            ),
        ),
        HighThroughputExecutor(
            label='alf_sampler_executor',
            max_workers=2,
            provider=GridEngineProvider(
                channel=LocalChannel(userhome=userhome),
                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                parallelism=1,
                walltime="01:00:00",
                scheduler_options='#$ -pe smp 80\n#$ -l ngpus=8',
                worker_init='source $HOME/.alfrc',
                launcher=SingleNodeLauncher(),
                cmd_timeout=30,
                queue='JG'
            ),
        )
    ]
)
