"""Single-host resources for the Cu(111) training example."""

from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider


LABELS = ('alf_sampler_executor', 'alf_QM_executor', 'alf_ML_executor')

# The sampler worker also builds surfaces. One worker per role limits local
# concurrency, while allowing VASP and model training to use their own processes.
config_local = Config(
    executors=[
        HighThroughputExecutor(label=label, address='127.0.0.1', max_workers_per_node=1,
                               provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1))
        for label in LABELS
    ],
    strategy='none',
    run_dir='runs/parsl',
)

config_debug = Config(
    executors=[ThreadPoolExecutor(label=label, max_threads=1) for label in LABELS],
    strategy='none',
    run_dir='runs/parsl-debug',
)
