"""HIPPYNN for ALF: training ensembles, loading them for sampling, and evaluating them at scale.

- :mod:`.interface`: the training tasks, ground- and excited-state, and the loaders the samplers use;
- :mod:`.lightning`: data-parallel training of one member on several GPUs and nodes, with PyTorch Lightning;
- :mod:`.final_evaluation`: the streaming final test (``stream_final_evaluation``), which imports PyTorch and
  HIPPYNN when it is imported, so it is loaded only when used;
- :mod:`.compat`: what differs between HIPPYNN releases.

Configurations name the tasks and loaders from here, such as
``"alframework.ml_interfaces.hippynn.train_HIPPYNN_ensemble_task"``. The earlier module paths, such as
``alframework.ml_interfaces.hippynn_interface``, still import, with a ``FutureWarning`` naming the new one.
"""
from alframework.ml_interfaces.hippynn.interface import (
    HIPNN_ASE_calculator,
    HIPNN_ASE_load_ensemble,
    load_hippynn_batched_ensemble,
    load_hippynn_excited_state_ensemble,
    train_excited_state_HIPPYNN_ensemble_multinode_task,
    train_excited_state_HIPPYNN_ensemble_task,
    train_HIPPYNN_ensemble_multinode_task,
    train_HIPPYNN_ensemble_task,
)

__all__ = ['HIPNN_ASE_calculator', 'HIPNN_ASE_load_ensemble', 'load_hippynn_batched_ensemble',
           'load_hippynn_excited_state_ensemble', 'train_excited_state_HIPPYNN_ensemble_multinode_task',
           'train_excited_state_HIPPYNN_ensemble_task', 'train_HIPPYNN_ensemble_multinode_task',
           'train_HIPPYNN_ensemble_task']
