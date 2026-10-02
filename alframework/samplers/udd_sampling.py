"""Uncertainty-driven dynamics sampler task."""

import copy

from parsl import python_app
from parsl.app.errors import RemoteExceptionWrapper

from alframework.samplers.mlmd_sampling import simple_mlmd_sampling_task


@python_app(executors=['alf_sampler_executor'])
def simple_udd_sampling_task(molecule_object, sampler_config, model_path, current_model_id, gpus_per_node):
    """Run MLMD sampling with uncertainty-driven dynamics enabled.

    This task intentionally reuses the standard MLMD sampling implementation.
    The only UDD-specific behavior is validating and injecting the
    ``udd_bias_weight`` option used by ``MLMD_calculator``.
    """
    if sampler_config.get('udd_bias_weight') is None:
        raise ValueError("simple_udd_sampling_task requires 'udd_bias_weight' in sampler_config.")

    udd_sampler_config = copy.deepcopy(sampler_config)
    udd_bias_weight = udd_sampler_config.pop('udd_bias_weight')
    udd_sampler_config.setdefault('MLMD_calculator_options', {})
    udd_sampler_config['MLMD_calculator_options']['udd_bias_weight'] = udd_bias_weight

    result = simple_mlmd_sampling_task.func(
        molecule_object,
        udd_sampler_config,
        model_path,
        current_model_id,
        gpus_per_node,
    )
    if isinstance(result, RemoteExceptionWrapper):
        result.reraise()
    return result
