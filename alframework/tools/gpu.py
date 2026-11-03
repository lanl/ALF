"""Select sampler devices within worker visibility, without changing it.

The CUDA/HIP runtime owns mask interpretation and local device numbering.
In particular, ROCr and HIP masks must not be independently reindexed here.
"""
import os


VISIBILITY_VARIABLES = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')


def _cuda_device(requested, count, gpus_per_node, masked):
    if count < 1:
        raise RuntimeError('No GPU is available within the worker visibility; check the executor allocation.')
    if requested not in (None, 'cuda'):
        try:
            index = int(requested.removeprefix('cuda:'))
        except ValueError as exc:
            raise ValueError(f'Invalid CUDA device: {requested!r}.') from exc
        if index < 0 or index >= count:
            raise ValueError(f'CUDA device {requested!r} is outside the {count} visible local devices.')
    else:
        width = count
        if not masked and gpus_per_node not in (None, 0):
            if not isinstance(gpus_per_node, int) or gpus_per_node < 1:
                raise ValueError('gpus_per_node must be a nonnegative integer or None.')
            width = min(width, gpus_per_node)
        rank = int(os.environ.get('PARSL_WORKER_RANK', '0'))
        if rank < 0:
            raise ValueError('PARSL_WORKER_RANK must be nonnegative.')
        index = rank % width
    return f'cuda:{index}'


def sampling_device(sampler_config, gpus_per_node=None):
    """Resolve a local Torch device; default to automatic GPU/CPU selection.

    An explicit non-CUDA device (including CPU) requires no Torch import.
    ``cuda:N`` pins a visible ordinal; ``cuda`` and automatic GPU selection
    distribute workers by rank. Existing masks take precedence over the
    unmasked ``gpus_per_node`` compatibility cap.
    """
    requested = sampler_config.get('device')
    if requested is not None:
        requested = str(requested)
        if requested != 'cuda' and not requested.startswith('cuda:'):
            return requested
    try:
        import torch
    except ImportError:
        count = 0
    else:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if requested is None and count == 0:
        return 'cpu'
    return _cuda_device(requested, count, gpus_per_node,
                        any(key in os.environ for key in VISIBILITY_VARIABLES))


def neurochem_device(sampler_config, gpus_per_node):
    """Resolve a CUDA ordinal for the Torch-free NeuroChem inference adapter.

    NeuroChem only targets CUDA. With no Torch runtime, CUDA mask entries
    determine the visible width; an unmasked legacy task supplies its width
    through ``gpus_per_node``. No physical identifiers are passed to the model.
    """
    requested = sampler_config.get('device') or 'cuda'
    if requested != 'cuda' and not requested.startswith('cuda:'):
        raise ValueError('NeuroChem sampling requires a CUDA device.')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask is None:
        if not isinstance(gpus_per_node, int) or gpus_per_node < 1:
            raise ValueError('Unmasked NeuroChem sampling requires positive gpus_per_node.')
        count = gpus_per_node
    else:
        count = 0
        for token in mask.split(','):
            token = token.strip()
            if not token or token in ('-1', 'NoDevFiles'):
                break
            if not token.isdecimal() and not token.startswith(('GPU-', 'MIG-')):
                raise ValueError('Unrecognized CUDA_VISIBLE_DEVICES entry for NeuroChem.')
            count += 1
    return _cuda_device(requested, count, gpus_per_node, mask is not None)
