"""Select sampler devices within worker visibility, without changing it.

The CUDA/HIP runtime owns mask interpretation and local device numbering.
In particular, ROCr and HIP masks must not be independently reindexed here.
The one exception is ``normalize_amd_visibility``, which removes a CUDA mask
that only repeats the ROCr mask on AMD nodes, where it would hide the GPU.
"""
import os


VISIBILITY_VARIABLES = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')
HIDDEN_MASKS = ('', '-1', 'NoDevFiles')
AMD_GPU_DRIVER = '/dev/kfd'


def _entries(mask):
    return [entry.strip() for entry in mask.split(',') if entry.strip()]


def normalize_amd_visibility(environ=os.environ, amd=None):
    """On an AMD GPU node, drop a ``CUDA_VISIBLE_DEVICES`` that only repeats ``ROCR_VISIBLE_DEVICES``.

    Parsl's HighThroughputExecutor gives a worker accelerator ``N`` by setting both masks to ``N``. On AMD,
    ROCr first narrows the node to that GPU, numbered 0, and HIP then applies ``CUDA_VISIBLE_DEVICES=N`` to
    what is left, so every worker but the one on GPU 0 sees no GPU (7 of 8 on a Frontier node). The repeat
    adds nothing there, so it is removed. NVIDIA nodes, where ``CUDA_VISIBLE_DEVICES`` is the mask that
    counts, and every other combination are left alone. ``amd`` defaults to whether the AMD GPU driver
    (``/dev/kfd``) is present. This must run before the process first uses the GPU runtime, which reads the
    masks once; importing ``alframework`` runs it. Returns whether the environment changed.
    """
    rocr, cuda = environ.get('ROCR_VISIBLE_DEVICES'), environ.get('CUDA_VISIBLE_DEVICES')
    if not rocr or cuda is None or environ.get('HIP_VISIBLE_DEVICES') is not None:
        return False
    if _entries(cuda) != _entries(rocr) or not _entries(rocr):
        return False
    if not (os.path.exists(AMD_GPU_DRIVER) if amd is None else amd):
        return False
    del environ['CUDA_VISIBLE_DEVICES']
    return True


def _assigned_masks(environ=os.environ):
    """Masks that give this process particular GPUs, as opposed to hiding them all."""
    return {key: environ[key] for key in VISIBILITY_VARIABLES
            if key in environ and environ[key].strip() not in HIDDEN_MASKS}


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
        assigned = _assigned_masks()
        if assigned:
            # Falling back to the CPU here would quietly run GPU work orders of magnitude slower.
            raise RuntimeError(
                f'This worker was given GPUs by {assigned}, but none is visible. On AMD GPUs this happens when '
                'CUDA_VISIBLE_DEVICES repeats ROCR_VISIBLE_DEVICES (as Parsl\'s available_accelerators sets '
                'them) and the GPU runtime started before alframework was imported. Set "device": "cpu" to '
                'run on the CPU on purpose.')
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
