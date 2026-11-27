"""LoRA on UMA's invariant radial MLPs and export to Fairchem's native format."""

from copy import deepcopy
import math
import os
from pathlib import Path

import torch
from torch.nn.utils import parametrize


class RadialLoRA(torch.nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        out_features, in_features = weight.shape
        self.a = torch.nn.Parameter(weight.new_empty(rank, in_features))
        self.b = torch.nn.Parameter(weight.new_zeros(out_features, rank))
        torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        return weight + self.scale * (self.b @ self.a)


def install_lora(model, rank, alpha):
    """Freeze the original backbone and adapt two invariant radial linears in each block."""
    backbone = model.backbone
    if backbone.regress_config.direct_forces:
        raise ValueError('UMA LoRA requires a conservative UMA-S checkpoint.')
    targets = {}
    for index, block in enumerate(backbone.blocks):
        radial = getattr(getattr(getattr(block, 'edge_wise', None), 'so2_conv_1', None), 'rad_func', None)
        net = getattr(radial, 'net', None)
        for layer in (0, 3):
            module = net[layer] if isinstance(net, torch.nn.Sequential) and len(net) > layer else None
            if type(module) is not torch.nn.Linear or rank > min(module.weight.shape):
                raise ValueError(f'Unsupported UMA LoRA target at block {index}, radial layer {layer}, rank {rank}.')
            targets[f'backbone.blocks.{index}.edge_wise.so2_conv_1.rad_func.net.{layer}'] = module
    if not targets:
        raise ValueError('The UMA checkpoint has no supported radial LoRA targets.')
    backbone.requires_grad_(False)
    for module in targets.values():
        parametrize.register_parametrization(module, 'weight', RadialLoRA(module.weight, rank, alpha))
    return targets


def _resume_config(model_config, rank, alpha):
    return {'_target_': f'{__name__}.rebuild_lora_model', '_recursive_': False,
            'model_config': deepcopy(model_config), 'rank': rank, 'alpha': alpha}


def initialize_lora_model(checkpoint_location, overrides, heads, rank, alpha):
    """Fairchem training initializer for an ALF LoRA round."""
    from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model

    model = initialize_finetuning_model(checkpoint_location, overrides, heads)
    install_lora(model, rank, alpha)
    model.finetune_model_full_config = _resume_config(model.finetune_model_full_config, rank, alpha)
    return model


def rebuild_lora_model(model_config, rank, alpha):
    """Hydra target for retained Fairchem training checkpoints."""
    from hydra.utils import instantiate

    model = instantiate(model_config)
    install_lora(model, rank, alpha)
    model.finetune_model_full_config = _resume_config(model_config, rank, alpha)
    return model


def _merge_weights(state, rank, alpha):
    factors = {}
    suffix = '.parametrizations.weight.original'
    for key in list(state):
        if key.endswith(suffix):
            name = key.removesuffix(suffix)
            base = state.pop(key)
            a = state.pop(f'{name}.parametrizations.weight.0.a')
            b = state.pop(f'{name}.parametrizations.weight.0.b')
            state[f'{name}.weight'] = base + (alpha / rank) * (b @ a)
            factors[name.removeprefix('module.')] = {'base': base, 'a': a, 'b': b}
    if not factors or any('.parametrizations.' in key for key in state):
        raise ValueError('The UMA LoRA checkpoint has an unexpected parameter layout.')
    return factors


def _publish(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_lora_checkpoint(source, merged_path, factors_path):
    """Publish a native checkpoint and the exact EMA factorization used for NTK."""
    checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    options = checkpoint.model_config
    rank, alpha = int(options['rank']), float(options['alpha'])
    regular = _merge_weights(checkpoint.model_state_dict, rank, alpha)
    ema = _merge_weights(checkpoint.ema_state_dict, rank, alpha)
    if regular.keys() != ema.keys():
        raise ValueError('UMA regular and EMA LoRA targets differ.')
    checkpoint.model_config = options['model_config']
    _publish(factors_path, {'rank': rank, 'alpha': alpha, 'targets': ema})
    _publish(merged_path, checkpoint)


def restore_ema_lora(model, path):
    """Restore LoRA in place on the native EMA model, keeping the deployed function."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    targets = install_lora(model, saved['rank'], saved['alpha'])
    if targets.keys() != saved['targets'].keys():
        raise ValueError('UMA LoRA factors do not match the checkpoint targets.')
    with torch.no_grad():
        for name, module in targets.items():
            weights = saved['targets'][name]
            registered = module.parametrizations.weight
            deployed = registered.original.detach().clone()
            registered.original.copy_(weights['base'])
            registered[0].a.copy_(weights['a'])
            registered[0].b.copy_(weights['b'])
            if not torch.equal(deployed, module.weight):
                raise ValueError(f'UMA LoRA factors do not reproduce {name}.')
