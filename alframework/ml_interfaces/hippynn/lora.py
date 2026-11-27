"""Low-rank adaptation (LoRA) of a HIPPYNN graph, for fine-tuning a pretrained model.

Every matrix-shaped weight W of the targeted nodes becomes W₀ + (α/r)·B·A:
- ``{name}_lora_base`` holds the frozen pretrained W₀;
- ``{name}_lora_A`` (r × in) starts random;
- ``{name}_lora_B`` (out × r) starts at zero, so the adapted model starts exactly as the pretrained one.

A forward pre-hook rebuilds W from them before each call. It's a module-level function, so the model
pickles as HIPPYNN's ``experiment_structure.pt`` requires; torch's own parametrizations don't pickle. The
other parameters of the targeted nodes, such as biases, are frozen. Parameters outside them, such as the
energy heads, keep their trainable state. Parameter containers (``ParameterList``), which are indexed
rather than called, are frozen but not adapted. A weight of more than two dimensions, such as HIP-NN's
interaction tensor ``[n_dist, out, in]``, is adapted as the matrix ``[n_dist·out, in]``.

The saved copy keeps this form, and every ALF loader reads it, since ALF is importable wherever it loads.
Its trainable parameters are exactly what LoRA trained (the factors and the heads), which is what NTK
linearizes in by default. :func:`merge_lora` folds the factors into plain weights, for use outside ALF.
"""

from functools import partial
import math

import torch

SUFFIXES = ("_lora_base", "_lora_A", "_lora_B")


def _compose(module, name):
    base = getattr(module, f"{name}_lora_base")
    update = getattr(module, f"{name}_lora_B") @ getattr(module, f"{name}_lora_A")
    return base + module._alf_lora_scale[name] * update.reshape(base.shape)


def _before(names, module, args):
    for name in names:
        setattr(module, name, _compose(module, name))


def _after(names, module, args, output):
    # Keep no autograd graph on the module between calls, so it stays copyable and picklable.
    for name in names:
        setattr(module, name, getattr(module, name).detach())


def node_prefixes(model, targets=None):
    """Parameter-name prefixes of the nodes to adapt: ``targets`` (node names), or every network node."""
    from hippynn.graphs.nodes.tags import Network

    by_node = {node: f"moddict.{key}." for node, key in model.names_dict.items()}
    if targets is None:
        prefixes = [prefix for node, prefix in by_node.items() if isinstance(node, Network)]
        if not prefixes:
            raise ValueError("The graph has no network node to adapt; name the nodes with lora_targets.")
        return prefixes
    names = {node.name: prefix for node, prefix in by_node.items()}
    missing = [name for name in targets if name not in names]
    if missing:
        raise ValueError(f"lora_targets {missing} are not nodes of the graph; it has {sorted(names)}.")
    return [names[name] for name in targets]


def apply_lora(model, *, rank=4, alpha=1.0, targets=None, seed=0):
    """Adapt ``model`` in place. Returns the names of the new low-rank parameters."""
    prefixes = node_prefixes(model, targets)
    generator = torch.Generator().manual_seed(int(seed))
    adapted = 0
    for module_name, module in list(model.named_modules()):
        if not any((module_name + ".").startswith(prefix) for prefix in prefixes):
            continue
        if isinstance(module, (torch.nn.ParameterList, torch.nn.ParameterDict)):
            # Containers are indexed, never called, so a forward hook couldn't rebuild their weights.
            continue
        names = [name for name, p in module.named_parameters(recurse=False)
                 if p.ndim >= 2 and p.is_floating_point() and not name.endswith(SUFFIXES)]
        if not names:
            continue
        module._alf_lora_scale = {}
        for name in names:
            weight = module._parameters.pop(name)
            rows, columns = math.prod(weight.shape[:-1]), weight.shape[-1]
            r = max(1, min(int(rank), rows, columns))
            a = torch.randn(r, columns, generator=generator, dtype=torch.float64) / math.sqrt(columns)
            module.register_parameter(f"{name}_lora_base", torch.nn.Parameter(weight.detach().clone(),
                                                                              requires_grad=False))
            module.register_parameter(f"{name}_lora_A", torch.nn.Parameter(a.to(weight.dtype).to(weight.device)))
            module.register_parameter(f"{name}_lora_B", torch.nn.Parameter(
                torch.zeros(rows, r, dtype=weight.dtype, device=weight.device)))
            module._alf_lora_scale[name] = float(alpha) / r
            setattr(module, name, _compose(module, name).detach())
        module.register_forward_pre_hook(partial(_before, tuple(names)))
        module.register_forward_hook(partial(_after, tuple(names)))
        adapted += len(names)
    if not adapted:
        raise ValueError("The targeted nodes have no matrix-shaped weights to adapt.")
    lora_names = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            is_factor = name.endswith(("_lora_A", "_lora_B"))
            parameter.requires_grad_(is_factor)
            if is_factor:
                lora_names.append(name)
    return lora_names


def is_lora_factor(name):
    return name.endswith(("_lora_A", "_lora_B"))


def lora_parameter_count(model):
    return sum(p.numel() for name, p in model.named_parameters() if is_lora_factor(name))


def merge_lora(model):
    """Fold every low-rank update into its weight, in place, leaving a plain HIPPYNN model."""
    merged = 0
    for module in model.modules():
        scales = getattr(module, "_alf_lora_scale", None)
        if not scales:
            continue
        for name in list(scales):
            weight = _compose(module, name).detach()
            for suffix in SUFFIXES:
                del module._parameters[f"{name}{suffix}"]
            module.__dict__.pop(name, None)
            module.register_parameter(name, torch.nn.Parameter(weight))
            merged += 1
        for hooks in (module._forward_pre_hooks, module._forward_hooks):
            for key in [k for k, hook in hooks.items() if getattr(hook, "func", None) in (_before, _after)]:
                del hooks[key]
        del module._alf_lora_scale
    return merged
