"""Sketched tangent rows of energy functionals, by forward mode or batched reverse mode.

Forward mode pushes each sketch direction through the model with ``torch.func.jvp``, batched over a chunk
of directions with ``vmap``. It costs about p evaluations and gives rows for every output at once. Batched
reverse mode backpropagates each output with ``is_grads_batched`` and projects the parameter gradients onto
the sketch. It costs about one backward pass per output, and works for any model that supports autograd to
third order. The default chooses forward mode when the adapter supports it and the outputs outnumber
half the sketch width.
"""

from dataclasses import dataclass

import torch
from torch.func import jacrev, jvp, vmap

from alframework.ml_interfaces.ntk.adapter import check_finite

ENGINES = ("auto", "forward", "reverse")


@dataclass(frozen=True)
class RowRequest:
    """Which functionals to evaluate, and which of them need sketched rows.

    ``energy`` and ``forces`` request rows for every state's energy and for every force component.
    ``force_probes`` ``[k, N, 3]`` request rows for the projections wᵀF. ``hvp_probes`` ``[k, N, 3]``
    request rows for every component of H v. ``bilinear_probes`` ``(u, v)``, each ``[k, N, 3]``, request
    rows for u_kᵀ H v_k. ``outputs`` names smooth model outputs (see ``NTKAdapter.outputs``), returned
    under the keys ``output:<name>``. Force values are returned whenever any derivative is needed, or
    when ``force_values`` is set.
    """

    energy: bool = True
    forces: bool = False
    force_probes: torch.Tensor = None
    hvp_probes: torch.Tensor = None
    bilinear_probes: tuple = None
    force_values: bool = False
    outputs: tuple = ()

    @property
    def row_keys(self):
        keys = []
        if self.energy:
            keys.append("energy")
        if self.forces:
            keys.append("forces")
        if self.force_probes is not None:
            keys.append("force_projection")
        if self.hvp_probes is not None:
            keys.append("hvp")
        if self.bilinear_probes is not None:
            keys.append("bilinear")
        keys.extend(f"output:{name}" for name in self.outputs)
        return tuple(keys)

    @property
    def needs_derivatives(self):
        return (self.forces or self.force_values or self.force_probes is not None
                or self.hvp_probes is not None or self.bilinear_probes is not None)

    def output_count(self, n_states, n_atoms):
        count = 0
        if self.energy:
            count += 1
        if self.forces:
            count += 3 * n_atoms
        if self.force_probes is not None:
            count += len(self.force_probes)
        if self.hvp_probes is not None:
            count += 3 * n_atoms * len(self.hvp_probes)
        if self.bilinear_probes is not None:
            count += len(self.bilinear_probes[0])
        return n_states * count + len(self.outputs)


def _assemble(energy, gradient, hvp_of, request):
    """Build the requested values from the energy ``[S]``, its gradient ``[S, N, 3]``, and an HVP function."""
    values = {"energy": energy}
    if gradient is None:
        return values
    forces = -gradient
    values["forces"] = forces
    if request.force_probes is not None:
        values["force_projection"] = torch.einsum("snc,knc->sk", forces, request.force_probes)
    if request.hvp_probes is not None:
        values["hvp"] = torch.stack([hvp_of(v) for v in request.hvp_probes], dim=1)  # [S, k, N, 3]
    if request.bilinear_probes is not None:
        u, v = request.bilinear_probes
        values["bilinear"] = torch.stack([(hvp_of(vk) * uk).sum((-1, -2)) for uk, vk in zip(u, v)], dim=1)
    return values


def _with_outputs(values, adapter, structure, positions, params, request):
    if request.outputs:
        outputs = adapter.outputs(structure, positions, params, tuple(request.outputs))
        values.update({f"output:{name}": outputs[name] for name in request.outputs})
    return values


def _functional_values(adapter, structure, request, params):
    """Values as pure functions of ``params``, for use inside ``torch.func`` transforms."""
    x0 = structure.positions

    def energy(x):
        return adapter.energies(structure, x, params)

    if not request.needs_derivatives:
        values = _assemble(energy(x0), None, None, request)
    else:
        gradient_of = jacrev(energy)

        def hvp_of(v):
            return jacrev(lambda x: (gradient_of(x) * v).sum((-1, -2)))(x0)

        values = _assemble(energy(x0), gradient_of(x0), hvp_of, request)
    return _with_outputs(values, adapter, structure, x0, params, request)


def _autograd_values(adapter, structure, request):
    """Values with autograd graphs to the live parameters, kept for differentiation."""
    x = structure.positions.detach().clone().requires_grad_(request.needs_derivatives)
    energy = adapter.energies(structure, x)
    if not request.needs_derivatives:
        return _with_outputs(_assemble(energy, None, None, request), adapter, structure, x, None, request)

    def gradient(outputs):
        rows = []
        for value in outputs:
            g = torch.autograd.grad(value, x, create_graph=True, allow_unused=True)[0] if value.requires_grad else None
            rows.append(torch.zeros_like(x) if g is None else g)
        return torch.stack(rows)

    g = gradient(energy)
    values = _assemble(energy, g, lambda v: gradient((g * v).sum((-1, -2))), request)
    return _with_outputs(values, adapter, structure, x, None, request)


def _forward_rows(adapter, structure, request, sketch, keys):
    params = adapter.parameter_values()
    if tuple(params) != sketch.names:
        raise ValueError("The sketch does not match the adapter's trainable parameters.")

    def values(prm):
        return _functional_values(adapter, structure, request, prm)

    def tangent(directions):
        return {key: value for key, value in jvp(values, (params,), (directions,))[1].items() if key in keys}

    with adapter.forward_context():
        primal = {key: value.detach() for key, value in values(params).items()}
        chunks = [vmap(tangent)(sketch.tangents(c)) for c in range(sketch.n_chunks)] if keys else []
    rows = {key: torch.cat([chunk[key] for chunk in chunks]).movedim(0, -1).detach() for key in keys}
    return primal, rows


def _flat_gradients(part, parameters, adapter):
    """Parameter gradients ``[len(part), P]`` of each output, batched when the model's backward allows it."""
    def flatten(grads, count):
        return torch.cat([(g if g is not None else torch.zeros(count, *p.shape, dtype=p.dtype, device=p.device))
                          .reshape(count, -1) for g, p in zip(grads, parameters)], dim=1)

    if not part.requires_grad:
        return flatten([None] * len(parameters), len(part))
    if getattr(adapter, "_ntk_batched_reverse", True):
        eye = torch.eye(len(part), dtype=part.dtype, device=part.device)
        try:
            grads = torch.autograd.grad(part, parameters, grad_outputs=eye, retain_graph=True,
                                        allow_unused=True, is_grads_batched=True)
            return flatten(grads, len(part))
        except RuntimeError:
            # Some backward formulas can't be batched. Remember that for this adapter and loop instead.
            adapter._ntk_batched_reverse = False
    rows = []
    for value in part:
        grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        rows.append(flatten([None if g is None else g[None] for g in grads], 1))
    return torch.cat(rows)


def _reverse_rows(adapter, structure, request, sketch, keys, output_chunk):
    live = adapter.live_parameters()
    if tuple(live) != sketch.names:
        raise ValueError("The sketch does not match the adapter's trainable parameters.")
    parameters = list(live.values())
    with adapter.reverse_context():
        values = _autograd_values(adapter, structure, request)
        rows = {}
        for key in keys:
            flat = values[key].reshape(-1)
            blocks = []
            for start in range(0, len(flat), output_chunk):
                gradient = _flat_gradients(flat[start:start + output_chunk], parameters, adapter)
                blocks.append(sketch.project(gradient.to(sketch.dtype)))
            rows[key] = torch.cat(blocks).reshape(*values[key].shape, sketch.width).detach()
    return {key: value.detach() for key, value in values.items()}, rows


def compute_rows(adapter, structure, request, sketch, *, engine="auto", output_chunk=256):
    """Return ``(values, rows)`` for one structure.

    ``values[key]`` holds the predictions (energies ``[S]``, forces ``[S, N, 3]``, projections ``[S, k]``,
    HVPs ``[S, k, N, 3]``). ``rows[key]`` holds the sketched rows, with an extra trailing ``[p]`` axis, for
    the keys in ``request.row_keys``.
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}.")
    keys = request.row_keys
    if engine == "auto":
        many = request.output_count(len(adapter.state_names), structure.n_atoms) > sketch.width // 2
        engine = "forward" if adapter.supports_forward and many else "reverse"
    if engine == "forward" and not adapter.supports_forward:
        raise ValueError(f"{type(adapter).__name__} does not support forward-mode rows.")
    with torch.inference_mode(False), torch.enable_grad():
        if engine == "forward":
            values, rows = _forward_rows(adapter, structure, request, sketch, keys)
        else:
            values, rows = _reverse_rows(adapter, structure, request, sketch, keys, output_chunk)
    check_finite("predictions or NTK rows", *values.values(), *rows.values())
    return values, rows
