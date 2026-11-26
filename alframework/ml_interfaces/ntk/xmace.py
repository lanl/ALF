"""X-MACE excited-state models (github.com/rhyan10/X-MACE) for ALF's NTK uncertainty.

X-MACE adds excited states to a ground-state MACE backbone, which can come from a foundation model:
E_s = E_backbone + scale · readout_s + n_atoms · shift_s. It installs as the package ``mace``, in place of
mace-torch. This adapter reuses the MACE adapter (neighbor lists, batched reverse-mode rows) and returns
the state energies ``[S]``.

Adiabatic state energies are not smooth where states cross, and neither are their NTK rows and Hessians.
:meth:`XMACENTKAdapter.outputs` exposes quantities that stay smooth there:

- ``ground_energy`` ``[1]``: the backbone surface.
- ``state_offsets`` ``[S]``: each state's energy relative to the backbone. For ``ExcitedMACE`` these are
  independent per-state readouts.
- ``latent`` ``[K]``: the permutation-invariant vector of ``AutoencoderExcitedMACE``.
- ``state_matrix``: the elements of the symmetric tridiagonal matrix whose eigenvalues
  ``AutoencoderExcitedMACE`` reports, in eV. It holds each block's diagonal, then its off-diagonal.
- ``smooth_nacs`` ``[N, P, 3]``: nonadiabatic couplings in X-MACE's smooth form, NAC × ΔE. This costs
  3N·P rows.
"""

import torch
from torch.func import functional_call

from alframework.ml_interfaces.ntk.mace import MACENTKAdapter

SMOOTH_OUTPUTS = ("ground_energy", "state_offsets", "latent", "state_matrix", "smooth_nacs")


def _subparameters(params, prefix):
    return {name[len(prefix):]: value for name, value in params.items() if name.startswith(prefix)}


class XMACENTKAdapter(MACENTKAdapter):
    """A frozen copy of an X-MACE model (``ExcitedMACE``, ``AutoencoderExcitedMACE``, or ``LEHExcitedMACE``).

    ``head`` selects one of a multi-head model's data sets. ``parameters`` defaults to the checkpoint's
    trainable parameters. ``"excited"`` keeps only the excited-state readouts and adapter, not the
    backbone. Use it for LoRA fine-tunes, which X-MACE saves with the low-rank weights merged.
    """

    def __init__(self, model, *, parameters="trainable", **options):
        if parameters == "excited":
            # The excited-state readouts and adapter only: right for LoRA fine-tunes, whose low-rank backbone
            # weights X-MACE merges into the backbone when training ends.
            def parameters(name, parameter):
                return not name.startswith("backbone.")
        super().__init__(model, parameters=parameters, **options)
        self.autoencoder = hasattr(self.model, "perm_decoders")

    def _forward(self, structure, positions, params):
        graph = dict(structure.extra["mace"])
        graph["positions"] = positions
        options = {"compute_force": False, "training": False}
        if params is None:
            return self.model(graph, **options)
        return functional_call(self.model, params, (graph,), options)

    def _head_index(self, structure):
        head = structure.extra["mace"].get("head")
        return 0 if head is None else int(head.reshape(-1)[0])

    def outputs(self, structure, positions, params=None, names=()):
        unknown = set(names) - set(SMOOTH_OUTPUTS)
        if unknown:
            raise ValueError(f"Unknown X-MACE outputs {sorted(unknown)}; choose from {list(SMOOTH_OUTPUTS)}.")
        output = self._forward(structure, positions, params)
        values = {}
        if "ground_energy" in names:
            values["ground_energy"] = output["ground_energy"].reshape(1)
        if "state_offsets" in names:
            values["state_offsets"] = (output["energy"] - output["ground_energy"][:, None]).reshape(-1)
        if "latent" in names or "state_matrix" in names:
            if not self.autoencoder:
                raise ValueError("latent and state_matrix need an AutoencoderExcitedMACE model.")
            latent = output["invariant_vals"]
            values["latent"] = latent.reshape(-1)
        if "state_matrix" in names:
            h = self._head_index(structure)
            decoder = self.model.perm_decoders[h]
            prefix = f"perm_decoders.{h}.decoder_nn."
            if params is None:
                elements = decoder.decoder_nn(latent)
            else:
                elements = functional_call(decoder.decoder_nn, _subparameters(params, prefix), (latent,))
            # In eV: scale the decoded matrix and add the baseline (backbone plus shift) to its diagonal.
            scale = self.model.delta_scale[h]
            baseline = (output["energy"] - scale * self._decoded(decoder, elements)).reshape(-1)
            diagonal = self._diagonal_mask(decoder, elements.shape[-1], elements.device)
            # The baseline is shared within each multiplicity block, so adding it to every diagonal
            # element shifts each block's eigenvalues by it.
            baseline_per_element = self._block_values(decoder, baseline)
            values["state_matrix"] = (scale * elements.reshape(-1)
                                      + diagonal * baseline_per_element)
        if "smooth_nacs" in names:
            if output.get("nacs") is None:
                raise ValueError("This X-MACE model predicts no NACs.")
            values["smooth_nacs"] = output["nacs"][: structure.n_atoms]
        return values

    @staticmethod
    def _decoded(decoder, elements):
        """The eigenvalues the decoder forms from ``elements``, without the backbone baseline."""
        energies, start = [], 0
        for n in decoder.block_sizes:
            diagonal = elements[:, start:start + n]
            off = elements[:, start + n:start + 2 * n - 1]
            matrix = torch.diag_embed(diagonal) + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
            energies.append(torch.linalg.eigvalsh(matrix))
            start += 2 * n - 1
        return torch.cat(energies, dim=1)

    @staticmethod
    def _diagonal_mask(decoder, size, device):
        mask, start = torch.zeros(size, device=device), 0
        for n in decoder.block_sizes:
            mask[start:start + n] = 1.0
            start += 2 * n - 1
        return mask

    @staticmethod
    def _block_values(decoder, per_state):
        """Spread a per-state vector (constant within each block) over the matrix elements."""
        pieces, state = [], 0
        for n in decoder.block_sizes:
            value = per_state[state]
            pieces.append(value.expand(2 * n - 1))
            state += n
        return torch.cat(pieces)
