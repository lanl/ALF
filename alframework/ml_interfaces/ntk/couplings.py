"""Physical nonadiabatic couplings and their uncertainty, from a scaled coupling and a gap.

A coupling model can predict the scaled coupling s = ΔE·d, which stays finite at
crossings. The physical coupling is d = s/ΔE, with ΔE from an energy model. For independent uncertainties σ_s
(from the coupling model) and σ_ΔE (from the energy model), first-order propagation gives

    σ_d² = (σ_s / ΔE)² + (s σ_ΔE / ΔE²)²,

which grows as the gap closes. That's the regime where nonadiabatic transitions happen, and where a small
error in s becomes a large error in d.
"""

import torch


def physical_coupling_uncertainty(scaled, scaled_sigma, gap, gap_sigma=0.0):
    """Return ``(d, σ_d)`` elementwise. ``gap`` and ``gap_sigma`` broadcast against ``scaled``."""
    scaled, scaled_sigma = torch.as_tensor(scaled), torch.as_tensor(scaled_sigma)
    gap = torch.as_tensor(gap, dtype=scaled.dtype)
    gap_sigma = torch.as_tensor(gap_sigma, dtype=scaled.dtype)
    if bool((gap == 0).any()):
        raise ValueError("The physical coupling is undefined at a zero gap; score the scaled coupling there.")
    coupling = scaled / gap
    sigma = torch.sqrt((scaled_sigma / gap) ** 2 + (scaled * gap_sigma / gap ** 2) ** 2)
    return coupling, sigma
