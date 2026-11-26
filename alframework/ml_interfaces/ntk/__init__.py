"""Model-agnostic neural tangent kernel (NTK) uncertainty.

One frozen potential, linearized in its weights θ, gives a Gaussian posterior over every quantity that is
linear in its energies. Each such quantity g has a sketched tangent row r_g = S ∂g/∂θ, where S is a
p × P Gaussian sketch, and a posterior variance λ r_gᵀ (A + λI)⁻¹ r_g. Here A sums the outer products of
the training rows, each divided by its observation scale.

The quantities are:

- each state's energy E_s;
- forces, as all components F_s or as random projections wᵀF_s;
- Hessian-vector products H_s v along probes;
- bilinear Hessian projections uᵀ H_s v;
- energy gaps E_{s+1} − E_s.

With independent Gaussian probes, E‖δF‖² = E_w[(wᵀδF)²] and E‖δH‖_F² = E_{u,v}[(uᵀδHv)²]. So a few
projections estimate the force and Hessian uncertainty without forming 3N force rows or a (3N)² Hessian.
This follows Projected Hessian Learning, arXiv:2603.04523. The energy-only basis is AdaptNTK,
arXiv:2609.00488.

Layout:

- ``adapter``: what a model must provide, and how structures are prepared.
- ``sketch``: seeded Gaussian sketch directions, generated in chunks.
- ``rows``: sketched rows by forward mode (``torch.func``) or batched reverse mode.
- ``posterior``: the information matrix, the ridge, and variances.
- ``estimator``: fitting on training geometries, scores, an ASE calculator and the packed evaluator.

Model adapters live with their models: ``alframework.ml_interfaces.hippynn.ntk`` and
``alframework.ml_interfaces.ntk.mace``.
"""

from alframework.ml_interfaces.ntk.adapter import NTKAdapter, Structure
from alframework.ml_interfaces.ntk.estimator import (NTKASECalculator, NTKEstimator, NTKPackedEvaluator,
                                                     reported_uncertainty)
from alframework.ml_interfaces.ntk.posterior import Posterior
from alframework.ml_interfaces.ntk.rows import RowRequest, compute_rows
from alframework.ml_interfaces.ntk.sketch import Sketch

__all__ = ["NTKAdapter", "Structure", "NTKEstimator", "NTKASECalculator", "NTKPackedEvaluator", "Posterior",
           "RowRequest", "compute_rows", "Sketch", "reported_uncertainty"]
