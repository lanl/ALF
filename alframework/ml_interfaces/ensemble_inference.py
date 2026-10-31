"""Physical ensemble predictions for sampling, independent of propagation.

An evaluator maps a list of ``ase.Atoms`` to unbiased per-member predictions.
It does not apply UDD, well/confinement potentials, thresholds, or any other
sampling policy; those belong to samplers and acquisition. Sampling screens
and ranks frames using these physical predictions only.

Prediction layout, for ``n_models`` members and ``n_systems`` structures:

- ``energies``: ``[n_models, n_systems]`` in eV.
- ``forces``: a list of ``n_systems`` arrays shaped ``[n_models, n_atoms_i, 3]``
  in eV/Angstrom. Structures may have different sizes.
"""

import numpy as np


class ASEEnsembleEvaluator:
    """Evaluate a list of ASE calculators one structure at a time.

    This adapter serves any ensemble loader that returns ASE calculators, such
    as ``hippynn_interface.HIPNN_ASE_load_ensemble``. It runs sequentially;
    native batched evaluators may implement the same ``predict`` method.
    """

    def __init__(self, calculators):
        if len(calculators) < 2:
            raise ValueError("An uncertainty ensemble requires at least two calculators.")
        self.calculators = list(calculators)

    @property
    def n_models(self):
        return len(self.calculators)

    def predict(self, atoms_list):
        energies = np.empty((self.n_models, len(atoms_list)))
        forces = []
        for system, atoms in enumerate(atoms_list):
            member_forces = np.empty((self.n_models, len(atoms), 3))
            for member, calculator in enumerate(self.calculators):
                calculator.calculate(atoms, properties=["energy", "forces"])
                energies[member, system] = calculator.results["energy"]
                member_forces[member] = calculator.results["forces"]
            forces.append(member_forces)
        return {"energies": energies, "forces": forces}


def ensemble_statistics(prediction):
    """Return per-structure uncertainty signals from member predictions.

    Population standard deviations across members, matching
    ``MLMD_calculator``: ``Es`` = std of energies; ``Fs`` and ``Fsmax`` = mean
    and max over atoms and components of the per-component force std; ``Fsrms``
    = root mean square of the same per-component force std.
    """
    energies = np.asarray(prediction["energies"], dtype=float)
    rows = []
    for system, member_forces in enumerate(prediction["forces"]):
        force_std = np.abs(np.std(np.asarray(member_forces, dtype=float), axis=0))
        rows.append({
            "Es": float(np.std(energies[:, system])),
            "Fs": float(np.mean(force_std)),
            "Fsmax": float(np.max(force_std)),
            "Fsrms": float(np.sqrt(np.mean(force_std ** 2))),
            "energy": float(np.mean(energies[:, system])),
        })
    return rows
