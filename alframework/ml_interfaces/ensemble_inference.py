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

# A gap_pair value meaning every adjacent pair: gap_std is the largest of their spreads.
ADJACENT_GAPS = 'adjacent'


class SelectedStateEvaluator:
    """Physical selected-state view of a multi-state evaluator, retaining all heads."""

    def __init__(self, evaluator, state, gap_pair=None):
        self.evaluator, self.selected_state, self.gap_pair = evaluator, state, gap_pair
        self.n_models = evaluator.n_models
        if not 0 <= state < evaluator.n_states:
            raise ValueError('Selected state is absent from the ensemble')
        if gap_pair == ADJACENT_GAPS:
            if evaluator.n_states < 2:
                raise ValueError('Adjacent gaps require at least two states')
        elif gap_pair is not None and not 0 <= gap_pair[0] < gap_pair[1] < evaluator.n_states:
            raise ValueError('Gap pair is absent from the ensemble')

    def predict(self, atoms_list):
        states = self.evaluator.predict_states(atoms_list)
        return dict(energies=states['state_energies'][:, self.selected_state],
                    forces=[f[:, self.selected_state] for f in states['state_forces']],
                    selected_state=self.selected_state, gap_pair=self.gap_pair, **states)

    def evaluate(self, *args):
        self.last_states = self.evaluator.evaluate_states(*args)
        energies, forces = self.last_states
        return energies[:, self.selected_state], forces[:, self.selected_state]


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
    if 'state_energies' in prediction:
        from alframework.tools.excited_state_tools import gap_key_for_pair
        state_energies = np.asarray(prediction['state_energies'], dtype=float)
        state_forces = prediction['state_forces']
        if state_energies.ndim != 3 or state_energies.shape[::2] != energies.shape:
            raise ValueError('Invalid member/state/system prediction axes')
        if not np.isfinite(state_energies).all():
            raise ValueError('Nonfinite state energies')
        for state in range(state_energies.shape[1]):
            stats = ensemble_statistics(dict(energies=state_energies[:, state],
                                            forces=[np.asarray(f)[:, state] for f in state_forces]))
            for row, values in zip(rows, stats):
                row.update({f'E{state}_mean': values['energy'], f'E{state}_stdev': values['Es'],
                            f'F{state}_stdev_mean': values['Fs'], f'F{state}_stdev_max': values['Fsmax'],
                            f'F{state}_stdev_rms': values['Fsrms']})
        for lower in range(state_energies.shape[1]):
            for upper in range(lower+1, state_energies.shape[1]):
                gaps = state_energies[:, upper] - state_energies[:, lower]
                key = gap_key_for_pair(lower, upper)
                for index, row in enumerate(rows):
                    row[key+'_mean'] = float(gaps[:, index].mean())
                    row[key+'_stdev'] = float(gaps[:, index].std())
        if isinstance(prediction.get('gap_pair'), str) and prediction['gap_pair'] == ADJACENT_GAPS:
            keys = [gap_key_for_pair(lower, lower + 1) + '_stdev' for lower in range(state_energies.shape[1] - 1)]
            for row in rows:
                spreads = [row[key] for key in keys]
                lower = int(np.argmax(spreads))  # The lowest pair wins a tie.
                row['gap_std'], row['gap_std_lower_state'] = spreads[lower], lower
        elif prediction.get('gap_pair') is not None:
            key = gap_key_for_pair(*prediction['gap_pair'])
            for row in rows:
                if key+'_stdev' not in row:
                    raise ValueError('Gap pair is absent from state predictions')
                row['gap_std'] = row[key+'_stdev']
    return rows
