"""Replay two independent, illustrative uncertainty signals through ALF."""

import json

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
from alframework.samplers.acquisition import acquisition_settings, assess_frame
from alframework.tools.geometry import minimum_distance


class TwoSpringCalculator(Calculator):
    """A central atom held by two springs, each with its own stretch score."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # Atoms 0 and 2 are bonded to atom 1. The rest length is 1 Å.
        bonds = self.atoms.positions[[0, 2]] - self.atoms.positions[1]
        lengths = np.linalg.norm(bonds, axis=1)
        stretch = lengths - 1.0
        outer_forces = -bonds * (stretch / lengths)[:, None]
        forces = np.stack((outer_forces[0], -outer_forces.sum(0), outer_forces[1]))
        left, right = np.maximum(stretch, 0.0)
        self.results = {
            "energy": float(0.5 * (stretch @ stretch)),
            "forces": forces,
            "alf_uncertainty": {"left_stretch": float(left), "right_stretch": float(right)},
            "alf_atomic_uncertainty": {
                "left_stretch": np.array([left, left, 0.0]),
                "right_stretch": np.array([0.0, right, right]),
            },
        }


def main():
    structures = {
        name: Atoms("HHeH", positions=[[-left, 0, 0], [0, 0, 0], [right, 0, 0]])
        for name, left, right in (("equilibrium", 1.0, 1.0), ("left_stretched", 1.5, 1.0),
                                  ("right_stretched", 1.0, 1.5))
    }
    config = {
        "distcut": 0.3,
        "acquisition": {
            "terms": {
                "left": {"uncertainty": "left_stretch", "cutoff": 0.25},
                "right": {"uncertainty": "right_stretch", "cutoff": 0.25},
            },
            "eligibility": ["left", "right"],
            "strategy": "largest",
        },
    }
    settings = acquisition_settings(config)
    prediction = ASEEnsembleEvaluator([TwoSpringCalculator()]).predict(list(structures.values()))
    report = []
    for (name, atoms), signals, arrays in zip(structures.items(), ensemble_statistics(prediction),
                                              prediction["atomic_uncertainty"]):
        decision = assess_frame(signals, minimum_distance(atoms), 0.0, config, settings)
        report.append({
            "structure": name,
            "scores": {score: signals[score] for score in arrays},
            "atomic_uncertainty": {score: values.tolist() for score, values in arrays.items()},
            "trigger_flags": decision["acquisition_trigger_flags"],
            "decision": decision["outcome"],
        })
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
