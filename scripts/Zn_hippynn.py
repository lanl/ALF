import sys

sys.path.append("..")

import ase
import numpy as np
import torch

from copy import deepcopy

from alframework.tools.database import Database as ZarrDatabase
from hippynn.databases import Database


def get_ase_atoms(arr_dict):
    mask = arr_dict["species"] > 0
    atoms = ase.Atoms(arr_dict["species"][mask])
    atoms.cell = arr_dict["cell"]
    atoms.positions = arr_dict["coordinates"][mask]
    atoms.energy = arr_dict["energy"]
    atoms.stress = arr_dict["stress"]
    atoms.atomenergies = arr_dict["atomenergies"][mask]
    atoms.forces = arr_dict["forces"][mask]
    return atoms


def predict_function(chunk, calculator):
    predicted_energies = []
    indices = []
    n_atoms = len(chunk["species"])
    for i in range(n_atoms):
        mol = {k: v[i] for k, v in chunk.items()}
        atoms = get_ase_atoms(mol)
        atoms.calc = calculator
        indices.append(mol["indices"])
        predicted_energies.append([atoms.get_potential_energy() / 27.2114])
    indices = np.stack(indices)
    predicted_energies = np.stack(predicted_energies)
    return {"indices": indices, "energy": predicted_energies}


class Score:
    def __init__(self, bad_percentile, training_percentile, calculator):
        self.bad_percentile = bad_percentile
        self.training_percentile = training_percentile
        self.bad_value = None
        self.training_value = None
        self.calculator = calculator

    @staticmethod
    def _get_diffs(predicted, data):
        pred_positions = np.lexsort((predicted["indices"][:, 1], predicted["indices"][:, 0]))
        data_positions = np.lexsort((data["indices"][:, 1], data["indices"][:, 0]))
        energy_diff = np.linalg.norm(np.abs(predicted["energy"][pred_positions] - data["energy"][data_positions]),
                                     axis=-1)
        return predicted["indices"][pred_positions], energy_diff

    def fit(self, data):
        predicted = predict_function(data, calculator=self.calculator)
        _, energy_diff = Score._get_diffs(predicted, data)
        self.bad_value = np.percentile(energy_diff, self.bad_percentile)
        self.training_value = np.percentile(energy_diff, self.training_percentile)

    def predict(self, chunk):
        if self.bad_value is None or self.training_value is None:
            raise "Fit the score function before prediction"
        predicted = predict_function(chunk, calculator=self.calculator)
        sorted_indices, energy_diff = Score._get_diffs(predicted, chunk)
        bad_indices = sorted_indices[energy_diff > self.bad_value]
        train_indices = sorted_indices[(energy_diff <= self.bad_value) & (energy_diff > self.training_value)]
        return bad_indices, train_indices


def get_hippynn_database(zarr_database, reduction):
    database = Database(reduction, inputs=['species', 'cell', 'coordinates'],
                        targets=['energyperatom', 'species', 'energy', 'atomenergies', 'forces'],
                        seed=101,
                        quiet=False,
                        allow_unfound=True,
                        )
    original_indices = reduction["indices"]
    torch.set_default_dtype(torch.float32)
    for k, v in database.arr_dict.items():
        if v.dtype == np.float64:
            database.arr_dict[k] = v.astype(np.float32)
    database.arr_dict["indices"] = np.arange(len(original_indices))
    database.make_random_split("train", int(0.9 * len(database)))
    database.split_the_rest("valid")
    database.splits["test"] = database.splits["valid"]
    return database


if __name__ == "__main__":
    zarr_database = ZarrDatabase.load_from_zarr("./data/Zn_zarr")
    zarr_database.create_initial_reduction("test_reduction", fraction=0.1, overwrite=True)

    n_iterations = 2

    for iteration in range(n_iterations):
        reduction = zarr_database.dump_reduction("test_reduction")
        database = get_hippynn_database(zarr_database, deepcopy(reduction))
        print(database.splits.keys())
        # train model using the reduction

        # load pre-trained calculator
        calculator = HIPNN_ASE_calculator("./data/HIPNN-Zn-03-1")

        score = Score(95, 75, calculator=calculator)
        score.fit(reduction)

        zarr_database.update_reduction("test_reduction", score, fraction=0.1, chunk_size=100)
    exit(0)
