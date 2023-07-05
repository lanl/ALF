import zarr
import numpy as np

from typing import List, Dict
from collections.abc import Iterable


def concatenate_different_arrays(array_list, stack=False):
    if stack:
        array_list = [a.reshape(1, *a.shape) for a in array_list]
    shapes = np.array([list(a.shape[1:]) for a in array_list])
    min_shape = np.min(shapes, axis=0)
    max_shape = np.max(shapes, axis=0)
    if np.all(min_shape == max_shape):
        return np.concatenate(array_list)
    else:
        number_of_elements = sum([a.shape[0] for a in array_list])
        result = np.zeros((number_of_elements, *max_shape), dtype=array_list[0].dtype)
        pos = 0
        for n, a in enumerate(array_list):
            slices = tuple([slice(pos, pos + a.shape[0])] + [slice(dim) for dim in a.shape[1:]])
            result[slices] = a
            pos += a.shape[0]
        return result


def check_dimensions(template, array, n_atoms):
    shape1 = template.shape
    shape2 = array.shape

    if len(shape1) != len(shape2):
        return False
    n_diff = 0
    for i, j in zip(shape1, shape2):
        if i != j:
            if n_diff == 1:
                return False
            if j != n_atoms:
                return False
    return True


class Database:
    def __init__(self, directory, property_names=None, allow_overwriting=False):
        store = zarr.DirectoryStore(directory)
        self.root = zarr.group(store=store, overwrite=allow_overwriting)
        self.db_size = 0
        if "data" not in self.root:
            self.root.create_group("data")
            self.root.create_group("reductions")
            self.root.create_group("global")
            if property_names is not None:
                self.property_names = property_names
        else:
            if "leaf_structure" in self.root["global"]:
                self.property_names = [i for i in self.root["global/leaf_structure"]]
                self.db_size = len(self.root["global/indices"])

    @classmethod
    def load_from_zarr(cls, directory):
        return cls(directory, allow_overwriting=False)

    def _add_first_element(self, item):
        n_atoms = len(item["species"])
        placeholder = zarr.zeros((1, 2), chunks=(100,), dtype="int")
        placeholder[0, 0] = n_atoms
        self.root["global"].array("indices", placeholder)

        placeholder = zarr.zeros((1,), chunks=(100,), dtype="float")
        placeholder[0] = item.get("global_property", 0)
        self.root["global"].array("global_property", placeholder)
        self.root["global"].create_group("leaf_structure")
        self.root["data"].create_group(n_atoms)

        if not hasattr(self, "property_names"):
            properties = list(item.keys())
            if "global_property" in properties:
                properties.remove("global_property")
            self.property_names = properties
        properties = self.property_names

        for key in properties:
            value = item[key]
            elem_example = zarr.zeros(item[key].shape, chunks=(10000,), dtype=value.dtype)
            self.root["global/leaf_structure"].array(key, elem_example)
            placeholder = zarr.zeros((1, *value.shape), chunks=(100,), dtype=value.dtype)
            placeholder[0] = value
            self.root[f"data/{n_atoms}"].array(key, placeholder)
        self.db_size = 1

    def add_instance(self, item):
        if isinstance(item, list):
            for i in item:
                self.add_instance(i)
        elif isinstance(item, dict):
            if self.db_size == 0:
                return self._add_first_element(item)
            self.root["global/global_property"].append([item.get("global_property", 0)])

            n_atoms = len(item["species"])
            if n_atoms not in self.root["data"]:
                group = self.root["data"].create_group(n_atoms)

                for key in self.property_names:
                    template_array = self.root[f"global/leaf_structure/{key}"][:]
                    value = item[key]
                    assert check_dimensions(template_array, value, n_atoms)
                    placeholder = zarr.zeros((1, *value.shape), chunks=(100,), dtype=value.dtype)
                    placeholder[0] = value
                    group.array(key, placeholder)
            else:
                for key in self.property_names:
                    value = item[key]
                    self.root[f"data/{n_atoms}/{key}"].append(value.reshape(1, *value.shape))
            loc_index = len(self.root[f"data/{n_atoms}/{self.property_names[0]}"]) - 1
            self.db_size += 1
            self.root["global/indices"].append([[n_atoms, loc_index]])

        else:
            raise NotImplementedError

    def add_property(self, name, item):
        pass

    def get_item(self, selection_index, pad_arrays=False):
        if isinstance(selection_index[0], Iterable):
            results = {}
            for i, idx in enumerate(selection_index):
                one_elem = self.get_item(idx, pad_arrays=False)
                if i == 0:
                    for key in one_elem:
                        results[key] = [one_elem[key]]
                else:
                    for key in one_elem:
                        results[key].append(one_elem[key])
            if pad_arrays:
                results = {k: concatenate_different_arrays(v, stack=True) for (k, v) in results.items()}
            return results

        elif isinstance(selection_index[0], int):
            # todo: check if index exists
            result = {}

            for key in self.root[f"data/{selection_index[0]}"]:
                result[key] = self.root[f"data/{selection_index[0]}/{key}"][selection_index[1]]
            return result
        else:
            raise NotImplementedError

    def _create_reduction(self, name, stage, reduction_indices):
        self.root[f"reductions/{name}"].create_group(stage)
        for group in self.root["data/"]:
            self.root[f"reductions/{name}/{stage}/"].create_group(group)
            pointer = self.root[f"reductions/{name}/{stage}/{group}"]
            pointer.array("indices", self.root[f"data/{group}/indices"])
            indices = pointer["indices"][:]
            _, positions, _ = np.intersect1d(indices, reduction_indices, return_indices=True)
            placeholder = zarr.zeros(positions.shape, chunks=(100,), dtype="int")
            placeholder[:] = positions
            pointer.array("positions", placeholder)

    def create_initial_reduction(self, name, fraction, overwrite=False):
        # specific variable for test set, preserve points from training on
        database_size = self.root["database_size"][0]
        subset_size = int(fraction * database_size)
        initial_subset_indices = np.random.choice(np.arange(database_size), subset_size)
        self.root["reductions"].create_group(name, overwrite=overwrite)
        self._create_reduction(name, "000", initial_subset_indices)

    def update_reduction(self, name, predicted_data: Dict, property_name, fraction, outlier_percentile):
        # suppose to have dictionary with indices and one of properties (e.g. forces, energies).

        last_reduction = self.get_last_reduction(name)
        current_stage = str(int(last_reduction.basename) + 1).zfill(3)

        # get all differences
        diffs = []
        for group in self.root["data"]:
            pointer = self.root[f"data/{group}"]
            group_indices = pointer["indices"][:]
            indices, group_positions, data_positions = np.intersect1d(group_indices, predicted_data["indices"],
                                                                      return_indices=True)
            database_values = pointer[property_name][group_positions]
            external_values = predicted_data[property_name][data_positions]
            diff = np.stack([indices, np.linalg.norm(database_values - external_values, axis=0)], axis=-1)
            diffs.append(diff)
        diffs = np.concatenate(diffs)

        # todo: there should be a selection logic here
        selected_indices = diffs[:, 0]

        self._create_reduction(name, current_stage, selected_indices)

    def get_last_reduction(self, name):
        g = self.root[f"reductions/{name}"]
        max_key = max(g.group_keys())
        return g[max_key]

    def dump_reduction(self, name, output_format):
        elements = {}
        last_reduction = self.get_last_reduction(name)

        for group in self.root["data/"]:
            positions = last_reduction[f"{group}/positions"][:]
            data_group = self.root[f"data/{group}"]
            for key in data_group:
                values = data_group[key][positions]
                if key not in elements:
                    elements[key] = []
                elements[key].append(values)
        elements = {k: concatenate_different_arrays(v) for (k, v) in elements.items()}
        # todo: inject to databse
        return elements


if __name__ == "__main__":
    # import os
    # full_file = os.path.join(d_dir, file)
    # x = pyanitools.anidataloader(full_file)
    # for c in x:
    #     batches.append(c)
    from pprint import pprint

    item = {
        "species": np.array([1, 8, 1]).astype(int),
        "energy": np.array([1]),
        "forces": np.random.random((3, 3))

    }

    another_item = {
        "species": np.array([1, 8, 1, 4, 5]).astype(int),
        "energy": np.array([5]),
        "forces": np.random.random((5, 3))

    }

    database = Database("./example_data", allow_overwriting=True)
    database.add_instance(item)
    database = Database.load_from_zarr("./example_data")
    print(database.property_names, database.db_size)

    database.add_instance(item)
    database.add_instance(another_item)

    for i in range(1000):
        database.add_instance(item)
        database.add_instance(another_item)

    pprint(database.root["global/indices"][:10])

    pprint(database.get_item([(3, 0), (5, 1)]))
    pprint(database.get_item([(3, 0), (5, 1)], pad_arrays=True))
    # pprint(database.get_item(1001))
    #
    # database.create_initial_reduction("lol", 0.5)
    #
    # dump = database.dump_reduction("lol", "json")
    #
    # pprint(dump)
    #
    # size = database.root["database_size"][0]
    #
    # indices = np.random.choice(np.arange(size), 500, replace=False)
    # energies = np.random.normal(loc=2, size=500)
    #
    # predicted_data = {"indices": indices, "energy": energies}
    #
    # database.update_reduction("lol", predicted_data, "energy", 0.1, 95)
