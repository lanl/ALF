import zarr
import numpy as np

from typing import List, Dict


def concatenate_different_arrays(array_list):
    shapes = np.array([list(a.shape) for a in array_list])
    min_shape = np.min(shapes, axis=0)
    max_shape = np.max(shapes, axis=0)
    if np.all(min_shape == max_shape):
        return np.stack(array_list)
    else:
        result = np.zeros((len(array_list), *max_shape))
        for n, a in enumerate(array_list):
            slices = tuple([n] + [slice(dim) for dim in a.shape])
            result[slices] = a
        return result


class Database:
    def __init__(self, data: List[Dict], directory, allow_overwriting=False):
        store = zarr.DirectoryStore(directory)
        self.root = zarr.group(store=store, overwrite=allow_overwriting)
        if "data" not in self.root:
            self.root.create_group("data")
            self.root.array("database_size", np.array([0], dtype="int"))
            self.root.create_group("reductions")
        self.add_instance(data)

    @classmethod
    def load_from_zarr(cls, directory):
        return cls([], directory, allow_overwriting=False)

    def add_instance(self, item):
        if isinstance(item, list):
            for i in item:
                self.add_instance(i)
        elif isinstance(item, dict):
            n_atoms = len(item["species"])
            data = self.root["data"]
            if n_atoms not in data:
                group = data.create_group(n_atoms)
                placeholder = zarr.zeros((1,), chunks=(100,), dtype="int")
                placeholder[0] = self.root["database_size"][0]
                group.array("indices", placeholder)
                for key, value in item.items():
                    placeholder = zarr.zeros((1, *item[key].shape), chunks=(100,))
                    placeholder[0] = value
                    group.array(key, placeholder)
            else:
                group = data[n_atoms]
                group["indices"].append(self.root["database_size"])
                for key, value in item.items():
                    group[key].append(value.reshape(1, *value.shape))
            self.root["database_size"][0] += 1

        else:
            raise NotImplementedError

    def get_item(self, selection_index):
        if isinstance(selection_index, list):
            selection_index = selection_index
            results = []

            data = self.root["data"]
            for key in data:
                group_index = data[key]["indices"][:]
                _, positions, _ = np.intersect1d(group_index, selection_index, return_indices=True)
                for pos in positions:
                    data_dict = {}
                    group = data[key]
                    for inner_key in group:
                        data_dict[inner_key] = group[inner_key][pos]
                results.append(data_dict)
            return results

        elif isinstance(selection_index, int):
            return self.get_item([selection_index])[0]
        else:
            raise NotImplementedError

    def create_reduction(self, name, fraction, overwrite=False):
        database_size = self.root["database_size"][0]
        subset_size = int(fraction * database_size)
        initial_subset_indices = np.random.choice(np.arange(database_size), subset_size)
        self.root["reductions"].create_group(name, overwrite=overwrite)
        self.root[f"reductions/{name}"].create_group("000")
        for group in self.root["data/"]:
            self.root[f"reductions/{name}/000/"].create_group(group)
            pointer = self.root[f"reductions/{name}/000/{group}"]
            pointer.array("indices", self.root[f"data/{group}/indices"])
            indices = pointer["indices"][:]
            _, positions, _ = np.intersect1d(indices, initial_subset_indices, return_indices=True)
            placeholder = zarr.zeros(positions.shape, chunks=(100,), dtype="int")
            placeholder[:] = positions
            pointer.array("positions", placeholder)

    def update_reduction(self, name):
        pass

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
        return elements


if __name__ == "__main__":
    from pprint import pprint

    item = {
        "species": np.array([1, 8, 1]),
        "energy": np.array([1]),
        "forces": np.random.random((3, 3))

    }

    another_item = {
        "species": np.array([1, 8, 1, 4, 5]),
        "energy": np.array([5]),
        "forces": np.random.random((5, 3))

    }

    database = Database([item], "./example_data", allow_overwriting=True)

    for i in range(1000):
        database.add_instance(item)
        database.add_instance(another_item)

    database = Database.load_from_zarr("./example_data")
    pprint(database.root["database_size"][0])

    pprint(database.get_item([0, 10, 1001])[0])
    pprint(database.get_item(1001))

    print(np.intersect1d([1, 3, 4, 3], np.array([3, 1, 2, 1]), return_indices=True))

    database.create_reduction("lol", 0.5)

    database.dump_reduction("lol", "json")
