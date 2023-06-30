import zarr
import numpy as np

from typing import List, Dict


class Database:
    def __init__(self, data: List[Dict], directory, allow_overwriting=False):
        store = zarr.DirectoryStore(directory)
        self.root = zarr.group(store=store, overwrite=allow_overwriting)
        if "data" not in self.root:
            self.root.create_group("data")
            self.root["data"].array("database_size", np.array([0]))
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
                placeholder = zarr.zeros((1,), chunks=(100,))
                placeholder[0] = data["database_size"][0]
                group.array("indices", placeholder)
                for key, value in item.items():
                    placeholder = zarr.zeros((1, *item[key].shape), chunks=(100,))
                    placeholder[0] = value
                    group.array(key, placeholder)
            else:
                group = data[n_atoms]
                group["indices"].append(data["database_size"])
                for key, value in item.items():
                    group[key].append(value.reshape(1, *value.shape))
            data["database_size"][0] += 1

        else:
            raise NotImplementedError

    def get_item(self, selection_index):
        if isinstance(selection_index, list):
            selection_index = set(selection_index)
            results = []

            data = self.root["data"]
            for key in data:
                if key != "database_size":
                    group_index = data[key]["indices"][:]
                    group_index_set = set(group_index)
                    index = group_index_set & selection_index
                    for idx in index:
                        position = np.where(group_index == idx)[0]
                        data_dict = {}
                        group = data[key]
                        for inner_key in group:
                            data_dict[inner_key] = group[inner_key][position]
                    results.append(data_dict)
            return results

        elif isinstance(selection_index, int):
            return self.get_item([selection_index])[0]
        else:
            raise NotImplementedError

    def create_reduction(self, name, fraction):
        pass

    def update_reduction(self, name):
        pass

    def dump_reductin(self, name, format):
        pass


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
    pprint(database.root["data"]["database_size"][0])

    pprint(database.get_item([0, 10, 1001])[0])
    pprint(database.get_item(1001))
