import numpy
import zarr
import numpy as np

from typing import List, Dict


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


def exclude_values(array, values=None):
    if values is None or len(values) == 0:
        return np.arange(len(array))
    else:
        result_mask = array != values[0]
        for value in values[1:]:
            result_mask &= (array != value)
        return np.where(result_mask)[0]


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
        group_name = f"{n_atoms:03}"
        placeholder = zarr.zeros((1, 2), chunks=(100,), dtype="int")
        placeholder[0, 0] = n_atoms
        self.root["global"].array("indices", placeholder)

        placeholder = zarr.zeros((1,), chunks=(100,), dtype="float")
        placeholder[0] = item.get("global_property", 0)
        self.root["global"].array("global_property", placeholder)
        self.root["global"].create_group("leaf_structure")
        self.root["data"].create_group(group_name)

        if not hasattr(self, "property_names"):
            properties = list(item.keys())
            if "global_property" in properties:
                properties.remove("global_property")
            self.property_names = properties
        properties = self.property_names

        for key in properties:
            value = item[key]
            elem_example = zarr.zeros(value.shape, chunks=(10000,), dtype=value.dtype)
            self.root["global/leaf_structure"].array(key, elem_example)
            placeholder = zarr.zeros((1, *value.shape), chunks=(100,), dtype=value.dtype)
            placeholder[0] = value
            self.root[f"data/{group_name}"].array(key, placeholder)
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
            group_name = f"{n_atoms:03}"
            if group_name not in self.root["data"]:
                group = self.root["data"].create_group(group_name)

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
                    self.root[f"data/{group_name}/{key}"].append(value.reshape(1, *value.shape))
            loc_index = len(self.root[f"data/{group_name}/{self.property_names[0]}"]) - 1
            self.db_size += 1
            self.root["global/indices"].append([[len(item["species"]), loc_index]])

        else:
            raise NotImplementedError

    def add_instance_and_property(self, name, example, group_dim=None):
        elem_example = zarr.zeros(example.shape, chunks=(10000,), dtype=example.dtype)
        self.root["global/leaf_structure"].array(name, elem_example)
        self.property_names.append(name)
        for key in self.root["data"]:
            shape = list(example.shape)
            if group_dim is not None:
                shape[group_dim] = int(key)

            filler = zarr.zeros((len(self.root[f"data/{key}/species"]), *shape),
                                chunks=(100,), dtype=example.dtype)
            self.root[f"data/{key}"].array(name, filler)

    def get_item(self, selection_index, pad_arrays=False):
        if isinstance(selection_index[0], numpy.ndarray) or isinstance(selection_index[0], list):
            selection_index = np.array(selection_index)
            results = {}
            groups = np.unique(selection_index[:, 0])
            for n_atom in groups:
                indices = selection_index[selection_index[:, 0] == n_atom]
                if len(results) == 0:
                    for key in self.root[f"data/{n_atom:03}"]:
                        results[key] = [self.root[f"data/{n_atom:03}/{key}"][indices[:, 1]]]
                    results["indices"] = [indices]
                else:
                    for key in self.root[f"data/{n_atom:03}"]:
                        results[key].append(self.root[f"data/{n_atom:03}/{key}"][indices[:, 1]])
                    results["indices"].append(indices)

            if pad_arrays:
                results = {k: concatenate_different_arrays(v) for (k, v) in results.items()}
            return results

        elif isinstance(selection_index[0], int):
            result = {}

            for key in self.root[f"data/{selection_index[0]}"]:
                result[key] = self.root[f"data/{selection_index[0]}/{key}"][selection_index[1]]
            return result
        else:
            raise NotImplementedError

    def create_initial_reduction(self, name, fraction, overwrite=False, exclude_global=(1,)):
        global_property = self.root["global/global_property"][:]
        valid_positions = exclude_values(global_property, exclude_global)
        selection_size = int(len(valid_positions) * fraction)
        selected_positions = np.random.choice(valid_positions, selection_size, replace=False)

        selected_indices = self.root["global/indices"][selected_positions]
        # 2 flags that the data point in a reduction
        self.root["global/global_property"][selected_indices] = 2
        self.root["reductions"].create_group(name, overwrite=overwrite)
        self.root[f"reductions/{name}"].array("000", selected_indices)

    # def update_reduction(self, name, predicted_data: Dict, property_name, fraction, outlier_percentile):
    #     # suppose to have dictionary with indices and one of properties (e.g. forces, energies).
    #
    #     last_reduction = self.get_last_reduction(name)
    #     current_stage = str(int(last_reduction.basename) + 1).zfill(3)
    #
    #     # get all differences
    #     diffs = []
    #     for group in self.root["data"]:
    #         pointer = self.root[f"data/{group}"]
    #         group_indices = pointer["indices"][:]
    #         indices, group_positions, data_positions = np.intersect1d(group_indices, predicted_data["indices"],
    #                                                                   return_indices=True)
    #         database_values = pointer[property_name][group_positions]
    #         external_values = predicted_data[property_name][data_positions]
    #         diff = np.stack([indices, np.linalg.norm(database_values - external_values, axis=0)], axis=-1)
    #         diffs.append(diff)
    #     diffs = np.concatenate(diffs)
    #
    #     # todo: there should be a selection logic here
    #     selected_indices = diffs[:, 0]
    #
    #     self._create_reduction(name, current_stage, selected_indices)

    def chunk_generator(self, chunk_size, exclude_global=(1., 2.)):
        global_property = self.root["global/global_property"][:]
        if exclude_global is not None:
            result_mask = global_property != exclude_global[0]
            for value in exclude_global[1:]:
                result_mask &= (global_property != value)
            valid_positions = np.where(result_mask)[0]
        else:
            valid_positions = np.arange(len(global_property))

        valid_indices = self.root["global/indices"][valid_positions]
        order = np.argsort(valid_indices[:, 0])
        sorted_indices = valid_indices[order]
        n_chunks = (len(sorted_indices)) // chunk_size + int(len(sorted_indices) % chunk_size != 0)
        borders = np.arange(1, n_chunks) * chunk_size
        chunk_indices = np.split(sorted_indices, borders)

        for chunk_idx in chunk_indices:
            yield self.get_item(chunk_idx, pad_arrays=True)

    def get_last_reduction(self, name):
        g = self.root[f"reductions/{name}"]
        max_key = max(g.array_keys())
        return g[max_key]

    def dump_reduction(self, name, output_format, stage=None):
        if stage is None:
            reduction = self.get_last_reduction(name)
        else:
            if isinstance(stage, int):
                stage = str(stage).zfill(3)
            reduction = self.root[f"reductions/{name}/{stage}"]

        data = self.get_item(reduction[:], pad_arrays=True)
        # todo: inject to databse
        return data


if __name__ == "__main__":
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

    database = Database("./example_data", allow_overwriting=True, property_names=("species", "forces"))
    database.add_instance(item)
    database = Database.load_from_zarr("./example_data")
    print(database.property_names)

    database.add_instance(item)
    database.add_instance(another_item)

    for i in range(3):
        database.add_instance(item)
        database.add_instance(another_item)

    pprint(database.root["global/indices"][:10])

    pprint(database.get_item([[3, 0], [5, 1]]))
    pprint(database.get_item([[5, 1], [3, 0]], pad_arrays=True))

    database.create_initial_reduction("lol", 0.5)

    dump = database.dump_reduction("lol", "json")

    for chunk in database.chunk_generator(2, exclude_global=None):
        pprint(chunk)

    database.add_instance_and_property("special_forces", item["forces"], group_dim=0)

    other_item = {
        "species": np.array([1, 8, 1, 4, 5]).astype(int),
        "energy": np.array([5]),
        "forces": np.random.random((5, 3)),
        "special_forces": np.random.random((5, 3))
    }

    database.add_instance(other_item)

    for chunk in database.chunk_generator(2, exclude_global=None):
        pprint(chunk)



    # size = database.root["database_size"][0]
    #
    # indices = np.random.choice(np.arange(size), 500, replace=False)
    # energies = np.random.normal(loc=2, size=500)
    #
    # predicted_data = {"indices": indices, "energy": energies}
    #
    # database.update_reduction("lol", predicted_data, "energy", 0.1, 95)
