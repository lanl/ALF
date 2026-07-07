import multiprocessing
import os
import re

import numpy as np
from parsl import python_app


def _source_model_dir(ML_config, model_path, current_model_id, ensemble_index):
    source_pattern = ML_config.get("fine_tune_source_model_path")
    if source_pattern is not None:
        return source_pattern.format(ensemble_index)

    source_model_id = ML_config.get("fine_tune_source_model_id", current_model_id)
    if source_model_id < 0:
        raise RuntimeError(
            "No source model is available for fine-tuning. Place a seed ensemble "
            "under models/model-0000 and start without status.txt."
        )
    return os.path.join(model_path.format(source_model_id), "model-{:02d}".format(ensemble_index))


def _prepare_database(
    h5_train_dir,
    db_info,
    energy_key,
    coordinates_key,
    species_key,
    force_key=None,
    cell_key=None,
    quadrupole_key=None,
    valid_size=0.1,
    test_size=0.1,
    remove_high_energy_cut=None,
    remove_high_energy_std=None,
    remove_high_forces_cut=None,
    remove_high_forces_std=None,
):
    import torch
    from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
    from hippynn.databases.database import prettyprint_arrays

    database = PyAniDirectoryDB(
        directory=h5_train_dir,
        allow_unfound=True,
        quiet=False,
        seed=np.random.randint(1e9),
        inputs=None,
        targets=None,
    )
    arrays = database.arr_dict

    arrays[species_key] = arrays[species_key].to(torch.int64)
    n_atoms = arrays[species_key].to(torch.bool).to(torch.int32).sum(dim=1)

    per_atom_key = energy_key + "peratom"
    if per_atom_key not in arrays and energy_key in arrays:
        arrays[per_atom_key] = arrays[energy_key] / n_atoms

    if quadrupole_key is not None and quadrupole_key in arrays:
        arrays[quadrupole_key] = arrays[quadrupole_key].reshape(-1, 9)

    for key, value in arrays.items():
        if value.dtype == torch.float64:
            arrays[key] = value.to(torch.float32)

    database.inputs = db_info["inputs"]
    database.targets = db_info["targets"]

    required_keys = set(database.inputs) | set(database.targets) | {"indices"}
    for key in list(arrays):
        if key not in required_keys:
            del arrays[key]

    if force_key is not None and force_key in arrays:
        database.remove_high_property(
            force_key,
            True,
            species_key=species_key,
            cut=remove_high_forces_cut,
            std_factor=remove_high_forces_std,
        )
    if energy_key in arrays:
        database.remove_high_property(
            energy_key,
            False,
            species_key=species_key,
            cut=remove_high_energy_cut,
            std_factor=remove_high_energy_std,
        )

    print("Array Shapes After Cleaning")
    prettyprint_arrays(database.arr_dict)

    database.make_random_split("valid", valid_size)
    database.make_random_split("test", test_size)
    database.split_the_rest("train")

    for split in ["train", "valid", "test"]:
        database.splits[split][energy_key] = database.splits[split][energy_key].to(torch.float32)
        if per_atom_key in database.splits[split]:
            database.splits[split][per_atom_key] = database.splits[split][per_atom_key].to(torch.float32)
        database.splits[split][coordinates_key] = database.splits[split][coordinates_key].to(torch.float32)
        if force_key is not None and force_key in database.splits[split]:
            database.splits[split][force_key] = database.splits[split][force_key].to(torch.float32)
        if cell_key is not None and cell_key in database.splits[split]:
            database.splits[split][cell_key] = database.splits[split][cell_key].to(torch.float32)

    return database


def fine_tune_HIPNN_model(
    model_dir,
    source_model_dir,
    h5_train_dir,
    energy_key,
    coordinates_key,
    species_key,
    cell_key=None,
    force_key=None,
    quadrupole_key=None,
    valid_size=0.1,
    test_size=0.1,
    learning_rate=1e-4,
    scheduler_options=None,
    controller_options=None,
    device_string="0",
    from_multiprocessing_nGPU=None,
    fine_tune_structure_file="experiment_structure.pt",
    fine_tune_checkpoint_file="best_checkpoint.pt",
    fine_tune_stopping_key=None,
    remove_high_energy_cut=None,
    remove_high_energy_std=None,
    remove_high_forces_cut=None,
    remove_high_forces_std=None,
    **unused_config,
):
    import torch
    from hippynn.experiment import SetupParams, setup_and_train
    from hippynn.experiment.controllers import PatienceController, RaiseBatchSizeOnPlateau
    from hippynn.experiment.serialization import load_checkpoint
    from hippynn.tools import active_directory, log_terminal

    scheduler_options = {} if scheduler_options is None else scheduler_options
    controller_options = {} if controller_options is None else controller_options
    model_dir = os.path.abspath(model_dir)
    source_model_dir = os.path.abspath(source_model_dir)
    h5_train_dir = os.path.abspath(h5_train_dir)

    if device_string.lower() == "from_multiprocessing":
        process = multiprocessing.current_process()
        os.environ["CUDA_VISIBLE_DEVICES"] = str((process._identity[-1] - 1) % from_multiprocessing_nGPU)
        train_device = "cuda:0"
    elif device_string.lower() == "cpu":
        train_device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_string
        train_device = "cuda:0"

    if train_device != "cpu":
        torch.cuda.set_device(train_device)

    os.makedirs(model_dir, exist_ok=True)
    source_structure = os.path.join(source_model_dir, fine_tune_structure_file)
    source_checkpoint = os.path.join(source_model_dir, fine_tune_checkpoint_file)
    if not os.path.exists(source_structure):
        raise FileNotFoundError(source_structure)
    if not os.path.exists(source_checkpoint):
        raise FileNotFoundError(source_checkpoint)

    with active_directory(model_dir):
        with log_terminal("training_log.txt", "wt"):
            print("Fine-tuning from HIPPYNN checkpoint:")
            print(source_model_dir)
            print("CUDA_VISIBLE_DEVICES: " + os.environ.get("CUDA_VISIBLE_DEVICES", ""))

            bundle = load_checkpoint(
                source_structure,
                source_checkpoint,
                restart_db=False,
                map_location="cpu",
            )
            training_modules = bundle["training_modules"]
            model, loss_module, model_evaluator = training_modules
            db_info = model_evaluator.db_info

            database = _prepare_database(
                h5_train_dir,
                db_info,
                energy_key,
                coordinates_key,
                species_key,
                force_key=force_key,
                cell_key=cell_key,
                quadrupole_key=quadrupole_key,
                valid_size=valid_size,
                test_size=test_size,
                remove_high_energy_cut=remove_high_energy_cut,
                remove_high_energy_std=remove_high_energy_std,
                remove_high_forces_cut=remove_high_forces_cut,
                remove_high_forces_std=remove_high_forces_std,
            )

            optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=learning_rate)
            scheduler = RaiseBatchSizeOnPlateau(optimizer=optimizer, **scheduler_options)
            controller = PatienceController(
                optimizer=optimizer,
                scheduler=scheduler,
                stopping_key=fine_tune_stopping_key or "Loss-Error",
                **controller_options,
            )
            experiment_params = SetupParams(controller=controller, device=train_device)

            print("Fine-tuning with fresh Adam optimizer.")
            print("Learning rate:", learning_rate)
            print("Controller Options:")
            print(controller_options, "\n\n")

            setup_and_train(
                training_modules=training_modules,
                database=database,
                setup_params=experiment_params,
            )


def fine_tune_HIPNN_model_wrapper(arg_dict):
    return fine_tune_HIPNN_model(**arg_dict)


@python_app(executors=["alf_ML_executor"])
def fine_tune_HIPPYNN_ensemble_task(
    ML_config,
    h5_dir,
    model_path,
    current_training_id,
    gpus_per_node,
    current_model_id=-1,
    remove_existing=False,
    h5_test_dir=None,
):
    p = multiprocessing.Pool(gpus_per_node)
    general_configuration = ML_config.copy()
    n_models = general_configuration.pop("n_models")
    params_list = [general_configuration.copy() for i in range(n_models)]

    for i, cur_dict in enumerate(params_list):
        cur_dict["model_dir"] = os.path.join(model_path.format(current_training_id), "model-{:02d}".format(i))
        cur_dict["source_model_dir"] = _source_model_dir(ML_config, model_path, current_model_id, i)
        cur_dict["h5_train_dir"] = h5_dir
        cur_dict["from_multiprocessing_nGPU"] = gpus_per_node

    p.map(fine_tune_HIPNN_model_wrapper, params_list)
    p.close()

    completed = []
    HIPNN_complete = re.compile("Training complete")
    for i in range(n_models):
        log_path = os.path.join(model_path.format(current_training_id), "model-{:02d}".format(i), "training_log.txt")
        with open(log_path, "r") as log_file:
            log = log_file.read()
        completed.append(len(HIPNN_complete.findall(log)) == 1)

    return completed, current_training_id
