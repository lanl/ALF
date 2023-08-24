import glob
import multiprocessing
import os
import re

import numpy as np
import torch
from ase import units
from ase.data import chemical_symbols
from hippynn.databases.database import Database
from hippynn.databases.h5_pyanitools import PyAniMethods
from hippynn.databases.restarter import Restartable
from parsl import python_app

from alframework.tools.database import Database as ZarrDatabase


class PyAniDB(Database, PyAniMethods, Restartable):
    def __init__(self, zarr_path, reduction_name, inputs, targets, *args, files=None, allow_unfound=False,
                 species_key="species",
                 quiet=False, **kwargs):
        self.files = files
        self.inputs = inputs
        self.targets = targets
        self.species_key = species_key
        zarr_database = ZarrDatabase.load_from_zarr(zarr_path)
        reduction = zarr_database.dump_reduction(reduction_name)

        super().__init__(reduction, inputs, targets, *args, **kwargs, quiet=quiet, allow_unfound=allow_unfound)
        self.restarter = self.make_restarter(zarr_path, reduction_name, inputs, targets, *args, files=files,
                                             quiet=quiet,
                                             species_key=species_key, **kwargs)
        original_indices = reduction["indices"]
        self.arr_dict["indices"] = np.arange(len(original_indices))

def get_hippynn_network(args, hipnn_order, network_params, periodic):
    from hippynn.graphs import networks
    if hipnn_order.lower() == 'scalar':
        network = networks.Hipnn("HIPNN", args, module_kwargs=network_params,
                                 periodic=periodic)
    elif hipnn_order.lower() == 'vector':
        network = networks.HipnnVec("HIPNN", args, module_kwargs=network_params,
                                    periodic=periodic)
    elif hipnn_order.lower() == 'quadratic':
        network = networks.HipnnQuad("HIPNN", args, module_kwargs=network_params,
                                     periodic=periodic)
    else:
        raise RuntimeError('Unrecognized hipnn_order parameter')
    return network


def train_HIPNN_model(model_dir,
                      zarr_path,
                      reduction_name,
                      energy_key,
                      coordinates_key,
                      species_key,
                      force_key=None,
                      cell_key=None,
                      electrostatics_flag=False,
                      charges_key=None,
                      dipole_key=None,
                      quadrupole_key=None,
                      electrostatics_weight=0,
                      hipnn_order='scalar',
                      network_params=None,
                      aca_params=None,
                      first_is_interacting=False,
                      pairwise_inference_parameters=None,
                      energy_mae_loss_weight=1e2,
                      energy_rmse_loss_weight=1e2,
                      total_energy_mae_loss_weight=0,
                      total_energy_rmse_loss_weight=0,
                      force_mae_loss_weight=1.0,
                      force_rmse_loss_weight=1.0,
                      charge_mae_loss_weight=1.0,
                      charge_rmse_loss_weight=1.0,
                      dipole_mae_loss_weight=1.0,
                      dipole_rmse_loss_weight=1.0,
                      quadrupole_mae_loss_weight=1.0,
                      quadrupole_rmse_loss_weight=1.0,
                      rbar_loss_weight=1.0,
                      l2_reg_loss_weight=1e-6,
                      learning_rate=5e-4,
                      valid_size=0.1,
                      test_size=0.1,
                      remove_high_energy_cut=None,
                      remove_high_energy_std=None,
                      remove_high_forces_cut=None,
                      remove_high_forces_std=None,
                      plot_every=50,
                      zarr_test_path=None,
                      scheduler_options={"max_batch_size": 128, "patience": 25, "factor": 0.5},
                      controller_options={"batch_size": 64, "eval_batch_size": 128, "max_epochs": 1000,
                                          "termination_patience": 55},
                      device_string='0',
                      from_multiprocessing_nGPU=None,
                      build_lammps_pickle=False
                      ):
    """

    Args:
        model_dir (str): Directory where HIPPYNN model will reside
        h5_train_dir (str): Path of the directory containing the training data in h5 format.
        energy_key (str): Energy key inside inside h5 dataset. Must match db name in 'properties_list'.
        coordinates_key (str): Coordinates key inside h5 dataset. Must match db name in 'properties_list'.
        species_key (str): Species key inside h5 dataset. Must match db name in 'properties_list'.
        force_key (str): Force key insdie h5 dataset. Must match db name in 'properties_list'. None if one don't want
                         to train with forces.
        cell_key (str): Cell key inside h5 dataset. Must match db name in 'properties_list'. Do not define if data is
                        not periodic.
        electrostatics_flag (bool): Flag to check if the electrostatic will be trained with the energy and forces.
                                    If True, at least one of "charges_key", "dipole_key", or "quadrupole_key" must
                                    also be True.
        charges_key (str): Per-atom charges key inside th h5 data. Must match db name in 'properties_list'. If None,
                           partial charges are not trained to.
        dipole_key (str): Molecular dipole key inside th h5 data. Must match db name in 'properties_list'. If None,
                          quadrupoles are not trained to.
        quadrupole_key (str): Molecular-quadrupole key inside th h5 data. Must match db name in 'properties_list'. If
                              None, quadrupoles are not trained to.
        electrostatics_weight (?):
        hipnn_order (str): Can be 'scalar', 'vector', or 'quadratic'
        network_params (dict): Dictonary that defines the model. See examples for details.
        aca_params (dict): Dictionary that defines parameters for charge training.
        first_is_interacting (bool):
        pairwise_inference_parameters (dict): Dictionary that defines any pairwise interference.
        energy_mae_loss_weight (float): MAE per atom energy loss weight.
        energy_rmse_loss_weight (float): RMSE per atomenergy loss weight.
        total_energy_mae_loss_weight (float): MAE total energy loss weight. Typicall only one of energy_mae_loss_weight
                                              and total_energy_mae_loss_eight are used (the other is set to 0).
        total_energy_rmse_loss_weight (float): MAE total energy loss weight. Typicall only one of energy_rmse_loss_weight
                                               and total_energy_rmse_loss_eight are used (the other is set to 0).
        force_mae_loss_weight (float): MAE force loss weight.
        force_rmse_loss_weight (float): RMSE force loss weight.
        charge_mae_loss_weight (float): MAE charge loss weight.
        charge_rmse_loss_weight (float): RMSE charge loss weight.
        dipole_mae_loss_weight (float): MAE dipole loss weight.
        dipole_rmse_loss_weight (float): RMSE dipole loss weight.
        quadrupole_mae_loss_weight (float): MAE quadrupole loss weight.
        quadrupole_rmse_loss_weight (float): RMSE quadrupole loss weight.
        rbar_loss_weight (float): Weight of rbar in loss.
        l2_reg_loss_weight (float): Weight of l2_reg in loss, typically 1e-6.
        learning_rate (float): Learning rate of the optimization algorithm. 
        valid_size (float): Percentage of the dataset from [0,1] that will be used for validation.
        test_size (float): Percentage of the dataset from [0,1] that will be used for test.
        remove_high_energy_cut (float): If none, all data is kept. If float, data with energies per atom that many
                                        standard deviations or larger are removed.
        remove_high_energy_std (float): If none, all data is kept. If float, data with energies per atom that many
                                        standard deviations or larger are removed.
        remove_high_forces_cut (float): If none, all data is kept. If float, data with forces that many standard
                                        deviations or larger are removed.
        remove_high_forces_std (float): If none, all data is kept. If float, data with forces that many standard
                                        deviations or larger are removed.
        plot_every (int): Every 'plot_every' epochs, generate plots.
        h5_test_dir (str): Path to a specific h5 test directory.
        scheduler_options (dict): Dictionary that defines the neural network scheduler.
        controller_options (dict): Dictionary that defines the controller.
        device_string (str): String that characterizes a cuda device.
        from_multiprocessing_nGPU (str): Setting that flag will cause hippynn to train on the gpu matching the rank of
                                         the multiprocessing subprocess.
        build_lammps_pickle (bool): Whether or not to build a pickle file fo rthe network enabling loading into lammps
                                    MLIAP. Requires building of lammps python interface.

        Returns:
             (None)
    """

    # Import Torch and configure GPU.
    # This must occur after the subprocess fork!!!

    args = locals()

    import pickle
    pickle.dump(args, open("args.pkl", "wb"))
    if device_string.lower() == "from_multiprocessing":
        process = multiprocessing.current_process()
        os.environ["CUDA_VISIBLE_DEVICES"] = str((process._identity[-1] - 1) % from_multiprocessing_nGPU)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_string

    device_cuda = 'cuda:0'
    torch.cuda.set_device(device_cuda)

    import hippynn
    from hippynn import plotting
    from hippynn.graphs import inputs, targets, physics, loss
    from hippynn.experiment.assembly import assemble_for_training
    from hippynn.pretraining import set_e0_values
    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
    from hippynn.experiment import SetupParams, setup_and_train
    from hippynn.databases.database import prettyprint_arrays

    with hippynn.tools.active_directory(model_dir):
        with hippynn.tools.log_terminal("training_log.txt", "wt"):

            print("CUDA_VISIBLE_DEVICES: " + os.environ["CUDA_VISIBLE_DEVICES"])
            species = inputs.SpeciesNode(db_name=species_key)
            positions = inputs.PositionsNode(db_name=coordinates_key)
            args = (species, positions)
            periodic = False
            if cell_key is not None:
                cell = inputs.CellNode(db_name=cell_key)
                args = (*args, cell)
                periodic = True
            network = get_hippynn_network(args, hipnn_order, network_params, periodic=periodic)

            print(electrostatics_flag)
            if not electrostatics_flag:  # Just the energy Nodes, i.e. standard HIPNN.
                henergy = targets.HEnergyNode("node_HEnergy", network, first_is_interacting)
                sys_energy = henergy.mol_energy
                sys_energy.db_name = energy_key

                en_peratom = physics.PerAtom("node_EperAtom", sys_energy)
                en_peratom.db_name = energy_key + "peratom"

                hierarchicality = henergy.hierarchicality
                hierarchicality = physics.PerAtom("RperAtom", hierarchicality)

                if force_key is not None:
                    force = physics.GradientNode("node_forces", (sys_energy, positions), sign=-1)
                    force.db_name = force_key

                ### Energy Contribution to the Loss graph 
                mae_energy = loss.MAELoss.of_node(sys_energy)
                rmse_energy = loss.MSELoss.of_node(sys_energy) ** (1 / 2)

                mae_energyperatom = loss.MAELoss.of_node(en_peratom)
                rmse_energyperatom = loss.MSELoss.of_node(en_peratom) ** (1 / 2)

                rbar = loss.Mean.of_node(hierarchicality)
                l2_reg = loss.l2reg(network)
                ### 

            else:  # For charge & energy training; we use the need 2 networks.
                # The 'first' network is responsible for the electrostatic predictions. 
                hcharge = targets.HChargeNode("node_HCharge", network)
                atom_charges = hcharge.atom_charges

                if charges_key is not None:  # Partial charges to train to.
                    atom_charges.db_name = charges_key

                system_charges = physics.AtomToMolSummer("node_sysCharges", atom_charges)
                q_hierarchicality = hcharge.charge_hierarchality

                if dipole_key is not None:
                    dipole = physics.DipoleNode("node_dipole", (hcharge, positions), db_name=dipole_key)

                if quadrupole_key is not None:
                    quadrupole = physics.QuadrupoleNode("node_quadrapole", (hcharge, positions), db_name=quadrupole_key)

                # Manually define indexers for Coulomb energy 
                coulomb_r_max = torch.tensor(aca_params["r_coulomb_cut"], device=device_cuda)
                from hippynn.graphs.nodes import indexers, pairs
                enc, padidxer = indexers.acquire_encoding_padding(
                    species,
                    species_set=network_params["possible_species"]
                )
                pairfinder = pairs.OpenPairIndexer(
                    'OpenPairFinder',
                    (positions, enc, padidxer),
                    dist_hard_max=coulomb_r_max
                )

                # Coulomb Screening with Combined energy. 
                from hippynn.layers.physics import LocalDampingCosine, CombineScreenings, WolfScreening
                energy_conv = aca_params["coulomb_const"]

                # Handle different types of screnings
                if aca_params["screening_type"] == "Wolf":
                    charge_screening = CombineScreenings(
                        (LocalDampingCosine(alpha=network_params["dist_soft_max"]), WolfScreening(alpha=0.05))
                    )
                elif aca_params["screening_type"] is None:
                    charge_screening = (LocalDampingCosine(alpha=network_params["dist_soft_max"]))
                else:
                    raise NotImplementedError("Only 'Wolf' or 'null' supported screening types.")

                coulomb_energy = physics.ScreenedCoulombEnergyNode(
                    "node_cEnergy",
                    (atom_charges,
                     pairfinder.pair_dist, pairfinder.pair_first, pairfinder.pair_second,
                     padidxer.mol_index, padidxer.n_molecules),
                    energy_conversion=energy_conv,
                    screening=charge_screening,
                    cutoff_distance=coulomb_r_max
                )

                # Energy network.

                periodic = False
                if cell_key is not None:
                    periodic = True
                network_energy = get_hippynn_network(network.parents, hipnn_order, network_params, periodic=periodic)

                henergy = targets.HEnergyNode("node_HEnergy", network_energy, first_is_interacting)
                sys_energy = henergy.mol_energy + coulomb_energy
                sys_energy.db_name = energy_key

                en_peratom = physics.PerAtom("node_EperAtom", sys_energy)
                en_peratom.db_name = energy_key + "peratom"

                hierarchicality = henergy.hierarchicality
                hierarchicality = physics.PerAtom("RperAtom", hierarchicality)

                if force_key is not None:
                    force = physics.GradientNode("node_forces", (sys_energy, positions), sign=-1)
                    force.db_name = force_key

                ### Contribution to the loss graph 
                mae_energy = loss.MAELoss.of_node(sys_energy)
                rmse_energy = loss.MSELoss.of_node(sys_energy) ** (1 / 2)

                mae_energyperatom = loss.MAELoss.of_node(en_peratom)
                rmse_energyperatom = loss.MSELoss.of_node(en_peratom) ** (1 / 2)

                rbar = loss.Mean.of_node(hierarchicality)
                l2_reg = loss.l2reg(network) + loss.l2reg(network_energy)
                ###

            loss_energy = energy_mae_loss_weight * mae_energyperatom + \
                          energy_rmse_loss_weight * rmse_energyperatom + \
                          total_energy_mae_loss_weight * mae_energy + \
                          total_energy_rmse_loss_weight * rmse_energy

            loss_regularization = rbar_loss_weight * rbar + \
                                  l2_reg_loss_weight * l2_reg

            loss_error = loss_energy
            validation_losses = {
                "T-MAE": mae_energy,
                "T-RMSE": rmse_energy,
                "TperAtom-MAE": mae_energyperatom,
                "TperAtom-RMSE": rmse_energyperatom,
                "T-Hier": rbar,
                "L2Reg": l2_reg,
                "Loss-Regularization": loss_regularization,
            }
            plots_to_make = (
                plotting.Hist2D.compare(sys_energy, saved=True),
                plotting.Hist2D.compare(en_peratom, saved=True),
            )

            if force_key is not None:
                mae_force = loss.MAELoss.of_node(force)
                rmse_force = loss.MSELoss.of_node(force) ** (1 / 2)
                loss_force = force_mae_loss_weight * mae_force + \
                             force_rmse_loss_weight * rmse_force
                loss_error = loss_error + loss_force
                validation_losses.update({
                    "ForceMAE": mae_force,
                    "ForceRMSE": rmse_force,
                })
                plots_to_make = plots_to_make + \
                                (plotting.Hist2D.compare(force, saved=True),)
            print(electrostatics_flag)
            if electrostatics_flag:
                if charges_key is not None:
                    rmse_charge = loss.MSELoss.of_node(atom_charges) ** (1 / 2)
                    mae_charge = loss.MAELoss.of_node(atom_charges)
                    loss_charge = charge_rmse_loss_weight * rmse_charge + charge_mae_loss_weight * mae_charge
                    loss_error = loss_error + loss_charge
                    validation_losses.update({
                        "ChargeRMSE": rmse_charge,
                        "ChargeMAE": mae_charge,
                    })
                    plots_to_make = plots_to_make + (plotting.Hist2D.compare(atom_charges, saved=True),)

                if dipole_key is not None:
                    rmse_dipole = loss.MSELoss.of_node(dipole) ** (1 / 2)
                    mae_dipole = loss.MAELoss.of_node(dipole)
                    loss_dipole = dipole_rmse_loss_weight * rmse_dipole + dipole_mae_loss_weight * mae_dipole
                    loss_error = loss_error + loss_dipole
                    validation_losses.update({
                        "DipoleRMSE": rmse_dipole,
                        "DipoleMAE": mae_dipole,
                    })
                    plots_to_make = plots_to_make + (plotting.Hist2D.compare(dipole, saved=True),)

                if quadrupole_key is not None:
                    rmse_quadrupole = loss.MSELoss.of_node(quadrupole) ** (1 / 2)
                    mae_quadrapole = loss.MAELoss.of_node(quadrupole)
                    loss_quadrapole = quadrupole_rmse_loss_weight * rmse_quadrupole + quadrupole_mae_loss_weight * mae_quadrapole
                    loss_error = loss_error + loss_quadrapole
                    validation_losses.update({
                        "QuadrupoleRMSE": rmse_quadrupole,
                        "QuadrupoleMAE": rmse_quadrupole,
                    })
                    plots_to_make = plots_to_make + (plotting.Hist2D.compare(quadrupole, saved=True),)

            # Final parts of the loss graph. 
            loss_train = loss_error + loss_regularization
            validation_losses.update({
                "Loss-Error": loss_error,
                "Loss": loss_train,
            })

            early_stopping_key = "Loss-Error"

            # Set up Plot maker 
            plots_to_make = plots_to_make + (
                plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="sense_0.pdf"),)
            for n_int in range(1, network_params["n_interaction_layers"]):
                plots_to_make = plots_to_make + (
                    plotting.SensitivityPlot(network.torch_module.sensitivity_layers[n_int],
                                             saved="sense_%i.pdf" % n_int),)

            if electrostatics_flag:
                for n_int in range(0, network_params[
                    "n_interaction_layers"]):  # Add sensitivity layers for the 'second' network.
                    plots_to_make = plots_to_make + (
                        plotting.SensitivityPlot(network_energy.torch_module.sensitivity_layers[n_int],
                                                 saved="sense_energy_%i.pdf" % n_int),)

            plot_maker = plotting.PlotMaker(
                *plots_to_make,
                plot_every=plot_every
            )

            training_modules, db_info = assemble_for_training(loss_train, validation_losses, plot_maker=plot_maker)
            model, loss_module, model_evaluator = training_modules

            database = PyAniDB(zarr_path, reduction_name, inputs=None,
                               targets=None,
                               seed=np.random.randint(1e9),
                               quiet=False,
                               allow_unfound=True,
                               species_key=species_key
                               )

            if database.arr_dict[energy_key].shape[-1] == 1:
                database.arr_dict[energy_key] = database.arr_dict[energy_key][..., 0]

            torch.set_default_dtype(torch.float32)
            # Ensure that species array is int64 for indexing
            arrays = database.arr_dict
            arrays[species_key] = arrays[species_key].astype('int64')
            n_atoms = arrays[species_key].astype(bool).astype(int).sum(axis=1)
            if not (energy_key + "peratom" in arrays):
                arrays[energy_key + "peratom"] = arrays[energy_key] / n_atoms

            # Ensure that float arrays are float32
            for k, v in arrays.items():
                if v.dtype == np.dtype('float64'):
                    arrays[k] = v.astype(np.float32)

            database.inputs = db_info["inputs"]
            database.targets = db_info["targets"]

            # Flatten Quadrupole data to match hippynn's internal representation. 
            if not (quadrupole_key == None) and electrostatics_flag:
                arrays[quadrupole_key] = arrays[quadrupole_key].reshape(-1, 9)

            for key in list(arrays):
                if not (key in db_info['inputs']) and not (key in db_info['targets']) and not (key == 'indices'):
                    del arrays[key]

            # Remove High energies and forces. 
            if force_key is not None:
                database.remove_high_property(force_key, True, species_key=species_key, cut=remove_high_forces_cut,
                                              std_factor=remove_high_forces_std)
            database.remove_high_property(energy_key, False, species_key=species_key, cut=remove_high_energy_cut,
                                          std_factor=remove_high_energy_std)

            print("Array Shapes After Cleaning")
            prettyprint_arrays(database.arr_dict)
            print(np.max(database.arr_dict[force_key]))
            print(np.min(database.arr_dict[force_key]))
            print(np.max(database.arr_dict[energy_key + "peratom"]))
            print(np.min(database.arr_dict[energy_key + "peratom"]))

            print(valid_size, test_size, len(database))
            database.make_random_split("valid", valid_size)

            if zarr_test_path is not None:
                database_test = PyAniDB(zarr_path, reduction_name, inputs=None,
                                        targets=None,
                                        seed=np.random.randint(1e9),
                                        quiet=False,
                                        allow_unfound=True
                                        )
                # Ensure that species array is int64 for indexing
                arrays_test = database_test.arr_dict
                arrays_test[species_key] = arrays_test[species_key].astype('int64')

                # Ensure that float arrays are float32
                for k, v in arrays_test.items():
                    if v.dtype == np.dtype('float64'):
                        arrays[k] = v.astype(np.float32)
                database_test.split_the_rest("test")
                database.split_the_rest("train")
                database.splits['test'] = database_test.splits['test']

                # Reshape Quadrupole data. 
                if quadrupole_key is not None and electrostatics_flag:
                    arrays[quadrupole_key] = arrays[quadrupole_key].reshape(-1, 9)
            else:
                database.make_random_split("test", test_size)
                database.split_the_rest("train")

            # Compute E0 values based on linear regression and remove from system energies
            if henergy.module_kwargs["first_is_interacting"]:
                from hippynn.networks.hipnn import compute_hipnn_e0
                from hippynn.graphs.nodes.tags import Encoder
                from hippynn.graphs import find_unique_relative
                from hippynn.graphs.nodes.base import _BaseNode
                encoder = None
                species_name = None
                energy_name = None
                energy_module = henergy
                if encoder is None:
                    encoder = find_unique_relative(energy_module, Encoder, "Constructing E0 Values")
                if species_name is None:
                    species_name = find_unique_relative(energy_module, inputs.SpeciesNode,
                                                        "Constructing E0 Values").db_name
                if energy_name is None:
                    energy_name = energy_module.main_output.db_name
                energy_module = energy_module.torch_module

                if isinstance(encoder, _BaseNode):
                    encoder = encoder.torch_module
                train_data = database.splits["train"]
                z_vals = train_data[species_key]
                t_vals = train_data[energy_key]

                eovals = compute_hipnn_e0(encoder, z_vals, t_vals).detach().numpy()
                print("Computed E0 energies from when first is interacting:", eovals)
                SELF_ENERGY_APPROX = {k: v for k, v in zip(network_params["possible_species"][1:], eovals)}
                SELF_ENERGY_APPROX[0] = 0
                for data_split in ["train", "valid", "test"]:
                    self_energy = np.vectorize(SELF_ENERGY_APPROX.__getitem__)(database.splits[data_split][species_key])
                    self_energySum = self_energy.sum(axis=1)  # Add up over atoms in system.
                    database.splits[data_split][energy_key] = (
                            database.splits[data_split][energy_key] - self_energySum).to(torch.float32)
                    database.splits[data_split]["energyperatom"] = (
                            database.splits[data_split]["energyperatom"] - self_energySum / torch.count_nonzero(
                        database.splits[data_split][species_key], dim=1)).to(torch.float32)
                    # Ensure that float arrays are float32
                    database.splits[data_split][coordinates_key] = database.splits[data_split][coordinates_key].to(
                        torch.float32)
                    if not (force_key is None):
                        database.splits[data_split][force_key] = database.splits[data_split][force_key].to(
                            torch.float32)
                training_modules.model.to(database.splits['train'][sys_energy.db_name].device)

            else:
                # Ensure that float arrays are float32
                for data_split in ["train", "valid", "test"]:
                    database.splits[data_split][energy_key] = (database.splits[data_split][energy_key]).to(
                        torch.float32)
                    database.splits[data_split]["energyperatom"] = database.splits[data_split]["energyperatom"].to(
                        torch.float32)
                    database.splits[data_split][coordinates_key] = database.splits[data_split][coordinates_key].to(
                        torch.float32)
                    if not (force_key is None):
                        database.splits[data_split][force_key] = database.splits[data_split][force_key].to(
                            torch.float32)

                training_modules.model.to(database.splits['train'][sys_energy.db_name].device)

                device = database.splits['train'][sys_energy.db_name].device
                database.send_to_device("cpu")
                set_e0_values(henergy, database, peratom=False, energy_name=energy_key, decay_factor=1e-2)
                database.send_to_device(device)

            optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=learning_rate)

            scheduler = RaiseBatchSizeOnPlateau(
                optimizer=optimizer,
                **scheduler_options
            )

            controller = PatienceController(
                optimizer=optimizer,
                scheduler=scheduler,
                stopping_key=early_stopping_key,
                **controller_options
            )
            experiment_params = SetupParams(controller=controller)

            print("Controller Options:")
            print(controller_options, "\n\n")

            print("Experiment Params:")
            print(experiment_params)

            # Parameters describing the training procedure.
            setup_and_train(
                training_modules=training_modules,
                database=database,
                setup_params=experiment_params,
            )

            if build_lammps_pickle:
                from hippynn.interfaces.lammps_interface import MLIAPInterface
                from hippynn.tools import device_fallback
                possible_symbols = [chemical_symbols[num] for num in network_params['possible_species']]
                possible_symbols.remove('X')
                unified = MLIAPInterface(henergy, possible_symbols, model_device=device_fallback)
                torch.save(unified, "lammps_model.pt")


def train_HIPNN_model_wrapper(arg_dict):
    """HIPNN train wrapper

    Args:
        arg_dict (dict): Dictionary containing training parameters.
    """
    return train_HIPNN_model(**arg_dict)


@python_app(executors=['alf_ML_executor'])
def train_HIPPYNN_ensemble_task(ML_config, zarr_path, reduction_name, model_path, current_training_id, gpus_per_node,
                                remove_existing=False, h5_test_dir=None):
    """Trains an ensemble of HIPPYNN neural networks.

    Args:
        ML_config (dict): Dictionary containig HIPPYNN parameters as defined in the ML config json.
        h5_dir (str): Path to the database that stores the training data.
        model_path (str): Path to the directory that will store the model.
        current_training_id (int): Integer indicating the model id.
        gpus_per_node (int): Number of GPUs per node.
        remove_existing (bool): If True deletes the previous model before training the new one.
        h5_test_dir (str or None): Path to the directory that stores the test data.

    """
    p = multiprocessing.Pool(gpus_per_node)
    general_configuration = ML_config.copy()
    n_models = general_configuration.pop('n_models')
    params_list = [general_configuration.copy() for i in range(n_models)]
    for i, cur_dict in enumerate(params_list):
        cur_dict['model_dir'] = model_path.format(current_training_id) + '/model-{:02d}'.format(i)
        cur_dict['zarr_path'] = zarr_path
        cur_dict['reduction_name'] = reduction_name
        cur_dict['from_multiprocessing_nGPU'] = gpus_per_node
    out = p.map(train_HIPNN_model_wrapper, params_list)
    completed = []
    HIPNN_complete = re.compile('Training complete')

    p.close()
    for i in range(n_models):
        log = open(model_path.format(current_training_id) + '/model-{:02d}/training_log.txt'.format(i), 'r').read()
        if len(HIPNN_complete.findall(log)) == 1:
            completed.append(True)
        else:
            completed.append(False)

    return completed, current_training_id


def HIPNN_ASE_calculator(HIPNN_model_directory, energy_key='energy', device="cuda:0"):
    """HIPPYNN calculator that can be used in ASE to run MD.

    Args:
        HIPNN_model_directory (str): Path of HIPPYNN model.
        energy_key (str): Key to retrieve the energy.
        device (str): Device responsible for loading the tensor into memory.

    Returns:
        (hippynn.interfaces.ase_interface.HippynCalculator): HippynnCalculator object that contains the HIPPYNN model.

    """
    from hippynn.experiment.serialization import load_checkpoint_from_cwd
    from hippynn.tools import active_directory
    from hippynn.interfaces.ase_interface import HippynnCalculator
    import torch

    with active_directory(HIPNN_model_directory):
        bundle = load_checkpoint_from_cwd(map_location="cpu", restore_db=False)
    model = bundle["training_modules"].model
    energy_node = model.node_from_name(energy_key)
    calc = HippynnCalculator(energy_node, en_unit=units.eV)
    calc.to(torch.float32)
    calc.to(torch.device(device))

    return calc


def HIPNN_ASE_load_ensemble(HIPNN_ensemble_directory, device="cuda:0"):
    """Loads an ensemble of HIPPYN neural networks.

    Args:
        HIPNN_ensemble_directory (str): Path of the directory containing the ensemble of HIPPYNN models.
        device (str):

    Returns:
        (list): A list of ASE calculators that uses a HIPPYNN model to calculate forces and energies.

    """
    model_list = []
    for cur_dir in glob.glob(HIPNN_ensemble_directory + '/model-*/'):
        model_list.append(HIPNN_ASE_calculator(cur_dir, device=device))

    return model_list
