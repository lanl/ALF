from __future__ import annotations

from parsl import python_app, bash_app, join_app
import sys
import multiprocessing
import re
import numpy as np
import glob
import os
from ase import units
from ase.data import chemical_symbols
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any
from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS, canonical_orders
from alframework.tools import ensemble_training
from alframework.tools.excited_state_tools import derive_state_property_table, derive_gap_property_table


def __getattr__(name):
    # Keep HIPPYNN/Torch imports after the training worker's device setup while
    # exposing the database class at a stable module path for checkpoint reload.
    if name != 'ALFPyAniDirectoryDB':
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from hippynn.databases.h5_pyanitools import PyAniDirectoryDB

    class ALFPyAniDirectoryDB(PyAniDirectoryDB):
        def __init__(self, *args, coordinates_key='coordinates', atom_keys=(),
                     excited_state_properties=None, retained_indices=None, **kwargs):
            self.coordinates_key = coordinates_key
            self.excited_state_properties = excited_state_properties
            self.retained_indices = retained_indices
            self.alf_frame_manifest = []
            self.atom_keys = tuple(dict.fromkeys(('coordinates', 'forces', 'charges', coordinates_key,
                                                 *(key for key in atom_keys if key is not None))))
            super().__init__(*args, **kwargs)
            # HIPPYNN persists this restarter in checkpoints; retain custom schema.
            self.restarter.kwargs.update(coordinates_key=coordinates_key, atom_keys=self.atom_keys,
                                        allow_unfound=kwargs.get('allow_unfound', False))
            if excited_state_properties is not None:
                self.restarter.kwargs.update(excited_state_properties=excited_state_properties,
                                            retained_indices=retained_indices)

        def extract_full_file(self, file, species_key='species', permit_only=None):
            # Newer HIPPYNN passes permit_only; older releases do not accept it.
            if self.excited_state_properties is None:
                extra = {} if permit_only is None else {'permit_only': permit_only}
                return super().extract_full_file(file, species_key=species_key, **extra)
            # The excited-state path selects its datasets explicitly below.
            states = validate_excited_state_training_properties(self.excited_state_properties)
            gaps = derive_gap_property_table(self.excited_state_properties)
            # Select datasets before reading: pair indices/NACs and optics may
            # have unrelated axes and can be much larger than training targets.
            import h5py
            from ase.data import atomic_numbers
            keys = {self.coordinates_key, species_key, TOPOLOGY_ATOM_IDS, '_id', 'pbc'}
            keys.update(row[key] for row in states for key in ('energy_db_name', 'force_db_name'))
            keys.update(row['gap_db_name'] for row in gaps)
            batches, paths = [], []
            with h5py.File(file, 'r') as handle:
                names = []
                if self.coordinates_key in handle or species_key in handle:
                    names.append('/')
                handle.visititems(lambda name, obj: names.append(name)
                                  if isinstance(obj, h5py.Group) and
                                  (self.coordinates_key in obj or species_key in obj) else None)
                for name in sorted(names):
                    group = handle[name]
                    if self.coordinates_key not in group or species_key not in group:
                        raise KeyError(f'{file}:{name}: missing required coordinates/species datasets')
                    batch = {key: group[key][()] for key in keys if key in group}
                    species = np.asarray(batch[species_key])
                    if species.dtype.kind in 'SUO':
                        species = np.array([atomic_numbers[v.decode() if isinstance(v, bytes) else str(v)]
                                            for v in species.reshape(-1)]).reshape(species.shape)
                    if species.ndim == 1:
                        species = species[None]
                    batch[species_key] = species
                    batches.append(batch)
                    paths.append(name)
            if not batches:
                raise ValueError(f'No structure groups in {file}')
            n_atoms_max = max(b[species_key].shape[-1] for b in batches)
            selected = []
            for group_index, batch in enumerate(batches):
                context = f'{file}:{paths[group_index]}'
                coordinates = np.asarray(batch[self.coordinates_key], dtype=np.float64)
                if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or min(coordinates.shape[:2]) < 1:
                    raise ValueError(f'{context}: coordinates must be nonempty [frames, atoms, 3]')
                count, atoms = coordinates.shape[:2]
                species = np.asarray(batch[species_key])
                if species.shape != (1, atoms):
                    raise ValueError(f'{context}: species must describe one composition per group')
                species_rows = np.broadcast_to(species, (count, atoms)).copy()
                orders = None
                if TOPOLOGY_ATOM_IDS in batch:
                    orders = canonical_orders(batch[TOPOLOGY_ATOM_IDS], count, atoms, context=context)
                    species_rows = np.take_along_axis(species_rows, orders, axis=1)
                    coordinates = np.take_along_axis(coordinates, orders[:, :, None], axis=1)
                if not np.all(species_rows == species_rows[0]):
                    raise ValueError(f'{context}: topology mappings change the atomic-number sequence')
                if 'pbc' in batch and np.any(batch['pbc']):
                    raise ValueError('Excited-state training requires nonperiodic data')
                out = {self.coordinates_key: coordinates, species_key: species_rows[:1]}
                for row in states:
                    ekey, fkey = row['energy_db_name'], row['force_db_name']
                    if ekey not in batch or fkey not in batch:
                        raise KeyError(f'{context}: missing required state energy/force datasets {ekey}, {fkey}')
                    energy = np.asarray(batch[ekey], dtype=np.float64)
                    if energy.shape == (count, 1):
                        energy = energy[:, 0]
                    if energy.shape != (count,):
                        raise ValueError(f'{context}:{ekey} must contain one energy per structure')
                    forces = np.asarray(batch[fkey], dtype=np.float64)
                    if forces.shape != (count, atoms, 3):
                        raise ValueError(f'{context}:{fkey} must have shape [frames, atoms, 3]')
                    if orders is not None:
                        forces = np.take_along_axis(forces, orders[:, :, None], axis=1)
                    out.update({ekey: energy, fkey: forces})
                for row in gaps:
                    key = row['gap_db_name']
                    if key not in batch:
                        raise KeyError(f'{context}: missing required datasets: {key}')
                    values = np.asarray(batch[key], dtype=np.float64).reshape(-1)
                    if not np.isfinite(values).all():
                        raise ValueError(f'{context}:{key} contains nonfinite values')
                    expected = (out[self.excited_state_properties[row['upper_energy_key']][0]]
                                - out[self.excited_state_properties[row['lower_energy_key']][0]])
                    if values.shape != (count,) or not np.allclose(values, expected, rtol=1e-7, atol=1e-7):
                        raise ValueError(f'{context}: gap {key} is inconsistent with upper minus lower energy')
                    out[key] = values
                for key, values in out.items():
                    if not np.isfinite(values).all():
                        raise ValueError(f'{context}:{key} contains nonfinite values')
                source_ids = np.asarray(batch.get('_id', [f'{context}:{i}' for i in range(count)])).reshape(-1)
                if len(source_ids) != count:
                    raise ValueError(f'{context}: ID count does not match coordinates')
                self.alf_frame_manifest.extend(dict(file=str(Path(file).resolve()), group=paths[group_index],
                                                    frame=i, id=v.decode() if isinstance(v, bytes) else str(v))
                                               for i, v in enumerate(source_ids))
                selected.append(out)
            return selected, n_atoms_max, sum(len(b[self.coordinates_key]) for b in selected)

        def load_arrays(self, allow_unfound=False, quiet=False):
            arrays = super().load_arrays(allow_unfound=allow_unfound, quiet=quiet)
            if self.excited_state_properties is not None and self.retained_indices is not None:
                indices = np.asarray(self.retained_indices, dtype=int)
                arrays = {key: value[indices] for key, value in arrays.items()}
                arrays['indices'] = indices
                self.alf_frame_manifest = [self.alf_frame_manifest[i] for i in indices]
            return arrays

        def determine_key_structure(self, batches, sys_count, n_atoms_max, species_key='species'):
            if not batches:
                raise ValueError('No structure groups in the ALF training dataset')
            if self.excited_state_properties is None and all(batch[species_key].shape[1] >= 7 for batch in batches):
                return super().determine_key_structure(batches, sys_count, n_atoms_max, species_key=species_key)

            keys = set(batches[0])
            atom_keys = set(self.atom_keys) | {species_key}
            shapes = {}
            padding = {}
            for key, value in batches[0].items():
                shapes[key] = [sys_count, *value.shape[1:]]
                padding[key] = [1] if key in atom_keys else []
                if padding[key]:
                    if value.ndim < 2:
                        raise ValueError(f'Atom property {key!r} has no atom axis')
                    shapes[key][1] = n_atoms_max
            for batch in batches:
                if set(batch) != keys:
                    raise ValueError('Inconsistent property keys across ALF structure groups')
                coordinates = batch[self.coordinates_key]
                n_frames, n_atoms = coordinates.shape[:2]
                if coordinates.shape != (n_frames, n_atoms, 3):
                    raise ValueError('ALF coordinates must have shape (frames, atoms, 3)')
                if batch[species_key].shape != (1, n_atoms):
                    raise ValueError('ALF species must describe one fixed composition per group')
                for key, value in batch.items():
                    expected = list(shapes[key][1:])
                    if key in atom_keys:
                        expected[0] = n_atoms
                    if list(value.shape[1:]) != expected:
                        raise ValueError(f'Unsupported trailing shape for {key!r}; configure its atom-property key')
                    if key != species_key and value.shape[0] != n_frames:
                        raise ValueError(f'Frame count mismatch for {key!r}')
            padding['sys_number'] = []
            return padding, shapes, self.coordinates_key

    ALFPyAniDirectoryDB.__qualname__ = 'ALFPyAniDirectoryDB'
    globals()[name] = ALFPyAniDirectoryDB
    return ALFPyAniDirectoryDB


def train_HIPNN_model(model_dir,
                      h5_train_dir,
                      energy_key,
                      coordinates_key,
                      species_key,
                      charges_key=None,
                      dipole_key=None,
                      quadrupole_key=None, 
                      network_params=None,
                      aca_params=None,
                      electrostatics_flag=False,
                      cell_key=None,
                      force_key=None,
                      hipnn_order = 'scalar',
                      first_is_interacting = False,
                      energy_mae_loss_weight = 1e2,
                      energy_rmse_loss_weight = 1e2,
                      total_energy_mae_loss_weight = 0,
                      total_energy_rmse_loss_weight = 0,
                      force_mae_loss_weight = 1.0,
                      force_rmse_loss_weight = 1.0,
                      charge_mae_loss_weight = 1.0,
                      charge_rmse_loss_weight = 1.0,
                      dipole_mae_loss_weight = 1.0,
                      dipole_rmse_loss_weight = 1.0,
                      quadrupole_mae_loss_weight = 1.0,
                      quadrupole_rmse_loss_weight = 1.0,
                      electrostatics_weight=0,
                      rbar_loss_weight = 1.0,
                      l2_reg_loss_weight = 1e-6,
                      pairwise_inference_parameters = None,
                      valid_size = .1,
                      test_size = .1,
                      plot_every = 50,
                      h5_test_dir = None,
                      learning_rate = 5e-4,
                      scheduler_options = {"max_batch_size":128,"patience":25,"factor":0.5},
                      controller_options = {"batch_size":64,"eval_batch_size":128,"max_epochs":1000,"termination_patience":55},
                      device_string = '0',
                      from_multiprocessing_nGPU = None,
                      build_lammps_pickle = False,
                      remove_high_energy_cut = None, 
                      remove_high_energy_std = None,
                      remove_high_forces_cut = None, 
                      remove_high_forces_std = None
                      ):
    """Train one HIPPYNN model on the HDF5 data in a directory.

    Dataset keys must match the database names in ``properties_list``.

    Args:
        model_dir (str): Directory where the HIPPYNN model will reside.
        h5_train_dir (str): Directory containing the HDF5 training data.
        energy_key (str): Energy key inside the HDF5 dataset.
        coordinates_key (str): Coordinates key inside the HDF5 dataset.
        species_key (str): Species key inside the HDF5 dataset.
        charges_key (str): Per-atom charges key. If None, partial charges are not trained to.
        dipole_key (str): Molecular dipole key. If None, dipoles are not trained to.
        quadrupole_key (str): Molecular quadrupole key. If None, quadrupoles are not trained to.
        network_params (dict): Defines the model. See the examples for details.
        aca_params (dict): Defines parameters for charge training.
        electrostatics_flag (bool): Whether electrostatics are trained with the energy and forces.
            If True, at least one of ``charges_key``, ``dipole_key``, or ``quadrupole_key`` must be set.
        cell_key (str): Cell key inside the HDF5 dataset. Leave as None if the data is not periodic.
        force_key (str): Force key inside the HDF5 dataset. Leave as None to remove force training.
        hipnn_order (str): ``'scalar'``, ``'vector'``, or ``'quadradic'``.
        first_is_interacting (bool): Passed to HIPPYNN's energy target node.
        energy_mae_loss_weight (float): MAE per-atom energy loss weight.
        energy_rmse_loss_weight (float): RMSE per-atom energy loss weight.
        total_energy_mae_loss_weight (float): MAE total energy loss weight. Typically only one of
            ``energy_mae_loss_weight`` and this weight is used, and the other is set to 0.
        total_energy_rmse_loss_weight (float): RMSE total energy loss weight. Typically only one of
            ``energy_rmse_loss_weight`` and this weight is used, and the other is set to 0.
        force_mae_loss_weight (float): MAE force loss weight.
        force_rmse_loss_weight (float): RMSE force loss weight.
        charge_mae_loss_weight (float): MAE charge loss weight.
        charge_rmse_loss_weight (float): RMSE charge loss weight.
        dipole_mae_loss_weight (float): MAE dipole loss weight.
        dipole_rmse_loss_weight (float): RMSE dipole loss weight.
        quadrupole_mae_loss_weight (float): MAE quadrupole loss weight.
        quadrupole_rmse_loss_weight (float): RMSE quadrupole loss weight.
        electrostatics_weight (float): Not currently used.
        rbar_loss_weight (float): Weight of rbar in the loss.
        l2_reg_loss_weight (float): Weight of L2 regularization in the loss. Typically 1e-6.
        pairwise_inference_parameters (dict): Not currently used.
        valid_size (float): Fraction of the data reserved for validation.
        test_size (float): Fraction of the data reserved for testing, when ``h5_test_dir`` is None.
        plot_every (int): Generate plots every this many epochs.
        h5_test_dir (str): Optional directory containing a separate test set.
        learning_rate (float): Initial optimizer learning rate.
        scheduler_options (dict): Learning-rate scheduler settings.
        controller_options (dict): Training loop settings.
        device_string (str): GPU to use, or ``'from_multiprocessing'`` to choose one from the worker process.
        from_multiprocessing_nGPU (int): Number of GPUs to distribute over with ``'from_multiprocessing'``.
        build_lammps_pickle (bool): Whether to build a pickle file for loading the network into LAMMPS
            MLIAP. Requires the LAMMPS Python interface.
        remove_high_energy_cut (float): Remove structures whose energy is farther than this from the mean.
            If None, this step is skipped.
        remove_high_energy_std (float): Remove structures whose energy is this many standard deviations
            or more above the mean. If None, this step is skipped.
        remove_high_forces_cut (float): Remove structures with a force farther than this from the mean.
            If None, this step is skipped.
        remove_high_forces_std (float): Remove structures with a force this many standard deviations or
            more above the mean. If None, this step is skipped.
    """
    #Import Torch and configure GPU. 
    #This must occur after the subprocess fork!!!
    import torch
    device_cuda = _ensemble_member_device(device_string, from_multiprocessing_nGPU)
    torch.cuda.set_device(device_cuda)
    
    import hippynn
    from hippynn import plotting
    from hippynn.graphs import inputs, networks, targets, physics, loss
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB
    from hippynn.experiment.assembly import assemble_for_training
    from hippynn.pretraining import set_e0_values
    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
    from hippynn.experiment import SetupParams, setup_and_train
    from hippynn.databases.database import  prettyprint_arrays
    
    
    with hippynn.tools.active_directory(model_dir):
        with hippynn.tools.log_terminal("training_log.txt","wt"):
            
            print("Training device: {}; CUDA_VISIBLE_DEVICES: {}".format(device_cuda, os.environ.get("CUDA_VISIBLE_DEVICES")))
            
            species = inputs.SpeciesNode(db_name=species_key)
            positions = inputs.PositionsNode(db_name=coordinates_key)
            
            if not(cell_key is None):
                cell = inputs.CellNode(db_name=cell_key)
                if hipnn_order.lower() == 'scalar':
                    network = networks.Hipnn("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)
                elif hipnn_order.lower() == 'vector':
                    network = networks.HipnnVec("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)
                elif hipnn_order.lower() == 'quadradic':
                    network = networks.HipnnQuad("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)
                else:
                    raise RuntimeError('Unrecognized hipnn_order parameter')
            else:
                if hipnn_order.lower() == 'scalar':
                    network = networks.Hipnn("HIPNN", (species, positions), module_kwargs=network_params, periodic=False)
                elif hipnn_order.lower() == 'vector':
                    network = networks.HipnnVec("HIPNN", (species, positions), module_kwargs=network_params, periodic=False)
                elif hipnn_order.lower() == 'quadradic':
                    network = networks.HipnnQuad("HIPNN", (species, positions), module_kwargs=network_params, periodic=False)
                else:
                    raise RuntimeError('Unrecognized hipnn_order parameter')
            print(electrostatics_flag)
            if not(electrostatics_flag): # Just the energy Nodes, i.e. standard HIPNN. 
                henergy = targets.HEnergyNode("node_HEnergy",network,first_is_interacting)
                sys_energy = henergy.mol_energy
                sys_energy.db_name = energy_key
                
                en_peratom = physics.PerAtom("node_EperAtom", sys_energy)
                en_peratom.db_name = energy_key+"peratom"
                
                hierarchicality = henergy.hierarchicality
                hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
                
                if not(force_key is None):
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
              
            else : # For charge & energy training; we use the need 2 networks.  
                # The 'first' network is responsible for the electrostatic predictions. 
                hcharge = targets.HChargeNode("node_HCharge", network)
                atom_charges = hcharge.atom_charges
            
                if not(charges_key is None): # Partial charges to train to. 
                    atom_charges.db_name = charges_key
            
                system_charges = physics.AtomToMolSummer("node_sysCharges", atom_charges)
                q_hierarchicality = hcharge.charge_hierarchality

                if not(dipole_key is None): 
                    dipole = physics.DipoleNode("node_dipole", (hcharge, positions), db_name=dipole_key)
                
                if not(quadrupole_key is None): 
                    quadrupole = physics.QuadrupoleNode("node_quadrapole", (hcharge, positions), db_name=quadrupole_key)
        
                # Manually define indexers for Coulomb energy 
                coulomb_r_max = torch.tensor(aca_params["r_coulomb_cut"], device=device_cuda) 
                from hippynn.graphs.nodes import indexers, pairs
                enc, padidxer = indexers.acquire_encoding_padding(
                    species, 
                    species_set = network_params["possible_species"]
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
                elif aca_params["screening_type"] == None:
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
                if not(cell_key is None):
                    cell = inputs.CellNode(db_name=cell_key)
                    if hipnn_order.lower() == 'scalar':
                        network_energy = networks.Hipnn("HIPNN", network.parents, module_kwargs=network_params, periodic=True)
                    elif hipnn_order.lower() == 'vector':
                        network_energy = networks.HipnnVec("HIPNN", network.parents, module_kwargs=network_params, periodic=True)
                    elif hipnn_order.lower() == 'quadradic':
                        network_energy = networks.HipnnQuad("HIPNN", network.parents, module_kwargs=network_params, periodic=True)
                    else:
                        raise RuntimeError('Unrecognized hipnn_order parameter')
                else:
                    if hipnn_order.lower() == 'scalar':
                        network_energy = networks.Hipnn("HIPNN", network.parents, module_kwargs=network_params, periodic=False)
                    elif hipnn_order.lower() == 'vector':
                        network_energy = networks.HipnnVec("HIPNN", network.parents, module_kwargs=network_params, periodic=False)
                    elif hipnn_order.lower() == 'quadradic':
                        network_energy = networks.HipnnQuad("HIPNN", network.parents, module_kwargs=network_params, periodic=False)
                    else:
                        raise RuntimeError('Unrecognized hipnn_order parameter')
            
                
                henergy = targets.HEnergyNode("node_HEnergy", network_energy,first_is_interacting)
                sys_energy = henergy.mol_energy + coulomb_energy 
                sys_energy.db_name = energy_key
                
                en_peratom = physics.PerAtom("node_EperAtom", sys_energy)
                en_peratom.db_name = energy_key+"peratom"
                
                hierarchicality = henergy.hierarchicality 
                hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
                
                if not(force_key == None):
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
                        
            if not(force_key == None):
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
                if not(charges_key == None):
                    rmse_charge =  loss.MSELoss.of_node(atom_charges) ** (1/2)
                    mae_charge =  loss.MAELoss.of_node(atom_charges)
                    loss_charge = charge_rmse_loss_weight*rmse_charge + charge_mae_loss_weight*mae_charge
                    loss_error = loss_error + loss_charge
                    validation_losses.update({
                        "ChargeRMSE": rmse_charge,
                        "ChargeMAE": mae_charge,
                    })
                    plots_to_make = plots_to_make + (plotting.Hist2D.compare(atom_charges, saved=True),)
                    
                if not(dipole_key == None):
                    rmse_dipole =  loss.MSELoss.of_node(dipole) ** (1/2)
                    mae_dipole = loss.MAELoss.of_node(dipole)
                    loss_dipole = dipole_rmse_loss_weight*rmse_dipole + dipole_mae_loss_weight*mae_dipole
                    loss_error = loss_error + loss_dipole
                    validation_losses.update({
                        "DipoleRMSE": rmse_dipole,
                        "DipoleMAE": mae_dipole,
                    })
                    plots_to_make = plots_to_make + (plotting.Hist2D.compare(dipole, saved=True),)
                    
                if not(quadrupole_key == None):
                    rmse_quadrupole =  loss.MSELoss.of_node(quadrupole) ** (1/2)
                    mae_quadrapole = loss.MAELoss.of_node(quadrupole)
                    loss_quadrapole = quadrupole_rmse_loss_weight*rmse_quadrupole + quadrupole_mae_loss_weight*mae_quadrapole
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
            plots_to_make = plots_to_make + (plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="sense_0.pdf"),)
            for n_int in range(1,network_params["n_interaction_layers"]):
                plots_to_make = plots_to_make +  (plotting.SensitivityPlot(network.torch_module.sensitivity_layers[n_int], saved="sense_%i.pdf"%n_int),)                
            
            if electrostatics_flag: 
                for n_int in range(0,network_params["n_interaction_layers"]): # Add sensitivity layers for the 'second' network. 
                    plots_to_make = plots_to_make +  (plotting.SensitivityPlot(network_energy.torch_module.sensitivity_layers[n_int], saved="sense_energy_%i.pdf"%n_int),)                
        
            plot_maker = plotting.PlotMaker(
                *plots_to_make,
                plot_every=plot_every
            )
            
            training_modules, db_info = assemble_for_training(loss_train, validation_losses, plot_maker=plot_maker)
            model, loss_module, model_evaluator= training_modules
        
            database_schema = dict(species_key=species_key, coordinates_key=coordinates_key,
                                   atom_keys=(coordinates_key, force_key, charges_key))
            database = ALFPyAniDirectoryDB(directory=h5_train_dir,allow_unfound=True,quiet=False,seed=np.random.randint(1e9),inputs=None,targets=None,**database_schema)
            
            #Ensure that species array is int64 for indexing
            arrays = database.arr_dict

            arrays[species_key] = arrays[species_key].to(torch.int64)   # or .long()

            min_dists = hippynn.pretraining.calculate_min_dists(database.arr_dict,species_key,coordinates_key,6.0,cell_name=cell_key,batch_size=256,device=device_cuda)
            min_dists_flag = min_dists>network_params["dist_soft_min"]
            for curK in database.arr_dict.keys():
                database.arr_dict[curK] = database.arr_dict[curK][min_dists_flag]
                
            n_atoms = (
              arrays[species_key]          # tensor
              .to(torch.bool)              # cast to bool
              .to(torch.int32)             # or .int() / .to(torch.int64) as you prefer
              .sum(dim=1)                  # sum along atom axis
             )

            min_dists = hippynn.pretraining.calculate_min_dists(database.arr_dict,species_name=species_key,positions_name=coordinates_key,dist_hard_max=6.0,cell_name=cell_key,batch_size=256,device=device_cuda)

            if not( energy_key+"peratom" in list(arrays.keys())):
                arrays[energy_key+"peratom"] =  arrays[energy_key] / n_atoms

            for k, v in arrays.items():
                if v.dtype == torch.float64:
                    arrays[k] = v.to(torch.float32)

            database.inputs = db_info["inputs"]
            database.targets = db_info["targets"]

            # Flatten Quadrupole data to match hippynn's internal representation. 
            if not(quadrupole_key == None) and electrostatics_flag:
                arrays[quadrupole_key] = arrays[quadrupole_key].reshape(-1,9)
    
            for key in list(arrays):
                if not(key in db_info['inputs']) and not(key in db_info['targets']) and not(key=='indices'):
                    del arrays[key]

            # Ensure that the "indices" array is unique and consecutive
            #num_samples = arrays[species_key].shape[0]
            #database.arr_dict["indices"] = torch.arange(num_samples, dtype=torch.int64)


            # Remove High energies and forces. 
            if force_key != None:
                database.remove_high_property(force_key, True, species_key=species_key, cut=remove_high_forces_cut, std_factor=remove_high_forces_std)
            database.remove_high_property(energy_key, False, species_key=species_key, cut=remove_high_energy_cut, std_factor=remove_high_energy_std)


            print("Array Shapes After Cleaning")
            prettyprint_arrays(database.arr_dict)
                        
            database.make_random_split("valid",valid_size)
            if not(h5_test_dir == None):
                database_test = ALFPyAniDirectoryDB(directory=h5_test_dir,allow_unfound=True,quiet=False,seed=np.random.randint(1e9),**db_info,**database_schema)
                #Ensure that species array is int64 for indexing
                arrays_test = database.arr_dict
                arrays_test[species_key] = arrays_test[species_key].astype('int64')
            
                #Ensure that float arrays are float32
                for k, v in arrays_test.items():
                    if v.dtype == np.dtype('float64'):
                        arrays[k] = v.astype(np.float32)
                database_test.split_the_rest("test")
                database.split_the_rest("train")
                database.splits['test'] = database_test.splits['test']
                                
                # Reshape Quadrupole data. 
                if not(quadrupole_key == None) and electrostatics_flag:
                    arrays[quadrupole_key] = arrays[quadrupole_key].reshape(-1,9)
            else:
                database.make_random_split("test",test_size)
                database.split_the_rest("train")
            
            #Compute E0 values based on linear regression and remove from system energies
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
                    species_name = find_unique_relative(energy_module, inputs.SpeciesNode, "Constructing E0 Values").db_name
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
                for data_split in ["train","valid","test"]: 
                    self_energy = np.vectorize(SELF_ENERGY_APPROX.__getitem__)(database.splits[data_split][species_key])
                    self_energySum = self_energy.sum(axis=1)  # Add up over atoms in system.
                    database.splits[data_split][energy_key] = (database.splits[data_split][energy_key] - self_energySum).to(torch.float32)
                    database.splits[data_split]["energyperatom"] = (database.splits[data_split]["energyperatom"] - self_energySum/torch.count_nonzero(database.splits[data_split][species_key], dim=1)).to(torch.float32)
                    #Ensure that float arrays are float32
                    database.splits[data_split][coordinates_key] = database.splits[data_split][coordinates_key].to(torch.float32)
                    if not(force_key is None):
                        database.splits[data_split][force_key] = database.splits[data_split][force_key].to(torch.float32)
                training_modules.model.to(database.splits['train'][sys_energy.db_name].device)

            else:
                #Ensure that float arrays are float32
                for data_split in ["train","valid","test"]:
                    database.splits[data_split][energy_key] = (database.splits[data_split][energy_key]).to(torch.float32)
                    database.splits[data_split]["energyperatom"] = database.splits[data_split]["energyperatom"].to(torch.float32)
                    database.splits[data_split][coordinates_key] = database.splits[data_split][coordinates_key].to(torch.float32)
                    if not(force_key is None):
                        database.splits[data_split][force_key] = database.splits[data_split][force_key].to(torch.float32)

                training_modules.model.to(database.splits['train'][sys_energy.db_name].device)
                set_e0_values(henergy, database, peratom=False, energy_name=energy_key, decay_factor=1e-2)
            
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
                torch.save(unified,"lammps_model.pt")
                
            
def _ensemble_member_device(device_string, n_gpus):
    """Return the Torch device for this training process.

    ``'from_multiprocessing'`` picks a local device from the pool worker's number, leaving the
    scheduler's visibility mask alone. ``'cuda:N'`` is a device already chosen for this process.
    Any other value is a mask for this process, as it always was.
    """
    if device_string.lower() == "from_multiprocessing":
        member = multiprocessing.current_process()._identity[-1] - 1
        return excited_training_device(device_string, member, n_gpus)
    if device_string.lower().startswith('cuda:'):
        return device_string
    os.environ["CUDA_VISIBLE_DEVICES"] = device_string
    return 'cuda:0'


def train_HIPNN_model_wrapper(arg_dict):
    return(train_HIPNN_model(**arg_dict))


def _ground_state_member_params(ML_config, h5_dir, model_path, current_training_id, gpus_per_node):
    general_configuration = ML_config.copy()
    n_models = general_configuration.pop('n_models')
    params_list = [general_configuration.copy() for i in range(n_models)]
    for i,cur_dict in enumerate(params_list):
        cur_dict['model_dir'] = model_path.format(current_training_id) + '/model-{:02d}'.format(i)
        cur_dict['h5_train_dir'] = h5_dir
        cur_dict['from_multiprocessing_nGPU'] = gpus_per_node
    return params_list


def _ground_state_member_complete(model_dir):
    log = open(model_dir + '/training_log.txt','r').read()
    return len(re.findall('Training complete', log)) == 1


@python_app(executors=['alf_ML_executor'])
def train_HIPPYNN_ensemble_task(ML_config,h5_dir,model_path,current_training_id,gpus_per_node,remove_existing=False,h5_test_dir=None):
    p = multiprocessing.Pool(gpus_per_node)
    params_list = _ground_state_member_params(ML_config, h5_dir, model_path, current_training_id, gpus_per_node)
    out = p.map(train_HIPNN_model_wrapper,params_list)
    p.close()
    completed = [_ground_state_member_complete(params['model_dir']) for params in params_list]
    return(completed, current_training_id)




def HIPNN_ASE_calculator(HIPNN_model_directory,energy_key='energy',device="cuda:0"):
    from hippynn.experiment.serialization import load_checkpoint
    from hippynn.interfaces.ase_interface import HippynnCalculator
    import torch
    member_directory = os.path.abspath(HIPNN_model_directory)
    bundle = load_checkpoint(os.path.join(member_directory, 'experiment_structure.pt'),
                             os.path.join(member_directory, 'best_checkpoint.pt'), map_location="cpu")

    model = bundle["training_modules"].model
    energy_node = model.node_from_name(energy_key)
    calc = HippynnCalculator(energy_node, en_unit=units.eV)
    calc.to(torch.float32)
    calc.to(torch.device(device))
    return(calc)
    
def HIPNN_ASE_load_ensemble(HIPNN_ensemble_directory,device="cuda:0"):
    model_list = []
    # Sorted, so member order is deterministic and matches the batched loader.
    for cur_dir in sorted(glob.glob(HIPNN_ensemble_directory + '/model-*/')):
        model_list.append(HIPNN_ASE_calculator(cur_dir,device=device))
    return(model_list)


# ---------------------------------------------------------------------------------------------
# Batched ensemble inference for batched dynamics engines (sampler ``batched_calculator``).
#
# ``load_hippynn_batched_ensemble`` loads the ``model-*`` members written by
# ``train_HIPPYNN_ensemble_task`` and evaluates many structures of different sizes together:
# one padded forward pass per member. Evaluator contract (used by
# ``alframework.samplers.torchsim_engine.TorchSimEngine``):
#
#     energies, forces = evaluator.evaluate(numbers, positions, cells, pbc, counts)
#
# Inputs are packed (unpadded) torch tensors: ``numbers`` [atoms], ``positions`` [atoms, 3] in
# Angstrom, ``cells`` [systems, 3, 3] in ASE's row-vector convention, ``pbc`` [3] shared by the
# batch, and ``counts`` the atoms per system. Outputs are unbiased per-member ``energies``
# [models, systems] in eV and ``forces`` [models, atoms, 3] in eV/Angstrom, in input atom order.
# No UDD, well potential, or thresholds are applied.
#
# Molecular checkpoints evaluate nonperiodic systems. Fully periodic systems use the
# checkpoint's cell input if it was trained with one; otherwise its open-boundary pair finders
# are replaced by HIPPYNN periodic pair finders, as HIPPYNN's own ASE calculator does, so results
# match ``HIPNN_ASE_load_ensemble``. Mixed periodicity is not supported. Members are loaded from
# absolute paths, without changing the working directory.
# ---------------------------------------------------------------------------------------------

def load_hippynn_batched_ensemble(model_directory, device="cuda:0", energy_key="energy", force_key="forces"):
    """Load a HIPPYNN ensemble for batched evaluation (``batched_calculator`` loader)."""
    return HippynnBatchedEnsemble(model_directory, device=device, energy_key=energy_key, force_key=force_key)


def _load_member(member_directory, device):
    # Absolute paths: no process-wide chdir, so concurrent loads in one process are safe.
    from hippynn.experiment.serialization import load_saved_tensors
    structure, state = load_saved_tensors(os.path.join(member_directory, "experiment_structure.pt"),
                                          os.path.join(member_directory, "best_model.pt"), map_location=device)
    model = structure["training_modules"].model
    model.load_state_dict(state)
    return model


def _target(model, key):
    matches = [node for node in model.nodes_to_compute if node.db_name == key]
    if len(matches) != 1:
        raise ValueError(f"HIPPYNN checkpoint must have exactly one output with db_name {key!r}.")
    return matches[0]


def _member_graph(model, energy_key, force_key, periodic):
    """Return a GraphModule computing (energy, forces), made periodic if requested."""
    from hippynn.graphs import GraphModule, copy_subgraph, find_relatives, find_unique_relative, get_subgraph, \
        replace_node
    from hippynn.graphs.nodes.inputs import CellNode, PositionsNode, SpeciesNode
    from hippynn.graphs.nodes.pairs import OpenPairIndexer, PeriodicPairIndexer

    outputs = [_target(model, energy_key), _target(model, force_key)]
    trained_cell = any(isinstance(node, CellNode) for node in model.input_nodes)
    if trained_cell and not periodic:
        raise ValueError("This HIPPYNN checkpoint was trained with a cell and requires fully periodic systems.")
    if trained_cell or not periodic:
        return GraphModule(list(model.input_nodes), outputs)
    why = "Building a periodic batched HIPPYNN evaluator"
    open_pairs = [node for node in get_subgraph(outputs) if isinstance(node, OpenPairIndexer)]
    outputs, subgraph = copy_subgraph(outputs, assume_inputed=open_pairs)
    copied_pairs = find_relatives(outputs, lambda node: node in subgraph and isinstance(node, OpenPairIndexer),
                                  why_desc=why)
    species = find_unique_relative(outputs, lambda node: node in subgraph and isinstance(node, SpeciesNode),
                                   why_desc=why)
    positions = find_unique_relative(outputs, lambda node: node in subgraph and isinstance(node, PositionsNode),
                                     why_desc=why)
    cell = CellNode("cell", db_name="cell")
    for pairs in copied_pairs:
        replace_node(pairs, PeriodicPairIndexer(pairs.name, (positions, species, cell),
                                                dist_hard_max=pairs.dist_hard_max), disconnect_old=True)
    return GraphModule([species, positions, cell], outputs)


class HippynnBatchedEnsemble:
    """Padded batched evaluation of HIPPYNN ensemble members on one device.

    Members are evaluated one after another, in ``model-*`` directory order, each on the
    whole batch. (``hippynn.graphs.make_ensemble`` does not preserve member order.)
    """

    def __init__(self, model_directory, device="cuda:0", energy_key="energy", force_key="forces"):
        import torch
        import hippynn
        if getattr(torch.version, "hip", None) is not None:
            # CUDA-specific custom kernels can be importable in a ROCm process; use portable kernels.
            hippynn.custom_kernels.set_custom_kernels("pytorch")
        self.device = torch.device(device)
        self.energy_key, self.force_key = energy_key, force_key
        directories = sorted(glob.glob(os.path.join(str(model_directory), "model-*/")))
        if len(directories) < 2:
            raise ValueError(f"An uncertainty ensemble requires at least two model-* members in {model_directory}.")
        self._members = [_load_member(directory, self.device) for directory in directories]
        self.n_models = len(self._members)
        self._graphs = {}

    def _member_graphs(self, periodic):
        if periodic not in self._graphs:
            from hippynn.graphs.nodes.inputs import CellNode, PositionsNode, SpeciesNode
            graphs = []
            for model in self._members:
                graph = _member_graph(model, self.energy_key, self.force_key, periodic)
                graph.to(self.device)
                graph.eval()
                kinds = []
                for node in graph.input_nodes:
                    kind = next(name for cls, name in ((SpeciesNode, "species"), (PositionsNode, "positions"),
                                                       (CellNode, "cell")) if isinstance(node, cls))
                    kinds.append(kind)
                graphs.append((graph, kinds))
            self._graphs[periodic] = graphs
        return self._graphs[periodic]

    def evaluate(self, numbers, positions, cells, pbc, counts):
        import torch
        pbc = [bool(value) for value in pbc]
        if any(pbc) and not all(pbc):
            raise ValueError("Batched HIPPYNN inference supports fully periodic or nonperiodic systems only.")
        counts_tensor = torch.as_tensor(counts, device=self.device)
        n_systems, max_atoms = len(counts), int(max(counts))
        mask = torch.arange(max_atoms, device=self.device).unsqueeze(0) < counts_tensor.unsqueeze(1)
        species = torch.zeros((n_systems, max_atoms), dtype=torch.long, device=self.device)
        coordinates = torch.zeros((n_systems, max_atoms, 3), dtype=torch.float32, device=self.device)
        species[mask] = numbers.to(self.device, torch.long)
        coordinates[mask] = positions.detach().to(self.device, torch.float32)
        values = {"species": species, "cell": cells.detach().to(self.device, torch.float32)}
        energies, forces = [], []
        for graph, kinds in self._member_graphs(all(pbc)):
            # Forces are autograd derivatives with respect to the coordinates input.
            values["positions"] = coordinates.clone().requires_grad_(True)
            with torch.enable_grad():
                energy, force = graph(*(values[kind] for kind in kinds))
            energies.append(energy.detach().reshape(n_systems))
            forces.append(force.detach().reshape(n_systems, max_atoms, 3)[mask])
        return torch.stack(energies), torch.stack(forces)


def _member_atomic_numbers(directory):
    """Read one member's atoms from its small file, since the summary lists every training frame.

    Models trained before that file existed only have the summary.
    """
    composition = directory / 'member_composition.json'
    if composition.exists():
        return json.loads(composition.read_text())['atomic_numbers']
    return json.loads((directory / 'training_summary.json').read_text())['data']['atomic_numbers']


def load_hippynn_excited_state_ensemble(model_directory, device, properties_list, hippynn_kernel="pytorch"):
    """Load molecular state heads in physical eV and eV/Angstrom units."""
    return HippynnExcitedStateEnsemble(model_directory, device, properties_list, hippynn_kernel)


class HippynnExcitedStateEnsemble:
    """Ordered member/state/system predictions; packed forces keep caller atom order."""

    def __init__(self, model_directory, device, properties_list, hippynn_kernel="pytorch"):
        import torch
        import hippynn
        from hippynn.graphs import GraphModule
        from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode
        self.state_table = validate_excited_state_training_properties(properties_list)
        self.n_states = len(self.state_table)
        self.device = torch.device(device)
        hippynn.custom_kernels.set_custom_kernels(hippynn_kernel)
        self.hippynn_kernel = hippynn_kernel
        self._graphs = []
        self._composition = None
        self._scales = []
        for row in self.state_table:
            scales = [float(properties_list[row[key]][2]) if len(properties_list[row[key]]) > 2 else 1.
                      for key in ('energy_key', 'force_key')]
            if any(not math.isfinite(s) or s <= 0 for s in scales):
                raise ValueError('Excited-state inference requires finite positive property scales')
            self._scales.append(scales)
        directories = sorted(Path(model_directory).glob('model-*/'))
        if len(directories) < 2:
            raise ValueError('Excited-state uncertainty requires at least two model-* members')
        for directory in directories:
            composition = sorted(_member_atomic_numbers(directory))
            if self._composition is not None and self._composition != composition:
                raise ValueError('Ensemble members disagree on molecular composition')
            self._composition = composition
            model = _load_member(str(directory.resolve()), self.device)
            outputs = [_target(model, row[key]) for key in ('energy_db_name', 'force_db_name')
                       for row in self.state_table]
            kinds = []
            for node in model.input_nodes:
                if isinstance(node, SpeciesNode):
                    kinds.append('species')
                elif isinstance(node, PositionsNode):
                    kinds.append('positions')
                else:
                    raise ValueError('Excited-state inference supports nonperiodic species/positions graphs only')
            graph = GraphModule(model.input_nodes, outputs).to(self.device)
            graph.eval()
            self._graphs.append((graph, kinds))
        self.n_models = len(self._graphs)

    def evaluate_states(self, numbers, positions, cells, pbc, counts):
        """Return tensors E[M,K,S], F[M,K,atoms,3], without changing input order."""
        import torch
        if bool(torch.as_tensor(pbc).any()):
            raise ValueError('Excited-state inference supports nonperiodic molecules only')
        if not counts or any(int(n) != len(self._composition) for n in counts):
            raise ValueError('Excited-state inference requires the trained molecular composition')
        numbers = torch.as_tensor(numbers, device=self.device, dtype=torch.long)
        positions = torch.as_tensor(positions, device=self.device)
        if numbers.shape != (sum(counts),) or positions.shape != (sum(counts), 3):
            raise ValueError('Invalid packed species/positions shape')
        if not bool(torch.isfinite(positions).all()):
            raise ValueError('Nonfinite molecular positions')
        species = numbers.reshape(len(counts), counts[0])
        if any(sorted(row) != self._composition for row in species.cpu().tolist()):
            raise ValueError('Molecule differs from trained composition')
        energies, forces = [], []
        for graph, kinds in self._graphs:
            dtype = next(p.dtype for p in graph.parameters() if p.is_floating_point())
            values = dict(species=species, positions=positions.detach().to(dtype).reshape(len(counts), counts[0], 3)
                          .clone().requires_grad_(True))
            with torch.enable_grad():
                outputs = graph(*(values[k] for k in kinds))
            energy = torch.stack([outputs[k].reshape(len(counts)) / self._scales[k][0]
                                  for k in range(self.n_states)])
            force = torch.stack([outputs[self.n_states+k].reshape(sum(counts), 3) / self._scales[k][1]
                                 for k in range(self.n_states)])
            if not bool(torch.isfinite(energy).all() and torch.isfinite(force).all()):
                raise FloatingPointError('Nonfinite excited-state prediction')
            energies.append(energy.detach()); forces.append(force.detach())
        return torch.stack(energies), torch.stack(forces)

    def predict_states(self, atoms_list):
        """Return state_energies[M,K,S] and a list of state_forces[M,K,atoms_i,3]."""
        import torch
        from alframework.tools.atom_order import validated_atom_ids
        if not atoms_list:
            raise ValueError('State prediction requires at least one molecule')
        for atoms in atoms_list:
            if TOPOLOGY_ATOM_IDS in atoms.arrays:
                validated_atom_ids(atoms.arrays[TOPOLOGY_ATOM_IDS], len(atoms))
        counts = [len(a) for a in atoms_list]
        energies, forces = self.evaluate_states(
            torch.as_tensor(np.concatenate([a.numbers for a in atoms_list])),
            torch.as_tensor(np.concatenate([a.positions for a in atoms_list])),
            None, np.stack([a.pbc for a in atoms_list]), counts)
        offsets = np.cumsum([0, *counts])
        return dict(state_energies=energies.cpu().numpy(),
                    state_forces=[forces[:, :, start:end].cpu().numpy()
                                  for start, end in zip(offsets[:-1], offsets[1:])])


# Multi-state molecular training. Ground-state entry points above are unchanged.

def validate_excited_state_training_properties(
    properties_list: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Validate flattened state targets and their HDF5 storage schemas."""

    state_table = derive_state_property_table(
        dict(properties_list or {}),
        require_forces=True,
    )
    database_names: list[str] = []
    for row in state_table:
        energy_schema = properties_list[str(row["energy_key"])]
        force_schema = properties_list[str(row["force_key"])]
        if len(energy_schema) < 2 or str(energy_schema[1]).lower() != "system":
            raise ValueError(
                f"{row['energy_key']} must be a system property in "
                "properties_list."
            )
        if len(force_schema) < 2 or str(force_schema[1]).lower() != "atomic":
            raise ValueError(
                f"{row['force_key']} must be an atomic property in "
                "properties_list."
            )
        database_names.extend(
            [str(row["energy_db_name"]), str(row["force_db_name"])]
        )
    gap_table = derive_gap_property_table(properties_list)
    database_names.extend(str(row["gap_db_name"]) for row in gap_table)
    if any(not name for name in database_names):
        raise ValueError("Excited-state HDF5 database names cannot be empty.")
    if len(database_names) != len(set(database_names)):
        raise ValueError(
            "Every excited-state energy, force, and gap requires a unique "
            "HDF5 database name."
        )
    return state_table


def validate_excited_state_training_config(
    ML_config: dict[str, Any],
    properties_list: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Validate the supported multi-state training contract."""

    config = dict(ML_config or {})
    state_table = validate_excited_state_training_properties(properties_list)
    n_models = int(config.get("n_models", 0))
    if n_models < 1:
        raise ValueError("n_models must be a positive integer.")

    cell_key = config.get("cell_key")
    if cell_key not in {None, "None"}:
        raise ValueError(
            "Excited-state HIPPYNN training currently supports only "
            "nonperiodic data; cell_key must be null."
        )
    if not bool(config.get("train_forces", True)):
        raise ValueError(
            "Excited-state HIPPYNN training requires train_forces=true."
        )
    if not bool(config.get("export_force_gradients", True)):
        raise ValueError(
            "Excited-state checkpoints require export_force_gradients=true "
            "so every F# node is available to sampling."
        )

    gap_config = config.get("gap_targets")
    if gap_config is not None and not isinstance(gap_config, dict):
        raise TypeError("gap_targets must be a dictionary when provided.")
    gap_options = dict(gap_config or {})
    gap_enabled = bool(gap_options.get("enabled", False))
    gap_mode = str(gap_options.get("mode", "derived")).strip().lower()
    if gap_mode != "derived":
        raise ValueError(
            "gap_targets.mode must be 'derived'; independent gap heads are "
            "not supported."
        )
    gap_weight = float(gap_options.get("weight", 1.0))
    if not np.isfinite(gap_weight) or gap_weight < 0:
        raise ValueError("gap_targets.weight must be finite and nonnegative.")
    configured_gaps = derive_gap_property_table(
        properties_list,
        gap_config=gap_options,
        require_properties=gap_enabled,
    )
    if gap_enabled and not configured_gaps:
        raise ValueError(
            "Enabled gap targets require at least one dE# property in "
            "properties_list."
        )

    exports = config.get("exports")
    if exports is not None:
        if not isinstance(exports, dict):
            raise TypeError("exports must be a dictionary when provided.")
        if bool(exports.get("csv", False)):
            raise ValueError(
                "CSV prediction export is not supported by the isolated "
                "excited-state trainer; PDF and PNG plots are supported."
            )

    network_choice = int(config.get("network_choice", 1))
    if network_choice not in {0, 1}:
        raise ValueError(
            "network_choice must be 0 (HipNN) or 1 (HipHopNN)."
        )
    if str(config.get("optimizer", "AdamW")).lower() != "adamw":
        raise ValueError(
            "The isolated excited-state trainer currently supports "
            "optimizer='AdamW' only."
        )
    if config.get('hippynn_kernel', 'pytorch') not in {'pytorch', 'auto', 'numba', 'cupy', 'triton', 'sparse'}:
        raise ValueError('hippynn_kernel must name a supported HIPPYNN kernel implementation')
    if config.get("n_atoms") is not None and int(config["n_atoms"]) < 1:
        raise ValueError("n_atoms must be positive when configured.")

    for key in (
        "batch_size",
        "eval_batch_size",
        "max_epochs",
        "termination_patience",
        "plot_frequency",
    ):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be a positive integer.")

    valid_size = float(config.get("valid_size", 0.1))
    test_size = float(config.get("test_size", 0.1))
    if (
        not np.isfinite(valid_size)
        or not np.isfinite(test_size)
        or valid_size <= 0
        or test_size <= 0
        or valid_size + test_size >= 1
    ):
        raise ValueError(
            "valid_size and test_size must be finite, positive, and sum to "
            "less than one."
        )

    for key in (
        "energy_weight",
        "force_weight",
        "l2_weight",
        "learning_rate",
    ):
        value = float(config.get(key, 1.0 if key != "l2_weight" else 2.0e-5))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if float(config.get("learning_rate", 5.0e-4)) <= 0:
        raise ValueError("learning_rate must be positive.")
    for key in (
        "remove_high_energy_cut",
        "remove_high_energy_std",
        "remove_high_forces_cut",
        "remove_high_forces_std",
    ):
        if config.get(key) is None:
            continue
        value = float(config[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative or null.")

    return state_table


def _h5_shards(h5_train_dir: str) -> list[str]:
    root = Path(h5_train_dir).expanduser()
    if root.is_file():
        paths = [str(root)]
    else:
        paths = sorted(
            set(
                glob.glob(str(root / "*.h5"))
                + glob.glob(str(root / "*.hdf5"))
            )
        )
    if not paths:
        raise FileNotFoundError(
            f"No ALF HDF5 shards were found in {str(root)!r}."
        )
    return paths


def load_excited_state_h5_arrays(h5_train_dir, properties_list, *, coordinates_key='coordinates',
                                species_key='species', configured_n_atoms=None,
                                configured_possible_species=None, return_database=False, seed=0):
    """Load explicit multi-state schemas through the shared restartable reader.

    HDF5 values already include properties_list unit conversions; do not apply
    those conversions again. Identity-aware reading is opt-in to this path.
    """
    states = validate_excited_state_training_properties(properties_list)
    gaps = derive_gap_property_table(properties_list)
    paths = _h5_shards(h5_train_dir)
    targets = [row[key] for row in states for key in ('energy_db_name', 'force_db_name')]
    targets += [row['gap_db_name'] for row in gaps]
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB
    database = ALFPyAniDirectoryDB(
        str(Path(paths[0]).parent), files=[str(Path(p).resolve()) for p in paths],
        inputs=[species_key, coordinates_key], targets=targets,
        coordinates_key=coordinates_key, species_key=species_key,
        atom_keys=[row['force_db_name'] for row in states],
        excited_state_properties=properties_list, seed=seed, quiet=True,
    )
    arrays = {key: value.numpy() for key, value in database.arr_dict.items()}
    species = arrays[species_key]
    if not np.all(species == species[0]):
        raise ValueError('Excited-state training requires one exact atomic-number sequence')
    if np.any(species <= 0):
        raise ValueError('Species must be positive atomic numbers')
    n_atoms = species.shape[1]
    if configured_n_atoms is not None and int(configured_n_atoms) != n_atoms:
        raise ValueError(f'Configured n_atoms={configured_n_atoms} does not match HDF5 atom count {n_atoms}')
    possible = list(configured_possible_species) if configured_possible_species is not None else [0, *sorted(set(species[0].tolist()))]
    if not possible or possible[0] != 0 or len(possible) != len(set(possible)) or min(possible) < 0:
        raise ValueError('possible_species must begin with padding species 0 and contain unique nonnegative integers')
    if not set(species[0]).issubset(possible):
        raise ValueError('possible_species is missing atomic numbers present in HDF5')
    summary = dict(n_structures=len(species), n_atoms=n_atoms, atomic_numbers=species[0].tolist(),
                   possible_species=possible, state_table=states, gap_table=gaps,
                   frames=database.alf_frame_manifest,
                   files=[dict(path=str(Path(p).resolve()), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in paths])
    if return_database:
        return arrays, summary, database
    # Keep bookkeeping indices outside the public numerical target contract.
    return {key: value for key, value in arrays.items() if key != 'indices'}, summary


def excited_training_device(device_string, model_id, gpus_per_node):
    """Select visible local ordinals without rewriting scheduler masks."""
    if str(device_string).lower() == 'cpu':
        return 'cpu'
    import torch
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError('No GPU is available within the training worker visibility')
    requested = str(device_string)
    if requested in {'from_multiprocessing', 'cuda', 'None'}:
        width = min(count, int(gpus_per_node)) if gpus_per_node else count
        if width < 1:
            raise ValueError('Training GPU count must be positive')
        requested = f'cuda:{int(model_id) % width}'
    from alframework.tools.gpu import _cuda_device
    return _cuda_device(requested, count, None, True)


def _minimum_structure_distances(coordinates: np.ndarray) -> np.ndarray:
    """Return the shortest nonperiodic pair distance in every structure."""

    count, atom_count, _ = coordinates.shape
    if atom_count < 2:
        return np.full(count, np.inf, dtype=np.float64)
    minima = np.full(count, np.inf, dtype=np.float64)
    for first in range(atom_count - 1):
        distances = np.linalg.norm(
            coordinates[:, first + 1 :, :]
            - coordinates[:, first : first + 1, :],
            axis=-1,
        )
        minima = np.minimum(minima, np.min(distances, axis=1))
    return minima


def _property_outlier_mask(
    values: np.ndarray,
    active: np.ndarray,
    threshold: float | None,
    *,
    standardized: bool,
) -> np.ndarray:
    """Apply production HIPPYNN's centered componentwise outlier formula."""

    failures = np.zeros(active.shape, dtype=bool)
    if threshold is None or not np.any(active):
        return failures
    selected = np.asarray(values, dtype=np.float64)[active]
    mean = float(np.mean(selected))
    if standardized:
        flattened = selected.reshape(-1)
        if flattened.size < 2:
            return failures
        deviation_scale = float(np.std(flattened, ddof=1))
        if not np.isfinite(deviation_scale) or deviation_scale == 0.0:
            return failures
        element_failures = (
            np.abs(selected - mean) / deviation_scale > float(threshold)
        )
    else:
        element_failures = np.abs(selected - mean) > float(threshold)
    structure_failures = np.any(
        element_failures.reshape(selected.shape[0], -1),
        axis=1,
    )
    failures[np.flatnonzero(active)] = structure_failures
    return failures


def _mask_training_arrays(
    arrays: dict[str, np.ndarray],
    keep: np.ndarray,
) -> dict[str, np.ndarray]:
    count = int(keep.shape[0])
    filtered: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.shape[0] != count:
            raise ValueError(
                f"Training array {key!r} has {array.shape[0]} structures; "
                f"expected {count}."
            )
        filtered[key] = array[keep]
    return filtered


def filter_excited_state_training_arrays(
    arrays: dict[str, np.ndarray],
    state_table: list[dict[str, Any]],
    *,
    coordinates_key: str,
    dist_soft_min: float,
    remove_high_energy_cut: float | None = None,
    remove_high_energy_std: float | None = None,
    remove_high_forces_cut: float | None = None,
    remove_high_forces_std: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Filter fixed-composition multi-state arrays before data splitting.

    Multi-state masks are computed from a common survivor pool and unioned,
    making the result independent of state order. The one-state path preserves
    production's force-then-energy cut/std sequence exactly.
    """

    coordinates = np.asarray(arrays[coordinates_key], dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"{coordinates_key} must have shape [structures, atoms, 3]."
        )
    count = int(coordinates.shape[0])
    minimum_distance = float(dist_soft_min)
    if not np.isfinite(minimum_distance) or minimum_distance < 0:
        raise ValueError("network_params.dist_soft_min must be nonnegative.")
    options = {
        "remove_high_energy_cut": remove_high_energy_cut,
        "remove_high_energy_std": remove_high_energy_std,
        "remove_high_forces_cut": remove_high_forces_cut,
        "remove_high_forces_std": remove_high_forces_std,
    }
    for key, value in options.items():
        if value is not None and (
            not np.isfinite(float(value)) or float(value) < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative or null.")

    active = _minimum_structure_distances(coordinates) > minimum_distance
    summary: dict[str, Any] = {
        "initial_structures": count,
        "dist_soft_min": minimum_distance,
        "removed_min_distance": int(count - np.count_nonzero(active)),
        "static_cut_failures": {},
        "standard_deviation_failures": {},
    }
    energy_names = [
        str(row["energy_db_name"]) for row in state_table
    ]
    force_names = [
        str(row["force_db_name"]) for row in state_table
    ]
    removed_static = 0
    removed_standardized = 0

    def apply_one(
        property_name: str,
        threshold: float | None,
        *,
        standardized: bool,
        summary_key: str,
    ) -> None:
        nonlocal active, removed_static, removed_standardized
        failures = _property_outlier_mask(
            arrays[property_name],
            active,
            threshold,
            standardized=standardized,
        )
        failure_count = int(np.count_nonzero(failures))
        summary[summary_key][property_name] = failure_count
        if standardized:
            removed_standardized += failure_count
        else:
            removed_static += failure_count
        active &= ~failures

    if len(state_table) == 1:
        # Production ground-state HIPPYNN filters force cut/std first, then
        # energy cut/std, recomputing statistics after every removal.
        apply_one(
            force_names[0],
            remove_high_forces_cut,
            standardized=False,
            summary_key="static_cut_failures",
        )
        apply_one(
            force_names[0],
            remove_high_forces_std,
            standardized=True,
            summary_key="standard_deviation_failures",
        )
        apply_one(
            energy_names[0],
            remove_high_energy_cut,
            standardized=False,
            summary_key="static_cut_failures",
        )
        apply_one(
            energy_names[0],
            remove_high_energy_std,
            standardized=True,
            summary_key="standard_deviation_failures",
        )
    else:
        static_failures = np.zeros(count, dtype=bool)
        for property_name, threshold in (
            *(
                (name, remove_high_forces_cut)
                for name in force_names
            ),
            *(
                (name, remove_high_energy_cut)
                for name in energy_names
            ),
        ):
            failures = _property_outlier_mask(
                arrays[property_name],
                active,
                threshold,
                standardized=False,
            )
            summary["static_cut_failures"][property_name] = int(
                np.count_nonzero(failures)
            )
            static_failures |= failures
        removed_static = int(
            np.count_nonzero(static_failures & active)
        )
        active &= ~static_failures

        standard_failures = np.zeros(count, dtype=bool)
        for property_name, threshold in (
            *(
                (name, remove_high_forces_std)
                for name in force_names
            ),
            *(
                (name, remove_high_energy_std)
                for name in energy_names
            ),
        ):
            failures = _property_outlier_mask(
                arrays[property_name],
                active,
                threshold,
                standardized=True,
            )
            summary["standard_deviation_failures"][property_name] = int(
                np.count_nonzero(failures)
            )
            standard_failures |= failures
        removed_standardized = int(
            np.count_nonzero(standard_failures & active)
        )
        active &= ~standard_failures

    summary["removed_static_cut_union"] = int(removed_static)
    summary["removed_standard_deviation_union"] = int(
        removed_standardized
    )
    summary["final_structures"] = int(np.count_nonzero(active))
    summary["per_state_failures"] = {
        str(int(row["state"])): {
            "energy": {
                "static_cut": int(
                    summary["static_cut_failures"].get(
                        str(row["energy_db_name"]),
                        0,
                    )
                ),
                "standard_deviation": int(
                    summary["standard_deviation_failures"].get(
                        str(row["energy_db_name"]),
                        0,
                    )
                ),
            },
            "forces": {
                "static_cut": int(
                    summary["static_cut_failures"].get(
                        str(row["force_db_name"]),
                        0,
                    )
                ),
                "standard_deviation": int(
                    summary["standard_deviation_failures"].get(
                        str(row["force_db_name"]),
                        0,
                    )
                ),
            },
        }
        for row in state_table
    }
    filtered = _mask_training_arrays(arrays, active)
    return filtered, summary


def validate_filtered_split_capacity(
    structure_count: int,
    *,
    test_size: float,
    valid_size: float,
) -> None:
    """Fail before HIPPYNN attempts to create an empty data split."""

    count = int(structure_count)
    test_count = int(float(test_size) * count)
    remaining = count - test_count
    adjusted_valid = float(valid_size) / (1.0 - float(test_size))
    valid_count = int(adjusted_valid * remaining)
    train_count = remaining - valid_count
    if min(test_count, valid_count, train_count) < 1:
        raise ValueError(
            "Excited-state filtering left too few structures for nonempty "
            f"train/valid/test splits: {count} survivors produce "
            f"train={train_count}, valid={valid_count}, test={test_count}."
        )


def compose_excited_state_loss(
    energy_error_terms: list[tuple[Any, Any]],
    force_error_terms: list[tuple[Any, Any]],
    l2_term: Any,
    *,
    gap_error_terms: list[tuple[Any, Any]] | None = None,
    n_atoms: int,
    energy_weight: float,
    force_weight: float,
    gap_weight: float = 1.0,
    l2_weight: float,
) -> Any:
    """Compose the fork-compatible multi-state E/F/gap/L2 objective."""

    if not energy_error_terms:
        raise ValueError("At least one state-energy loss term is required.")
    atom_count = int(n_atoms)
    if atom_count < 1:
        raise ValueError("n_atoms must be positive.")
    total = sum(
        float(energy_weight) * (rmse + mae)
        for rmse, mae in energy_error_terms
    )
    force_normalizer = math.sqrt(3.0 * float(atom_count))
    total = total + sum(
        float(force_weight) * (rmse + mae) / force_normalizer
        for rmse, mae in force_error_terms
    )
    total = total + sum(
        float(gap_weight) * (rmse + mae)
        for rmse, mae in (gap_error_terms or [])
    )
    return total + float(l2_weight) * l2_term


def _build_shared_network(
    species_node: Any,
    positions_node: Any,
    *,
    network_choice: int,
    network_params: dict[str, Any],
) -> Any:
    from hippynn.graphs import networks

    parents = (species_node, positions_node)
    if int(network_choice) == 0:
        return networks.Hipnn(
            "excited_state_HipNN",
            parents,
            module_kwargs=dict(network_params),
            periodic=False,
        )
    if int(network_choice) == 1:
        return networks.HipHopnn(
            "excited_state_HipHopNN",
            parents,
            module_kwargs=dict(network_params),
            periodic=False,
        )
    raise ValueError("network_choice must be 0 (HipNN) or 1 (HipHopNN).")


def build_excited_state_training_graph(
    network: Any,
    positions_node: Any,
    state_table: list[dict[str, Any]],
    *,
    gap_table: list[dict[str, Any]] | None = None,
    n_atoms: int,
    energy_weight: float,
    force_weight: float,
    gap_weight: float = 1.0,
    l2_weight: float,
    first_is_interacting: bool = False,
) -> tuple[
    list[tuple[dict[str, Any], Any]],
    list[tuple[dict[str, Any], Any]],
    Any,
    dict[str, Any],
]:
    """Attach state heads/gradients and construct the shared training loss."""

    from hippynn.graphs import loss, physics, targets

    energy_outputs: list[tuple[dict[str, Any], Any]] = []
    force_outputs: list[tuple[dict[str, Any], Any]] = []
    gap_outputs: list[tuple[dict[str, Any], Any]] = []
    energy_error_terms: list[tuple[Any, Any]] = []
    force_error_terms: list[tuple[Any, Any]] = []
    gap_error_terms: list[tuple[Any, Any]] = []
    validation_losses: dict[str, Any] = {}
    for row in state_table:
        state = int(row["state"])
        energy_head = targets.HEnergyNode(
            f"state_{state}_energy_head",
            network,
            first_is_interacting=bool(first_is_interacting),
        )
        energy_output = energy_head.mol_energy
        energy_output.db_name = str(row["energy_db_name"])
        force_output = physics.GradientNode(
            f"state_{state}_force_gradient",
            (energy_output, positions_node),
            sign=-1,
            db_name=str(row["force_db_name"]),
        )
        energy_outputs.append((row, energy_output))
        force_outputs.append((row, force_output))

        energy_rmse = loss.MSELoss.of_node(energy_output) ** 0.5
        energy_mae = loss.MAELoss.of_node(energy_output)
        force_rmse = loss.MSELoss.of_node(force_output) ** 0.5
        force_mae = loss.MAELoss.of_node(force_output)
        energy_error_terms.append((energy_rmse, energy_mae))
        force_error_terms.append((force_rmse, force_mae))
        validation_losses[f"sE{state}_RMSE"] = energy_rmse
        validation_losses[f"sE{state}_MAE"] = energy_mae
        validation_losses[f"F{state}_RMSE"] = force_rmse
        validation_losses[f"F{state}_MAE"] = force_mae

    energy_by_state = {
        int(row["state"]): output for row, output in energy_outputs
    }
    for row in gap_table or []:
        gap_key = str(row["gap_key"])
        gap_output = (
            energy_by_state[int(row["upper_state"])]
            - energy_by_state[int(row["lower_state"])]
        )
        gap_output.name = f"derived_{gap_key}"
        gap_output.db_name = str(row["gap_db_name"])
        gap_outputs.append((row, gap_output))
        gap_rmse = loss.MSELoss.of_node(gap_output) ** 0.5
        gap_mae = loss.MAELoss.of_node(gap_output)
        gap_error_terms.append((gap_rmse, gap_mae))
        validation_losses[f"{gap_key}_RMSE"] = gap_rmse
        validation_losses[f"{gap_key}_MAE"] = gap_mae

    l2_term = loss.l2reg(network)
    total_loss = compose_excited_state_loss(
        energy_error_terms,
        force_error_terms,
        l2_term,
        gap_error_terms=gap_error_terms,
        n_atoms=n_atoms,
        energy_weight=energy_weight,
        force_weight=force_weight,
        gap_weight=gap_weight,
        l2_weight=l2_weight,
    )
    validation_losses["L2"] = l2_term
    validation_losses["Loss"] = total_loss
    return (
        energy_outputs,
        force_outputs,
        total_loss,
        validation_losses,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def model_member_seed(ML_config: dict[str, Any], model_id: int) -> int:
    """Return the deterministic seed assigned to an ensemble member."""

    return int(ML_config.get("random_seed", 114514)) + int(model_id)


def train_single_excited_state_model(
    *,
    model_id: int,
    model_dir: str,
    h5_train_dir: str,
    properties_list: dict[str, list[Any]],
    ML_config: dict[str, Any],
    from_multiprocessing_nGPU: int | None = None,
) -> dict[str, Any]:
    """Train one shared-trunk multi-state HIPPYNN model member."""

    config = dict(ML_config or {})
    state_table = validate_excited_state_training_config(
        config,
        properties_list,
    )
    gap_options = dict(config.get("gap_targets") or {})
    gap_table = (
        derive_gap_property_table(
            properties_list,
            gap_config=gap_options,
            require_properties=True,
        )
        if bool(gap_options.get("enabled", False))
        else []
    )
    selected_device = excited_training_device(config.get('device_string', 'from_multiprocessing'),
                                                model_id, from_multiprocessing_nGPU)
    recorded_device = {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}

    # Torch and HIPPYNN must be imported only after the child selects its GPU.
    import torch
    import hippynn
    from hippynn import plotting
    from hippynn.experiment import SetupParams, setup_training, train_model
    from hippynn.experiment.controllers import (
        PatienceController,
        RaiseBatchSizeOnPlateau,
    )
    from hippynn.graphs import inputs

    # Torch kernels execute on the selected CPU/GPU and support force
    # derivatives without HIPPYNN's optional accelerated-kernel adapters.
    # Other implementations remain an explicit choice; never silently retry.
    active_kernel = hippynn.custom_kernels.set_custom_kernels(config.get('hippynn_kernel', 'pytorch'))

    device = torch.device(selected_device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(int(config.get('torch_threads', 1)))
    seed = model_member_seed(config, model_id)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_default_dtype(torch.float32)
    hippynn.settings.WARN_LOW_DISTANCES = False
    hippynn.settings.TRANSPARENT_PLOT = True

    coordinates_key = str(config.get("coordinates_key", "coordinates"))
    species_key = str(config.get("species_key", "species"))
    network_params = dict(config.get("network_params") or {})
    configured_n_atoms = config.get("n_atoms")
    configured_species = network_params.get("possible_species")
    arrays, data_summary, database = load_excited_state_h5_arrays(
        h5_train_dir,
        properties_list,
        coordinates_key=coordinates_key,
        species_key=species_key,
        configured_n_atoms=(
            None if configured_n_atoms is None else int(configured_n_atoms)
        ),
        configured_possible_species=configured_species, return_database=True, seed=seed,
    )
    if "dist_soft_min" not in network_params:
        raise ValueError(
            "network_params.dist_soft_min is required for production-style "
            "excited-state training filtering."
        )
    arrays, filtering_summary = filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key=coordinates_key,
        dist_soft_min=float(network_params["dist_soft_min"]),
        remove_high_energy_cut=config.get("remove_high_energy_cut"),
        remove_high_energy_std=config.get("remove_high_energy_std"),
        remove_high_forces_cut=config.get("remove_high_forces_cut"),
        remove_high_forces_std=config.get("remove_high_forces_std"),
    )
    validate_filtered_split_capacity(
        int(arrays[coordinates_key].shape[0]),
        test_size=float(config.get("test_size", 0.1)),
        valid_size=float(config.get("valid_size", 0.1)),
    )
    data_summary["n_structures_before_filtering"] = int(
        data_summary["n_structures"]
    )
    data_summary["n_structures"] = int(
        arrays[coordinates_key].shape[0]
    )
    data_summary["filtering"] = filtering_summary
    network_params["possible_species"] = data_summary["possible_species"]
    n_atoms = int(data_summary["n_atoms"])

    model_root = Path(model_dir).expanduser().resolve()
    if model_root.exists() and any(model_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty model directory {model_root}."
        )
    model_root.mkdir(parents=True, exist_ok=True)

    with hippynn.tools.active_directory(str(model_root)):
        with hippynn.tools.log_terminal("training_log.txt", "wt"):
            print(f"Model ID: {int(model_id)}")
            print(f"Model seed: {seed}")
            print(f"CUDA_VISIBLE_DEVICES: {recorded_device}")
            print(f"Training device: {device}")
            print("Training filters:")
            print(json.dumps(filtering_summary, indent=2))

            species_node = inputs.SpeciesNode(db_name=species_key)
            positions_node = inputs.PositionsNode(db_name=coordinates_key)
            positions_node.requires_grad = True
            network = _build_shared_network(
                species_node,
                positions_node,
                network_choice=int(config.get("network_choice", 1)),
                network_params=network_params,
            )

            plotters: list[Any] = []
            exports = dict(config.get("exports") or {})
            export_pdf = bool(exports.get("pdf", True))
            export_png = bool(exports.get("png", False))
            (
                energy_outputs,
                force_outputs,
                total_loss,
                validation_losses,
            ) = build_excited_state_training_graph(
                network,
                positions_node,
                state_table,
                gap_table=gap_table,
                n_atoms=n_atoms,
                energy_weight=float(config.get("energy_weight", 1.0)),
                force_weight=float(config.get("force_weight", 1.0)),
                gap_weight=float(gap_options.get("weight", 1.0)),
                l2_weight=float(config.get("l2_weight", 2.0e-5)),
                first_is_interacting=bool(
                    config.get("first_is_interacting", False)
                ),
            )

            plot_dir = model_root / str(
                exports.get("output_subdir", "plots")
            )
            if export_pdf or export_png:
                plot_dir.mkdir(parents=True, exist_ok=True)
            for row, output in energy_outputs:
                if export_pdf:
                    plotters.append(
                        plotting.Hist2D.compare(
                            output,
                            saved=str(
                                plot_dir / f"{str(row['energy_key'])}.pdf"
                            ),
                            shown=False,
                        )
                    )
                if export_png:
                    plotters.append(
                        plotting.Hist2D.compare(
                            output,
                            saved=str(
                                plot_dir / f"{str(row['energy_key'])}.png"
                            ),
                            shown=False,
                        )
                    )
            for row, output in force_outputs:
                if export_pdf:
                    plotters.append(
                        plotting.Hist2D.compare(
                            output,
                            saved=str(
                                plot_dir / f"{str(row['force_key'])}.pdf"
                            ),
                            shown=False,
                        )
                    )
                if export_png:
                    plotters.append(
                        plotting.Hist2D.compare(
                            output,
                            saved=str(
                                plot_dir / f"{str(row['force_key'])}.png"
                            ),
                            shown=False,
                        )
                    )
            plot_maker = (
                plotting.PlotMaker(
                    *plotters,
                    plot_every=int(config.get("plot_frequency", 50)),
                )
                if plotters
                else None
            )

            training_modules, db_info = hippynn.experiment.assemble_for_training(
                total_loss,
                validation_losses,
                plot_maker=plot_maker,
            )
            database.inputs, database.targets = db_info['inputs'], db_info['targets']
            database.arr_dict = {key: torch.as_tensor(value) for key, value in arrays.items()}
            database.make_trainvalidtest_split(test_size=float(config.get('test_size', .1)),
                                               valid_size=float(config.get('valid_size', .1)))
            retained = arrays['indices'].tolist()
            database.restarter.args = (database.directory, database.inputs, database.targets)
            database.restarter.kwargs.update(retained_indices=retained,
                                             test_size=float(config.get('test_size', .1)),
                                             valid_size=float(config.get('valid_size', .1)))
            split_indices = {name: split['indices'].tolist() for name, split in database.splits.items()}
            fraction = float((config.get('controller_options') or {}).get('fraction_train_eval',
                                                                         config.get('fraction_train_eval', .1)))
            if not 0 < fraction <= 1 or int(len(split_indices['train']) * fraction) < 1:
                raise ValueError('fraction_train_eval selects no training frames; increase it for this small dataset')
            manifest = dict(files=data_summary['files'], frames=data_summary['frames'],
                            retained_indices=retained, splits=split_indices,
                            filtered_indices=sorted(set(range(data_summary['n_structures_before_filtering'])) - set(retained)))
            Path('training_input_manifest.json').write_text(json.dumps(manifest, indent=2))
            initial_parameters = [p.detach().clone() for p in training_modules.model.parameters()]

            optimizer = torch.optim.AdamW(
                training_modules.model.parameters(),
                lr=float(config.get("learning_rate", 5.0e-4)),
            )
            scheduler = RaiseBatchSizeOnPlateau(
                optimizer=optimizer,
                **dict(
                    config.get("scheduler_options")
                    or {
                        "max_batch_size": 128,
                        "patience": 15,
                        "factor": 0.5,
                    }
                ),
            )
            controller_options = dict(config.get("controller_options") or {})
            controller_options.setdefault(
                "batch_size",
                int(config.get("batch_size", 64)),
            )
            controller_options.setdefault(
                "eval_batch_size",
                int(config.get("eval_batch_size", 128)),
            )
            controller_options.setdefault(
                "max_epochs",
                int(config.get("max_epochs", 1000)),
            )
            controller_options.setdefault(
                "termination_patience",
                int(config.get("termination_patience", 55)),
            )
            controller_options.setdefault(
                "fraction_train_eval",
                float(config.get("fraction_train_eval", 0.1)),
            )
            controller = PatienceController(
                optimizer=optimizer,
                scheduler=scheduler,
                stopping_key="Loss",
                **controller_options,
            )
            setup_params = SetupParams(
                controller=controller,
                device=device,
            )
            training_modules, controller, metric_tracker = setup_training(
                training_modules=training_modules,
                setup_params=setup_params,
            )
            metric_tracker = train_model(
                training_modules,
                database,
                controller,
                metric_tracker,
                callbacks=None,
                batch_callbacks=None,
            )
            if not all(torch.isfinite(p).all() for p in training_modules.model.parameters()):
                raise RuntimeError('Training produced nonfinite model parameters')

            summary = {
                "model_id": int(model_id),
                "seed": seed,
                "device": str(device),
                "hippynn_kernel": active_kernel,
                "cuda_visible_devices": recorded_device,
                "state_table": state_table,
                "gap_table": gap_table,
                "gap_targets": {
                    "enabled": bool(gap_table),
                    "weight": float(gap_options.get("weight", 1.0)),
                    "mode": "derived",
                },
                "data": data_summary,
                "metric": _json_safe(metric_tracker.best_metric_values),
                "epochs": int(controller.current_epoch),
                "parameters_changed": any(not torch.equal(before.cpu(), after.detach().cpu())
                                          for before, after in zip(initial_parameters, training_modules.model.parameters())),
                "split_counts": {key: len(values) for key, values in split_indices.items()},
                "average_epoch_time": float(
                    np.average(metric_tracker.epoch_times)
                ),
            }
            Path("training_summary.json").write_text(
                json.dumps(_json_safe(summary), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            Path("member_composition.json").write_text(
                json.dumps({"atomic_numbers": data_summary["atomic_numbers"]}) + "\n",
                encoding="utf-8",
            )

    return {
        "model_id": int(model_id),
        "model_dir": str(model_root),
        "device": recorded_device,
        "seed": seed,
    }


def _excited_state_training_worker(
    arg_dict: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = train_single_excited_state_model(**arg_dict)
        return {"ok": True, "payload": payload}
    except Exception as exc:
        return {
            "ok": False,
            "payload": {
                "model_id": int(arg_dict["model_id"]),
                "model_dir": str(arg_dict["model_dir"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def _training_complete(model_dir: str) -> bool:
    log_path = Path(model_dir) / "training_log.txt"
    return (
        log_path.exists()
        and "Training complete" in log_path.read_text(encoding="utf-8")
    )


def _validate_completed_member(directory, state_table, gap_table=None):
    """Reload one member using an absolute checkpoint path, without changing cwd."""
    keys = [row[key] for row in state_table for key in ('energy_db_name', 'force_db_name')]
    keys += [row['gap_db_name'] for row in (gap_table or [])]
    model = _load_member(Path(directory).resolve(), 'cpu')
    for key in keys:
        _target(model, key)


def _validate_completed_ensemble(ensemble_dir, state_table, gap_table=None):
    members = sorted(Path(ensemble_dir).glob('model-*'))
    if not members:
        raise RuntimeError('No completed HIPPYNN model members were found')
    for directory in members:
        _validate_completed_member(directory, state_table, gap_table)


def _excited_state_gap_table(config, properties_list):
    gap_options = dict(config.get("gap_targets") or {})
    return (
        derive_gap_property_table(
            properties_list,
            gap_config=gap_options,
            require_properties=True,
        )
        if bool(gap_options.get("enabled", False))
        else []
    )


def _excited_state_member_params(config, h5_dir, ensemble_dir, gpus_per_node, properties_list):
    return [
        {
            "model_id": model_id,
            "model_dir": os.path.join(
                ensemble_dir,
                f"model-{model_id:02d}",
            ),
            "h5_train_dir": str(h5_dir),
            "properties_list": dict(properties_list),
            "ML_config": config,
            "from_multiprocessing_nGPU": (
                int(gpus_per_node) if int(gpus_per_node) > 0 else None
            ),
        }
        for model_id in range(int(config["n_models"]))
    ]


def _excited_state_member_complete(result: dict[str, Any]) -> bool:
    """Judge one member from its worker's result, and write down why if it failed."""
    payload = result["payload"]
    model_dir = Path(payload["model_dir"]).expanduser().resolve()
    if bool(result["ok"]):
        member_complete = _training_complete(str(model_dir))
        if not member_complete:
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "training_error.json").write_text(
                json.dumps(
                    {
                        "model_id": int(payload["model_id"]),
                        "model_dir": str(model_dir),
                        "error_type": "IncompleteTrainingError",
                        "error": (
                            "HIPPYNN returned without a single "
                            "'Training complete' marker."
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return member_complete
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_error.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    return False


def _worker_count(ML_config, *, n_models, gpus_per_node):
    if str(ML_config.get('device_string', 'from_multiprocessing')).lower() == 'cpu':
        return 1
    import torch
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError('No GPU is available within the training worker visibility')
    width = min(count, gpus_per_node) if gpus_per_node > 0 else count
    if str(ML_config.get('device_string', '')).startswith('cuda:'):
        return 1
    return min(width, n_models)


def train_excited_state_ensemble(
    *,
    ML_config: dict[str, Any],
    h5_dir: str,
    model_path: str,
    current_training_id: int,
    gpus_per_node: int,
    properties_list: dict[str, list[Any]],
) -> tuple[list[bool], int]:
    """Train all model members and preserve ALF's promotion contract."""

    config = dict(ML_config or {})
    state_table = validate_excited_state_training_config(
        config,
        properties_list,
    )
    gap_table = _excited_state_gap_table(config, properties_list)
    n_models = int(config["n_models"])
    ensemble_dir = model_path.format(int(current_training_id))
    params_list = _excited_state_member_params(config, h5_dir, ensemble_dir, gpus_per_node, properties_list)

    root = Path(ensemble_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty ensemble directory {root}')
    context = multiprocessing.get_context('spawn')
    # Consecutive member IDs map to consecutive visible ordinals. Pool chunks
    # contain at most one member per GPU; every queued chunk has the same mapping.
    processes = _worker_count(config, n_models=n_models, gpus_per_node=int(gpus_per_node))
    pool = context.Pool(processes=processes)
    try:
        raw_results = []
        for start in range(0, n_models, processes):
            raw_results.extend(pool.map(_excited_state_training_worker, params_list[start:start + processes]))
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()

    completed = [_excited_state_member_complete(result) for result in raw_results]

    if all(completed):
        try:
            if gap_table:
                _validate_completed_ensemble(
                    ensemble_dir,
                    state_table,
                    gap_table,
                )
            else:
                _validate_completed_ensemble(ensemble_dir, state_table)
        except Exception as exc:
            completed = [False for _ in completed]
            ensemble_root = Path(ensemble_dir).expanduser().resolve()
            ensemble_root.mkdir(parents=True, exist_ok=True)
            (ensemble_root / "training_error.json").write_text(
                json.dumps(
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return completed, int(current_training_id)


def validate_excited_state_task_options(
    *,
    remove_existing: bool,
    h5_test_dir: str | None,
) -> None:
    """Reject task options that would delete or silently ignore data."""

    if bool(remove_existing):
        raise ValueError(
            "remove_existing=True is not supported because this trainer will "
            "not delete or overwrite model data."
        )
    if h5_test_dir is not None:
        raise ValueError(
            "h5_test_dir is not supported by the isolated excited-state "
            "trainer; use valid_size and test_size splits."
        )


@python_app(executors=["alf_ML_executor"])
def train_excited_state_HIPPYNN_ensemble_task(
    ML_config,
    h5_dir,
    model_path,
    current_training_id,
    gpus_per_node,
    properties_list,
    remove_existing=False,
    h5_test_dir=None,
):
    """Parsl entry point for isolated multi-state HIPPYNN ensembles."""

    validate_excited_state_task_options(
        remove_existing=remove_existing,
        h5_test_dir=h5_test_dir,
    )
    return train_excited_state_ensemble(
        ML_config=dict(ML_config or {}),
        h5_dir=str(h5_dir),
        model_path=str(model_path),
        current_training_id=int(current_training_id),
        gpus_per_node=int(gpus_per_node),
        properties_list=dict(properties_list or {}),
    )


# ---------------------------------------------------------------------------------------------
# Multi-node training. One app per member lets ``alf_ML_executor`` place members on any number of
# nodes. Select these tasks with ``ML_task``; the tasks above stay the single-node default.
# ---------------------------------------------------------------------------------------------


def _train_ground_state_member(arg_dict, attempt):
    """Run in a fresh child process: choose this worker's GPU, then train one member."""
    arg_dict = dict(arg_dict)
    device = ensemble_training.member_device(arg_dict.get('device_string', 'from_multiprocessing'),
                                             arg_dict.get('from_multiprocessing_nGPU'))
    arg_dict['device_string'] = device
    with ensemble_training.member_placement(arg_dict['model_dir'], attempt, device) as record:
        train_HIPNN_model(**arg_dict)
        record['completed'] = _ground_state_member_complete(arg_dict['model_dir'])
    return record['completed']


def _train_excited_state_member(arg_dict, attempt):
    """Run in a fresh child process: choose this worker's GPU, train one member, then reload it."""
    config = dict(arg_dict['ML_config'])
    device = ensemble_training.member_device(config.get('device_string', 'from_multiprocessing'),
                                             arg_dict.get('from_multiprocessing_nGPU'))
    config['device_string'] = device
    arg_dict = dict(arg_dict, ML_config=config)
    with ensemble_training.member_placement(arg_dict['model_dir'], attempt, device) as record:
        result = _excited_state_training_worker(arg_dict)
        record['completed'] = _excited_state_member_complete(result)
        if record['completed']:
            state_table = validate_excited_state_training_config(config, arg_dict['properties_list'])
            gap_table = _excited_state_gap_table(config, arg_dict['properties_list'])
            try:
                _validate_completed_member(arg_dict['model_dir'], state_table, gap_table)
            except Exception as exc:
                record['completed'] = False
                (Path(arg_dict['model_dir']) / 'training_error.json').write_text(
                    json.dumps({'error_type': type(exc).__name__, 'error': str(exc),
                                'traceback': traceback.format_exc()}, indent=2), encoding='utf-8')
    return record['completed']


@python_app(executors=['alf_ML_executor'])
def _ground_state_member_task(arg_dict, shards, attempt):
    from alframework.ml_interfaces import hippynn_interface
    from alframework.tools import ensemble_training
    ensemble_training.wait_for_shards(shards)
    return ensemble_training.run_member_in_child(hippynn_interface._train_ground_state_member, arg_dict, attempt)


@python_app(executors=['alf_ML_executor'])
def _excited_state_member_task(arg_dict, shards, attempt):
    from alframework.ml_interfaces import hippynn_interface
    from alframework.tools import ensemble_training
    ensemble_training.wait_for_shards(shards)
    return ensemble_training.run_member_in_child(hippynn_interface._train_excited_state_member, arg_dict, attempt)


def _ml_executor():
    import parsl
    return parsl.dfk().executors.get('alf_ML_executor')


@join_app
def train_HIPPYNN_ensemble_multinode_task(ML_config, h5_dir, model_path, current_training_id, gpus_per_node,
                                          member_retries=1):
    """Train a ground-state ensemble with one app per member, so it can span any number of nodes.

    This body runs on the coordinator and doesn't train. The ML interfaces guide covers the Parsl setup.
    """
    params_list = _ground_state_member_params(ML_config, h5_dir, model_path, current_training_id, gpus_per_node)
    shards = _h5_shards(h5_dir)
    return ensemble_training.train_ensemble_across_nodes(
        lambda member, attempt: _ground_state_member_task(params_list[member], shards, attempt),
        len(params_list), model_path.format(current_training_id), current_training_id, gpus_per_node,
        ML_config.get('device_string', 'from_multiprocessing'), member_retries, _ml_executor())


@join_app
def train_excited_state_HIPPYNN_ensemble_multinode_task(ML_config, h5_dir, model_path, current_training_id,
                                                        gpus_per_node, properties_list, remove_existing=False,
                                                        h5_test_dir=None, member_retries=1):
    """Train an excited-state ensemble with one app per member, so it can span any number of nodes.

    This body runs on the coordinator and doesn't train. The ML interfaces guide covers the Parsl setup.
    """
    validate_excited_state_task_options(remove_existing=remove_existing, h5_test_dir=h5_test_dir)
    config = dict(ML_config or {})
    properties_list = dict(properties_list or {})
    # A bad configuration fails here, rather than once per member and attempt.
    validate_excited_state_training_config(config, properties_list)
    _excited_state_gap_table(config, properties_list)
    ensemble_dir = model_path.format(int(current_training_id))
    root = Path(ensemble_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty ensemble directory {root}')
    params_list = _excited_state_member_params(config, h5_dir, ensemble_dir, gpus_per_node, properties_list)
    shards = _h5_shards(h5_dir)
    return ensemble_training.train_ensemble_across_nodes(
        lambda member, attempt: _excited_state_member_task(params_list[member], shards, attempt),
        len(params_list), ensemble_dir, int(current_training_id), gpus_per_node,
        config.get('device_string', 'from_multiprocessing'), member_retries, _ml_executor())
