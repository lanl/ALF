"""Small CPU-trained checkpoints for dependency and serialization regressions."""

from pathlib import Path


def train_cpu_ensemble(directory, members=2, *, periodic=False):
    import h5py
    import numpy as np
    import torch
    from hippynn.experiment import SetupParams, setup_and_train
    from hippynn.experiment.assembly import assemble_for_training
    from hippynn.experiment.controllers import Controller
    from hippynn.graphs import inputs, loss, networks, physics, targets
    from hippynn.tools import active_directory

    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB

    directory = Path(directory).resolve()
    data = directory / "training-data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(29)
    coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    coordinates = coordinates[None] + rng.normal(0.0, 0.02, (12, 3, 3))
    with h5py.File(data / "data.h5", "w") as handle:
        group = handle.create_group("water")
        group["species"] = np.array(["O", "H", "H"], dtype="S")
        group["coordinates"] = coordinates
        group["energy"] = (coordinates**2).sum(axis=(1, 2))
        group["forces"] = -2 * coordinates
        if periodic:
            group["cell"] = np.tile(np.eye(3) * 8.0, (len(coordinates), 1, 1))

    reports = []
    for member in range(members):
        torch.manual_seed(member + 13)
        species = inputs.SpeciesNode(db_name="species")
        positions = inputs.PositionsNode(db_name="coordinates")
        parents = (species, positions)
        if periodic:
            parents += (inputs.CellNode(db_name="cell"),)
        network = (networks.HipnnVec if periodic else networks.Hipnn)(
            "hipnn",
            parents,
            periodic=periodic,
            module_kwargs=dict(
                possible_species=[0, 1, 8],
                n_features=4,
                n_sensitivities=4,
                dist_soft_min=0.5,
                dist_soft_max=3.0,
                dist_hard_max=3.5,
                n_interaction_layers=1,
                n_atom_layers=1,
                sensitivity_type="inverse",
                resnet=True,
            ),
        )
        energy = targets.HEnergyNode("henergy", network).mol_energy
        energy.db_name = "energy"
        forces = physics.GradientNode("forces", (energy, positions), sign=-1)
        forces.db_name = "forces"
        objective = loss.MSELoss.of_node(energy) + loss.MSELoss.of_node(forces)
        modules, info = assemble_for_training(objective, {"Loss": objective})
        initial = [p.detach().clone() for p in modules.model.parameters()]
        database = ALFPyAniDirectoryDB(str(data), seed=member, quiet=True, **info)
        database.make_random_split("valid", 0.2)
        database.make_random_split("test", 0.2)
        database.split_the_rest("train")
        controller = Controller(
            optimizer=torch.optim.Adam(modules.model.parameters(), lr=0.001),
            scheduler=None,
            batch_size=4,
            eval_batch_size=4,
            max_epochs=2,
            stopping_key="Loss",
            fraction_train_eval=1.0,
        )
        output = directory / f"model-{member:02d}"
        output.mkdir()
        with active_directory(str(output)):
            setup_and_train(modules, database, SetupParams(controller=controller, device="cpu"))
        assert any(not torch.equal(before, after) for before, after in zip(initial, modules.model.parameters()))
        reports.append(dict(member=output.name, epochs=controller.current_epoch))
    return reports
