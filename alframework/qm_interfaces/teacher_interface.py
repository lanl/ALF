"""Label structures with a foundation model, such as MACE or UMA, in place of QM.

A teacher is a pretrained checkpoint as its authors publish it; a model ALF has fine-tuned is used directly,
not as a teacher. A teacher whose task was trained only on non-periodic structures (UMA's ``omol``) refuses
periodic ones unless ``teacher.allow_periodic`` is true.

Select ``teacher_batch_task`` (with ``"batching": {"mode": "native"}``) or ``teacher_task`` as the
``QM_task``, and describe the teacher in the QM configuration's ``teacher`` block::

    {"teacher": {"loader": "alframework.qm_interfaces.teacher_interface.load_mace_teacher",
                 "name": "mace-mp-0-small", "checkpoint": "/models/mace-mp-0-small.model",
                 "dtype": "float64"},
     "batching": {"mode": "native", "batch_size": 32}}

A loader is called as ``loader(settings, device)`` and returns a teacher: an object with
``evaluate(atoms_list)`` returning energies in eV and forces in eV/Å, in input order, plus
``package``, ``supported_numbers`` (or ``None``), ``supports_charge_spin``, and optionally
``supports_periodic`` (``True`` if absent). Labels count as
converged when they are finite. Every labeled molecule records the teacher in
``label_provenance``, which storage writes onto the HDF5 shard.
"""
import hashlib
import json
import math
import os
import threading
from importlib import import_module, metadata
from pathlib import Path

import numpy as np
from parsl import python_app
from alframework.tools.task_timeline import timed

TEACHER_PROPERTIES = {"energy": "system", "forces": "atomic"}
DTYPES = ("float32", "float64")
_TEACHERS = {}
# Models such as MACE keep state between calls, so a process uses its teacher from one thread at a time.
_LOCK = threading.RLock()


def _resolve(path, master_directory=None):
    path = Path(str(path)).expanduser()
    if not path.is_absolute() and master_directory:
        path = Path(master_directory) / path
    return path


def _load_function(name):
    module, function = str(name).rsplit(".", 1)
    return getattr(import_module(module), function)


def teacher_settings(QM_config, master_directory=None):
    """Return the validated ``teacher`` block, with the checkpoint as an absolute path."""
    block = (QM_config or {}).get("teacher")
    if not isinstance(block, dict):
        raise ValueError("Teacher labeling needs a 'teacher' block in the QM configuration.")
    settings = dict(block)
    for key in ("loader", "checkpoint"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ValueError(f"teacher.{key} must name a {'loader function' if key == 'loader' else 'local file'}.")
    settings["checkpoint"] = str(_resolve(settings["checkpoint"], master_directory))
    settings.setdefault("name", Path(settings["checkpoint"]).stem)
    settings.setdefault("dtype", "float64")
    if settings["dtype"] not in DTYPES:
        raise ValueError(f"teacher.dtype must be one of {DTYPES}, not {settings['dtype']!r}.")
    for key in ("allow_mixed_labels", "allow_periodic"):
        if not isinstance(settings.setdefault(key, False), bool):
            raise ValueError(f"teacher.{key} must be true or false.")
    if not isinstance(settings.setdefault("options", {}), dict):
        raise ValueError("teacher.options must be a dictionary.")
    return settings


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(provenance):
    """What must match for two labels to come from the same teacher."""
    return (provenance.get("checkpoint_sha256"), provenance.get("task"), provenance.get("loader"))


def validate_teacher_config(QM_config, master_config=None):
    """Check a teacher configuration when ALF starts, so that a mistake doesn't fail every task.

    Args:
        QM_config (dict): The QM configuration, with its ``teacher`` block.
        master_config (dict): For ``properties_list``, ``master_directory``, and ``h5_dir``.

    """
    master_config = master_config or {}
    settings = teacher_settings(QM_config, master_config.get("master_directory"))
    if not Path(settings["checkpoint"]).is_file():
        raise ValueError(f"Teacher checkpoint not found: {settings['checkpoint']}")
    loader = _load_function(settings["loader"])
    check = getattr(loader, "validate_settings", None)
    if check is not None:
        check(settings)
    properties = master_config.get("properties_list")
    if properties is not None:
        _validate_properties(properties)
    h5_dir = master_config.get("h5_dir")
    if h5_dir and not settings["allow_mixed_labels"]:
        _refuse_mixed_labels(settings, h5_dir)


def _validate_properties(properties):
    if "energy" not in properties:
        raise ValueError("Teacher labeling needs an 'energy' entry in properties_list.")
    for key, spec in properties.items():
        if key not in TEACHER_PROPERTIES:
            raise ValueError(f"A teacher labels only {sorted(TEACHER_PROPERTIES)}, not {key!r}.")
        if str(spec[1]).lower() != TEACHER_PROPERTIES[key]:
            raise ValueError(f"properties_list[{key!r}] must be '{TEACHER_PROPERTIES[key]}'.")
        if float(spec[2]) != 1.0:
            raise ValueError(f"properties_list[{key!r}] has unit factor {spec[2]}, but teachers give eV and eV/Å, "
                             "so it must be 1.0. A factor copied from a QM configuration would rescale every label.")


def _refuse_mixed_labels(settings, h5_dir):
    from alframework.tools.tools import read_label_provenance

    expected = (file_sha256(settings["checkpoint"]), settings.get("task"), settings["loader"])
    for path in sorted(Path(h5_dir).glob("*.h5")):
        sources = read_label_provenance(path)
        if not sources or any(_identity(source) != expected for source in sources):
            found = "labels with no teacher recorded, such as QM labels" if not sources else "labels from another teacher"
            raise ValueError(f"{path} holds {found}, so a student would train on mixed reference levels. "
                             "Use a new h5_dir, or set teacher.allow_mixed_labels to true.")


def _device(settings, gpus_per_node):
    if settings.get("device"):
        return str(settings["device"])
    from alframework.tools.gpu import sampling_device
    return sampling_device({}, gpus_per_node)


def load_teacher(settings, device):
    """Return this worker's teacher for these settings, loading it the first time."""
    key = json.dumps(settings, sort_keys=True) + "|" + device
    if key not in _TEACHERS:
        _TEACHERS.clear()  # Hold one teacher per worker, so a changed setting frees the old one's memory.
        teacher = _load_function(settings["loader"])(settings, device)
        teacher.provenance = {
            "teacher": settings["name"], "loader": settings["loader"], "package": teacher.package,
            "package_version": _version(teacher.package), "checkpoint": settings["checkpoint"],
            "checkpoint_sha256": file_sha256(settings["checkpoint"]), "task": settings.get("task"),
            "dtype": settings["dtype"], "adapter": type(teacher).__name__, "units": {"energy": "eV", "forces": "eV/Å"},
            "charge_spin": "from molecule metadata 'charge' and 'mult'; defaults 0 and 1",
            "allow_periodic": settings["allow_periodic"], "options": settings["options"],
        }
        _TEACHERS[key] = teacher
    return _TEACHERS[key]


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


class TorchSimTeacher:
    """Evaluate a batch in one forward pass of a TorchSim model."""

    adapter = "torchsim"

    def __init__(self, model, package, supported_numbers=None, supports_charge_spin=False, supports_periodic=True):
        self.model, self.package = model, package
        self.supported_numbers = None if supported_numbers is None else {int(z) for z in supported_numbers}
        self.supports_charge_spin, self.supports_periodic = supports_charge_spin, supports_periodic

    def evaluate(self, atoms_list):
        import torch_sim as ts

        extras = {"system_extras_map": {"charge": "charge", "spin": "spin"}} if self.supports_charge_spin else {}
        state = ts.io.atoms_to_state(list(atoms_list), device=self.model.device, dtype=self.model.dtype, **extras)
        output = self.model(state)
        energies = output["energy"].detach().cpu().double().numpy().reshape(-1)
        forces = output["forces"].detach().cpu().double().numpy()
        ends = np.cumsum([len(atoms) for atoms in atoms_list])
        return list(energies), np.split(forces, ends[:-1])


class AseTeacher:
    """Evaluate one structure at a time with an ASE calculator."""

    adapter = "ase"

    def __init__(self, calculator, package, supported_numbers=None, supports_charge_spin=False, supports_periodic=True):
        self.calculator, self.package = calculator, package
        self.supported_numbers = None if supported_numbers is None else {int(z) for z in supported_numbers}
        self.supports_charge_spin, self.supports_periodic = supports_charge_spin, supports_periodic

    def evaluate(self, atoms_list):
        energies, forces = [], []
        for atoms in atoms_list:
            atoms = atoms.copy()
            atoms.calc = self.calculator
            energies.append(float(atoms.get_potential_energy()))
            forces.append(np.asarray(atoms.get_forces(), dtype=float))
        return energies, forces


def _torch_dtype(settings):
    import torch
    return torch.float64 if settings["dtype"] == "float64" else torch.float32


def load_mace_teacher(settings, device):
    """A MACE model, through MACE's own TorchSim interface. MACE models assume neutral singlets."""
    import torch
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(settings["checkpoint"], device=torch.device(device), dtype=_torch_dtype(settings),
                              compute_forces=True, compute_stress=False, **settings["options"])
    return TorchSimTeacher(model, "mace-torch", supported_numbers=model.z_table.zs)


def load_uma_teacher(settings, device):
    """A UMA model, through fairchem's own TorchSim interface. ``task`` picks the UMA head."""
    load_uma_teacher.validate_settings(settings)
    import torch
    from fairchem.core.calculate.torchsim_interface import FairChemModel

    model = FairChemModel(settings["checkpoint"], device=torch.device(device), dtype=_torch_dtype(settings),
                          task_name=settings["task"], **settings["options"])
    # Only the molecular task takes total charge and spin multiplicity.
    return TorchSimTeacher(model, "fairchem-core", supports_charge_spin=settings["task"] == "omol",
                           supports_periodic=uma_task_supports_periodic(settings["task"]))


def uma_task_supports_periodic(task):
    """Whether a UMA task was trained on periodic structures. fairchem's ``omol`` data are all aperiodic."""
    return task != "omol"


def _require_task(settings):
    if not isinstance(settings.get("task"), str) or not settings["task"]:
        raise ValueError("A UMA teacher needs teacher.task, such as 'omol' for molecules or 'omat' for materials. "
                         "There is no default, because each task is a different level of theory.")


load_uma_teacher.validate_settings = _require_task


def _fail(molecule, reason):
    molecule.set_converged_flag(False)
    molecule.update_metadata({"labeling_error": reason})


def _unsupported(teacher, molecule, settings):
    atoms = molecule.get_atoms()
    if atoms.get_pbc().any() and not getattr(teacher, "supports_periodic", True) and not settings["allow_periodic"]:
        return ("This teacher's task was trained on non-periodic structures, such as UMA's omol, but this structure "
                "is periodic. Set teacher.allow_periodic to true to label periodic structures anyway")
    if teacher.supported_numbers is not None:
        missing = sorted(set(atoms.get_atomic_numbers().tolist()) - teacher.supported_numbers)
        if missing:
            return f"Teacher has no parameters for atomic numbers {missing}"
    charge, mult = molecule.get_metadata().get("charge", 0), molecule.get_metadata().get("mult", 1)
    if not teacher.supports_charge_spin and (charge != 0 or mult != 1):
        return f"Teacher assumes neutral singlets, but this structure has charge {charge} and multiplicity {mult}"
    return None


def _atoms(molecule, teacher):
    atoms = molecule.get_atoms().copy()
    if teacher.supports_charge_spin:
        atoms.info["charge"] = int(molecule.get_metadata().get("charge", 0))
        atoms.info["spin"] = int(molecule.get_metadata().get("mult", 1))
    return atoms


def _evaluate(teacher, molecules, batch_error=None):
    """Label molecules, splitting a failing batch in half until each failure is isolated."""
    try:
        energies, forces = teacher.evaluate([_atoms(molecule, teacher) for molecule in molecules])
    except Exception as exc:
        _free_memory()
        if len(molecules) == 1:
            first = f" (batch error: {batch_error})" if batch_error else ""
            return _fail(molecules[0], f"{type(exc).__name__}: {exc}{first}")
        half = len(molecules) // 2
        for part in (molecules[:half], molecules[half:]):
            _evaluate(teacher, part, batch_error or f"{type(exc).__name__}: {exc}")
        return None
    for molecule, energy, force in zip(molecules, energies, forces):
        energy, force = float(energy), np.asarray(force, dtype=float)
        molecule.store_results({"energy": energy, "forces": force})
        if math.isfinite(energy) and np.isfinite(force).all() and force.shape == (len(molecule.get_atoms()), 3):
            molecule.set_converged_flag(True)
        else:
            _fail(molecule, "Teacher returned non-finite or misshapen results")
    return None


def _free_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def label_with_teacher(molecule_objects, QM_config, properties_list=None, gpus_per_node=None,
                       master_directory=None):
    """Label molecules with the configured teacher, returning them in input order.

    Args:
        molecule_objects (list): MoleculesObject instances to label.
        QM_config (dict): The QM configuration, with its ``teacher`` block.
        properties_list (dict): ALF's property table, checked against what a teacher provides.
        gpus_per_node (int): Caps automatic GPU choice, as for samplers.
        master_directory (str): Resolves a relative checkpoint path.

    Returns:
        (list): The same molecules, each labeled or marked unconverged with a ``labeling_error``.
    """
    molecule_objects = list(molecule_objects)
    settings = teacher_settings(QM_config, master_directory)
    if properties_list is not None:
        _validate_properties(properties_list)
    with _LOCK:
        teacher = load_teacher(settings, _device(settings, gpus_per_node))
        groups = {}
        for molecule in molecule_objects:
            molecule.update_metadata({"label_provenance": teacher.provenance})
            reason = _unsupported(teacher, molecule, settings)
            if reason is not None:
                _fail(molecule, reason)
            else:
                groups.setdefault(tuple(bool(p) for p in molecule.get_atoms().get_pbc()), []).append(molecule)
        for group in groups.values():
            _evaluate(teacher, group)
    return molecule_objects


@python_app(executors=['alf_QM_executor'])
@timed('QM')
def teacher_task(molecule_object, QM_config, properties_list, gpus_per_node=None, master_directory=None):
    """Label one structure with the configured teacher. Also used by ``"mode": "sequential"`` batching."""
    return label_with_teacher([molecule_object], QM_config, properties_list, gpus_per_node, master_directory)[0]


@python_app(executors=['alf_QM_executor'])
@timed('QM')
def teacher_batch_task(molecule_objects, QM_config, properties_list, gpus_per_node=None, master_directory=None):
    """Label a batch with one forward pass of the configured teacher, for ``"mode": "native"`` batching."""
    return label_with_teacher(molecule_objects, QM_config, properties_list, gpus_per_node, master_directory)


# LabelingCoordinator calls this at startup. The docs build replaces python_app with the plain function.
for _task in (teacher_task, teacher_batch_task):
    getattr(_task, "func", _task).validate_config = validate_teacher_config
