"""PySCF labeling on the CPU: the ASE calculator and ``pyscf_task``, for molecules and periodic cells.

These test ALF's wrapper, not PySCF's physics. Unit, sign, and stress conversions use a stand-in SCF with
known numbers. Tests marked ``needs_pyscf`` run real PySCF, with a minimal basis or a tiny cell, once for
each path ALF uses, so a PySCF release that changes what ALF calls is caught; they check that the path
runs and returns finite results of the right shape, never accuracy. Accuracy is for ``--test_qm`` and the
qualification rounds. They skip where PySCF isn't installed, as in CI's minimal job.
"""

import types

import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.qm_interfaces import pyscf as pi
from alframework.qm_interfaces.pyscf import calculator as pc
from alframework.qm_interfaces.pyscf.calculator import PySCFCalculator, validate_pyscf_settings
from alframework.tools.labeling import LabelingCoordinator
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import store_current_data

WATER = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
MOLECULE = {"basis": "sto-3g", "xc": "pbe"}
PERIODIC = {"periodic": {"basis": "gth-szv", "pseudo": "gth-pade"}}
STRESS = {"periodic": {"basis": "gth-szv", "pseudo": "gth-pade", "df": "fftdf", "ke_cutoff": 40}}
GROUND = {"E0": ["energy", "system", 1.0], "F0": ["forces", "atomic", 1.0]}
# A doublet with no degenerate orbitals. OH's degenerate pi orbitals left its SCF oscillating near
# 1e-10 Eh, and on CI it once failed to converge in 100 cycles.
NH2 = [[0.0, 0.0, 0.0], [1.024, 0.0, 0.0], [-0.2303, 0.9978, 0.0]]

try:
    import pyscf  # noqa: F401
    HAVE_PYSCF = True
except ImportError:
    HAVE_PYSCF = False
# CI's minimal job has no PySCF; the settings and refusal tests below still run there.
needs_pyscf = pytest.mark.skipif(not HAVE_PYSCF, reason="needs PySCF")


def hydrogen_cell():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.05, 0.02]], cell=[3.5, 3.6, 3.7], pbc=True)
    atoms.positions += 1.2
    return atoms


def energy(atoms, **settings):
    atoms = atoms.copy()
    atoms.calc = PySCFCalculator(**settings)
    return atoms.get_potential_energy()


# The calculator: molecules.


@needs_pyscf
def test_calculator_works_with_ase_dynamics():
    from ase import units
    from ase.md.verlet import VelocityVerlet

    atoms = WATER.copy()
    atoms.calc = PySCFCalculator(**MOLECULE)
    VelocityVerlet(atoms, timestep=0.25 * units.fs).run(2)
    assert np.isfinite(atoms.get_potential_energy()) and np.isfinite(atoms.get_forces()).all()
    assert atoms.calc.metadata["qm_backend"] == "pyscf" and atoms.calc.metadata["qm_system"] == "molecule"
    assert "RKS" in atoms.calc.metadata["qm_method"]


@needs_pyscf
def test_impossible_multiplicity_is_refused():
    radical = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])
    with pytest.raises(ValueError, match="impossible"):
        energy(radical, **MOLECULE)


# The calculator: periodic cells.


@needs_pyscf
def test_periodic_cell_runs_with_multigrid():
    atoms = hydrogen_cell()
    atoms.calc = PySCFCalculator(**PERIODIC)
    assert atoms.get_forces().shape == (2, 3) and np.isfinite(atoms.get_forces()).all()
    assert atoms.calc.metadata["qm_system"] == "periodic"
    assert "multigrid" in atoms.calc.metadata["qm_numint"]


# The conversions ALF applies to PySCF's results, with a stand-in SCF. Round unit constants make each
# conversion visible.

NIST = types.SimpleNamespace(HARTREE2EV=27.0, BOHR=0.5)


class FakeSCF:
    converged, e_tot, e_free, entropy = True, -2.0, None, 0.0

    def __init__(self, gradient=None, stress=None, **attributes):
        self.gradient, self.stress = gradient, stress
        self.__dict__.update(attributes)

    def kernel(self):
        return self.e_tot

    def Gradients(self):
        return types.SimpleNamespace(kernel=lambda: self.gradient, get_stress=lambda: self.stress)


def with_fake_scf(monkeypatch, scf):
    fake_pyscf = types.SimpleNamespace(lib=types.SimpleNamespace(num_threads=lambda n: None), __version__="fake")
    monkeypatch.setattr(PySCFCalculator, "_modules", lambda self: (fake_pyscf, NIST, None))
    monkeypatch.setattr(PySCFCalculator, "_build", lambda self, kind, charge, mult, pyscf: scf)


def test_energy_and_forces_are_converted_to_ev_and_ev_per_angstrom(monkeypatch):
    gradient = np.arange(9.0).reshape(3, 3) - 4.0
    with_fake_scf(monkeypatch, FakeSCF(gradient=gradient))
    atoms = WATER.copy()
    atoms.calc = PySCFCalculator(**MOLECULE)
    assert atoms.get_potential_energy() == pytest.approx(-2.0 * 27.0)
    np.testing.assert_allclose(atoms.get_forces(), -gradient * 27.0 / 0.5)


def test_stress_is_symmetrized_converted_and_in_voigt_order(monkeypatch):
    sigma = np.arange(1.0, 10.0).reshape(3, 3)  # Not symmetric, so the average shows.
    with_fake_scf(monkeypatch, FakeSCF(gradient=np.zeros((2, 3)), stress=sigma))
    atoms = hydrogen_cell()
    atoms.calc = PySCFCalculator(**STRESS)
    symmetric = 0.5 * (sigma + sigma.T) * 27.0 / 0.5 ** 3
    xx, yy, zz, yz, xz, xy = (symmetric[i, j] for i, j in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)])
    np.testing.assert_allclose(atoms.get_stress(), [xx, yy, zz, yz, xz, xy])


def test_smeared_energy_reported_is_the_free_energy(monkeypatch):
    with_fake_scf(monkeypatch, FakeSCF(gradient=np.zeros((2, 3)), e_free=-1.5, entropy=0.1))
    atoms = hydrogen_cell()
    atoms.calc = PySCFCalculator(periodic=dict(PERIODIC["periodic"], smearing_eV=2.0))
    assert atoms.get_potential_energy() == pytest.approx(-1.5 * 27.0)
    assert atoms.calc.metadata["smearing_entropy"] == 0.1 and "free energy" in atoms.calc.metadata["energy_is"]


@pytest.mark.parametrize(
    "atoms,settings,properties,message",
    [
        (Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[4, 4, 4], pbc=[True, True, False]), PERIODIC,
         ["energy"], "all or none"),
        (hydrogen_cell(), dict(PERIODIC, nroots=1, density_fit=False), ["energy"], "molecules only"),
        (WATER, MOLECULE, ["stress"], "periodic cells only"),
        (hydrogen_cell(), PERIODIC, ["stress"], "fftdf"),
    ],
    ids=["mixed_pbc", "periodic_excited", "molecule_stress", "multigrid_stress"],
)
def test_calculations_it_cannot_do_are_refused(atoms, settings, properties, message):
    atoms = atoms.copy()
    calculator = PySCFCalculator(**settings)
    with pytest.raises(ValueError, match=message):
        calculator.calculate(atoms, properties)


@needs_pyscf
def test_charged_periodic_cell_is_refused():
    atoms = hydrogen_cell()
    atoms.info.update(charge=2, mult=1)
    with pytest.raises(ValueError, match="neutral"):
        energy(atoms, **PERIODIC)


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"backend": "cuda"}, "backend"),
        ({"basis_set": "sto-3g"}, "Unknown PySCF settings"),
        ({"periodic": {"df": "pw"}}, "periodic.df"),
        ({"periodic": {"kpts": [2, 2]}}, "kpts"),
        ({"periodic": {"kpts": [2, 2, 2]}}, "Gamma point only"),
        ({"periodic": {"df": "gdf"}}, "no CPU forces"),
        ({"periodic": {"smearing": 0.01}}, "Unknown periodic"),
        ({"nroots": 1}, "density fitting"),
        ({"nroots": -1}, "nroots"),
        ({"multiplicity": 0}, "multiplicity"),
    ],
)
def test_bad_settings_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        validate_pyscf_settings(settings)


def test_gdf_and_k_points_are_allowed_on_the_gpu():
    options = validate_pyscf_settings({"backend": "gpu", "periodic": {"df": "gdf", "kpts": [2, 2, 2]}})
    assert options["periodic"]["df"] == "gdf" and options["nroots"] == 0


@needs_pyscf
def test_gpu_backend_moves_the_method_to_gpu4pyscf(monkeypatch):
    moved = []
    monkeypatch.setattr(pc, "_single_device_backend", lambda pyscf: (None, None))
    real_import = pc.importlib.import_module
    fake_cupy = types.SimpleNamespace(ndarray=type("ndarray", (), {}), asnumpy=np.asarray)
    monkeypatch.setattr(pc, "importlib", types.SimpleNamespace(
        import_module=lambda name: fake_cupy if name == "cupy" else real_import(name)))
    monkeypatch.setattr(pc, "_to_gpu", lambda mf: moved.append(type(mf).__name__) or mf)
    atoms = WATER.copy()
    atoms.calc = PySCFCalculator(backend="gpu", **MOLECULE)
    atoms.get_forces()
    assert moved and "RKS" in moved[0]
    assert atoms.calc.metadata["qm_backend"] == "gpu4pyscf"


@needs_pyscf
def test_gpu_calculation_does_not_write_inherited_cpu_checkpoint(monkeypatch):
    """An unused temporary checkpoint must not make an otherwise valid label fail."""
    import pyscf.scf.chkfile

    inherited = []
    monkeypatch.setattr(pc, "_single_device_backend", lambda pyscf: (None, None))
    real_import = pc.importlib.import_module
    fake_cupy = types.SimpleNamespace(ndarray=type("ndarray", (), {}), asnumpy=np.asarray)
    monkeypatch.setattr(pc, "importlib", types.SimpleNamespace(
        import_module=lambda name: fake_cupy if name == "cupy" else real_import(name)))
    def move(mf):
        inherited.append(mf.chkfile)
        return mf
    monkeypatch.setattr(pc, "_to_gpu", move)
    def locked(*args, **kwargs):
        raise BlockingIOError("inherited temporary checkpoint is locked")
    monkeypatch.setattr(pyscf.scf.chkfile, "save_mol", locked)
    monkeypatch.setattr(pyscf.scf.chkfile, "dump_scf", locked)
    atoms = WATER.copy()
    atoms.calc = PySCFCalculator(backend="gpu", **MOLECULE)
    assert np.isfinite(atoms.get_forces()).all()
    assert inherited and inherited[0]


@needs_pyscf
def test_cpu_build_keeps_its_checkpoint():
    import pyscf

    calc = PySCFCalculator(**MOLECULE)
    calc.atoms = WATER.copy()
    mf = calc._build("molecule", 0, 1, pyscf)
    assert mf.chkfile


@needs_pyscf
def test_gpu_backend_switches_on_multigrid_after_the_move(monkeypatch):
    """PySCF's CPU multigrid integrator has no GPU counterpart, so GPU4PySCF's own is switched on after to_gpu."""
    from pyscf.pbc.dft import rks

    events = []
    monkeypatch.setattr(pc, "_single_device_backend", lambda pyscf: (None, None))
    real_import = pc.importlib.import_module
    fake_cupy = types.SimpleNamespace(ndarray=type("ndarray", (), {}), asnumpy=np.asarray)
    monkeypatch.setattr(pc, "importlib", types.SimpleNamespace(
        import_module=lambda name: fake_cupy if name == "cupy" else real_import(name)))
    monkeypatch.setattr(pc, "_to_gpu", lambda mf: events.append("to_gpu") or mf)
    original = rks.RKS.multigrid_numint
    monkeypatch.setattr(rks.RKS, "multigrid_numint",
                        lambda self, mesh=None: events.append("multigrid") or original(self, mesh))
    atoms = hydrogen_cell()
    atoms.calc = PySCFCalculator(backend="gpu", **PERIODIC)
    atoms.get_potential_energy()
    assert events == ["to_gpu", "multigrid"]


@needs_pyscf
def test_hybrid_functional_with_periodic_multigrid_is_refused():
    with pytest.raises(ValueError, match="pure functionals only"):
        energy(hydrogen_cell(), xc="b3lyp", **PERIODIC)


# The labeling task.


def molecule(atoms, index=0):
    return MoleculesObject(atoms.copy(), f"mol-0000-{index:010d}")


@needs_pyscf
def test_task_stores_the_ground_state_under_the_names_trainers_read(tmp_path):
    from alframework.ml_interfaces.fine_tuning import shard_atoms

    labeled = [pi.label_pyscf_molecule(molecule(WATER, i), MOLECULE, GROUND) for i in range(3)]
    assert all(m.check_convergence() for m in labeled)
    assert set(labeled[0].get_results()) == {"E0", "F0"}
    shard = tmp_path / "data-0000.h5"
    store_current_data(str(shard), labeled, GROUND)
    with h5py.File(shard) as handle:
        group = handle["H02_O01"]
        assert {"energy", "forces"} <= set(group) and "cell" not in group
    frames = shard_atoms([shard])
    assert len(frames) == 3 and frames[0].info["REF_energy"] == pytest.approx(labeled[0].get_results()["E0"])
    hippynn = pytest.importorskip("hippynn")
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB

    database = ALFPyAniDirectoryDB(directory=str(tmp_path), inputs=["species", "coordinates"],
                                   targets=["energy", "forces"], quiet=True, seed=0)
    assert len(database.arr_dict["energy"]) == 3 and hippynn


@needs_pyscf
def test_task_labels_a_periodic_cell_with_its_stress(tmp_path):
    properties = dict(GROUND, stress=["stress", "system", 1.0])
    labeled = pi.label_pyscf_molecule(molecule(hydrogen_cell()), STRESS, properties)
    assert labeled.check_convergence(), labeled.get_metadata().get("qm_error")
    assert labeled.get_results()["stress"].shape == (6,)
    assert labeled.get_metadata()["qm_system"] == "periodic"
    shard = tmp_path / "data-0000.h5"
    store_current_data(str(shard), [labeled], properties)
    with h5py.File(shard) as handle:
        group = handle["H02"]
        assert {"energy", "forces", "stress", "cell"} <= set(group)


@needs_pyscf
def test_task_labels_excited_states_and_gaps():
    properties = {"E0": ["E0", "system", 1.0], "E1": ["E1", "system", 1.0], "F0": ["F0", "atomic", 1.0],
                  "F1": ["F1", "atomic", 1.0], "dE01": ["dE01", "system", 1.0]}
    labeled = pi.label_pyscf_molecule(molecule(WATER), dict(MOLECULE, nroots=1, density_fit=False,
                                                             energy_offset_eV=1.5), properties)
    results = labeled.get_results()
    assert labeled.check_convergence()
    assert results["dE01"] == pytest.approx(results["E1"] - results["E0"]) and results["dE01"] > 0
    assert labeled.get_metadata()["energy_offset_eV"] == 1.5 and labeled.get_metadata()["n_states"] == 2
    plain = pi.label_pyscf_molecule(molecule(WATER), dict(MOLECULE, nroots=1, density_fit=False), properties)
    for key in ("E0", "E1"):
        assert results[key] == pytest.approx(plain.get_results()[key] - 1.5, abs=1e-8)


@needs_pyscf
def test_failed_scf_marks_the_structure_unconverged():
    labeled = pi.label_pyscf_molecule(molecule(WATER), dict(MOLECULE, scf_max_cycle=1), GROUND)
    assert not labeled.check_convergence() and labeled.get_results() == {}
    assert labeled.get_metadata()["qm_failed_stage"] == "scf"


@needs_pyscf
def test_open_shell_structure_runs_unrestricted_dft():
    radical = MoleculesObject(Atoms("NH2", positions=NH2), "mol-0000-0000000000")
    radical.update_metadata({"mult": 2})
    labeled = pi.label_pyscf_molecule(radical, MOLECULE, GROUND)
    assert labeled.check_convergence(), labeled.get_metadata().get("qm_error")
    assert labeled.get_metadata()["mult"] == 2 and "UKS" in labeled.get_metadata()["qm_method"]


@pytest.mark.parametrize(
    "settings,properties,message",
    [
        (MOLECULE, {"E0": ["energy", "system", 1.0], "E1": ["e1", "system", 1.0]}, "nroots \\+ 1"),
        (MOLECULE, dict(GROUND, dipole=["dipole", "system", 1.0]), "not \\['dipole'\\]"),
        (MOLECULE, dict(GROUND, stress=["stress", "atomic", 1.0]), "system property"),
        ({"basis": ""}, GROUND, "basis"),
    ],
    ids=["too_many_states", "unknown_property", "atomic_stress", "empty_basis"],
)
def test_configuration_mistakes_stop_alf_at_startup(settings, properties, message):
    with pytest.raises(ValueError, match=message):
        LabelingCoordinator(pi.pyscf_task, "pyscf", settings, {"properties_list": properties})


def test_task_works_with_sequential_batching():
    coordinator = LabelingCoordinator(pi.pyscf_task, "pyscf", dict(MOLECULE, batching={"mode": "sequential",
                                                                                       "batch_size": 2}),
                                      {"properties_list": GROUND})
    assert coordinator is not None


@pytest.mark.parametrize("extra", [{"compute_nacr": True}, {"_compute_optical": True}, {"solvent": {"method": "C-PCM"}}],
                         ids=["nacr", "optical", "solvent"])
def test_cpu_state_path_refuses_what_only_the_gpu_gives(extra):
    from alframework.qm_interfaces.pyscf.gpu4pyscf import GPU4PySCFDependencyError, run_gpu4pyscf_states

    with pytest.raises(GPU4PySCFDependencyError, match="GPU backend"):
        run_gpu4pyscf_states(WATER.copy(), dict(extra, nroots=2), device_index=0, backend="cpu")


@needs_pyscf
def test_periodic_cell_runs_with_smearing():
    settings = {"periodic": {"basis": "gth-szv", "pseudo": "gth-pade", "smearing_eV": 2.0}}
    atoms = hydrogen_cell()
    atoms.calc = PySCFCalculator(**settings)
    assert np.isfinite(atoms.get_forces()).all()
    assert atoms.calc.metadata["smearing_eV"] == 2.0 and atoms.calc.metadata["smearing_entropy"] >= 0


@needs_pyscf
@pytest.mark.parametrize("batching", [{"mode": "sequential", "batch_size": 2, "max_wait_seconds": 0.05}, None],
                         ids=["sequential", "unbatched"])
def test_pyscf_task_labels_through_the_real_coordinator(tmp_path, monkeypatch, batching):
    """Bootstrap and a retraining round, with real PySCF labels stored as energy and forces."""
    from parsl import python_app

    from tests.helpers import driver
    from tests.helpers.driver import MODEL_PATTERN, run_alf_driver

    monkeypatch.setattr(driver, "PROPERTIES", GROUND)

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        import zlib
        atoms = WATER.copy()
        atoms.rattle(0.03, seed=zlib.crc32(moleculeid.encode()))
        return MoleculesObject(atoms, moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        (tmp_path / MODEL_PATTERN.format(current_training_id)).mkdir(parents=True)
        return [True, True], current_training_id

    qm_config = dict(MOLECULE, **({"batching": batching} if batching else {}))
    run = run_alf_driver(tmp_path, monkeypatch, builder=builder, sampler=sampler, qm=pi.pyscf_task, ml=ml,
                         qm_config=qm_config, bootstrap_set=3, parallel_samplers=2, target_queued_QM=2,
                         save_h5_threshold=2, maximum_completed_QM=6, timeout=300)
    assert run.status["current_model_id"] >= 1 and run.status["lifetime_failed_QM_tasks"] == 0
    shards = sorted((tmp_path / "h5").glob("data-*.h5"))
    assert len(shards) >= 2
    for shard in shards:
        with h5py.File(shard) as handle:
            for group in handle.values():
                assert {"energy", "forces"} <= set(group) and np.isfinite(group["energy"][()]).all()
