"""GPU4PySCF excited-state labeling.

GPU4PySCF, PySCF, and CuPy are replaced with fakes, so this needs no GPU. It checks ALF's settings,
unit conversions, failure handling, and storage, not the quantum chemistry itself.
"""

import importlib
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from ase import Atoms

import alframework.qm_interfaces.gpu4pyscf_interface as gpu4pyscf_module
from alframework.qm_interfaces.gpu4pyscf_interface import (
    DEFAULT_GPU4PYSCF_CONFIG,
    NACR_PROPERTY_KEY,
    GPU4PySCFDependencyError,
    GPU4PySCFDeviceError,
    GPU4PySCFGradientError,
    GPU4PySCFNACError,
    GPU4PySCFOutputError,
    GPU4PySCFSCFError,
    GPU4PySCFTDAError,
    _require_scf_convergence,
    _require_tda_convergence,
    _to_numpy,
    convert_gpu4pyscf_atomic_units,
    excited_excited_nac_pairs,
    gpu4pyscf_excited_state_task,
    label_gpu4pyscf_excited_state_molecule,
    run_gpu4pyscf_states,
    select_gpu4pyscf_device,
    validate_gpu4pyscf_config,
    validate_gpu4pyscf_properties,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import build_input_dict, store_current_data


@pytest.mark.parametrize('solvent', [{}, 'dichloromethane', {'method':'IEF-PCM'}, {'optical_dielectric':0.}, {'optical_dielectric':10.}, {'lebedev_order':30}, {'unknown':1}])
def test_invalid_solvent_cannot_silently_label_in_vacuum(solvent):
    with pytest.raises(ValueError):
        validate_gpu4pyscf_config({'solvent': solvent})
    assert validate_gpu4pyscf_config({'solvent': {'enabled': False}})['xc'] == DEFAULT_GPU4PYSCF_CONFIG['xc']


def _properties(nroots=1, *, include_gap=False, include_nacr=False):
    properties = {}
    for state in range(nroots + 1):
        properties[f"sE{state}"] = [
            f"state_{state}_energy",
            "system",
            1.0,
        ]
        properties[f"F{state}"] = [
            f"state_{state}_forces",
            "atomic",
            1.0,
        ]
    if include_gap:
        properties["dE01"] = ["gap_01", "system", 1.0]
    if include_nacr:
        properties[NACR_PROPERTY_KEY] = ["nacr", "pair_atomic", 1.0]
    return properties


def _molecule(molecule_id="water"):
    return MoleculesObject(
        Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2390, 0.9266, 0.0],
            ],
        ),
        molecule_id,
    )


def _device(index=1):
    return {
        "index": index,
        "label": f"cuda:{index}",
        "visible_count": 4,
        "worker_rank": index,
        "cuda_visible_devices": "0,1,2,3",
    }


def _predictions(nroots=1):
    states = nroots + 1
    energies = np.arange(states, dtype=np.float64) - 90.0
    forces = np.arange(states * 3 * 3, dtype=np.float64).reshape(states, 3, 3)
    return energies, forces


def _no_nac(atom_count=3):
    """Empty coupling array returned when compute_nacr is disabled."""

    return np.empty((0, atom_count, 3), dtype=np.float64)


def _no_pairs():
    return np.empty((0, 2), dtype=np.int64)


def _nac_predictions(nroots, atom_count=3):
    """Deterministic coupling vectors for every excited-excited pair."""

    pairs = np.asarray(excited_excited_nac_pairs(nroots), dtype=np.int64).reshape(-1, 2)
    nacr = np.arange(len(pairs) * atom_count * 3, dtype=np.float64).reshape(len(pairs), atom_count, 3)
    return nacr, pairs


@pytest.fixture
def fake_backend(monkeypatch):
    """Skip GPU selection, and let a test choose what the quantum chemistry run does."""
    monkeypatch.setattr(gpu4pyscf_module, "select_gpu4pyscf_device", lambda count: _device())

    def install(run):
        monkeypatch.setattr(gpu4pyscf_module, "run_gpu4pyscf_states", run)

    return install


def _returns(energies, forces, nacr=None, pairs=None, versions=None):
    """A backend run that returns these outputs, with no couplings unless given."""
    nacr = _no_nac() if nacr is None else nacr
    pairs = _no_pairs() if pairs is None else pairs
    return lambda *args, **kwargs: (energies, forces, nacr, pairs, versions or {})


def _raises(error):
    """A backend run that fails with this error."""

    def run(*args, **kwargs):
        raise error

    return run


def test_defaults_are_five_root_cam_b3lyp():
    options = validate_gpu4pyscf_config({})

    assert options == DEFAULT_GPU4PYSCF_CONFIG
    assert options["xc"] == "cam-b3lyp"
    assert options["basis"] == "6-31g*"
    assert options["nroots"] == 5
    assert options["multiplicity"] == 1
    assert options["density_fit"] is True


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"xc": ""}, "xc"),
        ({"basis": ""}, "basis"),
        ({"charge": 0.5}, "charge"),
        ({"multiplicity": 3}, "singlet"),
        ({"density_fit": "yes"}, "density_fit"),
        (
            {"density_fit": False, "auxbasis": "def2-universal-jkfit"},
            "requires density_fit",
        ),
        ({"grids_level": -1}, "grids_level"),
        ({"nroots": 0}, "nroots"),
        ({"scf_conv_tol": np.nan}, "scf_conv_tol"),
        ({"tda_conv_tol": 0.0}, "tda_conv_tol"),
        ({"scf_max_cycle": 0}, "scf_max_cycle"),
        ({"tda_max_cycle": 2.5}, "tda_max_cycle"),
        ({"num_threads": False}, "num_threads"),
        ({"max_memory_mb": -1}, "max_memory_mb"),
        ({"verbosity": -1}, "verbosity"),
        ({"energy_offset_eV": np.inf}, "energy_offset_eV"),
    ],
    ids=[
        "empty_xc",
        "empty_basis",
        "fractional_charge",
        "triplet",
        "non_boolean_density_fit",
        "auxbasis_without_density_fit",
        "negative_grids_level",
        "zero_roots",
        "nan_scf_tolerance",
        "zero_tda_tolerance",
        "zero_scf_cycles",
        "fractional_tda_cycles",
        "boolean_threads",
        "negative_memory",
        "negative_verbosity",
        "infinite_energy_offset",
    ],
)
def test_malformed_settings_are_rejected(updates, message):
    with pytest.raises(ValueError, match=message):
        validate_gpu4pyscf_config(updates)


def test_valid_settings_are_kept():
    options = validate_gpu4pyscf_config(
        {
            "xc": "pbe0",
            "basis": "def2-svp",
            "charge": 1,
            "density_fit": False,
            "grids_level": 4,
            "nroots": 2,
            "max_memory_mb": 16000,
        }
    )

    assert options["xc"] == "pbe0"
    assert options["basis"] == "def2-svp"
    assert options["charge"] == 1
    assert options["density_fit"] is False
    assert options["nroots"] == 2
    assert options["max_memory_mb"] == pytest.approx(16000)


def test_properties_must_cover_every_state():
    state_table, gap_table = validate_gpu4pyscf_properties(
        _properties(5, include_gap=True),
        nroots=5,
    )
    assert [row["state"] for row in state_table] == list(range(6))
    assert [row["gap_key"] for row in gap_table] == ["dE01"]

    with pytest.raises(ValueError, match=r"nroots \+ 1"):
        validate_gpu4pyscf_properties(_properties(1), nroots=5)

    missing_force = _properties(1)
    del missing_force["F1"]
    with pytest.raises(ValueError, match="Missing.*F1"):
        validate_gpu4pyscf_properties(missing_force, nroots=1)


def test_wrong_or_unknown_properties_are_rejected():
    wrong_energy = _properties(1)
    wrong_energy["sE1"][1] = "atomic"
    with pytest.raises(ValueError, match="system property"):
        validate_gpu4pyscf_properties(wrong_energy, nroots=1)

    wrong_force = _properties(1)
    wrong_force["F1"][1] = "system"
    with pytest.raises(ValueError, match="atomic property"):
        validate_gpu4pyscf_properties(wrong_force, nroots=1)

    unsupported = _properties(1)
    unsupported["dipole"] = ["dipole", "system", 1.0]
    with pytest.raises(ValueError, match="unsupported keys"):
        validate_gpu4pyscf_properties(unsupported, nroots=1)


def test_scf_tda_convergence_flags_are_checked_strictly():
    _require_scf_convergence(SimpleNamespace(converged=np.bool_(True)))
    with pytest.raises(GPU4PySCFSCFError, match="one Boolean scalar"):
        _require_scf_convergence(SimpleNamespace(converged=1))
    with pytest.raises(GPU4PySCFSCFError, match="did not converge"):
        _require_scf_convergence(SimpleNamespace(converged=np.bool_(False)))

    _require_tda_convergence(
        SimpleNamespace(converged=np.asarray([True, True], dtype=bool)),
        2,
    )
    with pytest.raises(GPU4PySCFTDAError) as exc_info:
        _require_tda_convergence(
            SimpleNamespace(converged=np.asarray([True, False], dtype=bool)),
            2,
        )
    assert exc_info.value.failed_states == (2,)
    with pytest.raises(GPU4PySCFTDAError, match="shape"):
        _require_tda_convergence(SimpleNamespace(converged=np.asarray(True)), 2)
    with pytest.raises(GPU4PySCFTDAError, match="Boolean"):
        _require_tda_convergence(SimpleNamespace(converged=np.asarray([1, 1])), 2)


def test_unit_conversion_turns_gradients_into_forces():
    energies, forces = convert_gpu4pyscf_atomic_units(
        np.asarray([1.0, 2.0]),
        np.asarray(
            [
                [[1.0, -1.0, 0.5]],
                [[-2.0, 0.0, 1.0]],
            ]
        ),
        state_count=2,
        atom_count=1,
        hartree_to_ev=4.0,
        bohr_to_angstrom=0.5,
    )

    np.testing.assert_allclose(energies, [4.0, 8.0])
    np.testing.assert_allclose(
        forces,
        [
            [[-8.0, 8.0, -4.0]],
            [[16.0, 0.0, -8.0]],
        ],
    )


@pytest.mark.parametrize(
    ("energies", "gradients", "message"),
    [
        (np.zeros(1), np.zeros((2, 1, 3)), "malformed energies"),
        (np.zeros(2), np.zeros((1, 1, 3)), "malformed gradients"),
        (np.asarray([0.0, np.nan]), np.zeros((2, 1, 3)), "non-finite"),
        (np.zeros(2), np.full((2, 1, 3), np.inf), "non-finite"),
    ],
    ids=["too_few_energies", "too_few_gradients", "nan_energy", "infinite_gradient"],
)
def test_unit_conversion_rejects_malformed_outputs(energies, gradients, message):
    with pytest.raises(GPU4PySCFOutputError, match=message):
        convert_gpu4pyscf_atomic_units(
            energies,
            gradients,
            state_count=2,
            atom_count=1,
            hartree_to_ev=1.0,
            bohr_to_angstrom=1.0,
        )


def test_cupy_arrays_are_copied_to_numpy():
    calls = []

    class FakeGPUArray:
        pass

    value = FakeGPUArray()
    fake_cupy = SimpleNamespace(
        ndarray=FakeGPUArray,
        asnumpy=lambda item: calls.append(item) or np.asarray([1.0, 2.0]),
    )

    converted = _to_numpy(value, fake_cupy)

    assert calls == [value]
    np.testing.assert_allclose(converted, [1.0, 2.0])


def test_worker_rank_picks_the_gpu(
    monkeypatch,
):
    selected = []

    class FakeDevice:
        def __init__(self, index):
            self.index = index

        def use(self):
            selected.append(self.index)

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: 4),
            Device=FakeDevice,
        )
    )
    monkeypatch.setattr(gpu4pyscf_module, "_import_cupy", lambda: fake_cupy)
    monkeypatch.setenv("PARSL_WORKER_RANK", "6")

    device = select_gpu4pyscf_device(4)

    assert device["index"] == 2
    assert device["worker_rank"] == 6
    assert selected == [2]

    fake_cupy.cuda.runtime.getDeviceCount = lambda: 1
    monkeypatch.setenv("PARSL_WORKER_RANK", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    pinned = select_gpu4pyscf_device(4)
    assert pinned["index"] == 0
    assert pinned["visible_count"] == 1
    assert pinned["cuda_visible_devices"] == "3"
    assert selected[-1] == 0

    # A mask listing several GPUs: the rank picks among them, whatever gpus_per_node says.
    fake_cupy.cuda.runtime.getDeviceCount = lambda: 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4,6,7")
    masked = select_gpu4pyscf_device(1)
    assert masked["index"] == 3
    assert masked["cuda_visible_devices"] == "3,4,6,7"

    fake_cupy.cuda.runtime.getDeviceCount = lambda: 0
    with pytest.raises(GPU4PySCFDeviceError, match="CPU fallback"):
        select_gpu4pyscf_device(4)


def test_missing_cupy_gives_a_clear_error(monkeypatch):
    real_import = importlib.import_module

    def block_cupy(name):
        if name == "cupy":
            raise ModuleNotFoundError("synthetic missing CuPy")
        return real_import(name)

    monkeypatch.setattr(gpu4pyscf_module.importlib, "import_module", block_cupy)

    with pytest.raises(GPU4PySCFDependencyError, match="working CuPy"):
        gpu4pyscf_module._import_cupy()


@pytest.fixture
def fake_gpu_stack(monkeypatch):
    """Stand-ins for CuPy, PySCF and GPU4PySCF that record what the backend asks of them."""
    calls = {
        "gradient_states": [],
        "threads": [],
        "device": [],
    }

    class FakeDevice:
        def __init__(self, index):
            self.index = index

        def use(self):
            calls["device"].append(self.index)

    class FakeGradient:
        def __init__(self, ground=False):
            self.ground = ground

        def kernel(self, state=None):
            if self.ground:
                assert state is None
                return np.full((3, 3), 0.25)
            calls["gradient_states"].append(state)
            return np.full((3, 3), float(state))

    class FakeTD:
        def __init__(self, equilibrium=True):
            self.nstates = None
            self.conv_tol = None
            self.max_cycle = None
            self.converged = np.asarray([True, True])
            self.e = np.asarray([0.1, 0.2])
            self.equilibrium = equilibrium
            if not equilibrium:
                self.e += .05

        def nac_method(self):
            def kernel(states):
                calls.setdefault('nac_pairs', []).append(states)
                return [np.zeros((3,3))]*3+[np.full((3,3), .125)]
            return SimpleNamespace(kernel=kernel)

        def kernel(self):
            calls["tda_kernel"] = True

        def nuc_grad_method(self):
            return FakeGradient()

    class FakeMF:
        def __init__(self):
            self.grids = SimpleNamespace(level=None)
            self.converged = np.bool_(True)
            self.e_tot = np.asarray(-2.0)
            self.with_solvent = SimpleNamespace()

        def PCM(self):
            calls['pcm'] = self.with_solvent
            return self

        def density_fit(self, **kwargs):
            calls["density_fit"] = kwargs
            return self

        def to_gpu(self):
            calls["to_gpu"] = True
            return self

        def kernel(self):
            calls["scf_kernel"] = True

        def TDA(self, equilibrium_solvation=True):
            calls.setdefault('response_modes', []).append(equilibrium_solvation)
            calls["tda"] = FakeTD(equilibrium_solvation)
            return calls["tda"]

        def nuc_grad_method(self):
            return FakeGradient(ground=True)

    fake_mf = FakeMF()
    fake_pyscf = SimpleNamespace(
        __version__="2.test",
        lib=SimpleNamespace(num_threads=lambda count: calls["threads"].append(count)),
        M=lambda **kwargs: calls.setdefault("molecule", kwargs) or SimpleNamespace(),
    )
    # setdefault returns the kwargs dictionary, which is sufficient as the
    # opaque molecule passed into FakeRKS.
    fake_gpu4pyscf = SimpleNamespace(__version__="1.test")
    fake_dft = SimpleNamespace(RKS=lambda mol, xc: calls.setdefault("rks", (mol, xc)) and fake_mf)
    fake_cupy = SimpleNamespace(
        __version__="13.test",
        ndarray=np.ndarray,
        asnumpy=np.asarray,
        cuda=SimpleNamespace(Device=FakeDevice),
    )
    fake_nist = SimpleNamespace(HARTREE2EV=2.0, BOHR=0.5)
    modules = {
        "pyscf": fake_pyscf,
        "gpu4pyscf": fake_gpu4pyscf,
        "gpu4pyscf.dft": fake_dft,
        "cupy": fake_cupy,
        "pyscf.data.nist": fake_nist,
    }
    real_import = importlib.import_module
    monkeypatch.setattr(
        gpu4pyscf_module.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else real_import(name),
    )
    calls['mf'] = fake_mf
    return calls


def test_failed_vertical_tda_is_not_accepted_as_a_complete_label(fake_gpu_stack, monkeypatch):
    original = fake_gpu_stack['mf'].TDA
    def tda(equilibrium_solvation=True):
        value = original(equilibrium_solvation)
        if not equilibrium_solvation:
            value.converged = np.array([False,True])
        return value
    monkeypatch.setattr(fake_gpu_stack['mf'], 'TDA', tda)
    options = validate_gpu4pyscf_config(dict(nroots=2, solvent={'enabled':True}))
    options['_compute_vertical'] = True
    with pytest.raises(GPU4PySCFTDAError) as error:
        run_gpu4pyscf_states(_molecule().atoms, options, device_index=0)
    assert error.value.stage == 'vertical_tda'


def test_pcm_sequential_batch_retains_successful_members(fake_backend, tmp_path):
    from tests.helpers.driver import loaded_parsl
    from alframework.tools.labeling import LabelingCoordinator
    def run(atoms, options, **kwargs):
        if atoms.positions[0,0] < 0:
            raise GPU4PySCFTDAError('synthetic vertical failure', stage='vertical_tda')
        return (*_predictions(2), _no_nac(), _no_pairs(), {})
    fake_backend(run)
    config=dict(nroots=2, solvent={'enabled':True}, batching=dict(mode='sequential',batch_size=4))
    props=_properties(2)
    with loaded_parsl(tmp_path/'runinfo', 'alf_QM_executor'):
        coordinator=LabelingCoordinator(gpu4pyscf_excited_state_task,
            'alframework.qm_interfaces.gpu4pyscf_interface.gpu4pyscf_excited_state_task',config)
        kwargs=[dict(QM_config=config,properties_list=props,gpus_per_node=1)]
        for index in range(4):
            molecule=_molecule(str(index))
            if index==1:molecule.atoms.positions[0,0]=-1.
            coordinator.submit(molecule,kwargs)
        coordinator.submit_ready(kwargs)
        result=coordinator.queue.task_list[-1].result(timeout=30)
    assert [m.get_moleculeid() for m in result]==['0','1','2','3']
    assert [m.check_convergence() for m in result]==[True,False,True,True]


def test_pcm_routes_equilibrium_forces_nacs_and_separate_vertical_spectrum(fake_gpu_stack, monkeypatch):
    calls = fake_gpu_stack
    monkeypatch.setattr(gpu4pyscf_module, 'transition_dipoles_staged', lambda td, cupy: np.full((2,3), 1. if td.equilibrium else 2.))
    options = validate_gpu4pyscf_config(dict(nroots=2, compute_nacr=True, solvent={'enabled':True}))
    options.update(_compute_optical=True, _compute_vertical=True, _compute_vertical_optical=True)
    energy, forces, nac, pairs, metadata = run_gpu4pyscf_states(_molecule().atoms, options, device_index=0)
    assert calls['pcm'].eps == 9.02 and calls['pcm'].eps_optical == 2.03
    assert calls['response_modes'] == [True, False]
    assert calls['nac_pairs'] == [(1,2)] and calls['gradient_states'] == [1,2]
    np.testing.assert_allclose(energy, [-4.,-3.8,-3.6])
    np.testing.assert_allclose(metadata['_vertical_excitation_energies'], [.3,.5])
    np.testing.assert_allclose(metadata['_transition_dipoles'], 1.)
    np.testing.assert_allclose(metadata['_vertical_transition_dipoles'], 2.)
    np.testing.assert_allclose(nac, .25)
    assert metadata['qm_nac_solver']=='solvent_pairwise' and metadata['s1_gradient_source']=='tda_gradient'


def test_vertical_units_offset_and_hdf5_roundtrip(fake_backend, tmp_path):
    props = _properties(2, include_gap=True)
    for key in gpu4pyscf_module.VERTICAL_PROPERTIES:
        props[key] = [key, 'system', 2.]
    # All energies share the same stored scale; offsets never touch excitations.
    for key in ('sE0','sE1','sE2','dE01'):
        props[key][2] = 2.
    values = dict(_vertical_excitation_energies=np.array([.3,.5]),
        _vertical_transition_dipoles=np.ones((2,3)), _vertical_oscillator_strengths=np.array([.1,.2]))
    fake_backend(_returns(*_predictions(2), versions=values))
    molecule = label_gpu4pyscf_excited_state_molecule(_molecule(), QM_config=dict(nroots=2, solvent={'enabled':True}, energy_offset_eV=-100.), properties_list=props)
    assert molecule.check_convergence()
    assert molecule.get_results()['sE0']==10.
    np.testing.assert_allclose(molecule.get_results()['vertical_excitation_energies'], [.3,.5])
    path = tmp_path/'labels.h5'
    store_current_data(str(path), [molecule], props)
    with h5py.File(path) as handle:
        group = next(iter(handle.values()))
        np.testing.assert_allclose(group['vertical_excitation_energies'][0], [.6,1.])


@pytest.mark.parametrize('property_name,value', [('vertical_excitation_energies',[float('nan'),1.]), ('vertical_transition_dipoles',[[1,2,3]]), ('vertical_oscillator_strengths',[-1.,0.])])
def test_invalid_vertical_output_is_unconverged(fake_backend, property_name, value):
    props = _properties(2)
    props[property_name] = [property_name,'system',1.]
    fake_backend(_returns(*_predictions(2), versions={'_'+property_name:value}))
    molecule = label_gpu4pyscf_excited_state_molecule(_molecule(), QM_config={'nroots':2}, properties_list=props)
    assert not molecule.check_convergence()
    assert property_name not in molecule.get_results()


def test_backend_runs_rks_tda_for_every_state(fake_gpu_stack):
    calls = fake_gpu_stack
    options = validate_gpu4pyscf_config(
        {
            "nroots": 2,
            "num_threads": 3,
            "auxbasis": "fake-jkfit",
        }
    )

    energies, forces, nacr, nac_pairs, versions = run_gpu4pyscf_states(
        _molecule().get_atoms(),
        options,
        device_index=2,
    )

    assert calls["device"] == [2]
    assert calls["threads"] == [3]
    assert calls["density_fit"] == {"auxbasis": "fake-jkfit"}
    assert calls["gradient_states"] == [1, 2]
    assert calls["tda"].nstates == 2
    assert calls["tda"].conv_tol == pytest.approx(1.0e-8)
    assert calls["tda"].max_cycle == 200
    np.testing.assert_allclose(energies, [-4.0, -3.8, -3.6])
    np.testing.assert_allclose(forces[0], -np.ones((3, 3)))
    np.testing.assert_allclose(forces[1], -np.full((3, 3), 4.0))
    np.testing.assert_allclose(forces[2], -np.full((3, 3), 8.0))
    assert {k: versions[k] for k in ("pyscf_version", "gpu4pyscf_version", "cupy_version", "s1_gradient_source")} == {
        "pyscf_version": "2.test",
        "gpu4pyscf_version": "1.test",
        "cupy_version": "13.test",
        # compute_nacr defaults off, so S1 still comes from the TDA gradient.
        "s1_gradient_source": "tda_gradient",
    }
    assert nacr.shape == (0, 3, 3)
    assert nac_pairs.shape == (0, 2)


def test_labeling_stores_every_state(fake_backend):
    energies, forces = _predictions(2)
    seen_numbers = []

    def fake_run(atoms, options, *, device_index):
        seen_numbers.append(atoms.get_atomic_numbers().tolist())
        assert options["nroots"] == 2
        assert device_index == 1
        return (
            energies.copy(),
            forces.copy(),
            _no_nac(),
            _no_pairs(),
            {
                "pyscf_version": "fake",
                "gpu4pyscf_version": "fake",
                "cupy_version": "fake",
            },
        )

    fake_backend(fake_run)
    molecule = _molecule()
    properties = _properties(2, include_gap=True)
    labeled = label_gpu4pyscf_excited_state_molecule(
        molecule,
        QM_config={"nroots": 2, "energy_offset_eV": -100.0},
        properties_list=properties,
        sampler_config={"energy_offset_eV": -50.0},
        gpus_per_node=4,
    )

    assert seen_numbers == [[8, 1, 1]]
    assert labeled.check_convergence() is True
    assert labeled.get_results()["sE0"] == pytest.approx(10.0)
    assert labeled.get_results()["sE1"] == pytest.approx(11.0)
    assert labeled.get_results()["sE2"] == pytest.approx(12.0)
    assert labeled.get_results()["dE01"] == pytest.approx(1.0)
    for state in range(3):
        np.testing.assert_allclose(labeled.get_results()[f"F{state}"], forces[state])
    metadata = labeled.get_metadata()
    assert metadata["qm_backend"] == "gpu4pyscf"
    assert metadata["qm_device"] == "cuda:1"
    assert metadata["qm_scf_converged"] is True
    assert metadata["qm_tda_converged"] is True
    assert metadata["nroots"] == 2
    assert metadata["n_states"] == 3
    assert metadata["energy_offset_source"] == "QM_config"
    assert metadata["gap_properties"] == ["dE01"]


def test_sampler_energy_offset_is_the_fallback(fake_backend):
    energies, forces = _predictions(1)
    fake_backend(_returns(energies, forces))

    labeled = label_gpu4pyscf_excited_state_molecule(
        _molecule(),
        QM_config={"nroots": 1},
        properties_list=_properties(1),
        sampler_config={"energy_offset_eV": -100.0},
    )

    assert labeled.get_results()["sE0"] == pytest.approx(10.0)
    assert labeled.get_metadata()["energy_offset_source"] == "sampler_config"


def test_failed_run_leaves_no_partial_labels(fake_backend):

    fake_backend(_raises(GPU4PySCFTDAError("synthetic root failure", failed_states=(2,))))
    molecule = _molecule("failed")
    properties = _properties(2, include_gap=True)
    molecule.store_results({key: np.asarray([123.0]) for key in properties})

    failed = label_gpu4pyscf_excited_state_molecule(
        molecule,
        QM_config={"nroots": 2},
        properties_list=properties,
    )

    assert failed.check_convergence() is False
    assert failed.get_results() == {}
    metadata = failed.get_metadata()
    assert metadata["qm_error_type"] == "GPU4PySCFTDAError"
    assert metadata["qm_failed_stage"] == "tda"
    assert metadata["qm_failed_states"] == [2]
    assert metadata["qm_scf_converged"] is True
    assert metadata["qm_tda_converged"] is False


@pytest.mark.parametrize(
    ("error", "stage", "scf_converged", "tda_converged"),
    [
        (
            GPU4PySCFDependencyError("missing stack"),
            "dependency",
            False,
            False,
        ),
        (
            GPU4PySCFSCFError("failed SCF", failed_states=(0,)),
            "scf",
            False,
            False,
        ),
        (
            GPU4PySCFGradientError("failed gradient", failed_states=(1,)),
            "gradient",
            True,
            True,
        ),
    ],
    ids=["dependency", "scf", "gradient"],
)
def test_failed_stage_is_reported(
    fake_backend,
    error,
    stage,
    scf_converged,
    tda_converged,
):

    fake_backend(_raises(error))
    failed = label_gpu4pyscf_excited_state_molecule(
        _molecule(f"{stage}-failure"),
        QM_config={"nroots": 1},
        properties_list=_properties(1),
    )

    metadata = failed.get_metadata()
    assert failed.check_convergence() is False
    assert metadata["qm_failed_stage"] == stage
    assert metadata["qm_scf_converged"] is scf_converged
    assert metadata["qm_tda_converged"] is tda_converged


def test_malformed_output_marks_frame_failed(fake_backend):
    _, forces = _predictions(1)
    fake_backend(_returns(np.zeros(1), forces))

    failed = label_gpu4pyscf_excited_state_molecule(
        _molecule(),
        QM_config={"nroots": 1},
        properties_list=_properties(1),
    )

    assert failed.check_convergence() is False
    assert failed.get_metadata()["qm_failed_stage"] == "output"
    assert failed.get_metadata()["qm_scf_converged"] is True
    assert failed.get_metadata()["qm_tda_converged"] is True


def test_invalid_molecules_never_reach_the_gpu(monkeypatch):
    selected = []
    monkeypatch.setattr(
        gpu4pyscf_module,
        "select_gpu4pyscf_device",
        lambda count: selected.append(count),
    )
    periodic = _molecule("periodic")
    periodic.get_atoms().set_cell(np.eye(3) * 10.0)
    periodic.get_atoms().set_pbc(True)

    failed = label_gpu4pyscf_excited_state_molecule(
        periodic,
        QM_config={"nroots": 1},
        properties_list=_properties(1),
    )
    assert failed.check_convergence() is False
    assert "nonperiodic" in failed.get_metadata()["qm_error"]

    nonfinite = _molecule("nonfinite")
    nonfinite.get_atoms().positions[0, 0] = np.nan
    failed = label_gpu4pyscf_excited_state_molecule(
        nonfinite,
        QM_config={"nroots": 1},
        properties_list=_properties(1),
    )
    assert failed.check_convergence() is False
    assert "finite" in failed.get_metadata()["qm_error"]
    assert selected == []


def test_labels_can_be_saved_to_hdf5(fake_backend, tmp_path):
    energies, forces = _predictions(1)
    fake_backend(_returns(energies, forces))
    properties = _properties(1, include_gap=True)
    labeled = label_gpu4pyscf_excited_state_molecule(
        _molecule("gpu4pyscf-water"),
        QM_config={"nroots": 1},
        properties_list=properties,
    )
    path = tmp_path / "gpu4pyscf-labels.h5"

    store_current_data(str(path), [labeled], properties)

    with h5py.File(path, "r") as handle:
        group = handle["H02_O01"]
        assert group["state_0_energy"][0] == pytest.approx(-90.0)
        assert group["state_1_energy"][0] == pytest.approx(-89.0)
        assert group["gap_01"][0] == pytest.approx(1.0)
        assert group["state_0_forces"].shape == (1, 3, 3)
        assert group["state_1_forces"].shape == (1, 3, 3)


def test_task_accepts_the_coordinator_inputs(fake_backend):
    energies, forces = _predictions(1)
    fake_backend(_returns(energies, forces))
    molecule = _molecule("task-contract")
    task_input = build_input_dict(
        gpu4pyscf_excited_state_task.func,
        [
            {
                "molecule_object": molecule,
                "QM_config": {"nroots": 1},
            },
            {"properties_list": _properties(1)},
            {"sampler_config": {}},
            {"gpus_per_node": 4},
        ],
        raise_on_fail=True,
    )

    labeled = gpu4pyscf_excited_state_task.func(**task_input)

    assert labeled.check_convergence() is True
    assert set(labeled.get_results()) == {"sE0", "F0", "sE1", "F1"}
    assert gpu4pyscf_excited_state_task.executors == ["alf_QM_executor"]


def test_nac_pairs_are_ordered_combinations():
    assert excited_excited_nac_pairs(1) == ()
    assert excited_excited_nac_pairs(2) == ((1, 2),)
    assert excited_excited_nac_pairs(3) == ((1, 2), (1, 3), (2, 3))
    # nroots=5 is the production keto/indigo setting: 10 pairs.
    assert len(excited_excited_nac_pairs(5)) == 10
    assert len(excited_excited_nac_pairs(6)) == 15


def test_nacr_is_off_by_default():
    assert validate_gpu4pyscf_config({})["compute_nacr"] is False
    assert validate_gpu4pyscf_config({"compute_nacr": True, "nroots": 2})["compute_nacr"] is True

    with pytest.raises(ValueError, match="compute_nacr requires nroots >= 2"):
        validate_gpu4pyscf_config({"compute_nacr": True, "nroots": 1})
    with pytest.raises(ValueError, match="compute_nacr must be a Boolean"):
        validate_gpu4pyscf_config({"compute_nacr": "yes"})


def test_nacr_property_must_match_the_setting():
    with_nacr = _properties(2, include_nacr=True)
    without = _properties(2)

    validate_gpu4pyscf_properties(with_nacr, nroots=2, compute_nacr=True)
    validate_gpu4pyscf_properties(without, nroots=2, compute_nacr=False)

    with pytest.raises(ValueError, match="compute_nacr requires properties_list"):
        validate_gpu4pyscf_properties(without, nroots=2, compute_nacr=True)
    with pytest.raises(ValueError, match="compute_nacr is false"):
        validate_gpu4pyscf_properties(with_nacr, nroots=2, compute_nacr=False)

    wrong_scope = _properties(2)
    wrong_scope[NACR_PROPERTY_KEY] = ["nacr", "atomic", 1.0]
    with pytest.raises(ValueError, match="must use the 'pair_atomic' scope"):
        validate_gpu4pyscf_properties(wrong_scope, nroots=2, compute_nacr=True)


def test_malformed_nacr_is_rejected():
    nacr, pairs = _nac_predictions(3)
    validated_nacr, validated_pairs = gpu4pyscf_module._validated_nac_outputs(nacr, pairs, nroots=3, atom_count=3)
    np.testing.assert_allclose(validated_nacr, nacr)
    np.testing.assert_array_equal(validated_pairs, pairs)

    with pytest.raises(GPU4PySCFOutputError, match="pair ordering"):
        gpu4pyscf_module._validated_nac_outputs(nacr, pairs[::-1], nroots=3, atom_count=3)
    with pytest.raises(GPU4PySCFOutputError, match="malformed coupling vectors"):
        gpu4pyscf_module._validated_nac_outputs(nacr[:, :2], pairs, nroots=3, atom_count=3)
    with pytest.raises(GPU4PySCFOutputError, match="malformed NAC pair index"):
        gpu4pyscf_module._validated_nac_outputs(nacr, pairs[:1], nroots=3, atom_count=3)
    nonfinite = nacr.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(GPU4PySCFOutputError, match="non-finite coupling"):
        gpu4pyscf_module._validated_nac_outputs(nonfinite, pairs, nroots=3, atom_count=3)


def test_labeling_stores_nacr(fake_backend):
    energies, forces = _predictions(2)
    nacr, pairs = _nac_predictions(2)
    fake_backend(_returns(energies, forces, nacr, pairs, {"s1_gradient_source": "nac_solver"}))

    labeled = label_gpu4pyscf_excited_state_molecule(
        _molecule(),
        QM_config={"nroots": 2, "compute_nacr": True},
        properties_list=_properties(2, include_nacr=True),
    )

    assert labeled.check_convergence() is True
    results = labeled.get_results()
    np.testing.assert_allclose(results[NACR_PROPERTY_KEY], nacr)
    assert results[NACR_PROPERTY_KEY].shape == (1, 3, 3)

    metadata = labeled.get_metadata()
    assert metadata["compute_nacr"] is True
    assert metadata["nac_pairs"] == [[1, 2]]
    assert metadata["nac_scope"] == "excited_excited"
    assert metadata["nacr_units"] == "angstrom^-1"
    assert metadata["qm_nac_converged"] is True
    assert metadata["s1_gradient_source"] == "nac_solver"


def test_labeling_without_nacr_stores_none(fake_backend):
    energies, forces = _predictions(2)
    fake_backend(_returns(energies, forces))

    labeled = label_gpu4pyscf_excited_state_molecule(
        _molecule(),
        QM_config={"nroots": 2},
        properties_list=_properties(2),
    )

    assert labeled.check_convergence() is True
    assert NACR_PROPERTY_KEY not in labeled.get_results()
    metadata = labeled.get_metadata()
    assert metadata["compute_nacr"] is False
    assert "nac_pairs" not in metadata


def test_nac_failure_marks_frame_failed(fake_backend):

    fake_backend(_raises(GPU4PySCFNACError("coupling solver exploded", failed_states=(1, 2))))

    labeled = label_gpu4pyscf_excited_state_molecule(
        _molecule(),
        QM_config={"nroots": 2, "compute_nacr": True},
        properties_list=_properties(2, include_nacr=True),
    )

    assert labeled.check_convergence() is False
    metadata = labeled.get_metadata()
    assert metadata["qm_failed_stage"] == "nac"
    # SCF and TDA completed before the coupling stage; NAC did not.
    assert metadata["qm_scf_converged"] is True
    assert metadata["qm_tda_converged"] is True
    assert metadata["qm_nac_converged"] is False
    assert metadata["qm_failed_states"] == [1, 2]
    assert NACR_PROPERTY_KEY not in labeled.get_results()


def test_nacr_survives_hdf5_storage(fake_backend, tmp_path):
    energies, forces = _predictions(2)
    nacr, pairs = _nac_predictions(2)
    fake_backend(_returns(energies, forces, nacr, pairs))
    properties = _properties(2, include_nacr=True)
    labeled = label_gpu4pyscf_excited_state_molecule(
        _molecule(),
        QM_config={"nroots": 2, "compute_nacr": True},
        properties_list=properties,
    )

    h5path = str(tmp_path / "nacr.h5")
    store_current_data(h5path, [labeled], properties)

    with h5py.File(h5path, "r") as handle:
        group = handle[list(handle.keys())[0]]
        # [frames, pairs, atoms, 3]
        assert group["nacr"].shape == (1, 1, 3, 3)
        atom_order = np.argsort(labeled.get_atoms().numbers, kind="stable")
        np.testing.assert_allclose(group["nacr"][0], nacr[:, atom_order])
        # The pair index is stored once per group, not per frame.
        assert group["nac_pairs"].shape == (1, 2)
        np.testing.assert_array_equal(group["nac_pairs"][...], pairs)
        assert group["state_0_forces"].shape == (1, 3, 3)


def test_optical_properties_are_stored(fake_backend, tmp_path):
    def runner(atoms, options, **kw):
        assert options["_compute_optical"]
        return (
            np.array([-10.0, -9.0]),
            np.zeros((2, len(atoms), 3)),
            None,
            None,
            {"_transition_dipoles": np.ones((1, 3)), "_oscillator_strengths": np.array([0.2])},
        )

    fake_backend(runner)
    props = _properties(1, include_gap=True)
    props.update(
        {k: [k, "system", 1.0] for k in ("transition_dipoles", "oscillator_strengths", "absolute_state_energies")}
    )
    m = label_gpu4pyscf_excited_state_molecule(
        _molecule(), QM_config={"nroots": 1, "energy_offset_eV": -8.0}, properties_list=props
    )
    assert m.check_convergence()
    assert m.get_results()["sE0"] == -2.0
    assert m.get_results()["dE01"] == 1.0
    np.testing.assert_array_equal(m.get_results()["absolute_state_energies"], [-10, -9])
    store_current_data(str(tmp_path / "optics.h5"), [m], props)
    with h5py.File(tmp_path / "optics.h5") as h5:
        g = next(iter(h5.values()))
        assert g["transition_dipoles"].shape == (1, 1, 3)
        assert g["oscillator_strengths"][0, 0] == pytest.approx(0.2)


@pytest.mark.parametrize("strength", [-0.1, np.nan])
def test_invalid_optics_mark_frame_failed(fake_backend, strength):
    fake_backend(
        lambda *a, **k: (
            np.array([-10.0, -9.0]),
            np.zeros((2, 3, 3)),
            None,
            None,
            {"_transition_dipoles": np.ones((1, 3)), "_oscillator_strengths": np.array([strength])},
        )
    )
    props = _properties(1)
    props["oscillator_strengths"] = ["oscillator_strengths", "system", 1.0]
    m = label_gpu4pyscf_excited_state_molecule(_molecule(), QM_config={"nroots": 1}, properties_list=props)
    assert not m.check_convergence()
    assert m.get_results() == {}


def test_single_gpu_setting_applies_only_during_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "gpu4pyscf.__config__", raising=False)
    config = SimpleNamespace(num_devices=4)
    fake = SimpleNamespace(__config__=config)
    seen = []

    def load(name):
        seen.append((name, config.num_devices))
        return name

    monkeypatch.setattr(gpu4pyscf_module.importlib, "import_module", load)
    gpu4pyscf_module._single_device_backend(fake)
    assert seen == [("gpu4pyscf", 1), ("gpu4pyscf.dft", 1)]
    assert config.num_devices == 4


def test_multi_gpu_backend_already_loaded_is_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpu4pyscf.__config__", SimpleNamespace(num_devices=4))
    with pytest.raises(GPU4PySCFDeviceError, match="multi-GPU"):
        gpu4pyscf_module._single_device_backend(SimpleNamespace())


def test_tight_tda_tolerance_keeps_small_corrections():
    config = validate_gpu4pyscf_config({"tda_conv_tol": 1e-8})
    assert config["tda_lindep"] < config["tda_conv_tol"] ** 2
