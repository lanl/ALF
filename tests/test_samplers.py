"""Legacy ASE sampling: MLMD_calculator, the well potential, UDD, mlmd_sampling, and the distance
screen it shares with batched sampling.
"""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from alframework.samplers import reactive_sampler
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator, Well_Potential
from alframework.samplers.batched_mlmd import minimum_distance, sample_trajectories
from alframework.samplers.mlmd_sampling import mlmd_sampling
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import annealing_schedule
from tests.helpers.fakes import FakeLangevin, FakeUncertaintyCalculator, FixedCalculator
from tests.helpers.interface_checks import check_ase_calculator_interface, check_energy_force_results


@pytest.mark.parametrize("sampler_name", ["neb_sampler", "dimer_sampler"])
@pytest.mark.parametrize(
    "config,rank,expected",
    [({}, "2", "cuda:2"), ({"device": "cuda:1"}, "2", "cuda:1"), ({"device": "cpu"}, "2", "cpu")],
    ids=["worker_rank", "explicit_gpu", "explicit_cpu"],
)
def test_reactive_sampler_selects_its_device_like_mlmd(monkeypatch, sampler_name, config, rank, expected):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 4))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("PARSL_WORKER_RANK", rank)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    requested = []

    class Loaded(Exception):
        pass

    def load_calculators(path, device):
        requested.append(device)
        raise Loaded

    monkeypatch.setattr(reactive_sampler, "load_module_from_config", lambda *args: load_calculators)
    with pytest.raises(Loaded):
        getattr(reactive_sampler, sampler_name)(SimpleNamespace(metadata={}), config, "model-{:04d}", 0, 3, 1, 4)
    assert requested == [expected]


def test_reactive_sampler_fails_without_a_gpu_unless_cpu_is_requested(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="No GPU"):
        reactive_sampler._reactive_device({}, 4)


def test_well_pulls_outlying_atoms_inward():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    calc = Well_Potential(r_start=1.0, force=2.0, origin=[0.0, 0.0, 0.0], mass_weighted=False)

    calc.calculate(atoms, properties=["energy", "forces"])

    assert calc.results["energy"] == 2.0
    np.testing.assert_allclose(calc.results["forces"], [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])


def test_well_forces_scale_with_mass():
    atoms = Atoms("HeH", positions=[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    calc = Well_Potential(r_start=1.0, force=1.0, origin=[0.0, 0.0, 0.0])

    calc.calculate(atoms, properties=["energy", "forces"])

    masses = atoms.get_masses()
    np.testing.assert_allclose(calc.results["forces"][:, 0], -masses)


def test_well_cannot_zero_energy_or_forces():
    atoms = Atoms("H", positions=[[2.0, 0.0, 0.0]])
    calc = Well_Potential(
        r_start=1.0,
        force=2.0,
        origin=[0.0, 0.0, 0.0],
        zero_properties=["energy", "potential_energy", "forces", "stress"],
    )

    calc.calculate(atoms, properties=["energy", "forces", "stress"])

    assert calc.zero_properties == ["stress"]
    assert "energy" in calc.implemented_properties
    assert "potential_energy" in calc.implemented_properties
    assert "forces" in calc.implemented_properties
    assert "stress" in calc.implemented_properties
    assert calc.results["energy"] == 2.0 * atoms.get_masses()[0]
    np.testing.assert_allclose(calc.results["forces"], [[-2.0 * atoms.get_masses()[0], 0.0, 0.0]])
    assert calc.results["stress"] == 0


def test_mlmd_calculator_averages_the_ensemble(fixed_calculator_factory):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    model_a = fixed_calculator_factory(energy=1.0, forces=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model_b = fixed_calculator_factory(energy=3.0, forces=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    calc = MLMD_calculator([model_a, model_b])

    check_ase_calculator_interface(
        calc, atoms, ["energy", "forces", "energy_stdev", "forces_stdev_mean", "forces_stdev_max"]
    )
    check_energy_force_results(calc.results, len(atoms))
    assert calc.results["energy"] == 2.0
    np.testing.assert_allclose(calc.results["forces"], [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert calc.results["energy_stdev"] == 1.0
    assert calc.results["forces_stdev_max"] == 1.0
    assert calc.results["forces_stdev_mean"] == 1.0 / 3.0
    assert "E_en_bias" not in calc.results
    assert "F_en_bias" not in calc.results


def test_mlmd_calculator_single_model_has_zero_uncertainty(fixed_calculator_factory):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    model = fixed_calculator_factory(energy=1.5, forces=[[0.1, 0.2, 0.3]])
    calc = MLMD_calculator([model])

    calc.calculate(atoms, ["energy", "forces", "energy_stdev", "forces_stdev_mean", "forces_stdev_max"])

    assert calc.results["energy"] == 1.5
    np.testing.assert_allclose(calc.results["forces"], [[0.1, 0.2, 0.3]])
    assert calc.results["energy_stdev"] == 0.0
    assert calc.results["forces_stdev_mean"] == 0.0
    assert calc.results["forces_stdev_max"] == 0.0


def test_mlmd_calculator_applies_udd_bias(fixed_calculator_factory):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    model_a = fixed_calculator_factory(energy=1.0, forces=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model_b = fixed_calculator_factory(energy=3.0, forces=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    calc = MLMD_calculator([model_a, model_b], udd_bias_weight=0.5)

    calc.calculate(atoms, ["energy", "forces"])

    assert calc.results["energy_stdev"] == 1.0
    assert calc.results["unbiased_energy"] == 2.0
    np.testing.assert_allclose(calc.results["unbiased_forces"], [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert calc.results["E_en_bias"] == 0.5
    np.testing.assert_allclose(calc.results["F_en_bias"], [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    assert calc.results["energy"] == 2.5
    np.testing.assert_allclose(calc.results["forces"], [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]])


def test_agreeing_models_get_no_udd_bias(fixed_calculator_factory):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    model_a = fixed_calculator_factory(energy=1.5, forces=[[0.1, 0.2, 0.3]])
    model_b = fixed_calculator_factory(energy=1.5, forces=[[0.1, 0.2, 0.3]])
    calc = MLMD_calculator([model_a, model_b], udd_bias_weight=0.5)

    calc.calculate(atoms, ["energy", "forces"])

    assert calc.results["energy_stdev"] == 0.0
    assert calc.results["E_en_bias"] == 0.0
    np.testing.assert_allclose(calc.results["F_en_bias"], [[0.0, 0.0, 0.0]])
    assert calc.results["energy"] == calc.results["unbiased_energy"]
    np.testing.assert_allclose(calc.results["forces"], calc.results["unbiased_forces"])


def test_udd_bias_ignores_the_well(fixed_calculator_factory):
    atoms = Atoms("H2", positions=[[0.2, 0.0, 0.0], [0.0, 0.0, 0.7]])
    model_a = fixed_calculator_factory(energy=1.0, forces=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model_b = fixed_calculator_factory(energy=3.0, forces=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    calc = MLMD_calculator(
        [model_a, model_b],
        well_params={"r_start": 0.0, "force": 10.0, "origin": [0.0, 0.0, 0.0], "mass_weighted": False},
        udd_bias_weight=0.5,
    )

    calc.calculate(atoms, ["energy", "forces"])

    assert calc.results["energy_stdev"] == 1.0
    assert calc.results["E_en_bias"] == 0.5
    np.testing.assert_allclose(calc.results["F_en_bias"], [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    assert calc.results["unbiased_energy"] > 2.0


def test_mlmd_sampling_density_schedule_changes_the_cell(monkeypatch):
    mlmd_sampling_module = importlib.import_module("alframework.samplers.mlmd_sampling")

    monkeypatch.setattr(mlmd_sampling_module, "Langevin", FakeLangevin)

    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    molecule = MoleculesObject(atoms, "h2-density")
    mass = np.sum(atoms.get_masses())
    initial_volume = atoms.get_volume()
    density_unit_conversion = 1.66054e-24 / 1.0e-24
    starting_density_value = density_unit_conversion * mass / initial_volume

    result = mlmd_sampling_module.mlmd_sampling(
        molecule,
        FakeUncertaintyCalculator(),
        dt=1000.0,
        maxt=2.0,
        Escut=1.0,
        Fscut=1.0,
        Ncheck=1,
        Tamp=0.0,
        Tper=1.0,
        Tsrt=300.0,
        Tend=300.0,
        Ramp=0.2,
        Rper=2.0,
        Rend=2.0,
        distcut=0.1,
    )

    expected_final_density = (1.0e-24 / 1.66054e-24) * annealing_schedule(
        t=1.0,
        tmax=2.0,
        amp=0.2,
        per=2.0,
        srt=starting_density_value,
        end=2.0,
    )

    assert result.get_atoms() is None
    assert len(result.get_metadata()["denss"]) == 2
    assert np.isclose(result.get_metadata()["denss"][-1], expected_final_density)
    assert np.isclose(result.get_metadata()["cell"].volume, mass / expected_final_density)


def test_udd_task_requires_a_bias_weight(h2_atoms):
    udd_sampling_module = importlib.import_module("alframework.samplers.udd_sampling")
    molecule = MoleculesObject(h2_atoms, "h2-udd")

    with pytest.raises(ValueError, match="udd_bias_weight"):
        error = udd_sampling_module.simple_udd_sampling_task.func(
            molecule,
            {"MLMD_calculator_options": {"well_params": None}},
            "models/model-{:04d}",
            0,
            1,
        )
        error.reraise()


def test_udd_task_passes_bias_to_the_sampler(monkeypatch, h2_atoms):
    udd_sampling_module = importlib.import_module("alframework.samplers.udd_sampling")
    molecule = MoleculesObject(h2_atoms, "h2-udd")
    captured = {}

    def fake_mlmd_task(molecule_object, sampler_config, model_path, current_model_id, gpus_per_node):
        captured["molecule_object"] = molecule_object
        captured["sampler_config"] = sampler_config
        captured["model_path"] = model_path
        captured["current_model_id"] = current_model_id
        captured["gpus_per_node"] = gpus_per_node
        return molecule_object

    monkeypatch.setattr(udd_sampling_module.simple_mlmd_sampling_task, "func", fake_mlmd_task)

    original_config = {
        "udd_bias_weight": 0.45,
        "MLMD_calculator_options": {"well_params": None},
        "dt": 0.25,
    }
    result = udd_sampling_module.simple_udd_sampling_task.func(
        molecule,
        original_config,
        "models/model-{:04d}",
        7,
        4,
    )

    assert result is molecule
    assert captured["molecule_object"] is molecule
    assert captured["model_path"] == "models/model-{:04d}"
    assert captured["current_model_id"] == 7
    assert captured["gpus_per_node"] == 4
    assert captured["sampler_config"]["dt"] == 0.25
    assert captured["sampler_config"]["MLMD_calculator_options"] == {
        "well_params": None,
        "udd_bias_weight": 0.45,
    }
    assert "udd_bias_weight" not in captured["sampler_config"]
    assert original_config["MLMD_calculator_options"] == {"well_params": None}


@pytest.mark.parametrize(
    "atoms, expected",
    [
        (Atoms("H2", positions=[[0.1, 0, 0], [9.9, 0, 0]], cell=[10] * 3, pbc=True), 0.2),
        (Atoms("H2", positions=[[0.1, 0, 0], [9.9, 0, 0]], cell=[10] * 3), 9.8),
        (Atoms("H3", positions=[[0, 0, 0], [0, 0, 0], [2, 0, 0]]), 0.0),
        (Atoms("H2", positions=[[0, 0, 0], [10, 0, 0]], cell=[10] * 3, pbc=True), 0.0),
        (
            Atoms("H2", positions=[[0.1, 0.1, 0], [9.9, 9.9, 0]], cell=[10] * 3, pbc=[True, False, False]),
            np.hypot(0.2, 9.8),
        ),
        (
            Atoms(
                "H2",
                scaled_positions=[[0.1, 0.01, 0.1], [0.1, 0.99, 0.1]],
                cell=[[8, 0, 0], [2, 7, 0], [1, 0.5, 6]],
                pbc=True,
            ),
            0.02 * np.sqrt(53),
        ),
        (Atoms("He", cell=[0.1] * 3, pbc=True), np.inf),
        (Atoms(), np.inf),
    ],
    ids=[
        "periodic_image",
        "nonperiodic_cell",
        "overlapping_atoms",
        "atoms_one_cell_apart",
        "partly_periodic",
        "triclinic_cell",
        "single_atom",
        "no_atoms",
    ],
)
def test_minimum_distance_handles_every_cell_type(atoms, expected):
    positions, cell, pbc = atoms.positions.copy(), atoms.cell.copy(), atoms.pbc.copy()
    assert minimum_distance(atoms) == pytest.approx(expected)
    np.testing.assert_array_equal(atoms.positions, positions)
    np.testing.assert_array_equal(atoms.cell, cell)
    np.testing.assert_array_equal(atoms.pbc, pbc)


def _all_pairs_minimum(atoms):
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min()) if len(atoms) > 1 else float("inf")


def _random_structures():
    rng = np.random.default_rng(7)
    cubic = Atoms("H64", positions=rng.uniform(0, 8, (64, 3)), cell=[8] * 3, pbc=True)
    skewed_cell = [[7.0, 0, 0], [3.5, 6.0, 0], [2.0, 2.5, 5.5]]
    skewed = Atoms("H50", scaled_positions=rng.uniform(0, 1, (50, 3)), cell=skewed_cell, pbc=True)
    tiny = Atoms("H2", scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.7, 0.4]], cell=[2.0, 2.2, 1.9], pbc=True)
    gas = Atoms("H3", positions=rng.uniform(0, 30, (3, 3)), cell=[30] * 3, pbc=True)  # no pair within 3 A
    slab = Atoms("H40", positions=rng.uniform(0, 9, (40, 3)), cell=[9, 9, 9], pbc=[True, True, False])
    molecule = Atoms("H30", positions=rng.uniform(0, 6, (30, 3)))
    coincident = Atoms("H3", positions=[[1, 1, 1], [1, 1, 1], [4, 4, 4]], cell=[10] * 3, pbc=True)
    return dict(cubic=cubic, skewed=skewed, tiny=tiny, gas=gas, slab=slab, molecule=molecule, coincident=coincident)


@pytest.mark.parametrize("name", list(_random_structures()))
def test_minimum_distance_matches_the_all_pairs_minimum_image(name):
    atoms = _random_structures()[name]
    assert minimum_distance(atoms) == pytest.approx(_all_pairs_minimum(atoms), rel=1e-12, abs=1e-12)


def test_minimum_distance_skips_the_all_pairs_matrix_for_dense_structures(monkeypatch):
    """The O(N^2) matrix is only for structures with no pair within the search radius."""
    atoms = _random_structures()["cubic"]
    expected = _all_pairs_minimum(atoms)
    monkeypatch.setattr(Atoms, "get_all_distances", lambda *args, **kwargs: pytest.fail("all-pairs matrix used"))
    assert minimum_distance(atoms) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "batched"])
@pytest.mark.parametrize(
    "positions, periodic, cutoff, rejected, expected",
    [
        ([[0.1, 0, 0], [9.9, 0, 0]], True, 0.5, True, 0.2),
        ([[0.1, 0, 0], [9.9, 0, 0]], False, 0.5, False, 9.8),
        ([[0, 0, 0], [0, 0, 0]], False, 0.5, True, 0.0),
        ([[0, 0, 0], [10, 0, 0]], True, 0.5, True, 0.0),
        ([[0, 0, 0], [1, 0, 0]], False, 1.0, False, 1.0),
    ],
    ids=[
        "across_periodic_boundary",
        "same_distance_without_periodicity",
        "overlapping_atoms",
        "overlapping_periodic_image",
        "exactly_at_cutoff",
    ],
)
def test_close_contacts_are_rejected(legacy, positions, periodic, cutoff, rejected, expected):
    atoms = Atoms("H2", positions=positions, cell=[10] * 3, pbc=periodic)
    molecule = MoleculesObject(atoms, "screen")
    models = [FixedCalculator(0.0, np.zeros((2, 3))) for _ in range(2)]
    config = dict(dt=0.5, maxt=0.001, Ncheck=1, Escut=1.0, Fscut=1.0, distcut=cutoff, min_time=0.0)
    if legacy:
        result = mlmd_sampling(
            molecule,
            MLMD_calculator(models, well_params=None),
            **config,
            Tamp=0.0,
            Tper=1.0,
            Tsrt=0.0,
            Tend=0.0,
            Ramp=None,
            Rper=None,
            Rend=None,
        )
    else:
        config.update(
            srt_temp=[0.0, 0.0],
            end_temp=[0.0, 0.0],
            amp_temp=[0.0, 0.0],
            per_temp=[1.0, 1.0],
            amp_dens=None,
            per_dens=None,
            end_dens=None,
            MLMD_calculator_options={"well_params": None},
        )
        (result,) = sample_trajectories([molecule], models, config)
        assert result.get_metadata()["trajectory_outcome"] == ("rejected" if rejected else "completed")
    assert bool(result.get_metadata()["distcrit"]) is rejected
    assert result.get_metadata()["distmin"] == pytest.approx(expected)
    assert result.get_atoms() is None
