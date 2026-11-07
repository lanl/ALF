"""Sampling windows: moving one run from base sampling to UDD or gap-seeking as labels are stored.

Runs the real coordinator and real trajectories against analytic fakes, so it needs no QM or trained
model. Tests that step trajectories need torch, and the packed-engine test skips engines not installed.
"""

import json

import numpy as np
import pytest
from ase import Atoms
from parsl import python_app

from alframework.samplers import batched_mlmd
from alframework.samplers.batched_mlmd import sample_trajectories
from alframework.samplers.mlmd_sampling import simple_mlmd_sampling_task
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_coordinator import SamplerCoordinator
from alframework.tools.sampling_windows import validated_windows, window_number, windowed_config
from tests.helpers.driver import MODEL_PATTERN, run_alf_driver
from tests.helpers.engines import DEVICE
from tests.helpers.fakes import CrossingStates
from tests.helpers.molecules import WATER

CUTOFFS = {"Escut": 1000.0, "Fscut": 1000.0}
UDD = {"MLMD_calculator_options": {"udd_bias_weight": 0.1}}
GAP_SEEKING = {"gap_seeking": {"enabled": True, "trigger_gap_threshold_eV": 0.05, "sigma": 3.5, "alpha_eV": 0.05}}
TWO_WINDOWS = [{"until_stored_labels": 10}, UDD]


def coordinator_for(**settings):
    return SamplerCoordinator(batched_mlmd.mlmd_trajectories_task, "batched", {**CUTOFFS, **settings})


def test_sampling_windows_are_optional():
    assert validated_windows({}) is None
    config = dict(CUTOFFS)
    assert coordinator_for().apply_window(config, 100) is config


@pytest.mark.parametrize(
    "windows,message",
    [
        ("later", "nonempty list"),
        ([], "nonempty list"),
        (["base"], "dictionary"),
        ([{"until_stored_labels": 10, "dt": 0.5}, {}], "can't set"),
        ([{"until_stored_labels": 10}], "must not set"),
        ([{}, {}], "requires an integer"),
        ([{"until_stored_labels": 0}, {}], "requires an integer"),
        ([{"until_stored_labels": 10.5}, {}], "requires an integer"),
        ([{"until_stored_labels": True}, {}], "requires an integer"),
        ([{"until_stored_labels": 10}, {"until_stored_labels": 10}, {}], "requires an integer"),
    ],
    ids=[
        "not_a_list",
        "empty",
        "window_not_a_dictionary",
        "unsupported_setting",
        "last_has_a_limit",
        "missing_limit",
        "zero_limit",
        "float_limit",
        "boolean_limit",
        "limits_not_increasing",
    ],
)
def test_sampling_windows_reject_invalid_lists(windows, message):
    with pytest.raises(ValueError, match=message):
        validated_windows({"sampling_windows": windows})


@pytest.mark.parametrize(
    "stored,expected",
    [(0, 0), (9, 0), (10, 1), (19, 1), (20, 2), (1000, 2)],
    ids=["none", "below_first", "at_first", "below_second", "at_second", "far_above"],
)
def test_sampling_window_changes_at_each_limit(stored, expected):
    windows = [{"until_stored_labels": 10}, {"until_stored_labels": 20}, {}]
    assert window_number(windows, stored) == expected


def test_sampling_window_replaces_settings_without_changing_the_base():
    base = {**CUTOFFS, "MLMD_calculator_options": {"well_params": {"r_start": 3.0}}}
    windows = [{"until_stored_labels": 10}, UDD]
    first, second = windowed_config(base, windows, 0), windowed_config(base, windows, 1)
    assert first["MLMD_calculator_options"] == base["MLMD_calculator_options"] and first["sampling_window"] == 0
    assert second["MLMD_calculator_options"] == {"udd_bias_weight": 0.1} and second["sampling_window"] == 1
    assert "until_stored_labels" not in first
    assert "sampling_window" not in base and base["MLMD_calculator_options"] == {"well_params": {"r_start": 3.0}}
    assert windows == [{"until_stored_labels": 10}, UDD]


def test_sampling_window_does_not_carry_over_from_the_previous_one():
    windows = [{"until_stored_labels": 10, **UDD}, {"until_stored_labels": 20, **GAP_SEEKING}, {}]
    assert "MLMD_calculator_options" not in windowed_config(CUTOFFS, windows, 1)
    assert "gap_seeking" not in windowed_config(CUTOFFS, windows, 2)


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"sampling_windows": [{"until_stored_labels": 10}, GAP_SEEKING]}, "excited_state"),
        (
            {
                "model_mode": "excited_state",
                "MLMD_calculator_options": {"udd_bias_weight": 0.1},
                "sampling_windows": [{"until_stored_labels": 10, "MLMD_calculator_options": {}}, GAP_SEEKING],
            },
            "udd_bias_weight",
        ),
        ({"sampling_windows": [{"until_stored_labels": 10}, {"gap_seeking": {"enabled": True}}]}, "requires"),
    ],
    ids=["ground_state", "with_udd", "missing_settings"],
)
def test_sampling_window_with_bad_settings_stops_the_run_at_startup(settings, message):
    with pytest.raises(ValueError, match=message):
        coordinator_for(**settings)


def test_sampling_windows_need_a_multi_trajectory_sampler():
    with pytest.raises(ValueError, match="molecule_objects"):
        SamplerCoordinator(simple_mlmd_sampling_task, "mlmd", {**CUTOFFS, "sampling_windows": TWO_WINDOWS})


def test_sampling_windows_are_fixed_on_reload():
    coordinator = coordinator_for(sampling_windows=TWO_WINDOWS)
    edited = {**CUTOFFS, "sampling_windows": [{"until_stored_labels": 99}, {}]}
    assert coordinator.keep_startup_settings(edited)["sampling_windows"] == TWO_WINDOWS


def test_sampling_window_change_is_announced_once(capsys):
    coordinator = coordinator_for(sampling_windows=TWO_WINDOWS)
    base = {**CUTOFFS, "sampling_windows": TWO_WINDOWS}
    coordinator.apply_window(base, 0)
    assert "window 0" in capsys.readouterr().out
    coordinator.apply_window(base, 5)
    assert capsys.readouterr().out == ""
    coordinator.apply_window(base, 12)
    assert "window 1" in capsys.readouterr().out


def test_coordinator_moves_through_sampling_windows_as_labels_are_stored(tmp_path, monkeypatch):
    start = {
        **CUTOFFS,
        "driver": {"strategy": "batched"},
        "batching": {"batch_size": 2, "max_wait_seconds": 0.05},
        "sampling_windows": TWO_WINDOWS,
    }
    seen = []

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(WATER.copy(), moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_objects, sampler_config):
        seen.append((sampler_config["sampling_window"], sampler_config.get("MLMD_calculator_options")))
        return molecule_objects

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        molecule_object.store_results({"energy": -1.0, "forces": np.zeros((len(molecule_object), 3))})
        molecule_object.set_converged_flag(True)
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        (tmp_path / MODEL_PATTERN.format(current_training_id)).mkdir(parents=True)
        # Edit the windows mid-run, as a user might. This must not move the switch.
        edited = dict(start, sampling_windows=[{"until_stored_labels": 100000}, {}])
        (tmp_path / "sampler.json").write_text(json.dumps(edited))
        return [True] * 4, current_training_id

    run = run_alf_driver(
        tmp_path,
        monkeypatch,
        builder=builder,
        sampler=sampler,
        qm=qm,
        ml=ml,
        sampler_config=start,
        stop_when=lambda status: status.get("lifetime_stored_labels", 0) >= 16,
        timeout=60,
        bootstrap_set=2,
        parallel_samplers=4,
        target_queued_QM=4,
        save_h5_threshold=1,
    )

    assert run.status["lifetime_stored_labels"] >= 16
    numbers = [number for number, _ in seen]
    assert numbers == sorted(numbers), "a run must not move back to an earlier window"
    assert 0 in numbers and 1 in numbers
    assert all(options is None for number, options in seen if number == 0)
    assert all(options == {"udd_bias_weight": 0.1} for number, options in seen if number == 1)


NEAR_CROSSING = [[0.01, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]]


def sample_in_window(engine, stored_labels):
    settings = dict(
        dt=0.1,
        maxt=0.001,
        Ncheck=2,
        distcut=0.1,
        min_time=0.0,
        random_seed=4,
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
        model_mode="excited_state",
        state_selection={"mode": "fixed", "state": 1},
        driver={"strategy": "batched", "engine": engine},
        batching={"batch_size": 2},
        sampling_windows=[{"until_stored_labels": 10}, GAP_SEEKING],
        **CUTOFFS,
    )
    config = coordinator_for(**settings).apply_window(settings, stored_labels)
    molecule = MoleculesObject(Atoms("OH2", positions=NEAR_CROSSING), "mol-0")
    return sample_trajectories([molecule], CrossingStates(), config, device=DEVICE)[0].get_metadata()


def check_gap_seeking_starts_in_the_later_window(engine):
    before, after = sample_in_window(engine, 9), sample_in_window(engine, 10)
    assert before["sampling_window"] == 0 and "gap_seeking_switched" not in before
    assert after["sampling_window"] == 1 and after["gap_seeking_switched"] is True
    assert not np.allclose(before["positions"], after["positions"])


def test_gap_seeking_starts_in_its_sampling_window_on_ase():
    pytest.importorskip("torch")
    check_gap_seeking_starts_in_the_later_window("ase")


def test_gap_seeking_starts_in_its_sampling_window_on_packed_engines(engine_name):
    check_gap_seeking_starts_in_the_later_window(engine_name)


def test_cohort_frames_record_their_sampling_window():
    pytest.importorskip("torch")
    settings = dict(
        dt=0.1,
        maxt=0.001,
        Ncheck=2,
        distcut=0.1,
        min_time=0.0,
        random_seed=4,
        Escut=1e-6,
        Fscut=1000.0,
        srt_temp=[0.0, 0.0],
        end_temp=[0.0, 0.0],
        amp_temp=[0.0, 0.0],
        per_temp=[1.0, 1.0],
        amp_dens=None,
        per_dens=None,
        end_dens=None,
        model_mode="excited_state",
        state_selection={"mode": "fixed", "state": 1},
        acquisition={"trajectory_policy": "continue", "selection": {"mode": "cohort_top_k", "cohort_size": 1, "k": 5}},
        sampling_window=1,
    )
    molecule = MoleculesObject(Atoms("OH2", positions=NEAR_CROSSING), "mol-0")
    result = sample_trajectories([molecule], CrossingStates(), settings)[0]
    candidates = result.get_metadata()["acquisition_candidates"]
    assert candidates and all(candidate["metadata"]["sampling_window"] == 1 for candidate in candidates)
