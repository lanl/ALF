import numpy as np
import pytest
from ase import Atoms

from alframework.samplers.batched_mlmd import sample_trajectories
from alframework.tools.cohorts import CohortPool
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.fakes import spring_ensemble

WATER = Atoms("OH2", positions=[[0, 0, 0], [1.2, 0, 0], [0.6, 1.0392, 0]])
SAMPLER = dict(dt=0.5, maxt=0.05, Ncheck=5, distcut=0.5, min_time=0.0, Escut=0.012, Fscut=100.0, random_seed=9,
               srt_temp=[2500.0, 3500.0], end_temp=[2500.0, 3500.0], amp_temp=[0.0, 200.0],
               per_temp=[0.01, 0.05], amp_dens=None, per_dens=None, end_dens=None)


def member(name):
    return MoleculesObject(Atoms("H"), name)


def cohort_config(cohort_size, k, policy="stop", cap=None):
    selection = {"mode": "cohort_top_k", "cohort_size": cohort_size, "k": k}
    if cap is not None:
        selection["max_candidates_per_trajectory"] = cap
    return dict(SAMPLER, acquisition={"trajectory_policy": policy, "selection": selection})


def inputs(count):
    molecules = []
    for index in range(count):
        atoms = WATER.copy()
        if index % 7 == 3:
            atoms.positions[2] = atoms.positions[1] + [0.0, 0.3, 0.0]   # distcut rejection
        if index % 7 == 5:
            atoms = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]])     # low spread: few or no candidates
        molecules.append(MoleculesObject(atoms, f"mol-0000-{index:010d}"))
    return molecules


def sampled(count, config):
    """Sample every trajectory once (alone) and return its result keyed by ID."""
    results = {}
    for molecule in inputs(count):
        copy = MoleculesObject(molecule.get_atoms().copy(), molecule.get_moleculeid())
        results[copy.get_moleculeid()] = sample_trajectories([copy], spring_ensemble(), dict(config))[0]
    return results


def selected_ids(frames):
    return [(frame.get_metadata()["frame_id"], frame.get_metadata()["cohort_rank"]) for frame in frames]


def replay(results, cohort_size, k, width, order):
    """Admit all trajectories, then return them in tasks of ``width`` in the given order."""
    pool = CohortPool({"cohort_size": cohort_size, "k": k})
    ids = sorted(results)
    for name in ids:
        pool.admit(member(name), model_id=0)
    tasks = [ids[start:start + width] for start in range(0, len(ids), width)]
    if order == "reversed":
        tasks.reverse()
    elif order == "shuffled":
        tasks = [tasks[i] for i in np.random.default_rng(width).permutation(len(tasks))]
    frames = []
    for task in tasks:
        for name in (reversed(task) if order == "reversed" else task):
            pool.record(results[name])
        frames.extend(pool.ready())
    return frames, pool


@pytest.mark.parametrize("policy,cap", [("stop", None), ("continue", None), ("continue", 2)])
def test_selection_is_identical_across_widths_and_completion_orders(policy, cap):
    results = sampled(50, cohort_config(50, 8, policy, cap))
    outcomes = {r.get_metadata()["trajectory_outcome"] for r in results.values()}
    assert {"eligible", "rejected"} <= outcomes
    counts = [len(r.get_metadata()["acquisition_candidates"]) for r in results.values()]
    if policy == "continue":
        assert max(counts) > 1
        if cap is not None:
            assert max(counts) <= cap
    else:
        assert max(counts) == 1
    reference, _ = replay(results, 50, 8, 1, "forward")
    assert len(reference) == 8
    for width in (1, 2, 8, 32, 50):
        for order in ("forward", "reversed", "shuffled"):
            frames, pool = replay(results, 50, 8, width, order)
            assert selected_ids(frames) == selected_ids(reference), (width, order)
            assert pool.status()["finished"] == 1


def test_selection_is_the_global_top_k_of_all_candidates():
    results = sampled(20, cohort_config(20, 5, "continue"))
    everything = [c for r in results.values() for c in r.get_metadata()["acquisition_candidates"]]
    expected = sorted(everything, key=lambda c: (-c["ranking_score"], c["trajectory_id"], c["step"]))[:5]
    frames, _ = replay(results, 20, 5, 4, "forward")
    assert [f.get_metadata()["frame_id"] for f in frames] == [c["frame_id"] for c in expected]
    frame = frames[0]
    assert frame.get_moleculeid() == "{}-s{:09d}".format(expected[0]["trajectory_id"], expected[0]["step"])
    np.testing.assert_array_equal(frame.get_atoms().get_positions(), expected[0]["positions"])
    assert frame.get_metadata()["ranking_score"] == expected[0]["ranking_score"]


def test_worker_pruning_does_not_change_the_selection():
    unpruned = sampled(14, cohort_config(14, 50, "continue"))
    pruned = sampled(14, cohort_config(14, 3, "continue"))
    full, _ = replay(unpruned, 14, 50, 14, "forward")
    small, _ = replay(pruned, 14, 3, 14, "forward")
    assert selected_ids(small) == selected_ids(full)[:3]


def candidate(trajectory_id, step, score):
    return {"trajectory_id": trajectory_id, "step": step, "frame_id": f"{trajectory_id}:step-{step:09d}",
            "time_ps": 0.0, "ranking_score": score, "signals": {"Es": 1.0}, "flags": {"Ecrit": True},
            "numbers": np.array([1, 1]), "positions": np.zeros((2, 3)), "cell": np.zeros((3, 3)),
            "pbc": np.array([False] * 3)}


def returned(trajectory_id, outcome, candidates=()):
    molecule = member(trajectory_id)
    molecule.update_atoms(None)
    molecule.update_metadata({"trajectory_outcome": outcome, "acquisition_candidates": list(candidates)})
    return molecule


def test_ties_fewer_than_k_empty_and_failed_members():
    pool = CohortPool({"cohort_size": 4, "k": 3})
    for name in ("b", "a", "c", "d"):
        pool.admit(member(name), model_id=0)
    pool.record(returned("b", "eligible", [candidate("b", 5, 2.0)]))
    # A later failure does not discard frames the trajectory already returned.
    pool.record(returned("a", "failed", [candidate("a", 10, 2.0), candidate("a", 5, 2.0)]))
    pool.record(returned("c", "rejected"))
    assert pool.ready() == []                            # "d" has not finished
    pool.record_failed([member("d")])   # whole-task failure: no candidates
    frames = pool.ready()
    assert [f.get_metadata()["frame_id"] for f in frames] == [
        "a:step-000000005", "a:step-000000010", "b:step-000000005"]
    summary = pool.finished[0]
    assert summary["outcomes"] == {"eligible": 1, "failed": 2, "rejected": 1} and summary["candidates"] == 3
    empty = CohortPool({"cohort_size": 1, "k": 3})
    empty.admit(member("e"), model_id=0)
    empty.record(returned("e", "completed"))
    assert empty.ready() == [] and empty.finished[0]["selected"] == []


def test_duplicate_and_unknown_reports_are_rejected():
    pool = CohortPool({"cohort_size": 2, "k": 1})
    pool.admit(member("a"), model_id=0)
    with pytest.raises(ValueError, match="already admitted"):
        pool.admit(member("a"), model_id=0)
    pool.record(returned("a", "completed"))
    with pytest.raises(ValueError, match="already reported"):
        pool.record(returned("a", "completed"))
    with pytest.raises(ValueError, match="not admitted"):
        pool.record(returned("z", "completed"))


def test_cohorts_larger_than_capacity_are_admitted_incrementally():
    pool = CohortPool({"cohort_size": 10, "k": 2})
    running, selections, admitted = [], [], 0
    while admitted < 10 or running:
        while admitted < 10 and len(running) < 3:          # parallel_samplers = 3
            molecule = member(f"t{admitted}")
            pool.admit(molecule, model_id=0)
            running.append(molecule.get_moleculeid())
            admitted += 1
        name = running.pop(0)
        pool.record(returned(name, "eligible", [candidate(name, 1, float(name[1:]))]))
        selections.append(pool.ready())
    assert sum(1 for frames in selections if frames) == 1
    assert [f.get_metadata()["trajectory_id"] for f in selections[-1]] == ["t9", "t8"]


def test_model_promotion_finishes_the_open_cohort_on_its_pinned_model():
    pool = CohortPool({"cohort_size": 3, "k": 1})
    first = [member(n) for n in ("a", "b")]
    assert [pool.admit(m, model_id=0) for m in first] == [0, 0]
    # A promotion happens: the open cohort still fills on model 0 so it can complete.
    late = member("c")
    assert pool.admit(late, model_id=1) == 0
    assert late.get_metadata()["cohort_id"] == "cohort-0000-000000"
    nxt = member("d")
    assert pool.admit(nxt, model_id=1) == 1
    assert nxt.get_metadata()["cohort_id"] == "cohort-0001-000000"
    for name in ("a", "b", "c"):
        pool.record(returned(name, "eligible", [candidate(name, 1, 1.0)]))
    frames = pool.ready()
    assert [f.get_metadata()["cohort_model_id"] for f in frames] == [0]
    assert [c["cohort_id"] for c in pool.status()["open"]] == ["cohort-0001-000000"]
