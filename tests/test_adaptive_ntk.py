"""Selection regressions independent of the model, workers, and reference labels."""

import numpy as np
import pytest

from alframework.samplers.acquisition import acquisition_settings, select_candidates
from alframework.samplers.adaptive_ntk import select_ntk_candidates
from alframework.samplers.batched_mlmd import _prune


def candidate(index, row):
    return {
        "trajectory_id": "trajectory", "step": index,
        "energy_ntk": {"basis_id": "frozen-model", "ridge": 0.25, "row": np.array(row)},
    }


def test_duplicate_rows_lose_priority_and_ties_use_frame_identity():
    a, b, c = [candidate(i, row) for i, row in enumerate(([2., 0.], [2., 0.], [0., 1.8]))]
    pool = [b, c, a]
    fixed = select_ntk_candidates(pool, 2, adaptive=False)
    adaptive = select_candidates(pool, {"selector": "adaptive_ntk", "k": 2})
    assert [item["step"] for item in fixed] == [0, 1]
    assert [item["step"] for item in adaptive] == [0, 2]
    assert adaptive[1]["adaptive_ntk"]["selection_score"] == pytest.approx(0.81)
    assert len(select_ntk_candidates(pool, 8)) == 3
    pool[0]["energy_ntk"]["basis_id"] = "different-checkpoint"
    with pytest.raises(ValueError, match="one fitted basis"):
        select_ntk_candidates(pool, 2)


def test_rank_one_scores_match_fresh_solves_past_the_refactor_boundary():
    rng = np.random.default_rng(8)
    rows = rng.normal(size=(43, 6))
    pool = [candidate(i, row) for i, row in enumerate(rows)]
    precision = np.eye(rows.shape[1])
    remaining = list(range(len(rows)))
    expected = []
    for _ in range(36):
        trial = rows[remaining]
        scores = 0.25 * np.einsum("ij,ij->i", trial, np.linalg.solve(precision, trial.T).T)
        index = int(np.argmax(scores))
        chosen = remaining.pop(index)
        expected.append((chosen, scores[index]))
        precision += np.outer(rows[chosen], rows[chosen])

    result = select_ntk_candidates(pool[::-1], 36)
    assert [item["step"] for item in result] == [index for index, _ in expected]
    np.testing.assert_allclose([item["adaptive_ntk"]["selection_score"] for item in result],
                               [score for _, score in expected], rtol=1e-11)
    assert [item["adaptive_ntk"]["selection_order"] for item in result] == list(range(36))
    np.testing.assert_array_equal([item["energy_ntk"]["row"] for item in pool], rows)
    assert all("adaptive_ntk" not in item for item in pool)
    assert [item["step"] for item in select_ntk_candidates(pool, 36)] == [item["step"] for item in result]


def test_md_cap_requires_configuration_and_keeps_initial_energy_scores():
    selection = {"mode": "cohort_top_k", "cohort_size": 2, "k": 1, "selector": "adaptive_ntk"}
    config = {"Escut": 0.1, "Fscut": 0.5, "acquisition": {"selection": selection}}
    with pytest.raises(ValueError, match="max_candidates_per_trajectory"):
        acquisition_settings(config)
    selection["max_candidates_per_trajectory"] = 2
    settings = acquisition_settings(config)["selection"]
    pool = [dict(candidate(i, [i]), ranking_score=3 - i) for i in range(3)]
    # The cap is independent of k and of a scalar used for eligibility.
    assert [item["step"] for item in _prune(pool, settings)] == [2, 1]
