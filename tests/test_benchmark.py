"""The GPU sizing benchmark: its suggestions from measurements, reading a run directory, and a tiny CPU run.

The suggestion tests use made-up measurements, so they check the rules, not any GPU.
"""

import json

import pytest

from alframework.benchmark import recommend


def _train(profile, batch, step, reserved=1.0, result="passed"):
    return dict(kind="training", profile=profile, batch=batch, train_step_s=step, peak_reserved_gib=reserved,
                result=result)


def _sample(profile, count, step, reserved=1.0, result="passed"):
    return dict(kind="sampling", profile=profile, trajectories=count, step_s=step, peak_reserved_gib=reserved,
                result=result)


def test_training_suggestions_pick_the_smallest_batch_near_the_best_throughput():
    # Throughput 100, 160, 190, 200 structures/s; 64 reaches 95% of the best, and 128 ran out of memory.
    records = [_train("hipnn64", 16, 0.16), _train("hipnn64", 32, 0.2), _train("hipnn64", 64, 64 / 190),
               _train("hipnn64", 128, 128 / 200), _train("hipnn64", 256, None, result="oom")]
    suggestion = recommend.training_suggestions(records, memory_gib=40, train_rows=6400, epochs=10, walltime_hours=1)
    assert suggestion["hipnn64"]["batch_size"] == 64
    assert suggestion["hipnn64"]["fitting_batches"] == [16, 32, 64, 128]
    assert suggestion["hipnn64"]["epoch_minutes"] == pytest.approx(100 * (64 / 190) / 60)
    assert suggestion["hipnn64"]["fits_walltime"]


def test_training_suggestions_leave_out_batches_over_the_memory_share():
    records = [_train("hiphop145", 8, 0.1, reserved=28), _train("hiphop145", 16, 0.15, reserved=37)]
    suggestion = recommend.training_suggestions(records, memory_gib=40, memory_fraction=0.8)["hiphop145"]
    assert suggestion["batch_size"] == 8 and suggestion["fitting_batches"] == [8]


def test_gpus_per_member_splits_the_run_batch_into_shares_that_fit():
    fitting = {8: {}, 16: {}}
    assert recommend.gpus_for_batch(16, fitting) == 1
    assert recommend.gpus_for_batch(32, fitting) == 2
    assert recommend.gpus_for_batch(64, fitting) == 4
    assert recommend.gpus_for_batch(36, fitting) is None, "no divisor up to 8 gives a share that was measured"


def test_sampling_suggestions_give_trajectories_per_gpu_and_parallel_samplers():
    records = [_sample("hipnn64", 1, 0.01), _sample("hipnn64", 16, 0.02), _sample("hipnn64", 64, 0.07),
               _sample("hipnn64", 256, None, result="oom")]
    suggestion = recommend.sampling_suggestions(records, memory_gib=40, sampler_gpus=4)["hipnn64"]
    assert suggestion["trajectories_per_gpu"] == 64 and suggestion["parallel_samplers"] == 256


def test_benchmark_reads_a_run_directory(tmp_path):
    from alframework.benchmark.__main__ import read_run

    (tmp_path / "master_config.json").write_text(json.dumps(dict(
        ML_config_path="ml.json", sampler_config_path="sampler.json", parallel_samplers=64,
        properties_list={"E0": [], "F0": [], "E1": [], "F1": [], "dE01": []})))
    (tmp_path / "ml.json").write_text(json.dumps(dict(network_choice=0, n_models=4, batch_size=64, gpus_per_member=2,
                                                      network_params={"n_features": 64, "dist_hard_max": 8.0})))
    (tmp_path / "sampler.json").write_text(json.dumps(dict(batching={"batch_size": 16})))
    (tmp_path / "reference.xyz").write_text("1\n\nH 0 0 0\n")
    run = read_run(tmp_path)
    assert (run["states"], run["members"], run["batch_size"], run["gpus_per_member"]) == (2, 4, 64, 2)
    assert (run["sampler_batch"], run["parallel_samplers"]) == (16, 64)
    assert run["network"]["network_params"]["dist_hard_max"] == 8.0
    assert run["structure"] == str(tmp_path / "reference.xyz")


@pytest.mark.parametrize("states", [1, 2])
def test_benchmark_runs_on_cpu_and_writes_every_measurement(tmp_path, states, capsys):
    pytest.importorskip("hippynn")
    from alframework.benchmark.__main__ import main

    output = tmp_path / "sizing.json"
    main(["--device", "cpu", "--atoms", "6", "--profiles", "hipnn64", "--states", str(states), "--members", "2",
          "--train-batches", "2", "4", "--sample-batches", "1", "2", "--steps", "1", "--warmup", "0",
          "--output", str(output)])
    result = json.loads(output.read_text())
    assert [r["result"] for r in result["records"]] == ["passed"] * 4, result["records"]
    assert result["training"]["hipnn64"]["batch_size"] in (2, 4)
    assert result["sampling"]["hipnn64"]["trajectories_per_gpu"] in (1, 2)
    printed = capsys.readouterr().out
    assert "Training (one GPU)" in printed and "Suggestions" in printed
