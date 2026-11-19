"""Progress plots run in their own process, one update at a time, with the coordinator's arguments."""

import json

from alframework.tools.progress_plots import ProgressPlots


def test_an_update_runs_the_utility_in_a_new_process_with_its_arguments(tmp_path):
    plots = ProgressPlots('tests.helpers.fakes.write_plot_marker')
    assert plots.update({'plot_dir': str(tmp_path), 'meta_n_cols': 3})
    plots.wait(timeout=60)
    assert plots.process.returncode == 0
    marker = json.loads((tmp_path / 'marker.json').read_text())
    assert marker['meta_n_cols'] == 3 and marker['pid'] != __import__('os').getpid()


def test_an_update_is_skipped_while_the_previous_one_runs(tmp_path):
    plots = ProgressPlots('tests.helpers.fakes.write_plot_marker')
    assert plots.update({'plot_dir': str(tmp_path), 'meta_n_cols': 1, 'sleep': 3})
    assert not plots.update({'plot_dir': str(tmp_path), 'meta_n_cols': 2})
    plots.wait(timeout=60)
    assert plots.update({'plot_dir': str(tmp_path), 'meta_n_cols': 2})
    plots.wait(timeout=60)
    assert json.loads((tmp_path / 'marker.json').read_text())['meta_n_cols'] == 2
