"""Exercise the online data path across sizes, isomers, label shards and checkpoints."""

import json
from pathlib import Path

import numpy as np
import pytest

from alframework.builders.external_geometry import molecular_collection_builder_task
from alframework.ml_interfaces.hippynn import interface as ml
from alframework.samplers.acquisition import acquisition_settings
from alframework.samplers.batched_mlmd import sample_trajectories
from alframework.tools.cohorts import CohortPool
from alframework.tools.config import load_config_file
from alframework.tools.data_shards import store_current_data
from alframework.tools.molecules import MOLECULE_ID
from tests.helpers.hippynn import TINY_NETWORK


def test_mixed_excited_state_acquisition_then_replay(tmp_path):
    pytest.importorskip('rdkit')
    pytest.importorskip('hippynn')
    from examples.molecular_excited_states.prepare import PROPERTIES, prepare
    from alframework.ml_interfaces.hippynn.ntk import load_hippynn_excited_state_ntk_estimator

    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps(['C=O', 'CCC=O', 'CC(=O)C']))
    run = prepare(panel, tmp_path / 'run', mode='random', pool_size=12, seed_size=8, local=True)
    builder = load_config_file(run / 'builder_config.json', str(run) + '/')
    ids = [f'bootstrap-{i:010d}' for i in range(24)]
    frames = molecular_collection_builder_task.func(moleculeids=ids, builder_config=builder, master_directory=run)
    assert [frame.get_metadata()[MOLECULE_ID] for frame in frames[:3]] == ['C=O', 'CCC=O', 'CC(C)=O']
    assert {len(frame) for frame in frames} == {4, 10}

    def label(frame):
        positions = frame.get_atoms().positions
        centered = positions - positions.mean(axis=0)
        forces = -.2 * centered
        energy = -float(frame.get_atoms().numbers.sum()) * 5. + .1 * np.square(centered).sum()
        frame.store_results(dict(E0=energy, E1=energy + 2., F0=forces, F1=forces, dE01=2.))
        frame.set_converged_flag(True)

    for frame in frames:
        label(frame)
    label_path = run / 'data' / 'data-0000.h5'
    label_path.parent.mkdir()
    store_current_data(str(label_path), frames, PROPERTIES)
    arrays, data = ml.load_excited_state_h5_arrays(str(label_path.parent), PROPERTIES, mixed_molecules=True)
    assert len(data['frames']) == len(frames)
    assert set(np.count_nonzero(arrays['species'], axis=1)) == {4, 10}
    np.testing.assert_allclose(arrays['dE01'], 2.)
    by_id = {frame.get_moleculeid(): frame for frame in frames}
    for idx, record in enumerate(data['frames']):
        source = by_id[record['id']]
        np.testing.assert_allclose(arrays['F1'][idx, :len(source)], source.get_results()['F1'])

    config = dict(n_models=1, network_choice=0, network_params=dict(TINY_NETWORK, possible_species=[0, 1, 6, 8]),
                  mixed_molecules=True, device_string='cpu', hippynn_kernel='pytorch',
                  max_epochs=1, batch_size=4, eval_batch_size=4, torch_threads=1, fraction_train_eval=1.,
                  exports={'pdf': False, 'png': False}, gap_targets={'enabled': True})
    directory = run / 'models' / 'model-0000'
    member = directory / 'model-00'
    ml.train_single_excited_state_model(model_id=0, model_dir=str(member), h5_train_dir=str(label_path.parent),
                                        properties_list=PROPERTIES, ML_config=config)
    manifest = json.loads((member / 'training_input_manifest.json').read_text())
    assert set(manifest['elemental_energies_eV']) == {'1', '6', '8'}
    assert manifest['mixed_molecules']
    models = ml.load_hippynn_excited_state_ensemble(directory, 'cpu', PROPERTIES, hippynn_kernel='pytorch')
    probes = [frames[0].get_atoms(), frames[1].get_atoms()]
    together = models.predict_states(probes)
    for i, atoms in enumerate(probes):
        separately = models.predict_states([atoms])
        np.testing.assert_allclose(together['state_energies'][:, :, i], separately['state_energies'][:, :, 0],
                                   atol=5e-5, rtol=1e-6)
        np.testing.assert_allclose(together['state_forces'][i], separately['state_forces'][0], atol=5e-5, rtol=1e-5)
    estimate = load_hippynn_excited_state_ntk_estimator(directory, PROPERTIES, sketch_dim=4)
    n_atoms = sum(len(by_id[data['frames'][index]['id']]) for index in manifest['splits']['train'])
    assert estimate.metadata['rows']['force'] == n_atoms * 2 * 3

    replay = molecular_collection_builder_task.func(
        moleculeids=[f'trajectory-{i:010d}' for i in range(16)], builder_config=builder,
        current_h5_id=1, h5_path=str(run / 'data/data-{:04d}.h5'), master_directory=run)
    assert any(Path(frame.get_metadata()['geometry_source']) == label_path for frame in replay)
    sampler = load_config_file(run / 'sampler_config.json', str(run) + '/')
    sampler.update(driver={'engine': 'ase', 'strategy': 'batched'}, maxt=.002, Ncheck=2, min_time=.0005,
                   max_force_cutoff=None, metadata_files='none', screen_audit_dir=str(run / 'audit'),
                   state_selection={'mode': 'fixed', 'state': 1})
    sampler['acquisition'].update(selection={'mode': 'cohort_top_k', 'cohort_size': 4, 'k': 2})
    cohort = CohortPool(acquisition_settings(sampler)['selection'])
    for frame in replay[:4]:
        cohort.admit(frame, 0)
    for frame in sample_trajectories(replay[:4], models, sampler, device='cpu'):
        cohort.record(frame)
    acquired = cohort.ready()
    assert len(acquired) == 2
    for frame in acquired:
        label(frame)
    next_shard = label_path.with_name('data-0001.h5')
    store_current_data(str(next_shard), acquired, PROPERTIES)
    _, retraining = ml.load_excited_state_h5_arrays(str(label_path.parent), PROPERTIES, mixed_molecules=True)
    assert len(retraining['frames']) == len(frames) + len(acquired)
    assert {row[MOLECULE_ID] for row in retraining['frames']} == {'C=O', 'CCC=O', 'CC(C)=O'}
