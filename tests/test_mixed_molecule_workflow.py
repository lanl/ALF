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


def stand_in_label(frame):
    """Two states 2 eV apart on a harmonic well, with one optical transition."""
    positions = frame.get_atoms().positions
    centered = positions - positions.mean(axis=0)
    forces = -.2 * centered
    energy = -float(frame.get_atoms().numbers.sum()) * 5. + .1 * np.square(centered).sum()
    frame.store_results(dict(E0=energy, E1=energy + 2., F0=forces, F1=forces, dE01=2.,
                             oscillator_strengths=np.array([.2]), transition_dipoles=np.ones((1, 3))))
    frame.set_converged_flag(True)


@pytest.fixture(scope='module')
def trained_run(tmp_path_factory):
    """A prepared three-molecule run: its bootstrap built, stand-in labeled, stored, and one HipNN member trained."""
    pytest.importorskip('rdkit')
    pytest.importorskip('hippynn')
    from types import SimpleNamespace
    from examples.molecular_excited_states.prepare import prepare
    from examples.molecular_excited_states.evaluate import run_properties

    tmp_path = tmp_path_factory.mktemp('mixed')
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps(['C=O', 'CCC=O', 'CC(=O)C',
                                 {'smiles': 'O=CO', 'split': 'validation'},
                                 {'smiles': 'CC=O', 'split': 'holdout'}]))
    run = prepare(panel, tmp_path / 'run', mode='random', pool_size=12, seed_size=8, local=True, optical=True)
    properties = run_properties(run)
    builder = load_config_file(run / 'builder_config.json', str(run) + '/')
    ids = [f'bootstrap-{i:010d}' for i in range(24)]
    frames = molecular_collection_builder_task.func(moleculeids=ids, builder_config=builder, master_directory=run)
    for frame in frames:
        stand_in_label(frame)
    label_path = run / 'data' / 'data-0000.h5'
    label_path.parent.mkdir()
    store_current_data(str(label_path), frames, properties, h5_options={'group_by': 'atom_count'})
    config = dict(n_models=1, network_choice=0, network_params=dict(TINY_NETWORK, possible_species=[0, 1, 6, 8]),
                  mixed_molecules=True, device_string='cpu', hippynn_kernel='pytorch',
                  max_epochs=1, batch_size=4, eval_batch_size=4, torch_threads=1, fraction_train_eval=1.,
                  exports={'pdf': False, 'png': False}, gap_targets={'enabled': True})
    directory = run / 'models' / 'model-0000'
    member = directory / 'model-00'
    ml.train_single_excited_state_model(model_id=0, model_dir=str(member), h5_train_dir=str(label_path.parent),
                                        properties_list=properties, ML_config=config)
    return SimpleNamespace(run=run, properties=properties, builder=builder, frames=frames, label_path=label_path,
                           config=config, directory=directory, member=member)


def test_mixed_excited_state_acquisition_then_replay(trained_run):
    from examples.molecular_excited_states.prepare import state_properties
    from alframework.ml_interfaces.hippynn.ntk import load_hippynn_excited_state_ntk_estimator

    run, properties, frames, label_path = trained_run.run, trained_run.properties, trained_run.frames, \
        trained_run.label_path
    assert properties == state_properties(1, optical=True)
    assert [frame.get_metadata()[MOLECULE_ID] for frame in frames[:3]] == ['C=O', 'CCC=O', 'CC(C)=O']
    assert {len(frame) for frame in frames} == {4, 10}

    arrays, data = ml.load_excited_state_h5_arrays(str(label_path.parent), properties, mixed_molecules=True)
    assert len(data['frames']) == len(frames)
    assert set(np.count_nonzero(arrays['species'], axis=1)) == {4, 10}
    np.testing.assert_allclose(arrays['dE01'], 2.)
    by_id = {frame.get_moleculeid(): frame for frame in frames}
    for idx, record in enumerate(data['frames']):
        source = by_id[record['id']]
        np.testing.assert_allclose(arrays['F1'][idx, :len(source)], source.get_results()['F1'])

    manifest = json.loads((trained_run.member / 'training_input_manifest.json').read_text())
    assert set(manifest['elemental_energies_eV']) == {'1', '6', '8'}
    assert manifest['mixed_molecules']
    models = ml.load_hippynn_excited_state_ensemble(trained_run.directory, 'cpu', properties, hippynn_kernel='pytorch')
    probes = [frames[0].get_atoms(), frames[1].get_atoms()]
    together = models.predict_states(probes)
    for i, atoms in enumerate(probes):
        separately = models.predict_states([atoms])
        np.testing.assert_allclose(together['state_energies'][:, :, i], separately['state_energies'][:, :, 0],
                                   atol=5e-5, rtol=1e-6)
        np.testing.assert_allclose(together['state_forces'][i], separately['state_forces'][0], atol=5e-5, rtol=1e-5)
    estimate = load_hippynn_excited_state_ntk_estimator(trained_run.directory, properties, sketch_dim=4)
    n_atoms = sum(len(by_id[data['frames'][index]['id']]) for index in manifest['splits']['train'])
    assert estimate.metadata['rows']['force'] == n_atoms * 2 * 3

    replay = molecular_collection_builder_task.func(
        moleculeids=[f'trajectory-{i:010d}' for i in range(16)], builder_config=trained_run.builder,
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
        stand_in_label(frame)
    store_current_data(str(label_path.with_name('data-0001.h5')), acquired, properties)
    _, retraining = ml.load_excited_state_h5_arrays(str(label_path.parent), properties, mixed_molecules=True)
    assert len(retraining['frames']) == len(frames) + len(acquired)
    assert {row[MOLECULE_ID] for row in retraining['frames']} == {'C=O', 'CCC=O', 'CC(C)=O'}
    second = run / 'models' / 'model-0001' / 'model-00'
    ml.train_single_excited_state_model(model_id=0, model_dir=str(second), h5_train_dir=str(label_path.parent),
                                        properties_list=properties, ML_config=trained_run.config)
    assert len(json.loads((second / 'training_input_manifest.json').read_text())['frames']) == len(frames) + len(acquired)


def test_oscillator_readout_fits_on_mixed_molecule_labels(trained_run):
    """The optical readout featurizes a mixed-molecule member, fits on its bootstrap labels and scores held-out ones."""
    from ase.io import write
    from alframework.builders.external_geometry import build_external_geometries
    from examples.molecular_excited_states.properties import PropertyFrame, featurize, fit, predict, score

    run, member, properties = trained_run.run, trained_run.member, trained_run.properties
    probes = [trained_run.frames[0].get_atoms(), trained_run.frames[1].get_atoms()]
    property_frames = [PropertyFrame('probe', str(i), atoms.numbers, atoms.positions, {}, None)
                       for i, atoms in enumerate(probes)]
    features, species, _ = featurize(property_frames, member, 'cpu', 2)
    assert features.shape[:2] == species.shape
    assert not features.requires_grad
    assert bool((features[species == 0] == 0).all())

    manifest_rows = json.loads((run / 'panel_manifest.json').read_text())['molecules']
    for split in ('validation', 'holdout'):
        paths = [row['seed_path'] for row in manifest_rows if row['split'] == split]
        held = build_external_geometries([f'{split}-0', f'{split}-1'],
                                         {'sources': paths, 'selection_mode': 'sequential'}, run)
        for frame in held:
            stand_in_label(frame)
        store_current_data(str(run / f'{split}.h5'), held, properties)
    optical = run / 'strengths.pt'
    fit('oscillator', member, run, [trained_run.label_path], [run / 'validation.h5'], optical, epochs=2)
    scored = score(optical, [run / 'holdout.h5'])
    assert scored['states'] == [1]
    assert 0 <= scored['mean_spectrum_overlap'] <= 1
    assert all(row['prediction'][0] >= 0 for row in scored['frames'])
    geometries = run / 'holdout.extxyz'
    write(geometries, [frame.get_atoms() for frame in held])
    bare = predict(optical, geometries)
    assert np.asarray([row['oscillator_strengths'] for row in bare['frames']]).shape == (2, 1)
