import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import write

from alframework.tools.molecules import MoleculesObject
from tests.helpers.engines import DEVICE, require_engine


def structures(periodic=False):
    cell = [[7, 0, 0], [1, 7, 0], [0, 0, 7]] if periodic else None
    reference = Atoms('OH', positions=[[0.2, 0, 0], [1.4, 0, 0]], cell=cell, pbc=periodic)
    query = Atoms('OHH', positions=[[0.2, 0, 0], [1.4, 0, 0], [6 if periodic else -1, 0, 0]],
                  cell=cell, pbc=periodic)
    return reference, query


def load_test_potential(model_directory, device, *, sigma):
    from alframework.samplers.engines.packed import ASECalculatorBatch

    return ASECalculatorBatch([LennardJones(sigma=sigma)])


@pytest.mark.parametrize('pbc', [(False, False, False), (True, True, False), (True, True, True)],
                         ids=['molecule', 'slab', 'periodic'])
def test_local_novelty_agrees_for_mixed_size_batch_and_ase(pbc):
    torch = pytest.importorskip('torch')
    require_engine('torchsim')
    from alframework.ml_interfaces.local_novelty import LocalEnvironmentNovelty
    from alframework.tools.geometry import neighbor_search

    reference, query = structures(any(pbc))
    for atoms in (reference, query):
        atoms.pbc = pbc
    novelty = LocalEnvironmentNovelty([reference], device=DEVICE, distance_block_size=1)
    atoms = [reference, query]
    numbers = torch.as_tensor(np.concatenate([a.numbers for a in atoms]), device=DEVICE)
    positions = torch.as_tensor(np.concatenate([a.positions for a in atoms]), device=DEVICE)
    cells = torch.as_tensor(np.stack([a.cell.array for a in atoms]), device=DEVICE)
    periodicity = torch.as_tensor(reference.pbc, device=DEVICE)
    scores, maxima = novelty.score_batch(numbers, positions, cells, periodicity, [2, 3],
                                         neighbor_search('torch_linked_cell'))
    assert scores.device == numbers.device and maxima.device == numbers.device
    expected = [novelty.score_atoms(a) for a in atoms]
    np.testing.assert_allclose(scores.cpu(), np.concatenate(expected), atol=1e-6)
    np.testing.assert_allclose(maxima.cpu(), [a.max() for a in expected], atol=1e-6)
    assert expected[0].max() < 1e-6
    assert expected[1].max() > 0.05
    # A periodic image, translation or atom permutation cannot change the local descriptor.
    moved = query[[2, 0, 1]]
    moved.translate([2, 3, 4])
    if any(pbc):
        moved.positions[0] += query.cell[0]
    np.testing.assert_allclose(novelty.score_atoms(moved), expected[1][[2, 0, 1]], atol=1e-6)
    with pytest.raises(ValueError, match='element is missing'):
        novelty.score_atoms(Atoms('He'))


def test_packed_hook_scores_once_and_retains_observations_for_survivors():
    torch = pytest.importorskip('torch')
    from alframework.samplers.engines.packed import PackedMLMDEngine
    from alframework.testing import check_evaluator

    class RecordingEvaluator:
        n_models = 1

        def __init__(self):
            self.calls = []

        def evaluate(self, numbers, positions, cells, pbc, counts):
            return positions.new_zeros((1, len(counts))), positions[None, :, :] * 2

        def observe_uncertainty_batch(self, numbers, positions, cells, pbc, counts, energies, forces):
            self.calls.append(list(counts))
            torch.testing.assert_close(forces[0], positions * 2)
            scores = positions[:, 0].abs().numpy()
            return [{'uncertainty': {'test_max': float(row.max())}, 'atomic_uncertainty': {'test': row}}
                    for row in np.split(scores, np.cumsum(counts)[:-1])]

    atoms = list(structures())
    evaluator = RecordingEvaluator()
    rngs = [np.random.RandomState(index) for index in range(2)]
    engine = PackedMLMDEngine(atoms, evaluator, 0.5, [100, 100], rngs)
    engine.observe_physical(0)
    assert evaluator.calls == []
    assert engine.observe(0)['prediction']['uncertainty'] == [{'test_max': 1.4}]
    engine.finish(0)
    scores = engine.observe(1)['prediction']['atomic_uncertainty'][0]['test']
    np.testing.assert_allclose(scores, [0.2, 1.4, 1.0])
    assert evaluator.calls == [[2, 3]]
    engine.scale_cell(1, 1.1)
    assert engine.observe(1)['prediction']['uncertainty'][0]['test_max'] == pytest.approx(1.54)
    assert evaluator.calls == [[2, 3], [3]]
    engine.finish(1)
    assert check_evaluator(RecordingEvaluator(), list(structures()))[1]['test_max'] == 1.4


def test_packed_hook_isolates_a_failed_structure():
    pytest.importorskip('torch')
    from alframework.samplers.engines.packed import PackedMLMDEngine

    class OneBadStructure:
        n_models = 1

        def evaluate(self, numbers, positions, cells, pbc, counts):
            return positions.new_zeros((1, len(counts))), positions.new_zeros((1, len(numbers), 3))

        def observe_uncertainty_batch(self, numbers, positions, cells, pbc, counts, energies, forces):
            if 3 in counts:
                raise ValueError('Bad structure')
            return [{'uncertainty': {'ok': 0.}}]

    engine = PackedMLMDEngine(list(structures()), OneBadStructure(), 0.5, [100, 100],
                              [np.random.RandomState(0), np.random.RandomState(1)])
    with pytest.raises(ValueError, match='Bad structure'):
        engine.observe(1)
    engine.finish(1)
    assert engine.observe(0)['prediction']['uncertainty'] == [{'ok': 0.}]
    engine.finish(0)


@pytest.mark.parametrize('engine', ['ase', 'torchsim', 'alchemi'])
def test_local_novelty_triggers_single_model_sampling_and_keeps_atom_scores(tmp_path, engine):
    pytest.importorskip('torch')
    from alframework.ml_interfaces.local_novelty import LocalEnvironmentNovelty, LocalNoveltyCalculator
    from alframework.samplers.batched_mlmd import sample_trajectories, worker_ensemble

    reference, query = structures()
    reference_path = tmp_path / 'reference.xyz'
    write(reference_path, reference)
    config = dict(dt=0.5, maxt=0.002, Ncheck=5, distcut=0.3, min_time=0., random_seed=5,
                  srt_temp=[10, 10], end_temp=[10, 10], amp_temp=[0, 0], per_temp=[1, 1],
                  amp_dens=None, per_dens=None, end_dens=None, driver={'engine': engine},
                  acquisition={'terms': {'novel': {'uncertainty': 'local_novelty_max', 'cutoff': 0.1}},
                               'eligibility': ['novel'], 'strategy': 'largest'})
    novelty = LocalEnvironmentNovelty([reference])
    if engine == 'ase':
        models = [LocalNoveltyCalculator(LennardJones(sigma=1.0), novelty)]
    else:
        require_engine(engine)
        require_engine('torchsim')
        config.update(batched_calculator='alframework.ml_interfaces.local_novelty.load_local_novelty_batched',
                      batched_calculator_options={'base_loader': f'{__name__}.load_test_potential',
                                                  'reference_path': str(reference_path),
                                                  'base_options': {'sigma': 1.0}})
        models = worker_ensemble(config, tmp_path, DEVICE)
    molecules = [MoleculesObject(atoms.copy(), f'mol-0000-{index:010d}')
                 for index, atoms in enumerate([reference, query])]
    quiet, picked = sample_trajectories(molecules, models, config, device=DEVICE)
    assert quiet.get_atoms() is None and quiet.get_metadata()['trajectory_outcome'] == 'completed'
    metadata = picked.get_metadata()
    assert metadata['acquisition_trigger_flags'] == {'novel': True}
    expected = novelty.score_atoms(picked.get_atoms())
    np.testing.assert_allclose(metadata['atomic_uncertainty']['local_novelty'], expected, atol=1e-6)
    assert metadata['local_novelty_max'] == pytest.approx(expected.max(), abs=1e-6)
