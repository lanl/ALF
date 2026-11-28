"""Uncertainty a model reports of its own, besides an ensemble's spread, chooses frames in both sampler families.

Local-environment novelty, scored against reference structures, is one such score; its tests come last.
"""
import pickle

import numpy as np
import pytest
from ase import Atoms

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics, member_uncertainty
from alframework.samplers import acquisition
from alframework.samplers.acquisition import acquisition_settings, assess_frame, require_members, uses_spread
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.tools.molecules import MoleculesObject
from alframework.tools.trajectory_summary import OWN_UNCERTAINTY_PREFIX, trajectory_row
from tests.helpers.engines import DEVICE, require_engine
from tests.helpers.fakes import PairSpringCalculator
from tests.helpers.molecules import SPRING_WATER

OWN = {"terms": {"stretch": {"uncertainty": "stretch_energy", "cutoff": 0.05}}, "eligibility": ["stretch"],
       "strategy": "largest"}


class SelfUncertainSpring(PairSpringCalculator):
    """One model that reports its own uncertainty: here its spring energy, which grows away from equilibrium."""

    def calculate(self, atoms=None, properties=("energy",), system_changes=()):
        super().calculate(atoms, properties, system_changes or ["positions"])
        self.results["alf_uncertainty"] = {"stretch_energy": float(self.results["energy"])}


@pytest.mark.parametrize("term, message", [
    ({"uncertainty": "x"}, 'must be {"uncertainty"'),
    ({"uncertainty": "", "cutoff": 1.0}, "must name what the model reports"),
    ({"uncertainty": "x", "cutoff": 0}, "positive number"),
    ({"uncertainty": "x", "cutoff": True}, "positive number"),
])
def test_a_term_of_your_own_must_name_an_uncertainty_and_a_cutoff(term, message):
    with pytest.raises(ValueError, match=message):
        acquisition.custom_terms({"acquisition": {"terms": {"mine": term}}})
    with pytest.raises(ValueError, match="built-in term"):
        acquisition.custom_terms({"acquisition": {"terms": {"energy": {"uncertainty": "x", "cutoff": 1.0}}}})


def test_a_run_of_only_its_own_terms_needs_no_spread_cutoffs():
    settings = acquisition_settings({"acquisition": OWN})
    assert settings["eligibility"] == ["stretch"] and settings["strategy"] == "largest"
    with pytest.raises(ValueError, match="requires Escut"):
        acquisition_settings({})  # the default energy and force terms still need them
    with pytest.raises(ValueError, match="Unknown acquisition eligibility terms \\['mine'\\]"):
        acquisition_settings({"acquisition": dict(OWN, eligibility=["mine"])})


@pytest.mark.parametrize("config, spread", [
    ({"Escut": 1.0, "Fscut": 1.0}, True),
    ({"acquisition": OWN}, False),
    ({"acquisition": dict(OWN, strategy="uncertainty"), "Escut": 1.0, "Fscut": 1.0}, True),
    ({"acquisition": dict(OWN, strategy="summed", weights={"stretch": 1.0})}, False),
    ({"acquisition": dict(OWN, strategy="summed", weights={"stretch": 1.0, "energy": 0.5}), "Escut": 1.0}, True),
], ids=["default", "own_terms", "own_eligibility_spread_ranking", "summed_own", "summed_with_energy"])
def test_uses_spread_says_whether_a_run_needs_two_members(config, spread):
    assert uses_spread(config) is spread
    one = [PairSpringCalculator(3.5)]
    if spread:
        with pytest.raises(ValueError, match="has one member, so no spread"):
            require_members(one, config)
    else:
        require_members(one, config)
    require_members([PairSpringCalculator(3.5)] * 2, config)


def test_a_frame_is_chosen_and_ranked_by_the_models_own_uncertainty():
    config = {"acquisition": OWN, "distcut": 0.5}
    settings = acquisition_settings(config)
    low = dict(Es=0.0, Fs=0.0, Fsmax=0.0, stretch_energy=0.01)
    high = dict(low, stretch_energy=0.2)
    assert assess_frame(low, 1.0, 1.0, config, settings)["outcome"] == "continue"
    decision = assess_frame(high, 1.0, 1.0, config, settings)
    assert decision["outcome"] == "eligible" and decision["ranking_score"] == pytest.approx(4.0)
    assert decision["acquisition_trigger_flags"] == {"stretch": True}
    assert decision["acquisition_cutoffs"] == {"stretch": 0.05}
    assert (decision["Ecrit"], decision["Fcrit"], decision["Fmcrit"]) == (False, False, False)
    assert assess_frame(high, 0.1, 1.0, config, settings)["outcome"] == "rejected", "distcut still applies"


def test_an_evaluators_uncertainty_joins_its_spread():
    prediction = {"energies": np.zeros((1, 2)), "forces": [np.zeros((1, 3, 3)), np.zeros((1, 2, 3))],
                  "uncertainty": [{"dropout": 0.1}, {"dropout": 0.3}]}
    rows = ensemble_statistics(prediction)
    assert [row["dropout"] for row in rows] == [0.1, 0.3] and rows[0]["Es"] == 0.0
    for bad, message in (([{"dropout": 0.1}], "for 1 structures, not 2"),
                         ([{"Es": 1.0}, {}], "name of a built-in signal"),
                         ([{"dropout": float("nan")}, {}], "not a finite number")):
        with pytest.raises(ValueError, match=message):
            ensemble_statistics(dict(prediction, uncertainty=bad))


def test_ase_ensembles_average_what_their_members_report():
    assert member_uncertainty([{"alf_uncertainty": {"a": 1.0}}, {"alf_uncertainty": {"a": 3.0, "b": 2.0}}, {}]) == \
        {"a": 2.0, "b": 2.0}
    evaluator = ASEEnsembleEvaluator([SelfUncertainSpring(3.0), SelfUncertainSpring(5.0)])
    prediction = evaluator.predict([SPRING_WATER.copy()])
    expected = np.mean(prediction["energies"][:, 0])
    assert prediction["uncertainty"][0]["stretch_energy"] == pytest.approx(expected)
    calculator = MLMD_calculator([SelfUncertainSpring(3.0)])
    atoms = SPRING_WATER.copy()
    calculator.calculate(atoms, properties=["energy", "forces", "energy_stdev", "forces_stdev_mean",
                                            "forces_stdev_max"])
    assert calculator.results["energy_stdev"] == 0.0
    assert calculator.results["alf_uncertainty"]["stretch_energy"] > 0


SAMPLER = dict(dt=0.5, maxt=0.5, Ncheck=5, distcut=0.3, min_time=0.0, random_seed=5,
               srt_temp=[3000.0, 3000.0], end_temp=[3000.0, 3000.0], amp_temp=[0.0, 0.0], per_temp=[1.0, 1.0],
               amp_dens=None, per_dens=None, end_dens=None)


def atom_scores(atoms):
    return {'position_score': np.abs(atoms.positions[:, 0]), 'species_score': atoms.numbers / 10.}


class AtomScoreSpring(PairSpringCalculator):
    def __init__(self, scale=1., names=('position_score', 'species_score')):
        super().__init__(3.5)
        self.scale, self.names = scale, names

    def calculate(self, atoms=None, properties=('energy',), system_changes=()):
        super().calculate(atoms, properties, system_changes or ['positions'])
        scores = atom_scores(atoms)
        self.results['alf_atomic_uncertainty'] = {name: self.scale * scores[name] for name in self.names}
        self.results['alf_uncertainty'] = {
            name + '_max': float(values.max()) for name, values in self.results['alf_atomic_uncertainty'].items()
        }


def test_ase_replay_preserves_two_sources_and_can_trigger_either_one():
    atoms = SPRING_WATER.copy()
    prediction = ASEEnsembleEvaluator([AtomScoreSpring()]).predict([atoms])
    scores = atom_scores(atoms)
    for name, values in scores.items():
        np.testing.assert_array_equal(prediction['atomic_uncertainty'][0][name], values)
    [signals] = ensemble_statistics(prediction)
    assert signals['Fsmax'] == 0.
    for triggered in scores:
        terms = {name: {'uncertainty': name + '_max', 'cutoff': 0.01 if name == triggered else 100.}
                 for name in scores}
        config = {'acquisition': {'terms': terms, 'eligibility': list(terms), 'strategy': 'largest'}}
        decision = assess_frame(signals, 2., 0., config, acquisition_settings(config))
        assert decision['outcome'] == 'eligible'
        assert decision['acquisition_trigger_flags'] == {name: name == triggered for name in scores}


def test_ase_averages_only_members_with_the_same_atom_score_name():
    atoms = SPRING_WATER.copy()
    models = [AtomScoreSpring(), AtomScoreSpring(scale=3., names=('position_score',))]
    scores = atom_scores(atoms)
    evaluator = ASEEnsembleEvaluator(models)
    [observed] = evaluator.predict([atoms])['atomic_uncertainty']
    calculator = MLMD_calculator(models)
    calculator.calculate(atoms, properties=['energy', 'forces'])
    for values in (observed, calculator.results['alf_atomic_uncertainty']):
        np.testing.assert_allclose(values['position_score'], 2 * scores['position_score'])
        np.testing.assert_array_equal(values['species_score'], scores['species_score'])


@pytest.mark.parametrize('mode', ['serial', 'first_trigger', 'cohort'])
def test_selected_atom_scores_round_trip_with_their_frame(tmp_path, mode):
    from alframework.samplers.batched_mlmd import sample_trajectories
    from alframework.samplers.mlmd_sampling import mlmd_sampling
    from alframework.tools.cohorts import CohortPool

    terms = {'local': {'uncertainty': 'position_score_max', 'cutoff': 0.01}}
    acquisition = {'terms': terms, 'eligibility': ['local'], 'strategy': 'largest'}
    [molecule] = _molecules(1)
    source = AtomScoreSpring()
    molecule_id = molecule.get_moleculeid()
    config = dict(SAMPLER, maxt=0.01, meta_dir=str(tmp_path), meta_dir_layout='flat', acquisition=acquisition)
    if mode == 'serial':
        sampler_keys = ('dt', 'maxt', 'Ncheck', 'distcut', 'min_time', 'meta_dir', 'acquisition')
        frame = mlmd_sampling(molecule, MLMD_calculator([source]),
                              **{key: config[key] for key in sampler_keys},
                              Tamp=0., Tper=1., Tsrt=3000., Tend=3000., Ramp=None, Rper=None, Rend=None)
    else:
        if mode == 'cohort':
            selection = {'mode': 'cohort_top_k', 'k': 2, 'cohort_size': 1}
            acquisition.update(selection=selection, trajectory_policy='continue')
            pool = CohortPool(selection)
            pool.admit(molecule, 0)
        [frame] = sample_trajectories([molecule], [source], config)
        if mode == 'cohort':
            pool.record(frame)
            frames = pool.ready()
            assert len(frames) == 2
            frame = frames[0]
    metadata = frame.get_metadata()
    assert metadata['acquisition_trigger_flags'] == {'local': True}
    assert metadata['acquisition_cutoffs'] == {'local': 0.01}
    expected = atom_scores(frame.get_atoms())
    signals = metadata['uncertainty'] if mode == 'serial' else metadata
    assert signals['position_score_max'] == pytest.approx(expected['position_score'].max())
    with (tmp_path / f'metadata-{molecule_id}.p').open('rb') as stream:
        persisted = pickle.load(stream)
    if mode == 'cohort':
        persisted = next(candidate['metadata'] for candidate in persisted['acquisition_candidates']
                         if candidate['frame_id'] == metadata['frame_id'])
    for name, values in expected.items():
        np.testing.assert_array_equal(metadata['atomic_uncertainty'][name], values)
        np.testing.assert_array_equal(persisted['atomic_uncertainty'][name], values)
    assert 'atomic_uncertainty' not in trajectory_row(molecule_id, metadata)


def test_packed_engine_keeps_atom_scores_aligned_after_retiring_a_system():
    pytest.importorskip('torch')
    from alframework.samplers.engines.packed import PackedMLMDEngine
    from alframework.testing import check_evaluator

    atoms = [SPRING_WATER.copy(), Atoms('H2', positions=[[0., 0., 0.], [1.2, 0., 0.]])]
    engine = PackedMLMDEngine(atoms, [AtomScoreSpring()], 0.5, [300., 300.],
                              [np.random.RandomState(0), np.random.RandomState(1)])
    before = engine.observe(1)['prediction']['atomic_uncertainty'][0]
    engine.finish(0)
    retained = engine.observe(1)['prediction']['atomic_uncertainty'][0]
    for name, values in before.items():
        np.testing.assert_array_equal(retained[name], values)
    engine.scale_cell(1, 1.1)
    after = engine.observe(1)['prediction']['atomic_uncertainty'][0]
    for name, values in atom_scores(engine.atoms(1)).items():
        np.testing.assert_array_equal(after[name], values)
    # The evaluator checker also reads the packed-only return path.
    assert check_evaluator(engine.evaluator, [engine.atoms(1)])[0]['species_score_max'] == 0.1
    engine.finish(1)


def _molecules(count):
    return [MoleculesObject(SPRING_WATER.copy(), f"mol-0000-{index:010d}") for index in range(count)]


@pytest.mark.parametrize("engine", ["ase", "torchsim"])
def test_one_model_chooses_frames_by_its_own_uncertainty_in_the_multi_trajectory_sampler(engine):
    from alframework.samplers.batched_mlmd import sample_trajectories
    from tests.helpers.engines import require_engine

    if engine != "ase":
        require_engine(engine)
    config = dict(SAMPLER, acquisition=OWN, driver={"engine": engine})
    results = sample_trajectories(_molecules(3), [SelfUncertainSpring(3.5)], config)
    outcomes = [molecule.get_metadata()["trajectory_outcome"] for molecule in results]
    assert "eligible" in outcomes
    for molecule in results:
        metadata = molecule.get_metadata()
        assert metadata["Es"] == 0.0, "one member has no spread"
        assert metadata["max_uncertainty"]["stretch_energy"] > 0
        row = trajectory_row(molecule.get_moleculeid(), metadata)
        assert row[OWN_UNCERTAINTY_PREFIX + "stretch_energy"] == pytest.approx(metadata["max_uncertainty"]["stretch_energy"])
        if metadata["trajectory_outcome"] == "eligible":
            assert metadata["acquisition_trigger_flags"] == {"stretch": True}
    with pytest.raises(ValueError, match="has one member, so no spread"):
        sample_trajectories(_molecules(1), [SelfUncertainSpring(3.5)], dict(SAMPLER, Escut=1.0, Fscut=1.0))


def test_one_model_chooses_frames_by_its_own_uncertainty_in_the_one_trajectory_sampler():
    from alframework.samplers.mlmd_sampling import mlmd_sampling

    schedule = dict(Tamp=0.0, Tper=1.0, Tsrt=3000.0, Tend=3000.0, Ramp=None, Rper=None, Rend=None)
    common = {key: SAMPLER[key] for key in ("dt", "maxt", "Ncheck", "distcut", "min_time")}
    [molecule] = _molecules(1)
    result = mlmd_sampling(molecule, MLMD_calculator([SelfUncertainSpring(3.5)]), acquisition=OWN, **common,
                           **schedule)
    metadata = result.get_metadata()
    assert result.get_atoms() is not None, "the frame whose stretch passed its cutoff goes to labeling"
    assert metadata["max_uncertainty"]["stretch_energy"] > 0.05 and metadata["Es"] == 0.0
    # Without the model's own terms the same single model would never trigger: its spread is 0.
    [molecule] = _molecules(1)
    quiet = mlmd_sampling(molecule, MLMD_calculator([SelfUncertainSpring(3.5)]), Escut=1.0, Fscut=1.0, **common,
                          **schedule)
    assert quiet.get_atoms() is None and quiet.get_metadata()["Ecrit"] is False


def test_the_summary_reads_the_largest_own_uncertainty_back(tmp_path):
    from alframework.tools.trajectory_summary import TrajectorySummary, read_trajectory_summary

    summary = TrajectorySummary(tmp_path / "trajectories.jsonl")
    plain, own = _molecules(2)
    plain.update_metadata({"trajectory_outcome": "completed", "Es": 0.1})
    own.update_metadata({"trajectory_outcome": "eligible", "max_uncertainty": {"stretch_energy": 0.4}})
    summary.append([plain, own])
    columns = read_trajectory_summary(tmp_path / "trajectories.jsonl")
    values = list(columns[OWN_UNCERTAINTY_PREFIX + "stretch_energy"])
    assert np.isnan(values[0]) and values[1] == pytest.approx(0.4)


def test_the_coordinator_refuses_one_member_when_the_run_reads_the_spread(tmp_path, monkeypatch):
    from tests.helpers.driver import run_alf_driver
    from tests.helpers.driver import LOOP, restart_tasks

    with pytest.raises(ValueError, match=r"The ML configuration \(n_models 1\) has one member"):
        run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, []), **LOOP,
                       sampler_config={"Escut": 1.0, "Fscut": 1.0}, ml_config={"n_models": 1})


def test_a_one_member_run_that_reads_its_own_uncertainty_trains_and_retrains(tmp_path, monkeypatch):
    from tests.helpers.driver import run_alf_driver
    from tests.helpers.driver import LOOP, restart_tasks

    run = run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, []), **LOOP,
                         sampler_config={"acquisition": OWN}, ml_config={"n_models": 1})
    assert run.status["current_model_id"] >= 1


# --- Local-environment novelty ----------------------------------------------------------------------
# Distance of each atom's radial environment to the nearest same-element reference environment.

def novelty_structures(periodic=False):
    """A reference OH, and a query that adds a hydrogen the reference has never seen."""
    cell = [[7, 0, 0], [1, 7, 0], [0, 0, 7]] if periodic else None
    reference = Atoms('OH', positions=[[0.2, 0, 0], [1.4, 0, 0]], cell=cell, pbc=periodic)
    query = Atoms('OHH', positions=[[0.2, 0, 0], [1.4, 0, 0], [6 if periodic else -1, 0, 0]],
                  cell=cell, pbc=periodic)
    return reference, query


def load_test_potential(model_directory, device, *, sigma):
    from ase.calculators.lj import LennardJones
    from alframework.samplers.engines.packed import ASECalculatorBatch

    return ASECalculatorBatch([LennardJones(sigma=sigma)])


NOVELTY_CELLS = pytest.mark.parametrize('pbc', [(False, False, False), (True, True, False), (True, True, True)],
                                        ids=['molecule', 'slab', 'periodic'])


def novelty_case(pbc):
    pytest.importorskip('torch')
    from alframework.ml_interfaces.local_novelty import LocalEnvironmentNovelty

    reference, query = novelty_structures(any(pbc))
    for atoms in (reference, query):
        atoms.pbc = pbc
    return reference, query, LocalEnvironmentNovelty([reference], device=DEVICE, distance_block_size=1)


@NOVELTY_CELLS
def test_local_novelty_batch_matches_single_structures(pbc):
    import torch
    from alframework.tools.geometry import neighbor_search

    require_engine('torchsim')
    reference, query, novelty = novelty_case(pbc)
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


@NOVELTY_CELLS
def test_local_novelty_scores_are_invariant_to_moving_the_atoms(pbc):
    """A periodic image, translation or atom permutation cannot change the local descriptor."""
    _, query, novelty = novelty_case(pbc)
    moved = query[[2, 0, 1]]
    moved.translate([2, 3, 4])
    if any(pbc):
        moved.positions[0] += query.cell[0]
    np.testing.assert_allclose(novelty.score_atoms(moved), novelty.score_atoms(query)[[2, 0, 1]], atol=1e-6)


def test_local_novelty_refuses_a_missing_element():
    _, _, novelty = novelty_case((False, False, False))
    with pytest.raises(ValueError, match='element is missing'):
        novelty.score_atoms(Atoms('He'))


class RecordingEvaluator:
    """A packed evaluator whose per-atom score is |x|, recording the structures of each scoring call."""

    n_models = 1

    def __init__(self):
        self.calls = []

    def evaluate(self, numbers, positions, cells, pbc, counts):
        return positions.new_zeros((1, len(counts))), positions[None, :, :] * 2

    def observe_uncertainty_batch(self, numbers, positions, cells, pbc, counts, energies, forces):
        import torch

        self.calls.append(list(counts))
        torch.testing.assert_close(forces[0], positions * 2)
        scores = positions[:, 0].abs().numpy()
        return [{'uncertainty': {'test_max': float(row.max())}, 'atomic_uncertainty': {'test': row}}
                for row in np.split(scores, np.cumsum(counts)[:-1])]


def recording_engine():
    pytest.importorskip('torch')
    from alframework.samplers.engines.packed import PackedMLMDEngine

    evaluator = RecordingEvaluator()
    engine = PackedMLMDEngine(list(novelty_structures()), evaluator, 0.5, [100, 100],
                              [np.random.RandomState(index) for index in range(2)])
    return engine, evaluator


def test_local_novelty_packed_hook_scores_each_frame_once():
    from alframework.testing import check_evaluator

    engine, evaluator = recording_engine()
    engine.observe_physical(0)
    assert evaluator.calls == []
    assert engine.observe(0)['prediction']['uncertainty'] == [{'test_max': 1.4}]
    engine.observe(1)
    assert evaluator.calls == [[2, 3]]
    engine.finish(0)
    engine.finish(1)
    assert check_evaluator(RecordingEvaluator(), list(novelty_structures()))[1]['test_max'] == 1.4


def test_local_novelty_packed_hook_keeps_observations_for_survivors():
    engine, evaluator = recording_engine()
    engine.observe(0)
    engine.finish(0)
    np.testing.assert_allclose(engine.observe(1)['prediction']['atomic_uncertainty'][0]['test'], [0.2, 1.4, 1.0])
    assert evaluator.calls == [[2, 3]]
    engine.scale_cell(1, 1.1)
    assert engine.observe(1)['prediction']['uncertainty'][0]['test_max'] == pytest.approx(1.54)
    assert evaluator.calls == [[2, 3], [3]]
    engine.finish(1)


def test_local_novelty_packed_hook_isolates_a_failed_structure():
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

    engine = PackedMLMDEngine(list(novelty_structures()), OneBadStructure(), 0.5, [100, 100],
                              [np.random.RandomState(0), np.random.RandomState(1)])
    with pytest.raises(ValueError, match='Bad structure'):
        engine.observe(1)
    engine.finish(1)
    assert engine.observe(0)['prediction']['uncertainty'] == [{'ok': 0.}]
    engine.finish(0)


@pytest.mark.parametrize('engine', ['ase', 'torchsim', 'alchemi'])
def test_local_novelty_triggers_single_model_sampling(tmp_path, engine):
    pytest.importorskip('torch')
    from ase.calculators.lj import LennardJones
    from ase.io import write
    from alframework.ml_interfaces.local_novelty import LocalEnvironmentNovelty, LocalNoveltyCalculator
    from alframework.samplers.batched_mlmd import sample_trajectories, worker_ensemble

    reference, query = novelty_structures()
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
    # The frame is chosen by its atom scores, and keeps them.
    expected = novelty.score_atoms(picked.get_atoms())
    np.testing.assert_allclose(metadata['atomic_uncertainty']['local_novelty'], expected, atol=1e-6)
    assert metadata['local_novelty_max'] == pytest.approx(expected.max(), abs=1e-6)
