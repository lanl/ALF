"""Check the paper pilots' composition and state-matching contracts."""

import json
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from examples.molecular_excited_states import evaluate, prepare, qualify
from examples.surface_training.custom_modules import perovskite_training_builder_task
from examples.surface_training import perovskite_benchmark as benchmark
from examples.surface_training import perovskite_bulk as bulk


def test_four_relaxed_end_members_and_separate_ordered_alloy(tmp_path, monkeypatch):
    pytest.importorskip('rdkit')
    from ase.calculators import vasp

    targets = {'CsPbBr3': 5.94, 'CsPbI3': 6.28, 'CsSnBr3': 5.86, 'CsSnI3': 6.14}
    seen = {}

    class QuadraticVasp(Calculator):
        implemented_properties = ['energy', 'free_energy']
        converged = True

        def __init__(self, *, directory, command, **options):
            super().__init__(directory=directory)
            assert command == 'vasp_std'
            assert options['xc'] == 'RPBE' and options['ivdw'] == 11 and 'ldipol' not in options
            assert options['kpts'][0] == options['kpts'][1] == options['kpts'][2] > 1
            composition = Path(directory).parent.name
            assert seen.setdefault(composition, options['kpts']) == options['kpts']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            target = targets[Path(self.directory).parent.name]
            energy = 10 * (atoms.cell.lengths()[0] - target) ** 2
            self.results = dict(energy=energy, free_energy=energy)

    monkeypatch.setattr(vasp, 'Vasp', QuadraticVasp)
    qm = benchmark.config('vasp_perovskite.json')
    directory = tmp_path / 'bulk'
    report = bulk.prepare_bulk(directory, qm=qm, scratch=tmp_path / 'scratch')
    assert set(report['training_end_members']) == set(targets)
    for composition, target in targets.items():
        assert read(directory / f'{composition}.extxyz').cell.lengths() == pytest.approx([target] * 3)
    orders = []
    for name in bulk.ALLOY_ORDERS:
        atoms = read(directory / f'CsPbSnI3-{name}.extxyz')
        assert Counter(atoms.get_chemical_symbols()) == {'Cs': 4, 'Pb': 2, 'Sn': 2, 'I': 12}
        assert atoms.cell.lengths() == pytest.approx([6.21, 12.42, 12.42])
        orders.append(atoms.get_chemical_symbols())
    assert orders[0] != orders[1]
    builder = benchmark.config('builder_perovskite.json')
    training = builder['materials']
    assert {entry['id'].split('_')[0] for entry in training} == set(targets)
    for entry in training:
        entry['bulk_file'] = str(directory / Path(entry['bulk_file']).name)
    bootstrap = Counter()
    for index in range(16):
        frame = perovskite_training_builder_task.func(f'bootstrap-{index:010d}', builder)
        material = frame.get_metadata()['surface_builder']['material_id']
        bootstrap[material, benchmark.termination(frame.get_atoms())] += 1
    assert len(bootstrap) == 16 and set(bootstrap.values()) == {1}

    pure = benchmark.prepare_heldout(1, bulk_dir=directory)
    alloy = benchmark.prepare_heldout(1, bulk_dir=directory, partition='alloy')
    assert len(pure) == 32 and len(alloy) == 8
    assert Counter(atoms.info['termination'] for atoms in pure) == {'CsX': 16, 'MX2': 16}
    assert Counter(atoms.info['site_metal'] for atoms in alloy) == {'Pb': 4, 'Sn': 4}
    for atoms in pure + alloy:
        assert len(atoms) <= 200 and atoms.pbc.all()
        assert atoms.cell.lengths()[2] - np.ptp(atoms.positions[:, 2]) == pytest.approx(20)
        atoms.calc = SinglePointCalculator(atoms, energy=0., forces=np.zeros_like(atoms.positions))

    class ZeroPotential(Calculator):
        implemented_properties = ['energy', 'forces']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = dict(energy=0., forces=np.zeros_like(atoms.positions))

    scores = benchmark.score(alloy, ZeroPotential())
    assert len(scores) == 4
    for score in scores.values():
        assert score['relative_energy_MAE_eV'] == 0.
        assert set(score['by_site_metal']) == {'Pb', 'Sn'}
    distances = benchmark.summarize_trajectory(alloy[:1])
    assert 'Pb_donor_distance_A' in distances and 'Sn_donor_distance_A' in distances


def test_perovskite_uma_benchmark_defaults_to_training_and_sampling_task(monkeypatch):
    trainer = benchmark.config('uma_perovskite.json')['fine_tune']
    sampler = benchmark.config('sampler_perovskite_uma.json')['batched_calculator_options']
    assert trainer['task'] == sampler['task'] == 'oc25'
    monkeypatch.setitem(sys.modules, 'fairchem.core', SimpleNamespace(
        FAIRChemCalculator=lambda model, task_name: (model, task_name)))
    monkeypatch.setitem(sys.modules, 'fairchem.core.units.mlip_unit', SimpleNamespace(
        load_predict_unit=lambda path, device: (path, device)))
    checkpoint = Path('uma-s-1p2p1.pt')
    expected = (str(checkpoint), 'cpu')
    assert benchmark.calculator('uma', checkpoint, 'cpu') == (expected, 'oc25')
    assert benchmark.calculator('uma', checkpoint, 'cpu', uma_task='oc20') == (expected, 'oc20')


def test_carbonyl_holdouts_and_basis_root_matching(monkeypatch):
    pytest.importorskip('rdkit')
    pyscf = pytest.importorskip('pyscf')
    molecules = prepare.load_panel(prepare.HERE / 'panel.json', 30)
    assert len(molecules) == 30 and max(row['n_atoms'] for row in molecules) == 28
    assert {row['name'] for row in molecules if row['split'] == 'holdout'} == {
        '2-methylpropanal', 'pentan-3-one', 'cycloheptanone', 'methacrolein', 'diacetyl', 'nonan-2-one'}
    assert Counter(row['family'] for row in molecules if row.get('qualification')) == {
        family: 1 for family in {row['family'] for row in molecules}
    }

    monkeypatch.setattr(pyscf.gto, 'intor_cross', lambda integral, left, right: np.eye(4))
    occ, vir = np.eye(4)[:, :1], np.eye(4)[:, 1:]
    amplitudes = np.eye(3)[:, None, :]
    left = (None, occ, vir, amplitudes, np.array([4., 5., 6.]), np.zeros(3))
    right = (None, occ, vir, -amplitudes[[1, 0, 2]], np.array([3.9, 4.1, 6.]), np.zeros(3))
    comparison = qualify.compare_roots(left, right, min_overlap=.8, gap_shift_eV=.2)
    assert comparison['root_matches_1_based'] == [2, 1, 3]
    assert comparison['matched_energy_shifts_eV'] == pytest.approx([.1, -1.1, 0.])
    assert comparison['flags'] == ['root_order_changed', 'vertical_energy_shift_large']
    assert comparison['s1_basis_sensitive']
    high_root_shift = (None, occ, vir, -amplitudes, np.array([4., 5., 7.]), np.zeros(3))
    comparison = qualify.compare_roots(left, high_root_shift, min_overlap=.8, gap_shift_eV=.2)
    assert comparison['flags'] == ['vertical_energy_shift_large']
    assert not comparison['s1_basis_sensitive']


def test_policy_comparison_checks_accepted_labels_and_counts_ntk_fit(tmp_path, monkeypatch):
    models, timelines = {}, {}
    for policy in ('ensemble', 'ntk', 'random'):
        models[policy] = tmp_path / policy
        member = models[policy] / 'model-00'
        member.mkdir(parents=True)
        (member / 'training_input_manifest.json').write_text(json.dumps({'frames': [{}, {}]}))
        timelines[policy] = tmp_path / f'{policy}-timeline'
        timelines[policy].mkdir()
        record = {'stage': 'sampler', 'start': 0., 'end': 2., 'seconds': 2., 'items': 1, 'outcome': 'ok'}
        if policy == 'ntk':
            record['phases'] = {'ntk_fit': 1.}
        (timelines[policy] / 'worker.jsonl').write_text(json.dumps(record) + '\n')

    monkeypatch.setattr(evaluate, 'evaluate', lambda *args, **kwargs: {
        'C=O': {'gap_MAE_eV': 0.1, 'state_force_MAE_eV_A': [0.2, 0.3]},
    })
    report = evaluate.compare_at_equal_budget(models, [], device='cpu', timelines=timelines)
    assert report['accepted_qm_labels'] == 2
    assert report['policies']['ntk']['mean_over_molecules'] == {'S1_force_MAE_eV_A': 0.3, 'S1_S0_gap_MAE_eV': 0.1}
    assert report['policies']['ntk']['timing']['ntk_fit_seconds_in_sampler'] == 1.
    assert report['policies']['ensemble']['timing']['ntk_fit_seconds_in_sampler'] == 0.
    (member / 'training_input_manifest.json').write_text(json.dumps({'frames': [{}]}))
    with pytest.raises(ValueError, match='equal accepted QM label counts'):
        evaluate.compare_at_equal_budget(models, [], device='cpu')


def test_chromophore_preset_builds_six_state_acquisition_from_one_schema(tmp_path):
    pytest.importorskip('rdkit')
    from alframework.qm_interfaces.pyscf.gpu4pyscf import validate_gpu4pyscf_properties
    from alframework.samplers.acquisition import acquisition_settings

    entries = prepare.load_panel(prepare.PANELS['chromophore'], 40)
    validation = prepare.load_panel(prepare.HERE / 'property_validation.json', 40, require_training=False)
    assert Counter(row['split'] for row in entries) == {'train': 27, 'holdout': 8}
    assert max(row['n_atoms'] for row in entries) == 40
    assert sum(row['n_atoms'] > 30 and row['split'] == 'train' for row in entries) == 2
    assert not {row['molecule_id'] for row in entries} & {row['molecule_id'] for row in validation}
    for row in [*entries, *validation]:
        row.update(seed_path=f'{row["name"]}.h5', reference_path=f'{row["name"]}.mol')
    prepare.configure(tmp_path, [*entries, *validation], mode='ntk', bootstrap_per_molecule=8,
                      seed_size=12, seed=42, local=True, nroots=5, model_width=64, optical=True,
                      extra_labels=24, gap_seeking_after=324)
    configs = {path.stem: json.loads(path.read_text()) for path in tmp_path.glob('*.json')}
    properties = configs['master_config']['properties_list']
    states, gaps = validate_gpu4pyscf_properties(properties, nroots=5, compute_nacr=False)
    sampler = configs['sampler_config']
    assert len(states) == 6 and len(gaps) == 5
    assert configs['hippynn_config']['network_params']['n_features'] == 64
    assert 'oscillator_strengths' in properties and 'transition_dipoles' in properties
    assert sampler['batched_calculator_options']['properties_list'] == properties
    assert sampler['batched_calculator_options']['scores']['gap_pairs'][-1] == [4, 5]
    assert len(acquisition_settings(sampler)['eligibility']) == 5
    assert set(configs['builder_config']['sources']) == {row['seed_path'] for row in entries if row['split'] == 'train'}


@pytest.fixture(scope='module')
def chalcogenide_preview(tmp_path_factory):
    """The chalcogenide example's preview run, with its seed, fixed-pool and holdout poses."""
    pytest.importorskip('rdkit')
    from examples.surface_training import chalcogenide

    run = chalcogenide.prepare(tmp_path_factory.mktemp('chalcogenide') / 'run', preview=True)
    poses = {name: read(run / f'{name}.extxyz', index=':') for name in ('seeds', 'fixed', 'holdout')}
    return chalcogenide, run, poses


def test_chalcogenide_bootstrap_balances_its_materials(chalcogenide_preview):
    _, _, poses = chalcogenide_preview
    bootstrap = poses['seeds'][:48]
    assert len(bootstrap) == 48
    assert set(Counter(atoms.info['material'] for atoms in bootstrap).values()) == {8}


def test_chalcogenide_hgse_is_held_out_of_training(chalcogenide_preview):
    _, _, poses = chalcogenide_preview
    assert all(atoms.info['composition'] != 'HgSe' for atoms in [*poses['seeds'], *poses['fixed']])
    assert sum(atoms.info['composition'] == 'HgSe' for atoms in poses['holdout']) == 16


def test_chalcogenide_poses_share_each_compositions_cell(chalcogenide_preview):
    chalcogenide, _, poses = chalcogenide_preview
    every = [*poses['seeds'], *poses['fixed'], *poses['holdout']]
    for composition in chalcogenide.LATTICES:
        frames = [atoms for atoms in every if atoms.info['composition'] == composition]
        for atoms in frames:
            assert len(atoms) <= 200 and atoms.pbc.all()
            np.testing.assert_allclose(atoms.cell.array, frames[0].cell.array)
            assert atoms.cell[2, 2] - np.ptp(atoms.positions[:, 2]) >= 20. - 1.e-6
            assert atoms.info['metal_donor_A'] > 0


def test_chalcogenide_builder_frames_get_slab_vasp_settings(chalcogenide_preview):
    from alframework.qm_interfaces.ase.vasp import vasp_input_for_structure

    chalcogenide, run, _ = chalcogenide_preview
    builder = json.loads((run / 'builder_config.json').read_text())
    sample = chalcogenide.chalcogenide_builder_task.func('test_builder', builder, master_directory=run)
    vasp = vasp_input_for_structure(sample.get_atoms(), json.loads((run / 'vasp_config.json').read_text()),
                                    sample.get_metadata())
    assert vasp['kpts'][2] == 1 and vasp['gamma'] and vasp['idipol'] == 3
    assert vasp['encut'] == 450 and vasp['sigma'] == .05


def test_chalcogenide_label_refuses_shards_outside_the_training_directory(chalcogenide_preview):
    chalcogenide, run, _ = chalcogenide_preview
    with pytest.raises(ValueError, match='outside the active training directory'):
        chalcogenide.label(run, 'holdout', run / 'runs' / 'h5' / 'data-0010.h5')
