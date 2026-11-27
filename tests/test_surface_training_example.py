"""Verify the shared periodic data and fixed-cell dynamics in the training example."""

import json
from pathlib import Path

import numpy as np
import pytest

from alframework.ml_interfaces.fine_tuning import ENERGY_KEY, FORCES_KEY, fine_tune_settings, shard_atoms
from alframework.ml_interfaces.fine_tuning.uma import UmaFineTuner
from alframework.qm_interfaces.ase.vasp import VASP_ase_calculator_task
from alframework.tools.data_shards import store_current_data
from alframework.tools.label_sources import read_label_sources
from alframework.tools.molecules import MoleculesObject
from examples.surface_training.custom_modules import surface_training_builder_task
from tests.helpers.fakes import FixedCalculator


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'surface_training'


def _config(filename):
    return json.loads((EXAMPLE / filename).read_text())


def _build(index, **overrides):
    config = _config('builder_config.json')
    config.update(overrides)
    return surface_training_builder_task.func(f'mol-boot-{index:010d}', config, str(EXAMPLE))


def test_example_builds_clean_slabs_and_all_requested_sites():
    candidates = [_build(i) for i in range(24)]
    metadata = [item.get_metadata()['surface_builder'] for item in candidates]
    assert {info['material_id'] for info in metadata} == {'clean_cu111', 'co_cu111'}
    assert {info.get('site_type') for info in metadata} == {None, 'ontop', 'bridge', 'hollow'}
    for item in candidates:
        atoms = item.get_atoms()
        assert atoms.pbc.all()
        assert item.get_metadata()['vacuum_axes'] == ['c']
        assert atoms.cell[2, 2] - np.ptp(atoms.positions[:, 2]) == pytest.approx(20, abs=0.2)
    clean = [item.get_atoms() for item, info in zip(candidates, metadata)
             if info['material_id'] == 'clean_cu111']
    np.testing.assert_array_equal(clean[0].cell, clean[1].cell)
    assert not np.allclose(clean[0].positions, clean[1].positions)
    np.testing.assert_array_equal(candidates[0].get_atoms().positions, _build(0).get_atoms().positions)


def test_example_stores_vasp_free_energy_and_periodic_cells(tmp_path, monkeypatch):
    import ase.calculators.vasp
    from ase.build import bulk

    masters = [_config(f'master_{name}.json') for name in ('uma', 'hippynn')]
    coefficients = masters[0]['properties_list']
    assert masters[1]['properties_list'] == coefficients
    assert len({master['status_path'] for master in masters}) == 2
    assert len({master['h5_path'] for master in masters}) == 2
    assert len({master['model_path'] for master in masters}) == 2

    vasp_constraints = {}

    class ExampleVasp(ase.calculators.vasp.Vasp):
        def __init__(self, directory, command, **options):
            super().__init__(directory=directory, command=command, **options)
            Path(directory).mkdir()
            assert options['xc'] == 'RPBE' and options['nsw'] == 0
            assert options['ediff'] == 1e-7 and options['nelm'] == 200

        def calculate(self, atoms, properties):
            assert atoms.pbc.all()
            assert set(properties) == {'free_energy', 'forces'}
            directory = Path(self.directory)
            vasp_constraints[directory.name] = list(atoms.constraints)
            self.spinpol, self.sort = False, np.arange(len(atoms))
            self.write_incar(atoms, directory=directory)
            self.write_kpoints(atoms=atoms, directory=directory)
            (directory / 'OUTCAR').write_text('aborting loop because EDIFF is reached\n')
            self.results = {'energy': 200, 'free_energy': -100.0, 'forces': np.ones((len(atoms), 3))}

    def check_mesh(molecule, expected):
        kpoints = (tmp_path / molecule.get_moleculeid() / 'KPOINTS').read_text().splitlines()
        assert kpoints[2].startswith('Gamma')
        assert tuple(map(int, kpoints[3].split())) == expected

    def incar(molecule):
        lines = (tmp_path / molecule.get_moleculeid() / 'INCAR').read_text().splitlines()
        return dict(tuple(part.strip() for part in line.split('=', 1)) for line in lines if '=' in line)

    monkeypatch.setattr(ase.calculators.vasp, 'Vasp', ExampleVasp)
    config = _config('vasp_config.json')
    cu = MoleculesObject(bulk('Cu', cubic=True, a=4), 'bulk-cu')
    ag_atoms = bulk('Ag', cubic=True, a=4)
    ag = MoleculesObject(ag_atoms, 'bulk-ag')
    ag.update_atoms(ag_atoms.repeat((2, 1, 1)))
    slabs = [_build(i, fixed_bottom_layers=1) for i in range(3)] + [_build(3, vacuum_A=32)]
    expected_meshes = [(11, 11, 11), (6, 11, 11)] + [(5, 5, 1)] * len(slabs)
    samples = [cu, ag, *slabs]
    labeled = [VASP_ase_calculator_task.func(item, config, str(tmp_path), coefficients) for item in samples]
    for item, mesh in zip(labeled, expected_meshes):
        check_mesh(item, mesh)
        assert 'LDIPOL' not in incar(item)
        if fixed := item.get_metadata().get('fixed_substrate_indices'):
            assert not vasp_constraints[item.get_moleculeid()]
            np.testing.assert_array_equal(item.get_atoms().constraints[0].get_indices(), fixed)
            np.testing.assert_array_equal(item.get_results()['forces'][fixed], np.ones((len(fixed), 3)))

    legacy = MoleculesObject(slabs[0].get_atoms().copy(), 'legacy-slab')
    legacy_config = {'QM_run_command': config['QM_run_command'], 'input': config['input'],
                     'slab_kspacing_A_inv': 0.18}
    result = VASP_ase_calculator_task.func(legacy, legacy_config, str(tmp_path), coefficients)
    check_mesh(result, (4, 4, 1))
    assert result.get_metadata()['label_source']['slab_kspacing_A_inv'] == 0.18

    representatives = {item.get_metadata()['surface_builder']['material_id']: item for item in slabs}
    assert set(representatives) == {'clean_cu111', 'co_cu111'}
    for item in [cu, *representatives.values()]:
        atoms = item.get_atoms().copy()
        expected_center = (atoms.get_center_of_mass(scaled=True)[2] + 0.5) % 1
        atoms.translate(atoms.cell[2] / 2)
        atoms.wrap()
        candidate = MoleculesObject(atoms, 'dipole-' + item.get_moleculeid())
        candidate.update_metadata(item.get_metadata())
        result = VASP_ase_calculator_task.func(
            candidate, dict(config, slab_dipole_correction=True), str(tmp_path), coefficients)
        tags = incar(result)
        assert result.get_metadata()['label_source']['slab_dipole_correction'] is True
        if item is cu:
            assert 'LDIPOL' not in tags
        else:
            assert tags['LDIPOL'] == '.TRUE.' and tags['IDIPOL'] == '3'
            np.testing.assert_allclose([float(value) for value in tags['DIPOL'].split()],
                                       [0.5, 0.5, expected_center], atol=1e-6)

    shard = tmp_path / 'data-0000.h5'
    assert store_current_data(str(shard), labeled, coefficients) == len(samples)
    source, = read_label_sources(shard)
    assert source['kpoint_spacing_A_inv'] == 0.15
    settings = fine_tune_settings(_config('uma_config.json'), tmp_path)
    uma = UmaFineTuner(settings)
    frames = shard_atoms(uma.prepare_data([shard], tmp_path), uma.settings['energy_dataset'],
                        uma.settings['forces_dataset'])
    assert len(frames) == len(samples)
    assert all(atoms.pbc.all() and atoms.info[ENERGY_KEY] == -100 for atoms in frames)
    assert all(np.all(atoms.arrays[FORCES_KEY] == 1) for atoms in frames)
    assert sorted(atoms.cell[2, 2] for atoms in frames) == sorted(item.get_atoms().cell[2, 2] for item in samples)


@pytest.mark.parametrize('name', ['uma', 'hippynn', 'ase-cohort'])
def test_example_samplers_keep_the_slab_cell(name, tmp_path, monkeypatch):
    from ase.io import read
    from alframework.samplers import mlmd_sampling

    candidate = _build(0, fixed_bottom_layers=int(name != 'uma'))
    fixed = candidate.get_metadata().get('fixed_substrate_indices', [])
    initial = candidate.get_atoms().positions.copy()
    if fixed:
        unperturbed = _build(0, fixed_bottom_layers=1, rattle_A=0).get_atoms()
        np.testing.assert_array_equal(initial[fixed], unperturbed.positions[fixed])
    cell = candidate.get_atoms().cell.array.copy()
    forces = np.zeros((len(candidate), 3))
    forces[0, 0] = 1.0
    members = [FixedCalculator(0, forces), FixedCalculator(0, -forces)]
    config = _config('sampler_uma.json' if name == 'uma' else 'sampler_hippynn.json')
    config.update(device='cpu', maxt=0.005, Ncheck=1, meta_dir=str(tmp_path), Fscut=0.01,
                  min_time=0.001, trajectory_frequency=1, trajectory_interval=1)
    if name == 'uma':
        pytest.importorskip('torch_sim')
        from alframework.samplers.batched_mlmd import sample_trajectories
        assert config['batched_calculator_options']['task'] == _config('uma_config.json')['fine_tune']['task']
        sample_trajectories([candidate], members, config, device='cpu')
    elif name == 'ase-cohort':
        from alframework.samplers.batched_mlmd import sample_trajectories
        from alframework.tools.cohorts import CohortPool

        selection = dict(mode='cohort_top_k', cohort_size=1, k=1)
        config.update(driver=dict(strategy='batched', engine='ase'), batching=dict(batch_size=2),
                      acquisition=dict(selection=selection))
        pool = CohortPool(selection)
        pool.admit(candidate, model_id=0)
        sample_trajectories([candidate], members, config, device='cpu')
        pool.record(candidate)
        candidate, = pool.ready()
    else:
        monkeypatch.setattr(mlmd_sampling, 'load_module_from_config', lambda *_: lambda *args, **kwargs: members)
        mlmd_sampling.simple_mlmd_sampling_task.func(candidate, config, 'model-{:04d}', 0, 0)
    assert candidate.get_metadata()['vacuum_axes'] == ['c']
    np.testing.assert_allclose(candidate.get_atoms().cell, cell, rtol=1e-7, atol=1e-8)
    if fixed:
        atoms = candidate.get_atoms()
        assert candidate.get_metadata()['fixed_substrate_indices'] == fixed
        np.testing.assert_array_equal(atoms.constraints[0].get_indices(), fixed)
        np.testing.assert_array_equal(atoms.positions[fixed], initial[fixed])
        assert not np.array_equal(atoms.positions, initial)
    trajectory, = tmp_path.rglob('*.traj')
    frames = read(trajectory, index=':')
    assert len(frames) > 1
    for atoms in frames:
        assert atoms.pbc.all()
        np.testing.assert_allclose(atoms.cell, cell, rtol=1e-7, atol=1e-8)
        np.testing.assert_array_equal(atoms.positions[fixed], initial[fixed])
