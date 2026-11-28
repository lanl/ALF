"""Prepare an ALF run for state-resolved learning from a SMILES panel."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from alframework.builders.rdkit_molecules import (
    collection_workflow_settings,
    load_molecule_panel as load_panel,
    prepare_rdkit_collection,
)
from alframework.tools.molecules import MOLECULE_ID


HERE = Path(__file__).resolve().parent
PANELS = {'carbonyl': HERE / 'panel.json', 'chromophore': HERE / 'chromophores.json'}


def state_properties(nroots, optical=False):
    scopes = {f'E{i}': 'system' for i in range(nroots + 1)}
    scopes.update({f'F{i}': 'atomic' for i in range(nroots + 1)})
    scopes.update({f'dE{i}{i + 1}': 'system' for i in range(nroots)})
    if optical:
        scopes.update(oscillator_strengths='system', transition_dipoles='system')
    return {key: [key, scope, 1.0] for key, scope in scopes.items()}


PROPERTIES = state_properties(1)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def configure(root, panel, *, mode, bootstrap_per_molecule, seed_size, seed, local, nroots=1,
              basis='6-31g*', model_width=145, optical=False, extra_labels=12, gap_seeking_after=None,
              collection_report=None):
    base = HERE.parent / 'AcAc_excited_pyseqm'
    master = json.loads((base / 'master_config.json').read_text())
    model = json.loads((base / 'hippynn_config.json').read_text())
    sampler = json.loads((base / 'sampler_config.json').read_text())
    n = sum(row['split'] == 'train' for row in panel)
    workflow = (collection_report['alf_configuration'] if collection_report is not None else
                collection_workflow_settings(panel, seed=seed, seed_size=seed_size,
                                             bootstrap_per_molecule=bootstrap_per_molecule))
    properties = state_properties(nroots, optical)
    pairs = [[i, i + 1] for i in range(nroots)]
    master.update(
        **workflow['master'],
        QM_task='alframework.qm_interfaces.pyscf.gpu4pyscf.gpu4pyscf_excited_state_task',
        properties_list=properties,
        retraining={'after': n * 4}, parallel_samplers=32,
        maximum_stored_labels=n * (bootstrap_per_molecule + extra_labels),
        maximum_builder_structures=8, trajectory_summary_path='status_plots/trajectories.jsonl',
    )
    model.pop('n_atoms')
    model['network_params'].pop('possible_species')
    model['network_params']['n_features'] = model_width
    model.update(n_models=4 if mode == 'ensemble' else 1, mixed_molecules=True, random_seed=seed,
                 gap_targets={'enabled': True, 'weight': 1.0})
    sampler.update(workflow['sampler'])
    sampler.update(state_selection={'mode': 'batch_cycle', 'states': list(range(nroots + 1))}, dt=.25, maxt=.1, Ncheck=10,
                   min_time=.005, Escut=25.0, Fscut=.15, random_seed=seed,
                   gap_uncertainty={'pairs': 'adjacent'}, gap_std_cut=.01,
                   velocity_initialization='maxwell_boltzmann_if_missing',
                   batching={'batch_size': 8, 'partial_policy': 'timeout', 'max_wait_seconds': 10})
    sampler['batched_calculator_options']['properties_list'] = properties
    if gap_seeking_after is not None:
        if not master['bootstrap_set'] < gap_seeking_after < master['maximum_stored_labels']:
            raise ValueError('Gap-seeking must start between bootstrap and the final label budget.')
        sampler['sampling_windows'] = [
            {'until_stored_labels': gap_seeking_after},
            {'gap_seeking': {'enabled': True, 'trigger_gap_threshold_eV': .1, 'sigma': 3.5, 'alpha_eV': .05}},
        ]
    acquisition = sampler['acquisition'] = {
        'strategy': 'random', 'trajectory_policy': 'continue',
        'selection': {'mode': 'cohort_top_k', 'cohort_size': 16, 'k': 8,
                      'max_candidates_per_trajectory': 4, 'group_by_selected_state': True},
    }
    if mode == 'ensemble':
        acquisition.update(strategy='largest', eligibility=['gap'])
    elif mode == 'ntk':
        sampler['batched_calculator'] = 'alframework.ml_interfaces.hippynn.ntk.load_hippynn_excited_state_ntk'
        sampler['batched_calculator_options'] = {
            'properties_list': properties, 'energy_scale': .05, 'force_scale': .1, 'sketch_dim': 64,
            'scores': {'forces': False, 'gap_pairs': pairs},
        }
        terms = {f'gap_ntk_{a}{b}': {'uncertainty': f'ntk_dE{a}{b}', 'cutoff': .01} for a, b in pairs}
        acquisition.update(strategy='largest', eligibility=list(terms), terms=terms)
    if local:
        base = HERE.parent / 'surface_training'
        master.update(parsl_configuration='parsl_configs.config_local',
                      parsl_debug_configuration='parsl_configs.config_debug', gpus_per_node=1)
    shutil.copyfile(base / 'parsl_configs.py', root / 'parsl_configs.py')
    write_json(root / 'master_config.json', master)
    write_json(root / 'hippynn_config.json', model)
    write_json(root / 'sampler_config.json', sampler)
    write_json(root / 'QM_config.json', {
        'xc': 'cam-b3lyp', 'basis': basis, 'nroots': nroots, 'charge': 0, 'multiplicity': 1,
        'density_fit': True, 'ground_state_grid_response': True,
    })
    write_json(root / 'builder_config.json', workflow['builder'])
    report = collection_report if collection_report is not None else {'molecules': panel}
    write_json(root / 'panel_manifest.json', {**report, 'acquisition': mode})


def prepare(panel, output, *, mode='ensemble', max_atoms=30, bootstrap_per_molecule=8, seed_size=12,
            pool_size=48, seed=42, local=False, max_conformers=1, validation_panel=None, **settings):
    if validation_panel is not None:
        records = load_panel(panel, max_atoms)
        validation = load_panel(validation_panel, max_atoms, require_training=False)
        if any(row['split'] != 'validation' for row in validation):
            raise ValueError('Separate validation entries must declare split=validation.')
        if {row[MOLECULE_ID] for row in records} & {row[MOLECULE_ID] for row in validation}:
            raise ValueError('A validation molecule already occurs in the main panel.')
        panel = [*records, *validation]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.panel-', dir=output.parent) as temporary:
        root = Path(temporary) / 'run'
        collection = prepare_rdkit_collection(
            dict(molecules=panel, max_atoms=max_atoms, pool_size=pool_size, seed_size=seed_size,
                 bootstrap_per_molecule=bootstrap_per_molecule, seed=seed, max_conformers=max_conformers),
            root, published_directory=output,
        )
        configure(root, collection['molecules'], mode=mode, bootstrap_per_molecule=bootstrap_per_molecule,
                  seed_size=seed_size, seed=seed, local=local, collection_report=collection, **settings)
        root.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--panel', type=Path, help='Override the preset with a custom manifest.')
    parser.add_argument('--validation-panel', type=Path, help='Add separate molecules with split=validation.')
    parser.add_argument('--preset', choices=PANELS, default='carbonyl')
    parser.add_argument('--mode', choices=('ensemble', 'ntk', 'random'), default='ensemble')
    parser.add_argument('--max-atoms', type=int)
    parser.add_argument('--model', choices=('hiphop64', 'hiphop145'))
    parser.add_argument('--basis', default='6-31g*')
    parser.add_argument('--nroots', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--optical', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--extra-labels', type=int, help='Post-bootstrap label budget per training molecule.')
    parser.add_argument('--gap-seeking-after', type=int, metavar='LABELS')
    parser.add_argument('--bootstrap-per-molecule', type=int, default=8)
    parser.add_argument('--seed-size', type=int, default=12)
    parser.add_argument('--pool-size', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local', action='store_true', help='Use one workstation GPU instead of Slurm.')
    options = vars(parser.parse_args())
    preset = options.pop('preset')
    broad = preset == 'chromophore'
    width = options.pop('model') or ('hiphop64' if broad else 'hiphop145')
    options['panel'] = options['panel'] or PANELS[preset]
    options['max_atoms'] = options['max_atoms'] if options['max_atoms'] is not None else (40 if broad else 30)
    options['optical'] = broad if options['optical'] is None else options['optical']
    options['extra_labels'] = options['extra_labels'] if options['extra_labels'] is not None else (24 if broad else 12)
    if options['extra_labels'] < 4:
        parser.error('--extra-labels must allow at least one round of four labels per molecule.')
    print(prepare(**options, model_width=int(width.removeprefix('hiphop')), max_conformers=3 if broad else 1))


if __name__ == '__main__':
    main()
