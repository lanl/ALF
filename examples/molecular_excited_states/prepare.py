"""Prepare an ALF run for state-resolved learning from a SMILES panel."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from ase.data import atomic_numbers

from alframework.builders.rdkit_molecule import _write_geometry_h5, prepare_rdkit_seed
from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS
from alframework.tools.files import cached_sha256
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


def load_panel(path, max_atoms, *, require_training=True):
    from rdkit import Chem

    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError('Provide a nonempty JSON list of SMILES or molecule records.')
    panel = {}
    for item in entries:
        record = {'smiles': item} if isinstance(item, str) else dict(item)
        smiles = record['smiles']
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(f'Expected one connected molecule for {smiles!r}.')
        if Chem.GetFormalCharge(molecule) or any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
            raise ValueError(f'Expected a neutral closed-shell molecule for {smiles!r}.')
        identity = Chem.MolToSmiles(molecule, isomericSmiles=True)
        if identity in panel:
            raise ValueError(f'Duplicate canonical SMILES across the panel: {identity}')
        count = Chem.AddHs(molecule).GetNumAtoms()
        if not 2 <= count <= max_atoms:
            raise ValueError(f'{identity}: {count} atoms including H, expected 2 to {max_atoms}.')
        split = record.get('split', 'train')
        if split not in {'train', 'validation', 'holdout'}:
            raise ValueError(f'{identity}: split must be train, validation or holdout.')
        panel[identity] = dict(record, split=split, n_atoms=count, molecule_id=identity)
    if require_training and not any(row['split'] == 'train' for row in panel.values()):
        raise ValueError('The panel needs at least one training molecule.')
    return list(panel.values())


def reference_conformers(mol, seed, limit):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Lipinski, rdMolAlign

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    heavy = Chem.RemoveHs(mol)
    count = 8 * limit if limit > 1 and Lipinski.NumRotatableBonds(heavy) else 1
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=count, params=parameters))
    if not ids:
        raise ValueError('RDKit could not embed a reference.')
    if AllChem.MMFFHasAllMoleculeParams(mol):
        energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        energies = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500)
    else:
        energies = [(0, 0.)] * len(ids)
    heavy = Chem.RemoveHs(mol)
    selected = []
    for conf_id, _ in sorted(zip(ids, energies), key=lambda item: item[1]):
        if all(rdMolAlign.GetBestRMS(Chem.Mol(heavy), heavy, conf_id, previous) >= .35 for previous in selected):
            selected.append(conf_id)
        if len(selected) == limit:
            break
    return selected


def interleave_conformer_seeds(paths, output, count, identity):
    pools = []
    for path in paths:
        with h5py.File(path) as handle:
            group = handle['molecule']
            numbers = [atomic_numbers[value.decode()] for value in group['species'][:]]
            pools.append((numbers, group[TOPOLOGY_ATOM_IDS][:], group['coordinates'][:], group['strata'][:]))
    numbers, atom_ids, *_ = pools[0]
    if any(z != numbers or not np.array_equal(ids, atom_ids) for z, ids, *_ in pools):
        raise ValueError('Conformer pools have incompatible atom order.')
    rows = [np.linspace(0, len(pool[2]) - 1, len(range(i, count, len(pools))), dtype=int)
            for i, pool in enumerate(pools)]
    selected = [(rows[i][round_index], i)
                for round_index, i in (divmod(index, len(pools)) for index in range(count))]
    _write_geometry_h5(output, np.stack([pools[i][2][row] for row, i in selected]), numbers, atom_ids,
                       np.arange(count), np.stack([pools[i][3][row] for row, i in selected]),
                       np.asarray([row for row, _ in selected]), identity)


def prepare_molecule(root, output, entry, pool_size, seed_size, seed, max_conformers=1):
    from rdkit import Chem

    identity = entry[MOLECULE_ID]
    digest = hashlib.sha256(f'{seed}|{identity}'.encode()).hexdigest()
    index = int(digest[:8], 16) & 0x7fffffff
    mol = Chem.AddHs(Chem.MolFromSmiles(identity))
    relative = Path(entry['split']) / digest[:16]
    (root / relative.parent).mkdir(exist_ok=True)
    path = root / 'reference.mol'
    try:
        conformers = reference_conformers(mol, index, min(max_conformers, seed_size))
        routes = [relative] + [relative / f'conformer-{i}' for i in range(1, len(conformers))]
        settings = dict(smiles=identity, reference_geometry=str(path), reference_charge=0,
                        pool_size=pool_size, seed_size=seed_size, max_embedding_rounds=20,
                        geometry_screen={'nonbonded_min_scale': 1.25})
        for i, (route, conf_id) in enumerate(zip(routes, conformers)):
            Chem.MolToMolFile(mol, str(path), confId=conf_id)
            prepare_rdkit_seed({**settings, 'seed': (index + i) & 0x7fffffff}, root / route,
                               molecule_identity=identity, published_directory=output / route)
        seed_path = relative / 'seed.h5'
        if len(routes) > 1:
            seed_path = relative / 'combined_seed.h5'
            interleave_conformer_seeds([root / route / 'seed.h5' for route in routes],
                                      root / seed_path, seed_size, identity)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f'Seed preparation failed for {identity}: {exc}') from exc
    finally:
        path.unlink(missing_ok=True)
    reference = relative / 'reference.mol'
    return dict(entry, seed_path=str(seed_path), reference_path=str(reference),
                seed_sha256=cached_sha256(root / seed_path), reference_sha256=cached_sha256(root / reference),
                conformer_manifests=[str(route / 'seed_manifest.json') for route in routes])


def configure(root, panel, *, mode, bootstrap_per_molecule, seed_size, seed, local, nroots=1,
              basis='6-31g*', model_width=145, optical=False, extra_labels=12, gap_seeking_after=None):
    base = HERE.parent / 'AcAc_excited_pyseqm'
    master = json.loads((base / 'master_config.json').read_text())
    model = json.loads((base / 'hippynn_config.json').read_text())
    sampler = json.loads((base / 'sampler_config.json').read_text())
    train = [row for row in panel if row['split'] == 'train']
    n = len(train)
    properties = state_properties(nroots, optical)
    pairs = [[i, i + 1] for i in range(nroots)]
    master.update(
        builder_task='alframework.builders.external_geometry.molecular_collection_builder_task',
        QM_task='alframework.qm_interfaces.pyscf.gpu4pyscf.gpu4pyscf_excited_state_task',
        properties_list=properties, bootstrap_set=n * bootstrap_per_molecule,
        bootstrap_spares=n * (seed_size - bootstrap_per_molecule),
        retraining={'after': n * 4}, parallel_samplers=32,
        maximum_stored_labels=n * (bootstrap_per_molecule + extra_labels),
        maximum_builder_structures=8, trajectory_summary_path='status_plots/trajectories.jsonl',
    )
    model.pop('n_atoms')
    model['network_params'].pop('possible_species')
    model['network_params']['n_features'] = model_width
    model.update(n_models=4 if mode == 'ensemble' else 1, mixed_molecules=True, random_seed=seed,
                 gap_targets={'enabled': True, 'weight': 1.0})
    sampler.update(state_selection={'mode': 'batch_cycle', 'states': list(range(nroots + 1))}, dt=.25, maxt=.1, Ncheck=10,
                   min_time=.005, Escut=25.0, Fscut=.15, random_seed=seed,
                   gap_uncertainty={'pairs': 'adjacent'}, gap_std_cut=.01,
                   velocity_initialization='maxwell_boltzmann_if_missing',
                   batching={'batch_size': 8, 'partial_policy': 'timeout', 'max_wait_seconds': 10},
                   topology_check={
                       'enabled': True, 'reference_format': 'auto', 'reference_charge': 0,
                       'references': {row[MOLECULE_ID]: row['reference_path'] for row in train},
                       'bond_min_scale': .85, 'bond_max_scale': 1.5, 'connectivity_scale': 1.25,
                   })
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
    write_json(root / 'builder_config.json', {
        'sources': [row['seed_path'] for row in train], 'selection_seed': seed,
        'replay_closed_shards': True,
    })
    write_json(root / 'panel_manifest.json', {'acquisition': mode, 'molecules': panel})


def prepare(panel, output, *, mode='ensemble', max_atoms=30, bootstrap_per_molecule=8, seed_size=12,
            pool_size=48, seed=42, local=False, max_conformers=1, validation_panel=None, **settings):
    if not 1 <= bootstrap_per_molecule <= seed_size <= pool_size:
        raise ValueError('Require 1 <= bootstrap-per-molecule <= seed-size <= pool-size.')
    records = load_panel(panel, max_atoms)
    if validation_panel is not None:
        validation = load_panel(validation_panel, max_atoms, require_training=False)
        if any(row['split'] != 'validation' for row in validation):
            raise ValueError('Separate validation entries must declare split=validation.')
        if {row[MOLECULE_ID] for row in records} & {row[MOLECULE_ID] for row in validation}:
            raise ValueError('A validation molecule already occurs in the main panel.')
        records.extend(validation)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.panel-', dir=output.parent) as temporary:
        root = Path(temporary) / 'run'
        root.mkdir()
        panel = [prepare_molecule(root, output, row, pool_size, seed_size, seed, max_conformers)
                 for row in records]
        configure(root, panel, mode=mode, bootstrap_per_molecule=bootstrap_per_molecule,
                  seed_size=seed_size, seed=seed, local=local, **settings)
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
