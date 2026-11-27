"""Create an ALF run directory for online S0/S1 learning from SMILES."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from alframework.builders.rdkit_molecule import prepare_rdkit_seed
from alframework.tools.molecules import MOLECULE_ID


HERE = Path(__file__).resolve().parent
PROPERTIES = {
    'E0': ['E0', 'system', 1.0], 'E1': ['E1', 'system', 1.0],
    'F0': ['F0', 'atomic', 1.0], 'F1': ['F1', 'atomic', 1.0], 'dE01': ['dE01', 'system', 1.0],
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_panel(path, max_atoms):
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
        if split not in {'train', 'holdout'}:
            raise ValueError(f'{identity}: split must be train or holdout.')
        panel[identity] = dict(record, split=split, n_atoms=count, molecule_id=identity)
    if not any(row['split'] == 'train' for row in panel.values()):
        raise ValueError('The panel needs at least one training molecule.')
    return list(panel.values())


def prepare_molecule(root, output, entry, pool_size, seed_size, seed):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    identity = entry[MOLECULE_ID]
    digest = hashlib.sha256(f'{seed}|{identity}'.encode()).hexdigest()
    index = int(digest[:8], 16) & 0x7fffffff
    mol = Chem.AddHs(Chem.MolFromSmiles(identity))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = index
    if AllChem.EmbedMolecule(mol, parameters) < 0:
        raise ValueError(f'Could not embed reference for {identity}.')
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    relative = Path(entry['split']) / digest[:16]
    (root / relative.parent).mkdir(exist_ok=True)
    path = root / 'reference.mol'
    Chem.MolToMolFile(mol, str(path))
    try:
        report = prepare_rdkit_seed(dict(smiles=identity, reference_geometry=str(path), reference_charge=0,
                                         seed=index, pool_size=pool_size, seed_size=seed_size,
                                         max_embedding_rounds=20,
                                         geometry_screen={'nonbonded_min_scale': 1.25}),
                                    root / relative, molecule_identity=identity,
                                    published_directory=output / relative)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f'Seed preparation failed for {identity}: {exc}') from exc
    finally:
        path.unlink()
    return dict(entry, seed_path=str(relative / 'seed.h5'), reference_path=str(relative / 'reference.mol'),
                seed_sha256=report['artifacts_sha256']['seed.h5'], reference_sha256=report['reference_sha256'])


def configure(root, panel, *, mode, bootstrap_per_molecule, seed_size, seed, local):
    base = HERE.parent / 'AcAc_excited_pyseqm'
    master = json.loads((base / 'master_config.json').read_text())
    model = json.loads((base / 'hippynn_config.json').read_text())
    sampler = json.loads((base / 'sampler_config.json').read_text())
    train = [row for row in panel if row['split'] == 'train']
    n = len(train)
    master.update(
        builder_task='alframework.builders.external_geometry.molecular_collection_builder_task',
        QM_task='alframework.qm_interfaces.pyscf.gpu4pyscf.gpu4pyscf_excited_state_task',
        properties_list=PROPERTIES, bootstrap_set=n * bootstrap_per_molecule,
        bootstrap_spares=n * (seed_size - bootstrap_per_molecule),
        retraining={'after': n * 4}, parallel_samplers=32,
        maximum_stored_labels=n * (bootstrap_per_molecule + 12),
        maximum_builder_structures=8, trajectory_summary_path='status_plots/trajectories.jsonl',
    )
    model.pop('n_atoms')
    model['network_params'].pop('possible_species')
    model.update(n_models=4 if mode == 'ensemble' else 1, mixed_molecules=True, random_seed=seed,
                 gap_targets={'enabled': True, 'weight': 1.0})
    sampler.update(state_selection={'mode': 'batch_cycle', 'states': [0, 1]}, dt=.25, maxt=.1, Ncheck=10,
                   min_time=.005, Escut=25.0, Fscut=.15, random_seed=seed,
                   gap_uncertainty={'pair': [0, 1]}, gap_std_cut=.01,
                   velocity_initialization='maxwell_boltzmann_if_missing',
                   batching={'batch_size': 8, 'partial_policy': 'timeout', 'max_wait_seconds': 10},
                   topology_check={
                       'enabled': True, 'reference_format': 'auto', 'reference_charge': 0,
                       'references': {row[MOLECULE_ID]: row['reference_path'] for row in train},
                       'bond_min_scale': .85, 'bond_max_scale': 1.5, 'connectivity_scale': 1.25,
                   })
    sampler['batched_calculator_options']['properties_list'] = PROPERTIES
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
            'properties_list': PROPERTIES, 'energy_scale': .05, 'force_scale': .1, 'sketch_dim': 64,
            'scores': {'forces': False, 'gap_pairs': [[0, 1]]},
        }
        acquisition.update(strategy='largest', eligibility=['gap_ntk'],
                           terms={'gap_ntk': {'uncertainty': 'ntk_dE01', 'cutoff': .01}})
    if local:
        base = HERE.parent / 'surface_training'
        master.update(parsl_configuration='parsl_configs.config_local',
                      parsl_debug_configuration='parsl_configs.config_debug', gpus_per_node=1)
    shutil.copyfile(base / 'parsl_configs.py', root / 'parsl_configs.py')
    write_json(root / 'master_config.json', master)
    write_json(root / 'hippynn_config.json', model)
    write_json(root / 'sampler_config.json', sampler)
    write_json(root / 'QM_config.json', {
        'xc': 'cam-b3lyp', 'basis': '6-31g*', 'nroots': 1, 'charge': 0, 'multiplicity': 1,
        'density_fit': True, 'ground_state_grid_response': True,
    })
    write_json(root / 'builder_config.json', {
        'sources': [row['seed_path'] for row in train], 'selection_seed': seed,
        'replay_closed_shards': True,
    })
    write_json(root / 'panel_manifest.json', {'acquisition': mode, 'molecules': panel})


def prepare(panel, output, *, mode='ensemble', max_atoms=30, bootstrap_per_molecule=8, seed_size=12,
            pool_size=48, seed=42, local=False):
    if not 1 <= bootstrap_per_molecule <= seed_size <= pool_size:
        raise ValueError('Require 1 <= bootstrap-per-molecule <= seed-size <= pool-size.')
    records = load_panel(panel, max_atoms)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.panel-', dir=output.parent) as temporary:
        root = Path(temporary) / 'run'
        root.mkdir()
        panel = [prepare_molecule(root, output, row, pool_size, seed_size, seed) for row in records]
        configure(root, panel, mode=mode, bootstrap_per_molecule=bootstrap_per_molecule,
                  seed_size=seed_size, seed=seed, local=local)
        root.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--panel', type=Path, default=HERE / 'panel.json')
    parser.add_argument('--mode', choices=('ensemble', 'ntk', 'random'), default='ensemble')
    parser.add_argument('--max-atoms', type=int, default=30)
    parser.add_argument('--bootstrap-per-molecule', type=int, default=8)
    parser.add_argument('--seed-size', type=int, default=12)
    parser.add_argument('--pool-size', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local', action='store_true', help='Use one workstation GPU instead of Slurm.')
    print(prepare(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
