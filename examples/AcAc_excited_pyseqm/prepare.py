"""Copy a fresh AcAc ALF example; optionally prepare CPU RDKit/FPS seeds.

Example: python prepare.py --output-dir /scratch/acac --generate-seeds
Then configure parsl_configs.py for your site and run from that directory:
python -m alframework --master master_config.json
Preparation never starts Parsl, QM, training or dynamics.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile


HERE = Path(__file__).resolve().parent


def prepare(output_dir, *, model='hiphop145', pool_size=None, seed_size=None, generate_seeds=False):
    if model not in {'hiphop145', 'hiphop64', 'hipnn64'}:
        raise ValueError('Unknown model profile')
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing existing run: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        for path in HERE.iterdir():
            if path.suffix in {'.json', '.xyz', '.py'}:
                shutil.copyfile(path, temporary/path.name)
        def update(name, callback):
            path = temporary/name
            config = json.loads(path.read_text())
            callback(config)
            path.write_text(json.dumps(config, indent=2)+'\n')
        def seed_settings(config):
            if pool_size is not None:
                config['pool_size'] = pool_size
            if seed_size is not None:
                config['seed_size'] = seed_size
        update('seed_keto.json', seed_settings)
        seed = json.loads((temporary/'seed_keto.json').read_text())
        from alframework.builders.rdkit_molecule import resolve_config
        resolve_config(seed, base_directory=temporary)
        update('master_config.json', lambda c: c.update(bootstrap_set=seed['seed_size']))
        def model_settings(config):
            if model != 'hiphop145':
                config['network_params']['n_features'] = 64
            if model == 'hipnn64':
                config['network_choice'] = 0
                for key in ('l_max', 'n_max'):
                    config['network_params'].pop(key, None)
        update('hippynn_config.json', model_settings)
        if generate_seeds:
            from alframework.builders.rdkit_molecule import prepare_rdkit_seed
            prepare_rdkit_seed(seed, temporary/'seeds', config_directory=temporary)
            # Retain fingerprint checking while removing staging-directory
            # paths from the library's generated configuration suggestions.
            import hashlib
            builder = temporary/'seeds/builder_config.json'
            builder.write_text(json.dumps(dict(seed_manifest_path='seeds/provenance.json', selection_seed=seed['seed']), indent=2)+'\n')
            path = temporary/'seeds/provenance.json'
            manifest = json.loads(path.read_text())
            manifest['settings']['reference_geometry'] = 'reference.xyz'
            manifest['configuration_sha256'] = hashlib.sha256(json.dumps(manifest['settings'], sort_keys=True, allow_nan=False).encode()).hexdigest()
            manifest['alf_configuration']['master']['builder_config_path'] = 'seeds/builder_config.json'
            manifest['alf_configuration']['sampler']['topology_check']['reference_conformer_path'] = 'seeds/reference.xyz'
            manifest['artifacts_sha256']['builder_config.json'] = hashlib.sha256(builder.read_bytes()).hexdigest()
            path.write_text(json.dumps(manifest, indent=2)+'\n')
        (temporary/'preparation.json').write_text(json.dumps(dict(
            model=model, seeds_prepared=generate_seeds,
            bootstrap_set=seed['seed_size'], campaign_submitted=False), indent=2)+'\n')
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', choices=['hiphop145', 'hiphop64', 'hipnn64'], default='hiphop145')
    parser.add_argument('--pool-size', type=int)
    parser.add_argument('--seed-size', type=int)
    parser.add_argument('--generate-seeds', action='store_true')
    args = parser.parse_args(argv)
    print(prepare(args.output_dir, model=args.model, pool_size=args.pool_size,
                  seed_size=args.seed_size, generate_seeds=args.generate_seeds))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
