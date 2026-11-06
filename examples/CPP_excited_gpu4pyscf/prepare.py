"""Copy a fresh gas-phase CPP ALF example with a minimized S0 reference.

python prepare.py --ring-size 6 --output-dir /scratch/cpp6 --generate-seeds
For other CPP sizes provide --reference, --energy-offset-ev and --ring-size.
Model choices: hiphop145 (default), hiphop64, hipnn64; all use four members.
Configure parsl_configs.py for your site, then run in the prepared directory:
python -m alframework --master master_config.json
GPU4PySCF workers require a compatible cuTENSOR installation/library path.
Preparation is CPU-only and submits no QM, training or dynamics jobs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


HERE = Path(__file__).resolve().parent


def prepare(output_dir, *, ring_size=9, reference=None, energy_offset_ev=None,
            count=1000, model='hiphop145', generate_seeds=False):
    if isinstance(ring_size, bool) or not isinstance(ring_size, int) or ring_size < 3:
        raise ValueError('ring_size must be an integer >= 3')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('count must be a positive integer')
    if model not in {'hiphop145', 'hiphop64', 'hipnn64'}:
        raise ValueError('Unknown model profile')
    if reference is None:
        reference = HERE/f'references/cpp{ring_size}_s0.xyz'
        if not reference.is_file():
            raise ValueError('Provide a minimized S0 reference and its energy offset for this CPP size')
        record = json.loads(reference.with_suffix('.json').read_text())
        if hashlib.sha256(reference.read_bytes()).hexdigest() != record['reference_sha256']:
            raise ValueError('Minimized reference fingerprint changed')
        if not record['success'] or not record['optimizer_converged']:
            raise ValueError('Reference optimization did not converge')
        offset = record['energy_offset_eV'] if energy_offset_ev is None else energy_offset_ev
    else:
        reference = Path(reference).expanduser().resolve(strict=True)
        if energy_offset_ev is None:
            raise ValueError('A custom reference requires an explicit energy offset; do not reuse the CPP9 offset')
        record = dict(reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                      optimization_verified=False, source='caller-supplied minimized S0 reference')
        offset = energy_offset_ev
    if isinstance(offset, bool) or not math.isfinite(float(offset)):
        raise ValueError('energy_offset_ev must be finite')
    from ase.io import read
    from alframework.builders.ring_molecule import DEFAULT_SETTINGS, infer_topology
    atoms = read(reference, parallel=False)
    if infer_topology(atoms, DEFAULT_SETTINGS['screening'])['nring'] != ring_size:
        raise ValueError('Reference topology disagrees with requested CPP size')
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing existing run: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        for path in HERE.iterdir():
            if path.suffix in {'.json', '.py'}:
                shutil.copyfile(path, temporary/path.name)
        shutil.copytree(HERE/'references', temporary/'references')
        shutil.copyfile(reference, temporary/'reference.xyz')
        if hashlib.sha256((temporary/'reference.xyz').read_bytes()).hexdigest() != record['reference_sha256']:
            raise RuntimeError('Reference changed during preparation')
        def update(name, callback):
            path = temporary/name
            config = json.loads(path.read_text())
            callback(config)
            path.write_text(json.dumps(config, indent=2)+'\n')
        update('master_config.json', lambda c: c.update(bootstrap_set=count))
        update('QM_config.json', lambda c: c.update(energy_offset_eV=float(offset)))
        def model_settings(config):
            config['n_atoms'] = len(atoms)
            if model != 'hiphop145':
                config['network_params']['n_features'] = 64
            if model == 'hipnn64':
                config['network_choice'] = 0
                for key in ('l_max', 'n_max'):
                    config['network_params'].pop(key, None)
        update('hippynn_config.json', model_settings)
        seed = dict(DEFAULT_SETTINGS, reference_geometry='reference.xyz', count=count)
        (temporary/'seed_config.json').write_text(json.dumps(seed, indent=2)+'\n')
        if generate_seeds:
            from alframework.builders.ring_molecule import prepare_ring_pool
            prepare_ring_pool(seed, temporary/'seeds', config_directory=temporary)
            builder = temporary/'seeds/builder_config.json'
            builder.write_text(json.dumps(dict(seed_manifest_path='seeds/provenance.json', selection_seed=seed['seed']), indent=2)+'\n')
            manifest_path = temporary/'seeds/provenance.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['artifacts_sha256']['builder_config.json'] = hashlib.sha256(builder.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        record.update(ring_size=ring_size, energy_offset_eV=float(offset), model=model,
                      seeds_prepared=generate_seeds, count=count, campaign_submitted=False,
                      solvent='gas phase; implicit solvent requires a separately qualified adapter')
        (temporary/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ring-size', type=int, default=9)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--energy-offset-ev', type=float)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--model', choices=['hiphop145', 'hiphop64', 'hipnn64'], default='hiphop145')
    parser.add_argument('--generate-seeds', action='store_true')
    args = parser.parse_args(argv)
    print(prepare(args.output_dir, ring_size=args.ring_size, reference=args.reference,
                  energy_offset_ev=args.energy_offset_ev, count=args.count,
                  model=args.model, generate_seeds=args.generate_seeds))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
