"""Example-owned replay policy; geometry loading belongs to ALF."""
import hashlib
import json
from pathlib import Path

import h5py
from parsl import python_app

SCHEMA = "alf.cpp_ring_pool.v1"

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_ring_molecules(moleculeids, builder_config, h5_path=None, current_h5_id=0, master_directory=None):
    """Sequential finite bootstrap, then deterministic closed-shard replay."""
    from alframework.builders.external_geometry import build_external_geometries
    root = Path(master_directory or '.').expanduser().resolve()
    path = Path(builder_config['seed_manifest_path']).expanduser()
    path = path if path.is_absolute() else root/path
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'completed':
        raise ValueError('Ring builder requires a completed preparation manifest')
    for name in ('seed.h5', 'reference.xyz'):
        if _sha256(path.parent/name) != manifest['artifacts_sha256'][name]:
            raise ValueError('Prepared ring input changed: '+name)
    if int(current_h5_id) == 0:
        replay = dict(sources=[str(path.parent/'seed.h5')], selection_mode='sequential')
    else:
        if h5_path is None:
            raise ValueError('Runtime ring replay requires h5_path')
        sources = []
        for index in range(int(current_h5_id)):
            shard = Path(str(h5_path).format(index))
            shard = shard if shard.is_absolute() else root/shard
            if not shard.is_file():
                raise FileNotFoundError(f'Closed label shard missing: {shard}')
            sources.append(str(shard))
        replay = dict(sources=sources, selection_mode='deterministic_hash',
                      selection_seed=builder_config.get('selection_seed', 42))
    molecules = build_external_geometries(moleculeids, replay, root)
    for molecule in molecules:
        molecule.update_metadata(dict(ring_manifest_sha256=_sha256(path), ring_size=manifest['ring_size'],
                                      replay_external_bootstrap=int(current_h5_id)==0))
        if int(current_h5_id) == 0:
            with h5py.File(path.parent/'seed.h5') as handle:
                metadata = molecule.get_metadata()
                group, frame = handle[metadata['geometry_group']], metadata['geometry_frame']
                molecule.update_metadata(dict(ring_tier=int(group['tier_index'][frame]),
                                              ring_rotation_angles_degrees=group['rotation_angles_degrees'][frame].tolist(),
                                              ring_rattle_rms_angstrom=float(group['rattle_rms_angstrom'][frame])))
    return molecules


@python_app(executors=['alf_sampler_executor'])
def ring_molecule_builder_task(moleculeid=None, moleculeids=None, builder_config=None,
                               h5_path=None, current_h5_id=0, master_directory=None):
    """Ordinary scalar/plural ALF builder; no generation or QM inside workers."""
    plural = moleculeids is not None
    molecules = build_ring_molecules(moleculeids if plural else [moleculeid], builder_config or {},
                                    h5_path, current_h5_id, master_directory)
    return molecules if plural else molecules[0]

