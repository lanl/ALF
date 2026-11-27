"""Single-member NTK for conservative molecular UMA fine-tunes."""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

from alframework.ml_interfaces.fine_tuning.uma import BASIS_FILE, LORA_FILE, MODEL_NAME, SPLIT_FILE
from alframework.ml_interfaces.ntk.adapter import NTKAdapter, resolve_device_dtype
from alframework.ml_interfaces.ntk.estimator import GroundStateNTKPackedEvaluator, NTKEstimator, read_basis_metadata
from alframework.ml_interfaces.torchsim_ensemble import packed_state
from alframework.tools.files import cached_sha256

__all__ = ['load_uma_ntk', 'load_uma_ntk_estimator', 'prepare_uma_ntk']


class UMANTKAdapter(NTKAdapter):
    """Use the deployed EMA potential for both ordinary MD and reverse-mode scoring."""

    @torch.inference_mode(False)
    def __init__(self, directory, split, provenance, device):
        from fairchem.core.common.utils import setup_imports
        from fairchem.core.units.mlip_unit.utils import load_inference_model

        setup_imports()
        self.device, self.dtype = resolve_device_dtype(device, torch.float32)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('UMA NTK requires CPU or CUDA/ROCm.')
        self.charge, self.spin = split['charge'], split['spin']
        checkpoint = directory / MODEL_NAME
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        ema, saved = load_inference_model(str(checkpoint), use_ema=True, preloaded_checkpoint=saved,
                                          overrides={'backbone': {'execution_mode': 'general'}})
        self.model = ema.module
        self.model.setup_tasks(saved.tasks_config)
        config = self.model.backbone.regress_config
        if config.direct_forces or not config.forces or 'energy' not in self.model.tasks:
            raise ValueError('UMA NTK requires a conservative omol energy and force fine-tune.')
        # Derive forces once outside the head from the denormalized energy.
        config.forces = config.stress = config.hessian = False
        self.model.backbone.requires_grad_(False)
        if split['lora']:
            from alframework.ml_interfaces.fine_tuning.uma_lora import restore_ema_lora

            restore_ema_lora(self.model, directory / LORA_FILE)
        self.model.to(device=self.device, dtype=self.dtype).eval()
        task = self.model.tasks['energy']
        self.normalizer = task.normalizer.to(self.device)
        self.references = task.element_references.to(self.device) if task.element_references is not None else None
        self._parameters = {name: value for name, value in self.model.named_parameters()
                            if value.is_floating_point() and value.requires_grad}
        if not self._parameters:
            raise ValueError('The UMA NTK checkpoint has no trainable head or LoRA parameters.')
        self._digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()

    def digest(self):
        return self._digest

    def live_parameters(self):
        return self._parameters

    def _data(self, numbers, positions, cells, pbc, counts):
        from fairchem.core.calculate.torchsim_interface import _simstate_to_atomicdata_batch

        if min(counts) < 2 or torch.as_tensor(pbc).any():
            raise ValueError('UMA NTK supports nonperiodic molecules with at least two atoms.')
        if cells is None:
            cells = torch.zeros((len(counts), 3, 3), device=self.device, dtype=self.dtype)
        state = packed_state(numbers, positions, cells, pbc, counts, device=self.device, dtype=self.dtype,
                             charge=self.charge, spin=self.spin)
        return _simstate_to_atomicdata_batch(state, 'omol', self.dtype)

    def prepare(self, structure):
        structure.extra['data'] = self._data(structure.numbers, structure.positions, structure.cell[None],
                                             structure.pbc, [structure.n_atoms])

    def _energy(self, data, positions):
        data.pos = positions
        energy = self.normalizer.denorm(self.model(data)['energy']['energy'])
        return self.references.undo_refs(data, energy) if self.references is not None else energy

    def energies(self, structure, positions, params=None):
        if params is not None:
            raise ValueError('UMA NTK supports reverse-mode derivatives.')
        return self._energy(structure.extra['data'].clone(), positions).reshape(1)

    @torch.inference_mode(False)
    def predict_batch(self, numbers, positions, cells, pbc, counts):
        data = self._data(numbers, positions, cells, pbc, counts)
        with torch.enable_grad():
            x = positions.detach().to(device=self.device, dtype=self.dtype).clone().requires_grad_(True)
            energies = self._energy(data, x)
            gradient, = torch.autograd.grad(energies.sum(), x)
        return energies.detach()[None], -gradient.detach()[None]


class UMANTKEstimator(NTKEstimator):
    def structure_from_atoms(self, atoms):
        charge, spin = self.adapter.charge, self.adapter.spin
        if atoms.info.get('charge', charge) != charge or atoms.info.get('spin', spin) != spin:
            raise ValueError('Structure charge and spin do not match the UMA NTK basis.')
        return super().structure_from_atoms(atoms)


def _member_artifacts(directory):
    directory = Path(directory)
    try:
        content = (directory / SPLIT_FILE).read_bytes()
        split = json.loads(content)
        if split.get('version') != 1 or split.get('task') != 'omol' or 'ntk' not in split:
            raise ValueError('UMA NTK requires an omol member trained with fine_tune.ntk.')
        files = (MODEL_NAME, LORA_FILE) if split['lora'] else (MODEL_NAME,)
        provenance = {'split_sha256': hashlib.sha256(content).hexdigest(),
                      'files': {name: cached_sha256(directory / name) for name in files}}
    except FileNotFoundError as exc:
        raise ValueError(f'Missing UMA NTK artifact {exc.filename}. Fine-tune with fine_tune.ntk.') from exc
    return directory, split, provenance


def _training_frames(path, recorded):
    from ase.db import connect

    with connect(str(path), use_lock_file=False) as database:
        for row, frame_id in zip(database.select(sort='id'), recorded, strict=True):
            if row.data.get('alf_frame') != frame_id:
                raise ValueError('UMA training frame order differs from the recorded split.')
            atoms = row.toatoms()
            atoms.info.update(row.data)
            yield atoms


def prepare_uma_ntk(member_directory, device='cpu'):
    """Fit a basis on the actual training database of a completed fine-tune."""
    directory, split, provenance = _member_artifacts(member_directory)
    for name in ('train', 'valid'):
        entry = split[name]
        if cached_sha256(directory / entry['file']) != entry['sha256']:
            raise ValueError(f'UMA {name} database differs from the recorded split.')
    train = split['train']
    adapter = UMANTKAdapter(directory, split, provenance, device)
    estimator = UMANTKEstimator(adapter, _training_frames(directory / train['file'], train['frames']),
                                engine='reverse', fit_provenance=provenance, **split['ntk'])
    estimator.save_basis(directory / BASIS_FILE)
    return estimator


def load_uma_ntk_estimator(member_directory, device='cpu', *, charge=None, spin=None):
    """Restore a verified member and basis without opening any training databases."""
    directory, split, provenance = _member_artifacts(member_directory)
    if any(value is not None and (type(value) is not int or value != split[key])
           for key, value in (('charge', charge), ('spin', spin))):
        raise ValueError('Sampler charge and spin must match the UMA NTK training split.')
    try:
        with (directory / BASIS_FILE).open('rb') as stream:
            metadata = read_basis_metadata(stream)
            if (metadata.get('fit_provenance') != provenance
                    or metadata['training_structure_count'] != len(split['train']['frames'])):
                raise ValueError('The UMA NTK basis does not match the checkpoint, factors, or training split.')
            stream.seek(0)
            return UMANTKEstimator.from_basis(UMANTKAdapter(directory, split, provenance, device), stream,
                                              engine='reverse')
    except FileNotFoundError as exc:
        raise ValueError(f'Missing UMA NTK basis {exc.filename}. Fine-tune with fine_tune.ntk.') from exc


def load_uma_ntk(model_directory, device='cuda:0', *, member=0, charge=None, spin=None,
                 force_score='projected', force_probes=8, energy_score=False):
    """Load one fine-tuned UMA member for packed molecular dynamics with NTK acquisition."""
    members = sorted(Path(model_directory).glob('model-*/'))
    if type(member) is not int or not 0 <= member < len(members):
        raise ValueError(f'{model_directory} has {len(members)} members, not member {member}.')
    estimator = load_uma_ntk_estimator(members[member], device=device, charge=charge, spin=spin)
    return GroundStateNTKPackedEvaluator(estimator, force_score=force_score, force_probes=force_probes,
                                        energy_score=energy_score)


if __name__ == '__main__':
    started = perf_counter()
    fitted = prepare_uma_ntk(*sys.argv[1:])
    peak = torch.cuda.max_memory_allocated(fitted.device) if fitted.device.type == 'cuda' else None
    print(json.dumps({'basis_id': fitted.basis_id, 'device': str(fitted.device), 'fit_seconds': perf_counter() - started,
                      'peak_gpu_bytes': peak, **fitted.metadata}, indent=2))
